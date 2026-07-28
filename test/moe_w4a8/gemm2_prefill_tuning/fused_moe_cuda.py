import torch
import time
from lmslim import quant_ops 
from typing import Optional, Type, Any, Callable, Dict, List, Tuple
import pandas as pd
import os
import json
import triton
import triton.language as tl
import pandas as pd
import logging
import math
import torch.nn.functional as F
# from per_token_group_test import per_token_group_quant_int8
import functools
import logging
from vllm.platforms import current_platform
import vllm.envs as envs
from vllm.logger import init_logger
from vllm import _custom_ops as ops
# import _custom_ops as ops
from lightop.moe import moe_gemm_w4a8, get_moe_cuda_config_w4a8
from vllm.model_executor.layers.fused_moe.fused_moe import moe_kernel_prepare_input
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)

logger = init_logger(__name__)
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def ceil_div(a, b):
    return (a + b - 1) // b
                 
def invoke_fused_moe_kernel_int8(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w4a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[List[int]] = None,
                            use_nn_moe: Optional[bool]=False) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w4a8 is True

    moe_gemm_w4a8(A, 
                    B, 
                    C,
                    A_scale, 
                    B_scale,
                    topk_weights if mul_routed_weight else None,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    top_k,
                    config)

# 执行具体计算的实现
def _fused_moegemm_helper_test_pure(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w4a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       config: Optional[Dict[str, Any]] = None,
                       mode: Optional[int]=0):
    # Check constraints. 
    assert use_int8_w4a8 is True

    assert hidden_states.shape[1] == w1.shape[2] * 2, "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    if use_nn_moe:
        E, _, N = w1.shape
    else:
        E, N, _ = w1.shape

    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
    
        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        # if use_int8_w8a8:
        #     M=curr_hidden_states.shape[0]
        #     #print(f'+++++++-------_{M}_--------+++++++')
        #     # config=get_moe_config(M,N,K)
        #     config,status=get_moe_cuda_config(E,M,N,K,device_name,num_cus,compute_type)
            
        if use_int8_w4a8:
            if use_nn_moe:
                _, _K, _N = w2.shape
            else:
                _, _N, _K = w2.shape
            # _M=curr_hidden_states.shape[0]
            # config=get_moe_config(_M,_N,_K)
            config1, config2, status=get_moe_cuda_config_w4a8(E,M,512,3584,7168,128,top_k_num,device_name,num_cus,compute_type)
            # print("config1",config1)
            # print("config2",config2)
            
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        # print("gemm1开始执行.....")
        # 该函数在 GEMM1 适配后调用汇编实现
        invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                qa1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config1,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w4a8=use_int8_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)

        # print("gemm1执行完成.....")
        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, 512))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, 512))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")
        
        qintermediate_cache2, qa2_scale = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w2,
            A_scale=a2_scale,
            B_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)
        # print("gemm2开始执行.....")
        invoke_fused_moe_kernel_int8(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                qa2_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config2,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w4a8=use_int8_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        # print("gemm2执行完成.....")
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        
        
    return out_hidden_states   

# 函数总入口
def fused_moe_impl( hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    gating_output: torch.Tensor,
                    topk: int,
                    renormalize: bool,
                    inplace: bool = False,
                    use_grouped_topk: bool = True,
                    num_expert_group: Optional[int] = 8,
                    topk_group: Optional[int] = 4,
                    custom_routing_function: Optional[Callable] = None,
                    use_fp8_w8a8: bool = False,
                    use_int8_w4a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None,
                    use_nn_moe: Optional[bool] = False,
                    moe_ep_size: Optional[int] = 1,
                    start_expert: Optional[int] = None,
                    end_expert: Optional[int] = None,
                    config:  Dict[str, Any] = None,
) -> torch.Tensor:
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    
    return _fused_moegemm_helper_test_pure(hidden_states=hidden_states,
                                                    w1=w1,
                                                    w2=w2,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    inplace=False,
                                                    use_int8_w4a8=use_int8_w4a8,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    block_shape=block_shape,
                                                    config=config,
                                                    per_channel_quant=True)



            







