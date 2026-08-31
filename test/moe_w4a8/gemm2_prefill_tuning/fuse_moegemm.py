import torch
import time
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
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

M_THRESHOLD = 128

os.environ["TRITON_CACHE_DIR"] = f".triton/cache"

logger = init_logger(__name__)
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

# gemm2 prefill 阶段的config
blockM_gemm2_prefill_configs = {
    0 : 16,
    1 : 32,
    2 : 64,

    3 : 16,
    4 : 32,
    5 : 16,

    6 : 32,
    7 : 32,
    8 : 64,
    9 : 64,

    10: 64,

    11: 64,
    12: 32,

    13: 128,
}

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

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if GROUP_SIZE_M ==1:
        num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)
    #NN load copy in register by broadcast
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        #+ (offs_bk[None, :] * stride_bk + offs_bn[:, None] * stride_bn)
        + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)
    )
    b_shifter = (offs_ak[:,None]%2)*4

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    # reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
    #                             tl.arange(0, 4)[:, None]).to(tl.uint8).reshape(8)
    

    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm

    b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn
        )

    b_scale = tl.load(b_scale_ptrs)
    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None],
            other=0.0,
        )
        b = tl.load(b_ptrs,cache_modifier="")

        b = tl.interleave(b,b)
        b = b.trans()
        b = (b >> b_shifter) & 0xF
    
        # We accumulate along the K dimension.
        b = (b[:,:] - 8 ).to(a_ptr.type.element_ty)
        accumulator += tl.dot(a, b) 

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]


    accumulator = (accumulator * a_scale[:, None] * b_scale[None,:]).to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def ceil_div(a, b):
    return (a + b - 1) // b


def moe_ep_align_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int, start_expert: int,
        end_expert: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    ops.ep_moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad, start_expert, 
                                 end_expert)
        
    return sorted_ids, expert_ids, num_tokens_post_pad

# 目前仅用来计算gemm1的结果
def invoke_fused_moe_kernel_int8_gemm1(A: torch.Tensor,
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
                                            compute_type: tl.dtype,
                                            use_fp8_w8a8: bool,
                                            use_int8_w4a8: bool,
                                            use_int8_w8a16: bool,
                                            use_int4_w4a16: bool,
                                            per_channel_quant: bool,
                                            block_shape: Optional[List[int]] = None,
                                            use_nn_moe: Optional[bool]=False,
                                            block_m: int=-1
                                            ) -> None:
    assert block_m != -1
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w4a8 is True
    
    M = A.shape[0]
    num_tokens = M * top_k

    triton_config = {   
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }
    triton_config["BLOCK_SIZE_M"] = block_m

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < triton_config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * triton_config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1] if not use_nn_moe else B.shape[2],
            A.shape[1],
            EM,
            num_tokens,
            A.stride(0),#strideam
            A.stride(1),#strideak
            B.stride(0),#stridebe
            B.stride(2) if not use_nn_moe else B.stride(1),#stridebk
            B.stride(1) if not use_nn_moe else B.stride(2),#stridebn
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,

            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,   
            **triton_config, 
        )

# 用于 gemm2 的正确性测试函数
def invoke_fused_moe_kernel_int8_gemm2_valid(A: torch.Tensor,
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
                                            compute_type: tl.dtype,
                                            use_fp8_w8a8: bool,
                                            use_int8_w4a8: bool,
                                            use_int8_w8a16: bool,
                                            use_int4_w4a16: bool,
                                            per_channel_quant: bool,
                                            block_shape: Optional[List[int]] = None,
                                            use_nn_moe: Optional[bool]=False,
                                            block_m: int=-1,
                                            delta: int=-1,
                                            mode_idx: int=-1,
                                            num_iters: int=10
                                            ) -> None:
    assert block_m != -1
    assert mode_idx != -1
    assert delta != -1

    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w4a8 is True
    
    M = A.shape[0]
    num_tokens = M * top_k

    triton_config = {   
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }
    triton_config["BLOCK_SIZE_M"] = block_m

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < triton_config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * triton_config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1] if not use_nn_moe else B.shape[2],
            A.shape[1],
            EM,
            num_tokens,
            A.stride(0),#strideam
            A.stride(1),#strideak
            B.stride(0),#stridebe
            B.stride(2) if not use_nn_moe else B.stride(1),#stridebk
            B.stride(1) if not use_nn_moe else B.stride(2),#stridebn
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,

            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,   
            **triton_config, 
        )
    
    out_cuda = torch.zeros_like(C)
    cuda_config = {
                    'MODE': mode_idx,
                    'DELTA': delta
                }
    moe_gemm_w4a8(A, 
                    B, 
                    out_cuda,
                    A_scale, 
                    B_scale,
                    topk_weights if mul_routed_weight else None,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    top_k,
                    cuda_config)
    torch.cuda.synchronize()

    is_true = (torch.allclose(C, out_cuda, rtol=1e-2, atol=1e-2))
    print("         测试结果： ", is_true)
    return is_true

# 用于 gemm2 性能测试的函数
def invoke_fused_moe_kernel_int8_gemm2_prof(A: torch.Tensor,
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
                                            compute_type: tl.dtype,
                                            use_fp8_w8a8: bool,
                                            use_int8_w4a8: bool,
                                            use_int8_w8a16: bool,
                                            use_int4_w4a16: bool,
                                            per_channel_quant: bool,
                                            block_shape: Optional[List[int]] = None,
                                            use_nn_moe: Optional[bool]=False,
                                            block_m: int=-1,
                                            mode_idx: int=-1,
                                            delta: int=-1,
                                            num_iters: int=10
                                            ) -> None:
    assert block_m != -1
    assert mode_idx != -1
    assert delta != -1

    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w4a8 is True
    
    out_cuda = torch.zeros_like(C)

    cuda_config = {
                    'MODE': mode_idx,
                    'DELTA': delta
                }

    # 预热一次
    for _ in range(3):
        moe_gemm_w4a8(
                        A, 
                        B, 
                        out_cuda,
                        A_scale, 
                        B_scale,
                        topk_weights if mul_routed_weight else None,
                        sorted_token_ids, 
                        expert_ids,
                        num_tokens_post_padded, 
                        top_k,
                        cuda_config
                        )
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda:   moe_gemm_w4a8(
                                                                        A, 
                                                                        B, 
                                                                        out_cuda,
                                                                        A_scale, 
                                                                        B_scale,
                                                                        topk_weights if mul_routed_weight else None,
                                                                        sorted_token_ids, 
                                                                        expert_ids,
                                                                        num_tokens_post_padded, 
                                                                        top_k,
                                                                        cuda_config
                                                                        ), quantiles=quantiles)


    return ms * 1000

# gemm2 正确性验证函数入口
def gemm2_kernel_valid(
                        hidden_states: torch.Tensor,
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
                        mode_idx: int=-1,
                        block_m: int=-1,
                        num_iters: int=10, # 重复多少次进行性能测试
                        delta: int=-1, 
                    ):
    # Check constraints. 
    # mode_idx block_m 必须被指定
    assert delta != -1
    assert mode_idx != -1
    assert block_m != -1

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
            block_shape=None)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, block_m,
                                    global_num_experts, expert_map))

        # 
        invoke_fused_moe_kernel_int8_gemm1(qcurr_hidden_states,
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
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w4a8=use_int8_w4a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                block_m=block_m
                                                )

        if activation == "silu":
            # d = intermediate_cache1.shape[-1] // 2
            # intermediate_cache2 = F.silu(intermediate_cache1[..., :d]) * intermediate_cache1[..., d:]
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
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

        # gemm2 进行精度校验
        is_true = invoke_fused_moe_kernel_int8_gemm2_valid(qintermediate_cache2,
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
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w4a8=use_int8_w4a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                block_m=block_m,
                                                mode_idx=mode_idx,
                                                num_iters=num_iters,
                                                delta=delta
                                                )


        return is_true

# gemm2 性能测试函数入口
def gemm2_kernel_prof(
                        hidden_states: torch.Tensor,
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
                        mode_idx: int = -1,
                        block_m: int = -1,
                        delta: int=-1, 
                    ):
    # Check constraints. 
    # mode_idx block_m 必须被指定
    assert mode_idx != -1
    assert block_m != -1

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
            block_shape=None)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, block_m,
                                    global_num_experts, expert_map))

        # 
        invoke_fused_moe_kernel_int8_gemm1(qcurr_hidden_states,
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
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w4a8=use_int8_w4a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                block_m=block_m
                                                )

        if activation == "silu":
            # d = intermediate_cache1.shape[-1] // 2
            # intermediate_cache2 = F.silu(intermediate_cache1[..., :d]) * intermediate_cache1[..., d:]
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
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

        # gemm2 进行性能测试
        time_us = invoke_fused_moe_kernel_int8_gemm2_prof(qintermediate_cache2,
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
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w4a8=use_int8_w4a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                block_m=block_m,
                                                mode_idx=mode_idx,
                                                delta=delta
                                                )


        return time_us

