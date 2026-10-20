# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
#import vllm.envs as envs
# from vllm import _custom_ops as ops
# from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
#     moe_align_block_size)
from ...vllm_compat.moe_align_block_size import moe_align_block_size
import torch
import triton
import triton.language as tl
from lightop import envs as lsenvs
# from vllm.utils import W8a8GetCacheJSON

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
from ..gemm.int8_utils import per_token_quant_int8
# from ...vllm_compat.fused_moe_cache import get_moe_cache
if use_lightop:
    from lightop import get_moe_cuda_marlin_config_w4a8, moe_gemm_marlin_w4a8, fuse_silu_mul_quant
    from .... import op

# @torch.compile
# def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
#     torch.sum(x, dim=1, out=out)
#     out.mul_(routed_scaling_factor)


# @triton.jit
# def _moe_sum_reduce_kernel(
#         input_ptr,
#         input_stride_0,
#         input_stride_1,
#         input_stride_2,
#         output_ptr,
#         output_stride_0,
#         output_stride_1,
#         token_num: int,
#         topk_num: int,
#         hidden_dim: int,
#         routed_scaling_factor: tl.constexpr,
#         BLOCK_M: tl.constexpr,
#         BLOCK_DIM: tl.constexpr,
#         NUM_STAGE: tl.constexpr,
# ):
#     input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
#     input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
#     output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

#     token_block_id = tl.program_id(0)
#     dim_block_id = tl.program_id(1)

#     token_start = token_block_id * BLOCK_M
#     token_end = min((token_block_id + 1) * BLOCK_M, token_num)

#     dim_start = dim_block_id * BLOCK_DIM
#     dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

#     offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

#     for token_index in range(token_start, token_end):
#         accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
#         input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
#         for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
#             tmp = tl.load(
#                 input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
#             )
#             accumulator += tmp
#         accumulator = accumulator * routed_scaling_factor
#         store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
#         tl.store(
#             store_t_ptr,
#             accumulator.to(input_ptr.dtype.element_ty),
#             mask=offs_dim < dim_end,
#         )


# def moe_sum_reduce_triton(
#         input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
# ):
#     assert input.is_contiguous()
#     assert output.is_contiguous()

#     token_num, topk_num, hidden_dim = input.shape
#     assert output.shape[0] == token_num and output.shape[1] == hidden_dim

#     if token_num <= 32:
#         BLOCK_M = 1
#         BLOCK_DIM = 512
#         NUM_STAGE = 2
#         num_warps = 4

#     elif token_num <= 128:
#         BLOCK_M = 1
#         BLOCK_DIM = 1024
#         NUM_STAGE = 0
#         num_warps = 2

#     elif token_num <= 4096:
#         BLOCK_M = 1
#         BLOCK_DIM = 2048
#         NUM_STAGE = 0
#         num_warps = 2
#     else:
#         BLOCK_M = 1
#         BLOCK_DIM = 2048
#         NUM_STAGE = 2
#         num_warps = 8

#     grid = (
#         triton.cdiv(token_num, BLOCK_M),
#         triton.cdiv(hidden_dim, BLOCK_DIM),
#     )

#     _moe_sum_reduce_kernel[grid](
#         input,
#         *input.stride(),
#         output,
#         *output.stride(),
#         token_num=token_num,
#         topk_num=topk_num,
#         hidden_dim=hidden_dim,
#         routed_scaling_factor=routed_scaling_factor,
#         BLOCK_M=BLOCK_M,
#         BLOCK_DIM=BLOCK_DIM,
#         NUM_STAGE=NUM_STAGE,
#         num_warps=num_warps,
#     )
#     return

# def moe_reduce_dispatch(
#     intermediate_cache3: torch.Tensor,
#     out_hidden_states: torch.Tensor,
#     begin_chunk_idx: int,
#     end_chunk_idx: int,
#     routed_scaling_factor: float,
#     shared_output: Optional[torch.Tensor] = None,
# ):
#     inter_cache_view = intermediate_cache3.view(*intermediate_cache3.shape)
#     n = intermediate_cache3.shape[0]

#     # 根据 n 大小选择不同的 reduce 实现
#     if 1 <= n <= 4:
#         moe_sum_reduce_torch_compile(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
#     elif 4 < n <= 1024:
#         moe_sum_reduce_triton(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
#     elif 1024 < n <= 32768:
#         ops.moe_sum_opt1(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
#     else:
#         ops.moe_sum(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
        
#     # 根据 shared_output 是否存在决定怎么更新
#     if shared_output is not None:
#         out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor).add_(shared_output[begin_chunk_idx:end_chunk_idx])
#     else:
#         out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

def moe_align_block_size_lightop(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ep_size: int = 8,
    num_token: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    local_num_experts = num_experts // ep_size
  
    if num_token:
        if num_token < block_size:
            max_num_tokens_padded = min(topk_ids.numel() * block_size, topk_ids.numel() + local_num_experts * (block_size - 1))
        else:
            max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        sorted_ids = torch.full((max_num_tokens_padded,), fill_value=topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    else:
        max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=topk_ids.device)
        sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    if expert_map is not None:
        expert_ids = torch.zeros((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    else:
        expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.zeros((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    op.moe_align_block_size(topk_ids, 
                            num_experts, 
                            block_size, 
                            sorted_ids,
                            expert_ids, 
                            num_tokens_post_pad,
                            expert_map,
                            expert_mask,
                            num_local_tokens,
                            expert_map is not None)
    return sorted_ids, expert_ids, num_tokens_post_pad



def fused_experts_impl_w4a8_marlin(hidden_states: torch.Tensor,
                                   w1: torch.Tensor,
                                   w2: torch.Tensor,
                                   topk_weights: torch.Tensor,
                                   topk_ids: torch.Tensor,
                                   workspace: torch.Tensor,
                                   global_reduce_buffer: torch.Tensor,
                                   inplace: bool = False,
                                   activation: str = "silu",
                                   apply_router_weight_on_input: bool = False,
                                   global_num_experts: int = -1,
                                   moe_cache_getter: Optional[Callable] = None,
                                   expert_map: Optional[torch.Tensor] = None,
                                   w1_scale: Optional[torch.Tensor] = None,
                                   w2_scale: Optional[torch.Tensor] = None,
                                   a1_scale: Optional[torch.Tensor] = None,
                                   a2_scale: Optional[torch.Tensor] = None,
                                   routed_scaling_factor: Optional[float] = 1.0,
                                   shared_output: Optional[torch.Tensor] = None,
                                   num_local_tokens: Optional[torch.Tensor] = None,
                                   expect_m: Optional[int] = -1, **_
                                   ):
    assert hidden_states.shape[1] == w1.shape[1] * 32, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // 4, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前只支持 bf16
    assert hidden_states.dtype in [torch.int8, torch.bfloat16]
    compute_type = torch.bfloat16
    assert use_lightop, "Please set LMSLIM_USE_LIGHTOP=1 to support w4a8_marlin"

    num_tokens, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 32 # 256
    N2 = N *2# 256
    if global_num_experts == -1:
        global_num_experts = E

    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = lsenvs.LMSLIM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    if moe_cache_getter is not None:
        cache13 = moe_cache_getter(top_k_num, 2 * N, K, device=hidden_states.device, dtype=compute_type)
    else:
        cache13 = torch.empty(
            (M * top_k_num * max(2 * N, K), ),
            device=hidden_states.device,
            dtype=compute_type,
        )
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1

    intermediate_cache1 = cache13[:M * top_k_num * N2].view(-1, N2)
    intermediate_cache3 = cache13[:M * top_k_num * K]

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N),
                                      device=hidden_states.device,
                                      dtype=compute_type)
    is_ep = expert_map is not None
    expert_mask=None
    ep_size=None
    if is_ep:
        expert_mask = torch.zeros((CHUNK_SIZE, top_k_num), dtype=torch.bool,  device=hidden_states.device, requires_grad=False)
        ep_size = global_num_experts // E

    if a1_scale is not None:
        out_hidden_states = torch.empty_like(hidden_states, dtype=compute_type)
    else:
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
        
        
        bs = tokens_in_chunk
        if num_local_tokens is not None and expect_m != -1:
            bs = expect_m

        intermediate_cache3 = intermediate_cache3.view(-1, K)
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk * top_k_num]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk * top_k_num]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk * top_k_num]

        config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w4a8(
            E,
            bs,
            N2,
            K,
            K,
            N,
            top_k_num,
            device_name,
            num_cus,
            tl.bfloat16,
            is_ep=is_ep,
            ep_size=ep_size)

        assert status, f'lightop unsupport this size E:{E}, N:{N}, K:{K}'
        block_size_m = config_marlin_0["BLOCK_SIZE_M"]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        if num_local_tokens is None and expert_mask is None:
            sorted_token_ids, expert_ids, num_tokens_post_padded =(
                moe_align_block_size(curr_topk_ids, block_size_m,
                                    global_num_experts, expert_map=expert_map))
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size_lightop(curr_topk_ids, block_size_m, global_num_experts, 
                                            expert_map = expert_map, 
                                            expert_mask = expert_mask[begin_chunk_idx:end_chunk_idx] if is_ep else None,
                                            num_local_tokens = num_local_tokens,
                                            ep_size=ep_size))

        if a1_scale is None:
            qcurr_hidden_states, qa1_scale = per_token_quant_int8(curr_hidden_states)
        else:
            qcurr_hidden_states = curr_hidden_states
            qa1_scale = a1_scale

        moe_gemm_marlin_w4a8(qcurr_hidden_states,
                         w1,
                         intermediate_cache1,
                         qa1_scale,
                         w1_scale,
                         curr_topk_weights if apply_router_weight_on_input else None,
                         sorted_token_ids,
                         expert_ids,
                         num_tokens_post_padded,
                         top_k_num,
                         workspace,
                         global_reduce_buffer,
                         config_marlin_0,
                         is_ep=is_ep
                         )

        if num_local_tokens is not None:
            qintermediate_cache2, qa2_scale = fuse_silu_mul_quant(intermediate_cache1, num_local_tokens, top_k_num)
        else:
            if activation == "silu":
                qintermediate_cache2, qa2_scale = fuse_silu_mul_quant(intermediate_cache1)
            elif activation == "gelu":
                torch.ops._C.gelu_and_mul(intermediate_cache2,
                                          intermediate_cache1.view(-1, N))
                qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)
            else:
                raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        intermediate_cache3 = moe_gemm_marlin_w4a8(qintermediate_cache2,
                                           w2,
                                           intermediate_cache3,
                                           qa2_scale,
                                           w2_scale,
                                           curr_topk_weights if not apply_router_weight_on_input else None,
                                           sorted_token_ids,
                                           expert_ids,
                                           num_tokens_post_padded,
                                           1,
                                           workspace,
                                           global_reduce_buffer,
                                           config_marlin_1,
                                           is_ep=is_ep
                                           ).view(-1, top_k_num, K)

        if is_ep:
            op.moe_sum(input=intermediate_cache3,
                       output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                       bias = None if shared_output is None else shared_output[begin_chunk_idx:end_chunk_idx],
                       expert_mask = expert_mask[:tokens_in_chunk],
                       num_local_tokens=num_local_tokens,
                       factor=routed_scaling_factor,
                       )
        elif use_lightop:
            if shared_output is not None:
                bias = shared_output[begin_chunk_idx:end_chunk_idx]
            else:
                bias = None
            op.moe_sum(input=intermediate_cache3.view(*intermediate_cache3.shape),
                       output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                       bias=bias,
                       expert_mask=None,
                       num_local_tokens=None,
                       factor=routed_scaling_factor)
        # else:
        #     moe_reduce_dispatch(
        #         intermediate_cache3,
        #         out_hidden_states,
        #         begin_chunk_idx,
        #         end_chunk_idx,
        #         1.0,
        #         None,
        #     )

    return out_hidden_states
