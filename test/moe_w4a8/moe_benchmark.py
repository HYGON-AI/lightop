# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
import pandas as pd
import functools
import json
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import torch
import triton
import triton.language as tl
from lightop import envs as lsenvs
from common.w8a8_cache_json import W8a8GetCacheJSON
import torch.nn.functional as F
import ast  

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME

if use_lightop:
    from lightop.moe import moe_gemm_w8a8, get_moe_cuda_config
    from moe_w8a8_prefill_gemm import backend
    from grouped_gemm import moe_gemm_decode_prof
from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)

import argparse

cache13 = None

blockM_configs = {
    0: 16,   # 2warp
    1: 32,
    2: 64,
    3: 16,   # 4warp
    4: 32,
    5: 64,
    6: 16,   # 4warp
    7: 32,
    8: 64,
    9: 32,   # 8warp
    10: 64,
    11: 32,  # 8warp
    12: 64,
    13: 64,  # 16warp
    14: 16,  # 2warp
    15: 32,
    16: 64,
    17: 16,  # 2warp
    18: 32,
    19: 64,
    20: 16,  # 2warp
    21: 32,
    22: 64
}

def get_config_mode(num_token):
    mode1 = 0
    mode2 = 0
    token_mode_config = {
        1: [0, 0],
        2: [0, 0],
        4: [3, 0],
        8: [0, 0],
        16: [0, 0],
        32: [0, 0],
        64: [3, 0],
        128: [0, 3],
        4096: [4, 4],
        6144: [11, 4],
        8192: [11, 4],
        16384: [11, 4],
        32768: [11, 4],
    }
    # 找到最接近的token数
    min_diff = float('inf')
    closest_token = None
    for token in token_mode_config.keys():
        diff = abs(token - num_token)
        if diff < min_diff:
            min_diff = diff
            closest_token = token
    # 获取对应的配置
    if closest_token is not None:
        mode1, mode2 = token_mode_config[closest_token]
    assert blockM_configs[mode1] == blockM_configs[mode2], "mode0 and mode1 blockm must be equal!"
    return mode1, mode2
def get_w8a8moe_json(E:int,m:int,n1:int,n2:int,k:int):
    #接续json
    moe_dict= W8a8GetCacheJSON().triton_moejson_dict if W8a8GetCacheJSON().triton_moejson_dict else {}
    if len(moe_dict)==0:
        configs=None
    elif f"1_{n1}_{n2}_{k}" in moe_dict:
        if m<=16:
            m_=m
        elif m<=32:
            m_= 16 #取值到最近的4的倍数
        elif m<=64:
            m_=256
        elif m<1024: #1024
            m_=1024
        elif m<8192:
            m_=8192
        else:
            m_=16384
        configs=moe_dict[f"{m_}_{n1}_{n2}_{k}"]
        config1=configs['config1']
        config2=configs['config2']
    else:
        configs =None
         
    if configs is None:
        if m<E:
            config1 = config2={
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                }
        else:
            config1=config2 = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }
        print("config1:",config1)
    return config1,config2


def moe_kernel_prepare_input(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (per_channel_quant
                    ), "int8 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            # assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            # assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    return A, A_scale

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
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
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
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m
    if GROUP_SIZE_M ==1:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

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
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[List[int]] = None,
                            use_nn_moe: Optional[bool]=False) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w8a8 is True
    
    M = A.shape[0]
    num_tokens = M * top_k

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
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
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2) if not use_nn_moe else B.stride(1),
            B.stride(1) if not use_nn_moe else B.stride(2),
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
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            **config,
        )

def fused_experts_impl_int8(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
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
                       use_nn_moe: Optional[bool] = False):
    # Check constraints.
    assert use_int8_w8a8 is True
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    use_nn_moe=False

    num_tokens, _ = hidden_states.shape
    E, N, K = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    global cache13  # 声明使用全局变量
    if cache13 is None:
        cache13 = torch.empty(CHUNK_SIZE * 8 * 7168, device=hidden_states.device, dtype=hidden_states.dtype)
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

        #get config if quant method is block-int8
        mode0=None
        mode1=None
        config=None
        config1=None
        config2=None
        status =False
        num_cus=0
        m=curr_hidden_states.shape[0]
        if use_lightop and per_channel_quant: #channel-wise cuda
            num_cus=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            config1,config2,status=get_moe_cuda_config(E,m,w1.shape[1],w1.shape[2],w2.shape[1],w2.shape[2],top_k_num,device_name,num_cus,compute_type)      
            config=config1
            if status is False:
                config1,config2=get_w8a8moe_json(m,E,N,w2.shape[1],w2.shape[2])
                config=config1

        elif use_lightop and m >= 4096 and device_name=='BW200': # block-wise cuda
            mode0, mode1 = get_config_mode(tokens_in_chunk)
            config = {"BLOCK_SIZE_M": blockM_configs[mode0]}
        elif use_lightop and  device_name=='K100_AI_120' and m <= 64: # block-wise cuda
            config = {"BLOCK_SIZE_M": 16}
        else:
            config1,config2=get_w8a8moe_json(m,E,N,w2.shape[1],w2.shape[2])
            config=config1

        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)
        
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))

        
        if use_lightop and per_channel_quant and status:

            moe_gemm_w8a8(qcurr_hidden_states,
                          w1,
                          intermediate_cache1,
                          qa1_scale,
                          w1_scale,
                          curr_topk_weights if apply_router_weight_on_input else None,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          top_k_num,
                          config1)
        elif use_lightop and block_shape is not None and device_name=='K100_AI_120' and m <= 64:
            moe_gemm_decode_prof.CUDA_W8A8_MOE1(qcurr_hidden_states,
                                                w1,
                                                intermediate_cache1,
                                                qa1_scale,
                                                w1_scale,
                                                sorted_token_ids,
                                                expert_ids,
                                                curr_topk_weights if apply_router_weight_on_input else None,
                                                num_tokens_post_padded)
        elif use_lightop and block_shape is not None and  device_name=='BW200' and m >=4096:
            backend.moe_w8a8_prefill_gemm(qcurr_hidden_states,
                                          w1,
                                          intermediate_cache1,
                                          qa1_scale,
                                          w1_scale,
                                          curr_topk_weights if apply_router_weight_on_input else None,
                                          sorted_token_ids,
                                          expert_ids,
                                          num_tokens_post_padded,
                                          top_k_num,
                                          mode0)
        else:
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
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)

        if activation == "silu":
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
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        
        if use_lightop and per_channel_quant and status:
            moe_gemm_w8a8(qintermediate_cache2,
                          w2,
                          intermediate_cache3,
                          qa2_scale,
                          w2_scale,
                          curr_topk_weights if not apply_router_weight_on_input else None,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          1,
                          config2)
        elif use_lightop and block_shape is not None and device_name=='K100_AI_120' and m <= 64:
            moe_gemm_decode_prof.CUDA_W8A8_MOE2(qintermediate_cache2,
                                                w2,
                                                intermediate_cache3,
                                                qa2_scale,
                                                w2_scale,
                                                sorted_token_ids,
                                                expert_ids,
                                                curr_topk_weights if not apply_router_weight_on_input else None,
                                                num_tokens_post_padded)
        elif use_lightop and block_shape is not None and device_name=='BW200'and m >=4096:
            #print("block*****")
            backend.moe_w8a8_prefill_gemm(qintermediate_cache2,
                                          w2,
                                          intermediate_cache3,
                                          qa2_scale,
                                          w2_scale,
                                          curr_topk_weights if not apply_router_weight_on_input else None,
                                          sorted_token_ids,
                                          expert_ids,
                                          num_tokens_post_padded,
                                          1,
                                          mode1)

        else:
            config=config2
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
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
    return intermediate_cache3


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

# 生成 [m, topk] 的整数张量，每行元素不重复
def generate_unique_int_tensor(m, topk, low=0, high=256):
    # 检查 topk 是否超过范围
    if topk > (high - low):
        raise ValueError(f"topk ({topk}) 不能超过范围 ({high - low})")
    
    # 生成张量
    tensor = torch.zeros((m, topk), dtype=torch.int)
    
    for i in range(m):
        # 对每一行生成不重复的随机整数
        row = torch.randperm(high - low)[:topk] + low
        tensor[i] = row
    
    return tensor


def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             n2:int,
                             k2:int,
                             topk:int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             device: str = "cuda",
                             block_size: Optional[list] = None,
                             best_config:Optional[list] = None,
                             per_channel_quant:Optional[bool] = False,
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    weight13 = to_int8(torch.randn((E,n ,k), device=device) * 5)
    weight2 = to_int8(torch.randn((E,n2 ,k2), device=device) * 5)

    if per_channel_quant is False:
        weight13_scale = (torch.randn((E,math.ceil(n/block_size[0]), math.ceil(k/block_size[1])), device=device,
                            dtype=torch.float32))
        weight2_scale = (torch.randn((E,math.ceil(n2/block_size[0]), math.ceil(k2/block_size[1])), device=device,
                            dtype=torch.float32))    
    else:
        weight13_scale = (torch.randn((E,n,1), device=device,
                            dtype=torch.float32))
        weight2_scale = (torch.randn((E,n2,1), device=device,
                            dtype=torch.float32))        
    
    print("input.dtype:",input.dtype)

    #q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    
    #topk_weights [m,topk]
    tensor = torch.rand((m, topk),device=device)
    topk_weights = F.softmax(tensor, dim=1)
    #topk_ids [m,topk]
    topk_ids=generate_unique_int_tensor(m=m, topk=topk,low=0,high=E).cuda()
    
    return input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids



def main():
    
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='这是一个示例脚本的描述')
    
    # 添加参数
    parser.add_argument('--E', type=int, default=256, help='E 参数')
    parser.add_argument('--n1-list', type=str, required=True, 
                       help='N1 参数列表，如 "[1536,768,384,384,192]"')
    parser.add_argument('--n2-list', type=str, required=True,
                       help='N2 参数列表，如 "[2048,2048,2048,4096,4096]"')
    parser.add_argument('--k-list', type=str, required=True,
                       help='K 参数列表，如 "[768,374,192,192,96]"')
    parser.add_argument('--topk', type=int, default=8, 
                       help='topk 参数')
    parser.add_argument('--out-dtype', type=torch.dtype, default=torch.bfloat16,
                       help='out-dtype')
    parser.add_argument('--use-channel-wise',action='store_true', default=False,
                       help='use-channel-wise')
    parser.add_argument('--block_size', type=list, default=[128,128],
                       help='block_size')
    
    # 解析参数
    args = parser.parse_args()
 
    def parse_list(list_str):
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析列表参数: {list_str}。错误: {str(e)}")
    
    
    # m_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    m_list=[8192]
    E=args.E 
    block_size=[128,128]
    out_dtype=args.out_dtype
    
    n_list_1=parse_list(args.n1_list) #[256,512]
    k_list_1=parse_list(args.n2_list) #[7168,7168]
    
    n_list_2=parse_list(args.n2_list) #[7168,7168]
    k_list_2=parse_list(args.k_list) # [128,256]
    
    topk=args.topk
    best_config = None

    
    per_channel_quant=args.use_channel_wise
    
    #初始化类
    tritonsingleton= W8a8GetCacheJSON()
    
    triton_json_dir=tritonsingleton.triton_json_dir
    print("triton_json_dir:",triton_json_dir)
    
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=lsenvs.LMSLIM_GPU_NAME
              
    moe_cost_gpu=[]
    _m=[]
    _n1=[]
    _n2=[]
    _k=[]  
    
    for i in range(0,len(k_list_1),1):
        if per_channel_quant:
            file_name = f"{triton_json_dir}/MOE_W8A8INT8_E={E}_N1={n_list_1[i]}_N2={n_list_2[i]}_K={k_list_2[i]}_TOPK{topk}_{device_name}.json"
            block_size=None
        else:
            file_name = f"{triton_json_dir}/MOE_BLOCKINT8[{block_size[0]},{block_size[1]}]_E={E}_N1={n_list_1[i]}_N2={n_list_2[i]}_K={k_list_2[i]}_TOPK{topk}_{device_name}.json"
        
        print("file_name:",file_name)
    
        tritonsingleton.moe_weight_shapes.append({E,n_list_1[i],n_list_2[i],k_list_2[i]})
        tritonsingleton.triton_moejson_dict=tritonsingleton.get_moeint8_triton_cache(file_name,E,n_list_1[i],n_list_2[i],k_list_2[i],topk)
     
    for i in range(0,len(k_list_1),1):

        #稳定性测试：跑1-32k m值
        # for m in range(1,32768,1):
        #     print("稳定性测试 m:{} n1:{} n2{} k:{} ".format(m,n_list_1[i],n_list_2[i],k_list_2[i]))
        #     input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=m,n=n_list_1[i],k=k_list_1[i],n2=n_list_2[i],k2=k_list_2[i],topk=topk,E=E,block_size=block_size,out_dtype=out_dtype,per_channel_quant=per_channel_quant)
        #     output=fused_experts_impl_int8(hidden_states=input,
        #                                 w1=weight13,
        #                                 w2=weight2,
        #                                 topk_weights=topk_weights,
        #                                 topk_ids=topk_ids,
        #                                 inplace=True,
        #                                 use_int8_w8a8=True,
        #                                 w1_scale=weight13_scale,
        #                                 w2_scale=weight2_scale,
        #                                 block_shape=block_size,
        #                                 per_channel_quant=per_channel_quant)

        #读取json文件
        for m in m_list:
            print("m:{} n1:{} n2:{} k:{} ".format(m,n_list_1[i],n_list_2[i],k_list_2[i]))
            input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=m,n=n_list_1[i],k=k_list_1[i],n2=n_list_2[i],k2=k_list_2[i],topk=topk,E=E,block_size=block_size,out_dtype=out_dtype,per_channel_quant=per_channel_quant)
            
            gpu_costtime=triton.testing.do_bench(lambda:fused_experts_impl_int8(hidden_states=input,
                                                                                        w1=weight13,
                                                                                        w2=weight2,
                                                                                        topk_weights=topk_weights,
                                                                                        topk_ids=topk_ids,
                                                                                        inplace=True,
                                                                                        use_int8_w8a8=True,
                                                                                        w1_scale=weight13_scale,
                                                                                        w2_scale=weight2_scale,
                                                                                        block_shape=block_size,
                                                                                        per_channel_quant=per_channel_quant),quantiles=None, return_mode="mean")*1000
            
            print("cost_time:{} us\n".format(gpu_costtime))
            moe_cost_gpu.append(gpu_costtime)           
            _m.append(m)
            _n1.append(n_list_1[i])
            _n2.append(n_list_2[i])
            _k.append(k_list_2[i])
            
    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({'m':_m,'n1':_n1,'n2':_n2,'k':_k,'MOE GPU耗时': moe_cost_gpu})

    # 将 DataFrame 写入 Excel 文件
    if per_channel_quant:
        excel_name="channel-wsie_output.xlsx"
    else:
        excel_name="block-wsie_output.xlsx"
        
    df.to_excel(excel_name, index=False)

    print("表格已保存到 {}文件中。".format(excel_name))           
            
        

if __name__ == "__main__":
    main()
    #qwen系列
    #脚本执行命令： python3 moe_benchmark.py  --E 128 --n1-list '[1536,768,384,384,192]'  --n2-list '[2048,2048,2048,4096,4096]' --k-list '[768,374,192,192,96]'

    #deepseek系列
    #python3 moe_benchmark.py  --E 256 --n1-list '[256]'  --n2-list '[7168]' --k-list '[128]' --use-channel-wise 0
    
    # channel-wise:
    # HIP_VISIBLE_DEVICES=3 python3 moe_w8a8/moe_benchmark.py  --E 256 --n1-list '[256]'  --n2-list '[7168]' --k-list '[128]' --use-channel-wise
    # block-wsie:
    # HIP_VISIBLE_DEVICES=3 python3 moe_w8a8/moe_benchmark.py  --E 256 --n1-list '[256]'  --n2-list '[7168]' --k-list '[128]'