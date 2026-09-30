# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
# from vllm import _custom_ops as ops
import torch
import torch.nn.functional as F
from lightop import envs as lsenvs
use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
from ..gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)

from lightop import get_moe_groupgemm_config_w4a8,moe_groupgemm_w4a8
from lightop import get_moe_groupgemm_marlin_config_w4a8, moe_groupgemm_marlin_w4a8

def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    E,N, K_half = tensor_int8.shape
    tensor_int8 = tensor_int8.to(torch.int8)
    high4 = tensor_int8 & 0XF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((E, N, K_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4.to(torch.int8)
    unpacked[:, :, 1::2] = low4.to(torch.int8)

    return unpacked

def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)
def native_w4a8_groupgemm(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):

    outputs = []
    start_idx = 0
    w13 = unpack_int8_to_int4(weight13) #torch.int32
    w2 = unpack_int8_to_int4(weight2)   #torch.int32
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # 模拟激活量化和反量化
        q_a1, qa1_scale = per_token_quant_int8(hidden_states[start_idx:end_idx])

        current_w13 = w13[i]
        current_w2 = w2[i]
        current_w13_s = w13_scale[i]
        current_w2_s = w2_scale[i]
        inter_out = native_w8a8_per_token_matmul(
            q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype
        )
        # print("inter_out: ", inter_out.shape)
        d = inter_out.shape[-1] // 2
        act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
        # print("act_out: ", act_out.shape)

        act_out_q, act_out_s = per_token_quant_int8(act_out)

        expert_out= native_w8a8_per_token_matmul(
            act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype
        )
        # print("expert_out: ", expert_out.shape)

        outputs.append(expert_out)
        start_idx = end_idx
    if len(outputs) > 0:
        expert_output = torch.cat(outputs, dim=0)
    else:
        assert hidden_states.numel() == 0, f"sorted_tokens: should be empty, but got {hidden_states.shape}"
        expert_output = hidden_states

    return expert_output

def groupgemm_impl_w4a8(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):

    M, K = hidden_states.shape
    if M == 0:
        return hidden_states

    if use_lightop:
        E, N1, K1 = weight13.shape
        _, N2, K2 = weight2.shape
        dtype = hidden_states.dtype
        cache13 = torch.empty(M * max(N1, K),device=hidden_states.device,dtype=dtype)
        intermediate_cache1 = cache13[:M * N1].view(-1, N1)
        intermediate_cache2 = torch.empty(M,N1//2,device=hidden_states.device,dtype=dtype)
        intermediate_cache3 = cache13[:M * K].view(-1, K)

        config1, config2, status = get_moe_groupgemm_marlin_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)

        tensor_problem_size = tokens_per_expert.to(dtype=torch.int32, device=hidden_states.device)
        tensor_expert_offsets = (torch.cumsum(tensor_problem_size, dim=0) - tensor_problem_size).to(dtype=torch.int32)

        q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)
        moe_groupgemm_w4a8(q_a1_all,
                           weight13,
                           intermediate_cache1,
                           qa1_scale_all,
                           w13_scale,
                           tensor_problem_size,
                           tensor_expert_offsets,
                           config1)
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
        qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)

        moe_groupgemm_w4a8(qintermediate_cache2,
                           weight2,
                           intermediate_cache3,
                           qa2_scale,
                           w2_scale,
                           tensor_problem_size,
                           tensor_expert_offsets,
                           config2)
    else:
        return native_w4a8_groupgemm(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert)

    return intermediate_cache3
import sys
def groupgemm_impl_w4a8_marlin(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):

    M, K = hidden_states.shape
    if M == 0:
        return hidden_states

    if use_lightop:

        E = weight13.shape[0]
        N1 = weight13.shape[2] // 4
        K1 = K//2
        N2 = K1*2
        K2 = N1 // 4

        dtype = hidden_states.dtype
        cache13 = torch.empty(M * max(N1, K),device=hidden_states.device,dtype=dtype)
        intermediate_cache1 = cache13[:M * N1].view(-1, N1)
        intermediate_cache2 = torch.empty(M,N1//2,device=hidden_states.device,dtype=dtype)
        intermediate_cache3 = cache13[:M * K].view(-1, K)

        config1, config2, status = get_moe_groupgemm_marlin_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)

        tensor_problem_size = tokens_per_expert.to(dtype=torch.int32, device=hidden_states.device)
        tensor_expert_offsets = (torch.cumsum(tensor_problem_size, dim=0) - tensor_problem_size).to(dtype=torch.int32)

        q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)
        moe_groupgemm_marlin_w4a8(q_a1_all,
                                  weight13,
                                  intermediate_cache1,
                                  qa1_scale_all,
                                  w13_scale,
                                  tensor_problem_size,
                                  tensor_expert_offsets,
                                  config1)
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
        qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)

        moe_groupgemm_marlin_w4a8(qintermediate_cache2,
                                  weight2,
                                  intermediate_cache3,
                                  qa2_scale,
                                  w2_scale,
                                  tensor_problem_size,
                                  tensor_expert_offsets,
                                  config2)
    else:
        raise

    return intermediate_cache3

