# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import pdb
import torch
import lightop

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

from lmslim.layers.gemm.int8_utils import (
   per_token_group_quant_int8, per_token_quant_int8)

from triton_w4a8 import invoke_fused_moe_kernel_w4a8
import triton
import triton.language as tl

from marlin_quant import unpack_int8_to_int4

from lightop import moe_gemm_marlin_w4a8, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

def fused_marlin_moe_valid(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1_triton: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2_triton: torch.Tensor, # 256, 256, 7168
                     w1_marlin: torch.Tensor,
                     w2_marlin: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     inplace: bool = False,
                     ) -> torch.Tensor:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1_marlin.shape[1] * 32, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2_marlin.shape[2] // 4, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1_marlin.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2_marlin.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前只支持 bf16
    assert hidden_states.dtype in [torch.bfloat16]

    M, K = hidden_states.shape
    E = w1_marlin.shape[0]
    N = w2_marlin.shape[1] * 32 # 256
    topk = topk_ids.shape[1] 


    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w4a8(E, M, 2 * N, K, K, N, topk, device_name, num_cus, tl.bfloat16)
    # debug
    # config_marlin_0["BLOCK_SIZE_M"] = 128
    # config_marlin_0["MODE"] = 1000
    # config_marlin_1["BLOCK_SIZE_M"] = 128
    # config_marlin_1["MODE"] = 1000

    print("         config_marlin_0: ", config_marlin_0)
    print("         config_marlin_1: ", config_marlin_1)

    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)
    # triton 算子始终使用 block_m=16 测试
    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts)

    # 建立 归约使用的 信号量 Tensor 当前最大开到 sms * 6 = 480 
    sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    workspace = torch.zeros(500, dtype=torch.int, device=hidden_states.device, requires_grad=False)
    # 建立用于 归约使用的 缓冲区 blokc数： sms * 6
    global_reduce_buffer = torch.zeros(sms * 6 * block_size_m * 512, dtype=torch.int, device=hidden_states.device, requires_grad=False)

    # [M*topk, N] [32*8, 256]
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K), ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N) # [M * topk, 2 * N]  [32*8, 512]
    intermediate_cache3 = intermediate_cache13[:M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)  # [M * topk, K]  [32*8, 7168]

    # 对A矩阵进行量化
    a_1, a_1_scale = per_token_quant_int8(hidden_states)
    # =========================== gemm1 marlin 计算 ===============================
    
    # config_marlin = {
    #                     "BLOCK_SIZE_M": block_size_m,
    #                     "thread_n": thread_n,
    #                     "thread_k": thread_k,
    #                     "blocks_per_cu": blocks_per_cu,
    #                     'MODE': 500,
    #                     'DELTA': 1
    #                 }
    '''
    print("a_1:", a_1.shape)
    print("w1_marlin:", w1_marlin.shape, w1_marlin.dtype, w1_marlin.stride())
    print("intermediate_cache1:", intermediate_cache1.shape)
    print("a_1_scale:", a_1_scale.shape)
    print("w1_scale:", w1_scale.shape)
    print("topk_weights:", topk_weights.shape)
    align = block_size_m
    print("sorted_token_ids:", sorted_token_ids[0:align])
    print("sorted_token_ids // topk:", sorted_token_ids[0:align] // topk)
    print("sorted_token_ids % topk:", sorted_token_ids[0:align] % topk)
    print("expert_ids:", expert_ids[:10])
    print("num_tokens_post_padded:", num_tokens_post_padded)
    print("topk:", topk)
    print("config_marlin_0:", config_marlin_0)
    print("topk_ids:", topk_ids.shape)
    '''
    moe_gemm_marlin_w4a8(a_1, w1_marlin, intermediate_cache1, a_1_scale, w1_scale, None, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, topk, workspace, global_reduce_buffer, config_marlin_0)
    # =========================== gemm1 triton 计算 ===============================
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}
    torch.cuda.synchronize()
    invoke_fused_moe_kernel_w4a8(
                                a_1,
                                w1_triton,
                                triton_gemm1_output,
                                a_1_scale,
                                w1_scale,
                                None, # w2_zero
                                topk_weights,
                                topk_ids,
                                sorted_token_ids_triton,
                                expert_ids_triton,
                                num_tokens_post_padded_triton,
                                False,  # 是否乘 top_weight
                                topk, # top_k
                                config,
                                compute_type=tl.bfloat16,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=True,
                                block_shape=None
                                )

    triton_gemm1_output = triton_gemm1_output.reshape((-1, 2 * N))
    # debug
    '''
    w1_int8_list = []
    for i in range(w1_triton.shape[0]):
        w1_int8 = unpack_int8_to_int4(w1_triton[i])
        w1_int8_list.append(w1_int8)
    w1_int8 = torch.stack(w1_int8_list, dim=0).to(torch.uint8) << 4
    w1_int8 = w1_int8.to(torch.int8)
    begin, end, expert_id = 0, 10, 0
    res = torch.matmul(a_1[sorted_token_ids[begin:end] // topk].cpu().to(torch.int32), w1_int8[expert_id, :, :].t().cpu().to(torch.int32))
    res0 = torch.matmul(a_1[sorted_token_ids[begin:end] // topk].cpu().to(torch.int32)[:, :128], w1_int8[expert_id, :, :].t().cpu().to(torch.int32)[:128, :])
    out = res.to(a_1.device) * a_1_scale[sorted_token_ids[begin:end] // topk] * w1_scale[expert_id].t()
    out_bf16 =  out.to(torch.bfloat16)
    torch.nonzero(intermediate_cache1 != triton_gemm1_output)
    torch.nonzero(intermediate_cache1[1] != triton_gemm1_output[1])
    torch.nonzero(intermediate_cache1[sorted_token_ids[begin:end]] != triton_gemm1_output[sorted_token_ids[begin:end]])
    # torch.nonzero(out_bf16 != intermediate_cache1)
    torch.nonzero(sorted_token_ids == 1)
    # pdb.set_trace()
    '''

    torch.testing.assert_close(intermediate_cache1, triton_gemm1_output, atol=1e-3, rtol=1e-3)
    print("         gemm1 success!!!!")
    torch.cuda.synchronize()
    # return
    # 激活函数
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)


    # 准备gemm2 需要的 A和A-scale 
    # [M*topk, N] [32*8, 256]
    a_2, a_2_scale = per_token_quant_int8(intermediate_cache2)
    torch.cuda.synchronize()
    '''
    print("a_2:", a_2.shape)
    print("w2_marlin:", w2_marlin.shape, w2_marlin.dtype, w2_marlin.stride())
    print("intermediate_cache3:", intermediate_cache3.shape)
    print("a_2_scale:", a_2_scale.shape)
    print("w2_scale:", w2_scale.shape)
    print("topk_weights:", topk_weights.shape)
    align = block_size_m
    print("sorted_token_ids:", sorted_token_ids[0:align])
    print("sorted_token_ids // topk:", sorted_token_ids[0:align] // topk)
    print("sorted_token_ids % topk:", sorted_token_ids[0:align] % topk)
    print("expert_ids:", expert_ids[:10])
    print("num_tokens_post_padded:", num_tokens_post_padded)
    print("topk:", topk)
    print("config_marlin_1:", config_marlin_1)
    print("topk_ids:", topk_ids.shape)
    '''
    moe_gemm_marlin_w4a8(a_2, w2_marlin, intermediate_cache3, a_2_scale, w2_scale, topk_weights, 
                         sorted_token_ids, expert_ids, num_tokens_post_padded, 1, workspace, global_reduce_buffer, config_marlin_1)
    # =========================== gemm2 triton 计算 ===============================
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))

    invoke_fused_moe_kernel_w4a8(a_2, w2_triton, triton_gemm2_output, a_2_scale, w2_scale,
                                None, # w2_zero
                                topk_weights,
                                topk_ids,
                                sorted_token_ids_triton,
                                expert_ids_triton,
                                num_tokens_post_padded_triton,
                                True,  # 是否乘 top_weight
                                1, # top_k
                                config,
                                compute_type=tl.bfloat16,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=True,
                                block_shape=None
                                )

    triton_gemm2_output = triton_gemm2_output.reshape((-1, K))
    torch.nonzero(intermediate_cache3 != triton_gemm2_output)
    
    # pdb.set_trace()
    torch.testing.assert_close(intermediate_cache3, triton_gemm2_output, atol=1e-2, rtol=1e-2)
    print("        gemm2 success!!!!")
    return (num_tokens_post_padded.item() * 7168 * 512 * 2),(num_tokens_post_padded.item()  * 256 * 7168 * 2), \
    (num_tokens_post_padded.item() * 7168 + (256 if (num_tokens_post_padded.item() / block_size_m) > 256 else (num_tokens_post_padded.item() / block_size_m)) * 512 * 3584  + M * 8 * 512 * 2 + num_tokens_post_padded.item() * 4 + (256 if (num_tokens_post_padded.item() / block_size_m) > 256 else (num_tokens_post_padded.item() / block_size_m)) * 512 * 4)/1024/1024/1024, \
    (num_tokens_post_padded.item() * 256 + (256 if (num_tokens_post_padded.item() / block_size_m) > 256 else (num_tokens_post_padded.item() / block_size_m)) * 128 * 7168 + M * 7168 * 2 + num_tokens_post_padded.item()  * 4 + (256 if (num_tokens_post_padded.item() / block_size_m) > 256 else (num_tokens_post_padded.item() / block_size_m)) * 7168 * 4)/1024/1024/1024
