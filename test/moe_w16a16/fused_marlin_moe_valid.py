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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from triton_w16a16 import invoke_fused_moe_kernel_w16a16
import triton
import triton.language as tl

from marlin_quant import unpack_int8_to_int4

from lightop import moe_gemm_marlin_w16a16, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w16a16

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

from vllm.model_executor.layers.activation import SiluAndMul
# def torch_moe(a, w1, w2, score, topk ,topk_weight1, topk_ids1):
#     B, D = a.shape
#     a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)      # [m*topk, k]
#     out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)  # [m*topk, k]
#     score = torch.softmax(score, dim=-1, dtype=torch.float32) # [m, e]==> [m, e]
#     topk_weight, topk_ids = torch.topk(score, topk) # [m, topk]
#     topk_weight = topk_weight.view(-1) # [m*topk]
#     topk_ids = topk_ids.view(-1) # [m*topk]
    
#     for i in range(w1.shape[0]): # 
#         mask = topk_ids == i # [m*topk]
#         if mask.sum():
#             out[mask] = SiluAndMul()(
#                 a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
#     return (out.view(B, -1, w2.shape[1]) *
#             topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def torch_moe(a, w1, w2, score, topk ,topk_weight1, topk_ids1):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)      # [m*topk, k]
    out = torch.zeros(B * topk, w1.shape[1], dtype=a.dtype, device=a.device)  # [m*topk, k]
    score = torch.softmax(score, dim=-1, dtype=torch.float32) # [m, e]==> [m, e]
    topk_weight, topk_ids = torch.topk(score, topk) # [m, topk]
    topk_weight1 = topk_weight1.view(-1) # [m*topk]
    topk_ids1 = topk_ids1.view(-1) # [m*topk]
    # import pdb
    # pdb.set_trace()
    for i in range(w1.shape[0]): # 
        mask = topk_ids1 == i # [m*topk]
        if mask.sum():
            out[mask] = (
                a[mask] @ w1[i].transpose(0, 1)) 
    return (out.to(out.dtype))

# def torch_moe(a, w1, w2, score, topk ,topk_weight, topk_ids):
#     M, K = a.shape #a:torch.Size([16, 2048]) M:16 K:2048
#     # import pdb
#     # pdb.set_trace()
#     # a.view(M, -1, K) torch.Size([16, 1, 2048])
#     # a.view(M, -1, K).repeat(1, topk, 1) torch.Size([16, 8, 2048])
#     # a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K) torch.Size([128, 2048])
#     a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)      # [m*topk, k]
#     #NT 128 * 768
#     out = torch.zeros(M * topk, w1.shape[1], dtype=a.dtype, device=a.device)  # [m*topk, k]
    
#     for i in range(w1.shape[0]): # 
#         mask = topk_ids == i # [m*topk]
#         if mask.sum():
#             out[mask] = a[mask] @ w1[i].transpose(0, 1)
#     return (out.to(out.dtype))

def fused_marlin_moe_valid(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1_triton: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2_triton: torch.Tensor, # 256, 256, 7168
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
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1_triton.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2_triton.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前只支持 bf16
    assert hidden_states.dtype in [torch.bfloat16]

    M, K = hidden_states.shape # 16 2048
    E = w1_triton.shape[0] # 128
    N = w1_triton.shape[1] // 2 # 768
    topk = topk_ids.shape[1] 
    # print("topk_ids",topk_ids)
    # print("topk_ids",topk_ids.size())
    # print("topk_weights",topk_weights)
    # print("topk_weights",topk_weights.size())   

    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w16a16(E, M, 2 * N, K, K, N, 8, device_name, num_cus, tl.bfloat16)
    print("E, M, 2 * N, K, K, N, 8, device_name, num_cus, tl.bfloat16",E, M, 2 * N, K, K, N, 8, device_name, num_cus, tl.bfloat16)
    # debug
    # config_marlin_0["BLOCK_SIZE_M"] = 128
    # config_marlin_0["MODE"]         = 1000
    # config_marlin_1["BLOCK_SIZE_M"] = 128
    # config_marlin_1["MODE"]         = 1000
    print("         config_marlin_0: ", config_marlin_0)
    print("         config_marlin_1: ", config_marlin_1)
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E
    # import pdb
    # pdb.set_trace()
    # sorted_token_ids : torch.Size([2048])
    # expert_ids : torch.Size([128])
    # num_tokens_post_padded : tensor([1264], device='cuda:0', dtype=torch.int32)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)
    # triton 算子始终使用 block_m=16 测试
    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts)

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

    # =========================== gemm1 marlin 计算 ===============================
    
    # config_marlin = {
    #                     "BLOCK_SIZE_M": block_size_m,
    #                     "thread_n": thread_n,
    #                     "thread_k": thread_k,
    #                     "blocks_per_cu": blocks_per_cu,
    #                     'MODE': 500,
    #                     'DELTA': 1
    #                 }

    moe_gemm_marlin_w16a16(hidden_states, w1_triton, intermediate_cache1, None, sorted_token_ids, 
                         expert_ids_triton, num_tokens_post_padded, topk, config_marlin_0)
    # print("w16a16 kernel 执行成功")
    # =========================== gemm1 triton 计算 ===============================
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}
    torch.cuda.synchronize()

    invoke_fused_moe_kernel_w16a16(
                                hidden_states,
                                w1_triton,
                                triton_gemm1_output,
                                None,# a_scale
                                None,# b_scale
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
                                use_int4_w4a8=False,
                                block_shape=None
                                )

    triton_gemm1_output = triton_gemm1_output.reshape((-1, 2 * N))

    torch.cuda.synchronize()
    torch.testing.assert_close(intermediate_cache1, triton_gemm1_output, atol=1e-2, rtol=1e-2)
    print("        cuda对比triton gemm1 success!!!!")


    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)


    # 准备gemm2 需要的 A和A-scale 
    # [M*topk, N] [32*8, 256]

    torch.cuda.synchronize()

    moe_gemm_marlin_w16a16(intermediate_cache2, w2_triton, intermediate_cache3, topk_weights, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, 1, config_marlin_1)
    # =========================== gemm2 triton 计算 ===============================
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    # topk_weights = torch.ones_like(topk_weights)
    invoke_fused_moe_kernel_w16a16(
                                intermediate_cache2,
                                w2_triton,
                                triton_gemm2_output,
                                None,# a_scale
                                None,# b_scale
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
                                use_int4_w4a8=False,
                                block_shape=None
                                )
    triton_gemm2_output = triton_gemm2_output.reshape((-1, K))

    torch.testing.assert_close(intermediate_cache3, triton_gemm2_output, atol=1e-2, rtol=1e-2)
    print("        Triton对比cuda gemm2 success!!!!")
    return (num_tokens_post_padded.item() * K * 2*N * 2),(num_tokens_post_padded.item()  * N * K * 2), \
           (num_tokens_post_padded.item() * K * 2 + (E if (num_tokens_post_padded.item() / block_size_m) > E else (num_tokens_post_padded.item() / block_size_m)) * 2*N * K * 2  + M * topk * 2*N * 2)/1024/1024/1024, \
           (num_tokens_post_padded.item() * N * 2 + (E if (num_tokens_post_padded.item() / block_size_m) > E else (num_tokens_post_padded.item() / block_size_m)) * N * K * 2 + M * topk * K * 2 )/1024/1024/1024
 