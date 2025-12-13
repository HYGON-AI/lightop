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

triton_opt_configs = {
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 2,
        "num_stages": 2
    },
    "2": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "4": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "8": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "16": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "24": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "32": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 2,
        "num_stages": 2
    },
    "48": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "64": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 2,
        "num_stages": 2
    },
    "96": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "128": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },    
    "256": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "512": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "1024": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "1536": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 2
    },
    "2048": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "3072": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "4096": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    }
}

triton_opt_configs_nn = {
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 2,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "2": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 64,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "4": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
        "num_ldmatrixes": 1
    },
    "8": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "16": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "24": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "32": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "48": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "64": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "96": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "128": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "256": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "390": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "512": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "1024": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "1536": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "2048": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "3072": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    },
    "4096": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "num_ldmatrixes": 1
    }
}


def get_triton_config(num_tokens: int, use_nn_moe: Optional[bool]=False):
    # 将字典的键转换为整数并排序
    num_tokens_range = sorted([int(k) for k in triton_opt_configs_nn.keys()]) if use_nn_moe else sorted([int(k) for k in triton_opt_configs.keys()])
    
    # 找到小于等于 num_tokens 的最大键
    matched_key = None
    for key in num_tokens_range:
        if key <= num_tokens:
            matched_key = key
        else:
            break
    
    # 如果找不到（num_tokens 小于所有键），返回最小的键对应的配置
    if matched_key is None:
        # 找距离 num_tokens 最近的键
        min_distance = float('inf')
        for key in num_tokens_range:
            distance = abs(key - num_tokens)
            if distance < min_distance:
                min_distance = distance
                matched_key = key
    
    # 返回对应的配置值
    return triton_opt_configs[str(matched_key)] if not use_nn_moe else triton_opt_configs_nn[str(matched_key)]

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
                     w1_marlin: torch.Tensor,
                     w2_marlin: torch.Tensor,
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
    assert hidden_states.dtype in [torch.float16,torch.bfloat16]
    compute_type = None
    if hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")
    M, K = hidden_states.shape # 16 2048
    E = w1_triton.shape[0] # 128
    N = w1_triton.shape[1] // 2 # 768

    topk = topk_ids.shape[1] 
    # print("topk_ids",topk_ids)
    # print("topk_ids",topk_ids.size())
    # print("topk_weights",topk_weights)
    # print("topk_weights",topk_weights.size())   

    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w16a16(E, M, 2 * N, K, K, N, 8, device_name, num_cus, hidden_states.dtype)
    # print("E, M, 2 * N, K, K, N, 8, device_name, num_cus, tl.bfloat16",E, M, 2 * N, K, K, N, 8, device_name, num_cus, hidden_states.dtype)
    # debug
    # config_marlin_0["BLOCK_SIZE_M"] = 64
    # config_marlin_0["MODE"] = 147
    # config_marlin_1["BLOCK_SIZE_M"] = 64
    # config_marlin_1["MODE"] = 183
    # print("         config_marlin_0: ", config_marlin_0)
    # print("         config_marlin_1: ", config_marlin_1)
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    use_nn_moe = True
    opt_config = get_triton_config(M, use_nn_moe)
    # print("triton opt_config: ", opt_config, "use_nn_moe: ", use_nn_moe)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)
    # triton 算子始终使用 block_m=16 测试
    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts)
    sorted_token_ids_triton_opt, expert_ids_triton_opt, num_tokens_post_padded_triton_opt = moe_align_block_size(topk_ids, opt_config["BLOCK_SIZE_M"], global_num_experts)

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

    moe_gemm_marlin_w16a16(hidden_states, w1_marlin, intermediate_cache1, None, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, topk, config_marlin_0)
    # print("w16a16 kernel 执行成功")
    # =========================== gemm1 triton 计算 ===============================
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    triton_opt_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
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
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=False,
                                block_shape=None
                                )
    # 将 w1_triton 从 [E, 2*N, K] 转置为 [E, K, 2*N]，并在显存上实际修改
    w1_triton_transposed = w1_triton.transpose(1, 2).contiguous()
    # import pdb
    # pdb.set_trace()
    invoke_fused_moe_kernel_w16a16(
                                hidden_states,
                                w1_triton_transposed,
                                triton_opt_gemm1_output,
                                None,# a_scale
                                None,# b_scale
                                None, # w2_zero
                                topk_weights,
                                topk_ids,
                                sorted_token_ids_triton_opt,
                                expert_ids_triton_opt,
                                num_tokens_post_padded_triton_opt,
                                False,  # 是否乘 top_weight
                                topk, # top_k
                                opt_config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=False,
                                block_shape=None,
                                use_nn_moe=use_nn_moe,
                                )

    triton_gemm1_output = triton_gemm1_output.reshape((-1, 2 * N))
    triton_opt_gemm1_output = triton_opt_gemm1_output.reshape((-1, 2 * N))
    torch.cuda.synchronize()
    
    torch.testing.assert_close(triton_opt_gemm1_output, triton_gemm1_output, atol=1e-2, rtol=1e-2)
    # print("        triton_opt对比triton gemm1 success!!!!")
    
    torch.testing.assert_close(intermediate_cache1, triton_gemm1_output, atol=1e-2, rtol=1e-2)
    # print("        cuda对比triton gemm1 success!!!!")
    
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
   
    moe_gemm_marlin_w16a16(intermediate_cache2, w2_marlin, intermediate_cache3,topk_weights, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, 1, config_marlin_1)
    # =========================== gemm2 triton 计算 ===============================
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    triton_opt_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    # import pdb
    # pdb.set_trace()
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
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=False,
                                block_shape=None
                                )
    # 将 w1_triton 从 [E, 2*N, K] 转置为 [E, K, 2*N]，并在显存上实际修改
    w2_triton_transposed = w2_triton.transpose(1, 2).contiguous()
    invoke_fused_moe_kernel_w16a16(
                                intermediate_cache2,
                                w2_triton_transposed,
                                triton_opt_gemm2_output,
                                None,# a_scale
                                None,# b_scale
                                None, # w2_zero
                                topk_weights,
                                topk_ids,
                                sorted_token_ids_triton_opt,
                                expert_ids_triton_opt,
                                num_tokens_post_padded_triton_opt,
                                True,  # 是否乘 top_weight
                                1, # top_k
                                opt_config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int8_w8a8 = False,
                                use_int4_w4a8=False,
                                block_shape=None,
                                use_nn_moe=use_nn_moe,
                                )
    torch.cuda.synchronize()
    triton_gemm2_output = triton_gemm2_output.reshape((-1, K))
    triton_opt_gemm2_output = triton_opt_gemm2_output.reshape((-1, K))
    
    torch.testing.assert_close(triton_opt_gemm2_output, triton_gemm2_output, atol=1e-2, rtol=1e-2)
    # print("        triton_opt对比triton gemm2 success!!!!")
    
    torch.testing.assert_close(intermediate_cache3, triton_gemm2_output, atol=1e-2, rtol=1e-2)
    # print("        Triton对比cuda gemm2 success!!!!")
    
    #N:384 K:2048
    return (num_tokens_post_padded.item() * K * 2*N * 2),(num_tokens_post_padded.item()  * N * K * 2), \
    (num_tokens_post_padded.item() * K * 2 + (E if (num_tokens_post_padded.item() / block_size_m) > E else (num_tokens_post_padded.item() / block_size_m)) * 2*N * K * 2  + M * topk * 2*N * 2)/1024/1024/1024, \
    (num_tokens_post_padded.item() * N * 2 + (E if (num_tokens_post_padded.item() / block_size_m) > E else (num_tokens_post_padded.item() / block_size_m)) * N * K * 2 + M * topk * K * 2 )/1024/1024/1024, \
    hidden_states, intermediate_cache2, triton_gemm1_output, triton_gemm2_output     


def fused_triton_moe(
                     hidden_states: torch.Tensor, # 32, 7168
                     intermediate_cache2: torch.Tensor, # 32*8, 256
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168, 
                     w1_marlin: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2_marlin: torch.Tensor, # 256, 256, 7168,
                     gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     inplace: bool = False,
                     use_nn_moe: Optional[bool]=False,
                     ) -> torch.Tensor:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch w1"
    
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前支持 bf16和fp16
    assert hidden_states.dtype in [torch.bfloat16, torch.float16]
    compute_type = None
    if hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")
    M, K = hidden_states.shape # 16 2048
    E = w1.shape[0] # 128
    N = w1.shape[1] // 2 if not use_nn_moe else (w1.shape[1] // 2) # 768
    topk = topk_ids.shape[1] 
    opt_config = get_triton_config(M, use_nn_moe)
    block_size_m = opt_config["BLOCK_SIZE_M"]
    # print("triton opt_config: ", opt_config, "use_nn_moe: ", use_nn_moe)

    if global_num_experts == -1:
        global_num_experts = E

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)

    # intermediate_cache2 = torch.empty(
    #     (M * topk_ids.shape[1], N),
    #     device=hidden_states.device,
    #     dtype=hidden_states.dtype,
    # )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K), ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N) # [M * topk, 2 * N]  [32*8, 512]
    intermediate_cache3 = intermediate_cache13[:M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)  # [M * topk, K]  [32*8, 7168]
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    w1 = w1.transpose(1, 2).contiguous() if use_nn_moe else w1
    w2 = w2.transpose(1, 2).contiguous() if use_nn_moe else w2
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: invoke_fused_moe_kernel_w16a16(
                                                                hidden_states,
                                                                w1,
                                                                triton_gemm1_output,
                                                                None,# a_scale
                                                                None,# b_scale
                                                                None, # w2_zero
                                                                topk_weights,
                                                                topk_ids,
                                                                sorted_token_ids,
                                                                expert_ids,
                                                                num_tokens_post_padded,
                                                                False,  # 是否乘 top_weight
                                                                topk, # top_k
                                                                opt_config,
                                                                compute_type=compute_type,
                                                                use_fp8_w8a8=False,
                                                                use_int8_w8a16=False,
                                                                use_int8_w8a8 = False,
                                                                use_int4_w4a8=False,
                                                                block_shape=None,
                                                                use_nn_moe=use_nn_moe,
                                                                ), quantiles=quantiles)

    gemm1_time_us = avg_ms * 1000
    # 激活函数
    # torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
    torch.cuda.synchronize()
    # import pdb
    # pdb.set_trace()
    # # opt_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: invoke_fused_moe_kernel_w16a16(
                                                                    intermediate_cache2,
                                                                    w2,
                                                                    triton_gemm2_output,
                                                                    None,# a_scale
                                                                    None,# b_scale
                                                                    None, # w2_zero
                                                                    topk_weights,
                                                                    topk_ids,
                                                                    sorted_token_ids,
                                                                    expert_ids,
                                                                    num_tokens_post_padded,
                                                                    True,  # 是否乘 top_weight
                                                                    1, # top_k
                                                                    opt_config,
                                                                    compute_type=compute_type,
                                                                    use_fp8_w8a8=False,
                                                                    use_int8_w8a16=False,
                                                                    use_int8_w8a8 = False,
                                                                    use_int4_w4a8=False,
                                                                    block_shape=None,
                                                                    use_nn_moe=use_nn_moe,
                                                                    ), quantiles=quantiles)


    gemm2_time_us =  avg_ms * 1000 
    triton_gemm1_output = triton_gemm1_output.reshape((-1, 2 * N))
    triton_gemm2_output = triton_gemm2_output.reshape((-1, K))
    return gemm1_time_us, gemm2_time_us, triton_gemm1_output, triton_gemm2_output