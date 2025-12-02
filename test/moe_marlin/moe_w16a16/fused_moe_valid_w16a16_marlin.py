# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional, Type,Any, Dict, List, Tuple

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

from triton_w16a16 import invoke_fused_moe_kernel_w16a16
import triton
import triton.language as tl

from lightop import moe_gemm_marlin_w16a16, get_moe_cuda_marlin_config_w16a16

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

blockM_all_configs = {
    103: 16, 
    104: 32, 
    106: 16, 
    111: 32, 
    115: 32, 
    117: 64, 
    120: 32, 
    121: 64, 
    123: 64, 
    125: 128,
    128: 16, 
    129: 16, 
    130: 16, 
    132: 32, 
    133: 32, 
    136: 64, 
    141: 32, 
    142: 64, 
    143: 32, 
    144: 64, 
    146: 64, 
    147: 64, 
    148: 64, 
    149: 128,
    153: 16, 
    154: 32, 
    156: 16, 
    161: 32, 
    167: 16, 
    168: 32, 
    169: 16, 
    175: 32, 
    177: 64, 
    180: 32, 
    181: 64, 
    183: 64, 
    185: 128,
    188: 16, 
    189: 16, 
    190: 16, 
    192: 32, 
    193: 32, 
    195: 64, 
}

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

def fused_moe_valid_w16a16_marlin(
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
                                    modes: Optional[List[int]]=None,
                                    gemm1_compare_failed_list = [],
                                    gemm2_compare_failed_list = [],
                                    impl_status_tune: bool = False
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
    mode0 = modes[0]
    mode1 = modes[1]
    config_marlin_0["BLOCK_SIZE_M"] = blockM_all_configs[mode0]
    config_marlin_0["MODE"]         = mode0
    config_marlin_0["DELTA"]        = 1
    config_marlin_1["BLOCK_SIZE_M"] = blockM_all_configs[mode1]
    config_marlin_1["MODE"]         = mode1
    config_marlin_1["DELTA"]        = int(blockM_all_configs[mode0]/blockM_all_configs[mode1])
    print("config_marlin_0: ", config_marlin_0)
    print("config_marlin_1: ", config_marlin_1)
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)
    # triton 算子始终使用 block_m=16 测试
    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts)

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

    # hidden_states = torch.ones_like(hidden_states)
    # w1_triton = torch.ones_like(w1_triton)
    # w1_marlin = torch.ones_like(w1_marlin)
    # w2_triton = torch.ones_like(w2_triton)
    # w2_marlin = torch.ones_like(w2_marlin)
    # topk_weights = torch.ones_like(topk_weights)
    
    # =========================== gemm1 marlin 计算 ===============================
    moe_gemm_marlin_w16a16(hidden_states, w1_marlin, intermediate_cache1, None, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, topk, config_marlin_0)
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

    if not impl_status_tune:
        if not (torch.allclose(intermediate_cache1, triton_gemm1_output, rtol=1e-2, atol=1e-2)):
            gemm1_compare_failed_list.append(config_marlin_0['MODE'])
            print(f"kernel{1} 精度检查不合格!!!")
            
            # torch.set_printoptions(profile="full")  
            # ftriton = open('cuda_output.txt', 'w')   
            # print(f'cuda_output: {intermediate_cache1}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {triton_gemm1_output}', file=ftorch)
            # ftorch.close
        else:
            print(f"kernel{1} 精度检查合格.")

    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    torch.cuda.synchronize()

    moe_gemm_marlin_w16a16(intermediate_cache2, w2_marlin, intermediate_cache3, topk_weights, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, 1, config_marlin_1)
    # =========================== gemm2 triton 计算 ===============================
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
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
    if not impl_status_tune:
        if not (torch.allclose(intermediate_cache3, triton_gemm2_output, rtol=1e-2, atol=1e-2)):
            gemm2_compare_failed_list.append(config_marlin_1['MODE'])
            print(f"kernel{2} 精度检查不合格!!!")
            # torch.set_printoptions(profile="full")  
            # ftriton = open('cuda_output.txt', 'w')   
            # print(f'cuda_output: {out_cuda}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {C}', file=ftorch)
            # ftorch.close
        else:
            print(f"kernel{2} 精度检查合格.")