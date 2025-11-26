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
    500: 16,
    501: 32,
    502: 64,
    503: 16,
    504: 32,
    506: 16,
    509: 32,
    510: 64,
    511: 32,
    513: 64,
    514: 32,
    515: 32,
    516: 64,
    517: 64,
    518: 32,
    519: 64,
    520: 32,
    521: 64,
    522: 64,
    523: 64,
    524: 128,
    525: 128,
    526: 16,
    527: 32,
    528: 16,
    529: 16,
    530: 16,
    531: 32,
    532: 32,
    533: 32,
    535: 64,
    536: 64,
    539: 32,
    540: 64,
    541: 32,
    542: 64,
    543: 32,
    544: 64,
    545: 64,
    546: 64,
    547: 64,
    548: 64,
    549: 128,
    550: 16,
    551: 32,
    552: 64,
    553: 16,
    554: 32,
    556: 16,
    559: 32,
    560: 64,
    561: 32,
    563: 64,
    564: 16,
    565: 32,
    566: 64,
    567: 16,
    568: 32,
    569: 16,
    570: 32,
    571: 64,
    574: 32,
    575: 32,
    576: 64,
    577: 64,
    578: 32,
    579: 64,
    580: 32,
    581: 64,
    582: 64,
    583: 64,
    584: 128,
    585: 128,
    586: 16,
    587: 32,
    588: 16,
    589: 16,
    590: 16,
    591: 32,
    592: 32,
    593: 32,
    594: 64,
    595: 64,
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

def fused_moe_valid_w16a16(
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

    # import pdb
    # pdb.set_trace()
    # hidden_states = torch.ones_like(hidden_states)
    # w1_triton = torch.ones_like(w1_triton)
    # w2_triton = torch.ones_like(w2_triton)
    # topk_weights = torch.ones_like(topk_weights)
    # =========================== gemm1 marlin 计算 ===============================
    moe_gemm_marlin_w16a16(hidden_states, w1_triton, intermediate_cache1, None, sorted_token_ids, 
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
            # print(f'cuda_output: {out_cuda}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {C}', file=ftorch)
            # ftorch.close
        else:
            print(f"kernel{1} 精度检查合格.")

    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    torch.cuda.synchronize()

    moe_gemm_marlin_w16a16(intermediate_cache2, w2_triton, intermediate_cache3, topk_weights, sorted_token_ids, 
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