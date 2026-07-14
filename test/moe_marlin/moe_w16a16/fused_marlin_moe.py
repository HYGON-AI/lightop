# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import torch
import lightop

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)

from lightop.moe import moe_gemm_marlin_w16a16,moe_gemm_w4a8, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w16a16

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
import triton
import triton.language as tl
def fused_marlin_moe(
                     hidden_states: torch.Tensor, # 32, 7168
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168, 
                     w1_marlin: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2_marlin: torch.Tensor, # 256, 256, 7168,
                     gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     inplace: bool = False) -> torch.Tensor:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch w1"
    
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前只支持 bf16
    assert hidden_states.dtype in [torch.bfloat16,torch.float16]

    M, K = hidden_states.shape # 16 2048
    E = w1.shape[0] # 128
    N = w1.shape[1] // 2 # 768
    topk = topk_ids.shape[1] 
    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w16a16(E, M, 2 * N, K, K, N, 8, device_name, num_cus, hidden_states.dtype)
    # debug
    # config_marlin_0["DELTA"] = 1
    # config_marlin_1["DELTA"] = 1
    # config_marlin_0["BLOCK_SIZE_M"] = 16
    # config_marlin_0["MODE"] = 304
    # config_marlin_1["BLOCK_SIZE_M"] = 16
    # config_marlin_1["MODE"] = 304
    # print("         config_marlin_0: ", config_marlin_0)
    # print("         config_marlin_1: ", config_marlin_1)
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts)

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
    
    torch.cuda.synchronize()

    moe_gemm_marlin_w16a16(hidden_states, w1_marlin, intermediate_cache1, None, sorted_token_ids, 
                         expert_ids, num_tokens_post_padded, topk, config_marlin_0)
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: moe_gemm_marlin_w16a16(hidden_states, 
                                                                                    w1_marlin, 
                                                                                    intermediate_cache1, 
                                                                                    None, 
                                                                                    sorted_token_ids, 
                                                                                    expert_ids, 
                                                                                    num_tokens_post_padded, 
                                                                                    topk, 
                                                                                    config_marlin_0), quantiles=quantiles)



    gemm1_time_us = avg_ms * 1000
    # 激活函数
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    torch.cuda.synchronize()

    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: moe_gemm_marlin_w16a16(intermediate_cache2, 
                                                                                    w2_marlin, 
                                                                                    intermediate_cache3,
                                                                                    topk_weights, 
                                                                                    sorted_token_ids, 
                                                                                    expert_ids, 
                                                                                    num_tokens_post_padded, 
                                                                                    1, 
                                                                                    config_marlin_1), quantiles=quantiles)

    gemm2_time_us =  avg_ms * 1000 
    return gemm1_time_us, gemm2_time_us
