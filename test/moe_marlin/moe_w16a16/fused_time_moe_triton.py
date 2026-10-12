# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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
from triton_w16a16_bench import invoke_fused_moe_kernel_w16a16
from lightop.moe import moe_gemm_marlin_w16a16,moe_gemm_w4a8, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
import triton
import triton.language as tl
def fused_time_moe_triton(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
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
    assert hidden_states.dtype in [torch.bfloat16]

    M, K = hidden_states.shape # 16 2048
    E = w1.shape[0] # 128
    N = w1.shape[1] // 2 # 768
    topk = topk_ids.shape[1]
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,"num_warps": 8,"num_stages": 2,"num_ldmatrixes": 1}
    block_size_m = config["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

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
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    torch.cuda.synchronize()
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()

    quantiles = [0.5,0.2,0.3]
    ms_gemm1,minms,maxms = triton.testing.do_bench(lambda:         invoke_fused_moe_kernel_w16a16(
                                hidden_states,
                                w1,
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
                                ),quantiles=quantiles)

    # end_event.record()
    # end_event.synchronize()
    # gemm1_time_us = start_event.elapsed_time(end_event) * 1000
    # 激活函数
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
    triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # torch.cuda.synchronize()
    # start_event.record()
    
    
    # topk_weights = torch.ones_like(topk_weights)
    ms_gemm2,minms,maxms = triton.testing.do_bench(lambda:         invoke_fused_moe_kernel_w16a16(
                                intermediate_cache2,
                                w2,
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
                                ),quantiles=quantiles)
    
   

    # end_event.record()
    # end_event.synchronize()
    # gemm2_time_us =  start_event.elapsed_time(end_event) * 1000 
    triton_gemm2_output = triton_gemm2_output.reshape((-1, K))    
    # return gemm1_time_us, gemm2_time_us
    return ms_gemm1,ms_gemm2




def Configen():
    import itertools
    BLOCK_M=[16]
    config_list = list(itertools.product(
        BLOCK_M
    ))
    configs = [
        {"BLOCK_SIZE_M":i[0]}
        for i in config_list 
    ]
    return configs
def do_bench_fused_time_moe_triton(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
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
    assert hidden_states.dtype in [torch.bfloat16]

    M, K = hidden_states.shape # 16 2048
    E = w1.shape[0] # 128
    N = w1.shape[1] // 2 # 768
    topk = topk_ids.shape[1]
    # config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,"num_warps": 8,"num_stages": 2,"num_ldmatrixes": 1}
    
    configs = Configen()
    # block_size_m = config["BLOCK_SIZE_M"]
    if global_num_experts == -1:
        global_num_experts = E

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
    triton_gemm1_output = torch.zeros_like(intermediate_cache1).reshape((M, topk, 2 * N))
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    best_time=float('inf')
    ms11=float('inf')
    ms22=float('inf')
    quantiles = [0.5,0.2,0.3]
    for i in range(0,len(configs)):
        config = configs[i]
        # print("config",config)
        ms_gemm1,minms,maxms = triton.testing.do_bench(lambda:         invoke_fused_moe_kernel_w16a16(
                                hidden_states,
                                w1,
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
                                ),quantiles=quantiles)

    # end_event.record()
    # end_event.synchronize()
    # gemm1_time_us = start_event.elapsed_time(end_event) * 1000
    # 激活函数
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
        triton_gemm2_output = torch.zeros_like(intermediate_cache3).reshape((M, topk, K))
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # torch.cuda.synchronize()
    # start_event.record()
    
    
    # topk_weights = torch.ones_like(topk_weights)
        ms_gemm2,minms,maxms = triton.testing.do_bench(lambda:         invoke_fused_moe_kernel_w16a16(
                                intermediate_cache2,
                                w2,
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
                                ),quantiles=quantiles)
    
   

    # end_event.record()
    # end_event.synchronize()
    # gemm2_time_us =  start_event.elapsed_time(end_event) * 1000 
        triton_gemm2_output = triton_gemm2_output.reshape((-1, K))    
    # return gemm1_time_us, gemm2_time_us
        if(ms_gemm1 + ms_gemm2 < best_time):
            # print("config",config)
            best_time = ms_gemm1 + ms_gemm2
            # print("total ms",best_time)
            best_config = config
            ms11 = ms_gemm1
            ms22 = ms_gemm2
            # print("kernel1 ms:",ms)
            # print("kernel2 ms:",ms2)
    return ms11,ms22