# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import pdb
import torch
import lightop
from lightop.activation import silu_and_mul_opt
from lightop.moe import moe_align_block_size

from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)

from triton_w4a8 import invoke_fused_moe_kernel_w4a8
import triton
import triton.language as tl

from marlin_quant import unpack_int8_to_int4

from lightop.moe import moe_gemm_marlin_w4a8, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8

device_name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

def fused_marlin_moe(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1_marlin: torch.Tensor,
                     w2_marlin: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,
                     ep_size: int = 8,
                     expert_map: Optional[torch.Tensor] = None,
                     inplace: bool = False,
                     ) -> torch.Tensor:
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

    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w4a8(
                                                                                E, 
                                                                                M, 
                                                                                2 * N, 
                                                                                K, 
                                                                                K, 
                                                                                N, 
                                                                                topk, 
                                                                                device_name, 
                                                                                num_cus, 
                                                                                tl.bfloat16,
                                                                                is_ep=expert_map is not None,
                                                                                ep_size=ep_size)

    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    assert expert_map is not None

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts, expert_map)
    
    # 建立 归约使用的 信号量 Tensor 当前最大开到 sms * 6 = 480 
    sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    workspace = torch.zeros(500, dtype=torch.int, device=hidden_states.device, requires_grad=False)
    # 建立用于 归约使用的 缓冲区 blokc数： sms * 6
    global_reduce_buffer = torch.zeros(sms * 6 * block_size_m * 512, dtype=torch.int, device=hidden_states.device, requires_grad=False)

    # 申明 cuda 算子使用的 独占返回张量
    cuda_gemm1_output = torch.zeros((M * topk, 2 * N), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    cuda_gemm2_output = torch.zeros((M * topk, K), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)

    # 
    cuda_gemm2_a = torch.zeros((M * topk, N), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    # 对A矩阵进行量化
    a_1, a_1_scale = per_token_quant_int8(hidden_states)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # =========================== gemm1 marlin 计算 ===============================
    moe_gemm_marlin_w4a8(
                        a_1,
                        w1_marlin, 
                        cuda_gemm1_output, 
                        a_1_scale, 
                        w1_scale, 
                        None, 
                        sorted_token_ids, 
                        expert_ids, 
                        num_tokens_post_padded, 
                        topk, 
                        workspace, 
                        global_reduce_buffer, 
                        config_marlin_0,
                        is_ep=expert_map is not None)
    end_event.record()
    end_event.synchronize()
    gemm1_time_us = start_event.elapsed_time(end_event) * 1000

    # 激活函数
    silu_and_mul_opt(cuda_gemm2_a, cuda_gemm1_output)

    # 准备gemm2 需要的 A和A-scale 
    # [M*topk, N] [32*8, 256]
    a_2, a_2_scale = per_token_quant_int8(cuda_gemm2_a)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    moe_gemm_marlin_w4a8(
                        a_2, 
                        w2_marlin, 
                        cuda_gemm2_output, 
                        a_2_scale, 
                        w2_scale, 
                        topk_weights, 
                        sorted_token_ids, 
                        expert_ids, 
                        num_tokens_post_padded, 
                        1, 
                        workspace, 
                        global_reduce_buffer, 
                        config_marlin_1,
                        is_ep=expert_map is not None)
    end_event.record()
    end_event.synchronize()
    gemm2_time_us =  start_event.elapsed_time(end_event) * 1000 
    return gemm1_time_us, gemm2_time_us