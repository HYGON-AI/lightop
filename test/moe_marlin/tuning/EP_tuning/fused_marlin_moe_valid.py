# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import pdb
import torch
import lightop
from typing import Any, Dict, List, Tuple, TypedDict
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

from lmslim.layers.gemm.int8_utils import (
   per_token_group_quant_int8, per_token_quant_int8)

from triton_w4a8 import invoke_fused_moe_kernel_w4a8
import triton
import triton.language as tl

from marlin_quant import unpack_int8_to_int4

from lightop import moe_gemm_marlin_w4a8, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8
from lightop import get_moe_cuda_marlin_config_w4a8, moe_gemm_marlin_w4a8
from lightop.activation import fuse_silu_mul_quant
from lightop import op as op
from vllm.platforms import current_platform

device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

# 应用 256 降到 32 的 moe_align 优化
def moe_align_block_size_lightop_new(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    num_token: Optional[int] = None,
    ep_size:int = 8,
    is_EP : bool  = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_num_experts = num_experts // ep_size
    if num_token:
        if num_token < block_size:
            max_num_tokens_padded = min(topk_ids.numel() * block_size, topk_ids.numel() + local_num_experts * (block_size - 1))
        else:
            max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        sorted_ids = torch.full((max_num_tokens_padded,), fill_value=topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    else:
        max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=topk_ids.device)
        sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    if expert_map is not None:
        expert_ids = torch.zeros((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    else:
        expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.zeros((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)    
    op.moe_align_block_size(topk_ids, 
                            num_experts, 
                            block_size, 
                            sorted_ids,
                            expert_ids, 
                            num_tokens_post_pad,
                            expert_map,
                            expert_mask,
                            None,
                            is_EP)
    return sorted_ids, expert_ids, num_tokens_post_pad

def fused_marlin_moe_valid(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1_triton: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2_triton: torch.Tensor, # 256, 256, 7168
                     w1_marlin: torch.Tensor,
                     w2_marlin: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     config_marlin_0,
                     config_marlin_1,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     inplace: bool = False,
                     ep_size=0
                     ) -> torch.Tensor:
    assert ep_size != 0
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
    
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    assert expert_map is not None
        
    # 应用 256 降到 32 的 moe_align 优化
    expert_mask = torch.zeros((M, topk), dtype=torch.bool,  device=hidden_states.device, requires_grad=False)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_lightop_new(
                                                                                    topk_ids,
                                                                                    block_size_m,
                                                                                    global_num_experts,
                                                                                    expert_map,
                                                                                    expert_mask,
                                                                                    num_token=M,
                                                                                    ep_size=ep_size,
                                                                                    is_EP=True)

    # triton 算子始终使用 block_m=16 测试
    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts, expert_map)

    # 建立 归约使用的 信号量 Tensor 当前最大开到 sms * 6 = 480 
    sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    workspace = torch.zeros(500, dtype=torch.int, device=hidden_states.device, requires_grad=False)
    # 建立用于 归约使用的 缓冲区 blokc数： sms * 6
    global_reduce_buffer = torch.zeros(sms * 6 * block_size_m * 512, dtype=torch.int, device=hidden_states.device, requires_grad=False)

    # 声明 triton 算子使用的 独占返回张量
    triton_gemm1_output = torch.zeros((M, topk, 2 * N), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    triton_gemm2_output = torch.zeros((M, topk, K), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    # 申明 cuda 算子使用的 独占返回张量
    cuda_gemm1_output = torch.zeros((M * topk, 2 * N), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    cuda_gemm2_output = torch.zeros((M * topk, K), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)

    triton_gemm2_a = torch.zeros((M * topk, N), dtype=torch.bfloat16, device=hidden_states.device, requires_grad=False)
    # 对A矩阵进行量化
    a_1, a_1_scale = per_token_quant_int8(hidden_states)

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
    # =========================== gemm1 triton 计算 ===============================
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}
    torch.cuda.synchronize()
    invoke_fused_moe_kernel_w4a8(
                                a_1, # torch.Size([1024, 7168])
                                w1_triton, # torch.Size([256, 4096, 3584])
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
    # torch.testing.assert_close(cuda_gemm1_output, triton_gemm1_output, atol=1e-3, rtol=1e-3)
    gemm1_falg = torch.allclose(cuda_gemm1_output, triton_gemm1_output, rtol=1e-3, atol=1e-3)
    torch.cuda.synchronize()

    # 激活函数
    torch.ops._C.silu_and_mul(triton_gemm2_a, triton_gemm1_output)

    # 准备gemm2 需要的 A和A-scale 
    # [M*topk, N] [32*8, 256]
    a_2, a_2_scale = per_token_quant_int8(triton_gemm2_a)
    torch.cuda.synchronize()

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
    # =========================== gemm2 triton 计算 ===============================
    invoke_fused_moe_kernel_w4a8(a_2, 
                                w2_triton, 
                                triton_gemm2_output, 
                                a_2_scale, 
                                w2_scale,
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
    gemm2_falg = torch.allclose(cuda_gemm2_output, triton_gemm2_output, rtol=1e-2, atol=1e-2)
    
    return (gemm1_falg, gemm2_falg)
