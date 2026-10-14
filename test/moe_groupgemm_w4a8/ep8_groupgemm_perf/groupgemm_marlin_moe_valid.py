# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import pdb
import torch
from typing import Any, Dict, List, Tuple, TypedDict

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)

from w4a8_moe_groupgemm_marlin_correctness import native_w8a8_per_token_matmul, unpack_int8_to_int4
import torch.nn.functional as F
import itertools

from lightop.moe import moe_groupgemm_marlin_w4a8
from lightop.moe import get_moe_groupgemm_marlin_config_w4a8

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

def groupgemm_marlin_moe_valid(
                     hidden_states: torch.Tensor, # 32, 7168 
                     weight13: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     weight2: torch.Tensor, # 256, 256, 7168
                     weight13_marlin: torch.Tensor,
                     weight2_marlin: torch.Tensor,
                     w13_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     tokens_per_expert: torch.Tensor,
                     ) -> torch.Tensor:
    assert hidden_states.shape[1] == weight13_marlin.shape[1] * 32, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == weight2_marlin.shape[2] // 4, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert weight13_marlin.is_contiguous(), "Expert weights1 must be contiguous"
    assert weight2_marlin.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前只支持 bf16
    assert hidden_states.dtype in [torch.bfloat16]

    outputs = []
    start_idx = 0
    w13 = unpack_int8_to_int4(weight13)
    w2 = unpack_int8_to_int4(weight2)
    inter_out_gemm1 = []
    act_out_gemm2 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # q_a1, qa1_scale = per_token_quant_int8(hidden_states[start_idx:end_idx])
        q_a1 = q_a1_all[start_idx:end_idx]
        qa1_scale = qa1_scale_all[start_idx:end_idx]

        current_w13 = w13[i]
        current_w2 = w2[i]
        current_w13_s = w13_scale[i]
        current_w2_s = w2_scale[i]
        inter_out = native_w8a8_per_token_matmul(
            q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype
        )
        inter_out_gemm1.append(inter_out)
        d = inter_out.shape[-1] // 2
        act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
        act_out_gemm2.append(act_out)
        act_out_q, act_out_s = per_token_quant_int8(act_out)

        expert_out= native_w8a8_per_token_matmul(
            act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype
        )
        outputs.append(expert_out)
        start_idx = end_idx
    if len(outputs) > 0:
        expert_output = torch.cat(outputs, dim=0)
    else:
        assert hidden_states.numel() == 0, f"sorted_tokens: should be empty, but got {hidden_states.shape}"
        expert_output = hidden_states

    if len(inter_out_gemm1) > 0:
        inter_out_gemm1_out_ref = torch.cat(inter_out_gemm1, dim=0)
        act_out_cat = torch.cat(act_out_gemm2, dim=0)
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1_out_ref)
        gemm2_out_cuda = torch.zeros_like(expert_output)
        act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    E = len(tokens_per_expert)
    M = hidden_states.size(0)
    N1 = weight13.size(1)
    K1 = hidden_states.size(1)//2
    N2 = K1*2
    K2 = N1 // 4

    dtype = hidden_states.dtype
    config1, config2, status = get_moe_groupgemm_marlin_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)
        
    problem_size = tokens_per_expert
    expert_offsets = list(itertools.accumulate(tokens_per_expert, initial=0))[:-1]
    assert len(expert_offsets) == len(problem_size)
    tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    tensor_expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=hidden_states.device)
    torch.cuda.synchronize()
    # print("tensor_problem_size: ", tensor_problem_size)
    # print("tensor_expert_offsets: ", tensor_expert_offsets)
    
    # # gemm1
    moe_groupgemm_marlin_w4a8(q_a1_all, 
                                weight13_marlin, 
                                gemm1_out_cuda,
                                qa1_scale_all, 
                                w13_scale,
                                tensor_problem_size, 
                                tensor_expert_offsets,
                                config1)
    torch.cuda.synchronize()
    
    if not (torch.allclose(inter_out_gemm1_out_ref, gemm1_out_cuda, rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm1 精度检查不合格！！！")
        torch.set_printoptions(profile="full")  
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {gemm1_out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success moe_groupgemm_w4a8 gemm1 精度检查合格")
    
    # gemm2
    moe_groupgemm_marlin_w4a8(act_out_q, 
                                weight2_marlin, 
                                gemm2_out_cuda,
                                act_out_s, 
                                w2_scale,
                                tensor_problem_size, 
                                tensor_expert_offsets,
                                config2)
    if not (torch.allclose(expert_output, gemm2_out_cuda, rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm2 精度检查不合格！！！")
        # torch.set_printoptions(profile="full")  
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success moe_groupgemm_w4a8 gemm2 精度检查合格")