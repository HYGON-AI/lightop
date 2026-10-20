# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Optional, Tuple, Union,Type
from . import quant_ops as ops
from ..layers.gemm.int8_utils import matmul_block_int8,per_token_quant_int8,per_token_group_quant_int8
import threading

import time
import json
import triton
import os
import math

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def triton_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             per_token_act_quant: bool,
                             per_out_channel_weight_quant: bool,
                             use_bias: bool,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             device: str = "cuda",
                             best_config:Optional[list] = None,
                             repeat:Optional[int] = 2):

    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)
    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((n_b_scales,1), device=device,
                           dtype=torch.float32))
    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    #warm up repeat次
    for i in range(repeat):
        triton_out=ops.triton_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias,best_config)
    torch.cuda.synchronize()

def triton_blockint8_gemm_helper(m: int,
                                n: int,
                                k: int,
                                block_size:list=[128,128],
                                use_bias: bool=False,
                                out_dtype: Type[torch.dtype] = torch.bfloat16,
                                device: str = "cuda:0",
                                best_config:Optional[dict] = None,
                                repeat:Optional[int] = 2):

    input = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    weight = to_int8(torch.randn((n ,k), device=device) * 5)
    weight_scale = (torch.randn((math.ceil(n/block_size[0]), math.ceil(k/block_size[1])), device=device,
                            dtype=torch.float32))

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])

    for i in range(repeat):
        triton_out=matmul_block_int8(q_input,weight,x_scale,weight_scale,block_size,out_dtype,bias,best_config)
    torch.cuda.synchronize()
