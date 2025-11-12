import torch
import time
from lmslim import quant_ops 
from typing import Optional, Type
import pandas as pd
import os
import json
from lightop import m_grouped_w8a8_gemm_nt_contig_asm
import torch
import random

def align(x, y):
    return ((x + y - 1) // y) * y

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    #每个group实际的M
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    #将M对齐到128
    aligned_ms = [align(actual_m, 256) for actual_m in actual_ms]
    #group的M的总和
    m = sum(aligned_ms)
    
    # a 是整体input
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)  
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        #表示M属于哪个group id
        m_indices[start:actual_end] = i        #[000...-1-1-1, 111...-1-1]           
        m_indices[actual_end:aligned_end] = -1

        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)
    a_int8 = to_int8(a)
    b_int8 = to_int8(b)
    a_scale = torch.randn(m, device='cuda', dtype=torch.float)
    b_scale = torch.randn((num_groups, n), device='cuda', dtype=torch.float)

    group_a = [a_int8, a_scale]
    group_b = [b_int8, b_scale]
    return m, group_a, group_b, m_indices, d, ref_d


def test_m_grouped_gemm_contiguous() -> None:
    num_groups = 16
    expected_m_per_group = 1024
    n = 512
    k = 512

    m, a, b, m_indices,d,ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k)

    d = m_grouped_w8a8_gemm_nt_contig_asm(a, b, d, m_indices)

test_m_grouped_gemm_contiguous()

