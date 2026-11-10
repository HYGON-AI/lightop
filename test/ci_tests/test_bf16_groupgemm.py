# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
from typing import Optional, Type
import pandas as pd
import os
from lightop.gemm_ops import grougemm_bf16_fp16
import torch
import random

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def cblas_gemm_torch(
    A: torch.Tensor,  # int8, shape: [batch, m, k]
    B: torch.Tensor,  # int8, shape: [batch, n, k]
    C: torch.Tensor,  # bf16, shape: [batch, m, n]
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    batch, m, k = A.shape
    batch_b, n, k_b = B.shape
    batch_c, m_c, n_c = C.shape
    
    assert batch == batch_b == batch_c, "Batch sizes must match"
    assert k == k_b, "K dimensions must match"
    assert m == m_c, "M dimensions must match"
    
    # 将 int8 转换为 float32 进行计算
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    
    # 将 bf16 转换为 float32 进行计算
    C_fp32 = C.to(torch.float32)
    
    # 执行矩阵乘法: sum = A @ B^T
    # A: [batch, m, k], B: [batch, n, k]
    # 需要计算 A @ B^T，所以先转置 B: [batch, k, n]
    # 结果: [batch, m, n]
    B_transposed = B_fp32.transpose(-2, -1)  # [batch, k, n]
    sum_fp32 = torch.bmm(A_fp32, B_transposed)  # [batch, m, n]
    
    # 应用 scaleA 和 scaleB
    # scaleA: [batch] -> [batch, 1, 1]
    # scaleB: [batch] -> [batch, 1, 1]
    
    # 应用 alpha, beta
    val_fp32 = alpha * sum_fp32 + beta * C_fp32
    
    # 转换为 bf16
    D = val_fp32.to(torch.bfloat16)
    
    return D


def cblas_gemm_torch_naive(
    A: torch.Tensor,  # int8, shape: [batch, m, k]
    B: torch.Tensor,  # int8, shape: [batch, n, k]
    C: torch.Tensor,  # bf16, shape: [batch, m, n]
    scaleA: torch.Tensor,  # float, shape: [batch]
    scaleB: torch.Tensor,  # float, shape: [batch]
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    batch, m, k = A.shape
    batch_b, n, k_b = B.shape
    batch_c, m_c, n_c = C.shape
    
    assert batch == batch_b == batch_c, "Batch sizes must match"
    assert k == k_b, "K dimensions must match"
    assert m == m_c, "M dimensions must match"
    assert n == n_c, "N dimensions must match"
    
    
    # 转换为 float32 进行计算
    A_fp32 = A.to(torch.float32).to("cpu")
    B_fp32 = B.to(torch.float32).to("cpu")
    C_fp32 = C.to(torch.float32).to("cpu")
    scaleA = scaleA.to("cpu")
    scaleB = scaleB.to("cpu")
    D = torch.zeros((batch, m, n), dtype=torch.bfloat16, device=A_fp32.device)
    # 按照 C++ 代码的循环逻辑
    for b in range(batch):
        for m_idx in range(m):
            for n_idx in range(n):
                # 计算 sum = sum(A[m, k] * B[n, k]) for k in range(k)
                sum_val = torch.tensor(0.0, dtype=torch.float32)
                for k_idx in range(k):
                    sum_val = sum_val + A_fp32[b, m_idx, k_idx] * B_fp32[b, n_idx, k_idx]
                
                # 应用 scaleA 和 scaleB
                sum_val = sum_val * scaleA[b] * scaleB[b]
                
                # 计算最终值
                val = alpha * sum_val + beta * C_fp32[b, m_idx, n_idx]
                
                # 转换为 bf16 并存储
                D[b, m_idx, n_idx] = val.to(torch.bfloat16)
    D = D.to(A.device)
    
    return D





if __name__ == "__main__":
    device = "cuda"
    group_nums = 8
    
    m1=[item if item < 17 else  1 << (item - 27) for item in range(1, 17)]
    m2=[item<<2  if item <17 else (item - 8)<<3  for item in range(5, 29)]
    m3=[2<<(item)  for item in range(7, 13)]

    m_list=[256, 256, 384, 512, 640, 768, 896, 1024, 2048]
    n_list=[256, 256, 384, 512, 640, 768, 896, 1024, 2048]
    k_list=[64,  512, 1024,7168, 78, 7168,1536,7168, 1152]
    batch = 4
    alphas = [1.0 for _ in range(group_nums)]
    betas = [0.0 for _ in range(group_nums)]

    # NT
    transa = False
    transb = True

    out_dtype = torch.bfloat16
    group_A = []
    group_B = []
    #group_C = []
    group_D = []
    group_D_ref = []
    for i in range(group_nums):
        list_size = len(m_list)
        max_index = list_size - 1
        index = random.randint(0, max_index-1)
        # index = 0
        m, n, k = m_list[index], n_list[index], k_list[index]
        print(f"m : {m}, n : {n}, k : {k}\n")
        a = torch.randn((batch, m, k), device=device, dtype=torch.bfloat16)
        b = torch.randn((batch, n, k), device=device, dtype=torch.bfloat16)

        # a = torch.ones((batch, m, k), device=device, dtype=torch.bfloat16)
        # b = torch.ones((batch, n, k), device=device, dtype=torch.bfloat16)


        res_shape = (batch, m, n)
        d = torch.zeros(res_shape, device=device, dtype=torch.bfloat16)
        c = torch.zeros(res_shape, device=device, dtype=torch.bfloat16) 
        group_D_ref.append(cblas_gemm_torch(a, b, c, 1.0, 0.0))
        group_A.append(a)
        group_B.append(b)
        group_D.append(d)
    
    group_D_res = grougemm_bf16_fp16(group_A, group_B, group_D, alphas, None, betas, out_dtype, transa, transb)
    # torch.cuda.synchronize()
    # group_D_res = grougemm_bf16_fp16(group_A, group_B, group_D, alphas, None, betas, out_dtype, transa, transb)
    # group_D_res = grougemm_bf16_fp16(group_A, group_B, group_D, alphas, None, betas, out_dtype, transa, transb)
    # group_D_res = grougemm_bf16_fp16(group_A, group_B, group_D, alphas, None, betas, out_dtype, transa, transb)
    # group_D_res = grougemm_bf16_fp16(group_A, group_B, group_D, alphas, None, betas, out_dtype, transa, transb)
    torch.cuda.synchronize()
    for i in range(group_nums):
        if not torch.allclose(group_D_ref[i], group_D_res[i], rtol=1e-2, atol=5e-2):
            print("精度检查不合格！！！") 
            print("group_D_ref:", group_D_ref[i])
            print("group_D_res:", group_D_res[i])
        else:
            print("精度检查合格！！！")

