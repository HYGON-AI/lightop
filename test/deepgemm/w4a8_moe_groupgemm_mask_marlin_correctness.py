import torch
import random
import time
import pandas as pd
import functools
import json
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
import torch
from lightop.gemm_ops import m_grouped_w4a8_gemm_nt_masked
import ast  
from lightop.quant import per_token_quant_int8
from w4a8_moe_groupgemm_marlin_quant import w4a8_2_marlin_weight  # pyright: ignore[reportMissingImports]

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    E,N, K_half = tensor_int8.shape
    tensor_int8 = tensor_int8.to(torch.int8)
    high4 = tensor_int8 & 0XF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((E, N, K_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4.to(torch.int8)
    unpacked[:, :, 1::2] = low4.to(torch.int8)

    return unpacked


def native_w8a8_perChannel_batch_matmul(q_a1_all, weight13, qa1_scale_all, w13_scale, output_dtype):

    A = q_a1_all.to(torch.float32)
    B = weight13.to(torch.float32)

    print("A shape:", A.shape)
    print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    print("C shape after bmm:", C.shape)
    print("qa1_scale_all shape:", qa1_scale_all.shape)
    print("w13_scale shape:", w13_scale.shape)
    C = qa1_scale_all * C * w13_scale.transpose(1,2)  # Broadcast per-column scale
    C = C.to(output_dtype)

    return C
           
def deepgemm_masked_impl_w4a8_ep_correctness(hidden_states, weight13, w13_scale, masked_m, expected_m_per_group):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
    
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(weight13.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(weight13[i])
        w1_marlin_list.append(w1_marlin_in)
    weight13_marlin = torch.stack(w1_marlin_list, dim=0)
    


    w13 = unpack_int8_to_int4(weight13)
    inter_out_gemm1 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)
    # print("q_a1_all:", q_a1_all.shape, qa1_scale_all.shape)
    # torch.set_printoptions(profile="full")
    # q_a1_all = torch.ones_like(q_a1_all)
    # q_a1_all[:, 128:192] = 0
    # print("q_a1_all:",q_a1_all.shape,q_a1_all)
    # qa1_scale_all = torch.ones_like(qa1_scale_all)

    inter_out_gemm1 = native_w8a8_perChannel_batch_matmul(q_a1_all, w13, qa1_scale_all, w13_scale, output_dtype=hidden_states.dtype)

    inter_out_gemm1[:, math.floor(expected_m_per_group * 0.7):, :] = 0

    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)
        # gemm2_out_cuda = torch.zeros_like(expert_output)
        # act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    E = len(masked_m)
    M = hidden_states.size(1)
    N1 = weight13.size(1)
    K1 = hidden_states.size(2)//2
    N2 = K1*2
    K2 = N1 // 4

        
    problem_size = masked_m
    tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    torch.cuda.synchronize()
    # print("tensor_problem_size: ", tensor_problem_size)
    # print("tensor_expert_offsets: ", tensor_expert_offsets)
    
    # # gemm1
    print("开始 m_grouped_w4a8_gemm_nt_masked gemm1...")
    print("q_a1_all:", q_a1_all.shape, q_a1_all.dtype)
    print("weight13_marlin:", weight13_marlin.shape, weight13_marlin.dtype)
    print("gemm1_out_cuda:", gemm1_out_cuda.shape, gemm1_out_cuda.dtype)
    print("qa1_scale_all:", qa1_scale_all.shape, qa1_scale_all.dtype)
    print("w13_scale:", w13_scale.shape, w13_scale.dtype)
    # torch.set_printoptions(profile="full") 
    print("M:", M, "N2:", N2)
    # row_values = torch.arange(-128, 128).reshape(-1, 1).to(hidden_states.device)
    # print("row_values:", row_values)
    # # 广播到所有列
    # q_a1_all = row_values.expand(M, N2).reshape(1, M, N2).to(hidden_states.device).to(torch.int8)

    # print("q_a1_all:", q_a1_all)
    m_grouped_w4a8_gemm_nt_masked((q_a1_all, qa1_scale_all), 
                                  (weight13_marlin, w13_scale),
                                    gemm1_out_cuda,
                                    tensor_problem_size, 
                                    expected_m_per_group,
                                    )
    torch.cuda.synchronize()
    
    if not (torch.allclose(inter_out_gemm1[:,0:math.floor(expected_m_per_group * 0.7),:], gemm1_out_cuda[:,0:math.floor(expected_m_per_group * 0.7),:], rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm1 精度检查不合格！！！")
        print("rocblas:", inter_out_gemm1)
        # torch.set_printoptions(profile="full") 
        print("asm:",gemm1_out_cuda)
        diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
        diff_indices = diff_indices.view(-1, inter_out_gemm1.shape[1])
        print("不同元素的索引:", diff_indices)
        # torch.set_printoptions(profile="full")  
        # diff_mask = inter_out_gemm1_out_ref != gemm1_out_cuda
        # diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        # print("存在差异的坐标与值如下：")
        # for idx in diff_indices:
        #     i, j = idx.tolist()
        #     print(f"坐标 ({i}, {j}): rocblas={inter_out_gemm1_out_ref[i, j]}, asm={gemm1_out_cuda[i, j]}")
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {gemm1_out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success moe_groupgemm_w4a8 gemm1 精度检查合格")
    
    return gemm1_out_cuda

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             expected_m_per_group: int = 256,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)

    weight13 = to_int8(torch.randn((E, n ,k // 2), device=device) )

    weight13_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(256)
    return input,weight13,weight13_scale,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    m_list=[4096]
    E=8
    expected_m_per_group=1024
    out_dtype=torch.bfloat16
    
    n_list=[4096, 7168] #[256,512]
    k_list=[7168, 2048] #[7168,7168]
    
    
    for i in range(0, len(k_list), 1):

        for m in m_list:
            print("m:{} expected_m:{} n:{} k:{} ".format(m, expected_m_per_group, n_list[i], k_list[i]))
            input,weight13,weight13_scale,masked_m=_fused_moe_helper(m=m,n=n_list[i],
                    k=k_list[i], E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            print("masked_m:", masked_m)
            print("input:",input.shape, input.dtype)
            print("weight13:",weight13.shape,weight13.dtype)
            print("weight13_scale:",weight13_scale.shape,weight13_scale.dtype)
            print("masked_m:", masked_m.shape, masked_m.dtype, masked_m)

            deepgemm_masked_impl_w4a8_ep_correctness(input,weight13, weight13_scale, masked_m, expected_m_per_group)

if __name__ == "__main__":
    main()

