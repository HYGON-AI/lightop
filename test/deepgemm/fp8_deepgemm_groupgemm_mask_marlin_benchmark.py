# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import pandas as pd
import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
import torch
from lightop.moe import moe_groupgemm_marlin_w4a8
from lightop.gemm_ops import m_grouped_fp8_gemm_nt_masked
from lightop.moe import get_moe_groupgemm_marlin_config_w4a8
import ast  
from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)
import argparse
from deepgemm_groupgemm_marlin_quant import weight8bit_nt_kpack2_marlin  # pyright: ignore[reportMissingImports]

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


def native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype):

    A = input.to(torch.float32)
    B = weight.to(torch.float32)

    # print("A shape:", A.shape)
    # print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    # print("C shape after bmm:", C.shape)
    # print("input_scale shape:", input_scale.shape)
    # print("weight_scale shape:", weight_scale.shape)
    C = input_scale * C * weight_scale.transpose(1,2)  # Broadcast per-column scale
    C = C.to(output_dtype)

    return C
           

def deepgemm_masked_impl_fp8_ep_benchmark(input, input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype):
    
    
    weight_marlin = weight8bit_nt_kpack2_marlin(weight)

    inter_out_gemm1 = native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype=out_dtype)

    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)

    torch.cuda.synchronize()

    
    m_grouped_fp8_gemm_nt_masked((input, input_scale), 
                                  (weight_marlin, weight_scale),
                                    gemm1_out_cuda,
                                    masked_m, 
                                    expected_m_per_group,
                                    )
    torch.cuda.synchronize()

    # warm up

    for _ in range(10):
        m_grouped_fp8_gemm_nt_masked((input, input_scale), 
            (weight_marlin, weight_scale),
            gemm1_out_cuda,
            masked_m, 
            expected_m_per_group,
            )
    # torch.cuda.synchronize()

    # gemm1
    asm_gemm1_cost = triton.testing.do_bench(
            lambda: m_grouped_fp8_gemm_nt_masked((input, input_scale), 
            (weight_marlin, weight_scale),
            gemm1_out_cuda,
            masked_m, 
            expected_m_per_group,
            ), quantiles=None, return_mode="mean")*1000
    
    
    if not (torch.allclose(inter_out_gemm1[:,0:math.floor(expected_m_per_group * 0.5),:], gemm1_out_cuda[:,0:math.floor(expected_m_per_group * 0.5),:], rtol=1e-2, atol=1e-2)):
        print("error deepgemm_groupgemm_fp8 gemm 精度检查不合格！！！")
        # torch.set_printoptions(profile="full") 
        print("rocblas:", inter_out_gemm1)
        print("asm:",gemm1_out_cuda)
        # diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
        # diff_indices = diff_indices.view(-1, inter_out_gemm1.shape[1])
        # print("不同元素的索引:", diff_indices)
        # torch.set_printoptions(profile="full")  
        diff_mask = inter_out_gemm1 != gemm1_out_cuda
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print("存在差异的坐标与值如下：")
        count = 0
        for idx in diff_indices:
            count += 1
            if count > 33:
                break
            _, i, j = idx.tolist()
            print(f"坐标 ({_}, {i}, {j}): rocblas={inter_out_gemm1[_, i, j]}, asm={gemm1_out_cuda[_, i, j]}")
    else:
        print("success deepgemm_groupgemm_fp8 gemm 精度检查合格")

    return asm_gemm1_cost


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
    input = torch.randint(-16, 17, (E, m, k), device=device).to(torch.float8_e4m3fn)

    input_scale = (torch.randn((E,m,1), device=device, dtype=torch.float32))

    weight = torch.randint(-16, 17, (E,n ,k),  device=device).to(torch.float8_e4m3fn)

    weight_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(256)
    return input,input_scale,weight,weight_scale,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 创建解析器对象


    
    m=4096
    n_list=[4096, 7168]
    k_list=[7168, 2048]
    
    # m_list=[8192]
    E=16 
    expected_m_per_group_list=[32,64,128,256,512,1024]
    #expected_m_per_group_list = [256]
    out_dtype=torch.bfloat16
    
    # n_list_1=n1_list #[256,12]
    # k_list_1=n2_list #[7168,7168]
    
    # n_list_2=n2_list #[7168,7168]
    # k_list_2=k_list # [128,256]
    # block_size=None

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    deepgemm_asm_costs = []

    df_tflops = []
    df_bandwidth = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(0,len(k_list),1):
            print("expected_m_per_group:{} n:{} k:{} ".format(expected_m_per_group, n_list[i],k_list[i]))
            input,input_scale,weight,weight_scale,masked_m=_fused_moe_helper(m=m, n=n_list[i],
                    k=k_list[i],E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            asm_cost_gemm = deepgemm_masked_impl_fp8_ep_benchmark(input, input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype)
            deepgemm_asm_costs.append(asm_cost_gemm)
            df_E.append(E)
            df_m.append(m)
            df_n.append(n_list[i])
            df_k.append(k_list[i])
            df_expected_m.append(expected_m_per_group)

            valid_m = masked_m.sum().item()
            print("valid_m:", valid_m, "masked_m:", masked_m)
            computes = 2 * valid_m * n_list[i] * k_list[i]
            tflops_asm = computes / (asm_cost_gemm / 1e6) / 1e12

            data_bytes = (valid_m * k_list[i] + k_list[i] * n_list[i] * E + valid_m * n_list[i] * 2)# bfloat16)   
            bandwidth_asm = data_bytes / (asm_cost_gemm / 1e6) / 1e9

            
            print("------------------------------------------------------")
            print("deepgemm asm cost: {:.2f} us, {:.2f} TFlops".format(asm_cost_gemm, tflops_asm))
            print("deepgemm asm bandwidth: {:.2f} GB/s".format(bandwidth_asm))
            print("------------------------------------------------------") 


            df_tflops.append(tflops_asm)
            df_bandwidth.append(bandwidth_asm)


    df = pd.DataFrame({'E':df_E, 'm':df_m,'n':df_n,'k':df_k,'expected_m':df_expected_m,'asm deepgemm 耗时(us)': deepgemm_asm_costs,'asm deepgemm TFlops':df_tflops,'asm deepgemm Bandwidth(GB/s)':df_bandwidth})
        # 将 DataFrame 写入 Excel 文件
    excel_name="DeepGEMM_masked_groupgemm_fp8.xlsx"
        
    df.to_excel(excel_name, index=False)

    print("表格已保存到 {}文件中。".format(excel_name)) 

if __name__ == "__main__":
    main()
    #dp2
    #HIP_VISIBLE_DEVICES=3 python deepgemm_groupgemm_fp8/w4a8_moe_groupgemm_marlin_correctness.py  --E 16 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'
    #dp4
    #HIP_VISIBLE_DEVICES=3 python deepgemm_groupgemm_fp8/w4a8_moe_groupgemm_marlin_correctness.py  --E 8 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'
