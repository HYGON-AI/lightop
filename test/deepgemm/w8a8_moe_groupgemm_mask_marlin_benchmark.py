import torch
import triton
import pandas as pd
import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
from vllm import _custom_ops as ops
import torch
from lightop import moe_groupgemm_marlin_w4a8
from lightop import m_grouped_w8a8_gemm_nt_masked
from lightop import get_moe_groupgemm_marlin_config_w4a8
import ast  
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)
import argparse
from deepgemm_groupgemm_marlin_quant import weight8bit_nt_kpack2_marlin  # pyright: ignore[reportMissingImports]

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

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

    # print("A shape:", A.shape)
    # print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    # print("C shape after bmm:", C.shape)
    # print("qa1_scale_all shape:", qa1_scale_all.shape)
    # print("w13_scale shape:", w13_scale.shape)
    C = qa1_scale_all * C * w13_scale.transpose(1,2)  # Broadcast per-column scale
    C = C.to(output_dtype)

    return C
           
def deepgemm_masked_impl_w8a8_ep_correctness(hidden_states, weight13, weight2, w13_scale, w2_scale, masked_m, expected_m_per_group):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
    
    outputs = []
    start_idx = 0
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(weight13.shape[0]):
        w1_marlin_in = weight8bit_nt_kpack2_marlin(weight13[i])
        w1_marlin_list.append(w1_marlin_in)
    weight13_marlin = torch.stack(w1_marlin_list, dim=0)
    
    # -------- 处理 gemm2 权重矩阵 ------------------
    # w2_marlin_list = []
    # for i in range(weight2.shape[0]):
    #     w2_marlin_in = weight8bit_nt_kpack2_marlin(weight2[i])
    #     w2_marlin_list.append(w2_marlin_in)
    # weight2_marlin = torch.stack(w2_marlin_list, dim=0)

    w13 = weight13.to(torch.int32)
    w2 = unpack_int8_to_int4(weight2)
    inter_out_gemm1 = []
    inter_out_gemm2 = []  # pyright: ignore[reportUnusedVariable]
    act_out_gemm2 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)


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

    dtype = hidden_states.dtype
    config1, config2, status = get_moe_groupgemm_marlin_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)
        
    problem_size = masked_m
    tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    torch.cuda.synchronize()

    m_grouped_w8a8_gemm_nt_masked((q_a1_all, qa1_scale_all), 
                                  (weight13_marlin, w13_scale),
                                    gemm1_out_cuda,
                                    tensor_problem_size, 
                                    expected_m_per_group,
                                    )
    # torch.cuda.synchronize()
    
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
    return
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
    
    return expert_output



def deepgemm_masked_impl_w8a8_ep_benchmark(hidden_states, weight13, w13_scale, masked_m, expected_m_per_group):
    

    
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(weight13.shape[0]):
        w1_marlin_in = weight8bit_nt_kpack2_marlin(weight13[i])
        w1_marlin_list.append(w1_marlin_in)
    weight13_marlin = torch.stack(w1_marlin_list, dim=0)
    


    w13 = weight13.to(torch.int32)
    inter_out_gemm1 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)


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

    dtype = hidden_states.dtype
        
    # problem_size = masked_m
    # tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    # warm up

    for _ in range(10):
        m_grouped_w8a8_gemm_nt_masked((q_a1_all, qa1_scale_all), 
            (weight13_marlin, w13_scale),
            gemm1_out_cuda,
            masked_m, 
            expected_m_per_group,
            )
    # torch.cuda.synchronize()

    # gemm1
    asm_gemm1_cost = triton.testing.do_bench(
            lambda: m_grouped_w8a8_gemm_nt_masked((q_a1_all, qa1_scale_all), 
            (weight13_marlin, w13_scale),
            gemm1_out_cuda,
            masked_m, 
            expected_m_per_group,
            ), quantiles=None, return_mode="mean")*1000

    
    # torch.cuda.synchronize()
    
    
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
    input = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)

    weight13 = to_int8(torch.randn((E,n ,k), device=device) )

    weight13_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    return input,weight13,weight13_scale,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 创建解析器对象


    
    m_list=4096
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
            input,weight13,weight13_scale,masked_m=_fused_moe_helper(m=m_list,n=n_list[i],
                    k=k_list[i],E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            asm_cost_gemm = deepgemm_masked_impl_w8a8_ep_benchmark(input, weight13, weight13_scale, masked_m, expected_m_per_group)
            deepgemm_asm_costs.append(asm_cost_gemm)
            df_E.append(E)
            df_m.append(m_list)
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
    excel_name="DeepGEMM_masked_groupgemm_w8a8.xlsx"
        
    df.to_excel(excel_name, index=False)

    print("表格已保存到 {}文件中。".format(excel_name)) 

if __name__ == "__main__":
    main()
    #dp2
    #HIP_VISIBLE_DEVICES=3 python moe_groupgemm_w4a8/w4a8_moe_groupgemm_marlin_correctness.py  --E 16 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'
    #dp4
    #HIP_VISIBLE_DEVICES=3 python moe_groupgemm_w4a8/w4a8_moe_groupgemm_marlin_correctness.py  --E 8 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'
