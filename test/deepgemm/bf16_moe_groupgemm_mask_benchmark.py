import torch
import random
import triton
import pandas as pd
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
from vllm import _custom_ops as ops
import torch
from lightop.gemm_ops import m_grouped_bf16_gemm_nt_masked
import ast  
import argparse

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count



def native_bf16_perChannel_batch_matmul(q_a1_all, weight13, output_dtype):

    A = q_a1_all.to(torch.float32)
    B = weight13.to(torch.float32)

    print("A shape:", A.shape)
    print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    print("C shape after bmm:", C.shape)
    C = C.to(output_dtype)
    return C
           
def deepgemm_masked_impl_bf16_correctness(input, weight13, masked_m, expected_m_per_group):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
    
    outputs = []
    start_idx = 0
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    print("weight13:", weight13.shape, weight13.dtype)
    


    inter_out_gemm = []

    
    inter_out_gemm = native_bf16_perChannel_batch_matmul(input, weight13, output_dtype=input.dtype)

    # inter_out_gemm[:, math.floor(expected_m_per_group * 0.7):, :] = 0


    if len(inter_out_gemm) > 0:
        gemm_out_cuda = torch.zeros_like(inter_out_gemm)
        # gemm2_out_cuda = torch.zeros_like(expert_output)
        # act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    E = len(masked_m)
    M = input.size(1)
    N1 = weight13.size(1)
    K1 = input.size(2)//2
    N2 = K1*2
    K2 = N1 // 4

    dtype = input.dtype
        
    # problem_size = masked_m
    # masked_m = torch.tensor(masked_m, dtype=torch.int32, device=input.device)
    torch.cuda.synchronize()
    # print("tensor_problem_size: ", tensor_problem_size)
    # print("tensor_expert_offsets: ", tensor_expert_offsets)
    
    # # gemm1
    print("开始 m_grouped_bf16_gemm_nt_masked ...")
    print("input:", input.shape, input.dtype)
    print("weight:", weight13.shape, weight13.dtype)
    print("gemm_out_cuda:", gemm_out_cuda.shape, gemm_out_cuda.dtype)
    print("masked_m:", masked_m.shape, masked_m.dtype, masked_m)
    print("expected_m_per_group:", expected_m_per_group)
    # torch.set_printoptions(profile="full") 
    print("M:", M, "N2:", N2)
    # row_values = torch.arange(-128, 128).reshape(-1, 1).to(input.device)
    # print("row_values:", row_values)
    # # 广播到所有列
    # q_a1_all = row_values.expand(M, N2).reshape(1, M, N2).to(input.device).to(torch.int8)

    # print("q_a1_all:", q_a1_all)
    # torch.set_printoptions(profile="full") 
    # print("weight13_marlin:", weight13_marlin)
    m_grouped_bf16_gemm_nt_masked(input, 
                                  weight13,
                                    gemm_out_cuda,
                                    masked_m, 
                                    expected_m_per_group,
                                    )
    torch.cuda.synchronize()

    for _ in range(10):
        m_grouped_bf16_gemm_nt_masked(input, 
            weight13,
            gemm_out_cuda,
            masked_m, 
            expected_m_per_group,
            )
    # torch.cuda.synchronize()

    # gemm
    asm_gemm_cost = triton.testing.do_bench(
            lambda: m_grouped_bf16_gemm_nt_masked(input, 
            weight13,
            gemm_out_cuda,
            masked_m, 
            expected_m_per_group,
            ), quantiles=None, return_mode="mean")*1000
    
    if not (torch.allclose(inter_out_gemm[:,0:math.floor(expected_m_per_group * 0.7),:], gemm_out_cuda[:,0:math.floor(expected_m_per_group * 0.7),:], rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm1 精度检查不合格！！！")
        # torch.set_printoptions(profile="full") 
        print("rocblas:", inter_out_gemm)
        print("asm:", gemm_out_cuda)
        # diff_indices = torch.where(inter_out_gemm != gemm_out_cuda)[0]
        # diff_indices = diff_indices.view(-1, inter_out_gemm.shape[1])
        # print("不同元素的索引:", diff_indices)
        # torch.set_printoptions(profile="full")  
        diff_mask = inter_out_gemm != gemm_out_cuda
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print("存在差异的坐标与值如下：")
        count = 0
        for idx in diff_indices:
            count += 1
            if count > 33:
                break
            _, i, j = idx.tolist()
            print(f"坐标 ({_}, {i}, {j}): rocblas={inter_out_gemm[_, i, j]}, asm={gemm_out_cuda[_, i, j]}")
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {gemm_out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success moe_groupgemm_w4a8 gemm1 精度检查合格")
    
    return asm_gemm_cost

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             expected_m_per_group: int = 256,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)
    weight = torch.randn((E, n, k), device=device).to(dtype=out_dtype)

    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(256)
    return input,weight,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    m_list=4096
    n_list=[4096, 7168]
    k_list=[7168, 2048]
    
    # m_list=[8192]
    E=8 
    expected_m_per_group_list=[32,64,128,256,512,1024]
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
            input,weight,masked_m=_fused_moe_helper(m=m_list,n=n_list[i],
                    k=k_list[i], E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            
            # weight13 = torch.ones_like(weight13) 

            # rows = torch.arange(-128, 128).unsqueeze(1)  # shape [256, 1]

            # # 扩展成每行重复 128 次，得到 shape [256, 128]
            # tensor = rows.expand(-1, 128).cuda()
            # weight13 = tensor.to(torch.int8).unsqueeze(0).expand(E, -1, -1)  # shape [1, 256, 128] then expand to [E, 256, 128]
            # print("weight13:", weight13)

            # input = torch.ones_like(input).to(torch.bfloat16)
            # weight = torch.ones_like(weight).to(torch.bfloat16)

            # row_values = torch.arange(0, 64).reshape(1, -1)
            # # # 广播到所有列
            # input = row_values.expand(input.size(1), input.size(2)).reshape(input.size(0), input.size(1), input.size(2)).to(input.device).to(torch.bfloat16)
            # weight = row_values.expand(weight.size(1), weight.size(2)).reshape(weight.size(0), weight.size(1), weight.size(2)).to(weight.device).to(torch.bfloat16)

            print("masked_m:", masked_m)
            print("input:",input.shape, input.dtype)
            print("weight13:",weight.shape,weight.dtype)
            print("masked_m:", masked_m.shape, masked_m.dtype, masked_m)

            
            
            asm_cost_gemm = deepgemm_masked_impl_bf16_correctness(input, weight, masked_m, expected_m_per_group)
            deepgemm_asm_costs.append(asm_cost_gemm)
            df_E.append(E)
            df_m.append(m_list)
            df_n.append(n_list[i])
            df_k.append(k_list[i])
            df_expected_m.append(expected_m_per_group)

            valid_m = masked_m.sum().item()
            computes = 2 * valid_m * n_list[i] * k_list[i]
            tflops_asm = computes / (asm_cost_gemm / 1e6) / 1e12

            data_bytes = (valid_m * k_list[i] + k_list[i] * n_list[i] * E + valid_m * n_list[i]) * 2 # bfloat16)   
            bandwidth_asm = data_bytes / (asm_cost_gemm / 1e6) / 1e9
            print("bandwidth_asm:", bandwidth_asm)

            
            print("------------------------------------------------------")
            print("deepgemm asm cost: {:.2f} us, {:.2f} TFlops, bandwidth: {:.2f} GB/s".format(asm_cost_gemm, tflops_asm, bandwidth_asm))
            print("------------------------------------------------------") 

            df_tflops.append(tflops_asm)
            df_bandwidth.append(bandwidth_asm)


    df = pd.DataFrame({'E':df_E, 'm':df_m,'n':df_n,'k':df_k,'expected_m':df_expected_m,'asm deepgemm 耗时(us)': deepgemm_asm_costs,'asm deepgemm TFlops':df_tflops,'asm deepgemm Bandwidth(GB/s)':df_bandwidth})
        # 将 DataFrame 写入 Excel 文件
    excel_name="DeepGEMM_masked_groupgemm_bf16.xlsx"
        
    df.to_excel(excel_name, index=False)

    print("表格已保存到 {}文件中。".format(excel_name)) 

if __name__ == "__main__":
    main()
    #HIP_VISIBLE_DEVICES=3 python bf16_moe_groupgemm_mask_correctness.py --E 8 --n1-list [4096] --n2-list [7168] --k-list [2048]
    #HIP_VISIBLE_DEVICES=3 python bf16_moe_groupgemm_mask_correctness.py --E 8 --n1-list [4096] --n2-list [7168] --k-list [2048]