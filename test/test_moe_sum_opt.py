import torch
import time
from lmslim import quant_ops 
from typing import Optional, Type
import pandas as pd
import os
import json
import triton
import triton.language as tl

from lightop import op as ops

@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def _test_helper(m: int,
                 topk: int,
                 hidden_size:int,
                 out_dtype: Type[torch.dtype] = torch.float16,
                 device: str = "cuda",
                 best_config:Optional[list] = None):
    input = (torch.randn((m,topk, hidden_size), device=device) * 5).to(dtype=out_dtype)
    # 为每个实现准备独立的输出张量，避免相互影响
    output_ops = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_triton = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_torch = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    
    # 使用一个未被修改的输出来计算 PyTorch 的基准结果
    output_pytorch_ref = torch.zeros_like(output_ops)
    moe_sum_reduce_torch_compile(input, output_pytorch_ref, 1.0)
    bias = torch.randn_like(output_ops)

    if out_dtype == torch.bfloat16:
        # bfloat16: output = sum_result * 2.5 + bias
        output_pytorch_ref.mul_(2.5).add_(bias)
    elif out_dtype == torch.float16:
        # float16: output = sum_result + (bias * 2.5)
        output_pytorch_ref.add_(bias.mul(2.5))
     
    # --- 1. 手写 CUDA 算子计时 ---
    # do_bench 自带预热，无需手动循环
    torch.cuda.synchronize()
    elapsed_time0_gpu = triton.testing.do_bench(lambda: ops.moe_sum(input, output_ops,bias), quantiles=None, return_mode="mean") * 1000
    print(f"cuda time: {elapsed_time0_gpu:.4f} us")
    torch.cuda.synchronize()
    
    if not (torch.allclose(output_ops, output_pytorch_ref, rtol=1e-2, atol=5e-2)):
        print("手写算子 精度检查不合格！！！")
    else:
        print("手写算子 精度检查合格")

    return elapsed_time0_gpu, 0, 0


def main():
    topk=8
    hidden_size =7168
    out_dtype = torch.float16
    bs=[]
    topks=[]
    hidden_sizes=[]
    
    vllm_reduce_cost_gpu=[]
    triton_reduce_cost_gpu=[]
    torch_reduce_cost_gpu=[]
    
    m_list =[1<<(item)  for item in range(0, 17)]
    # m_list =[128,256]
    for m in m_list:
        print("m:{} ".format(m))
    
        elapsed_time0_gpu,elapsed_time1_gpu,elapsed_time2_gpu =_test_helper(m=m,topk=topk,hidden_size=hidden_size,out_dtype=out_dtype,best_config=None)
        bs.append(m)
        topks.append(topk)
        hidden_sizes.append(hidden_size)
        
        vllm_reduce_cost_gpu.append(elapsed_time0_gpu)
        triton_reduce_cost_gpu.append(elapsed_time1_gpu)
        torch_reduce_cost_gpu.append(elapsed_time2_gpu)

    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({'m':bs,'topk':topks,'hidden_size':hidden_sizes,'vllm moe_reduce GPU耗时': vllm_reduce_cost_gpu,'triton moe_reduce GPU耗时': triton_reduce_cost_gpu,'torch moe_reduce GPU耗时': torch_reduce_cost_gpu})

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('output_reduce.xlsx', index=False)

if __name__ == "__main__":
    main()