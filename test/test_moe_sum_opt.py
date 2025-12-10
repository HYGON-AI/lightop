import torch
import time
from lmslim import quant_ops 
from typing import Optional, Type
import pandas as pd
import os
import json
import triton
import triton.language as tl
import pdb
from lightop import op as ops

def moe_sum_reduce_torch_compile(x, out):
    torch.sum(x, dim=1, out=out)

def _test_helper(m: int,
                 topk: int,
                 hidden_size:int,
                 out_dtype: Type[torch.dtype] = torch.bfloat16,
                 device: str = "cuda",
                 best_config:Optional[list] = None):
    # 数据准备
    input = (torch.randn((m,topk, hidden_size), device=device) * 5).to(dtype=out_dtype)
    output_ops = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_triton = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_torch = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    factor = 2.5
    
    output_pytorch_ref = torch.zeros_like(output_ops)
    moe_sum_reduce_torch_compile(input, output_pytorch_ref)
    

    bias = torch.randn_like(output_ops)
    
    num_local_tokens = [m]
    num_local_tokens_tensor = torch.tensor(num_local_tokens, device=device, dtype=torch.int32)
    
    # if out_dtype == torch.bfloat16:
    #     # bfloat16: output = sum_result * 2.5 + bias
    #     output_pytorch_ref.mul_(factor).add_(bias)
    # elif out_dtype == torch.float16:
    #     factor_inverse = 1.0 / factor
    #     # float16: output = sum_result + (bias * 2.5)
    #     output_pytorch_ref.add_(bias.mul(factor_inverse))
      
    # --- 1. 精度检查 ---
    output_ops.zero_()
    ops.moe_sum(input, output_ops, None, None, None, factor)
    torch.cuda.synchronize()
    # pdb.set_trace()
    if not (torch.allclose(output_ops, output_pytorch_ref, rtol=1e-2, atol=1e-2)):
        print(f"❌❌ 手写算子 精度检查不合格! (m={m})")
    else:
        print(f"✅✅ 手写算子 精度检查合格! (m={m})")

    # --- 2. 预热 (10轮) ---
    for _ in range(10):
        ops.moe_sum(input, output_ops, None, None, None, factor)
    torch.cuda.synchronize()

    # --- 3. 性能测试 (100轮) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        ops.moe_sum(input, output_ops, None, None, None, factor)
    end_event.record()
    torch.cuda.synchronize()

    # 计算平均耗时 (elapsed_time 返回的是毫秒 ms，转换为微秒 us)
    total_time_ms = start_event.elapsed_time(end_event)
    elapsed_time0_gpu = (total_time_ms / 100) * 1000
    print(f"cuda time: {elapsed_time0_gpu:.4f} us")

    return elapsed_time0_gpu


def main():
    topk = 8
    hidden_size = 2048
    out_dtype = torch.bfloat16
    bs = []
    topks = []
    hidden_sizes = []
    
    vllm_reduce_cost_gpu = []
    triton_reduce_cost_gpu = []
    torch_reduce_cost_gpu = []
    print(f"Testing dtype: {out_dtype}") 
    
    # m_list = [1]
    #deepseek 带bias
    # m_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384]
    #qwen-30b  不带bias
    m_list =  [1,2,5,7,16,32,64,128,256,323,512,1024,2048,4096,8192,12800]
    for m in m_list:
        print("-" * 30)
        print("Batch size m: {}".format(m))
    
        elapsed_time0_gpu  = _test_helper(
            m=m, topk=topk, hidden_size=hidden_size, out_dtype=out_dtype, best_config=None
        )
        
        bs.append(m)
        topks.append(topk)
        hidden_sizes.append(hidden_size)
        
        vllm_reduce_cost_gpu.append(elapsed_time0_gpu)
    

    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({
        'm': bs,
        'topk': topks,
        'hidden_size': hidden_sizes,
        '耗时(us)': vllm_reduce_cost_gpu,
    })

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('output_reduce.xlsx', index=False)
    print("\nBenchmark finished. Results saved to output_reduce.xlsx")

if __name__ == "__main__":
    main()