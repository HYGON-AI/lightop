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

@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float 
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    if token_num<=32:
        BLOCK_M = 1
        BLOCK_DIM = 512
        NUM_STAGE = 2
        num_warps = 4
        
    elif token_num<=128:
        BLOCK_M = 1
        BLOCK_DIM = 1024
        NUM_STAGE = 0
        num_warps = 2
        
    elif token_num<=4096:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 0
        num_warps = 2
    else:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 2
        num_warps = 8          
        

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


def _test_helper(m: int,
                 topk: int,
                 hidden_size:int,
                 out_dtype: Type[torch.dtype] = torch.float16,
                 device: str = "cuda",
                 best_config:Optional[list] = None):
    # --- 数据准备 (不变) ---
    input = (torch.randn((m,topk, hidden_size), device=device) * 5).to(dtype=out_dtype)
    # 为每个实现准备独立的输出张量，避免相互影响
    output_ops = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_triton = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    output_torch = torch.zeros((m, hidden_size), device=device, dtype=out_dtype)
    
    # --- 计算正确性基准 ---
    # 使用一个未被修改的输出来计算 PyTorch 的基准结果
    output_pytorch_ref = torch.zeros_like(output_ops)
    moe_sum_reduce_torch_compile(input, output_pytorch_ref, 1.0)
    
    # --- 1. 手写 CUDA 算子计时 ---
    # do_bench 自带预热，无需手动循环
    torch.cuda.synchronize()
    elapsed_time0_gpu = triton.testing.do_bench(lambda: ops.moe_sum(input, output_ops), quantiles=None, return_mode="mean") * 1000
    print(f"cuda time: {elapsed_time0_gpu:.4f} us")
    torch.cuda.synchronize()
    
    if not (torch.allclose(output_ops, output_pytorch_ref, rtol=1e-2, atol=5e-2)):
        print("手写算子 精度检查不合格！！！")
    else:
        print("手写算子 精度检查合格")

    # --- 2. Triton 算子计时 ---
    # do_bench 自带预热，无需手动循环
    torch.cuda.synchronize()
    elapsed_time1_gpu = triton.testing.do_bench(lambda: moe_sum_reduce_triton(input, output_triton, 1.0), quantiles=None, return_mode="mean") * 1000
    print(f"triton_time: {elapsed_time1_gpu:.4f} us")
    torch.cuda.synchronize()
    
    if not (torch.allclose(output_triton, output_pytorch_ref, rtol=1e-3, atol=1e-3)):
        print("triton 精度检查不合格！！！")
    else:
        print("triton 精度检查合格")

    # --- 3. torch.compile 计时 (修正后) ---
    # **关键：先调用一次以触发 JIT 编译**
    moe_sum_reduce_torch_compile(input, output_torch, 1.0)
    torch.cuda.synchronize() # **关键：等待编译和首次运行完成**
    
    # 现在进行实际的性能测试
    elapsed_time2_gpu = triton.testing.do_bench(lambda: moe_sum_reduce_torch_compile(input, output_torch, 1.0), quantiles=None, return_mode="mean") * 1000
    print(f"torch_compile_time: {elapsed_time2_gpu:.4f} us\n")
    torch.cuda.synchronize()

    # torch.compile 的精度检查 (它的输出应该和基准完全一致)
    if not (torch.allclose(output_torch, output_pytorch_ref, rtol=1e-3, atol=1e-3)):
        print("torch.compile 精度检查不合格！！！")
    else:
        print("torch.compile 精度检查合格")

    return elapsed_time0_gpu, elapsed_time1_gpu, elapsed_time2_gpu


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