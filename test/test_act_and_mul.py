# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import torch.nn.functional as F
import time
from lightop.activation import (
    gelu_and_mul_opt,
    gelu_tanh_and_mul_opt,
    silu_and_mul_opt,
)
from vllm import _custom_ops as vllm_ops

def benchmark_op(func, output_tensor, input_tensor, warmup=10, runs=100):
    # 预热
    for _ in range(warmup):
        func(output_tensor, input_tensor)
    torch.cuda.synchronize()
    
    # 性能测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        func(output_tensor, input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    
    # us
    avg_time_us = (start_event.elapsed_time(end_event) * 1000) / runs
    return avg_time_us

def main():
    parser = argparse.ArgumentParser(description="Benchmark custom activation and multiplication ops.")
    parser.add_argument(
        "--act", 
        type=str, 
        choices=["silu", "gelu", "gelu_tanh"], 
        default="silu",
        help="选择要测试的激活函数类型 (默认: silu)"
    )
    args = parser.parse_args()

    if args.act == "silu":
        baseline_func = torch.ops._C.silu_and_mul
        custom_func = silu_and_mul_opt
    elif args.act == "gelu":
        baseline_func = torch.ops._C.gelu_and_mul
        custom_func = gelu_and_mul_opt
    elif args.act == "gelu_tanh":
        baseline_func = torch.ops._C.gelu_tanh_and_mul
        custom_func = gelu_tanh_and_mul_opt

  
    dtype = torch.bfloat16
    device = "cuda"
    d = 4096
    num_tokens_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    print(f"开始验证 {args.act}_and_mul 精度与性能测试...\n")
    print("| num_tokens | d | 精度验证 | 基准耗时 (us) | 自定义算子耗时 (us) | 相对提升 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for num_tokens in num_tokens_list:
        # input 最后维度是 2*d
        input_tensor = torch.randn((num_tokens, 2 * d), dtype=dtype, device=device)
        output_baseline = torch.empty((num_tokens, d), dtype=dtype, device=device)
        output_custom = torch.empty((num_tokens, d), dtype=dtype, device=device)
        
        # 1. 精度验证
        baseline_func(output_baseline, input_tensor)
        custom_func(output_custom, input_tensor)
        
        is_correct = torch.allclose(output_baseline, output_custom, atol=1e-2, rtol=1e-2)
        acc_str = "✅ Pass" if is_correct else "❌ Fail"
        
        if not is_correct:
            max_diff = torch.max(torch.abs(output_baseline - output_custom)).item()
            acc_str += f" (Max diff: {max_diff:.4f})"
        
        # 2. 性能测试
        baseline_time = benchmark_op(baseline_func, output_baseline, input_tensor)
        custom_time = benchmark_op(custom_func, output_custom, input_tensor)
        
        # 计算加速比
        speedup = baseline_time / custom_time if custom_time > 0 else 0
        
        # 打印表格行
        print(f"| {num_tokens:<10} | {d:<4} | {acc_str:<8} | {baseline_time:<12.2f} | {custom_time:<17.2f} | {speedup:<7.2f}x |")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要可用的 CUDA 环境来运行此测试"
    main()