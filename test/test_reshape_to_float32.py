# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from lightop.tensor import reshape_to_float

def torch_baseline(x, vocab_size):
    reshaped = x.reshape(x.shape[0], -1)
    return reshaped[:, :vocab_size].float()

def benchmark(fn, *args, warmup=10, iters=100):
    """
    性能测试框架，返回平均耗时（微秒 us）
    """
    # 1. 预热 (Warmup)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # 2. 测试 (Benchmark)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()

    # elapsed_time 返回的是毫秒 (ms)
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms / iters) * 1000.0
    
    return avg_time_us

def main():
    # 测试参数设置
    B = 256
    world_size = 8
    chunk_size = 18992
    
    # 词表大小，可以模拟切片行为 (比如 150000 略小于 8 * 18992)
    # 如果想测试完全不切片，可以设为 world_size * chunk_size
    vocab_size = 8 * 18992
    
    print(f"=== 测试参数 ===")
    print(f"目标形状: [{B}, {world_size}, {chunk_size}]")
    print(f"目标词表大小: {vocab_size}")
    print(f"数据类型: bfloat16 -> float32\n")

    # 构造不连续的输入张量 (Non-contiguous Tensor)
    # 先按照 [B, chunk_size, world_size] 生成连续内存
    x_raw = torch.randn(B, chunk_size, world_size, dtype=torch.bfloat16, device='cuda')
    # 再将 chunk_size 和 world_size 的维度转置回来，此时逻辑形状对了，但物理内存不再连续
    x = x_raw.transpose(1, 2)
    
    print("=== 输入状态检查 ===")
    print(f"当前输入形状: {x.shape}")
    print(f"当前输入是否连续 (is_contiguous): {x.is_contiguous()}\n") # 这里会打印 False
    # ==============================================

    print("=== 1. 精度验证 ===")
    out_torch = torch_baseline(x, vocab_size)
    out_lightop = reshape_to_float(x, vocab_size)

    # 验证形状和数据类型
    assert out_lightop.shape == out_torch.shape, f"形状不匹配: {out_lightop.shape} vs {out_torch.shape}"
    assert out_lightop.dtype == torch.float32, f"输出类型必须是 float32，但得到了 {out_lightop.dtype}"

    # 验证精度差异
    diff = torch.abs(out_torch - out_lightop).max().item()
    if diff == 0.0:
        print("✅ 精度校验通过！结果完全一致 (Max diff = 0.0)")
    elif diff < 1e-5:
        print(f"✅ 精度校验通过！结果基本一致 (Max diff = {diff})")
    else:
        print(f"❌ 精度校验失败！存在显著差异 (Max diff = {diff})")

    print("\n=== 2. 性能测试 ===")
    # 测原生 PyTorch
    torch_time = benchmark(torch_baseline, x, vocab_size)
    print(f"PyTorch 原生实现: {torch_time:.2f} us")

    # 测自定义 CUDA 算子
    lightop_time = benchmark(reshape_to_float, x, vocab_size)
    print(f"Lightop CUDA算子: {lightop_time:.2f} us")

    # 计算加速比
    speedup = torch_time / lightop_time
    print(f"\n🚀 加速比: {speedup:.2f}x")

if __name__ == "__main__":
    main()