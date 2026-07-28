import torch
import time
import pandas as pd
from typing import Tuple
from lightop.tensor import fill_zeros

def test_correctness(shape: Tuple[int, int], dtype=torch.float32):
    print(f"\n[Correctness Test] shape={shape}, dtype={dtype}")

    #  fill_zeros
    out_custom = fill_zeros(shape, dtype, device="cuda")

    # PyTorch 内置 zeros
    out_torch = torch.zeros(shape, dtype=dtype, device="cuda")

    # 比较最大绝对误差
    max_diff = (out_custom - out_torch).abs().max().item()
    print(f"  Max abs diff: {max_diff:.6f}")
    assert max_diff == 0, "❌ Fill zeros does not match torch.zeros!"
    print("  ✅ Correctness passed!")


def benchmark(shape: Tuple[int, int], dtype=torch.float32, warmup=10, repeat=100):
    print(f"\n[Performance Test] shape={shape}, dtype={dtype}")

    out = torch.empty(shape, device="cuda", dtype=dtype)

    # 预热
    for _ in range(warmup):
        fill_zeros(shape, dtype, device="cuda")
        torch.cuda.synchronize()

    # 测自定义 kernel时间
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fill_zeros(shape, dtype, device="cuda")
    torch.cuda.synchronize()
    t1 = time.time()
    custom_time = (t1 - t0) / repeat * 1e6  # us
    print(f" 优化后: {custom_time:.2f} us")

    # 测 torch.zeros
    for _ in range(warmup):
        torch.zeros(shape, dtype=dtype, device="cuda")
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        torch.zeros(shape, dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    t1 = time.time()
    torch_time = (t1 - t0) / repeat * 1e6  # us
    print(f"原始torch:       {torch_time:.2f} us")

    print(f"  加速比 (torch/custom): {torch_time/custom_time:.2f}x")

    return {
        "batch_size": shape[0],
        "hidden_size": shape[1],
        "dtype": dtype,
        "custom_avg_time_us": custom_time,
        "torch_avg_time_us": torch_time,
        "speedup_ratio": torch_time / custom_time
    }

# 将性能测试结果写入 Excel
def save_results_to_excel(results, filename="performance_results.xlsx"):
    # 将测试结果转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 保存到 Excel 文件
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(0)

    # 用于收集所有测试结果
    all_results = []

    # 测试不同 hidden size
    for hs in [1024, 2048, 4096, 8192, 16384]:
        for bs in [32, 64, 128, 256, 1000]:
            # test_correctness((bs, hs), dtype=torch.float32)
            test_correctness((bs, hs), dtype=torch.float16)
            test_correctness((bs, hs), dtype=torch.bfloat16)

            # 性能测试并收集结果
            result = benchmark((bs, hs), dtype=torch.float16)
            all_results.append(result)

            result = benchmark((bs, hs), dtype=torch.bfloat16)
            all_results.append(result)

    # 保存所有测试结果到 Excel 文件
    save_results_to_excel(all_results)
