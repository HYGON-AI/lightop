# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import csv

import pytest

from lightop.kvcache import reshape_and_cache_cuda


# -----------------------------------------------------------
# 辅助函数：手动计时
# -----------------------------------------------------------
def benchmark_manual(func, warmup=10, rep=100):
    # 预热
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # 正式计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func()
    end_event.record()
    torch.cuda.synchronize()

    # elapsed_time 返回的是毫秒(ms)
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms / rep) * 1000.0
    return avg_us

# -----------------------------------------------------------
# 生成输入
# 固定测试维度见 FIXED_NUM_HEADS / FIXED_HEAD_DIM
# -----------------------------------------------------------
def reshape_and_cache_torch(key, value, key_cache, value_cache, slot_mapping):
    """PyTorch baseline for the auto cache dtype and unique nonnegative slots.

    Negative slots are padding and leave the cache unchanged. Both caches use
    the lightop layouts: key [B, H, S, D], value [B, H, D, S].
    """
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    block_size = key_cache.shape[2]
    block_ids = slots // block_size
    block_offsets = slots % block_size
    key_cache[block_ids, :, block_offsets, :] = key[valid]
    value_cache[block_ids, :, :, block_offsets] = value[valid]


def make_inputs(T, num_heads=4, head_size=128, dtype=torch.float16):
    device = "cuda"
    block_size = 64
    # Keep enough cache blocks for the largest token case while avoiding the
    # excessive allocation used by the legacy benchmark.
    num_blocks = 4666

    # 这里的 T_min 保持为 1
    T_min = 1

    # 生成 Key 和 Value
    # Shape: [num_tokens, num_heads, head_size]
    key = torch.randn(T, num_heads, head_size, device=device, dtype=dtype) * 0.1
    value = torch.randn_like(key)

    # 生成 Slot Mapping
    slot_mapping = torch.arange(T_min, T_min + T, device=device)

    # 随机设置一些 invalid slot (-1)
    invalid_cnt = max(1, T // 50)
    rand_idx = torch.randperm(T)[:invalid_cnt]
    slot_mapping[rand_idx] = -1

    # PyTorch reference caches use the same layout as lightop.
    key_cache_torch = torch.zeros(
        num_blocks, num_heads, block_size, head_size,
        device=device, dtype=dtype
    )
    value_cache_torch = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        device=device, dtype=dtype
    )

    # lightop key/value cache layout:
    #   key:   [num_blocks, num_heads, block_size, head_size]
    #   value: [num_blocks, num_heads, head_size, block_size]
    key_cache_lightop = torch.zeros(
        num_blocks, num_heads, block_size, head_size,
        device=device, dtype=dtype
    )
    value_cache_lightop = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        device=device, dtype=dtype
    )

    k_scale = torch.tensor([1.0], device=device)
    v_scale = torch.tensor([1.0], device=device)

    return (
        key, value,
        key_cache_torch, value_cache_torch,
        key_cache_lightop, value_cache_lightop,
        slot_mapping,
        k_scale, v_scale,
        head_size, block_size, num_heads
    )


FIXED_NUM_HEADS = 8
FIXED_HEAD_DIM = 128
TS = [8, 16, 32, 64, 88, 128, 256, 356, 420, 512, 1024, 2048, 4096, 8192, 16384, 32768, 41000]


def run_case(T):
    (key, value,
     key_cache, value_cache,
     key_cache2, value_cache2,
     slot_mapping,
     k_scale, v_scale,
     head_size, block_size, num_heads) = make_inputs(
        T,
        num_heads=FIXED_NUM_HEADS,
        head_size=FIXED_HEAD_DIM,
        dtype=torch.bfloat16
    )

    invalid_cnt = (slot_mapping < 0).sum().item()

    # ---------------------
    # Torch kernel
    # ---------------------
    def run_torch():
        reshape_and_cache_torch(
            key, value, key_cache, value_cache, slot_mapping
        )

    # ---------------------
    # custom kernel
    # ---------------------
    def run_custom():
        reshape_and_cache_cuda(
            key, value,
            key_cache2, value_cache2,
            slot_mapping,
            "auto",
            k_scale, v_scale
        )

    # ---------------------
    # 精度检查
    # ---------------------
    # 先运行一次填充数据，确保功能正确
    run_torch()
    run_custom()

    diff_key = (key_cache - key_cache2).abs().max().item()
    diff_value = (value_cache - value_cache2).abs().max().item()
    acc_pass = (diff_key < 1e-3 and diff_value < 1e-3)
    assert acc_pass, (
        f"T={T}: key max diff {diff_key:.4e}, "
        f"value max diff {diff_value:.4e} (阈值 1e-3)"
    )

    time_torch = benchmark_manual(run_torch, warmup=10, rep=1000)
    time_custom = benchmark_manual(run_custom, warmup=10, rep=1000)
    speedup = time_torch / time_custom
    # Effective K/V payload bandwidth, excluding indexing/temporary traffic.
    total_bytes = 4 * (T - invalid_cnt) * num_heads * head_size * key.element_size()
    bw_torch = total_bytes / (time_torch * 1e3)
    bw_custom = total_bytes / (time_custom * 1e3)

    return {
        "T": T,
        "invalid_cnt": invalid_cnt,
        "time_torch": time_torch,
        "time_custom": time_custom,
        "speedup": speedup,
        "bw_torch": bw_torch,
        "bw_custom": bw_custom,
        "num_heads": num_heads,
        "head_size": head_size,
        "block_size": block_size,
    }


# -----------------------------------------------------------
# 测试主流程
# -----------------------------------------------------------
def run_sweep(csv_writer, benchmark_name="Benchmark"):
    print(f"\n======= Running sweep: {benchmark_name} =======")

    print(f"Token list: {TS}")
    print(f"Params: Heads={FIXED_NUM_HEADS}, HeadDim={FIXED_HEAD_DIM}")
    print("Settings: Warmup=10, Rep=1000")

    # -----------------------------------------------------
    # 写入 CSV 表头
    # -----------------------------------------------------
    # 列标题 (增加带宽数据)
    csv_writer.writerow([
        "Token",
        "NumHeads",
        "HeadSize",
        "BlockSize",
        "InvalidCnt",
        "Torch Time (us)",
        "Custom Time (us)",
        "Speedup (Torch / Custom)",
        "Torch BW (GB/s)",   # New
        "Custom BW (GB/s)", # New
        "Accuracy Pass"
    ])

    # -----------------------------------------------------
    # 打印 Console 表头
    # -----------------------------------------------------
    # 定义格式
    header_fmt = "{:<8} {:<10} {:<12} {:<12} {:<10} {:<12} {:<12} {:<8}"
    row_fmt    = header_fmt

    print("-" * 100)
    print(header_fmt.format("Token", "Invalid", "Torch(us)", "Custom(us)", "Speedup", "Torch(GB/s)", "Cus(GB/s)", "Result"))
    print("-" * 100)

    # 用于存储所有用例的加速比
    speedups = []

    # -----------------------------------------------------
    # sweep loop
    # -----------------------------------------------------
    for T in TS:
        result = run_case(T)

        def fmt(value):
            return f"{value:.3f}"

        # 打印一行简略表格信息
        print(row_fmt.format(
            result["T"], result["invalid_cnt"], fmt(result["time_torch"]),
            fmt(result["time_custom"]), fmt(result["speedup"]), fmt(result["bw_torch"]),
            fmt(result["bw_custom"]), "PASS"
        ))

        # 写入 CSV
        csv_writer.writerow([
            result["T"],
            result["num_heads"],
            result["head_size"],
            result["block_size"],
            result["invalid_cnt"],
            result["time_torch"],
            result["time_custom"],
            result["speedup"],
            result["bw_torch"],
            result["bw_custom"],
            "PASS",
        ])
        speedups.append(result["speedup"])

    # -----------------------------------------------------
    # 计算并打印平均加速比
    # -----------------------------------------------------
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print("-" * 100)
        print(f"Average Speedup across {len(speedups)} test cases: {avg_speedup:.4f}x")
        print("-" * 100)


# ======================================================
# 入口：执行测试并写 CSV
# ======================================================
@pytest.mark.parametrize("T", TS)
def test_reshape_cache_cuda(T):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    run_case(T)


if __name__ == "__main__":
    fname = "reshape_and_cache_bench_fixed.csv"
    with open(fname, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # 执行指定配置的 Sweep
        run_sweep(writer, f"H{FIXED_NUM_HEADS}_D{FIXED_HEAD_DIM}")

    print(f"\nCSV saved => {fname}")
