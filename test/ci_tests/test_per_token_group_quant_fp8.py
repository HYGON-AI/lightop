# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
import time
import csv
import pytest
from lightop.quant import per_token_group_quant_fp8

@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
    y_ptr += y_ptr_offset
    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_fp8(x, group_size, eps=1e-5, use_ue8m0=False):
    """简化的 Triton Baseline 封装"""
    # 强制 e4m3fn 配置
    fp8_max = 448.0
    fp8_min = -448.0
    dtype = torch.float8_e4m3fn

    x_q = torch.empty_like(x, dtype=dtype)
    shape_s = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape_s, device=x.device, dtype=torch.float32)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_group_quant_fp8[(M,)](
        x, x_q, x_s, group_size, x.shape[1], x.stride(0),
        eps, fp8_min, fp8_max, use_ue8m0,
        BLOCK=BLOCK, num_warps=num_warps, num_stages=1
    )
    return x_q, x_s


NUM_TOKENS_LIST = [65536]


def run_case(num_tokens, headdim=128, group_size=128, eps=1e-5, use_ue8m0=False):
    warmup_iters = 10
    bench_iters = 100

    # 初始化张量
    x = torch.randn((num_tokens, headdim), dtype=torch.bfloat16, device='cuda')

    # 1. 精度验证
    # Baseline (Triton)
    triton_q, triton_s = triton_per_token_group_quant_fp8(x, group_size, eps, use_ue8m0)

    # Custom Op (LightOp)
    custom_q, custom_s = per_token_group_quant_fp8(
        x, group_size, eps=eps
    )

    # 对比 (Scale 要求误差很小, FP8 可以通过查看 uint8 值是否一致或者只差 1 ULP)
    scale_match = torch.allclose(triton_s, custom_s, atol=1e-5, rtol=1e-5)
    # 将 FP8 视图转为 int8 比较，由于有舍入模式的不同，容忍最多 1 ULP 的差距
    q_diff = (triton_q.view(torch.int8).float() - custom_q.view(torch.int8).float()).abs().max().item()
    q_match = (q_diff <= 1.0)

    assert scale_match, (
        f"Scale 不匹配: max_abs_diff="
        f"{(triton_s - custom_s).abs().max().item():.6e}"
    )
    assert q_match, f"Q 量化结果不一致: max diff {q_diff} ULP (容忍 1 ULP)"

    # 2. 性能测试 - Triton
    for _ in range(warmup_iters):
        triton_per_token_group_quant_fp8(x, group_size, eps, use_ue8m0)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(bench_iters):
        triton_per_token_group_quant_fp8(x, group_size, eps, use_ue8m0)
    torch.cuda.synchronize()
    triton_us = (time.perf_counter() - start_time) * 1e6 / bench_iters

    # 3. 性能测试 - LightOp
    for _ in range(warmup_iters):
        per_token_group_quant_fp8(x, group_size, eps=eps)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(bench_iters):
        per_token_group_quant_fp8(x, group_size, eps=eps)
    torch.cuda.synchronize()
    custom_us = (time.perf_counter() - start_time) * 1e6 / bench_iters

    speedup = triton_us / custom_us if custom_us > 0 else 0.0

    return {
        "num_tokens": num_tokens,
        "headdim": headdim,
        "q_diff": q_diff,
        "triton_us": triton_us,
        "custom_us": custom_us,
        "speedup": speedup,
    }


def benchmark_and_verify():
    headdim = 128
    group_size = 128
    eps = 1e-5
    use_ue8m0 = False
    results_data = []

    print(f"{'='*80}")
    print(f"{'num_tokens':<12} | {'headdim':<8} | {'精度 (Q)':<10} | {'精度 (Scale)':<12} | {'Triton (us)':<12} | {'LightOp (us)':<12} | {'加速比':<8}")
    print(f"{'-'*80}")

    for num_tokens in NUM_TOKENS_LIST:
        result = run_case(num_tokens, headdim, group_size, eps, use_ue8m0)

        print(f"{result['num_tokens']:<12} | {result['headdim']:<8} | {'✅':<10} | {'✅':<12} | {result['triton_us']:<12.2f} | {result['custom_us']:<12.2f} | {result['speedup']:<.2f}x")

        results_data.append({
            "Num Tokens": result["num_tokens"],
            "Head Dim": headdim,
            "Group Size": group_size,
            "Q Correct": "Yes",
            "Q Max Diff": result["q_diff"],
            "Scale Correct": "Yes",
            "Triton (us)": round(result["triton_us"], 2),
            "LightOp (us)": round(result["custom_us"], 2),
            "Speedup": round(result["speedup"], 2),
        })

    print(f"{'='*80}")

    csv_filename = "per_token_group_quant_fp8_results.csv"
    fieldnames = [
        "Num Tokens", "Head Dim", "Group Size", "Q Correct", "Q Max Diff",
        "Scale Correct", "Triton (us)", "LightOp (us)", "Speedup"
    ]
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"CSV saved => {csv_filename}")


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
def test_per_token_group_quant_fp8(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    run_case(num_tokens)


if __name__ == "__main__":
    benchmark_and_verify()
