# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance test for ``rms_norm_opt``."""

import torch

from lightop import op


DEVICE = "cuda"
EPSILON = 1e-5
WARMUP_ITERS = 10
TIMED_ITERS = 100
# (num_tokens, hidden_size): mid + fused (>=16384) + small edge cases
TEST_SHAPES = (
    (46464, 64),
    (46464, 128),
    (46464, 256),
    (46464, 512),
    (2904, 1024),
    (1024, 2048),
    (1024, 4096),
    (1024, 8192),
    (1024, 16384),
    (1024, 32768),
    (512, 65536),
    (256, 131072),
    (5, 512),
    (9, 256),
    (17, 128),
    (33, 64),
    # irregular vs xpu-perf (batch!=1024, odd / non-pow2 hidden)
    (7, 100),
    (17, 1000),
    (33, 2047),
    (65, 5000),
    (7, 8191),
    (17, 10000),
    (3, 16385),
    (513, 32767),
    (999, 999),
)
TEST_DTYPES = (
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 2e-2, 2e-2),
    (torch.bfloat16, 2e-2, 2e-2),
)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    return (
        x_fp32 * torch.rsqrt(variance + epsilon) * weight.float()
    ).to(x.dtype)


def run_case(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    timed_iters: int,
) -> dict:
    torch.manual_seed(0)
    x = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=dtype
    )
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    out = torch.empty_like(x)

    # Correctness must be checked before benchmarking.
    expected = torch_rms_norm(x, weight, EPSILON)
    op.rms_norm_opt(out, x, weight, EPSILON)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    max_abs_error = (out.float() - expected.float()).abs().max().item()

    # Skip heavy timing on very large fused dims.
    if timed_iters <= 0:
        return {
            "torch_us": float("nan"),
            "lightop_us": float("nan"),
            "speedup": float("nan"),
            "bandwidth_gbs": float("nan"),
            "max_abs_error": max_abs_error,
        }

    torch_us = benchmark_us(
        lambda: torch_rms_norm(x, weight, EPSILON), timed_iters
    )
    lightop_us = benchmark_us(
        lambda: op.rms_norm_opt(out, x, weight, EPSILON), timed_iters
    )

    # Logical traffic: read x and weight once, then write out once.
    logical_bytes = (
        x.numel() * x.element_size()
        + weight.numel() * weight.element_size()
        + out.numel() * out.element_size()
    )
    bandwidth_gbs = logical_bytes / (lightop_us * 1.0e3)

    return {
        "torch_us": torch_us,
        "lightop_us": lightop_us,
        "speedup": torch_us / lightop_us,
        "bandwidth_gbs": bandwidth_gbs,
        "max_abs_error": max_abs_error,
    }


def benchmark_us(fn, timed_iters: int = TIMED_ITERS) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / timed_iters


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")

    print(
        f"{'Shape':<18} | {'Dtype':<8} | {'Correctness':<11} | "
        f"{'Max abs':>10} | {'Torch(us)':>10} | {'LightOp(us)':>11} | "
        f"{'Speedup':>8} | {'Effective GB/s':>14}"
    )
    print("-" * 112)
    for dtype, rtol, atol in TEST_DTYPES:
        dtype_name = str(dtype).replace("torch.", "")
        for num_tokens, hidden_size in TEST_SHAPES:
            timed_iters = TIMED_ITERS if hidden_size <= 8192 else 0
            result = run_case(
                num_tokens, hidden_size, dtype, rtol, atol, timed_iters
            )
            shape = f"[{num_tokens}, {hidden_size}]"
            print(
                f"{shape:<18} | {dtype_name:<8} | {'PASS':<11} | "
                f"{result['max_abs_error']:>10.4e} | "
                f"{result['torch_us']:>10.3f} | "
                f"{result['lightop_us']:>11.3f} | "
                f"{result['speedup']:>8.2f} | "
                f"{result['bandwidth_gbs']:>14.3f}"
            )


if __name__ == "__main__":
    main()
