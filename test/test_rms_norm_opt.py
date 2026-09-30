# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance test for ``rms_norm_opt``."""

import torch

from lightop import op


DEVICE = "cuda"
DTYPE = torch.bfloat16
EPSILON = 1e-5
RTOL = 1e-2
ATOL = 1e-2
WARMUP_ITERS = 10
TIMED_ITERS = 100
TEST_SHAPES = (
    # (2904, 1024),
    # (46464, 512),
    # (46464, 256),
    # (46464, 128),
    (46464, 64),
    # (5, 512),
    # (9, 256),
    # (17, 128),
    # (33, 64),
)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    return (
        x_fp32 * torch.rsqrt(variance + epsilon) * weight.float()
    ).to(x.dtype)


def benchmark_us(fn) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / TIMED_ITERS


def run_case(num_tokens: int, hidden_size: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=DTYPE
    )
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
    out = torch.empty_like(x)

    # Correctness must be checked before benchmarking.
    expected = torch_rms_norm(x, weight, EPSILON)
    op.rms_norm_opt(out, x, weight, EPSILON)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    max_abs_error = (out.float() - expected.float()).abs().max().item()

    torch_us = benchmark_us(
        lambda: torch_rms_norm(x, weight, EPSILON)
    )
    lightop_us = benchmark_us(
        lambda: op.rms_norm_opt(out, x, weight, EPSILON)
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")

    print(
        f"{'Shape':<18} | {'Dtype':<8} | {'Correctness':<11} | "
        f"{'Max abs':>10} | {'Torch(us)':>10} | {'LightOp(us)':>11} | "
        f"{'Speedup':>8} | {'Effective GB/s':>14}"
    )
    print("-" * 112)
    for num_tokens, hidden_size in TEST_SHAPES:
        result = run_case(num_tokens, hidden_size)
        shape = f"[{num_tokens}, {hidden_size}]"
        print(
            f"{shape:<18} | {'BF16':<8} | {'PASS':<11} | "
            f"{result['max_abs_error']:>10.4e} | "
            f"{result['torch_us']:>10.3f} | "
            f"{result['lightop_us']:>11.3f} | "
            f"{result['speedup']:>8.2f} | "
            f"{result['bandwidth_gbs']:>14.3f}"
        )


if __name__ == "__main__":
    main()
