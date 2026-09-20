"""Correctness and performance test for ``fused_add_rms_norm_opt``."""

import csv

import pytest
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
    (2904, 1024),
    (46464, 512),
    (46464, 256),
    (46464, 128),
    (46464, 64),
    (5, 512),
    (9, 256),
    (17, 128),
    (33, 64),
)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    return (
        x_fp32 * torch.rsqrt(variance + epsilon) * weight.float()
    ).to(x.dtype)


def torch_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve the FP32 sum until normalization; round only the outputs.
    summed = x.float() + residual.float()
    x_out = torch_rms_norm(summed, weight, epsilon).to(x.dtype)
    return x_out, summed.to(residual.dtype)


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
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    # Correctness must be checked before benchmarking.
    expected_x, expected_residual = torch_fused_add_rms_norm(
        x, residual, weight, EPSILON
    )
    actual_x = x.clone()
    actual_residual = residual.clone()
    op.fused_add_rms_norm_opt(
        actual_x, actual_residual, weight, EPSILON
    )
    torch.testing.assert_close(
        actual_x, expected_x, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        actual_residual, expected_residual, rtol=RTOL, atol=ATOL
    )
    max_abs_error = max(
        (actual_x.float() - expected_x.float()).abs().max().item(),
        (
            actual_residual.float() - expected_residual.float()
        ).abs().max().item(),
    )

    torch_us = benchmark_us(
        lambda: torch_fused_add_rms_norm(
            x, residual, weight, EPSILON
        )
    )
    perf_x = x.clone()
    perf_residual = residual.clone()
    lightop_us = benchmark_us(
        lambda: op.fused_add_rms_norm_opt(
            perf_x, perf_residual, weight, EPSILON
        )
    )

    # Logical traffic: read and write x and residual, and read weight once.
    logical_bytes = (
        2 * x.numel() * x.element_size()
        + 2 * residual.numel() * residual.element_size()
        + weight.numel() * weight.element_size()
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
    results_data = []

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

        results_data.append({
            "Shape": shape,
            "Num Tokens": num_tokens,
            "Hidden Size": hidden_size,
            "Dtype": "BF16",
            "Correctness": "PASS",
            "Max Abs Error": result["max_abs_error"],
            "Torch (us)": result["torch_us"],
            "LightOp (us)": result["lightop_us"],
            "Speedup": result["speedup"],
            "Effective GB/s": result["bandwidth_gbs"],
        })

    csv_filename = "fused_add_rms_norm_opt_results.csv"
    fieldnames = [
        "Shape", "Num Tokens", "Hidden Size", "Dtype", "Correctness",
        "Max Abs Error", "Torch (us)", "LightOp (us)", "Speedup",
        "Effective GB/s"
    ]
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nCSV saved => {csv_filename}")


@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    TEST_SHAPES,
    ids=[f"{n}x{h}" for n, h in TEST_SHAPES],
)
def test_fused_add_rms_norm_opt(num_tokens, hidden_size) -> None:
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    run_case(num_tokens, hidden_size)


if __name__ == "__main__":
    main()
