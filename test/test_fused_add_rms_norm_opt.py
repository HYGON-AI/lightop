"""Correctness and performance test for ``fused_add_rms_norm_opt``."""

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
    # The BF16 residual addition is rounded before the FP32 RMSNorm.
    residual_out = x + residual
    x_out = torch_rms_norm(residual_out, weight, epsilon)
    return x_out, residual_out


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
