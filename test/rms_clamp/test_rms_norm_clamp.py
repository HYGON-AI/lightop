"""Optional-clamp tests for plain and residual RMSNorm."""

import torch

from lightop import op
from lightop.rmsnorm import fused_add_rms_norm, rms_norm


EPSILON = 1e-6
LIMIT = 0.5
DTYPES = (torch.bfloat16, torch.float32)
WARMUP_ITERS = 10
TIMED_ITERS = 100


def dtype_name(dtype: torch.dtype) -> str:
    return "BF16" if dtype == torch.bfloat16 else "FP32"


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-2, 1e-2) if dtype == torch.bfloat16 else (2e-4, 2e-4)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, limit: float = -1.0
) -> torch.Tensor:
    output = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPSILON)
        * weight.float()
    )
    if limit >= 0.0:
        output = torch.clamp(output, -limit, limit)
    return output.to(x.dtype)


def test_plain_rms_norm_clamp() -> None:
    for dtype in DTYPES:
        rtol, atol = tolerance(dtype)
        for shape in ((17, 2048), (5, 65)):
            torch.manual_seed(10 + shape[-1])
            x = (torch.randn(shape, device="cuda", dtype=dtype) * 2.0).contiguous()
            weight = (
                torch.randn(shape[-1], device="cuda", dtype=dtype) * 0.5 + 1.0
            ).contiguous()

            omitted = torch.empty_like(x)
            explicit = torch.empty_like(x)
            op.rms_norm_opt(omitted, x, weight, EPSILON)
            op.rms_norm_opt(explicit, x, weight, EPSILON, limit=-1.0)
            torch.testing.assert_close(omitted, explicit, rtol=0, atol=0)
            torch.testing.assert_close(
                omitted, torch_rms_norm(x, weight), rtol=rtol, atol=atol
            )

            clamped = torch.empty_like(x)
            op.rms_norm_opt(clamped, x, weight, EPSILON, limit=LIMIT)
            torch.testing.assert_close(
                clamped,
                torch_rms_norm(x, weight, LIMIT),
                rtol=rtol,
                atol=atol,
            )
            assert clamped.float().abs().max().item() <= LIMIT

            inplace_input = x.clone()
            inplace_output = rms_norm(
                inplace_input, weight, EPSILON, limit=LIMIT
            )
            assert inplace_output.data_ptr() == inplace_input.data_ptr()
            torch.testing.assert_close(
                inplace_output,
                torch_rms_norm(x, weight, LIMIT),
                rtol=rtol,
                atol=atol,
            )

            wrapper_out = torch.empty_like(x)
            returned = rms_norm(
                x, weight, EPSILON, out=wrapper_out, limit=LIMIT
            )
            assert returned.data_ptr() == wrapper_out.data_ptr()
            torch.testing.assert_close(
                returned,
                torch_rms_norm(x, weight, LIMIT),
                rtol=rtol,
                atol=atol,
            )

            zero = torch.empty_like(x)
            op.rms_norm_opt(zero, x, weight, EPSILON, limit=0.0)
            torch.testing.assert_close(zero, torch.zeros_like(zero))

    for bad_limit in (-2.0, float("inf"), float("nan")):
        try:
            op.rms_norm_opt(omitted, x, weight, EPSILON, limit=bad_limit)
        except RuntimeError:
            pass
        else:
            raise AssertionError("rms_norm_opt accepted an invalid limit")


def test_fused_add_rms_norm_clamp() -> None:
    for dtype in DTYPES:
        rtol, atol = tolerance(dtype)
        for shape in ((17, 2048), (5, 65)):
            torch.manual_seed(20 + shape[-1])
            x = (torch.randn(shape, device="cuda", dtype=dtype) * 2.0).contiguous()
            residual = (torch.randn_like(x) * 2.0).contiguous()
            weight = (
                torch.randn(shape[-1], device="cuda", dtype=dtype) * 0.5 + 1.0
            ).contiguous()
            expected_residual = x + residual
            expected = torch_rms_norm(expected_residual, weight, LIMIT)

            actual_x, actual_residual = x.clone(), residual.clone()
            op.fused_add_rms_norm_opt(
                actual_x, actual_residual, weight, EPSILON, limit=LIMIT
            )
            torch.testing.assert_close(
                actual_residual, expected_residual, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_x, expected, rtol=rtol, atol=atol
            )
            assert actual_x.float().abs().max().item() <= LIMIT

            omitted_x, omitted_residual = x.clone(), residual.clone()
            explicit_x, explicit_residual = x.clone(), residual.clone()
            op.fused_add_rms_norm_opt(
                omitted_x, omitted_residual, weight, EPSILON
            )
            op.fused_add_rms_norm_opt(
                explicit_x, explicit_residual, weight, EPSILON, limit=-1.0
            )
            torch.testing.assert_close(omitted_x, explicit_x, rtol=0, atol=0)
            torch.testing.assert_close(
                omitted_residual, explicit_residual, rtol=0, atol=0
            )

            wrapper_x, wrapper_residual = x.clone(), residual.clone()
            returned_x, returned_residual = fused_add_rms_norm(
                wrapper_x, wrapper_residual, weight, EPSILON, limit=LIMIT
            )
            assert returned_x is wrapper_x and returned_residual is wrapper_residual
            torch.testing.assert_close(
                returned_x, expected, rtol=rtol, atol=atol
            )
            torch.testing.assert_close(
                returned_residual, expected_residual, rtol=0, atol=0
            )

    for bad_limit in (-2.0, float("inf"), float("nan")):
        try:
            op.fused_add_rms_norm_opt(
                wrapper_x, wrapper_residual, weight, EPSILON, limit=bad_limit
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("fused_add_rms_norm_opt accepted an invalid limit")


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


def run_benchmark() -> None:
    shape = (4096, 2048)
    print(
        f"{'Case':<16} | {'Shape':<14} | {'Dtype':<8} | "
        f"{'Correctness':<11} | {'Mean us':>10} | {'Effective GB/s':>14}"
    )
    print("-" * 86)
    for dtype in DTYPES:
        torch.manual_seed(100)
        x = (torch.randn(shape, device="cuda", dtype=dtype) * 1e-3).contiguous()
        residual = (torch.randn_like(x) * 1e-3).contiguous()
        weight = torch.ones(shape[-1], device="cuda", dtype=dtype)
        out = torch.empty_like(x)
        default_x, default_residual = x.clone(), residual.clone()
        clamp_x, clamp_residual = x.clone(), residual.clone()
        plain_bytes = 2 * x.numel() * x.element_size() + weight.numel() * weight.element_size()
        fused_bytes = 4 * x.numel() * x.element_size() + weight.numel() * weight.element_size()
        cases = (
            ("plain_default", lambda: op.rms_norm_opt(out, x, weight, EPSILON), plain_bytes),
            ("plain_clamp", lambda: op.rms_norm_opt(out, x, weight, EPSILON, LIMIT), plain_bytes),
            ("fused_default", lambda: op.fused_add_rms_norm_opt(default_x, default_residual, weight, EPSILON), fused_bytes),
            ("fused_clamp", lambda: op.fused_add_rms_norm_opt(clamp_x, clamp_residual, weight, EPSILON, LIMIT), fused_bytes),
        )
        for name, fn, logical_bytes in cases:
            mean_us = benchmark_us(fn)
            bandwidth_gbs = logical_bytes / (mean_us * 1e3)
            print(
                f"{name:<16} | {'[4096, 2048]':<14} | {dtype_name(dtype):<8} | "
                f"{'PASS':<11} | {mean_us:>10.3f} | {bandwidth_gbs:>14.3f}"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    test_plain_rms_norm_clamp()
    test_fused_add_rms_norm_clamp()
    print("Plain and residual RMSNorm optional-clamp checks passed.")
    run_benchmark()


if __name__ == "__main__":
    main()
