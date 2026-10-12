# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional-clamp tests for RMSNorm dynamic per-token INT8 quantization."""

import torch

from lightop import op
from lightop.norm import lm_faster_rmsquant
from lightop.rmsnorm import rms_norm_dynamic_per_token_quant, rms_norm_per_token_quant


EPSILON = 1e-6
LIMIT = 0.5
DTYPES = (torch.bfloat16, torch.float32)
WARMUP_ITERS = 10
TIMED_ITERS = 100


def dtype_name(dtype: torch.dtype) -> str:
    return "BF16" if dtype == torch.bfloat16 else "FP32"


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (2e-2, 2e-4) if dtype == torch.bfloat16 else (2e-4, 2e-5)


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


def torch_dynamic_int8_quant(
    normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = normalized.float().abs().amax(dim=-1, keepdim=True)
    inv_scale = torch.where(
        absmax == 0.0, torch.zeros_like(absmax), 127.0 / absmax
    )
    quantized = (
        torch.round(normalized.float() * inv_scale)
        .clamp(-128, 127)
        .to(torch.int8)
    )
    scales = torch.clamp(absmax / 127.0, min=1.19209e-7)
    return quantized, scales


def run_case(
    dtype: torch.dtype,
    shape: tuple[int, int],
    with_residual: bool,
    update_input: bool,
    limit: float,
) -> None:
    torch.manual_seed(30 + shape[-1] + int(with_residual))
    x = (torch.randn(shape, device="cuda", dtype=dtype) * 2.0).contiguous()
    residual = (torch.randn_like(x) * 2.0).contiguous() if with_residual else None
    weight = (
        torch.randn(shape[-1], device="cuda", dtype=dtype) * 0.5 + 1.0
    ).contiguous()
    input_before = x.clone()
    expected_source = input_before if residual is None else input_before + residual
    expected_normalized = torch_rms_norm(expected_source, weight, limit)
    expected_q, expected_scales = torch_dynamic_int8_quant(expected_normalized)
    rtol, atol = tolerance(dtype)

    actual_q, actual_scales = rms_norm_dynamic_per_token_quant(
        x,
        weight,
        EPSILON,
        torch.int8,
        residual=residual,
        update_input=update_input,
        limit=limit,
    )
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=1)
    torch.testing.assert_close(
        actual_scales, expected_scales, rtol=rtol, atol=atol
    )
    if residual is not None:
        torch.testing.assert_close(residual, expected_source, rtol=0, atol=0)
    if update_input:
        torch.testing.assert_close(
            x, expected_normalized, rtol=rtol, atol=max(atol, 1e-4)
        )
        if limit >= 0.0:
            assert x.float().abs().max().item() <= limit


def test_dynamic_per_token_quant_clamp() -> None:
    for dtype in DTYPES:
        for shape in ((17, 2048), (5, 65)):
            for with_residual in (False, True):
                for update_input in (False, True):
                    run_case(dtype, shape, with_residual, update_input, LIMIT)

        shape = (17, 2048)
        torch.manual_seed(300)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        omitted_x, explicit_x = x.clone(), x.clone()
        omitted_q, omitted_scales = rms_norm_dynamic_per_token_quant(
            omitted_x, weight, EPSILON, torch.int8, update_input=False
        )
        explicit_q, explicit_scales = rms_norm_dynamic_per_token_quant(
            explicit_x,
            weight,
            EPSILON,
            torch.int8,
            update_input=False,
            limit=-1.0,
        )
        torch.testing.assert_close(omitted_q, explicit_q, rtol=0, atol=0)
        torch.testing.assert_close(omitted_scales, explicit_scales, rtol=0, atol=0)

        wrapper_q, wrapper_scales = rms_norm_per_token_quant(
            x.clone(),
            weight,
            EPSILON,
            dtype=torch.int8,
            update_input=False,
            limit=LIMIT,
        )
        expected = torch_rms_norm(x, weight, LIMIT)
        expected_q, expected_scales = torch_dynamic_int8_quant(expected)
        rtol, atol = tolerance(dtype)
        torch.testing.assert_close(wrapper_q, expected_q, rtol=0, atol=1)
        torch.testing.assert_close(
            wrapper_scales, expected_scales, rtol=rtol, atol=atol
        )

        model_q, model_scales = lm_faster_rmsquant(
            x.clone(),
            weight,
            EPSILON,
            torch.int8,
            update_input=False,
            limit=LIMIT,
        )
        torch.testing.assert_close(model_q, expected_q, rtol=0, atol=1)
        torch.testing.assert_close(
            model_scales, expected_scales, rtol=rtol, atol=atol
        )

    try:
        rms_norm_per_token_quant(
            x,
            weight,
            EPSILON,
            dtype=torch.float8_e4m3fn,
            update_input=False,
            limit=LIMIT,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("FP8 rms_norm_per_token_quant accepted limit")

    for bad_limit in (-2.0, float("inf"), float("nan")):
        try:
            rms_norm_dynamic_per_token_quant(
                x,
                weight,
                EPSILON,
                torch.int8,
                update_input=False,
                limit=bad_limit,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("dynamic per-token quant accepted invalid limit")


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
        torch.manual_seed(400)
        x = (torch.randn(shape, device="cuda", dtype=dtype) * 1e-3).contiguous()
        weight = torch.ones(shape[-1], device="cuda", dtype=dtype)
        output = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty((shape[0], 1), device="cuda", dtype=torch.float32)
        logical_bytes = (
            x.numel() * x.element_size()
            + weight.numel() * weight.element_size()
            + output.numel()
            + scales.numel() * scales.element_size()
        )
        cases = (
            ("quant_default", lambda: op.rms_norm_dynamic_per_token_quant(output, x, weight, scales, EPSILON, None, False)),
            ("quant_clamp", lambda: op.rms_norm_dynamic_per_token_quant(output, x, weight, scales, EPSILON, None, False, LIMIT)),
        )
        for name, fn in cases:
            mean_us = benchmark_us(fn)
            bandwidth_gbs = logical_bytes / (mean_us * 1e3)
            print(
                f"{name:<16} | {'[4096, 2048]':<14} | {dtype_name(dtype):<8} | "
                f"{'PASS':<11} | {mean_us:>10.3f} | {bandwidth_gbs:>14.3f}"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    test_dynamic_per_token_quant_clamp()
    print("Dynamic per-token INT8 optional-clamp checks passed.")
    run_benchmark()


if __name__ == "__main__":
    main()
