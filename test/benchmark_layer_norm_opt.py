# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark inference-only LayerNorm kernels used by Qwen3-VL.

Examples:
    python test/benchmark_layer_norm_opt.py
    python test/benchmark_layer_norm_opt.py --dtype bf16 --tokens 288 2904

The script validates correctness before timing. It compares ``layer_norm``
against PyTorch LayerNorm and compares ``fused_add_layer_norm`` against the
unfused PyTorch expression ``LayerNorm(x + residual)``.
"""

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from lightop.layernorm import fused_add_layer_norm, layer_norm


EPSILON = 1e-5
DEFAULT_TOKENS = (1, 16, 256, 288, 1024, 2904)
DEFAULT_HIDDEN_SIZES = (1024, 1152)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "both"),
        default="both",
        help="Activation and bias dtype (default: both).",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKENS),
        help="Token counts to benchmark.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_SIZES),
        help="Last-dimension sizes to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    return parser.parse_args()


def benchmark_us(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def torch_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, epsilon)


def torch_add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = x + residual
    return torch_layer_norm(residual_out, weight, bias, epsilon), residual_out


def effective_bandwidth_gbs(logical_bytes: int, latency_us: float) -> float:
    return logical_bytes / (latency_us * 1.0e3)


def run_case(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float,
    warmup: int,
    iters: int,
) -> list[dict[str, object]]:
    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden_size), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    bias = torch.randn(hidden_size, device="cuda", dtype=dtype)

    expected = torch_layer_norm(x, weight, bias, epsilon)
    actual = torch.empty_like(x)
    layer_norm(x, weight, bias, epsilon, out=actual)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    plain_max_error = (actual.float() - expected.float()).abs().max().item()

    expected_add, expected_residual = torch_add_layer_norm(
        x, residual, weight, bias, epsilon
    )
    actual_add = x.clone()
    actual_residual = residual.clone()
    fused_add_layer_norm(
        actual_add, actual_residual, weight, bias, epsilon
    )
    torch.testing.assert_close(
        actual_residual, expected_residual, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_add, expected_add, rtol=1e-2, atol=1e-2
    )
    add_max_error = max(
        (actual_add.float() - expected_add.float()).abs().max().item(),
        (
            actual_residual.float() - expected_residual.float()
        ).abs().max().item(),
    )

    torch_plain_us = benchmark_us(
        lambda: torch_layer_norm(x, weight, bias, epsilon), warmup, iters
    )
    custom_out = torch.empty_like(x)
    custom_plain_us = benchmark_us(
        lambda: layer_norm(x, weight, bias, epsilon, out=custom_out), warmup, iters
    )

    torch_add_us = benchmark_us(
        lambda: torch_add_layer_norm(x, residual, weight, bias, epsilon),
        warmup,
        iters,
    )
    custom_x = x.clone()
    custom_residual = residual.clone()
    custom_add_us = benchmark_us(
        lambda: fused_add_layer_norm(
            custom_x, custom_residual, weight, bias, epsilon
        ),
        warmup,
        iters,
    )

    element_bytes = x.element_size()
    plain_bytes = (
        2 * x.numel() + weight.numel() + bias.numel()
    ) * element_bytes
    fused_bytes = (
        4 * x.numel() + weight.numel() + bias.numel()
    ) * element_bytes
    return [
        {
            "op": "layer_norm",
            "tokens": num_tokens,
            "hidden": hidden_size,
            "dtype": str(dtype).removeprefix("torch."),
            "torch_us": torch_plain_us,
            "lightop_us": custom_plain_us,
            "speedup": torch_plain_us / custom_plain_us,
            "max_error": plain_max_error,
            "bandwidth": effective_bandwidth_gbs(
                plain_bytes, custom_plain_us
            ),
        },
        {
            "op": "add_layer_norm",
            "tokens": num_tokens,
            "hidden": hidden_size,
            "dtype": str(dtype).removeprefix("torch."),
            "torch_us": torch_add_us,
            "lightop_us": custom_add_us,
            "speedup": torch_add_us / custom_add_us,
            "max_error": add_max_error,
            "bandwidth": effective_bandwidth_gbs(
                fused_bytes, custom_add_us
            ),
        },
    ]


def print_header() -> None:
    print(
        f"{'Op':<16} {'Shape':<16} {'Dtype':<9} "
        f"{'Torch(us)':>11} {'LightOp(us)':>12} "
        f"{'Speedup':>9} {'Max abs':>11} {'Effective GB/s':>15}"
    )
    print("-" * 106)


def print_result(result: dict[str, object]) -> None:
    shape = f"[{result['tokens']}, {result['hidden']}]"
    print(
        f"{result['op']:<16} {shape:<16} {result['dtype']:<9} "
        f"{result['torch_us']:>11.3f} {result['lightop_us']:>12.3f} "
        f"{result['speedup']:>9.2f} {result['max_error']:>11.3e} "
        f"{result['bandwidth']:>15.2f}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if any(value <= 0 for value in args.tokens + args.hidden_sizes):
        raise ValueError("tokens and hidden sizes must be positive")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtypes = tuple(dtype_map.values()) if args.dtype == "both" else (
        dtype_map[args.dtype],
    )

    print_header()
    for dtype in dtypes:
        for hidden_size in args.hidden_sizes:
            for num_tokens in args.tokens:
                for result in run_case(
                    num_tokens,
                    hidden_size,
                    dtype,
                    args.epsilon,
                    args.warmup,
                    args.iters,
                ):
                    print_result(result)


if __name__ == "__main__":
    main()
