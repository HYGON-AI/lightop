# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional-clamp tests for fused RMSNorm + RoPE + FP8 QKV quantization."""

import torch

from lightop import op
from lightop.rmsnorm import fuse_rmsnorm_rope_quant_qkv


EPSILON = 1e-6
LIMIT = 8.0
DTYPES = (torch.bfloat16, torch.float32)
Q_C_DIM = 128
KV_C_DIM = 512
ROT_DIM = 64
CACHE_ENTRY = KV_C_DIM + ROT_DIM
WARMUP_ITERS = 10
TIMED_ITERS = 100


def dtype_name(dtype: torch.dtype) -> str:
    return "BF16" if dtype == torch.bfloat16 else "FP32"


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-2, 1e-2) if dtype == torch.bfloat16 else (2e-4, 2e-4)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, limit: float = -1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    output = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPSILON)
        * weight.float()
    )
    if limit >= 0.0:
        output = torch.clamp(output, -limit, limit)
    return output.to(x.dtype), output


def make_inputs(tokens: int, heads: int, dtype: torch.dtype) -> dict:
    torch.manual_seed(3000 + tokens + heads)
    q = (
        torch.randn(
            tokens, heads, Q_C_DIM + ROT_DIM, device="cuda", dtype=dtype
        )
        * 2.0
    ).contiguous()
    k = (
        torch.randn(tokens, KV_C_DIM + ROT_DIM, device="cuda", dtype=dtype)
        * 2.0
    ).contiguous()
    weight = torch.linspace(4.0, 12.0, KV_C_DIM, device="cuda", dtype=dtype)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) % 128
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, ROT_DIM, 2, device="cuda", dtype=torch.float32) / ROT_DIM)
    )
    freqs = torch.outer(torch.arange(128, device="cuda", dtype=torch.float32), inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
    block_size = 16
    num_blocks = (tokens + block_size - 1) // block_size
    return {
        "positions": positions,
        "q_c": q[..., :Q_C_DIM],
        "q_pe": q[..., Q_C_DIM:],
        "q_quant": torch.empty_like(q, dtype=torch.float8_e4m3fn),
        "q_scale": torch.empty(tokens, heads, 1, device="cuda", dtype=torch.float32),
        "k_pe": k[..., KV_C_DIM:],
        "kv_c": k[..., :KV_C_DIM],
        "kv_c_before": k[..., :KV_C_DIM].clone(),
        "kv_c_normed": torch.empty(tokens, KV_C_DIM, device="cuda", dtype=dtype),
        "weight": weight,
        "cos_sin_cache": cos_sin_cache,
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        "kv_cache": torch.zeros(
            num_blocks,
            block_size,
            CACHE_ENTRY,
            device="cuda",
            dtype=torch.uint8,
        ),
    }


def native_args(data: dict) -> tuple:
    return (
        data["positions"],
        data["q_c"],
        data["q_pe"],
        data["q_quant"].view(torch.uint8),
        data["q_scale"],
        data["k_pe"],
        data["kv_c"],
        data["kv_c_normed"],
        data["weight"],
        data["cos_sin_cache"],
        data["slot_mapping"],
        data["kv_cache"],
        "fp8_e4m3",
        1.0,
        False,
        EPSILON,
    )


def wrapper_args(data: dict) -> tuple:
    args = list(native_args(data))
    args[3] = data["q_quant"]
    return tuple(args)


def selected_cache(data: dict) -> torch.Tensor:
    cache = data["kv_cache"].view(-1, CACHE_ENTRY)[data["slot_mapping"]]
    return cache.view(torch.float8_e4m3fn)


def test_fuse_rmsnorm_rope_quant_qkv_clamp() -> None:
    for dtype in DTYPES:
        rtol, atol = tolerance(dtype)
        omitted = make_inputs(64, 32, dtype)
        explicit = make_inputs(64, 32, dtype)
        op.fuse_rmsnorm_rope_quant_qkv(*native_args(omitted))
        op.fuse_rmsnorm_rope_quant_qkv(*native_args(explicit), limit=-1.0)
        for key in (
            "q_pe",
            "q_quant",
            "q_scale",
            "k_pe",
            "kv_c_normed",
            "kv_cache",
        ):
            torch.testing.assert_close(omitted[key], explicit[key], rtol=0, atol=0)

        clamped = make_inputs(64, 32, dtype)
        fuse_rmsnorm_rope_quant_qkv(*wrapper_args(clamped), limit=LIMIT)
        expected, expected_fp32 = torch_rms_norm(
            clamped["kv_c_before"], clamped["weight"], LIMIT
        )
        torch.testing.assert_close(
            clamped["kv_c_normed"], expected, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(clamped["q_pe"], omitted["q_pe"], rtol=0, atol=0)
        torch.testing.assert_close(clamped["k_pe"], omitted["k_pe"], rtol=0, atol=0)
        torch.testing.assert_close(
            clamped["q_quant"].view(torch.uint8),
            omitted["q_quant"].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            clamped["q_scale"], omitted["q_scale"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            selected_cache(clamped)[:, :KV_C_DIM].float(),
            expected_fp32.to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )
        assert clamped["kv_c_normed"].float().abs().max().item() <= LIMIT

        zero = make_inputs(16, 32, dtype)
        fuse_rmsnorm_rope_quant_qkv(*wrapper_args(zero), limit=0.0)
        torch.testing.assert_close(
            zero["kv_c_normed"], torch.zeros_like(zero["kv_c_normed"])
        )
        torch.testing.assert_close(
            selected_cache(zero)[:, :KV_C_DIM].float(),
            torch.zeros_like(selected_cache(zero)[:, :KV_C_DIM].float()),
        )

    for bad_limit in (-2.0, float("inf"), float("nan")):
        invalid = make_inputs(4, 32, torch.bfloat16)
        try:
            op.fuse_rmsnorm_rope_quant_qkv(
                *native_args(invalid), limit=bad_limit
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "fuse_rmsnorm_rope_quant_qkv accepted an invalid limit"
            )


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
    tokens, heads = 2048, 128
    print(
        f"{'Case':<24} | {'Tokens/heads':<12} | {'Dtype':<8} | "
        f"{'Correctness':<11} | {'Mean us':>10} | {'Effective GB/s':>14}"
    )
    print("-" * 96)
    for dtype in DTYPES:
        omitted = make_inputs(tokens, heads, dtype)
        clamped = make_inputs(tokens, heads, dtype)
        logical_bytes = (
            2 * omitted["q_pe"].numel() * omitted["q_pe"].element_size()
            + 2 * omitted["k_pe"].numel() * omitted["k_pe"].element_size()
            + 2 * omitted["kv_c"].numel() * omitted["kv_c"].element_size()
            + tokens * CACHE_ENTRY
            + omitted["q_c"].numel() * omitted["q_c"].element_size()
            + omitted["q_quant"].numel() * omitted["q_quant"].element_size()
            + omitted["q_scale"].numel() * omitted["q_scale"].element_size()
        )
        cases = (
            ("rms_rope_quant_default", lambda: op.fuse_rmsnorm_rope_quant_qkv(*native_args(omitted))),
            ("rms_rope_quant_clamp", lambda: op.fuse_rmsnorm_rope_quant_qkv(*native_args(clamped), limit=LIMIT)),
        )
        for name, fn in cases:
            mean_us = benchmark_us(fn)
            bandwidth_gbs = logical_bytes / (mean_us * 1e3)
            print(
                f"{name:<24} | {'2048/128':<12} | {dtype_name(dtype):<8} | "
                f"{'PASS':<11} | {mean_us:>10.3f} | {bandwidth_gbs:>14.3f}"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    test_fuse_rmsnorm_rope_quant_qkv_clamp()
    print("Fused RMSNorm + RoPE + QKV quant optional-clamp checks passed.")
    run_benchmark()


if __name__ == "__main__":
    main()
