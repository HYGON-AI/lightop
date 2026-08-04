"""BF16 optional-clamp tests for fused RMSNorm + contiguous RoPE."""

import torch

from lightop import op
from lightop.rmsnorm import fused_rms_norm_rope_contiguous


EPSILON = 1e-6
LIMIT = 8.0
DTYPE = torch.bfloat16
Q_C_DIM = 128
KV_C_DIM = 512
ROT_DIM = 64
CACHE_ENTRY = KV_C_DIM + ROT_DIM
WARMUP_ITERS = 10
TIMED_ITERS = 100


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


def make_inputs(tokens: int, heads: int, cache_dtype_str: str) -> dict:
    torch.manual_seed(2000 + tokens + heads)
    q = (
        torch.randn(
            tokens, heads, Q_C_DIM + ROT_DIM, device="cuda", dtype=DTYPE
        )
        * 2.0
    ).contiguous()
    k = (
        torch.randn(tokens, KV_C_DIM + ROT_DIM, device="cuda", dtype=DTYPE)
        * 2.0
    ).contiguous()
    weight = torch.linspace(4.0, 12.0, KV_C_DIM, device="cuda", dtype=DTYPE)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) % 128
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, ROT_DIM, 2, device="cuda", dtype=torch.float32) / ROT_DIM)
    )
    freqs = torch.outer(torch.arange(128, device="cuda", dtype=torch.float32), inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)
    block_size = 16
    num_blocks = (tokens + block_size - 1) // block_size
    cache_dtype = DTYPE if cache_dtype_str == "bf16" else torch.uint8
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        CACHE_ENTRY,
        device="cuda",
        dtype=cache_dtype,
    )
    return {
        "positions": positions,
        "q_pe": q[..., Q_C_DIM:],
        "k_pe": k[..., KV_C_DIM:],
        "kv_c": k[..., :KV_C_DIM],
        "kv_c_before": k[..., :KV_C_DIM].clone(),
        "kv_c_normed": torch.empty(tokens, KV_C_DIM, device="cuda", dtype=DTYPE),
        "weight": weight,
        "cos_sin_cache": cos_sin_cache,
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        "kv_cache": kv_cache,
        "kv_cache_dtype_str": cache_dtype_str,
    }


def op_args(data: dict) -> tuple:
    return (
        data["positions"],
        data["q_pe"],
        data["k_pe"],
        data["kv_c"],
        data["kv_c_normed"],
        data["weight"],
        data["cos_sin_cache"],
        data["slot_mapping"],
        data["kv_cache"],
        data["kv_cache_dtype_str"],
        1.0,
        False,
        EPSILON,
    )


def selected_cache(data: dict) -> torch.Tensor:
    cache = data["kv_cache"].view(-1, CACHE_ENTRY)[data["slot_mapping"]]
    if data["kv_cache_dtype_str"] == "fp8_e4m3":
        return cache.view(torch.float8_e4m3fn)
    return cache


def assert_cache_matches(data: dict, expected: torch.Tensor) -> None:
    actual = selected_cache(data)[:, :KV_C_DIM]
    if data["kv_cache_dtype_str"] == "fp8_e4m3":
        torch.testing.assert_close(
            actual.float(), expected.to(torch.float8_e4m3fn).float(), rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_fused_rms_norm_rope_contiguous_clamp() -> None:
    for cache_dtype_str in ("bf16", "fp8_e4m3"):
        omitted = make_inputs(64, 32, cache_dtype_str)
        explicit = make_inputs(64, 32, cache_dtype_str)
        op.fused_rms_norm_rope_contiguous(*op_args(omitted))
        op.fused_rms_norm_rope_contiguous(*op_args(explicit), limit=-1.0)
        for key in ("q_pe", "k_pe", "kv_c_normed", "kv_cache"):
            torch.testing.assert_close(omitted[key], explicit[key], rtol=0, atol=0)

        clamped = make_inputs(64, 32, cache_dtype_str)
        fused_rms_norm_rope_contiguous(*op_args(clamped), limit=LIMIT)
        expected = torch_rms_norm(
            clamped["kv_c_before"], clamped["weight"], LIMIT
        )
        torch.testing.assert_close(
            clamped["kv_c_normed"], expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(clamped["q_pe"], omitted["q_pe"], rtol=0, atol=0)
        torch.testing.assert_close(clamped["k_pe"], omitted["k_pe"], rtol=0, atol=0)
        assert_cache_matches(clamped, expected)
        assert clamped["kv_c_normed"].float().abs().max().item() <= LIMIT

        zero = make_inputs(16, 32, cache_dtype_str)
        fused_rms_norm_rope_contiguous(*op_args(zero), limit=0.0)
        torch.testing.assert_close(
            zero["kv_c_normed"], torch.zeros_like(zero["kv_c_normed"])
        )
        torch.testing.assert_close(
            selected_cache(zero)[:, :KV_C_DIM].float(),
            torch.zeros_like(selected_cache(zero)[:, :KV_C_DIM].float()),
        )

    for bad_limit in (-2.0, float("inf"), float("nan")):
        invalid = make_inputs(4, 32, "bf16")
        try:
            op.fused_rms_norm_rope_contiguous(
                *op_args(invalid), limit=bad_limit
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "fused_rms_norm_rope_contiguous accepted an invalid limit"
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
    omitted = make_inputs(tokens, heads, "fp8_e4m3")
    clamped = make_inputs(tokens, heads, "fp8_e4m3")
    logical_bytes = (
        2 * omitted["q_pe"].numel() * omitted["q_pe"].element_size()
        + 2 * omitted["k_pe"].numel() * omitted["k_pe"].element_size()
        + 2 * omitted["kv_c"].numel() * omitted["kv_c"].element_size()
        + tokens * CACHE_ENTRY
    )
    cases = (
        ("rms_rope_default", lambda: op.fused_rms_norm_rope_contiguous(*op_args(omitted))),
        ("rms_rope_clamp", lambda: op.fused_rms_norm_rope_contiguous(*op_args(clamped), limit=LIMIT)),
    )
    print(
        f"{'Case':<24} | {'Tokens/heads':<12} | {'Dtype':<8} | "
        f"{'Correctness':<11} | {'Mean us':>10} | {'Effective GB/s':>14}"
    )
    print("-" * 96)
    for name, fn in cases:
        mean_us = benchmark_us(fn)
        bandwidth_gbs = logical_bytes / (mean_us * 1e3)
        print(
            f"{name:<24} | {'2048/128':<12} | {'BF16':<8} | "
            f"{'PASS':<11} | {mean_us:>10.3f} | {bandwidth_gbs:>14.3f}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    test_fused_rms_norm_rope_contiguous_clamp()
    print("BF16 fused RMSNorm + contiguous RoPE optional-clamp checks passed.")
    run_benchmark()


if __name__ == "__main__":
    main()
