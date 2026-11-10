# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for fused RoPE, including head-size 72."""

import argparse
from math import prod

import pytest
import torch

from lightop.fuse_rotary_positional_embedding import rotary_embedding_fuse


OPTIMIZED_HEAD_SIZES = (64, 72, 128, 256)
FALLBACK_HEAD_SIZES = (80, 96)
HEAD72_CASES = (
    ("q_only", (5,), 8, 8, False, False),
    ("flat_mha", (5,), 16, 16, True, True),
    ("gqa", (5,), 32, 8, False, True),
    ("batch_gqa", (2, 3), 16, 4, False, True),
)
DTYPES = (torch.float16, torch.bfloat16)
ROPE_STYLES = ((True, "neox"), (False, "gpt_j"))
RTOL = 2e-2
ATOL = 2e-2


def _dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "bf16"


def _build_cos_sin_cache(
    max_position: int, rot_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    half_dim = rot_dim // 2
    positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
    frequencies = torch.arange(half_dim, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (frequencies / half_dim))
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype).contiguous()


def _rope_reference(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    is_neox: bool,
    head_size: int,
) -> torch.Tensor:
    """Out-of-place PyTorch reference using the LightOp cache layout."""
    original_shape = tensor.shape
    tensor_3d = tensor.reshape(positions.numel(), -1, head_size)
    cache_per_token = cache.index_select(0, positions.reshape(-1))
    cos = cache_per_token[:, None, : head_size // 2]
    sin = cache_per_token[:, None, head_size // 2 :]

    if is_neox:
        x = tensor_3d[..., : head_size // 2]
        y = tensor_3d[..., head_size // 2 : head_size]
        rotated = torch.cat((x * cos - y * sin, y * cos + x * sin), dim=-1)
    else:
        x = tensor_3d[..., 0:head_size:2]
        y = tensor_3d[..., 1:head_size:2]
        rotated = torch.stack(
            (x * cos - y * sin, y * cos + x * sin), dim=-1
        ).flatten(-2)

    return rotated.reshape(original_shape)


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def _run_correctness_case(
    case_name: str,
    positions_shape: tuple[int, ...],
    num_heads: int,
    num_kv_heads: int,
    flatten_heads: bool,
    use_key: bool,
    dtype: torch.dtype,
    is_neox: bool,
    head_size: int = 72,
) -> tuple[float, float]:
    torch.manual_seed(2026 + head_size)
    num_tokens = prod(positions_shape)
    position_values = [4, 1, 7, 0, 3, 6][:num_tokens]
    positions = torch.tensor(
        position_values, device="cuda", dtype=torch.int64
    ).reshape(positions_shape)
    cache = _build_cos_sin_cache(16, head_size, dtype)

    query = torch.randn(
        *positions_shape, num_heads, head_size, device="cuda", dtype=dtype
    )
    if flatten_heads:
        query = query.reshape(*positions_shape, num_heads * head_size)

    key = None
    if use_key:
        key = torch.randn(
            *positions_shape, num_kv_heads, head_size,
            device="cuda", dtype=dtype,
        )
        if flatten_heads:
            key = key.reshape(*positions_shape, num_kv_heads * head_size)

    query_ref = _rope_reference(query.clone(), positions, cache, is_neox, head_size)
    key_ref = (
        _rope_reference(key.clone(), positions, cache, is_neox, head_size)
        if key is not None
        else None
    )

    query_out, key_out = rotary_embedding_fuse(
        positions, query, key, head_size, cache, is_neox
    )
    torch.cuda.synchronize()

    assert query_out is query
    assert key_out is key
    query_diff = _max_abs_diff(query_out, query_ref)
    key_diff = _max_abs_diff(key_out, key_ref) if key_ref is not None else 0.0
    torch.testing.assert_close(query_out, query_ref, rtol=RTOL, atol=ATOL)
    if key_ref is not None:
        torch.testing.assert_close(key_out, key_ref, rtol=RTOL, atol=ATOL)
    return query_diff, key_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gpt_j"])
@pytest.mark.parametrize(
    "case_name,positions_shape,num_heads,num_kv_heads,flatten_heads,use_key",
    HEAD72_CASES,
    ids=[case[0] for case in HEAD72_CASES],
)
def test_rotary_embedding_fuse_head72_matches_torch(
    case_name: str,
    positions_shape: tuple[int, ...],
    num_heads: int,
    num_kv_heads: int,
    flatten_heads: bool,
    use_key: bool,
    dtype: torch.dtype,
    is_neox: bool,
) -> None:
    query_diff, key_diff = _run_correctness_case(
        case_name, positions_shape, num_heads, num_kv_heads,
        flatten_heads, use_key, dtype, is_neox,
    )
    style = "neox" if is_neox else "gpt_j"
    print(
        f"[Correctness Test] case={case_name}, dtype={_dtype_name(dtype)}, "
        f"style={style}, Q max diff={query_diff:.6f}, "
        f"K max diff={key_diff:.6f}  ✅ PASS"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gpt_j"])
@pytest.mark.parametrize(
    "head_size",
    OPTIMIZED_HEAD_SIZES + FALLBACK_HEAD_SIZES,
    ids=[
        *(f"optimized_{size}" for size in OPTIMIZED_HEAD_SIZES),
        *(f"fallback_{size}" for size in FALLBACK_HEAD_SIZES),
    ],
)
def test_rotary_embedding_fuse_head_size_dispatch(
    dtype: torch.dtype, is_neox: bool, head_size: int
) -> None:
    query_diff, key_diff = _run_correctness_case(
        "supported_heads", (3,), 16, 4, False, True,
        dtype, is_neox, head_size,
    )
    style = "neox" if is_neox else "gpt_j"
    print(
        f"[Correctness Test] head_size={head_size}, "
        f"dtype={_dtype_name(dtype)}, style={style}, "
        f"Q max diff={query_diff:.6f}, K max diff={key_diff:.6f}  ✅ PASS"
    )


@torch.no_grad()
def _benchmark_op(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    is_neox: bool,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(warmup):
        rotary_embedding_fuse(positions, query, key, 72, cache, is_neox)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        rotary_embedding_fuse(positions, query, key, 72, cache, is_neox)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def _benchmark_case(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    is_neox: bool,
    warmup: int,
    repeat: int,
) -> float:
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cache = _build_cos_sin_cache(max(num_tokens + 16, 128), 72, dtype)
    query = torch.randn(
        num_tokens, num_heads, 72, device="cuda", dtype=dtype
    )
    key = torch.randn(
        num_tokens, num_kv_heads, 72, device="cuda", dtype=dtype
    )
    return _benchmark_op(
        positions, query, key, cache, is_neox, warmup, repeat
    )


def _run_cli_correctness() -> None:
    print("\n=== 1. 精度验证 ===")
    print(
        f"{'Case':>12} | {'DType':>6} | {'Style':>6} | "
        f"{'Q Max Diff':>12} | {'K Max Diff':>12} | {'Result':>8}"
    )
    print("-" * 79)
    passed = 0
    total = len(HEAD72_CASES) * len(DTYPES) * len(ROPE_STYLES)
    for case in HEAD72_CASES:
        case_name, positions_shape, q_heads, kv_heads, flattened, use_key = case
        for dtype in DTYPES:
            for is_neox, style in ROPE_STYLES:
                q_diff, k_diff = _run_correctness_case(
                    case_name, positions_shape, q_heads, kv_heads,
                    flattened, use_key, dtype, is_neox,
                )
                passed += 1
                print(
                    f"{case_name:>12} | {_dtype_name(dtype):>6} | {style:>6} | "
                    f"{q_diff:12.6f} | {k_diff:12.6f} | {'✅ PASS':>8}"
                )
    print(f"\n🎉 精度测试通过：{passed}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument(
        "--tokens", nargs="+", type=int,
        default=[1, 8, 16, 128, 512, 2048, 4096],
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--is-neox", action="store_true")
    parser.add_argument("--name", default="current")
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip the correctness section and only run the benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print("\n=== 测试参数 ===")
    print(f"Environment : {args.name}")
    print(f"dtype       : {args.dtype}")
    print("head_size   : 72")
    print("rot_dim     : 72")
    print(f"num_heads   : {args.num_heads}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"is_neox     : {args.is_neox}")
    print(f"warmup      : {args.warmup}")
    print(f"repeat      : {args.repeat}")

    if not args.skip_correctness:
        _run_cli_correctness()

    print("\n=== 2. 性能测试 ===")
    print(f"{'Tokens':>12} | {'LightOp Time (us)':>18}")
    print("-" * 34)
    for num_tokens in args.tokens:
        latency_us = _benchmark_case(
            num_tokens, args.num_heads, args.num_kv_heads,
            dtype, args.is_neox, args.warmup, args.repeat,
        )
        print(f"{num_tokens:12d} | {latency_us:18.3f}")


if __name__ == "__main__":
    main()
