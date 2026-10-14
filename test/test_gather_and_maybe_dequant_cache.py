#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for the LightOp gather-cache operator."""

import statistics
from collections.abc import Callable

import pytest
import torch

from lightop import gather_and_maybe_dequant_cache


DEVICE = "cuda"
OUTPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
HEAD_DIMS = (320, 576)
WARMUP = 10
REPEAT = 50
ROUNDS = 5
BENCHMARK_CACHE_DTYPES = ("auto", "fp8", "fp8_e5m2")


def _fp8_dtype(device: str) -> torch.dtype:
    properties = torch.cuda.get_device_properties(device)
    arch_name = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    if arch_name.startswith("gfx94"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _is_fp8_fnuz(device: str) -> bool:
    properties = torch.cuda.get_device_properties(device)
    arch_name = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    return arch_name.startswith("gfx94")


def _decode_e5m2_bytes(data: torch.Tensor, fnuz: bool) -> torch.Tensor:
    """Independently decode raw OCP/FNUZ E5M2 bytes into FP32."""
    bits = data.to(torch.int32)
    sign = torch.where((bits & 0x80) == 0, 1.0, -1.0)
    exponent = (bits >> 2) & 0x1F
    mantissa = bits & 0x03
    bias = 16.0 if fnuz else 15.0

    subnormal = mantissa.float() * (2.0 ** (1.0 - bias - 2.0))
    normal = (1.0 + mantissa.float() / 4.0) * torch.pow(
        2.0, exponent.float() - bias
    )
    decoded = sign * torch.where(exponent == 0, subnormal, normal)

    if fnuz:
        # FNUZ has one unsigned zero.  The otherwise-negative-zero encoding
        # is NaN, while exponent 0x1f remains a finite exponent.
        return torch.where(bits == 0x80, float("nan"), decoded)

    decoded = torch.where(
        (exponent == 0x1F) & (mantissa == 0),
        sign * float("inf"),
        decoded,
    )
    return torch.where(
        (exponent == 0x1F) & (mantissa != 0), float("nan"), decoded
    )


def _small_metadata(
    device: str,
    seq_starts_values: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    block_table = torch.tensor(
        [[3, 1, 5], [4, 2, 0]], dtype=torch.int32, device=device
    )
    cu_seq_lens = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    token_to_seq = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )
    seq_starts = (
        None
        if seq_starts_values is None
        else torch.tensor(seq_starts_values, dtype=torch.int32, device=device)
    )
    return block_table, cu_seq_lens, token_to_seq, seq_starts


def _make_cache(
    values: torch.Tensor,
    kv_cache_dtype: str,
    output_dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    if kv_cache_dtype == "auto":
        return values.to(output_dtype)
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return values.to(_fp8_dtype(device)).view(torch.uint8)
    if kv_cache_dtype == "fp8_e5m2":
        # Raw bytes avoid relying on PyTorch exposing the platform's FNUZ
        # dtype.  Repetition covers every possible E5M2 encoding.
        return (
            torch.arange(values.numel(), dtype=torch.int64, device=device)
            .remainder(256)
            .to(torch.uint8)
            .view(values.shape)
        )
    raise ValueError(f"unsupported test cache dtype: {kv_cache_dtype}")


def _gather_reference(
    src_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    seq_starts_values: list[int] | None,
    kv_cache_dtype: str,
    output_dtype: torch.dtype,
    scale_value: float,
    device: str,
) -> torch.Tensor:
    block_size = src_cache.size(1)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    sequence_ids = torch.arange(
        len(seq_lens), dtype=torch.int64, device=device
    ).repeat_interleave(seq_lens_tensor)
    cumulative_lens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            seq_lens_tensor.cumsum(0)[:-1],
        )
    )
    token_offsets = torch.arange(
        sum(seq_lens), dtype=torch.int64, device=device
    ) - cumulative_lens.repeat_interleave(seq_lens_tensor)
    if seq_starts_values is None:
        cache_positions = token_offsets
    else:
        starts = torch.tensor(
            seq_starts_values, dtype=torch.int64, device=device
        )
        cache_positions = token_offsets + starts.repeat_interleave(
            seq_lens_tensor
        )
    table_indices = cache_positions // block_size
    slot_ids = cache_positions % block_size
    block_ids = block_table[sequence_ids, table_indices].to(torch.int64)
    gathered = src_cache[block_ids, slot_ids]
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        gathered = (
            gathered.contiguous().view(_fp8_dtype(device)).float()
            * scale_value
        )
    elif kv_cache_dtype == "fp8_e5m2":
        gathered = (
            _decode_e5m2_bytes(gathered, fnuz=_is_fp8_fnuz(device))
            * scale_value
        )
    return gathered.to(output_dtype)


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_starts_values", [None, [1, 2]])
@pytest.mark.parametrize("scale_value", [1.0, 1.5])
@pytest.mark.parametrize(
    "kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"]
)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache(
    head_dim: int,
    seq_starts_values: list[int] | None,
    scale_value: float,
    kv_cache_dtype: str,
    output_dtype: torch.dtype,
) -> None:
    """Cover every output type, head size, scale, and start offset."""
    device = DEVICE
    block_size = 4
    num_blocks = 6
    seq_lens = [5, 3]
    num_tokens = sum(seq_lens)
    block_table, cu_seq_lens, token_to_seq, seq_starts = _small_metadata(
        device, seq_starts_values
    )
    values = torch.linspace(
        -3.0,
        3.0,
        num_blocks * block_size * head_dim,
        dtype=torch.float32,
        device=device,
    ).view(num_blocks, block_size, head_dim)
    src_cache = _make_cache(values, kv_cache_dtype, output_dtype, device)
    scale = torch.tensor(scale_value, dtype=torch.float32, device=device)
    dst = torch.empty(
        (num_tokens, head_dim), dtype=output_dtype, device=device
    )

    gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )
    torch.cuda.synchronize(device)

    expected = _gather_reference(
        src_cache,
        block_table,
        seq_lens,
        seq_starts_values,
        kv_cache_dtype,
        output_dtype,
        scale_value,
        device,
    )
    torch.testing.assert_close(
        dst, expected, rtol=0.0, atol=0.0, equal_nan=True
    )


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("float16", "auto"),
        ("bfloat16", "auto"),
        ("fp8", "fp8_e4m3"),
        ("fp8_ds_mla", "fp8_e4m3"),
    ],
)
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache_dtype_aliases(
    alias: str,
    canonical: str,
) -> None:
    """Keep every cache-dtype spelling accepted by the clean vLLM dispatch."""
    device = DEVICE
    head_dim = 576
    num_tokens = 8
    block_table, cu_seq_lens, token_to_seq, seq_starts = _small_metadata(
        device, [1, 2]
    )
    values = torch.linspace(
        -2.0,
        2.0,
        6 * 4 * head_dim,
        dtype=torch.float32,
        device=device,
    ).view(6, 4, head_dim)
    src_cache = _make_cache(values, canonical, torch.bfloat16, device)
    scale = torch.tensor(1.25, dtype=torch.float32, device=device)
    alias_dst = torch.empty(
        (num_tokens, head_dim), dtype=torch.bfloat16, device=device
    )
    canonical_dst = torch.empty_like(alias_dst)

    for cache_dtype, output in (
        (alias, alias_dst),
        (canonical, canonical_dst),
    ):
        gather_and_maybe_dequant_cache(
            src_cache,
            output,
            block_table,
            cu_seq_lens,
            token_to_seq,
            num_tokens,
            cache_dtype,
            scale,
            seq_starts,
        )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(alias_dst, canonical_dst, rtol=0.0, atol=0.0)


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache_rejects_invalid_contract() -> None:
    """Reject unsafe dtype/storage combinations before launching the kernel."""
    device = DEVICE
    head_dim = 320
    block_table, cu_seq_lens, token_to_seq, _ = _small_metadata(device, None)
    src_cache = torch.zeros(
        (6, 4, head_dim), dtype=torch.bfloat16, device=device
    )
    dst = torch.empty((8, head_dim), dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match="token_to_seq must be int32"):
        gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq.to(torch.int64),
            8,
            "auto",
            scale,
        )

    with pytest.raises(RuntimeError, match="scale must be float32"):
        gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            8,
            "auto",
            scale.to(torch.float64),
        )

    with pytest.raises(
        RuntimeError, match="FP8 cache modes require one-byte"
    ):
        gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            8,
            "fp8_e5m2",
            scale,
        )

    with pytest.raises(
        RuntimeError, match="auto cache mode requires src_cache and dst"
    ):
        gather_and_maybe_dequant_cache(
            src_cache.float(),
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            8,
            "auto",
            scale,
        )


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e5m2"])
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache_original_workload(
    kv_cache_dtype: str,
) -> None:
    """Validate the shape and FP32 output used by vLLM's original unit test."""
    device = DEVICE
    torch.manual_seed(0)

    batch_size = 8
    block_size = 16
    num_blocks = 1024
    sequence_length = 512
    head_dim = 576
    num_tokens = batch_size * sequence_length
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)

    values = torch.randn(
        num_blocks,
        block_size,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    src_cache = _make_cache(values, kv_cache_dtype, torch.float32, device)
    cu_seq_lens = torch.arange(
        0,
        num_tokens + 1,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )
    token_to_seq = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(sequence_length)
    block_table = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=device
    )
    for batch_id in range(batch_size):
        block_table[batch_id] = torch.randperm(
            num_blocks, dtype=torch.int64, device=device
        ).to(torch.int32)

    dst = torch.empty(
        (num_tokens, head_dim), dtype=torch.float32, device=device
    )
    gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        None,
    )
    torch.cuda.synchronize(device)

    expected = _gather_reference(
        src_cache,
        block_table,
        [sequence_length] * batch_size,
        None,
        kv_cache_dtype,
        torch.float32,
        0.1,
        device,
    )
    torch.testing.assert_close(
        dst, expected, rtol=0.0, atol=0.0, equal_nan=True
    )


def _median_kernel_us(fn: Callable[[], None]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPEAT):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / REPEAT)
    return statistics.median(samples)


@torch.inference_mode()
def _benchmark_case(kv_cache_dtype: str, head_dim: int) -> float:
    """Check one production-shaped BF16 case before measuring its latency."""
    output_dtype = torch.bfloat16
    torch.manual_seed(20260813)

    batch_size = 8
    sequence_length = 512
    block_size = 16
    num_blocks = 1024
    num_tokens = batch_size * sequence_length

    values = torch.randn(
        num_blocks,
        block_size,
        head_dim,
        device=DEVICE,
        dtype=torch.float32,
    )
    src_cache = _make_cache(
        values, kv_cache_dtype, output_dtype, DEVICE
    )
    block_table = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=DEVICE
    )
    for batch_id in range(batch_size):
        block_table[batch_id] = torch.randperm(
            num_blocks, dtype=torch.int64, device=DEVICE
        ).to(torch.int32)
    cu_seq_lens = torch.arange(
        0,
        num_tokens + 1,
        sequence_length,
        dtype=torch.int32,
        device=DEVICE,
    )
    token_to_seq = torch.arange(
        batch_size, dtype=torch.int32, device=DEVICE
    ).repeat_interleave(sequence_length)
    scale = torch.tensor(0.1, dtype=torch.float32, device=DEVICE)
    dst = torch.empty(
        (num_tokens, head_dim), dtype=output_dtype, device=DEVICE
    )

    def run_lightop() -> None:
        gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            num_tokens,
            kv_cache_dtype,
            scale,
            None,
        )

    # Never record a fast but incorrect implementation as the benchmark.
    run_lightop()
    torch.cuda.synchronize()
    expected = _gather_reference(
        src_cache,
        block_table,
        [sequence_length] * batch_size,
        None,
        kv_cache_dtype,
        output_dtype,
        0.1,
        DEVICE,
    )
    torch.testing.assert_close(
        dst, expected, rtol=0.0, atol=0.0, equal_nan=True
    )
    return _median_kernel_us(run_lightop)


def _run_standalone_correctness() -> int:
    """Run the same matrix as pytest when this file is used as a script."""
    case_count = 0
    for output_dtype in OUTPUT_DTYPES:
        for kv_cache_dtype in ("auto", "fp8_e4m3", "fp8_e5m2"):
            for scale_value in (1.0, 1.5):
                for seq_starts_values in (None, [1, 2]):
                    for head_dim in HEAD_DIMS:
                        test_gather_and_maybe_dequant_cache(
                            head_dim,
                            seq_starts_values,
                            scale_value,
                            kv_cache_dtype,
                            output_dtype,
                        )
                        case_count += 1

    for alias, canonical in (
        ("float16", "auto"),
        ("bfloat16", "auto"),
        ("fp8", "fp8_e4m3"),
        ("fp8_ds_mla", "fp8_e4m3"),
    ):
        test_gather_and_maybe_dequant_cache_dtype_aliases(alias, canonical)
        case_count += 1

    test_gather_and_maybe_dequant_cache_rejects_invalid_contract()
    case_count += 1

    for kv_cache_dtype in ("auto", "fp8", "fp8_e5m2"):
        test_gather_and_maybe_dequant_cache_original_workload(kv_cache_dtype)
        case_count += 1

    return case_count


def main() -> None:
    assert torch.cuda.is_available(), "this test requires a GPU"
    assert torch.version.hip is not None, "LightOp requires ROCm"

    properties = torch.cuda.get_device_properties(0)
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(
        f"device={torch.cuda.get_device_name(0)}, "
        f"gcn={getattr(properties, 'gcnArchName', 'unknown')}"
    )

    case_count = _run_standalone_correctness()
    print(f"correctness=PASS, cases={case_count}")
    print(
        f"warmup={WARMUP}, repeat={REPEAT}, rounds={ROUNDS}, "
        "tokens=4096, dst=bfloat16"
    )
    print()
    print("| cache dtype | head dim | median (us) | correctness |")
    print("| --- | ---: | ---: | --- |")
    for kv_cache_dtype in BENCHMARK_CACHE_DTYPES:
        for head_dim in HEAD_DIMS:
            median_us = _benchmark_case(kv_cache_dtype, head_dim)
            print(
                f"| {kv_cache_dtype} | {head_dim} | "
                f"{median_us:.3f} | PASS |"
            )


if __name__ == "__main__":
    main()
