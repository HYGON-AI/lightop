# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import statistics

import pytest
import torch

from lightop import op


ENTRY_BYTES_WITH_ROPE = 656
ENTRY_BYTES_NO_ROPE = 528
ENTRY_BYTES = ENTRY_BYTES_WITH_ROPE
NOPE_DIM = 512
HEAD_DIM_WITH_ROPE = 576
HEAD_DIM_NO_ROPE = 512
HEAD_DIM = HEAD_DIM_WITH_ROPE
NUM_GROUPS = 4
GROUP_SIZE = NOPE_DIM // NUM_GROUPS
FP8_DTYPE = torch.float8_e4m3fn
TARGET_KV_TOKENS = 81920
WARMUP = 5
REPEAT = 20
ROUNDS = 5


def _make_packed_cache(
    num_physical_tokens: int,
    entry_bytes: int = ENTRY_BYTES,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    torch.manual_seed(20260717)
    rope_dim = head_dim - NOPE_DIM
    assert rope_dim >= 0
    assert entry_bytes == NOPE_DIM + NUM_GROUPS * 4 + rope_dim * 2

    fp8_values = (
        torch.randn(
            num_physical_tokens,
            NOPE_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        * 3.0
    ).to(FP8_DTYPE)
    scales = (
        0.25
        + torch.rand(
            num_physical_tokens,
            NUM_GROUPS,
            device="cuda",
            dtype=torch.float32,
        )
        * 1.75
    ).contiguous()

    packed = torch.empty(
        (num_physical_tokens, entry_bytes),
        device="cuda",
        dtype=torch.uint8,
    )
    packed[:, :NOPE_DIM].copy_(fp8_values.view(torch.uint8))
    packed[:, NOPE_DIM : NOPE_DIM + NUM_GROUPS * 4].copy_(
        scales.view(torch.uint8)
    )
    if rope_dim > 0:
        rope = torch.randn(
            num_physical_tokens,
            rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        packed[:, NOPE_DIM + NUM_GROUPS * 4 :].copy_(rope.view(torch.uint8))
    return packed.view(num_physical_tokens, 1, 1, entry_bytes)


def _time_cuda(fn) -> float:
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


def _kv_reference(
    packed_cache: torch.Tensor,
    page_table_flat: torch.Tensor,
) -> torch.Tensor:
    entry_bytes = packed_cache.shape[-1]
    rope_bytes = entry_bytes - NOPE_DIM - NUM_GROUPS * 4
    assert rope_bytes >= 0 and rope_bytes % 2 == 0
    head_dim = NOPE_DIM + rope_bytes // 2

    rows = packed_cache.view(-1, entry_bytes).index_select(
        0, page_table_flat.to(torch.int64)
    )
    fp8_values = rows[:, :NOPE_DIM].contiguous().view(FP8_DTYPE).float()
    scales = (
        rows[:, NOPE_DIM : NOPE_DIM + NUM_GROUPS * 4]
        .contiguous()
        .view(torch.float32)
    )
    nope = (
        fp8_values.view(-1, NUM_GROUPS, GROUP_SIZE) * scales[:, :, None]
    ).reshape(-1, NOPE_DIM).to(torch.bfloat16)

    output = torch.empty(
        (page_table_flat.numel(), 1, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output[:, 0, :NOPE_DIM] = nope
    if head_dim > NOPE_DIM:
        rope = rows[:, NOPE_DIM + NUM_GROUPS * 4 :].contiguous().view(
            torch.bfloat16
        )
        output[:, 0, NOPE_DIM:] = rope
    return output


@pytest.mark.parametrize(
    ("entry_bytes", "head_dim"),
    [
        (ENTRY_BYTES_WITH_ROPE, HEAD_DIM_WITH_ROPE),
        (ENTRY_BYTES_NO_ROPE, HEAD_DIM_NO_ROPE),
    ],
)
@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_prefill_gather_and_upconvert_matches_native(
    entry_bytes, head_dim
) -> None:
    packed_cache = _make_packed_cache(257, entry_bytes, head_dim)
    page_table_flat = torch.tensor(
        [256, 0, 17, 17, 128, 3, 200, 42, 1, 255, 99, 5, 6, 7, 8, 9, 10],
        device="cuda",
        dtype=torch.int32,
    )
    kv_dst = torch.empty(
        (page_table_flat.numel(), 1, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )

    op.prefill_gather_and_upconvert_fp8_kv_cache(
        packed_cache,
        page_table_flat,
        kv_dst,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        kv_dst,
        _kv_reference(packed_cache, page_table_flat),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("entry_bytes", "head_dim"),
    [
        (ENTRY_BYTES_WITH_ROPE, HEAD_DIM_WITH_ROPE),
        (ENTRY_BYTES_NO_ROPE, HEAD_DIM_NO_ROPE),
    ],
)
@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_prefill_gather_and_upconvert_empty_kv(entry_bytes, head_dim) -> None:
    packed_cache = _make_packed_cache(1, entry_bytes, head_dim)
    page_table_flat = torch.empty(0, device="cuda", dtype=torch.int32)
    kv_dst = torch.empty(
        (0, 1, head_dim), device="cuda", dtype=torch.bfloat16
    )

    op.prefill_gather_and_upconvert_fp8_kv_cache(
        packed_cache,
        page_table_flat,
        kv_dst,
    )
    torch.cuda.synchronize()
    assert kv_dst.numel() == 0


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_prefill_gather_and_upconvert_rejects_flat_src_cache() -> None:
    rows = 41
    packed_cache = _make_packed_cache(
        rows, ENTRY_BYTES_NO_ROPE, HEAD_DIM_NO_ROPE
    ).reshape(-1)
    page_table_flat = torch.arange(rows, device="cuda", dtype=torch.int32)
    kv_dst = torch.empty(
        (rows, 1, HEAD_DIM_NO_ROPE), device="cuda", dtype=torch.bfloat16
    )

    with pytest.raises(RuntimeError, match="1D src_cache"):
        op.prefill_gather_and_upconvert_fp8_kv_cache(
            packed_cache, page_table_flat, kv_dst
        )


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_prefill_gather_and_upconvert_target_shape() -> None:
    """Compare the exact GLM P0 KV shape with the original Triton kernel."""
    from sglang.srt.layers.attention.nsa.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    num_physical_tokens = TARGET_KV_TOKENS + 17
    packed_u8 = _make_packed_cache(num_physical_tokens)
    packed_fp8 = packed_u8.view(FP8_DTYPE)
    page_table_flat = torch.randperm(
        num_physical_tokens, device="cuda", dtype=torch.int64
    )[:TARGET_KV_TOKENS].to(torch.int32).contiguous()
    lightop_kv = torch.empty(
        (TARGET_KV_TOKENS, 1, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )

    baseline_kv = dequantize_k_cache_paged(packed_fp8, page_table_flat)
    op.prefill_gather_and_upconvert_fp8_kv_cache(
        packed_fp8,
        page_table_flat,
        lightop_kv,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lightop_kv, baseline_kv, rtol=0.0, atol=0.0)

    latest_baseline_kv = None

    def run_dequant_only() -> None:
        nonlocal latest_baseline_kv
        latest_baseline_kv = dequantize_k_cache_paged(
            packed_fp8, page_table_flat
        )

    def run_lightop() -> None:
        op.prefill_gather_and_upconvert_fp8_kv_cache(
            packed_fp8,
            page_table_flat,
            lightop_kv,
        )

    dequant_us = _time_cuda(run_dequant_only)
    lightop_us = _time_cuda(run_lightop)
    assert latest_baseline_kv is not None
    print(
        "\nP0 target benchmark "
        f"kv=[{TARGET_KV_TOKENS},1,576]: "
        f"dequant_only={dequant_us:.3f} us, "
        f"lightop={lightop_us:.3f} us, "
        f"speedup={dequant_us / lightop_us:.3f}x"
    )
    assert lightop_us < dequant_us
