#!/usr/bin/env python3
"""Regression tests for fused decode gather and compact-index generation."""

import pytest
import torch

from lightop import op


ROWS = 48
REQUESTS = 8
TOPK = 2048
HEAD_DIM = 512
ENTRY_BYTES_NO_ROPE = 528
ENTRY_BYTES_WITH_ROPE = 656
ENTRY_BYTES = ENTRY_BYTES_WITH_ROPE
PHYSICAL_ROWS = 33772


def _make_packed_cache(entry_bytes: int = ENTRY_BYTES) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260720)
    cache = torch.empty(
        (PHYSICAL_ROWS, entry_bytes), dtype=torch.uint8, device="cuda"
    )

    # Avoid the two E4M3FN NaN codes. The production quantizer emits finite
    # cache bytes, and exhaustive finite-code conversion is covered separately.
    raw = torch.randint(
        0, 126, (PHYSICAL_ROWS, HEAD_DIM),
        dtype=torch.uint8, device="cuda", generator=generator,
    )
    raw.bitwise_xor_(
        torch.randint(
            0, 2, raw.shape, dtype=torch.uint8, device="cuda",
            generator=generator,
        ) << 7
    )
    scales = torch.rand(
        (PHYSICAL_ROWS, 4), dtype=torch.float32, device="cuda",
        generator=generator,
    ) + 0.125
    cache[:, :HEAD_DIM].copy_(raw)
    cache[:, HEAD_DIM : HEAD_DIM + 16].copy_(
        scales.contiguous().view(torch.uint8)
    )
    if entry_bytes > HEAD_DIM + 16:
        cache[:, HEAD_DIM + 16 :].zero_()
    return cache


@pytest.mark.parametrize(
    "entry_bytes", [ENTRY_BYTES_NO_ROPE, ENTRY_BYTES_WITH_ROPE]
)
@torch.no_grad()
def test_dp_padded_indices_are_bitwise_exact(entry_bytes: int) -> None:
    cache = _make_packed_cache(entry_bytes)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    token_ids = torch.randint(
        0, PHYSICAL_ROWS, (ROWS, TOPK),
        dtype=torch.int32, device="cuda", generator=generator,
    )
    token_ids[:, 31::257] = -1
    token_ids[:, 73::263] = PHYSICAL_ROWS + 7
    # The target P0 is eight real rows followed by DP-padding rows.  It is not
    # eight request lengths repeated over six rows each.
    token_ids[REQUESTS:] = -1

    request_lengths = torch.tensor(
        [2048, 2047, 1536, 1024, 17, 0, 2000, 1],
        dtype=torch.int32,
        device="cuda",
    )
    # Sparse FlashMLA's exact mask is the per-row physical index sentinel.  A
    # shorter lengths tensor must not be inferred as request-major grouping.
    id_only_lengths = torch.full(
        (ROWS,), TOPK, dtype=torch.int32, device="cuda"
    )

    expected = torch.empty(
        (ROWS, TOPK, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    actual = torch.empty_like(expected)
    compact = torch.empty((ROWS, TOPK), dtype=torch.int32, device="cuda")

    op.decode_gather_and_up_convert(
        cache, token_ids, expected, id_only_lengths
    )
    op.decode_gather_and_up_convert_with_indices(
        cache, token_ids, actual, request_lengths, compact
    )
    torch.cuda.synchronize()

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    expected_compact = torch.arange(
        ROWS * TOPK, dtype=torch.int32, device="cuda"
    ).view(ROWS, TOPK)
    invalid = (
        (token_ids < 0)
        | (token_ids >= PHYSICAL_ROWS)
    )
    expected_compact.masked_fill_(invalid, -1)
    assert torch.equal(compact, expected_compact)


@pytest.mark.parametrize(
    "entry_bytes", [ENTRY_BYTES_NO_ROPE, ENTRY_BYTES_WITH_ROPE]
)
@torch.no_grad()
def test_vector_alignment_contract_is_checked(entry_bytes: int) -> None:
    token_ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")
    compact = torch.empty((1, 1), dtype=torch.int32, device="cuda")
    aligned_dst = torch.empty(
        (1, 1, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )

    misaligned_cache = torch.empty(
        (entry_bytes + 1,), dtype=torch.uint8, device="cuda"
    )[1:]
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        op.decode_gather_and_up_convert_with_indices(
            misaligned_cache, token_ids, aligned_dst, lengths, compact
        )

    aligned_cache = torch.empty(
        (entry_bytes,), dtype=torch.uint8, device="cuda"
    )
    misaligned_dst = torch.empty(
        (HEAD_DIM + 1,), dtype=torch.bfloat16, device="cuda"
    )[1:].view(1, 1, HEAD_DIM)
    with pytest.raises(RuntimeError, match="32-byte aligned"):
        op.decode_gather_and_up_convert_with_indices(
            aligned_cache, token_ids, misaligned_dst, lengths, compact
        )


@torch.no_grad()
def test_with_indices_rejects_528_to_576() -> None:
    cache = torch.empty(
        (1, ENTRY_BYTES_NO_ROPE), dtype=torch.uint8, device="cuda"
    )
    token_ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")
    dst = torch.empty((1, 1, 576), dtype=torch.bfloat16, device="cuda")
    compact = torch.empty((1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="576 requires"):
        op.decode_gather_and_up_convert_with_indices(
            cache, token_ids, dst, lengths, compact
        )


if __name__ == "__main__":
    for entry_bytes in (ENTRY_BYTES_NO_ROPE, ENTRY_BYTES_WITH_ROPE):
        test_dp_padded_indices_are_bitwise_exact(entry_bytes)
        test_vector_alignment_contract_is_checked(entry_bytes)
    test_with_indices_rejects_528_to_576()
    print("PASS")
