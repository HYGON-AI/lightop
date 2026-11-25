# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the Triton sparse-mask TopK against a torch reference.

The reference below re-implements the selection rule of
``topk_transform_decode_kernel_sparse_mask`` / ``sparse_mask_exact_topk``
directly from its documented behaviour:

* a masked-out position behaves like a ``+0.0`` score;
* the fast path (``positive_count <= topk`` and
  ``positive_count + masked_out_count >= topk``) emits every strictly positive
  valid position and fills the rest with the lowest-index masked-out
  positions in ascending logical order;
* otherwise the generic path ranks every valid position by score.

The Triton path must produce the same selected *set*; the reference contract
leaves the output order unspecified.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lightop.attention import (  # noqa: E402
    fast_topk_transform_sparse_mask_fused,
    fast_topk_transform_sparse_mask_fused_v2,
    sparse_mask_topk_triton_enabled,
)

TOPK = 2048


def _unpack_mask(nonzero_mask: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(16, device=nonzero_mask.device, dtype=torch.int32)
    bits = (nonzero_mask.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(nonzero_mask.shape[0], -1)[:, :width].to(torch.bool)


def reference_sparse_mask_topk(
    score: torch.Tensor,
    nonzero_mask: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows, width = score.shape
    masked = _unpack_mask(nonzero_mask, width)
    in_range = torch.arange(width, device=score.device) < lengths.unsqueeze(1)
    valid = masked & in_range
    positive = valid & (score > 0.0)
    neg_inf = torch.full_like(score, float("-inf"))
    pos_score = torch.where(positive, score, neg_inf)
    valid_score = torch.where(valid, score, neg_inf)
    invalid = in_range & ~masked

    out = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
    for row in range(rows):
        n_pos = int(positive[row].sum())
        n_zero = int(invalid[row].sum())
        if n_pos <= topk and n_pos + n_zero >= topk:
            picked = torch.argsort(-pos_score[row], stable=True)[:topk]
            chosen = picked.tolist()
            need = topk - n_pos
            if need > 0:
                fill = torch.nonzero(invalid[row], as_tuple=False).flatten()[:need]
                chosen = chosen[:n_pos] + fill.tolist()
        else:
            chosen = torch.argsort(-valid_score[row], stable=True)[:topk].tolist()
        chosen = (chosen + [0] * topk)[:topk]
        out[row] = page_table[row, torch.tensor(chosen, device=score.device)]
    return out


def _make_inputs(rows: int, width: int, density: float, lengths_mode: str = "full"):
    device = "cuda"
    torch.manual_seed(rows * 131 + width + int(density * 100))
    score = torch.randn(rows, width, device=device, dtype=torch.float32) * 4.0
    mask_words = (width + 15) // 16
    keep = torch.rand(rows, mask_words, device=device) < density
    bits = torch.zeros(rows, mask_words * 16, dtype=torch.int32, device=device)
    bits[:, : mask_words * 16] = keep.repeat_interleave(16, dim=1)
    nonzero_mask = torch.zeros(rows, mask_words, dtype=torch.int16, device=device)
    for bit in range(16):
        nonzero_mask |= (bits[:, bit::16][:, :mask_words] << bit).to(torch.int16)
    if lengths_mode == "full":
        lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    else:
        lengths = torch.randint(
            width // 2, width + 1, (rows,), device=device, dtype=torch.int32
        )
    page_table = (
        torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1).contiguous()
    )
    cu_seqlens_q = torch.arange(0, rows + 1, device=device, dtype=torch.int32)
    return score, nonzero_mask, lengths, page_table, cu_seqlens_q


def _as_set(out: torch.Tensor) -> torch.Tensor:
    return out.to(torch.int64).sort(dim=-1).values


CASES = [
    # rows, width, density, lengths_mode
    (4, 4096, 1.00, "full"),
    (4, 8192, 0.50, "full"),
    (2, 8192, 0.10, "full"),
    (4, 16384, 0.25, "trunc"),
    (3, 32768, 0.75, "full"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a DCU device")
@pytest.mark.parametrize("rows,width,density,lengths_mode", CASES)
def test_matches_reference(rows, width, density, lengths_mode):
    score, nonzero_mask, lengths, page_table, cu_seqlens_q = _make_inputs(
        rows, width, density, lengths_mode
    )
    expected = reference_sparse_mask_topk(score, nonzero_mask, lengths, page_table, TOPK)
    got = fast_topk_transform_sparse_mask_fused_v2(
        score, nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
    )
    assert got.shape == (rows, TOPK)
    assert got.dtype == torch.int32
    assert torch.equal(_as_set(got), _as_set(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a DCU device")
@pytest.mark.parametrize("rows,width,density,_mode", CASES)
def test_matches_hip_kernel(rows, width, density, _mode):
    """The HIP kernel is the production reference; compare as a set."""
    score, nonzero_mask, lengths, page_table, cu_seqlens_q = _make_inputs(
        rows, width, density, "full"
    )
    expected = fast_topk_transform_sparse_mask_fused(
        score, nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
    )
    got = fast_topk_transform_sparse_mask_fused_v2(
        score, nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
    )
    assert torch.equal(_as_set(got), _as_set(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a DCU device")
def test_deterministic():
    score, nonzero_mask, lengths, page_table, cu_seqlens_q = _make_inputs(4, 16384, 0.25)
    first = fast_topk_transform_sparse_mask_fused_v2(
        score, nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
    )
    second = fast_topk_transform_sparse_mask_fused_v2(
        score, nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
    )
    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a DCU device")
def test_rejects_bad_shape_and_dtype():
    score, nonzero_mask, lengths, page_table, cu_seqlens_q = _make_inputs(2, 4096, 1.0)
    with pytest.raises(ValueError):
        fast_topk_transform_sparse_mask_fused_v2(
            score.to(torch.bfloat16), nonzero_mask, lengths, page_table, cu_seqlens_q, TOPK
        )
    with pytest.raises(ValueError):
        fast_topk_transform_sparse_mask_fused_v2(
            score, nonzero_mask[:, :-1], lengths, page_table, cu_seqlens_q, TOPK
        )


def test_switch_env_parsing(monkeypatch):
    monkeypatch.delenv("LIGHTOP_SPARSE_MASK_TOPK_TRITON", raising=False)
    assert sparse_mask_topk_triton_enabled() is False
    monkeypatch.setenv("LIGHTOP_SPARSE_MASK_TOPK_TRITON", "1")
    assert sparse_mask_topk_triton_enabled() is True
    monkeypatch.setenv("LIGHTOP_SPARSE_MASK_TOPK_TRITON", "0")
    assert sparse_mask_topk_triton_enabled() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
