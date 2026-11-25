# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of the sparse-mask TopK transform.

Alternative to :func:`lightop.fuse_topk_transform.fast_topk_transform_sparse_mask_fused`,
which is served by the HIP kernel ``topk_transform_decode_kernel_sparse_mask``.

Motivation
----------
That kernel launches one block per row (``row = blockIdx.x``). With the four
query rows of an EAGLE target-verify step, only 4 of 64 CUs are busy (6.25 %),
and the generic path rescans the row seven times inside one block with roughly
ninety block-wide barriers. On gfx938 it costs 1.648 ms per call and 43.9 % of
a decode step.

This module restores parallelism in the row dimension (grid = rows x blocks)
and reproduces the reference selection rule, including its two paths:

* **Fast path** (what production hits). The condition is
  ``positive_count + masked_out_count >= topk``: emit every valid position
  whose score is strictly positive, then fill the remaining slots with the
  lowest-index masked-out positions, in ascending logical order.
  Masked-out positions are the implicit ``+0.0`` entries of the contract
  ("Masked-out positions are reconstructed as positive zero").
* **Generic path** otherwise (including "fewer candidates than topk"): rank
  every valid position by score and take the top ``topk``.

Only the generic row subset (rare) pays for an extra top-k; the fast path
needs a single mask probe plus one top-k.
"""

from typing import Optional, Tuple

import torch

try:  # Triton is optional: the torch fallback keeps this module importable.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


#: Positions per program for the mask probe pass (32 KB of fp32 per program).
MASK_BLOCK = 8192
#: Warps per program for the mask probe pass.
MASK_WARPS = 8


def is_available() -> bool:
    """Return whether the Triton fast path can run on this build."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


if _TRITON_AVAILABLE:

    @triton.jit
    def _mask_probe_kernel(
        score_ptr,
        mask_ptr,
        length_ptr,
        pos_score_ptr,
        valid_score_ptr,
        invalid_ptr,
        width,
        mask_width,
        BLOCK: tl.constexpr,
    ):
        """Classify each position once, in a single coalesced sweep.

        ``pos_score``   score where the position is valid and ``> 0``, else -inf
        ``valid_score`` score where the position is valid, else -inf
        ``invalid``     1 for in-range masked-out positions (the +0.0 pool)
        """
        row = tl.program_id(0)
        pid = tl.program_id(1)

        off = pid * BLOCK + tl.arange(0, BLOCK)
        in_range = off < width

        word = tl.load(mask_ptr + row * mask_width + off // 16, mask=in_range, other=0)
        word = word.to(tl.int32)
        masked = ((word >> (off % 16).to(tl.int32)) & 1) != 0

        length = tl.load(length_ptr + row).to(tl.int32)
        valid = masked & (off < length)

        score = tl.load(score_ptr + row * width + off, mask=in_range, other=0.0)
        positive = valid & (score > 0.0)
        neg_inf = float("-inf")

        tl.store(pos_score_ptr + row * width + off,
                 tl.where(positive, score, neg_inf), mask=in_range)
        tl.store(valid_score_ptr + row * width + off,
                 tl.where(valid, score, neg_inf), mask=in_range)
        tl.store(invalid_ptr + row * width + off,
                 tl.where(in_range & (off < length) & (~masked), 1, 0).to(tl.uint8),
                 mask=in_range)


def _probe_triton(score: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
    rows, width = score.shape
    pos_score = torch.empty_like(score)
    valid_score = torch.empty_like(score)
    invalid = torch.empty((rows, width), dtype=torch.uint8, device=score.device)
    grid = (rows, triton.cdiv(width, MASK_BLOCK))
    _mask_probe_kernel[grid](
        score, mask, lengths, pos_score, valid_score, invalid,
        width, mask.shape[1], BLOCK=MASK_BLOCK, num_warps=MASK_WARPS,
    )
    return pos_score, valid_score, invalid


def _probe_torch(score: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
    """Pure-torch equivalent of :func:`_probe_triton` (fallback / CPU tests)."""
    rows, width = score.shape
    words = torch.arange(16, device=score.device, dtype=torch.int32)
    bits = (mask.to(torch.int32).unsqueeze(-1) >> words) & 1
    masked = bits.reshape(rows, -1)[:, :width].to(torch.bool)
    in_range = torch.arange(width, device=score.device) < lengths.unsqueeze(1)
    valid = masked & in_range
    neg_inf = torch.full_like(score, float("-inf"))
    pos_score = torch.where(valid & (score > 0.0), score, neg_inf)
    valid_score = torch.where(valid, score, neg_inf)
    invalid = (in_range & ~masked).to(torch.uint8)
    return pos_score, valid_score, invalid


def _probe(score: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
    if is_available() and score.is_cuda:
        return _probe_triton(score, mask, lengths)
    return _probe_torch(score, mask, lengths)


def _expand_page_table(page_table: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor],
                       rows: int) -> torch.Tensor:
    if page_table.dim() != 2:
        raise ValueError("page_table_size_1 must be 2-D")
    if page_table.shape[0] == rows:
        return page_table
    if cu_seqlens_q is None:
        raise ValueError(
            "page_table_size_1 has a different row count than score and "
            "cu_seqlens_q was not provided"
        )
    return page_table[cu_seqlens_q[1:] - 1]


def sparse_mask_topk_triton(
    score: torch.Tensor,
    nonzero_mask: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    topk: int,
) -> torch.Tensor:
    """Sparse-mask TopK plus page-table transform (Triton path).

    Same contract as :func:`fast_topk_transform_sparse_mask_fused`; the result
    is a ``(rows, topk)`` int32 tensor of page ids. Ordering is unspecified by
    the reference contract; the selected set matches.
    """
    if score.dim() != 2 or score.dtype != torch.float32:
        raise ValueError("score must be float32 [rows, width]")
    rows, width = score.shape
    if not 0 < topk <= width:
        raise ValueError("topk must be in (0, width]")
    if lengths.dtype != torch.int32 or tuple(lengths.shape) != (rows,):
        raise ValueError("lengths must be int32 [rows]")
    if (nonzero_mask.dtype != torch.int16 or nonzero_mask.dim() != 2
            or tuple(nonzero_mask.shape) != (rows, (width + 15) // 16)):
        raise ValueError("nonzero_mask must be int16 [rows, ceil(width/16)]")

    pos_score, valid_score, invalid = _probe(score, nonzero_mask, lengths)
    table = _expand_page_table(page_table_size_1, cu_seqlens_q, rows)

    positive_count = (pos_score > 0.0).sum(dim=1)
    invalid_count = invalid.sum(dim=1).to(positive_count.dtype)

    # Fast path applies when the positives fit in topk and positives plus the
    # implicit-zero pool can fill it; otherwise the reference ranks every valid
    # position by score (its generic path). Both paths are evaluated lazily so
    # the common case pays for a single top-k.
    fast = (positive_count <= topk) & ((positive_count + invalid_count) >= topk)

    # --- fast rows: positives by score, then lowest-index masked-out positions
    out = torch.gather(
        table, 1, torch.topk(pos_score, topk, dim=-1, sorted=True).indices.long()
    ).to(torch.int32)
    if bool((~fast).any()):
        slow_rows = (~fast).nonzero(as_tuple=True)[0]
        slow_idx = torch.topk(
            valid_score[slow_rows], topk, dim=-1, sorted=True
        ).indices.long()
        out[slow_rows] = torch.gather(table[slow_rows], 1, slow_idx).to(torch.int32)

    if bool((fast & ((topk - positive_count) > 0)).any()):
        need = (topk - positive_count).clamp(min=0)
        rank = torch.cumsum(invalid.to(torch.int32), dim=1)
        take = fast.unsqueeze(1) & (invalid > 0) & (rank <= need.unsqueeze(1))
        src_row, src_col = take.nonzero(as_tuple=True)
        if src_row.numel():
            slot = positive_count[src_row] + rank[src_row, src_col] - 1
            out[src_row, slot.long()] = table[src_row, src_col]

    return out


__all__ = [
    "sparse_mask_topk_triton",
    "is_available",
    "MASK_BLOCK",
    "MASK_WARPS",
]
