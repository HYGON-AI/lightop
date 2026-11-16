# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Fused metadata kernel for paged attention inference (CUDA implementation).

This module implements the fused metadata preparation that combines:
  1. Prefix sum over sequence lengths → ``cache_seqlens_int32``, ``cu_seqlens_k``
  2. Page table gather from ``req_to_token`` → ``page_table``
  3. (optional) SWA page table gather via ``full_to_swa_mapping``

The Triton counterpart ``_fused_metadata_kernel_general`` serves as the
correctness baseline and performance reference.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import op


# ============================================================================
# Public API
# ============================================================================


def fused_metadata_kernel_general(
    # ---- inputs -----------------------------------------------------------
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    # ---- outputs ----------------------------------------------------------
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    swa_page_table: Optional[torch.Tensor] = None,
    full_to_swa_mapping: Optional[torch.Tensor] = None,
    # ---- scalars (inferred from tensors if omitted) -----------------------
    B: Optional[int] = None,
    max_seq_pages: Optional[int] = None,
    page_size: int = 1,
    seq_len_delta: int = 0,
    use_swa: bool = False,
) -> None:
    """
    Fused metadata preparation for paged attention.

    This single kernel replaces the common pattern of::

        cache_seqlens_int32 = seq_lens + seq_len_delta                     # element-wise
        cu_seqlens_k        = seq_lens.cumsum(0)                          # exclusive prefix-sum
        page_table          = req_to_token[req_pool_indices] // page_size  # gather + div

    **Shapes**  (``seq_lens`` and ``req_pool_indices`` accept ``int32`` or
    ``int64``; all other tensors are ``int32``, on the same CUDA device)::

        seq_lens            : [B]
        req_to_token        : [num_reqs, max_tokens]
        req_pool_indices    : [B]
        cache_seqlens_int32 : [B]                    (output, overwritten)
        cu_seqlens_k        : [B + 1]                (output, overwritten)
        page_table          : [B, max_seq_pages]     (output, overwritten)
        swa_page_table      : [B, max_seq_pages]     (optional output)
        full_to_swa_mapping : [num_tokens]           (optional input)

    Args:
        seq_lens:          Sequence lengths for each batch element.  [B] int32
        req_to_token:      Logical→physical page mapping.          [num_reqs, max_tokens] int32
        req_pool_indices:  For each batch element, which row of
                           ``req_to_token`` to use.                 [B] int32
        cache_seqlens_int32: Output — ``seq_len + seq_len_delta``. [B] int32
        cu_seqlens_k:      Output — exclusive prefix-sum of
                           ``cache_seqlens_int32``.                 [B + 1] int32
        page_table:        Output — physical page ids per batch
                           row.                                    [B, max_seq_pages] int32
        swa_page_table:    Optional output — SWA page ids.         [B, max_seq_pages] int32
        full_to_swa_mapping: Optional input — maps logical page → SWA slot.
                             Supports int32 and int64.
        B:                 Number of sequences. If ``None`` inferred from
                           ``seq_lens.size(0)``.
        max_seq_pages:     Number of pages per sequence. If ``None``
                           inferred from ``page_table.size(1)``.
        page_size:         Number of logical tokens per physical page.
                           Must be a power of two.
        seq_len_delta:     Added to each ``seq_lens`` element before
                           storing to ``cache_seqlens_int32``.
        use_swa:           If ``True``, build ``swa_page_table``.
                           Requires ``swa_page_table`` and
                           ``full_to_swa_mapping``.
    """
    if B is None:
        B = seq_lens.size(0)
    if max_seq_pages is None:
        max_seq_pages = page_table.size(1)

    # Ensure contiguous — required for coalesced memory access
    seq_lens = seq_lens.contiguous()
    req_to_token = req_to_token.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    cache_seqlens_int32 = cache_seqlens_int32.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    page_table = page_table.contiguous()
    if swa_page_table is not None:
        swa_page_table = swa_page_table.contiguous()
    if full_to_swa_mapping is not None:
        full_to_swa_mapping = full_to_swa_mapping.contiguous()

    op.fused_metadata_kernel_general(
        seq_lens,
        req_to_token,
        req_pool_indices,
        cache_seqlens_int32,
        cu_seqlens_k,
        page_table,
        swa_page_table,
        full_to_swa_mapping,
        B,
        max_seq_pages,
        page_size,
        seq_len_delta,
        use_swa,
    )


def dsa_cp_prefill_metadata_plan(seq_lens_cpu, extend_lens_cpu, cp_size, cp_rank,
                                 topk, padded_tokens, page_size=64, has_prefix=False):
    seq = [int(x) for x in seq_lens_cpu]
    extend = [int(x) for x in extend_lens_cpu]
    if not (len(seq) == len(extend) <= 64 and 1 < cp_size <= 64 and 0 <= cp_rank < cp_size):
        raise ValueError("Unsupported DSA CP batch or rank")
    if not (0 < topk <= 2048 and 0 < page_size <= 1024 and page_size & (page_size - 1) == 0):
        raise ValueError("Unsupported DSA CP TopK or page size")
    selected, counts, begin = [], [], 0
    for i, (length, q) in enumerate(zip(seq, extend)):
        if not 0 <= q <= length <= 2147483647:
            raise ValueError("Invalid DSA CP lengths")
        first = (cp_rank - begin % cp_size) % cp_size
        count = max(0, (q - first + cp_size - 1) // cp_size)
        if count:
            selected.append(i)
            counts.append(count)
        begin += q
    tokens = sum(counts)
    padded = max(tokens, int(padded_tokens))
    total_k = sum(seq[i] for i in selected)
    if begin > 2147483647 or total_k > 2147483647 or not 0 <= padded <= 1048576 or padded * topk > 2147483647:
        raise ValueError("DSA CP metadata exceeds supported integer bounds")
    return dict(selected=selected, counts=counts, tokens=tokens, padded=padded,
                width=max((seq[i] for i in selected), default=0), total_k=total_k,
                max_q=max(counts, default=1), cp_size=cp_size, cp_rank=cp_rank,
                topk=topk, page_size=page_size, has_prefix=bool(has_prefix), batch=len(seq))


def fused_dsa_cp_prefill_metadata(seq_lens, req_to_token, req_pool_indices,
                                  extend_lens, plan):
    b, t, p, w = len(plan["selected"]), plan["tokens"], plan["padded"], plan["width"]
    if (not seq_lens.is_cuda or seq_lens.ndim != 1 or seq_lens.numel() != plan["batch"]
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or req_to_token.dtype != torch.int32 or req_to_token.ndim != 2
        or req_to_token.shape[1] < w or req_to_token.stride(1) != 1
        or req_pool_indices.dtype not in (torch.int32, torch.int64)
        or req_pool_indices.ndim != 1 or req_pool_indices.numel() != plan["batch"]
        or extend_lens.dtype != torch.int32 or extend_lens.ndim != 1
        or extend_lens.numel() != plan["batch"]
        or any(x.device != seq_lens.device or not x.is_contiguous()
               for x in (seq_lens, req_pool_indices, extend_lens))
        or req_to_token.device != seq_lens.device):
        raise ValueError("Unsupported DSA CP tensor layout")
    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=seq_lens.device)
    pages = empty((b, w))
    real = pages if plan["page_size"] == 1 else empty((b, (w + plan["page_size"] - 1) // plan["page_size"]))
    out = [empty(b, seq_lens.dtype), empty(b), empty(b+1), empty(b+1), empty(t),
           empty(p), empty(p+1), empty(p+1), pages, real, empty(t), empty(t),
           empty(t), empty(p), empty(plan["total_k"] if plan["has_prefix"] else 0), empty(b*8+1)]
    op.fused_dsa_cp_prefill_metadata(seq_lens, req_to_token, req_pool_indices,
                                   extend_lens, out, plan["cp_size"], plan["cp_rank"],
                                   plan["topk"], plan["page_size"])
    return dict(indexer_seq_lens=out[0], cache_seqlens_int32=out[1], cu_seqlens_k=out[2],
                cu_seqlens_q=out[3], dsa_seqlens_expanded=out[4],
                dsa_cache_seqlens_int32=out[5], dsa_cu_seqlens_k=out[6],
                dsa_cu_seqlens_q=out[7], page_table_1=out[8], real_page_table=out[9],
                indexer_k_start_end=(out[10],out[11]), token_to_batch_idx=out[12],
                topk_indices_offset=out[13], page_table_1_flattened=out[14] if plan["has_prefix"] else None)
