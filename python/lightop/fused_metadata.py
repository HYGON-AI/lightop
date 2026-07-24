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
