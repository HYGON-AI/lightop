# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4 packed KV dequantization and sparse Prefill metadata on HCU."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import torch

from . import op


def dsv41_pool_pairs_norm(previous, current, previous_score, score, weight, eps):
    """Pair softmax pooling, BF16 rounding and FP32 RMSNorm for CP Prefill.

    Pair inputs are finite FP32 [N,512] with contiguous columns; row-strided
    projection views are supported. Weight is contiguous FP32/BF16 [512].
    Returns a new BF16 pre-RoPE latent; no input or ring state is modified.
    """
    return op.dsv41_pool_pairs_norm(
        previous, current, previous_score, score, weight, eps
    )


def dsv41_rope_fake_quant_store(
    input: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """RoPE64 + per-16 E4M3/FP4 round-trip + V4 cache store in one launch.

    Input is contiguous BF16 [N,512]; freqs is complex64 [N,32] or
    interleaved FP32 [N,64]. Indices are valid contiguous int32/int64 slots.
    The V4 uint8 cache keeps FP8 nope and BF16 rope with per-64 UE8M0
    scales. Input is not modified. Negative indices suppress writes.
    """
    if freqs.is_complex():
        freqs = torch.view_as_real(freqs).flatten(-2)
    op.dsv41_rope_fake_quant_store(input, freqs, cache, indices, page_size)


def dequantize_k_cache_paged_hcu(
    cache: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
    page_size: int,
    fp8_fnuz: bool = False,
) -> None:
    """Gather valid token IDs from V4 packed KV into BF16 [N, 1, 512].

    The contiguous uint8 cache has 448 FP8 + 64 BF16 data per token,
    followed by 8 scale bytes per token at page_size * 576 in each page.
    IDs must be nonnegative and within the cache; no GPU-to-host validation.
    Output rows may be strided, but columns must be contiguous. V4.1 FP4
    and 512-dimensional FP8 cache layouts are not supported by this API.
    """
    op.dequantize_k_cache_paged_hcu(cache, token_ids, output, page_size, fp8_fnuz)


def combine_topk_swa_indices_hcu(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_pos: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    out_indices: torch.Tensor,
    out_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
) -> None:
    """Rebase selected TopK indices and append the causal positional SWA.

    All tensors are device INT32. Request starts must be nondecreasing,
    cover every output row and may include a cross-chunk offset. Query
    positions are absolute. Negative TopK entries remain -1. Only the
    scanned prefix is written; initialize unused output padding to -1.
    Out lengths count the prefix including holes, as in the Triton path.
    """
    op.combine_topk_swa_indices_hcu(
        topk_indices,
        query_start_loc,
        query_pos,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        out_indices,
        out_lens,
        window_size,
        compress_ratio,
        topk,
    )


@lru_cache(maxsize=1)
def _sparse_prefill_metadata_backend():
    if getattr(op, "SPARSE_PREFILL_METADATA_ABI", 0) == 1:
        return op

    # Support deployments that already installed the standalone validation build.
    name = f"{__package__}._sparse_prefill_metadata"
    try:
        backend = import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        backend = op
    if getattr(backend, "SPARSE_PREFILL_METADATA_ABI", 0) != 1:
        raise RuntimeError(
            "sparse Prefill metadata requires a rebuilt LightOp native extension"
        )
    return backend


def sparse_prefill_metadata_backend_info() -> dict:
    backend = _sparse_prefill_metadata_backend()
    return {
        "module": backend.__file__,
        "abi": backend.SPARSE_PREFILL_METADATA_ABI,
        "build_id": getattr(backend, "BUILD_ID", "monolithic"),
    }


def dsv4_build_cp_sparse_query_metadata(
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    query_positions: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical lengths, real lengths, and positions with CP padding at -1."""
    return _sparse_prefill_metadata_backend().dsv4_build_cp_sparse_query_metadata(
        extend_seq_lens, extend_start_loc, query_positions, cp_size, cp_rank
    )


def dsv4_pack_compact_sparse_token_ids(
    per_req_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    total_compressed: int,
    *,
    out_ids: torch.Tensor | None = None,
    out_base: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack physical IDs and compressed offsets using the caller's exact extent.

    The optional output buffers must not alias the inputs.
    """
    return _sparse_prefill_metadata_backend().dsv4_pack_compact_sparse_token_ids(
        per_req_ids,
        seq_lens,
        compress_ratio,
        page_size,
        total_compressed,
        out_ids,
        out_base,
    )


def dsv41_build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    window: int,
    total_swa: int,
    *,
    query_seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather SWA IDs and optionally write the physical query prefix.

    Returns token IDs, first positions, gather lengths, and offsets.
    """
    return _sparse_prefill_metadata_backend().dsv41_build_swa_token_ids(
        seq_lens,
        extend_seq_lens,
        req_pool_indices,
        req_to_token,
        full_to_swa,
        window,
        total_swa,
        query_seq_lens,
        query_start_loc,
    )


def dsv4_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    *,
    out_indices: torch.Tensor | None = None,
    out_lens: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine causal TopK and SWA indices, preserving caller-owned padding."""
    rows = topk_indices.shape[0]
    initialize_padding = out_indices is None
    if out_indices is None:
        width = ((topk + window_size + 127) // 128) * 128
        out_indices = torch.empty(
            (rows, width), dtype=torch.int32, device=topk_indices.device
        )
    if out_lens is None:
        out_lens = torch.empty(rows, dtype=torch.int32, device=topk_indices.device)
    _sparse_prefill_metadata_backend().combine_topk_swa_indices_hcu(
        topk_indices,
        query_start_loc,
        query_positions,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        out_indices,
        out_lens,
        window_size,
        compress_ratio,
        topk,
        initialize_padding,
    )
    return out_indices, out_lens
