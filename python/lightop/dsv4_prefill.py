# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4 packed KV dequantization and sparse-prefill index combination on HCU."""

import torch

from . import op


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
        topk_indices, query_start_loc, query_pos, seq_lens, gather_lens,
        compressed_base, swa_base, out_indices, out_lens,
        window_size, compress_ratio, topk,
    )
