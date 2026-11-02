# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LightOp MQA APIs and optional paged-MQA backend selection."""

import os
from functools import lru_cache

import torch

from ..gemmopt import (
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    mqa_logits,
    paged_mqa_logits as _paged_mqa_logits_nonpersistent,
    page_mqa_logits_sparse_mask,
)


def get_paged_mqa_logits_fp4_metadata(
    context_lens, block_kv, num_sms, rows_per_request=1
):
    """Plan the FP4 paged-MQA schedule, including paired route lengths."""
    from .. import op

    if not hasattr(op, "get_paged_mqa_logits_fp4_metadata"):
        raise RuntimeError(
            "get_paged_mqa_logits_fp4_metadata unavailable: rebuild/install "
            "LightOp with the FP4 schedule planner"
        )
    return op.get_paged_mqa_logits_fp4_metadata(
        context_lens, block_kv, num_sms, rows_per_request
    )


def get_paged_sparse_mqa_logits_metadata(
    context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
    qk_dtype, sparse_block_kv, num_sms=None,
):
    """DeepGEMM `get_paged_sparse_mqa_logits_metadata` on HCU.

    Plans `fp8_fp4_paged_sparse_mqa_logits`: per request-aligned pair of query
    tokens, the first min(M, ceil(context_len / sparse_block_kv)) entries of
    each row of sparse_kv_block_indices int32 [Q,M] (unique, ascending logical
    blocks, M % 4 == 0, M <= 4096) are merged, mapped through block_table int32
    [Q,pages] at page_kv and cut into KV splits of 640 (packed FP4 qk_dtype,
    int8/uint8) or 512 (FP8 E4M3) positions, followed by a persistent schedule
    of num_sms (default: the CU count) entries per wave. indices int32 [Q] is
    the request of each token; a request's tokens are consecutive and paired
    tokens share their block_table row. The uint8 result has DeepGEMM's byte
    layout. Call once per stream before CUDA graph capture.
    """
    from .. import op

    if not hasattr(op, "get_paged_sparse_mqa_logits_metadata"):
        raise RuntimeError(
            "get_paged_sparse_mqa_logits_metadata unavailable: rebuild/install "
            "LightOp with the paged sparse MQA metadata planner"
        )
    return op.get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
        qk_dtype, sparse_block_kv, num_sms,
    )


def paged_mqa_logits_persisten(
    q, fused_kv_cache, weights, context_lens, block_table,
    max_context_len, *, group_size=1, persistent_ctas=0,
):
    """Dense paged MQA for group sizes 1 through 5.

    Equal physical KV pages in one group are loaded once. Different pages
    are handled independently, and a final partial group is valid. INT8 is
    supported on gfx936/gfx938; FP8 requires gfx938.
    """
    from .. import op

    if not hasattr(op, "paged_mqa_logits_persisten"):
        raise RuntimeError(
            "paged_mqa_logits_persisten requires a rebuilt LightOp extension"
        )
    return op.paged_mqa_logits_persisten(
        q, fused_kv_cache, weights, context_lens, block_table, None,
        max_context_len, True, 4, persistent_ctas, group_size,
    )


def _persistent_supported(q, fused_kv_cache, weights, context_lens,
                          block_table, schedule_meta, max_context_len,
                          clean_logits, is_prefill):
    if (schedule_meta is not None or not clean_logits or is_prefill
            or q.ndim != 4 or q.shape[0] == 0 or q.shape[1] != 1
            or q.shape[2] not in (8, 16, 32, 64) or q.shape[3] != 128
            or not q.is_contiguous() or weights.dtype != torch.float32
            or context_lens.dtype != torch.int32
            or block_table.dtype != torch.int32
            or block_table.ndim != 2 or block_table.shape[0] != q.shape[0]
            or max_context_len <= 0
            or max_context_len > block_table.shape[1] * 64):
        return False
    props = torch.cuda.get_device_properties(q.device)
    arch = props.gcnArchName.split(":")[0]
    if arch == "gfx936":
        return (props.multi_processor_count == 80
                and q.dtype == torch.int8
                and fused_kv_cache.dtype == torch.int8)
    if arch == "gfx938" and props.multi_processor_count in (64, 72):
        return ((q.dtype == torch.int8
                 and fused_kv_cache.dtype == torch.int8)
                or (q.dtype == torch.float8_e4m3fn
                    and fused_kv_cache.dtype == torch.uint8))
    return False


def paged_mqa_logits(
    q, fused_kv_cache, weights, context_lens, block_table,
    schedule_meta, max_context_len, clean_logits=True, is_prefill=False,
):
    """Select paged-MQA backend in LightOp, without changing SGLang.

    LIGHTOP_PAGED_MQA_LOGITS_BACKEND=0 selects the original kernel (default);
    1 selects the persistent kernel for supported inputs. Set
    LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE to 1..5 (default 1). Unsupported
    inputs use the original kernel.
    """
    backend = os.getenv("LIGHTOP_PAGED_MQA_LOGITS_BACKEND", "0")
    if backend not in ("0", "1"):
        raise ValueError("LIGHTOP_PAGED_MQA_LOGITS_BACKEND must be 0 or 1")
    if backend == "1" and _persistent_supported(
        q, fused_kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, clean_logits, is_prefill,
    ):
        group_size = int(os.getenv("LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE", "1"))
        if not 1 <= group_size <= 5:
            raise ValueError("LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE must be 1..5")
        return paged_mqa_logits_persisten(
            q, fused_kv_cache, weights, context_lens, block_table,
            max_context_len, group_size=group_size,
        )
    return _paged_mqa_logits_nonpersistent(
        q, fused_kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, clean_logits, is_prefill,
    )


def paged_mqa_logits_fp4(q, q_sf, kv_cache, weights, context_lens, block_table,
                         schedule_meta=None, max_context_len=None,
                         table_block_size=64, out=None, rows_per_request=1):
    """DeepGEMM `fp8_fp4_paged_mqa_logits` with clean_logits false, on HCU.

    q is packed E2M1 (int8/uint8) plus q_sf int32 UE8M0, laid out per head as
    [B,1,H,64] and [B,1,H]. kv_cache is the fused FP4 index-K pool
    [pages,64,1,68] uint8, weights fp32 [B,H], context_lens int32 [B]. Column j
    of row b scores pool slot block_table[b, j // table_block_size] *
    table_block_size + j % table_block_size, for a block table already at
    table_block_size (64 dense, 8 when the table holds candidate blocks).
    Columns at or past context_lens[b] are left unwritten. Returns FP32
    [B,max_context_len] with a row stride rounded up to the split size, so
    callers must respect stride(0). schedule_meta comes from
    get_paged_mqa_logits_metadata(context_lens, 64, CUs) and may be None.

    rows_per_request n > 1 (table_block_size 64, H <= 32 only) declares every
    n rows one request's draft tokens sharing the block table row of the
    first; with at least PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS requests the
    rows are then scored in pairs, one KV read per pair, and fewer run as
    n = 1. Results are identical either way. A schedule must be planned on
    paged_mqa_logits_route_lens(context_lens, n) instead of context_lens.
    """
    from .. import op
    if not hasattr(op, "paged_mqa_logits_fp4"):
        raise RuntimeError(
            "paged_mqa_logits_fp4 unavailable: the installed LightOp native "
            "extension does not provide this operator; rebuild/install a version "
            "with FP4 paged MQA support"
        )
    if max_context_len is None:
        max_context_len = int(block_table.shape[1]) * table_block_size
    return op.paged_mqa_logits_fp4(
        q, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta,
        max_context_len, table_block_size, out, rows_per_request,
    )


def paged_mqa_logits_route_lens(context_lens, rows_per_request):
    """int32 lengths of the routes paged_mqa_logits_fp4 scores with
    rows_per_request n, the schedule's input: when it pairs rows (enough
    requests), per request the max of rows (2 j, 2 j + 1), an odd n's last row
    alone, [B / n * ceil(n / 2)]; otherwise context_lens itself, [B]."""
    from .. import op

    lens = context_lens.reshape(-1)
    if (rows_per_request > 1 and lens.numel() // rows_per_request
            >= op.PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS):
        lens = lens.reshape(-1, rows_per_request)
        if rows_per_request % 2:
            lens = torch.nn.functional.pad(lens, (0, 1))
        lens = lens.unflatten(1, (-1, 2)).amax(-1).flatten()
    return lens.to(torch.int32).contiguous()


def dsv41_candidate_block_table(logits, seq_lens, page_table, page_size,
                                block_size, out_phys_blocks, out_valid_lens):
    """Level one of the DSV4.1 two-level indexer: keeps the topk best blocks of
    `block_size` (must be 8) positions by block max, always including the block
    holding the row's newest position.

    logits is the FP32 [B,W] dense logits, seq_lens int32 [B] the live width of
    each row, page_table int32 [B,pages] at page_size. Writes out_phys_blocks
    [B,topk] int32, the kept blocks ascending as pool slots / block_size with
    zeros past min(n, topk), and out_valid_lens [B] int32, the sparse row
    length 8 (kept - 1) + (len - 1) % 8 + 1.
    """
    from .. import op
    if not hasattr(op, "dsv41_candidate_block_table"):
        raise RuntimeError(
            "dsv41_candidate_block_table unavailable: rebuild or upgrade the "
            "LightOp native extension"
        )
    return op.dsv41_candidate_block_table(
        logits, seq_lens, page_table, page_size, block_size, out_phys_blocks,
        out_valid_lens,
    )


def dsv41_select_candidate_blocks(logits, seq_lens, topk_blocks, block_size=8):
    """Select logical blocks and expand their mask in one HCU kernel.

    Returns bool [B,W] and ascending int32 [B,topk_blocks] with -1 padding.
    Selected blocks expand all eight positions, clipped at W, not seq_lens.
    Inputs have the same FP32 aligned-row and int32 length contract as the
    candidate block-table operator; no physical page table is needed.
    """
    from .. import op
    if not hasattr(op, "dsv41_select_candidate_blocks"):
        raise RuntimeError(
            "dsv41_select_candidate_blocks unavailable: rebuild or upgrade "
            "the LightOp native extension"
        )
    out_mask = torch.empty(logits.shape, dtype=torch.bool, device=logits.device)
    out_block_ids = torch.empty(
        (logits.shape[0], topk_blocks), dtype=torch.int32, device=logits.device
    )
    op.dsv41_select_candidate_blocks(
        logits, seq_lens, block_size, out_mask, out_block_ids
    )
    return out_mask, out_block_ids


def dsv41_ragged_topk_postprocess(scores, selected, offsets, slots, out_pages,
                                  out_raw=None, mask_scores=False):
    """Filter/sort ragged selections and write physical and optional raw IDs.

    Selection itself is unchanged. Negative IDs and INT_MAX become padding;
    consumer masking keeps scores strictly greater than -inf (NaNs removed,
    +inf retained). Output matrices are int32 and may be strided views.
    Scores are float32, selected/offsets int32, slots int32 or int64. Top-k
    is at most 2048. Inputs are read-only; writes are confined to output views.
    """
    from .. import op

    op.dsv41_ragged_topk_postprocess(
        scores, selected, offsets, slots, out_pages, out_raw, mask_scores,
    )


def dsv41_remap_indices_to_staging(indices, selected_pages, page_size, max_pages,
                                   static_selected_pages=False):
    """Map token IDs to compact staging slots, preserving all negative IDs.

    Selected pages may be unordered; valid destination pages must be unique.
    With static_selected_pages, negative entries are padding. Otherwise normal
    PyTorch negative page-index wrapping applies. Valid token pages must be
    below max_pages. Missing pages map to -page_size + the token offset.
    """
    from .. import op

    return op.dsv41_remap_indices_to_staging(
        indices, selected_pages, page_size, max_pages, static_selected_pages,
    )


def dsv41_build_swa_token_ids(seq_lens, extend_seq_lens, req_pool_indices,
                              req_to_token, full_to_swa, swa_window, total_swa):
    """Build SWA IDs and request metadata without host reads.

    Lengths/request IDs/table are device int32; full_to_swa is int64.
    total_swa must equal sum(min(seq, extend + swa_window - 1)), supplied
    from CPU metadata. Table/mapping indices must be valid, as in SGLang.
    """
    from .. import op

    return op.dsv41_build_swa_token_ids(
        seq_lens, extend_seq_lens, req_pool_indices, req_to_token, full_to_swa,
        swa_window, total_swa,
    )


def dsv41_pair_partners(kv, score, req, pos, pad, state, ring_size):
    """Select FP32 compression partners, then update the packed request ring.

    Returns partner KV and score [T,D]. State [S,2D] is updated in place;
    its final row is the padding sentinel. KV/score may be column-slice views.
    Retained writes must have unique live slots, as in SGLang's request batches.
    Two ordered HCU launches preserve all old-state reads before any write.
    """
    from .. import op
    if not hasattr(op, "dsv41_pair_partners"):
        raise RuntimeError("dsv41_pair_partners requires a rebuilt LightOp extension")
    return op.dsv41_pair_partners(kv, score, req, pos, pad, state, ring_size)


__all__ = [
    "fp8_fp4_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "get_paged_mqa_logits_fp4_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "mqa_logits",
    "paged_mqa_logits",
    "paged_mqa_logits_persisten",
    "paged_mqa_logits_fp4",
    "paged_mqa_logits_route_lens",
    "dsv41_candidate_block_table",
    "dsv41_select_candidate_blocks",
    "dsv41_pair_partners",
    "dsv41_build_swa_token_ids",
    "dsv41_remap_indices_to_staging",
    "dsv41_ragged_topk_postprocess",
    "page_mqa_logits_sparse_mask",
]


# Sparse-only native, source-pinned and validated for this package.
@lru_cache(maxsize=1)
def _prefill_current_sparse_backend():
    from importlib import import_module
    module_name = __package__.split(".attention", 1)[0] + "._paged_sparse_mqa"
    try:
        backend = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        # A regular full-wheel build exports the same implementation in op.
        from .. import op as backend
    if backend.PAGED_SPARSE_MQA_LOGITS_ABI != 3:
        raise RuntimeError("Sparse MQA ABI mismatch")
    return backend


_paged_sparse_mqa_backend = _prefill_current_sparse_backend

def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv, num_sms=None):
    """DeepGEMM `get_paged_sparse_mqa_logits_metadata` on HCU.

    Plans `fp8_fp4_paged_sparse_mqa_logits`: per request-aligned pair of query
    tokens, the first min(M, ceil(context_len / sparse_block_kv)) entries of
    each row of sparse_kv_block_indices int32 [Q,M] (unique, ascending logical
    blocks, M % 4 == 0, M <= 4096) are merged, mapped through block_table int32
    [Q,pages] at page_kv and cut into KV splits of 640 (packed FP4 qk_dtype,
    int8/uint8) or 512 (FP8 E4M3) positions, followed by a persistent schedule
    of num_sms (default: the CU count) entries per wave. indices int32 [Q] is
    the request of each token; a request's tokens are consecutive and paired
    tokens share their block_table row. The uint8 result has DeepGEMM's byte
    layout. Call once per stream before CUDA graph capture.
    """
    op = _prefill_current_sparse_backend()
    if not hasattr(op, 'get_paged_sparse_mqa_logits_metadata'):
        raise RuntimeError('get_paged_sparse_mqa_logits_metadata unavailable: rebuild/install LightOp with the paged sparse MQA metadata planner')
    return op.get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv, num_sms)

def sparse_fp8_fp4_paged_mqa_logits(q, fused_kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv, *, context_lens=None, sparse_kv_block_indices=None, output=None, num_sms=None, is_prefill=False):
    """Consume official paged sparse MQA metadata and return FP32 logits.

    q=(payload, packed UE8M0 scales), supporting MXFP4 or MXFP8, D=128 and
    H=4..32 in steps of four. weights is FP32 [R,H]. The result is FP32
    [R,num_max_sparse_blocks*sparse_block_kv] with a padded row stride.
    is_prefill=True selects the rewritten Prefill kernel; False retains decode.
    Metadata records request-aligned query pairs, their KV candidate unions
    and each query's sparse output slots. KV is read directly through those
    physical blocks; no dense logits temporary is formed.

    context_lens and sparse_kv_block_indices may be supplied together to mask
    individual tokens in a partial logical tail block. Without them the
    consumer follows the official metadata's whole-block visibility. They
    do not rebuild metadata or affect its scheduling. Unselected/padded
    output slots are -inf in both modes. num_sms must match the planner.
    """
    if not isinstance(q, (tuple, list)) or len(q) != 2:
        raise TypeError('q must be (payload, packed UE8M0 scales)')
    backend = _prefill_current_sparse_backend()
    if getattr(backend, 'PAGED_SPARSE_MQA_LOGITS_ABI', 0) != 3:
        raise RuntimeError('paged sparse MQA requires a rebuilt LightOp extension')
    return backend.fp8_fp4_paged_sparse_mqa_logits(q[0], q[1], fused_kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv, context_lens, sparse_kv_block_indices, output, num_sms, is_prefill)
fp8_fp4_paged_sparse_mqa_logits = sparse_fp8_fp4_paged_mqa_logits

if "__all__" in globals():
    __all__ = list(dict.fromkeys([*__all__, "get_paged_sparse_mqa_logits_metadata", "sparse_fp8_fp4_paged_mqa_logits", "fp8_fp4_paged_sparse_mqa_logits"]))

# Native sparse Prefill preparation; GPU computation lives in sparse_page_mqa/*.cu.
def _cdiv(a, b):
    return (a + b - 1) // b


@lru_cache(maxsize=1)
def _native_prefill_fusion_backend():
    from importlib import import_module
    name = __package__.split(".attention", 1)[0] + "._prefill_sparse_fusion"
    try:
        backend = import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        from .. import op as backend
    if getattr(backend, "PREFILL_SPARSE_FUSION_ABI", 0) != 1:
        raise RuntimeError("native sparse Prefill fusion requires a rebuilt LightOp extension")
    return backend


def _index_sort_backend():
    return _native_prefill_fusion_backend()


def _index_sort_options(tensor, kind):
    return (_native_prefill_fusion_backend(), 1, 512) if supports_prefill_sparse_fusion(tensor) else None


def supports_prefill_sparse_fusion(tensor):
    if not tensor.is_cuda:
        return False
    arch = torch.cuda.get_device_properties(tensor.device).gcnArchName.split(":")[0]
    return arch in ("gfx92a", "gfx936", "gfx938")



def _prefill_sparse_query_lanes(q):
    """Choose a query layout by row intervals, without device-value reads."""
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
    rows = q.shape[0]
    if arch == "gfx936" and q.shape[1] == 32:
        if 768 < rows <= 1536 or rows > 2560:
            return 4
    return 8


def prepare_prefill_sparse_query(q, raw_weights, freqs_cis, positions, rope_dim,
                                weight_scale, compress_ratio, width, *, query_tile=None, num_warps=4):
    """Native RoPE/FP4 packing, BF16 weight scaling and per-row metadata."""
    if not supports_prefill_sparse_fusion(q):
        raise ValueError("native sparse fusion requires gfx92a, gfx936 or gfx938")
    if query_tile is not None and query_tile not in (1, 2, 4, 8, 16, 32):
        raise ValueError("query_tile must be a power of two <=32")
    return tuple(_native_prefill_fusion_backend().prefill_sparse_native_prepare_query(
        q, raw_weights, freqs_cis, positions, rope_dim, weight_scale,
        compress_ratio, width, True,
        None, None, None, 0, 64, None, _prefill_sparse_query_lanes(q)))


def _prepare_prefill_sparse_request(q, raw_weights, freqs_cis, positions, rope_dim,
                                   weight_scale, compress_ratio, width, source,
                                   req_to_token, request_id, context_len, page_size, row_flags=None):
    """Native combined Q/weights/row/candidate/page preparation for H=32."""
    return tuple(_native_prefill_fusion_backend().prefill_sparse_native_prepare_query(
        q, raw_weights, freqs_cis, positions, rope_dim, weight_scale,
        compress_ratio, width, True, source,
        req_to_token, request_id, context_len, page_size, row_flags,
        _prefill_sparse_query_lanes(q)))


def prepare_prefill_sparse_candidates(source, width, req_to_token, request_id,
                                      context_len, compress_ratio, page_size, *, num_warps=4, row_flags=None):
    """Native sanitize/sort/pad and request page-table preparation."""
    return tuple(_native_prefill_fusion_backend().prefill_sparse_native_prepare_candidates(
        source, width, req_to_token, request_id, context_len, compress_ratio, page_size, row_flags))


def finalize_prefill_sparse_indices(compact, blocks, lengths, valid_lens, *,
                                    out_logical=None, req_to_token=None, request_id=None,
                                    compress_ratio=1, out_page_indices=None, num_warps=None,
                                    candidate_flags=None):
    """Native exact integer ordering/mapping after the unchanged TopK operator."""
    if out_logical is None:
        out_logical = torch.empty_like(compact)
    _native_prefill_fusion_backend().prefill_sparse_native_finalize(
        compact, blocks, lengths, valid_lens, out_logical, out_page_indices,
        req_to_token if out_page_indices is not None else None,
        request_id if out_page_indices is not None else None, compress_ratio, candidate_flags)
    return out_logical


def prefill_sparse_mqa_fused(
    q,
    raw_weights,
    freqs_cis,
    positions,
    *,
    weight_scale,
    rope_dim,
    k_cache,
    req_to_token,
    request_pool_indices,
    context_lens_per_request,
    rows_per_request,
    request_block_ids,
    compress_ratio,
    page_size,
    num_max_sparse_blocks,
    index_topk,
    out_page_indices,
    out_raw_indices=None,
    budget_bytes=1 << 30,
    combine_preparation=True,
):
    """MXFP4 sparse consumer after the two unchanged projection GEMMs.

    Geometry and metadata are refreshed on each call, including capture/replay;
    no cache of data values survives a forward. K is read in its original
    paged format by the existing gfx936 BF16/gfx938 guarded FP8/BF16 scorer.
    As required by native metadata, warm up once on the capture stream before
    its first Graph capture. Request IDs/positions must reference valid input
    rows/frequencies; this API does not copy device values to the host.
    """
    from ..fuse_topk_transform import topk_transform_ragged_v2

    rows, heads, _ = q.shape
    width = num_max_sparse_blocks
    if (
        len(rows_per_request) != len(context_lens_per_request)
        or len(request_block_ids) != len(rows_per_request)
        or sum(rows_per_request) != rows
    ):
        raise ValueError("ragged requests do not match projected queries")
    if budget_bytes <= 0 or index_topk not in (512, 1024, 2048):
        raise ValueError("positive scratch budget and supported TopK required")
    if (
        page_size <= 0
        or page_size % 8
        or any(
            count < 0 or length < 0
            for count, length in zip(rows_per_request, context_lens_per_request)
        )
    ):
        raise ValueError(
            "nonnegative request sizes and a page size divisible by eight required"
        )
    if (
        request_pool_indices.numel() != len(rows_per_request)
        or request_pool_indices.dtype not in (torch.int32, torch.int64)
        or not request_pool_indices.is_contiguous()
    ):
        raise ValueError("one contiguous integer request ID per request is required")
    if (
        req_to_token.ndim != 2
        or req_to_token.stride(1) != 1
        or req_to_token.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("request map must have contiguous integer columns")
    for output in [out_page_indices] + (
        [] if out_raw_indices is None else [out_raw_indices]
    ):
        if (
            output.ndim != 2
            or output.shape[0] < rows
            or output.shape[1] < index_topk
            or output.dtype != torch.int32
            or output.stride(1) != 1
        ):
            raise ValueError(
                "outputs must be int32 matrices with sufficient rows/columns"
            )
    tensors = [
        raw_weights,
        freqs_cis,
        positions,
        k_cache,
        req_to_token,
        request_pool_indices,
        out_page_indices,
    ]
    if out_raw_indices is not None:
        tensors.append(out_raw_indices)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all sparse fusion inputs/outputs must share one device")
    for count, length, source in zip(
        rows_per_request, context_lens_per_request, request_block_ids
    ):
        if source is not None and (
            source.shape[0] != count or source.ndim != 2 or source.dtype != torch.int32
        ):
            raise ValueError("published candidate rows must match requests")
        if count and length and source is None:
            raise ValueError("published block IDs required")
    combined = combine_preparation and heads == 32
    if combined:
        if (
            q.dtype != torch.bfloat16
            or raw_weights.dtype != torch.bfloat16
            or raw_weights.shape != (rows, heads)
        ):
            raise ValueError(
                "query and raw weights must preserve BF16 projection outputs"
            )
        if (
            positions.numel() != rows
            or rope_dim < 0
            or rope_dim > 128
            or rope_dim % 2
            or compress_ratio <= 0
            or width <= 0
            or width % 4
            or width > 4096
        ):
            raise ValueError("invalid fused query geometry")
        if not freqs_cis.is_complex() or not supports_prefill_sparse_fusion(q):
            raise ValueError("complex frequencies and gfx936/gfx938 required")
    else:
        query_data = prepare_prefill_sparse_query(
            q,
            raw_weights,
            freqs_cis,
            positions,
            rope_dim,
            weight_scale,
            compress_ratio,
            width,
        )
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    # Match the original conservative chunk budget; page tables remain broadcast
    # views because native metadata accepts stride0==0 (no R-by-pages copy).
    offset = 0
    for request, (count, length, source) in enumerate(
        zip(rows_per_request, context_lens_per_request, request_block_ids)
    ):
        end = offset + count
        if count == 0 or length == 0:
            out_page_indices[offset:end, :index_topk].fill_(-1)
            if out_raw_indices is not None:
                out_raw_indices[offset:end, :index_topk].fill_(-1)
            offset = end
            continue
        request_id = request_pool_indices[request : request + 1]
        sort_options = _index_sort_options(q, "candidate")
        row_flags = (
            torch.empty((count,), device=q.device, dtype=torch.int32)
            if sort_options is not None
            else None
        )
        if source.stride(1) != 1 or source.device != q.device:
            raise ValueError(
                "published candidates must have contiguous columns on the query device"
            )
        if combined:
            payload, sf, weights, lengths, valid_lens, pairs, zeros, blocks, pages = (
                _prepare_prefill_sparse_request(
                    q[offset:end],
                    raw_weights[offset:end],
                    freqs_cis,
                    positions[offset:end],
                    rope_dim,
                    weight_scale,
                    compress_ratio,
                    width,
                    source,
                    req_to_token,
                    request_id,
                    length,
                    page_size,
                    row_flags=row_flags,
                )
            )
        else:
            payload, sf, weights, lengths, valid_lens, pairs, zeros = [
                v[offset:end] for v in query_data
            ]
            pairs = query_data[5]
            blocks, pages = prepare_prefill_sparse_candidates(
                source,
                width,
                req_to_token,
                request_id,
                length,
                compress_ratio,
                page_size,
                row_flags=row_flags,
            )
        splits = _cdiv(width, 80)
        row_bytes = (
            width * (8 * 4 + 4)
            + pages.numel() * 4
            + splits * (16 + 80 * 8 + 16)
            + heads * (64 + 4 + 4)
            + index_topk * 16
        )
        step = max(1, budget_bytes // max(1, row_bytes))
        for begin in range(0, count, step):
            stop = min(begin + step, count)
            start = offset + begin
            finish = offset + stop
            block_chunk = blocks[begin:stop]
            logits = _prefill_current_sparse_backend().prefill_sparse_fp8_fp4_paged_mqa_logits(
                payload[begin:stop], sf[begin:stop], k_cache, weights[begin:stop],
                lengths[begin:stop], pages[None, :].expand(stop - begin, -1),
                pairs[: stop - begin], block_chunk, width, 8, num_sms, True, 0,
            )
            compact = torch.empty(
                (stop - begin, index_topk), device=q.device, dtype=torch.int32
            )
            topk_transform_ragged_v2(
                logits,
                valid_lens[begin:stop],
                out_offsets=zeros[begin:stop],
                out_indices=compact,
            )
            finalize_prefill_sparse_indices(
                compact,
                block_chunk,
                lengths[begin:stop],
                valid_lens[begin:stop],
                out_logical=None
                if out_raw_indices is None
                else out_raw_indices[start:finish, :index_topk],
                req_to_token=req_to_token,
                request_id=request_id,
                compress_ratio=compress_ratio,
                out_page_indices=out_page_indices[start:finish, :index_topk],
                candidate_flags=None if row_flags is None else row_flags[begin:stop],
            )
            del compact
            del logits
        offset = end


__all__ = list(dict.fromkeys([*__all__, "supports_prefill_sparse_fusion", "prepare_prefill_sparse_query", "prepare_prefill_sparse_candidates", "finalize_prefill_sparse_indices", "prefill_sparse_mqa_fused"]))
