"""LightOp MQA APIs and optional paged-MQA backend selection."""

import os

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
    "page_mqa_logits_sparse_mask",
]
