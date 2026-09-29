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


def paged_mqa_logits_fp4(q, weights, slots, lens, table, page_size, out=None):
    """Score BF16 queries against the FP4 indexer pool on gfx928/936/938.

    q [B,H,128] and weights [B,H] are BF16; slots [B,L] and lens [B]
    are contiguous int64. table is uint8 [pages,page_size*68], storing all
    E2M1 payload rows followed by all per-32 E8M0 scale rows in each page.
    The caller guarantees valid pool slots for j < min(lens[b], L).
    Returns FP32 [B,L] with -inf for invisible positions. Numerics follow
    DeepGEMM's MXFP4 MQA logits: each per-32 block scale is folded into the
    K operand ahead of one FP32 accumulator over all 128 channels, and relu,
    the head weight and the head reduction stay in FP32.
    An optional contiguous FP32 out supports allocation-free graph replay.
    Inputs must share a device. Unsupported shapes/layouts raise an error.
    gfx938 H32 workloads with at least 48K scores use lossless FP8 MMAC;
    other workloads use BF16 MMAC. Q values that cannot be encoded exactly
    retain BF16 computation. Query conversion runs on device on every replay.
    KV block scale exponents spanning more than 11 E8M0 steps within a token,
    or a common FP32 factor that is too extreme to represent normally, also
    select the BF16 path.
    """
    from .. import op
    if not hasattr(op, "paged_mqa_logits_fp4"):
        raise RuntimeError(
            "paged_mqa_logits_fp4 unavailable: the installed LightOp native "
            "extension does not provide this operator; rebuild/install a version "
            "with FP4 paged MQA support"
        )
    return op.paged_mqa_logits_fp4(q, weights, slots, lens, table, page_size, out)

__all__ = [
    "fp8_fp4_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "mqa_logits",
    "paged_mqa_logits",
    "paged_mqa_logits_persisten",
    "paged_mqa_logits_fp4",
    "page_mqa_logits_sparse_mask",
]
