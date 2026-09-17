"""Direct bindings for existing LightOp MQA APIs."""

from ..gemmopt import (
    get_paged_mqa_logits_metadata,
    mqa_logits,
    paged_mqa_logits,
    page_mqa_logits_sparse_mask,
)


def paged_mqa_logits_fp4(q, weights, slots, lens, table, page_size, out=None):
    """Score BF16 queries against the FP4 indexer pool on gfx928/936/938.

    q [B,H,128] and weights [B,H] are BF16; slots [B,L] and lens [B]
    are contiguous int64. table is uint8 [pages,page_size*68], storing all
    E2M1 payload rows followed by all per-32 E8M0 scale rows in each page.
    The caller guarantees valid pool slots for j < min(lens[b], L).
    Returns FP32 [B,L] with -inf for invisible positions. Dot, weighted
    head products and final head sum round to BF16 like the SGLang reference.
    An optional contiguous FP32 out supports allocation-free graph replay.
    Inputs must share a device. Unsupported shapes/layouts raise an error.
    gfx938 H32 workloads with at least 48K scores use lossless FP8 MMAC;
    other workloads use BF16 MMAC. Q values that cannot be encoded exactly
    retain BF16 computation. Query conversion runs on device on every replay.
    Wide or extreme KV block scales also select the BF16 fallback.
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
    "get_paged_mqa_logits_metadata",
    "mqa_logits",
    "paged_mqa_logits",
    "paged_mqa_logits_fp4",
    "page_mqa_logits_sparse_mask",
]
