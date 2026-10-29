"""BF16 fused routing with the DeepGEMM public calling convention."""

from .. import op


def bf16_mega_gate_hcu(
    x,
    weight,
    num_topk,
    use_shared_as_routed,
    num_shared_experts,
    routed_scaling_factor,
    ep_rank,
    scoring_func="sqrtsoftplus",
    mask=None,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    to_physical_map=None,
    logical_count=None,
    unmapped_topk_idx=None,
    force_random=None,
    out=None,
):
    """Return ``(topk_idx: INT64, topk_weights: FP32)`` of shape [M,num_topk].

    Parameter names, order, defaults, output order and dtypes follow DeepGEMM.
    HCU supports contiguous BF16 [M,H]/[E,H], optional FP32 bias [E], optional
    BOOL token mask [M], and preallocated ``out=(ids, weights)``. H is positive
    and 256-aligned; E is 4-aligned in [4,512]; num_topk is in [1,min(E,32)].
    Masked tokens produce -1 IDs and zero weights. Weights are normalized and
    multiplied by routed_scaling_factor. The tested LightOp numerical contract
    is retained: rank-order FP32 sum, no added epsilon, and stable ID ties.

    Shared-as-routed, image, fixed, physical-map and random routing are not
    implemented on HCU and raise explicitly. num_shared_experts is ignored
    when use_shared_as_routed=False, as upstream does.

    Scratch is managed internally. Warm each stream before capture; graphs
    captured on the same stream share counters and must replay sequentially.
    """
    result = op.bf16_mega_gate_hcu(
        x,
        weight,
        num_topk,
        use_shared_as_routed,
        num_shared_experts,
        routed_scaling_factor,
        ep_rank,
        scoring_func,
        mask,
        bias,
        image_bias,
        image_token_mask,
        fix_routing_mask,
        to_physical_map,
        logical_count,
        unmapped_topk_idx,
        force_random,
        out,
    )
    return (out[0], out[1]) if out is not None else result


__all__ = ["bf16_mega_gate_hcu"]
