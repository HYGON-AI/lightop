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
    *,
    return_logits=False,
    renormalize=True,
    apply_scale=True,
):
    """Return ``(topk_idx: INT64, topk_weights: FP32)`` of shape [M,num_topk].

    By default, parameter names, order, defaults, output order and dtypes
    follow DeepGEMM. The keyword-only extensions allow frameworks to request
    ``(ids, weights, logits)`` with ``return_logits=True`` and independently
    control normalization/scaling. Logits are FP32 [M,E] before scoring/bias.
    These extensions use the same kernel and do not recompute the router GEMM.
    HCU supports contiguous BF16 [M,H]/[E,H], optional FP32 bias [E], optional
    BOOL token mask [M], and preallocated ``out=(ids, weights)``. H is positive
    and 256-aligned; E is 4-aligned in [4,512]; num_topk is in [1,min(E,32)].
    Masked tokens produce -1 IDs and zero weights. By default weights are
    normalized and multiplied by routed_scaling_factor. The tested LightOp numerical contract
    is retained: rank-order FP32 sum, no added epsilon, and stable ID ties.

    Shared-as-routed, image, fixed, physical-map and random routing are not
    implemented on HCU and raise explicitly. num_shared_experts is ignored
    when use_shared_as_routed=False, as upstream does.

    Scratch is managed internally. Warm each stream before capture; graphs
    captured on the same stream share counters and must replay sequentially.
    """
    if return_logits or not renormalize or not apply_scale:
        # Match the native DeepGEMM-style entry's routing restrictions before
        # using its shared implementation to expose logits/scaling controls.
        if scoring_func != "sqrtsoftplus":
            raise ValueError("scoring_func must be sqrtsoftplus")
        if ep_rank < 0:
            raise ValueError("ep_rank must be nonnegative")
        if use_shared_as_routed:
            raise NotImplementedError("HCU mega gate does not support shared-as-routed experts")
        if any(value is not None for value in (
            image_bias, image_token_mask, fix_routing_mask, to_physical_map,
            logical_count, unmapped_topk_idx, force_random,
        )):
            raise NotImplementedError(
                "HCU mega gate does not support image, fixed, physical-map or random routing"
            )
        if out is not None and len(out) != 2:
            raise ValueError("out must be (ids_int64, weights_fp32)")
        weights, ids, logits = op._bf16_mega_gate_hcu(
            x, weight, bias, num_topk, renormalize, routed_scaling_factor,
            apply_scale, out=None if out is None else (out[1], out[0]), mask=mask,
        )
        if out is not None:
            ids, weights = out
        return (ids, weights, logits) if return_logits else (ids, weights)
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
