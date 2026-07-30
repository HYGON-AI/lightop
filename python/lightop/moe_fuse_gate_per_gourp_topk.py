from . import op


def moe_fuse_gate_per_gourp_topk(
    x,
    bias,
    num_expert_group,
    topk_per_group,
    num_pad_experts,
    num_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output=True,
):
    """Run the per-group top-k router.

    The public name intentionally keeps ``gourp`` for frontend compatibility.
    """
    return op.moe_fuse_gate_per_gourp_topk(
        x,
        bias,
        num_expert_group,
        topk_per_group,
        num_pad_experts,
        num_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
