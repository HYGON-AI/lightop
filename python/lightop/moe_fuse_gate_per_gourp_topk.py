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
    if x.dim() == 3:
        if x.size(1) != num_expert_group:
            raise ValueError(
                "for a 3D input, x.shape[1] must equal "
                f"num_expert_group, but got {x.size(1)} and "
                f"{num_expert_group}"
            )
        x = x.reshape(x.size(0), -1)

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
