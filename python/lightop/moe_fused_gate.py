import torch
import enum
import torch.nn as nn
#from lightop import op
from . import op

def moe_fused_gate(
    x,
    bias,
    num_expert_group,
    topk_group,
    topk,
    n_share_experts_fusion,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output=True,
    output_indices_int64=False,
):
    return op.moe_fused_gate(
        x,
        bias,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
        output_indices_int64,
    )


def kimi_k2_moe_fused_gate(
    x,
    bias,
    topk,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    return op.kimi_k2_moe_fused_gate(
        x,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def moe_fused_gate_sqrtsoftplus(
    x,
    bias,
    topk,
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    return op.moe_fused_gate_sqrtsoftplus(
        x,
        bias,
        topk,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def moe_fused_gate_sqrtsoftplus_v41_hcu(
    x,
    bias,
    topk,
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    return op.moe_fused_gate_sqrtsoftplus_v41_hcu(
        x,
        bias,
        topk,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
