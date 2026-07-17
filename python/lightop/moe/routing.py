"""MoE routing, expert alignment, and expert reduction APIs."""

from .. import op
from .._lmslim_native.vllm_compat.moe_align_block_size import moe_align_block_size
from ..moe_align_sum import moe_sum
from ..moe_fused_gate import (
    kimi_k2_moe_fused_gate,
    moe_fused_gate,
    moe_fused_gate_sqrtsoftplus,
)


def moe_align_block_size_out(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    expert_map=None,
    expert_mask=None,
    num_local_tokens=None,
    is_ep=False,
):
    """Run expert alignment into caller-provided output tensors."""
    return op.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
        expert_mask,
        num_local_tokens,
        is_ep,
    )

__all__ = [
    "kimi_k2_moe_fused_gate",
    "moe_align_block_size",
    "moe_align_block_size_out",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_sum",
]
