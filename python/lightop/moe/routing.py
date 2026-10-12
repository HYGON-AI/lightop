# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE routing, expert alignment, and expert reduction APIs."""

from .. import op
from .._lmslim_native.vllm_compat.moe_align_block_size import moe_align_block_size
from ..moe_align_sum import moe_sum
from ..moe_fused_gate import (
    kimi_k2_moe_fused_gate,
    moe_fused_gate,
    moe_fused_gate_sqrtsoftplus,
    moe_fused_gate_sqrtsoftplus_v41_hcu,
)


topk_ids_postprocess = op.topk_ids_postprocess
topk_softmax = op.topk_softmax


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
    is_fuse_fill=True,
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
        is_fuse_fill,
    )

__all__ = [
    "kimi_k2_moe_fused_gate",
    "moe_align_block_size",
    "moe_align_block_size_out",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_fused_gate_sqrtsoftplus_v41_hcu",
    "moe_sum",
    "topk_ids_postprocess",
    "topk_softmax",
]
