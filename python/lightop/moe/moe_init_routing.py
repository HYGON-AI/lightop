# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE routing: sort tokens by assigned expert and gather rows into expert-
contiguous groups, feeding group_matmul_gelu_quant.

This op is pure data movement (sort + gather + bincount), with no
multiply-add, so its ceiling is the device's D2D memory-copy bandwidth.
Measured against that ceiling, the native torch implementation below
already reaches roughly 88%-99% of it; two rounds of custom-kernel and
micro-optimization attempts did not produce a reproducible gain across the
full benchmark shape matrix, so this is the implementation as delivered.
"""

import torch


def moe_init_routing(x: torch.Tensor, topk_ids: torch.Tensor, expert_num: int):
    """Group tokens by expert.

    x: [seq, hidden]. topk_ids: [seq, topk] expert assignment per token slot.
    Returns (expanded_x, expanded_row_idx, group_list, sorted_expert_ids):
    expanded_x: [seq*topk, hidden] rows reordered into expert-contiguous
    blocks. expanded_row_idx: [seq*topk] original row index per output row.
    group_list: [expert_num] row count per expert. sorted_expert_ids:
    [seq*topk] expert id per output row (non-decreasing).
    """
    assert x.is_cuda and topk_ids.is_cuda

    seq, hidden = x.shape
    topk = topk_ids.shape[1]

    flat_expert_ids = topk_ids.reshape(-1)
    flat_row_idx = torch.arange(seq, device=x.device).repeat_interleave(topk)

    sort_order = torch.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[sort_order]
    expanded_row_idx = flat_row_idx[sort_order]

    expanded_x = x.index_select(0, expanded_row_idx)
    group_list = torch.bincount(sorted_expert_ids, minlength=expert_num)

    return expanded_x, expanded_row_idx, group_list, sorted_expert_ids


__all__ = ["moe_init_routing"]
