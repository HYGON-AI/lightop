# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for moe_init_routing."""

import pytest
import torch

from lightop.moe.moe_init_routing import moe_init_routing


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("seq,topk,expert_num,hidden", [
    (1, 1, 4, 16),
    (37, 3, 8, 64),
    (256, 4, 16, 128),
])
def test_matches_reference(seq, topk, expert_num, hidden):
    torch.manual_seed(0)
    x = torch.randn(seq, hidden, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, expert_num, (seq, topk), dtype=torch.int64, device="cuda")

    expanded_x, expanded_row_idx, group_list, sorted_expert_ids = moe_init_routing(
        x, topk_ids, expert_num
    )

    # Each output row must be exactly the source row it claims to come from.
    assert torch.equal(expanded_x, x.index_select(0, expanded_row_idx))

    # Output rows must be grouped by expert, non-decreasing expert id.
    assert torch.all(sorted_expert_ids[1:] >= sorted_expert_ids[:-1])

    # group_list must match the per-expert row counts actually emitted.
    assert torch.equal(group_list, torch.bincount(sorted_expert_ids, minlength=expert_num))

    # Every (token, slot) pair in topk_ids must appear exactly once in the
    # expanded output, i.e. this is a permutation of seq*topk rows.
    assert expanded_row_idx.numel() == seq * topk
    counts_per_token = torch.bincount(expanded_row_idx, minlength=seq)
    assert torch.equal(counts_per_token, torch.full((seq,), topk, device="cuda"))


def test_empty_input():
    x = torch.randn(0, 8, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros(0, 2, dtype=torch.int64, device="cuda")
    expanded_x, expanded_row_idx, group_list, sorted_expert_ids = moe_init_routing(x, topk_ids, 4)
    assert expanded_x.shape == (0, 8)
    assert expanded_row_idx.numel() == 0
    assert torch.equal(group_list, torch.zeros(4, dtype=group_list.dtype, device="cuda"))
    assert sorted_expert_ids.numel() == 0
