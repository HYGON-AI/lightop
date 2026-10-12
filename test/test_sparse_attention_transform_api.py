# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for sparse-attention top-k index transforms."""

import pytest
import torch

import lightop
import lightop.attention
import lightop.sampling
from lightop.attention import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    topk_transform_512,
)


SPARSE_ATTENTION_APIS = [
    "fast_topk_transform_fused",
    "fast_topk_transform_ragged_fused",
    "topk_transform_512",
]


@pytest.mark.parametrize("name", SPARSE_ATTENTION_APIS)
def test_sparse_transform_has_single_attention_owner(name: str) -> None:
    assert getattr(lightop.attention, name) is getattr(lightop, name)
    assert not hasattr(lightop.sampling, name)


def _expected_raw_indices(
    scores: torch.Tensor, lengths: torch.Tensor, topk: int
) -> torch.Tensor:
    positions = torch.arange(scores.shape[1], device=scores.device)
    masked = scores.masked_fill(
        positions.unsqueeze(0) >= lengths.unsqueeze(1),
        float("-inf"),
    )
    return torch.topk(masked, topk, dim=1).indices.to(torch.int32)


def test_fast_topk_transform_fused_matches_torch() -> None:
    torch.manual_seed(51)
    batch_size, width, topk = 2, 2064, 2048
    scores = torch.randn(
        (batch_size, width), device="cuda", dtype=torch.float32
    )
    lengths = torch.tensor(
        [2048, 2050], device="cuda", dtype=torch.int32
    )
    page_table = (
        torch.arange(width, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    page_table[1].add_(10000)
    cu_seqlens_q = torch.tensor(
        [0, 1, 2], device="cuda", dtype=torch.int32
    )

    actual = fast_topk_transform_fused(
        scores, lengths, page_table, cu_seqlens_q, topk
    )
    raw_indices = _expected_raw_indices(scores, lengths, topk)
    expected = torch.gather(page_table, 1, raw_indices.long())
    torch.cuda.synchronize()

    assert torch.equal(
        actual.sort(dim=1).values,
        expected.sort(dim=1).values,
    )


def test_fast_topk_transform_ragged_fused_matches_torch() -> None:
    torch.manual_seed(52)
    batch_size, width, topk = 2, 2064, 2048
    scores = torch.randn(
        (batch_size, width), device="cuda", dtype=torch.float32
    )
    lengths = torch.tensor(
        [2048, 2050], device="cuda", dtype=torch.int32
    )
    offsets = torch.tensor(
        [100, 10000], device="cuda", dtype=torch.int32
    )

    actual = fast_topk_transform_ragged_fused(
        scores, lengths, offsets, topk
    )
    expected = _expected_raw_indices(scores, lengths, topk)
    expected = expected + offsets.unsqueeze(1)
    torch.cuda.synchronize()

    assert torch.equal(
        actual.sort(dim=1).values,
        expected.sort(dim=1).values,
    )
