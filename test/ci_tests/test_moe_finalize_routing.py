# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for moe_finalize_routing (BF16 and FP8 paths)."""

import pytest
import torch

from lightop.moe.moe_finalize_routing import build_src_to_dst, moe_finalize_routing


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _reference_finalize(fmt, data, scale, topk_weights, src_to_dst):
    """Independent (non-Triton) per-token reference."""
    seq, topk = topk_weights.shape
    s2d = src_to_dst.view(seq, topk)
    rows = data.index_select(0, s2d.reshape(-1)).float()
    if fmt == "fp8":
        rows = rows * scale.index_select(0, s2d.reshape(-1))[:, None]
    rows = rows.view(seq, topk, -1)
    out = (topk_weights.float()[:, :, None] * rows).sum(dim=1)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("seq,topk,expert_num,hidden", [
    (5, 2, 4, 64),
    (129, 4, 8, 256),
])
def test_bf16_matches_reference(seq, topk, expert_num, hidden):
    torch.manual_seed(0)
    device = "cuda"
    topk_ids = torch.randint(0, expert_num, (seq, topk), dtype=torch.int64, device=device)
    src_to_dst = build_src_to_dst(topk_ids)
    data = torch.randn(seq * topk, hidden, dtype=torch.bfloat16, device=device)
    weights = torch.rand(seq, topk, device=device)

    out = moe_finalize_routing("bf16", data, None, weights, src_to_dst)
    out_ref = _reference_finalize("bf16", data, None, weights, src_to_dst)

    assert torch.allclose(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("seq,topk,expert_num,hidden", [
    (5, 2, 4, 64),
    (129, 4, 8, 256),
])
def test_fp8_matches_reference(seq, topk, expert_num, hidden):
    torch.manual_seed(0)
    device = "cuda"
    topk_ids = torch.randint(0, expert_num, (seq, topk), dtype=torch.int64, device=device)
    src_to_dst = build_src_to_dst(topk_ids)
    data_f = torch.randn(seq * topk, hidden, device=device)
    scale = data_f.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    data = (data_f / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.rand(seq, topk, device=device)

    out = moe_finalize_routing("fp8", data, scale, weights, src_to_dst)
    out_ref = _reference_finalize("fp8", data, scale, weights, src_to_dst)

    assert torch.allclose(out.float(), out_ref.float(), rtol=5e-2, atol=5e-2)


def test_unsupported_layout_falls_back_without_raising():
    """A non-unit-stride view must still produce a correct (fallback) result."""
    device = "cuda"
    seq, topk, hidden = 5, 2, 64
    topk_ids = torch.randint(0, 4, (seq, topk), dtype=torch.int64, device=device)
    src_to_dst = build_src_to_dst(topk_ids)
    data_wide = torch.randn(seq * topk, hidden * 2, dtype=torch.bfloat16, device=device)
    data = data_wide[:, ::2]  # stride(1) == 2, not supported by the fast kernel
    weights = torch.rand(seq, topk, device=device)

    out = moe_finalize_routing("bf16", data, None, weights, src_to_dst)
    out_ref = _reference_finalize("bf16", data, None, weights, src_to_dst)
    assert torch.allclose(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)
