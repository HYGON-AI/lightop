# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for group_matmul_gelu_quant (BF16 and FP8 paths)."""

import pytest
import torch

from lightop.moe.group_matmul_gelu_quant import (
    group_matmul_gelu_quant,
    group_matmul_gelu_quant_fp8,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _reference_gelu_quant(y):
    """Independent (non-Triton) reference for the fused epilogue."""
    act = torch.nn.functional.gelu(y.float(), approximate="none")
    amax = act.abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / 127.0
    quant = torch.round(act / scale[:, None]).clamp(-127.0, 127.0).to(torch.int8)
    return quant, scale


def _random_group_list(expert_num, seq, device):
    counts = torch.zeros(expert_num, dtype=torch.int64, device=device)
    for _ in range(seq):
        counts[torch.randint(0, expert_num, (1,), device=device)] += 1
    return counts


@pytest.mark.parametrize("seq,expert_num,k,n", [
    (33, 4, 32, 32),
    (257, 8, 64, 128),
])
def test_bf16_matches_reference(seq, expert_num, k, n):
    torch.manual_seed(0)
    device = "cuda"
    group_list = _random_group_list(expert_num, seq, device)
    x = torch.randn(seq, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(expert_num, k, n, dtype=torch.bfloat16, device=device)

    q, scale = group_matmul_gelu_quant(x, w, group_list)

    # Independent dense reference: same per-expert matmul, done with a plain
    # Python loop instead of torch.mm's fused path, then the epilogue above.
    y_ref = torch.empty(seq, n, dtype=torch.bfloat16, device=device)
    start = 0
    for e, count in enumerate(group_list.tolist()):
        if count:
            y_ref[start:start + count] = x[start:start + count].float() @ w[e].float()
        start += count
    q_ref, scale_ref = _reference_gelu_quant(y_ref)

    assert torch.allclose(scale, scale_ref, rtol=1e-3, atol=1e-6)
    assert (q.to(torch.int32) - q_ref.to(torch.int32)).abs().max() <= 1


@pytest.mark.parametrize("seq,expert_num,k,n", [
    # k and n must be multiples of the kernel's fixed BK=64 / BN=128 tiles.
    (33, 4, 64, 128),
    (257, 8, 64, 128),
])
def test_fp8_matches_reference(seq, expert_num, k, n):
    torch.manual_seed(0)
    device = "cuda"
    group_list = _random_group_list(expert_num, seq, device)
    x = torch.randn(seq, k, device=device)
    w = torch.randn(expert_num, k, n, device=device)

    x_scale = x.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    x_q = (x / x_scale[:, None]).to(torch.float8_e4m3fn)
    w_scale = w.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    w_q = (w / w_scale[:, None, :]).to(torch.float8_e4m3fn)

    q, scale = group_matmul_gelu_quant_fp8(x_q, x_scale, w_q, w_scale, group_list)

    y_ref = torch.empty(seq, n, dtype=torch.bfloat16, device=device)
    start = 0
    for e, count in enumerate(group_list.tolist()):
        if count:
            xf = x_q[start:start + count].float() * x_scale[start:start + count, None]
            wf = w_q[e].float() * w_scale[e][None, :]
            y_ref[start:start + count] = (xf @ wf).to(torch.bfloat16)
        start += count
    q_ref, scale_ref = _reference_gelu_quant(y_ref)

    # FP8 matmul + bf16 rounding introduces more noise than BF16 matmul, so
    # this tolerance is looser than the BF16 test above.
    assert torch.allclose(scale, scale_ref, rtol=5e-2, atol=1e-5)
    assert (q.to(torch.int32) - q_ref.to(torch.int32)).abs().max() <= 2
