# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from lightop.norm import gated_rms_norm_per_token_fp8_quant

EPS = 1e-6
FP8_MAX = 448.0
pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires HIP"
)


def inputs(tokens, heads, gate_layout):
    x = torch.randn(tokens * heads, 128, device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(tokens, heads, 384, device=x.device, dtype=x.dtype)
    gate = projection[:, :, -128:]
    if gate_layout == "flat_strided":
        gate = gate.view(tokens * heads, 128)
    elif gate_layout == "contiguous":
        gate = gate.contiguous().view(tokens * heads, 128)
    weight = torch.randn(128, device=x.device, dtype=x.dtype) * 0.25 + 1
    return x, gate, weight


def reference(x, gate, weight, heads):
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + EPS)
    gated = (normalized * weight.float()) * F.silu(gate.reshape_as(x).float())
    rounded = gated.to(torch.bfloat16).float().reshape(-1, heads * 128)
    absmax = rounded.abs().amax(dim=-1, keepdim=True)
    inv = torch.where(absmax == 0, 0.0, FP8_MAX / absmax)
    q = (rounded * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, absmax / FP8_MAX


def assert_matches(actual, expected):
    q, scale = actual
    q_expected, scale_expected = expected
    assert q.shape == q_expected.shape and q.dtype == torch.float8_e4m3fn
    assert scale.shape == scale_expected.shape and scale.dtype == torch.float32
    torch.testing.assert_close(scale, scale_expected, rtol=8e-3, atol=0)
    target = q_expected.float() * scale_expected
    observed = q.float() * scale
    torch.testing.assert_close(observed, target, rtol=0.12, atol=0.002)
    assert (observed - target).norm() <= 8e-3 * target.norm()


@pytest.mark.parametrize(
    "tokens,heads,gate_layout",
    [(1, 24, "contiguous"), (4, 24, "flat_strided"), (128, 24, "strided"),
     (512, 24, "strided"), (1024, 24, "strided"), (4, 12, "contiguous"),
     (4, 48, "strided")],
)
def test_matches_pytorch(tokens, heads, gate_layout):
    torch.manual_seed(1918)
    x, gate, weight = inputs(tokens, heads, gate_layout)
    assert_matches(
        gated_rms_norm_per_token_fp8_quant(x, gate, weight, EPS, num_heads=heads),
        reference(x, gate, weight, heads),
    )


def test_zero_scales_and_empty_batch():
    x, gate, weight = inputs(2, 24, "contiguous")
    gate.zero_()
    q, scale = gated_rms_norm_per_token_fp8_quant(x, gate, weight, EPS, num_heads=24)
    assert torch.count_nonzero(q.float()).item() == 0
    assert torch.count_nonzero(scale).item() == 0
    q, scale = gated_rms_norm_per_token_fp8_quant(x[:0], gate[:0], weight, EPS, num_heads=24)
    assert q.shape == (0, 3072)
    assert scale.shape == (0, 1)


def test_graph_replays_new_input():
    torch.manual_seed(1919)
    x, gate, weight = inputs(4, 24, "strided")
    gated_rms_norm_per_token_fp8_quant(x, gate, weight, EPS, num_heads=24)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gated_rms_norm_per_token_fp8_quant(x, gate, weight, EPS, num_heads=24)
    x.neg_()
    graph.replay()
    assert_matches(result, reference(x, gate, weight, 24))


def test_rejects_invalid_heads_and_gate_shape():
    x, gate, weight = inputs(2, 24, "strided")
    with pytest.raises(ValueError, match="complete tokens"):
        gated_rms_norm_per_token_fp8_quant(x[:-1], gate, weight, EPS, num_heads=24)
    with pytest.raises(ValueError, match="Gate shape"):
        gated_rms_norm_per_token_fp8_quant(x, gate[:, :-1], weight, EPS, num_heads=24)
