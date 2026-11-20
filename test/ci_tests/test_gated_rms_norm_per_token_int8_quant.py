# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Check the HCU gated normalization and quantization against PyTorch."""

import pytest
import torch
from torch.nn import functional as F

from lightop.norm import gated_rms_norm_per_token_int8_quant

EPS = 1e-6
HEAD_DIM = 128
pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires HIP"
)


def _inputs(tokens, heads, strided=False):
    x = torch.randn(
        (tokens * heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(HEAD_DIM, dtype=x.dtype, device=x.device) * 0.25 + 1
    if not strided:
        return x, torch.randn_like(x), weight
    width = heads * HEAD_DIM
    projection = torch.randn((tokens, width * 3), dtype=x.dtype, device=x.device)
    gate = projection[:, -width:].view(tokens, heads, HEAD_DIM)
    return x, gate, weight


def _reference(x, gate, weight, heads):
    x = x.float()
    norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    gated = (norm * weight.float()) * F.silu(gate.reshape_as(x).float())
    rounded = gated.to(torch.bfloat16).float().reshape(-1, heads * HEAD_DIM)
    absmax = rounded.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    return (rounded * (127.0 / absmax)).round().to(torch.int8), absmax / 127.0


def _assert_matches(actual, reference):
    q, scale = actual
    expected_q, expected_scale = reference
    assert q.shape == expected_q.shape and q.dtype == torch.int8
    assert scale.shape == expected_scale.shape and scale.dtype == torch.float32
    torch.testing.assert_close(scale, expected_scale, rtol=8e-3, atol=0)
    assert (q.to(torch.int16) - expected_q.to(torch.int16)).abs().max().item() <= 1
    reference_values = expected_q.float() * expected_scale
    error = (q.float() * scale - reference_values).norm()
    assert error <= 1.5e-3 * reference_values.norm()


@pytest.mark.parametrize(
    "tokens,heads,strided",
    [(1, 24, False), (4, 24, False), (64, 24, True), (128, 24, True),
     (1024, 24, True), (2048, 24, True), (4, 12, False), (4, 48, False)],
)
def test_matches_pytorch(tokens, heads, strided):
    torch.manual_seed(19)
    x, gate, weight = _inputs(tokens, heads, strided)
    actual = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=heads)
    _assert_matches(actual, _reference(x, gate, weight, heads))


def test_zero_scale_floor_and_empty_batch():
    x, gate, weight = _inputs(2, 24)
    gate.zero_()
    q, scale = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    assert torch.count_nonzero(q).item() == 0
    torch.testing.assert_close(scale, torch.full_like(scale, 1e-10 / 127), rtol=1e-7, atol=0)
    q, scale = gated_rms_norm_per_token_int8_quant(x[:0], gate[:0], weight, EPS, num_heads=24)
    assert q.shape == (0, 3072)
    assert scale.shape == (0, 1)


def test_rejects_incomplete_token_and_mismatched_gate():
    x, gate, weight = _inputs(2, 24)
    with pytest.raises(ValueError, match="complete tokens"):
        gated_rms_norm_per_token_int8_quant(x[:-1], gate[:-1], weight, EPS, num_heads=24)
    with pytest.raises(ValueError, match="Gate shape"):
        gated_rms_norm_per_token_int8_quant(x, gate[:-1], weight, EPS, num_heads=24)


def test_graph_replays_new_input():
    torch.manual_seed(23)
    x, gate, weight = _inputs(4, 24)
    before, _ = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    x.neg_()
    gate.mul_(0.5)
    graph.replay()
    _assert_matches(actual, _reference(x, gate, weight, 24))
    assert not torch.equal(before, actual[0])
