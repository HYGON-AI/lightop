# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Public LightOp API checks; the oracle does not depend on SGLang or Triton."""

import pytest
import torch

from lightop.activation import swiglu_oai_fp8_quant_contiguous

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(x, alpha=1.702, beta=1.0, limit=7.0):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    activated = (gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16().float()
    # Force one round-to-nearest FP32 division. Tensor/scalar division may
    # multiply by a pre-rounded reciprocal instead, shifting a scale by 1 ULP.
    scale = (activated.abs().amax(dim=-1, keepdim=True).double() / 448.0).float()
    # Finite-precision contract: invert the rounded FP32 scale, then multiply.
    # Replacing this by division changes FP8 midpoint rounding.
    inverse = torch.where(
        scale > 0, (1.0 / scale.double()).float(), torch.zeros_like(scale)
    )
    normalized = (activated.double() * inverse.double()).float()
    quantized = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale


@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),
        (5, 3),
        (7, 8),
        (7, 127),
        (17, 384),
        (5, 385),
        (17, 3072),
        (5, 3073),
        (3, 8192),
    ],
)
def test_formula_and_tails(rows, cols):
    torch.manual_seed(2026 + cols)
    x = (torch.randn(rows, 2 * cols, device="cuda") * 5).bfloat16()
    x[0].zero_()
    q, scale = swiglu_oai_fp8_quant_contiguous(x)
    qr, sr = reference(x)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (rows, cols)
    assert scale.dtype == torch.float32 and scale.shape == (rows, 1)
    torch.testing.assert_close(scale, sr, atol=1e-7, rtol=1e-6)
    a, b = q.float() * scale, qr.float() * sr
    assert (a - b).norm() / b.norm().clamp_min(1e-20) < 2e-3
    assert torch.isfinite(a).all()
    assert scale[0].item() == 0.0 and q[0].float().count_nonzero().item() == 0


def test_empty_and_unaligned_storage():
    x = torch.empty((0, 6144), device="cuda", dtype=torch.bfloat16)
    q, s = swiglu_oai_fp8_quant_contiguous(x)
    assert q.shape == (0, 3072) and s.shape == (0, 1)
    x = torch.randn(4 * 6144 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(4, 6144)
    q, s = swiglu_oai_fp8_quant_contiguous(x)
    qa, sa = swiglu_oai_fp8_quant_contiguous(x.clone())
    assert torch.equal(q.view(torch.uint8), qa.view(torch.uint8))
    assert torch.equal(s, sa)


def test_updated_graph_replay():
    x = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = swiglu_oai_fp8_quant_contiguous(x)
    for factor in [0.73, 1.49, 0.0]:
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        qr, sr = swiglu_oai_fp8_quant_contiguous(x)
        assert torch.equal(q.view(torch.uint8), qr.view(torch.uint8))
        assert torch.equal(s, sr)


@pytest.mark.parametrize(
    "alpha,beta,limit", [(1.0, 0.0, 4.0), (1.702, -0.5, 7.0), (0.5, 2.0, 2.0)]
)
def test_parameters(alpha, beta, limit):
    torch.manual_seed(42)
    x = torch.randn((17, 768), device="cuda", dtype=torch.bfloat16)
    q, s = swiglu_oai_fp8_quant_contiguous(x, alpha, beta, limit)
    qr, sr = reference(x, alpha, beta, limit)
    torch.testing.assert_close(s, sr, rtol=1e-6, atol=1e-7)
    assert (q.float() * s - qr.float() * sr).norm() / (qr.float() * sr).norm() < 2e-3


def test_input_guards():
    for x in [
        torch.randn((3, 7), device="cuda", dtype=torch.bfloat16),
        torch.randn((3, 8), device="cuda"),
        torch.randn((3, 8), dtype=torch.bfloat16),
        torch.randn((8, 8), device="cuda", dtype=torch.bfloat16).t(),
    ]:
        with pytest.raises(RuntimeError):
            swiglu_oai_fp8_quant_contiguous(x)
