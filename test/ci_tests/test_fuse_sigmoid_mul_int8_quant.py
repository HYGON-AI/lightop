# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightop.activation import fuse_sigmoid_mul_int8_quant


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires an HCU",
)


def reference(x, gate):
    gated = (x.float() * gate.reshape(x.shape).float().sigmoid()).to(x.dtype).float()
    amax = gated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    quantized = torch.round(gated * (127.0 / amax)).to(torch.int8)
    return quantized, amax / 127.0


@pytest.mark.parametrize("tokens,width", [(1, 3072), (4, 3072), (128, 3072), (512, 3072), (16, 640)])
def test_matches_pytorch(tokens, width):
    torch.manual_seed(42)
    attn = torch.randn((tokens, width), dtype=torch.bfloat16, device="cuda")
    if width == 3072:
        gate = torch.randn((tokens, 12, 512), dtype=attn.dtype, device=attn.device)[..., 256:]
    else:
        gate = torch.randn((tokens, 2 * width), dtype=attn.dtype, device=attn.device)[:, width:]
    expected, expected_scale = reference(attn, gate)
    quantized, scale = fuse_sigmoid_mul_int8_quant(attn, gate)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0.008)
    assert int((quantized.to(torch.int16) - expected).abs().max()) <= 1


@pytest.mark.parametrize("width", [640, 3072])
def test_exact_quantization_and_zero_row(width):
    ramp = ((torch.arange(width, device="cuda") % 255) - 127).to(torch.bfloat16)
    attn = torch.stack((torch.zeros_like(ramp), ramp * 2, ramp, ramp * 0.5))
    gate = torch.zeros_like(attn)
    original = attn.clone()
    quantized, scale = fuse_sigmoid_mul_int8_quant(attn, gate)
    expected, expected_scale = reference(attn, gate)
    torch.testing.assert_close(quantized, expected, atol=0, rtol=0)
    torch.testing.assert_close(scale[1:], expected_scale[1:], atol=0, rtol=0)
    torch.testing.assert_close(scale[:1], expected_scale[:1], atol=0, rtol=torch.finfo(torch.float32).eps)
    torch.testing.assert_close(attn, original, atol=0, rtol=0)


def test_empty_and_invalid_gate():
    attn = torch.empty((0, 3072), dtype=torch.bfloat16, device="cuda")
    quantized, scale = fuse_sigmoid_mul_int8_quant(attn, attn.reshape(0, 12, 256))
    assert quantized.shape == (0, 3072) and quantized.dtype == torch.int8
    assert scale.shape == (0, 1) and scale.dtype == torch.float32
    with pytest.raises(ValueError, match="Gate shape"):
        fuse_sigmoid_mul_int8_quant(attn, attn[:, :1])


def test_graph_replay_with_updated_input():
    attn = torch.ones((4, 3072), dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros((4, 12, 512), dtype=attn.dtype, device=attn.device)[..., 256:]
    fuse_sigmoid_mul_int8_quant(attn, gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantized, scale = fuse_sigmoid_mul_int8_quant(attn, gate)
    for value in (0.25, -1.5):
        attn.fill_(value)
        graph.replay()
        expected, expected_scale = reference(attn, gate)
        torch.testing.assert_close(quantized, expected, atol=0, rtol=0)
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
