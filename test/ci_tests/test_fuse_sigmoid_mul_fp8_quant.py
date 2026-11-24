# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightop.activation import fuse_sigmoid_mul_fp8_quant

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires HIP"
)


def make_input(tokens, heads, head_dim, layout):
    width = heads * head_dim
    x = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16)
    if layout == "flat":
        return x, torch.randn_like(x)
    if layout == "flat-strided":
        storage = torch.randn((tokens, width + 16), device=x.device, dtype=x.dtype)
        return x, storage[:, 16:]
    storage = torch.randn((tokens, heads, 2 * head_dim), device=x.device, dtype=x.dtype)
    return x, storage[..., head_dim:]


def reference(x, gate):
    gated = (x.float() * torch.sigmoid(gate.reshape_as(x).float())).to(torch.bfloat16).float()
    amax = gated.abs().amax(dim=1, keepdim=True)
    inv = torch.where(amax == 0, 0.0, 448.0 / amax)
    return (gated * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), amax / 448.0


def check_reference(actual, expected):
    q, scale = actual
    ref_q, ref_scale = expected
    assert q.dtype == torch.float8_e4m3fn and q.shape == ref_q.shape
    assert scale.dtype == torch.float32 and scale.shape == ref_scale.shape
    torch.testing.assert_close(scale, ref_scale, rtol=8e-3, atol=0)
    observed = q.float() * scale
    target = ref_q.float() * ref_scale
    torch.testing.assert_close(observed, target, rtol=0.12, atol=0.002)
    assert (observed - target).norm() <= 8e-3 * target.norm()


@pytest.mark.parametrize(
    "tokens,heads,head_dim,layout",
    [(1, 12, 256, "flat"), (4, 12, 256, "flat-strided"),
     (128, 12, 256, "head-strided"), (1024, 12, 256, "head-strided"),
     (17, 5, 128, "head-strided"), (3, 3, 257, "head-strided")],
)
def test_matches_pytorch(tokens, heads, head_dim, layout):
    torch.manual_seed(417)
    x, gate = make_input(tokens, heads, head_dim, layout)
    x_before, gate_before = x.clone(), gate.clone()
    check_reference(fuse_sigmoid_mul_fp8_quant(x, gate), reference(x, gate))
    assert torch.equal(x.view(torch.int16), x_before.view(torch.int16))
    assert torch.equal(gate.view(torch.int16), gate_before.view(torch.int16))


def test_exact_codes_and_zero_scale():
    x = torch.zeros((2, 3072), device="cuda", dtype=torch.bfloat16)
    x[0, 1::2].neg_()
    x[1, :6] = torch.tensor([448, -448, 7, -7, 0.125, -0.125], device=x.device, dtype=x.dtype)
    gate = torch.zeros_like(x)
    q, scale = fuse_sigmoid_mul_fp8_quant(x, gate)
    ref_q, ref_scale = reference(x, gate)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32))
    assert scale[0].item() == 0.0


def test_empty_and_invalid_gate():
    x, gate = make_input(0, 12, 256, "head-strided")
    q, scale = fuse_sigmoid_mul_fp8_quant(x, gate)
    assert q.shape == (0, 3072) and scale.shape == (0, 1)
    x, gate = make_input(2, 12, 256, "head-strided")
    with pytest.raises(ValueError, match="Gate shape"):
        fuse_sigmoid_mul_fp8_quant(x, gate[:, :-1])


def test_graph_replays_changed_inputs():
    torch.manual_seed(418)
    x, gate = make_input(4, 12, 256, "head-strided")
    fuse_sigmoid_mul_fp8_quant(x, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fuse_sigmoid_mul_fp8_quant(x, gate)
    x.neg_()
    gate.mul_(0.5)
    graph.replay()
    check_reference(result, reference(x, gate))
