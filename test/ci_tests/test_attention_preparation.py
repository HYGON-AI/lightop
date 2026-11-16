import pytest
import torch

from lightop.activation import fused_clamp_smooth
from lightop.quant import per_token_dynamic_quant_int8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="HIP GPU required",
)


def assert_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows", [0, 1, 18, 193, 3072, 5120])
@pytest.mark.parametrize("stride", [512, 576])
def test_clamp_smooth_preserves_bf16_rounding(rows, stride):
    torch.manual_seed(131103)
    backing = torch.randn((rows, stride), device="cuda", dtype=torch.bfloat16) * 16
    x = backing[:, :512]
    smooth = torch.rand(512, device="cuda", dtype=torch.float32) + 0.1
    original = backing.clone()
    expected = x.clamp(-8, 8) * smooth.bfloat16()
    actual = fused_clamp_smooth(x, smooth, -8, 8)
    assert_bits(actual, expected)
    assert_bits(backing, original)


@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 576, 3072])
def test_per_head_quant_and_gate(rows):
    torch.manual_seed(131104)
    x = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    smooth = torch.rand(128, device="cuda") + 0.5
    weights = torch.randn(rows, device="cuda")
    if rows:
        x[0].zero_()
    value = x.float() * smooth
    maximum = value.abs().amax(-1)
    scale = maximum / torch.full_like(maximum, 127)
    inverse = torch.where(maximum == 0, 0., torch.full_like(maximum, 127) / maximum)
    expected = (value * inverse[:, None]).round().clamp(-128, 127).to(torch.int8)
    actual, actual_scale, gate = per_token_dynamic_quant_int8(x, smooth, weights=weights)
    assert_bits(actual, expected)
    assert_bits(actual_scale, scale)
    assert_bits(gate, weights * scale)
    legacy, legacy_scale = per_token_dynamic_quant_int8(x, smooth)
    assert_bits(actual, legacy)
    assert_bits(actual_scale, legacy_scale)


def test_preparation_graph_updates():
    x = torch.randn((18, 576), device="cuda", dtype=torch.bfloat16)
    smooth = torch.rand(512, device="cuda")
    q = torch.randn((576, 128), device="cuda", dtype=torch.bfloat16)
    qs = torch.rand(128, device="cuda")
    w = torch.randn(576, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_clamp_smooth(x[:, :512], smooth, -8, 8)
        per_token_dynamic_quant_int8(q, qs, weights=w)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fused_clamp_smooth(x[:, :512], smooth, -8, 8)
            quant, scales, gate = per_token_dynamic_quant_int8(q, qs, weights=w)
        for _ in range(3):
            x.normal_(); smooth.uniform_(); q.normal_(); qs.uniform_(); w.normal_()
            graph.replay()
            assert_bits(result, x[:, :512].clamp(-8, 8) * smooth.bfloat16())
            native_q, native_s = per_token_dynamic_quant_int8(q, qs)
            assert_bits(quant, native_q)
            assert_bits(scales, native_s)
            assert_bits(gate, w * native_s)
    torch.cuda.current_stream().wait_stream(stream)
