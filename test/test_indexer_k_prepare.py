"""Indexer K preparation: BF16 stages, page-planar layout and graph replay."""
import pytest
import torch
from lightop import op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def hadamard():
    value = torch.ones(1, 1, device="cuda", dtype=torch.bfloat16)
    while value.shape[0] < 128:
        value = torch.cat((torch.cat((value, value), 1),
                           torch.cat((value, -value), 1)), 0)
    return value


def reference(key, weight, bias, positions, trig, neox):
    value = torch.nn.functional.layer_norm(key.float(), (128,), weight, bias, 1e-6).bfloat16()
    rotary = value[..., :64]
    cosine, sine = trig[positions].chunk(2, -1)
    if neox:
        partner = torch.cat((-rotary[..., 32:], rotary[..., :32]), -1)
        cosine = torch.cat((cosine, cosine), -1).unsqueeze(1)
        sine = torch.cat((sine, sine), -1).unsqueeze(1)
    else:
        partner = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), -1).flatten(-2)
        cosine = cosine.repeat_interleave(2, -1).unsqueeze(1)
        sine = sine.repeat_interleave(2, -1).unsqueeze(1)
    value[..., :64] = rotary * cosine + partner * sine
    value = torch.nn.functional.linear(value, hadamard()) * (128 ** -0.5)
    scale = value.float().abs().amax(-1).clamp_min(1e-6) / 127
    scaled = value.float() / scale[..., None]
    rounded = torch.where(scaled >= 0, (scaled + 0.5).floor(), (scaled - 0.5).ceil())
    return rounded.clamp(-127, 127).to(torch.int8).squeeze(1), scale.squeeze(1)


@pytest.mark.parametrize("tokens", [10, 15, 20])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_indexer_k_graph_and_cache_layout(tokens, neox, index_dtype, with_bias):
    torch.manual_seed(20261020)
    key = torch.randn(tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda")
    bias = torch.randn(128, device="cuda") if with_bias else None
    positions = torch.arange(tokens, device="cuda", dtype=index_dtype)
    trig = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    locations = (torch.arange(tokens, device="cuda", dtype=index_dtype) * 67 + 63) % 2048
    scratch = torch.full((32, 8448), 37, device="cuda", dtype=torch.uint8)
    owner = scratch.clone()

    def run():
        op.indexer_k_norm_rope_hadamard_store(
            key, weight, bias, positions, trig, locations, scratch, owner,
            neox, 1e-6, 128 ** -0.5, 1e-6,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        key.copy_(torch.randn_like(key))
        positions.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_scale = reference(key, weight, bias, positions.long(), trig, neox)
        pages, offsets = locations.long() // 64, locations.long() % 64
        actual = scratch[:, :8192].view(torch.int8).view(32, 64, 128)[pages, offsets]
        scale = scratch[:, 8192:].view(torch.float32)[pages, offsets]
        assert (actual.short() - expected.short()).abs().max() <= 1
        torch.testing.assert_close(scale, expected_scale, rtol=0.008, atol=1e-9)
        assert torch.equal(scratch, owner)
        touched = torch.zeros_like(scratch, dtype=torch.bool)
        for page, offset in zip(pages.cpu().tolist(), offsets.cpu().tolist()):
            touched[page, offset * 128:(offset + 1) * 128] = True
            touched[page, 8192 + offset * 4:8196 + offset * 4] = True
        assert torch.all(scratch[~touched] == 37)


def test_empty_and_padding_leave_cache_untouched():
    for tokens in [0, 10, 15, 20]:
        key = torch.randn(tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(128, device="cuda")
        positions = torch.full((tokens,), -100, device="cuda", dtype=torch.int64)
        locations = torch.full((tokens,), -1, device="cuda", dtype=torch.int64)
        trig = torch.zeros(16, 64, device="cuda", dtype=torch.bfloat16)
        cache = torch.full((4, 8448), 37, device="cuda", dtype=torch.uint8)
        op.indexer_k_norm_rope_hadamard_store(
            key, weight, None, positions, trig, locations, cache, cache,
            False, 1e-6, 128 ** -0.5, 1e-6,
        )
        torch.cuda.synchronize()
        assert torch.all(cache == 37)
