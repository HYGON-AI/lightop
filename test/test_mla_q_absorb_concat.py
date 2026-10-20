# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Q-absorb + Q RoPE + concat: exact rotary rounding and dynamic replay."""
import pytest
import torch
from lightop import op


def require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":")[0] != "gfx936":
        pytest.skip("validated on gfx936")


def rotate(x, cache, positions, neox):
    cos, sin = cache[positions].split(32, dim=-1)
    if neox:
        cos, sin = cos.repeat(1, 2), sin.repeat(1, 2)
        partner = torch.cat((-x[..., 32:], x[..., :32]), dim=-1)
    else:
        cos, sin = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
        partner = torch.stack((-x[..., 1::2], x[..., 0::2]), -1).flatten(-2)
    return x * cos[:, None, :] + partner * sin[:, None, :]


def reference(x, weight, rope, key, positions, cache, neox):
    nope = torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
    return torch.cat((nope, rotate(rope, cache, positions, neox)), dim=-1), rotate(key, cache, positions, neox)


def extras(rows):
    key = torch.randn(rows, 576, device="cuda", dtype=torch.bfloat16)[:, None, 512:]
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 13 + 7
    cache = torch.randn(1024, 64, device="cuda", dtype=torch.bfloat16)
    return key, positions, cache


@pytest.mark.parametrize("rows", [10, 15, 20])
@pytest.mark.parametrize("x_head_stride", [128, 192])
@pytest.mark.parametrize("rope_head_stride", [64, 192])
@pytest.mark.parametrize("kind", ["random", "zero", "ones", "one_hot", "small", "large"])
@pytest.mark.parametrize("neox", [False, True])
def test_values(rows, x_head_stride, rope_head_stride, kind, neox):
    require_hcu()
    torch.manual_seed(1933 + rows)
    x = torch.randn(rows, 128, x_head_stride, device="cuda", dtype=torch.bfloat16)[..., :128]
    weight = torch.randn(128, 512, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    rope = torch.randn(rows, 128, rope_head_stride, device="cuda", dtype=torch.bfloat16)[..., :64]
    if kind == "zero":
        x.zero_()
    elif kind == "ones":
        x.fill_(1)
        weight.fill_(1)
    elif kind == "one_hot":
        x.zero_()
        x[..., 107] = 1
    elif kind == "small":
        x.mul_(.0001)
        weight.mul_(.0001)
    elif kind == "large":
        x.mul_(128)
    key, positions, cache = extras(rows)
    originals = [t.clone() for t in (x, weight, rope, key, positions, cache)]
    y = op.mla_q_absorb_rope_concat(x, weight, rope, positions, cache, neox)
    expected, expected_key = reference(x, weight, rope, key, positions, cache, neox)
    assert y.shape == (rows, 128, 576) and y.is_contiguous()
    torch.testing.assert_close(y[..., :512], expected[..., :512], rtol=.008, atol=.00390625)
    assert torch.equal(y[..., 512:].view(torch.int16), expected[..., 512:].view(torch.int16))
    for value, original in zip((x, weight, rope, key, positions, cache), originals):
        assert torch.equal(value, original)
    if kind == "one_hot":
        torch.testing.assert_close(y[..., :512], weight[:, 107, :].expand(rows, -1, -1), rtol=0, atol=0)
    elif kind == "ones":
        assert torch.all(y[..., :512] == 128)


@pytest.mark.parametrize("rows", [10, 15, 20])
@pytest.mark.parametrize("neox", [False, True])
def test_shared_q_storage_and_graph_mutation(rows, neox):
    require_hcu()
    raw = torch.randn(rows, 128, 192, device="cuda", dtype=torch.bfloat16)
    x, rope = raw[..., :128], raw[..., 128:]
    weight = torch.randn(128, 512, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    key, positions, cache = extras(rows)
    for _ in range(3):
        op.mla_q_absorb_rope_concat(x, weight, rope, positions, cache, neox)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = op.mla_q_absorb_rope_concat(x, weight, rope, positions, cache, neox)
    for _ in range(5):
        raw.copy_(torch.randn_like(raw))
        weight.copy_(torch.randn_like(weight))
        key.copy_(torch.randn_like(key))
        cache.copy_(torch.randn_like(cache))
        positions.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_key = reference(x, weight, rope, key, positions, cache, neox)
        torch.testing.assert_close(y[..., :512], expected[..., :512], rtol=.008, atol=.00390625)
        assert torch.equal(y[..., 512:].view(torch.int16), expected[..., 512:].view(torch.int16))


def test_invalid_inputs_are_rejected():
    require_hcu()
    x = torch.empty(20, 128, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(128, 512, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    rope = torch.empty(20, 128, 64, device="cuda", dtype=torch.bfloat16)
    key, positions, cache = extras(20)
    for xx, ww, rr in [
        (x[:5], weight, rope[:5]),
        (x.float(), weight.float(), rope.float()),
        (x, weight.contiguous(), rope),
        (x, weight, rope.cpu()),
        (x, weight, rope[..., :32]),
    ]:
        with pytest.raises(RuntimeError):
            op.mla_q_absorb_rope_concat(xx, ww, rr, positions, cache, True)
    for pp, cc in [(positions.int(), cache), (positions.cpu(), cache), (positions, cache.float()), (positions, cache[:, :32])]:
        with pytest.raises(RuntimeError):
            op.mla_q_absorb_rope_concat(x, weight, rope, pp, cc, True)
