# SPDX-License-Identifier: Apache-2.0
"""MiniMax rotary64 unit-scale E4M3 writeback and graph metadata replay."""

import pytest
import torch

from lightop import op
from lightop.attention import minimax_qk_norm_rope_fp8_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a HIP device")
def test_minimax_qk_fp8_cache_graph():
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx938") or not hasattr(op, "minimax_qk_norm_rope_fp8_cache"):
        pytest.skip("requires the gfx938 MiniMax FP8 build")

    qkv = torch.ones((2, 9856 + 128), device="cuda", dtype=torch.bfloat16)[:, :9856]
    qkv[:, 8704:9216].fill_(512)
    weights = [torch.zeros(128, device="cuda", dtype=qkv.dtype) for _ in range(4)]
    positions = torch.zeros(2, device="cuda", dtype=torch.int64)
    cos_sin = torch.zeros((1, 64), device="cuda")
    cos_sin[:, :32].fill_(1)
    locations = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0, 0, 5, 6], device="cuda", dtype=torch.int32)
    key = torch.zeros((8, 4, 128), device="cuda", dtype=torch.uint8)
    value = torch.zeros_like(key)
    index = torch.zeros((8, 1, 128), device="cuda", dtype=torch.uint8)
    fp8 = torch.float8_e4m3fn

    def launch():
        return minimax_qk_norm_rope_fp8_cache(
            qkv, weights, positions, cos_sin, 1e-6, locations, mapping,
            key.view(fp8), value.view(fp8), index.view(fp8),
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, iq = launch()
    for cache in (key, value, index):
        cache.zero_()
    locations[0] = 3
    graph.replay()

    assert tuple(q.shape) == (2, 8192)
    assert tuple(iq.shape) == (2, 512)
    for query in (q, iq):
        assert query.dtype == fp8
        assert torch.all(query.view(torch.uint8) == 0x38)
    for cache, slot, code in ((key, 6, 0x38), (value, 6, 0x7E), (index, 3, 0x38)):
        expected = torch.zeros_like(cache)
        expected[slot].fill_(code)
        assert torch.equal(cache, expected)


def _require_gfx938():
    if not torch.cuda.is_available() or not getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", ""
    ).startswith("gfx938"):
        pytest.skip("requires gfx938")


def _projection_inputs(rows):
    # Rank-one INT8 inputs give an independent closed-form INT32 GEMM oracle.
    lhs = torch.arange(rows, device="cuda") % 7 - 3
    rhs = torch.arange(9856, device="cuda") % 11 - 5
    left = lhs.to(torch.int8)[:, None].expand(-1, 6144).contiguous()
    right = rhs.to(torch.int8)[:, None].expand(-1, 6144).contiguous()
    left_scale = ((lhs.abs() + 1).float() / 512).contiguous()
    right_scale = ((rhs.abs() + 1).float() / 256).contiguous()
    return left, left_scale, right, right_scale, (lhs[:, None] * rhs * 6144).float()


@pytest.mark.parametrize("rows", [64, 80, 96, 128, 144])
def test_minimax_installed_transposed_qkv_to_fp8_cache(rows):
    from lightop.attention import minimax31_project_qkv

    _require_gfx938()
    left, ls, right, rs, acc = _projection_inputs(rows)
    qkv = minimax31_project_qkv(left, ls, right, rs)
    assert qkv.shape == (rows, 9856)
    assert qkv.stride() == (1, rows)
    reference = (acc * ls[:, None] * rs).to(torch.bfloat16)
    torch.testing.assert_close(qkv, reference, atol=0, rtol=0)

    weights = [torch.linspace(0, 0.5, 128, device="cuda").to(qkv.dtype)] * 4
    positions = torch.arange(rows, device="cuda", dtype=torch.long)
    angles = torch.arange(rows * 32, device="cuda").view(rows, 32) * 0.001
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    locations = positions + 1
    locations[::7] = 0
    mapping = torch.arange(rows, -1, -1, device="cuda", dtype=torch.int32)

    def output(value):
        cache = tuple(
            torch.zeros((rows + 1, heads, 128), device="cuda", dtype=torch.uint8)
            for heads in (4, 4, 1)
        )
        query = minimax_qk_norm_rope_fp8_cache(
            value, weights, positions, cos_sin, 1e-6, locations, mapping,
            *(c.view(torch.float8_e4m3fn) for c in cache),
        )
        return tuple(q.view(torch.uint8) for q in query) + cache

    for actual, ref in zip(output(qkv), output(reference)):
        assert torch.equal(actual, ref)


def test_minimax_shared_plan_concurrent_scale_submission():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from lightop.attention import minimax31_project_qkv
    from lightop.minimax31_decode_lt import warmup

    _require_gfx938()
    rows = 80
    left, ls, right, rs, acc = _projection_inputs(rows)
    stream = torch.cuda.Stream()
    ready = Barrier(2, timeout=30)
    with torch.cuda.stream(stream):
        warmup(None, left.device, rows, 9856, 6144, 26690)
    inputs = [(ls * a, rs * b) for a, b in ((1, 4), (2, 8))]
    torch.cuda.synchronize()

    def submit(scales):
        a, b = scales
        with torch.cuda.stream(stream):
            outputs = []
            for _ in range(12):
                ready.wait()
                outputs.append(minimax31_project_qkv(left, a, right, b))
            return outputs

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(submit, inputs))
    stream.synchronize()
    for stream_outputs, (a, b) in zip(outputs, inputs):
        expected = (acc * a[:, None] * b).to(torch.bfloat16)
        for result in stream_outputs:
            torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_minimax_lt_validates_k_before_loading_native_library(monkeypatch):
    from lightop import minimax31_decode_lt as lt

    def unexpected(*args):
        pytest.fail("invalid operands reached Lt plan selection")

    monkeypatch.setattr(lt, "_plan", unexpected)
    a = torch.empty((2, 4), dtype=torch.int8)
    b = torch.empty((3, 2), dtype=torch.int8)
    with pytest.raises(ValueError, match="same K"):
        lt.project(None, a, torch.ones(2), b, torch.ones(3), 0)
