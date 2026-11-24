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
