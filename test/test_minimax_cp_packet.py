# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent raw-byte CP packet/cache oracles; no framework baseline dependency."""

import pytest
import torch

from lightop import op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("rows,scale", [(0, 1.0), (17, 1.0), (2048, 0.37)])
def test_packet_bytes_and_immutable_inputs(rows, scale):
    torch.manual_seed(42)
    source = torch.randn((rows, 1152), device="cuda", dtype=torch.bfloat16) * 20
    before = source.clone()
    k, v, ik = source[:, :512], source[:, 512:1024], source[:, 1024:]
    actual = op.minimax_cp_fp8_pack(k, v, ik, scale, 1.7, True, True)
    expected = torch.cat(
        [
            (k / scale)
            .to(torch.bfloat16)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8),
            (v / 1.7)
            .to(torch.bfloat16)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8),
            ik.view(torch.uint8),
        ],
        dim=1,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(source, before)


def test_scatter_graph_changed_packet_order_and_slots():
    torch.manual_seed(83)
    rows, capacity = 127, 200
    packet = torch.randint(0, 256, (rows, 1280), device="cuda", dtype=torch.uint8)
    order = torch.randperm(rows, device="cuda")
    # Match allocator output: one writer per slot, including the slot-zero case.
    slots = (torch.randperm(capacity - 1, device="cuda")[:rows] + 1).contiguous()
    slots[0] = 0
    backs = [
        torch.empty((capacity, n + 16), device="cuda", dtype=torch.uint8)
        for n in (512, 512, 256)
    ]
    caches = [b[:, :n] for b, n in zip(backs, (512, 512, 256))]
    op.minimax_cp_cache_scatter(packet, order, slots, *caches)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.minimax_cp_cache_scatter(packet, order, slots, *caches)
    for replay in range(3):
        packet.random_(0, 256)
        order.copy_(order.roll(1))
        slots.copy_(slots.roll(2))
        for b in backs:
            b.fill_(31 + replay)
        expected = [b.clone() for b in backs]
        ordered = packet.index_select(0, order)
        for b, start, width in zip(expected, (0, 512, 1024), (512, 512, 256)):
            valid = slots > 0 if start < 1024 else slots >= 0
            b[slots[valid], :width] = ordered[valid, start : start + width]
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(backs, expected):
            assert torch.equal(actual, reference)
