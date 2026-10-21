# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone installed-LightOp correctness coverage for K3 early EP steps.

Run in fresh processes with (PACKED, TUNED) = (0, 0), (1, 0), (1, 1).
The variable names are LIGHTOP_SITU_EP_PACKED and LIGHTOP_SITU_EP_TUNED.
The environment switch is cached at first invocation in the C++ extension.
"""

import lightop
import pytest
import torch


@pytest.mark.parametrize("hint", [-1, 1, 4, 5, 19, 20])
@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "uniform",
        "first_hot",
        "last_hot",
        "interior_empty",
        "dense",
        "cross_up",
        "cross_down",
    ],
)
def test_situ_ep_schedule(hint, kind):
    torch.manual_seed(17)
    e, m, h = 28, 129, 3072
    x = torch.randn((e, m, 2 * h), device="cuda", dtype=torch.bfloat16)
    if kind == "empty":
        values = [0] * e
    elif kind == "uniform":
        values = [3] * e
    elif kind == "first_hot":
        values = [m] + [0] * 27
    elif kind == "last_hot":
        values = [0] * 27 + [m]
    elif kind in ("dense", "cross_down"):
        values = [19] * e
    elif kind == "cross_up":
        values = [3] * e
    else:
        values = [i * 3 if i % 3 else 0 for i in range(e)]
    replay_values = values[::-1]
    if kind == "cross_up":
        replay_values = [19] * e
    elif kind == "cross_down":
        replay_values = [3] * e
    counts = torch.tensor(values, device="cuda", dtype=torch.int32)
    q = torch.full((e, m, h), -91, device="cuda", dtype=torch.int8)
    scales = torch.ones((e, m, 1), device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lightop.fuse_situ_mul_quant_ep(
            x, counts, 4.0, 25.0, expect_m=hint, output=q, scales=scales
        )
    # Alter counts after capture; replay must not cache the routing metadata.
    q.fill_(-91)
    scales.fill_(1)
    counts.copy_(torch.tensor(replay_values, device="cuda", dtype=torch.int32))
    graph.replay()
    for expert, n in enumerate(replay_values):
        assert torch.equal(q[expert, n:], torch.full_like(q[expert, n:], -91))
        assert torch.equal(scales[expert, n:], torch.ones_like(scales[expert, n:]))
        if not n:
            continue
        g, u = x[expert, :n].float().chunk(2, -1)
        y = (
            100.0
            * (2 * torch.sigmoid(g * 0.5) - 1)
            * torch.sigmoid(g)
            * (2 * torch.sigmoid(u * 0.08) - 1)
        )
        a = y.abs().amax(-1, keepdim=True)
        s = torch.where(a > 0, a / 127.0, torch.ones_like(a))
        ref = torch.round(y / s).clamp(-128, 127).to(torch.int8)
        assert (q[expert, :n].int() - ref.int()).abs().max().item() <= 1
        assert torch.allclose(scales[expert, :n], s, rtol=1e-4, atol=1e-7)
