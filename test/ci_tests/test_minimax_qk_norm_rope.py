# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 intermediate oracle and changed-input graph coverage."""

import pytest
import torch

from lightop.attention import minimax_qk_norm_rope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def make(
    m, rotary=64, neox=True, strided=False, kind="random", cos_dtype=torch.float32
):
    width = 9856
    pack = torch.randn(
        (m, width * (2 if strided else 1)), device="cuda", dtype=torch.bfloat16
    )
    if strided:
        pack = pack[:, ::2]
    xs = [pack[:, :8192], pack[:, 8192:8704], pack[:, 9216:9728], pack[:, 9728:9856]]
    ws = [torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    if kind == "zero":
        for x in xs:
            x.zero_()
    if kind == "large":
        for x in xs:
            x.mul_(1000)
    if kind == "minus_one_weight":
        for w in ws:
            w.fill_(-1)
    positions = (torch.arange(m, device="cuda") * 7 + 131072).to(torch.int64)
    maximum = 131080 + max(m * 7, 1)
    angles = torch.randn((maximum, rotary // 2), device="cuda")
    cs = torch.cat([angles.cos(), angles.sin()], dim=1).to(cos_dtype)
    return xs + ws + [positions, cs, 1e-5, 128, rotary, neox]


def reference(a):
    out = []
    rotary = a[12]
    half = rotary // 2
    neox = a[13]
    cos = a[9][a[8], :half].double().unsqueeze(1)
    sin = a[9][a[8], half:].double().unsqueeze(1)
    for x, w in zip(a[:4], a[4:8]):
        x = x.reshape(x.shape[0], x.shape[1] // 128, 128).double()
        z = (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + a[10])
            * (1 + w.double())
        )
        z = z.to(torch.bfloat16).double()
        r = z[..., :rotary]
        if neox:
            partner = torch.cat([-r[..., half:], r[..., :half]], dim=-1)
            c = torch.cat([cos, cos], -1)
            s = torch.cat([sin, sin], -1)
        else:
            partner = torch.stack([-r[..., 1::2], r[..., ::2]], dim=-1).flatten(-2)
            c = cos.repeat_interleave(2, -1)
            s = sin.repeat_interleave(2, -1)
        y = (
            torch.cat([r * c + partner * s, z[..., rotary:]], -1)
            .to(torch.bfloat16)
            .flatten(1)
        )
        out.append(y)
    return out


def assert_oracle(a, actual):
    for got, expected in zip(actual, reference(a)):
        torch.testing.assert_close(got, expected, rtol=0.016, atol=0.015625)
        relative_rms = (
            (got.float() - expected.float()).square().mean()
            / (expected.float().square().mean() + 1e-20)
        ).sqrt()
        if got.numel():
            assert relative_rms.item() <= 0.003


@pytest.mark.parametrize(
    "rows,rotary,neox,strided,kind",
    [
        (0, 64, True, False, "random"),
        (17, 64, True, False, "random"),
        (31, 64, False, True, "random"),
        (17, 128, True, False, "random"),
        (17, 0, True, False, "random"),
        (17, 64, True, False, "zero"),
        (17, 64, True, False, "minus_one_weight"),
        (2048, 64, True, False, "random"),
    ],
)
def test_qk_bf16_rounding(rows, rotary, neox, strided, kind):
    torch.manual_seed(42)
    args = make(rows, rotary, neox, strided, kind)
    assert_oracle(args, minimax_qk_norm_rope(*args))


def test_qk_graph_changed_input():
    torch.manual_seed(73)
    args = make(17, 64)
    minimax_qk_norm_rope(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = minimax_qk_norm_rope(*args)
    for _ in range(3):
        for x in args[:4]:
            x.copy_(torch.randn_like(x))
        for weight in args[4:8]:
            weight.copy_(torch.randn_like(weight) * 0.1)
        args[8].add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert_oracle(args, actual)


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("rotary", [128, 64])
def test_qk_unaligned_cos_cache_falls_back(cos_dtype, neox, rotary):
    torch.manual_seed(126)
    args = make(17, rotary=rotary, neox=neox, cos_dtype=cos_dtype)
    original = args[9]
    storage = torch.empty(
        original.numel() + 1, device=original.device, dtype=original.dtype
    )
    args[9] = storage[1:].view_as(original)
    args[9].copy_(original)
    assert args[9].is_contiguous()
    assert args[9].data_ptr() % 16 != 0

    actual = minimax_qk_norm_rope(*args)
    assert_oracle(args, actual)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = minimax_qk_norm_rope(*args)
    for _ in range(2):
        # Keep the offset pointer while changing the live cache contents.
        args[9].neg_()
        args[8].add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert_oracle(args, actual)
