# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native transform+INT8 parity, pitched layouts, and changing-input graphs."""
# ruff: noqa: E402 -- Check Triton before importing LightOp, which also imports it.

import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton reference kernels required")
import triton.language as tl
from triton.language.extra import libdevice

from lightop.quant.transforms import (
    _rope,
    is_transform_quant_available,
    rope_per_token_quant,
    scale_per_token_quant_int8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_transform_quant_available(),
    reason="native transform quant extension required",
)


@triton.jit
def _reference_rope(
    X,
    F,
    P,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    INVERSE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, B) * 2
    token = row // H
    head = row % H
    base = token * ST + head * SH + D - RD
    a = tl.load(X + base + d, d < RD, 0).to(tl.float32)
    b = tl.load(X + base + d + 1, d < RD, 0).to(tl.float32)
    pos = tl.load(P + token)
    c = tl.load(F + pos * RD + d, d < RD, 0)
    s = tl.load(F + pos * RD + d + 1, d < RD, 0)
    bi = b * s
    ai = a * s
    if INVERSE:
        ar = tl.fma(a, c, bi)
        br = tl.fma(b, c, -ai)
    else:
        ar = tl.fma(a, c, -bi)
        br = tl.fma(b, c, ai)
    tl.store(X + base + d, ar, d < RD)
    tl.store(X + base + d + 1, br, d < RD)


@triton.jit
def _reference_quant(X, Q, S, K: tl.constexpr, AWAY: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    v = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    a = tl.maximum(tl.max(tl.abs(v)), 1e-10)
    v = v * (127 / a)
    if AWAY:
        v = libdevice.round(v)
    else:
        v = libdevice.nearbyint(v)
    tl.store(Q + row * K + col, v.to(tl.int8), col < K)
    tl.store(S + row, a / 127)


def quant_ref(x, away):
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    if x.numel():
        k = x.shape[-1]
        block = triton.next_power_of_2(k)
        _reference_quant[(x.numel() // k,)](
            x,
            q,
            s,
            k,
            away,
            block,
            num_warps=min(max(block // 256, 1), 8),
        )
    return q, s


def rope_ref(x, f, p, groups, rd, inverse):
    # Keep the FP32 FMA result before converting to the activation dtype.
    # HIP may otherwise lower FMA+FP16 store to v_fma_mixlo_f16 and omit
    # this intermediate rounding.
    y = x.float().contiguous()
    if y.numel():
        _reference_rope[(y.shape[0] * y.shape[1],)](
            y,
            torch.view_as_real(f),
            p,
            *y.shape,
            rd,
            y.stride(0),
            y.stride(1),
            inverse,
            triton.next_power_of_2(rd // 2),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return quant_ref(
        y.to(x.dtype).reshape(y.shape[0], groups, x.shape[1] // groups * x.shape[2]),
        True,
    )


def equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a, b), (
            a.shape,
            int((a != b).sum()),
            float((a.float() - b.float()).abs().max()),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,groups,rd,pad",
    [
        ((0, 64, 512), 8, 64, 128),
        ((1, 4, 96), 2, 48, 0),
        ((7, 6, 128), 3, 32, 16),
        ((40, 64, 512), 8, 64, 128),
        ((48, 64, 512), 8, 64, 128),
        ((96, 64, 512), 8, 64, 64),
        ((128, 64, 512), 8, 64, 64),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_quant_exact(dtype, shape, groups, rd, pad, inverse):
    t, h, d = shape
    torch.manual_seed(3100 + t)
    # Both head and token padding are deliberate.
    storage = torch.randn(t, h + 2, d + 8, device="cuda", dtype=dtype)
    x = storage[:, :h, :d]
    before = storage.clone()
    f = torch.polar(
        torch.ones(256, rd // 2, device="cuda"),
        torch.randn(256, rd // 2, device="cuda"),
    )
    p = (torch.arange(t, device="cuda", dtype=torch.int32) * 3) % 256
    pitch = h // groups * d + pad
    result = rope_per_token_quant(x, f, p, groups, rd, inverse, pitch)
    equal(result, rope_ref(x, f, p, groups, rd, inverse))
    assert torch.equal(storage, before)
    assert result[0].stride(1) == pitch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(0, 8, 1024), (1, 2, 33), (7, 3, 128), (40, 8, 1024), (48, 8, 1024)]
)
@pytest.mark.parametrize("group_major", [False, True])
@pytest.mark.parametrize("materialize_output", [False, True])
def test_scale_quant_exact(dtype, shape, group_major, materialize_output):
    torch.manual_seed(5400 + shape[0])
    t, g, r = shape
    raw = torch.randint(-100000, 100000, (t, g, r), device="cuda", dtype=torch.int32)
    xs = torch.rand(t, g, 1, device="cuda") * 0.02
    ws = torch.rand(g * r, 1, device="cuda") * 0.03
    ref = ((raw.float() * xs) * ws.view(1, g, r)).to(dtype)
    raw_arg = raw.transpose(0, 1).contiguous() if group_major else raw
    before = raw_arg.clone()
    result = scale_per_token_quant_int8(
        raw_arg,
        xs,
        ws,
        output_dtype=dtype,
        group_major=group_major,
        materialize_output=materialize_output,
    )
    if materialize_output:
        equal(result, (ref, *quant_ref(ref.flatten(1), False)))
    else:
        assert result[0] is None
        equal(result[1:], quant_ref(ref.flatten(1), False))
    assert torch.equal(raw_arg, before)


def test_rounding_rules_and_zero_rows():
    x = torch.zeros(2, 1, 16, device="cuda", dtype=torch.bfloat16)
    values = torch.tensor([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda")
    x[0, 0, :7] = values
    f = torch.ones(2, 4, device="cuda", dtype=torch.complex64)
    p = torch.arange(2, device="cuda", dtype=torch.int64)
    q, s = rope_per_token_quant(x, f, p, rope_dim=8)
    assert q[0, 0, :7].tolist() == [127, 1, 2, 3, -1, -2, -3]
    assert not q[1].any()
    raw = torch.ones(2, 1, 16, device="cuda", dtype=torch.int32)
    ws = x[0].float().view(16, 1)
    xs = torch.tensor([1, 0], device="cuda", dtype=torch.float32).view(2, 1, 1)
    out, q, s = scale_per_token_quant_int8(raw, xs, ws)
    assert q[0, :7].tolist() == [127, 0, 2, 2, 0, -2, -2]
    assert not q[1].any()


@pytest.mark.parametrize("threads", [64, 128, 256, 512, 1024])
def test_pitched_output_and_graph(threads):
    x = torch.randn(48, 64, 512, device="cuda", dtype=torch.bfloat16)
    f = torch.polar(
        torch.ones(256, 32, device="cuda"), torch.randn(256, 32, device="cuda")
    )
    p = torch.arange(48, device="cuda", dtype=torch.int64)
    storage = torch.full((48, 8, 4224), 77, device="cuda", dtype=torch.int8)
    q = storage[..., :4096]
    s = torch.empty(48, 8, 1, device="cuda")

    def run():
        _rope(x, f, p, q, s, 8, 64, True, threads, False)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in range(3):
        x.normal_()
        p.add_(17).remainder_(256)
        graph.replay()
        equal((q, s), rope_ref(x, f, p, 8, 64, True))
        assert torch.all(storage[..., 4096:] == 77)


@pytest.mark.parametrize("splits", [1, 2, 4, 8])
@pytest.mark.parametrize("threads", [128, 256, 512, 1024])
@pytest.mark.parametrize("materialize_output", [False, True])
def test_scale_graph(splits, threads, materialize_output):
    raw = torch.randint(
        -100000, 100000, (40, 8, 1024), device="cuda", dtype=torch.int32
    )
    xs = torch.rand(40, 8, 1, device="cuda") * 0.02
    ws = torch.rand(8192, 1, device="cuda") * 0.03
    scale_per_token_quant_int8(
        raw,
        xs,
        ws,
        splits=splits,
        threads=threads,
        materialize_output=materialize_output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = scale_per_token_quant_int8(
            raw,
            xs,
            ws,
            splits=splits,
            threads=threads,
            materialize_output=materialize_output,
        )
    for seed in range(3):
        raw.random_(-100000, 100000)
        xs.uniform_(0, 0.02)
        graph.replay()
        ref = ((raw.float() * xs) * ws.view(1, 8, 1024)).bfloat16()
        if materialize_output:
            equal(result, (ref, *quant_ref(ref.flatten(1), False)))
        else:
            assert result[0] is None
            equal(result[1:], quant_ref(ref.flatten(1), False))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_unaligned_contiguous_input(dtype):
    shape = (40, 64, 512)
    storage = torch.randn(40 * 64 * 512 + 1, device="cuda", dtype=dtype)
    x = storage[1:].view(shape)
    before = storage.clone()
    f = torch.ones(256, 32, device="cuda", dtype=torch.complex64)
    p = torch.arange(40, device="cuda", dtype=torch.int64)
    equal(
        rope_per_token_quant(x, f, p, 8, 64, True, 4224),
        rope_ref(x, f, p, 8, 64, True),
    )
    assert torch.equal(storage, before)


def test_scale_unaligned_contiguous_input():
    raw = torch.randint(
        -100000, 100000, (40 * 8192 + 1,), device="cuda", dtype=torch.int32
    )[1:].view(40, 8, 1024)
    xs = torch.rand(40, 8, 1, device="cuda") * 0.02
    ws = torch.rand(8193, device="cuda")[1:].view(8192, 1) * 0.03
    # Keep the deliberately unaligned weight storage after multiplication.
    weight_storage = torch.empty(8193, device="cuda")
    weight_storage[1:].copy_(ws.flatten())
    ws = weight_storage[1:].view(8192, 1)
    ref = ((raw.float() * xs) * ws.view(1, 8, 1024)).bfloat16()
    equal(
        scale_per_token_quant_int8(raw, xs, ws, splits=1),
        (ref, *quant_ref(ref.flatten(1), False)),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_unaligned_frequency_storage(dtype):
    x = torch.randn(40, 64, 512, device="cuda", dtype=dtype)
    storage = torch.polar(
        torch.ones(256 * 32 + 1, device="cuda"),
        torch.randn(256 * 32 + 1, device="cuda"),
    )
    f = storage[1:].view(256, 32)
    p = torch.arange(40, device="cuda", dtype=torch.int64)
    equal(
        rope_per_token_quant(x, f, p, 8, 64, True, 4224),
        rope_ref(x, f, p, 8, 64, True),
    )


@pytest.mark.parametrize("rows", [40, 48, 86, 87, 90, 91, 96, 105, 106, 128])
def test_automatic_rope_threads_and_graph(rows, monkeypatch):
    import lightop.quant.transforms as transforms

    x = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
    f = torch.polar(
        torch.ones(66000, 32, device="cuda"), torch.randn(66000, 32, device="cuda")
    )
    p = torch.arange(rows, device="cuda", dtype=torch.int64) + 65536
    calls = []
    native = transforms._rope

    def audited(*args):
        calls.append(args[-2])
        return native(*args)

    monkeypatch.setattr(transforms, "_rope", audited)
    arch = transforms._rope_device_arch(x.device.index)
    expected_threads = 256
    if arch == "gfx936":
        if 87 <= rows <= 90:
            expected_threads = 128
        elif 91 <= rows <= 105:
            expected_threads = 64
    rope_per_token_quant(x, f, p, 8, 64, True, 4160)
    assert calls[-1] == expected_threads
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rope_per_token_quant(x, f, p, 8, 64, True, 4160)
    assert calls[-1] == expected_threads
    for phase in range(3):
        x.normal_()
        x[phase].zero_()
        p.add_(7)
        graph.replay()
        equal(actual, rope_ref(x, f, p, 8, 64, True))
