# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP8 transform parity, source preservation, layouts and changing-input graphs."""

import pytest
import torch

from lightop.quant import per_token_quant_fp8
from lightop.quant.transforms import (
    is_transform_quant_available,
    rope_per_token_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_transform_quant_available(),
    reason="native transform FP8 extension required",
)


def quant(x):
    if not x.numel():
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((*x.shape[:-1], 1), device=x.device),
        )
    return per_token_quant_fp8(x.contiguous())


def reference(x, frequencies, positions, groups, rope_dim, inverse):
    y = x.float().contiguous()
    if y.numel():
        freq = torch.view_as_real(frequencies[positions.long()])[:, None]
        real = y[..., -rope_dim::2]
        imag = y[..., -rope_dim + 1 :: 2]
        fr, fi = freq[..., 0], freq[..., 1]
        ri, ii = real * fi, imag * fi
        # Keep the FP32 FMA rounding and signed-zero result before rounding
        # through BF16/FP16. Triton's fast-math reference may erase -0.0.
        out_real = torch.addcmul(ii if inverse else -ii, real, fr)
        out_imag = torch.addcmul(-ri if inverse else ri, imag, fr)
        y[..., -rope_dim::2] = out_real
        y[..., -rope_dim + 1 :: 2] = out_imag
    return quant(
        y.to(x.dtype).reshape(x.shape[0], groups, x.shape[1] // groups * x.shape[2])
    )


def equal(actual, expected):
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,groups,rd",
    [
        ((0, 64, 512), 8, 64),
        ((1, 64, 512), 8, 64),
        ((7, 6, 128), 3, 32),
        ((40, 64, 512), 8, 64),
        ((48, 64, 512), 8, 64),
        ((96, 64, 512), 8, 64),
        ((128, 64, 512), 8, 64),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("major", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_rope_fp8_exact(dtype, shape, groups, rd, inverse, major, position_dtype):
    t, h, d = shape
    storage = torch.randn((t, h + 2, d + 8), device="cuda", dtype=dtype)
    x = storage[:, :h, :d]
    before = storage.clone()
    f = torch.polar(
        torch.ones(66000, rd // 2, device="cuda"),
        torch.randn(66000, rd // 2, device="cuda"),
    )
    p = torch.arange(t, device="cuda", dtype=position_dtype) * 3 + 65536
    expected = reference(x, f, p, groups, rd, inverse)
    if major:
        expected = tuple(v.transpose(0, 1).contiguous() for v in expected)
    for threads in (None, 64, 128, 256, 512, 1024):
        equal(
            rope_per_token_quant(
                x,
                f,
                p,
                groups,
                rd,
                inverse,
                dtype=torch.float8_e4m3fn,
                group_major=major,
                threads=threads,
            ),
            expected,
        )
    assert torch.equal(storage, before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [0, 1, 8, 40, 48])
@pytest.mark.parametrize("major", [False, True])
def test_strided_fp8_exact(dtype, rows, major):
    base = torch.randn(
        (8, rows, 1032) if major else (rows, 8, 1032), device="cuda", dtype=dtype
    )
    x = base[..., :1024].transpose(0, 1) if major else base[..., :1024]
    if rows:
        x[0].zero_()
    before = base.clone()
    expected = quant(x.contiguous().reshape(rows, 8192))
    for threads in (128, 256, 512, 1024):
        equal(per_token_quant_fp8(x, quant_ndim=2, threads=threads), expected)
    assert torch.equal(base, before)


@pytest.mark.parametrize("rows", [8, 40, 48, 63, 64, 65, 96, 127, 128, 129])
@pytest.mark.parametrize("major", [False, True])
def test_rope_fp8_graph_changed_inputs(rows, major, monkeypatch):
    import lightop.quant.transforms as transforms
    from lightop.quant.transforms import _rope_device_arch

    calls = []
    native = transforms._rope

    def audited(*args):
        calls.append(args[-2])
        return native(*args)

    monkeypatch.setattr(transforms, "_rope", audited)
    x = torch.randn((rows, 64, 512), device="cuda", dtype=torch.bfloat16)
    f = torch.polar(
        torch.ones(66000, 32, device="cuda"), torch.randn(66000, 32, device="cuda")
    )
    p = torch.arange(rows, device="cuda", dtype=torch.int64) + 65536
    rope_per_token_quant(
        x, f, p, 8, 64, True, dtype=torch.float8_e4m3fn, group_major=major
    )
    expected_threads = (
        64
        if _rope_device_arch(x.device.index) == "gfx938" and major and 64 <= rows <= 128
        else 256
    )
    assert calls[-1] == expected_threads
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rope_per_token_quant(
            x, f, p, 8, 64, True, dtype=torch.float8_e4m3fn, group_major=major
        )
    assert calls[-1] == expected_threads
    pointers = [v.data_ptr() for v in actual]
    for phase in range(4):
        x.normal_()
        p.add_(7)
        x[phase].zero_()
        before = x.clone()
        expected = reference(x, f, p, 8, 64, True)
        if major:
            expected = tuple(v.transpose(0, 1).contiguous() for v in expected)
        graph.replay()
        equal(actual, expected)
        assert torch.equal(x, before)
        assert pointers == [v.data_ptr() for v in actual]
