# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for masked, limited BF16 SwiGLU FP8 quantization."""

import pytest
import torch

from lightop.activation import fuse_silu_mul_fp8_quant_ep
from lightop import op


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _buffers(x, fp8type):
    dtype = torch.float8_e4m3fn if fp8type == 0 else torch.float8_e5m2
    q = torch.empty((*x.shape[:2], x.shape[2] // 2), dtype=dtype, device=x.device)
    q.view(torch.uint8).fill_(165)
    s = torch.full((*x.shape[:2], 1), -123.0, dtype=torch.float32, device=x.device)
    return q, s


def _run_into(x, q, s, counts, fp8type=0):
    op.fuse_silu_mul_fp8_quant_ep(x, q.view(torch.uint8), s, fp8type,
                                counts, None, 1, -1, 10.0)


def _check_against_unmasked(x, counts, q, s, fp8type):
    # The unmasked path retains the existing implementation and numerical
    # materialization boundaries. Compare valid rows exactly, not by tolerance.
    d = x.shape[-1] // 2
    reference_x = x
    if d % 8:
        # The legacy large-width path loads vectors of up to 32 elements.
        # Zero-pad to that alignment so it is a safe tail-case oracle.
        padded_d = (d + 31) // 32 * 32
        reference_x = torch.zeros((*x.shape[:2], 2 * padded_d), device=x.device, dtype=x.dtype)
        reference_x[..., :d].copy_(x[..., :d])
        reference_x[..., padded_d:padded_d+d].copy_(x[..., d:])
    expected_q, expected_s = fuse_silu_mul_fp8_quant_ep(reference_x, fp8type=fp8type, limit=10.0)
    for expert, count in enumerate(counts.tolist()):
        assert torch.equal(q[expert, :count].view(torch.uint8),
                           expected_q[expert, :count, :d].view(torch.uint8))
        assert torch.equal(s[expert, :count], expected_s[expert, :count])
        assert (q[expert, count:].view(torch.uint8) == 165).all()
        assert (s[expert, count:] == -123.0).all()


@pytest.mark.parametrize('capacity', [0, 1, 33, 129])
@pytest.mark.parametrize('width', [128, 1024, 2048, 2304, 3072, 4096, 4104, 8192, 16384, 32768])
@pytest.mark.parametrize('fp8type', [0, 1])
@pytest.mark.parametrize('distribution', ['random', 'zero', 'tiny', 'wide', 'clamp_boundary'])
def test_masked_matches_existing_quantization(capacity, width, fp8type, distribution):
    torch.manual_seed(20261019)
    x = torch.randn(4, capacity, 2 * width, dtype=torch.bfloat16, device='cuda')
    if distribution == 'zero':
        x.zero_()
    elif distribution == 'tiny':
        x.mul_(1e-7)
    elif distribution == 'wide':
        x.mul_(100)
    elif distribution == 'clamp_boundary':
        values = torch.tensor([-10.0625, -10., -9.9375, 0., 9.9375, 10., 10.0625, 100.],
                              dtype=x.dtype, device=x.device)
        x.copy_(values.repeat(x.numel() // 8).view_as(x))
    before = x.clone()
    counts = torch.tensor([0, min(1, capacity), max(0, capacity-1), capacity],
                          dtype=torch.int32, device='cuda')
    q, s = _buffers(x, fp8type)
    _run_into(x, q, s, counts, fp8type)
    _check_against_unmasked(x, counts, q, s, fp8type)
    assert torch.equal(x, before)


@pytest.mark.parametrize('width', [512, 2304, 8192])
def test_nondefault_stream_graph_updates_input_and_counts(width):
    x = torch.randn(4, 257, 2 * width, dtype=torch.bfloat16, device='cuda')
    counts = torch.zeros(4, dtype=torch.int32, device='cuda')
    q, s = _buffers(x, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_into(x, q, s, counts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run_into(x, q, s, counts)
        for n in [0, 1, 8, 16, 32, 128, 129, 257]:
            x.mul_(-0.75)
            counts.copy_(torch.tensor([0, 1 if n else 0, n//2, n], dtype=counts.dtype, device=x.device))
            q.view(torch.uint8).fill_(165)
            s.fill_(-123.0)
            graph.replay()
            _check_against_unmasked(x, counts, q, s, 0)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def test_empty_experts():
    x = torch.empty(0, 17, 4608, dtype=torch.bfloat16, device='cuda')
    counts = torch.empty(0, dtype=torch.int32, device='cuda')
    q, s = fuse_silu_mul_fp8_quant_ep(x, tokens_per_expert=counts, limit=10.0)
    assert q.shape == (0, 17, 2304)
    assert s.shape == (0, 17, 1)


@pytest.mark.parametrize('width', [1, 7, 127, 255, 511, 513, 1023, 1025, 2047, 2049, 2305, 4095])
@pytest.mark.parametrize('fp8type', [0, 1])
def test_unaligned_width_tails(width, fp8type):
    x = torch.randn(4, 17, width * 2, dtype=torch.bfloat16, device='cuda')
    counts = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device='cuda')
    q, s = _buffers(x, fp8type)
    _run_into(x, q, s, counts, fp8type)
    _check_against_unmasked(x, counts, q, s, fp8type)


def test_many_experts_skewed_routing():
    x = torch.randn(48, 2048, 4608, dtype=torch.bfloat16, device='cuda')
    counts = torch.tensor([0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 511, 1024, 2048] * 3,
                          dtype=torch.int32, device='cuda')
    q, s = _buffers(x, 0)
    _run_into(x, q, s, counts)
    _check_against_unmasked(x, counts, q, s, 0)
