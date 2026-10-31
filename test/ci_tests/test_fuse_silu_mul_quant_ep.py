# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the masked BF16 SwiGLU INT8 quantization path.

Mirrors test_fuse_silu_mul_fp8_quant_ep.py: the unmasked call keeps the previous
implementation, so every valid row of the masked launch must match it exactly
and the padding rows must stay untouched.
"""

import pytest
import torch

from lightop.activation import fuse_silu_mul_quant_ep
from lightop import op


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# Sentinels prove that a masked launch never writes a padding row.
Q_SENTINEL = 127
S_SENTINEL = -123.0


def _buffers(x):
    q = torch.empty((*x.shape[:2], x.shape[2] // 2), dtype=torch.int8, device=x.device)
    q.fill_(Q_SENTINEL)
    s = torch.full((*x.shape[:2], 1), S_SENTINEL, dtype=torch.float32, device=x.device)
    return q, s


def _run_into(x, q, s, counts):
    op.fuse_silu_mul_quant_ep(x, q, s, counts, None, 1, -1)


def _check_against_unmasked(x, counts, q, s):
    # The unmasked path retains the existing implementation and numerical
    # materialization boundaries. Compare valid rows exactly, not by tolerance.
    d = x.shape[-1] // 2
    reference_x = x
    if d % 32:
        # The legacy path loads vectors of up to 32 elements. Zero-pad to that
        # alignment so the oracle itself is a safe tail case.
        padded_d = (d + 31) // 32 * 32
        reference_x = torch.zeros((*x.shape[:2], 2 * padded_d), device=x.device, dtype=x.dtype)
        reference_x[..., :d].copy_(x[..., :d])
        reference_x[..., padded_d:padded_d + d].copy_(x[..., d:])
    expected_q, expected_s = fuse_silu_mul_quant_ep(reference_x)
    for expert, count in enumerate(counts.tolist()):
        assert torch.equal(q[expert, :count], expected_q[expert, :count, :d])
        assert torch.equal(s[expert, :count], expected_s[expert, :count])
        assert (q[expert, count:] == Q_SENTINEL).all()
        assert (s[expert, count:] == S_SENTINEL).all()


@pytest.mark.parametrize('capacity', [0, 1, 33, 129])
@pytest.mark.parametrize('width', [128, 1024, 2048, 2304, 3072, 4096, 4104, 8192])
@pytest.mark.parametrize('distribution', ['random', 'zero', 'tiny', 'wide'])
def test_masked_matches_existing_quantization(capacity, width, distribution):
    torch.manual_seed(20261029)
    x = torch.randn(4, capacity, 2 * width, dtype=torch.bfloat16, device='cuda')
    if distribution == 'zero':
        x.zero_()
    elif distribution == 'tiny':
        x.mul_(1e-7)
    elif distribution == 'wide':
        x.mul_(100)
    before = x.clone()
    counts = torch.tensor([0, min(1, capacity), max(0, capacity - 1), capacity],
                          dtype=torch.int32, device='cuda')
    q, s = _buffers(x)
    _run_into(x, q, s, counts)
    _check_against_unmasked(x, counts, q, s)
    assert torch.equal(x, before)


@pytest.mark.parametrize('width', [1, 7, 127, 255, 511, 513, 1023, 1025, 2047, 2049, 2305, 4095])
def test_unaligned_width_tails(width):
    x = torch.randn(4, 17, width * 2, dtype=torch.bfloat16, device='cuda')
    counts = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device='cuda')
    q, s = _buffers(x)
    _run_into(x, q, s, counts)
    _check_against_unmasked(x, counts, q, s)


@pytest.mark.parametrize('width', [512, 2303, 2304, 8192])
def test_nondefault_stream_graph_updates_input_and_counts(width):
    x = torch.randn(4, 257, 2 * width, dtype=torch.bfloat16, device='cuda')
    counts = torch.zeros(4, dtype=torch.int32, device='cuda')
    q, s = _buffers(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_into(x, q, s, counts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run_into(x, q, s, counts)
        for n in [0, 1, 8, 16, 32, 128, 129, 257]:
            x.mul_(-0.75)
            counts.copy_(torch.tensor([0, 1 if n else 0, n // 2, n], dtype=counts.dtype, device=x.device))
            q.fill_(Q_SENTINEL)
            s.fill_(S_SENTINEL)
            graph.replay()
            _check_against_unmasked(x, counts, q, s)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def test_empty_experts():
    x = torch.empty(0, 17, 4608, dtype=torch.bfloat16, device='cuda')
    counts = torch.empty(0, dtype=torch.int32, device='cuda')
    q, s = fuse_silu_mul_quant_ep(x, tokens_per_expert=counts)
    assert q.shape == (0, 17, 2304)
    assert s.shape == (0, 17, 1)


def test_many_experts_skewed_routing():
    x = torch.randn(48, 512, 4608, dtype=torch.bfloat16, device='cuda')
    counts = torch.tensor([0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 256, 384, 512] * 3,
                          dtype=torch.int32, device='cuda')
    q, s = _buffers(x)
    _run_into(x, q, s, counts)
    _check_against_unmasked(x, counts, q, s)


def _assert_valid_rows_match(x, counts, got_q, got_s, ref_q, ref_s):
    """Compare only the valid rows: the legacy kernels leave padding rows
    untouched, so a whole-tensor compare would read uninitialized memory."""
    d = x.shape[-1] // 2
    for expert, count in enumerate(counts.tolist()):
        assert torch.equal(got_q[expert, :count, :d], ref_q[expert, :count, :d])
        assert torch.equal(got_s[expert, :count], ref_s[expert, :count])


@pytest.mark.parametrize('d', [4104, 8192])
def test_over_limit_width_falls_back(d):
    """d > 4096 is outside the masked dispatch, so both calls take the legacy
    kernels and must still agree row for row.

    Widths here stay a multiple of 8: the legacy kernel's vector tail is only
    defined for aligned widths, which is why the unaligned case above compares
    against a padded oracle instead.
    """
    x = torch.randn(3, 17, 2 * d, dtype=torch.bfloat16, device='cuda')
    counts = torch.tensor([0, 5, 17], dtype=torch.int32, device='cuda')
    masked_q, masked_s = fuse_silu_mul_quant_ep(x, tokens_per_expert=counts)
    plain_q, plain_s = fuse_silu_mul_quant_ep(x)
    _assert_valid_rows_match(x, counts, masked_q, masked_s, plain_q, plain_s)


def test_bf16_guard_keeps_other_dtypes_on_the_old_path():
    """The masked dispatch is BF16-only; FP16 must not be silently routed to it
    and must produce the same valid rows as the unmasked call."""
    x = torch.randn(3, 17, 4608, dtype=torch.float16, device='cuda')
    counts = torch.tensor([0, 5, 17], dtype=torch.int32, device='cuda')
    masked_q, masked_s = fuse_silu_mul_quant_ep(x, tokens_per_expert=counts)
    plain_q, plain_s = fuse_silu_mul_quant_ep(x)
    _assert_valid_rows_match(x, counts, masked_q, masked_s, plain_q, plain_s)
