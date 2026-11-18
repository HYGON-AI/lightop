# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise regression against the deployed SGLang HIP native operator."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires a ROCm GPU",
)


def test_attention_export():
    from lightop import attention
    from lightop._dsv41_prep import dsv41_indexer_rope_fake_quant

    assert "dsv41_indexer_rope_fake_quant" in attention.__all__
    assert attention.dsv41_indexer_rope_fake_quant is dsv41_indexer_rope_fake_quant


@pytest.fixture
def ops():
    from lightop.attention import dsv41_indexer_rope_fake_quant

    native = pytest.importorskip("sgl_kernel")
    if not hasattr(native, "rope_fp4_fake_quant"):
        pytest.skip("requires sgl_kernel.rope_fp4_fake_quant reference")
    return dsv41_indexer_rope_fake_quant, native.rope_fp4_fake_quant


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def inputs(tokens, heads, index_dtype=torch.int64):
    q = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    phase = torch.randn(max(32, tokens * 2), 32, device="cuda")
    freqs = torch.polar(torch.ones_like(phase), phase)
    positions = torch.randint(
        len(freqs), (tokens,), device="cuda", dtype=index_dtype
    )
    return q, freqs, positions


@pytest.mark.parametrize("tokens", [0, 1, 3, 17, 128, 4096])
@pytest.mark.parametrize("heads", [1, 7, 32, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_native_bitwise(ops, tokens, heads, index_dtype):
    fused, native = ops
    q, freqs, positions = inputs(tokens, heads, index_dtype)
    saved = [t.clone() for t in (q, freqs, positions)]
    y = fused(q, freqs, positions)
    assert y.shape == q.shape and y.is_contiguous()
    if tokens:
        assert_bits(y, native(q, freqs[positions.long()], 64, False))
    for value, original in zip((q, freqs, positions), saved):
        assert torch.equal(value, original)


@pytest.mark.parametrize("magnitude", [0.0, 1e-38, 1e-35, 1e-6, 1.0, 1000.0, 1e35])
def test_quantization_range(ops, magnitude):
    fused, native = ops
    q, _, positions = inputs(65, 32)
    q *= magnitude
    freqs = torch.ones(130, 32, device="cuda", dtype=torch.complex64)
    assert_bits(fused(q, freqs, positions), native(q, freqs[positions], 64, False))


def test_ties_and_signed_zero(ops):
    fused, native = ops
    values = [6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5,
              -6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5]
    q = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
    q = q.repeat(8).view(1, 1, 128).expand(17, 32, 128).clone()
    freqs = torch.ones(17, 32, device="cuda", dtype=torch.complex64)
    positions = torch.arange(17, device="cuda")
    assert_bits(fused(q, freqs, positions), native(q, freqs, 64, False))


def test_unaligned_contiguous_input(ops):
    fused, native = ops
    q = torch.randn(17 * 7 * 128 + 1, device="cuda", dtype=torch.bfloat16)
    q = q[1:].view(17, 7, 128)
    _, freqs, positions = inputs(17, 7)
    assert q.is_contiguous() and q.data_ptr() % 8 != 0
    assert_bits(fused(q, freqs, positions), native(q, freqs[positions], 64, False))


def test_invalid_metadata(ops):
    fused, _ = ops
    q, freqs, positions = inputs(17, 7)
    cases = [(q.float(), freqs, positions),
             (q, freqs[:, ::2], positions),
             (q, freqs, positions.cpu()),
             (q.transpose(0, 1), freqs, positions)]
    for args in cases:
        with pytest.raises(RuntimeError):
            fused(*args)


def test_graph_replay_and_nondefault_stream(ops):
    fused, native = ops
    q, freqs, positions = inputs(65, 32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(q, freqs, positions)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = fused(q, freqs, positions)
        for _ in range(3):
            q.normal_()
            positions.random_(len(freqs))
            graph.replay()
            assert_bits(y, native(q, freqs[positions], 64, False))
    torch.cuda.current_stream().wait_stream(stream)
