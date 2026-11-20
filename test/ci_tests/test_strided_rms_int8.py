# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent row scales, source preservation, edge cases and graph replay."""

import pytest
import torch

from lightop import rms_norm_dynamic_per_token_quant_sglang

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="HIP required"
)


def _quant_with_norm_output(x, weight, epsilon=1e-6):
    normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    q, scales = rms_norm_dynamic_per_token_quant_sglang(
        x,
        weight,
        epsilon,
        update_input=False,
        norm_output=normalized,
    )
    return normalized, q, scales


def same_bits(x, y):
    dtype = torch.int32 if x.dtype == torch.float32 else torch.int16
    return torch.equal(x.view(dtype), y.view(dtype))


@pytest.mark.parametrize("rows", [0, 1, 8, 40, 48, 96])
@pytest.mark.parametrize("width", [128, 512, 768, 1280, 1536, 3072, 5120, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_strided_rms_quant_matches_existing_native(rows, width, dtype):
    torch.manual_seed(rows + width)
    storage = torch.randn((rows, width + 512), device="cuda", dtype=dtype)
    x = storage[:, :width]
    if rows:
        x[0].zero_()
    if rows > 1:
        x[1].mul_(1e-5)
    before = storage.clone()
    weight = torch.randn(width, device="cuda", dtype=dtype)
    expected = x.clone()
    eq, es = rms_norm_dynamic_per_token_quant_sglang(expected, weight, 1e-6)
    actual, aq, ass = _quant_with_norm_output(x, weight, 1e-6)
    assert same_bits(actual, expected)
    assert torch.equal(aq, eq) and same_bits(ass, es)
    assert same_bits(storage, before)


@pytest.mark.parametrize("rows", [8, 40, 48])
def test_strided_rms_graph_changed_inputs(rows):
    storage = torch.randn((rows, 1792), device="cuda", dtype=torch.bfloat16)
    x = storage[:, :1280]
    weight = torch.randn(1280, device="cuda", dtype=x.dtype)
    _quant_with_norm_output(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, aq, ass = _quant_with_norm_output(x, weight)
    pointers = [t.data_ptr() for t in (actual, aq, ass)]
    for phase in range(4):
        storage.normal_()
        x[phase].zero_()
        before = storage.clone()
        expected = x.clone()
        eq, es = rms_norm_dynamic_per_token_quant_sglang(expected, weight, 1e-6)
        graph.replay()
        assert same_bits(actual, expected)
        assert torch.equal(aq, eq) and same_bits(ass, es)
        assert same_bits(storage, before)
        assert pointers == [t.data_ptr() for t in (actual, aq, ass)]


def test_strided_rms_rejects_column_stride():
    x = torch.randn((8, 2560), device="cuda", dtype=torch.bfloat16)[:, ::2]
    weight = torch.ones(1280, device="cuda", dtype=x.dtype)
    with pytest.raises(RuntimeError, match="unit .*stride"):
        _quant_with_norm_output(x, weight)
