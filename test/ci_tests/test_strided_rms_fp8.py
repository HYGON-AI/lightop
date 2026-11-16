# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Strided FP8 RMSNorm matches the existing FP8 normalization contract."""

import pytest
import torch

from lightop.norm import rms_norm_per_token_fp8_quant
from lightop.rmsnorm import RMS_NORM_FP8_QUANT_VERSION

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or RMS_NORM_FP8_QUANT_VERSION < 2,
    reason="native strided RMS+FP8 extension required",
)


def _quant_with_norm_output(x, weight):
    normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    q, scales = rms_norm_per_token_fp8_quant(
        x,
        weight,
        epsilon=1e-6,
        update_input=False,
        norm_output=normalized,
    )
    return normalized, q, scales


def check(x, weight, actual):
    expected = x.contiguous().clone()
    if x.shape[0]:
        q, scale = rms_norm_per_token_fp8_quant(
            expected, weight, epsilon=1e-6, update_input=True
        )
    else:
        q = torch.empty_like(expected, dtype=torch.float8_e4m3fn)
        scale = torch.empty((0, 1), device=x.device)
    assert torch.equal(actual[0].view(torch.int16), expected.view(torch.int16))
    assert torch.equal(actual[1].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(actual[2].view(torch.int32), scale.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [0, 1, 8, 40, 48, 96])
@pytest.mark.parametrize("width", [128, 512, 1280, 1536, 3072, 5120, 8192, 16384])
def test_strided_rms_fp8_exact(dtype, rows, width):
    base = torch.randn((rows, width + 512), device="cuda", dtype=dtype)
    x = base[:, :width]
    if rows:
        x[0].zero_()
    if rows > 1:
        x[1].mul_(1e-5)
    weight = torch.randn(width, device="cuda", dtype=dtype)
    before = base.clone()
    check(x, weight, _quant_with_norm_output(x, weight))
    assert torch.equal(base, before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_strided_rms_fp8_graph_changed_inputs(dtype):
    base = torch.randn((48, 1792), device="cuda", dtype=dtype)
    x = base[:, :1280]
    weight = torch.randn(1280, device="cuda", dtype=dtype)
    _quant_with_norm_output(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _quant_with_norm_output(x, weight)
    pointers = [v.data_ptr() for v in actual]
    for phase in range(4):
        base.normal_()
        x[phase].zero_()
        before = base.clone()
        graph.replay()
        check(x, weight, actual)
        assert torch.equal(base, before)
        assert pointers == [v.data_ptr() for v in actual]
