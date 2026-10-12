# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and mutation tests for the LMSlim-native RMSQuant API."""

import inspect

import pytest
import torch

from lightop import norm
from lightop._lmslim_native.quantize import quant_ops


def _reference_rmsquant(x, weight, epsilon):
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    normalized = normalized * weight.float()
    scales = (normalized.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(
        1.19209e-7
    )
    quantized = torch.round(normalized / scales).clamp(-128, 127).to(torch.int8)
    return quantized, scales, normalized


def test_lm_faster_rmsquant_is_direct_public_binding():
    assert norm.lm_faster_rmsquant is quant_ops.lm_faster_rmsquant
    assert inspect.signature(norm.lm_faster_rmsquant) == inspect.signature(
        quant_ops.lm_faster_rmsquant
    )


@pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
def test_lm_faster_rmsquant_matches_torch(hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(hidden_size)
    epsilon = 1e-6
    x = torch.randn((8, hidden_size), device="cuda", dtype=torch.float16)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    expected_q, expected_scales, expected_normalized = _reference_rmsquant(
        x, weight, epsilon
    )

    actual_q, actual_scales = norm.lm_faster_rmsquant(
        x, weight, epsilon, torch.int8, update_input=True
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0.02, atol=1e-5)
    torch.testing.assert_close(x.float(), expected_normalized, rtol=0.02, atol=0.02)


def test_lm_faster_rmsquant_preserves_residual_update_semantics():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(23)
    epsilon = 1e-6
    x = torch.randn((4, 1024), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((1024,), device="cuda", dtype=torch.float16)
    expected_residual = (x + residual).to(torch.float16)
    expected_q, expected_scales, expected_normalized = _reference_rmsquant(
        expected_residual, weight, epsilon
    )

    actual_q, actual_scales = norm.lm_faster_rmsquant(
        x,
        weight,
        epsilon,
        torch.int8,
        residual=residual,
        update_input=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0.02, atol=1e-5)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(x.float(), expected_normalized, rtol=0.02, atol=0.02)
