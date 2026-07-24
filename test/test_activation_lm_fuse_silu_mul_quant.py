"""Numerical tests for the LMSlim-native fused SiLU-mul INT8 API."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from lightop import activation
from lightop._lmslim_native.quantize import quant_ops


def _reference_silu_mul_quant(x):
    gate, value = x.float().chunk(2, dim=-1)
    activated = F.silu(gate) * value
    scales = (activated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(
        1.19209e-7
    )
    quantized = torch.round(activated / scales).clamp(-128, 127).to(torch.int8)
    return quantized, scales


def test_lm_fuse_silu_mul_quant_is_direct_public_binding():
    assert activation.lm_fuse_silu_mul_quant is quant_ops.lm_fuse_silu_mul_quant
    assert inspect.signature(activation.lm_fuse_silu_mul_quant) == inspect.signature(
        quant_ops.lm_fuse_silu_mul_quant
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (17, 8192), (128, 2048)])
def test_lm_fuse_silu_mul_quant_matches_torch(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(shape[0] + shape[1])
    x = torch.randn(shape, device="cuda", dtype=dtype)
    expected_q, expected_scales = _reference_silu_mul_quant(x)

    actual_q, actual_scales = activation.lm_fuse_silu_mul_quant(x)
    torch.cuda.synchronize()

    assert actual_q.dtype is torch.int8
    assert actual_scales.dtype is torch.float32
    assert actual_q.shape == (shape[0], shape[1] // 2)
    assert actual_scales.shape == (shape[0], 1)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0.02, atol=1e-5)
