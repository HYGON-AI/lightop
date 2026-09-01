"""Correctness tests for inference-only optimized LayerNorm kernels."""

import pytest
import torch
import torch.nn.functional as F

from lightop import op
from lightop.layernorm import fused_add_layer_norm, layer_norm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (288, 1024), (7, 1152)])
def test_layer_norm_opt(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    bias = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected = F.layer_norm(
        x.float(), (shape[-1],), weight.float(), bias.float(), 1e-5
    ).to(dtype)

    out = torch.empty_like(x)
    assert layer_norm(x, weight, bias, 1e-5, out=out) is out
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    native_out = torch.empty_like(x)
    op.layer_norm_opt(native_out, x, weight, bias, 1e-5)
    torch.testing.assert_close(native_out, out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (288, 1024), (7, 1152)])
def test_fused_add_layer_norm_opt(
    dtype: torch.dtype, shape: tuple[int, int]
) -> None:
    torch.manual_seed(1)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    bias = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected_residual = x + residual
    expected_x = F.layer_norm(
        expected_residual.float(), (shape[-1],), weight.float(), bias.float(),
        1e-5,
    ).to(dtype)

    actual_x = x.clone()
    actual_residual = residual.clone()
    returned_x, returned_residual = fused_add_layer_norm(
        actual_x, actual_residual, weight, bias, 1e-5
    )
    assert returned_x is actual_x
    assert returned_residual is actual_residual
    torch.testing.assert_close(
        actual_residual, expected_residual, rtol=0, atol=0
    )
    torch.testing.assert_close(actual_x, expected_x, rtol=1e-2, atol=1e-2)


def test_layer_norm_opt_clamp() -> None:
    x = torch.randn((3, 1152), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1152, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1152, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    op.layer_norm_opt(out, x, weight, bias, 1e-5, 0.5)
    assert out.float().abs().max().item() <= 0.5


@pytest.mark.parametrize("shape", [(2, 3, 1024), (5, 1000)])
def test_layer_norm_opt_inplace_and_fallback(
    shape: tuple[int, ...]
) -> None:
    """Cover multidimensional input, in-place output, and generic fallback."""
    torch.manual_seed(2)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    expected = F.layer_norm(
        x.float(), (shape[-1],), weight.float(), bias.float(), 1e-5
    ).to(x.dtype)

    actual = x.clone()
    returned = layer_norm(actual, weight, bias, 1e-5)
    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("hidden_size", [1024, 1152])
def test_layer_norm_opt_constant_input(hidden_size: int) -> None:
    """For zero-variance input, normalized values are zero and output=bias."""
    x = torch.full(
        (4, hidden_size), 3.0, device="cuda", dtype=torch.float16
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    layer_norm(x, weight, bias, 1e-5, out=out)
    torch.testing.assert_close(
        out, bias.expand_as(out), rtol=0, atol=1e-3
    )
