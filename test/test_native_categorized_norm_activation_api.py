"""Verify categorized exports for native norm and activation operators."""

import torch

import lightop.activation
import lightop.norm
from lightop import op as lightop_native
from lightop.activation import (
    gelu_and_mul_opt,
    gelu_tanh_and_mul_opt,
    silu_and_mul_opt,
)
from lightop.norm import fused_add_rms_norm_opt, rms_norm_opt


def test_native_categorized_exports() -> None:
    assert rms_norm_opt is lightop_native.rms_norm_opt
    assert fused_add_rms_norm_opt is lightop_native.fused_add_rms_norm_opt
    assert gelu_and_mul_opt is lightop_native.gelu_and_mul_opt
    assert gelu_tanh_and_mul_opt is lightop_native.gelu_tanh_and_mul_opt
    assert silu_and_mul_opt is lightop_native.silu_and_mul_opt
    assert "rms_norm_opt" in lightop.norm.__all__
    assert "fused_add_rms_norm_opt" in lightop.norm.__all__
    assert "gelu_and_mul_opt" in lightop.activation.__all__
    assert "gelu_tanh_and_mul_opt" in lightop.activation.__all__
    assert "silu_and_mul_opt" in lightop.activation.__all__


def test_rms_norm_opt_categorized_api_matches_native() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    categorized_out = torch.empty_like(x)
    native_out = torch.empty_like(x)

    rms_norm_opt(categorized_out, x, weight, 1e-6)
    lightop_native.rms_norm_opt(native_out, x, weight, 1e-6)

    torch.testing.assert_close(categorized_out, native_out, rtol=0, atol=0)


def test_fused_add_rms_norm_opt_categorized_api_matches_native() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    categorized_x = x.clone()
    categorized_residual = residual.clone()
    native_x = x.clone()
    native_residual = residual.clone()

    fused_add_rms_norm_opt(
        categorized_x, categorized_residual, weight, 1e-6
    )
    lightop_native.fused_add_rms_norm_opt(
        native_x, native_residual, weight, 1e-6
    )

    torch.testing.assert_close(categorized_x, native_x, rtol=0, atol=0)
    torch.testing.assert_close(
        categorized_residual, native_residual, rtol=0, atol=0
    )


def test_silu_and_mul_opt_categorized_api_matches_native() -> None:
    torch.manual_seed(2)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    categorized_out = torch.empty(4, 128, device="cuda", dtype=x.dtype)
    native_out = torch.empty_like(categorized_out)

    silu_and_mul_opt(categorized_out, x)
    lightop_native.silu_and_mul_opt(native_out, x)

    torch.testing.assert_close(categorized_out, native_out, rtol=0, atol=0)


def test_gelu_and_mul_categorized_apis_match_native() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

    for categorized_api, native_api in (
        (gelu_and_mul_opt, lightop_native.gelu_and_mul_opt),
        (gelu_tanh_and_mul_opt, lightop_native.gelu_tanh_and_mul_opt),
    ):
        categorized_out = torch.empty(4, 128, device="cuda", dtype=x.dtype)
        native_out = torch.empty_like(categorized_out)

        categorized_api(categorized_out, x)
        native_api(native_out, x)

        torch.testing.assert_close(categorized_out, native_out, rtol=0, atol=0)
