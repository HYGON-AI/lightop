"""Verify categorized FP8 quantization uses native LightOp kernels."""

import pytest
import torch

from lightop import op as lightop_native
from lightop.quant import per_token_group_quant_fp8 as category_group_api
from lightop.quant import per_token_quant_fp8 as category_token_api


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
def test_per_token_group_quant_fp8_category_matches_native(input_dtype) -> None:
    assert category_group_api.__module__ == "lightop.quant.fp8"

    torch.manual_seed(20260721)
    input_tensor = torch.randn(
        (8, 256), device="cuda", dtype=input_dtype
    )
    input_tensor[0].zero_()

    group_size = 128
    eps = 1e-10
    category_q, category_scale = category_group_api(
        input_tensor,
        group_size,
        eps=eps,
        dtype=torch.float8_e4m3fn,
        column_major_scales=False,
    )

    native_q = torch.empty_like(
        input_tensor, dtype=torch.float8_e4m3fn
    )
    native_scale = torch.empty(
        input_tensor.shape[:-1]
        + (input_tensor.shape[-1] // group_size,),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    lightop_native.per_token_group_quant_fp8(
        native_q,
        input_tensor,
        native_scale,
        group_size,
        eps,
        False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        native_scale, category_scale, rtol=1e-6, atol=1e-8
    )
    fp8_ulp_diff = (
        native_q.view(torch.uint8).to(torch.int16)
        - category_q.view(torch.uint8).to(torch.int16)
    ).abs()
    assert fp8_ulp_diff.max().item() <= 1

    expanded_native_scale = native_scale.repeat_interleave(group_size, dim=-1)
    expanded_category_scale = category_scale.repeat_interleave(
        group_size, dim=-1
    )
    native_dequant = native_q.float() * expanded_native_scale
    category_dequant = category_q.float() * expanded_category_scale
    torch.testing.assert_close(
        native_dequant, category_dequant, rtol=0.125, atol=1e-3
    )


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
def test_per_token_quant_fp8_category_matches_native(input_dtype) -> None:
    assert category_token_api.__module__ == "lightop.quant.fp8"

    torch.manual_seed(20260721)
    input_tensor = torch.randn(
        (8, 257), device="cuda", dtype=input_dtype
    )
    input_tensor[0].zero_()

    category_q, category_scale = category_token_api(input_tensor)
    native_q = torch.empty_like(
        input_tensor, dtype=torch.float8_e4m3fn
    )
    native_scale = torch.empty(
        input_tensor.shape[:-1] + (1,),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    lightop_native.per_token_quant_fp8(
        native_q, input_tensor, native_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(category_scale, native_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        category_q.float(), native_q.float(), rtol=0, atol=0
    )
