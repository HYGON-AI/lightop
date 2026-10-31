# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm + clamp + smooth + INT8 quant compatibility tests."""

from lightop.norm import fused_add_rms_norm_opt as categorized_fused_add_rms_norm_opt
from lightop.norm import rms_norm_opt as categorized_rms_norm_opt
import pytest
import torch

from lightop.quant import per_token_dynamic_quant_int8
from lightop.rmsnorm import rms_norm_dynamic_per_token_quant_sglang


def _quant_reference(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_absmax = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scales = row_absmax / 127.0
    quantized = (
        torch.round(values.float() * (127.0 / row_absmax))
        .clamp(-128, 127)
        .to(torch.int8)
    )
    return quantized, scales


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("update_input", [False, True])
def test_rms_norm_clamp_smooth_quant_matches_reference_numerics(
    with_residual: bool,
    update_input: bool,
) -> None:
    torch.manual_seed(20260808)
    epsilon = 1e-5
    limit = 8.0
    dtype = torch.bfloat16
    original_input = (
        torch.randn((5, 1024), device="cuda", dtype=dtype) * 12.0
    )
    input_ = original_input.clone()
    weight = torch.randn((1024,), device="cuda", dtype=dtype)
    smooth_scale = torch.empty(
        (1024,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)
    original_residual = (
        torch.randn_like(input_) if with_residual else None
    )
    residual = (
        original_residual.clone() if original_residual is not None else None
    )
    norm_output = torch.empty_like(input_)

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input_,
        weight,
        epsilon,
        residual=residual,
        update_input=update_input,
        limit=limit,
        smooth_scale=smooth_scale,
        norm_output=norm_output,
        clamp_norm_output=True,
    )

    baseline_input = original_input.clone()
    baseline_residual = (
        original_residual.clone()
        if original_residual is not None
        else None
    )
    if baseline_residual is None:
        normalized = torch.empty_like(baseline_input)
        categorized_rms_norm_opt(
            normalized, baseline_input, weight, epsilon, limit
        )
    else:
        categorized_fused_add_rms_norm_opt(
            baseline_input, baseline_residual, weight, epsilon, limit
        )
        normalized = baseline_input
    expected_q, expected_scales = per_token_dynamic_quant_int8(
        normalized,
        smooth_scale,
        round_after_smooth=True,
    )

    torch.testing.assert_close(norm_output, normalized, rtol=0, atol=0)
    torch.testing.assert_close(quantized, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        scales, expected_scales.unsqueeze(-1), rtol=1e-5, atol=0
    )
    if update_input:
        torch.testing.assert_close(input_, normalized, rtol=0, atol=0)
    else:
        torch.testing.assert_close(input_, original_input, rtol=0, atol=0)
    if residual is not None:
        torch.testing.assert_close(
            residual, baseline_residual, rtol=0, atol=0
        )


def test_default_path_remains_rms_norm_then_quant() -> None:
    torch.manual_seed(7)
    input_ = torch.randn((3, 1024), device="cuda", dtype=torch.bfloat16)
    original_input = input_.clone()
    weight = torch.randn((1024,), device="cuda", dtype=torch.bfloat16)

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input_, weight, 1e-5, update_input=False
    )

    rstd = torch.rsqrt(
        original_input.float().square().mean(dim=-1, keepdim=True) + 1e-5
    )
    normalized = (original_input.float() * rstd * weight.float()).to(
        torch.bfloat16
    )
    expected_q, expected_scales = _quant_reference(normalized)
    torch.testing.assert_close(quantized, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=1e-5, atol=0)
    torch.testing.assert_close(input_, original_input, rtol=0, atol=0)


@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_output_can_remain_unclamped_for_router_branch(
    with_residual: bool,
) -> None:
    torch.manual_seed(99)
    input_ = torch.randn(
        (2, 1024), device="cuda", dtype=torch.bfloat16
    ) * 32.0
    original_input = input_.clone()
    original_residual = (
        torch.randn_like(input_) * 4.0 if with_residual else None
    )
    residual = (
        original_residual.clone() if original_residual is not None else None
    )
    weight = torch.randn((1024,), device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.empty(
        (1024,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)
    norm_output = torch.empty_like(input_)
    limit = 0.5

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input_,
        weight,
        1e-5,
        residual=residual,
        update_input=False,
        limit=limit,
        smooth_scale=smooth_scale,
        norm_output=norm_output,
        clamp_norm_output=False,
    )

    baseline_input = original_input.clone()
    baseline_residual = (
        original_residual.clone()
        if original_residual is not None
        else None
    )
    if baseline_residual is None:
        expected_norm = torch.empty_like(baseline_input)
        categorized_rms_norm_opt(
            expected_norm, baseline_input, weight, 1e-5, -1.0
        )
    else:
        categorized_fused_add_rms_norm_opt(
            baseline_input, baseline_residual, weight, 1e-5, -1.0
        )
        expected_norm = baseline_input
    expected_q, expected_scales = per_token_dynamic_quant_int8(
        expected_norm,
        smooth_scale,
        round_after_smooth=True,
        limit=limit,
    )

    assert (expected_norm.abs() > limit).any()
    torch.testing.assert_close(norm_output, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(quantized, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        scales, expected_scales.unsqueeze(-1), rtol=1e-5, atol=0
    )
    torch.testing.assert_close(input_, original_input, rtol=0, atol=0)
    if residual is not None:
        torch.testing.assert_close(
            residual,
            baseline_residual,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("hidden_size", "with_residual"),
    [(1536, False), (6144, True)],
)
def test_fused_result_matches_existing_lightop_norm_chain(
    hidden_size: int,
    with_residual: bool,
) -> None:
    torch.manual_seed(20260809 + hidden_size)
    epsilon = 1e-5
    limit = 8.0
    input_ = torch.randn(
        (3, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        (hidden_size,), device="cuda", dtype=torch.bfloat16
    )
    smooth_scale = torch.empty(
        (hidden_size,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)
    residual = torch.randn_like(input_) if with_residual else None

    baseline_norm = input_.clone()
    baseline_residual = residual.clone() if residual is not None else None
    if baseline_residual is None:
        baseline_output = torch.empty_like(baseline_norm)
        categorized_rms_norm_opt(
            baseline_output,
            baseline_norm,
            weight,
            epsilon,
            limit,
        )
    else:
        categorized_fused_add_rms_norm_opt(
            baseline_norm,
            baseline_residual,
            weight,
            epsilon,
            limit,
        )
        baseline_output = baseline_norm
    expected_q, expected_scales = per_token_dynamic_quant_int8(
        baseline_output,
        smooth_scale,
        round_after_smooth=True,
    )

    fused_input = input_.clone()
    fused_residual = residual.clone() if residual is not None else None
    fused_norm = torch.empty_like(fused_input)
    actual_q, actual_scales = rms_norm_dynamic_per_token_quant_sglang(
        fused_input,
        weight,
        epsilon,
        residual=fused_residual,
        update_input=False,
        limit=limit,
        smooth_scale=smooth_scale,
        norm_output=fused_norm,
        clamp_norm_output=True,
    )

    torch.testing.assert_close(fused_norm, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales,
        expected_scales.unsqueeze(-1),
        rtol=1e-5,
        atol=0,
    )
    if fused_residual is not None:
        torch.testing.assert_close(
            fused_residual,
            baseline_residual,
            rtol=0,
            atol=0,
        )
