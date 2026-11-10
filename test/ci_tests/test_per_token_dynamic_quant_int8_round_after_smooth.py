# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional post-smooth rounding in dynamic INT8 quantization."""

import pytest
import torch

from lightop.quant import (
    per_token_dynamic_quant_int8,
    per_token_quant_int8,
)


def _reference_quant(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
    *,
    round_after_smooth: bool,
    limit: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()
    if limit >= 0.0:
        x_float = x_float.clamp(-limit, limit).to(x.dtype).float()
    scale = smooth_scale.float()
    if round_after_smooth:
        scale = smooth_scale.to(x.dtype).float()
    x_scaled = x_float * scale
    if round_after_smooth:
        x_scaled = x_scaled.to(x.dtype).float()

    row_max = x_scaled.abs().amax(dim=-1, keepdim=True)
    if round_after_smooth:
        row_max = torch.clamp_min(row_max, 1e-10)
    scales = (row_max / 127.0).squeeze(-1)
    inv_scale = torch.where(
        row_max == 0.0, torch.zeros_like(row_max), 127.0 / row_max
    )
    x_q = torch.round(x_scaled * inv_scale).clamp(-128, 127).to(torch.int8)
    return x_q, scales


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2026)
    x = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.empty(
        (128,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)
    return x, smooth_scale


def test_round_after_smooth_defaults_to_false() -> None:
    x, smooth_scale = _make_inputs()

    default_q, default_scales = per_token_dynamic_quant_int8(x, smooth_scale)
    false_q, false_scales = per_token_dynamic_quant_int8(
        x, smooth_scale, round_after_smooth=False
    )

    torch.testing.assert_close(default_q, false_q, rtol=0, atol=0)
    torch.testing.assert_close(default_scales, false_scales, rtol=0, atol=0)


def test_round_after_smooth_matches_bf16_intermediate_rounding() -> None:
    x, smooth_scale = _make_inputs()

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x, smooth_scale, round_after_smooth=True
    )
    expected_q, expected_scales = _reference_quant(
        x, smooth_scale, round_after_smooth=True
    )
    unrounded_q, _ = _reference_quant(
        x, smooth_scale, round_after_smooth=False
    )

    assert not torch.equal(expected_q, unrounded_q)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0)


def test_round_after_smooth_zero_row_uses_legacy_scale_floor() -> None:
    x = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones((128,), device="cuda", dtype=torch.float32)
    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x, smooth_scale, round_after_smooth=True
    )
    expected_q, expected_scales = _reference_quant(
        x, smooth_scale, round_after_smooth=True
    )
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0)


def test_clamp_is_applied_before_smooth() -> None:
    x, smooth_scale = _make_inputs()
    x.mul_(16)
    limit = 1.3

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x,
        smooth_scale,
        round_after_smooth=True,
        limit=limit,
    )
    expected_q, expected_scales = _reference_quant(
        x,
        smooth_scale,
        round_after_smooth=True,
        limit=limit,
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0)


def test_empty_token_batch_is_a_noop() -> None:
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones((128,), device="cuda", dtype=torch.float32)

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x,
        smooth_scale,
        round_after_smooth=True,
        limit=1.3,
    )

    assert actual_q.shape == x.shape
    assert actual_scales.shape == (0,)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("round_after_smooth", "limit"),
    [
        (False, -1.0),  # Historical defaults: FP32 scale/product, no clamp.
        (False, 1.3),  # Clamp with the historical FP32 scale/product.
        (True, -1.0),  # Activation-dtype smooth rounding, no clamp.
        (True, 1.3),  # Clamp followed by activation-dtype smooth rounding.
    ],
)
def test_large_hidden_matches_lmslim_bitwise(
    dtype: torch.dtype,
    round_after_smooth: bool,
    limit: float,
) -> None:
    """The strided 16K path must preserve both numerical contracts."""

    torch.manual_seed(20260810)
    hidden_size = 16384
    x = (
        torch.randn((3, hidden_size), device="cuda", dtype=dtype) * 8
    )
    smooth_scale = torch.empty(
        (hidden_size,), device="cuda", dtype=torch.float32
    ).uniform_(0.05, 2.0)

    if not round_after_smooth and limit < 0.0:
        # Exercise the public API defaults, not just equivalent explicit args.
        actual_q, actual_scales = per_token_dynamic_quant_int8(
            x, smooth_scale
        )
    else:
        actual_q, actual_scales = per_token_dynamic_quant_int8(
            x,
            smooth_scale,
            round_after_smooth=round_after_smooth,
            limit=limit,
        )

    reference_input = x
    if limit >= 0.0:
        assert bool((x.abs() > limit).any())
        reference_input = x.float().clamp(-limit, limit).to(dtype)
    if round_after_smooth:
        # The stored FP32 scale is cast to activation dtype and
        # materializes the product in that dtype before lmslim quantization.
        reference_input = reference_input * smooth_scale.to(dtype)
    else:
        reference_input = reference_input.float() * smooth_scale.float()
    expected_q, expected_scales = per_token_quant_int8(reference_input)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales,
        expected_scales.squeeze(-1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("hidden_size", [12288, 32768])
def test_additional_large_hidden_widths_match_lmslim_bitwise(
    hidden_size: int,
) -> None:
    """Cover the 1024-thread medium branch and two-pass upper bound."""

    torch.manual_seed(20260811)
    x = torch.randn(
        (1, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    smooth_scale = torch.empty(
        (hidden_size,), device="cuda", dtype=torch.float32
    ).uniform_(0.05, 2.0)

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x, smooth_scale, round_after_smooth=True
    )
    expected_input = x * smooth_scale.to(torch.bfloat16)
    expected_q, expected_scales = per_token_quant_int8(expected_input)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales,
        expected_scales.squeeze(-1),
        rtol=0,
        atol=0,
    )


def test_large_hidden_zero_row_uses_lmslim_scale_floor() -> None:
    hidden_size = 16384
    x = torch.zeros(
        (1, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    smooth_scale = torch.ones(
        (hidden_size,), device="cuda", dtype=torch.float32
    )

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x, smooth_scale, round_after_smooth=True
    )
    expected_q, expected_scales = per_token_quant_int8(x)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales,
        expected_scales.squeeze(-1),
        rtol=0,
        atol=0,
    )


def test_large_hidden_accepts_16_byte_aligned_storage_offset() -> None:
    """Do not require accidental 32-byte alignment from BF16 inputs."""

    torch.manual_seed(20260812)
    hidden_size = 8208
    storage = torch.randn(
        (hidden_size + 8,), device="cuda", dtype=torch.bfloat16
    )
    offset = 8 if storage.data_ptr() % 32 == 0 else 0
    x = storage[offset : offset + hidden_size].view(1, hidden_size)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 == 0
    assert x.data_ptr() % 32 == 16
    smooth_scale = torch.empty(
        (hidden_size,), device="cuda", dtype=torch.float32
    ).uniform_(0.05, 2.0)

    actual_q, actual_scales = per_token_dynamic_quant_int8(
        x,
        smooth_scale,
        round_after_smooth=True,
        limit=1.3,
    )
    expected_input = (
        x.float().clamp(-1.3, 1.3).to(torch.bfloat16)
        * smooth_scale.to(torch.bfloat16)
    )
    expected_q, expected_scales = per_token_quant_int8(expected_input)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales,
        expected_scales.squeeze(-1),
        rtol=0,
        atol=0,
    )
