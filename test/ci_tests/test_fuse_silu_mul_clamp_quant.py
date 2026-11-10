# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from lightop import (
    fuse_silu_mul_clamp_quant,
    fuse_silu_mul_clamp_quant_ep,
)


def _reference(
    input: torch.Tensor,
    limit: float,
    use_gelu: bool = False,
    smooth_scale: torch.Tensor | None = None,
):
    hidden = input.shape[-1] // 2
    if use_gelu:
        gate = input[..., :hidden]
        up = input[..., hidden:]
        activated = F.gelu(gate, approximate="none") * up
        activated = activated.clamp(min=-limit, max=limit)
    else:
        gate = input[..., :hidden].clamp(max=limit)
        up = input[..., hidden:].clamp(min=-limit, max=limit)
        activated = F.silu(gate) * up

    if smooth_scale is not None:
        activated = activated * smooth_scale.to(dtype=input.dtype)

    absmax = activated.float().abs().amax(dim=-1, keepdim=True)
    absmax = absmax.clamp_min(1.0e-10)
    scales = absmax / 127.0
    quantized = torch.round(activated.float() / scales)
    quantized = quantized.clamp(-128, 127).to(torch.int8)
    return quantized, scales


@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("hidden", [128, 768, 1536, 2048, 12288])
@pytest.mark.parametrize("use_gelu", [False, True])
def test_fuse_silu_mul_clamp_quant_glm_shapes(tokens, hidden, use_gelu):
    torch.manual_seed(1234)
    limit = 10.0
    input = (
        torch.randn((tokens, 2 * hidden), device="cuda", dtype=torch.bfloat16)
        * 4
    )
    # Exercise both clamp boundaries, not only the normal random range.
    input[0, :8] = torch.tensor(
        [-12.0, -10.0, -9.5, 0.0, 9.5, 10.0, 10.5, 12.0],
        device=input.device,
        dtype=input.dtype,
    )
    input[0, hidden : hidden + 8] = torch.tensor(
        [-12.0, -10.0, -9.5, 0.0, 9.5, 10.0, 10.5, 12.0],
        device=input.device,
        dtype=input.dtype,
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(
        input, limit, use_gelu=use_gelu
    )
    expected_q, expected_scale = _reference(input, limit, use_gelu=use_gelu)

    assert torch.max(
        torch.abs(actual_q.to(torch.int16) - expected_q.to(torch.int16))
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8
    )


def test_fuse_silu_mul_clamp_quant_zero_row_scale():
    input = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(input, 10.0)

    assert torch.count_nonzero(actual_q).item() == 0
    torch.testing.assert_close(
        actual_scale,
        torch.full_like(actual_scale, 1.0e-10 / 127.0),
        rtol=1.0e-6,
        atol=0,
    )


@pytest.mark.parametrize("hidden", [128, 513, 1536])
@pytest.mark.parametrize("expect_m", [-1, 1, 3, 32])
@pytest.mark.parametrize("use_gelu", [False, True])
def test_fuse_silu_mul_clamp_quant_ep_masked(hidden, expect_m, use_gelu):
    torch.manual_seed(2026)
    experts, tokens = 4, 9
    limit = 10.0
    mask_m = torch.tensor(
        [tokens, 5, 1, 0], device="cuda", dtype=torch.int32
    )
    input = (
        torch.randn(
            (experts, tokens, 2 * hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 4
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant_ep(
        input, limit, mask_m, expect_m, use_gelu=use_gelu
    )
    dispatched_q, dispatched_scale = fuse_silu_mul_clamp_quant(
        input,
        limit,
        mask_m=mask_m,
        expect_m=expect_m,
        use_gelu=use_gelu,
    )
    expected_q, expected_scale = _reference(input, limit, use_gelu=use_gelu)
    valid = (
        torch.arange(tokens, device="cuda").unsqueeze(0)
        < mask_m.unsqueeze(1)
    )

    assert torch.max(
        torch.abs(
            actual_q[valid].to(torch.int16)
            - expected_q[valid].to(torch.int16)
        )
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale[valid],
        expected_scale[valid],
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert torch.equal(dispatched_q[valid], actual_q[valid])
    torch.testing.assert_close(
        dispatched_scale[valid],
        actual_scale[valid],
        rtol=0,
        atol=0,
    )


def test_fuse_silu_mul_clamp_quant_ep_requires_mask():
    input = torch.empty((2, 4, 256), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="mask_m is required"):
        fuse_silu_mul_clamp_quant(input, 10.0, expect_m=2)


def test_fuse_gelu_mul_clamp_smooth_quant_matches_reference_ordering():
    torch.manual_seed(20260807)
    input = (
        torch.randn((17, 2048), device="cuda", dtype=torch.bfloat16) * 4
    )
    smooth_scale = torch.empty(
        (1024,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    expected_q, expected_scale = _reference(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )

    assert torch.max(
        torch.abs(actual_q.to(torch.int16) - expected_q.to(torch.int16))
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8
    )


def test_fuse_gelu_mul_clamp_smooth_quant_ep_masked():
    torch.manual_seed(20260808)
    experts, tokens, hidden = 16, 8, 1024
    input = torch.randn(
        (experts, tokens, 2 * hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )
    smooth_scale = torch.empty(
        (hidden,), device="cuda", dtype=torch.float32
    ).uniform_(0.5, 2.0)
    mask_m = torch.tensor(
        [8, 7, 6, 5, 4, 3, 2, 1, 8, 6, 4, 2, 7, 5, 3, 0],
        device="cuda",
        dtype=torch.int32,
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant_ep(
        input,
        8.0,
        mask_m,
        expect_m=4,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    expected_q, expected_scale = _reference(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    valid = (
        torch.arange(tokens, device="cuda").unsqueeze(0)
        < mask_m.unsqueeze(1)
    )

    assert torch.max(
        torch.abs(
            actual_q[valid].to(torch.int16)
            - expected_q[valid].to(torch.int16)
        )
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale[valid],
        expected_scale[valid],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_fuse_gelu_mul_clamp_quant_default_preserves_legacy_api():
    torch.manual_seed(20260809)
    input = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)

    default_q, default_scale = fuse_silu_mul_clamp_quant(
        input, 8.0, use_gelu=True
    )
    none_q, none_scale = fuse_silu_mul_clamp_quant(
        input, 8.0, use_gelu=True, smooth_scale=None
    )

    assert torch.equal(default_q, none_q)
    torch.testing.assert_close(default_scale, none_scale, rtol=0, atol=0)
