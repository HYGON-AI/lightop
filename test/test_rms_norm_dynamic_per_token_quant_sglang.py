import pytest
import torch

from lightop import rms_norm_dynamic_per_token_quant_sglang


EPSILON = 1.0e-5


def _reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    value = input.float()
    residual_reference = None
    if residual is not None:
        value = value + residual.float()
        residual_reference = value.to(input.dtype)

    variance = value.square().mean(dim=-1, keepdim=True)
    normalized = (
        value * torch.rsqrt(variance + EPSILON) * weight.float()
    ).to(input.dtype)
    absmax = normalized.float().abs().amax(dim=-1, keepdim=True)
    scale = absmax.clamp_min(1.0e-10) / 127.0
    quantized = (
        torch.round(normalized.float() / scale)
        .clamp(-128, 127)
        .to(torch.int8)
    )
    return quantized, scale, residual_reference


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize("hidden_size", [1536, 4096])
def test_sglang_rms_quant_matches_native(
    num_tokens: int,
    hidden_size: int,
) -> None:
    torch.manual_seed(20260715)
    input = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    weight = (
        1.0
        + 0.1
        * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    ).contiguous()
    input_reference = input.clone()
    q_reference, scale_reference, _ = _reference(input_reference, weight)

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input=input,
        weight=weight,
        epsilon=EPSILON,
        update_input=True,
    )

    normalized_reference = (
        input_reference.float()
        * torch.rsqrt(
            input_reference.float().square().mean(dim=-1, keepdim=True)
            + EPSILON
        )
        * weight.float()
    ).to(input.dtype)
    torch.testing.assert_close(
        input, normalized_reference, rtol=2.0e-2, atol=2.0e-2
    )
    torch.testing.assert_close(
        scales, scale_reference, rtol=2.0e-3, atol=1.0e-7
    )
    assert torch.max(
        torch.abs(quantized.to(torch.int16) - q_reference.to(torch.int16))
    ).item() <= 1


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_sglang_rms_quant_preserves_input_when_update_disabled() -> None:
    torch.manual_seed(20260715)
    input = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    original = input.clone()
    q_reference, scale_reference, _ = _reference(original, weight)

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input=input,
        weight=weight,
        epsilon=EPSILON,
        update_input=False,
    )

    assert torch.equal(input, original)
    torch.testing.assert_close(
        scales, scale_reference, rtol=2.0e-3, atol=1.0e-7
    )
    assert torch.max(
        torch.abs(quantized.to(torch.int16) - q_reference.to(torch.int16))
    ).item() <= 1


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_sglang_rms_quant_residual_contract() -> None:
    torch.manual_seed(20260715)
    input = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input)
    weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    q_reference, scale_reference, residual_reference = _reference(
        input.clone(), weight, residual.clone()
    )

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input=input,
        weight=weight,
        epsilon=EPSILON,
        residual=residual,
        update_input=False,
    )

    assert residual_reference is not None
    torch.testing.assert_close(
        residual, residual_reference, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        scales, scale_reference, rtol=2.0e-3, atol=1.0e-7
    )
    assert torch.max(
        torch.abs(quantized.to(torch.int16) - q_reference.to(torch.int16))
    ).item() <= 1


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_sglang_rms_quant_zero_row_scale_floor() -> None:
    input = torch.zeros(3, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

    quantized, scales = rms_norm_dynamic_per_token_quant_sglang(
        input=input,
        weight=weight,
        epsilon=EPSILON,
        update_input=True,
    )

    assert torch.count_nonzero(quantized).item() == 0
    torch.testing.assert_close(
        scales,
        torch.full_like(scales, 1.0e-10 / 127.0),
        rtol=1.0e-6,
        atol=0.0,
    )
