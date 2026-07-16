import pytest
import torch

from lightop import mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang


Q_COLS = 1536
KV_COLS = 512
ROPE_COLS = 64
EPSILON = 1.0e-5


def _rms_norm_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    value_fp32 = value.float()
    variance = value_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (
        value_fp32
        * torch.rsqrt(variance + epsilon)
        * weight.float()
    ).to(value.dtype)


def _per_token_int8_reference(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    quantized = torch.round(value.float() / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_mla_packed_rms_norm_quant_matches_native(num_tokens: int) -> None:
    torch.manual_seed(20260715)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    packed = torch.randn(
        num_tokens,
        Q_COLS + KV_COLS + ROPE_COLS,
        device=device,
        dtype=dtype,
    )
    q_weight = (
        1.0 + 0.1 * torch.randn(Q_COLS, device=device, dtype=dtype)
    ).contiguous()
    kv_weight = (
        1.0 + 0.1 * torch.randn(KV_COLS, device=device, dtype=dtype)
    ).contiguous()

    original = packed.clone()
    q_reference = _rms_norm_reference(
        original[:, :Q_COLS], q_weight, EPSILON
    )
    kv_reference = _rms_norm_reference(
        original[:, Q_COLS : Q_COLS + KV_COLS], kv_weight, EPSILON
    )
    q_int8_reference, q_scale_reference = _per_token_int8_reference(q_reference)

    q_int8, q_scales = mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
        packed_input=packed,
        q_weight=q_weight,
        kv_weight=kv_weight,
        q_epsilon=EPSILON,
        kv_epsilon=EPSILON,
    )

    torch.testing.assert_close(
        packed[:, :Q_COLS], q_reference, rtol=2.0e-2, atol=2.0e-2
    )
    torch.testing.assert_close(
        packed[:, Q_COLS : Q_COLS + KV_COLS],
        kv_reference,
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    torch.testing.assert_close(
        q_scales, q_scale_reference, rtol=2.0e-3, atol=1.0e-7
    )
    assert torch.max(
        torch.abs(q_int8.to(torch.int16) - q_int8_reference.to(torch.int16))
    ).item() <= 1
    assert torch.equal(
        packed[:, Q_COLS + KV_COLS :],
        original[:, Q_COLS + KV_COLS :],
    )


@pytest.mark.skipif(torch.version.hip is None, reason="LightOp requires ROCm")
def test_mla_packed_rms_norm_quant_zero_input_scale_floor() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    packed = torch.zeros(
        3,
        Q_COLS + KV_COLS + ROPE_COLS,
        device=device,
        dtype=dtype,
    )
    packed[:, Q_COLS + KV_COLS :] = torch.randn(
        3, ROPE_COLS, device=device, dtype=dtype
    )
    rope_reference = packed[:, Q_COLS + KV_COLS :].clone()
    q_weight = torch.ones(Q_COLS, device=device, dtype=dtype)
    kv_weight = torch.ones(KV_COLS, device=device, dtype=dtype)

    q_int8, q_scales = mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
        packed_input=packed,
        q_weight=q_weight,
        kv_weight=kv_weight,
        q_epsilon=EPSILON,
        kv_epsilon=EPSILON,
    )

    assert torch.count_nonzero(q_int8).item() == 0
    assert torch.count_nonzero(packed[:, : Q_COLS + KV_COLS]).item() == 0
    torch.testing.assert_close(
        q_scales,
        torch.full_like(q_scales, 1.0e-10 / 127.0),
        rtol=1.0e-6,
        atol=0.0,
    )
    assert torch.equal(packed[:, Q_COLS + KV_COLS :], rope_reference)
