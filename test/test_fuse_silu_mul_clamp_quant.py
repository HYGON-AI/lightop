import pytest
import torch
import torch.nn.functional as F

from lightop import fuse_silu_mul_clamp_quant


def _reference(input: torch.Tensor, limit: float):
    hidden = input.shape[-1] // 2
    gate = input[:, :hidden].clamp(max=limit)
    up = input[:, hidden:].clamp(min=-limit, max=limit)
    activated = F.silu(gate) * up

    absmax = activated.float().abs().amax(dim=-1, keepdim=True)
    absmax = absmax.clamp_min(1.0e-10)
    scales = absmax / 127.0
    quantized = torch.round(activated.float() / scales)
    quantized = quantized.clamp(-128, 127).to(torch.int8)
    return quantized, scales


@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("hidden", [128, 768, 1536, 2048, 12288])
def test_fuse_silu_mul_clamp_quant_glm_shapes(tokens, hidden):
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

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(input, limit)
    expected_q, expected_scale = _reference(input, limit)

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
