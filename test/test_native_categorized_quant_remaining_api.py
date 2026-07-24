"""Runtime checks for the remaining categorized quantization operator."""

import pytest
import torch

from lightop import op as lightop_native
from lightop.quant import quantize_nope_fp8_rope_bf16_pack


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device is required")


def test_quantize_nope_pack_is_direct_native_binding() -> None:
    assert quantize_nope_fp8_rope_bf16_pack is (
        lightop_native.quantize_nope_fp8_rope_bf16_pack
    )


def test_quantize_nope_pack_matches_native_outputs() -> None:
    _require_cuda()
    torch.manual_seed(41)
    inputs = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)

    outputs = quantize_nope_fp8_rope_bf16_pack(inputs, 1e-8)
    native_outputs = lightop_native.quantize_nope_fp8_rope_bf16_pack(
        inputs, 1e-8
    )
    torch.cuda.synchronize()

    assert [tensor.shape for tensor in outputs] == [
        (3, 448),
        (3, 64),
        (3, 7),
    ]
    assert [tensor.dtype for tensor in outputs] == [
        torch.uint8,
        torch.bfloat16,
        torch.uint8,
    ]
    for output, native_output in zip(outputs, native_outputs):
        assert torch.equal(output, native_output)
