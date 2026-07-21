"""INT8 quantization APIs."""

import torch

from .. import op as lightop_native
from .._lmslim_native.layers.gemm.int8_utils import per_token_quant_int8
from ..wan_quant import (
    destindex_copy_quantize_int8_cuda,
    destindex_dequantize_int8_cuda,
)


def per_token_dynamic_quant_int8(
    x: torch.Tensor, smooth_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply smooth scaling and native per-token dynamic INT8 quantization."""
    if smooth_scale.shape[-1] != x.shape[-1]:
        raise ValueError("smooth_scale and x must have the same last dimension")

    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(
        x.shape[:-1], device=x.device, dtype=torch.float32
    )
    lightop_native.per_token_dynamic_quant_int8(
        x_q, x, scales, smooth_scale
    )
    return x_q, scales


__all__ = [
    "destindex_copy_quantize_int8_cuda",
    "destindex_dequantize_int8_cuda",
    "per_token_dynamic_quant_int8",
    "per_token_quant_int8",
]
