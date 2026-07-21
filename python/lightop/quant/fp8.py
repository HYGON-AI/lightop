"""FP8 quantization and conversion APIs backed by native LightOp kernels."""

from typing import Optional, Tuple

import torch

from .. import op as lightop_native
from ..gemmopt import fp8_to_16

quantize_nope_fp8_rope_bf16_pack = (
    lightop_native.quantize_nope_fp8_rope_bf16_pack
)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token group with the native LightOp FP8 kernel."""
    if column_major_scales:
        raise NotImplementedError(
            "native LightOp per_token_group_quant_fp8 only supports "
            "row-major scales"
        )

    if out_q is None:
        out_q = torch.empty_like(x, device=x.device, dtype=dtype)
    elif out_q.shape != x.shape or out_q.dtype != dtype:
        raise ValueError("out_q must match x.shape and the requested dtype")

    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    lightop_native.per_token_group_quant_fp8(
        out_q,
        x,
        scales,
        group_size,
        eps,
        False,
    )
    return out_q, scales


def per_token_quant_fp8(x):
    """Quantize each token with the native LightOp FP8 kernel."""
    x_q = torch.empty_like(
        x, device=x.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(
        x.shape[:-1] + (1,), device=x.device, dtype=torch.float32
    )
    lightop_native.per_token_quant_fp8(x_q, x, scales)
    return x_q, scales


__all__ = [
    "fp8_to_16",
    "per_token_group_quant_fp8",
    "per_token_quant_fp8",
    "quantize_nope_fp8_rope_bf16_pack",
]
