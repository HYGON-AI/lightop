# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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
    out_scale: Optional[torch.Tensor] = None,
    use_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token group with optional output buffers."""
    if column_major_scales:
        raise NotImplementedError(
            "native LightOp per_token_group_quant_fp8 only supports "
            "row-major scales"
        )

    expected_scale_shape = x.shape[:-1] + (x.shape[-1] // group_size,)

    if out_q is None:
        out_q = torch.empty_like(x, device=x.device, dtype=dtype)
    elif (
        out_q.shape != x.shape
        or out_q.device != x.device
        or out_q.dtype != dtype
        or not out_q.is_contiguous()
    ):
        raise ValueError(
            "out_q must be contiguous and match x.shape, x.device, "
            "and the requested dtype"
        )

    if out_scale is None:
        out_scale = torch.empty(
            expected_scale_shape,
            device=x.device,
            dtype=torch.float32,
        )
    elif (
        out_scale.shape != expected_scale_shape
        or out_scale.device != x.device
        or out_scale.dtype != torch.float32
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            "out_scale must be contiguous float32 and match the expected "
            "scale shape and x.device"
        )

    lightop_native.per_token_group_quant_fp8(
        out_q,
        x,
        out_scale,
        group_size,
        eps,
        use_ue8m0,
    )
    return out_q, out_scale


def per_token_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token with optional caller-provided output buffers."""
    expected_scale_shape = x.shape[:-1] + (1,)

    if out_q is None:
        out_q = torch.empty_like(x, device=x.device, dtype=dtype)
    elif (
        out_q.shape != x.shape
        or out_q.device != x.device
        or out_q.dtype != dtype
        or not out_q.is_contiguous()
    ):
        raise ValueError(
            "out_q must be contiguous and match x.shape, x.device, "
            "and the requested dtype"
        )

    if out_scale is None:
        out_scale = torch.empty(
            expected_scale_shape,
            device=x.device,
            dtype=torch.float32,
        )
    elif (
        out_scale.shape != expected_scale_shape
        or out_scale.device != x.device
        or out_scale.dtype != torch.float32
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            "out_scale must be contiguous float32 and match the expected "
            "scale shape and x.device"
        )

    lightop_native.per_token_quant_fp8(out_q, x, out_scale)
    return out_q, out_scale


__all__ = [
    "fp8_to_16",
    "per_token_group_quant_fp8",
    "per_token_quant_fp8",
    "quantize_nope_fp8_rope_bf16_pack",
]
