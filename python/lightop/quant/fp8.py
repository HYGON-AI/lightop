# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FP8 quantization and conversion APIs backed by native LightOp kernels."""

from typing import Optional, Tuple

import torch

from .. import op as lightop_native
from ..gemmopt import fp8_to_16

quantize_nope_fp8_rope_bf16_pack = lightop_native.quantize_nope_fp8_rope_bf16_pack


PER_TOKEN_QUANT_FP8_VERSION = getattr(lightop_native, "PER_TOKEN_QUANT_FP8_VERSION", 1)


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
            "native LightOp per_token_group_quant_fp8 only supports row-major scales"
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
    *,
    quant_ndim: int = 1,
    threads: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize complete token rows with optional caller-provided buffers.

    quant_ndim=1 retains the existing last-dimension reduction and output shape.
    quant_ndim=2 treats [T,G,R] as T complete G*R rows and returns contiguous
    [T,G*R] plus FP32 [T,1] scales. It reads token/group-major or padded storage
    directly, preserving source storage and avoiding a floating transpose/copy.

    The strided path supports BF16/FP16 to E4M3, unit inner stride and complete
    rows up to 16384 columns. Padded 2D rows also support quant_ndim=1. E4M3 uses
    absmax/448, including zero scale for zero rows. threads controls the strided
    path; contiguous quant_ndim=1 retains the original dtype/launch dispatch.
    Empty strided batches return empty outputs without launching a kernel.
    """
    if quant_ndim == 1 and x.ndim >= 1:
        output_shape = x.shape
    elif quant_ndim == 2 and x.ndim == 3:
        output_shape = (x.shape[0], x.shape[1] * x.shape[2])
    else:
        raise ValueError("quant_ndim must be 1, or 2 for [T,G,R] input")
    if output_shape[-1] <= 0:
        raise ValueError("token rows must have positive width")
    expected_scale_shape = output_shape[:-1] + (1,)

    if out_q is None:
        if quant_ndim == 1 and x.is_contiguous():
            out_q = torch.empty_like(x, device=x.device, dtype=dtype)
        else:
            out_q = torch.empty(output_shape, device=x.device, dtype=dtype)
    elif (
        out_q.shape != output_shape
        or out_q.device != x.device
        or out_q.dtype != dtype
        or not out_q.is_contiguous()
    ):
        raise ValueError(
            "out_q must be contiguous and match the output shape, x.device, "
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

    if quant_ndim == 1 and (x.is_contiguous() or PER_TOKEN_QUANT_FP8_VERSION < 2):
        lightop_native.per_token_quant_fp8(out_q, x, out_scale)
    else:
        if PER_TOKEN_QUANT_FP8_VERSION < 2:
            raise RuntimeError("native FP8 quantization needs quant_ndim support")
        lightop_native.per_token_quant_fp8(out_q, x, out_scale, quant_ndim, threads)
    return out_q, out_scale


__all__ = [
    "fp8_to_16",
    "per_token_group_quant_fp8",
    "per_token_quant_fp8",
    "quantize_nope_fp8_rope_bf16_pack",
]
