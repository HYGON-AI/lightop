# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""INT8 quantization APIs."""

import os

import torch

from ._per_token_int8_triton import per_token_quant_int8_segments_kernel as _segments_int8

from .. import op as lightop_native
from .._lmslim_native.layers.gemm.int8_utils import (
    per_token_group_quant_int8,
    per_token_quant_int8 as _per_token_quant_int8_triton,
)
from ..wan_quant import (
    destindex_copy_quantize_int8_cuda,
    destindex_dequantize_int8_cuda,
)

# Older installed LightOp packages may use the independently built extension.
_wide_int8 = getattr(lightop_native, "per_token_quant_int8_minimax_out", None)
if _wide_int8 is None:
    try:
        from ..lightop_per_token_int8_candidate import per_token_quant_int8_minimax_out as _wide_int8
    except ImportError:
        pass
_wide_int8_enabled = os.environ.get("LIGHTOP_PER_TOKEN_INT8_NATIVE", "1") == "1"
_int8_method = os.environ.get("LIGHTOP_PER_TOKEN_INT8_METHOD", "segments")
if _int8_method not in ("segments", "native", "triton"):
    raise ValueError("LIGHTOP_PER_TOKEN_INT8_METHOD must be segments, native, or triton")


def per_token_quant_int8(x):
    """Apply symmetric per-token INT8 quantization.

    The DeepSeek-V4 prefill workload uses a native HIP fast path for its
    BF16 ``[262144, 128]`` query matrix. MiniMax BF16 dense rows of widths
    3072, 6144 and 12288 use an exact three-segment reduction.
    LIGHTOP_PER_TOKEN_INT8_METHOD selects segments, native, or original triton. Other inputs retain the existing
    Triton implementation and its public behavior.
    """
    if (
        x.ndim >= 1
        and x.is_cuda
        and torch.version.hip is not None
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] == 128
        and x.numel() // 128 == 262144
    ):
        x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
        scales = torch.empty(
            x.shape[:-1] + (1,), device=x.device, dtype=torch.float32
        )
        lightop_native.per_token_quant_int8_out(x_q, x, scales)
        return x_q, scales
    if (
        _wide_int8_enabled and _int8_method == "segments"
        and x.ndim >= 2 and x.is_cuda and torch.version.hip is not None
        and x.dtype == torch.bfloat16 and x.is_contiguous()
        and x.shape[-1] in (3072, 6144, 12288)
    ):
        rows, width = x.numel() // x.shape[-1], x.shape[-1]
        x_q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
        if rows:
            warps = (2 if rows >= 512 else 8) if width == 3072 else (4 if width == 6144 else 8)
            _segments_int8[(rows,)](x, x_q, scales, D=width, B=width // 3,
                                   num_warps=warps, num_stages=1)
        return x_q, scales
    if (
        _wide_int8_enabled and _int8_method == "native" and _wide_int8 is not None
        and x.ndim >= 2 and x.is_cuda and torch.version.hip is not None
        and x.dtype == torch.bfloat16 and x.is_contiguous()
        and x.shape[-1] in (3072, 6144)
        and x.numel() // x.shape[-1] >= 1024
        and x.data_ptr() % 16 == 0
    ):
        x_q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
        _wide_int8(x_q, x, scales, 64)
        return x_q, scales
    return _per_token_quant_int8_triton(x)


def per_token_dynamic_quant_int8(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
    round_after_smooth: bool = False,
    limit: float = -1.0,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Apply optional clamp, smooth scaling, and per-token INT8 quantization.

    ``limit=-1`` preserves the historical smooth+quant behavior.  A
    non-negative limit clamps the activation before smooth scaling.  The
    smoothed values can optionally be rounded to the input dtype before
    quantization.
    """
    if smooth_scale.shape[-1] != x.shape[-1]:
        raise ValueError("smooth_scale and x must have the same last dimension")

    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(
        x.shape[:-1], device=x.device, dtype=torch.float32
    )
    if (
        weights is not None
        and getattr(lightop_native, "per_token_quant_int8_out_supports_gate", False)
        and torch.version.hip is not None
        and not round_after_smooth and limit < 0
        and x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == 128
        and x.is_contiguous() and x.data_ptr() % 16 == 0
        and smooth_scale.dtype == torch.float32 and smooth_scale.is_contiguous()
        and smooth_scale.numel() == 128 and smooth_scale.data_ptr() % 32 == 0
        and smooth_scale.device == x.device
        and weights.dtype == torch.float32 and weights.is_contiguous()
        and weights.numel() == scales.numel() and weights.device == x.device
    ):
        adjusted_weights = torch.empty_like(weights, memory_format=torch.contiguous_format)
        lightop_native.per_token_quant_int8_out(
            x_q, x, scales, smooth_scale, weights, adjusted_weights
        )
        return x_q, scales, adjusted_weights
    lightop_native.per_token_dynamic_quant_int8(
        x_q, x, scales, smooth_scale, round_after_smooth, limit
    )
    if weights is not None:
        return x_q, scales, weights * scales.reshape_as(weights)
    return x_q, scales


__all__ = [
    "destindex_copy_quantize_int8_cuda",
    "destindex_dequantize_int8_cuda",
    "per_token_dynamic_quant_int8",
    "per_token_group_quant_int8",
    "per_token_quant_int8",
]
