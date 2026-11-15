# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Existing FP8 RMSNorm semantics with independent output from strided rows."""

import torch

from .. import op

_native = getattr(op, "rms_norm_strided_per_token_fp8_quant_out", None)
if _native is None:
    try:
        from .._strided_rms_quant import (
            rms_norm_strided_per_token_fp8_quant_out as _native,
        )
    except (ImportError, OSError):
        pass


def is_strided_rms_fp8_quant_available():
    return _native is not None


def rms_norm_strided_per_token_fp8_quant(input, weight, epsilon=1e-6):
    """Preserve padded BF16/FP16 source rows and return norm, E4M3 and FP32 scale.

    Reuse rms_norm_per_token_fp8_quant's arithmetic, reduction and scale floor
    of 1/(448*512). Normalized values round to the activation dtype before FP8
    quantization. Each output has independent contiguous storage.
    """
    if _native is None:
        raise RuntimeError("native strided RMS+FP8 extension is not installed")
    normalized = torch.empty(input.shape, device=input.device, dtype=input.dtype)
    q = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((input.shape[0], 1), device=input.device, dtype=torch.float32)
    _native(input, weight, normalized, q, scales, epsilon)
    return normalized, q, scales
