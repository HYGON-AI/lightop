# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm plus complete-row INT8 quantization from padded or sliced rows."""

import torch

from .. import op

_native = getattr(op, "rms_norm_strided_per_token_quant_out", None)
if _native is None:
    try:
        from .._strided_rms_quant import rms_norm_strided_per_token_quant_out as _native
    except (ImportError, OSError):
        pass


def is_strided_rms_quant_available():
    return _native is not None


def rms_norm_strided_per_token_quant(input, weight, epsilon=1e-6):
    """Return normalized floating rows, INT8 rows and FP32 [rows,1] scales.

    Read a BF16/FP16 [rows,columns] view with aligned row stride and unit
    column stride. Preserve all source storage, including its padding. The
    native reduction and activation rounding are the same as LightOp's
    rms_norm_dynamic_per_token_quant_sglang; quantization observes those
    rounded floating values. No residual sum, clamp or GEMM is performed.
    """
    if _native is None:
        raise RuntimeError("native strided RMS+INT8 extension is not installed")
    normalized = torch.empty(input.shape, device=input.device, dtype=input.dtype)
    quantized = torch.empty(input.shape, device=input.device, dtype=torch.int8)
    scales = torch.empty((input.shape[0], 1), device=input.device, dtype=torch.float32)
    _native(input, weight, normalized, quantized, scales, epsilon)
    return normalized, quantized, scales
