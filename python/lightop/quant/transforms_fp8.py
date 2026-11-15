# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""RoPE and layout transforms fused with complete-row E4M3 quantization."""

import torch

from .. import op
from .transforms_int8 import _select_rope_quant_threads

_rope = getattr(op, "rope_per_token_quant_fp8_out", None)
_strided = getattr(op, "per_token_quant_fp8_strided_out", None)
if _rope is None or _strided is None:
    try:
        from .. import _transform_quant
    except (ImportError, OSError):
        pass
    else:
        _rope = _rope or getattr(_transform_quant, "rope_per_token_quant_fp8_out", None)
        _strided = _strided or getattr(
            _transform_quant, "per_token_quant_fp8_strided_out", None
        )


def is_transform_fp8_quant_available():
    return _rope is not None and _strided is not None


def rope_per_token_quant_fp8(
    input,
    frequencies,
    positions,
    groups=1,
    rope_dim=None,
    inverse=False,
    *,
    group_major=False,
    threads=None,
):
    """Round rotated head tails to the input dtype and quantize each head group.

    Read BF16/FP16 [T,H,D], preserve source storage and write independent
    E4M3 [T,G,H/G*D] (or [G,T,H/G*D]) with FP32 per-group row scales.
    Match per_token_quant_fp8: scale=absmax/448 and zero rows have zero scale.
    Positions must be valid indices into contiguous complex64 frequencies.
    threads=None selects calibrated launch geometry; explicit values override it.
    """
    if _rope is None:
        raise RuntimeError("native RoPE+FP8 extension is not installed")
    if input.ndim != 3 or groups <= 0 or input.shape[1] % groups:
        raise ValueError("input must be [T,H,D] with H divisible by groups")
    tokens, heads, dim = input.shape
    width = heads // groups * dim
    rows = (groups, tokens) if group_major else (tokens, groups)
    q = torch.empty((*rows, width), device=input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((*rows, 1), device=input.device, dtype=torch.float32)
    if rope_dim is None:
        rope_dim = frequencies.shape[1] * 2
    if threads is None:
        threads = _select_rope_quant_threads(
            input, groups, rope_dim, torch.float8_e4m3fn, group_major
        )
    _rope(
        input,
        frequencies,
        positions,
        q,
        scales,
        groups,
        rope_dim,
        inverse,
        threads,
        group_major,
    )
    return q, scales


def per_token_quant_fp8_strided(input, *, threads=256):
    """Quantize a complete [T,G,R] token row directly from either group layout.

    Accept token-major or group-major storage with unit inner stride, including
    padded rows. Return E4M3 [T,G*R] and FP32 [T,1] without a floating transpose.
    """
    if _strided is None:
        raise RuntimeError("native strided FP8 extension is not installed")
    if input.ndim != 3:
        raise ValueError("input must be [T,G,R]")
    tokens, groups, dim = input.shape
    q = torch.empty(
        (tokens, groups * dim), device=input.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty((tokens, 1), device=input.device, dtype=torch.float32)
    _strided(input, q, scales, threads)
    return q, scales
