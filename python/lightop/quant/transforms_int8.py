# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native transforms followed by complete-row symmetric INT8 quantization.

The regular setup.py/wheel build provides these operators through lightop.op.
Callers can check is_transform_quant_available() when supporting older packages.
"""

from functools import lru_cache

import torch

from .. import op

_rope = getattr(op, "rope_per_token_quant_int8_out", None)
_scale = getattr(op, "scale_per_token_quant_int8_out", None)
_scale_only = getattr(op, "scale_per_token_quant_int8_only_out", None)


# Shape calibration is independent of model and decode-stage names. Unknown
# devices and layouts keep the general 256-thread implementation.
_ROPE_THREAD_CONFIGS = {
    ("gfx936", torch.int8, False): {
        **dict.fromkeys(range(87, 91), 128),
        **dict.fromkeys(range(91, 106), 64),
    },
    ("gfx938", torch.float8_e4m3fn, True): {
        **dict.fromkeys(range(64, 129), 64),
    },
}


@lru_cache(maxsize=None)
def _rope_device_arch(device_index):
    properties = torch.cuda.get_device_properties(device_index)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


def _select_rope_quant_threads(input, groups, rope_dim, quant_dtype, group_major=False):
    if (
        not input.is_cuda
        or input.dtype != torch.bfloat16
        or input.shape[1:] != (64, 512)
        or groups != 8
        or rope_dim != 64
        or input.stride(2) != 1
    ):
        return 256
    key = (_rope_device_arch(input.device.index), quant_dtype, group_major)
    return _ROPE_THREAD_CONFIGS.get(key, {}).get(input.shape[0], 256)


def is_transform_quant_available():
    return _rope is not None and _scale is not None and _scale_only is not None


def rope_per_token_quant_int8(
    input,
    frequencies,
    positions,
    groups=1,
    rope_dim=None,
    inverse=False,
    output_pitch=None,
    *,
    threads=None,
):
    """RoPE on each head's tail, then INT8 quantize each complete head group.

    Input is BF16/FP16 [T,H,D], with positive row/head strides and unit inner
    stride. Frequencies are complex64 [positions,rope_dim/2]. Positions must
    be valid table indices, as for the original RoPE operation. The rotated
    tail rounds to the input dtype before absmax/quantization. INT8 uses
    round-half-away-from-zero; scales have a 1e-10 absmax floor.

    Returns independent INT8 [T,groups,H/groups*D] and FP32 [T,groups,1].
    output_pitch can pad each INT8 row for a consumer's matrix layout.
    The input and its padding are preserved. No GEMM is performed.
    threads=None selects a calibrated shape/device configuration; an explicit
    thread count keeps the caller-selected launch geometry. Complete group
    widths must be at most 16384, with even head and rotary dimensions.
    Positions may be INT32 or INT64. Empty token batches return empty outputs
    without launching a kernel.
    """
    if _rope is None:
        raise RuntimeError("native RoPE+INT8 extension is not installed")
    if input.ndim != 3 or groups <= 0 or input.shape[1] % groups:
        raise ValueError("input must be [T,H,D] with H divisible by groups")
    width = input.shape[1] // groups * input.shape[2]
    pitch = width if output_pitch is None else output_pitch
    if pitch < width:
        raise ValueError("output_pitch must be at least the group width")
    rope_dim = frequencies.shape[1] * 2 if rope_dim is None else rope_dim
    if threads is None:
        threads = _select_rope_quant_threads(input, groups, rope_dim, torch.int8)
    storage = torch.empty(
        (input.shape[0], groups, pitch), device=input.device, dtype=torch.int8
    )
    quantized = storage[..., :width]
    scales = torch.empty(
        (input.shape[0], groups, 1), device=input.device, dtype=torch.float32
    )
    _rope(
        input,
        frequencies,
        positions,
        quantized,
        scales,
        groups,
        rope_dim,
        inverse,
        threads,
    )
    return quantized, scales


def scale_per_token_quant_int8(
    accum,
    input_scale,
    weight_scale,
    *,
    output_dtype=torch.bfloat16,
    materialize_output=True,
    group_major=False,
    splits=None,
    threads=512,
):
    """Scale grouped INT32 accumulators, round, and quantize a full token row.

    Accumulators are [T,G,R] (or [G,T,R] with group_major), input scales
    [T,G,1], and weight scales contain G*R FP32 channels. Compute
    (FP32(accum)*input_scale)*weight_scale, then round to output_dtype
    before computing absmax and round-to-nearest-even INT8 over all G*R
    channels. Returns floating [T,G,R], INT8 [T,G*R], FP32 [T,1].
    With materialize_output=False the floating result is None: rounding
    still occurs in registers, with no floating allocation or store.

    For small batches, independent CTAs repeat the read-only reduction and
    write disjoint output slices; this needs no cross-CTA synchronization
    or scratch and remains safe under changing-input graph replay. Full token
    widths must be at most 16384. Empty token batches return empty outputs
    without launching a kernel. Inputs and outputs have independent storage.
    """
    if _scale is None:
        raise RuntimeError("native scale+INT8 extension is not installed")
    if accum.ndim != 3:
        raise ValueError("accum must be a grouped 3D INT32 tensor")
    tokens, groups = (
        (accum.shape[1], accum.shape[0]) if group_major else accum.shape[:2]
    )
    rank = accum.shape[2]
    width = groups * rank
    if splits is None:
        splits = 1
    output = (
        torch.empty((tokens, groups, rank), device=accum.device, dtype=output_dtype)
        if materialize_output
        else None
    )
    quantized = torch.empty((tokens, width), device=accum.device, dtype=torch.int8)
    scales = torch.empty((tokens, 1), device=accum.device, dtype=torch.float32)
    if materialize_output:
        _scale(
            accum,
            input_scale,
            weight_scale,
            output,
            quantized,
            scales,
            group_major,
            splits,
            threads,
        )
    else:
        if _scale_only is None:
            raise RuntimeError("native quantized-only scale extension is not installed")
        _scale_only(
            accum,
            input_scale,
            weight_scale,
            quantized,
            scales,
            output_dtype,
            group_major,
            splits,
            threads,
        )
    return output, quantized, scales
