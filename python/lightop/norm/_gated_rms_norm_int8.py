# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gated_rmsnorm_int8_kernel(
    X, Z, W, Q, S, eps,
    X_ROW: tl.constexpr,
    Z_TOKEN: tl.constexpr,
    Z_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_HEADS)
    cols = tl.arange(0, HEAD_DIM)
    mask = heads[:, None] < HEADS
    row = token * HEADS + heads
    x = tl.load(X + row[:, None] * X_ROW + cols[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN + heads[:, None] * Z_HEAD + cols[None, :],
                mask, 0).to(tl.float32)
    weight = tl.load(W + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
    gated = (x * rstd[:, None] * weight[None, :]) * (z * tl.sigmoid(z))
    gated = gated.to(tl.bfloat16).to(tl.float32)
    row_absmax = tl.max(tl.abs(gated), axis=1)
    absmax = tl.maximum(tl.max(row_absmax, axis=0), 1e-10)
    quantized = libdevice.nearbyint(gated * (127.0 / absmax)).to(tl.int8)
    tl.store(Q + row[:, None] * HEAD_DIM + cols[None, :], quantized, mask)
    tl.store(S + token, absmax / 127.0)


@triton.jit
def _gated_tile(X, Z, w, token, head, col, eps,
                X_ROW: tl.constexpr, Z_TOKEN: tl.constexpr, Z_HEAD: tl.constexpr):
    x = tl.load(X + (token * 24 + head[:, None]) * X_ROW + col[None, :]).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN + head[:, None] * Z_HEAD + col[None, :]).to(tl.float32)
    gate = z * tl.sigmoid(z)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) * 0.0078125 + eps)
    return ((x * inv[:, None] * w[None, :]) * gate).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _quantize_store(Y, out, inv, head, col):
    q = libdevice.nearbyint(Y * inv).to(tl.int8)
    tl.store(out + head[:, None] * 128 + col[None, :], q)


@triton.jit
def _gated_exact_heads_kernel(X, Z, W, Q, S, eps,
                              X_ROW: tl.constexpr, Z_TOKEN: tl.constexpr, Z_HEAD: tl.constexpr):
    token = tl.program_id(0)
    h = tl.arange(0, 8)
    c = tl.arange(0, 128)
    w = tl.load(W + c).to(tl.float32)
    y0 = _gated_tile(X, Z, w, token, h, c, eps, X_ROW, Z_TOKEN, Z_HEAD)
    y1 = _gated_tile(X, Z, w, token, h + 8, c, eps, X_ROW, Z_TOKEN, Z_HEAD)
    y2 = _gated_tile(X, Z, w, token, h + 16, c, eps, X_ROW, Z_TOKEN, Z_HEAD)
    largest = tl.maximum(tl.maximum(tl.abs(y0), tl.abs(y1)), tl.abs(y2))
    amax = tl.maximum(tl.max(tl.max(largest, 1), 0), 1e-10)
    inv = 127.0 / amax
    out = Q + token * 3072
    _quantize_store(y0, out, inv, h, c)
    _quantize_store(y1, out, inv, h + 8, c)
    _quantize_store(y2, out, inv, h + 16, c)
    tl.store(S + token, amax / 127.0)


@lru_cache(maxsize=None)
def _is_gfx936(device):
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":")[0] == "gfx936"


def gated_rms_norm_per_token_int8_quant(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm and SiLU gate, then quantize each token to INT8.

    Input has shape `(tokens * num_heads, 128)`. Gate has the same shape
    or shape `(tokens, num_heads, 128)` with a unit last-dimension stride.
    Weight is shared across heads and has shape `(128,)`. All three tensors
    use BF16 on the same HCU device. Quantization observes the BF16-rounded
    gated output and returns INT8 activations of shape
    `(tokens, num_heads * 128)` and FP32 scales of shape `(tokens, 1)`.
    """
    if input.ndim != 2 or input.shape[1] != 128 or not 1 <= num_heads <= 64:
        raise ValueError("Expected input with 128 columns and 1 to 64 value heads")
    rows, head_dim = input.shape
    if rows % num_heads:
        raise ValueError("Input rows must contain complete tokens")
    tokens = rows // num_heads
    if (
        torch.version.hip is None
        or not input.is_cuda
        or input.dtype != torch.bfloat16
        or input.stride(-1) != 1
        or gate.device != input.device
        or gate.dtype != input.dtype
        or gate.ndim not in (2, 3)
        or gate.stride(-1) != 1
        or weight.device != input.device
        or weight.dtype != input.dtype
        or weight.shape != (head_dim,)
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected BF16 GPU tensors on one device with a contiguous last dimension")
    if gate.shape == input.shape:
        gate_token_stride, gate_head_stride = num_heads * gate.stride(0), gate.stride(0)
    elif gate.shape == (tokens, num_heads, head_dim):
        gate_token_stride, gate_head_stride = gate.stride(0), gate.stride(1)
    else:
        raise ValueError("Gate shape must match input rows or be [tokens, num_heads, 128]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Expected a finite positive epsilon")

    quantized = torch.empty((tokens, num_heads * head_dim), dtype=torch.int8, device=input.device)
    scale = torch.empty((tokens, 1), dtype=torch.float32, device=input.device)
    if tokens:
        num_warps = 8 if num_heads == 24 and tokens <= 128 else 4
        with torch.cuda.device(input.device):
            if num_heads == 24 and _is_gfx936(input.device):
                if tokens <= 64:
                    num_warps = 8
                elif tokens <= 512:
                    num_warps = 4
                else:
                    num_warps = 2
                _gated_exact_heads_kernel[(tokens,)](
                    input, gate, weight, quantized, scale, epsilon, input.stride(0),
                    gate_token_stride, gate_head_stride, num_warps=num_warps,
                )
            else:
                _gated_rmsnorm_int8_kernel[(tokens,)](
                    input, gate, weight, quantized, scale, epsilon, input.stride(0),
                    gate_token_stride, gate_head_stride, num_heads, triton.next_power_of_2(num_heads),
                    head_dim, num_warps=num_warps,
                )
    return quantized, scale
