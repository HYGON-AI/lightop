# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_scale(absmax):
    denominator = tl.where(absmax == 0.0, 1.0, absmax)
    return tl.where(absmax == 0.0, 0.0, tl.div_rn(448.0, denominator))


@triton.jit
def _quantize(value, inv, DTYPE: tl.constexpr):
    return tl.clamp(value * inv, -448.0, 448.0).to(DTYPE)


@triton.jit
def _sigmoid_mul_fp8_kernel(
    attn_ptr,
    gate_ptr,
    quant_ptr,
    scale_ptr,
    gate_stride_row,
    gate_stride_head,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    attn_off = row * WIDTH + col
    gate_off = row * gate_stride_row + (col // HEAD_DIM) * gate_stride_head + col % HEAD_DIM
    attn = tl.load(attn_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    gated = (attn * tl.sigmoid(gate)).to(attn_ptr.dtype.element_ty).to(tl.float32)
    absmax = tl.max(tl.abs(gated), 0)
    inv = _inverse_scale(absmax)
    quantized = _quantize(gated, inv, quant_ptr.dtype.element_ty)
    tl.store(quant_ptr + attn_off, quantized, mask=mask)
    tl.store(scale_ptr + row, absmax / 448.0)


@triton.jit
def _gate_segment(attn, gate, col, head_stride, HEAD_DIM: tl.constexpr):
    a = tl.load(attn + col).to(tl.float32)
    g = tl.load(gate + (col // HEAD_DIM) * head_stride + col % HEAD_DIM).to(tl.float32)
    return (a * tl.sigmoid(g)).to(attn.dtype.element_ty).to(tl.float32)


@triton.jit
def _sigmoid_mul_fp8_3072_kernel(
    attn_ptr, gate_ptr, quant_ptr, scale_ptr, gate_stride_row, gate_stride_head,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 1024)
    attn = attn_ptr + row * 3072
    gate = gate_ptr + row * gate_stride_row
    quant = quant_ptr + row * 3072

    x0 = _gate_segment(attn, gate, col, gate_stride_head, HEAD_DIM)
    x1 = _gate_segment(attn, gate, col + 1024, gate_stride_head, HEAD_DIM)
    x2 = _gate_segment(attn, gate, col + 2048, gate_stride_head, HEAD_DIM)
    largest = tl.maximum(tl.maximum(tl.abs(x0), tl.abs(x1)), tl.abs(x2))
    absmax = tl.max(largest, 0)
    inv = _inverse_scale(absmax)

    tl.store(quant + col, _quantize(x0, inv, quant_ptr.dtype.element_ty))
    tl.store(quant + col + 1024, _quantize(x1, inv, quant_ptr.dtype.element_ty))
    tl.store(quant + col + 2048, _quantize(x2, inv, quant_ptr.dtype.element_ty))
    tl.store(scale_ptr + row, absmax / 448.0)


def fuse_sigmoid_mul_fp8_quant(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 attention output after multiplying by sigmoid(gate).

    Attention output is contiguous [tokens, hidden]. Gate is [tokens, hidden]
    or a [tokens, heads, head_dim] view with unit last-dimension stride.
    Returns E4M3FN [tokens, hidden] and FP32 scales [tokens, 1].
    The scale is the BF16-rounded gated row's absmax divided by 448.
    A zero row returns a zero scale.
    The input tensors are read-only.
    """
    if (
        attn_output.ndim != 2
        or torch.version.hip is None
        or not attn_output.is_cuda
        or attn_output.dtype != torch.bfloat16
        or not attn_output.is_contiguous()
        or gate.device != attn_output.device
        or gate.dtype != attn_output.dtype
        or gate.ndim not in (2, 3)
        or gate.stride(-1) != 1
    ):
        raise ValueError("Expected contiguous 2D BF16 attention output and BF16 gate on one GPU")

    tokens, width = attn_output.shape
    if gate.ndim == 2 and gate.shape == attn_output.shape and width > 0:
        head_dim = width
        gate_stride_head = 0
    elif gate.ndim == 3 and gate.shape[0] == tokens and gate.shape[1] * gate.shape[2] == width > 0:
        head_dim = gate.shape[2]
        gate_stride_head = gate.stride(1)
    else:
        raise ValueError("Gate shape must match the attention output")

    quantized = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
    scales = torch.empty((tokens, 1), dtype=torch.float32, device=attn_output.device)
    if tokens:
        args = (
            attn_output,
            gate,
            quantized,
            scales,
            gate.stride(0),
            gate_stride_head,
        )
        with torch.cuda.device(attn_output.device):
            if width == 3072:
                _sigmoid_mul_fp8_3072_kernel[(tokens,)](*args, head_dim, num_warps=4)
            else:
                _sigmoid_mul_fp8_kernel[(tokens,)](
                    *args, width, head_dim, triton.next_power_of_2(width), num_warps=4
                )
    return quantized, scales
