# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE finalize: gather expert outputs, dequantize, and weighted-sum
by routing weight, writing the result back in original token order.

For token t with top-k experts, this computes:
    out[t] = sum_k topk_weights[t, k] * dequant(data[src_to_dst[t*topk+k]])

in bf16 x hidden tiles. The Triton kernel below does the gather,
dequantization, and the weighted top-k reduction in one launch, with no
intermediate tensor written to device memory; this op is memory-bound
(roughly 0.5 FLOP/byte), so avoiding that extra traffic is the whole
optimization.

Falls back to a plain-torch reference path for inputs that don't satisfy
the kernel's layout assumptions (unit column stride). That fallback is
ordinary dispatch, not error handling: an actual Triton launch/compile
failure is raised rather than silently degraded to the slow path, because
this op is meant to run close to peak memory bandwidth and a silent
fallback would hide that kind of regression from the caller.
"""

import torch
import triton
import triton.language as tl


_BLOCK_H_BF16, _NUM_WARPS_BF16 = 512, 4
_BLOCK_H_FP8, _NUM_WARPS_FP8 = 2048, 2

_FMT_BF16 = 0
_FMT_FP8 = 1

_TOK_CHUNK = 1024


@triton.jit
def _finalize_kernel(
    data_ptr,
    scale_ptr,
    weights_ptr,
    src_to_dst_ptr,
    out_ptr,
    H,
    stride_d,
    TOPK: tl.constexpr,
    FMT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program owns one token and one contiguous hidden-dim tile."""
    token = tl.program_id(0).to(tl.int64)
    cols = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    mask = cols < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for k in range(TOPK):
        # src_to_dst is loaded as int64: at the largest benchmark shape,
        # row * stride_d exceeds the signed 32-bit range.
        row = tl.load(src_to_dst_ptr + token * TOPK + k).to(tl.int64)
        weight = tl.load(weights_ptr + token * TOPK + k).to(tl.float32)
        value = tl.load(
            data_ptr + row * stride_d + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if FMT == 1:
            value *= tl.load(scale_ptr + row).to(tl.float32)
        acc += weight * value

    tl.store(out_ptr + token * H + cols, acc.to(tl.bfloat16), mask=mask)


def _finalize_fallback(fmt, data, scale, topk_weights, src_to_dst):
    """Plain-torch reference path, chunked over tokens to bound peak memory
    at large shapes (each token's output only depends on its own rows)."""
    seq, topk = topk_weights.shape
    s2d = src_to_dst.view(seq, topk)
    w = topk_weights.float()
    out = None
    for t0 in range(0, seq, _TOK_CHUNK):
        t1 = min(t0 + _TOK_CHUNK, seq)
        idx = s2d[t0:t1].reshape(-1)
        rows = data.index_select(0, idx).float()
        if fmt == "fp8":
            rows = rows * scale.index_select(0, idx)[:, None]
        if out is None:
            out = torch.empty(seq, rows.shape[1], dtype=torch.bfloat16, device=data.device)
        c = t1 - t0
        out[t0:t1] = torch.bmm(w[t0:t1].unsqueeze(1), rows.view(c, topk, -1)).squeeze(1)
    return out


def build_src_to_dst(topk_ids: torch.Tensor) -> torch.Tensor:
    """[seq, topk] expert ids -> [seq*topk] int64 inverse permutation.

    Uses the same stable-sort-by-expert-id convention as moe_init_routing,
    which produced the grouped expert outputs this op reads from.
    """
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    s2d = torch.empty_like(order)
    s2d[order] = torch.arange(order.numel(), device=order.device)
    return s2d


def moe_finalize_routing(fmt, data, scale, topk_weights, src_to_dst):
    """Fused MoE finalize, returning ``[seq, hidden]`` bf16.

    fmt: "bf16" or "fp8". data: [M, H] bf16, or [M, H] float8_e4m3fn, rows
    in grouped-expert order. scale: None for bf16, [M] fp32 for fp8.
    topk_weights: [seq, topk] fp32 routing weights. src_to_dst: [seq*topk]
    int, from build_src_to_dst.
    """
    if fmt not in ("bf16", "fp8"):
        raise ValueError(fmt)

    seq, topk = topk_weights.shape
    layout_supported = (
        data.is_cuda and topk_weights.is_cuda and src_to_dst.is_cuda
        and data.ndim == 2 and topk > 0 and seq > 0
        and src_to_dst.ndim == 1 and src_to_dst.numel() == seq * topk
        and data.stride(1) == 1 and topk_weights.stride(1) == 1
        and (fmt == "bf16" or scale is not None)
    )
    if not layout_supported:
        return _finalize_fallback(fmt, data, scale, topk_weights, src_to_dst)

    weights = topk_weights.float().contiguous()
    s2d = src_to_dst.to(torch.int64).contiguous()
    if scale is not None:
        scale = scale.contiguous()
    hidden = data.shape[1]
    out = torch.empty((seq, hidden), dtype=torch.bfloat16, device=data.device)

    if fmt == "fp8":
        block_h, num_warps = _BLOCK_H_FP8, _NUM_WARPS_FP8
    else:
        block_h, num_warps = _BLOCK_H_BF16, _NUM_WARPS_BF16

    grid = (seq, triton.cdiv(hidden, block_h))
    _finalize_kernel[grid](
        data,
        out if scale is None else scale,  # dummy valid pointer when unused
        weights,
        s2d,
        out,
        hidden,
        data.stride(0),
        TOPK=topk,
        FMT=_FMT_FP8 if fmt == "fp8" else _FMT_BF16,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return out


__all__ = ["build_src_to_dst", "moe_finalize_routing"]
