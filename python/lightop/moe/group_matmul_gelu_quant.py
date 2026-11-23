# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Grouped matmul + GELU + dynamic int8 quantization for MoE expert FFNs.

Computes, for tokens already grouped by expert (see moe_init_routing):
    y = gelu(x @ w[expert])
    q, scale = per_row_dynamic_int8_quant(y)

BF16 dispatches each expert's row slice to torch.mm (rocBLAS), which is
faster across this op's benchmark shapes than a single torch._grouped_mm
call. FP8 uses a Triton grouped-GEMM kernel: this ROCm/DTK build has no
grouped FP8 GEMM API (torch._scaled_grouped_mm), and looping torch._scaled_mm
per expert (hipBLASLt) is dominated by per-call launch overhead at these
shapes, so a single Triton launch covering every expert is faster.

GELU activation and per-row int8 quantization are fused into one Triton
kernel so the GEMM output is read once and no fp32 intermediate is
materialized between activation and quantization.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice as hip_libdevice


# ---------------------------------------------------------------------------
# Fused GELU + per-row dynamic int8 quantization epilogue
# ---------------------------------------------------------------------------

@triton.jit
def _gelu_quant_kernel(
    y_ptr,
    q_ptr,
    scale_ptr,
    N: tl.constexpr,
    stride_ym,
    stride_yn,
    stride_qm,
    stride_qn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    values = tl.load(
        y_ptr + row * stride_ym + cols * stride_yn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    act = 0.5 * values * (1.0 + tl.erf(values * 0.7071067811865476))
    amax = tl.max(tl.abs(act), axis=0)
    scale = tl.maximum(amax, 1.0e-12) / 127.0
    quant = hip_libdevice.rint(act / scale)
    quant = tl.minimum(tl.maximum(quant, -127.0), 127.0)

    tl.store(
        q_ptr + row * stride_qm + cols * stride_qn,
        quant.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + row, scale)


def _gelu_quant_epilogue(y: torch.Tensor, num_warps: int):
    rows, cols = y.shape
    q = torch.empty((rows, cols), dtype=torch.int8, device=y.device)
    scale = torch.empty(rows, dtype=torch.float32, device=y.device)
    if rows == 0:
        return q, scale
    if cols <= 0 or cols > 4096:
        raise ValueError(f"supported column count is 1..4096, got {cols}")

    block_n = triton.next_power_of_2(cols)
    _gelu_quant_kernel[(rows,)](
        y,
        q,
        scale,
        N=cols,
        stride_ym=y.stride(0),
        stride_yn=y.stride(1),
        stride_qm=q.stride(0),
        stride_qn=q.stride(1),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return q, scale


# ---------------------------------------------------------------------------
# FP8 grouped matmul (Triton)
# ---------------------------------------------------------------------------

_FP8_BM, _FP8_BN, _FP8_BK = 64, 128, 64


@triton.jit
def _grouped_mm_fp8_kernel(
    x_ptr, w_ptr, y_ptr, xs_ptr, ws_ptr,
    tile_expert_ptr, tile_row_start_ptr, tile_row_end_ptr,
    K,
    stride_xm, stride_xk, stride_we, stride_wk, stride_wn,
    stride_ym, stride_yn, stride_wse,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    e = tl.load(tile_expert_ptr + pid_t)
    if e < 0:
        return
    e64 = e.to(tl.int64)
    row_start = tl.load(tile_row_start_ptr + pid_t)
    row_end = tl.load(tile_row_end_ptr + pid_t)

    # Row/expert offsets are int64: at the largest benchmark shape
    # (M*K ~ 1e10, E*K*N ~ 2e10) both exceed the signed 32-bit range.
    rm = (row_start + tl.arange(0, BM)).to(tl.int64)
    m_mask = rm < row_end
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + e64 * stride_we + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc += tl.dot(a, b)
        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    xs = tl.load(xs_ptr + rm, mask=m_mask, other=0.0)
    ws = tl.load(ws_ptr + e64 * stride_wse + rn)
    acc = acc * xs[:, None] * ws[None, :]

    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


def _build_tile_table(group_list: torch.Tensor, m: int):
    """Tile -> (expert, row range) table, built on-device (no host sync).

    Sized to an upper bound of cdiv(m, BM) + num_experts; padding entries
    are marked expert=-1 and the kernel exits early for them.
    """
    e_num = group_list.shape[0]
    t_max = triton.cdiv(m, _FP8_BM) + e_num
    cnt = group_list.to(torch.int64)
    tiles = (cnt + _FP8_BM - 1) // _FP8_BM
    cum_tiles = torch.cumsum(tiles, 0)
    row_end_e = torch.cumsum(cnt, 0)
    row_start_e = row_end_e - cnt

    t = torch.arange(t_max, device=group_list.device)
    e = torch.searchsorted(cum_tiles, t, right=True)
    valid = e < e_num
    e_c = e.clamp(max=e_num - 1)
    tile_in_e = t - (cum_tiles[e_c] - tiles[e_c])

    tile_expert = torch.where(valid, e_c, -1).to(torch.int32)
    tile_row_start = (row_start_e[e_c] + tile_in_e * _FP8_BM).to(torch.int64)
    tile_row_end = row_end_e[e_c].to(torch.int64)
    return tile_expert, tile_row_start, tile_row_end, t_max


def _grouped_mm_fp8(x, w, group_list, x_scale, w_scale):
    m, k = x.shape
    e_num, k2, n = w.shape
    assert k == k2 and k % _FP8_BK == 0 and n % _FP8_BN == 0

    y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    tile_expert, tile_row_start, tile_row_end, t_max = _build_tile_table(group_list, m)

    grid = (t_max, n // _FP8_BN)
    _grouped_mm_fp8_kernel[grid](
        x, w, y, x_scale, w_scale,
        tile_expert, tile_row_start, tile_row_end,
        k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), w_scale.stride(0),
        BM=_FP8_BM, BN=_FP8_BN, BK=_FP8_BK,
    )
    return y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def group_matmul_gelu_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    group_list: torch.Tensor,
    num_warps: int = 4,
):
    """BF16 grouped matmul + GELU + per-row int8 quantization.

    x: [M, K] bf16, rows grouped by expert (see moe_init_routing). w:
    [E, K, N] bf16. group_list: [E] int, row count per expert, sums to M.
    Returns (q, scale): q is [M, N] int8, scale is [M] fp32.
    """
    assert x.is_cuda and w.is_cuda and group_list.is_cuda

    counts = group_list.tolist()
    y = torch.empty((x.shape[0], w.shape[2]), dtype=torch.bfloat16, device=x.device)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            torch.mm(x[start:start + count], w[expert], out=y[start:start + count])
        start += count
    return _gelu_quant_epilogue(y, num_warps)


def group_matmul_gelu_quant_fp8(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    group_list: torch.Tensor,
    num_warps: int = 4,
):
    """FP8 grouped matmul + GELU + per-row int8 quantization.

    x_q: [M, K] float8_e4m3fn, rows grouped by expert. x_scale: [M] fp32
    per-token scale. w_q: [E, K, N] float8_e4m3fn. w_scale: [E, N] fp32
    per-output-channel scale. group_list: [E] int, row count per expert.
    Returns (q, scale): q is [M, N] int8, scale is [M] fp32.
    """
    assert x_q.dtype == torch.float8_e4m3fn and w_q.dtype == torch.float8_e4m3fn

    y = _grouped_mm_fp8(x_q, w_q, group_list, x_scale, w_scale)
    return _gelu_quant_epilogue(y, num_warps)


__all__ = ["group_matmul_gelu_quant", "group_matmul_gelu_quant_fp8"]
