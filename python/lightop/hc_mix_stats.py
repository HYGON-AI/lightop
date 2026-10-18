# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision GEMM with fused RMS statistics (``lightop.gemm_ops``)."""

from typing import Optional

import torch
from torch import Tensor

from . import op


def hc_mix_stats(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Optional[Tensor]:
    """Compute an RMS-normalized projection using two internal HCU kernels.

    Args:
        x: Contiguous BF16 tensor ``[M,K]``, with M >= 0 and K > 0.
        weight: Contiguous FP32 tensor ``[N,K]``, N >= 0, on the same
            supported HCU GPU as x. Unaligned storage-offset views
            are supported; unresolved negative view bits are not.
        eps: Positive finite epsilon representable as a nonzero FP32 value.

    Returns:
        New contiguous FP32 ``[M,N]`` tensor on x.device, approximating
        ``(x.float() @ weight.T) * rsqrt(mean(x.float()**2, -1)[:,None] + eps)``.
        Returns ``None`` before allocation or launch for unsupported architecture,
        device, dtype, shape, layout/view, or epsilon.
        Supported empty M/N returns an empty Tensor, never ``None``.

    Raises:
        RuntimeError: Allocation, device-query or kernel execution failures.
            OOM and execution errors are not converted to ``None``.

    Notes:
        Inference-only: call under ``torch.no_grad()`` or ``torch.inference_mode()``.
        There is no backward implementation or runtime gradient-mode check;
        gradient requirements do not select the Torch fallback.
        Weights are split online as hi=BF16(weight),
        lo=BF16((weight-hi.float())*256). The first kernel accumulates
        high/low GEMM and row-square partials over slices of 256 K elements;
        the second reduces them and applies RMS scaling. Workspace is internal,
        per-call and Torch-allocated. Both kernels run on the current stream
        and support graph replay with updated inputs and weights.

        Arbitrary K/N tails and empty M/N are supported; no input is modified
        or implicitly copied. Launch limits: M <= 2**31-1, K <= 65535*256,
        N <= 65535*24, subject to available memory. Finite, BF16-representable
        high/low weights and non-overflowing FP32 reductions are the numerical
        domain; NaN/Inf/overflow are not sanitized. Reduction order is fixed
        across M and uses no floating-point atomics. Empty outputs require no
        kernel launches. See the operator documentation for hardware validation.
    """
    return op.hc_mix_stats(x, weight, eps)


def _check_split_weight(weight: Tensor):
    if (weight.dtype != torch.float32 or weight.ndim != 2
            or not weight.is_contiguous() or weight.is_neg()):
        raise ValueError("weight must be a contiguous FP32 matrix with resolved view bits")


@torch.no_grad()
def split_bf16_hc_weight(weight: Tensor):
    """Prepare unscaled BF16 high/middle/low once after loading FP32 weights."""
    _check_split_weight(weight)
    high = weight.bfloat16()
    residual = weight - high.float()
    middle = residual.bfloat16()
    low = (residual - middle.float()).bfloat16()
    return high, middle, low


@torch.no_grad()
def split_tf32_hc_weight(weight: Tensor):
    """Prepare a bit-truncated TF32 high component and an FP32 residual."""
    _check_split_weight(weight)
    high = (weight.view(torch.int32) & -8192).view(torch.float32)
    return high, weight - high


def hc_mix_stats_sinkhorn_bf16x3(x, weight_parts, scale, base,
                               sinkhorn_iters, rms_eps, hc_eps, hc_mult=4):
    """Fused compensated BF16 projection, RMS and Sinkhorn.

    x: contiguous BF16 [M,20480]. Supported M ranges depend on the device;
    see the operator documentation. weight_parts: three contiguous BF16
    [24,20480] tensors, prepared by split_bf16_hc_weight.
    scale/base: contiguous FP32 [3]/[24]. All tensors must share x.device.
    hc_mult=4; positive sinkhorn_iters and positive finite FP32 epsilons.
    Returns FP32 pre[M,4], post[M,4], comb[M,4,4]. Inference only.
    Invalid metadata raises; no implicit copies, fallback or backward.
    Current stream and allocator-owned scratch support graph replay.
    """
    return op.hc_mix_stats_sinkhorn_bf16x3(
        x, weight_parts, scale, base, sinkhorn_iters, rms_eps, hc_eps, hc_mult)


def hc_mix_stats_sinkhorn_deepgemm(x_flat, weight_parts, hc_scale, hc_base,
                                 sinkhorn_iters, rms_eps, hc_eps, hc_mult=4):
    """Fused compensated TF32 projection, RMS and Sinkhorn.

    Uses native TF32 MMAC with no DeepGEMM dependency. x_flat is contiguous
    BF16 [M,20480], including M=0. weight_parts is the two contiguous FP32
    [24,20480] tensors from split_tf32_hc_weight. The residual is explicitly
    rounded to TF32 in the kernel. Other contracts match the BF16x3 entry.
    """
    return op.hc_mix_stats_sinkhorn_deepgemm(
        x_flat, weight_parts, hc_scale, hc_base, sinkhorn_iters, rms_eps, hc_eps, hc_mult)
