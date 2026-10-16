# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision GEMM with fused RMS statistics (``lightop.gemm_ops``)."""

from typing import Optional

from torch import Tensor

from . import op


def hc_mix_stats(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Optional[Tensor]:
    """Compute an RMS-normalized projection using two internal HCU kernels.

    Args:
        x: Contiguous BF16 tensor ``[M,K]``, with M >= 0 and K > 0.
        weight: Contiguous FP32 tensor ``[N,K]``, N >= 0, on the same
            gfx928/gfx936/gfx938 GPU as x. Unaligned storage-offset views
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
        across M and uses no floating-point atomics. gfx938 is runtime-tested;
        gfx928/936 have architecture-specific implementations pending hardware
        validation. Empty outputs require no kernel launches.
    """
    return op.hc_mix_stats(x, weight, eps)
