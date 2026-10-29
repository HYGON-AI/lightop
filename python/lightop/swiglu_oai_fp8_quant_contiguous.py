# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Contiguous BF16 OAI activation and per-token FP8 E4M3FN quantization.

The BF16 activation rounding and zero-row scale=0 are part of the contract.
This API uses split [gate | up] layout; it is distinct from masked expert APIs.
"""

from . import op


def swiglu_oai_fp8_quant_contiguous(input, alpha=1.702, beta=1.0, limit=7.0):
    """Return FP8 [M,I] and FP32 [M,1] for contiguous BF16 [M,2I]."""
    native = getattr(op, "swiglu_oai_fp8_quant_contiguous", None)
    if native is None:
        # An isolated native extension supports incremental validation with an
        # existing LightOp build. A normal full build exposes the op directly.
        from .lightop_oai_contig import swiglu_oai_fp8_quant_contiguous as native
    return tuple(native(input, alpha, beta, limit))
