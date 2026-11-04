# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public binding for the source-preserving LMSlim SiLU-mul quant API."""

from .._lmslim_native.quantize.quant_ops import lm_fuse_silu_mul_quant
from .. import op as lightop_native

__all__ = ["lm_fuse_silu_mul_quant"]

if hasattr(lightop_native, "fused_clamp_smooth"):
    fused_clamp_smooth = lightop_native.fused_clamp_smooth
    __all__.append("fused_clamp_smooth")
