# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public binding for the source-preserving LMSlim RMSQuant implementation."""

from .._lmslim_native.quantize.quant_ops import lm_faster_rmsquant

__all__ = ["lm_faster_rmsquant"]
