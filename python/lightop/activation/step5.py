# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Step5 post-SiLU clamp and BF16-rounded rowwise FP8 quantization."""
import torch

from .. import op


def step5_swiglustep_fp8_quant(input, expert_ids, *, output=None, scales=None, limit=7.):
    """BF16 [M,3072] -> E4M3FN [M,1536], FP32 [M,1] scales.

    Computes BF16(min(silu(gate),limit)*clamp(up,-limit,limit)) before
    rowwise amax. Every negative expert ID writes zero FP8 and 1e-10/448
    scale, including holes between experts. Router weights are not applied.
    Uses the current stream; no input mutation, scalar readback or JIT.
    """
    if output is None:
        output = torch.empty((input.shape[0], 1536), device=input.device, dtype=torch.float8_e4m3fn)
    if scales is None:
        scales = torch.empty((input.shape[0], 1), device=input.device, dtype=torch.float32)
    op.step5_swiglustep_fp8_quant(input, expert_ids, output, scales, float(limit))
    return output, scales
