# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Masked OAI SwiGLU followed by per-row E4M3FN quantization."""

from typing import Optional

import torch

from lightop import op


def swiglu_oai_fp8_quant_masked(
    input: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """Quantize live rows of contiguous BF16 ``[experts, capacity, 2*width]``.

    Split the last axis into gate/up, let ``g = min(gate, limit)``, then
    compute ``g * sigmoid(alpha*g) * (clamp(up, -limit, limit) + 1)`` in FP32.
    Each row uses scale ``max(absmax, 1e-10) / 448`` and E4M3FN output.
    Width must be a multiple of eight in [8, 4096].

    Device int32 counts are clamped to [0, capacity]. Padding in supplied
    outputs is untouched; newly allocated padding is unspecified. Supplied
    outputs must be contiguous and must not share storage with any input or
    each other. Passing persistent buffers is suitable for graph replay with
    updated inputs/counts. Launch selection never reads counts on the host.
    """
    if input.ndim != 3 or input.shape[-1] % 2:
        raise ValueError("Expected [experts, capacity, 2*width]")
    experts, capacity, twice_width = input.shape
    width = twice_width // 2
    if output is None:
        output = torch.empty(
            (experts, capacity, width), device=input.device, dtype=torch.float8_e4m3fn
        )
    if scales is None:
        scales = torch.empty(
            (experts, capacity, 1), device=input.device, dtype=torch.float32
        )
    if experts == 16:
        op.swiglu_oai_fp8_quant_masked_flat(
            input, output, scales, tokens_per_expert, alpha, limit, 256, 256
        )
    else:
        op.swiglu_oai_fp8_quant_masked(
            input, output, scales, tokens_per_expert, alpha, limit, 32, 256
        )
    return output, scales
