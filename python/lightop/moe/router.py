# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BF16 router projection on gfx938."""


def bf16_router_gemm(x, weight):
    """Return FP32 router logits x @ weight.T with shape [64, 128].

    Requires contiguous, 16-byte-aligned BF16 tensors x[64, 6144] and
    weight[128, 6144] on the same gfx938 device. Accumulation and output
    are FP32. Scoring, normalization and top-k selection belong to the
    caller. Unsupported inputs raise RuntimeError.
    """
    from .. import _router_gemm

    return _router_gemm.forward(x, weight)
