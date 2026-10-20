# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE Marlin APIs."""

from ..gemmopt import (
    moe_gemm_marlin_w16a16,
    moe_gemm_marlin_w4a8,
    moe_gemm_marlin_w8a8,
    moe_gemm_marlin_w8a8_blockwise_fp8,
    moe_gemm_marlin_w8a8_fp8,
    moe_groupgemm_marlin_w4a8,
    moe_marlin_w4a16,
)

__all__ = [
    "moe_gemm_marlin_w16a16",
    "moe_gemm_marlin_w4a8",
    "moe_gemm_marlin_w8a8",
    "moe_gemm_marlin_w8a8_blockwise_fp8",
    "moe_gemm_marlin_w8a8_fp8",
    "moe_groupgemm_marlin_w4a8",
    "moe_marlin_w4a16",
]
