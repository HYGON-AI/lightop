# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Core GEMM and batched GEMM API exports."""

from ..gemmopt import fused_bmm, gemm

__all__ = ["fused_bmm", "gemm"]
