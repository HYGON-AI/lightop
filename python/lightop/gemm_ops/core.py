"""Core GEMM and batched GEMM API exports."""

from ..gemmopt import fused_bmm, gemm

__all__ = ["fused_bmm", "gemm"]
