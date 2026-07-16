"""Mixture-of-Experts APIs."""

from .config import get_moe_cuda_config, get_moe_cuda_marlin_config
from .gemm import moe_gemm_w8a8
from .marlin import (
    moe_gemm_marlin_w8a8,
    moe_gemm_marlin_w8a8_blockwise_fp8,
    moe_gemm_marlin_w8a8_fp8,
)
from .permutation import (
    fuse_groupgemm_permute_quant,
    groupgemm_permute,
    groupgemm_unpermute,
)
from .routing import (
    moe_align_block_size,
    moe_fused_gate,
    moe_fused_gate_sqrtsoftplus,
    moe_sum,
)

__all__ = [
    "fuse_groupgemm_permute_quant",
    "get_moe_cuda_config",
    "get_moe_cuda_marlin_config",
    "groupgemm_permute",
    "groupgemm_unpermute",
    "moe_align_block_size",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_gemm_marlin_w8a8",
    "moe_gemm_marlin_w8a8_blockwise_fp8",
    "moe_gemm_marlin_w8a8_fp8",
    "moe_gemm_w8a8",
    "moe_sum",
]
