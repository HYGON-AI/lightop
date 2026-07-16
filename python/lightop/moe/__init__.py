"""Mixture-of-Experts APIs."""

from importlib import import_module

from .config import (
    get_moe_cuda_config,
    get_moe_cuda_config_w16a16,
    get_moe_cuda_config_w4a8,
    get_moe_cuda_marlin_config,
    get_moe_cuda_marlin_config_w16a16,
    get_moe_cuda_marlin_config_w4a16,
    get_moe_cuda_marlin_config_w4a8,
    get_moe_groupgemm_config_w4a8,
    get_moe_groupgemm_marlin_config_w4a8,
)
from .gemm import (
    moe_gemm_w16a16,
    moe_gemm_w4a8,
    moe_gemm_w8a8,
    moe_groupgemm_w4a8,
)
from .marlin import (
    moe_gemm_marlin_w16a16,
    moe_gemm_marlin_w4a8,
    moe_gemm_marlin_w8a8,
    moe_gemm_marlin_w8a8_blockwise_fp8,
    moe_gemm_marlin_w8a8_fp8,
    moe_groupgemm_marlin_w4a8,
    moe_marlin_w4a16,
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

_LMSLIM_NATIVE_APIS = {
    "fused_experts_impl_fp8_marlin",
    "fused_experts_impl_int8",
    "fused_experts_impl_int8_marlin",
    "fused_experts_impl_w4a8",
    "fused_experts_impl_w4a8_marlin",
    "groupgemm_impl_w4a8",
    "groupgemm_impl_w4a8_marlin",
}


def __getattr__(name):
    if name not in _LMSLIM_NATIVE_APIS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.native"), name)
    globals()[name] = value
    return value


__all__ = [
    "fused_experts_impl_fp8_marlin",
    "fused_experts_impl_int8",
    "fused_experts_impl_int8_marlin",
    "fused_experts_impl_w4a8",
    "fused_experts_impl_w4a8_marlin",
    "groupgemm_impl_w4a8",
    "groupgemm_impl_w4a8_marlin",
    "fuse_groupgemm_permute_quant",
    "get_moe_cuda_config",
    "get_moe_cuda_config_w16a16",
    "get_moe_cuda_config_w4a8",
    "get_moe_cuda_marlin_config",
    "get_moe_cuda_marlin_config_w16a16",
    "get_moe_cuda_marlin_config_w4a16",
    "get_moe_cuda_marlin_config_w4a8",
    "get_moe_groupgemm_config_w4a8",
    "get_moe_groupgemm_marlin_config_w4a8",
    "groupgemm_permute",
    "groupgemm_unpermute",
    "moe_align_block_size",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_gemm_marlin_w16a16",
    "moe_gemm_marlin_w4a8",
    "moe_gemm_marlin_w8a8",
    "moe_gemm_marlin_w8a8_blockwise_fp8",
    "moe_gemm_marlin_w8a8_fp8",
    "moe_gemm_w16a16",
    "moe_gemm_w4a8",
    "moe_gemm_w8a8",
    "moe_groupgemm_marlin_w4a8",
    "moe_groupgemm_w4a8",
    "moe_marlin_w4a16",
    "moe_sum",
]
