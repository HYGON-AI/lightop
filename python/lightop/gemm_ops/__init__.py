# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public GEMM and Grouped GEMM APIs."""

from importlib import import_module

from ..hc_mix_stats import (
    hc_mix_stats_sinkhorn_bf16x3,
    hc_mix_stats_sinkhorn_deepgemm,
    split_bf16_hc_weight,
    split_tf32_hc_weight,
)
from .core import fused_bmm, gemm, hc_mix_stats
from .grouped import (
    grougemm_bf16_fp16,
    grougemm_w8a8_pertensor_asm,
    groupgemm,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
    m_grouped_w16a16_gemm_nt_contig_asm,
    m_grouped_w4a8_gemm_nt_masked,
    m_grouped_w8a8_gemm_nt_contig_asm,
    m_grouped_w8a8_gemm_nt_masked,
    m_grouped_w8a8_gemm_nt_nopad_contig_asm,
)
from .quantized import (
    awq_gemm_marlin_weight_repack,
    awq_marlin_repack,
    awq_marlin_repack_w4a8,
    gemm_awq_w4a16_marlin,
    gemm_awq_w4a16_marlin_dequant,
    gemm_w4a16,
    gemm_w8a16_fp8,
    gemm_w8a8_asm,
    gemm_w8a8_smooth,
    gemm_w8a8_wgrad_asm,
    gemm_w8a8_xgrad_asm,
)

_NATIVE_APIS = {
    "BlockSize",
    "hipblaslt_w8a8_blockwise_gemm",
    "hipblaslt_w8a8_channelwise_gemm",
    "hipblaslt_w8a8_channelwise_gemm_kme",
    "hipblaslt_w8a8_gemm",
    "rocblas_scaled_mm_nn",
    "triton_int8_gemm_helper",
    "triton_scaled_mm",
}


def __getattr__(name):
    if name not in _NATIVE_APIS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.native"), name)
    globals()[name] = value
    return value

__all__ = [
    "awq_gemm_marlin_weight_repack",
    "awq_marlin_repack",
    "awq_marlin_repack_w4a8",
    "fused_bmm",
    "gemm",
    "hc_mix_stats",
    "hc_mix_stats_sinkhorn_bf16x3",
    "hc_mix_stats_sinkhorn_deepgemm",
    "gemm_awq_w4a16_marlin",
    "gemm_awq_w4a16_marlin_dequant",
    "gemm_w4a16",
    "gemm_w8a16_fp8",
    "gemm_w8a8_asm",
    "gemm_w8a8_smooth",
    "gemm_w8a8_wgrad_asm",
    "gemm_w8a8_xgrad_asm",
    "grougemm_bf16_fp16",
    "grougemm_w8a8_pertensor_asm",
    "groupgemm",
    "hipblaslt_w8a8_blockwise_gemm",
    "hipblaslt_w8a8_channelwise_gemm",
    "hipblaslt_w8a8_channelwise_gemm_kme",
    "hipblaslt_w8a8_gemm",
    "m_grouped_bf16_gemm_nt_masked",
    "m_grouped_fp8_gemm_nt_masked",
    "m_grouped_w16a16_gemm_nt_contig_asm",
    "m_grouped_w4a8_gemm_nt_masked",
    "m_grouped_w8a8_gemm_nt_contig_asm",
    "m_grouped_w8a8_gemm_nt_masked",
    "m_grouped_w8a8_gemm_nt_nopad_contig_asm",
    "rocblas_scaled_mm_nn",
    "split_bf16_hc_weight",
    "split_tf32_hc_weight",
    "triton_int8_gemm_helper",
    "triton_scaled_mm",
]
