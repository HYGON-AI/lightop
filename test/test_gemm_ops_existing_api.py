"""Compatibility checks for existing LightOp GEMM category exports."""

from importlib import import_module

import pytest

import lightop
import lightop.gemm_ops

EXISTING_TOP_LEVEL_APIS = [
    "awq_gemm_marlin_weight_repack",
    "awq_marlin_repack",
    "awq_marlin_repack_w4a8",
    "fused_bmm",
    "gemm",
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
    "m_grouped_bf16_gemm_nt_masked",
    "m_grouped_fp8_gemm_nt_masked",
    "m_grouped_w16a16_gemm_nt_contig_asm",
    "m_grouped_w4a8_gemm_nt_masked",
    "m_grouped_w8a8_gemm_nt_contig_asm",
    "m_grouped_w8a8_gemm_nt_masked",
    "m_grouped_w8a8_gemm_nt_nopad_contig_asm",
]


@pytest.mark.parametrize("name", EXISTING_TOP_LEVEL_APIS)
def test_existing_gemm_api_is_direct_reexport(name: str) -> None:
    assert getattr(lightop.gemm_ops, name) is getattr(lightop, name)


def test_importing_gemm_ops_does_not_shadow_top_level_gemm() -> None:
    assert callable(lightop.gemm)
    assert lightop.gemm_ops.gemm is lightop.gemm


def test_removed_gemm_subpackage_cannot_shadow_top_level_function() -> None:
    public_gemm = lightop.gemm

    with pytest.raises(ModuleNotFoundError):
        import_module("lightop.gemm")

    assert lightop.gemm is public_gemm
    assert callable(lightop.gemm)
