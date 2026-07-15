"""Compatibility checks for existing LightOp normalization exports."""

import pytest

import lightop
import lightop.norm


EXISTING_NORM_APIS = [
    "AddLn",
    "CeluGN",
    "fuse_fill_rms_x2_concat",
    "fused_add_rms_norm",
    "fused_rms_norm_contiguous",
    "gemma_fused_add_rmsnorm",
    "gemma_rms_norm_per_token_fp8_quant",
    "gemma_rmsnorm",
    "GroupNorm",
    "LayerNorm",
    "LayerNormMixedFp32Weight",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_token_fp8_quant",
    "rms_norm_per_token_quant",
    "RMSNorm",
    "rmsnorm_backward",
    "rmsnorm_forward",
    "RMSNormAdd",
    "RMSNormAddDropout",
    "RMSNormAddDropoutBias",
    "RMSNormDropout",
    "RMSNormTorch",
]


@pytest.mark.parametrize("name", EXISTING_NORM_APIS)
def test_existing_norm_api_is_direct_reexport(name):
    assert getattr(lightop.norm, name) is getattr(lightop, name)


def test_norm_public_api_inventory():
    assert set(lightop.norm.__all__) == set(EXISTING_NORM_APIS) | {
        "lm_faster_rmsquant"
    }
