"""Compatibility checks for existing LightOp normalization exports."""

import pytest

import lightop
from lightop import op as lightop_native
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

NATIVE_NORM_APIS = [
    "fused_add_rms_norm_opt",
    "layer_norm_fwd_1pass_opt",
    "rms_norm_no_weight",
    "rms_norm_opt",
    "rmsnorm_forward_autograd",
]


@pytest.mark.parametrize("name", EXISTING_NORM_APIS)
def test_existing_norm_api_is_direct_reexport(name):
    assert getattr(lightop.norm, name) is getattr(lightop, name)


@pytest.mark.parametrize("name", NATIVE_NORM_APIS)
def test_native_norm_api_is_direct_binding(name):
    assert getattr(lightop.norm, name) is getattr(lightop_native, name)


def test_norm_public_api_inventory():
    assert set(lightop.norm.__all__) == set(
        EXISTING_NORM_APIS + NATIVE_NORM_APIS
    ) | {"lm_faster_rmsquant"}
