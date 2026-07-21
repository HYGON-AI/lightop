"""Normalization and residual-fusion APIs."""

from importlib import import_module

from .existing import (
    AddLn,
    CeluGN,
    GroupNorm,
    LayerNorm,
    LayerNormMixedFp32Weight,
    layer_norm_fwd_1pass_opt,
    RMSNorm,
    RMSNormAdd,
    RMSNormAddDropout,
    RMSNormAddDropoutBias,
    RMSNormDropout,
    RMSNormTorch,
    fuse_fill_rms_x2_concat,
    fused_add_rms_norm,
    fused_add_rms_norm_opt,
    fused_rms_norm_contiguous,
    gemma_fused_add_rmsnorm,
    gemma_rms_norm_per_token_fp8_quant,
    gemma_rmsnorm,
    rms_norm_dynamic_per_token_quant,
    rms_norm_no_weight,
    rms_norm_per_token_fp8_quant,
    rms_norm_per_token_quant,
    rms_norm_opt,
    rmsnorm_backward,
    rmsnorm_forward,
    rmsnorm_forward_autograd,
)

_NATIVE_APIS = {"lm_faster_rmsquant"}


def __getattr__(name):
    if name not in _NATIVE_APIS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.native"), name)
    globals()[name] = value
    return value


__all__ = [
    "AddLn",
    "CeluGN",
    "fuse_fill_rms_x2_concat",
    "fused_add_rms_norm",
    "fused_add_rms_norm_opt",
    "fused_rms_norm_contiguous",
    "gemma_fused_add_rmsnorm",
    "gemma_rms_norm_per_token_fp8_quant",
    "gemma_rmsnorm",
    "GroupNorm",
    "LayerNorm",
    "LayerNormMixedFp32Weight",
    "layer_norm_fwd_1pass_opt",
    "lm_faster_rmsquant",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_no_weight",
    "rms_norm_per_token_fp8_quant",
    "rms_norm_per_token_quant",
    "rms_norm_opt",
    "RMSNorm",
    "rmsnorm_backward",
    "rmsnorm_forward",
    "rmsnorm_forward_autograd",
    "RMSNormAdd",
    "RMSNormAddDropout",
    "RMSNormAddDropoutBias",
    "RMSNormDropout",
    "RMSNormTorch",
]
