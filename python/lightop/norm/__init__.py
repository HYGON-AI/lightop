"""Normalization and residual-fusion APIs."""

from importlib import import_module

from .existing import (
    AddLn,
    CeluGN,
    GroupNorm,
    LayerNorm,
    LayerNormMixedFp32Weight,
    RMSNorm,
    RMSNormAdd,
    RMSNormAddDropout,
    RMSNormAddDropoutBias,
    RMSNormDropout,
    RMSNormTorch,
    fuse_fill_rms_x2_concat,
    fused_add_rms_norm,
    fused_rms_norm_contiguous,
    gemma_fused_add_rmsnorm,
    gemma_rms_norm_per_token_fp8_quant,
    gemma_rmsnorm,
    rms_norm_dynamic_per_token_quant,
    rms_norm_per_token_fp8_quant,
    rms_norm_per_token_quant,
    rmsnorm_backward,
    rmsnorm_forward,
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
    "fused_rms_norm_contiguous",
    "gemma_fused_add_rmsnorm",
    "gemma_rms_norm_per_token_fp8_quant",
    "gemma_rmsnorm",
    "GroupNorm",
    "LayerNorm",
    "LayerNormMixedFp32Weight",
    "lm_faster_rmsquant",
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
