"""Direct bindings for existing LightOp normalization APIs."""

from .. import op as lightop_native
from ..groupnorm import CeluGN, GroupNorm
from ..layernorm import AddLn, LayerNorm, LayerNormMixedFp32Weight
from ..rmsnorm import (
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

rms_norm_opt = lightop_native.rms_norm_opt
fused_add_rms_norm_opt = lightop_native.fused_add_rms_norm_opt

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
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_token_fp8_quant",
    "rms_norm_opt",
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
