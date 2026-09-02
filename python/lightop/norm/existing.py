"""Direct bindings for existing LightOp normalization APIs."""

from .. import op as lightop_native
from ..groupnorm import CeluGN, GroupNorm
from ..layernorm import (
    AddLn,
    LayerNorm,
    LayerNormMixedFp32Weight,
    fused_add_layer_norm,
    layer_norm,
)
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
    rms_norm,
    rms_norm_per_token_fp8_quant,
    rms_norm_per_token_quant,
    rmsnorm_backward,
    rmsnorm_forward,
)

rms_norm_opt = lightop_native.rms_norm_opt
fused_add_rms_norm_opt = lightop_native.fused_add_rms_norm_opt
layer_norm_opt = lightop_native.layer_norm_opt
fused_add_layer_norm_opt = lightop_native.fused_add_layer_norm_opt
layer_norm_fwd_1pass_opt = lightop_native.layer_norm_fwd_1pass_opt
rms_norm_no_weight = lightop_native.rms_norm_no_weight
rmsnorm_forward_autograd = lightop_native.rmsnorm_forward_autograd

__all__ = [
    "AddLn",
    "CeluGN",
    "fuse_fill_rms_x2_concat",
    "fused_add_rms_norm",
    "fused_add_rms_norm_opt",
    "fused_add_layer_norm_opt",
    "fused_add_layer_norm",
    "fused_rms_norm_contiguous",
    "gemma_fused_add_rmsnorm",
    "gemma_rms_norm_per_token_fp8_quant",
    "gemma_rmsnorm",
    "GroupNorm",
    "LayerNorm",
    "LayerNormMixedFp32Weight",
    "layer_norm_fwd_1pass_opt",
    "layer_norm_opt",
    "layer_norm",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm",
    "rms_norm_no_weight",
    "rms_norm_per_token_fp8_quant",
    "rms_norm_opt",
    "rms_norm_per_token_quant",
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
