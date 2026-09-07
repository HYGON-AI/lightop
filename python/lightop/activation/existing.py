"""Direct bindings for existing LightOp activation and dropout APIs."""

from .. import op as lightop_native
from ..droppath import DropPath
from ..fuse_bias_swiglu import FuseBiasSwiGLU
from ..fuse_silu_mul_quant import (
    fuse_silu_and_mul,
    fuse_silu_and_mul_ep,
    fuse_silu_mul_fp8_quant,
    fuse_silu_mul_fp8_quant_ep,
    fuse_silu_mul_per_token_quant,
    fuse_silu_mul_quant,
    fuse_silu_mul_quant_ep,
    relu2,
)
from ..fuse_situ_mul_quant import (
    fuse_situ_mul_quant_contiguous,
    fuse_situ_mul_quant_ep,
)
from ..fuseadddropout import AddDropout
from ..fusebiasdropoutadd import FuseBiasDropoutAdd
from ..fusebiasgelu import FuseBiasGelu
from ..fusereludropout import ReluDropout
from ..groupnorm import GNSilu
from ..layernorm import LnAddDropout, LnAddDropoutBias, LnDropout, LnDropPath

gelu_and_mul_opt = lightop_native.gelu_and_mul_opt
gelu_tanh_and_mul_opt = lightop_native.gelu_tanh_and_mul_opt
silu_and_mul_opt = lightop_native.silu_and_mul_opt

__all__ = [
    "AddDropout",
    "DropPath",
    "fuse_silu_and_mul",
    "fuse_silu_and_mul_ep",
    "fuse_silu_mul_fp8_quant",
    "fuse_silu_mul_fp8_quant_ep",
    "fuse_silu_mul_per_token_quant",
    "fuse_silu_mul_quant",
    "fuse_silu_mul_quant_ep",
    "fuse_situ_mul_quant_contiguous",
    "fuse_situ_mul_quant_ep",
    "FuseBiasDropoutAdd",
    "FuseBiasGelu",
    "FuseBiasSwiGLU",
    "GNSilu",
    "gelu_and_mul_opt",
    "gelu_tanh_and_mul_opt",
    "LnAddDropout",
    "LnAddDropoutBias",
    "LnDropout",
    "LnDropPath",
    "relu2",
    "ReluDropout",
    "silu_and_mul_opt",
]
