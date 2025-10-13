import os
from .utils import gfx, LIGHTOP_ROOT_DIR, LIGHTOP_ASM_DIR
from .rmsnorm import RMSNorm, RMSNormTorch, RMSNormDropout, RMSNormAddDropout, RMSNormAddDropoutBias, RMSNormAdd,rmsnorm_forward,rmsnorm_backward,rms_norm_dynamic_per_token_quant
from .layernorm import LayerNorm, LnDropout, LnAddDropout, LnAddDropoutBias, LnDropPath, AddLn
from .droppath import DropPath
from .fuse_bias_swiglu import FuseBiasSwiGLU
from .fuse_rotary_positional_embedding import RotaryEmbedding,apply_rotary_pos_emb_index_fused
from .fuseadddropout import AddDropout
from .fusebiasdropoutadd import FuseBiasDropoutAdd
from .fusebiasgelu import FuseBiasGelu
from .fuseactmode import fuseactmode
# from .fuselinearbias import LinearBias
from .fusereludropout import ReluDropout
from .fusesoftmax import FuseSoftmax,FuseLogSoftmax
from .lossctc import CTCLoss
from .lossrnnt import RNNTLoss
from .scaled_masked_softmax import FusedScaleMaskSoftmax,FusedScaleMaskSoftmaxDropout,AttnMaskType
from .groupnorm import GroupNorm ,CeluGN,GNSilu, groupnorm_backward
from .reduce import sum, mean,topk
from .gemmopt import gemm, gemm_w4a16, w4tow16,gemm_w8a16_fp8,fp8_to_16,moe_gemm_w8a8,gemm_w8a8_smooth, moe_marlin_w4a16, awq_marlin_repack, gemm_w8a8_asm, gemm_w8a8_wgrad_asm, gemm_w8a8_xgrad_asm, moe_gemm_w4a8,moe_gemm_marlin_w4a8,groupgemm
from .moe_fused_gate import moe_fused_gate
from .blockwise_scaling import blockwise_scaled_cast_transpose
from .config import get_moe_cuda_config, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8
from .config_smooth_gemm import gemm_w8a8_smooth_config
from .miopen.miopen_mha import miopen_scaled_dot_product_attention
from .miopen.miopen_conv_fusion import miopen_conv_bias, miopen_conv_bias_add, miopen_conv_bias_relu, miopen_conv_bias_add_relu, miopenConvBias, miopenConvBiasAdd, miopenConvBiasRelu, miopenConvBiasAddRelu
from .miopen.miopen_groupnorm import miopen_groupnorm, miopen_add_groupnorm, miopenGroupNorm
from .miopen.miopen_layernorm import miopen_layernorm, miopen_add_layernorm, miopen_skip_layernorm, miopenLayerNorm
from .miopen.miopen_moe import miopen_moe_gather, miopen_moe_scatter_dynamic_quant, miopen_moe_swiglu_dynamic_quant
from .miopen.miopen_rms import miopen_add_rms_norm_dynamic_quant, miopen_head_rms_norm, miopen_rms_norm_dynamic_quant_without_residual
from .miopen.miopen_gemm_bias import miopen_gemm_bias
from .miopen.miopen_transform_tensor import miopen_transform_tensor

try:
    from .version import version, git_hash, git_branch, dtk, abi, torch_version, dcu_version  # noqa: F401
    __version__, __dcu_version__ = version, dcu_version
except ImportError:
    pass
