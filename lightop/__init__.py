import os
from .utils import gfx, LIGHTOP_ROOT_DIR, LIGHTOP_ASM_DIR
from .rmsnorm import RMSNorm, RMSNormTorch, RMSNormDropout, RMSNormAddDropout, RMSNormAddDropoutBias, RMSNormAdd,rmsnorm_forward,rmsnorm_backward,rms_norm_dynamic_per_token_quant,fused_rms_norm_contiguous
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
from .groupnorm import GroupNorm ,CeluGN,GNSilu
from .reduce import sum, mean,topk
from .gemmopt import gemm, gemm_w4a16, w4tow16,gemm_w8a16_fp8,fp8_to_16,moe_gemm_w8a8,gemm_w8a8_smooth, moe_marlin_w4a16, awq_marlin_repack, gemm_w8a8_asm, gemm_w8a8_wgrad_asm, \
      gemm_w8a8_xgrad_asm, moe_gemm_w4a8, moe_gemm_marlin_w4a8, groupgemm, moe_groupgemm_w4a8, moe_groupgemm_marlin_w4a8, awq_marlin_repack_w4a8
from .moe_fused_gate import moe_fused_gate
from .fuse_silu_mul_quant import fuse_silu_mul_quant
from .ds_cat import ds_cat
from .blockwise_scaling import blockwise_scaled_cast_transpose
from .config import get_moe_cuda_config, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8, get_moe_groupgemm_config_w4a8, get_moe_groupgemm_marlin_config_w4a8
from .config_smooth_gemm import gemm_w8a8_smooth_config
from .permute_unpermute import groupgemm_permute, groupgemm_unpermute
from .moe_sum import moe_sum, moe_align_block_size
try:
    from .version import version, git_hash, git_branch, dtk, abi, torch_version, dcu_version  # noqa: F401
    __version__, __dcu_version__ = version, dcu_version
except ImportError:
    pass
