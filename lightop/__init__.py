import os
import subprocess
import importlib
from pathlib import Path
from torch.utils.cpp_extension import ROCM_HOME
ROOT_DIR = Path(__file__).parent.resolve()

def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None
    
def get_dtk_version(ROCM_HOME):
    dtk_version = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk_version = ''.join(dtk_version.split('.'))
    dtk_version = int(dtk_version)
    return dtk_version

dtk_version = get_dtk_version(ROCM_HOME) #动态获取运行时的dtk版本
from . import build_config
if build_config.BUILD_MIOPEN and dtk_version >= 25042:
    op = importlib.import_module('.op_core_miopen', __name__)   #包含core+miopen
else:
    op = importlib.import_module('.op', __name__)         #只包含core


from .utils import gfx, LIGHTOP_ROOT_DIR, LIGHTOP_ASM_DIR
from .rmsnorm import RMSNorm, RMSNormTorch, RMSNormDropout, RMSNormAddDropout, RMSNormAddDropoutBias, RMSNormAdd,rmsnorm_forward,rmsnorm_backward,rms_norm_dynamic_per_token_quant,fused_rms_norm_contiguous,fused_rms_norm_rope_contiguous,rms_norm_per_token_fp8_quant
from .rmsnorm import fuse_rmsnorm_rope_quant_qkv,fused_add_rms_norm,gemma_fused_add_rmsnorm,gemma_rmsnorm
from .rmsnorm import fuse_fill_rms_x2_concat
from .grid_sample import GridSample, grid_sample_forward, grid_sample_backward
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
from .gemmopt import gemm, gemm_w4a16, w4tow16,gemm_w8a16_fp8,fp8_to_16,moe_gemm_w8a8, moe_gemm_marlin_w8a8, moe_gemm_marlin_w8a8_fp8, moe_gemm_marlin_w8a8_blockwise_fp8, gemm_w8a8_smooth, moe_marlin_w4a16, awq_marlin_repack, gemm_w8a8_asm, gemm_w8a8_wgrad_asm, \
      gemm_w8a8_xgrad_asm, moe_gemm_w4a8, moe_gemm_marlin_w4a8, groupgemm, moe_groupgemm_w4a8, moe_groupgemm_marlin_w4a8, awq_marlin_repack_w4a8, m_grouped_w4a8_gemm_nt_masked, grougemm_w8a8_pertensor_asm, m_grouped_w8a8_gemm_nt_masked, \
      m_grouped_w8a8_gemm_nt_contig_asm, mqa_logits,moe_gemm_marlin_w16a16,moe_gemm_w16a16,m_grouped_bf16_gemm_nt_masked,m_grouped_w16a16_gemm_nt_contig_asm,m_grouped_w8a8_gemm_nt_nopad_contig_asm, \
      grougemm_bf16_fp16, m_grouped_fp8_gemm_nt_masked, fused_bmm
from .gemmopt import gemm_awq_w4a16_marlin, gemm_awq_w4a16_marlin_dequant, awq_gemm_marlin_weight_repack
from .moe_fused_gate import moe_fused_gate
from .fuse_silu_mul_quant import fuse_silu_mul_quant, fuse_silu_mul_quant_ep, fuse_silu_and_mul, fuse_silu_mul_fp8_quant, fuse_silu_mul_fp8_quant_ep, moe_swiglu_dynamic_quant
from .ds_cat import ds_cat
from .blockwise_scaling import blockwise_scaled_cast_transpose
from .config import get_moe_cuda_config, get_moe_cuda_marlin_config, get_moe_cuda_config_w4a8, get_moe_cuda_marlin_config_w4a8, get_moe_cuda_marlin_config_w16a16, \
      get_moe_cuda_config_w16a16, get_moe_groupgemm_config_w4a8, get_moe_groupgemm_marlin_config_w4a8,get_batched_gemm_w8a8_config
from .config_smooth_gemm import gemm_w8a8_smooth_config
from .config_deepgemm import gemm_w8a8_deepgemm_masked_config
from .permute_unpermute import groupgemm_permute, groupgemm_unpermute, fuse_groupgemm_permute_quant
from .moe_align_sum import moe_sum, moe_align_block_size
from .batch_linear_assignment import batch_linear_assignment
from .fill_zeros import fill_zeros
from .fuse_rmsnorm_rope import rms_rotary_embedding_fuse, rms_rotary_embedding_fuse_with_kv_store, rms_rotary_embedding_fuse_with_kv_store_fake, split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
from .reshape_and_cache_cuda import reshape_and_cache_cuda
from .fuse_rms_mrope_cuda import fuse_rms_mrope_cuda
from .wan_quant import destindex_copy_quantize_int8_cuda, destindex_dequantize_int8_cuda
from .upsample import bilinear_upsample, bilinear_upsample_backward
try:
    from .version import version, git_hash, git_branch, dtk, abi, torch_version, dcu_version  # noqa: F401
    __version__, __dcu_version__ = version, dcu_version
except ImportError:
    pass

if dtk_version >= 25042:
    from .miopen.miopen_mha import miopen_scaled_dot_product_attention
    from .miopen.miopen_conv_fusion import miopen_conv_bias, miopen_conv_bias_add, miopen_conv_bias_relu, miopen_conv_bias_add_relu, miopenConvBias, miopenConvBiasAdd, miopenConvBiasRelu, miopenConvBiasAddRelu
    from .miopen.miopen_groupnorm import miopen_groupnorm, miopen_add_groupnorm, miopenGroupNorm
    from .miopen.miopen_layernorm import miopen_layernorm, miopen_add_layernorm, miopen_skip_layernorm, miopenLayerNorm
    from .miopen.miopen_moe import miopen_moe_gather, miopen_moe_scatter_dynamic_quant, miopen_moe_swiglu_dynamic_quant
    from .miopen.miopen_rms import miopen_add_rms_norm_dynamic_quant, miopen_head_rms_norm, miopen_rms_norm_dynamic_quant_without_residual
    from .miopen.miopen_gemm_bias import miopen_gemm_bias
    from .miopen.miopen_transform_tensor import miopen_transform_tensor
    from .miopen.miopen_fused_adam import miopen_adamw
