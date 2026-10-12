# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for existing LightOp QKV APIs."""

from .. import op as lightop_native
from ..fuse_rmsnorm_rope import (
    gammarms_rotary_embedding_fuse_with_kv_store,
    gammarms_rotary_embedding_fuse_with_kv_store_fake,
    mimo_v2_split_rope_vscale_kv_store,
    rms_rotary_embedding_fuse,
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
    split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
)
from ..rmsnorm import (
    fuse_rmsnorm_rope_quant_qkv,
    fused_rms_norm_rope_contiguous,
)

fuse_layernorm_rotary_embedding = (
    lightop_native.fuse_layernorm_rotary_embedding
)
fused_deepseek_v4_qnorm_rope_kvnorm_rope = (
    lightop_native.fused_deepseek_v4_qnorm_rope_kvnorm_rope
)
fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32 = (
    lightop_native.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32
)

gemma_qkv_split_rms_rotary_embedding_fuse = (
    lightop_native.gemma_qkv_split_rms_rotary_embedding_fuse
)

__all__ = [
    "fuse_layernorm_rotary_embedding",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
    "gemma_qkv_split_rms_rotary_embedding_fuse",
    "gammarms_rotary_embedding_fuse_with_kv_store",
    "gammarms_rotary_embedding_fuse_with_kv_store_fake",
    "mimo_v2_split_rope_vscale_kv_store",
    "rms_rotary_embedding_fuse",
    "rms_rotary_embedding_fuse_with_kv_store",
    "rms_rotary_embedding_fuse_with_kv_store_fake",
    "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
    "fuse_rmsnorm_rope_quant_qkv",
    "fused_rms_norm_rope_contiguous",
]
