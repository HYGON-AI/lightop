# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Attention APIs."""

# Bumped when the paged_mqa_logits_fp4 signature changes; consumers feature-detect
# on it because an installed extension may predate the current ABI.
PAGED_MQA_LOGITS_FP4_ABI = 3

from ..minimax_qk_norm_rope import minimax_qk_norm_rope, supports_minimax_qk_norm_rope

from ..dsv4_prefill import dsv41_rope_fake_quant_store
from .mhc import mhc_repeat_cp_sglang, supports_mhc_repeat_cp_sglang
from .mqa import (
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    get_paged_mqa_logits_fp4_metadata,
    get_paged_sparse_mqa_logits_metadata,
    mqa_logits,
    paged_mqa_logits,
    paged_mqa_logits_persisten,
    paged_mqa_logits_fp4,
    paged_mqa_logits_route_lens,
    page_mqa_logits_sparse_mask,
    dsv41_candidate_block_table,
    dsv41_select_candidate_blocks,
    dsv41_pair_partners,
    dsv41_build_swa_token_ids,
    dsv41_remap_indices_to_staging,
    dsv41_ragged_topk_postprocess,
)
from .qkv import (
    fuse_layernorm_rotary_embedding,
    fuse_rmsnorm_rope_quant_qkv,
    fused_deepseek_v4_qnorm_rope_kvnorm_rope,
    fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32,
    fused_rms_norm_rope_contiguous,
    gemma_qkv_split_rms_rotary_embedding_fuse,
    gammarms_rotary_embedding_fuse_with_kv_store,
    gammarms_rotary_embedding_fuse_with_kv_store_fake,
    mimo_v2_split_rope_vscale_kv_store,
    rms_rotary_embedding_fuse,
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
    split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
)
from .sparse import (
    convert_req_index_to_global_index,
    fast_topk_transform_fused,
    fast_topk_transform_sparse_mask_fused,
    fast_topk_transform_ragged_fused,
    top_k_per_row_decode,
    top_k_per_row_prefill,
    topk_transform_512,
)
from .softmax import (
    FusedScaleMaskSoftmax,
    FusedScaleMaskSoftmaxDropout,
    FuseLogSoftmax,
    FuseSoftmax,
)
from .rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb_index_fused,
    fuse_rms_mrope_cuda,
    rotary_embedding_deepseek_fuse,
    rotary_embedding_fuse,
)

__all__ = [
    "minimax_qk_norm_rope",
    "supports_minimax_qk_norm_rope",

    "FusedScaleMaskSoftmax",
    "FusedScaleMaskSoftmaxDropout",
    "FuseLogSoftmax",
    "FuseSoftmax",
    "apply_rotary_pos_emb_index_fused",
    "convert_req_index_to_global_index",
    "fast_topk_transform_fused",
    "fast_topk_transform_sparse_mask_fused",
    "fast_topk_transform_ragged_fused",
    "fp8_fp4_mqa_logits",
    "fuse_layernorm_rotary_embedding",
    "fuse_rms_mrope_cuda",
    "fuse_rmsnorm_rope_quant_qkv",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
    "fused_rms_norm_rope_contiguous",
    "gemma_qkv_split_rms_rotary_embedding_fuse",
    "get_paged_mqa_logits_metadata",
    "get_paged_mqa_logits_fp4_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "gammarms_rotary_embedding_fuse_with_kv_store",
    "gammarms_rotary_embedding_fuse_with_kv_store_fake",
    "mhc_repeat_cp_sglang",
    "mimo_v2_split_rope_vscale_kv_store",
    "mqa_logits",
    "paged_mqa_logits",
    "paged_mqa_logits_persisten",
    "paged_mqa_logits_fp4",
    "paged_mqa_logits_route_lens",
    "dsv41_candidate_block_table",
    "dsv41_select_candidate_blocks",
    "dsv41_pair_partners",
    "dsv41_build_swa_token_ids",
    "dsv41_remap_indices_to_staging",
    "dsv41_ragged_topk_postprocess",
    "dsv41_rope_fake_quant_store",
    "page_mqa_logits_sparse_mask",
    "rms_rotary_embedding_fuse",
    "rms_rotary_embedding_fuse_with_kv_store",
    "rms_rotary_embedding_fuse_with_kv_store_fake",
    "RotaryEmbedding",
    "rotary_embedding_deepseek_fuse",
    "rotary_embedding_fuse",
    "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
    "supports_mhc_repeat_cp_sglang",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_transform_512",
]

from .mqa import (
    fp8_fp4_paged_sparse_mqa_logits,
    prefill_sparse_mqa_fused,
    sparse_fp8_fp4_paged_mqa_logits,
)

__all__ = list(dict.fromkeys([*__all__, "fp8_fp4_paged_sparse_mqa_logits", "sparse_fp8_fp4_paged_mqa_logits", "prefill_sparse_mqa_fused"]))

# Older runtime op wheels may predate these optional, unrelated helpers.
from .. import op as _attention_native
for _helper in ("dsv41_build_swa_token_ids", "dsv41_remap_indices_to_staging", "dsv41_ragged_topk_postprocess"):
    if not hasattr(_attention_native, _helper):
        globals().pop(_helper, None)
        __all__ = [name for name in __all__ if name != _helper]
del _helper, _attention_native
