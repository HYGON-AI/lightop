# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks for existing LightOp attention exports."""

import pytest

import lightop
from lightop.attention import mqa
from lightop import op as lightop_native
import lightop.attention


EXISTING_ATTENTION_APIS = [
    "FusedScaleMaskSoftmax",
    "FusedScaleMaskSoftmaxDropout",
    "FuseLogSoftmax",
    "FuseSoftmax",
    "apply_rotary_pos_emb_index_fused",
    "fast_topk_transform_fused",
    "fast_topk_transform_ragged_fused",
    "fuse_rms_mrope_cuda",
    "fuse_rmsnorm_rope_quant_qkv",
    "fused_rms_norm_rope_contiguous",
    "gammarms_rotary_embedding_fuse_with_kv_store",
    "gammarms_rotary_embedding_fuse_with_kv_store_fake",
    "mimo_v2_split_rope_vscale_kv_store",
    "mqa_logits",
    "rms_rotary_embedding_fuse",
    "rms_rotary_embedding_fuse_with_kv_store",
    "rms_rotary_embedding_fuse_with_kv_store_fake",
    "RotaryEmbedding",
    "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
    "topk_transform_512",
]

PREEXISTING_CATEGORY_ONLY_APIS = [
    "rotary_embedding_deepseek_fuse",
]

NEW_ATTENTION_APIS = [
    "fuse_layernorm_rotary_embedding",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
    "get_paged_mqa_logits_metadata",
    "paged_mqa_logits",
    "fp8_fp4_mqa_logits",
    "convert_req_index_to_global_index",
    "dsv41_build_swa_token_ids",
    "dsv41_candidate_block_table",
    "dsv41_pair_partners",
    "dsv41_ragged_topk_postprocess",
    "dsv41_remap_indices_to_staging",
    "dsv41_rope_fake_quant_store",
    "dsv41_select_candidate_blocks",
    "fast_topk_transform_sparse_mask_fused",
    "fp8_fp4_paged_sparse_mqa_logits",
    "gemma_qkv_split_rms_rotary_embedding_fuse",
    "get_paged_mqa_logits_fp4_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "mhc_repeat_cp_sglang",
    "minimax_qk_norm_rope",
    "page_mqa_logits_sparse_mask",
    "paged_mqa_logits_fp4",
    "paged_mqa_logits_persisten",
    "paged_mqa_logits_route_lens",
    "prefill_sparse_mqa_fused",
    "rotary_embedding_fuse",
    "sparse_fp8_fp4_paged_mqa_logits",
    "supports_mhc_repeat_cp_sglang",
    "supports_minimax_qk_norm_rope",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
]


@pytest.mark.parametrize("name", EXISTING_ATTENTION_APIS)
def test_existing_attention_api_is_direct_reexport(name):
    assert getattr(lightop.attention, name) is getattr(lightop, name)


def test_preexisting_native_attention_api_is_direct_binding():
    assert lightop.attention.rotary_embedding_deepseek_fuse is (
        lightop_native.rotary_embedding_deepseek_fuse
    )


@pytest.mark.parametrize(
    "name", ["get_paged_mqa_logits_metadata", "paged_mqa_logits"]
)
def test_paged_mqa_api_reuses_existing_wrapper(name):
    assert getattr(lightop.attention, name) is getattr(mqa, name)


@pytest.mark.parametrize("name", NEW_ATTENTION_APIS[:3])
def test_new_native_attention_api_is_direct_binding(name):
    assert getattr(lightop.attention, name) is getattr(lightop_native, name)


def test_attention_public_api_inventory():
    assert set(lightop.attention.__all__) == set(
        EXISTING_ATTENTION_APIS
        + PREEXISTING_CATEGORY_ONLY_APIS
        + NEW_ATTENTION_APIS
    )
