"""Compatibility checks for existing LightOp attention exports."""

import pytest

import lightop
import lightop.attention


EXISTING_ATTENTION_APIS = [
    "FusedScaleMaskSoftmax",
    "FusedScaleMaskSoftmaxDropout",
    "FuseLogSoftmax",
    "FuseSoftmax",
    "apply_rotary_pos_emb_index_fused",
    "fuse_rms_mrope_cuda",
    "fuse_rmsnorm_rope_quant_qkv",
    "fused_rms_norm_rope_contiguous",
    "gammarms_rotary_embedding_fuse_with_kv_store",
    "gammarms_rotary_embedding_fuse_with_kv_store_fake",
    "mimo_v2_split_rope_vscale_kv_store",
    "mqa_logits",
    "rms_rotary_embedding_fuse",
    "rms_rotary_embedding_fuse_with_kv_store",
]


@pytest.mark.parametrize("name", EXISTING_ATTENTION_APIS)
def test_existing_attention_api_is_direct_reexport(name):
    assert getattr(lightop.attention, name) is getattr(lightop, name)


def test_attention_public_api_inventory():
    assert set(lightop.attention.__all__) == set(EXISTING_ATTENTION_APIS)
