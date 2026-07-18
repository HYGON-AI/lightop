"""Compatibility checks for existing LightOp attention exports."""

import pytest

import lightop
import lightop.attention


EXISTING_ATTENTION_APIS = [
    "apply_rotary_pos_emb_index_fused",
    "fuse_rms_mrope_cuda",
]


@pytest.mark.parametrize("name", EXISTING_ATTENTION_APIS)
def test_existing_attention_api_is_direct_reexport(name):
    assert getattr(lightop.attention, name) is getattr(lightop, name)


def test_attention_public_api_inventory():
    assert set(lightop.attention.__all__) == set(EXISTING_ATTENTION_APIS)
