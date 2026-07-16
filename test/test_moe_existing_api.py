"""Compatibility checks for the categorized LightOp MoE exports."""

import lightop
import lightop.moe
import lightop.moe.permutation
import lightop.moe.routing


PERMUTATION_APIS = [
    "fuse_groupgemm_permute_quant",
    "groupgemm_permute",
    "groupgemm_unpermute",
]


ROUTING_APIS = [
    "moe_align_block_size",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_sum",
]


def test_routing_apis_are_direct_reexports():
    for name in ROUTING_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(lightop.moe.routing, name)
        assert categorized is getattr(lightop, name)


def test_permutation_apis_are_direct_reexports():
    for name in PERMUTATION_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(lightop.moe.permutation, name)
        assert categorized is getattr(lightop, name)


def test_moe_public_api_inventory():
    assert set(lightop.moe.__all__) == set(ROUTING_APIS + PERMUTATION_APIS)
