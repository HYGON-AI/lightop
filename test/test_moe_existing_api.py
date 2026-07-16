"""Compatibility checks for the categorized LightOp MoE exports."""

import lightop
import lightop.moe
import lightop.moe.config
import lightop.moe.gemm
import lightop.moe.marlin
import lightop.moe.permutation
import lightop.moe.routing


W8A8_CONFIG_APIS = [
    "get_moe_cuda_config",
    "get_moe_cuda_marlin_config",
]


W8A8_GEMM_APIS = [
    "moe_gemm_w8a8",
]


W8A8_MARLIN_APIS = [
    "moe_gemm_marlin_w8a8",
    "moe_gemm_marlin_w8a8_blockwise_fp8",
    "moe_gemm_marlin_w8a8_fp8",
]


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


def test_w8a8_apis_are_direct_reexports():
    modules = {
        lightop.moe.config: W8A8_CONFIG_APIS,
        lightop.moe.gemm: W8A8_GEMM_APIS,
        lightop.moe.marlin: W8A8_MARLIN_APIS,
    }
    for module, names in modules.items():
        for name in names:
            categorized = getattr(lightop.moe, name)
            assert categorized is getattr(module, name)
            assert categorized is getattr(lightop, name)


def test_moe_public_api_inventory():
    expected = (
        ROUTING_APIS
        + PERMUTATION_APIS
        + W8A8_CONFIG_APIS
        + W8A8_GEMM_APIS
        + W8A8_MARLIN_APIS
    )
    assert set(lightop.moe.__all__) == set(expected)
