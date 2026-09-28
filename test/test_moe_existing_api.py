"""Compatibility checks for the categorized LightOp MoE exports."""

import lightop
import lightop.moe
import lightop.moe.config
import lightop.moe.expert_parallel
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


W4A8_CONFIG_APIS = [
    "get_moe_cuda_config_w4a8",
    "get_moe_cuda_marlin_config_w4a8",
]


W4A8_GEMM_APIS = [
    "moe_gemm_w4a8",
]


W4A8_MARLIN_APIS = [
    "moe_gemm_marlin_w4a8",
]


GROUPGEMM_CONFIG_APIS = [
    "get_moe_groupgemm_config_w4a8",
    "get_moe_groupgemm_marlin_config_w4a8",
]


GROUPGEMM_APIS = [
    "moe_groupgemm_w4a8",
]


GROUPGEMM_MARLIN_APIS = [
    "moe_groupgemm_marlin_w4a8",
]


OTHER_MOE_CONFIG_APIS = [
    "get_moe_cuda_config_w16a16",
    "get_moe_cuda_marlin_config_w16a16",
]


CATEGORY_ONLY_CONFIG_APIS = [
    "get_moe_cuda_marlin_config_w4a16",
]


OTHER_MOE_GEMM_APIS = [
    "moe_gemm_w16a16",
]


OTHER_MOE_MARLIN_APIS = [
    "moe_gemm_marlin_w16a16",
    "moe_marlin_w4a16",
]


LMSLIM_REQUIRED_MOE_APIS = [
    "get_moe_cuda_config",
    "get_moe_cuda_config_w4a8",
    "get_moe_cuda_marlin_config",
    "get_moe_cuda_marlin_config_w4a8",
    "get_moe_groupgemm_config_w4a8",
    "get_moe_groupgemm_marlin_config_w4a8",
    "moe_gemm_marlin_w4a8",
    "moe_gemm_marlin_w8a8",
    "moe_gemm_marlin_w8a8_fp8",
    "moe_gemm_w4a8",
    "moe_gemm_w8a8",
    "moe_groupgemm_marlin_w4a8",
    "moe_groupgemm_w4a8",
]


LMSLIM_MIGRATED_INT8_APIS = [
    "fused_experts_impl_fp8_marlin",
    "fused_experts_impl_int8",
    "fused_experts_impl_int8_marlin",
]


LMSLIM_MIGRATED_W4A8_APIS = [
    "fused_experts_impl_w4a8",
    "fused_experts_impl_w4a8_marlin",
]


LMSLIM_MIGRATED_GROUPGEMM_APIS = [
    "groupgemm_impl_w4a8",
    "groupgemm_impl_w4a8_marlin",
]


LMSLIM_MIGRATED_W4A16_APIS = [
    "fused_experts_impl_int4",
    "fused_experts_impl_w4a16",
    "fused_experts_w4a16",
]


LMSLIM_OPTIONAL_INT4_CUDA_APIS = [
    "fused_experts_cuda",
    "fused_experts_impl_cuda",
]


PERMUTATION_APIS = [
    "fuse_groupgemm_permute_quant",
    "groupgemm_permute",
    "groupgemm_unpermute",
]


ROUTING_APIS = [
    "kimi_k2_moe_fused_gate",
    "moe_align_block_size",
    "moe_align_block_size_out",
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_fused_gate_sqrtsoftplus_v41_hcu",
    "moe_sum",
    "topk_ids_postprocess",
    "topk_softmax",
]


EXPERT_PARALLEL_APIS = [
    "ep_build_m_indices",
    "ep_gather",
    "ep_scatter",
]


LEGACY_TOP_LEVEL_ROUTING_APIS = [
    "moe_fused_gate",
    "moe_fused_gate_sqrtsoftplus",
    "moe_fused_gate_sqrtsoftplus_v41_hcu",
    "moe_sum",
]


def test_routing_apis_are_categorized_exports():
    for name in ROUTING_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(lightop.moe.routing, name)

    for name in LEGACY_TOP_LEVEL_ROUTING_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(lightop, name)


def test_expert_parallel_apis_are_categorized_exports():
    for name in EXPERT_PARALLEL_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(lightop.moe.expert_parallel, name)


def test_moe_align_uses_migrated_lmslim_implementation():
    assert lightop.moe.moe_align_block_size.__module__ == (
        "lightop._lmslim_native.vllm_compat.moe_align_block_size"
    )


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


def test_w4a8_apis_are_direct_reexports():
    modules = {
        lightop.moe.config: W4A8_CONFIG_APIS,
        lightop.moe.gemm: W4A8_GEMM_APIS,
        lightop.moe.marlin: W4A8_MARLIN_APIS,
    }
    for module, names in modules.items():
        for name in names:
            categorized = getattr(lightop.moe, name)
            assert categorized is getattr(module, name)
            assert categorized is getattr(lightop, name)


def test_groupgemm_apis_are_direct_reexports():
    modules = {
        lightop.moe.config: GROUPGEMM_CONFIG_APIS,
        lightop.moe.gemm: GROUPGEMM_APIS,
        lightop.moe.marlin: GROUPGEMM_MARLIN_APIS,
    }
    for module, names in modules.items():
        for name in names:
            categorized = getattr(lightop.moe, name)
            assert categorized is getattr(module, name)
            assert categorized is getattr(lightop, name)


def test_other_existing_moe_apis_are_direct_reexports():
    modules = {
        lightop.moe.config: OTHER_MOE_CONFIG_APIS,
        lightop.moe.gemm: OTHER_MOE_GEMM_APIS,
        lightop.moe.marlin: OTHER_MOE_MARLIN_APIS,
    }
    for module, names in modules.items():
        for name in names:
            categorized = getattr(lightop.moe, name)
            assert categorized is getattr(module, name)
            assert categorized is getattr(lightop, name)


def test_category_only_config_api_is_exported():
    for name in CATEGORY_ONLY_CONFIG_APIS:
        assert getattr(lightop.moe, name) is getattr(lightop.moe.config, name)


def test_lmslim_required_moe_apis_are_available_from_category():
    for name in LMSLIM_REQUIRED_MOE_APIS:
        assert getattr(lightop.moe, name) is getattr(lightop, name)


def test_lmslim_int8_implementations_are_source_migrated():
    native = __import__("lightop.moe.native", fromlist=["dummy"])
    for name in LMSLIM_MIGRATED_INT8_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(native, name)
        assert "lightop._lmslim_native.layers.fused_moe" in categorized.__module__


def test_lmslim_w4a8_implementations_are_source_migrated():
    native = __import__("lightop.moe.native", fromlist=["dummy"])
    for name in LMSLIM_MIGRATED_W4A8_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(native, name)
        assert "lightop._lmslim_native.layers.fused_moe" in categorized.__module__


def test_lmslim_groupgemm_implementations_are_source_migrated():
    native = __import__("lightop.moe.native", fromlist=["dummy"])
    for name in LMSLIM_MIGRATED_GROUPGEMM_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(native, name)
        assert "lightop._lmslim_native.layers.groupgemm" in categorized.__module__


def test_lmslim_w4a16_implementations_are_source_migrated():
    native = __import__("lightop.moe.native", fromlist=["dummy"])
    for name in LMSLIM_MIGRATED_W4A16_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(native, name)
        assert "lightop._lmslim_native.layers.fused_moe" in categorized.__module__


def test_optional_int4_cuda_interfaces_are_lazy_callables():
    native = __import__("lightop.moe.native", fromlist=["dummy"])
    for name in LMSLIM_OPTIONAL_INT4_CUDA_APIS:
        categorized = getattr(lightop.moe, name)
        assert categorized is getattr(native, name)
        assert callable(categorized)


def test_moe_public_api_inventory():
    expected = (
        ROUTING_APIS
        + EXPERT_PARALLEL_APIS
        + PERMUTATION_APIS
        + W8A8_CONFIG_APIS
        + W8A8_GEMM_APIS
        + W8A8_MARLIN_APIS
        + W4A8_CONFIG_APIS
        + W4A8_GEMM_APIS
        + W4A8_MARLIN_APIS
        + GROUPGEMM_CONFIG_APIS
        + GROUPGEMM_APIS
        + GROUPGEMM_MARLIN_APIS
        + OTHER_MOE_CONFIG_APIS
        + CATEGORY_ONLY_CONFIG_APIS
        + OTHER_MOE_GEMM_APIS
        + OTHER_MOE_MARLIN_APIS
        + LMSLIM_MIGRATED_INT8_APIS
        + LMSLIM_MIGRATED_W4A8_APIS
        + LMSLIM_MIGRATED_GROUPGEMM_APIS
        + LMSLIM_MIGRATED_W4A16_APIS
        + LMSLIM_OPTIONAL_INT4_CUDA_APIS
    )
    assert set(lightop.moe.__all__) == set(expected)
