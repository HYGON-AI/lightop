#!/usr/bin/env python3
"""Run the pytest-only cases that were blank in all_test_result.xlsx.

These tests define pytest test functions and have no ``__main__`` entry point,
so they must be run through pytest rather than ``python test/<file>.py``.
"""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_native_categorized_quant_remaining_api.py",
    "test/test_native_categorized_attention_remaining_api.py",
    "test/test_vision_existing_api.py",
    "test/test_public_envs.py",
    "test/test_native_categorized_norm_remaining_api.py",
    "test/test_native_categorized_moe_api.py",
    "test/test_kvcache_existing_api.py",
    "test/test_attention_existing_api.py",
    "test/test_native_categorized_norm_activation_api.py",
    "test/test_gemm_ops_existing_api.py",
    "test/test_quant_per_token_quant_fp8.py",
    "test/test_quant_per_token_group_quant_fp8.py",
    "test/test_moe_existing_api.py",
    "test/test_prefill_gather_and_upconvert_fp8_kv_cache.py",
    "test/test_categorized_operator_imports.py",
    "test/test_quant_blockwise_scaled_cast_transpose_api.py",
    "test/test_gemm_ops_rocblas_scaled_mm_nn.py",
    "test/test_rms_norm_dynamic_per_token_quant_sglang.py",
    "test/test_quant_destindex_copy_quantize_int8_cuda.py",
    "test/test_quant_per_token_quant_int8.py",
    "test/test_sampling_existing_api.py",
    "test/test_no_lmslim_runtime_dependency.py",
    "test/test_quant_destindex_dequantize_int8_cuda.py",
    "test/test_gemm_ops_hipblaslt_w8a8_channelwise_gemm.py",
    "test/test_quant_w4tow16.py",
    "test/test_fuse_silu_mul_clamp_quant.py",
    "test/test_activation_existing_api.py",
    "test/test_quant_fp8_to_16.py",
    "test/test_sparse_attention_transform_api.py",
    "test/test_gemm_ops_triton_scaled_mm.py",
    "test/test_per_token_group_quant_fp8_api_alignment.py",
    "test/test_loss_existing_api.py",
    "test/test_mla_qkv_a_rms_norm_dynamic_per_token_quant.py",
    "test/test_norm_existing_api.py",
    "test/test_w8a8_config_dir.py",
    "test/test_norm_lm_faster_rmsquant.py",
    "test/test_activation_lm_fuse_silu_mul_quant.py",
    "test/test_gemm_ops_triton_int8_gemm_helper.py",
    "test/test_tensor_existing_api.py",
    "test/test_native_categorized_attention_api.py",
    "test/test_gemm_ops_lazy_native_import.py",
    "test/test_native_categorized_kvcache_remaining_api.py",
    "test/test_moe_align_block_size_out.py",
    "test/test_gemm_ops_hipblaslt_w8a8_channelwise_gemm_kme.py",
    "test/test_gemm_ops_hipblaslt_w8a8_blockwise_gemm.py",
    "test/test_gemm_ops_hipblaslt_w8a8_gemm.py",
]


if __name__ == "__main__":
    raise SystemExit(run_category("blank-pytest", RUNNABLE_TEST_FILES))
