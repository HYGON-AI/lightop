#!/usr/bin/env python3
"""Run every lightop.norm migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_norm_existing_api.py",
    "test/test_native_categorized_norm_remaining_api.py",
    "test/test_norm_lm_faster_rmsquant.py",
]

SCRIPT_TEST_FILES = [
    "test/test_ln_add.py",
    "test/test_celu_groupnorm.py",
    "test/test_fuse_fill_rms_x2_concat.py",
    "test/test_fuse_rmsnorm_contiguous.py",
    "test/test_gemma_fused_add_rmsnorm.py",
    "test/test_gemma_rms_norm_fp8_quant.py",
    "test/test_groupnorm.py",
    "test/test_layernorm.py",
    "test/test_layernorm_mixed_fp32_weight.py",
    "test/test_layer_norm_fwd_1pass.py",
    "test/test_rms_norm_dynamic_per_token_quant.py",
    "test/test_rms_norm_fp8_quant.py",
    "test/test_rmsnorm.py",
    "test/test_rn_add.py",
    "test/test_rn_add_dropout.py",
    "test/test_rn_add_dropout_bias.py",
    "test/test_rn_dropout.py",
    "test/test_fuse_silu_mul_fp8_quant.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "norm",
            RUNNABLE_TEST_FILES,
            SCRIPT_TEST_FILES,
        )
    )
