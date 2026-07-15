#!/usr/bin/env python3
"""Run every lightop.norm migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_norm_existing_api.py",
    "test/test_norm_lm_faster_rmsquant.py",
]

# These files now import public APIs through lightop.norm, but their historical
# execution style is not yet suitable for the category-level pytest command.
PENDING_ENTRY_TEST_FILES = [
    ("test/test_ln_add.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_celu_groupnorm.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_fuse_fill_rms_x2_concat.py", "test function requires unparametrized arguments"),
    ("test/test_fuse_rmsnorm_contiguous.py", "executes cases during module import; move them into pytest functions"),
    ("test/test_gemma_fused_add_rmsnorm.py", "cases only run from __main__"),
    ("test/test_gemma_rms_norm_fp8_quant.py", "FP8 cases only run from __main__"),
    ("test/test_groupnorm.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_layernorm.py", "cases only run from __main__"),
    ("test/test_layernorm_mixed_fp32_weight.py", "cases only run from __main__"),
    ("test/test_rms_norm_dynamic_per_token_quant.py", "cases only run from __main__"),
    ("test/test_rms_norm_fp8_quant.py", "FP8 cases only run from __main__"),
    ("test/test_rmsnorm.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_rn_add.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_rn_add_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_rn_add_dropout_bias.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_rn_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_fuse_silu_mul_fp8_quant.py", "RMSNorm FP8 cases only run from __main__ despite the legacy filename"),
    ("test/test_fuse_rmsnorm_rope_contiguous.py", "vLLM-dependent cases only run from __main__"),
    ("test/test_fuse_rmsnorm_rope_quant_qkv.py", "vLLM-dependent cases only run from __main__"),
]


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "norm",
            RUNNABLE_TEST_FILES,
            PENDING_ENTRY_TEST_FILES,
        )
    )
