#!/usr/bin/env python3
"""Run every lightop.activation migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_activation_existing_api.py",
    "test/test_activation_lm_fuse_silu_mul_quant.py",
]

# These files now import public APIs through lightop.activation, but their
# historical execution style is not yet suitable for the category pytest run.
PENDING_ENTRY_TEST_FILES = [
    ("test/test_add_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_droppath.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_fuse_silu_mul.py", "single smoke case only runs from __main__"),
    ("test/test_fuse_silu_mul_quant.py", "cases only run from __main__"),
    ("test/test_fuse_silu_mul_quant_ep.py", "EP cases only run from __main__"),
    ("test/test_bias_dropout_add.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_bias_gelu.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_bias_swiglu.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_gn_silu.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_ln_add_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_ln_add_dropout_bias.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_ln_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_ln_droppath.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/test_relu_dropout.py", "benchmark functions require arguments; add pytest parametrization"),
    ("test/moe_marlin/tuning/EP_tuning/fused_marlin_moe_valid.py", "validation script has no pytest entry"),
    ("test/moe_marlin/tuning/EP_tuning/fused_marlin_moe.py", "helper implementation; no standalone test entry"),
]


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "activation",
            RUNNABLE_TEST_FILES,
            PENDING_ENTRY_TEST_FILES,
        )
    )
