#!/usr/bin/env python3
"""Run every lightop.activation migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_activation_existing_api.py",
    "test/test_activation_lm_fuse_silu_mul_quant.py",
]

SCRIPT_TEST_FILES = [
    "test/test_add_dropout.py",
    "test/test_droppath.py",
    "test/test_fuse_silu_mul.py",
    "test/test_fuse_silu_mul_quant.py",
    "test/test_fuse_silu_mul_quant_ep.py",
    "test/test_bias_dropout_add.py",
    "test/test_bias_gelu.py",
    "test/test_bias_swiglu.py",
    "test/test_gn_silu.py",
    "test/test_ln_add_dropout.py",
    "test/test_ln_add_dropout_bias.py",
    "test/test_ln_dropout.py",
    "test/test_ln_droppath.py",
    "test/test_relu_dropout.py",
]

if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "activation",
            RUNNABLE_TEST_FILES,
            SCRIPT_TEST_FILES,
        )
    )
