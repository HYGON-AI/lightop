#!/usr/bin/env python3
"""Run every lightop.activation migration acceptance case."""

from category_test_runner import run_category


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "activation",
            [
                "test/test_activation_existing_api.py",
                "test/test_activation_lm_fuse_silu_mul_quant.py",
            ],
        )
    )
