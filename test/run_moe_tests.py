#!/usr/bin/env python3
"""Run the core lightop.moe migration acceptance cases."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_moe_existing_api.py",
    "test/test_moe_fused_gate_sqrtsoftplus.py",
    "test/test_fuse_groupgemm_permute_quant.py",
]

SCRIPT_TEST_FILES = [
    "test/test_kimi_k2_moe_fused_gate.py",
    "test/test_moe_align_block_size.py",
    "test/test_moe_sum_opt.py",
    "test/test_fused_moe_gate.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "moe",
            RUNNABLE_TEST_FILES,
            SCRIPT_TEST_FILES,
        )
    )
