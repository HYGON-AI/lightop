#!/usr/bin/env python3
"""Run every lightop.attention migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_attention_existing_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_rotary_positional_embedding.py",
    "test/test_rms_mrope.py",
    "test/test_fuse_rmsnorm_rope_quant_qkv.py",
    "test/test_fuse_rmsnorm_rope_contiguous.py",
    "test/test_scaled_masked_softmax.py",
    "test/test_scaled_masked_softmax_dropout.py",
    "test/test_softmax.py",
    "test/test_fuse_gammarmsnorm_rope_qkv_kernel.py",
    "test/test_fuse_rmsnorm_rope_qkv_kernel.py",
    "test/test_mimo_v2_split_rope_vscale_kv_store.py",
    "test/mqa_logits/test_mqa_logits.py",
    "test/test_fused_rms_rope.py",
    "test/test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "attention",
            RUNNABLE_TEST_FILES,
            SCRIPT_TEST_FILES,
        )
    )
