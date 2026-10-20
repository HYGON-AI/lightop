#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run every lightop.kvcache migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_kvcache_existing_api.py",
    "test/test_native_categorized_kvcache_remaining_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_gather_and_maybe_dequant_cache.py",
    "test/test_fused_metadata.py",
    "test/test_reshape_cache_cuda.py",
    "test/test_fused_store_index_k_cache_fp8.py",
    "test/test_set_mla_kv_buffer.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("kvcache", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
