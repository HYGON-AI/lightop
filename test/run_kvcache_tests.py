#!/usr/bin/env python3
"""Run every lightop.kvcache migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_kvcache_existing_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_fused_metadata.py",
    "test/test_reshape_cache_cuda.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("kvcache", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
