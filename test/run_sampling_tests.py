#!/usr/bin/env python3
"""Run every lightop.sampling migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_sampling_existing_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_topk_transform_512.py",
    "test/test_topk_transform_1024.py",
    "test/test_topk_topp_sampler.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("sampling", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
