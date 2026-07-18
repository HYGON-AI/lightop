#!/usr/bin/env python3
"""Run every lightop.tensor migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_tensor_existing_api.py",
    "test/test_batch_linear_assignment.py",
]

SCRIPT_TEST_FILES = [
    "test/test_concat.py",
    "test/test_fill_zeros.py",
    "test/test_sum.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("tensor", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
