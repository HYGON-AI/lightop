#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run every lightop.tensor migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_tensor_existing_api.py",
    "test/test_batch_linear_assignment.py",
]

SCRIPT_TEST_FILES = [
    "test/test_concat.py",
    "test/test_fill_zeros.py",
    "test/test_reshape_to_float32.py",
    "test/test_sum.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("tensor", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
