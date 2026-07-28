#!/usr/bin/env python3
"""Run every lightop.vision migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_vision_existing_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_blinear_sample.py",
    "test/test_grid_sample.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("vision", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
