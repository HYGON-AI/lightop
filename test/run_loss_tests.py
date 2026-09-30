#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run every lightop.loss migration acceptance case."""

from category_test_runner import run_category


PYTEST_TEST_FILES = [
    "test/test_loss_existing_api.py",
]

SCRIPT_TEST_FILES = [
    "test/test_lossctc.py",
    "test/test_lossrnnt.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("loss", PYTEST_TEST_FILES, SCRIPT_TEST_FILES)
    )
