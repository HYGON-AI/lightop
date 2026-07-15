#!/usr/bin/env python3
"""Run every lightop.norm migration acceptance case."""

from category_test_runner import run_category


if __name__ == "__main__":
    raise SystemExit(
        run_category(
            "norm",
            [
                "test/test_norm_existing_api.py",
                "test/test_norm_lm_faster_rmsquant.py",
            ],
        )
    )
