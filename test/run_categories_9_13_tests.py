#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run all migration acceptance suites for categories 9 through 13."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = [
    "test/run_kvcache_tests.py",
    "test/run_sampling_tests.py",
    "test/run_tensor_tests.py",
    "test/run_vision_tests.py",
    "test/run_loss_tests.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightOp category 9-13 migration acceptance suites."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="run one full build before all category suites",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="explicit build/lib.* directory forwarded to each suite",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=16,
        help="MAX_JOBS used with --build (default: 16)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list every case without building or testing",
    )
    return parser.parse_args()


def run(command: list[str], env: dict[str, str]) -> bool:
    print(f"\n{'=' * 78}\n{' '.join(command)}\n{'=' * 78}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode == 0


def main() -> int:
    args = parse_args()
    env = os.environ.copy()

    if args.build and not args.list_cases:
        env["MAX_JOBS"] = str(args.max_jobs)
        if not run([sys.executable, "setup.py", "build"], env):
            return 1

    common_args: list[str] = []
    if args.build_dir is not None:
        common_args.extend(["--build-dir", str(args.build_dir)])
    if args.list_cases:
        common_args.append("--list-cases")

    results = []
    for runner in RUNNERS:
        passed = run([sys.executable, runner, *common_args], env)
        results.append((runner, passed))
        if not passed:
            break

    print(f"\n{'=' * 78}\nCATEGORY 9-13 TEST SUMMARY\n{'=' * 78}")
    for runner, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {runner}")
    return 0 if len(results) == len(RUNNERS) and all(
        passed for _, passed in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
