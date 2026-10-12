# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared runner for category-level LightOp acceptance suites."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_args(category: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run all {category} migration acceptance cases."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="run `python setup.py build` before testing",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="explicit build/lib.* directory to use",
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
        help="list runnable and pending legacy cases without building or testing",
    )
    return parser.parse_args()


def find_build_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser().resolve()]
    else:
        abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = sorted((ROOT / "build").glob(f"lib.*{abi}"))
        if not candidates:
            candidates = sorted((ROOT / "build").glob("lib.*"))
    if not candidates or not candidates[-1].is_dir():
        raise SystemExit("no build/lib.* directory found; rerun with --build")

    build_dir = candidates[-1]
    if not list((build_dir / "lightop").glob("op*.so")):
        raise SystemExit(
            f"compiled lightop/op*.so is missing from {build_dir}; rerun with --build"
        )
    return build_dir


def run_streamed(name: str, command: Sequence[str], env: dict[str, str]) -> bool:
    print(f"\n{'=' * 78}\n[{name}] {' '.join(command)}\n{'=' * 78}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    print(f"[{name}] {'PASSED' if return_code == 0 else 'FAILED'}")
    return return_code == 0


def print_case_inventory(
    category: str,
    pytest_test_files: Sequence[str],
    script_test_files: Sequence[str],
) -> None:
    print(f"lightop.{category} pytest files ({len(pytest_test_files)}):")
    for path in pytest_test_files:
        print(f"  PYTEST   {path}")

    print(f"lightop.{category} standalone Python test files ({len(script_test_files)}):")
    for path in script_test_files:
        print(f"  PYTHON   {path}")

def run_category(
    category: str,
    pytest_test_files: Sequence[str],
    script_test_files: Sequence[str] = (),
) -> int:
    args = parse_args(category)
    env = os.environ.copy()

    if args.list_cases:
        print_case_inventory(category, pytest_test_files, script_test_files)
        return 0

    if args.build:
        build_env = env.copy()
        build_env["MAX_JOBS"] = str(args.max_jobs)
        if not run_streamed(
            "full build", [sys.executable, "setup.py", "build"], build_env
        ):
            return 1

    build_dir = find_build_dir(args.build_dir)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(build_dir)
        if not old_pythonpath
        else os.pathsep.join((str(build_dir), old_pythonpath))
    )
    print(f"Using LightOp build: {build_dir}")

    results: list[tuple[str, bool]] = []
    if pytest_test_files:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            *[str(ROOT / path) for path in pytest_test_files],
        ]
        results.append(
            (
                "pytest acceptance cases",
                run_streamed(f"lightop.{category} pytest cases", command, env),
            )
        )

    for path in script_test_files:
        results.append(
            (
                path,
                run_streamed(
                    f"lightop.{category} {path}",
                    [sys.executable, str(ROOT / path)],
                    env,
                ),
            )
        )

    passed = all(result for _, result in results)
    print(f"\n{'=' * 78}\n{category.upper()} TEST SUMMARY\n{'=' * 78}")
    for name, result in results:
        print(f"{'PASS' if result else 'FAIL'}  {name}")
    return 0 if passed else 1
