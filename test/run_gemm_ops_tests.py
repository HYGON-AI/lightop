#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run the LightOp GEMM migration and numerical regression suites."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LightOp optionally, then run gemm_ops migration tests."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="run `python setup.py build` before testing",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="skip the longer W4A16 and W8A16 FP8 legacy regression scripts",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="explicit build/lib.* directory to prepend to PYTHONPATH",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=16,
        help="MAX_JOBS used with --build (default: 16)",
    )
    return parser.parse_args()


def find_build_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        build_dir = explicit.expanduser().resolve()
        if not build_dir.is_dir():
            raise SystemExit(f"build directory does not exist: {build_dir}")
        candidates = [build_dir]
    else:
        abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = sorted((ROOT / "build").glob(f"lib.*{abi}"))
        if not candidates:
            candidates = sorted((ROOT / "build").glob("lib.*"))
        if not candidates:
            raise SystemExit("no build/lib.* directory found; rerun with --build")

    build_dir = candidates[-1].resolve()
    if not list((build_dir / "lightop").glob("op*.so")):
        raise SystemExit(
            f"compiled lightop/op*.so is missing from {build_dir}; rerun with --build"
        )
    return build_dir


def stream_command(
    name: str,
    command: Sequence[str],
    env: dict[str, str],
    validator: Callable[[str], tuple[bool, str]] | None = None,
) -> bool:
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
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    return_code = process.wait()
    output = "".join(output_lines)

    if return_code != 0:
        print(f"[{name}] FAILED: exit code {return_code}")
        return False
    if validator is not None:
        valid, reason = validator(output)
        if not valid:
            print(f"[{name}] FAILED: {reason}")
            return False
    print(f"[{name}] PASSED")
    return True


def validate_w4a16(output: str) -> tuple[bool, str]:
    passed = sum(line.strip() == "test pass" for line in output.splitlines())
    if passed != 20:
        return False, f"expected 20 W4A16 passes, observed {passed}"
    return True, ""


def validate_w8a16_fp8(output: str) -> tuple[bool, str]:
    results = [
        line.strip() for line in output.splitlines() if line.strip() in {"True", "False"}
    ]
    if len(results) != 10:
        return False, f"expected 10 W8A16 FP8 comparisons, observed {len(results)}"
    if "False" in results:
        return False, "at least one W8A16 FP8 comparison failed"
    return True, ""


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    results: list[tuple[str, bool]] = []

    if args.build:
        build_env = env.copy()
        build_env["MAX_JOBS"] = str(args.max_jobs)
        ok = stream_command(
            "full build", [sys.executable, "setup.py", "build"], build_env
        )
        results.append(("full build", ok))
        if not ok:
            return 1

    build_dir = find_build_dir(args.build_dir)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(build_dir)
        if not old_pythonpath
        else os.pathsep.join((str(build_dir), old_pythonpath))
    )
    print(f"Using LightOp build: {build_dir}")

    migration_tests = sorted(TEST_DIR.glob("test_gemm_ops_*.py"))
    results.append(
        (
            "gemm_ops API and native tests",
            stream_command(
                "gemm_ops API and native tests",
                [sys.executable, "-m", "pytest", "-q", *map(str, migration_tests)],
                env,
            ),
        )
    )

    if not args.skip_legacy:
        results.append(
            (
                "W4A16 legacy regression",
                stream_command(
                    "W4A16 legacy regression",
                    [sys.executable, str(TEST_DIR / "ci_tests" / "test_gemm.py")],
                    env,
                    validate_w4a16,
                ),
            )
        )
        results.append(
            (
                "W8A16 FP8 legacy regression",
                stream_command(
                    "W8A16 FP8 legacy regression",
                    [sys.executable, str(TEST_DIR / "test_w8a16.py")],
                    env,
                    validate_w8a16_fp8,
                ),
            )
        )

    print(f"\n{'=' * 78}\nGEMM_OPS TEST SUMMARY\n{'=' * 78}")
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    failed = [name for name, passed in results if not passed]
    if failed:
        print(f"\nFAILED: {len(failed)} test group(s)")
        return 1
    print(f"\nPASSED: all {len(results)} test group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
