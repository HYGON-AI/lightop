#!/usr/bin/env python3
"""Run the 10-case LightOp post-integration quick validation.

From /workspace/projects/cuda-repo/lightop_limslim/lightop_repo, run:

    python lightop/tools/run_quick_operator_tests.py

Select one HCU/DTK device:

    ROCR_VISIBLE_DEVICES=0 python lightop/tools/run_quick_operator_tests.py

Optional arguments supported by run_operator_tests.py can be appended, e.g.:

    python lightop/tools/run_quick_operator_tests.py --timeout 300

If the current directory is the lightop subdirectory, the shorter equivalent
is:

    python tools/run_quick_operator_tests.py

Results are written to test_results/after_<timestamp>/, including per-case
logs, passed_cases.txt, failed_cases.txt, summary.txt, and summary.json.
"""

from __future__ import annotations

from pathlib import Path
import sys

from run_operator_tests import main as run_operator_tests


def main() -> int:
    manifest = (
        Path(__file__).resolve().parent
        / "operator_tests"
        / "after_quick_cases.txt"
    )
    sys.argv[1:1] = ["--stage", "after", "--manifest", str(manifest)]
    return run_operator_tests()


if __name__ == "__main__":
    raise SystemExit(main())
