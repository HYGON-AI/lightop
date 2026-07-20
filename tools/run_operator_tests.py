#!/usr/bin/env python3
"""Run one LightOp build's operator cases without installing another build."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime


DEFAULT_FAILURE_PATTERNS = (
    r"Traceback \(most recent call last\):",
    r"AssertionError",
    r"(?im)^\s*(?:test\s+)?fail(?:ed)?\b",
    r"(?i)\b\d+\s+failed(?:,|\s|$)",
)


@dataclass
class Result:
    case: str
    status: str
    returncode: int | None
    duration_seconds: float
    log: str
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run operator tests for exactly one LightOp package build."
    )
    parser.add_argument(
        "--stage", choices=("before", "after"), default="after",
    )
    parser.add_argument(
        "--package",
        help="Optional wheel or import directory; defaults to installed LightOp.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Defaults to tools/operator_tests/<stage>_cases.txt.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--failure-pattern", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def load_cases(
    repo_root: Path,
    manifest: Path,
    includes: list[str],
    excludes: list[str],
) -> list[Path]:
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    cases: list[Path] = []
    seen: set[str] = set()
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        if includes and not any(fnmatch.fnmatch(name, item) for item in includes):
            continue
        if any(fnmatch.fnmatch(name, item) for item in excludes):
            continue
        case = (repo_root / name).resolve()
        if repo_root not in case.parents or not case.is_file():
            raise ValueError(f"manifest case does not exist: {name}")
        if name not in seen:
            seen.add(name)
            cases.append(case)
    if not cases:
        raise ValueError("manifest selected no cases")
    return cases


def prepare_package(spec: str | None, output: Path, python: str) -> Path | None:
    install_log = output / "install.log"
    if spec is None:
        install_log.write_text(
            "Using LightOp installed in the current Python environment.\n",
            encoding="utf-8",
        )
        return None
    source = Path(spec).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"package does not exist: {source}")
    if source.is_dir():
        install_log.write_text(
            f"Using import directory without installation: {source}\n",
            encoding="utf-8",
        )
        return source
    if source.suffix != ".whl":
        raise ValueError(f"package file is not a wheel: {source}")

    target = output / "_package"
    target.mkdir(parents=True, exist_ok=True)
    command = [
        python, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-deps", "--upgrade",
        "--target", str(target), str(source),
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    install_log.write_text(
        "$ " + " ".join(command) + "\n\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"wheel installation failed; see {install_log}")
    return target


def package_environment(package_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    # Operator migration tests must not auto-load environment-provided vLLM
    # plugins. Older vllm_hcu releases import the removed lmslim package and
    # can otherwise make an isolated LightOp package fail during collection.
    env["VLLM_PLUGINS"] = ""
    if package_root is None:
        env.pop("PYTHONPATH", None)
        return env
    # Deliberately replace PYTHONPATH so another LightOp checkout cannot leak in.
    env["PYTHONPATH"] = str(package_root)
    lib_dir = package_root / "lightop" / "lib"
    if lib_dir.is_dir():
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            str(lib_dir) + (os.pathsep + existing if existing else "")
        )
    return env


def safe_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", path)


def run_case(
    case: Path,
    repo_root: Path,
    package_root: Path | None,
    output: Path,
    python: str,
    timeout: float,
    failure_patterns: list[re.Pattern[str]],
) -> Result:
    relative = case.relative_to(repo_root).as_posix()
    log_path = output / "cases" / (safe_name(relative) + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [python, str(case)]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=repo_root,
            env=package_environment(package_root),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=None if timeout <= 0 else timeout, check=False,
        )
        case_output = completed.stdout
        returncode = completed.returncode
        if returncode:
            status, reason = "FAILED", f"exit code {returncode}"
        else:
            matched = next(
                (item.pattern for item in failure_patterns if item.search(case_output)),
                None,
            )
            if matched:
                status, reason = "FAILED", f"matched failure regex: {matched}"
            else:
                status, reason = "PASSED", ""
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        case_output = stdout + stderr
        returncode = None
        status, reason = "TIMEOUT", f"exceeded {timeout:g}s timeout"

    duration = time.monotonic() - started
    header = (
        f"case: {relative}\ncommand: {' '.join(command)}\n"
        f"status: {status}\nreason: {reason}\n"
        f"duration_seconds: {duration:.3f}\n" + "=" * 80 + "\n"
    )
    log_path.write_text(header + case_output, encoding="utf-8", errors="replace")
    return Result(
        case=relative, status=status, returncode=returncode,
        duration_seconds=round(duration, 3), log=str(log_path), reason=reason,
    )


def write_results(
    output: Path,
    stage: str,
    package: str | None,
    manifest: Path,
    results: list[Result],
) -> dict[str, int]:
    counts = {"total": len(results), "passed": 0, "failed": 0, "timeout": 0}
    for result in results:
        counts[result.status.lower()] += 1
    passed = [item for item in results if item.status == "PASSED"]
    failed = [item for item in results if item.status != "PASSED"]
    (output / "passed_cases.txt").write_text(
        "".join(f"{x.case}\t{x.duration_seconds:.3f}s\t{x.log}\n" for x in passed),
        encoding="utf-8",
    )
    (output / "failed_cases.txt").write_text(
        "".join(f"{x.case}\t{x.status}\t{x.reason}\t{x.log}\n" for x in failed),
        encoding="utf-8",
    )
    payload = {
        "stage": stage,
        "package": (
            str(Path(package).expanduser().resolve())
            if package else "<current-installed-environment>"
        ),
        "manifest": str(manifest),
        "counts": counts,
        "cases": [asdict(item) for item in results],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"LightOp {stage} operator test summary", "=" * 72,
        f"package: {payload['package']}", f"manifest: {manifest}",
        f"total={counts['total']} passed={counts['passed']} "
        f"failed={counts['failed']} timeout={counts['timeout']}",
        "", f"FAILED/TIMEOUT ({len(failed)})",
        *[f"  {x.case}: {x.status} ({x.reason}) -> {x.log}" for x in failed],
        "", f"PASSED ({len(passed)})", *[f"  {x.case}" for x in passed],
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = (
        args.manifest.resolve() if args.manifest
        else repo_root / "tools" / "operator_tests" / f"{args.stage}_cases.txt"
    )
    try:
        cases = load_cases(repo_root, manifest, args.include, args.exclude)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"stage={args.stage} manifest={manifest} selected={len(cases)}")
    for case in cases:
        print(f"  {case.relative_to(repo_root).as_posix()}")
    if args.dry_run:
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve() if args.output
        else repo_root / "test_results" / f"{args.stage}_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        package_root = prepare_package(args.package, output, args.python)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    patterns = [
        re.compile(item)
        for item in (*DEFAULT_FAILURE_PATTERNS, *args.failure_pattern)
    ]
    results: list[Result] = []
    for index, case in enumerate(cases, 1):
        relative = case.relative_to(repo_root).as_posix()
        print(f"[{index}/{len(cases)}] {relative} ... ", end="", flush=True)
        result = run_case(
            case, repo_root, package_root, output,
            args.python, args.timeout, patterns,
        )
        results.append(result)
        print(
            f"{result.status} ({result.duration_seconds:.3f}s)"
            + (f" - {result.reason}" if result.reason else "")
        )

    counts = write_results(output, args.stage, args.package, manifest, results)
    print("\n" + "=" * 72)
    print(
        f"total={counts['total']} passed={counts['passed']} "
        f"failed={counts['failed']} timeout={counts['timeout']}"
    )
    print(f"results={output}")
    if (counts["failed"] or counts["timeout"]) and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
