#!/usr/bin/env python3
"""Run every LightOp operator test with HCU-aware device allocation.

Ordinary test programs run on one stable idle HCU.  Test programs that create a
torch.distributed process group run through torchrun after every detected HCU is
idle, so they can exercise their real multi-rank path.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


INIT_PROCESS_GROUP = re.compile(r"(?:torch\.distributed|dist)\.init_process_group\s*\(")


class DeviceError(RuntimeError):
    """Raised when HCU availability cannot satisfy a test."""


def parse_hy_smi(output: str) -> dict[int, float]:
    """Return the maximum of VRAM% and HCU% for every HCU in hy-smi output."""
    result: dict[int, float] = {}
    vram_column, hcu_column = 5, 6
    for raw_line in output.splitlines():
        line = raw_line.strip()
        columns = re.split(r"\s+", line)
        if line.startswith("HCU") and "VRAM%" in columns and "HCU%" in columns:
            vram_column = columns.index("VRAM%")
            hcu_column = columns.index("HCU%")
            continue
        if not re.match(r"^\d+\s+", line) or len(columns) <= max(vram_column, hcu_column):
            continue
        try:
            device_id = int(columns[0])
            usage = []
            for column in (vram_column, hcu_column):
                value = columns[column].rstrip("%")
                usage.append(0.0 if value in {"", "N/A", "NA", "--"} else float(value))
            result[device_id] = max(usage)
        except ValueError:
            continue
    return result


def query_usage() -> dict[int, float]:
    proc = subprocess.run(["hy-smi"], text=True, capture_output=True, check=False)
    if proc.returncode:
        raise DeviceError(proc.stderr.strip() or proc.stdout.strip() or "hy-smi failed")
    usage = parse_hy_smi(proc.stdout)
    if not usage:
        raise DeviceError("hy-smi did not report any HCU usage")
    return usage


def wait_for_devices(
    count: int,
    *,
    require_all: bool,
    threshold: float,
    wait_seconds: int,
    stable_samples: int,
    sample_interval: float,
) -> list[int]:
    deadline = time.monotonic() + wait_seconds
    last_usage: dict[int, float] = {}
    while time.monotonic() < deadline:
        samples: list[dict[int, float]] = []
        for index in range(stable_samples):
            sample = query_usage()
            last_usage = sample
            samples.append(sample)
            if index + 1 < stable_samples:
                time.sleep(sample_interval)

        stable_ids = set(samples[0])
        for sample in samples[1:]:
            stable_ids.intersection_update(sample)
        idle = sorted(
            device_id
            for device_id in stable_ids
            if all(sample[device_id] <= threshold for sample in samples)
        )
        if require_all:
            all_ids = set(last_usage)
            if all_ids and set(idle) == all_ids:
                return sorted(all_ids)
        elif len(idle) >= count:
            return idle[:count]

        usage_text = ", ".join(
            f"HCU{device_id}={usage:g}%" for device_id, usage in sorted(last_usage.items())
        )
        print(f"Waiting for {'all HCUs' if require_all else f'{count} idle HCU(s)'}: {usage_text}")
        time.sleep(sample_interval)

    requirement = "all HCUs" if require_all else f"{count} idle HCU(s)"
    raise DeviceError(f"Timed out after {wait_seconds}s waiting for {requirement}")


def find_test_programs(test_root: Path) -> list[Path]:
    programs = sorted(path for path in test_root.rglob("test_*.py") if path.is_file())
    # This is the repository's distributed all-reduce test even though its
    # filename is benchmark_* rather than test_*.
    allreduce = test_root / "benchmark_allreduce.py"
    if allreduce.is_file():
        programs.append(allreduce)
    return programs


def is_distributed_test(path: Path) -> bool:
    return bool(INIT_PROCESS_GROUP.search(path.read_text(encoding="utf-8", errors="ignore")))


def test_command(path: Path, multi_hcu_count: int) -> list[str]:
    if is_distributed_test(path):
        return [
            "torchrun",
            "--nnodes=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            f"--nproc-per-node={multi_hcu_count}",
            str(path),
        ]
    return [sys.executable, str(path)]


def run_program(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    """Run one program while allowing its stdout/stderr to reach Actions live."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        # torchrun launches worker processes.  End the whole session so a
        # timed-out test cannot leave workers holding an HCU for later cases.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def test_environment(test_root: Path, devices: Iterable[int] | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if devices is None:
        env.pop("HIP_VISIBLE_DEVICES", None)
        # All distributed workers are local to the test container. The Runner
        # hostname is not resolvable inside every pre-built test image.
        env["MASTER_ADDR"] = "127.0.0.1"
        env["GLOO_SOCKET_IFNAME"] = "lo"
    else:
        env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    env["PYTHONPATH"] = f"{test_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["PYTORCH_TEST_WITH_ROCM"] = "1"
    env["CONTINUE_THROUGH_ERROR"] = "True"
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def write_junit(report: Path, passed: list[str], failed: list[tuple[str, str]]) -> None:
    suite = ET.Element(
        "testsuite",
        name="lightop.functional",
        tests=str(len(passed) + len(failed)),
        failures=str(len(failed)),
        errors="0",
        skipped="0",
    )
    for name in passed:
        ET.SubElement(suite, "testcase", classname="lightop.functional", name=name)
    for name, message in failed:
        case = ET.SubElement(suite, "testcase", classname="lightop.functional", name=name)
        ET.SubElement(case, "failure", message=message)
    report.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(report, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--usage-threshold", type=float, default=5.0)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument(
        "--all-hcu-wait-seconds",
        type=int,
        default=600,
        help="Maximum wait for the one distributed test to obtain every HCU.",
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=600,
        help="Maximum runtime for a single-HCU test program.",
    )
    parser.add_argument(
        "--distributed-test-timeout-seconds",
        type=int,
        default=1800,
        help="Maximum runtime for a distributed multi-HCU test program.",
    )
    parser.add_argument("--stable-samples", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    args = parser.parse_args()

    programs = find_test_programs(args.test_root)
    if not programs:
        print(f"No test programs found under {args.test_root}", file=sys.stderr)
        return 2

    distributed_programs = [program for program in programs if is_distributed_test(program)]
    print(
        f"Discovered {len(programs)} test programs under {args.test_root}; "
        f"{len(distributed_programs)} require every HCU.",
        flush=True,
    )
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    for number, program in enumerate(programs, start=1):
        name = str(program.relative_to(args.test_root))
        distributed = is_distributed_test(program)
        try:
            if distributed:
                devices = wait_for_devices(
                    1,
                    require_all=True,
                    threshold=args.usage_threshold,
                    wait_seconds=args.all_hcu_wait_seconds,
                    stable_samples=args.stable_samples,
                    sample_interval=args.sample_interval,
                )
                env = test_environment(args.test_root, None)
                device_text = f"all HCUs ({','.join(map(str, devices))})"
                timeout_seconds = args.distributed_test_timeout_seconds
            else:
                devices = wait_for_devices(
                    1,
                    require_all=False,
                    threshold=args.usage_threshold,
                    wait_seconds=args.wait_seconds,
                    stable_samples=args.stable_samples,
                    sample_interval=args.sample_interval,
                )
                env = test_environment(args.test_root, devices)
                device_text = str(devices[0])
                timeout_seconds = args.test_timeout_seconds
            command = test_command(program, len(devices))
            print(
                f"[{number}/{len(programs)}] START {name} "
                f"[HCU: {device_text}; timeout: {timeout_seconds}s]",
                flush=True,
            )
            started = time.monotonic()
            result = run_program(
                command,
                cwd=args.test_root,
                env=env,
                timeout_seconds=timeout_seconds,
            )
            elapsed = time.monotonic() - started
            if result.returncode:
                failed.append((name, f"Exited with status {result.returncode}"))
                print(
                    f"[{number}/{len(programs)}] FAIL {name} "
                    f"[status: {result.returncode}; elapsed: {elapsed:.1f}s]",
                    flush=True,
                )
            else:
                passed.append(name)
                print(
                    f"[{number}/{len(programs)}] PASS {name} "
                    f"[elapsed: {elapsed:.1f}s]",
                    flush=True,
                )
        except subprocess.TimeoutExpired:
            failed.append((name, f"Timed out after {timeout_seconds}s"))
            print(
                f"[{number}/{len(programs)}] TIMEOUT {name} "
                f"[limit: {timeout_seconds}s]",
                flush=True,
            )
        except (DeviceError, OSError) as exc:
            failed.append((name, str(exc)))
            print(f"[{number}/{len(programs)}] ERROR {name}: {exc}", flush=True)

    write_junit(args.report, passed, failed)
    print(f"Passed ({len(passed)}): {' '.join(passed) or 'none'}", flush=True)
    print(f"Failed ({len(failed)}): {' '.join(name for name, _ in failed) or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
