#!/usr/bin/env python3
"""Create a first-pass LightOp/DCU profile digest from hipprof text.

This helper is intentionally conservative. It extracts simple timing signals
from hipprof-like text, compares a candidate against an optional baseline, and
writes a markdown digest that an agent must still inspect and refine.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TIME_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_:$<>,.~+\- /\\()[\]]+?)"
    r"\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>ns|us|ms|s|%)?\b",
    re.IGNORECASE,
)


@dataclass
class Signal:
    name: str
    value_ms: float
    raw: str


def to_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1_000.0
    if unit == "s":
        return value * 1_000.0
    return value


def read_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def interesting_line(line: str, kernel_regex: Optional[re.Pattern[str]]) -> bool:
    text = line.strip()
    if not text:
        return False
    lowered = text.lower()
    if kernel_regex and kernel_regex.search(text):
        return True
    return any(
        token in lowered
        for token in (
            "kernel",
            "hip",
            "memcpy",
            "memset",
            "launch",
            "time",
            "gpu",
        )
    )


def parse_signals(text: str, kernel_regex: str = "") -> list[Signal]:
    regex = re.compile(kernel_regex) if kernel_regex else None
    signals: list[Signal] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not interesting_line(line, regex):
            continue
        for match in TIME_RE.finditer(line):
            unit = match.group("unit") or "ms"
            if unit == "%":
                continue
            value = float(match.group("value"))
            name = match.group("name").strip(" |,\t")
            if not name:
                name = line[:80]
            signals.append(Signal(name=name, value_ms=to_ms(value, unit), raw=line))
    return signals


def pick_top(signals: Iterable[Signal]) -> Optional[Signal]:
    items = list(signals)
    if not items:
        return None
    return max(items, key=lambda signal: signal.value_ms)


def classify(top: Optional[Signal], text: str) -> tuple[str, str, str]:
    lowered = (top.raw if top else text[:500]).lower()
    if not top:
        return ("Unknown", "Low", "Re-run profiling or inspect the raw benchmark log.")
    if "memcpy" in lowered or "copy" in lowered:
        return ("Host-device or device copy overhead", "Medium", "Remove the copy, reuse buffers, or fuse the producer/consumer path.")
    if "memset" in lowered:
        return ("Initialization overhead", "Medium", "Avoid unnecessary memset or fold initialization into the kernel.")
    if "launch" in lowered or "hiplaunch" in lowered:
        return ("Launch overhead", "Medium", "Fuse adjacent LightOp calls or route to a fused binding.")
    if "kernel" in lowered:
        return ("Kernel execution", "Medium", "Inspect the hot kernel memory layout, launch shape, and config branch.")
    return ("Mixed HIP runtime cost", "Low", "Inspect hipprof categories and collect deeper counters if needed.")


def fmt_signal(signal: Optional[Signal]) -> str:
    if not signal:
        return "missing"
    return f"{signal.value_ms:.6g} ms"


def pct_delta(candidate: Optional[Signal], baseline: Optional[Signal]) -> str:
    if not candidate or not baseline or baseline.value_ms == 0:
        return ""
    delta = (candidate.value_ms - baseline.value_ms) * 100.0 / baseline.value_ms
    return f"{delta:+.2f}%"


def render(args: argparse.Namespace) -> str:
    text = read_text(args.hipprof)
    baseline_text = read_text(args.baseline_hipprof)
    benchmark_log = read_text(args.benchmark_log)

    signals = parse_signals(text, args.kernel_regex)
    baseline_signals = parse_signals(baseline_text, args.kernel_regex)
    top = pick_top(signals)
    baseline_top = pick_top(baseline_signals)
    bottleneck, confidence, next_edit = classify(top, text)

    op = args.operator or "<operator>"
    version = args.version or "<version>"
    lines: list[str] = []
    lines.append(f"### DCU Profile Digest: {op} @ {version}")
    lines.append("")
    lines.append("Environment")
    lines.append(f"- LightOp root: {args.lightop_root or '<fill in>'}")
    lines.append(f"- Repo commit: {args.commit or '<fill in>'}")
    lines.append(f"- GPU / gfx: {args.gpu or '<fill in>'}")
    lines.append(f"- DTK / ROCm / PyTorch: {args.runtime or '<fill in>'}")
    lines.append(f"- Build command: {args.build_command or '<fill in>'}")
    lines.append(f"- Benchmark command: {args.benchmark_command or '<fill in>'}")
    lines.append(f"- Shape / dtype: {args.shape or '<fill in>'}")
    lines.append(f"- Baseline or parent profile: {args.baseline_hipprof or '<none>'}")
    lines.append(f"- Candidate profile: {args.hipprof or '<fill in>'}")
    lines.append(f"- Benchmark log: {args.benchmark_log or '<none>'}")
    lines.append("")
    lines.append("Headline")
    lines.append(f"- Bottleneck class: {bottleneck}")
    lines.append(f"- Dominant time sink or hotspot: {top.raw if top else 'unknown'}")
    lines.append(f"- Confidence: {confidence}")
    lines.append("- Why this profile is valid: first-pass hipprof text was parsed; inspect raw output before finalizing.")
    lines.append("")
    lines.append("Evidence")
    lines.append("| Signal | Baseline | Candidate | Delta | Interpretation |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(
        f"| Top parsed time sink | {fmt_signal(baseline_top)} | {fmt_signal(top)} | {pct_delta(top, baseline_top)} | {bottleneck} |"
    )
    if benchmark_log:
        lines.append("| Benchmark log | present | present |  | Use synchronized benchmark numbers for final speedup claims. |")
    lines.append("")
    lines.append("Profiler Hotspots")
    for signal in sorted(signals, key=lambda item: item.value_ms, reverse=True)[:8]:
        lines.append(f"- {signal.raw}")
    if not signals:
        lines.append("- No timing rows parsed; inspect hipprof.txt manually.")
    lines.append("")
    lines.append("Counter / ISA Analysis")
    lines.append("- Counter source: <fill in if rocprof/rocprof-compute was used>")
    lines.append("- Hot instruction window: <fill in if ISA was inspected>")
    lines.append("- Suspected codegen or resource issue: <fill in>")
    lines.append("- Instruction-level or launch-level edit: <fill in>")
    lines.append("")
    lines.append("Inference Chain")
    lines.append(f"1. Measured signal: {top.raw if top else 'no parsed dominant signal'}")
    lines.append(f"2. Likely mechanism: {bottleneck}")
    lines.append("3. Why other hypotheses are weaker: <fill in after inspecting raw profile>")
    lines.append("4. Risk: <fill in>")
    lines.append("5. Validation command after edit: <fill in>")
    lines.append("")
    lines.append("Next Edit")
    lines.append(f"- {next_edit}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a first-pass LightOp/DCU profile digest")
    parser.add_argument("--hipprof", type=Path, required=True, help="Candidate hipprof output")
    parser.add_argument("--baseline-hipprof", type=Path, help="Baseline or parent hipprof output")
    parser.add_argument("--benchmark-log", type=Path, help="Synchronized benchmark log")
    parser.add_argument("--kernel-regex", default="", help="Regex used to focus kernel lines")
    parser.add_argument("--operator", default="", help="LightOp operator name")
    parser.add_argument("--version", default="", help="Candidate version label")
    parser.add_argument("--lightop-root", default="", help="LightOp checkout root")
    parser.add_argument("--commit", default="", help="LightOp commit")
    parser.add_argument("--gpu", default="", help="GPU/gfx")
    parser.add_argument("--runtime", default="", help="DTK/ROCm/PyTorch versions")
    parser.add_argument("--build-command", default="", help="Build command")
    parser.add_argument("--benchmark-command", default="", help="Benchmark command")
    parser.add_argument("--shape", default="", help="Shape and dtype")
    parser.add_argument("--output", type=Path, required=True, help="Digest output path")
    args = parser.parse_args()

    digest = render(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(digest, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
