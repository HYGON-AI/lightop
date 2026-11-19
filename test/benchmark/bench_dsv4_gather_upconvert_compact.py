# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit HCU benchmark for the public DPSKV4 compact gather interface.

    python test/benchmark/bench_dsv4_gather_upconvert_compact.py --help
    python test/benchmark/bench_dsv4_gather_upconvert_compact.py --json-output gather.json
    python test/benchmark/bench_dsv4_gather_upconvert_compact.py --mode graph --rounds 7
    python test/benchmark/bench_dsv4_gather_upconvert_compact.py --shapes 24,128,512,64,6

Shapes are Q,SWA_K,EXTRA_K,EXTRA_PAGE,REUSE_GROUP. Each case checks the legacy
single and dual gather paths, combined gather, and query reuse against the same
independent reference before timing. Full valid lengths make layout conversion
explicit: legacy paths include concatenating the two BF16 outputs and remapping
their compact indices to the combined output. All four paths produce the same
logical rows; query reuse may point multiple compact indices at a shared row.

All workspaces are preallocated. Timing uses events on one current stream;
eager timings include launch gaps, graph timings include replay overhead. Path
order rotates each round. Report median, quartile spread, min/max, and every
round, without asserting a performance threshold. Shared and independent index
patterns expose both reuse opportunities and its fallback cost. These are
operator measurements, not end-to-end decode performance.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys


DEFAULT_SHAPES = [
    (5, 192, 0, 256, 5),
    (6, 128, 64, 2, 6),
    (12, 128, 512, 64, 6),
    (24, 128, 512, 128, 6),
    (48, 128, 1024, 256, 6),
    (20, 192, 0, 256, 5),
]


def parse_shape(value):
    try:
        q, ak, bk, page, group = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected Q,SWA_K,EXTRA_K,EXTRA_PAGE,REUSE_GROUP") from exc
    if (q < 1 or ak < 1 or bk < 0 or page not in (2, 64, 128, 256)
            or group not in (5, 6) or q % group or (group == 6 and ak < 5)):
        raise argparse.ArgumentTypeError("invalid shape, page size or complete reuse group")
    return q, ak, bk, page, group


def summarize(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return {
        "median_us": statistics.median(samples),
        "iqr_us": quartiles[2] - quartiles[0],
        "min_us": min(samples),
        "max_us": max(samples),
        "round_us": samples,
    }


def make_case(torch, op, compact_gather, shape, pattern):
    # Direct execution puts test/benchmark, rather than test/, on sys.path.
    test_dir = str(Path(__file__).resolve().parents[1])
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    from ci_tests.test_dsv4_gather_upconvert_k_cache import _make_cache, _reference

    q, ak, bk, page, group = shape
    a = _make_cache((ak + 255) // 256, 256, seed=31)
    b = _make_cache(max(1, (bk + page - 1) // page), page, seed=37)

    def packed(raw, size):
        return torch.as_strided(raw, (raw.shape[0], size, 1, 584),
                                (raw.stride(0), 584, 584, 1))

    ai = torch.arange(ak, device="cuda", dtype=torch.int32).repeat(q, 1)
    bi = torch.arange(bk, device="cuda", dtype=torch.int32).repeat(q, 1)
    if pattern == "independent":
        # Every adjacent row has a different ordering, so it cannot be shared.
        for row in range(q):
            ai[row] = ai[row].roll(-(row % group))
            if bk:
                bi[row] = bi[row].roll(-(row % group))
    al = torch.full((q,), ak, device="cuda", dtype=torch.int32)
    bl = torch.full((q,), bk, device="cuda", dtype=torch.int32)
    expected = torch.cat((_reference(a, ai, al, 256)[0],
                          _reference(b, bi, bl, page)[0]), dim=1)
    legacy_a = torch.empty((q, ak, 1, 512), device="cuda", dtype=torch.bfloat16)
    legacy_b = torch.empty((q, bk, 1, 512), device="cuda", dtype=torch.bfloat16)
    a_ids = torch.empty((q, 1, ak), device="cuda", dtype=torch.int32)
    b_ids = torch.empty((q, 1, bk), device="cuda", dtype=torch.int32)
    joined = torch.empty_like(expected)
    joined_ids = torch.empty((q, 1, ak + bk), device="cuda", dtype=torch.int32)
    row_ids = torch.arange(q, device="cuda", dtype=torch.int32).view(q, 1, 1)
    a_offset, b_offset = row_ids * bk, row_ids * ak + ak
    compact_outputs = {name: (torch.empty_like(joined), torch.empty_like(joined_ids))
                       for name in ("combined", "query_reuse")}
    pa, pb = packed(a, 256), packed(b, page)

    def remap_legacy():
        torch.cat((legacy_a, legacy_b), dim=1, out=joined)
        torch.add(a_ids, a_offset, out=joined_ids[:, :, :ak])
        if bk:
            torch.add(b_ids, b_offset, out=joined_ids[:, :, ak:])
        return joined, joined_ids

    def legacy_single_chain():
        op.dsv4_gather_upconvert_k_cache_paged(a, ai, al, legacy_a, a_ids, 256)
        if bk:
            op.dsv4_gather_upconvert_k_cache_paged(b, bi, bl, legacy_b, b_ids, page)
        return remap_legacy()

    def legacy_dual_chain():
        op.dsv4_gather_upconvert_dual_k_cache_paged(
            pa, ai, al, legacy_a, a_ids, 256,
            pb, bi, bl, legacy_b, b_ids, page,
        )
        return remap_legacy()

    def combined():
        out, ids = compact_outputs["combined"]
        compact_gather(pa, ai, al, pb, bi, bl, out, ids, page, reuse_group=1)
        return out, ids

    def query_reuse():
        out, ids = compact_outputs["query_reuse"]
        compact_gather(pa, ai, al, pb, bi, bl, out, ids, page, reuse_group=group)
        return out, ids

    calls = {"legacy_single_chain": legacy_single_chain,
             "legacy_dual_chain": legacy_dual_chain,
             "combined": combined, "query_reuse": query_reuse}
    def check_output(name, output):
        out, ids = output
        actual = out.view(-1, 1, 512)[ids.long()].reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg=lambda message: f"{name}: {message}")
    for name, call in calls.items():
        check_output(name, call())
    return calls, check_output


def measure(torch, calls, check_output, args):
    stream = torch.cuda.current_stream()
    timed = {}
    for name, call in calls.items():
        for _ in range(args.warmup):
            call()
        stream.synchronize()
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = call()
            # Reject empty/broken captures rather than reporting a false speedup.
            output[0].fill_(float("nan"))
            output[1].fill_(-1)
            graph.replay()
            stream.synchronize()
            check_output(name, output)
            timed[name] = graph.replay
        else:
            timed[name] = call
    samples = {name: [] for name in timed}
    names = list(timed)
    for round_id in range(args.rounds):
        order = names[round_id % len(names):] + names[:round_id % len(names)]
        for name in order:
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record(stream)
            for _ in range(args.iterations):
                timed[name]()
            end.record(stream)
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / args.iterations)
    return {name: summarize(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+", default=DEFAULT_SHAPES)
    parser.add_argument("--patterns", nargs="+", choices=("shared", "independent"),
                        default=["shared", "independent"])
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.rounds < 3 or args.iterations < 1 or args.warmup < 1:
        parser.error("rounds must be >=3; iterations and warmup must be positive")

    import torch
    from lightop import op
    from lightop.kvcache import dsv4_gather_upconvert_compact_k_cache_paged

    if not torch.cuda.is_available() or not torch.version.hip:
        parser.error("requires an HCU device, HIP PyTorch and a matching LightOp build")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    result = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "lightop": importlib.metadata.version("lightop"),
        "device": str(props), "arch": getattr(props, "gcnArchName", None),
        "mode": args.mode, "rounds": args.rounds, "iterations": args.iterations,
        "warmup": args.warmup, "cases": [],
    }
    for shape in args.shapes:
        for pattern in args.patterns:
            calls, check_output = make_case(
                torch, op, dsv4_gather_upconvert_compact_k_cache_paged, shape, pattern,
            )
            record = {"shape": shape, "pattern": pattern, "correctness": "passed",
                      "timing": measure(torch, calls, check_output, args)}
            result["cases"].append(record)
            print(json.dumps(record), flush=True)
            if args.json_output:
                args.json_output.write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
