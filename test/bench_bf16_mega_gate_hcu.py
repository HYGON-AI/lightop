# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare public BF16 mega gate against GEMM + the existing HCU router.

Run explicitly; keep performance measurements out of default pytest/CI:

    python test/bench_bf16_mega_gate_hcu.py
    python test/bench_bf16_mega_gate_hcu.py --shapes 1025,384,5120,6 --return-logits
    python test/bench_bf16_mega_gate_hcu.py --mode graph --json-output mega_gate.json

Shapes are M,E,H,K. Both paths use identical BF16 inputs, FP32 correction bias,
normalization and scaling, and return INT64 IDs + FP32 weights. The split path
includes its INT32-to-INT64 ID conversion. --return-logits also checks the FP32
logits returned by both paths; it does not save internal logits work when off.
No padding mask is used in this benchmark.

Eager events enclose public calls (including allocation and launch gaps), not
just native kernel time. Graph mode captures --graph-calls public calls and
reports replay time per call, excluding capture and CPU allocation overhead.
Both paths are warmed up and checked before timing. Measurement order alternates
each round; output contains the median, min/max and all per-round averages.
These are operator measurements, not end-to-end inference throughput.
"""

import argparse
import importlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import torch

import lightop
from lightop import gemm_ops, op
from lightop.moe import bf16_mega_gate_hcu, moe_fused_gate_sqrtsoftplus_v41_hcu


DEFAULT_SHAPES = [(m, 384, 5120, 6) for m in (1, 8, 32, 64, 128, 256, 512, 1025)]


def parse_shape(value):
    try:
        m, e, h, k = (int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected M,E,H,K") from exc
    if m < 1 or e not in (256, 384) or h <= 0 or h % 256 or not 1 <= k <= 16:
        raise argparse.ArgumentTypeError(
            "comparison requires M>0, E=256/384, H>0 divisible by 256, K=1..16"
        )
    return m, e, h, k


def check_outputs(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype
        if a.dtype == torch.int64:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)


def make_calls(shape, scale, return_logits):
    m, e, h, k = shape
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(e, h, device="cuda", dtype=torch.bfloat16) * 0.03
    bias = torch.randn(e, device="cuda", dtype=torch.float32) * 0.05

    def split():
        logits = gemm_ops.gemm(x, weight, out_dtype=torch.float32)
        weights, ids = moe_fused_gate_sqrtsoftplus_v41_hcu(
            logits, bias, topk=k, renormalize=True,
            routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=True,
        )
        result = (ids.long(), weights)
        return (*result, logits) if return_logits else result

    def fused():
        return bf16_mega_gate_hcu(
            x, weight, k, False, 0, scale, 0, bias=bias,
            return_logits=return_logits,
        )

    expected = split()
    check_outputs(fused(), expected)
    # Independently check routing semantics on the split path's FP32 logits.
    logits = gemm_ops.gemm(x, weight, out_dtype=torch.float32)
    scores = torch.nn.functional.softplus(logits, threshold=20).sqrt()
    ids = (scores + bias).argsort(dim=-1, descending=True, stable=True)[:, :k]
    selected = scores.gather(1, ids)
    total = torch.zeros_like(selected[:, :1])
    for i in range(k):
        total = total + selected[:, i:i + 1]
    check_outputs(expected[:2], (ids, selected / total * scale))
    return {"split": split, "fused": fused}, expected


def capture(fn, count, expected):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(count):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()
    check_outputs(output, expected)
    return graph


def measure(fn, iters, calls_per_iteration):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iters * calls_per_iteration)


def benchmark(calls, expected, mode, args):
    # Warm the same stream used by capture, including the native counter cache.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in calls.values():
            for _ in range(args.warmup):
                fn()
        stream.synchronize()
        graphs = {}
        if mode == "graph":
            graphs = {name: capture(fn, args.graph_calls, expected)
                      for name, fn in calls.items()}
            timed = {name: graph.replay for name, graph in graphs.items()}
        else:
            timed = calls
        for fn in timed.values():
            fn()
        stream.synchronize()
        samples = {name: [] for name in timed}
        for round_id in range(args.rounds):
            order = list(timed) if round_id % 2 == 0 else list(reversed(timed))
            for name in order:
                samples[name].append(measure(
                    timed[name], args.iters, args.graph_calls if graphs else 1
                ))
    torch.cuda.current_stream().wait_stream(stream)
    result = {name: {"median_us": statistics.median(values),
                     "min_us": min(values), "max_us": max(values),
                     "round_us": values} for name, values in samples.items()}
    result["speedup"] = result["split"]["median_us"] / result["fused"]["median_us"]
    result["latency_change_percent"] = (1 / result["speedup"] - 1) * 100
    return result


def metadata(args):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    version = importlib.import_module("lightop.version")
    try:
        source_sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_sha = None
    return {"device": props.name, "arch": getattr(props, "gcnArchName", None),
            "torch": torch.__version__, "hip": torch.version.hip,
            "lightop": lightop.__file__, "extension": op.__file__,
            "wheel_version": getattr(version, "hcu_version", None),
            "wheel_git_hash": getattr(version, "git_hash", None),
            "dtk": getattr(version, "dtk", None), "script_git_sha": source_sha,
            "settings": vars(args)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=DEFAULT_SHAPES)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--graph-calls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--return-logits", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.graph_calls) < 1 or args.rounds < 3:
        parser.error("warmup/iters/graph-calls must be positive; rounds must be >=3")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be finite and positive for the split router")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("requires a HIP GPU")
    report = {"metadata": metadata(args), "results": []}
    print(json.dumps(report["metadata"], indent=2), flush=True)
    print("M,E,H,K  mode  split_us[min,max]  fused_us[min,max]  speedup", flush=True)
    modes = ("eager", "graph") if args.mode == "both" else (args.mode,)
    with torch.inference_mode():
        for shape in args.shapes:
            torch.manual_seed(args.seed)
            calls, expected = make_calls(shape, args.scale, args.return_logits)
            for mode in modes:
                result = benchmark(calls, expected, mode, args)
                report["results"].append({"shape": shape, "mode": mode, **result})
                split, fused = result["split"], result["fused"]
                print(f"{','.join(map(str, shape))} {mode} "
                      f"{split['median_us']:.3f}[{split['min_us']:.3f},{split['max_us']:.3f}] "
                      f"{fused['median_us']:.3f}[{fused['min_us']:.3f},{fused['max_us']:.3f}] "
                      f"{result['speedup']:.3f}x", flush=True)
                if args.json_output:
                    Path(args.json_output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
