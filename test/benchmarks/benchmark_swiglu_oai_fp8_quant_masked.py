# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in masked OAI quantization benchmark; no additional dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

import lightop
from lightop import op
from lightop.activation import swiglu_oai_fp8_quant_masked


def check(x, counts, q, scales):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    values = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7.0, 7.0) + 1)
    ref_scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    ref = (values / ref_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    live = torch.arange(x.shape[1], device=x.device)[None, :] < counts[:, None]
    torch.testing.assert_close(scales[live], ref_scale[live], rtol=2e-6, atol=1e-12)
    a = q[live].view(torch.uint8).to(torch.int16)
    b = ref[live].view(torch.uint8).to(torch.int16)
    assert ((a & 127) - (b & 127)).abs().max().item() <= 1
    nonzero = ((a & 127) != 0) & ((b & 127) != 0)
    assert torch.equal(a[nonzero] & 128, b[nonzero] & 128)
    assert (q.view(torch.uint8)[~live] == 165).all().item()
    assert (scales[~live] == -123).all().item()


def run_case(capacity, width, distribution, args):
    x = torch.randn(16, capacity, 2 * width, device="cuda", dtype=torch.bfloat16)
    # Equal live-row totals: balanced across 16 experts vs concentrated on four.
    per_expert = capacity // 4
    raw = [per_expert] * 16 if distribution == "balanced" else [4 * per_expert] * 4 + [0] * 12
    counts = torch.tensor(raw, device=x.device, dtype=torch.int32)
    q = torch.empty(16, capacity, width, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(16, capacity, 1, device=x.device, dtype=torch.float32)
    variants = {"public": lambda: swiglu_oai_fp8_quant_masked(x, counts, q, scales)}
    configs = [("general", 32, 256), ("flat", 256, 256)]
    if args.sweep:
        configs = [("general", blocks, threads)
                   for blocks in (16, 32, 64)
                   for threads in (64, 128, 256)] + [
                       ("flat", blocks, threads) for blocks in (128, 256, 512)
                       for threads in (64, 128, 256)]
    for kind, blocks, threads in configs:
        fn = getattr(op, "swiglu_oai_fp8_quant_masked" + ("_flat" if kind == "flat" else ""))
        variants[f"{kind}_blocks{blocks}_threads{threads}"] = (
            lambda fn=fn, blocks=blocks, threads=threads:
            fn(x, q, scales, counts, 1.702, 7.0, blocks, threads)
        )
    runners = {}
    for name, fn in variants.items():
        q.view(torch.uint8).fill_(165)
        scales.fill_(-123)
        fn()
        check(x, counts, q, scales)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.iterations):
                    fn()
            q.view(torch.uint8).fill_(165)
            scales.fill_(-123)
            graph.replay()
            check(x, counts, q, scales)
            runners[name] = graph.replay
        else:
            def repeat(fn=fn):
                for _ in range(args.iterations):
                    fn()
            runners[name] = repeat
    samples = {name: [] for name in runners}
    names = list(runners)
    for repeat in range(args.repeats):
        # Rotate measurement order to reduce a fixed warm-clock/order bias.
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.iterations)
    medians = {name: statistics.median(times) for name, times in samples.items()}
    baseline = medians["general_blocks32_threads256"]
    return {
        "shape": list(x.shape), "counts": raw, "live_rows": sum(raw),
        "distribution": distribution, "correctness": "FP32 oracle + padding passed",
        "median_us": medians, "samples_us": samples,
        "speedup_vs_general32x256": {name: baseline / value for name, value in medians.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--mode", choices=["eager", "graph"], default="graph")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(c < 4 for c in args.capacities) or min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("capacities must be >=4; timing counts must be positive")
    if not 8 <= args.width <= 4096 or args.width % 8:
        parser.error("width must be a multiple of eight in [8,4096]")
    if not torch.cuda.is_available():
        parser.error("requires a GPU")
    torch.manual_seed(args.seed)
    native_path = Path(op.__file__)
    result = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        "package": lightop.__file__, "native_path": str(native_path),
        "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "GPU event microbenchmark with persistent buffers; no allocations/oracle/count copies timed. Not end-to-end performance.",
        "cases": [],
    }
    for capacity in args.capacities:
        for distribution in ("balanced", "skewed"):
            case = run_case(capacity, args.width, distribution, args)
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
