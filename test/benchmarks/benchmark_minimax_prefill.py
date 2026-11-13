# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in four-op microbenchmark; see README_minimax_prefill.md for scope."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import lightop
from lightop.activation import swiglu_oai_fp8_quant_contiguous
from lightop.attention import minimax_qk_norm_rope
from lightop.kvcache import minimax_cp_cache_scatter
from lightop.quant import minimax_cp_fp8_pack


def qk_case(rows):
    packed = torch.randn((rows, 9856), device="cuda", dtype=torch.bfloat16)
    xs = [packed[:, :8192], packed[:, 8192:8704], packed[:, 9216:9728], packed[:, 9728:9856]]
    weights = [torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in xs]
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    angles = torch.randn((rows, 32), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

    def unfused():
        cos, sin = cos_sin[positions].chunk(2, dim=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        outputs = []
        for x, weight in zip(xs, weights):
            x = x.reshape(rows, -1, 128).float()
            y = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
                 * (1.0 + weight.float())).bfloat16()
            a, b = y[..., :32].float(), y[..., 32:64].float()
            rotary = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).bfloat16()
            outputs.append(torch.cat((rotary, y[..., 64:]), dim=-1).flatten(1))
        return tuple(outputs)

    def fused():
        return minimax_qk_norm_rope(*xs, *weights, positions, cos_sin, 1e-5, 128, 64, True)

    def check(actual, expected):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0.016, atol=0.015625)
            relative_rms = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            assert relative_rms.item() <= 0.003

    return unfused, fused, check, {"packed_width": 9856, "rotary_dim": 64, "head_dim": 128}


def pack_case(rows):
    source = torch.randn((rows, 1152), device="cuda", dtype=torch.bfloat16) * 20
    k, v, ik = source[:, :512], source[:, 512:1024], source[:, 1024:]

    def unfused():
        return torch.cat(((k / 0.37).bfloat16().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8),
                          (v / 1.7).bfloat16().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8),
                          ik.view(torch.uint8)), dim=-1)

    def fused():
        return minimax_cp_fp8_pack(k, v, ik, 0.37, 1.7)

    def check(a, b):
        assert torch.equal(a, b)

    return unfused, fused, check, {"k_width": 512, "v_width": 512, "index_width": 128, "k_scale": 0.37, "v_scale": 1.7}


def scatter_case(rows):
    packet = torch.randint(0, 256, (rows, 1280), device="cuda", dtype=torch.uint8)
    order = torch.randperm(rows, device="cuda")
    # Allocator-like unique positive slots: no device mask/nonzero allocation in timing.
    slots = torch.randperm(rows, device="cuda") + 1
    unfused_caches = [torch.full((rows + 1, width), 31, device="cuda", dtype=torch.uint8)
                      for width in (512, 512, 256)]
    fused_caches = [x.clone() for x in unfused_caches]

    def unfused():
        ordered = packet.index_select(0, order)
        for cache, start, width in zip(unfused_caches, (0, 512, 1024), (512, 512, 256)):
            cache.index_copy_(0, slots, ordered[:, start:start + width])
        return tuple(unfused_caches)

    def fused():
        minimax_cp_cache_scatter(packet, order, slots, *fused_caches)
        return tuple(fused_caches)

    def check(actual, expected):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))

    return unfused, fused, check, {"packet_bytes_per_row": 1280, "slots": "unique_positive"}


def oai_case(rows):
    x = torch.randn((rows, 6144), device="cuda", dtype=torch.bfloat16) * 5
    x[0].zero_()

    def unfused():
        gate, up = x.float().chunk(2, dim=-1)
        gate, up = gate.clamp(max=7), up.clamp(-7, 7)
        activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).bfloat16().float()
        scale = activated.abs().amax(-1, keepdim=True) / 448.0
        inverse = torch.where(scale > 0, scale.reciprocal(), torch.zeros_like(scale))
        return (activated * inverse).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    def fused():
        return swiglu_oai_fp8_quant_contiguous(x)

    def check(actual, expected):
        q, s = actual
        qr, sr = expected
        torch.testing.assert_close(s, sr, rtol=1e-6, atol=1e-7)
        a, b = q.float() * s, qr.float() * sr
        assert ((a - b).norm() / b.norm().clamp_min(1e-20)).item() < 2e-3
        assert s[0].item() == 0 and q[0].float().count_nonzero().item() == 0

    return unfused, fused, check, {"input_width": 6144, "alpha": 1.702, "beta": 1.0, "limit": 7.0}


def runner(fn, mode, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if mode == "eager":
        return fn, None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    # Retain graph outputs and allocation lifetime through all timed replays.
    return graph.replay, (graph, outputs)


def measure(fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 2048, 8192])
    parser.add_argument("--op", choices=["all", "qk", "pack", "scatter", "oai"], default="all")
    parser.add_argument("--mode", choices=["eager", "graph"], default="graph")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("rows, warmup, iterations and repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new result filename")
    if not torch.cuda.is_available():
        parser.error("an available HCU with HIP PyTorch is required")
    props = torch.cuda.get_device_properties(0)
    result = {"torch": torch.__version__, "hip": torch.version.hip,
              "lightop": lightop.__file__, "native": getattr(lightop.op, "__file__", None),
              "device": str(props), "args": {**vars(args), "output": str(args.output)},
              "baseline": "unfused PyTorch operations; not a previous LightOp kernel or end-to-end model",
              "results": []}
    cases = {"qk": qk_case, "pack": pack_case, "scatter": scatter_case, "oai": oai_case}
    for name, make in cases.items():
        if args.op not in ("all", name):
            continue
        for rows in args.rows:
            torch.manual_seed(args.seed)
            reference, fused, check, shape = make(rows)
            check(fused(), reference())
            baseline_run, baseline_keep = runner(reference, args.mode, args.warmup)
            fused_run, fused_keep = runner(fused, args.mode, args.warmup)
            if args.mode == "graph":
                baseline_run()
                fused_run()
                torch.cuda.synchronize()
                check(fused_keep[1], baseline_keep[1])
            times = {"unfused_us": [], "fused_us": []}
            for repeat in range(args.repeats):
                order = [("unfused_us", baseline_run), ("fused_us", fused_run)]
                for label, fn in (order if repeat % 2 == 0 else reversed(order)):
                    times[label].append(measure(fn, args.iterations))
            row = {"op": name, "rows": rows, "shape": shape, **times,
                   "unfused_median_us": statistics.median(times["unfused_us"]),
                   "fused_median_us": statistics.median(times["fused_us"]), "correctness": "passed"}
            row["speedup"] = row["unfused_median_us"] / row["fused_median_us"]
            result["results"].append(row)
            print(json.dumps(row), flush=True)
            # Do not retain previous cases' graph pools and inputs.
            del baseline_run, fused_run, baseline_keep, fused_keep, reference, fused, check
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
