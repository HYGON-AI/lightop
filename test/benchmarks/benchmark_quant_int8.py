# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in INT8 quantization comparisons; no inference-framework dependency.

GPU events measure a graph containing repeated API calls. Compilation, table
initialization, correctness checks and host allocation are outside timing.
This is a warm-buffer microbenchmark, not an individual E2E kernel duration.
"""
import argparse
import hashlib
import importlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import lightop
from lightop import op
from lightop.quant import int8 as quant_api

oai = importlib.import_module("lightop.activation.swiglu_oai_int8_quant")


@triton.jit
def oai_triton_reference(X, Q, S, Counts, H: tl.constexpr, M: tl.constexpr,
                         MASKED: tl.constexpr, BLOCK: tl.constexpr):
    # Original split gate/up arithmetic: FP32 OAI -> BF16 -> row INT8.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    live = True
    if MASKED:
        live = row % M < tl.load(Counts + row // M)
    g = tl.load(X + row * (2 * H) + cols, live & (cols < H), 0).to(tl.float32)
    u = tl.load(X + row * (2 * H) + H + cols, live & (cols < H), 0).to(tl.float32)
    g = tl.minimum(g, 7.0)
    u = tl.maximum(tl.minimum(u, 7.0), -7.0)
    y = (g * tl.sigmoid(1.702 * g) * (u + 1.0)).to(X.dtype.element_ty).to(tl.float32)
    y = tl.where(cols < H, y, 0.0)
    scale = tl.where(live, tl.maximum(tl.max(tl.abs(y), 0), 1e-10) / 127.0, 1.0)
    q = libdevice.nearbyint(tl.div_rn(y, scale))
    tl.store(Q + row * H + cols, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), cols < H)
    tl.store(S + row, scale)


def oai_torch_reference(x):
    g, u = x.float().chunk(2, -1)
    g = g.clamp(max=7.0)
    y = (g * torch.sigmoid(1.702 * g) * (u.clamp(-7.0, 7.0) + 1)).bfloat16().float()
    scales = y.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    return (y / scales).round().clamp(-127, 127).to(torch.int8), scales


def check_oai(actual, expected, live=None):
    q, s = actual
    rq, rs = expected
    if live is not None:
        q, s, rq, rs = q[live], s[live], rq[live], rs[live]
    torch.testing.assert_close(s, rs, rtol=1e-6, atol=1e-12)
    delta = (q.short() - rq.short()).abs()
    if delta.numel():
        assert delta.max().item() <= 1
        assert (delta != 0).float().mean().item() <= 1e-4


def measure(functions, args):
    graphs = {}
    for name, fn in functions.items():
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iterations):
                fn()
        graphs[name] = graph
    samples = {name: [] for name in functions}
    for repeat in range(args.repeats):
        names = list(graphs)
        if repeat % 2:
            names.reverse()
        for name in names:
            graph = graphs[name]
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.iterations)
    return {name: {"median_us": statistics.median(v), "samples_us": v}
            for name, v in samples.items()}


def per_token_case(rows, width, args):
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    x[0].zero_()
    baseline = quant_api._per_token_quant_int8_triton
    expected = baseline(x)
    functions = {"original_triton": lambda: baseline(x)}
    original_method, original_enabled = quant_api._int8_method, quant_api._wide_int8_enabled
    try:
        quant_api._wide_int8_enabled = True
        for method in ("triton", "segments", "native"):
            def call(method=method):
                quant_api._int8_method = method
                return quant_api.per_token_quant_int8(x)
            actual = call()
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            functions["public_" + method] = call
        timing = measure(functions, args)
    finally:
        quant_api._int8_method, quant_api._wide_int8_enabled = original_method, original_enabled
    return {"operator": "per_token_quant_int8", "shape": list(x.shape),
            "native_fast_path_eligible": width in (3072, 6144) and rows >= 1024 and quant_api._wide_int8 is not None,
            "timings": timing, "correctness": "exact vs original Triton"}


def oai_case(rows, masked, distribution, args):
    shape = (16, rows, 6144) if masked else (rows, 6144)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    counts, live = None, None
    if masked:
        counts = torch.full((16,), rows, device="cuda", dtype=torch.int32)
        if distribution == "skewed":
            counts.copy_(torch.arange(16, device="cuda", dtype=torch.int32) * rows // 15)
        if distribution == "decode_sparse":
            counts.copy_(torch.tensor([0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 64],
                                      device="cuda", dtype=torch.int32).clamp(max=rows))
        live = torch.arange(rows, device="cuda")[None, :] < counts[:, None]
        x[~live] = float("nan")
    expected = oai_torch_reference(x)
    outputs = {}
    def buffers(name):
        q = torch.full((*shape[:-1], 3072), 99, device="cuda", dtype=torch.int8)
        s = torch.full((*shape[:-1], 1), -9.0, device="cuda", dtype=torch.float32)
        outputs[name] = q, s
        return q, s
    q, s = buffers("original_triton")
    def reference():
        oai_triton_reference[(x.numel() // 6144,)](
            x, q, s, counts, 3072, rows, masked, 4096, num_warps=4)
        return q, s
    functions = {"original_triton": reference}
    entries = {"public_masked": oai.swiglu_oai_int8_quant_masked,
               "public_masked_flat": oai.swiglu_oai_int8_quant_masked_flat} if masked else {
                   "public_contiguous": oai.swiglu_oai_int8_quant_contiguous}
    for name, fn in entries.items():
        oq, oscale = buffers(name)
        def call(fn=fn, oq=oq, oscale=oscale):
            if masked:
                return fn(x, counts, output=oq, scales=oscale)
            return fn(x, output=oq, scales=oscale)
        functions[name] = call
    for name, fn in functions.items():
        actual = fn()
        check_oai(actual, expected, live)
        if masked:
            if name == "public_masked_flat":
                assert (actual[0][~live] == 99).all() and (actual[1][~live] == -9).all()
            else:
                assert (actual[0][~live] == 0).all() and (actual[1][~live] == 1).all()
    return {"operator": "swiglu_oai_int8", "shape": list(shape),
            "distribution": distribution,
            "live_rows": int(counts.sum()) if masked else rows,
            "timings": measure(functions, args),
            "correctness": "independent PyTorch reference; padding contracts checked",
            "padding_note": "masked_flat skips padding; baseline and masked write zero/unit padding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "per-token", "oai"], default="all")
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 2048, 6144])
    parser.add_argument("--widths", type=int, nargs="+", default=[3072, 6144, 12288])
    parser.add_argument("--contiguous-rows", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--capacities", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows + args.widths + args.contiguous_rows + args.capacities
           + [args.warmup, args.iterations, args.repeats]) < 1:
        parser.error("shapes and timing counts must be positive")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("requires a HIP GPU")
    torch.manual_seed(args.seed)
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "-c", "safe.directory=" + str(repo), "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    artifacts = {}
    for module in (op, oai, quant_api):
        path = Path(module.__file__)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    native = oai._native
    if getattr(native, "__file__", None):
        path = Path(native.__file__)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    native_module = getattr(quant_api._wide_int8, "__module__", None)
    if native_module:
        path = Path(importlib.import_module(native_module).__file__)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [Path(__file__).resolve(),
                                  repo / "csrc/per_token_quant_int8_minimax.cu",
                                  repo / "csrc/swiglu_oai_int8_quant.cu",
                                  repo / "csrc/swiglu_oai_int8_quant_tables.cu"]}
    dtk = Path("/opt/dtk/.dtk_version")
    result = {"revision": revision, "torch": torch.__version__, "hip": torch.version.hip,
              "dtk": dtk.read_text().strip() if dtk.exists() else None,
              "device": str(torch.cuda.get_device_properties(0)), "artifact_sha256": artifacts,
              "source_sha256": source_hashes,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "method": "alternating-order GPU events; one graph of repeated calls; warm buffers; not E2E kernel timing",
              "cases": []}
    def record(case):
        result["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)
    if args.suite in ("all", "per-token"):
        for rows in args.rows:
            for width in args.widths:
                record(per_token_case(rows, width, args))
    if args.suite in ("all", "oai"):
        for rows in args.contiguous_rows:
            record(oai_case(rows, False, "full", args))
        for rows in args.capacities:
            for distribution in ("full", "skewed", "decode_sparse"):
                record(oai_case(rows, True, distribution, args))


if __name__ == "__main__":
    main()
