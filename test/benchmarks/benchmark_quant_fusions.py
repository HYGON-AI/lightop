# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 transform/quantization comparisons on HIP; no serving framework.

Each pair has identical input values and output layouts and must pass bitwise
checks before timing. Standalone Triton RoPE/scale stages model the split
preparation chain; the INT8 RoPE baseline explicitly uses half-away rounding.
FP8 and scale-to-INT8 baselines reuse existing LightOp quantization. RMS pairs
compare copy + existing fused RMS/quant against direct strided RMS/quant.
These are warm-buffer operator comparisons, not historical service A/B results.

Build and install the regular LightOp package first (python3 setup.py install
or scripts/build_hcu_wheel.sh), using matching DTK, HIP PyTorch and Python ABI.
The benchmark also needs Triton for independent split-stage references.

Run from the repository root, for example::

    python3 test/benchmarks/benchmark_quant_fusions.py --tokens 40 48 --output /tmp/quant-fusions.json
    python3 test/benchmarks/benchmark_quant_fusions.py --suite rope-int8 --tokens 40 48 64 96 128 --output /tmp/rope-int8.json
    python3 test/benchmarks/benchmark_quant_fusions.py --suite rope-fp8 --tokens 40 48 64 96 128 --output /tmp/rope-fp8.json

Defaults: BF16, RoPE [T,64,512] with 8 groups and 64 rotary channels;
INT32 scale [T,8,1024]; RMS widths 1280/1536/3072. Both FP8 group layouts,
INT32 floating-output materialization modes, and padded inputs are covered.
Use --rms-widths to vary RMS columns (INT8 <=8192, FP8 <=16384).

Default timing uses 20 warmup calls, 50 calls per captured graph, and 7 repeats
with alternating variant order. Compilation, setup and parity checks are
outside timing. Output JSON records every sample, medians, latency reduction,
package/source versions, HIP/device properties, and native binary SHA256.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Delay GPU imports so --help works without a HIP/PyTorch installation.
torch = triton = tl = libdevice = None

SUITES = (
    "all",
    "rope-int8",
    "scale-int8",
    "rope-fp8",
    "strided-fp8",
    "rms-int8",
    "rms-fp8",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--tokens", nargs="+", type=int, default=[40, 48])
    parser.add_argument("--rms-widths", nargs="+", type=int, default=[1280, 1536, 3072])
    parser.add_argument("--int8-pitch-padding", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.tokens + args.rms_widths + [args.warmup, args.iterations, args.repeats]
        )
        < 1
    ):
        parser.error("shapes and timing counts must be positive")
    if args.device < 0 or args.int8_pitch_padding < 0 or args.int8_pitch_padding % 16:
        parser.error(
            "device must be nonnegative; pitch padding must be a nonnegative multiple of 16"
        )
    if any(width % 16 or width > 16384 for width in args.rms_widths):
        parser.error("RMS widths must be multiples of 16 and at most 16384")
    if args.suite == "rms-int8" and max(args.rms_widths) > 8192:
        parser.error("INT8 RMS widths must be at most 8192")
    return parser, args


def load_runtime():
    global torch, triton, tl, libdevice
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    import lightop
    from lightop import op, rms_norm_dynamic_per_token_quant_sglang
    from lightop.norm import rms_norm_per_token_fp8_quant
    from lightop.quant import per_token_quant_fp8, per_token_quant_int8, transforms

    return SimpleNamespace(
        lightop=lightop,
        op=op,
        transforms=transforms,
        quant_int8=per_token_quant_int8,
        quant_fp8=per_token_quant_fp8,
        norm_int8=rms_norm_dynamic_per_token_quant_sglang,
        norm_fp8=rms_norm_per_token_fp8_quant,
    )


def reference_kernels():
    @triton.jit
    def rope(
        X,
        Y,
        F,
        P,
        H: tl.constexpr,
        D: tl.constexpr,
        RD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK) * 2
        base = row * D + D - RD
        pos = tl.load(P + row // H).to(tl.int64)
        real = tl.load(X + base + col, col < RD, 0).to(tl.float32)
        imag = tl.load(X + base + col + 1, col < RD, 0).to(tl.float32)
        cosine = tl.load(F + pos * RD + col, col < RD, 0)
        sine = tl.load(F + pos * RD + col + 1, col < RD, 0)
        ri, ii = real * sine, imag * sine
        # Inverse RoPE: retain separate products and explicit FP32 FMA.
        out_real = tl.fma(real, cosine, ii)
        # Unary minus can lower to 0 - x and erase -0.0. Flip the sign bit
        # to match native RoPE before the BF16/FP8 rounding boundaries.
        negative_ri = (ri.to(tl.uint32, bitcast=True) ^ 0x80000000).to(
            tl.float32, bitcast=True
        )
        out_imag = tl.fma(imag, cosine, negative_ri)
        tl.store(Y + base + col, out_real, col < RD)
        tl.store(Y + base + col + 1, out_imag, col < RD)

    @triton.jit
    def quant_away(
        X, Q, S, WIDTH: tl.constexpr, PITCH: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        value = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(value)), 1e-10)
        quantized = libdevice.round(value * (127.0 / absmax)).to(tl.int8)
        tl.store(Q + row * PITCH + col, quantized, col < WIDTH)
        tl.store(S + row, absmax / 127.0)

    @triton.jit
    def scale(
        ACC,
        XS,
        WS,
        Y,
        TOKENS: tl.constexpr,
        GROUPS: tl.constexpr,
        RANK: tl.constexpr,
        MAJOR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        group = col // RANK
        offset = token * (GROUPS * RANK) + col
        if MAJOR:
            offset = (group * TOKENS + token) * RANK + col % RANK
        value = tl.load(ACC + offset, col < GROUPS * RANK, 0).to(tl.float32)
        xs = tl.load(XS + token * GROUPS + group, col < GROUPS * RANK, 0)
        ws = tl.load(WS + col, col < GROUPS * RANK, 0)
        value = (value * xs) * ws
        # Store the BF16 rounding boundary before the separate INT8 quantizer.
        tl.store(Y + token * (GROUPS * RANK) + col, value, col < GROUPS * RANK)

    return SimpleNamespace(rope=rope, quant_away=quant_away, scale=scale)


def check_bits(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("output count differs")
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if value is None or reference is None:
            if value is not reference:
                raise AssertionError(
                    f"output {index}: floating materialization differs"
                )
            continue
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError(f"output {index}: shape/dtype differs")
        if not torch.equal(
            value.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        ):
            raise AssertionError(
                f"output {index}: bitwise mismatch; refusing to report timing"
            )


def measure_pair(unfused, fused, args):
    functions = {"unfused": unfused, "fused": fused}
    expected = tuple(
        None if value is None else value.clone(memory_format=torch.contiguous_format)
        for value in unfused()
    )
    check_bits(fused(), expected)
    runners, retained_outputs = {}, {}
    for name, fn in functions.items():
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iterations):
                outputs = fn()
        graph.replay()
        retained_outputs[name] = outputs
        check_bits(outputs, expected)
        runners[name] = graph
    samples = {name: [] for name in functions}
    for repeat in range(args.repeats):
        names = list(runners)
        if repeat % 2:
            names.reverse()
        for name in names:
            runners[name].replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / args.iterations)
    for outputs in retained_outputs.values():
        check_bits(outputs, expected)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    if min(medians.values()) <= 0:
        raise RuntimeError("timer resolution insufficient; increase --iterations")
    return {
        "correctness": "bitwise output checks passed before/after graph replay",
        "median_us": medians,
        "samples_us": samples,
        "speedup": medians["unfused"] / medians["fused"],
        "latency_reduction_pct": 100.0 * (1.0 - medians["fused"] / medians["unfused"]),
    }


def rope_case(api, kernels, tokens, fp8, major, args):
    groups, heads, dim, rotary = 8, 64, 512, 64
    x = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    before = x.clone()
    rotated = x.clone()
    frequencies = torch.polar(
        torch.ones(65536 + tokens, rotary // 2, device=x.device, dtype=torch.float32),
        torch.randn(65536 + tokens, rotary // 2, device=x.device, dtype=torch.float32),
    )
    positions = torch.arange(tokens, device=x.device, dtype=torch.int64) + 65536
    width = heads // groups * dim
    pitch = width + args.int8_pitch_padding

    def unfused():
        # Prefix values are immutable and initialized outside timing. Each call
        # overwrites the tail from x, so replays never apply RoPE repeatedly.
        kernels.rope[(tokens * heads,)](
            x,
            rotated,
            torch.view_as_real(frequencies),
            positions,
            heads,
            dim,
            rotary,
            triton.next_power_of_2(rotary // 2),
            num_warps=1,
            enable_fp_fusion=False,
        )
        if fp8:
            q, scales = api.quant_fp8(rotated.view(tokens, groups, width))
            if major:
                q, scales = (
                    q.transpose(0, 1).contiguous(),
                    scales.transpose(0, 1).contiguous(),
                )
            return q, scales
        storage = torch.empty(
            (tokens, groups, pitch), device=x.device, dtype=torch.int8
        )
        q = storage[..., :width]
        scales = torch.empty((tokens, groups, 1), device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(width)
        kernels.quant_away[(tokens * groups,)](
            rotated,
            q,
            scales,
            width,
            pitch,
            block,
            num_warps=min(max(block // 256, 1), 8),
        )
        return q, scales

    def fused():
        return api.transforms.rope_per_token_quant(
            x,
            frequencies,
            positions,
            groups,
            rotary,
            True,
            dtype=torch.float8_e4m3fn if fp8 else torch.int8,
            group_major=major,
            output_pitch=None if fp8 else pitch,
        )

    result = measure_pair(unfused, fused, args)
    check_bits((x,), (before,))
    return {
        "operator": "rope_per_token_quant",
        "quant_dtype": "fp8" if fp8 else "int8",
        "shape": list(x.shape),
        "groups": groups,
        "rope_dim": rotary,
        "output_layout": "group-major" if major else "token-major",
        "output_pitch": width if fp8 else pitch,
        "baseline": "standalone inverse RoPE + "
        + (
            "existing LightOp FP8 quant + optional quantized layout copies"
            if fp8
            else "standalone half-away INT8 quant into the same pitched layout"
        ),
        **result,
    }


def scale_case(api, kernels, tokens, major, materialize, args):
    groups, rank = 8, 1024
    shape = (groups, tokens, rank) if major else (tokens, groups, rank)
    raw = torch.randint(-100000, 100000, shape, device="cuda", dtype=torch.int32)
    xs = torch.rand(tokens, groups, 1, device="cuda", dtype=torch.float32) * 0.02
    ws = torch.rand(groups * rank, device="cuda", dtype=torch.float32) * 0.03
    if major:
        raw[:, 0].zero_()
    else:
        raw[0].zero_()
    before = raw.clone()
    floating = torch.empty(
        (tokens, groups, rank), device=raw.device, dtype=torch.bfloat16
    )

    def unfused():
        kernels.scale[(tokens,)](
            raw,
            xs,
            ws,
            floating,
            tokens,
            groups,
            rank,
            major,
            triton.next_power_of_2(groups * rank),
            num_warps=8,
            enable_fp_fusion=False,
        )
        q, scales = api.quant_int8(floating.view(tokens, groups * rank))
        return floating if materialize else None, q, scales

    def fused():
        return api.transforms.scale_per_token_quant_int8(
            raw,
            xs,
            ws,
            group_major=major,
            materialize_output=materialize,
        )

    result = measure_pair(unfused, fused, args)
    check_bits((raw,), (before,))
    return {
        "operator": "scale_per_token_quant_int8",
        "shape": list(shape),
        "input_layout": "group-major" if major else "token-major",
        "materialize_output": materialize,
        "baseline": "standalone INT32 scale -> BF16 store + existing LightOp RNE INT8 quant",
        **result,
    }


def strided_fp8_case(api, tokens, major, args):
    shape = (8, tokens, 1032) if major else (tokens, 8, 1032)
    storage = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = storage[..., :1024].transpose(0, 1) if major else storage[..., :1024]
    x[0].zero_()
    before = storage.clone()
    result = measure_pair(
        lambda: api.quant_fp8(x.contiguous().view(tokens, 8192)),
        lambda: api.quant_fp8(x, quant_ndim=2),
        args,
    )
    check_bits((storage,), (before,))
    return {
        "operator": "per_token_quant_fp8",
        "quant_ndim": 2,
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "input_layout": "group-major" if major else "token-major",
        "baseline": "materialize token-major floating rows + existing LightOp FP8 quant",
        **result,
    }


def rms_case(api, tokens, width, fp8, args):
    storage = torch.randn(tokens, width + 512, device="cuda", dtype=torch.bfloat16)
    x = storage[:, :width]
    x[0].zero_()
    before = storage.clone()
    weight = torch.randn(width, device=x.device, dtype=x.dtype)

    def unfused():
        normalized = x.clone(memory_format=torch.contiguous_format)
        if fp8:
            q, scales = api.norm_fp8(
                normalized, weight, epsilon=1e-6, update_input=True
            )
        else:
            q, scales = api.norm_int8(normalized, weight, 1e-6)
        return normalized, q, scales

    def fused():
        normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        fn = api.norm_fp8 if fp8 else api.norm_int8
        q, scales = fn(x, weight, 1e-6, update_input=False, norm_output=normalized)
        return normalized, q, scales

    result = measure_pair(unfused, fused, args)
    check_bits((storage,), (before,))
    return {
        "operator": (
            "rms_norm_per_token_fp8_quant"
            if fp8
            else "rms_norm_dynamic_per_token_quant_sglang"
        ),
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "baseline": "copy padded rows + existing native fused RMSNorm/quant (same rounding/floor)",
        **result,
    }


def file_record(module):
    path = getattr(module, "__file__", None)
    if path is None:
        return None
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def environment(api, args):
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    try:
        version = importlib.metadata.version("lightop")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(api.lightop, "__version__", "unavailable")
    symbols = {
        "rope_quant": api.transforms._rope,
        "scale_int8": api.transforms._scale,
        "quant_fp8": api.op.per_token_quant_fp8,
        "rms_int8": api.op.rms_norm_dynamic_per_token_quant_sglang,
        "rms_fp8": api.op.rms_norm_fp8_quant,
    }
    binaries = {"lightop.op": file_record(api.op)}
    bindings = {}
    for name, fn in symbols.items():
        module = getattr(fn, "__module__", None)
        bindings[name] = module
        if module and module not in binaries:
            binaries[module] = file_record(sys.modules.get(module))
    properties = torch.cuda.get_device_properties(args.device)
    return {
        "revision": revision,
        "package": api.lightop.__file__,
        "version": str(version),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "hip": torch.version.hip,
        "device": str(properties),
        "device_arch": getattr(properties, "gcnArchName", "unavailable"),
        "compute_units": properties.multi_processor_count,
        "activation_dtype": "torch.bfloat16",
        "native_bindings": bindings,
        "native_artifacts": binaries,
        "benchmark": file_record(sys.modules[__name__]),
    }


def cases(api, kernels, args):
    for tokens in args.tokens:
        if args.suite in ("all", "rope-int8"):
            yield rope_case(api, kernels, tokens, False, False, args)
        if args.suite in ("all", "scale-int8"):
            for major in (False, True):
                for materialize in (False, True):
                    yield scale_case(api, kernels, tokens, major, materialize, args)
        if args.suite in ("all", "rope-fp8"):
            for major in (False, True):
                yield rope_case(api, kernels, tokens, True, major, args)
        if args.suite in ("all", "strided-fp8"):
            for major in (False, True):
                yield strided_fp8_case(api, tokens, major, args)
        for fp8 in (False, True):
            if args.suite in ("all", "rms-fp8" if fp8 else "rms-int8"):
                for width in args.rms_widths:
                    if fp8 or width <= 8192:
                        yield rms_case(api, tokens, width, fp8, args)


def main():
    parser, args = parse_args()
    try:
        api = load_runtime()
    except (ImportError, OSError) as error:
        parser.error(
            f"requires matching HIP PyTorch, Triton and installed LightOp: {error}"
        )
    if torch.version.hip is None or not torch.cuda.is_available():
        parser.error("requires a HIP GPU")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    report = {
        "status": "running",
        "environment": environment(api, args),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "note": (
            "GPU events per call in repeated CUDA/HIP graphs; public allocations are captured, "
            "not replayed on CPU. Compilation, setup and bitwise checks are outside timing. "
            "Split scratch buffers are persistent; RoPE prefixes are immutable. "
            "Warm-buffer microbenchmarks, not historical service A/B or end-to-end speedups. "
            "RMS compares copy+existing fused RMS/quant; INT8 RMS widths above 8192 are FP8-only."
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({key: value for key, value in report.items() if key != "cases"}),
        flush=True,
    )
    try:
        with torch.inference_mode():
            for case in cases(api, reference_kernels(), args):
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                args.output.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
