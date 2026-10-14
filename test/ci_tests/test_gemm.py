# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GEMM and W4A16 correctness and GPU timing checks on supported HCU GPUs.

Run both suites and save GPU timings to CSV: python -m pytest -v test/ci_tests/test_gemm.py
Run only GPU timings: python -m pytest -v test/ci_tests/test_gemm.py -k gpu_timing
Run both correctness suites and timing: python test/ci_tests/test_gemm.py --benchmark
Run only W4A16: python test/ci_tests/test_gemm.py --op w4a16 --benchmark
Select the GPU with HIP_VISIBLE_DEVICES before starting Python.
CSV defaults to test-results/gemm_gpu_timing_<timestamp>.csv in the repository.
Set GEMM_CSV_PATH to override the CSV destination (pytest and standalone).
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import statistics

import pytest
import torch
from lightop import gemm_ops
from lightop.quant import w4tow16

def _timing_row(op, shape, input_dtype, output_dtype, lightop_us, torch_us):
    return {"gemm_gpu_op": op, "gemm_gpu_shape": shape,
            "gemm_gpu_input": str(input_dtype).removeprefix("torch."),
            "gemm_gpu_output": str(output_dtype).removeprefix("torch."),
            "lightop_mean_us": lightop_us, "torch_mean_us": torch_us,
            "speedup": torch_us / lightop_us}


def _write_gpu_csv(rows):
    default = Path(__file__).resolve().parents[2] / "test-results" / (
        f"gemm_gpu_timing_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    path = Path(os.environ.get("GEMM_CSV_PATH") or default).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["op", "M", "N", "K", "input_dtype", "output_dtype",
              "lightop_mean_us", "torch_mean_us", "speedup"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            m, n, k = row["gemm_gpu_shape"]
            writer.writerow({"op": row["gemm_gpu_op"], "M": m, "N": n, "K": k,
                             "input_dtype": row["gemm_gpu_input"],
                             "output_dtype": row["gemm_gpu_output"],
                             "lightop_mean_us": row["lightop_mean_us"],
                             "torch_mean_us": row["torch_mean_us"],
                             "speedup": row["speedup"]})
    return path


class _GpuTimingCsv:
    """Save GPU timings after tests; only print the CSV destination."""

    def pytest_terminal_summary(self, terminalreporter):
        rows = []
        for report in terminalreporter.stats.get("passed", []):
            props = dict(report.user_properties)
            if report.when == "call" and "gemm_gpu_op" in props:
                rows.append(props)
        if not rows:
            return
        terminalreporter.write_line(f"GPU timing CSV: {_write_gpu_csv(rows)}")


@pytest.fixture(scope="module", autouse=True)
def _register_gpu_timing_summary(request):
    manager = request.config.pluginmanager
    name = "lightop-gemm-gpu-timing-summary"
    if not manager.hasplugin(name):
        manager.register(_GpuTimingCsv(), name)


W4A16_SHAPES = [(m, n, k) for n, k in
                ((3328, 6144), (6144, 3072), (24576, 6144), (6144, 12288))
                for m in (1, 2, 4, 8, 16)]


def _w4a16_inputs(shape):
    torch.manual_seed(0)
    m, n, k = shape
    a = torch.rand(m, k, device="cuda", dtype=torch.float16)
    scale = torch.rand(n, device="cuda", dtype=torch.float16) * 0.1
    zero = torch.rand(n, device="cuda", dtype=torch.float16)
    packed = torch.randint(2**32, (n, k // 8), device="cuda").to(torch.uint32)
    weight = w4tow16(packed, scale, zero)
    return a, packed, scale, zero, weight


@pytest.mark.parametrize("shape", W4A16_SHAPES)
def test_w4a16(shape):
    a, packed, scale, zero, weight = _w4a16_inputs(shape)
    expected = torch.mm(a, weight.t())
    actual = gemm_ops.gemm_w4a16(a, packed.t(), scale, zero)
    assert actual.dtype == a.dtype and actual.device == a.device
    assert actual.shape == expected.shape
    # Preserve the original W4A16 tolerance, but fail pytest on a mismatch.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


# Cover draft/target router shapes, M tiling, partial tiles and empty outputs.
GEMM_SHAPES = [(m, n, 5120) for m in (1, 4, 12, 16, 17, 24, 32, 256)
               for n in (128, 384, 512)] + [(3, 19, 32), (17, 33, 96),
                                          (3, 1, 32), (0, 384, 32), (3, 0, 32)]
GEMM_CASES = [(torch.bfloat16, shape) for shape in GEMM_SHAPES] + [
    (torch.float16, shape) for shape in ((1, 128, 5120), (17, 33, 96), (256, 384, 5120))
]


def _inputs(shape, dtype):
    torch.manual_seed(42)
    m, n, k = shape
    return (torch.randn(m, k, device="cuda", dtype=dtype),
            torch.randn(n, k, device="cuda", dtype=dtype))


def _reference(a, weight):
    # Compute from the already-rounded inputs; avoid a BF16 output reference.
    return (a.double() @ weight.double().t()).float()


@pytest.mark.parametrize("dtype,shape", GEMM_CASES)
def test_gemm_out_dtype(dtype, shape):
    a, weight = _inputs(shape, dtype)
    reference = _reference(a, weight)
    fp32 = gemm_ops.gemm(a, weight, out_dtype=torch.float32)
    assert fp32.dtype == torch.float32
    assert fp32.shape == reference.shape and fp32.device == a.device
    torch.testing.assert_close(fp32, reference, atol=2e-4, rtol=2e-5)

    # Exercise the original two-argument API, explicit None and input dtype.
    original = gemm_ops.gemm(a, weight)
    assert original.dtype == dtype
    torch.testing.assert_close(original, gemm_ops.gemm(a, weight, out_dtype=None),
                               atol=0, rtol=0)
    torch.testing.assert_close(original, gemm_ops.gemm(a, weight, out_dtype=dtype),
                               atol=0, rtol=0)
    torch.testing.assert_close(original, fp32.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(fp32, gemm_ops.gemm(a, weight.t(), out_dtype=torch.float32),
                               atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 128, 5120), (17, 384, 5120), (256, 512, 5120)])
def test_gemm_graph_replay(shape):
    a, weight = _inputs(shape, torch.bfloat16)
    gemm_ops.gemm(a, weight, out_dtype=torch.float32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gemm_ops.gemm(a, weight, out_dtype=torch.float32)
    # Replay must consume current contents, not just reproduce capture output.
    for _ in range(2):
        a.normal_()
        weight.normal_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, _reference(a, weight), atol=2e-4, rtol=2e-5)


def test_gemm_invalid_arguments():
    a, weight = _inputs((3, 19, 32), torch.bfloat16)
    with pytest.raises(RuntimeError, match="out_dtype"):
        gemm_ops.gemm(a, weight, out_dtype=torch.float64)
    with pytest.raises(RuntimeError, match="divisible by 32"):
        gemm_ops.gemm(a[:, :31].contiguous(), weight[:, :31].contiguous())
    with pytest.raises(RuntimeError, match="matching BF16 or FP16"):
        gemm_ops.gemm(a, weight.float())
    with pytest.raises(RuntimeError, match="contiguous"):
        gemm_ops.gemm(a.t(), weight)


def _gpu_mean_us(fn, repeat=30, rounds=5):
    """GPU event time per GEMM in graph replay, in microseconds.

    Prewarming and graph capture are outside the timed region. Replaying a
    batch avoids measuring Python dispatch gaps between individual GEMMs.
    Values describe warm graph replay; they are not end-to-end API latency.
    """
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    samples = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeat)
    return statistics.mean(samples)


GPU_TIMING_SHAPES = [(m, n, 5120) for m in (1, 12, 24, 256)
                     for n in (128, 384, 512)]


def _report_gpu_timing(op, shape, input_dtype, output_dtype, fn, reference_fn,
                       record_property):
    lightop_us = _gpu_mean_us(fn)
    torch_us = _gpu_mean_us(reference_fn)
    speedup = torch_us / lightop_us
    record_property("gemm_gpu_op", op)
    record_property("gemm_gpu_shape", shape)
    record_property("gemm_gpu_input", str(input_dtype).removeprefix("torch."))
    record_property("gemm_gpu_output", str(output_dtype).removeprefix("torch."))
    record_property("lightop_mean_us", lightop_us)
    record_property("torch_mean_us", torch_us)
    record_property("speedup", speedup)


@pytest.mark.parametrize("shape", GPU_TIMING_SHAPES)
@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["default_bf16", "fp32"])
def test_gemm_gpu_timing(shape, out_dtype, record_property):
    a, weight = _inputs(shape, torch.bfloat16)
    if out_dtype is None:
        reference_fn = lambda: torch.mm(a, weight.t())
    else:
        reference_fn = lambda: torch.mm(a, weight.t(), out_dtype=out_dtype)
    _report_gpu_timing("gemm", shape, a.dtype, out_dtype or a.dtype,
                       lambda: gemm_ops.gemm(a, weight, out_dtype=out_dtype),
                       reference_fn, record_property)


@pytest.mark.parametrize("shape", W4A16_SHAPES)
def test_w4a16_gpu_timing(shape, record_property):
    a, packed, scale, zero, weight = _w4a16_inputs(shape)
    # The torch baseline uses an already dequantized weight; dequantization
    # and data preparation are excluded from both timing regions.
    _report_gpu_timing("w4a16", shape, a.dtype, a.dtype,
                       lambda: gemm_ops.gemm_w4a16(a, packed.t(), scale, zero),
                       lambda: torch.mm(a, weight.t()), record_property)



def benchmark_gemm(repeat):
    rows = []
    for m in (1, 12, 24, 256):
        for n in (128, 384, 512):
            a, weight = _inputs((m, n, 5120), torch.bfloat16)
            lightop_us = _gpu_mean_us(
                lambda: gemm_ops.gemm(a, weight, out_dtype=torch.float32), repeat)
            torch_us = _gpu_mean_us(
                lambda: torch.mm(a, weight.t(), out_dtype=torch.float32), repeat)
            rows.append(_timing_row("gemm", (m, n, 5120), a.dtype,
                                    torch.float32, lightop_us, torch_us))
    return rows


def benchmark_w4a16(repeat):
    rows = []
    for shape in W4A16_SHAPES:
        a, packed, scale, zero, weight = _w4a16_inputs(shape)
        lightop_us = _gpu_mean_us(
            lambda: gemm_ops.gemm_w4a16(a, packed.t(), scale, zero), repeat)
        torch_us = _gpu_mean_us(lambda: torch.mm(a, weight.t()), repeat)
        rows.append(_timing_row("w4a16", shape, a.dtype, a.dtype, lightop_us, torch_us))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightOp GEMM and W4A16 correctness and performance")
    parser.add_argument("--op", choices=("all", "gemm", "w4a16"), default="all",
                        help="Run both operators by default; optionally select one")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True,
                        help="Measure selected operators by default; --no-benchmark disables timing")
    parser.add_argument("--repeat", type=int, default=30, help="GEMMs per benchmark graph")
    args = parser.parse_args()
    timing_rows = []
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.op in ("all", "gemm"):
        for dtype, shape in GEMM_CASES:
            test_gemm_out_dtype(dtype, shape)
        for shape in ((1, 128, 5120), (17, 384, 5120), (256, 512, 5120)):
            test_gemm_graph_replay(shape)
        test_gemm_invalid_arguments()
        print("GEMM correctness: PASS (36 cases)", flush=True)
        if args.benchmark:
            timing_rows.extend(benchmark_gemm(args.repeat))
    if args.op in ("all", "w4a16"):
        for shape in W4A16_SHAPES:
            test_w4a16(shape)
        print("W4A16 correctness: PASS (20 cases)", flush=True)
        if args.benchmark:
            timing_rows.extend(benchmark_w4a16(args.repeat))

    if timing_rows:
        print(f"GPU timing CSV: {_write_gpu_csv(timing_rows)}", flush=True)
