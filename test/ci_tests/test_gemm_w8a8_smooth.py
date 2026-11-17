# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pytest correctness cases and standalone benchmarks for channel INT8 GEMM.

The opt-in input-scale-first epilogue preserves two FP32 rounding steps before
BF16 conversion. It is a numerical-compatibility port, not a faster MMAC body.
Benchmark explicitly with --ordered-scale-benchmark; default pytest runs only
correctness. Reproduce its gfx936 CO using:
python scripts/generate_w8a8_input_scale_first.py --check
"""

# python3 -m pytest -v test/ci_tests/test_gemm_w8a8_smooth.py

import argparse
import json
import os
import sys

import pytest
import torch

from lightop import gemm_ops, gemmopt


def check_case(m, k, n, dtype=torch.bfloat16, *, direct=False, split_k=1,
               zero=False, bias=False, full_range=False):
    torch.manual_seed(13)
    low, high = (-128, 128) if full_range else (-3, 4)
    a = torch.randint(low, high, (m, k), device="cuda", dtype=torch.int8)
    if zero:
        a.zero_()
    b = torch.randint(low, high, (n, k), device="cuda", dtype=torch.int8).T
    sa = torch.rand((m, 1), device="cuda", dtype=torch.float32) * 0.05 + 0.005
    sb = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.05 + 0.005
    sa[0] = 0
    sb[0] = 0
    offset = torch.randn((n,), device="cuda", dtype=dtype) if bias else None
    if direct:
        from lightop._lmslim_native.layers.gemm.int8_utils import matmul_int8

        config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
                  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "SPLIT_K": split_k,
                  "num_stages": 0, "num_warps": 4}
        actual = matmul_int8(a, sa, b, sb, dtype, config)
        if offset is not None:
            actual += offset
    else:
        actual = gemm_ops.triton_scaled_mm(a, b, sa, sb, dtype, offset)
    expected = ((a.float() @ b.float()) * sa * sb.T).to(dtype)
    if offset is not None:
        expected += offset
    torch.cuda.synchronize()
    rtol = {torch.bfloat16: 0.008, torch.float16: 0.001, torch.float32: 1e-5}[dtype]
    for start in range(0, m, 256):
        torch.testing.assert_close(actual[start:start+256], expected[start:start+256],
                                   atol=2e-6, rtol=rtol)
    return {"shape": [m, k, n], "dtype": str(dtype), "direct": direct,
            "split_k": split_k, "zero": zero, "bias": bias,
            "full_range": full_range, "correctness": "PASS"}


def _correctness_cases():
    cases = [
        (4096, 1280, 32768, {}),
        (4046, 1280, 32768, {}),
        (4095, 1280, 32768, {}),
        (4096, 1280, 32768, {"bias": True}),
        (17, 129, 257, {"direct": True}),
        (33, 65, 97, {"direct": True, "zero": True}),
        (17, 129, 257, {"direct": True, "split_k": 2}),
        (17, 129, 257, {"direct": True, "full_range": True}),
        (31, 129, 97, {"direct": True, "dtype": torch.float16}),
        (31, 129, 97, {"direct": True, "dtype": torch.float32}),
        (16, 5120, 384, {}),
    ]
    shapes = [(8192, 5120), (5120, 4608), (6144, 25600),
              (2304, 5120), (5120, 1792), (1280, 4096)]
    cases += [(m, k, n, {}) for m in (4096, 4046) for k, n in shapes]
    cases += [(4095, 5120, 4608, {}),
              (4096, 5120, 4608, {"full_range": True}),
              (4046, 5120, 1792, {"zero": True}),
              (4096, 5120, 1792, {"bias": True})]
    cases += [(16, k, n, {}) for k, n in [(1280, 4096), (1280, 32768),
               (2304, 5120), (5120, 1792), (5120, 4608), (8192, 5120)]]
    cases += [(128, 5120, 512, {}), (15, 1280, 4096, {}),
              (17, 1280, 4096, {}), (127, 5120, 512, {}),
              (129, 5120, 512, {}), (16, 5120, 1792, {"bias": True}),
              (16, 2304, 5120, {"full_range": True}),
              (16, 1280, 4096, {"zero": True}), (4047, 1280, 32768, {})]
    covered = {(m, k, n) for m, k, n, kwargs in cases if not kwargs}
    cases += [(m, k, n, {})
              for m, k, n in gemmopt._W8A8_SMOOTH_GFX936_CACHED_ALGOS
              if (m, k, n) not in covered]
    cases += [(32, 2304, 5120, {})]
    return cases


def correctness():
    with torch.cuda.stream(torch.cuda.Stream()):
        return [check_case(m, k, n, **kwargs)
                for m, k, n, kwargs in _correctness_cases()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/ROCm device")
@pytest.mark.parametrize(
    "m,k,n,kwargs",
    [pytest.param(
        m, k, n, kwargs,
        id="_".join([f"M{m}", f"K{k}", f"N{n}"]
                    + [f"{key}={value}" for key, value in sorted(kwargs.items())]),
    ) for m, k, n, kwargs in _correctness_cases()],
)
def test_gemm_w8a8_smooth_correctness(m, k, n, kwargs):
    with torch.cuda.stream(torch.cuda.Stream()):
        check_case(m, k, n, **kwargs)


def _ordered_inputs(m, k, n, *, zero=False, full_range=False):
    torch.manual_seed(131)
    # Bound every integer partial sum below 2**24, making the independent
    # FP32 dot an exact oracle for the INT32 accumulator.
    a = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-3, 4, (n, k), device="cuda", dtype=torch.int8).T
    if zero:
        a.zero_()
    if full_range:
        a.fill_(-128)
        b.fill_(127)
        assert 128 * 127 * k < 2**24
    sa = (torch.rand((m, 1), device="cuda", dtype=torch.float32) - 0.5) * 0.1
    sb = (torch.rand((n, 1), device="cuda", dtype=torch.float32) - 0.5) * 0.1
    sa[0] = 0
    sb[0] = 0
    return a, b, sa, sb


def _require_ordered_gfx():
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "gcnArchName", "").split(":")[0] != "gfx936":
        pytest.skip("input_scale_first specializes gfx936")


def _ordered_reference(a, b, sa, sb):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        accumulator = a.float() @ b.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return ((accumulator * sa) * sb.T).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
@pytest.mark.parametrize("m,k,n,kind", [
    (17, 128, 256, "random"), (255, 256, 512, "random"),
    (256, 128, 256, "zero"), (257, 256, 512, "full_range"),
    (4992, 5376, 21504, "random"), (4992, 5376, 28672, "random"),
    (4992, 14336, 5376, "random"), (4992, 7168, 5376, "random"),
    (8192, 5376, 21504, "random"), (8192, 5376, 28672, "random"),
    (8192, 14336, 5376, "random"), (8192, 7168, 5376, "random"),
] + [(m, k, n, "random") for m in (37760, 41856, 52544)
     for n, k in ((21504, 5376), (28672, 5376), (5376, 14336), (5376, 7168))])
def test_gemm_w8a8_input_scale_first(m, k, n, kind):
    _require_ordered_gfx()
    with torch.cuda.stream(torch.cuda.Stream()):
        a, b, sa, sb = _ordered_inputs(m, k, n, zero=kind == "zero",
                                      full_range=kind == "full_range")
        supported, actual = gemm_ops.gemm_w8a8_smooth(
            a, b, sa, sb, None, torch.bfloat16, input_scale_first=True)
        assert supported
        expected = _ordered_reference(a, b, sa, sb)
        torch.cuda.synchronize()
        assert actual.dtype == torch.bfloat16 and actual.shape == (m, n)
        assert actual.is_contiguous() and torch.isfinite(actual).all()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
def test_gemm_w8a8_input_scale_first_rounding_boundary():
    """Stock scale-product association fails every output at this BF16 tie."""
    _require_ordered_gfx()
    a = torch.ones((257, 128), dtype=torch.int8, device="cuda")
    weight = torch.zeros((256, 128), dtype=torch.int8, device="cuda")
    weight[:, 0], weight[:, 1] = 127, 10  # INT32 dot is exactly 137.
    b = weight.T
    sa = torch.full((257, 1), 0.06446823477745056, device="cuda", dtype=torch.float32)
    sb = torch.full((256, 1), 0.07585038244724274, device="cuda", dtype=torch.float32)
    accumulator = torch.full((257, 256), 137., device="cuda", dtype=torch.float32)
    expected = ((accumulator * sa) * sb.T).bfloat16()
    folded = (accumulator * (sa * sb.T)).bfloat16()
    assert torch.all(expected != folded)
    supported, actual = gemm_ops.gemm_w8a8_smooth(
        a, b, sa, sb, None, torch.bfloat16, input_scale_first=True)
    assert supported and torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_w8a8_input_scale_first_legacy_default(dtype):
    a, b, sa, sb = _ordered_inputs(257, 128, 256)
    old_ok, old = gemm_ops.gemm_w8a8_smooth(a, b, sa, sb, None, dtype)
    new_ok, explicit_default = gemm_ops.gemm_w8a8_smooth(
        a, b, sa, sb, None, dtype, input_scale_first=False)
    assert old_ok == new_ok
    if old_ok:
        assert torch.equal(old, explicit_default)
    # Preserve the old 11-position-argument native call as well.
    native_old = gemmopt.op.gemm_w8a8_smooth_asm(
        a, b, sa, sb, None, dtype, 257, 256, 128, True, 8)
    native_default = gemmopt.op.gemm_w8a8_smooth_asm(
        a, b, sa, sb, None, dtype, 257, 256, 128, True, 8, False)
    assert torch.equal(native_old, native_default)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
@pytest.mark.parametrize("case", ["fp16", "bias", "small_m", "tail_n", "tail_k",
    "a_stride", "b_layout", "scale_dtype", "scale_shape", "scale_stride",
    "scale_device", "a_dimension", "b_dimension", "diagnostic_bypass"])
def test_gemm_w8a8_input_scale_first_unsupported(case):
    m, k, n = (32, 2304, 5120) if case == "diagnostic_bypass" else (257, 128, 256)
    if case == "small_m": m = 16
    if case == "tail_n": n = 257
    if case == "tail_k": k = 129
    a, b, sa, sb = _ordered_inputs(m, k, n)
    dtype, bias = torch.bfloat16, None
    if case == "fp16": dtype = torch.float16
    if case == "bias": bias = torch.ones(n, dtype=dtype, device="cuda")
    if case == "a_stride": a = torch.zeros((m, k*2), dtype=torch.int8, device="cuda")[:, ::2]
    if case == "b_layout": b = b.contiguous()
    if case == "scale_dtype": sa = sa.bfloat16()
    if case == "scale_shape": sa = sa.view(-1)
    if case == "scale_stride": sa = torch.ones((m, 2), device="cuda", dtype=torch.float32)[:, :1]
    if case == "scale_device": sa = sa.cpu()
    if case == "a_dimension": a = a.unsqueeze(0)
    if case == "b_dimension": b = b.unsqueeze(0)
    assert gemm_ops.gemm_w8a8_smooth(
        a, b, sa, sb, bias, dtype, input_scale_first=True) == (False, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
def test_gemm_w8a8_input_scale_first_other_arch(monkeypatch):
    from types import SimpleNamespace
    a, b, sa, sb = _ordered_inputs(257, 128, 256)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(gcnArchName="gfx938"))
    assert gemm_ops.gemm_w8a8_smooth(
        a, b, sa, sb, None, torch.bfloat16, input_scale_first=True) == (False, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
def test_gemm_w8a8_input_scale_first_native_validation():
    a, b, sa, sb = _ordered_inputs(257, 128, 256)
    with pytest.raises(RuntimeError, match="BF16 output"):
        gemmopt.op.gemm_w8a8_smooth_asm(
            a, b, sa, sb, None, torch.float16, 257, 256, 128, False, 8, True)
    with pytest.raises(RuntimeError, match="FP32 scales"):
        gemmopt.op.gemm_w8a8_smooth_asm(
            a, b, sa.bfloat16(), sb, None, torch.bfloat16, 257, 256, 128, False, 8, True)
    with pytest.raises(RuntimeError, match="HCU tensors"):
        gemmopt.op.gemm_w8a8_smooth_asm(
            a.cpu(), b.cpu(), sa.cpu(), sb.cpu(), None, torch.bfloat16,
            257, 256, 128, False, 8, True)


def test_gemm_w8a8_input_scale_first_flag_type():
    with pytest.raises(TypeError, match="must be bool"):
        gemm_ops.gemm_w8a8_smooth(None, None, None, None, None,
                                 torch.bfloat16, input_scale_first="yes")


def test_gemm_w8a8_input_scale_first_packaged_objects():
    import hashlib
    from pathlib import Path
    import lightop
    root = Path(lightop.__file__).parent / "hsa/gfx936"
    stem = "gemm_w8a8_smooth_256x256x128_WGM8_TN_BF16"
    expected = {stem + ".co": "7811b52cb012f7a209189722bffbfdf1a12d1b60f28f760b879ad090c4b2bbf6",
                stem + "_INPUT_SCALE_FIRST.co": "e3d9adfbbc518be344ac38e266fbfd9ed237a53d35653ac990588ec5d556f19b"}
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
def test_gemm_w8a8_input_scale_first_intermediate_overflow():
    _require_ordered_gfx()
    a = torch.ones((257, 128), dtype=torch.int8, device="cuda")
    b = torch.ones((256, 128), dtype=torch.int8, device="cuda").T
    sa = torch.full((257, 1), 1e38, dtype=torch.float32, device="cuda")
    sb = torch.full((256, 1), 1e-20, dtype=torch.float32, device="cuda")
    # Both scales are normal finite FP32 numbers. Sequential scaling must
    # preserve the intermediate overflow instead of reassociating to 128e18.
    expected = _ordered_reference(a, b, sa, sb)
    assert torch.isposinf(expected).all()
    supported, actual = gemm_ops.gemm_w8a8_smooth(
        a, b, sa, sb, None, torch.bfloat16, input_scale_first=True)
    assert supported and torch.equal(actual, expected)


def benchmark_ordered_scale():
    import statistics
    _require_ordered_gfx()
    test_gemm_w8a8_input_scale_first_rounding_boundary()
    rows = []
    for m in (4992, 8192):
        for n, k in ((21504, 5376), (28672, 5376), (5376, 14336), (5376, 7168)):
            a, b, sa, sb = _ordered_inputs(m, k, n)
            def call(enabled):
                ok, result = gemm_ops.gemm_w8a8_smooth(
                    a, b, sa, sb, None, torch.bfloat16, input_scale_first=enabled)
                assert ok
                return result
            expected = _ordered_reference(a, b, sa, sb)
            assert torch.equal(call(True), expected)
            assert torch.isfinite(expected).all()
            del expected
            for _ in range(10):
                call(False)
                call(True)
            torch.cuda.synchronize()
            times = {False: [], True: []}
            for repeat in range(20):
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); result = call(enabled); end.record(); end.synchronize()
                    times[enabled].append(start.elapsed_time(end) * 1000)
            traffic = m*k + n*k + 2*m*n + 4*(m+n)
            row = {"shape_MNK": [m, n, k], "dtype": "bfloat16",
                   "reference": "sequential FP32 scales, strict equality",
                   "correctness": "PASS", "physical_gpu": os.getenv("HIP_VISIBLE_DEVICES"),
                   "mean_us": {str(v): statistics.mean(times[v]) for v in (False, True)},
                   "median_us": {str(v): statistics.median(times[v]) for v in (False, True)},
                   "p90_us": {str(v): sorted(times[v])[17] for v in (False, True)},
                   "logical_traffic_gbs": traffic/(statistics.mean(times[True])*1000),
                   "command": " ".join(sys.argv)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def benchmark(m, k=1280, n=32768):
    torch.manual_seed(13)
    a = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-3, 4, (n, k), device="cuda", dtype=torch.int8).T
    sa = torch.rand((m, 1), device="cuda") * 0.05
    sb = torch.rand((n, 1), device="cuda") * 0.05
    for _ in range(10):
        gemm_ops.triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        output = gemm_ops.triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
    end.record()
    end.synchronize()
    mean_us = start.elapsed_time(end) * 10
    byte_count = m*k + n*k + 2*m*n + 4*(m+n)
    return {"test_file": __file__, "reference": "PyTorch FP32 dot/scales then output dtype",
            "shape": [m, k, n], "dtype": "bfloat16", "correctness": "PASS",
            "mean_us": mean_us, "logical_traffic_gbs": byte_count/(mean_us*1000),
            "useful_tops": 2*m*n*k/(mean_us*1e6),
            "gpu": os.getenv("HIP_VISIBLE_DEVICES"), "command": " ".join(sys.argv)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--ordered-scale-benchmark", action="store_true")
    args = parser.parse_args()
    shapes = [(16, 1280, 4096), (16, 1280, 32768), (16, 2304, 5120),
              (16, 5120, 1792), (16, 5120, 4608), (16, 8192, 5120), (128, 5120, 512)]
    shapes += [(m, k, n) for m in (4096, 4046)
               for k, n in [(1280, 32768), (8192, 5120), (5120, 4608),
                            (6144, 25600), (2304, 5120), (5120, 1792), (1280, 4096)]]
    if args.ordered_scale_benchmark:
        result = {"ordered_scale_performance": benchmark_ordered_scale()}
    else:
        result = {"correctness": correctness(),
                  "performance": [benchmark(m, k, n) for m, k, n in shapes]}
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
