# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pytest correctness cases and standalone benchmarks for channel INT8 GEMM."""

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
    args = parser.parse_args()
    shapes = [(16, 1280, 4096), (16, 1280, 32768), (16, 2304, 5120),
              (16, 5120, 1792), (16, 5120, 4608), (16, 8192, 5120), (128, 5120, 512)]
    shapes += [(m, k, n) for m in (4096, 4046)
               for k, n in [(1280, 32768), (8192, 5120), (5120, 4608),
                            (6144, 25600), (2304, 5120), (5120, 1792), (1280, 4096)]]
    result = {"correctness": correctness(),
              "performance": [benchmark(m, k, n) for m, k, n in shapes]}
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
