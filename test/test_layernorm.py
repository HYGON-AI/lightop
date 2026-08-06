# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for LayerNorm.

The test includes irregular-shape coverage that is not represented by standard
benchmark shapes.
"""

import time

import torch
import torch.nn as nn

from lightop import op
from lightop.norm import LayerNorm

DEVICE = "cuda"
EPS = 1e-5

class TorchLayerNormModule(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)


class FuseLayerNormModule(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        self.ln = LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)


def _layernorm_allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _layernorm_get_data(batchsize, seqs, cols, use_half=False):
    dtype = torch.float16 if use_half else torch.float32
    x = torch.randn(
        (batchsize, seqs, cols), device="cuda", dtype=dtype, requires_grad=True
    )
    grad_y = torch.randn_like(x)
    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)
    return x1, x2, grad_y


def _layernorm_get_model(cols, use_half=False, mode="full"):
    elementwise_affine = mode != "no_affine"
    torch_model = TorchLayerNormModule(cols, elementwise_affine=elementwise_affine).cuda()
    fuse_model = FuseLayerNormModule(cols, elementwise_affine=elementwise_affine).cuda()

    if use_half:
        torch_model.half()
        fuse_model.half()

    if mode == "full":
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.ln.weight)
            fuse_model.ln.bias.copy_(torch_model.ln.bias)
    elif mode == "no_bias":
        torch_model.ln.bias = None
        fuse_model.ln.bias = None
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.ln.weight)

    torch_model.train()
    fuse_model.train()
    return torch_model, fuse_model


def _run_layernorm_benchmark(mode, batchsize, seqs, cols, repeat_num, use_half=False):
    mode_title = {
        "full": "包含 Weight 和 Bias (Fwd+Bwd)",
        "no_bias": "仅包含 Weight (Fwd+Bwd)",
        "no_affine": "无 Weight 和 Bias (Fwd+Bwd)",
    }[mode]
    precision = "fp16" if use_half else "fp32"
    print(
        f"--- Running Test: {mode_title} | {precision}, "
        f"batch={batchsize}, seqs={seqs}, cols={cols} ---"
    )

    x1, x2, grad_y = _layernorm_get_data(batchsize, seqs, cols, use_half=use_half)
    f1, f2 = _layernorm_get_model(cols, use_half=use_half, mode=mode)

    y1_check = f1(x1)
    y1_check.backward(grad_y)
    y2_check = f2(x2)
    y2_check.backward(grad_y)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f1.zero_grad()
        x1.grad = None
        y1 = f1(x1)
        y1.backward(grad_y)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print(f"  Torch LayerNorm average time: {torch_time:.4f} ms")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f2.zero_grad()
        x2.grad = None
        y2 = f2(x2)
        y2.backward(grad_y)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print(f"  Fuse LayerNorm average time:  {fuse_time:.4f} ms")

    print("\n  Correctness Check:")
    rtol, atol = (2e-2, 2e-2) if use_half else (1e-5, 1e-5)
    x1_final, x2_final, grad_y_final = _layernorm_get_data(
        batchsize, seqs, cols, use_half
    )
    f1_check, f2_check = _layernorm_get_model(cols, use_half, mode)
    y1_final = f1_check(x1_final)
    y2_final = f2_check(x2_final)
    y1_final.backward(grad_y_final)
    y2_final.backward(grad_y_final)

    print(
        f"    Forward output close: "
        f"{_layernorm_allclose(y1_final, y2_final, rtol=rtol, atol=atol)}"
    )
    print(
        f"    Input grad close:     "
        f"{_layernorm_allclose(x1_final.grad, x2_final.grad, rtol=rtol, atol=atol)}"
    )

    if mode in ["full", "no_bias"]:
        w_rtol, w_atol = (2e-1, 2e-1) if use_half else (1e-4, 1e-4)
        weight_grad_close = _layernorm_allclose(
            f1_check.ln.weight.grad,
            f2_check.ln.weight.grad,
            rtol=w_rtol,
            atol=w_atol,
        )
        print(f"    Weight grad close:    {weight_grad_close}")
    else:
        weight_check = f1_check.ln.weight is None and f2_check.ln.weight is None
        print(f"    Weight grad is None:  {weight_check}")

    if mode == "full":
        bias_grad_close = _layernorm_allclose(
            f1_check.ln.bias.grad,
            f2_check.ln.bias.grad,
            rtol=w_rtol,
            atol=w_atol,
        )
        print(f"    Bias grad close:      {bias_grad_close}")
    else:
        bias_check = f1_check.ln.bias is None and f2_check.ln.bias is None
        print(f"    Bias grad is None:    {bias_check}")


def run_layernorm_benchmark_suite():
    shape_cases = [
        (1, 4096, 1024),
        (1, 2048, 2048),
        (1, 2048, 4096),
        (1, 2048, 8192),
        (1, 2048, 16384),
        (1, 1024, 32768),
        (1, 512, 65536),
        (1, 256, 131072),
        (1024, 1, 16384),
        (1024, 1, 32768),
        (7, 1, 1000),
        (17, 1, 2047),
        (33, 1, 5000),
        (65, 1, 8191),
        (7, 1, 8208),
        (17, 1, 10000),
        (3, 1, 16385),
        (513, 1, 4095),
    ]
    modes = ["full", "no_affine"]
    separator = "\n" + "=" * 80 + "\n"

    for use_half in [False, True]:
        for batchsize, seqs, cols in shape_cases:
            repeat_num = 200 if cols <= 8192 else 20
            for mode in modes:
                print(separator)
                _run_layernorm_benchmark(
                    mode, batchsize, seqs, cols, repeat_num, use_half
                )

# Irregular-shape LayerNorm correctness coverage.
# Odd dims: mid path + around fused threshold (>=16384)
# LayerNorm asserts cols >= 64 on HCU; keep smaller odds for softmax/rms only.
ODD_DIMS = (
    1,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
    16383,
    16385,
    32767,
)

# Non-power-of-2 / not vec-aligned (4 or 8)
NON_POW2_DIMS = (
    6,
    12,
    48,
    100,
    384,
    768,
    1000,
    1536,
    3000,
    5000,
    6000,
    8208,   # >8192 and %16==0 — LN mid welford stride path
    10000,
    12000,
    20000,
    25000,
    48000,
)

# Batches / rows not used by xpu-perf (which fixes batch=1024)
BATCHES = (1, 3, 7, 17, 33, 65, 128, 256, 513, 1000, 1025)

DTYPES = (
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 2e-2, 2e-2),
    (torch.bfloat16, 2e-2, 2e-2),
)

def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


# Representative (batch, dim) pairs mixing irregular batch and dim
MIXED_CASES = (
    (1, 16385),
    (3, 16384),
    (7, 8191),
    (17, 10000),
    (33, 5000),
    (65, 4095),
    (128, 3000),
    (256, 1536),
    (513, 1023),
    (1000, 768),
    (1025, 511),
    (7, 32767),
    (3, 48000),
    (17, 20000),
    (1025, 16383),
    (7, 8208),
    (33, 63),
    (65, 100),
    (257, 2047),
    (999, 999),
)

def _check_layernorm(batch: int, dim: int, dtype: torch.dtype, rtol: float, atol: float):
    # HCU lightop LayerNorm requires cols >= 64 (same as torch on this backend).
    if dim < 64:
        return True, 0.0
    torch.manual_seed(0)
    x = torch.randn(batch, dim, device=DEVICE, dtype=dtype)
    w = torch.randn(dim, device=DEVICE, dtype=dtype)
    b = torch.randn(dim, device=DEVICE, dtype=dtype)
    y_ref = torch.nn.functional.layer_norm(
        x.float(), (dim,), w.float(), b.float(), EPS
    ).to(dtype)
    y = op.layernorm_forward_autograd(x, w, b, EPS, False)
    return torch.allclose(y, y_ref, rtol=rtol, atol=atol), _max_abs(y, y_ref)


def _iter_irregular_cases():
    # Fixed batch=7 over odd / non-pow2 dims (batch itself irregular)
    for dim in ODD_DIMS + NON_POW2_DIMS:
        yield 7, dim, "odd/non_pow2"
    # Fixed dim=1000 (non-pow2) over irregular batches
    for batch in BATCHES:
        yield batch, 1000, "batch_sweep"
    # Fixed dim=16385 (odd fused+) over irregular batches
    for batch in (1, 3, 7, 17, 33, 65, 128, 256, 513):
        yield batch, 16385, "odd_fused_batch"
    # Hand-picked mixed cases
    for batch, dim in MIXED_CASES:
        yield batch, dim, "mixed"


def run_irregular_layernorm_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")

    failures = []
    for batch, dim, tag in _iter_irregular_cases():
        for dtype, rtol, atol in DTYPES:
            ok, max_abs = _check_layernorm(batch, dim, dtype, rtol, atol)
            print(
                f"[{'PASS' if ok else 'FAIL'}] layer_norm {tag} "
                f"dtype={dtype} batch={batch} dim={dim} max_abs={max_abs:.4e}"
            )
            if not ok:
                failures.append(("layer_norm", tag, dtype, batch, dim, max_abs))

    if failures:
        raise AssertionError(f"irregular LayerNorm failures: {failures}")
    print("All irregular LayerNorm checks passed.")


def main():
    run_layernorm_benchmark_suite()
    run_irregular_layernorm_suite()


if __name__ == "__main__":
    main()
