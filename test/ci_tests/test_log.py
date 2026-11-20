#!/usr/bin/env python3
"""Accuracy: lightop.log vs torch.log."""
from __future__ import annotations

import torch
from lightop import log, log_out


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def _check(got: torch.Tensor, ref: torch.Tensor, tag: str, dtype: torch.dtype) -> None:
    atol, rtol = _tol(dtype)
    got_f = got.float()
    ref_f = ref.float()
    nan_ok = bool(torch.equal(got_f.isnan(), ref_f.isnan()))
    inf_ok = bool(torch.equal(got_f.isinf(), ref_f.isinf()))
    finite = (~got_f.isnan()) & (~ref_f.isnan()) & (~got_f.isinf()) & (~ref_f.isinf())
    max_abs = float((got_f[finite] - ref_f[finite]).abs().max()) if finite.any() else 0.0
    close = True
    if finite.any():
        close = bool(torch.allclose(got_f[finite], ref_f[finite], atol=atol, rtol=rtol))
    ok = nan_ok and inf_ok and close
    print(
        f"[{'PASS' if ok else 'FAIL'}] log {tag} dtype={dtype} "
        f"max_abs={max_abs:.6g} nan_ok={nan_ok} inf_ok={inf_ok} "
        f"(atol={atol}, rtol={rtol})"
    )
    assert ok, f"log {tag} dtype={dtype} max_abs={max_abs} nan_ok={nan_ok} inf_ok={inf_ok}"


def test_log() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(2)
    shapes = [(128, 4096), (1, 1024), (7, 1024), (131, 33)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in shapes:
            x = torch.rand(shape, device="cuda", dtype=dtype) + 0.1
            _check(log(x), torch.log(x), f"shape={shape} positive", dtype)
            out = torch.empty_like(x)
            _check(log_out(x, out), torch.log(x), f"shape={shape} out=", dtype)
        z = torch.zeros((8, 16), device="cuda", dtype=dtype)
        _check(log(z), torch.log(z), "zeros", dtype)
        n = -torch.rand((8, 16), device="cuda", dtype=dtype) - 0.1
        _check(log(n), torch.log(n), "negatives", dtype)
    print("All log checks passed.")


if __name__ == "__main__":
    test_log()
