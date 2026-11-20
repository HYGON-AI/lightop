#!/usr/bin/env python3
"""Accuracy: lightop.gelu(approximate='tanh') vs torch."""
from __future__ import annotations

import torch
from lightop import gelu


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def test_gelu_tanh() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(128, 4096, device="cuda", dtype=dtype).clamp_(-3, 3)
        got = gelu(x, approximate="tanh")
        ref = torch.nn.functional.gelu(x, approximate="tanh")
        atol, rtol = _tol(dtype)
        max_abs = float((got.float() - ref.float()).abs().max())
        ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
        print(
            f"[{'PASS' if ok else 'FAIL'}] gelu(tanh) dtype={dtype} "
            f"max_abs={max_abs:.6g} (atol={atol}, rtol={rtol})"
        )
        assert ok, f"gelu(tanh) dtype={dtype} max_abs={max_abs}"


if __name__ == "__main__":
    test_gelu_tanh()
    print("All gelu(tanh) checks passed.")
