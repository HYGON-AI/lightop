#!/usr/bin/env python3
"""Accuracy: lightop.gelu(approximate='none' / erf) vs torch."""
from __future__ import annotations

import torch
from lightop import gelu


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    # fp16/bf16: same ::erf path as torch; allow 1 ULP-scale noise after cast.
    return 1e-3, 1e-3


def test_gelu_none() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(128, 4096, device="cuda", dtype=dtype).clamp_(-3, 3)
        got = gelu(x, approximate="none")
        ref = torch.nn.functional.gelu(x, approximate="none")
        atol, rtol = _tol(dtype)
        max_abs = float((got.float() - ref.float()).abs().max())
        ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
        print(
            f"[{'PASS' if ok else 'FAIL'}] gelu(none) dtype={dtype} "
            f"max_abs={max_abs:.6g} (atol={atol}, rtol={rtol})"
        )
        assert ok, f"gelu(none) dtype={dtype} max_abs={max_abs}"


if __name__ == "__main__":
    test_gelu_none()
    print("All gelu(none) checks passed.")
