#!/usr/bin/env python3
"""Accuracy: lightop.silu vs torch."""
from __future__ import annotations

import torch
from lightop import silu


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def test_silu() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(2)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(128, 4096, device="cuda", dtype=dtype).clamp_(-3, 3)
        got = silu(x)
        ref = torch.nn.functional.silu(x)
        atol, rtol = _tol(dtype)
        max_abs = float((got.float() - ref.float()).abs().max())
        ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
        print(
            f"[{'PASS' if ok else 'FAIL'}] silu dtype={dtype} "
            f"max_abs={max_abs:.6g} (atol={atol}, rtol={rtol})"
        )
        assert ok, f"silu dtype={dtype} max_abs={max_abs}"


if __name__ == "__main__":
    test_silu()
    print("All silu checks passed.")
