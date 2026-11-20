#!/usr/bin/env python3
"""Accuracy: lightop.index_select vs torch.index_select."""
from __future__ import annotations

import torch
from lightop import index_select


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def test_index_select() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(5)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(256, 1024, device="cuda", dtype=dtype)
        index = torch.randint(0, 256, (64,), device="cuda", dtype=torch.int64)
        got = index_select(src, 0, index)
        ref = torch.index_select(src, 0, index)
        atol, rtol = _tol(dtype)
        max_abs = float((got.float() - ref.float()).abs().max())
        ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
        print(
            f"[{'PASS' if ok else 'FAIL'}] index_select dtype={dtype} "
            f"max_abs={max_abs:.6g} (atol={atol}, rtol={rtol})"
        )
        assert ok, f"index_select dtype={dtype} max_abs={max_abs}"


if __name__ == "__main__":
    test_index_select()
    print("All index_select checks passed.")
