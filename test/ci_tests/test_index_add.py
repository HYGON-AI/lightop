#!/usr/bin/env python3
"""Accuracy: lightop.index_add vs torch.index_add.

Duplicate indices use atomics; GPU fp16/bf16 accumulation order is
non-deterministic (see pytorch#91184). Align with community tests:
atol=rtol=1e-1 for half/bfloat16 on CUDA.
"""
from __future__ import annotations

import torch
from lightop import index_add


def _tol(dtype: torch.dtype, device: str = "cuda"):
    # Low-precision types on GPU have non-deterministic accumulation order,
    # leading to larger rounding differences vs torch.index_add.
    # Community / PyTorch practice: atol=rtol=1e-1 for half/bf16 on CUDA.
    # See: https://github.com/pytorch/pytorch/issues/91184
    if device == "cuda" and dtype in (torch.float16, torch.bfloat16, torch.half):
        return 1e-1, 1e-1
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def test_index_add() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(6)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        dst = torch.randn(128, 512, device="cuda", dtype=dtype)
        src = torch.randn(64, 512, device="cuda", dtype=dtype)
        # unique
        idx_u = torch.randperm(128, device="cuda")[:64]
        got_u = index_add(dst, 0, idx_u, src, assume_unique=True)
        ref_u = dst.index_add(0, idx_u, src)
        max_abs_u = float((got_u.float() - ref_u.float()).abs().max())

        # duplicates (atomic add)
        idx_d = torch.randint(0, 128, (64,), device="cuda", dtype=torch.int64)
        got_d = index_add(dst, 0, idx_d, src, assume_unique=False)
        ref_d = dst.index_add(0, idx_d, src)
        atol, rtol = _tol(dtype, device="cuda")
        max_abs_d = float((got_d.float() - ref_d.float()).abs().max())
        ok_u = torch.allclose(got_u.float(), ref_u.float(), atol=atol, rtol=rtol)
        ok_d = torch.allclose(got_d.float(), ref_d.float(), atol=atol, rtol=rtol)
        ok = ok_u and ok_d
        print(
            f"[{'PASS' if ok else 'FAIL'}] index_add dtype={dtype} "
            f"unique_max_abs={max_abs_u:.6g} dup_max_abs={max_abs_d:.6g} "
            f"(atol={atol}, rtol={rtol})"
        )
        assert ok, (
            f"index_add dtype={dtype} unique={max_abs_u} dup={max_abs_d}"
        )


if __name__ == "__main__":
    test_index_add()
    print("All index_add checks passed.")
