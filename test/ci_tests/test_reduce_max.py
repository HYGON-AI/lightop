#!/usr/bin/env python3
"""Accuracy: lightop.reduce_max vs torch.max(dim=-1)."""
from __future__ import annotations

import torch
from lightop import reduce_max


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


def test_reduce_max() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(3)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for rows, cols in ((64, 8192), (128, 4096), (1, 1024)):
            x = torch.randn(rows, cols, device="cuda", dtype=dtype)
            values, indices = reduce_max(x)
            ref_v, ref_i = torch.max(x, dim=-1)
            atol, rtol = _tol(dtype)
            max_abs = float((values.float() - ref_v.float()).abs().max())
            v_ok = torch.allclose(values.float(), ref_v.float(), atol=atol, rtol=rtol)
            i_ok = torch.equal(indices, ref_i)
            ok = v_ok and i_ok
            print(
                f"[{'PASS' if ok else 'FAIL'}] reduce_max dtype={dtype} "
                f"shape=({rows},{cols}) max_abs={max_abs:.6g} idx_match={i_ok} "
                f"(atol={atol}, rtol={rtol})"
            )
            assert ok, f"reduce_max dtype={dtype} max_abs={max_abs} idx_match={i_ok}"


if __name__ == "__main__":
    test_reduce_max()
    print("All reduce_max checks passed.")
