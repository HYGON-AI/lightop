"""Correctness tests for lightop.embedding."""

from __future__ import annotations

import torch

from lightop import embedding


def _check_case(rows: int, dim: int, count: int, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    weight = torch.randn(rows, dim, device="cuda", dtype=dtype)
    index = torch.randint(0, rows, (count,), device="cuda", dtype=torch.int64)
    if count >= 2:
        index[0] = 0
        index[1] = rows - 1
    dst = torch.empty(count, dim, device="cuda", dtype=dtype)

    got = embedding(dst, index, weight)
    ref = weight.index_select(0, index)

    assert got.data_ptr() == dst.data_ptr()
    atol = rtol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)


def test_embedding() -> None:
    assert torch.cuda.is_available()

    # One-block path, including a non-vector-aligned tail.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        _check_case(rows=257, dim=777, count=33, dtype=dtype, seed=10)

    # Large-dimension path and the gather-style path used for very wide rows.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        _check_case(rows=8, dim=4096, count=5, dtype=dtype, seed=20)
        _check_case(rows=4, dim=65536, count=3, dtype=dtype, seed=30)

    # Large-batch path for dimensions that do not fill one block per row.
    _check_case(rows=64, dim=256, count=8193, dtype=torch.bfloat16, seed=40)


if __name__ == "__main__":
    test_embedding()
    print("All embedding checks passed.")
