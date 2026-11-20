"""Correctness tests for lightop.gather."""

from __future__ import annotations

import torch

from lightop import gather


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)


def _check(got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, tag: str) -> None:
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol, msg=tag)


def test_gather() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(11)

    # General element-wise index path.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(17, 1024, device="cuda", dtype=dtype)
        index = torch.randint(0, 17, (9, 1024), device="cuda", dtype=torch.int64)
        dst = torch.empty_like(index, dtype=dtype)
        got = gather(dst, index, src, special=False)
        ref = torch.gather(src, 0, index)
        _check(got, ref, dtype, f"general dtype={dtype}")

    # Explicit row-broadcast path.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(32, 2048, device="cuda", dtype=dtype)
        row_ids = torch.randint(0, 32, (13, 1), device="cuda", dtype=torch.int64)
        index = row_ids.expand(13, 2048).contiguous()
        dst = torch.empty(13, 2048, device="cuda", dtype=dtype)
        got = gather(dst, index, src, special=True)
        ref = src.index_select(0, row_ids[:, 0])
        _check(got, ref, dtype, f"special dtype={dtype}")

    # Very wide general and special paths.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(4, 65536, device="cuda", dtype=dtype)
        index = torch.randint(0, 4, (2, 65536), device="cuda", dtype=torch.int64)
        dst = torch.empty(2, 65536, device="cuda", dtype=dtype)
        got = gather(dst, index, src, special=False)
        ref = torch.gather(src, 0, index)
        _check(got, ref, dtype, f"wide-general dtype={dtype}")

        row_ids = torch.tensor([[1], [3]], device="cuda", dtype=torch.int64)
        row_index = row_ids.expand(2, 32768).contiguous()
        wide_src = src[:, :32768].contiguous()
        wide_dst = torch.empty(2, 32768, device="cuda", dtype=dtype)
        got = gather(wide_dst, row_index, wide_src, special=True)
        ref = wide_src.index_select(0, row_ids[:, 0])
        _check(got, ref, dtype, f"wide-special dtype={dtype}")

    # Large-batch special dispatch, including the half/bfloat16 medium branch.
    src = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    row_ids = torch.randint(0, 64, (4097, 1), device="cuda", dtype=torch.int64)
    index = row_ids.expand(4097, 1024).contiguous()
    dst = torch.empty(4097, 1024, device="cuda", dtype=torch.bfloat16)
    got = gather(dst, index, src, special=True)
    ref = src.index_select(0, row_ids[:, 0])
    _check(got, ref, torch.bfloat16, "large-batch-special")


if __name__ == "__main__":
    test_gather()
    print("All gather checks passed.")
