"""Correctness tests for lightop.scatter."""

from __future__ import annotations

import torch

from lightop import scatter


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)


def _check(got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, tag: str) -> None:
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol, msg=tag)


def _reference(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    ref = dst.clone()
    for row in range(src.size(0)):
        ref[index[row, 0], :] = src[row, :]
    return ref


def test_scatter() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(12)

    # General element-wise path with unique destination rows.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(32, 1024, device="cuda", dtype=dtype)
        dst = torch.randn(64, 1024, device="cuda", dtype=dtype)
        row_ids = torch.randperm(64, device="cuda")[:32]
        index = row_ids.view(-1, 1).expand(32, 1024).contiguous()
        got = scatter(dst.clone(), index, src, special=False)
        ref = _reference(dst, index, src)
        _check(got, ref, dtype, f"general dtype={dtype}")

    # Explicit row-copy path.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(24, 2048, device="cuda", dtype=dtype)
        dst = torch.randn(48, 2048, device="cuda", dtype=dtype)
        row_ids = torch.randperm(48, device="cuda")[:24]
        index = row_ids.view(-1, 1).expand(24, 2048).contiguous()
        got = scatter(dst.clone(), index, src, special=True)
        ref = _reference(dst, index, src)
        _check(got, ref, dtype, f"special dtype={dtype}")

    # Large-dimension multi-block path.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        src = torch.randn(4, 32768, device="cuda", dtype=dtype)
        dst = torch.randn(8, 32768, device="cuda", dtype=dtype)
        row_ids = torch.tensor([7, 1, 5, 3], device="cuda", dtype=torch.int64)
        index = row_ids.view(-1, 1).expand(4, 32768).contiguous()
        got = scatter(dst.clone(), index, src, special=True)
        ref = _reference(dst, index, src)
        _check(got, ref, dtype, f"large-dim dtype={dtype}")

    # Large-batch special dispatch.
    src = torch.randn(2049, 128, device="cuda", dtype=torch.float16)
    dst = torch.randn(2112, 128, device="cuda", dtype=torch.float16)
    row_ids = torch.arange(2049, device="cuda", dtype=torch.int64)
    index = row_ids.view(-1, 1).expand(2049, 128).contiguous()
    got = scatter(dst.clone(), index, src, special=True)
    ref = _reference(dst, index, src)
    _check(got, ref, torch.float16, "large-batch-special")


if __name__ == "__main__":
    test_scatter()
    print("All scatter checks passed.")
