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

    # Row-uniform but non-contiguous index (stride(1) == 0).
    #
    # The host-side dispatch now derives the special path from the layout alone, and the
    # special kernels are the only ones that read a single id per row (index[row*stride0]).
    # Every case above builds its index with .contiguous(), i.e. stride(1) == 1, so those
    # cases take the general element-wise path and would keep passing even if the special
    # kernels were broken. Reusing the expanded index without .contiguous() is what
    # actually exercises scatter_special_small_dim / scatter_special_large_dim: the general
    # kernel indexes a densely packed array and would read the wrong ids here.
    #   cols=1024, fp32  -> row_bytes == 4096  -> scatter_special_small_dim
    #   cols=32768, fp32 -> row_bytes == 131072 -> scatter_special_large_dim
    for cols, src_rows, dst_rows, tag in (
        (1024, 24, 48, "small-dim"),
        (32768, 4, 8, "large-dim"),
    ):
        src = torch.randn(src_rows, cols, device="cuda", dtype=torch.float32)
        dst = torch.randn(dst_rows, cols, device="cuda", dtype=torch.float32)
        row_ids = torch.randperm(dst_rows, device="cuda")[:src_rows]
        index = row_ids.view(-1, 1).expand(src_rows, cols)  # stride(1) == 0, do not materialize
        assert index.stride(1) == 0
        got = scatter(dst.clone(), index, src)  # no special flag: the layout selects it
        ref = _reference(dst, index, src)
        _check(got, ref, torch.float32, f"row-uniform-noncontig {tag}")

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
