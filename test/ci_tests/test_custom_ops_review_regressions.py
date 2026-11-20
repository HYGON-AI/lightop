# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issues raised in the custom-ops review.

Each test pins one silently-wrong-result bug that the review found:

* scatter: the large-dim ``special`` path rounded both the block count and the
  per-block chunk down, so a row tail (e.g. float ``cols=1025``) was never copied.
* topk (power-of-two fast path, k=8/16): the candidate threshold used to be
  refreshed only once per vector chunk, which could evict a legitimate top-K
  element and return a smaller value instead.
* index_select / index_add: the grid remap ``(bid * 4097) % batch`` is only a
  bijection when ``gcd(4097, batch) == 1``; batches such as 17, 241 or 8330 broke
  it and left output rows unwritten (or write them twice).
* reduce_max / reduce_min and log: kernels must run on the tensor's device and
  reject inputs/outputs that live on different devices.
"""

import pytest
import torch

from lightop.ops_basic import (
    index_add,
    index_select,
    log,
    log_out,
    reduce_max,
    reduce_min,
    scatter,
)
from lightop.reduce import topk


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
MULTI_DEVICE = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least two HCU devices"
)


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-6, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)


def _row_reference(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    ref = dst.clone()
    for row in range(src.size(0)):
        ref[index[row, 0], :] = src[row, :]
    return ref


# ------------------------------------------------------------------ scatter --
@CUDA
@pytest.mark.parametrize(
    ("dtype", "cols"),
    (
        (torch.float32, 1025),   # row_bytes=4100 -> 1 block x 1024 elems
        (torch.float32, 4097),   # 4 blocks x 1024 elems < 4097
        (torch.float16, 4100),   # 2 blocks x 2048 elems
        (torch.bfloat16, 8193),  # 4 blocks x 2048 elems
    ),
)
def test_scatter_special_large_dim_writes_row_tail(dtype: torch.dtype, cols: int) -> None:
    torch.manual_seed(2026)
    src_rows, dst_rows = 4, 8
    src = torch.randn(src_rows, cols, device="cuda", dtype=dtype)
    dst = torch.randn(dst_rows, cols, device="cuda", dtype=dtype)
    row_ids = torch.tensor([7, 1, 5, 3], device="cuda", dtype=torch.int64)

    # Row-uniform index (expand keeps stride(1) == 0) selects the special path.
    index = row_ids.view(-1, 1).expand(src_rows, cols)
    got = scatter(dst.clone(), index, src)
    ref = _row_reference(dst, index, src)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(
        got.float(), ref.float(), atol=atol, rtol=rtol,
        msg=f"scatter special large-dim dtype={dtype} cols={cols}",
    )
    # Pin the tail columns explicitly: the bug left exactly these untouched.
    for row in range(src_rows):
        torch.testing.assert_close(
            got[row_ids[row].item(), cols - 1].float(),
            src[row, cols - 1].float(),
            atol=atol, rtol=rtol,
            msg=f"scatter last column row={row} dtype={dtype} cols={cols}",
        )


# --------------------------------------------------------------------- topk --
@CUDA
@pytest.mark.parametrize("k", (2, 4, 8))
@pytest.mark.parametrize("dim", (2048, 8192))
def test_topk_power2_duplicate_heavy_rows(k: int, dim: int) -> None:
    """Many ties stress the insertion threshold; a stale one loses a true top-k value."""
    torch.manual_seed(4242)
    # Small integer alphabet => lots of duplicates, so the buffer minimum moves
    # repeatedly inside a single vector chunk.
    x = torch.randint(0, 6, (8, dim), device="cuda", dtype=torch.int32).to(torch.float32)
    values, indices = topk(x, k, dim=-1, largest=True, sorted=False)
    ref_values, _ = torch.topk(x, k, dim=-1, largest=True, sorted=True)

    torch.testing.assert_close(
        torch.gather(x, -1, indices), values, atol=0, rtol=0, msg="topk value/index mismatch"
    )
    torch.testing.assert_close(
        torch.sort(values, dim=-1, descending=True).values, ref_values,
        atol=0, rtol=0, msg=f"topk k={k} dim={dim} duplicate-heavy",
    )


@CUDA
@pytest.mark.parametrize("k", (4, 8))
def test_topk_power2_staircase_rows(k: int) -> None:
    """Descending staircase with a dip: probes the eviction of the current minimum."""
    torch.manual_seed(99)
    dim = 2048
    base = torch.arange(dim, 0, -1, device="cuda", dtype=torch.float32).view(1, dim)
    x = base.repeat(4, 1)
    # After the first K candidates, drop a value just below the K-th and then a
    # slightly larger one: the stale threshold used to swap in the smaller value.
    x[:, k] = x[:, k - 1] - 2.0
    x[:, k + 1] = x[:, k - 1] - 1.0

    values, indices = topk(x, k, dim=-1, largest=True, sorted=False)
    ref_values, _ = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    torch.testing.assert_close(
        torch.sort(values, dim=-1, descending=True).values, ref_values,
        atol=0, rtol=0, msg=f"topk staircase k={k}",
    )
    torch.testing.assert_close(
        torch.gather(x, -1, indices), values, atol=0, rtol=0, msg="topk staircase indices"
    )


@CUDA
@pytest.mark.xfail(
    reason="k=16 power-of-two path has a pre-existing 8+8 merge defect "
    "(python never dispatches k=16 here; tracked separately)",
    strict=False,
)
def test_topk_power2_k16_skewed_lane_is_known_broken() -> None:
    """Documents the k=16 defect instead of hiding it: one lane owning >16 of the
    global top values returns garbage in slots 8..15."""
    from lightop import op

    dim = 65536
    x = torch.full((4, dim), -1000.0, device="cuda", dtype=torch.float32)
    ramp = torch.arange(100.0, 80.0, -1, device="cuda")
    for t in range(20):
        x[:, 256 * (t // 4) + (t % 4)] = ramp[t]

    values = torch.empty(4, 16, device="cuda", dtype=x.dtype)
    indices = torch.empty(4, 16, device="cuda", dtype=torch.int64)
    op.topk_power2_opt(values, indices, x, 16)
    ref = torch.topk(x, 16, dim=-1, largest=True, sorted=True).values
    torch.testing.assert_close(
        torch.sort(values, dim=-1, descending=True).values, ref, atol=0, rtol=0
    )


# ------------------------------------------------------------- index_select --
@CUDA
@pytest.mark.parametrize("dst_batch", (17, 34, 241, 4896))
def test_index_select_large_dim_batch_shares_factor_with_remap(dst_batch: int) -> None:
    """4097 = 17 x 241: these batch sizes used to collapse the block remap."""
    torch.manual_seed(7)
    src_batch, dim = 8, 4096
    src = torch.randn(src_batch, dim, device="cuda", dtype=torch.bfloat16)
    # Output rows = len(index) = dst_batch while index values stay inside src_batch.
    index = torch.randint(0, src_batch, (dst_batch,), device="cuda", dtype=torch.int64)

    got = index_select(src, 0, index)
    ref = torch.index_select(src, 0, index)
    torch.testing.assert_close(got, ref, atol=0, rtol=0, msg=f"index_select dst_batch={dst_batch}")


# ---------------------------------------------------------------- index_add --
@CUDA
@pytest.mark.parametrize("src_batch", (8330,))
def test_index_add_duplicate_indices_accumulate(src_batch: int) -> None:
    """src_batch shares a factor with the remap multiplier; duplicates must still sum."""
    torch.manual_seed(11)
    dim, dst_batch = 64, 4096
    dst = torch.zeros(dst_batch, dim, device="cuda", dtype=torch.float32)
    src = torch.randn(src_batch, dim, device="cuda", dtype=torch.float32)
    # Deliberately duplicated destinations, including a hot row.
    index = torch.randint(0, dst_batch, (src_batch,), device="cuda", dtype=torch.int64)
    index[: src_batch // 4] = 7

    got = index_add(dst.clone(), 0, index, src)
    ref = torch.zeros_like(dst).index_add(0, index, src)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4, msg="index_add duplicate accumulate")


# ------------------------------------------------------- reduce_max/min, log --
@CUDA
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_reduce_minmax_matches_torch(dtype: torch.dtype) -> None:
    torch.manual_seed(2027)
    for rows, cols in ((17, 16381), (16381, 7), (1024, 4096)):
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        got_max, got_max_idx = reduce_max(x)
        ref_max, ref_max_idx = torch.max(x, dim=-1)
        torch.testing.assert_close(got_max, ref_max, atol=0, rtol=0, msg="reduce_max value")
        torch.testing.assert_close(got_max_idx, ref_max_idx, atol=0, rtol=0, msg="reduce_max index")

        got_min, got_min_idx = reduce_min(x)
        ref_min, ref_min_idx = torch.min(x, dim=-1)
        torch.testing.assert_close(got_min, ref_min, atol=0, rtol=0, msg="reduce_min value")
        torch.testing.assert_close(got_min_idx, ref_min_idx, atol=0, rtol=0, msg="reduce_min index")


@CUDA
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_log_matches_torch(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    x = torch.rand(1024, device="cuda", dtype=dtype) + 0.5
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(log(x), torch.log(x), atol=atol, rtol=rtol, msg="log")

    out = torch.empty_like(x)
    log_out(x, out)
    torch.testing.assert_close(out, torch.log(x), atol=atol, rtol=rtol, msg="log_out")


@MULTI_DEVICE
def test_log_rejects_cross_device_output() -> None:
    x = torch.rand(64, device="cuda:0", dtype=torch.float32) + 0.5
    out = torch.empty(64, device="cuda:1", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        log_out(x, out)


@MULTI_DEVICE
def test_reduce_max_out_rejects_cross_device_output() -> None:
    x = torch.randn(8, 16, device="cuda:0", dtype=torch.float32)
    values = torch.empty(8, device="cuda:1", dtype=torch.float32)
    indices = torch.empty(8, device="cuda:1", dtype=torch.int64)
    from lightop.ops_basic import reduce_max_out

    with pytest.raises(RuntimeError):
        reduce_max_out(x, values, indices)


if __name__ == "__main__":
    test_scatter_special_large_dim_writes_row_tail(torch.float32, 1025)
    test_topk_power2_duplicate_heavy_rows(8, 2048)
    test_topk_power2_staircase_rows(8)
    test_index_select_large_dim_batch_shares_factor_with_remap(17)
    test_index_add_duplicate_indices_accumulate(8330)
    test_reduce_minmax_matches_torch(torch.float32)
    test_log_matches_torch(torch.float32)
    print("All custom-ops review regression checks passed.")
