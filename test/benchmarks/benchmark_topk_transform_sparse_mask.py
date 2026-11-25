# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark: HIP sparse-mask TopK vs the Triton implementation.

Usage (on a DCU host with LightOp installed)::

    python test/benchmarks/benchmark_topk_transform_sparse_mask.py
    python test/benchmarks/benchmark_topk_transform_sparse_mask.py --density 0.25
    python test/benchmarks/benchmark_topk_transform_sparse_mask.py --quick

The table printed at the end is meant to be pasted into the PR description
together with the environment line.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lightop.attention import (  # noqa: E402
    fast_topk_transform_sparse_mask_fused,
    fast_topk_transform_sparse_mask_fused_v2,
)

TOPK = 2048


def bench(fn, warmup: int = 20, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def make_inputs(rows: int, width: int, density: float):
    device = "cuda"
    torch.manual_seed(0)
    score = torch.randn(rows, width, device=device, dtype=torch.float32) * 4.0
    mask_words = (width + 15) // 16
    keep = torch.rand(rows, mask_words, device=device) < density
    bits = torch.zeros(rows, mask_words * 16, dtype=torch.int32, device=device)
    bits[:, : mask_words * 16] = keep.repeat_interleave(16, dim=1)
    nonzero_mask = torch.zeros(rows, mask_words, dtype=torch.int16, device=device)
    for bit in range(16):
        nonzero_mask |= (bits[:, bit::16][:, :mask_words] << bit).to(torch.int16)
    lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    page_table = (
        torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1).contiguous()
    )
    cu_seqlens_q = torch.arange(0, rows + 1, device=device, dtype=torch.int32)
    return score, nonzero_mask, lengths, page_table, cu_seqlens_q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=1.0,
                        help="fraction of positions whose mask bit is set")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"# device={name} arch={arch} topk={TOPK} density={args.density}")

    shapes = [(4, 65536), (4, 131072)] if args.quick else [
        (1, 65536), (1, 131072), (1, 524288),
        (4, 65536), (4, 131072), (4, 262144), (4, 524288),
    ]

    print("\n| rows | width | HIP (ms) | Triton (ms) | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for rows, width in shapes:
        score, mask, lengths, page_table, cu = make_inputs(rows, width, args.density)

        # Sanity: never report a timing for a divergent implementation.
        expected = fast_topk_transform_sparse_mask_fused(
            score, mask, lengths, page_table, cu, TOPK)
        got = fast_topk_transform_sparse_mask_fused_v2(
            score, mask, lengths, page_table, cu, TOPK)
        if not torch.equal(expected.sort(dim=-1).values, got.sort(dim=-1).values):
            raise SystemExit(
                f"selected set mismatch at rows={rows} width={width} "
                f"density={args.density}; refusing to time"
            )

        hip = bench(lambda: fast_topk_transform_sparse_mask_fused(
            score, mask, lengths, page_table, cu, TOPK))
        tri = bench(lambda: fast_topk_transform_sparse_mask_fused_v2(
            score, mask, lengths, page_table, cu, TOPK))
        print(f"| {rows} | {width} | {hip:.3f} | {tri:.3f} | {hip / tri:.2f}x |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
