# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Plain vs deduplicating DPSKV4 dual gather/upconvert on verify-like shapes.

Example (VERIFY step of the fake_128k_c56 trace: bs=7, 4 verify tokens):
    python test/benchmarks/bench_dsv4_gather_dedup.py --requests 7 --tokens 4 \
        --swa-topk 128 --extra-topk 1024 --extra-overlap 1.0   # C128-like
    python test/benchmarks/bench_dsv4_gather_dedup.py --requests 7 --tokens 4 \
        --swa-topk 128 --extra-topk 512 --extra-overlap 0.8    # C4-like
"""

import argparse

import torch

from lightop import op

TOKEN_BYTES = 576 + 8


def _cache(num_pages: int, page_size: int) -> torch.Tensor:
    stride = (page_size * TOKEN_BYTES + 575) // 576 * 576
    raw = torch.randint(0, 256, (num_pages, stride), dtype=torch.uint8, device="cuda")
    # Keep scale bytes in a sane range so the conversion takes the normal path.
    scales = raw[:, page_size * 576 : page_size * 576 + page_size * 8]
    scales.copy_(torch.randint(120, 135, scales.shape, dtype=torch.uint8, device="cuda"))
    return torch.as_strided(
        raw, (num_pages, page_size, 1, TOKEN_BYTES), (stride, TOKEN_BYTES, TOKEN_BYTES, 1)
    )


def _swa_indices(args, num_locations: int) -> torch.Tensor:
    perm = torch.randperm(num_locations, device="cuda", dtype=torch.int64)
    rows = []
    for request in range(args.requests):
        base = request * (args.swa_topk + args.tokens) + 1
        for token in range(args.tokens):
            positions = torch.arange(base + token, base + token + args.swa_topk, device="cuda")
            rows.append(perm[positions % num_locations])
    return torch.stack(rows).to(torch.int32).contiguous()


def _extra_indices(args, num_locations: int) -> torch.Tensor:
    rows = []
    for _ in range(args.requests):
        shared = torch.randperm(num_locations, device="cuda")[: args.extra_topk]
        for _ in range(args.tokens):
            row = shared.clone()
            replace = torch.rand(args.extra_topk, device="cuda") >= args.extra_overlap
            row[replace] = torch.randint(
                0, num_locations, (int(replace.sum()),), device="cuda"
            )
            rows.append(row)
    return torch.stack(rows).to(torch.int32).contiguous()


def _time(fn, iters: int) -> float:
    """Eager wall time per call; includes host launch overhead."""
    for _ in range(10):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _time_graph(fn, iters: int, calls_per_graph: int = 20) -> float:
    """GPU time per call with launches replayed from a CUDA graph (production)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    replays = max(1, iters // calls_per_graph)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * calls_per_graph)


def _kernel_times(fn, iters: int) -> dict:
    """Average device time per call of every kernel fn launches."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    times = {}
    for event in prof.key_averages():
        total = getattr(event, "device_time_total", None)
        if total is None:
            total = event.cuda_time_total
        if total > 0 and event.count >= iters:
            times[event.key] = total / iters
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=7)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--swa-topk", type=int, default=128)
    parser.add_argument("--extra-topk", type=int, default=1024)
    parser.add_argument("--extra-overlap", type=float, default=0.9)
    parser.add_argument("--swa-pages", type=int, default=527)
    parser.add_argument("--swa-page-size", type=int, default=256)
    parser.add_argument("--extra-pages", type=int, default=4096)
    parser.add_argument("--extra-page-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)
    queries = args.requests * args.tokens
    swa_cache = _cache(args.swa_pages, args.swa_page_size)
    extra_cache = _cache(args.extra_pages, args.extra_page_size)
    swa_locations = args.swa_pages * args.swa_page_size
    extra_locations = args.extra_pages * args.extra_page_size
    swa_indices = _swa_indices(args, swa_locations)
    extra_indices = _extra_indices(args, extra_locations)
    swa_lengths = torch.full((queries,), args.swa_topk, dtype=torch.int32, device="cuda")
    extra_lengths = torch.full((queries,), args.extra_topk, dtype=torch.int32, device="cuda")

    def outputs(topk):
        return (
            torch.empty((queries, topk, 1, 512), dtype=torch.bfloat16, device="cuda"),
            torch.empty((queries, 1, topk), dtype=torch.int32, device="cuda"),
        )

    swa_out, swa_compact = outputs(args.swa_topk)
    extra_out, extra_compact = outputs(args.extra_topk)
    swa_map = torch.zeros(swa_locations, dtype=torch.int64, device="cuda")
    extra_map = torch.zeros(extra_locations, dtype=torch.int64, device="cuda")
    swa_epoch = torch.zeros(2, dtype=torch.int64, device="cuda")
    extra_epoch = torch.zeros(2, dtype=torch.int64, device="cuda")

    def plain():
        op.dsv4_gather_upconvert_dual_k_cache_paged(
            swa_cache, swa_indices, swa_lengths, swa_out, swa_compact, args.swa_page_size,
            extra_cache, extra_indices, extra_lengths, extra_out, extra_compact,
            args.extra_page_size,
        )

    def dedup():
        op.dsv4_gather_upconvert_dual_dedup_k_cache_paged(
            swa_cache, swa_indices, swa_lengths, swa_out, swa_compact,
            args.swa_page_size, swa_map, swa_epoch,
            extra_cache, extra_indices, extra_lengths, extra_out, extra_compact,
            args.extra_page_size, extra_map, extra_epoch,
        )

    # Correctness spot check: every compact target must hold the plain result.
    plain()
    ref_swa, ref_extra = swa_out.clone(), extra_out.clone()
    ref_swa_compact, ref_extra_compact = swa_compact.clone(), extra_compact.clone()
    dedup()
    torch.cuda.synchronize()
    for ref, ref_compact, out, compact in (
        (ref_swa, ref_swa_compact, swa_out, swa_compact),
        (ref_extra, ref_extra_compact, extra_out, extra_compact),
    ):
        resolved = out.view(-1, 512)[compact.view(-1).long()]
        expected = ref.view(-1, 512)[ref_compact.view(-1).long()]
        assert torch.equal(resolved.view(torch.int16), expected.view(torch.int16))

    rows = queries * (args.swa_topk + args.extra_topk)
    unique = int(swa_indices.unique().numel() + extra_indices.unique().numel())
    print(
        f"Q={queries} swa_topk={args.swa_topk} extra_topk={args.extra_topk} "
        f"rows={rows} unique={unique} ({unique / rows:.2%})"
    )
    results = {}
    for name, fn in (("plain", plain), ("dedup", dedup)):
        eager_us = _time(fn, args.iters)
        graph_us = _time_graph(fn, args.iters)
        kernels = _kernel_times(fn, args.iters)
        results[name] = graph_us
        print(
            f"{name:5s}: graph {graph_us:7.2f} us | eager {eager_us:7.2f} us | "
            f"kernel sum {sum(kernels.values()):7.2f} us"
        )
        for key, value in sorted(kernels.items(), key=lambda kv: -kv[1]):
            print(f"         {value:7.2f} us  {key[:100]}")
    print(f"speedup (graph): x{results['plain'] / results['dedup']:.2f}")


if __name__ == "__main__":
    main()
