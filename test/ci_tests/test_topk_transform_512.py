# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import time

import torch

from lightop.attention import (
    topk_transform_512,
    topk_transform_512_sparse_mask,
)

MAX_CONTEXT_LEN = 512 * 1024
TOPK = 512
PAGE_SIZE = 64


def topk_transform_512_reference(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    page_size: int,
):
    batch_size = scores.size(0)
    positions = torch.arange(scores.size(1), device=scores.device)
    valid_scores = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    masked_scores = scores.masked_fill(~valid_scores, float("-inf"))
    selected = torch.topk(masked_scores, TOPK, dim=1, sorted=False).indices

    sequential = positions[:TOPK].unsqueeze(0).expand(batch_size, -1)
    short_rows = seq_lens.unsqueeze(1) <= TOPK
    raw_indices = torch.where(short_rows, sequential, selected)
    valid_indices = raw_indices < seq_lens.unsqueeze(1)
    safe_raw_indices = raw_indices.masked_fill(~valid_indices, 0)

    page_slots = torch.div(
        safe_raw_indices, page_size, rounding_mode="floor"
    )
    page_offsets = safe_raw_indices.remainder(page_size).to(torch.int32)
    page_indices = (
        torch.gather(page_tables, 1, page_slots) * page_size + page_offsets
    )
    page_indices.masked_fill_(~valid_indices, -1)
    raw_indices = raw_indices.to(torch.int32)
    raw_indices.masked_fill_(~valid_indices, -1)

    return page_indices, raw_indices


def pack_nonzero_mask(nonzero: torch.Tensor) -> torch.Tensor:
    padded_width = ((nonzero.shape[1] + 15) // 16) * 16
    padded = torch.zeros(
        nonzero.shape[0], padded_width, dtype=torch.bool, device=nonzero.device
    )
    padded[:, : nonzero.shape[1]] = nonzero
    bit_weights = 1 << torch.arange(16, dtype=torch.int32, device=nonzero.device)
    return (
        padded.view(nonzero.shape[0], -1, 16).to(torch.int32)
        .mul(bit_weights)
        .sum(dim=-1)
        .to(torch.int16)
    )


def sparse_mask_reference(
    scores: torch.Tensor,
    nonzero: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    page_size: int,
):
    raw_rows = []
    for row, length_tensor in enumerate(seq_lens.cpu()):
        length = int(length_tensor)
        if length <= TOPK:
            selected = torch.arange(length, device=scores.device)
        else:
            live = torch.nonzero(nonzero[row, :length], as_tuple=False).flatten()
            if live.numel() >= TOPK:
                ranks = torch.topk(scores[row, live], TOPK, sorted=False).indices
                selected = live[ranks]
            else:
                zero = torch.nonzero(~nonzero[row, :length], as_tuple=False).flatten()
                selected = torch.cat((live, zero[: TOPK - live.numel()]))
            selected = selected.sort().values
        raw = torch.full((TOPK,), -1, dtype=torch.int32, device=scores.device)
        raw[: selected.numel()] = selected.to(torch.int32)
        raw_rows.append(raw)
    raw_indices = torch.stack(raw_rows)
    valid = raw_indices >= 0
    safe = raw_indices.clamp_min(0)
    slots = torch.div(safe, page_size, rounding_mode="floor").long()
    physical_pages = torch.gather(page_tables, 1, slots)
    page_indices = physical_pages * page_size + safe.remainder(page_size)
    page_indices.masked_fill_(~valid | (physical_pages < 0), -1)
    return page_indices, raw_indices


def make_inputs(seq_lens_host, row_padding=17):
    max_seq_len = max(max(seq_lens_host), TOPK)
    batch_size = len(seq_lens_host)
    score_storage = torch.randn(
        batch_size,
        max_seq_len + row_padding,
        dtype=torch.float32,
        device="cuda",
    )
    scores = score_storage[:, :max_seq_len]
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_tables = torch.stack(
        [
            torch.randperm(num_pages, device="cuda", dtype=torch.int64).to(
                torch.int32
            )
            for _ in range(batch_size)
        ]
    )
    return scores, seq_lens, page_tables


def check_case(seq_lens_host, return_raw_indices):
    scores, seq_lens, page_tables = make_inputs(seq_lens_host)
    out_page_indices = torch.empty(
        len(seq_lens_host), TOPK, dtype=torch.int32, device="cuda"
    )
    out_raw_indices = (
        torch.empty_like(out_page_indices) if return_raw_indices else None
    )

    topk_transform_512(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        PAGE_SIZE,
        out_raw_indices,
    )
    ref_page_indices, ref_raw_indices = topk_transform_512_reference(
        scores, seq_lens, page_tables, PAGE_SIZE
    )
    torch.cuda.synchronize()

    assert torch.equal(
        out_page_indices.sort(dim=1).values,
        ref_page_indices.sort(dim=1).values,
    )
    if out_raw_indices is not None:
        assert torch.equal(
            out_raw_indices.sort(dim=1).values,
            ref_raw_indices.sort(dim=1).values,
        )

        valid = out_raw_indices >= 0
        safe_raw_indices = out_raw_indices.clamp_min(0)
        page_slots = torch.div(
            safe_raw_indices, PAGE_SIZE, rounding_mode="floor"
        ).to(torch.long)
        page_offsets = safe_raw_indices.remainder(PAGE_SIZE)
        transformed = (
            torch.gather(page_tables, 1, page_slots) * PAGE_SIZE + page_offsets
        )
        transformed.masked_fill_(~valid, -1)
        assert torch.equal(out_page_indices, transformed)


def check_invalid_page_size():
    scores, seq_lens, page_tables = make_inputs([TOPK], row_padding=0)
    out_page_indices = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")
    try:
        topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            63,
        )
    except RuntimeError as error:
        assert "power of two" in str(error)
        return
    raise AssertionError("non-power-of-two page_size should be rejected")


def benchmark(callable_fn, warmup=10, repeat=100):
    for _ in range(warmup):
        callable_fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        callable_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


def run_performance(batch_size=4, seq_len=8192):
    scores, seq_lens, page_tables = make_inputs([seq_len] * batch_size)
    out_page_indices = torch.empty(
        batch_size, TOPK, dtype=torch.int32, device="cuda"
    )
    out_raw_indices = torch.empty_like(out_page_indices)

    def run_lightop():
        topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            PAGE_SIZE,
            out_raw_indices,
        )

    def run_reference():
        topk_transform_512_reference(scores, seq_lens, page_tables, PAGE_SIZE)

    reference_us = benchmark(run_reference)
    lightop_us = benchmark(run_lightop)
    logical_bytes = (
        scores.numel() * scores.element_size()
        + seq_lens.numel() * seq_lens.element_size()
        + batch_size * TOPK * page_tables.element_size()
        + out_page_indices.numel() * out_page_indices.element_size()
        + out_raw_indices.numel() * out_raw_indices.element_size()
    )
    effective_gbps = logical_bytes / (lightop_us * 1e-6) / 1e9

    print("\nPerformance (10 warmup, 100 iterations)")
    print(f"shape: scores=[{batch_size}, {seq_len}], dtype=float32")
    print(f"PyTorch reference: {reference_us:.2f} us")
    print(f"LightOp:           {lightop_us:.2f} us")
    print(f"Speedup:           {reference_us / lightop_us:.2f}x")
    print(f"Effective BW:      {effective_gbps:.2f} GB/s")


def test_topk_transform_512():
    assert torch.cuda.is_available(), "CUDA/ROCm device is required"
    torch.manual_seed(2026)

    seq_lens_cases = [
        [0, 1, 127, 511],
        [512, 513, 1024, 4096],
        [777, 2049, 8192],
        [65536, 131072, MAX_CONTEXT_LEN - 1, MAX_CONTEXT_LEN],
    ]
    for seq_lens_host in seq_lens_cases:
        for return_raw_indices in (False, True):
            check_case(seq_lens_host, return_raw_indices)
            print(
                f"PASS seq_lens={seq_lens_host}, "
                f"return_raw_indices={return_raw_indices}"
            )

    check_invalid_page_size()
    print("PASS invalid page_size check")


@torch.no_grad()
def test_topk_transform_512_sparse_mask():
    assert torch.cuda.is_available(), "CUDA/ROCm device is required"
    torch.manual_seed(20261102)
    width = 1043
    seq_lens = torch.tensor([127, 513, 1031], dtype=torch.int32, device="cuda")
    scores = torch.rand(3, width, dtype=torch.float32, device="cuda") + 0.25
    nonzero = torch.zeros(3, width, dtype=torch.bool, device="cuda")
    nonzero[0, 1:127:3] = True
    nonzero[1, :513:5] = True
    nonzero[2, :1031] = torch.rand(1031, device="cuda") > 0.25
    # Tail bits and scores must not affect selection.
    nonzero[0, 127:] = True
    nonzero[1, 513:] = True
    scores[0, 127:] = 1.0e6
    scores[1, 513:] = 1.0e6
    mask = pack_nonzero_mask(nonzero)

    for page_size in (1, 64):
        num_pages = (width + page_size - 1) // page_size
        page_tables = torch.arange(
            num_pages, dtype=torch.int32, device="cuda"
        ).repeat(3, 1)
        page_tables[1].add_(num_pages)
        page_tables[2, ::7] = -1
        expected_page, expected_raw = sparse_mask_reference(
            scores, nonzero, seq_lens, page_tables, page_size
        )
        for return_raw in (False, True):
            actual_page = torch.empty(3, TOPK, dtype=torch.int32, device="cuda")
            actual_raw = torch.empty_like(actual_page) if return_raw else None
            topk_transform_512_sparse_mask(
                scores,
                mask,
                seq_lens,
                page_tables,
                actual_page,
                page_size,
                actual_raw,
            )
            torch.cuda.synchronize()
            assert torch.equal(actual_page, expected_page)
            if actual_raw is not None:
                assert torch.equal(actual_raw, expected_raw)


def test_long_rows_exact_adaptive_sampler():
    if __package__:
        from .test_adaptive_topk import assert_fixed_pairs, fixed_call, fixed_inputs
    else:
        from test_adaptive_topk import assert_fixed_pairs, fixed_call, fixed_inputs

    for length in (131071, 131072, 131073, 262144, 262656, 262657):
        for kind in ("dense", "negative", "zero", "sparse", "narrow",
                     "overflow", "underfill", "special"):
            args, (_, expected_raw) = fixed_inputs(kind, length, TOPK)
            mapped, raw = fixed_call(args, TOPK)
            assert_fixed_pairs(mapped, raw, args, expected_raw)

    # Small-batch retry guard and the DSpark query-row regimes on either side.
    if __package__:
        from .test_adaptive_topk import fixed_reference
    else:
        from test_adaptive_topk import fixed_reference
    generator = torch.Generator().manual_seed(20261028)
    length = 262144
    for rows in (1, 5, 6, 7, 12, 18, 24):
        scores = torch.randn(rows, length, generator=generator)
        lengths = torch.full((rows,), length, dtype=torch.int32)
        pages = torch.arange((length + PAGE_SIZE - 1) // PAGE_SIZE,
                             dtype=torch.int32)[None, :].repeat(rows, 1) * 3 + 17
        _, expected_raw = fixed_reference(scores, lengths, pages, TOPK, PAGE_SIZE)
        args = (scores.cuda(), lengths.cuda(), pages.cuda())
        mapped, raw = fixed_call(args, TOPK)
        assert_fixed_pairs(mapped, raw, args, expected_raw)


def test_long_rows_graph_replay():
    if __package__:
        from .test_adaptive_topk import (
            assert_fixed_pairs, fixed_call, fixed_inputs, fixed_reference,
        )
    else:
        from test_adaptive_topk import (
            assert_fixed_pairs, fixed_call, fixed_inputs, fixed_reference,
        )

    args, _ = fixed_inputs("dense", 262657, TOPK)
    fixed_call(args, TOPK)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped, raw = fixed_call(args, TOPK)
    for kind in ("zero", "dense", "overflow", "underfill", "narrow", "special"):
        replacement, (_, expected_raw) = fixed_inputs(kind, 262657, TOPK)
        for dest, source in zip(args, replacement):
            dest.copy_(source)
        graph.replay()
        assert_fixed_pairs(mapped, raw, args, expected_raw)
    args[1].copy_(torch.tensor(
        [131071, 131072, 131073, 262655, 262656, 262657],
        dtype=torch.int32, device="cuda",
    ))
    _, expected_raw = fixed_reference(*(x.cpu() for x in args), TOPK, PAGE_SIZE)
    graph.replay()
    assert_fixed_pairs(mapped, raw, args, expected_raw)


def run_sparse_performance(batch_size=6, seq_len=131072, nonzero_per_row=256):
    scores, seq_lens, page_tables = make_inputs([seq_len] * batch_size)
    if not 0 <= nonzero_per_row <= seq_len:
        raise ValueError("nonzero_per_row must be in [0, seq_len]")
    scores.zero_()
    nonzero = torch.zeros_like(scores, dtype=torch.bool)
    if nonzero_per_row:
        positions = torch.linspace(
            0, seq_len - 1, nonzero_per_row, device="cuda"
        ).round().long()
        nonzero[:, positions] = True
        scores[:, positions] = torch.rand(
            batch_size, nonzero_per_row, device="cuda"
        ) + 0.25
    mask = pack_nonzero_mask(nonzero)
    out_page_indices = torch.empty(
        batch_size, TOPK, dtype=torch.int32, device="cuda"
    )
    out_raw_indices = torch.empty_like(out_page_indices)

    def run_sparse():
        topk_transform_512_sparse_mask(
            scores,
            mask,
            seq_lens,
            page_tables,
            out_page_indices,
            PAGE_SIZE,
            out_raw_indices,
        )

    def run_dense():
        topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            PAGE_SIZE,
            out_raw_indices,
        )

    dense_us = benchmark(run_dense)
    sparse_us = benchmark(run_sparse)
    print("\nSparse-mask performance (10 warmup, 100 iterations)")
    print(f"shape: scores=[{batch_size}, {seq_len}], dtype=float32")
    print(
        f"nonzero/row: {nonzero_per_row} "
        f"(zero_ratio={1.0 - nonzero_per_row / seq_len:.4%})"
    )
    print(f"Dense Top512:       {dense_us:.2f} us")
    print(f"Sparse-mask Top512: {sparse_us:.2f} us")
    print(f"Speedup:            {dense_us / sparse_us:.2f}x")


if __name__ == "__main__":
    test_topk_transform_512()
    test_topk_transform_512_sparse_mask()
    test_long_rows_exact_adaptive_sampler()
    test_long_rows_graph_replay()
    run_performance()
    run_performance(batch_size=6, seq_len=131072)
    run_sparse_performance(nonzero_per_row=256)
    run_sparse_performance(nonzero_per_row=4096)
