# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact integer references, ragged CP/page tails and mutable graph inputs."""

import argparse
import json
import math
from pathlib import Path

import pytest
import torch


def native():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires HIP hardware")
    from lightop import dsv4_prefill

    assert dsv4_prefill.sparse_prefill_metadata_backend_info()["abi"] == 1
    return dsv4_prefill


def spaced(x):
    shape = tuple(2 * n + 1 for n in x.shape)
    storage = torch.empty(shape, dtype=x.dtype, device=x.device)
    view = storage[tuple(slice(1, 1 + 2 * n, 2) for n in x.shape)]
    view.copy_(x)
    return view


def cp_reference(extend, starts, positions, size, rank):
    counts = []
    for length, start in zip(extend.cpu().tolist(), starts.cpu().tolist()):
        # Enumerating global query slots is independent of the phase formula.
        indices = torch.arange(start, start + length, dtype=torch.int64)
        counts.append(int((indices.remainder(size) == rank).sum()))
    real = torch.tensor(counts, dtype=torch.int32)
    physical = real.clone()
    if counts:
        physical[-1] += positions.numel() - sum(counts)
    clean = positions.cpu().clone()
    clean[sum(counts) :] = -1
    return physical, real, clean


def test_cp_exact():
    op = native()
    cases = [
        [],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [7],
        [8],
        [127],
        [128],
        [129],
        [4095],
        [4096],
        [32767],
        [32768],
        [32769],
        [131072],
        [65535, 1, 65537, 0, 131072],
        [0, 1, 0, 7, 128, 3, 0],
        ([0, 1, 2, 3, 4, 5, 7, 31] * 8),
    ]
    for lengths in cases:
        starts = []
        offset = 13
        for length in lengths:
            starts.append(offset)
            offset += length
        for size in [1, 2, 3, 4, 5, 8, 16]:
            for rank in range(size):
                q = math.ceil(sum(lengths) / size) + (3 if lengths else 0)
                for stride in [False, True]:
                    extend = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                    beginning = torch.tensor(starts, dtype=torch.int32, device="cuda")
                    positions = torch.arange(q, dtype=torch.int32, device="cuda") + 117
                    if stride:
                        extend, beginning, positions = map(
                            spaced, (extend, beginning, positions)
                        )
                    got = op.dsv4_build_cp_sparse_query_metadata(
                        extend, beginning, positions, size, rank
                    )
                    for actual, expected in zip(
                        got, cp_reference(extend, beginning, positions, size, rank)
                    ):
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )


def pack_reference(source, seq, ratio, page):
    source = source.cpu()
    lengths = [max(x // ratio, 1) for x in seq.cpu().tolist()]
    result = []
    bases = []
    offset = 0
    for r, length in enumerate(lengths):
        bases.append(offset)
        offset += length
        token = torch.arange(length, dtype=torch.int64)
        ids = source[r, token // page if ratio == 4 else token].long()
        if ratio == 4:
            ids = ids * page + token.remainder(page)
        result.append(ids.clamp_min(0).int())
    return (
        torch.cat(result) if result else torch.empty(0, dtype=torch.int32),
        torch.tensor(bases, dtype=torch.int32),
    )


def test_pack_exact():
    op = native()
    for lengths in [
        [],
        [0],
        [1],
        [3],
        [4],
        [5],
        [127],
        [128],
        [129],
        [131072],
        [4194304],
        [0, 1, 3, 4, 5, 127, 128, 129, 4095, 32769],
        ([1, 7, 511, 4096] * 16),
    ]:
        for ratio in [4, 128]:
            for page in [1, 7, 64, 128]:
                counts = [max(n // ratio, 1) for n in lengths]
                total = sum(counts)
                columns = max(
                    (math.ceil(n / page) if ratio == 4 else n for n in counts),
                    default=1,
                )
                for dtype in [torch.int32, torch.int64]:
                    source = (
                        torch.arange(
                            len(lengths) * columns, dtype=dtype, device="cuda"
                        ).reshape(len(lengths), columns)
                        + 11
                    )
                    if source.numel():
                        source[:, 0] = -1
                    seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                    for stride in [False, True]:
                        ids = spaced(source) if stride else source
                        lens = spaced(seq) if stride else seq
                        out = torch.empty(total + 1, dtype=torch.int32, device="cuda")[
                            1:
                        ]
                        bases = torch.empty(
                            len(lengths) + 1, dtype=torch.int32, device="cuda"
                        )[1:]
                        actual = op.dsv4_pack_compact_sparse_token_ids(
                            ids, lens, ratio, page, total, out_ids=out, out_base=bases
                        )
                        expected = pack_reference(ids, lens, ratio, page)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
                        assert actual[0].data_ptr() == out.data_ptr()


def test_swa_query_prefix_fusion():
    op = native()
    from test_dsv41_build_swa_token_ids import make_case, reference

    for batch, width in [(0, 1), (1, 1), (1, 127), (1, 65536), (8, 8192), (64, 2048)]:
        for stride in [False, True]:
            for window in [1, 128, 256]:
                args = make_case(batch, width, window, stride)
                queries = torch.arange(
                    batch, dtype=torch.int32, device="cuda"
                ).remainder(7)
                if stride:
                    queries = spaced(queries)
                starts = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
                actual = op.dsv41_build_swa_token_ids(
                    *args, query_seq_lens=queries, query_start_loc=starts
                )
                for a, b in zip(actual, reference(*args)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                expected = torch.cat(
                    [torch.zeros(1, dtype=torch.int32), queries.cpu().cumsum(0).int()]
                )
                torch.testing.assert_close(starts.cpu(), expected, rtol=0, atol=0)
    # Exercise the bulk eight-chain gather with odd tails, several requests,
    # physical token permutations and independently strided tables/mappings.
    for batch, width in [(1, 1048579), (8, 262144)]:
        for stride in [False, True]:
            args = list(make_case(batch, width, 128, stride, query_len=width))
            args[3].copy_((args[3].long() * 31).remainder(batch * width).int())
            queries = torch.arange(batch, dtype=torch.int32, device="cuda") * 13
            starts = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
            actual = op.dsv41_build_swa_token_ids(
                *args, query_seq_lens=queries, query_start_loc=starts
            )
            for a, b in zip(actual, reference(*args)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(
                starts.cpu(),
                torch.cat(
                    [torch.zeros(1, dtype=torch.int32), queries.cpu().cumsum(0).int()]
                ),
                rtol=0,
                atol=0,
            )


def combine_reference(
    topk,
    starts,
    seq,
    gather,
    compressed,
    swa,
    window,
    ratio,
    k,
    positions,
    initial=None,
):
    topk = topk.cpu()
    starts = starts.cpu().tolist()
    seq = seq.cpu().tolist()
    gather = gather.cpu().tolist()
    compressed = compressed.cpu().tolist()
    swa = swa.cpu().tolist()
    positions = positions.cpu().tolist() if positions is not None else None
    rows = topk.shape[0]
    width = ((k + window + 127) // 128) * 128
    out = (
        torch.full((rows, width), -1, dtype=torch.int32)
        if initial is None
        else initial.cpu().clone()
    )
    lens = torch.empty(rows, dtype=torch.int32)
    for r in range(len(seq)):
        start = starts[r] - starts[0]
        end = starts[r + 1] - starts[0]
        for row in range(start, end):
            pos = (
                positions[row]
                if positions is not None
                else seq[r] - (end - start) + row - start
            )
            nk = min((pos + 1) // ratio, k)
            ns = min(pos + 1, window)
            lens[row] = nk + ns
            selected = topk[row, :nk]
            out[row, :nk] = torch.where(selected >= 0, selected + compressed[r], -1)
            out[row, nk : nk + ns] = (
                torch.arange(ns, dtype=torch.int32)
                + swa[r]
                + pos
                - ns
                + 1
                - (seq[r] - gather[r])
            )
    return out, lens


def test_combine_exact():
    op = native()
    for rows, k in [
        (0, 0),
        (1, 0),
        (1, 1),
        (7, 7),
        (31, 512),
        (4095, 2048),
        (4096, 2048),
    ]:
        for ratio in [1, 3, 4, 128]:
            for window in [0, 1, 128, 256]:
                starts = torch.tensor(
                    [31, 31, 31 + rows // 2, 31 + rows],
                    dtype=torch.int32,
                    device="cuda",
                )
                seq = torch.full((3,), rows * 8 + 512, dtype=torch.int32, device="cuda")
                gather = seq.clone()
                compressed = torch.tensor(
                    [0, 17, 50000], dtype=torch.int32, device="cuda"
                )
                swa = torch.tensor(
                    [0, 1000000, 2000000], dtype=torch.int32, device="cuda"
                )
                topk = torch.arange(
                    rows * (k + 3), dtype=torch.int32, device="cuda"
                ).reshape(rows, k + 3)[:, :k]
                if topk.numel():
                    topk[:, 0] = -1
                for explicit in [False, True]:
                    positions = (
                        torch.arange(rows, dtype=torch.int32, device="cuda") * 8 + 117
                        if explicit
                        else None
                    )
                    if explicit and rows:
                        positions[-1] = -1
                    args = (
                        topk,
                        starts,
                        seq,
                        gather,
                        compressed,
                        swa,
                        window,
                        ratio,
                        k,
                    )
                    actual = op.dsv4_combine_topk_swa_indices(
                        *args, query_positions=positions
                    )
                    expected = combine_reference(*args, positions)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
                    if rows <= 31:
                        width = actual[0].shape[1]
                        out = torch.full(
                            (rows, width + 17), 777, dtype=torch.int32, device="cuda"
                        )[:, :width]
                        lens = torch.empty(rows, dtype=torch.int32, device="cuda")
                        expected = combine_reference(*args, positions, out)
                        actual = op.dsv4_combine_topk_swa_indices(
                            *args,
                            query_positions=positions,
                            out_indices=out,
                            out_lens=lens
                        )
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


def test_mutable_graph_and_stream():
    op = native()
    device = "cuda"
    extend = torch.tensor([9, 7], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 9], dtype=torch.int32, device=device)
    positions = torch.tensor([100, 200], dtype=torch.int32, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = op.dsv4_build_cp_sparse_query_metadata(
                extend, starts, positions, 8, 0
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.dsv4_build_cp_sparse_query_metadata(extend, starts, positions, 8, 0)
    for lengths, beginnings in [([8, 8], [0, 8]), ([1, 15], [0, 1]), ([9, 7], [0, 9])]:
        extend.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        starts.copy_(torch.tensor(beginnings, dtype=torch.int32, device=device))
        positions.add_(7)
        graph.replay()
        for a, b in zip(out, cp_reference(extend, starts, positions, 8, 0)):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    seq = torch.tensor([256, 512], dtype=torch.int32, device=device)
    source = torch.arange(8, dtype=torch.int32, device=device).reshape(2, 4)
    for _ in range(3):
        op.dsv4_pack_compact_sparse_token_ids(source, seq, 4, 64, 192)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.dsv4_pack_compact_sparse_token_ids(source, seq, 4, 64, 192)
    for values in [[512, 256], [384, 384], [256, 512]]:
        seq.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        source.add_(1)
        graph.replay()
        for a, b in zip(out, pack_reference(source, seq, 4, 64)):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    from test_dsv41_build_swa_token_ids import make_case, reference

    args = list(make_case(2, 1024, query_len=128))
    queries = torch.tensor([0, 7], dtype=torch.int32, device=device)
    qstarts = torch.empty(3, dtype=torch.int32, device=device)
    for _ in range(3):
        op.dsv41_build_swa_token_ids(
            *args, query_seq_lens=queries, query_start_loc=qstarts
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.dsv41_build_swa_token_ids(
            *args, query_seq_lens=queries, query_start_loc=qstarts
        )
    for _ in range(3):
        args[4].add_(3)
        args[1][0].sub_(1)
        args[1][1].add_(1)
        queries.copy_(queries.flip(0))
        graph.replay()
        for a, b in zip(out, reference(*args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(
            qstarts.cpu(),
            torch.cat(
                [torch.zeros(1, dtype=torch.int32), queries.cpu().cumsum(0).int()]
            ),
            rtol=0,
            atol=0,
        )
    idx = torch.arange(16 * 512, dtype=torch.int32, device=device).reshape(16, 512)
    starts = torch.tensor([91, 91, 99, 107], dtype=torch.int32, device=device)
    seq = torch.full((3,), 4096, dtype=torch.int32, device=device)
    gather = seq.clone()
    comp = torch.tensor([0, 10000, 20000], dtype=torch.int32, device=device)
    swa = torch.tensor([30000, 40000, 50000], dtype=torch.int32, device=device)
    pos = torch.arange(16, dtype=torch.int32, device=device) * 8 + 111
    values = (idx, starts, seq, gather, comp, swa, 128, 4, 512)
    for _ in range(3):
        op.dsv4_combine_topk_swa_indices(*values, query_positions=pos)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.dsv4_combine_topk_swa_indices(*values, query_positions=pos)
    for last in [-1, 111, -1]:
        idx.add_(1)
        pos[-1] = last
        graph.replay()
        for a, b in zip(out, combine_reference(*values, pos)):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


def test_invalid_metadata():
    op = native()
    x = torch.tensor([8], dtype=torch.int32, device="cuda")
    pos = torch.arange(1, dtype=torch.int32, device="cuda")
    for size, rank in [(0, 0), (8, -1), (8, 8)]:
        with pytest.raises(RuntimeError):
            op.dsv4_build_cp_sparse_query_metadata(x, x, pos, size, rank)
    with pytest.raises(RuntimeError):
        op.dsv4_build_cp_sparse_query_metadata(x.cpu(), x, pos, 8, 0)
    source = torch.ones((1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        op.dsv4_pack_compact_sparse_token_ids(source, x, 0, 64, 2)
    with pytest.raises(RuntimeError):
        op.dsv4_pack_compact_sparse_token_ids(source, x, 4, 0, 2)
    with pytest.raises(RuntimeError):
        op.dsv4_pack_compact_sparse_token_ids(source.float(), x, 4, 64, 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    checks = []
    for function in [
        test_cp_exact,
        test_pack_exact,
        test_swa_query_prefix_fusion,
        test_combine_exact,
        test_mutable_graph_and_stream,
        test_invalid_metadata,
    ]:
        function()
        checks.append(function.__name__)
        print(function.__name__ + ": PASS", flush=True)
    result = dict(
        correctness="PASS",
        checks=checks,
        backend=native().sparse_prefill_metadata_backend_info(),
    )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
