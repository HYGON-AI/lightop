# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""KPool TopK checks compare the complete order, including tied boundaries."""

import pytest
import torch
from lightop import fast_kpool_topk_transform_fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(score, lengths, pool_size, seq_lens, page_table=None,
              topk_indices_offset=None, row_starts=None, out_rows=None,
              page_table_row_index=None):
    score = score.cpu()
    lengths, seq_lens = lengths.cpu(), seq_lens.cpu()
    page_table = None if page_table is None else page_table.cpu()
    offsets = None if topk_indices_offset is None else topk_indices_offset.cpu().reshape(-1)
    starts = None if row_starts is None else row_starts.cpu()
    rowmap = None if page_table_row_index is None else page_table_row_index.cpu()
    rows, width = score.shape
    output = torch.full((rows if out_rows is None else out_rows, 2048 + pool_size - 1),
                        -1, dtype=torch.int32)
    for row in range(rows):
        start = 0 if starts is None else max(0, int(starts[row]))
        length = max(0, min(int(lengths[row]), max(0, width - start)))
        bits = score[row, start:start + length].view(torch.int32).to(torch.int64) & 0xffffffff
        keys = torch.where(bits & 0x80000000 != 0, (~bits) & 0xffffffff,
                           bits | 0x80000000)
        groups = torch.argsort(keys, descending=True, stable=True)[:2048 // pool_size]
        raw = (groups[:, None] * pool_size + torch.arange(pool_size)).flatten()
        tail = torch.arange(length * pool_size,
                            length * pool_size + int(seq_lens[row]) % pool_size)
        raw = torch.cat((raw, tail)).long()
        if page_table is not None:
            table_row = row if rowmap is None else int(rowmap[row])
            if not 0 <= table_row < page_table.shape[0]:
                continue
            valid = raw < page_table.shape[1]
            mapped = torch.full(raw.shape, -1, dtype=torch.int32)
            mapped[valid] = page_table[table_row, raw[valid]]
        elif offsets is not None:
            mapped = (raw + offsets[row]).int()
        else:
            mapped = raw.int()
        output[row, :mapped.numel()] = mapped
    return output


def run(**kwargs):
    return fast_kpool_topk_transform_fused(topk=2048, **kwargs)


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("length_case", ["empty", "one", "below", "equal", "above",
                                        "4097", "22528", "22529", "65537",
                                        "131073", "262145", "327680", "327681"])
@pytest.mark.parametrize("pattern", ["unique", "ties", "equal", "bits"])
def test_complete_order(pool_size, length_case, pattern):
    torch.manual_seed(20261123)
    k = 2048 // pool_size
    n = {"empty": 0, "one": 1, "below": k-1, "equal": k,
         "above": k+1}.get(length_case)
    if n is None:
        n = int(length_case)
    width = max(n + 7, 8)
    # Noncontiguous rows, contiguous columns, and nonzero row starts.
    storage = torch.empty((3, width + 11), device="cuda")
    score = storage[:, :width]
    if pattern == "unique":
        for row in range(3):
            score[row].copy_(torch.randperm(width, device="cuda").float())
    elif pattern == "ties":
        score.copy_(torch.randint(-2, 3, score.shape, device="cuda"))
    elif pattern == "equal":
        score.zero_()
    else:
        # Includes signed zero, infinities, subnormals and distinct signed NaNs.
        patterns = torch.tensor([0, -2147483648, 0x7f800000, -0x00800000,
                                 1, -2147483647, 0x7fc00001, 0x7fc01000,
                                 -0x003fffff, -0x003ff000, 0x7f800001,
                                 0x7fa00000, 0x7fbfffff, 0x7fc00000,
                                 -0x007fffff, -0x00600000], dtype=torch.int32)
        values = patterns.repeat((width + patterns.numel() - 1) // patterns.numel())[:width].view(torch.float32)
        score.copy_(values.to("cuda").expand_as(score))
    lengths = torch.tensor([n, max(0, n - 3), n], dtype=torch.int32, device="cuda")
    args = dict(score=score, lengths=lengths, pool_size=pool_size,
                seq_lens=lengths * pool_size + torch.tensor([0, 1, pool_size-1],
                                      dtype=torch.int32, device="cuda"),
                row_starts=torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
                out_rows=5)
    expected = reference(**args)
    for _ in range(5):
        assert torch.equal(run(**args).cpu(), expected)


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("mapping", ["paged", "offset"])
def test_graph_replay_and_input_changes(pool_size, mapping):
    torch.manual_seed(20261123)
    n = 4096
    score = torch.zeros((3, n), device="cuda")
    lengths = torch.tensor([n, 2048, 3000], dtype=torch.int32, device="cuda")
    args = dict(score=score, lengths=lengths, pool_size=pool_size,
                seq_lens=lengths * pool_size + torch.tensor([1, pool_size-1, 0],
                                           dtype=torch.int32, device="cuda"))
    if mapping == "paged":
        args["page_table"] = torch.randperm(n * pool_size + pool_size,
                                  dtype=torch.int32, device="cuda")[None].repeat(2, 1)
        args["page_table_row_index"] = torch.tensor([1, 0, -1],
                                            dtype=torch.int32, device="cuda")
    else:
        args["topk_indices_offset"] = torch.tensor([[100], [300], [500]],
                                                dtype=torch.int32, device="cuda")
    run(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(**args)
    expected = reference(**args)
    for _ in range(50):
        graph.replay()
        assert torch.equal(actual.cpu(), expected)
    # Graph must recompute both selected membership and canonical ordering.
    score.copy_(torch.randint(-2, 3, score.shape, device="cuda"))
    expected = reference(**args)
    for _ in range(5):
        graph.replay()
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("pool_size", [4, 16])
def test_graph_length_transitions(pool_size):
    torch.manual_seed(20261123)
    score = torch.randn((3, 4097), device="cuda")
    lengths = torch.tensor([0, 1, 4097], dtype=torch.int32, device="cuda")
    seq_lens = lengths * pool_size + 1
    args = dict(score=score, lengths=lengths, pool_size=pool_size,
                seq_lens=seq_lens, out_rows=5)
    run(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(**args)
    for values in ([0, 1, 4097], [512, 513, 4096], [4096, 2, 0], [4097, 2048, 300]):
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        seq_lens.copy_(lengths * pool_size + pool_size - 1)
        expected = reference(**args)
        for _ in range(5):
            graph.replay()
            assert torch.equal(actual.cpu(), expected)
    score.zero_()
    expected = reference(**args)
    graph.replay()
    assert torch.equal(actual.cpu(), expected)
