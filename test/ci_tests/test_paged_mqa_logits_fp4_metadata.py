# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FP4 scheduling boundaries, one-million-token contexts and live graph replay."""

from bisect import bisect_right
from itertools import accumulate

import pytest
import torch


def reference_schedule(lengths, rows_per_request, ctas):
    batch = len(lengths)
    # Use the public eight-request threshold in the independent reference.
    paired = rows_per_request > 1 and batch // rows_per_request >= 8
    if paired:
        lengths = [
            max(lengths[start + offset : start + min(offset + 2, rows_per_request)])
            for start in range(0, len(lengths), rows_per_request)
            for offset in range(0, rows_per_request, 2)
        ]
    cumulative = list(accumulate((length + 255) // 256 for length in lengths))
    active_ctas = (320 if ctas == 640 and cumulative[-1] < 8192 else
                   128 if ctas == 256 and cumulative[-1] < 1024 else ctas)
    if ctas == 256 and cumulative[-1] >= 1024 and not paired and rows_per_request > 1 and batch <= 64:
        largest_route = (max(lengths) + 255) // 256
        if largest_route * batch <= 2 * cumulative[-1]:
            active_ctas = 192
    quotient, remainder = divmod(cumulative[-1], active_ctas)
    result = []
    for cta in range(ctas + 1):
        active_cta = min(cta, active_ctas)
        target = active_cta * quotient + min(active_cta, remainder)
        row = bisect_right(cumulative, target)
        previous = cumulative[row - 1] if row else 0
        result.append([row, target - previous])
    return torch.tensor(result, dtype=torch.int32)


def plan(lengths, rows_per_request=1):
    from lightop.attention import get_paged_mqa_logits_fp4_metadata

    return get_paged_mqa_logits_fp4_metadata(
        lengths, 64, torch.cuda.get_device_properties(lengths.device).multi_processor_count,
        rows_per_request,
    )


@pytest.mark.parametrize("rows", [1, 8, 64, 384, 1024, 1025, 1088, 1500, 2048,
                                 2049, 3000, 4096, 5000, 8192, 8193, 16384])
@pytest.mark.parametrize("pattern", ["zero", "boundaries", "1m"])
def test_fp4_schedule_rows_and_context_lengths(rows, pattern):
    values = [0] if pattern == "zero" else (
        [1048576] if pattern == "1m" else [0, 1, 255, 256, 257, 1048575, 1048576, 0]
    )
    lengths = torch.tensor([values[i % len(values)] for i in range(rows)],
                           device="cuda", dtype=torch.int32)
    actual = plan(lengths)
    expected = reference_schedule(lengths.cpu().tolist(), 1, actual.shape[0] - 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("requests,rows_per_request", [(1, 6), (7, 6), (8, 6),
                                                       (64, 6), (8, 7), (4096, 7)])
def test_fp4_schedule_pairing(requests, rows_per_request):
    lengths = torch.randint(0, 1048577, (requests * rows_per_request,),
                            device="cuda", dtype=torch.int32)
    actual = plan(lengths, rows_per_request)
    expected = reference_schedule(lengths.cpu().tolist(), rows_per_request,
                                  actual.shape[0] - 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fp4_schedule_limit():
    with pytest.raises(RuntimeError, match="route rows exceed"):
        plan(torch.ones(16385, device="cuda", dtype=torch.int32))


def test_fp4_schedule_replay_reads_current_lengths():
    lengths = torch.ones(384, device="cuda", dtype=torch.int32)
    plan(lengths, 6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan(lengths, 6)
    for value in [0, 257, 1048576]:
        lengths.fill_(value)
        graph.replay()
        expected = reference_schedule(lengths.cpu().tolist(), 6, actual.shape[0] - 1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fp4_schedule_replay_crosses_cta_work_threshold():
    lengths = torch.zeros(42, device="cuda", dtype=torch.int32)
    plan(lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan(lengths)
    for splits in [1023, 1024, 1025, 0, 1024]:
        lengths.copy_(torch.tensor([splits * 256] + [0] * 41, device="cuda", dtype=torch.int32))
        graph.replay()
        expected = reference_schedule(lengths.cpu().tolist(), 1, actual.shape[0] - 1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows,rows_per_request,tb", [(16, 1, 8), (1025, 1, 64),
                                                    (48, 6, 64), (56, 7, 64)])
def test_fp4_schedule_covers_same_logits_as_grid(rows, rows_per_request, tb):
    from lightop.attention import paged_mqa_logits_fp4

    heads, width = 16, 513
    q = torch.randint(0, 256, (rows, 1, heads, 64), device="cuda", dtype=torch.uint8)
    q_sf = torch.full((rows, 1, heads), 0x7f7f7f7f, device="cuda", dtype=torch.int32)
    pool = torch.randint(0, 256, (9, 4352), device="cuda", dtype=torch.uint8)
    pool[:, 4096:].fill_(127)
    weights = torch.rand((rows, heads), device="cuda")
    lengths = torch.randint(0, width + 1, (rows,), device="cuda", dtype=torch.int32)
    table = torch.arange((width + tb - 1) // tb, device="cuda", dtype=torch.int32)
    table = table.expand(rows, -1).contiguous()
    args = (q, q_sf, pool, weights, lengths, table)
    scheduled = paged_mqa_logits_fp4(*args, plan(lengths, rows_per_request),
                                   width, tb, None, rows_per_request)
    grid = paged_mqa_logits_fp4(*args, None, width, tb, None, rows_per_request)
    valid = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(scheduled[valid], grid[valid], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [31, 32, 33, 63, 65])
@pytest.mark.parametrize("pattern", ["zero", "boundaries", "1m"])
def test_fp4_wave_scan_boundaries(rows, pattern):
    values = [0] if pattern == "zero" else (
        [1048576] if pattern == "1m" else [0, 1, 255, 256, 257, 1048575, 1048576, 0]
    )
    lengths = torch.tensor([values[i % len(values)] for i in range(rows)],
                           device="cuda", dtype=torch.int32)
    actual = plan(lengths)
    expected = reference_schedule(lengths.cpu().tolist(), 1, actual.shape[0] - 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fp4_seven_request_schedule_crosses_cta_threshold():
    lengths = torch.zeros(42, device="cuda", dtype=torch.int32)
    plan(lengths, 6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan(lengths, 6)
    for splits in [1023, 1024, 1025, 0, 2048]:
        lengths.copy_(torch.tensor([splits * 256] + [0] * 41, device="cuda", dtype=torch.int32))
        graph.replay()
        expected = reference_schedule(lengths.cpu().tolist(), 6, actual.shape[0] - 1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("requests,rows", [(7, 6), (5, 7), (4, 16)])
def test_multirow_plan_reacts_to_live_length_imbalance(requests, rows):
    batch = requests * rows
    lengths = torch.full((batch,), 32 * 256, device="cuda", dtype=torch.int32)
    plan(lengths, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        schedule = plan(lengths, rows)
    if schedule.shape[0] != 257:
        pytest.skip("the density-dependent policy targets the 256-CTA device")

    # Same total work at, above and below twice the mean. Replay the same graph
    # in both directions so the choice is based on the live route lengths.
    for splits, active in [([64] + [32]*(batch-2) + [0], 192),
                           ([65, 31] + [32]*(batch-3) + [0], 256),
                           ([32]*batch, 192), ([0]*(batch-1) + [1023], 128)]:
        lengths.copy_(torch.tensor(splits, device="cuda", dtype=torch.int32) * 256)
        graph.replay()
        result = schedule.cpu()
        assert (result[:-1] != result[1:]).any(dim=1).sum().item() == active
        expected = reference_schedule([n*256 for n in splits], rows, 256)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("splits", [8191, 8192, 8193])
def test_gfx936_density_boundary(splits):
    """The live split count selects density without changing schedule coverage."""
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx936") or props.multi_processor_count != 80:
        pytest.skip("gfx936 80-CU density policy")
    lengths = [256 * (splits // 16 + (i < splits % 16)) for i in range(16)]
    tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    actual = plan(tensor)
    assert actual.shape == (641, 2)
    torch.testing.assert_close(actual.cpu(), reference_schedule(lengths, 1, 640),
                               rtol=0, atol=0)


def test_gfx936_density_replay_reads_live_work():
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx936") or props.multi_processor_count != 80:
        pytest.skip("gfx936 80-CU density policy")
    lengths = torch.ones(16, device="cuda", dtype=torch.int32)
    plan(lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan(lengths)
    for splits in (8191, 8192, 8193, 0, 8192):
        values = [256 * (splits // 16 + (i < splits % 16)) for i in range(16)]
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(actual.cpu(), reference_schedule(values, 1, 640),
                                   rtol=0, atol=0)
