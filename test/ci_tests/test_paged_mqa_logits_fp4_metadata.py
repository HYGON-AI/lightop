"""FP4 scheduling boundaries, one-million-token contexts and live graph replay."""

from bisect import bisect_right
from itertools import accumulate

import pytest
import torch


def reference_schedule(lengths, rows_per_request, ctas):
    if rows_per_request > 1 and len(lengths) // rows_per_request >= 8:
        lengths = [
            max(lengths[start + offset : start + min(offset + 2, rows_per_request)])
            for start in range(0, len(lengths), rows_per_request)
            for offset in range(0, rows_per_request, 2)
        ]
    cumulative = list(accumulate((length + 255) // 256 for length in lengths))
    quotient, remainder = divmod(cumulative[-1], ctas)
    result = []
    for cta in range(ctas + 1):
        target = cta * quotient + min(cta, remainder)
        row = bisect_right(cumulative, target)
        previous = cumulative[row - 1] if row else 0
        result.append([row, target - previous])
    return torch.tensor(result, dtype=torch.int32)


def plan(lengths, rows_per_request=1):
    from lightop import op

    return op.get_paged_mqa_logits_fp4_metadata(
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


@pytest.mark.parametrize("rows,rows_per_request,tb", [(16, 1, 8), (1025, 1, 64),
                                                    (48, 6, 64), (56, 7, 64)])
def test_fp4_schedule_covers_same_logits_as_grid(rows, rows_per_request, tb):
    from lightop import op

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
    scheduled = op.paged_mqa_logits_fp4(*args, plan(lengths, rows_per_request),
                                       width, tb, None, rows_per_request)
    grid = op.paged_mqa_logits_fp4(*args, None, width, tb, None, rows_per_request)
    valid = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(scheduled[valid], grid[valid], rtol=0, atol=0)
