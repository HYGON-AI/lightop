import pytest
import torch

from lightop import op
from lightop.attention import mqa_logits_topk_sampled_int8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="sampled INT8 MQA TopK requires gfx936 HIP",
)


@pytest.fixture(autouse=True)
def supported_device(monkeypatch):
    monkeypatch.setenv("LIGHTOP_MQA_LOGITS_BETA", "1")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx936"):
        pytest.skip("sampled INT8 MQA TopK requires gfx936")


def make_inputs(rows=17, width=262144):
    device = "cuda"
    q = torch.randint(-127, 128, (rows, 32, 128), device=device, dtype=torch.int8)
    k = torch.randint(-127, 128, (width, 128), device=device, dtype=torch.int8)
    weights = torch.empty((rows, 35), device=device)[:, :32]
    weights.uniform_(-0.000015, 0.00002)
    scale = torch.empty(width, device=device).uniform_(0.001, 0.02)
    starts = torch.zeros(rows, device=device, dtype=torch.int32)
    ends = torch.full_like(starts, width)
    offsets = torch.arange(rows, device=device, dtype=torch.int32) * 3
    return q, k, weights, scale, starts, ends, offsets


def dense_reference(args):
    q, k, weights, scale, starts, ends, offsets = args
    logits = op.mqa_logits(
        q, k, weights, starts, ends, q.shape[0], k.shape[0],
        32, 128, scale, True, None,
    )
    result = torch.empty((q.shape[0], 2048), device=q.device, dtype=torch.int32)
    op.topk_transform_ragged_v2_interface(
        logits, ends - starts, result, offsets, starts,
    )
    return result, logits


def outputs(args):
    q = args[0]
    return (
        torch.empty((q.shape[0], 2048), device=q.device),
        torch.empty(q.shape[0], device=q.device),
        torch.empty((q.shape[0], 2), device=q.device, dtype=torch.int32),
    )


def assert_exact(args, result, values, threshold):
    expected, logits = dense_reference(args)
    assert torch.equal(result.sort(-1).values, expected.sort(-1).values)
    positions = result - args[-1][:, None] + args[-3][:, None]
    selected = logits.gather(1, positions.long().clamp(0, logits.shape[1] - 1))
    selected.masked_fill_(result < 0, -float("inf"))
    assert torch.equal(selected.contiguous().view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(threshold.view(torch.uint8), values[:, -1].contiguous().view(torch.uint8))
    finite = torch.isfinite(values[:, 1:]) & torch.isfinite(values[:, :-1])
    assert bool(torch.all(~finite | (values[:, :-1] >= values[:, 1:])))


@pytest.mark.parametrize("width", [4099, 131072, 262143, 262144, 869760, 1048576])
def test_updates_and_optional_outputs(width):
    args = make_inputs(width=width)
    q, k, weights, scale, starts, ends, offsets = args
    values, threshold, stats = outputs(args)
    for update in range(3):
        q.random_(-127, 128)
        k.random_(-127, 128)
        weights.uniform_(-0.000015, 0.00002)
        scale.uniform_(0.001, 0.02)
        starts.zero_()
        ends.fill_(width)
        ends[:5] = torch.tensor([0, 1, 2047, 2048, 2049], device=q.device)
        starts[5:] = 127
        offsets.add_(13)
        if update == 2:
            q.zero_()
        result = mqa_logits_topk_sampled_int8(*args, values, threshold, stats)
        assert_exact(args, result, values, threshold)
    result = mqa_logits_topk_sampled_int8(*args)
    expected, _ = dense_reference(args)
    assert torch.equal(result.sort(-1).values, expected.sort(-1).values)


@pytest.mark.parametrize("ties", [4097, 6000, 6001, 8191, 16383, 16384, 16385])
def test_exact_ties_across_partial_waves(ties):
    args = make_inputs()
    q, k, weights, scale, _, _, offsets = args
    q.fill_(1)
    k.fill_(-1)
    k[:ties].fill_(1)
    weights.fill_(1)
    scale.fill_(1)
    values, threshold, stats = outputs(args)
    expected = torch.arange(2048, dtype=torch.int32, device=q.device)[None, :] + offsets[:, None]
    for _ in range(32 if ties == 6000 else 3):
        result = mqa_logits_topk_sampled_int8(*args, values, threshold, stats)
        assert torch.equal(result.sort(-1).values, expected)
        assert_exact(args, result, values, threshold)
    if ties == 6000:
        assert bool(torch.all(stats[:, 0] == 0))


def test_nonfinite_exact_fallback():
    args = make_inputs(rows=5)
    args[2][0, 0] = float("nan")
    args[2][1, 0] = float("inf")
    args[2][2, 0] = -float("inf")
    args[3][211] = float("nan")
    args[3][993] = float("inf")
    values, threshold, stats = outputs(args)
    result = mqa_logits_topk_sampled_int8(*args, values, threshold, stats)
    assert_exact(args, result, values, threshold)
    assert bool(torch.all(stats[:, 0] != 0))


def test_graph_updates_inputs_bounds_and_routes():
    args = make_inputs(rows=65)
    values, threshold, stats = outputs(args)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            mqa_logits_topk_sampled_int8(*args, values, threshold, stats)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = mqa_logits_topk_sampled_int8(*args, values, threshold, stats)
    for update in range(3):
        q, k, weights, scale, starts, ends, offsets = args
        q.random_(-127, 128)
        k.random_(-127, 128)
        weights.uniform_(-0.000015, 0.00002)
        scale.uniform_(0.001, 0.02)
        starts.zero_()
        ends.fill_(k.shape[0])
        if update == 1:
            ends[::3] = 2000
            starts[1::3] = k.shape[0] // 2
            offsets.add_(13)
        if update == 2:
            q.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(args, result, values, threshold)


def test_reject_aliasing_optional_outputs():
    args = make_inputs()
    values, threshold, stats = outputs(args)
    with pytest.raises(RuntimeError, match="outputs alias"):
        mqa_logits_topk_sampled_int8(*args, values, values.flatten()[:args[0].shape[0]], stats)


@pytest.mark.parametrize("width", [4099, 262144, 869760])
def test_unsorted_indices_preserve_selected_set(width):
    args = make_inputs(rows=65, width=width)
    for update in range(3):
        args[0].random_(-127, 128)
        if update == 1:
            args[4][::3] = width // 2
        if update == 2:
            args[0].zero_()
        ordered = mqa_logits_topk_sampled_int8(*args, fallback_workspace_rows=64)
        unordered = mqa_logits_topk_sampled_int8(*args, fallback_workspace_rows=64, sort_topk=False)
        expected, _ = dense_reference(args)
        assert torch.equal(unordered.sort(-1).values, ordered.sort(-1).values)
        assert torch.equal(unordered.sort(-1).values, expected.sort(-1).values)
    values, threshold, _ = outputs(args)
    with pytest.raises(RuntimeError, match="sorted optional outputs"):
        mqa_logits_topk_sampled_int8(*args, values, threshold, sort_topk=False)


@pytest.mark.parametrize("rows,width", [(5, 4099), (17, 65535), (65, 262144)])
@pytest.mark.parametrize("reuse_full_window", [False, True])
def test_direct_candidate_emit_preserves_scores_and_routes(rows, width, reuse_full_window):
    args = make_inputs(rows=rows, width=width)
    values, threshold, stats = outputs(args)
    reference_stats = torch.empty_like(stats)
    kwargs = dict(fallback_workspace_rows=64, reuse_full_window=reuse_full_window, min_sample_k=65535)
    for update in range(3):
        args[0].random_(-127, 128)
        args[1].random_(-127, 128)
        if update == 1:
            args[4][::3] = width // 2
        if update == 2:
            args[0].zero_()
        reference = mqa_logits_topk_sampled_int8(*args, stats=reference_stats, **kwargs)
        result = mqa_logits_topk_sampled_int8(
            *args, values, threshold, stats, direct_candidate_emit=True, **kwargs,
        )
        assert torch.equal(result.sort(-1).values, reference.sort(-1).values)
        assert torch.equal(stats, reference_stats)
        assert_exact(args, result, values, threshold)
