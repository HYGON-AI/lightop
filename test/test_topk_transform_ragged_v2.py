# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unordered ragged Top512/1024/2048 contracts and a PyTorch benchmark.

NaN scores are unsupported. Metadata values obey the caller contract:
0 <= start, 0 <= length, start + length <= width, 0 <= output offset,
and every mapped ID fits int32.
"""

import argparse
import json
import sys

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="native ragged TopK requires HIP",
)


def call(scores, lengths, offsets, out, starts=None):
    # Defer LightOp import so --baseline-only works before the new op is built.
    import lightop

    return lightop.topk_transform_ragged_v2(
        scores, lengths, out_offsets=offsets, out_indices=out,
        row_starts=starts,
    )


def assert_selection(scores, lengths, offsets, out, starts=None, exact=False):
    """Check selected membership and the cutoff without prescribing tie IDs."""
    scores, lengths, offsets, out = (
        tensor.cpu() for tensor in (scores, lengths, offsets, out)
    )
    starts = torch.zeros_like(lengths) if starts is None else starts.cpu()
    k = out.shape[1]
    for row in range(scores.shape[0]):
        length, start, offset = int(lengths[row]), int(starts[row]), int(offsets[row])
        count = min(length, k)
        valid = out[row] >= 0
        assert int(valid.sum()) == count
        assert torch.all(out[row, ~valid] == -1)
        ids = out[row, valid].long() - offset
        assert torch.all((ids >= 0) & (ids < length))
        assert ids.unique().numel() == count
        if count == 0:
            continue
        values = scores[row, start:start + length]
        reference = torch.topk(values, count, sorted=False)
        if exact:
            assert torch.equal(ids.sort().values, reference.indices.sort().values)
        else:
            cutoff = reference.values.min()
            assert torch.all(values[ids] >= cutoff)
            # Every score strictly above the cutoff must be selected.
            assert int((values[ids] > cutoff).sum()) == int((values > cutoff).sum())


def inputs(lengths, k, width=None, starts=None, kind="mixed", padded=True):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    starts_cpu = torch.zeros_like(lengths) if starts is None else torch.tensor(starts, dtype=torch.int32)
    rows = lengths.numel()
    if width is None:
        width = max(1, int((lengths + starts_cpu).max())) if rows else 17
    storage = torch.full((rows, width + (19 if padded else 0)), float("inf"))
    generator = torch.Generator().manual_seed(42)
    for row in range(rows):
        length, start = int(lengths[row]), int(starts_cpu[row])
        values = torch.randperm(length, generator=generator).float()
        if kind == "negative":
            values = -values - 1
        elif kind == "positive":
            values = values + 1
        elif kind == "mixed":
            values = values - length // 2
        elif kind == "ties":
            values = values.remainder(17) - 8
        elif kind == "zero":
            values.zero_()
        elif kind == "negative_infinity":
            values.fill_(float("-inf"))
        elif kind == "infinities":
            values = values.remainder(5) - 2
            values[::3] = float("inf")
            values[1::3] = float("-inf")
        storage[row, start:start + length] = values
    scores = storage.cuda()[:, :width]
    offsets = (torch.arange(rows, dtype=torch.int32) * (width + 101) + 37).cuda()
    out = torch.full((rows, k), -777, dtype=torch.int32, device="cuda")
    return scores, lengths.cuda(), offsets, out, None if starts is None else starts_cpu.cuda()


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("kind", ["negative", "mixed", "positive"])
@pytest.mark.parametrize("width", [1024, 8192, 65539])
def test_unique_scores_match_pytorch(k, kind, width):
    args = inputs([width, width - 7, min(k - 1, width), min(k, width), 0],
                  k, width=width, kind=kind)
    assert call(*args) is None
    assert_selection(*args, exact=True)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("kind", ["ties", "zero", "negative_infinity", "infinities"])
def test_tied_scores_membership(k, kind):
    args = inputs([8192, k + 1, k, k - 1, 1, 0], k, kind=kind)
    call(*args)
    assert_selection(*args)


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_boundaries_starts_offsets_and_row_padding(k):
    lengths = [0, 1, 511, 512, 513, 1023, 1024, 1025]
    args = inputs(lengths, k, width=1049, starts=[0, 7, 3, 4, 1, 9, 5, 11])
    # Storage padding and every column outside each window contain +inf.
    assert args[0].stride(0) > args[0].shape[1]
    original_scores = args[0].clone()
    call(*args)
    torch.testing.assert_close(args[0], original_scores, rtol=0, atol=0)
    assert_selection(*args, exact=True)
    for row, length in enumerate(lengths):
        assert torch.all(args[3][row, min(length, k):] == -1)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("rows,width", [(0, 17), (0, 0), (3, 0)])
def test_empty_rows_and_width(k, rows, width):
    args = inputs([0] * rows, k, width=width, padded=False)
    call(*args)
    assert args[3].shape == (rows, k)
    assert torch.all(args[3] == -1)


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_large_legal_output_offset(k):
    args = inputs([7], k, width=13, starts=[3])
    args[2].fill_(2**31 - 8)
    call(*args)
    assert_selection(*args, exact=True)


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_current_nondefault_stream(k):
    args = inputs([8192, 4096, k - 1, 0], k)
    stream = torch.cuda.Stream()
    ready = torch.cuda.Event()
    ready.record()
    stream.wait_event(ready)
    with torch.cuda.stream(stream):
        # Score updates are enqueued with the op; launching on the default stream can
        # race the new values and would fail this stream contract.
        for _ in range(49):
            args[0].neg_()
        call(*args)
        observed = args[3].clone()
    stream.synchronize()
    assert_selection(args[0], args[1], args[2], observed, args[4], exact=True)


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_graph_replay_updates_scores_lengths_and_starts(k):
    args = inputs([8192, 4096, k, 0], k, width=8211, starts=[0, 3, 5, 7])
    call(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*args)
    for lengths, starts, kind in (
        ([1, k - 1, k + 1, 0], [7, 2, 5, 0], "negative"),
        ([8192, k + 1, k, 3], [9, 11, 0, 5], "ties"),
        ([0, 17, 512, 1024], [0, 5, 3, 7], "negative_infinity"),
        ([8192, 4096, k - 1, 2], [1, 3, 5, 9], "positive"),
    ):
        replacement = inputs(lengths, k, width=8211, starts=starts, kind=kind)
        for destination, source in zip(args[:3], replacement[:3]):
            destination.copy_(source)
        args[4].copy_(replacement[4])
        args[2].add_(19)
        args[3].fill_(-777)
        graph.replay()
        assert_selection(*args, exact=kind in ("negative", "positive"))


def test_graph_replay_large_sampling_path():
    args = inputs([65536, 32768], 512, width=65539, starts=[1, 3])
    call(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*args)
    for lengths, kind in (([32768, 65536], "ties"),
                          ([65536, 32768], "negative"),
                          ([0, 511], "negative_infinity")):
        replacement = inputs(lengths, 512, width=65539, starts=[1, 3], kind=kind)
        args[0].copy_(replacement[0])
        args[1].copy_(replacement[1])
        args[3].fill_(-777)
        graph.replay()
        assert_selection(*args, exact=kind == "negative")


@pytest.mark.parametrize("bad", [
    "score_dtype", "score_dim", "score_cpu", "score_column_stride", "score_row_stride",
    "length_dtype", "length_dim", "length_shape", "length_cpu", "length_stride",
    "offset_dtype", "offset_dim", "offset_shape", "offset_cpu", "offset_stride",
    "start_dtype", "start_dim", "start_shape", "start_cpu", "start_stride",
    "out_dtype", "out_dim", "out_rows", "out_cpu", "out_stride", "out_row_stride",
])
def test_reject_invalid_tensor_contracts(bad):
    args = list(inputs([8192, 8192], 512, starts=[0, 0]))
    index = {"score": 0, "length": 1, "offset": 2, "out": 3, "start": 4}
    field, defect = bad.split("_", 1)
    i = index[field]
    tensor = args[i]
    if defect == "dtype":
        args[i] = tensor.to(torch.float16 if field == "score" else torch.int64)
    elif defect == "dim":
        args[i] = tensor.unsqueeze(0)
    elif defect == "cpu":
        args[i] = tensor.cpu()
    elif defect in ("shape", "rows"):
        args[i] = tensor[:1]
    elif defect == "column_stride":
        args[i] = torch.empty((2, 16384), device="cuda")[:, ::2]
    elif defect == "row_stride" and field == "score":
        args[i] = torch.empty((8192,), device="cuda").as_strided((2, 8192), (0, 1))
    elif defect == "row_stride":
        args[i] = torch.empty((2, 531), dtype=torch.int32, device="cuda")[:, :512]
    elif field == "out":
        args[i] = torch.empty((2, 1024), dtype=torch.int32, device="cuda")[:, ::2]
    else:
        args[i] = torch.empty((4,), dtype=torch.int32, device="cuda")[::2]
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        call(*args)


@pytest.mark.parametrize("k", [7, 256, 2049])
def test_reject_unsupported_topk(k):
    args = inputs([8192, 8192], k)
    with pytest.raises((RuntimeError, ValueError)):
        call(*args)


@pytest.mark.parametrize("alias", ["score", "length", "offset", "start"])
def test_reject_output_aliasing(alias):
    args = list(inputs([512, 512], 512, width=512, starts=[0, 0], padded=False))
    if alias == "score":
        args[3] = args[0].view(torch.int32)
    else:
        i = {"length": 1, "offset": 2, "start": 4}[alias]
        args[i] = args[3].view(-1)[:2]
        args[i].zero_()
    with pytest.raises((RuntimeError, ValueError)):
        call(*args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two devices required")
@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_reject_mixed_devices(index):
    args = list(inputs([8192, 8192], 512, starts=[0, 0]))
    args[index] = args[index].to("cuda:1")
    with pytest.raises((RuntimeError, ValueError)):
        call(*args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two devices required")
def test_score_device_differs_from_current_device():
    with torch.cuda.device(1):
        args = inputs([8192, 512], 512)
    with torch.cuda.device(0):
        call(*args)
        assert torch.cuda.current_device() == 0
    assert_selection(*args, exact=True)


def pytorch_unfused(scores, lengths, offsets, out, starts=None):
    """Materialize masking, TopK, remapping, and final int32 output."""
    rows, width = scores.shape
    out.fill_(-1)
    if rows == 0 or width == 0:
        return
    k = min(out.shape[1], width)
    positions = torch.arange(width, dtype=torch.int64, device=scores.device)[None, :]
    if starts is None:
        valid = positions < lengths[:, None]
    else:
        valid = (positions >= starts[:, None]) & (positions < (starts + lengths)[:, None])
    masked = scores.masked_fill(~valid, float("-inf"))
    selected = torch.topk(masked, k, dim=1).indices
    selected_valid = valid.gather(1, selected)
    local_indices = selected if starts is None else selected - starts[:, None]
    mapped = torch.where(selected_valid,
                         local_indices + offsets[:, None], -1)
    out[:, :k].copy_(mapped.to(torch.int32))


def measure(fn, warmup=10, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/HIP device is required")
    print(json.dumps({"device": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "hip": torch.version.hip,
                      "warmup": args.warmup, "repeats": args.repeats,
                      "bandwidth": "full score matrix + output/metadata normalization; "
                      "NOT measured HBM or actual traffic"}))
    shapes = [(512, width, 512) for width in (1024, 8192, 65536, 131072)]
    shapes.append((512, 8192, 1024))
    shapes.append((512, 8192, 2048))
    if args.profile_case:
        shapes = [tuple(int(value) for value in args.profile_case.split(","))]
        if len(shapes[0]) != 3:
            raise ValueError("--profile-case must be rows,width,k")
    for rows, width, k in shapes:
        if rows < 0 or width < 0 or k not in (512, 1024, 2048):
            raise ValueError("invalid benchmark shape")
        generator = torch.Generator(device="cuda").manual_seed(42)
        scores = torch.randn((rows, width), generator=generator, device="cuda")
        offsets = torch.arange(rows, dtype=torch.int32, device="cuda") * (width + 17) + 31
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        length_modes = ["full"]
        if not args.profile_case and k == 512 and width in (1024, 8192):
            length_modes.append("causal")
        for length_mode in length_modes:
            if length_mode == "full":
                lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            else:
                lengths = torch.div(
                    torch.arange(1, rows + 1, dtype=torch.int32, device="cuda") * width,
                    rows, rounding_mode="floor")
            case_args = scores, lengths, offsets, out
            cases = [("pytorch_unfused", lambda: pytorch_unfused(*case_args))]
            if not args.baseline_only:
                cases.append(("lightop_ragged_v2", lambda: call(*case_args)))
            # Correctness runs before timing, including the baseline-only path.
            for name, fn in cases:
                fn()
                assert_selection(*case_args)
                elapsed_us = measure(fn, args.warmup, args.repeats)
                logical_payload_bytes = rows * width * 4 + rows * k * 4 + rows * 12
                print(json.dumps({"case": name, "rows": rows, "width": width, "k": k,
                                  "length_mode": length_mode, "mean_us": elapsed_us,
                                  "logical_payload_bytes": logical_payload_bytes,
                                  "effective_GB_s": logical_payload_bytes /
                                  (elapsed_us * 1000)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--profile-case", help="single benchmark shape rows,width,k")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    cli_args, pytest_args = parser.parse_known_args()
    if cli_args.warmup < 0 or cli_args.repeats <= 0:
        parser.error("warmup must be nonnegative and repeats positive")
    if cli_args.benchmark or cli_args.baseline_only or cli_args.profile_case:
        benchmark(cli_args)
    else:
        sys.exit(pytest.main([__file__, *pytest_args]))
