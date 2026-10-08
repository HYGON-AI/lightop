# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Dense BF16 Q + packed FP4 K, tested through the installed Lightop API.

Run: python -m pytest -q test/ci_tests/test_fp8_fp4_mqa_logits.py
No JIT build or standalone-prototype dependency is used here.
"""

import argparse
import json
import os
import statistics

import pytest
import torch


@pytest.fixture(scope="module")
def logits_op():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a HIP GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip(f"BF16/FP4 MQA requires gfx936/gfx938, got {arch}")
    from lightop.attention import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits


def _case(t, h, length, width):
    rng = torch.Generator().manual_seed(20260924)
    q = torch.randn(t, h, 128, generator=rng).bfloat16()
    payload = torch.randint(256, (length, 64), generator=rng,
                            dtype=torch.uint8).view(torch.int8)
    scales = torch.randint(118, 137, (length, 4), generator=rng,
                           dtype=torch.uint8).view(torch.int32).reshape(length)
    weights = torch.randn(t, h, generator=rng)
    starts = [(row * 37) % (length + 1) for row in range(t)]
    ends = [start + min(length - start, width, (row * 29) % (width + 1))
            for row, start in enumerate(starts)]
    if t:
        ends[0] = min(length, width)
    if t > 1:
        ends[1] = starts[1]  # An empty window with nonempty K.
    return (q, payload, scales, weights,
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(ends, dtype=torch.int32))


def _reference(q, payload, scales, weights, starts, ends, width):
    # Independent E2M1 decode; do not call any Lightop decoder/kernel.
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    raw = payload.view(torch.uint8).long()
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(-1, 128)
    exponents = scales.view(torch.uint8).reshape(-1, 4).float() - 127
    k = (table[codes] * torch.exp2(exponents).repeat_interleave(32, 1))
    k = k.bfloat16().float()
    result = torch.full((q.shape[0], width), -torch.inf)
    for row in range(q.shape[0]):
        start, end = starts[row].item(), ends[row].item()
        dots = q[row].float() @ k[start:end].T
        result[row, :end - start] = (dots.relu() * weights[row, :, None]).sum(0)
    return result


def _call(op, tensors, width, **kwargs):
    q, payload, scales, weights, starts, ends = tensors
    args = dict(q=(q, None), kv=(payload, scales), weights=weights,
                cu_seq_len_k_start=starts, cu_seq_len_k_end=ends,
                clean_logits=False, max_seqlen_k=width)
    args.update(kwargs)
    return op(**args)


def _contiguous_offset_copy(source, offset):
    storage = torch.empty(source.numel() + offset, dtype=source.dtype,
                          device=source.device)
    view = storage[offset:].view_as(source)
    view.copy_(source)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


@pytest.mark.parametrize("t,h,length,width", [
    (1, 1, 1, 4), (33, 3, 257, 260), (64, 8, 256, 256),
    (32, 4, 65, 80), (17, 4, 256, 256), (17, 16, 65, 80),
    (5, 4, 0, 8), (0, 1, 0, 4), (0, 16, 513, 64),
    (65, 16, 1025, 260), (32, 4, 768, 128), (7, 2, 513, 4),
    (129, 32, 4097, 512), (33, 64, 513, 260),
])
def test_compressed_logits(logits_op, t, h, length, width):
    tensors = _case(t, h, length, width)
    expected = _reference(*tensors, width)
    actual = _call(logits_op, [x.cuda() for x in tensors], width)
    assert actual.dtype == torch.float32
    assert actual.shape == (t, width)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-2)


@pytest.mark.parametrize("ratio", [1, 2])
def test_packed_requests_and_compression_ratio(logits_op, ratio):
    # Multiple requests share K but use their own origins; W is smaller than L.
    width = 260
    tensors = list(_case(67, 16, 1031, width))
    starts = torch.tensor([3] * 33 + [517] * 34, dtype=torch.int32)
    lengths = torch.arange(67, dtype=torch.int32).remainder(33).add(1)
    lengths = (lengths * 7) // ratio
    tensors[-2:] = starts, starts + lengths
    expected = _reference(*tensors, width)
    actual = _call(logits_op, [x.cuda() for x in tensors], width)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-2)


def test_signed_weights_and_local_topk(logits_op):
    q = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    q[:, :, 0] = 1
    payload = torch.zeros(8, 64, dtype=torch.int8)
    payload[:, 0] = torch.arange(8, dtype=torch.int8)
    scales = torch.full((8, 4), 127, dtype=torch.uint8).view(torch.int32).reshape(8)
    weights = torch.tensor([[1., -2.], [1., 2.]])
    starts = torch.tensor([2, 3], dtype=torch.int32)
    ends = torch.tensor([7, 8], dtype=torch.int32)
    tensors = (q, payload, scales, weights, starts, ends)
    actual = _call(logits_op, [x.cuda() for x in tensors], 8).cpu()
    torch.testing.assert_close(actual, _reference(*tensors, 8), rtol=0, atol=0)
    # Negative head weights must not be clamped. The output index is local.
    assert (actual[0, :5] < 0).all()
    assert actual.topk(2, dim=-1).indices.tolist() == [[0, 1], [4, 3]]
    assert (actual.topk(2, dim=-1).indices + starts[:, None]).tolist() == [[2, 3], [7, 6]]


@pytest.mark.parametrize("invalid,match", [
    ("fp4_q", "only .*BF16 Q, None"),
    ("fp8_q", "only .*BF16 Q, None"),
    ("q_scale", "only .*BF16 Q, None"),
    ("q_shape", "q must be contiguous BF16"),
    ("q_strided", "q must be contiguous BF16"),
    ("cpu_q", "q must be contiguous BF16"),
    ("no_heads", "head count"),
    ("k_dtype", "K must be contiguous packed"),
    ("k_strided", "K must be contiguous packed"),
    ("scale_dtype", "K scale must be packed"),
    ("scale_shape", "K scale must be packed"),
    ("weights_dtype", "weights must be contiguous FP32"),
    ("weights_strided", "weights must be contiguous FP32"),
    ("starts_dtype", "ks/ke must be contiguous int32"),
    ("ends_shape", "ks/ke must be contiguous int32"),
    ("clean", "clean_logits=True is unsupported"),
    ("width_zero", "max_seqlen_k must be positive"),
    ("width_negative", "max_seqlen_k must be positive"),
    ("width_unaligned", "max_seqlen_k must be positive"),
    ("width_overflow", "max_seqlen_k must be positive"),
])
def test_reject_invalid_inputs(logits_op, invalid, match):
    tensors = [x.cuda() for x in _case(3, 2, 8, 8)]
    q, payload, scales, weights, starts, ends = tensors
    overrides = {
        "fp4_q": dict(q=(q.view(torch.int8), None)),
        "fp8_q": dict(q=(q.to(torch.float8_e4m3fn), None)),
        "q_scale": dict(q=(q, torch.ones(1, device="cuda"))),
        "q_shape": dict(q=(q[:, :, :64].contiguous(), None)),
        "q_strided": dict(q=(q.transpose(0, 1), None)),
        "cpu_q": dict(q=(q.cpu(), None)),
        "no_heads": dict(q=(q[:, :0].contiguous(), None)),
        "k_dtype": dict(kv=(payload.float(), scales)),
        "k_strided": dict(kv=(payload[::2], scales[::2])),
        "scale_dtype": dict(kv=(payload, scales.long())),
        "scale_shape": dict(kv=(payload, scales[:, None])),
        "weights_dtype": dict(weights=weights.bfloat16()),
        "weights_strided": dict(weights=weights.transpose(0, 1)),
        "starts_dtype": dict(cu_seq_len_k_start=starts.long()),
        "ends_shape": dict(cu_seq_len_k_end=ends[:1]),
        "clean": dict(clean_logits=True),
        "width_zero": dict(max_seqlen_k=0),
        "width_negative": dict(max_seqlen_k=-4),
        "width_unaligned": dict(max_seqlen_k=7),
        "width_overflow": dict(max_seqlen_k=1 << 31),
    }
    with pytest.raises(RuntimeError, match=match):
        _call(logits_op, tensors, 8, **overrides[invalid])


def test_accept_aligned_contiguous_storage_offsets(logits_op):
    tensors = _case(3, 2, 8, 8)
    expected = _reference(*tensors, 8)
    device_tensors = [x.cuda() for x in tensors]
    device_tensors[0] = _contiguous_offset_copy(device_tensors[0], 2)
    device_tensors[1] = _contiguous_offset_copy(device_tensors[1], 4)
    assert device_tensors[0].data_ptr() % 4 == 0
    assert device_tensors[1].data_ptr() % 4 == 0
    actual = _call(logits_op, device_tensors, 8)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-2)


def test_current_stream_and_graph_replay(logits_op):
    tensors = _case(33, 4, 257, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        device_tensors = [x.cuda() for x in tensors]
        actual = _call(logits_op, device_tensors, 128)
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), _reference(*tensors, 128),
                               rtol=3e-3, atol=3e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _call(logits_op, device_tensors, 128)
    # A replay must consume new data, not a result cached during capture.
    device_tensors[3].mul_(0.5)
    graph.replay()
    expected_inputs = list(tensors)
    expected_inputs[3] = tensors[3] * 0.5
    torch.testing.assert_close(captured.cpu(), _reference(*expected_inputs, 128),
                               rtol=3e-3, atol=3e-2)


def test_large_bf16_graph_replay_with_q_and_k_scale_updates(logits_op):
    # T*L previously triggered the optional FP8 path. The BF16-only kernel
    # must use the current Q and K scales on every graph replay.
    tensors = list(_case(32, 8, 1024, 512))
    rng = torch.Generator().manual_seed(41)
    tensors[0] = (torch.randint(-8, 9, (32, 8, 128), generator=rng)
                  .float() * 0.5).bfloat16()
    tensors[2] = torch.randint(123, 131, (1024, 4), generator=rng,
                               dtype=torch.uint8).view(torch.int32).reshape(1024)
    tensors[4] = torch.zeros(32, dtype=torch.int32)
    tensors[5] = torch.full((32,), 512, dtype=torch.int32)
    device_tensors = [x.cuda() for x in tensors]
    actual = _call(logits_op, device_tensors, 512)
    torch.testing.assert_close(actual.cpu(), _reference(*tensors, 512),
                               rtol=3e-3, atol=3e-2)

    # Change Q in place and verify replay consumes the new BF16 value.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(logits_op, device_tensors, 512)
    device_tensors[0][0, 0, 0] = 1.0078125
    graph.replay()
    changed = list(tensors)
    changed[0] = tensors[0].clone()
    changed[0][0, 0, 0] = 1.0078125
    torch.testing.assert_close(captured.cpu(), _reference(*changed, 512),
                               rtol=3e-3, atol=3e-2)

    # Restore Q, then change one K scale byte in the first tile.
    device_tensors[0][0, 0, 0] = tensors[0][0, 0, 0]
    device_tensors[2].view(torch.uint8).reshape(-1, 4)[0, 0] = 110
    graph.replay()
    changed = list(tensors)
    changed[2] = tensors[2].clone()
    changed[2].view(torch.uint8).reshape(-1, 4)[0, 0] = 110
    torch.testing.assert_close(captured.cpu(), _reference(*changed, 512),
                               rtol=3e-3, atol=3e-2)


def test_bf16_onehot_matrix_layout(logits_op):
    # Every K row has exactly one nonzero channel. This catches the 16-byte
    # rotation of the K LDS layout and MMAC register hazards that constant
    # Q/K values can mask.
    t, h, length = 32, 8, 1024
    q = torch.zeros(t, h, 128, dtype=torch.bfloat16)
    payload = torch.zeros(length, 64, dtype=torch.int8)
    for row in range(t):
        q[row, 0, row] = 1
    for col in range(length):
        channel = col % 128
        payload[col, channel // 2] = 2 if channel % 2 == 0 else 32
    scales = torch.full((length, 4), 127, dtype=torch.uint8).view(torch.int32).reshape(length)
    weights = torch.zeros(t, h)
    weights[:, 0] = 1
    starts = torch.zeros(t, dtype=torch.int32)
    ends = torch.full((t,), length, dtype=torch.int32)
    actual = _call(logits_op, [x.cuda() for x in
                               (q, payload, scales, weights, starts, ends)], length).cpu()
    expected = (torch.arange(t)[:, None] ==
                torch.arange(length)[None, :].remainder(128)).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("t,h,length,width", [
    (33, 8, 1025, 260), (64, 16, 512, 260), (65, 32, 513, 260),
])
def test_large_bf16_tails_and_compressed_windows(
        logits_op, t, h, length, width):
    tensors = list(_case(t, h, length, width))
    rng = torch.Generator().manual_seed(42)
    tensors[0] = (torch.randint(-8, 9, (t, h, 128), generator=rng)
                  .float() * 0.5).bfloat16()
    tensors[2] = torch.randint(123, 131, (length, 4), generator=rng,
                               dtype=torch.uint8).view(torch.int32).reshape(length)
    actual = _call(logits_op, [x.cuda() for x in tensors], width)
    torch.testing.assert_close(actual.cpu(), _reference(*tensors, width),
                               rtol=3e-3, atol=3e-2)


def test_public_exports(logits_op):
    import lightop
    from lightop.attention.mqa import fp8_fp4_mqa_logits
    from lightop.gemmopt import fp8_fp4_mqa_logits as gemmopt_fp8_fp4_mqa_logits

    assert logits_op is getattr(lightop, "fp8_fp4_mqa_logits")
    assert logits_op is fp8_fp4_mqa_logits
    assert logits_op is gemmopt_fp8_fp4_mqa_logits
    assert callable(getattr(lightop.op, "fp8_fp4_mqa_logits"))


def test_stale_extension_has_actionable_error(logits_op, monkeypatch):
    import lightop

    monkeypatch.delattr(lightop.op, "fp8_fp4_mqa_logits")
    with pytest.raises(RuntimeError, match="rebuild/install Lightop"):
        logits_op((None, None), (None, None), None, None, None, False, 4)


@pytest.fixture(scope="module")
def candidate_logits_op(logits_op):
    assert getattr(logits_op, "supports_candidate_blocks", False), (
        "rebuild/install LightOp with candidate_blocks support")
    return logits_op


def _candidate_case(query_lens, key_lens, width=None):
    width = width or max(4, (max(key_lens, default=0) + 3) // 4 * 4)
    t = sum(query_lens)
    tensors = list(_case(t, 32, 17 + sum(key_lens) + 11 * len(key_lens), width))
    rng = torch.Generator().manual_seed(20261003)
    tensors[2] = torch.randint(124, 131, (tensors[1].shape[0], 4),
                               generator=rng, dtype=torch.uint8).view(torch.int32).reshape(-1)
    tensors[3] *= .01
    starts, ends, origin = [], [], 17
    for rows, length in zip(query_lens, key_lens):
        starts.extend([origin] * rows)
        ends.extend(origin + min(length, (row * 17 + 1) % (length + 1))
                    for row in range(rows))
        origin += length + 11
    tensors[-2:] = (torch.tensor(starts, dtype=torch.int32),
                   torch.tensor(ends, dtype=torch.int32))
    return [x.cuda() for x in tensors], width


def _candidate_tables(query_lens, key_lens, k, strided=False):
    rng = torch.Generator().manual_seed(20261004)
    tables = []
    for rows, length in zip(query_lens, key_lens):
        storage = torch.full((rows + 2, k + 5 if strided else k), -1,
                             dtype=torch.int32)
        for row in range(rows):
            n = min(k, (length + 7) // 8)
            storage[row + 1, :n] = torch.randperm((length + 7) // 8,
                                                  generator=rng)[:n].int()
        storage = storage.cuda()
        tables.append(storage[1:rows + 1, :k])
    return tables


def _candidate_mask(tensors, width, blocks, query_lens):
    # Build the oracle from request-relative block ids, independently of the
    # operator's request/row map. Invalid ids scatter into one discarded column.
    starts, ends = tensors[-2:]
    masks, row_start = [], 0
    for table, rows in zip(blocks, query_lens):
        mask = torch.zeros(rows, width + 1, dtype=torch.bool, device=starts.device)
        if rows and table.shape[1]:
            columns = table.long()[:, :, None] * 8 + torch.arange(
                8, device=starts.device)[None, None, :]
            visible = ends[row_start:row_start + rows] - starts[row_start:row_start + rows]
            valid = ((table[:, :, None] >= 0) & (columns < width)
                     & (columns < visible[:, None, None]))
            destinations = torch.where(valid, columns, width).flatten(1)
            mask.scatter_(1, destinations, True)
        masks.append(mask[:, :width])
        row_start += rows
    return torch.cat(masks) if masks else torch.zeros(
        0, width, dtype=torch.bool, device=starts.device)


def _candidate_call(op, tensors, width, blocks, query_lens):
    return _call(op, tensors, width, candidate_blocks=blocks,
                 candidate_query_lens=query_lens)


def _assert_candidate_close(op, tensors, width, blocks, query_lens, actual=None):
    mask = _candidate_mask(tensors, width, blocks, query_lens)
    expected = _call(op, tensors, width).masked_fill_(~mask, -torch.inf)
    if actual is None:
        actual = _candidate_call(op, tensors, width, blocks, query_lens)
    assert actual.shape == expected.shape and actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.isneginf(actual[~mask]).all()
    return expected, actual


@pytest.mark.parametrize("query_lens,key_lens,k", [
    ([33], [257], 5), ([3, 0, 4], [17, 0, 269], 7),
    ([1, 2, 0, 9], [9, 511, 23, 513], 3), ([65], [4097], 17),
    ([0], [32], 5),
])
def test_candidate_blocks_match_dense(candidate_logits_op, query_lens, key_lens, k):
    tensors, width = _candidate_case(query_lens, key_lens)
    blocks = _candidate_tables(query_lens, key_lens, k)
    _assert_candidate_close(candidate_logits_op, tensors, width, blocks, query_lens)


def test_candidate_strided_tail_tables_and_per_row_origins(candidate_logits_op):
    query_lens, key_lens = [3, 2], [127, 259]
    tensors, width = _candidate_case(query_lens, key_lens)
    # A candidate block remains local to each row even when ks differs within a request.
    shifts = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int32, device="cuda")
    tensors[-2] += shifts
    tensors[-1] += shifts
    blocks = _candidate_tables(query_lens, key_lens, 5, strided=True)
    for table in blocks:
        assert table.stride(0) > table.shape[1] and table.stride(1) == 1
        assert table.storage_offset() > 0
    _assert_candidate_close(candidate_logits_op, tensors, width, blocks, query_lens)


@pytest.mark.parametrize("mode", ["empty", "invalid", "tail", "fullcover"])
def test_candidate_empty_invalid_and_causal_tails(candidate_logits_op, mode):
    lengths = [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 129]
    query_lens = [len(lengths)]
    tensors, width = _candidate_case(query_lens, [129])
    tensors[-1] = tensors[-2] + torch.tensor(lengths, dtype=torch.int32, device="cuda")
    if mode == "empty":
        table = torch.empty(len(lengths), 0, dtype=torch.int32, device="cuda")
    elif mode == "invalid":
        table = torch.tensor([-1, -2, torch.iinfo(torch.int32).max],
                             dtype=torch.int32, device="cuda").repeat(len(lengths), 1)
    elif mode == "tail":
        last = (torch.tensor(lengths, dtype=torch.int32, device="cuda") - 1) // 8
        # A partial newest block contains future columns; a distant block must be ignored.
        table = torch.stack((last, torch.full_like(last, width // 8 + 9)), dim=1)
    else:
        table = torch.arange((width + 7) // 8, dtype=torch.int32,
                             device="cuda").repeat(len(lengths), 1)
    expected, actual = _assert_candidate_close(
        candidate_logits_op, tensors, width, [table], query_lens)
    if mode in ("empty", "invalid"):
        assert torch.isneginf(actual).all()
    assert torch.isneginf(expected[0]).all()


@pytest.mark.parametrize("length", [0, 9])
def test_candidate_key_array_and_window_bounds(candidate_logits_op, length):
    tensors = list(_case(4, 32, length, 16))
    tensors[-2] = torch.tensor([0, max(0, length - 3), length, length], dtype=torch.int32)
    tensors[-1] = torch.full((4,), length, dtype=torch.int32)
    tensors = [x.cuda() for x in tensors]
    table = torch.tensor([0, 1, torch.iinfo(torch.int32).max, -1],
                         dtype=torch.int32, device="cuda").repeat(4, 1)
    _assert_candidate_close(candidate_logits_op, tensors, 16, [table], [4])


def _valid_topk(logits, k):
    count = min(k, logits.shape[1])
    ranked = logits.topk(count, dim=-1).indices
    valid = torch.isfinite(logits.gather(1, ranked))
    ordered = torch.where(valid, ranked, logits.shape[1]).sort(dim=-1).values
    result = torch.full((logits.shape[0], k), -1, dtype=torch.int64, device=logits.device)
    result[:, :count] = torch.where(ordered < logits.shape[1], ordered, -1)
    return result


@pytest.mark.parametrize("k", [1, 7, 64, 511, 512])
def test_candidate_exact_no_tie_topk_and_underfill(candidate_logits_op, k):
    # Exactly representable integer scores: K row j encodes j in 12 binary channels.
    t, width, length = 2, 1024, 2080
    q = torch.zeros(t, 32, 128, dtype=torch.bfloat16)
    q[:, 0, :12] = torch.exp2(torch.arange(12).float()).bfloat16()
    bits = (torch.arange(length)[:, None] >> torch.arange(12)[None, :]) & 1
    payload = torch.zeros(length, 64, dtype=torch.uint8)
    payload[:, :6] = (bits[:, 0::2] * 2 + bits[:, 1::2] * 32).byte()
    scales = torch.full((length, 4), 127, dtype=torch.uint8).view(torch.int32).reshape(-1)
    weights = torch.zeros(t, 32)
    weights[:, 0] = 1
    starts = torch.tensor([17, 1033], dtype=torch.int32)
    ends = starts + torch.tensor([997, 1001], dtype=torch.int32)
    tensors = [x.cuda() for x in (q, payload.view(torch.int8), scales, weights, starts, ends)]
    table = torch.full((t, 128), -1, dtype=torch.int32, device="cuda")
    table[0] = torch.arange(128, dtype=torch.int32, device="cuda")
    table[1, :16] = torch.arange(64, 80, dtype=torch.int32, device="cuda")
    expected, actual = _assert_candidate_close(
        candidate_logits_op, tensors, width, [table], [t])
    expected_indices, actual_indices = _valid_topk(expected, k), _valid_topk(actual, k)
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(torch.where(actual_indices >= 0, actual_indices + tensors[-2][:, None], -1),
                       torch.where(expected_indices >= 0, expected_indices + tensors[-2][:, None], -1))
    assert (actual_indices[1] >= 0).sum().item() == min(k, 128)


def test_candidate_mixed_fp4_scale_boundaries(candidate_logits_op):
    t, length = 33, 176
    q = torch.ones(t, 32, 128, dtype=torch.bfloat16)
    codes = torch.arange(length)[:, None].add(torch.arange(64)[None, :]).remainder(16)
    payload = (codes | ((codes + 1).remainder(16) << 4)).byte().view(torch.int8)
    limits = torch.tensor([0, 1, 2, 126, 127, 128, 251, 252, 253, 254, 255], dtype=torch.uint8)
    scales = limits[(torch.arange(length)[:, None] + torch.arange(4)[None, :]) % len(limits)]
    scales = scales.contiguous().view(torch.int32).reshape(length)
    weights = torch.ones(t, 32)
    starts = torch.arange(t, dtype=torch.int32).mul(5).remainder(length)
    ends = torch.full((t,), length, dtype=torch.int32)
    tensors = [x.cuda() for x in (q, payload, scales, weights, starts, ends)]
    blocks = [torch.arange(22, dtype=torch.int32, device="cuda").repeat(t, 1)]
    mask = _candidate_mask(tensors, length, blocks, [t])
    expected = _call(candidate_logits_op, tensors, length).masked_fill_(~mask, -torch.inf)
    actual = _candidate_call(candidate_logits_op, tensors, length, blocks, [t])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True)
    assert torch.isneginf(actual[~mask]).all()


def test_candidate_all_heads_onehot_exact(candidate_logits_op):
    t, width, origin, length = 32, 128, 17, 145
    q = torch.zeros(t, 32, 128, dtype=torch.bfloat16)
    for head in range(32):
        q[:, head, head] = 1
    payload = torch.zeros(length, 64, dtype=torch.uint8)
    for key in range(length):
        channel = key % 128
        payload[key, channel // 2] = 2 << (4 * (channel % 2))
    scales = torch.full((length, 4), 127, dtype=torch.uint8).view(torch.int32).reshape(-1)
    signed = torch.arange(1, 33).float() * torch.where(torch.arange(32) % 2 == 0, 1, -1)
    weights = torch.diag(signed)
    starts = torch.full((t,), origin, dtype=torch.int32)
    ends = starts + width
    tensors = [x.cuda() for x in (q, payload.view(torch.int8), scales, weights, starts, ends)]
    blocks = [torch.arange(16, dtype=torch.int32, device="cuda").repeat(t, 1)]
    expected = (((torch.arange(width) + origin) % 128)[None, :] == torch.arange(t)[:, None]).float()
    expected *= signed[:, None]
    for actual in (_call(candidate_logits_op, tensors, width),
                   _candidate_call(candidate_logits_op, tensors, width, blocks, [t])):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [
    "missing_lens", "missing_blocks", "dtype", "cpu", "ndim", "rows",
    "column_stride", "request_count", "lens_sum", "negative_lens", "heads",
])
def test_candidate_reject_invalid_inputs(candidate_logits_op, invalid):
    tensors, width = _candidate_case([3], [65])
    blocks = _candidate_tables([3], [65], 3)
    query_lens = [3]
    if invalid == "missing_lens":
        query_lens = None
    elif invalid == "missing_blocks":
        blocks = None
    elif invalid == "dtype":
        blocks = [blocks[0].long()]
    elif invalid == "cpu":
        blocks = [blocks[0].cpu()]
    elif invalid == "ndim":
        blocks = [blocks[0][0]]
    elif invalid == "rows":
        blocks = [blocks[0][:-1]]
    elif invalid == "column_stride":
        blocks = [torch.zeros(3, 6, dtype=torch.int32, device="cuda")[:, ::2]]
    elif invalid == "request_count":
        blocks = blocks * 2
    elif invalid == "lens_sum":
        query_lens = [2]
    elif invalid == "negative_lens":
        query_lens = [-1]
    else:
        tensors[0] = tensors[0][:, :16].contiguous()
        tensors[3] = tensors[3][:, :16].contiguous()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _candidate_call(candidate_logits_op, tensors, width, blocks, query_lens)


def test_candidate_nondefault_stream_and_graph_block_updates(candidate_logits_op):
    query_lens, key_lens = [33], [1025]
    tensors, width = _candidate_case(query_lens, key_lens)
    blocks = _candidate_tables(query_lens, key_lens, 16, strided=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actual = _candidate_call(candidate_logits_op, tensors, width, blocks, query_lens)
    stream.synchronize()
    _assert_candidate_close(candidate_logits_op, tensors, width, blocks, query_lens, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _candidate_call(candidate_logits_op, tensors, width, blocks, query_lens)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    _assert_candidate_close(candidate_logits_op, tensors, width, blocks, query_lens, captured)
    before = captured.clone()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        blocks[0].add_(1).remainder_((key_lens[0] + 7) // 8)
        graph.replay()
    stream.synchronize()
    _assert_candidate_close(candidate_logits_op, tensors, width, blocks, query_lens, captured)
    assert not torch.equal(before, captured)


def _timing(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(100):
        begin.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    return dict(mean_us=statistics.mean(samples), p50_us=statistics.median(samples),
                p90_us=sorted(samples)[90], stddev_us=statistics.pstdev(samples),
                warmup=10, repeats=100)


def _benchmark(profile_shape, profile_mode):
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("benchmark requires a HIP GPU")
    from lightop.attention import fp8_fp4_mqa_logits as op
    assert getattr(op, "supports_candidate_blocks", False), "rebuild/install candidate_blocks support"
    props = torch.cuda.get_device_properties(0)
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip,
                          device=props.name, gfx=props.gcnArchName,
                          hip_visible_devices=os.getenv("HIP_VISIBLE_DEVICES")), sort_keys=True), flush=True)
    shapes = {"4K": 4096, "32K": 32768, "64K": 65536, "128K": 131072}
    lengths = shapes.values() if profile_shape == "all" else [shapes[profile_shape]]
    for length in lengths:
        t, h = 512, 32
        tensors, _ = _candidate_case([t], [length], width=length)
        tensors[1], tensors[2] = tensors[1][:length], tensors[2][:length]
        tensors[-2].zero_()
        tensors[-1] = length - 4096 + torch.arange(t, dtype=torch.int32, device="cuda") * 8 + 1
        blocks = _candidate_tables([t], [length], 2048)
        mask = _candidate_mask(tensors, length, blocks, [t])
        invalid = ~mask
        _assert_candidate_close(op, tensors, length, blocks, [t])
        # Match framework work on BOTH sides: original in-place candidate mask
        # remains after the operator; no clone on just the dense side.
        operations = {
            "dense_masked": lambda: _call(op, tensors, length).masked_fill_(invalid, -torch.inf),
            "candidate_masked": lambda: _candidate_call(op, tensors, length, blocks, [t]).masked_fill_(invalid, -torch.inf),
        }
        modes = operations if profile_mode == "both" else [profile_mode + "_masked"]
        for mode in modes:
            result = _timing(operations[mode])
            # A common minimum tensor-IO footprint, NOT measured DRAM traffic:
            # input footprint + logits initialization + in-place mask read/write.
            footprint = sum(x.numel() * x.element_size() for x in tensors)
            footprint += 3 * t * length * 4 + invalid.numel()
            if mode == "candidate_masked":
                footprint += sum(x.numel() * x.element_size() for x in blocks)
            result.update(mode=mode, t=t, h=h, length=length, candidate_blocks=2048,
                          block_size=8, dtype="BF16-Q/FP4-K/FP32-W",
                          correctness="dense+expanded-causal-mask rtol=atol=1e-5 passed",
                          nominal_tensor_io_bytes=footprint,
                          nominal_effective_gb_s=footprint / (result["mean_us"] * 1000),
                          bandwidth_note="minimum logical tensor footprint; not measured DRAM bandwidth")
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile-shape", choices=("4K", "32K", "64K", "128K", "all"), default="64K")
    parser.add_argument("--profile-mode", choices=("both", "dense", "candidate"), default="both")
    args, pytest_args = parser.parse_known_args()
    if args.benchmark:
        _benchmark(args.profile_shape, args.profile_mode)
    else:
        raise SystemExit(pytest.main([__file__, *pytest_args]))
