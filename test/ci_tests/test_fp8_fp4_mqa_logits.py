# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Dense BF16 Q + packed FP4 K, tested through the installed Lightop API.

Run: python -m pytest -q test/ci_tests/test_fp8_fp4_mqa_logits.py
No JIT build or standalone-prototype dependency is used here.
"""

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
