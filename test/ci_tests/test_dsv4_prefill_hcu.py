# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""LightOp DSV4 Prefill parity tests and warmed, synchronized Triton comparison."""
import argparse
import json
import os
import statistics
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import triton
from lightop import dequantize_k_cache_paged_hcu, combine_topk_swa_indices_hcu

def _lightop_dequant(cache, ids, page_size, out=None, fnuz=False):
    if out is None:
        out = torch.empty((ids.numel(), 1, 512), dtype=torch.bfloat16, device=cache.device)
    dequantize_k_cache_paged_hcu(cache.view(torch.uint8), ids, out, page_size, fnuz)
    return out


def _dequant_reference(cache, ids, page_size, fnuz=False):
    cache = cache.view(torch.uint8)
    pages = ids.long() // page_size
    offsets = ids.long() % page_size
    data = cache[pages[:, None], offsets[:, None] * 576 + torch.arange(576, device=cache.device)]
    codes = cache[pages[:, None], page_size * 576 + offsets[:, None] * 8 + torch.arange(7, device=cache.device)]
    scales = torch.exp2(codes.float() - 127)
    scales = torch.where(codes == 0, 0, scales)
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    nope = data[:, :448].contiguous().view(dtype).float().reshape(-1, 7, 64)
    nope = (nope * scales[:, :, None]).reshape(ids.numel(), 448).bfloat16()
    rope = data[:, 448:].contiguous().view(torch.bfloat16)
    return torch.cat((nope, rope), dim=1).unsqueeze(1)


def _triton_dequant(cache, ids, page_size, out=None):
    import sglang.kernels.ops.attention.dsv4.dequant_k_cache as dequant

    with patch.object(dequant, '_use_hcu_dequant_k_cache_paged', False, create=True):
        return dequant.dequantize_k_cache_paged(cache, ids, page_size, out=out)


@pytest.fixture
def sglang_dequant():
    pytest.importorskip('sglang', reason='optional SGLang Triton comparison')
    import sglang.kernels.ops.attention.dsv4.dequant_k_cache as dequant

    return dequant


@pytest.fixture
def sglang_sparse_utils():
    pytest.importorskip('sglang', reason='optional SGLang HCU routing test')
    import sglang.srt.layers.attention.dsv4.sparse_prefill_utils as sparse_utils

    return sparse_utils


def combine_topk_swa_indices(topk_indices, query_start_loc, query_pos, seq_lens, gather_lens, compressed_base, swa_base, window_size, compress_ratio, topk, out_indices=None, out_lens=None):
    rows = topk_indices.size(0)
    width = ((topk + window_size + 127) // 128) * 128
    if out_indices is None:
        out_indices = torch.full((rows, width), -1, dtype=torch.int32, device=topk_indices.device)
    if out_lens is None:
        out_lens = torch.empty(rows, dtype=torch.int32, device=topk_indices.device)
    combine_topk_swa_indices_hcu(topk_indices, query_start_loc, query_pos, seq_lens, gather_lens, compressed_base, swa_base, out_indices, out_lens, window_size, compress_ratio, topk)
    return out_indices, out_lens


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HCU packed V4 KV LightOp kernel"
)


def _case(page_size=64, num_tokens=333, index_dtype=torch.int32, device="cuda"):
    pages = max(8, -(-num_tokens // page_size))
    page_bytes = -(-(page_size * 584) // 576) * 576
    cache = torch.randint(0, 256, (pages, page_bytes), device=device, dtype=torch.uint8)
    ids = torch.randint(0, pages * page_size, (num_tokens,), device=device, dtype=index_dtype)
    return cache, ids


@pytest.mark.parametrize("fnuz", [False, True])
@pytest.mark.parametrize("page_size", [1, 2, 16, 64, 256])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [0, 1, 3, 333])
def test_matches_torch(fnuz, page_size, index_dtype, rows):
    cache, ids = _case(page_size, rows, index_dtype)
    storage = torch.full((rows * 2 + 1, 1, 521), -77, device="cuda", dtype=torch.bfloat16)
    out = storage[: rows * 2 : 2, :, :512]
    expected = _dequant_reference(cache, ids, page_size, fnuz)
    actual = _lightop_dequant(cache.view(torch.bfloat16), ids, page_size, out=out, fnuz=fnuz)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    # RoPE is copied bit-for-bit, including NaNs and signed zero.
    assert torch.equal(actual[:, :, 448:].view(torch.int16), expected[:, :, 448:].view(torch.int16))
    assert (storage[1::2] == -77).all()
    assert (storage[:, :, 512:] == -77).all()


@pytest.mark.parametrize("page_size", [1, 2, 16, 64, 256])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [0, 1, 3, 333])
def test_matches_triton(sglang_dequant, page_size, index_dtype, rows):
    cache, ids = _case(page_size, rows, index_dtype)
    fnuz = sglang_dequant.fp8_dtype == torch.float8_e4m3fnuz
    actual = _lightop_dequant(cache, ids, page_size, fnuz=fnuz)
    expected = _triton_dequant(cache, ids, page_size)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("fnuz", [False, True])
def test_all_fp8_codes_and_scale_boundaries(fnuz):
    page_size = 256
    cache, ids = _case(page_size, 256)
    data = cache[0, :page_size * 576].view(page_size, 576)
    data[:, :448] = torch.arange(256, device="cuda", dtype=torch.uint8)[:, None]
    scales = cache[0, page_size * 576 : page_size * 584].view(page_size, 8)
    scales[:, :7] = torch.tensor([0, 1, 119, 127, 135, 254, 255], dtype=torch.uint8, device="cuda")
    ids.copy_(torch.arange(256, dtype=ids.dtype, device="cuda"))
    got = _lightop_dequant(cache, ids, page_size, fnuz=fnuz)
    expected = _dequant_reference(cache, ids, page_size, fnuz)
    torch.testing.assert_close(got, expected, atol=0, rtol=0, equal_nan=True)


def test_current_stream_graph_and_duplicate_ids():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cache, ids = _case()
        ids[::2] = 0
        expected = _dequant_reference(cache, ids, 64)
        out = torch.empty_like(expected)
        _lightop_dequant(cache, ids, 64, out=out)
    stream.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _lightop_dequant(cache, ids, 64, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two HCU devices")
def test_tensor_device_not_current_device():
    stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream):
        cache, ids = _case(device="cuda:1")
        expected = _dequant_reference(cache, ids, 64)
        out = torch.empty_like(expected)
    with torch.cuda.stream(stream), torch.cuda.device(0):
        _lightop_dequant(cache, ids, 64, out=out)
        assert torch.cuda.current_device() == 0
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("bad", ["dtype", "id_stride", "output_stride", "page_size", "device"])
def test_invalid_metadata_rejected(bad):
    cache, ids = _case()
    out = torch.empty((ids.numel(), 1, 512), device="cuda", dtype=torch.bfloat16)
    page_size = 64
    if bad == "dtype":
        ids = ids.float()
    elif bad == "id_stride":
        ids = ids.repeat_interleave(2)[::2]
    elif bad == "output_stride":
        out = torch.empty((ids.numel(), 1, 1024), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    elif bad == "page_size":
        page_size = 0
    else:
        ids = ids.cpu()
    with pytest.raises(RuntimeError):
        dequantize_k_cache_paged_hcu(cache, ids, out, page_size, False)



DEVICE = "cuda"
WINDOW = 128
# flash_mla_sparse_fwd reads rows in 128-wide tiles; the combined width is
# padded to that multiple.
TOPK_ALIGNMENT = 128


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def _reference(
    topk_indices,
    query_start_loc,
    query_pos,
    seq_lens,
    gather_lens,
    compressed_base,
    swa_base,
    window_size,
    compress_ratio,
    topk,
):
    num_tokens = topk_indices.shape[0]
    width = -(-(topk + window_size) // TOPK_ALIGNMENT) * TOPK_ALIGNMENT
    out = torch.full((num_tokens, width), -1, dtype=torch.int32, device=DEVICE)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    qsl = query_start_loc.tolist()
    base = qsl[0]
    for r in range(seq_lens.shape[0]):
        gather_start = int(seq_lens[r]) - int(gather_lens[r])
        for token_idx in range(qsl[r] - base, qsl[r + 1] - base):
            pos = int(query_pos[token_idx])
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            vals = topk_indices[token_idx, :topk_len]
            out[token_idx, :topk_len] = torch.where(
                vals >= 0, vals + compressed_base[r], torch.full_like(vals, -1)
            )
            window = torch.arange(swa_len, dtype=torch.int32, device=DEVICE)
            out[token_idx, topk_len : topk_len + swa_len] = (
                swa_base[r] + window + pos - swa_len + 1 - gather_start
            )
            lens[token_idx] = topk_len + swa_len
    return out, lens


def _check(**kwargs):
    got_idx, got_lens = combine_topk_swa_indices(**kwargs)
    ref_idx, ref_lens = _reference(**kwargs)
    assert torch.equal(got_lens, ref_lens), (got_lens.tolist(), ref_lens.tolist())
    assert torch.equal(got_idx, ref_idx)


def _trailing_case(seq_lens, extend_lens, topk, compress_ratio, seed=0):
    """Rows are the trailing ``extend_lens[r]`` tokens of request ``r``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    query_pos = []
    for seq_len, extend_len in zip(seq_lens, extend_lens):
        query_pos.extend(range(seq_len - extend_len, seq_len))
    num_tokens = len(query_pos)
    topk_indices = torch.randint(
        0, 1 << 20, (num_tokens, topk), generator=gen, dtype=torch.int32
    ).to(DEVICE)
    starts = [0]
    for extend_len in extend_lens:
        starts.append(starts[-1] + extend_len)
    gather_lens = [min(s, e + WINDOW - 1) for s, e in zip(seq_lens, extend_lens)]
    return dict(
        topk_indices=topk_indices,
        query_start_loc=_i32(starts),
        query_pos=_i32(query_pos),
        seq_lens=_i32(seq_lens),
        gather_lens=_i32(gather_lens),
        compressed_base=_i32([1000 * r for r in range(len(seq_lens))]),
        swa_base=_i32([5000 + 300 * r for r in range(len(seq_lens))]),
        window_size=WINDOW,
        compress_ratio=compress_ratio,
        topk=topk,
    )


@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize(
    "seq_lens, extend_lens",
    [([96, 144], [3, 2]), ([7, 300, 1000], [7, 130, 5]), ([1], [1])],
)
def test_trailing_extend_matches_reference(seq_lens, extend_lens, compress_ratio):
    _check(
        **_trailing_case(seq_lens, extend_lens, topk=64, compress_ratio=compress_ratio)
    )


def test_negative_one_holes_are_kept():
    case = _trailing_case([512, 640], [4, 4], topk=64, compress_ratio=4)
    topk_indices = case["topk_indices"]
    # Holes inside the scanned prefix (every row scans the full top-k here).
    topk_indices[:, 0] = -1
    topk_indices[:, 5] = -1
    topk_indices[3, 10:20] = -1
    _check(**case)
    got_idx, got_lens = combine_topk_swa_indices(**case)
    assert (got_idx[:, 0] == -1).all()
    assert (got_idx[:, 5] == -1).all()
    assert (got_idx[3, 10:20] == -1).all()
    # The scanned prefix still counts the holes.
    assert int(got_lens[0]) == 64 + WINDOW


def test_non_trailing_query_positions():
    # Two requests; each rank of a two-way interleave holds every other row of
    # the extend, so the query positions are not the trailing tokens. The
    # query_start_loc carries a cross-chunk offset that the kernel rebases.
    seq_lens = [96, 144]
    extend_lens = [6, 4]
    query_pos = [90, 92, 94, 140, 142]
    starts = [10, 13, 15]
    num_tokens = len(query_pos)
    topk = 32
    topk_indices = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=DEVICE
    ).view(num_tokens, topk)
    gather_lens = [min(s, e + WINDOW - 1) for s, e in zip(seq_lens, extend_lens)]
    _check(
        topk_indices=topk_indices,
        query_start_loc=_i32(starts),
        query_pos=_i32(query_pos),
        seq_lens=_i32(seq_lens),
        gather_lens=_i32(gather_lens),
        compressed_base=_i32([0, 24]),
        swa_base=_i32([48, 181]),
        window_size=WINDOW,
        compress_ratio=4,
        topk=topk,
    )


def test_swa_only_layer():
    case = _trailing_case([200, 50], [2, 2], topk=0, compress_ratio=4)
    case["topk_indices"] = torch.zeros((4, 1), dtype=torch.int32, device=DEVICE)
    _check(**case)
    _, got_lens = combine_topk_swa_indices(**case)
    assert got_lens.tolist() == [WINDOW, WINDOW, 49, 50]


@pytest.mark.parametrize("topk", [0, 1, 31, 64, 512, 2048])
@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_ragged_empty_requests_and_strided_rows(topk, compress_ratio):
    case = _trailing_case([7, 9, 600, 3], [0, 9, 7, 0], topk, compress_ratio)
    case["query_start_loc"] += 123
    storage = torch.full((32, topk + 7), -99, dtype=torch.int32, device=DEVICE)
    view = storage[::2, :topk]
    view.copy_(case["topk_indices"])
    view[:, ::5] = -1
    case["topk_indices"] = view
    _check(**case)


def test_preallocated_output_preserves_padding_and_reuses_storage():
    case = _trailing_case([8, 800], [8, 4], topk=31, compress_ratio=4)
    reference, ref_lens = _reference(**case)
    rows, width = reference.shape
    storage = torch.full((rows * 2, width + 7), -77, dtype=torch.int32, device=DEVICE)
    out = storage[::2, :width]
    out.fill_(-1)
    lens = torch.empty(rows, dtype=torch.int32, device=DEVICE)
    for _ in range(2):
        actual, actual_lens = combine_topk_swa_indices(
            **case, out_indices=out, out_lens=lens
        )
        assert actual.data_ptr() == out.data_ptr()
        assert actual_lens.data_ptr() == lens.data_ptr()
        assert torch.equal(actual, reference)
        assert torch.equal(actual_lens, ref_lens)
    assert (storage[1::2] == -77).all()
    assert (storage[:, width:] == -77).all()


def test_zero_tokens():
    _check(**_trailing_case([7, 9], [0, 0], topk=64, compress_ratio=4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU LightOp dispatch only")
def test_hcu_does_not_launch_triton():
    """Exercise LightOp's native entrypoint without a private SGLang router."""
    import lightop

    case = _trailing_case([512], [5], topk=64, compress_ratio=4)
    expected, expected_lens = _reference(**case)
    actual = torch.full_like(expected, -1)
    actual_lens = torch.full_like(expected_lens, -77)
    with patch.object(
        triton.runtime.jit.JITFunction,
        "run",
        side_effect=AssertionError("LightOp native dispatch must not launch Triton"),
    ) as triton_launch:
        lightop.combine_topk_swa_indices_hcu(
            case["topk_indices"], case["query_start_loc"], case["query_pos"],
            case["seq_lens"], case["gather_lens"], case["compressed_base"],
            case["swa_base"], actual, actual_lens, case["window_size"],
            case["compress_ratio"], case["topk"],
        )
    triton_launch.assert_not_called()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_lens, expected_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU LightOp stream and graph tests")
def test_hcu_current_stream_and_graph_replay():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        case = _trailing_case([512], [8], topk=64, compress_ratio=4)
        ref_idx, ref_lens = _reference(**case)
        out = torch.full_like(ref_idx, -1)
        lens = torch.empty_like(ref_lens)
        combine_topk_swa_indices(**case, out_indices=out, out_lens=lens)
    stream.synchronize()
    assert torch.equal(out, ref_idx)
    assert torch.equal(lens, ref_lens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        combine_topk_swa_indices(**case, out_indices=out, out_lens=lens)
    out.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref_idx)
    assert torch.equal(lens, ref_lens)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="HCU multi-device guard test",
)
def test_hcu_tensor_device_not_current_device():
    case = _trailing_case([512], [8], topk=64, compress_ratio=4)
    ref_idx, ref_lens = _reference(**case)
    case = {
        key: value.to("cuda:1") if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }
    stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream), torch.cuda.device(0):
        out, lens = combine_topk_swa_indices(**case)
        assert torch.cuda.current_device() == 0
    stream.synchronize()
    assert torch.equal(out.cpu(), ref_idx.cpu())
    assert torch.equal(lens.cpu(), ref_lens.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU LightOp validation only")
@pytest.mark.parametrize(
    "bad_input", ["dtype", "metadata_stride", "output_width", "column_stride"]
)
def test_hcu_rejects_invalid_inputs(bad_input):

    case = _trailing_case([512], [4], topk=64, compress_ratio=4)
    ref_idx, ref_lens = _reference(**case)
    if bad_input == "dtype":
        case["seq_lens"] = case["seq_lens"].long()
    elif bad_input == "metadata_stride":
        case["query_pos"] = case["query_pos"].repeat_interleave(2)[::2]
    elif bad_input == "output_width":
        ref_idx = ref_idx[:, :-1]
    else:
        case["topk_indices"] = case["topk_indices"].repeat_interleave(2, dim=1)[
            :, ::2
        ]
    with pytest.raises(RuntimeError):
        combine_topk_swa_indices_hcu(**case, out_indices=ref_idx, out_lens=ref_lens)



def dequant_case(n,p):
    pages=max(8,triton.cdiv(n,p)); b=triton.cdiv(p*584,576)*576
    cache=torch.randint(0,256,(pages,b),device='cuda',dtype=torch.uint8)
    cache[:,p*576:p*584].view(pages,p,8)[:,:,:7]=127
    ids=torch.arange(n,device='cuda',dtype=torch.int32)
    out=torch.empty((n,1,512),device='cuda',dtype=torch.bfloat16)
    return cache,ids,out

def combine_case(n,k,ratio,reqs=1):
    per=n//reqs
    starts=torch.arange(reqs+1,device='cuda',dtype=torch.int32)*per+37
    pos=(torch.arange(per,device='cuda',dtype=torch.int32)*8+8192).repeat(reqs)
    seq=torch.full((reqs,),8192+per*8,device='cuda',dtype=torch.int32)
    gather=torch.full_like(seq,per*8+127)
    base=torch.arange(reqs,device='cuda',dtype=torch.int32)*(int(seq[0])//ratio)
    swa=reqs*(int(seq[0])//ratio)+torch.arange(reqs,device='cuda',dtype=torch.int32)*(per*8+127)
    idx=torch.randint(-1,8192//ratio,(n,k),device='cuda',dtype=torch.int32)
    out=torch.full((n,triton.cdiv(k+128,128)*128),-1,device='cuda',dtype=torch.int32)
    lens=torch.empty(n,device='cuda',dtype=torch.int32)
    return idx,starts,pos,seq,gather,base,swa,out,lens

def combine_triton(case,k,ratio):
    from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import _combine_topk_swa_indices_kernel

    idx,starts,pos,seq,gather,base,swa,out,lens=case
    _combine_topk_swa_indices_kernel[(seq.numel(),128)](out,out.stride(0),lens,idx,idx.stride(0),starts,pos,seq,gather,base,swa,top_k=k,COMPRESS_RATIO=ratio,WINDOW_SIZE=128,PADDED_TOP_K=triton.next_power_of_2(k))


def paired_measure(reference, candidate):
    functions = {'triton': reference, 'lightop': candidate}
    graphs = {}
    for name, fn in functions.items():
        for _ in range(10): fn()
        torch.cuda.synchronize()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            for _ in range(100): fn()
    samples = {name: [] for name in functions}
    for round_index in range(5):
        names = ('triton', 'lightop') if round_index % 2 == 0 else ('lightop', 'triton')
        for name in names:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name].replay(); end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    return {name: {'mean_us': statistics.mean(values), 'samples_us': values, 'CV': statistics.stdev(values) / statistics.mean(values)} for name, values in samples.items()}


def _check_dequant_outputs(reference, candidate, ref_out, out):
    reference()
    expected = ref_out.clone()
    # Every sentinel differs from the expected value, including NaN targets.
    out.copy_(torch.where(torch.isnan(expected), 0, float('nan')))
    candidate()
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)


def _check_combine_outputs(reference, candidate, ref_out, ref_lens, out, lens):
    reference()
    expected, expected_lens = ref_out.clone(), ref_lens.clone()
    out.fill_(-1)
    scanned = torch.arange(out.shape[1], device=out.device)[None, :] < expected_lens[:, None]
    out.masked_fill_(scanned, -2)
    lens.fill_(-1)
    candidate()
    assert torch.equal(out, expected)
    assert torch.equal(lens, expected_lens)


@pytest.mark.parametrize('write', ['all', 'none', 'skip_nan', 'skip_finite'])
def test_dequant_parity_detects_missing_writes(write):
    expected = torch.tensor([0, -0., 1, -3, float('inf'), -float('inf'), float('nan')], dtype=torch.bfloat16)
    ref_out, out = torch.empty_like(expected), torch.empty_like(expected)
    reference = lambda: ref_out.copy_(expected)

    def candidate():
        if write == 'all':
            out.copy_(expected)
        elif write == 'skip_nan':
            out[:-1].copy_(expected[:-1])
        elif write == 'skip_finite':
            out[1:].copy_(expected[1:])

    if write == 'all':
        _check_dequant_outputs(reference, candidate, ref_out, out)
    else:
        with pytest.raises(AssertionError):
            _check_dequant_outputs(reference, candidate, ref_out, out)


@pytest.mark.parametrize('write', ['all', 'none', 'indices_only', 'lens_only', 'skip_hole', 'bad_padding'])
def test_combine_parity_detects_missing_writes(write):
    expected = torch.tensor([[-1, 4, 7, -1], [8, -1, -1, -1]], dtype=torch.int32)
    expected_lens = torch.tensor([3, 1], dtype=torch.int32)
    ref_out, out = torch.empty_like(expected), torch.empty_like(expected)
    ref_lens, lens = torch.empty_like(expected_lens), torch.empty_like(expected_lens)

    def reference():
        ref_out.copy_(expected)
        ref_lens.copy_(expected_lens)

    def candidate():
        if write in ('all', 'indices_only', 'bad_padding'):
            out.copy_(expected)
        elif write == 'skip_hole':
            out[0, 1:3].copy_(expected[0, 1:3])
            out[1, 0] = expected[1, 0]
        if write in ('all', 'lens_only', 'skip_hole', 'bad_padding'):
            lens.copy_(expected_lens)
        if write == 'bad_padding':
            out[0, -1] = 0

    if write == 'all':
        _check_combine_outputs(reference, candidate, ref_out, ref_lens, out, lens)
    else:
        with pytest.raises(AssertionError):
            _check_combine_outputs(reference, candidate, ref_out, ref_lens, out, lens)


def benchmark(tokens, profile_only=False):
    import sglang.kernels.ops.attention.dsv4.dequant_k_cache as dequant

    torch.manual_seed(42)
    results = []
    for n in tokens:
        for p in (64, 256):
            cache, ids, out = dequant_case(n, p)
            ref_out = torch.empty_like(out)
            fnuz = dequant.fp8_dtype == torch.float8_e4m3fnuz
            reference = lambda: _triton_dequant(cache, ids, p, out=ref_out)
            candidate = lambda: dequantize_k_cache_paged_hcu(cache, ids, out, p, fnuz)
            _check_dequant_outputs(reference, candidate, ref_out, out)
            if profile_only:
                for _ in range(10): reference(); candidate()
                torch.cuda.synchronize()
                continue
            timings = paired_measure(reference, candidate)
            traffic = n * (576 + 7 + 1024 + 4)
            for item in timings.values(): item['effective_GB_s'] = traffic / item['mean_us'] / 1000
            result = {'op': 'dequant', 'N': n, 'page_size': p, 'dtype': f'FP8_{"FNUZ" if fnuz else "FN"}+BF16->BF16, ids INT32', 'correct': True, 'logical_bytes': traffic, 'timings': timings}
            result['speedup'] = timings['triton']['mean_us'] / timings['lightop']['mean_us']
            results.append(result); print(json.dumps(result), flush=True)
        for k, ratio in ((512, 4), (2048, 128)):
            case = combine_case(n, k, ratio)
            ref_case = (*case[:-2], torch.full_like(case[-2], -1), torch.empty_like(case[-1]))
            reference = lambda: combine_triton(ref_case, k, ratio)
            candidate = lambda: combine_topk_swa_indices_hcu(*case, 128, ratio, k)
            _check_combine_outputs(reference, candidate, *ref_case[-2:], *case[-2:])
            if profile_only:
                for _ in range(10): reference(); candidate()
                torch.cuda.synchronize()
                continue
            timings = paired_measure(reference, candidate)
            scanned = int(case[-1].sum())
            traffic = 4 * (2 * scanned - n * 128 + n * 8)
            for item in timings.values(): item['effective_GB_s'] = traffic / item['mean_us'] / 1000
            result = {'op': 'combine', 'N': n, 'topk': k, 'ratio': ratio, 'dtype': 'INT32', 'correct': True, 'logical_bytes': traffic, 'timings': timings}
            result['speedup'] = timings['triton']['mean_us'] / timings['lightop']['mean_us']
            results.append(result); print(json.dumps(result), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-performance', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--tokens', type=int, nargs='+', default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--output', default='dsv4_prefill_hcu_benchmark.json')
    args = parser.parse_args()
    if not args.profile_only:
        code = pytest.main([__file__, '-q'])
        if code: raise SystemExit(code)
    if not args.skip_performance:
        results = benchmark(args.tokens, args.profile_only)
        if not args.profile_only:
            Path(args.output).write_text(json.dumps({'test': __file__, 'reference': 'SGLang original Triton; Torch edge-case oracle', 'warmup': 10, 'timed_iterations': 100, 'rounds': 5, 'physical_card': os.getenv('HIP_VISIBLE_DEVICES'), 'torch': torch.__version__, 'hip': torch.version.hip, 'gfx': torch.cuda.get_device_properties(0).gcnArchName, 'results': results}, indent=2))
