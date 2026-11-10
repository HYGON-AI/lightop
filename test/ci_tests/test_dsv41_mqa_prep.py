# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Exact MQA prep/gather tests and synchronized I4096 operator comparisons.

Run correctness: python -m pytest -q test/test_dsv41_mqa_prep.py
Run correctness, then performance on an idle card:
  HIP_VISIBLE_DEVICES=0 python test/test_dsv41_mqa_prep.py --benchmark --json /path/results.json
The caller must capture hy-smi before benchmarking; this test never stops services.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch


@pytest.fixture(scope="module")
def prep_ops():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a HIP GPU")
    import lightop

    for name in ("dsv41_mqa_query_lens", "dsv41_mqa_prefill_metadata",
                 "dsv41_gather_index_k_fp4"):
        assert callable(getattr(lightop, name, None)), f"rebuild LightOp: missing {name}"
    return lightop


def _case(cp_size=8, cp_rank=0, dtype=torch.int64, tail=False,
          extend=(13, 0, 19, 7), prefix=(1, 7, 3, 9), mapping_seed=17):
    """Keep original global row ownership, not re-interleaved tail rows."""
    starts = []
    global_positions = []
    owners = []
    for request, (length, cached) in enumerate(zip(extend, prefix)):
        starts.append(len(global_positions))
        global_positions.extend(range(cached, cached + length))
        owners.extend([request] * length)
    chosen = [i for i in range(len(global_positions))
              if i % cp_size == cp_rank and (not tail or i % 5 in (1, 2, 4))]
    positions = torch.tensor([global_positions[i] for i in chosen], dtype=torch.int64)
    counts = torch.tensor([sum(owners[i] == request for i in chosen)
                           for request in range(len(extend))], dtype=torch.int32)
    seq_lens = torch.tensor([p + n for p, n in zip(prefix, extend)], dtype=dtype)
    request_ids = torch.tensor([6, 1, 4, 2][:len(extend)], dtype=torch.int64)
    width = max(seq_lens.tolist(), default=0) + 1
    generator = torch.Generator().manual_seed(mapping_seed)
    table = torch.randint(0, 4096, (8, width), generator=generator, dtype=torch.int32)
    return dict(starts=torch.tensor(starts, dtype=torch.int32),
                lengths=torch.tensor(extend, dtype=torch.int32),
                seq_lens=seq_lens, q_lens=counts, positions=positions,
                req_pool_indices=request_ids, req_to_token=table,
                cp_size=cp_size, cp_rank=cp_rank, tail=tail)


def _metadata_reference(case, ratio):
    ks, ke, visible, slots = [], [], [], []
    query_start = 0
    key_start = 0
    for request, (length, queries) in enumerate(zip(case["seq_lens"].tolist(),
                                                  case["q_lens"].tolist())):
        keys = length // ratio
        for pos in case["positions"][query_start:query_start + queries].tolist():
            count = (pos + 1) // ratio
            ks.append(key_start)
            ke.append(key_start + count)
            visible.append(count)
        request_id = int(case["req_pool_indices"][request])
        slots.extend(int(case["req_to_token"][request_id, j * ratio]) // ratio
                     for j in range(keys))
        query_start += queries
        key_start += keys
    return (torch.tensor(ks, dtype=torch.int32), torch.tensor(ke, dtype=torch.int32),
            torch.tensor(visible, dtype=torch.int32), torch.tensor(slots, dtype=torch.int64))


def _gpu_args(case, ratio):
    lengths = case["seq_lens"].tolist()
    counts = case["q_lens"].tolist()
    return dict(seq_lens=case["seq_lens"].cuda(), q_lens=case["q_lens"].cuda(),
                positions=case["positions"].cuda(),
                req_pool_indices=case["req_pool_indices"].cuda(),
                req_to_token=case["req_to_token"].cuda(), ratio=ratio,
                num_k_slots=sum(n // ratio for n in lengths),
                max_request_work=max(counts + [n // ratio for n in lengths], default=0))


def _assert_exact(actual, expected):
    torch.cuda.synchronize()
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size,cp_rank", [(c, r) for c in (1, 2, 4, 8, 16)
                                           for r in range(c)])
def test_query_lens_original_cp_ownership(prep_ops, cp_size, cp_rank):
    case = _case(cp_size, cp_rank)
    actual = prep_ops.dsv41_mqa_query_lens(case["starts"].cuda(),
                                         case["lengths"].cuda(), cp_size, cp_rank)
    _assert_exact((actual,), (case["q_lens"],))


@pytest.mark.parametrize("cp_size,cp_rank", [(c, r) for c in (1, 2, 4, 8, 16)
                                           for r in range(c)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tail", [False, True])
def test_prefill_metadata_exact(prep_ops, cp_size, cp_rank, ratio, dtype, tail):
    case = _case(cp_size, cp_rank, dtype, tail)
    args = _gpu_args(case, ratio)
    if not tail:
        args["q_lens"] = prep_ops.dsv41_mqa_query_lens(
            case["starts"].cuda(), case["lengths"].cuda(), cp_size, cp_rank)
    # Tail counts are already selected by original owners; they are not CP-split again.
    actual = prep_ops.dsv41_mqa_prefill_metadata(**args)
    _assert_exact(actual, _metadata_reference(case, ratio))


@pytest.mark.parametrize("ratio", [1, 2])
def test_same_lengths_different_request_mapping(prep_ops, ratio):
    first = _case(cp_size=4, cp_rank=1, mapping_seed=3)
    second = _case(cp_size=4, cp_rank=1, mapping_seed=91)
    second["req_pool_indices"] = first["req_pool_indices"].flip(0).contiguous()
    a = prep_ops.dsv41_mqa_prefill_metadata(**_gpu_args(first, ratio))
    b = prep_ops.dsv41_mqa_prefill_metadata(**_gpu_args(second, ratio))
    _assert_exact(a, _metadata_reference(first, ratio))
    _assert_exact(b, _metadata_reference(second, ratio))
    assert not torch.equal(a[-1], b[-1]), "slots must follow the current request allocation"


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("extend,prefix", [((0,), (0,)), ((1,), (0,)),
                                         ((0, 0), (3, 7)), ((), ())])
def test_empty_queries_and_keys(prep_ops, ratio, extend, prefix):
    case = _case(16, 15, extend=extend, prefix=prefix)
    actual = prep_ops.dsv41_mqa_prefill_metadata(**_gpu_args(case, ratio))
    _assert_exact(actual, _metadata_reference(case, ratio))
    counts = prep_ops.dsv41_mqa_query_lens(case["starts"].cuda(),
                                          case["lengths"].cuda(), 16, 15)
    _assert_exact((counts,), (case["q_lens"],))


def test_nonempty_query_without_compressed_keys(prep_ops):
    # ratio=2: a genuine Q row at position zero sees zero compressed K.
    case = _case(1, 0, extend=(1,), prefix=(0,))
    actual = prep_ops.dsv41_mqa_prefill_metadata(**_gpu_args(case, 2))
    _assert_exact(actual, _metadata_reference(case, 2))
    assert actual[0].shape == (1,) and actual[-1].shape == (0,)
    empty_cache = torch.empty((0, 128 * 68), dtype=torch.uint8, device="cuda")
    gathered = prep_ops.dsv41_gather_index_k_fp4(empty_cache, actual[-1], 128)
    _assert_exact(gathered, (torch.empty((0, 64), dtype=torch.int8),
                            torch.empty((0,), dtype=torch.int32)))


def _cache(page_size, padded=False, offset=False):
    generator = torch.Generator().manual_seed(20261005 + page_size)
    row_stride = page_size * 68 + (13 if padded else 0)
    raw = torch.randint(256, (5 * row_stride + (11 if offset else 0),),
                        generator=generator, dtype=torch.uint8)
    return torch.as_strided(raw, (5, page_size * 68), (row_stride, 1),
                            storage_offset=11 if offset else 0)


def _gather_reference(cache, slots, page_size):
    payload = torch.empty((slots.numel(), 64), dtype=torch.uint8)
    scale_bytes = torch.empty((slots.numel(), 4), dtype=torch.uint8)
    for row, slot in enumerate(slots.tolist()):
        page, offset = divmod(slot, page_size)
        payload[row] = cache[page, offset * 64:(offset + 1) * 64]
        start = page_size * 64 + offset * 4
        scale_bytes[row] = cache[page, start:start + 4]
    return payload.view(torch.int8), scale_bytes.view(torch.int32).reshape(-1)


def _gpu_cache(cache):
    # Preserve page stride and storage offset; .cuda() alone would compact this view.
    storage = torch.empty(cache.storage_offset() + cache.shape[0] * cache.stride(0),
                          dtype=torch.uint8, device="cuda")
    view = torch.as_strided(storage, cache.shape, cache.stride(), cache.storage_offset())
    view.copy_(cache)
    return view


@pytest.mark.parametrize("page_size", [1, 2, 7, 128])
@pytest.mark.parametrize("padded,offset", [(False, False), (True, False),
                                         (False, True), (True, True)])
@pytest.mark.parametrize("empty", [False, True])
def test_gather_byte_exact(prep_ops, page_size, padded, offset, empty):
    cache = _cache(page_size, padded, offset)
    values = [] if empty else [0, page_size - 1, page_size, 5 * page_size - 1,
                               page_size, 2 * page_size + 1 if page_size > 1 else 2]
    slots = torch.tensor(values, dtype=torch.int64)
    storage = torch.empty(slots.numel() + 3, dtype=torch.int64, device="cuda")
    gpu_slots = storage[3:]
    gpu_slots.copy_(slots)
    actual = prep_ops.dsv41_gather_index_k_fp4(_gpu_cache(cache), gpu_slots, page_size)
    _assert_exact(actual, _gather_reference(cache, slots, page_size))


def test_current_stream(prep_ops):
    case = _case(8, 3)
    cache = _cache(128, padded=True, offset=True)
    stream = torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        starts, lengths = case["starts"].cuda(), case["lengths"].cuda()
        args = _gpu_args(case, 2)
        args["q_lens"] = prep_ops.dsv41_mqa_query_lens(starts, lengths, 8, 3)
        metadata = prep_ops.dsv41_mqa_prefill_metadata(**args)
        # Make a valid subset of physical slots while retaining producer ordering.
        slots = metadata[-1].remainder(5 * 128)
        gathered = prep_ops.dsv41_gather_index_k_fp4(_gpu_cache(cache), slots, 128)
        done.record(stream)
    torch.cuda.current_stream().wait_event(done)
    reference = _metadata_reference(case, 2)
    _assert_exact(metadata, reference)
    _assert_exact(gathered, _gather_reference(cache, reference[-1] % (5 * 128), 128))


@pytest.mark.parametrize("ratio", [1, 2])
def test_existing_mqa_logits_topk_and_physical_mapping(prep_ops, ratio):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip("existing BF16/FP4 MQA requires gfx936/gfx938")
    # Only five local Q rows, but windows above 512 exercise actual TopK selection.
    case = _case(8, 0, prefix=(689, 719, 691, 713))
    args = _gpu_args(case, ratio)
    fused = prep_ops.dsv41_mqa_prefill_metadata(**args)
    reference = tuple(t.cuda() for t in _metadata_reference(case, ratio))
    generator = torch.Generator().manual_seed(20261005)
    cache_cpu = torch.randint(256, (33, 128 * 68), dtype=torch.uint8, generator=generator)
    # UE8M0 exponent bytes are finite and ordinary: random full-range bytes may encode NaNs.
    cache_cpu[:, 128 * 64:] = torch.randint(
        120, 134, (33, 128 * 4), dtype=torch.uint8, generator=generator)
    cache = cache_cpu.cuda()
    old_k = _torch_gather(cache, reference[-1], 128)
    new_k = prep_ops.dsv41_gather_index_k_fp4(cache, fused[-1], 128)
    tokens = args["positions"].numel()
    q = torch.randn(tokens, 4, 128, generator=generator).bfloat16().cuda()
    weights = torch.randn(tokens, 4, generator=generator).bfloat16().float().cuda()
    width = max(n // ratio for n in case["seq_lens"].tolist())
    width = (width + 3) // 4 * 4

    def logits(metadata, kv):
        return prep_ops.fp8_fp4_mqa_logits(
            q=(q, None), kv=kv, weights=weights,
            cu_seq_len_k_start=metadata[0], cu_seq_len_k_end=metadata[1],
            clean_logits=False, max_seqlen_k=width)

    old_logits, new_logits = logits(reference, old_k), logits(fused, new_k)
    torch.cuda.synchronize()
    torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
    valid = torch.arange(width, device="cuda")[None, :] < fused[2][:, None]
    assert torch.isfinite(new_logits[valid]).all()
    assert torch.isneginf(new_logits[~valid]).all()
    def select(scores, metadata):
        indices = torch.empty((tokens, 512), dtype=torch.int32, device="cuda")
        prep_ops.topk_transform_ragged_v2(scores, metadata[2],
                                         out_offsets=metadata[0], out_indices=indices)
        mapped = metadata[-1][indices.clamp(min=0).long()].int()
        return indices, torch.where(indices >= 0, mapped, -1)

    # Existing native TopK guarantees the exact set (including its tie policy),
    # not output order: adaptive_topk.h emits waves through atomic reservations.
    # Repeat the *same old input* to expose ordering variation independently of prep.
    selections = [select(old_logits, reference) for _ in range(3)]
    selections.append(select(new_logits, fused))
    torch.cuda.synchronize()
    baseline_ids, baseline_physical = selections[0]
    order_changes = [int((ids != baseline_ids).sum()) for ids, _ in selections[1:3]]
    print(f"NATIVE_TOPK_ORDER_DIAGNOSTIC ratio={ratio} old_repeat_changes={order_changes}")
    for ids, physical in selections[1:]:
        torch.testing.assert_close(ids.sort(dim=1).values,
                                   baseline_ids.sort(dim=1).values, rtol=0, atol=0)
        torch.testing.assert_close(physical.sort(dim=1).values,
                                   baseline_physical.sort(dim=1).values, rtol=0, atol=0)
    scores_cpu, lengths_cpu, offsets_cpu = (tensor.cpu() for tensor in
                                           (old_logits, reference[2], reference[0]))
    for indices, _ in selections:
        for row, output in enumerate(indices.cpu()):
            length, offset = int(lengths_cpu[row]), int(offsets_cpu[row])
            valid_ids = output[output >= 0].long() - offset
            count = min(length, 512)
            assert valid_ids.numel() == count
            assert valid_ids.unique().numel() == count
            assert ((valid_ids >= 0) & (valid_ids < length)).all()
            assert (output[output < 0] == -1).all()
            values = scores_cpu[row, :length]
            cutoff = values.topk(count).values.min()
            assert (values[valid_ids] >= cutoff).all()
            assert int((values[valid_ids] > cutoff).sum()) == int((values > cutoff).sum())


@pytest.mark.parametrize("bad", ["ratio", "launch", "query_dtype", "query_shape",
                                  "seq_shape", "positions_dtype", "request_dtype",
                                  "table_dtype", "table_layout", "host_positions"])
def test_invalid_metadata_structure(prep_ops, bad):
    args = _gpu_args(_case(), 1)
    if bad == "ratio":
        args["ratio"] = 4
    elif bad == "launch":
        args["max_request_work"] = 0
    elif bad == "query_dtype":
        args["q_lens"] = args["q_lens"].long()
    elif bad == "query_shape":
        args["q_lens"] = args["q_lens"][:-1]
    elif bad == "seq_shape":
        args["seq_lens"] = args["seq_lens"][None, :]
    elif bad == "positions_dtype":
        args["positions"] = args["positions"].int()
    elif bad == "request_dtype":
        args["req_pool_indices"] = args["req_pool_indices"].int()
    elif bad == "table_dtype":
        args["req_to_token"] = args["req_to_token"].long()
    elif bad == "table_layout":
        args["req_to_token"] = args["req_to_token"][:, ::2]
    else:
        args["positions"] = args["positions"].cpu()
    with pytest.raises(RuntimeError):
        prep_ops.dsv41_mqa_prefill_metadata(**args)


@pytest.mark.parametrize("bad", ["size", "rank", "dtype", "shape", "stride"])
def test_invalid_query_structure(prep_ops, bad):
    starts, lengths = torch.arange(4, dtype=torch.int32, device="cuda"), torch.ones(
        4, dtype=torch.int32, device="cuda")
    cp_size, cp_rank = 8, 0
    if bad == "size":
        cp_size = 0
    elif bad == "rank":
        cp_rank = 8
    elif bad == "dtype":
        starts = starts.long()
    elif bad == "shape":
        lengths = lengths[:-1]
    else:
        starts = torch.arange(8, dtype=torch.int32, device="cuda")[::2]
    with pytest.raises(RuntimeError):
        prep_ops.dsv41_mqa_query_lens(starts, lengths, cp_size, cp_rank)


@pytest.mark.parametrize("bad", ["page", "cache_dtype", "cache_width", "slots_dtype",
                                  "slots_stride", "cache_columns"])
def test_invalid_gather_structure(prep_ops, bad):
    page_size = 128
    cache = _gpu_cache(_cache(page_size))
    slots = torch.tensor([0, 129], dtype=torch.int64, device="cuda")
    if bad == "page":
        page_size = 0
    elif bad == "cache_dtype":
        cache = cache.to(torch.int8)
    elif bad == "cache_width":
        cache = cache[:, :-1]
    elif bad == "slots_dtype":
        slots = slots.int()
    elif bad == "slots_stride":
        slots = torch.arange(4, dtype=torch.int64, device="cuda")[::2]
    else:
        cache = torch.empty((5, page_size * 136), dtype=torch.uint8,
                            device="cuda")[:, ::2]
    with pytest.raises(RuntimeError):
        prep_ops.dsv41_gather_index_k_fp4(cache, slots, page_size)


def _timed(call, useful_bytes, warmup=10, repeats=100, rounds=3):
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    median = statistics.median(samples)
    return dict(round_mean_us=samples, mean_us=statistics.mean(samples),
                median_us=median, relative_range=(max(samples) - min(samples)) / median,
                useful_bytes=useful_bytes, effective_GBps=useful_bytes / median / 1000)


def _torch_gather(cache, slots, page_size):
    page, offset = (slots // page_size).unsqueeze(-1), slots % page_size
    payload_cols = offset[:, None] * 64 + torch.arange(64, device=cache.device)
    scale_cols = page_size * 64 + offset[:, None] * 4 + torch.arange(4, device=cache.device)
    return (cache[page, payload_cols].view(torch.int8),
            cache[page, scale_cols].contiguous().view(torch.int32).squeeze(-1))


def _benchmark():
    import lightop

    properties = torch.cuda.get_device_properties(0)
    results = []
    for batch in (1, 4):
        extend = (4096,) if batch == 1 else (1021, 1031, 997, 1047)
        case = _case(8, 0, extend=extend, prefix=(0,) * batch)
        starts_gpu, lengths_gpu = case["starts"].cuda(), case["lengths"].cuda()
        for ratio in (1, 2):
            args = _gpu_args(case, ratio)
            lengths = case["seq_lens"].tolist()
            q_cpu = case["q_lens"].tolist()
            key_starts = []
            cursor = 0
            for length in lengths:
                key_starts.append(cursor)
                cursor += length // ratio

            def torch_prep():
                q_lens = torch.tensor(q_cpu, dtype=torch.int32, device="cuda")
                chunks = [args["req_to_token"][args["req_pool_indices"][b],
                          torch.arange(length // ratio, device="cuda") * ratio].long() // ratio
                          for b, length in enumerate(lengths)]
                slots = torch.cat(chunks)
                visible = ((args["positions"] + 1) // ratio).int()
                ks = torch.repeat_interleave(torch.tensor(key_starts, dtype=torch.int32,
                                                          device="cuda"),
                                             q_lens.long(), output_size=case["positions"].numel())
                return ks, ks + visible, visible, slots

            def fused_first_use_prep():
                q_lens = lightop.dsv41_mqa_query_lens(starts_gpu, lengths_gpu, 8, 0)
                return lightop.dsv41_mqa_prefill_metadata(**dict(args, q_lens=q_lens))

            expected = _metadata_reference(case, ratio)
            _assert_exact(torch_prep(), expected)
            prepared = fused_first_use_prep()
            _assert_exact(prepared, expected)
            cache = torch.randint(256, (33, 128 * 68), dtype=torch.uint8, device="cuda")
            slots = prepared[-1]
            _assert_exact(lightop.dsv41_gather_index_k_fp4(cache, slots, 128),
                          tuple(x.cpu() for x in _torch_gather(cache, slots, 128)))
            tokens, keys = args["positions"].numel(), args["num_k_slots"]
            # Unique useful input/output bytes, not a claim about actual cache traffic.
            metadata_bytes = tokens * (8 + 12) + keys * (4 + 8) + batch * (8 + 4 + 8)
            gather_bytes = keys * 68 * 2
            results.append(dict(input_len=4096, batch=batch, cp=8, cp_rank=0,
                                local_queries=tokens, ratio=ratio, keys=keys, page_size=128,
                                correctness="exact", torch_prep=_timed(torch_prep, metadata_bytes),
                                fused_first_use_prep=_timed(fused_first_use_prep, metadata_bytes),
                                fused_ratio_metadata=_timed(
                                    lambda: lightop.dsv41_mqa_prefill_metadata(**args), metadata_bytes),
                                torch_gather=_timed(lambda: _torch_gather(cache, slots, 128), gather_bytes),
                                fused_gather=_timed(
                                    lambda: lightop.dsv41_gather_index_k_fp4(cache, slots, 128), gather_bytes)))
    return dict(environment=dict(torch=torch.__version__, hip=torch.version.hip,
                                 device=properties.name, gfx=properties.gcnArchName,
                                 HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES")),
                warmup=10, timed_iterations=100, rounds=3, operators=results,
                scope="Synchronized operator wall-clock including allocation and launch; not E2E.",
                reuse="First-use prep includes query kernel. Framework reuses query counts per forward "
                      "and ratio metadata across layers; gather still runs every layer.",
                bandwidth="Unique useful read+write bytes / median latency; excludes repeated CTA "
                          "prefix reads and is not measured hardware DRAM bandwidth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    status = pytest.main([__file__, "-q"])
    if status:
        raise SystemExit(status)
    if options.benchmark:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise SystemExit("benchmark requires HIP GPU; skipped tests are not a correctness pass")
        report = _benchmark()
        serialized = json.dumps(report, indent=2) + "\n"
        if options.json:
            options.json.write_text(serialized)
        print(serialized, flush=True)
