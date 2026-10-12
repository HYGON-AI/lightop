"""get_paged_sparse_mqa_logits_metadata against a reference of DeepGEMM's semantics.

Q blocks claim their KV-split range with a global atomic (as in DeepGEMM), so the
metadata is compared after renumbering each Q block's splits from zero; everything
else, including the balanced schedule, must match exactly.
"""

import random

import numpy as np
import pytest
import torch

SPLIT_KV = {"fp4": 640, "fp8": 512}
QK_DTYPE = {"fp4": torch.int8, "fp8": torch.float8_e4m3fn}
INVALID_SLOT = 0xFFFF


def ceil_div(a, b):
    return (a + b - 1) // b


def num_sms_default():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(request_sizes, request_context_lens, sparse_block_kv, num_max_sparse_blocks,
                page_kv, seed, block_table_extra_cols=0):
    """DeepGEMM test_attention.py style paged inputs: consecutive tokens of a request
    keep ~80% of the previous token's sparse blocks."""
    rng = random.Random(seed)
    indices = [r for r, size in enumerate(request_sizes) for _ in range(size)]
    context_lens = [length + offset for length, size in zip(request_context_lens, request_sizes)
                    for offset in range(size)]
    rows, previous, previous_request = [], None, None
    for context_len, request in zip(context_lens, indices):
        num_available = ceil_div(context_len, sparse_block_kv)
        num_sparse = min(num_available, num_max_sparse_blocks)
        if num_sparse == num_available:
            blocks = list(range(num_available))
        elif request != previous_request:
            blocks = rng.sample(range(num_available), num_sparse)
        else:
            kept = [b for b in previous if b < num_available]
            kept = rng.sample(kept, min(round(num_sparse * 0.8), len(kept)))
            kept_set = set(kept)
            while len(kept_set) < num_sparse:
                kept_set.add(rng.randrange(num_available))
            blocks = list(kept_set)
        blocks.sort()
        rows.append(blocks + [blocks[-1] if blocks else 0] * (num_max_sparse_blocks - num_sparse))
        previous, previous_request = blocks, request

    num_pages = [ceil_div(length + size - 1, page_kv) if length + size > 1 else 1
                 for length, size in zip(request_context_lens, request_sizes)]
    total_pages = sum(num_pages)
    pool = torch.randperm(total_pages, generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    request_table = torch.zeros((len(request_sizes), max(num_pages) + block_table_extra_cols), dtype=torch.int32)
    begin = 0
    for r, n in enumerate(num_pages):
        request_table[r, :n] = pool[begin:begin + n]
        begin += n
    idx = torch.tensor(indices, dtype=torch.int32)
    return dict(
        context_lens=torch.tensor(context_lens, dtype=torch.int32),
        block_table=request_table[idx.long()].contiguous(),
        indices=idx,
        sparse=torch.tensor(rows, dtype=torch.int32).reshape(len(indices), num_max_sparse_blocks),
    )


def reference(context_lens, block_table, indices, page_kv, sparse, split_kv, sparse_block_kv, num_sms):
    """DeepGEMM's paged metadata with Q blocks' splits allocated in Q order."""
    lens, table, idx = context_lens.numpy(), block_table.numpy(), indices.numpy()
    sparse = sparse.numpy()
    num_q, m = sparse.shape
    blocks_per_split = split_kv // sparse_block_kv
    blocks_per_page = page_kv // sparse_block_kv
    splits, q_block_infos = [], [(0, 0)] * num_q
    for q in range(num_q):
        start = q
        while start > 0 and idx[start - 1] == idx[q]:
            start -= 1
        if (q - start) % 2:
            continue
        num_tokens = 2 if q + 1 < num_q and idx[q + 1] == idx[q] else 1
        rows = [sparse[q + t, :min(m, ceil_div(int(lens[q + t]), sparse_block_kv))].astype(np.int64)
                for t in range(num_tokens)]
        if num_tokens == 1:
            rows.append(np.zeros(0, dtype=np.int64))
        merged = np.union1d(rows[0], rows[1])
        slots, present = [], []
        for row in rows:
            slot = np.searchsorted(row, merged)
            slots.append(slot)
            present.append((slot < len(row)) & (row[np.minimum(slot, max(len(row) - 1, 0))] == merged)
                           if len(row) else np.zeros(len(merged), dtype=bool))
        physical = (table[q, merged // blocks_per_page].astype(np.int64) * blocks_per_page
                    + merged % blocks_per_page) & 0xFFFFFFFF
        num_splits = ceil_div(len(merged), blocks_per_split)
        q_block_infos[q] = (len(splits), num_splits)
        for s in range(num_splits):
            lo, hi = s * blocks_per_split, min((s + 1) * blocks_per_split, len(merged))
            bases = [int(slots[t][lo]) for t in range(2)]
            infos = []
            for k in range(lo, hi):
                offsets = [int(slots[t][k]) - bases[t] if present[t][k] else INVALID_SLOT for t in range(2)]
                infos.append((int(physical[k]), offsets[0] | (offsets[1] << 16)))
            infos += [(0, INVALID_SLOT | (INVALID_SLOT << 16))] * (blocks_per_split - len(infos))
            header = (q, hi - lo, bases[0], bases[1] if num_tokens == 2 else 0xFFFFFFFF)
            splits.append((header, infos))

    entries = []
    for q in range(num_q):
        base, n = q_block_infos[q]
        num_entries = ceil_div(n, 8)
        num_tokens = 2 if q + 1 < num_q and idx[q + 1] == idx[q] else 1
        for e in range(num_entries):
            end = base + n // num_entries + (e < n % num_entries)
            entries.append((base, end, q, num_tokens))
            base = end
    num_waves = max(1, ceil_div(len(entries), num_sms))
    entries += [(0, 0, 0, 0)] * (num_waves * num_sms - len(entries))
    if num_waves > 1:
        for w in range(num_waves):
            wave = entries[w * num_sms:(w + 1) * num_sms]
            order = sorted(range(num_sms), key=lambda i: (wave[i][1] - wave[i][0], i))
            out = [None] * num_sms
            for rank, i in enumerate(order):
                dst = num_sms - 1 - rank if num_waves == 2 and w == 1 else \
                    (rank + num_sms - w * num_sms // num_waves) % num_sms
                out[dst] = wave[i]
            entries[w * num_sms:(w + 1) * num_sms] = out
    return splits, entries, num_waves


def decode(metadata, split_kv, sparse_block_kv, num_sms):
    raw = metadata.cpu().numpy()
    words = raw[:16].view(np.uint32)
    num_splits, num_waves, unaligned = int(words[0]), int(words[1]), int(words[2])
    blocks_per_split = split_kv // sparse_block_kv
    split_words = 4 + 2 * blocks_per_split
    body = raw[16:16 + num_splits * split_words * 4].view(np.uint32).reshape(num_splits, split_words)
    splits = [(tuple(int(x) for x in row[:4]), [(int(row[4 + 2 * k]), int(row[5 + 2 * k]))
                                                for k in range(blocks_per_split)]) for row in body]
    begin = 16 + num_splits * split_words * 4
    sched = raw[begin:begin + num_waves * num_sms * 16].view(np.uint32).reshape(-1, 4)
    entries = [tuple(int(x) for x in row) for row in sched]
    return splits, entries, num_waves, unaligned


def canonicalize(splits, entries):
    """Renumber every Q block's splits from zero and map entries onto them."""
    first = {}
    for i, (header, _) in enumerate(splits):
        first.setdefault(header[0], i)
    grouped = {}
    for i, (header, infos) in enumerate(splits):
        assert i - first[header[0]] == len(grouped.setdefault(header[0], [])), "Q block splits not contiguous"
        grouped[header[0]].append((header, infos))
    canon_entries = [(0, 0, 0, 0) if e == (0, 0, 0, 0) else
                     (e[0] - first[e[2]], e[1] - first[e[2]], e[2], e[3]) for e in entries]
    return grouped, canon_entries


def run_case(inputs, page_kv, fmt, sparse_block_kv, num_sms=None, check=True):
    from lightop import op

    num_sms = num_sms or num_sms_default()
    dev = {k: v.cuda() for k, v in inputs.items()}
    metadata = op.get_paged_sparse_mqa_logits_metadata(
        dev["context_lens"], dev["block_table"], dev["indices"], page_kv, dev["sparse"],
        QK_DTYPE[fmt], sparse_block_kv, num_sms)
    num_q, m = inputs["sparse"].shape
    assert metadata.dtype == torch.uint8 and metadata.numel() == op.get_paged_sparse_mqa_logits_metadata_bytes(
        num_q, m, sparse_block_kv, SPLIT_KV[fmt], num_sms)
    if not check:
        return metadata
    splits, entries, num_waves, unaligned = decode(metadata, SPLIT_KV[fmt], sparse_block_kv, num_sms)
    ref_splits, ref_entries, ref_waves = reference(
        inputs["context_lens"], inputs["block_table"], inputs["indices"], page_kv, inputs["sparse"],
        SPLIT_KV[fmt], sparse_block_kv, num_sms)
    assert unaligned == 0 and num_waves == ref_waves and len(splits) == len(ref_splits)
    grouped, canon_entries = canonicalize(splits, entries)
    ref_grouped, ref_canon_entries = canonicalize(ref_splits, ref_entries)
    assert grouped.keys() == ref_grouped.keys()
    for q in ref_grouped:
        assert grouped[q] == ref_grouped[q], f"KV splits of Q block {q} differ"
    assert canon_entries == ref_canon_entries
    return metadata


def random_requests(num_q_tokens, avg_kv_len, sparse_block_kv, seed, sizes=(2, 6)):
    rng = random.Random(seed)
    request_sizes = []
    while sum(request_sizes) < num_q_tokens:
        request_sizes.append(min(rng.randint(*sizes), num_q_tokens - sum(request_sizes)))
    lens = [rng.randint(int(0.7 * avg_kv_len) // sparse_block_kv, int(1.3 * avg_kv_len) // sparse_block_kv)
            * sparse_block_kv for _ in request_sizes]
    return request_sizes, lens


@pytest.mark.parametrize("fmt", ["fp4", "fp8"])
@pytest.mark.parametrize("sparse_block_kv", [8, 16])
@pytest.mark.parametrize("num_max_sparse_blocks", [4, 128, 512, 2048, 4096])
@pytest.mark.parametrize("num_q_tokens,avg_kv_len", [(1, 64), (3, 640), (37, 4096), (512, 16384),
                                                      (161, 131072), (64, 1 << 20)])
def test_matches_reference(fmt, sparse_block_kv, num_max_sparse_blocks, num_q_tokens, avg_kv_len):
    if avg_kv_len >= 131072 and num_max_sparse_blocks < 512:
        pytest.skip("covered by the shorter contexts")
    seed = num_q_tokens * 7919 + avg_kv_len + sparse_block_kv + num_max_sparse_blocks
    sizes, lens = random_requests(num_q_tokens, avg_kv_len, sparse_block_kv, seed)
    inputs = make_inputs(sizes, lens, sparse_block_kv, num_max_sparse_blocks, 64, seed)
    run_case(inputs, 64, fmt, sparse_block_kv)


@pytest.mark.parametrize("fmt,sparse_block_kv", [("fp4", 8), ("fp8", 16)])
@pytest.mark.parametrize("num_max_sparse_blocks", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("avg_kv_len", [4096, 131072, 1 << 20])
@pytest.mark.parametrize("num_q_tokens", [1, 4, 8, 16, 32, 64, 96])
def test_decode_batches(fmt, sparse_block_kv, num_max_sparse_blocks, avg_kv_len, num_q_tokens):
    """EP/DP decode: up to 16 requests of 6 tokens (MTP) per rank."""
    seed = num_q_tokens * 131 + avg_kv_len + num_max_sparse_blocks
    size = 6 if num_q_tokens >= 6 else num_q_tokens
    sizes, lens = random_requests(num_q_tokens, avg_kv_len, sparse_block_kv, seed, sizes=(size, size))
    run_case(make_inputs(sizes, lens, sparse_block_kv, num_max_sparse_blocks, 64, seed), 64, fmt, sparse_block_kv)


@pytest.mark.parametrize("cus_multiple,offset", [(1, -1), (1, 0), (1, 1), (4, 0), (4, 1)])
@pytest.mark.parametrize("request_size", [1, 3, 6])
def test_cta_count_boundaries(cus_multiple, offset, request_size):
    """CTAs are interleaved above one per CU and take tickets above four per CU."""
    num_q_tokens = num_sms_default() * cus_multiple + offset
    sizes, lens = random_requests(num_q_tokens, 20000, 8, num_q_tokens, sizes=(request_size, request_size))
    run_case(make_inputs(sizes, lens, 8, 2048, 64, num_q_tokens), 64, "fp4", 8)


@pytest.mark.parametrize("request_sizes,request_lens", [
    ([1], [0]), ([2], [0]), ([1, 1, 1], [1, 7, 9]), ([5], [1]), ([2, 3], [0, 8]),
    ([1], [1 << 20]), ([2], [(1 << 20) - 1]), ([100], [300]), ([1] * 300, [17] * 300),
    ([7] * 40, [5000] * 40), ([130], [64]),
])
def test_edge_cases(request_sizes, request_lens):
    inputs = make_inputs(request_sizes, request_lens, 8, 1024, 64, 1)
    run_case(inputs, 64, "fp4", 8)


@pytest.mark.parametrize("page_kv", [16, 48, 64, 256])
def test_page_sizes(page_kv):
    sizes, lens = random_requests(97, 30000, 16, page_kv)
    run_case(make_inputs(sizes, lens, 16, 1024, page_kv, page_kv, block_table_extra_cols=3), page_kv, "fp8", 16)


@pytest.mark.parametrize("num_sms", [1, 7, 80, 128, 256])
@pytest.mark.parametrize("num_q_tokens", [3, 200, 2000])
def test_schedule_widths(num_sms, num_q_tokens):
    sizes, lens = random_requests(num_q_tokens, 60000, 8, num_sms + num_q_tokens)
    run_case(make_inputs(sizes, lens, 8, 2048, 64, num_sms), 64, "fp4", 8, num_sms)


def test_many_short_and_long_requests():
    rng = random.Random(5)
    sizes = [rng.randint(1, 6) for _ in range(1500)]
    lens = [rng.choice([rng.randint(1, 512), rng.randint(1, 1 << 20)]) for _ in sizes]
    run_case(make_inputs(sizes, lens, 8, 512, 64, 5), 64, "fp4", 8)


def test_repeated_calls_and_graph_replay():
    from lightop import op

    sizes, lens = random_requests(256, 50000, 8, 11)
    inputs = make_inputs(sizes, lens, 8, 1024, 64, 11)
    for _ in range(3):
        run_case(inputs, 64, "fp4", 8)
    dev = {k: v.cuda() for k, v in inputs.items()}
    args = (dev["context_lens"], dev["block_table"], dev["indices"], 64, dev["sparse"], torch.int8, 8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        op.get_paged_sparse_mqa_logits_metadata(*args)  # warm up this stream's workspace
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = op.get_paged_sparse_mqa_logits_metadata(*args)
    for scale in (1, 0, 3):
        dev["context_lens"].copy_((inputs["context_lens"] // max(scale, 1)) * min(scale, 1))
        graph.replay()
        torch.cuda.synchronize()
        current = dict(inputs, context_lens=dev["context_lens"].cpu())
        splits, entries, num_waves, _ = decode(captured, 640, 8, num_sms_default())
        ref_splits, ref_entries, ref_waves = reference(
            current["context_lens"], current["block_table"], current["indices"], 64, current["sparse"],
            640, 8, num_sms_default())
        assert num_waves == ref_waves
        grouped, canon = canonicalize(splits, entries)
        ref_grouped, ref_canon = canonicalize(ref_splits, ref_entries)
        assert grouped == ref_grouped and canon == ref_canon


def test_invalid_arguments():
    from lightop import op

    inputs = {k: v.cuda() for k, v in make_inputs([2], [100], 8, 8, 64, 0).items()}
    args = [inputs["context_lens"], inputs["block_table"], inputs["indices"], 64, inputs["sparse"], torch.int8, 8]
    with pytest.raises(RuntimeError, match="sparse_block_kv"):
        op.get_paged_sparse_mqa_logits_metadata(*args[:6], 32)
    with pytest.raises(RuntimeError, match="page_kv"):
        op.get_paged_sparse_mqa_logits_metadata(*args[:3], 60, *args[4:])
    with pytest.raises(RuntimeError, match="qk_dtype"):
        op.get_paged_sparse_mqa_logits_metadata(*args[:5], torch.bfloat16, 8)
    with pytest.raises(RuntimeError, match="num_max_sparse_blocks"):
        op.get_paged_sparse_mqa_logits_metadata(*args[:4], inputs["sparse"][:, :6].contiguous(), *args[5:])
