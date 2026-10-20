"""Correctness, graph replay and paired timing for fused candidate masks."""
import argparse
import json
import statistics
import time

import torch
from lightop import op
from lightop.attention import dsv41_select_candidate_blocks


def baseline(x, lens, k):
    from sglang.kernels.ops.attention.dsv4.lightop_indexer import candidate_block_table
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        _expand_blocks_kernel, _hcu_topk_tables,
    )
    import triton
    b, w = x.shape
    table, _ = _hcu_topk_tables(b, (w + 7) // 8, x.device)
    ids, valid = candidate_block_table(x, lens, table, 8, k, 8)
    mask = torch.zeros((b, w), device=x.device, dtype=torch.bool)
    if b:
        _expand_blocks_kernel[(b,)](
            mask, ids, valid, k, w, mask.stride(0), ids.stride(0), valid.stride(0),
            BLOCK_K=triton.next_power_of_2(k), TABLE_LENS=True,
        )
    return mask, ids


def oracle(x, lens, k):
    data, lengths = x.cpu(), lens.cpu().tolist()
    mask = torch.zeros(data.shape, dtype=torch.bool)
    ids = torch.full((data.shape[0], k), -1, dtype=torch.int32)
    for row, length in enumerate(lengths):
        length = min(length, data.shape[1])
        n = (length + 7) // 8
        if not n:
            continue
        keys = torch.stack([data[row, i*8:min((i+1)*8, length)].amax() for i in range(n)])
        keys[-1] = torch.inf
        chosen = keys.argsort(descending=True, stable=True)[:k].sort().values
        ids[row, :chosen.numel()] = chosen.to(torch.int32)
        for i in chosen.tolist():
            mask[row, i*8:min((i+1)*8, data.shape[1])] = True
    return mask.to(x.device), ids.to(x.device)


def correctness():
    torch.manual_seed(17)
    cases = [(0, 1024, 2048), (9, 29, 7), (12, 1024, 2048),
             (12, 8192, 64), (4, 20000, 2048), (3, 65536, 2048),
             (3, 131072, 2048), (4, 1024, 1)]
    for b, w, k in cases:
        x = torch.randn(b, ((w+3)//4)*4, device='cuda')[:, :w]
        lengths = ([0, 1, 7, 8, 9, w-1, w, w+8] * 2)[:b]
        lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
        for tied in (False, True):
            if tied: x.fill_(0)
            got = dsv41_select_candidate_blocks(x, lens, k)
            for expected in (baseline(x, lens, k), oracle(x, lens, k)):
                for a, e in zip(got, expected): torch.testing.assert_close(a, e, rtol=0, atol=0)
    x = torch.randn(4, 20000, device='cuda')
    lens = torch.full((4,), 20000, device='cuda', dtype=torch.int32)
    for _ in range(3): dsv41_select_candidate_blocks(x, lens, 2048)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = dsv41_select_candidate_blocks(x, lens, 2048)
    for values in ([0, 1, 16000, 20000], [20000, 0, 9, 8]):
        lens.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        graph.replay()
        for a, e in zip(got, baseline(x, lens, 2048)):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    mask = torch.empty_like(x, dtype=torch.bool)
    ids = torch.empty((4, 64), dtype=torch.int32, device='cuda')
    for inputs in [(x.half(), lens, 8, mask, ids), (x, lens.long(), 8, mask, ids),
                   (x, lens, 4, mask, ids), (x, lens, 8, mask[:, :-1], ids),
                   (x[:, ::2], lens, 8, mask[:, ::2], ids)]:
        try: op.dsv41_select_candidate_blocks(*inputs)
        except RuntimeError: pass
        else: raise AssertionError('Invalid inputs were accepted')
    torch.cuda.synchronize()
    print('Correctness: exact old-composition and Torch parity; graph replay and 5 guards passed')


def timed(fn):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e4


def benchmark():
    results = []
    torch.manual_seed(17)
    for b, w in [(4096, 1024), (1024, 8192), (128, 20000), (128, 65536), (64, 131072)]:
        x = torch.randn(b, w, device='cuda')
        lens = torch.randint(0, w + 1, (b,), device='cuda', dtype=torch.int32)
        old, new = [], []
        for r in range(5):
            order = [('old', lambda: baseline(x, lens, 2048)),
                     ('new', lambda: dsv41_select_candidate_blocks(x, lens, 2048))]
            for name, fn in order[::1 if r % 2 == 0 else -1]:
                (old if name == 'old' else new).append(timed(fn))
        old_us, new_us = statistics.mean(old), statistics.mean(new)
        results.append(dict(shape=[b, w], dtype='float32', baseline_mean_us=old_us,
            fused_mean_us=new_us, speedup=old_us/new_us, p50_us=statistics.median(new),
            p90_us=sorted(new)[-1], cv=statistics.pstdev(new)/new_us,
            effective_GB_s=b*(5*w+4*2048+4)/new_us/1000, old_samples=old, new_samples=new))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if args.profile_only:
        x = torch.randn(128, 65536, device='cuda')
        lens = torch.full((128,), 65536, dtype=torch.int32, device='cuda')
        for _ in range(10): dsv41_select_candidate_blocks(x, lens, 2048)
        torch.cuda.synchronize()
    else:
        correctness()
        if args.benchmark: benchmark()
