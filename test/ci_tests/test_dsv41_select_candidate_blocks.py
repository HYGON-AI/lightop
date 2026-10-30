# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness, graph replay and paired timing for fused candidate masks."""
import argparse
import json
import statistics
import time

import pytest
import torch
from lightop.attention import dsv41_select_candidate_blocks


def baseline(x, lens, k):
    """Independent GPU reference without SGLang's private implementation."""
    b, w = x.shape
    blocks = (w + 7) // 8
    ids = torch.full((b, k), -1, dtype=torch.int32, device=x.device)
    if not b or not blocks:
        return torch.zeros_like(x, dtype=torch.bool), ids
    lengths = lens.clamp(0, w)
    valid_blocks = (lengths + 7) // 8
    values = torch.full((b, blocks * 8), -torch.inf, device=x.device, dtype=x.dtype)
    positions = torch.arange(w, device=x.device)
    values[:, :w] = x.masked_fill(positions[None, :] >= lengths[:, None], -torch.inf)
    scores = values.reshape(b, blocks, 8).amax(-1)
    block_ids = torch.arange(blocks, device=x.device)
    scores.masked_fill_(block_ids[None, :] == valid_blocks[:, None] - 1, torch.inf)
    selected = scores.argsort(dim=1, descending=True, stable=True)[:, :min(k, blocks)]
    selected = selected.masked_fill(selected >= valid_blocks[:, None], blocks).sort(1).values
    live = selected < blocks
    ids[:, :selected.shape[1]] = selected.masked_fill(~live, -1).to(torch.int32)
    active = torch.zeros((b, blocks), dtype=torch.int32, device=x.device)
    active.scatter_add_(1, selected.clamp_max(blocks - 1), live.to(torch.int32))
    return active.bool().repeat_interleave(8, dim=1)[:, :w], ids


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
    torch.cuda.synchronize()
    print('Correctness: exact Torch GPU and scalar-oracle parity; graph replay passed')


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
            order = [('torch', lambda: baseline(x, lens, 2048)),
                     ('new', lambda: dsv41_select_candidate_blocks(x, lens, 2048))]
            for name, fn in order[::1 if r % 2 == 0 else -1]:
                (old if name == 'torch' else new).append(timed(fn))
        old_us, new_us = statistics.mean(old), statistics.mean(new)
        results.append(dict(shape=[b, w], dtype='float32', baseline_mean_us=old_us,
            fused_mean_us=new_us, speedup=old_us/new_us, p50_us=statistics.median(new),
            p90_us=sorted(new)[-1], cv=statistics.pstdev(new)/new_us,
            effective_GB_s=b*(5*w+4*2048+4)/new_us/1000, torch_samples=old, new_samples=new))
    print(json.dumps(results, indent=2))


def test_select_candidate_blocks():
    correctness()


@pytest.mark.parametrize('case, error', [
    ('logits_dtype', 'logits must be FP32'),
    ('logits_rank', 'logits must be FP32'),
    ('logits_stride', 'logits must be FP32'),
    ('seq_lens_dtype', 'seq_lens must be contiguous int32'),
    ('seq_lens_size', 'seq_lens must be contiguous int32'),
    ('seq_lens_stride', 'seq_lens must be contiguous int32'),
    ('seq_lens_device', 'seq_lens must be contiguous int32'),
    ('block_size', 'block_size must be 8'),
    ('topk_blocks_zero', 'topk'),
    ('topk_blocks_negative', 'negative dimension'),
])
def test_select_candidate_blocks_invalid_parameters(case, error):
    x = torch.randn(4, 1024, device='cuda')
    lens = torch.full((4,), 1024, dtype=torch.int32, device='cuda')
    topk_blocks, block_size = 64, 8
    if case == 'logits_dtype':
        x = x.half()
    elif case == 'logits_rank':
        x = x.unsqueeze(0)
    elif case == 'logits_stride':
        x = x[:, ::2]
    elif case == 'seq_lens_dtype':
        lens = lens.long()
    elif case == 'seq_lens_size':
        lens = lens[:-1]
    elif case == 'seq_lens_stride':
        lens = lens.repeat_interleave(2)[::2]
    elif case == 'seq_lens_device':
        lens = lens.cpu()
    elif case == 'block_size':
        block_size = 4
    elif case == 'topk_blocks_zero':
        topk_blocks = 0
    elif case == 'topk_blocks_negative':
        topk_blocks = -1

    with pytest.raises(RuntimeError, match=error):
        dsv41_select_candidate_blocks(x, lens, topk_blocks, block_size=block_size)


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
