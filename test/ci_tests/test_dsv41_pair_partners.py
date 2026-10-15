# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact pair/ring-state tests and synchronized Torch comparisons.

HIP_VISIBLE_DEVICES=7 python test/test_dsv41_pair_partners.py --benchmark --json results.json
Tests use the installed package selected by the caller; they never stop services.
"""
import argparse
import json
from pathlib import Path
import statistics

import pytest
import torch


def torch_reference(kv, score, req, pos, pad, state, ring):
    """Mirror SGLang's _low_ratio_pair_partners, including sentinel mutation."""
    count, dim = kv.shape
    if count == 0:
        return kv, score
    read_pos = (pos - 1).masked_fill(pad, -1)
    read_loc = req * ring + read_pos % ring
    read_loc.masked_fill_(read_pos < 0, -1)
    carried = state[read_loc]
    adjacent = torch.zeros_like(pad)
    adjacent[1:] = ((req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1)
                    & ~pad[1:] & ~pad[:-1])
    partner_kv = torch.where(adjacent[:, None], torch.roll(kv, 1, 0), carried[:, :dim])
    partner_score = torch.where(adjacent[:, None], torch.roll(score, 1, 0), carried[:, dim:])
    keep = ~pad
    if count > ring:
        keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
    write_pos = pos.masked_fill(~keep, -1)
    write_loc = req * ring + write_pos % ring
    write_loc.masked_fill_(write_pos < 0, -1)
    state[write_loc] = torch.cat([kv, score], dim=-1)
    state[-1, :dim].zero_()
    state[-1, dim:].fill_(-torch.inf)
    return partner_kv, partner_score


def make_case(count, dim=512, ring=2, split=False, multi=False, padded=False,
              req_dtype=torch.int32, pos_dtype=torch.int64, start=3):
    packed = torch.randn((17 * ring + 1, 2 * dim), device='cuda')
    # Noncanonical old sentinel deliberately checks read-before-reset ordering.
    data = torch.randn((count, 2 * dim), device='cuda')
    kv, score = data[:, :dim], data[:, dim:]
    if not split:
        kv, score = kv.contiguous(), score.contiguous()
    req = torch.zeros(count, dtype=req_dtype, device='cuda')
    pos = torch.arange(start, count + start, dtype=pos_dtype, device='cuda')
    pad = torch.zeros(count, dtype=torch.bool, device='cuda')
    if multi and count:
        step = max(1, (count + 6) // 7)
        req = torch.div(torch.arange(count, device='cuda'), step, rounding_mode='floor').to(req_dtype)
        pos = (torch.arange(count, device='cuda') % step + start).to(pos_dtype)
    if padded and count:
        pad[max(0, count - min(3, count)):] = True
    return kv, score, req, pos, pad, packed, ring


def get_op():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires a HIP GPU')
    from lightop.attention import dsv41_pair_partners
    return dsv41_pair_partners


def compare_case(op, case):
    kv, score, req, pos, pad, state, ring = case
    expected_state = state.clone()
    expected = torch_reference(kv, score, req, pos, pad, expected_state, ring)
    actual = op(kv, score, req, pos, pad, state, ring)
    for output, reference in zip((*actual, state), (*expected, expected_state)):
        torch.testing.assert_close(output, reference, atol=0, rtol=0, equal_nan=True)


def test_correctness():
    op = get_op()
    torch.manual_seed(17)
    for count in (0, 1, 2, 5, 17, 73, 80, 512, 4096, 32768):
        for ring in (2, 8, 16):
            for split in (False, True):
                compare_case(op, make_case(count, ring=ring, split=split))
                compare_case(op, make_case(count, ring=ring, split=split, multi=True,
                                          padded=True, start=0, req_dtype=torch.int64,
                                          pos_dtype=torch.int32))
    for dim in (1, 127, 513):
        compare_case(op, make_case(73, dim=dim, ring=8, split=True, multi=True))
    for req_type in (torch.int32, torch.int64):
        for pos_type in (torch.int32, torch.int64):
            compare_case(op, make_case(80, req_dtype=req_type, pos_dtype=pos_type))
    for start in (-3, -1):
        for ring in (2, 8):
            compare_case(op, make_case(5, ring=ring, split=True, start=start))


def test_stateful_chunks_and_stream():
    op = get_op()
    case = make_case(80, ring=2, split=True)
    kv, score, req, pos, pad, state, ring = case
    expected_state = state.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(4):
        expected = torch_reference(kv, score, req, pos, pad, expected_state, ring)
        with torch.cuda.stream(stream):
            actual = op(kv, score, req, pos, pad, state, ring)
        torch.cuda.current_stream().wait_stream(stream)
        for output, reference in zip((*actual, state), (*expected, expected_state)):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
        pos.add_(80)
        stream.wait_stream(torch.cuda.current_stream())


def test_graph_replay():
    op = get_op()
    kv, score, req, pos, pad, state, ring = make_case(73, split=True)
    for _ in range(3):
        op(kv, score, req, pos, pad, state, ring)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(kv, score, req, pos, pad, state, ring)
    for iteration in range(3):
        state.normal_()
        req.fill_(iteration)
        pos.add_(80)
        kv.add_(1)
        score.sub_(1)
        expected_state = state.clone()
        expected = torch_reference(kv, score, req, pos, pad, expected_state, ring)
        graph.replay()
        for output, reference in zip((*actual, state), (*expected, expected_state)):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)


def test_invalid_metadata():
    op = get_op()
    case = list(make_case(5))
    for index, invalid in ((0, case[0].cpu()), (1, case[1].to(torch.float16)),
                           (2, case[2].float()), (3, case[3][::2]),
                           (4, case[4].to(torch.int32)), (5, case[5][:, :-1]), (6, 0)):
        wrong = case.copy()
        wrong[index] = invalid
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            op(*wrong)


def benchmark():
    op = get_op()
    results = []
    for count in (1, 5, 73, 80, 512, 4096, 32768):
        for split in (False, True):
            case = make_case(count, split=split)
            torch_state = case[5].clone()
            native_state = case[5].clone()
            def original():
                return torch_reference(*case[:5], torch_state, case[6])
            def candidate():
                return op(*case[:5], native_state, case[6])
            samples = {'torch': [], 'lightop': []}
            for run in (original, candidate):
                for _ in range(10):
                    run()
            for round_id in range(5):
                order = (('torch', original), ('lightop', candidate))
                for name, run in (order if round_id % 2 == 0 else reversed(order)):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        output = run()
                    end.record()
                    torch.cuda.synchronize()
                    samples[name].append(start.elapsed_time(end) * 10)
            means = {name: statistics.mean(values) for name, values in samples.items()}
            logical_bytes = (4 * count * 512 + 2 * min(count, 2) * 512 + 2 * 512) * 4
            row = dict(tokens=count, D=512, ring=2, split=split, mean_us=means,
                       speedup=means['torch']/means['lightop'], samples_us=samples,
                       logical_bytes=logical_bytes,
                       effective_GB_s={name: logical_bytes / value / 1000 for name, value in means.items()},
                       cv={name: statistics.stdev(values)/means[name] for name, values in samples.items()},
                       p50_us={name: statistics.median(values) for name, values in samples.items()})
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    for test in (test_correctness, test_stateful_chunks_and_stream, test_graph_replay, test_invalid_metadata):
        test()
        print(test.__name__ + ': PASS', flush=True)
    result = dict(correctness='PASS', reference='SGLang Torch pair/ring composition',
                  dtype='float32', rows=benchmark() if args.benchmark else [])
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
