# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact Engram Prefill checks and synchronized Torch/HCU comparisons."""

import argparse
import itertools
import json
import statistics
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="requires a HIP GPU"
)


def reference(input_ids, positions, history, row, starts, token_map, multipliers,
              primes, offsets, num_real, pad_id, image_token_id=None, mm_pad_shift=1000000):
    # SGLang EngramHasher._torch_hash_ids / compute_engram_hash_ids (EXTEND).
    t = torch.arange(num_real, device=input_ids.device)
    shifts = torch.arange(4, device=input_ids.device)
    offset = t - starts[row]
    from_batch = shifts[None, :] <= offset[:, None]
    in_batch = input_ids.long()[(t[:, None] - shifts).clamp_min(0)]
    hist_col = (3 - shifts[None, :] + offset[:, None]).clamp(0, 2)
    from_hist = history.long()[row].gather(1, hist_col)
    tokens = torch.where(from_batch, in_batch, from_hist)
    if num_real < input_ids.numel():
        tokens = torch.cat((tokens, tokens.new_zeros(input_ids.numel() - num_real, 4)))
    blocked = positions.long()[:, None] < shifts
    if num_real < input_ids.numel():
        blocked[num_real:] = True
    if image_token_id is not None:
        tokens = tokens.masked_fill(tokens >= mm_pad_shift, image_token_id)
        blocked = (blocked | (tokens == image_token_id)).int().cummax(-1).values.bool()
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed[:, None, :] * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, 4):
        rolling = rolling ^ products[..., i]
        hashes.append(rolling[..., None] % primes[:, i-1])
    return torch.cat(hashes, -1) + offsets, tokens


def inputs(lengths, padding=0, dtype=torch.int64, position_dtype=torch.int64,
           history_dtype=torch.int32, image=False):
    n = sum(lengths) + padding
    lens = torch.tensor(lengths, device='cuda')
    starts = (lens.cumsum(0) - lens).long()
    row = torch.repeat_interleave(torch.arange(len(lengths), device='cuda'), lens)
    positions = torch.cat([torch.arange(k, device='cuda') + (19 if i % 2 else 0)
                           for i, k in enumerate(lengths)] + [torch.zeros(padding, device='cuda')]).to(position_dtype)
    ids = torch.randint(0, 129280, (n,), device='cuda', dtype=dtype)
    history = torch.randint(0, 129280, (len(lengths), 3), device='cuda', dtype=history_dtype)
    token_map = torch.randperm(129280, device='cuda') % 99092
    multipliers = torch.tensor([[46561817735917, 87622468935781, 19095818674273, 7206622809691],
                                [34243927720121, 82121828472819, 19621999308271, 66823698728231]], device='cuda')
    primes = torch.arange(16000001, 16000049, device='cuda').reshape(2, 3, 8)
    offsets = torch.arange(24, device='cuda')[None, :].expand(2, -1).contiguous() * 16000100
    if image:
        ids[::7] = 1000007
        history[:, 1] = 1000010
    return (ids, positions, history, row, starts, token_map, multipliers, primes, offsets,
            sum(lengths), 2, 17 if image else None, 1000000)


def correctness(candidate, check_rejections=True):
    torch.manual_seed(20261026)
    cases = 0
    for lengths, padding in [([0], 0), ([0], 5), ([1], 0), ([2,0,1,7], 3),
                             ([63,64,65], 7), ([513,7], 0), ([4096], 0),
                             ([3,32765], 0), ([65536], 0)]:
        for dtype, pos_type, hist_type, image in itertools.product(
                (torch.int32, torch.int64), (torch.int32, torch.int64),
                (torch.int32, torch.int64), (False, True)):
            data = inputs(lengths, padding, dtype, pos_type, hist_type, image)
            before = [x.clone() for x in data[:9]]
            expected = reference(*data)
            actual = candidate(*data)
            for a, b in zip(actual, expected):
                assert a.dtype == torch.int64
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for a, b in zip(data[:9], before):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            # Caller history commit stays separate; verify its values on ragged requests.
            last = (data[4] + torch.tensor(lengths, device='cuda') - 1).clamp(0, max(data[0].numel()-1, 0))
            if data[0].numel():
                for tokens in (actual[1], expected[1]):
                    committed = tokens[last, :3].flip(-1).to(hist_type)
                    if tokens is actual[1]:
                        actual_history = committed
                    else:
                        torch.testing.assert_close(actual_history, committed, atol=0, rtol=0)
            cases += 1
    data = inputs([513], 3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected, actual = reference(*data), candidate(*data)
    torch.cuda.current_stream().wait_stream(stream)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if check_rejections:
        data = inputs([17])
        for index, bad in [(0, data[0].float()), (0, data[0].cpu()),
                           (1, torch.empty(34, device='cuda', dtype=torch.int64)[::2]),
                           (7, data[7][:1].contiguous())]:
            invalid = list(data)
            invalid[index] = bad
            try:
                candidate(*invalid)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unsupported input was accepted')
    torch.cuda.synchronize()
    print(json.dumps(dict(phase='correctness', cases=cases+1, exact=True)), flush=True)


def test_engram_hash_ids_hcu():
    from lightop.engram import engram_hash_ids_hcu

    correctness(engram_hash_ids_hcu)


def timing(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1000)
        gpu.append(start_event.elapsed_time(end_event) * 1000)
    return dict(mean_us=statistics.mean(wall), p50_us=statistics.median(wall),
                p90_us=sorted(wall)[90], cv=statistics.stdev(wall)/statistics.mean(wall),
                gpu_event_us=statistics.mean(gpu))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--rows', type=int, nargs='+', default=[4096,8192,16384,32768,65536])
    args = parser.parse_args()
    candidate = reference
    if not args.baseline_only:
        from lightop.engram import engram_hash_ids_hcu
        candidate = engram_hash_ids_hcu
    if args.profile_only:
        for n in args.rows:
            data = inputs([n])
            for _ in range(10):
                candidate(*data)
            torch.cuda.synchronize()
        return
    correctness(candidate, check_rejections=not args.baseline_only)
    if args.correctness_only:
        return
    for n in args.rows:
        data = inputs([n])
        # Unique logical traffic: IDs/positions, four map reads, hashes/tokens.
        # This excludes repeated cache reads and allocator/temporary traffic.
        logical_bytes = n * (2 + 4 + 48 + 4) * 8
        for round_id in range(3):
            order = [('torch', reference), ('hcu', candidate)]
            if round_id % 2:
                order.reverse()
            for name, fn in order:
                result = timing(lambda: fn(*data))
                print(json.dumps(dict(phase='benchmark', tokens=n, round=round_id, method=name,
                                      **result, logical_GBs=logical_bytes/result['mean_us']/1000)), flush=True)


if __name__ == '__main__':
    main()
