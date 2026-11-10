# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paired timing for the masked EP act-and-mul quant kernels.

Both paths are timed in the same process on the same buffers: the call without
``tokens_per_expert`` keeps the legacy ``grid(E, 128)`` kernels, the call with it
takes the masked kernel. Every shape is also checked for exact equality on the
valid rows before it is timed, so a regression cannot hide behind a faster
number.

This script is not collected by pytest (the file name does not start with
``test_``); run it explicitly:

    python test/bench_fuse_silu_mul_quant_ep_masked.py
    python test/bench_fuse_silu_mul_quant_ep_masked.py --op fp8 --iters 200
    python test/bench_fuse_silu_mul_quant_ep_masked.py --shapes 385,8,2304 --counts uniform

Reported columns:

    E, T, d      shape of the [E, capacity, 2*d] input
    valid        sum(tokens_per_expert); rows the masked kernel actually walks
    pad%         1 - valid / (E*T), how much of the legacy grid is padding
    legacy_us    mean of the call without tokens_per_expert
    masked_us    mean of the call with tokens_per_expert
    speedup      legacy_us / masked_us

Timing uses CUDA events around ``--iters`` back-to-back calls after
``--warmup`` calls, so it includes the allocation the public wrapper performs
(the same way the numbers in the PR description were taken). Pass
``--pure-kernel`` to time the op on pre-allocated buffers instead.
"""

import argparse
import torch

from lightop.activation import fuse_silu_mul_quant_ep, fuse_silu_mul_fp8_quant_ep
from lightop import op

DEFAULT_SHAPES = [(64, 4, 2304), (128, 4, 2304), (385, 2, 2304), (385, 8, 2304)]


def build_counts(E, T, mode, seed):
    """Row counts per expert.

    ``skewed`` mirrors an EP decode batch: most experts hold 0-1 rows and a few
    hold many, which is where the legacy grid wastes the most work.
    ``uniform`` spreads the rows evenly for the lowest-padding case.
    """
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    if mode == 'uniform':
        counts = (torch.rand(E, generator=g) * (T + 1)).long()
    else:
        hot = max(1, E // 8)
        counts = torch.zeros(E, dtype=torch.long)
        counts[:hot] = (torch.rand(hot, generator=g) * (T + 1)).long()
        counts[hot:] = (torch.rand(E - hot, generator=g) < 0.5).long()
    return counts.clamp_(0, T).to(dtype=torch.int32, device='cuda')


def run_int8(x, counts=None):
    return fuse_silu_mul_quant_ep(x, tokens_per_expert=counts)


def run_fp8(x, counts=None):
    return fuse_silu_mul_fp8_quant_ep(x, fp8type=0, tokens_per_expert=counts, limit=10.0)


def time_calls(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0        # microseconds


def time_pure_kernel(op_name, x, counts, warmup, iters):
    E, T, H = x.shape
    d = H // 2
    q = torch.empty((E, T, d), dtype=torch.int8 if op_name == 'int8' else torch.uint8,
                    device=x.device)
    s = torch.empty((E, T, 1), dtype=torch.float32, device=x.device)
    if op_name == 'int8':
        call = lambda c: op.fuse_silu_mul_quant_ep(x, q, s, c, None, 1, -1)
    else:
        call = lambda c: op.fuse_silu_mul_fp8_quant_ep(x, q, s, 0, c, None, 1, -1, 10.0)
    return (time_calls(lambda: call(None), warmup, iters),
            time_calls(lambda: call(counts), warmup, iters))


def check_exact(fn, x, counts):
    d = x.shape[-1] // 2
    masked_q, masked_s = fn(x, counts)
    plain_q, plain_s = fn(x)
    masked_q = masked_q.view(torch.uint8) if masked_q.dtype != torch.int8 else masked_q
    plain_q = plain_q.view(torch.uint8) if plain_q.dtype != torch.int8 else plain_q
    for expert, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        if not torch.equal(masked_q[expert, :count, :d], plain_q[expert, :count, :d]):
            raise AssertionError(f'codes differ at expert {expert} (count {count})')
        if not torch.equal(masked_s[expert, :count], plain_s[expert, :count]):
            raise AssertionError(f'scale differs at expert {expert} (count {count})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--op', choices=['int8', 'fp8'], default='int8')
    ap.add_argument('--shapes', nargs='+', default=None,
                    help='E,T,d triplets; default %s' % (DEFAULT_SHAPES,))
    ap.add_argument('--counts', choices=['skewed', 'uniform'], default='skewed')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--pure-kernel', action='store_true',
                    help='time the op on pre-allocated buffers instead of the wrapper')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('requires a GPU')
    if args.op == 'int8':
        fn = run_int8
    else:
        fn = run_fp8

    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = []
        for spec in args.shapes:
            E, T, d = (int(v) for v in spec.split(','))
            shapes.append((E, T, d))

    torch.manual_seed(args.seed)
    print(f'op={args.op} counts={args.counts} iters={args.iters} warmup={args.warmup} '
          f'{"pure-kernel" if args.pure_kernel else "wrapper"}')
    print(f'device={torch.cuda.get_device_name(0)} arch={getattr(torch.cuda.get_device_properties(0), "gcnArchName", "n/a")} '
          f'hip={torch.version.hip}')
    print()
    print('%6s %4s %7s %8s %7s %12s %12s %9s' %
          ('E', 'T', 'd', 'valid', 'pad%', 'legacy_us', 'masked_us', 'speedup'))
    for E, T, d in shapes:
        x = torch.randn(E, T, 2 * d, dtype=torch.bfloat16, device='cuda')
        counts = build_counts(E, T, args.counts, args.seed)
        check_exact(fn, x, counts)
        if args.pure_kernel:
            legacy_us, masked_us = time_pure_kernel(args.op, x, counts, args.warmup, args.iters)
        else:
            legacy_us = time_calls(lambda: fn(x), args.warmup, args.iters)
            masked_us = time_calls(lambda: fn(x, counts), args.warmup, args.iters)
        valid = int(counts.sum())
        pad = 100.0 * (1 - valid / max(1, E * T))
        print('%6d %4d %7d %8d %6.1f%% %12.1f %12.1f %8.1fx' %
              (E, T, d, valid, pad, legacy_us, masked_us, legacy_us / masked_us))
        del x
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
