# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Correctness-first checks and synchronized Prefill fusion comparisons."""
import argparse
import json
import os
import statistics
import time

import torch
import lightop

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_compressed_kv
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4, rope_tail


def inputs(n, index_dtype=torch.int64, page_size=64):
    x = torch.randn(n, 512, device='cuda', dtype=torch.bfloat16)
    phase = torch.randn(n, 32, device='cuda')
    freqs = torch.polar(torch.ones_like(phase), phase)
    loc = (torch.randperm(n, device='cuda') + 1).to(index_dtype)
    page_bytes = ((page_size * 584 + 575) // 576) * 576
    cache = torch.full(((n + page_size) // page_size + 1, page_bytes), 165,
                       dtype=torch.uint8, device='cuda')
    return x, freqs, loc, cache


def reference(x, freqs, cache, loc, page_size):
    latent = fake_quant_compressed_kv(rope_tail(x, freqs, 64))
    fused_store_cache(latent, cache, loc, page_size=page_size, type='flashmla')


def baseline(x, freqs, cache, loc, page_size):
    latent = _rope_fq4(x, freqs, 64, compressed_kv=True)
    fused_store_cache(latent, cache, loc, page_size=page_size, type='flashmla')


def candidate(x, freqs, cache, loc, page_size):
    lightop.dsv41_rope_fake_quant_store(x, freqs, cache, loc, page_size)


def correctness(baseline_only):
    torch.manual_seed(20261024)
    cases = 0
    for n in (0, 1, 63, 64, 65, 513):
        for index_dtype in (torch.int32, torch.int64):
            for page_size in (64, 128):
                x, freqs, loc, cache = inputs(n, index_dtype, page_size)
                expected, current = cache.clone(), cache.clone()
                reference(x, freqs, expected, loc, page_size)
                baseline(x, freqs, current, loc, page_size)
                torch.testing.assert_close(current, expected, atol=0, rtol=0)
                if not baseline_only:
                    actual = cache.clone()
                    candidate(x, freqs, actual, loc, page_size)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                cases += 1
    for magnitude in (0., 2**-10, .5, 1., 2., 4., 6., 100., 1000.):
        x, freqs, loc, cache = inputs(65)
        x.fill_(magnitude)
        x[:, 1::2].neg_()
        expected, actual = cache.clone(), cache.clone()
        reference(x, freqs, expected, loc, 64)
        (baseline if baseline_only else candidate)(x, freqs, actual, loc, 64)
        if baseline_only and not torch.equal(actual, expected):
            decoded = []
            for packed in (expected, actual):
                out = torch.empty((loc.numel(), 1, 512), device='cuda', dtype=torch.bfloat16)
                lightop.dequantize_k_cache_paged_hcu(packed, loc, out, 64, False)
                decoded.append(out)
            print(json.dumps(dict(phase='baseline_edge_difference', magnitude=magnitude,
                                  mismatched_cache_bytes=(actual != expected).sum().item(),
                                  decoded_max_abs=(decoded[0].float() - decoded[1].float()).abs().max().item())), flush=True)
            # Diagnose the old AOT implementation separately. The candidate
            # still must match the literal reference cache byte for byte.
            torch.testing.assert_close(decoded[0], decoded[1], atol=0, rtol=0)
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        cases += 1
    if not baseline_only:
        # Unit scale and identity RoPE expose every FP4 tie-to-even boundary.
        x, freqs, loc, cache = inputs(3)
        ties = torch.tensor([6., .25, .75, 1.25, 1.75, 2.5, 3.5, 5.,
                             -6., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.],
                            dtype=torch.bfloat16, device='cuda')
        x.copy_(ties.repeat(32).expand_as(x))
        freqs.fill_(1)
        expected, actual = cache.clone(), cache.clone()
        reference(x, freqs, expected, loc, 64)
        before = x.clone()
        candidate(x, torch.view_as_real(freqs).flatten(-2), actual, loc, 64)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(x, before, atol=0, rtol=0)
        cases += 1
        # Negative slots must not write any cache bytes or modify the input.
        x, freqs, loc, cache = inputs(65)
        loc[::2] = -1
        keep = loc >= 0
        expected, actual = cache.clone(), cache.clone()
        reference(x[keep], freqs[keep], expected, loc[keep], 64)
        candidate(x, freqs, actual, loc, 64)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        cases += 1
        x, freqs, loc, cache = inputs(65)
        for bad_x in (x.float(), x[:, ::2], x.cpu()):
            try:
                candidate(bad_x, freqs, cache, loc, 64)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unsupported input was accepted')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        expected, actual = cache.clone(), cache.clone()
        with torch.cuda.stream(stream):
            reference(x, freqs, expected, loc, 64)
            candidate(x, freqs, actual, loc, 64)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    print(json.dumps(dict(phase='correctness', cases=cases,
                          cache_bytes='candidate exact' if not baseline_only else 'random exact; zero edge checked after dequantization',
                          candidate_checked=not baseline_only)), flush=True)


def timing(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    values = torch.tensor(samples, dtype=torch.float64)
    mean = statistics.mean(samples)
    return dict(mean_us=mean, p50_us=values.quantile(.5).item(),
                p90_us=values.quantile(.9).item(), cv=statistics.stdev(samples) / mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--rows', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192, 32768])
    args = parser.parse_args()
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip,
                          gfx=torch.cuda.get_device_properties(0).gcnArchName,
                          selected_card=os.environ.get('HIP_VISIBLE_DEVICES'),
                          lightop=lightop.__file__, sglang_reference=reference.__module__)), flush=True)
    if args.profile_only:
        for n in args.rows:
            x, freqs, loc, cache = inputs(n)
            fn = baseline if args.baseline_only else candidate
            for _ in range(10):
                fn(x, freqs, cache, loc, 64)
            torch.cuda.synchronize()
        return
    correctness(args.baseline_only)
    if args.correctness_only:
        return
    for n in args.rows:
        x, freqs, loc, cache = inputs(n)
        for repeat in range(3):
            results = {}
            variants = [('baseline', baseline)]
            if not args.baseline_only:
                variants.append(('candidate', candidate))
            if repeat % 2:
                variants.reverse()
            for name, fn in variants:
                stats = timing(lambda: fn(x, freqs, cache, loc, 64))
                stats['effective_GB_s'] = n * (1024 + 256 + 8 + 583) / stats['mean_us'] / 1000
                results[name] = stats
            print(json.dumps(dict(phase='benchmark', rows=n, repeat=repeat,
                                  dtype='bf16', page_size=64, **results)), flush=True)


if __name__ == '__main__':
    main()
