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
from lightop.attention import dsv41_rope_fake_quant_store
from lightop.rmsnorm import dsv41_pool_pairs_norm


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
    """Torch oracle: BF16 RoPE, per-16 FP4 round-trip, then V4 cache packing."""
    if x.shape[0] == 0:
        return
    tail = torch.view_as_complex(x[:, -64:].float().reshape(-1, 32, 2))
    rotated = torch.view_as_real(tail * freqs).flatten(-2).to(x.dtype)
    blocks = torch.cat((x[:, :-64], rotated), dim=-1).float().reshape(-1, 32, 16)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) * (1. / 6.)).clamp(2**-9, 448.)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = (blocks / scale).clamp(-6., 6.)
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2., .5, torch.where(magnitude < 4., 1., 2.))
    rounded = torch.round(magnitude / step) * step * torch.sign(scaled)
    latent = (rounded * scale).flatten(-2).to(x.dtype)

    # V4 pages contain 576 data bytes/token and a separate 8-byte scale row.
    # Only seven scale bytes are written; the eighth and page padding stay intact.
    nope = latent[:, :448].float().reshape(-1, 7, 64)
    # Match DTK's HIP_FP8_TYPE_FNUZ clamp, including HCU's native E4M3 encoding.
    gfx = torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0] if torch.version.hip else ''
    fp8_max = 224. if gfx and gfx not in ('gfx950', 'gfx1200', 'gfx1201') else 448.
    fp8_dtype = torch.float8_e4m3fnuz if gfx == 'gfx942' else torch.float8_e4m3fn
    raw_scale = nope.abs().amax(dim=-1).clamp_min(1e-4) / fp8_max
    bits = raw_scale.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(torch.int32)
    physical_scale = (exponent << 23).view(torch.float32)
    data = (nope / physical_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    data = data.view(torch.uint8).flatten(-2)
    rope = latent[:, 448:].contiguous().view(torch.uint8)
    page, offset = loc.long() // page_size, loc.long() % page_size
    cache[page[:, None], offset[:, None] * 576 + torch.arange(448, device=x.device)] = data
    cache[page[:, None], offset[:, None] * 576 + 448 + torch.arange(128, device=x.device)] = rope
    cache[page[:, None], page_size * 576 + offset[:, None] * 8 + torch.arange(7, device=x.device)] = exponent.to(torch.uint8)


def baseline(x, freqs, cache, loc, page_size):
    reference(x, freqs, cache, loc, page_size)


def candidate(x, freqs, cache, loc, page_size):
    dsv41_rope_fake_quant_store(x, freqs, cache, loc, page_size)


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
                          cache_bytes='exact',
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


def pool_norm_inputs(n, strided=False, weight_dtype=torch.float32):
    width = 1024 if strided else 512
    tensors = [torch.randn(n, width, device='cuda')[:, :512] for _ in range(4)]
    weight = (torch.rand(512, device='cuda') + .5).to(weight_dtype)
    return (*tensors, weight, 1e-6)


def pool_norm_reference(previous, current, previous_score, score, weight, eps):
    probabilities = torch.stack((previous_score, score), dim=1).softmax(dim=1)
    pooled = (torch.stack((previous, current), dim=1) * probabilities).sum(dim=1)
    pooled = pooled.bfloat16().float()
    inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + eps)
    return ((pooled * inv_rms) * weight.float()).bfloat16()


def assert_pool_norm_close(actual, expected):
    """Allow at most one BF16 representable step; signed zeros compare equal."""
    if (actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16
            or actual.shape != expected.shape or actual.device != expected.device):
        raise AssertionError('Pool/norm results must have matching BF16 metadata')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('Pool/norm results must be finite')
    actual_bits = actual.contiguous().view(torch.int16).to(torch.int32)
    expected_bits = expected.contiguous().view(torch.int16).to(torch.int32)
    actual_order = torch.where(actual_bits < 0, -32768 - actual_bits, actual_bits)
    expected_order = torch.where(expected_bits < 0, -32768 - expected_bits, expected_bits)
    ulps = (actual_order - expected_order).abs()
    torch.testing.assert_close(ulps, torch.zeros_like(ulps), rtol=0, atol=1)


def pool_norm_main(args):
    torch.manual_seed(20261024)
    candidate = dsv41_pool_pairs_norm if not args.baseline_only else pool_norm_reference
    if args.profile_only:
        for n in args.rows:
            data = pool_norm_inputs(n, strided=True)
            for _ in range(10):
                candidate(*data)
            torch.cuda.synchronize()
        return
    cases = 0
    for n in tuple(dict.fromkeys((0, 1,63, 64, 65, 513, *args.rows))):
        for strided in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                data = pool_norm_inputs(n, strided, dtype)
                before = [x.clone() for x in data[:-1]]
                expected, actual = pool_norm_reference(*data), candidate(*data)
                if not torch.equal(expected, actual):
                    print(json.dumps(dict(phase='pool_norm_difference', rows=n, strided=strided,
                                          weight_dtype=str(dtype), mismatches=(expected != actual).sum().item(),
                                          max_abs=(expected.float() - actual.float()).abs().max().item())), flush=True)
                assert_pool_norm_close(actual, expected)
                for original, unchanged in zip(data[:-1], before):
                    torch.testing.assert_close(original, unchanged, rtol=0, atol=0)
                cases += 1
    for magnitude in (0., 1e-6, 1., 1000.):
        data = list(pool_norm_inputs(65, True))
        data[0].fill_(magnitude)
        data[1].fill_(-magnitude)
        data[2].fill_(1000. if magnitude else 0.)
        data[3].fill_(-1000. if magnitude else 0.)
        assert_pool_norm_close(candidate(*data), pool_norm_reference(*data))
        cases += 1
    if not args.baseline_only:
        data = pool_norm_inputs(65)
        for replacement, index in ((data[0].double(), 0), (data[0][:, ::2], 0),
                                   (data[4][::2], 4), (0., 5)):
            invalid = list(data)
            invalid[index] = replacement
            try:
                candidate(*invalid)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unsupported pool/norm metadata accepted')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected, actual = pool_norm_reference(*data), candidate(*data)
        torch.cuda.current_stream().wait_stream(stream)
        assert_pool_norm_close(actual, expected)
        cases += 1
    print(json.dumps(dict(phase='pool_norm_correctness', cases=cases, max_bf16_ulp=1)), flush=True)
    if args.correctness_only:
        return
    for n in args.rows:
        for strided in (False, True):
            data = pool_norm_inputs(n, strided)
            for repeat in range(3):
                variants = [('baseline', pool_norm_reference)]
                if not args.baseline_only:
                    variants.append(('candidate', candidate))
                if repeat % 2:
                    variants.reverse()
                results = {}
                for label, fn in variants:
                    stats = timing(lambda: fn(*data))
                    stats['effective_GB_s'] = n * 512 * (4 * 4 + 2) / stats['mean_us'] / 1000
                    results[label] = stats
                print(json.dumps(dict(phase='pool_norm_benchmark', rows=n, strided=strided,
                                      repeat=repeat, **results)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--pool-norm', action='store_true')
    parser.add_argument('--rows', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192, 32768])
    args = parser.parse_args()
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip,
                          gfx=torch.cuda.get_device_properties(0).gcnArchName,
                          selected_card=os.environ.get('HIP_VISIBLE_DEVICES'),
                          lightop=lightop.__file__, reference_impl='torch')), flush=True)
    if args.pool_norm:
        pool_norm_main(args)
        return
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
