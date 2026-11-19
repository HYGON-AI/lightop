# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare indexed prefill Q preparation with the native gather + RoPE path."""

import argparse
import json
import statistics

import torch
from sgl_kernel import rope_fp4_fake_quant

from lightop.attention import dsv41_indexer_rope_fake_quant


def time_us(fn, iterations, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 4096, 8192, 32768])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens + [args.heads, args.iterations, args.repeats]) < 1:
        parser.error("all sizes and repeat counts must be positive")
    torch.manual_seed(11172026)
    for tokens in args.tokens:
        q = torch.randn(tokens, args.heads, 128, device="cuda", dtype=torch.bfloat16)
        phase = torch.randn(max(131072, tokens), 32, device="cuda")
        freqs = torch.polar(torch.ones_like(phase), phase)
        positions = torch.randint(len(freqs), (tokens,), device="cuda")
        baseline = lambda: rope_fp4_fake_quant(q, freqs[positions], 64, False)
        fused = lambda: dsv41_indexer_rope_fake_quant(q, freqs, positions)
        assert torch.equal(baseline().view(torch.int16), fused().view(torch.int16))
        old = time_us(baseline, args.iterations, args.repeats)
        new = time_us(fused, args.iterations, args.repeats)
        print(json.dumps({"tokens": tokens, "heads": args.heads,
                          "native_with_gather_us": old, "fused_us": new,
                          "speedup": old / new}), flush=True)


if __name__ == "__main__":
    main()
