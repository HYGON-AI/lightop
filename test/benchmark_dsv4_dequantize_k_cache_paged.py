"""Benchmark DSV4 paged K-cache dequantization against SGLang."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from test_dsv4_dequantize_k_cache_paged import (
    _load_ops,
    _make_fixture,
)


PREFILL_WORKLOADS = (
    (256, 32768),
    (256, 32895),
    (64, 8192),
    (64, 16384),
    (64, 24576),
    (64, 32768),
    (2, 256),
    (2, 512),
    (2, 768),
    (2, 1024),
)
INDEX_DTYPES = (torch.int32, torch.int64)


def _elapsed_us(function, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / repeat


def _benchmark_case(
    page_size: int,
    rows: int,
    index_dtype: torch.dtype,
    *,
    warmup: int,
    repeat: int,
    rounds: int,
) -> dict:
    reference, candidate = _load_ops()
    fixture = _make_fixture(page_size, rows, index_dtype)

    def run_reference() -> None:
        reference(
            fixture["cache"],
            fixture["page_table"],
            page_size,
            out=fixture["reference_out"],
        )

    def run_candidate() -> None:
        candidate(
            fixture["cache"],
            fixture["page_table"],
            fixture["actual_out"],
            page_size,
        )

    for _ in range(warmup):
        run_reference()
        run_candidate()
    torch.cuda.synchronize()
    assert torch.equal(
        fixture["actual_out"].view(torch.int16),
        fixture["reference_out"].view(torch.int16),
    )

    samples = {"sglang": [], "lightop": []}
    for round_index in range(rounds):
        functions = (
            ("sglang", run_reference),
            ("lightop", run_candidate),
        )
        if round_index % 2:
            functions = tuple(reversed(functions))
        for name, function in functions:
            samples[name].append(_elapsed_us(function, repeat))

    torch.cuda.synchronize()
    assert torch.equal(
        fixture["actual_out"].view(torch.int16),
        fixture["reference_out"].view(torch.int16),
    )
    means = {
        name: statistics.fmean(values) for name, values in samples.items()
    }
    return {
        "page_size": page_size,
        "rows": rows,
        "index_dtype": str(index_dtype),
        "round_means_us": samples,
        "mean_us": means,
        "speedup": means["sglang"] / means["lightop"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.repeat, args.rounds) < 1:
        parser.error("warmup, repeat, and rounds must be positive")

    results = [
        _benchmark_case(
            page_size,
            rows,
            index_dtype,
            warmup=args.warmup,
            repeat=args.repeat,
            rounds=args.rounds,
        )
        for page_size, rows in PREFILL_WORKLOADS
        for index_dtype in INDEX_DTYPES
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
