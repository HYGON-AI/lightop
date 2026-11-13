# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in router benchmark; see README_minimax_router_gemm.md."""

import argparse
import json
import statistics

import torch

from lightop.moe import bf16_router_gemm


def measure(fn, mode, warmup, iterations, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    stream.synchronize()
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                output = fn()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(output, fn())
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if graph is not None:
                graph.replay()
            else:
                for _ in range(iterations):
                    fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.pstdev(samples),
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations and repeats must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("requires HIP and a gfx938 GPU")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    if device.gcnArchName.split(":")[0] != "gfx938":
        parser.error("requires gfx938")
    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(64, 6144, dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn(128, 6144, dtype=torch.bfloat16, device="cuda", generator=gen)
    reference = (x.double() @ weight.double().T).float()
    operations = {
        "router_bf16_in_fp32_out": lambda: bf16_router_gemm(x, weight),
        "torch_bf16_in_bf16_out": lambda: x @ weight.T,
        "torch_fp32_in_fp32_out_including_casts": lambda: x.float() @ weight.float().T,
    }
    tolerances = [(1e-5, 3e-4), (4e-3, 3e-4), (1e-5, 3e-4)]
    errors = {}
    for (name, fn), (rtol, atol) in zip(operations.items(), tolerances):
        output = fn()
        torch.testing.assert_close(output.float(), reference, rtol=rtol, atol=atol)
        errors[name] = {
            "output_dtype": str(output.dtype),
            "max_abs_error": (output.float() - reference).abs().max().item(),
        }
    results = {}
    for mode in ("eager", "graph"):
        results[mode] = {
            name: measure(fn, mode, args.warmup, args.iterations, args.repeats)
            for name, fn in operations.items()
        }
    print(json.dumps({
        "device": device.name, "arch": device.gcnArchName,
        "torch": torch.__version__, "hip": torch.version.hip,
        "x_shape": list(x.shape), "weight_shape": list(weight.shape),
        "seed": 42, **vars(args), "correctness": errors, "timings": results,
    }, indent=2))


if __name__ == "__main__":
    main()
