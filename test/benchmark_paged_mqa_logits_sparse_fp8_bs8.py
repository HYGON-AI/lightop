"""Benchmark the GLM-5.2 BS8/MTP314 grouped FP8 Page-MQA shape.

This is a hardware performance harness, not a CI latency assertion.  Run it
against two LightOp installations with the same command to compare kernels.
"""

import argparse
import math
from collections.abc import Callable

import torch

from lightop import attention as lightop_attention


def _make_inputs(
    requests: int,
    group_size: int,
    context_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    rows = requests * group_size
    capacity = math.ceil((context_len + group_size - 1) / 64) * 64
    pages_per_request = capacity // 64
    num_pages = requests * pages_per_request

    q = (torch.randn(
        (rows, 1, 32, 128), device="cuda", dtype=torch.float32
    ) * 0.125).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, 32), device="cuda", dtype=torch.float32)

    # Packed page ABI: 64*128 FP8 bytes, then 64 float scales.  Restrict raw
    # FP8 bytes to finite E4M3FN encodings so NaN propagation does not distort
    # the production-shape timing comparison.
    kv_cache = torch.empty(
        (num_pages, 64, 1, 132), device="cuda", dtype=torch.uint8
    )
    packed_bytes = kv_cache.view(num_pages, -1)
    packed_bytes[:, : 64 * 128].random_(0, 0x7F)
    packed_bytes[:, 64 * 128 :].view(torch.float32).uniform_(0.0005, 0.002)

    request_pages = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    ).view(requests, pages_per_request)
    block_table = request_pages.repeat_interleave(group_size, dim=0).contiguous()
    row_offsets = torch.arange(
        group_size, device="cuda", dtype=torch.int32
    ).repeat(requests)
    context_lens = row_offsets.add(context_len)
    return q, kv_cache, weights, context_lens, block_table, capacity


def _elapsed_us(fn: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _run_context(
    api: str,
    context_len: int,
    requests: int,
    group_size: int,
    persistent_ctas: int,
    warmup: int,
    iterations: int,
    target_steps: int,
) -> None:
    q, kv_cache, weights, context_lens, block_table, capacity = _make_inputs(
        requests, group_size, context_len
    )

    if api == "grouped":
        grouped_op = lightop_attention.page_mqa_logits_sparse_mask_grouped

        def run_mqa() -> object:
            return grouped_op(
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                schedule_meta=None,
                max_context_len=capacity,
                clean_logits=True,
                num_warps=4,
                persistent_ctas=persistent_ctas,
                group_size=group_size,
            )

    else:
        persistent_op = lightop_attention.paged_mqa_logits_length_masked

        def run_mqa() -> object:
            return persistent_op(
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                schedule_meta=None,
                max_context_len=capacity,
                clean_logits=True,
                num_warps=4,
                persistent_ctas=128,
            )

    for _ in range(warmup):
        run_mqa()
    torch.cuda.synchronize()
    eager_us = _elapsed_us(run_mqa, iterations)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run_mqa()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    graph_us = _elapsed_us(graph.replay, iterations)

    # Keep captured outputs live until all replays and timing events finish.
    assert graph_outputs is not None
    print(
        f"api={api} context={context_len} capacity={capacity} "
        f"rows={requests * group_size} group={group_size} "
        f"persistent_ctas={persistent_ctas} "
        f"eager_us={eager_us:.3f} graph_us={graph_us:.3f} "
        f"estimated_output_mqa_ms={graph_us * target_steps / 1000.0:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api", choices=("grouped", "persistent"), default="grouped"
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--persistent-ctas", type=int, default=0)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[45000, 45500, 46000]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-tokens", type=int, default=1000)
    parser.add_argument("--accepted-tokens", type=float, default=3.3)
    parser.add_argument("--hit-rate", type=float, default=0.77)
    args = parser.parse_args()

    if args.output_tokens <= 0:
        parser.error("--output-tokens must be positive")
    if args.accepted_tokens <= 0:
        parser.error("--accepted-tokens must be positive")
    if not 0.0 <= args.hit_rate <= 1.0:
        parser.error("--hit-rate must be in [0, 1]")
    target_steps = math.ceil(args.output_tokens / args.accepted_tokens)

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", props.name).split(":")[0]
    print(
        f"device={arch}/{props.multi_processor_count}CU "
        f"requests={args.requests} mtp_rows={args.group_size} "
        f"mtp=3/1/4 output={args.output_tokens} "
        f"accept={args.accepted_tokens} hit_rate={args.hit_rate} "
        f"estimated_target_steps={target_steps}"
    )
    for context_len in args.contexts:
        _run_context(
            args.api,
            context_len,
            args.requests,
            args.group_size,
            args.persistent_ctas,
            args.warmup,
            args.iterations,
            target_steps,
        )


if __name__ == "__main__":
    main()
