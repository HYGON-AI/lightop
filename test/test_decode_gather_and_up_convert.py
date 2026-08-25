#!/usr/bin/env python3
"""Target-shape test for ``decode_gather_and_up_convert``.

The workload mirrors the decode tensors captured from GLM:

* physical KV cache: ``[33772, 64, 1, 656]`` uint8
* NSA Top-K indices: ``[48, 1, 2048]`` int32
* CUDA-graph output storage: ``[48, 2063, 512]`` bfloat16

``q=[48, 1, 64, 512]`` is intentionally not allocated because it is not an
input of this LightOp operator.  The test first checks exact BF16 correctness
against a PyTorch reference, then reports 10-warmup/100-repeat latency and
effective bandwidth as required by the LightOp test convention.
"""

import math
import os
import random
import statistics
from dataclasses import dataclass

import pytest
import torch

from lightop import op


SEED = 20260716
BATCH_SIZE = 48
NUM_CACHE_BLOCKS = 33772
BLOCK_SIZE = 64
NUM_CACHE_HEADS = 1
ENTRY_BYTES_NO_ROPE = 528
ENTRY_BYTES_WITH_ROPE = 656
ENTRY_BYTES = ENTRY_BYTES_WITH_ROPE
HEAD_DIM = 512
NUM_GROUPS = 4
GROUP_SIZE = HEAD_DIM // NUM_GROUPS
TOPK = 2048
WORKSPACE_TOPK = 2063
PACK_CHUNK_ROWS = 4096
REFERENCE_CHUNK_ROWS = 2048
WARMUP = 10
REPEAT = 100
ROUNDS = 5
OUTPUT_SENTINEL = -123.0
FP8_DTYPE = torch.float8_e4m3fn

KV_CACHE_SHAPE = (
    NUM_CACHE_BLOCKS,
    BLOCK_SIZE,
    NUM_CACHE_HEADS,
    ENTRY_BYTES,
)
INDICES_SHAPE = (BATCH_SIZE, 1, TOPK)
WORKSPACE_SHAPE = (BATCH_SIZE, WORKSPACE_TOPK, HEAD_DIM)
NUM_PHYSICAL_TOKENS = NUM_CACHE_BLOCKS * BLOCK_SIZE * NUM_CACHE_HEADS


@dataclass
class TestCase:
    kv_cache: torch.Tensor
    indices: torch.Tensor
    token_ids: torch.Tensor
    valid_lengths: torch.Tensor
    output_storage: torch.Tensor
    output: torch.Tensor


def _require_enough_memory() -> None:
    cache_bytes = math.prod(KV_CACHE_SHAPE)
    workspace_bytes = math.prod(WORKSPACE_SHAPE) * 2
    required_bytes = cache_bytes + workspace_bytes + 512 * 1024**2
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        raise RuntimeError(
            "target-shape test needs at least "
            f"{required_bytes / 1024**3:.2f} GiB free, but only "
            f"{free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB "
            "is available"
        )


def _make_random_topk() -> torch.Tensor:
    """Generate random, per-request unique physical cache row IDs."""
    rng = random.Random(SEED)
    indices_cpu = [
        rng.sample(range(NUM_PHYSICAL_TOKENS), TOPK)
        for _ in range(BATCH_SIZE)
    ]
    return torch.tensor(indices_cpu, dtype=torch.int32, device="cuda").unsqueeze(1)


def _pack_selected_cache_rows(
    cache_rows: torch.Tensor,
    unique_ids: torch.Tensor,
    entry_bytes: int = ENTRY_BYTES,
) -> None:
    """Initialize only rows referenced by Top-K, keeping setup memory bounded."""
    codebook = torch.tensor(
        [
            -448.0,
            -64.0,
            -8.0,
            -2.0,
            -1.0,
            -0.5,
            -0.125,
            -0.03125,
            0.0,
            0.03125,
            0.125,
            0.5,
            1.0,
            2.0,
            8.0,
            64.0,
            448.0,
        ],
        dtype=torch.float32,
        device="cuda",
    ).to(FP8_DTYPE)
    dim_ids = torch.arange(HEAD_DIM, dtype=torch.int64, device="cuda")
    group_ids = torch.arange(NUM_GROUPS, dtype=torch.int64, device="cuda")

    for physical_ids in unique_ids.split(PACK_CHUNK_ROWS):
        physical_ids = physical_ids.to(torch.int64)
        fp8_code_ids = (
            physical_ids[:, None] * 131 + dim_ids[None, :] * 17
        ) % codebook.numel()
        fp8_values = codebook[fp8_code_ids].contiguous()
        scales = 0.25 + (
            (physical_ids[:, None] * 13 + group_ids[None, :] * 7) % 29
        ).to(torch.float32) / 16.0

        packed_rows = torch.full(
            (physical_ids.numel(), entry_bytes),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        packed_rows[:, :HEAD_DIM].copy_(fp8_values.view(torch.uint8))
        packed_rows[:, HEAD_DIM : HEAD_DIM + NUM_GROUPS * 4].copy_(
            scales.contiguous().view(torch.uint8)
        )
        cache_rows.index_copy_(0, physical_ids, packed_rows)


def _make_case() -> TestCase:
    assert torch.cuda.is_available(), "this test requires a CUDA/HIP device"
    _require_enough_memory()
    torch.manual_seed(SEED)

    indices = _make_random_topk()
    assert tuple(indices.shape) == INDICES_SHAPE
    token_ids = indices[:, 0, :]
    unique_ids = torch.unique(token_ids.reshape(-1).to(torch.int64))

    kv_cache = torch.empty(KV_CACHE_SHAPE, dtype=torch.uint8, device="cuda")
    cache_rows = kv_cache.view(NUM_PHYSICAL_TOKENS, ENTRY_BYTES)
    _pack_selected_cache_rows(cache_rows, unique_ids)

    valid_lengths = torch.full(
        (BATCH_SIZE,), TOPK, dtype=torch.int32, device="cuda"
    )
    output_storage = torch.full(
        WORKSPACE_SHAPE,
        OUTPUT_SENTINEL,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = output_storage[:, :TOPK]
    torch.cuda.synchronize()

    return TestCase(
        kv_cache=kv_cache,
        indices=indices,
        token_ids=token_ids,
        valid_lengths=valid_lengths,
        output_storage=output_storage,
        output=output,
    )


def _reference_from_packed_rows(packed_rows: torch.Tensor) -> torch.Tensor:
    fp8_values = (
        packed_rows[:, :HEAD_DIM].contiguous().view(FP8_DTYPE).to(torch.float32)
    )
    scales = (
        packed_rows[:, HEAD_DIM : HEAD_DIM + NUM_GROUPS * 4]
        .contiguous()
        .view(torch.float32)
    )
    return (
        fp8_values.view(-1, NUM_GROUPS, GROUP_SIZE) * scales[:, :, None]
    ).reshape(-1, HEAD_DIM).to(torch.bfloat16)


def _verify_correctness(case: TestCase) -> float:
    op.decode_gather_and_up_convert(
        case.kv_cache,
        case.token_ids,
        case.output,
        case.valid_lengths,
    )
    torch.cuda.synchronize()

    cache_rows = case.kv_cache.view(NUM_PHYSICAL_TOKENS, ENTRY_BYTES)
    max_abs_diff = 0.0
    for batch_id in range(BATCH_SIZE):
        for start in range(0, TOPK, REFERENCE_CHUNK_ROWS):
            end = min(start + REFERENCE_CHUNK_ROWS, TOPK)
            physical_ids = case.token_ids[batch_id, start:end].to(torch.int64)
            packed_rows = cache_rows.index_select(0, physical_ids)
            expected = _reference_from_packed_rows(packed_rows)
            actual = case.output[batch_id, start:end]

            mismatch = actual != expected
            mismatch_count = int(mismatch.sum().item())
            if mismatch_count:
                first = mismatch.nonzero()[0]
                row = int(first[0].item())
                col = int(first[1].item())
                physical_id = int(physical_ids[row].item())
                raise AssertionError(
                    "decode_gather_and_up_convert mismatch: "
                    f"batch={batch_id}, topk={start + row}, dim={col}, "
                    f"physical_id={physical_id}, actual={actual[row, col].item()}, "
                    f"expected={expected[row, col].item()}, "
                    f"chunk_mismatches={mismatch_count}"
                )
            chunk_max = float(
                (actual.float() - expected.float()).abs().max().item()
            )
            max_abs_diff = max(max_abs_diff, chunk_max)

    untouched_tail = case.output_storage[:, TOPK:]
    if not bool((untouched_tail == OUTPUT_SENTINEL).all().item()):
        raise AssertionError("operator overwrote the 2063-workspace padding tail")
    return max_abs_diff


def _benchmark_once(case: TestCase) -> tuple[float, float, float]:
    def run() -> None:
        op.decode_gather_and_up_convert(
            case.kv_cache,
            case.token_ids,
            case.output,
            case.valid_lengths,
        )

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(REPEAT):
        run()
    end_event.record()
    torch.cuda.synchronize()

    average_us = start_event.elapsed_time(end_event) * 1000.0 / REPEAT
    selected_tokens = BATCH_SIZE * TOPK
    effective_bytes = selected_tokens * (
        HEAD_DIM + NUM_GROUPS * 4 + 4 + HEAD_DIM * 2
    ) + BATCH_SIZE * 4
    effective_bandwidth_gbs = effective_bytes / (average_us * 1.0e3)
    tokens_per_second = selected_tokens / (average_us * 1.0e-6)
    return average_us, effective_bandwidth_gbs, tokens_per_second


def _make_small_cache(
    num_physical_tokens: int, entry_bytes: int = ENTRY_BYTES
) -> torch.Tensor:
    kv_cache = torch.empty(
        (num_physical_tokens, entry_bytes), dtype=torch.uint8, device="cuda"
    )
    physical_ids = torch.arange(
        num_physical_tokens, dtype=torch.int64, device="cuda"
    )
    _pack_selected_cache_rows(kv_cache, physical_ids, entry_bytes)
    return kv_cache


def _small_reference(
    kv_cache: torch.Tensor,
    token_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, width = token_ids.shape
    num_physical_tokens = kv_cache.shape[0]
    expected = torch.zeros(
        (batch_size, width, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    for batch_id in range(batch_size):
        valid_width = min(max(int(valid_lengths[batch_id].item()), 0), width)
        if valid_width == 0:
            continue
        ids = token_ids[batch_id, :valid_width].to(torch.int64)
        valid_mask = (ids >= 0) & (ids < num_physical_tokens)
        if not bool(valid_mask.any().item()):
            continue
        valid_positions = valid_mask.nonzero().flatten()
        packed_rows = kv_cache.index_select(0, ids[valid_mask])
        expected[batch_id, valid_positions] = _reference_from_packed_rows(
            packed_rows
        )
    return expected


def _run_small_case(
    token_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    workspace_extra: int,
    entry_bytes: int = ENTRY_BYTES,
) -> None:
    kv_cache = _make_small_cache(128, entry_bytes)
    batch_size, width = token_ids.shape
    output_storage = torch.full(
        (batch_size, width + workspace_extra, HEAD_DIM),
        OUTPUT_SENTINEL,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = output_storage[:, :width]
    op.decode_gather_and_up_convert(
        kv_cache, token_ids, output, valid_lengths
    )
    torch.cuda.synchronize()

    expected = _small_reference(kv_cache, token_ids, valid_lengths)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    if workspace_extra:
        untouched_tail = output_storage[:, width:]
        if not bool((untouched_tail == OUTPUT_SENTINEL).all().item()):
            raise AssertionError("operator overwrote workspace padding tail")


def test_decode_gather_and_up_convert_target_shape() -> None:
    case = _make_case()
    max_abs_diff = _verify_correctness(case)
    assert max_abs_diff == 0.0


@pytest.mark.parametrize("entry_bytes", [ENTRY_BYTES_WITH_ROPE, ENTRY_BYTES_NO_ROPE])
def test_decode_gather_and_up_convert_invalid_and_ragged(entry_bytes: int) -> None:
    token_ids = (
        torch.arange(4 * 19, dtype=torch.int32, device="cuda").reshape(4, 19)
        % 128
    )
    token_ids[0, 3] = -1
    token_ids[0, 7] = 128
    token_ids[3, 18] = -99
    valid_lengths = torch.tensor([19, 9, 0, 19], dtype=torch.int32, device="cuda")
    _run_small_case(
        token_ids, valid_lengths, workspace_extra=5, entry_bytes=entry_bytes
    )

    # Keep the public int64 index path covered as well.
    _run_small_case(
        token_ids.to(torch.int64),
        valid_lengths,
        workspace_extra=5,
        entry_bytes=entry_bytes,
    )


def test_decode_gather_and_up_convert_no_rope_528_stride() -> None:
    token_ids = (
        torch.arange(3 * 37, dtype=torch.int32, device="cuda").reshape(3, 37)
        % 128
    )
    token_ids[1, 5] = -1
    token_ids[2, 17] = 128
    valid_lengths = torch.tensor([37, 19, 37], dtype=torch.int32, device="cuda")
    _run_small_case(
        token_ids,
        valid_lengths,
        workspace_extra=5,
        entry_bytes=ENTRY_BYTES_NO_ROPE,
    )


@pytest.mark.parametrize("entry_bytes", [ENTRY_BYTES_WITH_ROPE, ENTRY_BYTES_NO_ROPE])
def test_decode_gather_and_up_convert_topk_2063(entry_bytes: int) -> None:
    width = WORKSPACE_TOPK
    token_ids = (
        torch.arange(2 * width, dtype=torch.int32, device="cuda").reshape(2, width)
        % 128
    )
    valid_lengths = torch.tensor(
        [width, width - 1], dtype=torch.int32, device="cuda"
    )
    _run_small_case(
        token_ids, valid_lengths, workspace_extra=3, entry_bytes=entry_bytes
    )


def test_decode_gather_and_up_convert_rejects_528_to_576() -> None:
    kv_cache = torch.empty(
        (1, ENTRY_BYTES_NO_ROPE), dtype=torch.uint8, device="cuda"
    )
    token_ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    valid_lengths = torch.ones((1,), dtype=torch.int32, device="cuda")
    dst = torch.empty((1, 1, 576), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="576 requires"):
        op.decode_gather_and_up_convert(kv_cache, token_ids, dst, valid_lengths)


def main() -> None:
    assert torch.cuda.is_available(), "this test requires a CUDA/HIP device"
    properties = torch.cuda.get_device_properties(0)
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(
        f"device={torch.cuda.get_device_name(0)}, "
        f"gcn={getattr(properties, 'gcnArchName', 'unknown')}"
    )
    print(
        f"kv_cache={KV_CACHE_SHAPE}, indices={INDICES_SHAPE}, "
        f"workspace={WORKSPACE_SHAPE}, seed={SEED}"
    )
    print(
        "tokens_per_block="
        f"{os.environ.get('LIGHTOP_DECODE_GATHER_TOKENS_PER_BLOCK', 'default')}"
    )

    case = _make_case()
    max_abs_diff = _verify_correctness(case)
    print(f"correctness=PASS, max_abs_diff={max_abs_diff:.1f} (exact BF16)")
    for entry_bytes in (ENTRY_BYTES_WITH_ROPE, ENTRY_BYTES_NO_ROPE):
        test_decode_gather_and_up_convert_invalid_and_ragged(entry_bytes)
        test_decode_gather_and_up_convert_topk_2063(entry_bytes)
    test_decode_gather_and_up_convert_no_rope_528_stride()
    test_decode_gather_and_up_convert_rejects_528_to_576()
    print("edge_cases=PASS (ragged/zero/invalid/int64/no-rope-528/topk2063)")

    results = [_benchmark_once(case) for _ in range(ROUNDS)]
    median_us = statistics.median(result[0] for result in results)
    median_bandwidth = statistics.median(result[1] for result in results)
    median_lookups = statistics.median(result[2] for result in results)
    print()
    print("| round | warmup | repeats | average (us) | effective BW (GB/s) | lookups/s |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for round_id, (average_us, bandwidth_gbs, tokens_per_second) in enumerate(
        results, start=1
    ):
        print(
            f"| {round_id} | {WARMUP} | {REPEAT} | {average_us:.3f} | "
            f"{bandwidth_gbs:.3f} | {tokens_per_second:.3e} |"
        )
    print(
        f"median: {median_us:.3f} us, {median_bandwidth:.3f} GB/s, "
        f"{median_lookups:.3e} lookups/s"
    )


if __name__ == "__main__":
    main()
