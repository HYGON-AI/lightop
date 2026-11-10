# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and captured-size tests for DPSKV4 LightOp gather/upconvert."""

import os

import pytest
import torch

from lightop import op


DIM_NOPE = 448
DIM_ROPE = 64
TOKEN_DATA_BYTES = 576
SCALE_BYTES = 8
# Keep the test aligned with SGLang's ``is_fp8_fnuz`` decision.  The active
# gfx928/gfx936/gfx938 targets use OCP E4M3FN; only gfx94x uses FNUZ there.
FP8 = (
    torch.float8_e4m3fnuz
    if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    else torch.float8_e4m3fn
)


def _captured_single_case(
    name: str,
    queries: int,
    topk: int,
    num_pages: int,
    page_size: int,
    page_stride_bytes: int,
) -> dict:
    return {
        "name": name,
        "op": "single",
        "queries": queries,
        "topk": topk,
        "num_pages": num_pages,
        "page_size": page_size,
        "page_stride_bytes": page_stride_bytes,
        "cache_shape": (num_pages, page_size, 1, TOKEN_DATA_BYTES + SCALE_BYTES),
        "cache_stride": (
            page_stride_bytes,
            TOKEN_DATA_BYTES + SCALE_BYTES,
            TOKEN_DATA_BYTES + SCALE_BYTES,
            1,
        ),
        "indices_shape": (queries, topk),
        "lengths_shape": (queries,),
        "output_shape": (queries, topk, 1, DIM_NOPE + DIM_ROPE),
        "compact_shape": (queries, 1, topk),
    }


def _captured_dual_case(
    name: str,
    queries: int,
    swa_topk: int,
    extra_topk: int,
    extra_page_size: int,
    extra_page_stride_bytes: int,
) -> dict:
    case = {
        "name": name,
        "op": "dual",
        "queries": queries,
        "swa": _captured_single_case(
            name=f"{name}_swa",
            queries=queries,
            topk=swa_topk,
            num_pages=527,
            page_size=256,
            page_stride_bytes=149760,
        ),
        "extra": _captured_single_case(
            name=f"{name}_extra",
            queries=queries,
            topk=extra_topk,
            num_pages=586,
            page_size=extra_page_size,
            page_stride_bytes=extra_page_stride_bytes,
        ),
    }
    return case


# End-to-end capture on 2026-08-23.  These groups are intentionally separated
# by service side: P=CP8/TP8/chunk32768 and D=TP8/DP8/DSpark.  The P records
# came from one 131072-token request plus two 128-token requests.  The D records
# came from two 128-input/32-output requests; CUDA Graph was disabled only for
# the capture run so runtime calls crossed the Python/C++ boundary.
PREFILL_P_CAPTURED_CASES = (
    _captured_single_case("p_single_q16_k128", 16, 128, 527, 256, 149760),
    _captured_single_case("p_single_q4096_k128", 4096, 128, 527, 256, 149760),
    _captured_dual_case("p_dual_q16_swa128_extra64_page2", 16, 128, 64, 2, 1728),
    _captured_dual_case("p_dual_q16_swa128_extra512_page64", 16, 128, 512, 64, 37440),
    _captured_dual_case("p_dual_q4096_swa128_extra256_page2", 4096, 128, 256, 2, 1728),
    _captured_dual_case("p_dual_q4096_swa128_extra512_page2", 4096, 128, 512, 2, 1728),
    _captured_dual_case("p_dual_q4096_swa128_extra768_page2", 4096, 128, 768, 2, 1728),
    _captured_dual_case("p_dual_q4096_swa128_extra1024_page2", 4096, 128, 1024, 2, 1728),
    _captured_dual_case("p_dual_q4096_swa128_extra512_page64", 4096, 128, 512, 64, 37440),
)

DECODE_D_CAPTURED_CASES = (
    _captured_single_case("d_single_q5_k192", 5, 192, 527, 256, 149760),
    _captured_single_case("d_single_q6_k128", 6, 128, 527, 256, 149760),
    _captured_dual_case("d_dual_q6_swa128_extra512_page64", 6, 128, 512, 64, 37440),
    _captured_dual_case("d_dual_q6_swa128_extra8256_page2", 6, 128, 8256, 2, 1728),
)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _make_cache(num_pages: int, page_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    page_stride = _round_up(page_size * (TOKEN_DATA_BYTES + SCALE_BYTES), 576)
    cache = torch.zeros(
        (num_pages, page_stride), dtype=torch.uint8, device="cuda"
    )
    for page in range(num_pages):
        for token in range(page_size):
            data_base = token * TOKEN_DATA_BYTES
            scale_base = page_size * TOKEN_DATA_BYTES + token * SCALE_BYTES
            nope = torch.randn(
                DIM_NOPE, generator=generator, device="cuda", dtype=torch.float32
            )
            rope = torch.randn(
                DIM_ROPE, generator=generator, device="cuda", dtype=torch.float32
            ).to(torch.bfloat16)
            # Keep scales close to one while covering several exact powers of two.
            scales = torch.randint(
                124,
                131,
                (DIM_NOPE // 64,),
                generator=generator,
                device="cuda",
                dtype=torch.uint8,
            )
            cache[page, data_base : data_base + DIM_NOPE].copy_(
                nope.to(FP8).view(torch.uint8)
            )
            cache[
                page,
                data_base + DIM_NOPE : data_base + TOKEN_DATA_BYTES,
            ].copy_(rope.view(torch.uint8))
            cache[
                page,
                scale_base : scale_base + DIM_NOPE // 64,
            ].copy_(scales)
    return cache


def _reference(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries, topk = indices.shape
    output = torch.zeros(
        (queries, topk, 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=cache.device,
    )
    compact = torch.zeros(
        (queries, 1, topk), dtype=torch.int32, device=cache.device
    )
    raw = cache.view(torch.uint8)
    page_stride = raw.stride(0)
    num_tokens = raw.shape[0] * page_size
    for query in range(queries):
        for column in range(topk):
            if column >= int(lengths[query]):
                continue
            compact[query, 0, column] = query * topk + column
            location = int(indices[query, column])
            if location < 0 or location >= num_tokens:
                continue
            page, in_page = divmod(location, page_size)
            data_base = in_page * TOKEN_DATA_BYTES
            scale_base = page_size * TOKEN_DATA_BYTES + in_page * SCALE_BYTES
            fp8 = raw[page, data_base : data_base + DIM_NOPE].view(FP8).float()
            scales_u8 = raw[
                page, scale_base : scale_base + DIM_NOPE // 64
            ].to(torch.int32)
            scales = torch.exp2((scales_u8 - 127).float()).repeat_interleave(64)
            output[query, column, 0, :DIM_NOPE] = (fp8 * scales).to(
                torch.bfloat16
            )
            output[query, column, 0, DIM_NOPE:] = raw[
                page,
                data_base + DIM_NOPE : data_base + TOKEN_DATA_BYTES,
            ].view(torch.bfloat16)
    return output, compact


def _assert_result(
    actual: torch.Tensor,
    actual_compact: torch.Tensor,
    expected: torch.Tensor,
    expected_compact: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        mismatch = actual != expected
        first = mismatch.nonzero()[0]
        index = tuple(int(v) for v in first)
        print(
            "FIRST_MISMATCH",
            index,
            "actual=", actual[index].item(),
            "expected=", expected[index].item(),
        )
        print("ACTUAL_HEAD", actual[0, 0, 0, :16].float().tolist())
        print("EXPECTED_HEAD", expected[0, 0, 0, :16].float().tolist())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_compact, expected_compact, rtol=0, atol=0)


def test_single() -> None:
    page_size = 4
    cache = _make_cache(3, page_size, seed=7)
    indices = torch.tensor(
        [[0, 4, 11, -1, 12], [3, 8, 6, 1, 2], [9, 2, 7, 5, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    lengths = torch.tensor([5, 3, 4], dtype=torch.int32, device="cuda")
    output = torch.empty(
        (3, 5, 1, DIM_NOPE + DIM_ROPE), dtype=torch.bfloat16, device="cuda"
    )
    compact = torch.empty((3, 1, 5), dtype=torch.int32, device="cuda")
    expected, expected_compact = _reference(cache, indices, lengths, page_size)
    op.dsv4_gather_upconvert_k_cache_paged(
        cache, indices, lengths, output, compact, page_size
    )
    _assert_result(output, compact, expected, expected_compact)


def test_dual_and_4d_view() -> None:
    swa_page_size = 2
    extra_page_size = 3
    swa_raw = _make_cache(4, swa_page_size, seed=11)
    extra_raw = _make_cache(3, extra_page_size, seed=13)
    # Exercise the same non-contiguous 4-D byte view used by SGLang/FlashMLA.
    swa_cache = torch.as_strided(
        swa_raw,
        size=(swa_raw.shape[0], swa_page_size, 1, TOKEN_DATA_BYTES + SCALE_BYTES),
        stride=(swa_raw.stride(0), TOKEN_DATA_BYTES + SCALE_BYTES, TOKEN_DATA_BYTES + SCALE_BYTES, 1),
    )
    swa_indices = torch.tensor(
        [[0, 7, 3, -1], [6, 1, 4, 2]], dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.tensor(
        [[8, 2, 5], [0, 9, 4]], dtype=torch.int32, device="cuda"
    )
    swa_lengths = torch.tensor([4, 3], dtype=torch.int32, device="cuda")
    extra_lengths = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    swa_output = torch.empty((2, 4, 1, 512), dtype=torch.bfloat16, device="cuda")
    extra_output = torch.empty((2, 3, 1, 512), dtype=torch.bfloat16, device="cuda")
    swa_compact = torch.empty((2, 1, 4), dtype=torch.int32, device="cuda")
    extra_compact = torch.empty((2, 1, 3), dtype=torch.int32, device="cuda")
    swa_expected, swa_compact_expected = _reference(
        swa_raw, swa_indices, swa_lengths, swa_page_size
    )
    extra_expected, extra_compact_expected = _reference(
        extra_raw, extra_indices, extra_lengths, extra_page_size
    )
    op.dsv4_gather_upconvert_dual_k_cache_paged(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_output,
        swa_compact,
        swa_page_size,
        extra_raw,
        extra_indices,
        extra_lengths,
        extra_output,
        extra_compact,
        extra_page_size,
    )
    _assert_result(
        swa_output, swa_compact, swa_expected, swa_compact_expected
    )
    _assert_result(
        extra_output,
        extra_compact,
        extra_expected,
        extra_compact_expected,
    )


def _validate_captured_single_case(case: dict) -> None:
    assert case["op"] == "single"
    assert case["cache_shape"] == (
        case["num_pages"],
        case["page_size"],
        1,
        TOKEN_DATA_BYTES + SCALE_BYTES,
    )
    assert case["cache_stride"] == (
        case["page_stride_bytes"],
        TOKEN_DATA_BYTES + SCALE_BYTES,
        TOKEN_DATA_BYTES + SCALE_BYTES,
        1,
    )
    assert case["page_stride_bytes"] >= (
        case["page_size"] * (TOKEN_DATA_BYTES + SCALE_BYTES)
    )
    assert case["indices_shape"] == (case["queries"], case["topk"])
    assert case["lengths_shape"] == (case["queries"],)
    assert case["output_shape"] == (case["queries"], case["topk"], 1, 512)
    assert case["compact_shape"] == (case["queries"], 1, case["topk"])


def _validate_captured_cases(cases: tuple[dict, ...], expected_prefix: str) -> None:
    names = set()
    for case in cases:
        assert case["name"].startswith(expected_prefix)
        assert case["name"] not in names
        names.add(case["name"])
        if case["op"] == "single":
            _validate_captured_single_case(case)
        else:
            assert case["op"] == "dual"
            assert case["swa"]["queries"] == case["queries"]
            assert case["extra"]["queries"] == case["queries"]
            _validate_captured_single_case(case["swa"])
            _validate_captured_single_case(case["extra"])


def test_prefill_p_captured_size_metadata() -> None:
    """Keep the nine P-side runtime sizes separate from D-side cases."""

    assert len(PREFILL_P_CAPTURED_CASES) == 9
    _validate_captured_cases(PREFILL_P_CAPTURED_CASES, "p_")


def test_decode_d_captured_size_metadata() -> None:
    """Keep the four D-side runtime sizes separate from P-side cases."""

    assert len(DECODE_D_CAPTURED_CASES) == 4
    _validate_captured_cases(DECODE_D_CAPTURED_CASES, "d_")


def _allocate_captured_cache(case: dict) -> torch.Tensor:
    raw = torch.zeros(
        (case["num_pages"], case["page_stride_bytes"]),
        dtype=torch.uint8,
        device="cuda",
    )
    return torch.as_strided(
        raw,
        size=case["cache_shape"],
        stride=case["cache_stride"],
    )


def _run_captured_single_case(case: dict) -> None:
    cache = _allocate_captured_cache(case)
    indices = torch.zeros(case["indices_shape"], dtype=torch.int32, device="cuda")
    lengths = torch.full(
        case["lengths_shape"], case["topk"], dtype=torch.int32, device="cuda"
    )
    output = torch.empty(case["output_shape"], dtype=torch.bfloat16, device="cuda")
    compact = torch.empty(case["compact_shape"], dtype=torch.int32, device="cuda")
    op.dsv4_gather_upconvert_k_cache_paged(
        cache, indices, lengths, output, compact, case["page_size"]
    )
    torch.cuda.synchronize()
    assert output[0, 0, 0, 0].item() == 0
    assert compact[0, 0, 0].item() == 0
    assert compact[-1, 0, -1].item() == case["queries"] * case["topk"] - 1


def _run_captured_dual_case(case: dict) -> None:
    swa = case["swa"]
    extra = case["extra"]
    swa_cache = _allocate_captured_cache(swa)
    extra_cache = _allocate_captured_cache(extra)
    swa_indices = torch.zeros(swa["indices_shape"], dtype=torch.int32, device="cuda")
    extra_indices = torch.zeros(
        extra["indices_shape"], dtype=torch.int32, device="cuda"
    )
    swa_lengths = torch.full(
        swa["lengths_shape"], swa["topk"], dtype=torch.int32, device="cuda"
    )
    extra_lengths = torch.full(
        extra["lengths_shape"], extra["topk"], dtype=torch.int32, device="cuda"
    )
    swa_output = torch.empty(
        swa["output_shape"], dtype=torch.bfloat16, device="cuda"
    )
    extra_output = torch.empty(
        extra["output_shape"], dtype=torch.bfloat16, device="cuda"
    )
    swa_compact = torch.empty(
        swa["compact_shape"], dtype=torch.int32, device="cuda"
    )
    extra_compact = torch.empty(
        extra["compact_shape"], dtype=torch.int32, device="cuda"
    )
    op.dsv4_gather_upconvert_dual_k_cache_paged(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_output,
        swa_compact,
        swa["page_size"],
        extra_cache,
        extra_indices,
        extra_lengths,
        extra_output,
        extra_compact,
        extra["page_size"],
    )
    torch.cuda.synchronize()
    assert swa_output[0, 0, 0, 0].item() == 0
    assert extra_output[0, 0, 0, 0].item() == 0
    assert swa_compact[-1, 0, -1].item() == swa["queries"] * swa["topk"] - 1
    assert (
        extra_compact[-1, 0, -1].item()
        == extra["queries"] * extra["topk"] - 1
    )


def _run_captured_cases(cases: tuple[dict, ...]) -> None:
    for case in cases:
        if case["op"] == "single":
            _run_captured_single_case(case)
        else:
            _run_captured_dual_case(case)
        torch.cuda.empty_cache()


_RUN_CAPTURED_SIZE_TESTS = os.environ.get(
    "LIGHTOP_RUN_DSV4_CAPTURED_SIZE_TESTS", "0"
) == "1"


@pytest.mark.skipif(
    not _RUN_CAPTURED_SIZE_TESTS,
    reason="set LIGHTOP_RUN_DSV4_CAPTURED_SIZE_TESTS=1 for multi-GB P-size smoke tests",
)
def test_prefill_p_captured_sizes_smoke() -> None:
    _run_captured_cases(PREFILL_P_CAPTURED_CASES)


@pytest.mark.skipif(
    not _RUN_CAPTURED_SIZE_TESTS,
    reason="set LIGHTOP_RUN_DSV4_CAPTURED_SIZE_TESTS=1 for captured D-size smoke tests",
)
def test_decode_d_captured_sizes_smoke() -> None:
    _run_captured_cases(DECODE_D_CAPTURED_CASES)


if __name__ == "__main__":
    test_single()
    test_dual_and_4d_view()
    print("DSV4_LIGHTOP_GATHER_CORRECT=True")
