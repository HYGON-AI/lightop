# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for DSV4 paged K-cache dequantization."""

from __future__ import annotations

import pytest
import torch


HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
TOKEN_DATA_BYTES = 576
SCALE_BYTES = 8
SUPPORTED_PAGE_SIZES = (2, 64, 256)
SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)
SPECIAL_SCALES = (0, 1, 2, 126, 127, 128, 253, 254, 255)


def _load_ops():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("DSV4 dequantization tests require ROCm")

    arch = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":", 1)[0]
    if arch != "gfx936":
        pytest.skip("DSV4 E4M3FN validation is scheduled for gfx936")

    from lightop.kvcache import dsv4_dequantize_k_cache_paged_out
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
        fp8_dtype,
    )

    assert fp8_dtype == torch.float8_e4m3fn
    return dequantize_k_cache_paged, dsv4_dequantize_k_cache_paged_out


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _make_output(
    rows: int,
    *,
    row_stride: int = HEAD_DIM,
    storage_offset: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    span = 0 if rows == 0 else (rows - 1) * row_stride + HEAD_DIM
    owner = torch.full(
        (storage_offset + span + 16,),
        17.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = owner.as_strided(
        (rows, 1, HEAD_DIM),
        (row_stride, HEAD_DIM, 1),
        storage_offset=storage_offset,
    )
    guard = torch.ones(owner.numel(), dtype=torch.bool, device=owner.device)
    if rows:
        guard.as_strided(
            (rows, 1, HEAD_DIM),
            (row_stride, HEAD_DIM, 1),
            storage_offset=storage_offset,
        ).fill_(False)
    return out, owner, guard


def _make_fixture(
    page_size: int,
    rows: int,
    index_dtype: torch.dtype,
    *,
    row_stride: int = HEAD_DIM,
    output_offset: int = 16,
    cache_offset: int = 32,
    special_values: bool = False,
    seed: int = 731,
) -> dict:
    num_pages = max(2, (rows + page_size - 1) // page_size + 1)
    page_bytes = _round_up(
        page_size * (TOKEN_DATA_BYTES + SCALE_BYTES), TOKEN_DATA_BYTES
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)

    cache_owner = torch.full(
        (cache_offset + num_pages * page_bytes + 64,),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    cache = cache_owner[
        cache_offset : cache_offset + num_pages * page_bytes
    ].view(num_pages, page_bytes)
    data = cache[:, : page_size * TOKEN_DATA_BYTES].view(
        num_pages, page_size, TOKEN_DATA_BYTES
    )
    scales = cache[
        :, page_size * TOKEN_DATA_BYTES : page_size * (TOKEN_DATA_BYTES + SCALE_BYTES)
    ].view(num_pages, page_size, SCALE_BYTES)
    locations = torch.arange(
        num_pages * page_size, device="cuda", dtype=torch.int64
    )

    if special_values:
        data[..., :NOPE_DIM].copy_(
            (locations % 256).to(torch.uint8).view(num_pages, page_size, 1)
        )
        scale_table = torch.tensor(
            SPECIAL_SCALES, dtype=torch.uint8, device="cuda"
        )
        scale_values = scale_table[(locations // 256) % len(SPECIAL_SCALES)]
        scales[..., : NOPE_DIM // 64].copy_(
            scale_values.view(num_pages, page_size, 1)
        )
        rope_bits = torch.arange(
            num_pages * page_size * ROPE_DIM,
            device="cuda",
            dtype=torch.int64,
        )
        rope_bits = (rope_bits % 65536).to(torch.int16).view(
            num_pages, page_size, ROPE_DIM
        )
        data[..., NOPE_DIM:].copy_(rope_bits.view(torch.uint8))
    else:
        nope = torch.randint(
            0,
            256,
            (num_pages, page_size, NOPE_DIM),
            generator=generator,
            dtype=torch.uint8,
            device="cuda",
        )
        nope.masked_fill_((nope & 0x7F) == 0x7F, 0x7E)
        data[..., :NOPE_DIM].copy_(nope)
        rope = torch.randn(
            (num_pages, page_size, ROPE_DIM),
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).to(torch.bfloat16)
        data[..., NOPE_DIM:].copy_(rope.view(torch.uint8))
        scales[..., : NOPE_DIM // 64].copy_(
            torch.randint(
                124,
                131,
                (num_pages, page_size, NOPE_DIM // 64),
                generator=generator,
                dtype=torch.uint8,
                device="cuda",
            )
        )
    scales[..., -1].fill_(0xCD)

    ids_owner = torch.full(
        (rows + 6,), -99, dtype=index_dtype, device="cuda"
    )
    page_table = ids_owner[3 : 3 + rows]
    if special_values:
        page_table.copy_(locations[:rows].to(index_dtype))
    else:
        page_table.copy_(
            ((torch.arange(rows, device="cuda") * 37) % locations.numel()).to(
                index_dtype
            )
        )

    reference_out, reference_owner, reference_guard = _make_output(
        rows, row_stride=row_stride, storage_offset=output_offset
    )
    actual_out, actual_owner, actual_guard = _make_output(
        rows, row_stride=row_stride, storage_offset=output_offset
    )
    return {
        "cache": cache,
        "cache_owner": cache_owner,
        "page_table": page_table,
        "ids_owner": ids_owner,
        "page_size": page_size,
        "reference_out": reference_out,
        "reference_owner": reference_owner,
        "reference_guard": reference_guard,
        "actual_out": actual_out,
        "actual_owner": actual_owner,
        "actual_guard": actual_guard,
    }


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def _compare_with_reference(fixture: dict) -> None:
    reference, candidate = _load_ops()
    cache_before = _bytes(fixture["cache_owner"]).clone()
    ids_before = _bytes(fixture["ids_owner"]).clone()
    reference_before = fixture["reference_owner"].clone()
    actual_before = fixture["actual_owner"].clone()

    fixture["reference_out"].fill_(float("nan"))
    fixture["actual_out"].fill_(float("nan"))
    reference_result = reference(
        fixture["cache"],
        fixture["page_table"],
        fixture["page_size"],
        out=fixture["reference_out"],
    )
    actual_result = candidate(
        fixture["cache"],
        fixture["page_table"],
        fixture["actual_out"],
        fixture["page_size"],
    )
    torch.cuda.synchronize()

    assert reference_result is fixture["reference_out"]
    assert actual_result is None
    assert torch.equal(
        fixture["actual_out"].view(torch.int16),
        fixture["reference_out"].view(torch.int16),
    )
    assert torch.equal(_bytes(fixture["cache_owner"]), cache_before)
    assert torch.equal(_bytes(fixture["ids_owner"]), ids_before)
    assert torch.equal(
        fixture["reference_owner"][fixture["reference_guard"]],
        reference_before[fixture["reference_guard"]],
    )
    assert torch.equal(
        fixture["actual_owner"][fixture["actual_guard"]],
        actual_before[fixture["actual_guard"]],
    )


@pytest.mark.parametrize("page_size", SUPPORTED_PAGE_SIZES)
@pytest.mark.parametrize("index_dtype", SUPPORTED_INDEX_DTYPES)
def test_matches_sglang_for_strided_outputs(page_size, index_dtype):
    for rows in (0, 1, 7, 8, 9, 33):
        fixture = _make_fixture(
            page_size,
            rows,
            index_dtype,
            row_stride=513,
            output_offset=1,
            cache_offset=2,
        )
        _compare_with_reference(fixture)


@pytest.mark.parametrize("page_size", SUPPORTED_PAGE_SIZES)
@pytest.mark.parametrize("index_dtype", SUPPORTED_INDEX_DTYPES)
def test_matches_sglang_for_special_fp8_values(page_size, index_dtype):
    fixture = _make_fixture(
        page_size,
        256 * len(SPECIAL_SCALES),
        index_dtype,
        special_values=True,
    )
    _compare_with_reference(fixture)


@pytest.mark.parametrize("page_size", SUPPORTED_PAGE_SIZES)
@pytest.mark.parametrize("index_dtype", SUPPORTED_INDEX_DTYPES)
def test_invalid_locations_produce_zero_rows(page_size, index_dtype):
    _, candidate = _load_ops()
    fixture = _make_fixture(page_size, 4, index_dtype, row_stride=513)
    num_locations = fixture["cache"].size(0) * page_size
    fixture["page_table"].copy_(
        torch.tensor(
            [-1, num_locations, torch.iinfo(index_dtype).max, -7],
            dtype=index_dtype,
            device="cuda",
        )
    )
    cache_before = fixture["cache_owner"].clone()
    ids_before = fixture["ids_owner"].clone()
    output_before = fixture["actual_owner"].clone()

    candidate(
        fixture["cache"],
        fixture["page_table"],
        fixture["actual_out"],
        page_size,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(fixture["actual_out"].view(torch.int16)) == 0
    assert torch.equal(fixture["cache_owner"], cache_before)
    assert torch.equal(fixture["ids_owner"], ids_before)
    assert torch.equal(
        fixture["actual_owner"][fixture["actual_guard"]],
        output_before[fixture["actual_guard"]],
    )


def _assert_rejected(candidate, args, watched) -> None:
    snapshots = [_bytes(tensor).clone() for tensor in watched]
    with pytest.raises(RuntimeError, match="DSV4 flat dequant"):
        candidate(*args)
    torch.cuda.synchronize()
    for tensor, snapshot in zip(watched, snapshots):
        assert torch.equal(_bytes(tensor), snapshot)


def test_rejects_invalid_metadata_before_writing_output():
    _, candidate = _load_ops()
    fixture = _make_fixture(64, 9, torch.int64)
    cache = fixture["cache"]
    page_table = fixture["page_table"]
    out = fixture["actual_out"]
    watched = (
        fixture["cache_owner"],
        fixture["ids_owner"],
        fixture["actual_owner"],
    )

    bad_cases = (
        (cache.float(), page_table, out, 64),
        (cache[:, ::2], page_table, out, 64),
        (cache, page_table.float(), out, 64),
        (cache, page_table.view(3, 3), out, 64),
        (cache, page_table, out.half(), 64),
        (cache, page_table, out[:, :, :-1], 64),
        (cache, page_table, out, 128),
    )
    for args in bad_cases:
        _assert_rejected(candidate, args, watched)

    overlapping = fixture["actual_owner"].as_strided(
        (9, 1, HEAD_DIM),
        (HEAD_DIM - 1, HEAD_DIM, 1),
        storage_offset=16,
    )
    _assert_rejected(
        candidate, (cache, page_table, overlapping, 64), watched
    )
