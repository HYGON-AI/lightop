# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Runtime checks for the remaining categorized KV-cache operators."""

import pytest
import torch

from lightop import op as lightop_native
from lightop.kvcache import (
    fuse_act_quant_and_store_index_k_cache,
    fuse_qk_quant_and_store_index_k_cache,
    fused_quantize_and_store_mla_kv_cache,
    quantize_nope_fp8_rope_bf16_pack_store,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device is required")


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert torch.equal(
            actual.view(torch.uint8), expected.view(torch.uint8)
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_remaining_kvcache_bindings_are_direct_reexports() -> None:
    names = [
        "fuse_act_quant_and_store_index_k_cache",
        "fuse_qk_quant_and_store_index_k_cache",
        "fused_quantize_and_store_mla_kv_cache",
        "quantize_nope_fp8_rope_bf16_pack_store",
    ]
    category_apis = [
        fuse_act_quant_and_store_index_k_cache,
        fuse_qk_quant_and_store_index_k_cache,
        fused_quantize_and_store_mla_kv_cache,
        quantize_nope_fp8_rope_bf16_pack_store,
    ]
    for name, category_api in zip(names, category_apis):
        assert category_api is getattr(lightop_native, name)


def test_fuse_act_quant_and_store_index_k_cache_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(21)
    page_size = 4
    hidden_size = 128
    inputs = torch.randn(
        (4, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    locations = torch.tensor(
        [0, 3, 4, 7], device="cuda", dtype=torch.int64
    )
    buffer = torch.zeros(
        (2, page_size * (hidden_size + 4)),
        device="cuda",
        dtype=torch.uint8,
    )
    native_buffer = buffer.clone()

    fuse_act_quant_and_store_index_k_cache(
        inputs, buffer, locations, page_size, 1e-5, False, True
    )
    lightop_native.fuse_act_quant_and_store_index_k_cache(
        inputs, native_buffer, locations, page_size, 1e-5, False, True
    )
    torch.cuda.synchronize()
    assert torch.equal(buffer, native_buffer)


def test_fuse_qk_quant_and_store_index_k_cache_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(22)
    page_size = 4
    hidden_size = 128
    query = torch.randn(
        (4, 2, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        (4, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    locations = torch.tensor(
        [0, 3, 4, 7], device="cuda", dtype=torch.int64
    )
    buffer = torch.zeros(
        (2, page_size * (hidden_size + 4)),
        device="cuda",
        dtype=torch.uint8,
    )
    native_buffer = buffer.clone()

    outputs = fuse_qk_quant_and_store_index_k_cache(
        query,
        key,
        buffer,
        locations,
        page_size,
        None,
        1.0,
        1.0,
        1e-5,
        False,
        True,
    )
    native_outputs = (
        lightop_native.fuse_qk_quant_and_store_index_k_cache(
            query,
            key,
            native_buffer,
            locations,
            page_size,
            None,
            1.0,
            1.0,
            1e-5,
            False,
            True,
        )
    )
    torch.cuda.synchronize()
    for output, native_output in zip(outputs, native_outputs):
        _assert_tensor_equal(output, native_output)
    assert torch.equal(buffer, native_buffer)


def test_fused_quantize_and_store_mla_kv_cache_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(23)
    k_nope = torch.randn(
        (2, 512), device="cuda", dtype=torch.bfloat16
    )
    k_rope = torch.randn(
        (2, 64), device="cuda", dtype=torch.bfloat16
    )
    locations = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    buffer = torch.zeros((4, 656), device="cuda", dtype=torch.uint8)
    native_buffer = buffer.clone()

    fused_quantize_and_store_mla_kv_cache(
        k_nope, k_rope, buffer, locations, "fp8_e4m3", 1e-5
    )
    lightop_native.fused_quantize_and_store_mla_kv_cache(
        k_nope, k_rope, native_buffer, locations, "fp8_e4m3", 1e-5
    )
    torch.cuda.synchronize()
    assert torch.equal(buffer, native_buffer)


@pytest.mark.parametrize("location_dtype", [torch.int32, torch.int64])
def test_quantize_nope_fp8_rope_bf16_pack_store_matches_native(
    location_dtype: torch.dtype,
) -> None:
    _require_cuda()
    torch.manual_seed(24)
    page_size = 4
    inputs = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
    locations = torch.tensor(
        [0, 5], device="cuda", dtype=location_dtype
    )
    page_bytes = page_size * (448 + 64 * 2 + 8)
    buffer = torch.zeros(
        (2, page_bytes), device="cuda", dtype=torch.uint8
    )
    native_buffer = buffer.clone()

    quantize_nope_fp8_rope_bf16_pack_store(
        inputs, buffer, locations, page_size, 1e-5
    )
    lightop_native.quantize_nope_fp8_rope_bf16_pack_store(
        inputs, native_buffer, locations, page_size, 1e-5
    )
    torch.cuda.synchronize()
    assert torch.equal(buffer, native_buffer)
