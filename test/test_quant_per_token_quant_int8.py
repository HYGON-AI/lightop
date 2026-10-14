# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration comparisons for LMSlim per-token INT8 quantization APIs."""

import inspect

import torch

import lightop.moe
import lightop.quant
from lightop._lmslim_native.layers.gemm.int8_utils import (
    per_token_group_quant_int8 as source_group_api,
    per_token_quant_int8 as source_api,
)
from lightop.quant import (
    per_token_group_quant_int8 as migrated_group_api,
    per_token_quant_int8 as migrated_api,
)


def test_int8_quantization_apis_have_unique_quant_ownership() -> None:
    for name in ("per_token_group_quant_int8", "per_token_quant_int8"):
        assert name in lightop.quant.__all__
        assert name not in lightop.moe.__all__


def test_per_token_quant_int8_matches_migrated_source() -> None:
    assert inspect.signature(migrated_api) == inspect.signature(source_api)

    torch.manual_seed(5)
    input_tensor = torch.randn((3, 257), device="cuda", dtype=torch.bfloat16)

    migrated_q, migrated_scale = migrated_api(input_tensor)
    source_q, source_scale = source_api(input_tensor)
    torch.cuda.synchronize()

    torch.testing.assert_close(migrated_q, source_q, rtol=0, atol=0)
    torch.testing.assert_close(migrated_scale, source_scale, rtol=0, atol=0)


def test_per_token_quant_int8_dsv4_prefill_fast_path() -> None:
    torch.manual_seed(23)
    input_tensor = torch.randn(
        (262144, 128), device="cuda", dtype=torch.bfloat16
    )
    input_tensor[0].zero_()

    expected_q, expected_scale = source_api(input_tensor)
    actual_q, actual_scale = migrated_api(input_tensor)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=0, atol=0
    )


def test_per_token_group_quant_int8_matches_migrated_source() -> None:
    assert inspect.signature(migrated_group_api) == inspect.signature(
        source_group_api
    )

    torch.manual_seed(6)
    input_tensor = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)

    migrated_q, migrated_scale = migrated_group_api(input_tensor, 128)
    source_q, source_scale = source_group_api(input_tensor, 128)
    torch.cuda.synchronize()

    torch.testing.assert_close(migrated_q, source_q, rtol=0, atol=0)
    torch.testing.assert_close(migrated_scale, source_scale, rtol=0, atol=0)
