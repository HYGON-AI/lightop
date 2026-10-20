# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility and correctness checks for LightOp tensor exports."""

import pytest
import torch

import lightop
from lightop import op as lightop_native
import lightop.tensor


EXISTING_TENSOR_APIS = [
    "batch_linear_assignment",
    "ds_cat",
    "fill_zeros",
    "mean",
    "sum",
]

NATIVE_TENSOR_APIS = ["reshape_to_float"]


@pytest.mark.parametrize("name", EXISTING_TENSOR_APIS)
def test_existing_tensor_api_is_direct_reexport(name):
    assert getattr(lightop.tensor, name) is getattr(lightop, name)


@pytest.mark.parametrize("name", NATIVE_TENSOR_APIS)
def test_native_tensor_api_is_direct_binding(name):
    assert getattr(lightop.tensor, name) is getattr(lightop_native, name)


def test_tensor_public_api_inventory():
    assert set(lightop.tensor.__all__) == set(
        EXISTING_TENSOR_APIS + NATIVE_TENSOR_APIS
    )


@pytest.mark.parametrize("name", ["mean", "sum"])
def test_tensor_reduction_matches_torch(name):
    torch.manual_seed(11)
    values = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    actual = getattr(lightop.tensor, name)(values, dim=-1)
    expected = getattr(torch, name)(values, dim=-1)
    assert torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reshape_to_float_matches_torch_for_noncontiguous_input(dtype):
    torch.manual_seed(42)
    source = torch.randn((3, 10, 2), device="cuda", dtype=dtype)
    values = source.transpose(1, 2)
    assert not values.is_contiguous()
    vocab_size = 17

    actual = lightop.tensor.reshape_to_float(values, vocab_size)
    expected = values.reshape(3, -1)[:, :vocab_size].float()

    assert actual.shape == (3, vocab_size)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
