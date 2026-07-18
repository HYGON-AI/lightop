"""Compatibility and correctness checks for LightOp tensor exports."""

import pytest
import torch

import lightop
import lightop.tensor


EXISTING_TENSOR_APIS = [
    "batch_linear_assignment",
    "ds_cat",
    "fill_zeros",
    "mean",
    "sum",
]


@pytest.mark.parametrize("name", EXISTING_TENSOR_APIS)
def test_existing_tensor_api_is_direct_reexport(name):
    assert getattr(lightop.tensor, name) is getattr(lightop, name)


def test_tensor_public_api_inventory():
    assert set(lightop.tensor.__all__) == set(EXISTING_TENSOR_APIS)


@pytest.mark.parametrize("name", ["mean", "sum"])
def test_tensor_reduction_matches_torch(name):
    torch.manual_seed(11)
    values = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    actual = getattr(lightop.tensor, name)(values, dim=-1)
    expected = getattr(torch, name)(values, dim=-1)
    assert torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
