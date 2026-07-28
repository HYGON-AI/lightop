"""Compatibility and correctness checks for LightOp sampling exports."""

import pytest
import torch

import lightop
import lightop.sampling


CATEGORIZED_SAMPLING_APIS = [
    "topk",
    "topk_transform_1024",
]

LEGACY_PROBABILITY_APIS = [
    "get_default_generators",
    "get_seed_and_offset",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_sampling_from_probs",
]


@pytest.mark.parametrize("name", CATEGORIZED_SAMPLING_APIS)
def test_existing_sampling_api_is_direct_reexport(name):
    assert getattr(lightop.sampling, name) is getattr(lightop, name)


def test_sampling_public_api_inventory():
    assert set(lightop.sampling.__all__) == set(
        CATEGORIZED_SAMPLING_APIS + LEGACY_PROBABILITY_APIS
    )


def test_topk_matches_torch():
    torch.manual_seed(10)
    values = torch.randn(16, 257, device="cuda")
    actual_values, actual_indices = lightop.sampling.topk(values, 17)
    expected_values, expected_indices = torch.topk(values, 17)
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_values, expected_values)


def test_legacy_top_k_probability_sampler_is_preserved():
    probs = torch.tensor(
        [[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]],
        dtype=torch.float32,
        device="cuda",
    )
    sampled = lightop.sampling.top_k_sampling_from_probs(
        probs,
        1,
        deterministic=True,
        seed=123,
        offset=0,
    )
    assert torch.equal(sampled.long(), probs.argmax(dim=-1))
