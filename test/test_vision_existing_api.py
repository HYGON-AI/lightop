"""Compatibility checks for existing LightOp vision exports."""

import pytest

import lightop
import lightop.vision


EXISTING_VISION_APIS = [
    "GridSample",
    "bilinear_upsample",
    "bilinear_upsample_backward",
    "grid_sample_backward",
    "grid_sample_forward",
]


@pytest.mark.parametrize("name", EXISTING_VISION_APIS)
def test_existing_vision_api_is_direct_reexport(name):
    assert getattr(lightop.vision, name) is getattr(lightop, name)


def test_vision_public_api_inventory():
    assert set(lightop.vision.__all__) == set(EXISTING_VISION_APIS)
