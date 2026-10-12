# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks for existing LightOp activation exports."""

import pytest

import lightop
import lightop.activation


EXISTING_ACTIVATION_APIS = [
    "AddDropout",
    "DropPath",
    "fuse_silu_and_mul",
    "fuse_silu_and_mul_ep",
    "fuse_silu_mul_fp8_quant",
    "fuse_silu_mul_fp8_quant_ep",
    "fuse_silu_mul_per_token_quant",
    "fuse_silu_mul_quant",
    "fuse_silu_mul_quant_ep",
    "FuseBiasDropoutAdd",
    "FuseBiasGelu",
    "FuseBiasSwiGLU",
    "GNSilu",
    "LnAddDropout",
    "LnAddDropoutBias",
    "LnDropout",
    "LnDropPath",
    "relu2",
    "ReluDropout",
]


@pytest.mark.parametrize("name", EXISTING_ACTIVATION_APIS)
def test_existing_activation_api_is_direct_reexport(name):
    assert getattr(lightop.activation, name) is getattr(lightop, name)


def test_activation_public_api_inventory():
    assert set(lightop.activation.__all__) == set(EXISTING_ACTIVATION_APIS) | {
        "lm_fuse_silu_mul_quant"
    }
