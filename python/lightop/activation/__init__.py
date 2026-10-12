# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Activation, gating, dropout and drop-path APIs."""

from importlib import import_module

from .existing import (
    AddDropout,
    DropPath,
    FuseBiasDropoutAdd,
    FuseBiasGelu,
    FuseBiasSwiGLU,
    GNSilu,
    gelu_and_mul_opt,
    gelu_tanh_and_mul_opt,
    LnAddDropout,
    LnAddDropoutBias,
    LnDropPath,
    LnDropout,
    ReluDropout,
    fuse_silu_and_mul,
    fuse_silu_and_mul_ep,
    fuse_silu_mul_fp8_quant,
    fuse_silu_mul_fp8_quant_ep,
    fuse_silu_mul_per_token_quant,
    fuse_silu_mul_quant,
    fuse_silu_mul_quant_ep,
    fuse_situ_mul_quant_contiguous,
    fuse_situ_mul_quant_ep,
    relu2,
    silu_and_mul_opt,
)

_NATIVE_APIS = {"lm_fuse_silu_mul_quant"}


def __getattr__(name):
    if name not in _NATIVE_APIS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.native"), name)
    globals()[name] = value
    return value


__all__ = [
    "AddDropout",
    "DropPath",
    "fuse_silu_and_mul",
    "fuse_silu_and_mul_ep",
    "fuse_silu_mul_fp8_quant",
    "fuse_silu_mul_fp8_quant_ep",
    "fuse_silu_mul_per_token_quant",
    "fuse_silu_mul_quant",
    "fuse_silu_mul_quant_ep",
    "fuse_situ_mul_quant_contiguous",
    "fuse_situ_mul_quant_ep",
    "FuseBiasDropoutAdd",
    "FuseBiasGelu",
    "FuseBiasSwiGLU",
    "GNSilu",
    "gelu_and_mul_opt",
    "gelu_tanh_and_mul_opt",
    "lm_fuse_silu_mul_quant",
    "LnAddDropout",
    "LnAddDropoutBias",
    "LnDropout",
    "LnDropPath",
    "relu2",
    "ReluDropout",
    "silu_and_mul_opt",
]
