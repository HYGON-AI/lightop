# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sampling and Top-K/Top-P APIs."""

from ..fuse_topk_transform import topk_transform_1024
from ..reduce import topk
from .markov import markov_gemv_add_argmax
from .probability import (
    get_default_generators,
    get_seed_and_offset,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_p_sampling_from_probs,
)

__all__ = [
    "markov_gemv_add_argmax",
    "get_default_generators",
    "get_seed_and_offset",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_sampling_from_probs",
    "topk",
    "topk_transform_1024",
]
