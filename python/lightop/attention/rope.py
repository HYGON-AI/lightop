# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Existing LightOp RoPE APIs.

``rotary_embedding_deepseek_fuse`` keeps its default in-place behavior.
With ``round_bf16_products=True``, pass separate ``query_out`` and (when key is present) ``key_out``
tensors with unit feature stride and non-overlapping rows. This mode accepts BF16
``[tokens, heads, 64]`` inputs, int64 one-dimensional positions, a BF16
``[max_position, 64]`` cos/sin cache and no offsets. It rounds each product
to BF16 before adding, matching the unfused BF16 tensor arithmetic.
Inputs are unchanged; the function still returns None. Position indices
must be within the supplied cache, as for the existing operator.
"""

from .. import op as lightop_native
from ..fuse_rms_mrope_cuda import fuse_rms_mrope_cuda
from ..fuse_rotary_positional_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb_index_fused,
    rotary_embedding_fuse,
)

ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI = getattr(
    lightop_native, "ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI", 0
)
rotary_embedding_deepseek_fuse = lightop_native.rotary_embedding_deepseek_fuse

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb_index_fused",
    "fuse_rms_mrope_cuda",
    "rotary_embedding_deepseek_fuse",
    "rotary_embedding_fuse",
]
