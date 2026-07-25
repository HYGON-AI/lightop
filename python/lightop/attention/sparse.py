"""Sparse-attention index transformation APIs."""

from .. import op as lightop_native
from ..fuse_topk_transform import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    topk_transform_512,
)

convert_req_index_to_global_index = (
    lightop_native.convert_req_index_to_global_index
)
top_k_per_row_decode = lightop_native.top_k_per_row_decode
top_k_per_row_prefill = lightop_native.top_k_per_row_prefill

__all__ = [
    "convert_req_index_to_global_index",
    "fast_topk_transform_fused",
    "fast_topk_transform_ragged_fused",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_transform_512",
]
