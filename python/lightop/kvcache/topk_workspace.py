# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native DSV4 C4 TopK-union workspace indexing."""

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Optional

import torch

from .. import op as lightop_native


@lru_cache(maxsize=1)
def _get_workspace_native():
    module_name = "lightop._dsv4_topk_workspace"
    if find_spec(module_name) is None:
        return lightop_native
    return import_module(module_name)


def dsv4_pack_c4_topk_workspace(
    full_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    topk: int,
    *,
    capacity: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected logical C4 rows and rebase TopK/SWA indices on the GPU.

    Each valid query prefix may contain at most ``topk`` compressed entries,
    followed by SWA entries. Negative indices are ignored. Compressed indices
    address ``full_ids``; SWA indices start at ``full_ids.numel()``. Duplicate
    logical rows are stored once in ascending logical order, while each query
    retains its original index order and multiplicity. Physical IDs must fit
    int32, as required by the paged dequantization path.

    Returns int32 packed IDs, remapped query rows, and a one-element int32 GPU
    count. Capacity bounds the union without reading its count on the host;
    its unused ID suffix is -1. Padding columns in remapped rows are -1. The
    caller must keep capacity plus its SWA region within the int32 range.
    All kernels are precompiled native code and use the current GPU stream.
    """
    if capacity is None:
        capacity = min(full_ids.numel(), combined_indices.shape[0] * topk)
    packed, remapped, count = _get_workspace_native().dsv4_pack_c4_topk_workspace(
        full_ids, combined_indices, combined_lens, topk, capacity
    )
    return packed, remapped, count


__all__ = ["dsv4_pack_c4_topk_workspace"]
