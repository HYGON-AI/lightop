# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native C1/C2/C4 TopK-union workspace indexing."""

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


def dsv4_pack_sparse_topk_workspace(
    full_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    topk: int,
    *,
    capacity: Optional[int] = None,
    swa_rows: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected compressed rows and rebase TopK/SWA indices on the GPU.

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
    Compression ratio and KV byte layout are deliberately not part of this
    indexing operation: C1/C2/C4 use the same logical-row contract. If provided,
    ``swa_rows`` bounds valid SWA indices. The count owns its storage rather
    than retaining the complete scan array.
    """
    if capacity is None:
        capacity = min(full_ids.numel(), combined_indices.shape[0] * topk)
    native = _get_workspace_native()
    if not hasattr(native, "dsv4_pack_sparse_topk_workspace"):
        raise RuntimeError("Rebuild LightOp with sparse C1/C2 TopK workspace support")
    packed, remapped, count = native.dsv4_pack_sparse_topk_workspace(
        full_ids,
        combined_indices,
        combined_lens,
        topk,
        capacity,
        -1 if swa_rows is None else swa_rows,
    )
    return packed, remapped, count


def dsv4_pack_c4_topk_workspace(
    full_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    topk: int,
    *,
    capacity: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility API for the original C4 workspace operator."""
    if capacity is None:
        capacity = min(full_ids.numel(), combined_indices.shape[0] * topk)
    return tuple(
        _get_workspace_native().dsv4_pack_c4_topk_workspace(
            full_ids, combined_indices, combined_lens, topk, capacity
        )
    )


def dsv4_dequantize_sparse_topk_out(
    cache: torch.Tensor,
    full_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    page_size: int,
    *,
    fp8_fnuz: bool = False,
) -> torch.Tensor:
    """Dequantize a full-capacity V4 TopK union without remapping Attention.

    ``out`` has one BF16 row per logical ``full_ids`` entry. Only selected
    compressed rows are written; query order, duplicates and SWA indices stay
    unchanged. The caller must keep the original logical row allocation and
    satisfy ``full_ids.numel() <= combined_indices.shape[0] * topk``.

    Supports gfx936/gfx938 and explicit E4M3FN/FNUZ encodings. The returned
    union count owns four bytes on the GPU; no host readback or synchronization
    is performed. This API does not implement the smaller packed layout.
    """
    native = _get_workspace_native()
    if not hasattr(native, "dsv4_dequantize_sparse_topk_out"):
        raise RuntimeError("Rebuild LightOp with native selected TopK dequantization")
    return native.dsv4_dequantize_sparse_topk_out(
        cache, full_ids, combined_indices, combined_lens, out, topk, page_size,
        fp8_fnuz,
    )


__all__ = [
    "dsv4_pack_sparse_topk_workspace", "dsv4_pack_c4_topk_workspace",
    "dsv4_dequantize_sparse_topk_out",
]
