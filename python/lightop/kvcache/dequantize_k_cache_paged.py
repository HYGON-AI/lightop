# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSV4 paged KV-cache dequantization APIs."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Optional

import torch

from .. import op as lightop_native


@lru_cache(maxsize=1)
def _get_counted_native():
    # A counted-only companion can preserve an installed core extension.
    # Once present, import errors must propagate instead of hiding ABI or
    # shared-library failures behind the full-core fallback.
    module_name = "lightop._dsv4_dequant_counted"
    if find_spec(module_name) is None:
        return lightop_native
    return import_module(module_name)


def dsv4_dequantize_k_cache_paged_out(
    cache_u8: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    *,
    valid_count: Optional[torch.Tensor] = None,
) -> None:
    """Dequantize selected DSV4 cache rows into caller-owned storage.

    Args:
        cache_u8: Contiguous uint8 cache with shape
            ``[num_pages, bytes_per_page]``. The first 448 values of each
            token use E4M3FN/OCP data with seven UE8M0 scales; the final 64
            BF16 values are stored as raw bytes.
        page_table: Contiguous one-dimensional int32 or int64 tensor of
            physical token locations.
        out: BF16 output tensor with shape ``[N, 1, 512]`` and inner stride 1.
            Row gaps and nonzero storage offsets are supported.
        page_size: Number of tokens per cache page. Supported values are 2,
            64, and 256.
        valid_count: Optional contiguous single-element int32 GPU tensor
            on the same device. Its value U must be in [0, N]. Only the
            first U already-packed page_table entries are read, and rows
            [U, N) of out are untouched. U is the union cardinality, not
            the per-query TopK value. None preserves the original full path.

    Note:
        All tensors must be on the same GPU and ``out`` must not overlap an
        input, including valid_count. Invalid active locations produce zero
        rows. This operator
        accepts E4M3FN/OCP bytes only; it does not infer the encoding from the
        uint8 cache dtype or fall back to another implementation.
    """
    if valid_count is None:
        lightop_native.dsv4_dequantize_k_cache_paged_out(
            cache_u8, page_table, out, page_size
        )
    else:
        _get_counted_native().dsv4_dequantize_k_cache_paged_out(
            cache_u8, page_table, out, page_size, valid_count
        )


def dequantize_k_cache_paged_hcu_counted(
    cache_u8: torch.Tensor,
    token_ids: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    fp8_fnuz: bool,
    valid_count: torch.Tensor,
) -> None:
    """Dequantize only a GPU-counted prefix of V4-layout C1/C2 KV rows.

    Unlike the original C4-specific API, any positive page size is accepted.
    E4M3FN and E4M3FNUZ are explicit encodings. Rows beyond ``valid_count``
    are neither read nor written. This is a same-stream operation with no
    device-to-host count transfer.
    """
    from .topk_workspace import _get_workspace_native

    native = _get_workspace_native()
    if not hasattr(native, "dequantize_k_cache_paged_hcu_counted"):
        raise RuntimeError("Rebuild LightOp with counted C1/C2 KV dequantization")
    native.dequantize_k_cache_paged_hcu_counted(
        cache_u8, token_ids, out, page_size, fp8_fnuz, valid_count
    )


__all__ = ["dsv4_dequantize_k_cache_paged_out", "dequantize_k_cache_paged_hcu_counted"]
