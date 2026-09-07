"""DSV4 paged KV-cache dequantization APIs."""

from __future__ import annotations

import torch

from .. import op as lightop_native


def dsv4_dequantize_k_cache_paged_out(
    cache_u8: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
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

    Note:
        All tensors must be on the same GPU and ``out`` must not overlap an
        input. Invalid physical locations produce zero rows. This operator
        accepts E4M3FN/OCP bytes only; it does not infer the encoding from the
        uint8 cache dtype or fall back to another implementation.
    """
    lightop_native.dsv4_dequantize_k_cache_paged_out(
        cache_u8, page_table, out, page_size
    )


__all__ = ["dsv4_dequantize_k_cache_paged_out"]
