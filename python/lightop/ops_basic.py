"""Embedding / gather / scatter helpers ported from custom_ops."""

from __future__ import annotations

import torch

from . import op


def embedding(dst: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gather rows: dst[i] = weight[index[i]]. Writes into preallocated dst."""
    return op.embedding_opt(dst, index.contiguous(), weight)


def gather(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor, special: bool = False
) -> torch.Tensor:
    """dst[r,c] = src[index[r,c], c] (or row-broadcast when special=True)."""
    return op.gather_opt(dst, index, src, special)


def scatter(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor, special: bool = False
) -> torch.Tensor:
    return op.scatter_opt(dst, index, src, special)
