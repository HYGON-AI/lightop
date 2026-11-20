"""Elementwise / index / embedding helpers ported from custom_ops."""

from __future__ import annotations

import torch

from . import op


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    x = x.contiguous()
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return op.gelu_opt(x, approximate)
    return torch.nn.functional.gelu(x, approximate=approximate)


def gelu_out(
    x: torch.Tensor, out: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    x = x.contiguous()
    return op.gelu_opt_out(x, out, approximate)


def silu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return op.silu_opt(x)
    return torch.nn.functional.silu(x)


def silu_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    return op.silu_opt_out(x, out)


def log(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return op.log_opt(x)
    return torch.log(x)


def log_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    return op.log_opt_out(x, out)


def reduce_max(x: torch.Tensor):
    """Row-wise argmax for contiguous 2D CUDA tensors; else torch.max(dim=-1)."""
    x = x.contiguous()
    if x.is_cuda and x.dim() == 2 and x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return op.reduce_max_opt(x)
    return torch.max(x, dim=-1)


def reduce_max_out(x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor):
    x = x.contiguous()
    return op.reduce_max_opt_out(x, values, indices)


def reduce_min(x: torch.Tensor):
    x = x.contiguous()
    if x.is_cuda and x.dim() == 2 and x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return op.reduce_min_opt(x)
    return torch.min(x, dim=-1)


def reduce_min_out(x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor):
    x = x.contiguous()
    return op.reduce_min_opt_out(x, values, indices)


def index_select(src: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    src = src.contiguous()
    index = index.contiguous()
    if (
        src.is_cuda
        and dim == 0
        and src.dim() == 2
        and index.dim() == 1
        and src.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return op.index_select_opt(src, dim, index)
    return torch.index_select(src, dim, index)


def index_select_out(
    src: torch.Tensor, dim: int, index: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    return op.index_select_opt_out(src.contiguous(), dim, index.contiguous(), out)


def index_add(
    dst: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    assume_unique: bool = False,
) -> torch.Tensor:
    dst = dst.contiguous()
    src = src.contiguous()
    index = index.contiguous()
    if (
        dst.is_cuda
        and dim == 0
        and dst.dim() == 2
        and src.dim() == 2
        and dst.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return op.index_add_opt(dst, dim, index, src, assume_unique)
    return dst.index_add(dim, index, src)


def index_add_out(
    dst: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    assume_unique: bool = False,
) -> torch.Tensor:
    return op.index_add_opt_out(
        dst, dim, index.contiguous(), src.contiguous(), assume_unique
    )


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
