"""Elementwise helpers ported from custom_ops."""

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
