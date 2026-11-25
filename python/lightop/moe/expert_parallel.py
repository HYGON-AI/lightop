# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native expert-parallel data movement APIs."""

from __future__ import annotations

from typing import Optional

import os

import torch

from .. import op as lightop_native

ep_build_m_indices = lightop_native.ep_build_m_indices
ep_gather = lightop_native.ep_gather
ep_scatter = lightop_native.ep_scatter


def moe_gather_sp_res_range(
    num_tokens: int,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> tuple[int, int, int]:
    if sp_size > 1:
        num_res_tokens_per_rank = (num_tokens + sp_size - 1) // sp_size
        res_token_start = sp_rank * num_res_tokens_per_rank
        res_token_end = min(res_token_start + num_res_tokens_per_rank, num_tokens)
    else:
        num_res_tokens_per_rank = 1
        res_token_start = 0
        res_token_end = num_tokens
    return res_token_start, res_token_end, num_res_tokens_per_rank


def moe_gather_should_apply_residual(
    residual_tokens: Optional[torch.Tensor],
    res_token_start: int,
    res_token_end: int,
    res_scale: float,
) -> bool:
    if residual_tokens is None:
        return False
    if not isinstance(residual_tokens, torch.Tensor):
        return False
    if residual_tokens.numel() == 0:
        return False
    if res_token_end <= res_token_start:
        return False
    if res_scale == 0.0:
        return False
    return True


def _build_dense_inv_perm_from_scatter_token_id(
    scatter_token_id: torch.Tensor,
    num_tokens: int,
    topk: int,
) -> torch.Tensor:
    """CPU/GPU helper: for each token, collect up to topk scatter-row indices."""
    device = scatter_token_id.device
    inv = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
    slot = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    ids = scatter_token_id.to(torch.int64)
    for row in range(ids.numel()):
        tok = int(ids[row].item())
        if tok < 0 or tok >= num_tokens:
            continue
        k = int(slot[tok].item())
        if k < topk:
            inv[tok, k] = row
            slot[tok] += 1
    return inv


def moe_gather(
    scatter_tokens: torch.Tensor,
    scatter_token_id: torch.Tensor,
    scatter_token_weight: torch.Tensor,
    num_tokens: int,
    convergent_tokens: Optional[torch.Tensor] = None,
    residual_tokens: Optional[torch.Tensor] = None,
    res_token_start: Optional[int] = None,
    res_token_end: Optional[int] = None,
    res_scale: float = 1.0,
    sp_size: int = 1,
    sp_rank: int = 0,
    inv_perm: Optional[torch.Tensor] = None,
    *,
    expert_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather expert outputs back to tokens (custom_ops-compatible API).

    Prefer this over ``ep_gather`` when inputs are dense expert-major
    ``scatter_token_id`` / ``scatter_token_weight`` rows.
    """
    if expert_out is not None:
        scatter_tokens = expert_out
    if output is not None:
        convergent_tokens = output

    if res_token_start is None or res_token_end is None:
        res_token_start, res_token_end, _ = moe_gather_sp_res_range(
            num_tokens, sp_size=sp_size, sp_rank=sp_rank
        )

    if inv_perm is None:
        dispatch_tokens = scatter_tokens.size(0)
        if dispatch_tokens > num_tokens:
            topk = dispatch_tokens // num_tokens
            if os.environ.get("MOE_GATHER_DENSE_LAYOUT", "") == "1":
                inv_perm = torch.Tensor()
            else:
                # Prefer native builder when available.
                try:
                    inv_perm = lightop_native.moe_gather_build_inv_perm(
                        scatter_token_id, num_tokens, topk
                    )
                except Exception:
                    inv_perm = _build_dense_inv_perm_from_scatter_token_id(
                        scatter_token_id, num_tokens, topk
                    )
        else:
            inv_perm = torch.Tensor()

    apply_residual = moe_gather_should_apply_residual(
        residual_tokens, res_token_start, res_token_end, res_scale
    )
    residual_arg = residual_tokens if apply_residual else torch.Tensor()

    return lightop_native.moe_gather(
        scatter_tokens,
        scatter_token_id,
        scatter_token_weight,
        convergent_tokens if convergent_tokens is not None else torch.Tensor(),
        num_tokens,
        residual_arg,
        res_token_start,
        res_token_end,
        float(res_scale),
        inv_perm if inv_perm is not None else torch.Tensor(),
    )


__all__ = [
    "ep_build_m_indices",
    "ep_gather",
    "ep_scatter",
    "moe_gather",
    "moe_gather_build_inv_perm",
    "moe_gather_sp_res_range",
    "moe_gather_should_apply_residual",
]

# re-export native builder
moe_gather_build_inv_perm = lightop_native.moe_gather_build_inv_perm
