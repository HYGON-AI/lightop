#!/usr/bin/env python3
"""Accuracy tests for lightop.moe_gather vs torch reference (not custom_ops)."""

from __future__ import annotations

import os

import torch

from lightop.moe import moe_gather


def _torch_moe_gather(
    scatter_tokens: torch.Tensor,
    scatter_token_id: torch.Tensor,
    scatter_token_weight: torch.Tensor,
    num_tokens: int,
    residual_tokens: torch.Tensor | None = None,
    res_token_start: int = 0,
    res_token_end: int | None = None,
    res_scale: float = 0.0,
) -> torch.Tensor:
    """Torch reference: weighted scatter-add in float32, then cast to bf16."""
    device = scatter_tokens.device
    hidden = scatter_tokens.size(1)
    out = torch.zeros(num_tokens, hidden, device=device, dtype=torch.float32)

    if residual_tokens is not None and res_scale != 0.0:
        if res_token_end is None:
            res_token_end = num_tokens
        n = res_token_end - res_token_start
        out[res_token_start:res_token_end] = residual_tokens[:n].float() * float(res_scale)

    ids = scatter_token_id.to(torch.int64)
    valid = (ids >= 0) & (ids < num_tokens)
    if valid.any():
        ids_v = ids[valid]
        # [N, H] * [N, 1]
        contrib = scatter_tokens[valid].float() * scatter_token_weight[valid].float().unsqueeze(-1)
        out.index_add_(0, ids_v, contrib)

    return out.to(torch.bfloat16)


def _make_ep_direct_case(num_tokens: int, hidden: int, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    scatter = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.arange(num_tokens, device=device, dtype=torch.int32)
    weights = torch.rand(num_tokens, device=device, dtype=torch.float32) * 0.5 + 0.5
    return scatter, ids, weights


def _make_topk_case(num_tokens: int, topk: int, hidden: int, seed: int = 0, shuffle: bool = True):
    torch.manual_seed(seed)
    device = "cuda"
    dispatch = num_tokens * topk
    scatter = torch.randn(dispatch, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.arange(num_tokens, device=device, dtype=torch.int32).repeat_interleave(topk)
    weights = torch.rand(dispatch, device=device, dtype=torch.float32) * 0.5 + 0.25
    if shuffle:
        perm = torch.randperm(dispatch, device=device)
        scatter = scatter[perm].contiguous()
        ids = ids[perm].contiguous()
        weights = weights[perm].contiguous()
    return scatter, ids, weights, topk


def _check(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    topk: int = 1,
    rtol: float = 2e-2,
):
    # Kernel accumulates in bf16; vs float32 torch, error grows with #adds (topk).
    atol = max(2e-2, 1.5e-2 * topk)
    diff = (got.float() - ref.float()).abs()
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs_err={max_err:.6f} "
        f"mean_abs_err={mean_err:.6f} atol={atol:.3g} (vs torch)"
    )
    assert ok, f"{name} failed vs torch: max_abs_err={max_err} atol={atol}"


def test_ep_direct():
    scatter, ids, weights = _make_ep_direct_case(256, 1536, seed=1)
    out = torch.zeros(256, 1536, device="cuda", dtype=torch.bfloat16)
    got = moe_gather(scatter, ids, weights, 256, convergent_tokens=out, res_scale=0.0)
    ref = _torch_moe_gather(scatter, ids, weights, 256)
    _check("ep_direct_256x1536", got, ref, topk=1)


def test_ep_direct_with_residual():
    scatter, ids, weights = _make_ep_direct_case(128, 1024, seed=2)
    residual = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(128, 1024, device="cuda", dtype=torch.bfloat16)
    got = moe_gather(
        scatter,
        ids,
        weights,
        128,
        convergent_tokens=out,
        residual_tokens=residual,
        res_token_start=0,
        res_token_end=128,
        res_scale=1.0,
    )
    ref = _torch_moe_gather(scatter, ids, weights, 128, residual, 0, 128, 1.0)
    _check("ep_direct_residual", got, ref, topk=1)


def test_topk_inv_perm():
    scatter, ids, weights, topk = _make_topk_case(64, 4, 1536, seed=3, shuffle=True)
    out = torch.zeros(64, 1536, device="cuda", dtype=torch.bfloat16)
    os.environ.pop("MOE_GATHER_DENSE_LAYOUT", None)
    got = moe_gather(scatter, ids, weights, 64, convergent_tokens=out, res_scale=0.0)
    ref = _torch_moe_gather(scatter, ids, weights, 64)
    _check("topk4_inv_perm_64x1536", got, ref, topk=topk)


def test_topk_dense_layout():
    scatter, ids, weights, topk = _make_topk_case(32, 8, 2048, seed=4, shuffle=False)
    out = torch.zeros(32, 2048, device="cuda", dtype=torch.bfloat16)
    os.environ["MOE_GATHER_DENSE_LAYOUT"] = "1"
    try:
        got = moe_gather(scatter, ids, weights, 32, convergent_tokens=out, res_scale=0.0)
    finally:
        os.environ.pop("MOE_GATHER_DENSE_LAYOUT", None)
    ref = _torch_moe_gather(scatter, ids, weights, 32)
    _check("topk8_dense_32x2048", got, ref, topk=topk)


def test_topk8_inv_perm_large():
    scatter, ids, weights, topk = _make_topk_case(128, 8, 1536, seed=5, shuffle=True)
    out = torch.zeros(128, 1536, device="cuda", dtype=torch.bfloat16)
    os.environ.pop("MOE_GATHER_DENSE_LAYOUT", None)
    got = moe_gather(scatter, ids, weights, 128, convergent_tokens=out, res_scale=0.0)
    ref = _torch_moe_gather(scatter, ids, weights, 128)
    _check("topk8_inv_perm_128x1536", got, ref, topk=topk)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    test_ep_direct()
    test_ep_direct_with_residual()
    test_topk_inv_perm()
    test_topk_dense_layout()
    test_topk8_inv_perm_large()
    print("All moe_gather accuracy checks passed (vs torch).")
