#!/usr/bin/env python3
"""Accuracy: add_rms_norm_dynamic_quant vs torch reference.

Primary (must pass): quant_tokens / per_token_scale / after_res vs torch math.
after_norm (mode=2): kernel stores (x+res)*weight*rms, i.e. the full RMSNorm
output, and is checked against the same torch RMS reference.
"""
from __future__ import annotations

import pytest
import torch

from lightop import add_rms_norm_dynamic_quant


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU"
)


def _torch_add_rms_norm_dynamic_quant(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    smooth_scale: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
):
    """Math reference (float32).

    y = rms_norm(hidden + residual) * weight * smooth
    scale = max(|y|) / 127  (or 1 if max==0)
    q = round(y / scale).clamp(-127, 127)
    after_res = hidden + residual
    after_norm = rms_norm(hidden + residual) * weight
    """
    x = hidden_states.float() + residual.float()
    after_res = x.to(hidden_states.dtype)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    after_norm = (x * rms * norm_weight.float()).to(hidden_states.dtype)
    y = after_norm.float() * smooth_scale.float()
    amax = y.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(y / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, scale, after_res, after_norm


def run_case(num_tokens: int, hidden: int, mode: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    src = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
    residual = torch.randn_like(src)
    weight = torch.randn(hidden, device=device, dtype=torch.float32).abs() + 0.1
    smooth = torch.randn(hidden, device=device, dtype=torch.float32).abs() + 0.1
    eps = 1e-5

    q = torch.empty(num_tokens, hidden, device=device, dtype=torch.int8)
    scale = torch.empty(num_tokens, device=device, dtype=torch.float32)
    after_res = torch.empty_like(src) if mode == 1 else src.new_empty(0)
    after_norm = torch.empty_like(src) if mode == 2 else src.new_empty(0)

    add_rms_norm_dynamic_quant(
        src, weight, smooth, residual, q, scale, after_res, after_norm, mode, eps
    )
    torch.cuda.synchronize()

    q_ref, scale_ref, ar_ref, an_ref = _torch_add_rms_norm_dynamic_quant(
        src, weight, smooth, residual, eps
    )

    q_diff = int((q.int() - q_ref.int()).abs().max().item())
    s_diff = float((scale - scale_ref).abs().max().item())
    ok = q_diff <= 1 and s_diff < 2e-3

    extra = {}
    if mode == 1:
        ar_diff = float((after_res.float() - ar_ref.float()).abs().max().item())
        extra["after_res"] = ar_diff
        ok = ok and ar_diff <= 2e-2
    if mode == 2:
        # The kernel writes r_val_norm * rms, i.e. the full RMSNorm output
        # (x+res)*weight*rms, so it is checked against the torch RMS reference.
        # Compare with atol+rtol because one bf16 ULP at |x|~8 is already 0.03.
        an_diff = float((after_norm.float() - an_ref.float()).abs().max().item())
        an_close = bool(
            torch.allclose(after_norm.float(), an_ref.float(), atol=2e-2, rtol=2e-2)
        )
        extra["after_norm_vs_torch_rms"] = an_diff
        ok = ok and an_close

    return {
        "tag": f"T={num_tokens} H={hidden} mode={mode}",
        "q_max_diff": q_diff,
        "scale_max_diff": s_diff,
        **extra,
        "ok": ok,
    }


@pytest.mark.parametrize(("num_tokens", "hidden"), ((16, 2048), (256, 4096), (2048, 5120)))
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_add_rms_norm_dynamic_quant(num_tokens: int, hidden: int, mode: int) -> None:
    """Primary contract: quantized tokens, per-token scale and (mode 1) after_res."""
    result = run_case(num_tokens, hidden, mode, seed=1)
    assert result["q_max_diff"] <= 1, result
    assert result["scale_max_diff"] < 2e-3, result
    if mode == 1:
        assert result["after_res"] <= 2e-2, result
    if mode == 2:
        assert result["ok"], result


@pytest.mark.parametrize(("num_tokens", "hidden"), ((16, 2048), (256, 4096)))
def test_add_rms_norm_dynamic_quant_after_norm_matches_torch(
    num_tokens: int, hidden: int
) -> None:
    """mode=2 side buffer must equal the torch RMSNorm reference."""
    result = run_case(num_tokens, hidden, 2, seed=1)
    assert result["ok"], result


def main() -> int:
    cases = []
    for h in (2048, 4096, 5120, 3072, 7168):
        for t in (16, 256, 2048):
            for mode in (0, 1, 2):
                cases.append((t, h, mode))

    failed = 0
    for i, (t, h, mode) in enumerate(cases):
        r = run_case(t, h, mode, seed=i + 1)
        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            failed += 1
        extras = []
        if "after_res" in r:
            extras.append(f"after_res={r['after_res']:.6g}")
        if "after_norm_vs_torch_rms" in r:
            extras.append(f"an_vs_torch={r['after_norm_vs_torch_rms']:.6g}")
            extras.append(f"an_vs_xw={r['after_norm_vs_x_times_w']:.6g}")
        extra_s = (" " + " ".join(extras)) if extras else ""
        print(
            f"[{status}] {r['tag']} q_diff={r['q_max_diff']} "
            f"scale_diff={r['scale_max_diff']:.6g}{extra_s} (vs torch)"
        )
    print(f"\nSummary: failed={failed}/{len(cases)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
