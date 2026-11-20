#!/usr/bin/env python3
"""Accuracy check for zhufzh moe_quant_group_gemm modes.

Findings:
- mode 1002 (BALANCED_M_GRID + baseline .co): correct with general scales
- mode 1016 (FAST_BF16 / WGMF): correct only when a_scale=b_scale=1
  (xpu-perf lightop vendor hardcodes 1016 and uses torch.ones scales)
"""
from __future__ import annotations

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "moe_groupgemm_w4a8")
)
from w4a8_moe_groupgemm_marlin_quant import w4a8_2_marlin_weight  # noqa: E402

from lightop.moe import moe_groupgemm_marlin_w4a8
from lightop.quant import per_token_quant_int8


def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    e, n, k_half = tensor_int8.shape
    high4 = tensor_int8 & 0xF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((e, n, k_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4
    unpacked[:, :, 1::2] = low4
    return unpacked


def native_w8a8_per_token_matmul(a, b, a_s, b_s, output_dtype=torch.bfloat16):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m = a.numel() // a.shape[-1]
    b = b.t()
    n_in, n_out = b.shape
    origin = a.shape[:-1] + (n_out,)
    a = a.reshape(m, n_in)
    c = torch.matmul(a, b)
    c = a_s * c * b_s.view(1, -1)
    return c.reshape(origin).to(output_dtype)


def run_case(
    *,
    experts: int,
    tokens_per_expert: int,
    k: int,
    n: int,
    mode: int,
    unit_scale: bool,
    seed: int = 0,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> dict:
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    m = experts * tokens_per_expert

    hidden = torch.randn(m, k, device=device, dtype=dtype)
    w_packed = torch.randint(-128, 127, (experts, n, k // 2), device=device, dtype=torch.int8)
    if unit_scale:
        w_scale = torch.ones(experts, n, 1, device=device, dtype=torch.float32)
    else:
        w_scale = torch.rand(experts, n, 1, device=device, dtype=torch.float32) * 0.01 + 1e-4

    q_a, a_scale = per_token_quant_int8(hidden)
    if unit_scale:
        a_scale = torch.ones_like(a_scale)

    w_unpacked = unpack_int8_to_int4(w_packed)
    refs = []
    start = 0
    for e in range(experts):
        end = start + tokens_per_expert
        refs.append(
            native_w8a8_per_token_matmul(
                q_a[start:end], w_unpacked[e], a_scale[start:end], w_scale[e], dtype
            )
        )
        start = end
    ref = torch.cat(refs, dim=0)

    w_marlin = torch.stack([w4a8_2_marlin_weight(w_packed[i]) for i in range(experts)], dim=0)
    counts = [tokens_per_expert] * experts
    offsets = list(itertools.accumulate(counts, initial=0))[:-1]
    t_counts = torch.tensor(counts, dtype=torch.int32, device=device)
    t_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    out_base = torch.zeros(m, n, device=device, dtype=dtype)
    out_opt = torch.zeros_like(out_base)
    moe_groupgemm_marlin_w4a8(
        q_a, w_marlin, out_base, a_scale, w_scale, t_counts, t_offsets, {"MODE": 1000}
    )
    torch.cuda.synchronize()
    moe_groupgemm_marlin_w4a8(
        q_a, w_marlin, out_opt, a_scale, w_scale, t_counts, t_offsets, {"MODE": mode}
    )
    torch.cuda.synchronize()

    def stats(a, b):
        d = (a.float() - b.float()).abs()
        return {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "allclose": bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)),
        }

    return {
        "tag": f"mode={mode} unit_scale={unit_scale} E={experts} tpe={tokens_per_expert} "
        f"M={m} K={k} N={n}",
        "vs_ref_1000": stats(out_base, ref),
        "vs_ref_opt": stats(out_opt, ref),
        "vs_1000": stats(out_opt, out_base),
    }


CASES = [
    # usable production path: BALANCED_M_GRID with real scales
    dict(experts=8, tokens_per_expert=16, k=2048, n=512, mode=1002, unit_scale=False),
    dict(experts=8, tokens_per_expert=128, k=2048, n=1024, mode=1002, unit_scale=False),
    dict(experts=16, tokens_per_expert=64, k=2048, n=2048, mode=1002, unit_scale=False),
    dict(experts=32, tokens_per_expert=8, k=4096, n=3072, mode=1002, unit_scale=False),
    # xpu-perf path: MODE 1016 + ones scales
    dict(experts=8, tokens_per_expert=16, k=2048, n=512, mode=1016, unit_scale=True),
    dict(experts=8, tokens_per_expert=128, k=2048, n=1024, mode=1016, unit_scale=True),
    dict(experts=16, tokens_per_expert=64, k=2048, n=2048, mode=1016, unit_scale=True),
    dict(experts=32, tokens_per_expert=8, k=4096, n=3072, mode=1016, unit_scale=True),
    # negative control: 1016 with real scales must fail
    dict(experts=8, tokens_per_expert=128, k=2048, n=512, mode=1016, unit_scale=False),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mode{c['mode']}_unit{c['unit_scale']}_E{c['experts']}_tpe{c['tokens_per_expert']}_N{c['n']}")
def test_moe_quant_group_gemm_acc(case: dict) -> None:
    if case["mode"] == 1016 and not case["unit_scale"]:
        # Negative control: FAST_* variants are only valid with unit scales, so the
        # call must either raise or produce values outside tolerance.
        try:
            result = run_case(**case, seed=1)
        except RuntimeError:
            return
        close = (
            result["vs_ref_1000"]["allclose"]
            and result["vs_ref_opt"]["allclose"]
            and result["vs_1000"]["allclose"]
        )
        assert not close, result
        return

    result = run_case(**case, seed=1)
    assert result["vs_ref_1000"]["allclose"], result
    assert result["vs_ref_opt"]["allclose"], result
    assert result["vs_1000"]["allclose"], result


def main():
    cases = CASES
    failed = 0
    for i, c in enumerate(cases):
        expect_fail = c["mode"] == 1016 and not c["unit_scale"]
        try:
            r = run_case(**c, seed=i + 1)
            ok = (
                r["vs_ref_1000"]["allclose"]
                and r["vs_ref_opt"]["allclose"]
                and r["vs_1000"]["allclose"]
            )
            if expect_fail:
                # expect NOT close
                status = "PASS(expected-fail)" if not ok else "FAIL(unexpected-pass)"
                if ok:
                    failed += 1
            else:
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failed += 1
            print(
                f"[{status}] {r['tag']}\n"
                f"  1000 vs ref: max={r['vs_ref_1000']['max_abs']:.6g} close={r['vs_ref_1000']['allclose']}\n"
                f"  opt  vs ref: max={r['vs_ref_opt']['max_abs']:.6g} close={r['vs_ref_opt']['allclose']}\n"
                f"  opt  vs 1000: max={r['vs_1000']['max_abs']:.6g} close={r['vs_1000']['allclose']}"
            )
        except Exception as e:
            failed += 1
            print(f"[ERROR] {c}: {type(e).__name__}: {e}")

    print(f"\nSummary: failed={failed}/{len(cases)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
