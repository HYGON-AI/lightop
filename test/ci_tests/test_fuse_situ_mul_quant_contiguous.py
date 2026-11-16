# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Final validation for lightop.fuse_situ_mul_quant_contiguous.

Correctness reference: sglang-das Triton implementation
`fuse_situ_mul_quant_contiguous` at
`sglang-das/python/sglang/srt/layers/moe/ep_moe/layer.py`.

Default shapes / betas come from a real Kimi K3 log
`[hidden=3072, situ_beta=4.0, situ_linear_beta=25.0, bf16]` with per-TP
token counts observed at DP0 across TP0..TP3.
"""
import argparse
import statistics
import sys
import time

import torch

import lightop  # noqa: F401
from lightop.activation import fuse_situ_mul_quant_contiguous

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    from triton.language.extra import libdevice  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _pytorch_reference(input_t, situ_beta, situ_linear_beta):
    tokens, doubled = input_t.shape
    hidden = doubled // 2
    x = input_t.to(torch.float32)
    gate = x[..., :hidden]
    up = x[..., hidden:]
    gate_tanh = 2.0 * torch.sigmoid(2.0 * gate / situ_beta) - 1.0
    up_tanh = 2.0 * torch.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    activated = (
        situ_beta * gate_tanh * torch.sigmoid(gate) * situ_linear_beta * up_tanh
    )
    amax = activated.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(activated / scale).clamp_(-127.0, 127.0).to(torch.int8)
    return q, scale


if _HAS_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _situ_kernel(
        input_ptr,
        output_ptr,
        scales_ptr,
        hidden: tl.constexpr,
        situ_beta: tl.constexpr,
        situ_linear_beta: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden
        row_input = input_ptr + row * (2 * hidden)
        gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(row_input + hidden + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
        up_tanh = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
        activated = (
            situ_beta
            * gate_tanh
            * tl.sigmoid(gate)
            * situ_linear_beta
            * up_tanh
        )
        amax = tl.max(tl.abs(activated), axis=0)
        scale = tl.where(amax > 0.0, amax / 127.0, 1.0)
        quantized = libdevice.rint(activated / scale)
        quantized = tl.maximum(-127.0, tl.minimum(127.0, quantized)).to(tl.int8)
        tl.store(output_ptr + row * hidden + offsets, quantized, mask=mask)
        tl.store(scales_ptr + row, scale)


    def triton_reference(input_t, situ_beta, situ_linear_beta):
        tokens, doubled = input_t.shape
        hidden = doubled // 2
        output = torch.empty(
            (tokens, hidden), dtype=torch.int8, device=input_t.device
        )
        scales = torch.empty(
            (tokens, 1), dtype=torch.float32, device=input_t.device
        )
        _situ_kernel[(tokens,)](
            input_t,
            output,
            scales,
            hidden=hidden,
            situ_beta=float(situ_beta),
            situ_linear_beta=float(situ_linear_beta),
            BLOCK_SIZE=triton.next_power_of_2(hidden),
        )
        return output, scales


def _bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _check_pair(name, got_q, got_s, ref_q, ref_s, atol_int=1, rtol_scale=1e-3):
    diff = (got_q.to(torch.int32) - ref_q.to(torch.int32)).abs()
    max_int_diff = int(diff.max().item())
    mismatch_pct = 100.0 * (diff > atol_int).float().mean().item()
    s_diff = (got_s.float() - ref_s.float()).abs()
    denom = torch.clamp(ref_s.float().abs(), min=1e-12)
    rel_s = (s_diff / denom).max().item()
    ok_int = mismatch_pct == 0.0 and max_int_diff <= atol_int
    ok_scale = rel_s <= rtol_scale
    status = "PASS" if (ok_int and ok_scale) else "FAIL"
    print(
        f"  [{name}] status={status} max_int_diff={max_int_diff} "
        f"beyond_atol_pct={mismatch_pct:.4f}% scale_max_rel_diff={rel_s:.2e}"
    )
    return ok_int and ok_scale


def run_case(tokens, hidden, dtype, situ_beta, situ_linear_beta):
    print(
        f"\n[shape] tokens={tokens} hidden={hidden} dtype={dtype} "
        f"beta={situ_beta} linear_beta={situ_linear_beta}"
    )
    torch.manual_seed(0)
    input_t = torch.randn(
        (tokens, 2 * hidden), dtype=dtype, device="cuda"
    ).contiguous()
    input_t[tokens // 2 : tokens // 2 + max(1, tokens // 16)].mul_(5.0)

    got_q, got_s = fuse_situ_mul_quant_contiguous(
        input_t, situ_beta, situ_linear_beta
    )

    ref_q_pt, ref_s_pt = _pytorch_reference(
        input_t, situ_beta, situ_linear_beta
    )
    ok_pt = _check_pair("pytorch", got_q, got_s, ref_q_pt, ref_s_pt)

    ok_tri = True
    tri_us = None
    if _HAS_TRITON:
        ref_q_tri, ref_s_tri = triton_reference(
            input_t, situ_beta, situ_linear_beta
        )
        ok_tri = _check_pair("triton ", got_q, got_s, ref_q_tri, ref_s_tri)

    def run_lightop():
        fuse_situ_mul_quant_contiguous(input_t, situ_beta, situ_linear_beta)

    lightop_us = _bench(run_lightop) * 1e6

    if _HAS_TRITON:
        def run_triton():
            triton_reference(input_t, situ_beta, situ_linear_beta)
        tri_us = _bench(run_triton) * 1e6

    bytes_per_row = 2 * (2 * hidden) + hidden + 4
    total_bytes = tokens * bytes_per_row
    tbps_lightop = total_bytes / (lightop_us * 1e-6) / 1e12
    print(
        f"  [perf] lightop  mean={lightop_us:.2f} us  effBW={tbps_lightop:.3f} TB/s"
    )
    if tri_us is not None:
        tbps_tri = total_bytes / (tri_us * 1e-6) / 1e12
        speedup = tri_us / lightop_us
        print(
            f"  [perf] triton   mean={tri_us:.2f} us  effBW={tbps_tri:.3f} TB/s"
            f"   speedup(triton/lightop)={speedup:.2f}x"
        )
    return {
        "ok_pt": ok_pt,
        "ok_tri": ok_tri,
        "lightop_us": lightop_us,
        "tri_us": tri_us,
        "eff_bw_tbps": tbps_lightop,
        "tokens": tokens,
        "hidden": hidden,
        "dtype": str(dtype).replace("torch.", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situ-beta", type=float, default=4.0)
    parser.add_argument("--situ-linear-beta", type=float, default=25.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokens", type=int, nargs="*", default=None)
    parser.add_argument("--hidden", type=int, nargs="*", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    # Real Kimi K3 per-TP token counts at hidden=3072 (2*hidden=6144). Values
    # come from a runtime log across TP0..TP3 in the same batch: 6656, 4096,
    # 2816, 1536. Small (128) and large (8192) sanity shapes are kept for
    # coverage of the dispatch table.
    default_shapes = [
        (128, 3072),
        (1536, 3072),
        (2816, 3072),
        (4096, 3072),
        (6656, 3072),
        (8192, 3072),
    ]
    if args.tokens or args.hidden:
        default_shapes = list(zip(args.tokens or [4096], args.hidden or [3072]))

    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"gcn={torch.cuda.get_device_properties(0).gcnArchName}"
    )

    rows = []
    for t, h in default_shapes:
        rows.append(run_case(t, h, dtype, args.situ_beta, args.situ_linear_beta))

    all_ok = all(r["ok_pt"] and r["ok_tri"] for r in rows)
    print("\n===== Summary (fuse_situ_mul_quant_contiguous) =====")
    print(
        f"{'tokens':>7} {'hidden':>7} {'dtype':>10}"
        f" {'lightop_us':>11} {'tri_us':>10} {'effBW_TBps':>12}"
        f" {'speedup':>8} {'status':>7}"
    )
    for r in rows:
        speedup = "-"
        if r["tri_us"] is not None:
            speedup = f"{r['tri_us']/r['lightop_us']:.2f}x"
        status = "PASS" if r["ok_pt"] and r["ok_tri"] else "FAIL"
        print(
            f"{r['tokens']:>7} {r['hidden']:>7} {r['dtype']:>10}"
            f" {r['lightop_us']:>11.2f}"
            f" {(-1 if r['tri_us'] is None else r['tri_us']):>10.2f}"
            f" {r['eff_bw_tbps']:>12.3f}"
            f" {speedup:>8} {status:>7}"
        )
    print("=================================================")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
