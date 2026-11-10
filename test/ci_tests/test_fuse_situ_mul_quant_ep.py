# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Final validation for lightop.fuse_situ_mul_quant_ep (v002 grid-stride).

Correctness reference: sglang-das Triton implementation
`fuse_situ_mul_quant_ep` at
`sglang-das/python/sglang/srt/layers/moe/ep_moe/layer.py`.

v002 changes:
- The kernel no longer touches padding rows (t >= masked_m[e]); the
  downstream masked GEMM only reads valid rows, so padding output/scale is
  a don't-care. Comparison against the Triton oracle is restricted to
  valid rows accordingly.
- Added a decode-shape sweep [E=28, T_pad=512, valid ~= 5-30 per expert]
  to exercise the new expect_m-clamped grid.
"""
import argparse
import statistics
import sys
import time

import torch

import lightop  # noqa: F401
from lightop.activation import fuse_situ_mul_quant_ep

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    from triton.language.extra import libdevice  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _pytorch_reference(input_t, masked_m, situ_beta, situ_linear_beta):
    experts, tokens, doubled = input_t.shape
    hidden = doubled // 2
    x = input_t.to(torch.float32)
    gate = x[..., :hidden]
    up = x[..., hidden:]
    gate_tanh = 2.0 * torch.sigmoid(2.0 * gate / situ_beta) - 1.0
    up_tanh = 2.0 * torch.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    activated = (
        situ_beta * gate_tanh * torch.sigmoid(gate) * situ_linear_beta * up_tanh
    )
    token_idx = torch.arange(tokens, device=input_t.device).view(1, tokens)
    valid = token_idx < masked_m.view(experts, 1)
    activated = activated * valid.unsqueeze(-1)

    amax = activated.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        valid.unsqueeze(-1) & (amax > 0),
        amax / 127.0,
        torch.ones_like(amax),
    )
    q = torch.round(activated / scale).clamp_(-127.0, 127.0).to(torch.int8)
    q = q * valid.unsqueeze(-1).to(torch.int8)
    return q, scale


if _HAS_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _situ_ep_kernel(
        input_ptr,
        output_ptr,
        scales_ptr,
        masked_m_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        situ_beta: tl.constexpr,
        situ_linear_beta: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        expert = row // tokens
        token = row - expert * tokens
        valid_row = token < tl.load(masked_m_ptr + expert)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (offsets < hidden) & valid_row
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
        scale = tl.where(valid_row & (amax > 0.0), amax / 127.0, 1.0)
        quantized = libdevice.rint(activated / scale)
        quantized = tl.maximum(-127.0, tl.minimum(127.0, quantized)).to(tl.int8)
        tl.store(
            output_ptr + row * hidden + offsets,
            quantized,
            mask=offsets < hidden,
        )
        tl.store(scales_ptr + row, scale)


    def triton_reference(input_t, masked_m, situ_beta, situ_linear_beta):
        experts, tokens, doubled = input_t.shape
        hidden = doubled // 2
        output = torch.empty(
            (experts, tokens, hidden), dtype=torch.int8, device=input_t.device
        )
        scales = torch.empty(
            (experts, tokens, 1),
            dtype=torch.float32,
            device=input_t.device,
        )
        _situ_ep_kernel[(experts * tokens,)](
            input_t,
            output,
            scales,
            masked_m,
            tokens=tokens,
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


def _check_pair_valid_only(name, got_q, got_s, ref_q, ref_s, masked_m,
                           atol_int=1, rtol_scale=1e-3):
    """Compare only the rows the downstream masked GEMM will actually read.

    v002 kernel leaves padding rows untouched; the Triton reference zeros
    them. The functional contract with the downstream w4a8 masked GEMM only
    covers `[0, masked_m[e])`, so restrict the compare to those rows.
    """
    experts, tokens = got_q.shape[0], got_q.shape[1]
    row_ok_q = True
    row_ok_s = True
    max_int_diff = 0
    max_rel_s = 0.0

    for e in range(experts):
        vm = int(masked_m[e].item())
        if vm == 0:
            continue
        diff = (
            got_q[e, :vm].to(torch.int32) - ref_q[e, :vm].to(torch.int32)
        ).abs()
        max_int_diff = max(max_int_diff, int(diff.max().item()))
        if (diff > atol_int).any():
            row_ok_q = False
        s_diff = (got_s[e, :vm].float() - ref_s[e, :vm].float()).abs()
        denom = torch.clamp(ref_s[e, :vm].float().abs(), min=1e-12)
        rel = (s_diff / denom).max().item()
        max_rel_s = max(max_rel_s, rel)
        if rel > rtol_scale:
            row_ok_s = False

    ok = row_ok_q and row_ok_s
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{name}] status={status} max_int_diff={max_int_diff} "
        f"scale_max_rel_diff={max_rel_s:.2e}"
    )
    return ok


def _make_masked_m(experts, tokens_per_expert, kind, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "full":
        return torch.full(
            (experts,), tokens_per_expert, dtype=torch.int32, device="cuda"
        )
    if kind == "half":
        return torch.full(
            (experts,), tokens_per_expert // 2, dtype=torch.int32, device="cuda"
        )
    if kind == "random":
        vals = torch.randint(
            0, tokens_per_expert + 1, (experts,), generator=g, dtype=torch.int32
        )
        return vals.to("cuda")
    if kind.startswith("sparse_"):
        empty_pct = int(kind.split("_")[1])
        assert 0 < empty_pct < 100
        keep_prob = 1.0 - empty_pct / 100.0
        keep = torch.rand(experts, generator=g) < keep_prob
        vals = torch.where(
            keep,
            torch.tensor(tokens_per_expert // 2, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )
        return vals.to("cuda")
    if kind.startswith("decode"):
        # Decode-style: T is padded (e.g. 512) but each expert has just a
        # handful of tokens. `decode_N` = clamp masked_m to at most N.
        cap = int(kind.split("_")[1])
        vals = torch.randint(
            0, cap + 1, (experts,), generator=g, dtype=torch.int32
        )
        return vals.to("cuda")
    raise ValueError(kind)


def run_case(experts, tokens_per_expert, hidden, dtype, situ_beta,
             situ_linear_beta, mask_kind, expect_m=-1):
    print(
        f"\n[shape] E={experts} T_pad={tokens_per_expert} H={hidden}"
        f" dtype={dtype} mask={mask_kind} expect_m={expect_m}"
        f" beta={situ_beta} linear_beta={situ_linear_beta}"
    )
    torch.manual_seed(0)
    input_t = torch.randn(
        (experts, tokens_per_expert, 2 * hidden), dtype=dtype, device="cuda"
    ).contiguous()
    masked_m = _make_masked_m(experts, tokens_per_expert, mask_kind, seed=1)
    num_valid_rows = int(masked_m.sum().item())
    num_empty_experts = int((masked_m == 0).sum().item())
    max_valid = int(masked_m.max().item())
    if num_valid_rows == 0:
        masked_m[0] = 1
        num_valid_rows = 1
        num_empty_experts = int((masked_m == 0).sum().item())
        max_valid = 1

    # If caller did not set expect_m explicitly, derive a reasonable hint
    # from the actual mask (mirrors what sglang-das does with expected_m).
    if expect_m == -1 and mask_kind.startswith("decode"):
        expect_m = max_valid

    got_q, got_s = fuse_situ_mul_quant_ep(
        input_t, masked_m, situ_beta, situ_linear_beta, expect_m=expect_m
    )

    ref_q_pt, ref_s_pt = _pytorch_reference(
        input_t, masked_m, situ_beta, situ_linear_beta
    )
    ok_pt = _check_pair_valid_only(
        "pytorch", got_q, got_s, ref_q_pt, ref_s_pt, masked_m
    )

    ok_tri = True
    tri_us = None
    if _HAS_TRITON:
        ref_q_tri, ref_s_tri = triton_reference(
            input_t, masked_m, situ_beta, situ_linear_beta
        )
        ok_tri = _check_pair_valid_only(
            "triton ", got_q, got_s, ref_q_tri, ref_s_tri, masked_m
        )

    def run_lightop():
        fuse_situ_mul_quant_ep(
            input_t, masked_m, situ_beta, situ_linear_beta, expect_m=expect_m
        )

    lightop_us = _bench(run_lightop) * 1e6

    if _HAS_TRITON:
        def run_triton():
            triton_reference(input_t, masked_m, situ_beta, situ_linear_beta)
        tri_us = _bench(run_triton) * 1e6

    bytes_per_row = 2 * (2 * hidden) + hidden + 4
    total_bytes = num_valid_rows * bytes_per_row
    tbps_lightop = total_bytes / (lightop_us * 1e-6) / 1e12
    print(
        f"  [perf] lightop  mean={lightop_us:.2f} us  effBW={tbps_lightop:.3f} TB/s"
        f"  valid_rows={num_valid_rows}  empty_experts={num_empty_experts}/{experts}"
        f"  max_valid={max_valid}"
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
        "experts": experts,
        "tokens": tokens_per_expert,
        "hidden": hidden,
        "dtype": str(dtype).replace("torch.", ""),
        "mask_kind": mask_kind,
        "empty_experts": num_empty_experts,
        "max_valid": max_valid,
        "expect_m": expect_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situ-beta", type=float, default=4.0)
    parser.add_argument("--situ-linear-beta", type=float, default=25.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    cases = [
        # Prefill-style (large T_pad, many valid rows).
        (28, 4096, 3072, "full",       -1),
        (28, 4096, 3072, "half",       -1),
        (28, 4096, 3072, "random",     -1),
        (28, 4096, 3072, "sparse_50",  -1),
        (28, 4096, 3072, "sparse_75",  -1),
        (28, 4096, 3072, "sparse_90",  -1),
        # Decode-style (T_pad=512 CUDA-graph capacity, very few valid rows
        # per expert). expect_m=-1 -> the run_case helper derives it from
        # the observed max valid_m, mirroring the sglang-das expected_m path.
        (28, 512, 3072, "decode_5",   -1),
        (28, 512, 3072, "decode_10",  -1),
        (28, 512, 3072, "decode_30",  -1),
        # Same decode shapes, but forcing expect_m=-1 with T_pad grid
        # (pre-v002 behavior) to size the improvement precisely.
        (28, 512, 3072, "decode_10",  512),
    ]

    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"gcn={torch.cuda.get_device_properties(0).gcnArchName}"
    )

    rows = [
        run_case(e, t, h, dtype, args.situ_beta, args.situ_linear_beta, m, x)
        for (e, t, h, m, x) in cases
    ]

    all_ok = all(r["ok_pt"] and r["ok_tri"] for r in rows)
    print("\n===== Summary (fuse_situ_mul_quant_ep) =====")
    print(
        f"{'E':>3} {'T':>5} {'H':>6} {'dtype':>10} {'mask':>10}"
        f" {'exp_m':>6} {'valid':>7} {'empty':>6}"
        f" {'lightop_us':>11} {'tri_us':>10} {'effBW_TBps':>12}"
        f" {'speedup':>8} {'status':>7}"
    )
    for r in rows:
        speedup = "-"
        if r["tri_us"] is not None:
            speedup = f"{r['tri_us']/r['lightop_us']:.2f}x"
        status = "PASS" if r["ok_pt"] and r["ok_tri"] else "FAIL"
        print(
            f"{r['experts']:>3} {r['tokens']:>5} {r['hidden']:>6} {r['dtype']:>10}"
            f" {r['mask_kind']:>10}"
            f" {r['expect_m']:>6} {r['max_valid']:>7} {r['empty_experts']:>6}"
            f" {r['lightop_us']:>11.2f}"
            f" {(-1 if r['tri_us'] is None else r['tri_us']):>10.2f}"
            f" {r['eff_bw_tbps']:>12.3f}"
            f" {speedup:>8} {status:>7}"
        )
    print("============================================")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
