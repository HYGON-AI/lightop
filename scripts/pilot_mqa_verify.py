#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 MQA logits module-split pilot verification (run on an HCU machine).

Build modes to verify, in order (develop installs shown; wheel works the same):

  1) Default baseline (must stay byte-for-byte unchanged):
       python setup.py develop
       python scripts/pilot_mqa_verify.py --baseline

  2) Dual-track split (op_full + lightop._ext.op_mqa_logits):
       LIGHTOP_ENABLE_MQA_MODULE=1 python setup.py develop
       python scripts/pilot_mqa_verify.py

  3) Selective single-module build (developer speedup demo):
       LIGHTOP_MODULES=mqa_logits python setup.py develop
       python scripts/pilot_mqa_verify.py --selective

  4) Pytest regression (any mode):
       python -m pytest -q test/mqa_logits/test_mqa_logits.py
       python -m pytest -q test/test_dsv41_mqa_prep.py

  5) Timing comparison (record for review):
       time python setup.py develop                          # full build
       touch csrc/mqa_logits/paged/paged_mqa_logits.cu
       time python setup.py develop                          # incremental
       LIGHTOP_MODULES=mqa_logits time python setup.py develop  # single module

NOTE: when switching build modes in a develop tree, remove stale artifacts
first (build/ and any python/lightop/*.so / python/lightop/_ext/*.so left over
from the previous mode), otherwise old .so files shadow the new layout.

Checks
------
* baseline : lightop.op is the monolith .so; every MQA symbol is present.
* split    : routing marker; symbol/constant parity op_full vs op_mqa_logits;
             aggregator routes MQA symbols to op_mqa_logits (object identity);
             non-migrated root dsv41 symbols still come from op_full;
             bit-exact A/B on random inputs (dsv41_mqa_query_lens in-process,
             mqa_logits in a subprocess so a .co load abort cannot kill us).
* selective: MQA symbols resolve without the monolith; unbuilt-family symbols
             are absent by design.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

# --- symbol inventory (must match csrc/mqa_logits/binding_mqa.cpp) ----------
MQA_FUNCS = [
    "mqa_logits",
    "get_paged_mqa_logits_metadata",
    "dsv41_mqa_query_lens",
    "dsv41_mqa_prefill_metadata",
    "dsv41_gather_index_k_fp4",
    "fp8_fp4_mqa_logits",
    "paged_mqa_logits",
    "paged_mqa_logits_persisten",
    "paged_mqa_logits_fp4",
    "get_paged_mqa_logits_fp4_metadata",
    "dsv41_candidate_block_table",
    "dsv41_select_candidate_blocks",
    "dequantize_index_k_int8_paged",
    "page_mqa_logits_sparse_mask",
    "page_mqa_logits_sparse_mask_grouped",
    "page_mqa_logits_sparse_mask_grouped_int8",
    # bind_paged_sparse_mqa group
    "fp8_fp4_paged_sparse_mqa_logits",
    "sparse_fp8_fp4_paged_mqa_logits",
    "get_paged_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata_bytes",
    "prefill_sparse_fp8_fp4_paged_mqa_logits",
    # bind_prefill_sparse_fusion group
    "prefill_sparse_native_prepare_query",
    "prefill_sparse_native_prepare_candidates",
    "prefill_sparse_native_finalize",
    "prefill_sparse_sort_candidates",
    "prefill_sparse_finalize_indices",
]
MQA_CONSTS = [
    "_fp8_fp4_mqa_logits_supports_candidates",
    "PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS",
    "PAGED_MQA_LOGITS_FP4_MAX_SCHEDULE_ROWS",
    "PAGED_SPARSE_MQA_LOGITS_ABI",
    "PAGED_SPARSE_MQA_METADATA_FORMAT",
    "PREFILL_SPARSE_SORT_ABI",
    "PREFILL_SPARSE_FUSION_ABI",
]
# Bound inside export.cpp's MQA region but DEFINED in root
# csrc/kernels/dsv41_pair_partners.cu -> intentionally kept in the monolith.
KEPT_IN_MONOLITH = [
    "dsv41_build_swa_token_ids",
    "dsv41_remap_indices_to_staging",
    "dsv41_ragged_topk_postprocess",
    "dsv41_pair_partners",
]

_RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    _RESULTS.append((ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def have_cuda() -> bool:
    import torch
    return torch.cuda.is_available() and torch.version.hip is not None


def ab_query_lens(full_mod, mqa_mod) -> None:
    """Bit-exact A/B for dsv41_mqa_query_lens (pure HIP kernel, no .co)."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(20261118)
    b = 16
    seq_lens = torch.randint(1, 65, (b,), generator=g, device="cuda", dtype=torch.int32)
    start_loc = torch.cat([
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seq_lens, 0, dtype=torch.int32)[:-1],
    ])
    ref = full_mod.dsv41_mqa_query_lens(start_loc, seq_lens, 1, 0)
    new = mqa_mod.dsv41_mqa_query_lens(start_loc, seq_lens, 1, 0)
    check(torch.equal(ref, new), "A/B bit-exact: dsv41_mqa_query_lens (random int32, B=16)")


def ab_mqa_logits_child() -> int:
    """Child process: bit-exact A/B for mqa_logits (FP16 asm path).

    Runs in a subprocess because a missing/mismatched .co can abort the
    process inside hipModuleLoad. Exit codes: 0=equal, 1=mismatch/error,
    2=skip (no .co for this gfx).
    """
    import torch
    import lightop  # noqa: F401  (sets LIGHTOP_ASM_DIR from package resources)
    from lightop.utils import LIGHTOP_ASM_DIR

    co = os.path.join(LIGHTOP_ASM_DIR, "mqa_logits", "test_mqa_logits.co")
    if not os.path.exists(co):
        print(f"SKIP: {co} not present for this gfx")
        return 2
    full_mod = try_import("lightop.op_full")
    mqa_mod = try_import("lightop._ext.op_mqa_logits")
    if full_mod is None or mqa_mod is None:
        print("SKIP: dual-track modules not both present")
        return 2
    m, h, n, d = 8, 4, 64, 128
    g = torch.Generator(device="cuda").manual_seed(20261118)
    q = torch.randn(m, h, d, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(n, d, generator=g, device="cuda", dtype=torch.float16)
    w = torch.rand(m, n, generator=g, device="cuda", dtype=torch.float32)
    ks = torch.zeros(m, dtype=torch.int32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    args = (q, k, w, ks, ke, m, n, h, d, None, True, None)
    ref = full_mod.mqa_logits(*args)
    new = mqa_mod.mqa_logits(*args)
    if torch.equal(ref, new):
        print("EQUAL")
        return 0
    print("MISMATCH")
    return 1


def ab_mqa_logits() -> None:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--ab-mqa-child"],
        capture_output=True, text=True, timeout=600,
    )
    out = (proc.stdout or "").strip().splitlines()
    tail = out[-1] if out else (proc.stderr or "").strip()[-200:]
    if proc.returncode == 0 and tail == "EQUAL":
        check(True, "A/B bit-exact: mqa_logits (FP16, M=8 H=4 N=64, asm .co path)")
    elif proc.returncode == 2:
        check(True, f"A/B mqa_logits skipped: {tail}")
    else:
        check(False, f"A/B mqa_logits failed (rc={proc.returncode}): {tail}")


def verify_baseline(op) -> None:
    so = getattr(op, "__file__", "") or ""
    check(so.endswith(".so"), f"baseline: lightop.op is the monolith .so ({so})")
    missing = [n for n in MQA_FUNCS + MQA_CONSTS + KEPT_IN_MONOLITH if not hasattr(op, n)]
    check(not missing, f"baseline: all {len(MQA_FUNCS) + len(MQA_CONSTS)} MQA symbols + "
                       f"{len(KEPT_IN_MONOLITH)} kept dsv41 symbols present"
                       + (f" (missing: {missing})" if missing else ""))


def verify_split(op, full_mod, mqa_mod) -> None:
    check(getattr(mqa_mod, "LIGHTOP_MQA_MODULE", 0) == 1,
          "split: op_mqa_logits carries routing marker LIGHTOP_MQA_MODULE=1")

    missing_new = [n for n in MQA_FUNCS + MQA_CONSTS if not hasattr(mqa_mod, n)]
    check(not missing_new, f"split: op_mqa_logits exposes full MQA surface "
                           f"({len(MQA_FUNCS)} funcs + {len(MQA_CONSTS)} consts)"
                           + (f" (missing: {missing_new})" if missing_new else ""))
    missing_full = [n for n in MQA_FUNCS + MQA_CONSTS if not hasattr(full_mod, n)]
    check(not missing_full, "split: op_full still exposes the same MQA surface (dual-track)"
                            + (f" (missing: {missing_full})" if missing_full else ""))

    bad_consts = [n for n in MQA_CONSTS
                  if hasattr(mqa_mod, n) and hasattr(full_mod, n)
                  and getattr(mqa_mod, n) != getattr(full_mod, n)]
    check(not bad_consts, "split: module constants identical op_full vs op_mqa_logits"
                          + (f" (differ: {bad_consts})" if bad_consts else ""))

    misrouted = [n for n in MQA_FUNCS + MQA_CONSTS
                 if hasattr(mqa_mod, n) and getattr(op, n, None) is not getattr(mqa_mod, n)]
    check(not misrouted, "split: aggregator routes every MQA symbol to op_mqa_logits"
                         + (f" (misrouted: {misrouted})" if misrouted else ""))

    kept_ok = all(hasattr(op, n) and getattr(op, n) is getattr(full_mod, n, None)
                  for n in KEPT_IN_MONOLITH)
    check(kept_ok, "split: non-migrated root dsv41 symbols still served by op_full")

    import lightop
    check(hasattr(lightop, "mqa_logits"), "split: wrapper-level lightop.mqa_logits importable")

    if have_cuda():
        ab_query_lens(full_mod, mqa_mod)
        ab_mqa_logits()
    else:
        check(True, "A/B skipped: no HIP device visible")


def verify_selective(op, mqa_mod) -> None:
    check(getattr(mqa_mod, "LIGHTOP_MQA_MODULE", 0) == 1,
          "selective: op_mqa_logits carries routing marker")
    missing = [n for n in MQA_FUNCS + MQA_CONSTS if not hasattr(op, n)]
    check(not missing, "selective: all MQA symbols resolve without the monolith"
                       + (f" (missing: {missing})" if missing else ""))
    leaked = [n for n in KEPT_IN_MONOLITH if hasattr(op, n)]
    check(not leaked, "selective: unbuilt-family symbols absent by design"
                      + (f" (stale artifacts? present: {leaked})" if leaked else ""))
    if have_cuda():
        import torch
        seq = torch.tensor([3, 5, 2], dtype=torch.int32, device="cuda")
        loc = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")
        out = op.dsv41_mqa_query_lens(loc, seq, 1, 0)
        check(out.numel() == 3, "selective: smoke call op.dsv41_mqa_query_lens works")
    else:
        check(True, "smoke call skipped: no HIP device visible")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="assert default monolith layout")
    ap.add_argument("--selective", action="store_true", help="assert single-module layout")
    ap.add_argument("--ab-mqa-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.ab_mqa_child:
        return ab_mqa_logits_child()

    import lightop
    op = lightop.op
    mqa_mod = try_import("lightop._ext.op_mqa_logits")
    full_mod = try_import("lightop.op_full")

    if args.baseline:
        verify_baseline(op)
    elif args.selective:
        if mqa_mod is None:
            check(False, "selective: lightop._ext.op_mqa_logits not importable "
                         "(build with LIGHTOP_MODULES=mqa_logits)")
        else:
            if full_mod is not None:
                print("[WARN] op_full still importable in selective mode "
                      "(stale .so from a previous build?)")
            verify_selective(op, mqa_mod)
    elif mqa_mod is not None and full_mod is not None:
        verify_split(op, full_mod, mqa_mod)
    elif mqa_mod is not None:
        print("[INFO] op_mqa_logits present without op_full -> selective layout")
        verify_selective(op, mqa_mod)
    else:
        print("[INFO] no split module found -> baseline layout")
        verify_baseline(op)

    failed = [msg for ok, msg in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed.")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        return 1
    print("Next: python -m pytest -q test/mqa_logits/test_mqa_logits.py "
          "test/test_dsv41_mqa_prep.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())