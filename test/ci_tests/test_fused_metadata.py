# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness & benchmark test for fused_metadata_kernel_general.

Compares the CUDA implementation (lightop) against a pure Triton
reference on the same inputs, then benchmarks both side-by-side
using CUDA events for accurate GPU timing.
"""

import argparse
import math
import sys
import time
from typing import Optional

import torch

# ---- Triton reference -------------------------------------------------
import triton
import triton.language as tl


@triton.jit
def _fused_metadata_kernel_general_triton(
    seq_lens, seq_lens_stride_0,
    req_to_token, req_to_token_stride_0, req_to_token_stride_1,
    req_pool_indices, req_pool_indices_stride_0,
    cache_seqlens_int32, cache_seqlens_int32_stride_0,
    cu_seqlens_k, cu_seqlens_k_stride_0,
    page_table, page_table_stride_0, page_table_stride_1,
    swa_page_table, swa_page_table_stride_0, swa_page_table_stride_1,
    full_to_swa_mapping, full_to_swa_mapping_stride_0,
    B, max_seq_pages,
    page_size: tl.constexpr,
    seq_len_delta: tl.constexpr,
    use_swa: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    if max_seq_pages == 0:
        return

    i = pid_b
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    if page_size == 1:
        col_idx = col_offsets
    else:
        col_idx = col_offsets << SHIFT

    rt_offsets = row_offset + col_idx * req_to_token_stride_1
    page_index = tl.load(req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg")

    if page_size == 1:
        page_table_val = page_index
    else:
        page_table_val = page_index >> SHIFT

    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_table_val, mask=mask, cache_modifier=".cg")

    if use_swa:
        swa_slot = tl.load(
            full_to_swa_mapping + page_index * full_to_swa_mapping_stride_0,
            mask=mask, other=0, cache_modifier=".cg",
        )
        if page_size == 1:
            swa_val = swa_slot
        else:
            swa_val = swa_slot >> SHIFT
        swa_offsets = i * swa_page_table_stride_0 + col_offsets * swa_page_table_stride_1
        tl.store(swa_page_table + swa_offsets, swa_val, mask=mask, cache_modifier=".cg")


def run_triton(
    seq_lens,
    req_to_token,
    req_pool_indices,
    cache_seqlens_int32,
    cu_seqlens_k,
    page_table,
    swa_page_table=None,
    full_to_swa_mapping=None,
    B=None,
    max_seq_pages=None,
    page_size=1,
    seq_len_delta=0,
    use_swa=False,
    BLOCK_COLS=128,
):
    if B is None:
        B = seq_lens.size(0)
    if max_seq_pages is None:
        max_seq_pages = page_table.size(1)

    SHIFT = 0
    if page_size > 1:
        SHIFT = int(math.log2(page_size))

    grid = (B, triton.cdiv(max_seq_pages, BLOCK_COLS))

    _fused_metadata_kernel_general_triton[grid](
        seq_lens, seq_lens.stride(0),
        req_to_token, req_to_token.stride(0), req_to_token.stride(1),
        req_pool_indices, req_pool_indices.stride(0),
        cache_seqlens_int32, cache_seqlens_int32.stride(0),
        cu_seqlens_k, cu_seqlens_k.stride(0),
        page_table, page_table.stride(0), page_table.stride(1),
        swa_page_table, swa_page_table.stride(0) if swa_page_table is not None else 0,
        swa_page_table.stride(1) if swa_page_table is not None else 0,
        full_to_swa_mapping, full_to_swa_mapping.stride(0) if full_to_swa_mapping is not None else 0,
        B, max_seq_pages,
        page_size=page_size,
        seq_len_delta=seq_len_delta,
        use_swa=use_swa,
        SHIFT=SHIFT,
        BLOCK_COLS=BLOCK_COLS,
    )


# ---- Helpers -----------------------------------------------------------

def make_inputs(B, num_reqs, max_seq_pages, page_size,
                seq_len_delta, device='cuda', use_swa=False):
    """Create fixed seed inputs for reproducible benchmarking."""
    max_tokens = max_seq_pages * page_size
    torch.manual_seed(42)
    seq_lens = torch.randint(1, max_tokens // 2 + 1, (B,), dtype=torch.int32, device=device)
    req_to_token = torch.randint(0, max_tokens, (num_reqs, max_tokens), dtype=torch.int32, device=device)
    req_pool_indices = torch.randint(0, num_reqs, (B,), dtype=torch.int32, device=device)

    cache_seqlens_int32 = torch.empty(B, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(B + 1, dtype=torch.int32, device=device)
    page_table = torch.empty(B, max_seq_pages, dtype=torch.int32, device=device)

    swa_page_table = None
    full_to_swa_mapping = None
    if use_swa:
        swa_page_table = torch.empty(B, max_seq_pages, dtype=torch.int32, device=device)
        full_to_swa_mapping = torch.randint(0, max_tokens, (max_tokens,), dtype=torch.int32, device=device)

    return dict(
        seq_lens=seq_lens, req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        cu_seqlens_k=cu_seqlens_k,
        page_table=page_table,
        swa_page_table=swa_page_table,
        full_to_swa_mapping=full_to_swa_mapping,
        B=B, max_seq_pages=max_seq_pages,
        page_size=page_size, seq_len_delta=seq_len_delta,
        use_swa=use_swa,
    )


def ref_copy(kw):
    """Pure-PyTorch reference."""
    B = kw['B']
    seq_len_delta = kw['seq_len_delta']
    page_size = kw['page_size']
    use_swa = kw['use_swa']

    seq_lens = kw['seq_lens']
    cs = seq_lens + seq_len_delta
    cu = torch.empty(B + 1, dtype=torch.int32, device=seq_lens.device)
    acc = 0
    for i in range(B):
        cu[i] = acc
        acc += cs[i].item()
    cu[B] = acc

    pt_ref = torch.empty_like(kw['page_table'])
    row_idx = kw['req_pool_indices']
    r2t = kw['req_to_token']
    max_tokens = r2t.size(1)
    for i in range(B):
        row = row_idx[i].item()
        for c in range(pt_ref.size(1)):
            token_off = c * page_size
            idx = r2t[row, token_off] if token_off < max_tokens else 0
            pt_ref[i, c] = idx // page_size

    result = dict(cache_seqlens_int32=cs, cu_seqlens_k=cu, page_table=pt_ref)
    if use_swa:
        swa_ref = torch.empty_like(kw['swa_page_table'])
        fts = kw['full_to_swa_mapping']
        for i in range(B):
            row = row_idx[i].item()
            for c in range(swa_ref.size(1)):
                token_off = c * page_size
                idx = r2t[row, token_off] if token_off < max_tokens else 0
                swa_ref[i, c] = fts[idx].item() // page_size
        result['swa_page_table'] = swa_ref
    return result


# ---- Tests -------------------------------------------------------------

def test_correctness(**kw):
    """Compare CUDA vs Triton vs PyTorch reference."""
    from lightop.kvcache import fused_metadata_kernel_general as cuda_fn

    def clone_kw():
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}

    # Triton
    inp_t = clone_kw()
    run_triton(**inp_t)
    triton_out = {
        'cache_seqlens_int32': inp_t['cache_seqlens_int32'].clone(),
        'cu_seqlens_k': inp_t['cu_seqlens_k'].clone(),
        'page_table': inp_t['page_table'].clone(),
    }
    if kw['use_swa']:
        triton_out['swa_page_table'] = inp_t['swa_page_table'].clone()

    # CUDA
    inp_c = clone_kw()
    cuda_fn(**inp_c)
    cuda_out = {
        'cache_seqlens_int32': inp_c['cache_seqlens_int32'].clone(),
        'cu_seqlens_k': inp_c['cu_seqlens_k'].clone(),
        'page_table': inp_c['page_table'].clone(),
    }
    if kw['use_swa']:
        cuda_out['swa_page_table'] = inp_c['swa_page_table'].clone()

    # Reference
    ref = ref_copy(kw)

    errors = []
    for key in triton_out:
        t, c, r = triton_out[key], cuda_out[key], ref[key]
        if not torch.equal(t, r):
            errors.append(f"TRITON vs REF {key}: max_diff={(t.long()-r.long()).abs().max().item()}")
        if not torch.equal(c, r):
            errors.append(f"CUDA  vs REF {key}: max_diff={(c.long()-r.long()).abs().max().item()}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASS (all outputs match reference)")
    return True


def benchmark_gpu_events(label, kw, fn, warmup=20, iters=200):
    """Benchmark with CUDA events for precise GPU timing."""
    inp = make_inputs(**kw)  # Use the same seed
    for _ in range(warmup):
        inp_i = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        fn(**inp_i)

    # Pre-allocate once, reuse for all iterations.
    inp = make_inputs(**kw)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    batch = 10
    for _ in range(iters // batch):
        start_event.record()
        for _ in range(batch):
            fn(**inp)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms / batch)

    times.sort()
    med = times[len(times)//2]
    return med


# ---- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench-only', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = 'cuda'
    from lightop.kvcache import fused_metadata_kernel_general as cuda_fn

    test_configs = [
        # (B, num_reqs, max_seq_pages, page_size, seq_len_delta, use_swa)
        (1,  64,   512,   32, 0, False),
        (2,  64,   512,   32, 0, False),
        (4,  64,   512,   32, 0, False),
        (8,  64,   512,   32, 0, False),
        (16, 64,   512,   32, 0, False),
        (32, 64,   512,   32, 0, False),
        (64, 128,  1024,  32, 0, False),
        # SWA variants
        (4,  64,   512,   32, 0, True),
        (8,  64,   512,   32, 0, True),
        # page_size = 1
        (8,  64,   4096,  1,  0, False),
        # seq_len_delta
        (8,  64,   512,   32, 4, False),
    ]

    if args.small:
        test_configs = [
            (2, 8,  64,  16, 0, False),
            (4, 8,  64,  16, 0, True),
        ]

    if args.large:
        test_configs = [
            (8,  64,   2048,  32, 0, False),
            (8,  128,  4096,  32, 0, False),
            (64, 128,  4096,  16, 0, False),
            (128, 128, 4096,  16, 0, False),
            (8,  64,   1024,  1,  0, False),
        ]

    passed, failed = 0, 0
    results = []

    for cfg in test_configs:
        B, num_reqs, max_seq_pages, page_size, seq_len_delta, use_swa = cfg
        label = (f"B={B:3d} R={num_reqs:3d} P={max_seq_pages:4d} "
                 f"ps={page_size:2d} d={seq_len_delta} swa={str(use_swa):5s}")
        print(f"\n{'=' * 72}")
        print(f"Config: {label}")
        print(f"{'=' * 72}")

        kw = dict(B=B, num_reqs=num_reqs, max_seq_pages=max_seq_pages,
                  page_size=page_size, seq_len_delta=seq_len_delta,
                  use_swa=use_swa, device=device)

        # Correctness
        if not args.bench_only:
            inp = make_inputs(**kw)
            ok = test_correctness(**inp)
            if ok: passed += 1
            else:  failed += 1

        # Benchmark
        inp = make_inputs(**kw)
        t_triton = benchmark_gpu_events(label, kw, run_triton)
        t_cuda   = benchmark_gpu_events(label, kw, cuda_fn)
        speedup  = t_triton / t_cuda if t_cuda > 0 else float('inf')

        print(f"  Triton: {t_triton:.3f} ms | CUDA: {t_cuda:.3f} ms | Speedup: {speedup:.2f}x")
        results.append(dict(label=label, triton_ms=t_triton, cuda_ms=t_cuda, speedup=speedup))

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"Correctness: {passed} passed, {failed} failed")
    print(f"\n{'Label':<55} {'Triton(ms)':<12} {'CUDA(ms)':<12} {'Speedup':<10}")
    print('-' * 89)
    for r in results:
        lbl = r['label']
        tm  = f"{r['triton_ms']:.3f}"
        cm  = f"{r['cuda_ms']:.3f}"
        sp  = f"{r['speedup']:.2f}x"
        print(f"{lbl:<55} {tm:<12} {cm:<12} {sp:<10}")

    avg = sum(r['speedup'] for r in results) / len(results)
    print(f"\n{'Average speedup':<55} {avg:<12.2f}x")

    if args.large:
        mem_traffic = sum(
            kw['B'] * kw['max_seq_pages'] * 8  # 4B read + 4B write per page
            for _, kw in [(cfg, dict(B=cfg[0], max_seq_pages=cfg[2]))]
        ) / len(test_configs) / 1024
        print(f"\nLarge config mem traffic: ~{mem_traffic:.0f} KB per launch")

    if failed > 0:
        sys.exit(1)
    if avg < 5.0 and not args.small:
        print(f"NOTE: avg {avg:.2f}x < 5x target")


if __name__ == '__main__':
    main()
