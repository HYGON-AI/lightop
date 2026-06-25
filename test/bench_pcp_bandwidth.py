#!/usr/bin/env python3
"""Measure memory bandwidth of the PCP fused kernel (pcp_2k_16k)."""

import time
import torch
import os
import sys
import torch.utils.cpp_extension as cppext

# ── Constants ─────────────────────────────────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
MAX_POS = 131072
BASE = 10000.0
EPS = 1e-6

# ── Build PCP kernel ──────────────────────────────────────────────────
kernel_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lightop", "csrc",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_kernel.cu",
)
with open(kernel_path) as f:
    kernel_source = f.read()

wrapper_code = """
#include <torch/extension.h>
namespace at { namespace native {
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
    at::Tensor& q, const at::Tensor& kv, at::Tensor& k_cache,
    const at::Tensor& slot_mapping, const at::Tensor& q_position_ids,
    const at::Tensor& kv_position_ids, const at::Tensor& cos_sin_cache,
    double eps, int64_t cache_block_size);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
        &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp,
        "PCP fused kernel");
}
"""
with open("/tmp/pcp_bw_combined.cu", "w") as f:
    f.write(kernel_source)
    f.write("\n")
    f.write(wrapper_code)

pcp_mod = cppext.load(
    name="lightop_pcp_bw_test",
    sources=["/tmp/pcp_bw_combined.cu"],
    extra_include_paths=[os.path.dirname(kernel_path)],
    extra_cflags=["-O3"],
    verbose=False,
)
pcp_op = pcp_mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp


def build_cos_sin_cache(max_pos, rope_dim, base):
    half = rope_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def bench(nq, nkv, nh, block_size=16, warmup=5, iters=20):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cos_sin_cache = build_cos_sin_cache(MAX_POS, ROPE_DIM, BASE).to(device)
    q = torch.randn(nq, nh, HEAD_DIM, dtype=dtype, device=device).contiguous()
    kv = torch.randn(nkv, HEAD_DIM, dtype=dtype, device=device).contiguous()
    q_pos = torch.randint(0, 8192, (nq,), dtype=torch.int64, device=device)
    kv_pos = torch.randint(0, 8192, (nkv,), dtype=torch.int64, device=device)

    n_blocks = (nkv + block_size - 1) // block_size
    block_stride = block_size * TOKEN_DATA_BYTES + block_size * 8
    block_stride = ((block_stride + 575) // 576) * 576
    slot_map = torch.zeros(nkv, dtype=torch.int64, device=device)
    for i in range(nkv):
        slot_map[i] = (i // block_size) * block_size + (i % block_size)

    # ── Calculate bytes ──────────────────────────────────────────
    # Q: read + write (in-place), [Nq, H, 512] bf16
    bytes_q = nq * nh * HEAD_DIM * 2  # bf16 = 2 bytes
    # KV: read only, [Nkv, 512] bf16
    bytes_kv_read = nkv * HEAD_DIM * 2
    # K cache writes:
    #   fp8 nope: [Nkv, 448] uint8
    bytes_kv_fp8 = nkv * NOPE_DIM
    #   bf16 rope: [Nkv, 128] bf16
    bytes_kv_bf16 = nkv * ROPE_DIM * 2
    #   scales: [Nkv, 8] uint8
    bytes_kv_scale = nkv * 8
    bytes_kv_write = bytes_kv_fp8 + bytes_kv_bf16 + bytes_kv_scale
    # position ids: read, int64
    bytes_pos_q = nq * 8
    bytes_pos_kv = nkv * 8

    total_bytes = bytes_q * 2 + bytes_kv_read + bytes_kv_write + bytes_pos_q + bytes_pos_kv
    total_gb = total_bytes / 1e9

    print(f"\n{'='*65}")
    print(f"PCP fused kernel bandwidth test: Q={nq}, KV={nkv}, H={nh}")
    print(f"{'='*65}")
    print(f"Data footprint:")
    print(f"  Q r+w:      {bytes_q/1e6:8.1f} MB (in-place, bf16)")
    print(f"  KV read:    {bytes_kv_read/1e6:8.1f} MB (bf16)")
    print(f"  K cache fp8:{bytes_kv_fp8/1e6:8.1f} MB (write, uint8)")
    print(f"  K cache bf16:{bytes_kv_bf16/1e6:8.1f} MB (write)")
    print(f"  K cache scale:{bytes_kv_scale/1e6:8.1f} MB (write)")
    print(f"  pos ids:    {(bytes_pos_q+bytes_pos_kv)/1e6:8.1f} MB (read, int64)")
    print(f"  ─────────────────────")
    print(f"  Total:      {total_gb:8.3f} GB")

    # ── Warmup ───────────────────────────────────────────────────
    for i in range(warmup):
        q_c = q.clone()
        kc = torch.zeros(n_blocks, block_stride, dtype=torch.uint8, device=device)
        pcp_op(q_c, kv, kc, slot_map, q_pos, kv_pos, cos_sin_cache, EPS, block_size)

    torch.cuda.synchronize()

    # ── Benchmark ────────────────────────────────────────────────
    times = []
    for i in range(iters):
        q_c = q.clone()
        kc = torch.zeros(n_blocks, block_stride, dtype=torch.uint8, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pcp_op(q_c, kv, kc, slot_map, q_pos, kv_pos, cos_sin_cache, EPS, block_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times = sorted(times)
    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    med_ms = times[len(times)//2] * 1000
    bw_avg = total_gb / (avg_ms / 1000)
    bw_min = total_gb / (min_ms / 1000)
    bw_med = total_gb / (med_ms / 1000)

    print(f"\nTiming ({warmup} warmup + {iters} iters):")
    print(f"  min:  {min_ms:.3f} ms  →  {bw_min:.1f} GB/s")
    print(f"  med:  {med_ms:.3f} ms  →  {bw_med:.1f} GB/s")
    print(f"  avg:  {avg_ms:.3f} ms  →  {bw_avg:.1f} GB/s")
    return avg_ms, bw_avg


if __name__ == "__main__":
    bench(nq=2048, nkv=16384, nh=128)
