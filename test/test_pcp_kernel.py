#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Test the PCP fused kernel against Triton golden reference.

The Triton kernels are self-contained inline copies from deepseek_v4_attention.py
to avoid pulling in the full vllm dependency chain.
"""

import math
import time
import torch
import triton
import triton.language as tl
import os
import sys

import torch.utils.cpp_extension as cppext

# ── Constants ──────────────────────────────────────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
HALF_ROPE_DIM = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
QUANT_BLOCK = 64
NUM_QUANT_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1  # 8
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
FP8_MAX = 448.0
MAX_POS = 131072
BASE = 10000.0
EPS = 1e-6

# ── Triton Golden Kernels (from deepseek_v4_attention.py) ──────────────────

@triton.jit
def _qnorm_rope_inplace_kernel(
    q_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_token,
    q_stride_head,
    cos_sin_stride_pos,
    num_tokens,
    num_heads,
    eps,
    HEAD_DIM_CONST: tl.constexpr,
    NOPE_DIM_CONST: tl.constexpr,
    HALF_ROPE_CONST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    if token_idx >= num_tokens or head_idx >= num_heads:
        return

    offs = tl.arange(0, HEAD_DIM_CONST)
    row_ptr = q_ptr + token_idx * q_stride_token + head_idx * q_stride_head

    x = tl.load(row_ptr + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM_CONST
    rrms = 1.0 / tl.sqrt(var + eps)
    x_norm = x * rrms

    pos = tl.load(positions_ptr + token_idx)
    cache_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos

    is_rope = offs >= NOPE_DIM_CONST
    rope_local = offs - NOPE_DIM_CONST
    pair_idx = tl.maximum(rope_local >> 1, 0)
    is_even = (rope_local & 1) == 0

    partner = tl.load(row_ptr + (offs ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    partner = partner * rrms
    cos_v = tl.load(cache_base + pair_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE_CONST + pair_idx, mask=is_rope, other=0.0)

    rot = tl.where(
        is_even,
        x_norm * cos_v - partner * sin_v,
        x_norm * cos_v + partner * sin_v,
    )
    out = tl.where(is_rope, rot, x_norm)
    tl.store(row_ptr + offs, out.to(row_ptr.dtype.element_ty))


@triton.jit
def _kv_rope_quant_insert_kernel(
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    kv_stride_token,
    k_cache_stride_block,
    cos_sin_stride_pos,
    num_tokens_insert,
    cache_block_size,
    HEAD_DIM_CONST: tl.constexpr,
    NOPE_DIM_CONST: tl.constexpr,
    HALF_ROPE_CONST: tl.constexpr,
    QUANT_BLOCK_CONST: tl.constexpr,
    NUM_QBLOCKS_CONST: tl.constexpr,
    TOKEN_BYTES_CONST: tl.constexpr,
    SCALE_BYTES_CONST: tl.constexpr,
    FP8_MAX_CONST: tl.constexpr,
    ROPE_DIM_CONST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= num_tokens_insert:
        return

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    offs = tl.arange(0, HEAD_DIM_CONST)
    row_ptr = kv_ptr + token_idx * kv_stride_token
    x = tl.load(row_ptr + offs).to(tl.float32)

    pos = tl.load(positions_ptr + token_idx)
    cache_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos

    is_rope = offs >= NOPE_DIM_CONST
    rope_local = offs - NOPE_DIM_CONST
    pair_idx = tl.maximum(rope_local >> 1, 0)
    is_even = (rope_local & 1) == 0

    partner = tl.load(row_ptr + (offs ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    cos_v = tl.load(cache_base + pair_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE_CONST + pair_idx, mask=is_rope, other=0.0)
    rot = tl.where(
        is_even,
        x * cos_v - partner * sin_v,
        x * cos_v + partner * sin_v,
    )
    x = tl.where(is_rope, rot, x)

    # Match CUDA/reference numerics: fp32 -> bf16 -> fp32 before quantization.
    x = x.to(tl.bfloat16).to(tl.float32)

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size
    block_base = k_cache_ptr + block_idx.to(tl.int64) * k_cache_stride_block
    token_data_ptr = block_base + pos_in_block * TOKEN_BYTES_CONST
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + NOPE_DIM_CONST
    token_scale_ptr = (
        block_base + cache_block_size * TOKEN_BYTES_CONST + pos_in_block * SCALE_BYTES_CONST
    )

    nope_mask = offs < NOPE_DIM_CONST
    scale = tl.full((HEAD_DIM_CONST,), 1.0, dtype=tl.float32)
    for b in tl.static_range(NUM_QBLOCKS_CONST):
        q_begin = b * QUANT_BLOCK_CONST
        q_end = q_begin + QUANT_BLOCK_CONST
        block_mask = (offs >= q_begin) & (offs < q_end)
        block_vals = tl.where(block_mask, x, 0.0)
        block_absmax = tl.maximum(tl.max(tl.abs(block_vals), axis=0), 1e-4)
        exponent = tl.ceil(tl.log2(block_absmax / FP8_MAX_CONST))
        block_scale = tl.exp2(exponent)
        scale = tl.where(block_mask, block_scale, scale)
        encoded = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
        tl.store(token_scale_ptr + b, encoded.to(tl.uint8))
    tl.store(token_scale_ptr + NUM_QBLOCKS_CONST, tl.zeros((), dtype=tl.uint8))

    x_scaled = tl.clamp(x / scale, -FP8_MAX_CONST, FP8_MAX_CONST)
    x_fp8 = x_scaled.to(tl.float8e4nv)
    x_u8 = x_fp8.to(tl.uint8, bitcast=True)
    tl.store(token_fp8_ptr + offs, x_u8, mask=nope_mask)

    rope_offs = tl.arange(0, ROPE_DIM_CONST)
    bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    rope_vals = tl.load(row_ptr + NOPE_DIM_CONST + rope_offs).to(tl.float32)
    rope_partner = tl.load(row_ptr + NOPE_DIM_CONST + (rope_offs ^ 1)).to(tl.float32)
    pair_idx64 = rope_offs >> 1
    rope_even = (rope_offs & 1) == 0
    rope_cos = tl.load(cache_base + pair_idx64)
    rope_sin = tl.load(cache_base + HALF_ROPE_CONST + pair_idx64)
    rope_rot = tl.where(
        rope_even,
        rope_vals * rope_cos - rope_partner * rope_sin,
        rope_vals * rope_cos + rope_partner * rope_sin,
    )
    rope_rot = rope_rot.to(tl.bfloat16)
    tl.store(bf16_ptr + rope_offs, rope_rot)


def qnorm_rope_inplace_triton(q, position_ids, cos_sin_cache, eps):
    """Q RMSNorm (no weight) + GPT-J RoPE, in-place."""
    assert q.ndim == 3 and q.shape[-1] == HEAD_DIM
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return
    _qnorm_rope_inplace_kernel[(num_tokens, q.shape[1])](
        q, position_ids, cos_sin_cache,
        q.stride(0), q.stride(1), cos_sin_cache.stride(0),
        num_tokens, q.shape[1], eps,
        HEAD_DIM_CONST=HEAD_DIM, NOPE_DIM_CONST=NOPE_DIM, HALF_ROPE_CONST=HALF_ROPE_DIM,
        num_warps=8, num_stages=1,
    )


def kv_rope_quant_insert_triton(kv, k_cache, slot_mapping, position_ids,
                                 cos_sin_cache, cache_block_size):
    """KV RoPE + FP8 quant + cache insert."""
    assert kv.ndim == 2 and kv.shape[-1] == HEAD_DIM
    num_tokens_insert = slot_mapping.numel()
    if num_tokens_insert == 0:
        return
    _kv_rope_quant_insert_kernel[(num_tokens_insert,)](
        kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
        kv.stride(0), k_cache.stride(0), cos_sin_cache.stride(0),
        num_tokens_insert, cache_block_size,
        HEAD_DIM_CONST=HEAD_DIM, NOPE_DIM_CONST=NOPE_DIM, HALF_ROPE_CONST=HALF_ROPE_DIM,
        QUANT_BLOCK_CONST=QUANT_BLOCK, NUM_QBLOCKS_CONST=NUM_QUANT_BLOCKS,
        TOKEN_BYTES_CONST=TOKEN_DATA_BYTES, SCALE_BYTES_CONST=SCALE_BYTES_PER_TOKEN,
        FP8_MAX_CONST=FP8_MAX, ROPE_DIM_CONST=ROPE_DIM,
        num_warps=8, num_stages=1,
    )


# ── Build cos_sin_cache ────────────────────────────────────────────────────

def build_cos_sin_cache(max_pos, rope_dim, base):
    half = rope_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1).contiguous()


# ── Build PCP kernel module ────────────────────────────────────────────────

kernel_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "./", "csrc",
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
with open("/tmp/pcp_kernel_combined.cu", "w") as f:
    f.write(kernel_source)
    f.write("\n")
    f.write(wrapper_code)

print("Compiling PCP fused kernel...")
pcp_mod = cppext.load(
    name="lightop_pcp_test",
    sources=["/tmp/pcp_kernel_combined.cu"],
    extra_include_paths=[os.path.dirname(kernel_path)],
    extra_cflags=["-O3"],
    verbose=False,
)
pcp_op = pcp_mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp
print("PCP kernel compiled successfully!")


# ── Test ───────────────────────────────────────────────────────────────────

def run_test(label, nq, nkv, nh, block_size=16):
    print(f"\n{'='*60}")
    print(f"Test: {label}  (Q={nq}, KV={nkv}, heads={nh})")
    print(f"{'='*60}")

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
    k_cache_cuda = torch.zeros(n_blocks, block_stride, dtype=torch.uint8, device=device)
    k_cache_triton = torch.zeros(n_blocks, block_stride, dtype=torch.uint8, device=device)

    slot_mapping = torch.zeros(nkv, dtype=torch.int64, device=device)
    for i in range(nkv):
        blk = i // block_size
        pos = i % block_size
        slot_mapping[i] = blk * block_size + pos

    # Run Triton golden
    torch.cuda.synchronize()
    t0 = time.time()
    q_triton = q.clone()
    qnorm_rope_inplace_triton(q_triton, q_pos, cos_sin_cache, EPS)
    kv_rope_quant_insert_triton(kv, k_cache_triton, slot_mapping, kv_pos, cos_sin_cache, block_size)
    torch.cuda.synchronize()
    t_triton = time.time() - t0

    # Run CUDA PCP fused
    torch.cuda.synchronize()
    t0 = time.time()
    q_cuda = q.clone()
    pcp_op(q_cuda, kv, k_cache_cuda, slot_mapping, q_pos, kv_pos, cos_sin_cache, EPS, block_size)
    torch.cuda.synchronize()
    t_cuda = time.time() - t0

    print(f"  Triton: {t_triton*1000:.2f}ms,  CUDA PCP: {t_cuda*1000:.2f}ms  speedup={t_triton/t_cuda:.2f}x")

    # Verify Q
    q_diff = (q_cuda.float() - q_triton.float()).abs()
    q_ok = q_diff.max().item() < 0.1
    print(f"  Q:  max_diff={q_diff.max().item():.6f}  {'OK' if q_ok else 'FAIL'}")

    # Verify K cache
    fp8_mismatches = 0
    for b in range(n_blocks):
        bc, bt = k_cache_cuda[b], k_cache_triton[b]
        for t in range(min(block_size, nkv - b * block_size)):
            off = t * TOKEN_DATA_BYTES
            fp8_mismatches += (bc[off:off+NOPE_DIM] != bt[off:off+NOPE_DIM]).sum().item()
    fp8_ok = fp8_mismatches == 0

    bf16_mismatches = 0
    for b in range(n_blocks):
        bc, bt = k_cache_cuda[b], k_cache_triton[b]
        for t in range(min(block_size, nkv - b * block_size)):
            off = t * TOKEN_DATA_BYTES + NOPE_DIM
            bf16_mismatches += (bc[off:off+ROPE_DIM*2] != bt[off:off+ROPE_DIM*2]).sum().item()
    bf16_ok = bf16_mismatches == 0

    print(f"  K fp8: mismatches={fp8_mismatches}  {'OK' if fp8_ok else 'FAIL'}")
    print(f"  K bf16: mismatches={bf16_mismatches}  {'OK' if bf16_ok else 'FAIL'}")

    all_ok = q_ok and fp8_ok and bf16_ok
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILURES'}")
    return all_ok


if __name__ == "__main__":
    all_passed = True

    # Small test first
    all_passed &= run_test("small_pcp", nq=8, nkv=32, nh=4)

    if all_passed:
        # Full-size PCP tests
        all_passed &= run_test("pcp_2k_8k", nq=2048, nkv=8192, nh=128)
        all_passed &= run_test("pcp_2k_16k", nq=2048, nkv=16384, nh=128)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
