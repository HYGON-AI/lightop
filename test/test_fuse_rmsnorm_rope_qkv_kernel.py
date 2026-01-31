#!/usr/bin/env python3
"""
测试 rms_rotary_embedding_fuse_with_kv_store：参考 test_fused_rms_rope.py 的基准测试结构。
验证 1) Q 经 RMS+RoPE 与 rms_rotary_embedding_fuse 一致  2) v_buffer 正确写入。
"""

import torch
import itertools
from vllm import _custom_ops as ops
from lightop import  rms_rotary_embedding_fuse_with_kv_store

# =====================================
# 1. 全局配置
# =====================================
torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16  # float16 | float32 | bfloat16
RMS_EPS = 1e-4
ROPE_THETA = 10000.0
PAGE_SIZE = 16

# 定义测试形状: (BATCH, SEQ, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM)
TEST_SHAPES = [
    (2, 8, 4, 4, 128, 128),
    (2, 16, 4, 4, 256, 128),
    (2, 8, 4, 4, 512, 128),
]
TEST_NEOX_STYLES = [True]
TEST_RESIDUALS = [True]

# =====================================
# 2. 辅助工具
# =====================================
def build_cos_sin_cache(max_position, dim, base=10000.0, dtype=torch.float32, device="cuda"):
    half_dim = dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    t = torch.arange(0, max_position, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, sin], dim=-1).contiguous()


def get_tensors_kv_store(batch, seq, h_q, h_k, dim, device, dtype, page_size, generate_residual):
    """生成 rms_rotary_embedding_fuse_with_kv_store 所需张量。q/k/v 为 [num_tokens, num_heads, head_dim]。"""
    num_tokens = batch * seq
    num_blocks = max(1, (num_tokens + page_size - 1) // page_size)

    q = torch.randn(num_tokens, h_q, dim, device=device, dtype=dtype).contiguous()
    k = torch.randn(num_tokens, h_k, dim, device=device, dtype=dtype).contiguous()
    v = torch.randn(num_tokens, h_k, dim, device=device, dtype=dtype).contiguous()

    positions = torch.arange(num_tokens, device=device, dtype=torch.int64).contiguous()
    weight_q = torch.ones(dim, device=device, dtype=dtype).contiguous()
    weight_k = torch.ones(dim, device=device, dtype=dtype).contiguous()

    res_q = torch.randn(num_tokens, h_q, dim, device=device, dtype=dtype).contiguous() if generate_residual else None
    res_k = torch.randn(num_tokens, h_k, dim, device=device, dtype=dtype).contiguous() if generate_residual else None

    k_buffer = torch.zeros(num_blocks, h_k, page_size, dim, device=device, dtype=dtype).contiguous()
    v_buffer = torch.zeros(num_blocks, h_k, page_size, dim, device=device, dtype=dtype).contiguous()
    kv_cache_loc = torch.arange(num_tokens, device=device, dtype=torch.int64).contiguous()

    return q, k, v, positions, weight_q, weight_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, num_tokens


# =====================================
# 3. 基准与融合 Kernel 封装
# =====================================
def baseline_kernel_adapter(q, k, v, positions, w_q, w_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, page_size, cache, is_neox, use_res):
    """基准：Q 用 rms_rotary_embedding_fuse，v_buffer 手动按 slot 写入 v。"""
    if use_res:
        ops.fused_add_rms_norm_opt(q, res_q, w_q, RMS_EPS)
        ops.fused_add_rms_norm_opt(k, res_k, w_k, RMS_EPS)
    else:
        ops.rms_norm_opt(q, q, w_q, RMS_EPS)
        ops.rms_norm_opt(k, k, w_k, RMS_EPS)
    ops.rotary_embedding(positions, q, k, w_q.shape[-1], cache, is_neox)
    num_tokens = q.size(0)
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        v_buffer[blk, :, off, :] = v[t, :, :].clone()


def fused_kernel_adapter(q, k, v, positions, w_q, w_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, page_size, cache, is_neox, use_res):
    """融合：rms_rotary_embedding_fuse_with_kv_store。"""
    rms_rotary_embedding_fuse_with_kv_store(
        positions, q, k, v, cache, page_size,
        k_buffer, v_buffer, kv_cache_loc,
        is_neox=is_neox, weight_q=w_q, weight_k=w_k,
        residual_q=res_q, residual_k=res_k, epsilon=RMS_EPS,
    )


# =====================================
# 4. 精度与比较
# =====================================
def get_precision_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 2e-2, 2e-2


def compare(a, b, name, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    print(f"   > {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch!"
    return max_diff


def compare_v_buffer(v_buffer, v, kv_cache_loc, page_size, num_tokens, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = 0.0
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        slot = v_buffer[blk, :, off, :]
        ref = v[t, :, :]
        d = (slot - ref).abs().max().item()
        max_diff = max(max_diff, d)
        assert torch.allclose(slot, ref, atol=atol, rtol=rtol), f"v_buffer token {t} mismatch!"
    print(f"   > v_buffer: max_diff={max_diff:.6e}")
    return max_diff
def compare_k_buffer(k_buffer, k, kv_cache_loc, page_size, num_tokens, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = 0.0
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        slot = k_buffer[blk, :, off, :]
        ref = k[t, :, :]
        d = (slot - ref).abs().max().item()
        max_diff = max(max_diff, d)
        assert torch.allclose(slot, ref, atol=atol, rtol=rtol), f"k_buffer token {t} mismatch!"
    print(f"   > k_buffer: max_diff={max_diff:.6e}")
    return max_diff

# =====================================
# 5. 单次测试流程
# =====================================
def run_single_case(shape_config, is_neox, use_residual):
    batch, seq, h_q, h_k, dim, rotary_dim = shape_config
    num_tokens = batch * seq

    q_org, k_org, v_org, positions, w_q, w_k, res_q_org, res_k_org, k_buf_org, v_buf_org, kv_cache_loc, num_tokens = get_tensors_kv_store(
        batch, seq, h_q, h_k, dim, DEVICE, DTYPE, PAGE_SIZE, use_residual
    )

    max_pos = max(num_tokens + 64, 256)
    cos_sin_cache = build_cos_sin_cache(max_pos, rotary_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    cos_sin_cache = cos_sin_cache.contiguous()

    # ---------------------------
    # Accuracy Test
    # ---------------------------
    print(f"🧪 Checking: Shape={shape_config}, Neox={is_neox}, Res={use_residual}")

    q_ref = q_org.clone()
    k_ref = k_org.clone()
    v_ref = v_org.clone()
    k_buf_ref = torch.zeros_like(k_buf_org)  # baseline 不写 k_buffer
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q_ref = res_q_org.clone() if use_residual else None
    r_k_ref = res_k_org.clone() if use_residual else None

    q_tgt = q_org.clone()
    k_tgt = k_org.clone()
    v_tgt = v_org.clone()
    k_buf_tgt = torch.zeros_like(k_buf_org)
    v_buf_tgt = torch.zeros_like(v_buf_org)
    r_q_tgt = res_q_org.clone() if use_residual else None
    r_k_tgt = res_k_org.clone() if use_residual else None

    # Run Baseline (Q: rms_rotary_embedding_fuse, v_buffer: manual)
    baseline_kernel_adapter(q_ref, k_ref, v_ref, positions, w_q, w_k, r_q_ref, r_k_ref, k_buf_ref, v_buf_ref, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual)
    # Run Fused
    fused_kernel_adapter(q_tgt, k_tgt, v_tgt, positions, w_q, w_k, r_q_tgt, r_k_tgt, k_buf_tgt, v_buf_tgt, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual)

    status = "Pass"
    max_diff_record = 0.0

    try:

        diff_q = compare(q_tgt, q_ref, "Query", DTYPE)
        max_diff_record = diff_q
        diff_k = compare(k_tgt, k_ref, "Key", DTYPE)
        max_diff_record = max(max_diff_record, diff_k)
        diff_v = compare_v_buffer(v_buf_tgt, v_org, kv_cache_loc, PAGE_SIZE, num_tokens, DTYPE)
        diff_k_buf = compare_k_buffer(k_buf_tgt, k_ref, kv_cache_loc, PAGE_SIZE, num_tokens, DTYPE)
        max_diff_record = max(max_diff_record, diff_v)
        assert q_tgt.isfinite().all(), "Query 存在 NaN/Inf"
        assert v_buf_tgt.isfinite().all(), "v_buffer 存在 NaN/Inf"
        assert k_buf_tgt.isfinite().all(), "k_buffer 存在 NaN/Inf"
        print("   ✅ Accuracy Verified")
    except AssertionError as e:
        status = "Fail"
        print(f"   ❌ Accuracy Failed: {e}")

    return {
        "Batch": batch, "Seq": seq, "H_Q": h_q, "H_K": h_k, "Dim": dim,
        "Neox": is_neox, "Resid": use_residual,
        "Status": status,
        "Max_Diff": max_diff_record,
    }


# =====================================
# 6. 主程序
# =====================================
def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    results = []
    combinations = list(itertools.product(TEST_SHAPES, TEST_NEOX_STYLES, TEST_RESIDUALS))

    print(f"🚀 rms_rotary_embedding_fuse_with_kv_store 测试 | DTYPE: {DTYPE}")
    print("=" * 80)

    for shape, neox, resid in combinations:
        res = run_single_case(shape, neox, resid)
        results.append(res)
        print("-" * 50)

    print("\n" + "=" * 80)
    fail = [r for r in results if r["Status"] == "Fail"]
    if fail:
        print(f"⚠️  共有 {len(fail)} 个 Case 精度未通过")
    else:
        print("✨ 所有 Case 精度验证通过。")
    print("=" * 80)


if __name__ == "__main__":
    main()
