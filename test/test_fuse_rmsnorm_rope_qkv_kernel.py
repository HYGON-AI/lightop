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
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64

# 定义测试形状: (BATCH, SEQ, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, ROTARY_DIM) 或带 HEAD_DIM_V 的 7 元组
TEST_SHAPES = [(1, d1, 8, 1, 128, 64) for d1 in (1 << i for i in range(1, 10))]
# q/k/v 为 2 维时的形状: q=[b*s, h_q*head_dim], k=[b*s, h_k*head_dim], v=[b*s, h_k*head_dim_v], k/v_buffer 仍 4 维
TEST_SHAPES_2D = [
    (2, 16, 4, 4, 128, 64),       # head_dim_v 默认 = head_dim
    (2, 16, 4, 4, 128, 64, 128),  # 显式 head_dim_v
]
TEST_NEOX_STYLES = [True]
TEST_RESIDUALS = [False]
# v_buffer 为 fp8 (uint8) 的精度验证
TEST_V_BUFFER_FP8 = [False, True]
# 是否测试 2D q/k/v（flattened last two dims）
TEST_USE_2D_QKV = [False, True]

# 性能测试配置
WARMUP_ITERS = 10
BENCH_ITERS = 100

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


def get_tensors_kv_store(batch, seq, h_q, h_k, dim, device, dtype, page_size, generate_residual, use_v_buffer_fp8=False, use_2d_qkv=False, head_dim_v=None):
    """生成 rms_rotary_embedding_fuse_with_kv_store 所需张量。
    3D: q/k/v 为 [num_tokens, num_heads, head_dim]；
    2D: q=[num_tokens, h_q*head_dim], k=[num_tokens, h_k*head_dim], v=[num_tokens, h_k*head_dim_v]。
    k_buffer shape=[num_blocks, kv_heads, page_size, head_dim]；v_buffer shape=[num_blocks, kv_heads, head_dim_v, page_size]。
    """
    num_tokens = batch * seq
    num_blocks = max(1, (num_tokens + page_size - 1) // page_size)
    dim_v = head_dim_v if head_dim_v is not None else dim

    if use_2d_qkv:
        q = torch.randn(num_tokens, h_q * dim, device=device, dtype=dtype).contiguous()
        k = torch.randn(num_tokens, h_k * dim, device=device, dtype=dtype).contiguous()
        v = torch.randn(num_tokens, h_k * dim_v, device=device, dtype=dtype).contiguous()
        res_q = torch.randn(num_tokens, h_q * dim, device=device, dtype=dtype).contiguous() if generate_residual else None
        res_k = torch.randn(num_tokens, h_k * dim, device=device, dtype=dtype).contiguous() if generate_residual else None
    else:
        q = torch.randn(num_tokens, h_q, dim, device=device, dtype=dtype)
        k = torch.randn(num_tokens, h_k, dim, device=device, dtype=dtype)
        v = torch.randn(num_tokens, h_k, dim_v, device=device, dtype=dtype)
        res_q = torch.randn(num_tokens, h_q, dim, device=device, dtype=dtype) if generate_residual else None
        res_k = torch.randn(num_tokens, h_k, dim, device=device, dtype=dtype)if generate_residual else None

    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    weight_q = torch.ones(dim, device=device, dtype=dtype)
    weight_k = torch.ones(dim, device=device, dtype=dtype)

    if use_v_buffer_fp8:
        k_buffer = torch.zeros(num_blocks, h_k, page_size, dim, device=device, dtype=torch.float8_e4m3fn)
        v_buffer = torch.zeros(num_blocks, h_k, dim_v, page_size, device=device, dtype=torch.float8_e4m3fn)
    else:
        k_buffer = torch.zeros(num_blocks, h_k, page_size, dim, device=device, dtype=dtype)
        v_buffer = torch.zeros(num_blocks, h_k, dim_v, page_size, device=device, dtype=dtype)
    kv_cache_loc = torch.arange(num_tokens, device=device, dtype=torch.int64)

    return q, k, v, positions, weight_q, weight_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, num_tokens


# =====================================
# 3. 基准与融合 Kernel 封装
# =====================================
def _quantize_to_fp8_e4m3(x):
    """将张量量化为 fp8 e4m3 (uint8 存储)，支持任意 shape。"""
    x_f = x.float().clamp(-448.0, 448.0)
    x_fp8 = x_f.to(torch.float8_e4m3fn)
    return x_fp8.view(torch.uint8)


def _quantize_v_to_fp8_e4m3(v_slot):
    """将 v 的一个 slot 量化为 fp8 e4m3 (uint8 存储)。"""
    return _quantize_to_fp8_e4m3(v_slot)

def baseline_kernel_adapter(q, k, v, positions, w_q, w_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, page_size, cache, is_neox, use_res, use_v_buffer_fp8=False):
    """基准：Q/K 用 rms_rotary_embedding_fuse，k_buffer/v_buffer 手动按 slot 写入（fp8 时先量化）。"""
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
        if use_v_buffer_fp8:
            k_buffer[blk, :, off, :] = (k[t, :, :].to(torch.float8_e4m3fn)).view(torch.uint8)
            v_buffer[blk, :, :, off] = (v[t, :, :].to(torch.float8_e4m3fn)).view(torch.uint8)
        else:
            k_buffer[blk, :, off, :] = k[t, :, :].clone()
            v_buffer[blk, :, :, off] = v[t, :, :].clone()


def fused_kernel_adapter(q, k, v, positions, w_q, w_k, res_q, res_k, k_buffer, v_buffer, kv_cache_loc, page_size, cache, is_neox, use_res, use_v_buffer_fp8=False):
    """融合：rms_rotary_embedding_fuse_with_kv_store。fp8 时返回 (q_out, k_out, v_out) 为 fp8。"""
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q, k, v, cache, k_buffer.size(-1), page_size,
        k_buffer, v_buffer, kv_cache_loc,
        is_neox=is_neox, weight_q=w_q, weight_k=w_k,
        output_dtype=DTYPE,
        residual_q=res_q, residual_k=res_k,
        k_scale=1.0 if use_v_buffer_fp8 else None,
        v_scale=1.0 if use_v_buffer_fp8 else None,
        epsilon=RMS_EPS,
    )
    return q_out, k_out, v_out


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
    """v_buffer shape [num_blocks, kv_heads, head_dim_v, page_size]，slot = v_buffer[blk, :, :, off]。"""
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = 0.0
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        slot = v_buffer[blk, :, :, off]
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


def dequantize_fp8_e4m3_to_float(x_fp8):
    """将 fp8 e4m3 (uint8 或 float8_e4m3fn 存储) 反量化为 float。"""
    if x_fp8.dtype == torch.uint8:
        return x_fp8.view(torch.float8_e4m3fn).float()
    return x_fp8.float()


def compare_fp8_tensor(tensor_fp8, ref, name, atol=0.25, rtol=0.1):
    """校验 fp8 张量：反量化后与 ref 比较，使用 fp8 放宽容差。"""
    dec = dequantize_fp8_e4m3_to_float(tensor_fp8)
    ref_f = ref.to(torch.float8_e4m3fn).float() if ref.dtype != torch.float32 else ref
    # ref_f = ref.float() if ref.dtype != torch.float32 else ref
    max_diff = (dec - ref_f).abs().max().item()
    assert torch.allclose(dec, ref_f, atol=atol, rtol=rtol), f"{name} fp8 mismatch! max_diff={max_diff:.6e}"
    print(f"   > {name}_fp8: max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})")
    return max_diff


def compare_k_buffer_fp8(k_buffer_uint8, k_ref, kv_cache_loc, page_size, num_tokens, atol=0.25, rtol=0.1):
    """校验 fp8 k_buffer：反量化后与 k_ref 逐 slot 比较。"""
    k_dequant = dequantize_fp8_e4m3_to_float(k_buffer_uint8)
    max_diff = 0.0
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        slot = k_dequant[blk, :, off, :]
        ref = k_ref[t, :, :].to(torch.float8_e4m3fn).float()
        d = (slot - ref).abs().max().item()
        max_diff = max(max_diff, d)
        assert torch.allclose(slot, ref, atol=atol, rtol=rtol), f"k_buffer_fp8 token {t} mismatch! max_diff={d:.6e}"
    print(f"   > k_buffer_fp8: max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})")
    return max_diff


def compare_v_buffer_fp8(v_buffer_uint8, v_ref, kv_cache_loc, page_size, num_tokens, atol=0.25, rtol=0.1):
    """校验 fp8 v_buffer：反量化后与 v_ref 逐 slot 比较。v_buffer shape [num_blocks, kv_heads, head_dim_v, page_size]。"""
    v_dequant = dequantize_fp8_e4m3_to_float(v_buffer_uint8)
    max_diff = 0.0
    for t in range(num_tokens):
        blk = int(kv_cache_loc[t].item()) // page_size
        off = int(kv_cache_loc[t].item()) % page_size
        slot = v_dequant[blk, :, :, off]
        ref = v_ref[t, :, :].to(torch.float8_e4m3fn).float()
        d = (slot - ref).abs().max().item()
        max_diff = max(max_diff, d)
        assert torch.allclose(slot, ref, atol=atol, rtol=rtol), f"v_buffer_fp8 token {t} mismatch! max_diff={d:.6e}"
    print(f"   > v_buffer_fp8: max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})")
    return max_diff


# =====================================
# 4.1 性能测试 (GPU Event 计时)
# =====================================
def benchmark_func(func, warmup=10, rep=100):
    """预热 + 多次执行，返回平均耗时 (us)。"""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(rep):
        func()
    end_event.record()
    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms * 1000) / rep
    return avg_us


# =====================================
# 5. 单次测试流程
# =====================================
def run_single_case(shape_config, is_neox, use_residual, use_v_buffer_fp8=False, use_2d_qkv=False):
    if len(shape_config) == 7:
        batch, seq, h_q, h_k, dim, rotary_dim, head_dim_v = shape_config
    else:
        batch, seq, h_q, h_k, dim, rotary_dim = shape_config
        head_dim_v = dim
    num_tokens = batch * seq

    q_org, k_org, v_org, positions, w_q, w_k, res_q_org, res_k_org, k_buf_org, v_buf_org, kv_cache_loc, num_tokens = get_tensors_kv_store(
        batch, seq, h_q, h_k, dim, DEVICE, DTYPE, PAGE_SIZE, use_residual,
        use_v_buffer_fp8=use_v_buffer_fp8, use_2d_qkv=use_2d_qkv,
        head_dim_v=head_dim_v if use_2d_qkv else None,
    )

    # 2D q/k/v 时 reshape 为 3D，供 baseline/fused（均按 3D）使用；k_buffer/v_buffer 保持 4 维
    if use_2d_qkv:
        q_org = q_org.view(num_tokens, h_q, dim)
        k_org = k_org.view(num_tokens, h_k, dim)
        v_org = v_org.view(num_tokens, h_k, head_dim_v)
        if res_q_org is not None:
            res_q_org = res_q_org.view(num_tokens, h_q, dim)
        if res_k_org is not None:
            res_k_org = res_k_org.view(num_tokens, h_k, dim)

    max_pos = num_tokens
    cos_sin_cache = build_cos_sin_cache(max_pos, rotary_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    cos_sin_cache = cos_sin_cache.contiguous()

    # ---------------------------
    # Accuracy Test
    # ---------------------------
    print(f"🧪 Checking: Shape={shape_config}, Neox={is_neox}, Res={use_residual}, V_buffer_fp8={use_v_buffer_fp8}, Use_2D_QKV={use_2d_qkv}")

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

    # Run Baseline (Q/K: rms_rotary_embedding_fuse, k_buffer/v_buffer: manual or fp8 quant)
    baseline_kernel_adapter(q_ref, k_ref, v_ref, positions, w_q, w_k, r_q_ref, r_k_ref, k_buf_ref, v_buf_ref, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual, use_v_buffer_fp8=use_v_buffer_fp8)
    # Run Fused（fp8 时返回 q_out, k_out, v_out 为 fp8）
    q_out, k_out, v_out = fused_kernel_adapter(q_tgt, k_tgt, v_tgt, positions, w_q, w_k, r_q_tgt, r_k_tgt, k_buf_tgt, v_buf_tgt, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual, use_v_buffer_fp8=use_v_buffer_fp8)

    status = "Pass"
    max_diff_record = 0.0

    try:
        if use_v_buffer_fp8:
            # q 输出 fp8：反量化后与 q_ref (rms+rope 后) 比较
            diff_q = compare_fp8_tensor(q_out, q_ref, "q")
            max_diff_record = max(max_diff_record, diff_q)
            # k 输出 fp8：反量化后与 k_ref 比较
            diff_k = compare_fp8_tensor(k_out, k_ref, "k")
            max_diff_record = max(max_diff_record, diff_k)
            # k_buffer fp8：反量化后与 k_ref 逐 slot 比较
            diff_k_buf = compare_k_buffer_fp8(k_buf_tgt, k_ref, kv_cache_loc, PAGE_SIZE, num_tokens)
            max_diff_record = max(max_diff_record, diff_k_buf)
            # v_buffer fp8：反量化后与 v_org 逐 slot 比较
            diff_v_buf = compare_v_buffer_fp8(v_buf_tgt, v_org, kv_cache_loc, PAGE_SIZE, num_tokens)
            max_diff_record = max(max_diff_record, diff_v_buf)
            # v 输出 fp8：反量化后与 v_org 比较
            diff_v = compare_fp8_tensor(v_out, v_org, "v")
            max_diff_record = max(max_diff_record, diff_v)
            # 反量化后无 NaN/Inf
            for name, t in [("v_buffer", v_buf_tgt), ("k_buffer", k_buf_tgt)]:
                dec = dequantize_fp8_e4m3_to_float(t)
                assert dec.isfinite().all(), f"{name}_fp8 反量化后存在 NaN/Inf"
        else:
            diff_q = compare(q_out, q_ref, "q", DTYPE)
            assert q_out.isfinite().all(), "q_buffer 存在 NaN/Inf"
            max_diff_record = max(max_diff_record, diff_q)
            diff_k = compare(k_out, k_ref, "k", DTYPE)
            assert k_out.isfinite().all(), "k_buffer 存在 NaN/Inf"
            max_diff_record = max(max_diff_record, diff_k)
            diff_v = compare(v_out, v_org, "v", DTYPE)
            assert v_out.isfinite().all(), "v_buffer 存在 NaN/Inf"
            max_diff_record = max(max_diff_record, diff_v)
            diff_k = compare_k_buffer(k_buf_tgt, k_ref, kv_cache_loc, PAGE_SIZE, num_tokens, DTYPE)
            assert k_buf_tgt.isfinite().all(), "v_buffer 存在 NaN/Inf"
            max_diff_record = max(max_diff_record, diff_k)
            diff_v = compare_v_buffer(v_buf_tgt, v_org, kv_cache_loc, PAGE_SIZE, num_tokens, DTYPE)
            assert v_buf_tgt.isfinite().all(), "v_buffer 存在 NaN/Inf"
            max_diff_record = max(max_diff_record, diff_v)
        print("   ✅ Accuracy Verified")
    except AssertionError as e:
        status = "Fail"
        print(f"   ❌ Accuracy Failed: {e}")

    # ---------------------------
    # 性能测试 (Benchmark)，含 fp8 路径
    # ---------------------------
    base_us = fuse_us = speedup = None
    try:
        def baseline_fn():
            q_b, k_b, v_b = q_org.clone(), k_org.clone(), v_org.clone()
            k_buf_b = torch.zeros_like(k_buf_org)
            v_buf_b = torch.zeros_like(v_buf_org)
            r_q = res_q_org.clone() if use_residual else None
            r_k = res_k_org.clone() if use_residual else None
            baseline_kernel_adapter(q_b, k_b, v_b, positions, w_q, w_k, r_q, r_k, k_buf_b, v_buf_b, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual, use_v_buffer_fp8=use_v_buffer_fp8)

        def fused_fn():
            q_f, k_f, v_f = q_org.clone(), k_org.clone(), v_org.clone()
            k_buf_f = torch.zeros_like(k_buf_org)
            v_buf_f = torch.zeros_like(v_buf_org)
            r_q = res_q_org.clone() if use_residual else None
            r_k = res_k_org.clone() if use_residual else None
            fused_kernel_adapter(q_f, k_f, v_f, positions, w_q, w_k, r_q, r_k, k_buf_f, v_buf_f, kv_cache_loc, PAGE_SIZE, cos_sin_cache, is_neox, use_residual, use_v_buffer_fp8=use_v_buffer_fp8)

        base_us = benchmark_func(baseline_fn, warmup=WARMUP_ITERS, rep=BENCH_ITERS)
        fuse_us = benchmark_func(fused_fn, warmup=WARMUP_ITERS, rep=BENCH_ITERS)
        speedup = base_us / fuse_us if fuse_us > 0 else 0.0
        fp8_tag = " [fp8]" if use_v_buffer_fp8 else ""
        print(f"   ⏱️  Base={base_us:.2f}us, Fuse={fuse_us:.2f}us, Speedup={speedup:.3f}x{fp8_tag}")
    except Exception as e:
        print(f"   ❌ Benchmark Error: {e}")

    return {
        "Batch": batch, "Seq": seq, "H_Q": h_q, "H_K": h_k, "Dim": dim, "RotDim": rotary_dim,
        "Neox": is_neox, "Resid": use_residual, "V_buffer_fp8": use_v_buffer_fp8, "Use_2D_QKV": use_2d_qkv,
        "Status": status,
        "Max_Diff": max_diff_record,
        "Base(us)": round(base_us, 2) if base_us is not None else None,
        "Fuse(us)": round(fuse_us, 2) if fuse_us is not None else None,
        "Speedup": round(speedup, 3) if speedup is not None else None,
    }


# =====================================
# 6. 主程序
# =====================================
def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    results = []
    # 3D 用 TEST_SHAPES；2D q/k/v 用 TEST_SHAPES_2D
    combinations = list(itertools.product(TEST_NEOX_STYLES, TEST_RESIDUALS, TEST_V_BUFFER_FP8, TEST_USE_2D_QKV))

    print(f"🚀 rms_rotary_embedding_fuse_with_kv_store 测试 | DTYPE: {DTYPE} | Warmup: {WARMUP_ITERS} | Reps: {BENCH_ITERS}")
    print("=" * 80)

    for neox, resid, v_fp8, use_2d in combinations:
        shapes = TEST_SHAPES_2D if use_2d else TEST_SHAPES
        for shape in shapes:
            res = run_single_case(shape, neox, resid, use_v_buffer_fp8=v_fp8, use_2d_qkv=use_2d)
            results.append(res)
            print("-" * 50)

    print("\n" + "=" * 80)
    fail = [r for r in results if r["Status"] == "Fail"]
    if fail:
        print(f"⚠️  共有 {len(fail)} 个 Case 精度未通过")
    else:
        print("✨ 所有 Case 精度验证通过。")

    speedups = [r["Speedup"] for r in results if r.get("Speedup") is not None]
    if speedups:
        print(f"📊 平均加速比: {sum(speedups) / len(speedups):.3f}x")
        print("\n性能汇总:")
        for r in results:
            b, s, hq, hk, d = r["Batch"], r["Seq"], r["H_Q"], r["H_K"], r["Dim"]
            rd = r.get("RotDim", d)
            base_us = r.get("Base(us)")
            fuse_us = r.get("Fuse(us)")
            sp = r.get("Speedup")
            fp8_tag = " [fp8]" if r.get("V_buffer_fp8") else ""
            qkv2d_tag = " [2D QKV]" if r.get("Use_2D_QKV") else ""
            if base_us is not None and fuse_us is not None and sp is not None:
                print(f"   Shape({b},{s},{hq},{hk},{d},{rd}){fp8_tag}{qkv2d_tag}: Base={base_us}us, Fuse={fuse_us}us, Speedup={sp}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
