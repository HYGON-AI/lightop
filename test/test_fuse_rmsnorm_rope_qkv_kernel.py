import os
import statistics
import sys
import time
import torch
import itertools
from lightop.attention import (
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
)

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64
TEST_SHAPES = [(128, 1, 8, 1, 128, 128), (128, 1, 8, 1, 128, 64)]
TEST_NEOX = [True]
TEST_RESIDUALS = [False, True]
TEST_V_FP8 = [True]

# 性能测试：更多 shape，便于评估不同规模
PERF_SHAPES = [
    (128, 1, 8, 1, 128, 128),
    (128, 1, 8, 1, 128, 64),
    (256, 1, 8, 1, 128, 128),
    (512, 1, 8, 1, 128, 128),
    (1024, 1, 8, 1, 128, 128),
]
PERF_WARMUP = 10
PERF_REPEAT = 100

def get_precision_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 1e-5, 1e-2

def compare_tensor(a, b, name, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    print(f"   > {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch!"
    return max_diff

def build_cos_sin_cache(max_pos, dim, base=10000.0, dtype=torch.float32, device="cuda"):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(0, max_pos, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def get_tensors(batch, seq, h_q, h_k, dim, page_size, use_residual, use_fp8):
    num_tokens = batch * seq
    num_blocks = max(1, (num_tokens + page_size - 1) // page_size)
    q = torch.randn(num_tokens, h_q, dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE)
    res_q = torch.randn(num_tokens, h_q, dim, device=DEVICE, dtype=DTYPE) if use_residual else None
    res_k = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE) if use_residual else None
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    w_q = torch.ones(dim, device=DEVICE, dtype=DTYPE)
    w_k = torch.ones(dim, device=DEVICE, dtype=DTYPE)
    if use_fp8:
        k_buf = torch.zeros(8991, h_k, page_size, dim, device=DEVICE, dtype=torch.float8_e4m3fn)
        v_buf = torch.zeros(8991, h_k, dim, page_size, device=DEVICE, dtype=torch.float8_e4m3fn)
    else:
        k_buf = torch.zeros(num_blocks, h_k, page_size, dim, device=DEVICE, dtype=DTYPE)
        v_buf = torch.zeros(num_blocks, h_k, dim, page_size, device=DEVICE, dtype=DTYPE)
    kv_loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    return q, k, v, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, num_tokens


def get_tensors_2d(batch, seq, h_q, h_k, dim, page_size, use_residual, use_fp8):
    """q,k,v 为 2D: [B*S, num_head*head_dim]，k_buffer/v_buffer 与 get_tensors 一致。"""
    q_3d, k_3d, v_3d, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, num_tokens = get_tensors(
        batch, seq, h_q, h_k, dim, page_size, use_residual, use_fp8
    )
    # [B*S, num_heads, head_dim] -> [B*S, num_heads*head_dim]
    q = q_3d.reshape(num_tokens, h_q * dim)
    k = k_3d.reshape(num_tokens, h_k * dim)
    v = v_3d.reshape(num_tokens, h_k * dim)
    res_q_2d = res_q.reshape(num_tokens, h_q * dim) if use_residual else None
    res_k_2d = res_k.reshape(num_tokens, h_k * dim) if use_residual else None
    return q, k, v, positions, w_q, w_k, res_q_2d, res_k_2d, k_buf, v_buf, kv_loc, num_tokens


def dequant_fp8(x):
    return x.view(torch.float8_e4m3fn).float() if x.dtype == torch.uint8 else x.float()

def compare_fp8_tensor(tensor_fp8, ref, name, atol=0.25, rtol=0.1):
    """校验 fp8 张量：反量化后与 ref 比较，使用 fp8 放宽容差。"""
    # ref_f = ref.float() if ref.dtype != torch.float32 else ref
    max_diff = (tensor_fp8 - ref).abs().max().item()
    assert torch.allclose(tensor_fp8, ref, atol=1e-5, rtol=1e-2), f"{name} fp8 mismatch! max_diff={max_diff:.6e}"
    print(f"   > {name}_fp8: max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})")
    return max_diff


def run_one(shape, is_neox, use_residual, use_fp8):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    # 基准：fake
    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref, v_buf_ref, k_buf_ref, v_buf_ref = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org), torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_ref = res_q.clone() if use_residual else None
    r_k_ref = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = rms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_ref, k_ref, v_ref, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_ref, residual_k=r_k_ref,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
    )

    # 被测：real
    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
    )
    atol, rtol = (0.25, 0.1) if use_fp8 else (1e-5, 1e-2)
    def cmp(a, b, name):
        if use_fp8:
            a = dequant_fp8(a)
            b = dequant_fp8(b) if b.dtype in (torch.float8_e4m3fn, torch.uint8) else b.float()
        compare_fp8_tensor(a, b, name)
    cmp(q_out, q_ref_out, "q")
    cmp(k_out, k_ref_out, "k")
    cmp(v_out, v_ref_out, "v")
    cmp(k_buf_tgt, k_buf_ref, "k_buf")
    cmp(v_buf_tgt, v_buf_ref, "v_buf")


def run_one_2d(shape, is_neox, use_residual, use_fp8):
    """2D 测试：q,k,v shape 为 [B*S, num_head*head_dim]，k_buffer/v_buffer 与 3D 一致。"""
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors_2d(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    # 基准：fake（2D 输入）
    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref = torch.zeros_like(k_buf_org)
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q_ref = res_q.clone() if use_residual else None
    r_k_ref = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = rms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_ref, k_ref, v_ref, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_ref, residual_k=r_k_ref,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
    )

    # 被测：real（2D 输入）
    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    print(f"k_buf_org: {k_buf_org.shape}, v_buf_org: {v_buf_org.shape}")
    print(f"k_buf_ref: {k_buf_ref.shape}, v_buf_ref: {v_buf_ref.shape}")
    print(f"q_org: {q_org.shape}, k_org: {k_org.shape}, v_org: {v_org.shape}")
    print(f"q_ref: {q_ref.shape}, k_ref: {k_ref.shape}, v_ref: {v_ref.shape}")
    print(f"q_tgt: {q_tgt.shape}, k_tgt: {k_tgt.shape}, v_tgt: {v_tgt.shape}")
    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
    )

    atol, rtol = (0.25, 0.1) if use_fp8 else (1e-5, 1e-2)

    def cmp(a, b, name):
        if use_fp8:
            a = dequant_fp8(a)
            b = dequant_fp8(b) if b.dtype in (torch.float8_e4m3fn, torch.uint8) else b.float()
        compare_fp8_tensor(a, b, name, atol, rtol)
        # else:
        #     compare_tensor(a, b, name, DTYPE)
        

    cmp(q_out, q_ref_out, "q_2d")
    cmp(k_out, k_ref_out, "k_2d")
    cmp(v_out, v_ref_out, "v_2d")
    cmp(k_buf_tgt, k_buf_ref, "k_buf_2d")
    cmp(v_buf_tgt, v_buf_ref, "v_buf_2d")


def run_benchmark(shape, is_neox, use_residual, use_fp8, warmup=PERF_WARMUP, repeat=PERF_REPEAT):
    """对 rms_rotary_embedding_fuse_with_kv_store 做性能测试，返回 (mean_ms, std_ms, num_tokens)。"""
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q, k, v, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, _ = get_tensors(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    def run_once():
        q_t, k_t, v_t = q.clone(), k.clone(), v.clone()
        k_buf_t = torch.zeros_like(k_buf)
        v_buf_t = torch.zeros_like(v_buf)
        r_q = res_q.clone() if use_residual else None
        r_k = res_k.clone() if use_residual else None
        rms_rotary_embedding_fuse_with_kv_store(
            positions, q_t, k_t, v_t, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
            k_buffer=k_buf_t, v_buffer=v_buf_t, kv_cache_loc=kv_loc, is_neox=is_neox,
            weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q, residual_k=r_k,
            k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
        )

    # warmup
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    # timed
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_ms, std_ms, num_tokens


def run_perf_tests():
    """运行性能测试并打印表格。"""
    if not torch.cuda.is_available():
        print("SKIP perf: no CUDA")
        return
    print("fuse_rmsnorm_rope_qkv kernel performance (real kernel)")
    print("  warmup=%d, repeat=%d" % (PERF_WARMUP, PERF_REPEAT))
    print("-" * 90)
    print(f"{'shape':<45} {'mean(ms)':>10} {'std(ms)':>10} {'tokens/s':>12}")
    print("-" * 90)
    for shape in PERF_SHAPES:
        batch, seq, h_q, h_k, dim, rot_dim = shape
        mean_ms, std_ms, num_tokens = run_benchmark(
            shape, is_neox=True, use_residual=False, use_fp8=True,
            warmup=PERF_WARMUP, repeat=PERF_REPEAT,
        )
        tokens_per_sec = num_tokens / (mean_ms * 1e-3) if mean_ms > 0 else 0
        shape_str = str(shape)
        print(f"{shape_str:<45} {mean_ms:>10.3f} {std_ms:>10.3f} {tokens_per_sec:>12.0f}")
    print("-" * 90)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    for neox, resid, v_fp8 in itertools.product(TEST_NEOX, TEST_RESIDUALS, TEST_V_FP8):
        for shape in TEST_SHAPES:
            print(f"Shape={shape}, Neox={neox}, Res={resid}, V_fp8={v_fp8}")
            run_one(shape, neox, resid, v_fp8)
            print("  OK (3D)")
            print(f"  2D [B*S, num_head*head_dim]: ", end="")
            run_one_2d(shape, neox, resid, v_fp8)
            print("  OK (2D)")
    print("All passed vs rms_rotary_embedding_fuse_with_kv_store_fake.")


if __name__ == "__main__":
    run_perf_tests()
    main()