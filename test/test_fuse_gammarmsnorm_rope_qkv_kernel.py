import os
import statistics
import sys
import time
import torch
import itertools
from lightop import (
    gammarms_rotary_embedding_fuse_with_kv_store,
)

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64
TEST_SHAPES = [(128, 1, 8, 1, 256, 128), (128, 1, 8, 1, 256, 64)]
TEST_NEOX = [True]
TEST_RESIDUALS = [False, True]
TEST_V_FP8 = [True]
TEST_INTERLEAVED = [False, True]

# 性能测试：更多 shape，便于评估不同规模
PERF_SHAPES = [
    (128, 1, 8, 1, 256, 128),
    (128, 1, 8, 1, 256, 64),
    (128, 1, 8, 1, 256, 128),
    (512, 1, 8, 1, 256, 128),
    (1024, 1, 8, 1, 256, 128),
]
PERF_WARMUP = 10
PERF_REPEAT = 100


def get_mrope_params(rot_dim, mode="balanced"):
    half = rot_dim // 2
    if mode == "balanced":
        m_t = half // 3
        m_h = half // 3
        m_w = half - m_t - m_h
    elif mode == "time_heavy":
        m_t = half // 2
        m_h = half // 4
        m_w = half - m_t - m_h
    elif mode == "rope":
        m_t = half
        m_h = 0
        m_w = 0
    else:
        raise ValueError
    return m_t, m_h, m_w


def gammarms_rotary_embedding_fuse_with_kv_store_fake(
    positions, query, key, v, cos_sin_cache, head_dim, page_size,
    k_buffer, v_buffer, kv_cache_loc, is_neox,
    weight_q, weight_k, output_dtype, residual_q, residual_k,
    epsilon, m_t, m_h, m_w, is_interleaved
):
    num_tokens = query.shape[0]
    rot_dim = cos_sin_cache.shape[1]
    rot_half = rot_dim // 2

    assert head_dim in (128, 256), f"fake only supports head_dim=128/256, got {head_dim}"
    assert rot_dim in (64, 128), f"fake only supports rot_dim=64/128, got {rot_dim}"
    assert positions.shape[0] == 3, f"positions must be [3, num_tokens], got {positions.shape}"

    # 输入既支持 3D [N, H, D]，也支持 2D [N, H*D]
    if query.dim() == 2:
        q_in = query.view(num_tokens, -1, head_dim)
        k_in = key.view(num_tokens, -1, head_dim)
        v_in = v.view(num_tokens, -1, head_dim)
        res_q_in = residual_q.view(num_tokens, -1, head_dim) if residual_q is not None else None
        res_k_in = residual_k.view(num_tokens, -1, head_dim) if residual_k is not None else None
        return_2d = True
    else:
        q_in = query
        k_in = key
        v_in = v
        res_q_in = residual_q
        res_k_in = residual_k
        return_2d = False

    # -------------------------------
    # 1. GammaRMSNorm + residual
    # -------------------------------
    def apply_norm(x, gamma, res):
        if res is not None:
            x = x + res
        x_f = x.float()
        inv_rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + epsilon)
        return (x_f * inv_rms * (gamma.float() + 1.0)).to(x.dtype)

    q = apply_norm(q_in, weight_q, res_q_in)
    k = apply_norm(k_in, weight_k, res_k_in)
    v_ref = v_in.to(q_in.dtype)

    # -------------------------------
    # 2. 构建 M-RoPE cos/sin
    # 这里按“最小修改”保留一个 head_dim 长度的缓存，
    # 但只有前 rot_half 的 cos 和 [head_dim//2 : head_dim//2+rot_half) 的 sin 有效
    # -------------------------------
    t_pos = positions[0]
    h_pos = positions[1]
    w_pos = positions[2]

    rope_half_layout = head_dim // 2
    my_s_cos_sin = torch.zeros(
        (num_tokens, head_dim),
        device=query.device,
        dtype=query.dtype
    )

    for i in range(rot_half):
        if is_interleaved:
            use_h = (i % 3 == 1) and (i <= 3 * m_h)
            use_w = (i % 3 == 2) and (i <= 3 * m_w)
            use_t = not (use_h or use_w)
        else:
            use_t = i < m_t
            use_h = (i >= m_t) and (i < (m_t + m_h))
            use_w = (i >= (m_t + m_h)) and (i < rot_half)

        # 对齐 Triton：mask + sum
        cos_val = torch.zeros_like(t_pos, dtype=query.dtype)
        sin_val = torch.zeros_like(t_pos, dtype=query.dtype)

        if use_t:
            cos_val = cos_val + cos_sin_cache[t_pos, i]
            sin_val = sin_val + cos_sin_cache[t_pos, i + rot_half]
        if use_h:
            cos_val = cos_val + cos_sin_cache[h_pos, i]
            sin_val = sin_val + cos_sin_cache[h_pos, i + rot_half]
        if use_w:
            cos_val = cos_val + cos_sin_cache[w_pos, i]
            sin_val = sin_val + cos_sin_cache[w_pos, i + rot_half]

        my_s_cos_sin[:, i] = cos_val
        my_s_cos_sin[:, rope_half_layout + i] = sin_val

    # -------------------------------
    # 3. RoPE
    # 只对前 rot_dim 维做 rope，其余维度保持不变
    # -------------------------------
    q_rope = q.clone()
    k_rope = k.clone()

    if is_neox:
        cos = my_s_cos_sin[:, :rot_half].unsqueeze(1)   # [N,1,rot_half]
        sin = my_s_cos_sin[:, rope_half_layout:rope_half_layout + rot_half].unsqueeze(1)

        q_low = q[:, :, :rot_half]
        q_high = q[:, :, rot_half:2 * rot_half]
        k_low = k[:, :, :rot_half]
        k_high = k[:, :, rot_half:2 * rot_half]

        q_rope[:, :, :rot_half] = q_low * cos - q_high * sin
        q_rope[:, :, rot_half:2 * rot_half] = q_low * sin + q_high * cos

        k_rope[:, :, :rot_half] = k_low * cos - k_high * sin
        k_rope[:, :, rot_half:2 * rot_half] = k_low * sin + k_high * cos

    else:
        cos = my_s_cos_sin[:, :rot_half].unsqueeze(1)
        sin = my_s_cos_sin[:, rope_half_layout:rope_half_layout + rot_half].unsqueeze(1)

        qe = q[:, :, 0:2 * rot_half:2]
        qo = q[:, :, 1:2 * rot_half:2]
        ke = k[:, :, 0:2 * rot_half:2]
        ko = k[:, :, 1:2 * rot_half:2]

        q_rope[:, :, 0:2 * rot_half:2] = qe * cos - qo * sin
        q_rope[:, :, 1:2 * rot_half:2] = qo * cos + qe * sin

        k_rope[:, :, 0:2 * rot_half:2] = ke * cos - ko * sin
        k_rope[:, :, 1:2 * rot_half:2] = ko * cos + ke * sin

    # -------------------------------
    # 4. FP8 模拟
    # -------------------------------
    def to_fp8(x):
        return x.to(torch.float8_e4m3fn)

    q_fp8 = to_fp8(q_rope)
    k_fp8 = to_fp8(k_rope)
    v_fp8 = to_fp8(v_ref)

    # -------------------------------
    # 5. KV cache 写入
    # K: [block, head, token_in_block, dim]
    # V: [block, head, dim, token_in_block]
    # -------------------------------
    loc_id = kv_cache_loc // page_size
    loc_offset = kv_cache_loc % page_size

    for i in range(num_tokens):
        k_buffer[loc_id[i], :, loc_offset[i], :] = k_fp8[i]
        v_buffer[loc_id[i], :, :, loc_offset[i]] = v_fp8[i]

    # -------------------------------
    # 6. 输出
    # real wrapper 对 2D 输入时 q_out/k_out/v_out 仍可能返回 3D，
    # 这里保留原来的 fake 行为：输入 2D 就返回 2D
    # -------------------------------
    q_ret = q_fp8.to(query.dtype)
    k_ret = k_fp8.to(query.dtype)
    v_ret = v_fp8.to(query.dtype)

    if return_2d:
        q_ret = q_ret.reshape(num_tokens, -1)
        k_ret = k_ret.reshape(num_tokens, -1)
        v_ret = v_ret.reshape(num_tokens, -1)

    return q_ret, k_ret, v_ret


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
    positions = torch.stack([
        torch.arange(num_tokens, device=DEVICE, dtype=torch.int64),  # t
        torch.arange(num_tokens, device=DEVICE, dtype=torch.int64),  # h
        torch.arange(num_tokens, device=DEVICE, dtype=torch.int64),  # w
    ], dim=0).contiguous()
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
    q = q_3d.reshape(num_tokens, h_q * dim)
    k = k_3d.reshape(num_tokens, h_k * dim)
    v = v_3d.reshape(num_tokens, h_k * dim)
    res_q_2d = res_q.reshape(num_tokens, h_q * dim) if use_residual else None
    res_k_2d = res_k.reshape(num_tokens, h_k * dim) if use_residual else None
    return q, k, v, positions, w_q, w_k, res_q_2d, res_k_2d, k_buf, v_buf, kv_loc, num_tokens


def dequant_fp8(x):
    if x.dtype == torch.float8_e4m3fn:
        return x.float()
    if x.dtype == torch.uint8:
        return x.view(torch.float8_e4m3fn).float()
    return x.float()


def compare_fp8_tensor(tensor_fp8, ref, name, atol=0.25, rtol=0.1):
    max_diff = (tensor_fp8 - ref).abs().max().item()
    mean_diff = (tensor_fp8 - ref).abs().mean().item()
    assert torch.allclose(tensor_fp8, ref, atol=atol, rtol=rtol), \
        f"{name} fp8 mismatch! max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    print(f"   > {name}_fp8: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} (atol={atol}, rtol={rtol})")
    return max_diff


def run_one(shape, is_neox, use_residual, use_fp8, is_interleaved):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    m_t, m_h, m_w = get_mrope_params(rot_dim, "balanced")

    # 基准：fake
    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref, v_buf_ref = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_ref = res_q.clone() if use_residual else None
    r_k_ref = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = gammarms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_ref, k_ref, v_ref, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_ref, residual_k=r_k_ref,
        epsilon=RMS_EPS, m_t=m_t, m_h=m_h, m_w=m_w, is_interleaved=is_interleaved,
    )

    # 被测：real
    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = gammarms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
        mrope_section_t=m_t,
        mrope_section_h=m_h,
        mrope_section_w=m_w,
        is_interleaved=is_interleaved,
    )

    atol, rtol = (0.25, 0.1) if use_fp8 else (1e-5, 1e-2)

    def cmp(a, b, name):
        if use_fp8:
            a = dequant_fp8(a)
            b = dequant_fp8(b)
            compare_fp8_tensor(a, b, name, atol=atol, rtol=rtol)
        else:
            compare_tensor(a, b, name, DTYPE)

    cmp(q_out, q_ref_out, "q")
    cmp(k_out, k_ref_out, "k")
    cmp(v_out, v_ref_out, "v")
    cmp(k_buf_tgt, k_buf_ref, "k_buf")
    cmp(v_buf_tgt, v_buf_ref, "v_buf")


def run_one_2d(shape, is_neox, use_residual, use_fp8, is_interleaved):
    """2D 测试：q,k,v shape 为 [B*S, num_head*head_dim]，k_buffer/v_buffer 与 3D 一致。"""
    batch, seq, h_q, h_k, dim, rot_dim = shape
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors_2d(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(batch * seq, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    m_t, m_h, m_w = get_mrope_params(rot_dim, "balanced")

    # 基准：fake（2D 输入）
    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref = torch.zeros_like(k_buf_org)
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q_ref = res_q.clone() if use_residual else None
    r_k_ref = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = gammarms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_ref, k_ref, v_ref, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_ref, residual_k=r_k_ref,
        epsilon=RMS_EPS, m_t=m_t, m_h=m_h, m_w=m_w, is_interleaved=is_interleaved,
    )

    # 被测：real（2D 输入）
    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = gammarms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
        mrope_section_t=m_t,
        mrope_section_h=m_h,
        mrope_section_w=m_w,
        is_interleaved=is_interleaved,
    )
    q_out = q_out.reshape(q_ref_out.shape)
    k_out = k_out.reshape(k_ref_out.shape)
    v_out = v_out.reshape(v_ref_out.shape)
    atol, rtol = (0.25, 0.1) if use_fp8 else (1e-5, 1e-2)

    def cmp(a, b, name):
        if use_fp8:
            a = dequant_fp8(a)
            b = dequant_fp8(b)
            compare_fp8_tensor(a, b, name, atol=atol, rtol=rtol)
        else:
            compare_tensor(a, b, name, DTYPE)

    cmp(q_out, q_ref_out, "q_2d")
    cmp(k_out, k_ref_out, "k_2d")
    cmp(v_out, v_ref_out, "v_2d")
    cmp(k_buf_tgt, k_buf_ref, "k_buf_2d")
    cmp(v_buf_tgt, v_buf_ref, "v_buf_2d")


def run_benchmark(shape, is_neox, use_residual, use_fp8, is_interleaved, warmup=PERF_WARMUP, repeat=PERF_REPEAT):
    """对 gammarms_rotary_embedding_fuse_with_kv_store 做性能测试，返回 (mean_ms, std_ms, num_tokens)。"""
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q, k, v, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, _ = get_tensors(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    m_t, m_h, m_w = get_mrope_params(rot_dim, "balanced")

    def run_once():
        q_t, k_t, v_t = q.clone(), k.clone(), v.clone()
        k_buf_t = torch.zeros_like(k_buf)
        v_buf_t = torch.zeros_like(v_buf)
        r_q = res_q.clone() if use_residual else None
        r_k = res_k.clone() if use_residual else None
        gammarms_rotary_embedding_fuse_with_kv_store(
            positions, q_t, k_t, v_t, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
            k_buffer=k_buf_t, v_buffer=v_buf_t, kv_cache_loc=kv_loc, is_neox=is_neox,
            weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q, residual_k=r_k,
            k_scale=1.0 if use_fp8 else None, v_scale=1.0 if use_fp8 else None, epsilon=RMS_EPS,
            mrope_section_t=m_t,
            mrope_section_h=m_h,
            mrope_section_w=m_w,
            is_interleaved=is_interleaved,
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
    print("-" * 110)
    print(f"{'shape':<45} {'interleaved':>12} {'mean(ms)':>10} {'std(ms)':>10} {'tokens/s':>12}")
    print("-" * 110)
    for shape in PERF_SHAPES:
        for is_interleaved in TEST_INTERLEAVED:
            batch, seq, h_q, h_k, dim, rot_dim = shape
            mean_ms, std_ms, num_tokens = run_benchmark(
                shape, is_neox=True, use_residual=False, use_fp8=True,
                is_interleaved=is_interleaved,
                warmup=PERF_WARMUP, repeat=PERF_REPEAT,
            )
            tokens_per_sec = num_tokens / (mean_ms * 1e-3) if mean_ms > 0 else 0
            shape_str = str(shape)
            print(f"{shape_str:<45} {str(is_interleaved):>12} {mean_ms:>10.3f} {std_ms:>10.3f} {tokens_per_sec:>12.0f}")
    print("-" * 110)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return

    for neox, resid, v_fp8, is_interleaved in itertools.product(
        TEST_NEOX, TEST_RESIDUALS, TEST_V_FP8, TEST_INTERLEAVED
    ):
        for shape in TEST_SHAPES:
            print(f"Shape={shape}, Neox={neox}, Res={resid}, V_fp8={v_fp8}, Interleaved={is_interleaved}")
            run_one(shape, neox, resid, v_fp8, is_interleaved)
            print("  OK (3D)")
            print("  2D [B*S, num_head*head_dim]: ", end="")
            run_one_2d(shape, neox, resid, v_fp8, is_interleaved)
            print("OK (2D)")
    print("All passed vs gammarms_rotary_embedding_fuse_with_kv_store_fake.")


if __name__ == "__main__":
    run_perf_tests()
    main()