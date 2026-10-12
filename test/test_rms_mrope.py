# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
from lightop.attention import (
    fuse_rms_mrope_cuda as categorized_fuse_rms_mrope_cuda,
)
from lightop.norm import fused_add_rms_norm_opt, rms_norm_opt
torch.manual_seed(0)
# ---------------- Triton mRoPE kernel ----------------
@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    # instead of (3, bsz, seq_len, head_dim), also supports interleaved rotary
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)

    # Updated stride calculation for half head_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = (((cos_offsets % 3) == 1) &
                  (cos_offsets <= 3 * mrope_section_h))
        w_mask = (((cos_offsets % 3) == 2) &
                  (cos_offsets <= 3 * mrope_section_w))
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets,
                       mask=first_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets,
                       mask=first_k_mask,
                       other=0).to(sin_row.dtype)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets,
                       mask=second_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets,
                       mask=second_k_mask,
                       other=0).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # Since cos and sin are now half-size,
    # we use the same cos_row and sin_row for both halves
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def triton_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim, interleaved):
    num_tokens, hidden_size = q.shape
    num_head = hidden_size // head_size
    # n_qh = n_kh = num_head
    n_qh = hidden_size // head_size

    n_kh =  k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)

    _triton_mrope_forward[(num_tokens,)](
        q, k, cos, sin,
        num_tokens, n_qh, n_kh, head_size, rotary_dim,
        pad_n_qh, pad_n_kh, pad_hd,
        mrope_section[0], mrope_section[1], mrope_section[2],
        interleaved,
    )
    return q, k




def compare_and_bench(
    dtype=torch.float16,
    num_tokens=1,
    n_q_heads=8,
    n_kv_heads=4,
    head_size=64,
    rotary_dim=64,
    mrope_section=(24, 20, 20),
    interleaved=True,
    use_residual=False,
):
    device = "cuda"
    

    hidden_size_q = n_q_heads * head_size
    hidden_size_k = n_kv_heads * head_size

    # --- Inputs (Master Copy) ---
    # 模拟真实场景：
    # 1. 输入数据通常是矩阵乘法的结果，方差较大。
    #    这里乘以 (head_size ** 0.5) 模拟 Linear 层的输出分布。
    scale_factor = head_size ** 0.5
    q = torch.randn(num_tokens, hidden_size_q, device=device, dtype=dtype) * scale_factor
    k = torch.randn(num_tokens, hidden_size_k, device=device, dtype=dtype) * scale_factor

    # 2. RMSNorm 的权重 (Gamma) 通常在 1.0 附近。
    #    原始的 randn (均值0) 不符合物理意义，容易导致数值下溢或测试不稳定。
    #    这里模拟训练后的权重：均值为 1.0，有少量波动。
    weight_q = torch.rand(head_size, device=device, dtype=dtype) * 0.2 + 0.9  # 范围 [0.9, 1.1]
    weight_k = torch.rand(head_size, device=device, dtype=dtype) * 0.2 + 0.9

    # 3. Residual 通常也具有较大的方差
    if use_residual:
        residual_q = torch.randn_like(q) * scale_factor
        residual_k = torch.randn_like(k) * scale_factor
    else:
        residual_q = None
        residual_k = None

    cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)

    # ==========================================
    # 1. 精度验证 (Correctness Check)
    # ==========================================
    
    # --- Baseline: RMSNorm + Triton mRoPE ---
    # Clone inputs to avoid modifying master copies
    q_rms = q.clone().view(num_tokens * n_q_heads, head_size)
    k_rms = k.clone().view(num_tokens * n_kv_heads, head_size)
    
    res_q_rms = residual_q.clone() if use_residual else None
    res_k_rms = residual_k.clone() if use_residual else None
    q_fused = q.clone()
    k_fused = k.clone()
    res_q_fused = residual_q.clone() if use_residual else None
    res_k_fused = residual_k.clone() if use_residual else None
    if use_residual:
        # lightop fused ops modify residual in-place
        fused_add_rms_norm_opt(q_rms, res_q_rms.view(num_tokens*n_q_heads, head_size), weight_q, 1e-6)
        fused_add_rms_norm_opt(k_rms, res_k_rms.view(num_tokens*n_kv_heads, head_size), weight_k, 1e-6)
    else:
        rms_norm_opt(q_rms, q_rms, weight_q, 1e-6)
        rms_norm_opt(k_rms, k_rms, weight_k, 1e-6)
    import pdb
    # pdb.set_trace()
    q_triton = q_rms.view(num_tokens, n_q_heads * head_size)
    k_triton = k_rms.view(num_tokens, n_kv_heads * head_size)
    triton_mrope(q_triton, k_triton, cos, sin, mrope_section, head_size, rotary_dim, interleaved)
    
    categorized_fuse_rms_mrope_cuda(q_fused, k_fused, cos, sin, list(mrope_section), head_size, rotary_dim,
                    interleaved, weight_q, weight_k, res_q_fused, res_k_fused)
    torch.cuda.synchronize()

    # --- Check ---
    atol = 2e-2 if dtype == torch.float16 else 1e-5
    q_max_err = (q_triton - q_fused).abs().max().item()
    k_max_err = (k_triton - k_fused).abs().max().item()
    print(q_fused)
    print(q_triton)
    print(f"\n=== Correctness check ({dtype}, interleaved={interleaved}, residual={use_residual}) ===")
    print("Q max err:", q_max_err)
    print("K max err:", k_max_err)
    if q_max_err > atol or k_max_err > atol:
        raise ValueError(f"Fusion mismatch exceeds tolerance {atol}")

    # ==========================================
    # 2. 性能测试 (Performance Benchmark)
    # ==========================================
    
    # 注意：bench 函数内部必须重新 clone 数据，否则 do_bench 多次迭代会累积修改数据，导致 NaN 或逻辑错误。
    # 为了公平对比，Baseline 和 Fused 都包含了 clone 的开销。
    
    def bench_baseline():
        # 1. Init (Aligned with correctness check)
        q_in = q.clone()
        k_in = k.clone()
        res_q_in = residual_q.clone() if use_residual else None
        res_k_in = residual_k.clone() if use_residual else None

        # 2. View
        q_rms_in = q_in.view(num_tokens * n_q_heads, head_size)
        k_rms_in = k_in.view(num_tokens * n_kv_heads, head_size)

        # 3. RMSNorm (Using _opt ops to match correctness check)
        if use_residual:
            fused_add_rms_norm_opt(q_rms_in, res_q_in.view(num_tokens*n_q_heads, head_size), weight_q, 1e-6)
            fused_add_rms_norm_opt(k_rms_in, res_k_in.view(num_tokens*n_kv_heads, head_size), weight_k, 1e-6)
        else:
            rms_norm_opt(q_rms_in, q_rms_in, weight_q, 1e-6)
            rms_norm_opt(k_rms_in, k_rms_in, weight_k, 1e-6)
        
        # 4. View back
        q_triton_in = q_rms_in.view(num_tokens, n_q_heads * head_size)
        k_triton_in = k_rms_in.view(num_tokens, n_kv_heads * head_size)
        
        # 5. Triton Kernel
        triton_mrope(q_triton_in, k_triton_in, cos, sin, mrope_section, head_size, rotary_dim, interleaved)

    def bench_fused():
        # 1. Init
        q_fused_in = q.clone()
        k_fused_in = k.clone()
        res_q_fused_in = residual_q.clone() if use_residual else None
        res_k_fused_in = residual_k.clone() if use_residual else None

        # 2. Fused Kernel
        categorized_fuse_rms_mrope_cuda(q_fused_in, k_fused_in, cos, sin, list(mrope_section), head_size, rotary_dim,
                        interleaved, weight_q, weight_k, res_q_fused_in, res_k_fused_in, 1e-6)

    # Warmup manually once
    bench_baseline()
    bench_fused()

    t_base = do_bench(bench_baseline)
    t_fused = do_bench(bench_fused)
    print(f"\n=== Performance ===")
    print(f"Baseline: {t_base*1e3:.3f} ms, Fused: {t_fused*1e3:.3f} ms, Speedup: {t_base/t_fused:.2f}x")


if __name__ == "__main__":
    compare_and_bench(
        dtype=torch.float16,
        num_tokens=20480,
        n_q_heads=16,
        n_kv_heads=2,
        head_size=128,
        rotary_dim=128,
        mrope_section=(24, 20, 20),
        interleaved=True,
        use_residual=False
    )
