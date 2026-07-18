"""
测试 split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant：
输入为合并的 qkv [num_tokens, q_size + kv_size + kv_size]，内部做 split + RMSNorm + RoPE + 写 KV cache。
参考实现：op.rms_rotary_embedding_fuse（分离 q/k/v 后） + 手动写 KV buffer。
q,k,v 为 2D：q [num_tokens, h_q*dim]，k/v [num_tokens, h_k*dim]。
Kernel 条件：head_size=128, (num_heads+num_kv_heads)<=128, rot_dim in (128, 64)。
"""
import itertools
import torch
from lightop.attention import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
from lightop import op as lightop_op

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64

# (batch, seq, h_q, h_k, head_dim, rot_dim)
TEST_SHAPES = [
    (8, 32, 12, 1, 128, 128),
    (8, 32, 12, 1, 128, 64),
    (8, 32, 12, 1, 128, 128),
]
TEST_NEOX = [False]
TEST_RESIDUALS = [False, True]
# cache 为 fp8 时 k_buffer 为 fp8，v_buffer 为 bf16
TEST_FP8_CACHE = [False, True]
# kv_cache_loc 为 None 时只做 split + RMSNorm + RoPE，不写 cache
TEST_KV_CACHE = [True, False]


def get_precision_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2


def compare_tensor(a, b, name, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = (a - b).abs().max().item()
    print(f"   > {name}: max_diff={max_diff:.6e}")
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch!"
    return max_diff


def dequant_fp8(x):
    """fp8 buffer 反量化为 float 便于与 ref 比较。"""
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return x.float()
    if x.dtype == torch.uint8:
        # 存的是 fp8 字节，按 float8_e5m2 解释后转 float（与 kernel/ref 一致）
        return x.view(torch.float8_e5m2).float()
    return x


def compare_fp8_tensor(a, b, name, atol=0.25, rtol=0.1):
    """比较 fp8 张量（反量化后转 float）。"""
    a_f = dequant_fp8(a).float()
    b_f = dequant_fp8(b).float()
    max_diff = (a_f - b_f).abs().max().item()
    print(f"   > {name}_fp8: max_diff={max_diff:.6e}")
    assert torch.allclose(a_f, b_f, atol=atol, rtol=rtol), f"{name} fp8 mismatch!"
    return max_diff


def build_cos_sin_cache(max_pos, dim, base=10000.0, dtype=torch.float32, device="cuda"):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(0, max_pos, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def get_tensors_split_qkv(batch, seq, h_q, h_k, dim, page_size, use_residual, use_fp8_cache=False, use_kv_cache=True):
    num_tokens = batch * seq
    num_blocks = max(1, (num_tokens + page_size - 1) // page_size)
    q_size = h_q * dim
    kv_size = h_k * dim
    # qkv: [num_tokens, q_size + kv_size + kv_size]
    q = torch.randn(num_tokens, h_q, dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE)
    qkv_flat = torch.cat([
        q.reshape(num_tokens, q_size),
        k.reshape(num_tokens, kv_size),
        v.reshape(num_tokens, kv_size),
    ], dim=-1)
    res_q = torch.randn(num_tokens, h_q, dim, device=DEVICE, dtype=DTYPE) if use_residual else None
    res_k = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=DTYPE) if use_residual else None
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    w_q = torch.randn(dim, device=DEVICE, dtype=DTYPE)
    w_k = torch.randn(dim, device=DEVICE, dtype=DTYPE)
    if use_kv_cache and use_fp8_cache:
        k_buf = torch.zeros(num_blocks, h_k, page_size, dim, device=DEVICE, dtype=torch.uint8)
        v_buf = torch.zeros(num_blocks, h_k, dim, page_size, device=DEVICE, dtype=torch.uint8)
    else:
        k_buf = torch.zeros(num_blocks, h_k, page_size, dim, device=DEVICE, dtype=DTYPE)
        v_buf = torch.zeros(num_blocks, h_k, dim, page_size, device=DEVICE, dtype=DTYPE)
    kv_loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64) if use_kv_cache else None
    return qkv_flat, q_size, kv_size, res_q, res_k, positions, w_q, w_k, k_buf, v_buf, kv_loc, num_tokens, h_q, h_k, dim


def ref_no_quant_2d(positions, q, k, v, cos_sin, head_dim, page_size, k_buf, v_buf, kv_loc,
                    is_neox, w_q, w_k, res_q, res_k, epsilon, use_fp8_cache=False):
    """参考：op.rms_rotary_embedding_fuse（q/k 为 2D）；kv_loc 非 None 时再写 kv buffer。"""
    lightop_op.rms_rotary_embedding_fuse(
        positions, q, k, head_dim, cos_sin, is_neox, w_q, w_k, res_q, res_k, epsilon
    )
    if kv_loc is not None:
        loc_id = kv_loc // page_size
        loc_offset = kv_loc % page_size
        num_tokens = k.shape[0]
        k_3d = k.reshape(num_tokens, -1, head_dim)
        v_3d = v.reshape(num_tokens, -1, head_dim)
        if use_fp8_cache:
            if k_buf.dtype == torch.uint8:
                k_buf = k_buf.view(torch.float8_e5m2)
                v_buf = v_buf.view(torch.float8_e5m2)
            k_buf[loc_id, :, loc_offset, :] = k_3d.to(torch.float8_e5m2)
            v_buf[loc_id, :, :, loc_offset] = v_3d.to(torch.float8_e5m2)
        else:
            k_buf[loc_id, :, loc_offset, :] = k_3d
            v_buf[loc_id, :, :, loc_offset] = v_3d
    return q, k, v


def run_one(shape, is_neox, use_residual, use_fp8_cache=False, use_kv_cache=True):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    qkv_flat, q_size, kv_size, res_q, res_k, positions, w_q, w_k, k_buf_org, v_buf_org, kv_loc, _, _, _, _ = get_tensors_split_qkv(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual, use_fp8_cache=use_fp8_cache, use_kv_cache=use_kv_cache
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    qkv_c = qkv_flat.clone()
    q_ref = qkv_c[:, :q_size].contiguous()
    k_ref = qkv_c[:, q_size:q_size+kv_size].contiguous()
    v_ref = qkv_c[:, q_size+kv_size:].contiguous()
    k_buf_ref = torch.zeros_like(k_buf_org)
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q = res_q.clone() if use_residual else None
    r_k = res_k.clone() if use_residual else None

    q_ref_out, k_ref_out, v_ref_out = ref_no_quant_2d(
        positions, q_ref, k_ref, v_ref, cos_sin, dim, PAGE_SIZE,
        k_buf_ref, v_buf_ref, kv_loc, is_neox, w_q, w_k, r_q, r_k, RMS_EPS,
        use_fp8_cache=use_fp8_cache if use_kv_cache else False,
    )
    # 被测：split_qkv 接口（合并 qkv 输入）
    qkv_tgt = qkv_flat.clone()
    k_buf_tgt = torch.zeros_like(k_buf_org)
    v_buf_tgt = torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(
        positions,
        qkv_tgt,
        q_size,
        kv_size,
        cos_sin,
        head_dim=dim,
        page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt,
        v_buffer=v_buf_tgt,
        kv_cache_loc=kv_loc,
        is_neox=is_neox,
        weight_q=w_q,
        weight_k=w_k,
        output_dtype=DTYPE,
        kv_cache_dtype=torch.float8_e5m2 if use_fp8_cache else DTYPE,
        residual_q=r_q_tgt,
        residual_k=r_k_tgt,
        k_scale=None,
        v_scale=None,
        epsilon=RMS_EPS,
    )
    if use_fp8_cache and k_buf_tgt.dtype == torch.uint8:
        k_buf_tgt = k_buf_tgt.view(torch.float8_e5m2)
        v_buf_tgt = v_buf_tgt.view(torch.float8_e5m2)
    def cmp(a, b, name):
        compare_tensor(a, b, name, DTYPE)

    def cmp_fp8(a, b, name):
        compare_fp8_tensor(a, b, name)

    cmp(q_out, q_ref_out, "q")
    cmp(k_out, k_ref_out, "k")
    cmp(v_out, v_ref_out, "v")
    if use_kv_cache:
        if use_fp8_cache:
            cmp_fp8(k_buf_tgt, k_buf_ref, "k_buf")
            cmp_fp8(v_buf_tgt, v_buf_ref, "v_buf")
        else:
            cmp(k_buf_tgt, k_buf_ref, "k_buf")
            cmp(v_buf_tgt, v_buf_ref, "v_buf")


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    print("split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant vs ref (op.rms_rotary_embedding_fuse + scatter)")
    for neox, resid, fp8_cache, kv_cache in itertools.product(
        TEST_NEOX, TEST_RESIDUALS, TEST_FP8_CACHE, TEST_KV_CACHE
    ):
        if not kv_cache and fp8_cache:
            continue  # kv_cache_loc=None 时不写 cache，fp8_cache 无意义
        for shape in TEST_SHAPES:
            print(f"  Shape={shape}, Neox={neox}, Res={resid}, fp8_cache={fp8_cache}, kv_cache={kv_cache}")
            run_one(shape, neox, resid, use_fp8_cache=fp8_cache, use_kv_cache=kv_cache)
            print("    OK")
    print("All passed.")


if __name__ == "__main__":
    main()
