"""
测试 rms_rotary_embedding_fuse_with_kv_store_no_quant：无 fp8 量化，Q/K/V 与 k_buffer/v_buffer 均为 bf16/fp16。
Kernel 条件：head_size=128, (num_heads+num_kv_heads)<=128, rot_dim in (128, 64)。
"""
import itertools
import torch
from lightop.attention import rms_rotary_embedding_fuse_with_kv_store
from lightop import op as lightop_op

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64

# (batch, seq, h_q, h_k, head_dim, rot_dim)，满足 qwen3: head_size=128, rot_dim in (128,64), num_heads+num_kv_heads<=128
TEST_SHAPES = [
    (128, 1, 8, 1, 128, 128),
    (128, 1, 8, 1, 128, 64),
    (64, 1, 4, 1, 128, 128),
]
TEST_NEOX = [True]
TEST_RESIDUALS = [False, True]


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


def build_cos_sin_cache(max_pos, dim, base=10000.0, dtype=torch.float32, device="cuda"):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(0, max_pos, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def get_tensors(batch, seq, h_q, h_k, dim, page_size, use_residual):
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
    k_buf = torch.zeros(num_blocks, h_k, page_size, dim, device=DEVICE, dtype=DTYPE)
    v_buf = torch.zeros(num_blocks, h_k, dim, page_size, device=DEVICE, dtype=DTYPE)
    kv_loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    return q, k, v, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, num_tokens


def get_tensors_2d(batch, seq, h_q, h_k, dim, page_size, use_residual):
    q_3d, k_3d, v_3d, positions, w_q, w_k, res_q, res_k, k_buf, v_buf, kv_loc, num_tokens = get_tensors(
        batch, seq, h_q, h_k, dim, page_size, use_residual
    )
    num_tokens = batch * seq
    q = q_3d.reshape(num_tokens, h_q * dim)
    k = k_3d.reshape(num_tokens, h_k * dim)
    v = v_3d.reshape(num_tokens, h_k * dim)
    res_q_2d = res_q.reshape(num_tokens, h_q * dim) if use_residual else None
    res_k_2d = res_k.reshape(num_tokens, h_k * dim) if use_residual else None
    return q, k, v, positions, w_q, w_k, res_q_2d, res_k_2d, k_buf, v_buf, kv_loc, num_tokens


def ref_no_quant_3d(positions, q, k, v, cos_sin, head_dim, page_size, k_buf, v_buf, kv_loc,
                    is_neox, w_q, w_k, res_q, res_k, epsilon):
    """参考实现：op.rms_rotary_embedding_fuse 再写 kv buffer。"""
    lightop_op.rms_rotary_embedding_fuse(
        positions, q, k, head_dim, cos_sin, is_neox, w_q, w_k, res_q, res_k, epsilon
    )
    loc_id = kv_loc // page_size
    loc_offset = kv_loc % page_size
    k_buf[loc_id, :, loc_offset, :] = k
    v_buf[loc_id, :, :, loc_offset] = v
    return q, k, v


def run_one(shape, is_neox, use_residual):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    # 参考：rms_rotary_embedding_fuse + scatter
    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref = torch.zeros_like(k_buf_org)
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q = res_q.clone() if use_residual else None
    r_k = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = ref_no_quant_3d(
        positions, q_ref, k_ref, v_ref, cos_sin, dim, PAGE_SIZE,
        k_buf_ref, v_buf_ref, kv_loc, is_neox, w_q, w_k, r_q, r_k, RMS_EPS
    )

    # 被测：no_quant kernel
    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_tgt = torch.zeros_like(k_buf_org)
    v_buf_tgt = torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=None, v_scale=None, epsilon=RMS_EPS,
    )

    def cmp(a, b, name):
        compare_tensor(a, b, name, DTYPE)

    cmp(q_out, q_ref_out, "q")
    cmp(k_out, k_ref_out, "k")
    cmp(v_out, v_ref_out, "v")
    cmp(k_buf_tgt, k_buf_ref, "k_buf")
    cmp(v_buf_tgt, v_buf_ref, "v_buf")


def ref_no_quant_2d(positions, q, k, v, cos_sin, head_dim, page_size, k_buf, v_buf, kv_loc,
                    is_neox, w_q, w_k, res_q, res_k, epsilon, h_q, h_k):
    lightop_op.rms_rotary_embedding_fuse(
        positions, q, k, head_dim, cos_sin, is_neox, w_q, w_k, res_q, res_k, epsilon
    )
    loc_id = kv_loc // page_size
    loc_offset = kv_loc % page_size
    num_tokens = q.shape[0]
    k_3d = k.view(num_tokens, k_buf.shape[1], -1)
    v_3d = v.view(num_tokens, v_buf.shape[1], -1)
    k_buf[loc_id, :, loc_offset, :] = k_3d
    v_buf[loc_id, :, :, loc_offset] = v_3d
    return q, k, v


def run_one_2d(shape, is_neox, use_residual):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    q_org, k_org, v_org, positions, w_q, w_k, res_q, res_k, k_buf_org, v_buf_org, kv_loc, _ = get_tensors_2d(
        batch, seq, h_q, h_k, dim, PAGE_SIZE, use_residual
    )
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin = cos_sin.to(DTYPE)
    cos_sin = cos_sin.contiguous()

    q_ref, k_ref, v_ref = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_ref = torch.zeros_like(k_buf_org)
    v_buf_ref = torch.zeros_like(v_buf_org)
    r_q = res_q.clone() if use_residual else None
    r_k = res_k.clone() if use_residual else None
    q_ref_out, k_ref_out, v_ref_out = ref_no_quant_2d(
        positions, q_ref, k_ref, v_ref, cos_sin, dim, PAGE_SIZE,
        k_buf_ref, v_buf_ref, kv_loc, is_neox, w_q, w_k, r_q, r_k, RMS_EPS, h_q, h_k
    )

    q_tgt, k_tgt, v_tgt = q_org.clone(), k_org.clone(), v_org.clone()
    k_buf_tgt = torch.zeros_like(k_buf_org)
    v_buf_tgt = torch.zeros_like(v_buf_org)
    r_q_tgt = res_q.clone() if use_residual else None
    r_k_tgt = res_k.clone() if use_residual else None
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_tgt, k_tgt, v_tgt, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=DTYPE, residual_q=r_q_tgt, residual_k=r_k_tgt,
        k_scale=None, v_scale=None, epsilon=RMS_EPS,
    )

    def cmp(a, b, name):
        compare_tensor(a, b, name, DTYPE)

    cmp(q_out, q_ref_out, "q_2d")
    cmp(k_out, k_ref_out, "k_2d")
    cmp(v_out, v_ref_out, "v_2d")
    cmp(k_buf_tgt, k_buf_ref, "k_buf_2d")
    cmp(v_buf_tgt, v_buf_ref, "v_buf_2d")


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    print("rms_rotary_embedding_fuse_with_kv_store_no_quant vs ref (rms_rotary_embedding_fuse + scatter)")
    for neox, resid in itertools.product(TEST_NEOX, TEST_RESIDUALS):
        for shape in TEST_SHAPES:
            print(f"  Shape={shape}, Neox={neox}, Res={resid}")
            run_one(shape, neox, resid)
            print("    OK (3D)")
            run_one_2d(shape, neox, resid)
            print("    OK (2D)")
    print("All passed.")


if __name__ == "__main__":
    main()
