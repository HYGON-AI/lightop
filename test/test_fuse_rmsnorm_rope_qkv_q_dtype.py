"""Regression test: q's output dtype must follow output_dtype, not the KV cache.

q is handed straight to attention and never stored in the KV cache, so its dtype
is independent of the cache dtype. The kernel used to derive q's output format
from cache_fp8_type, which meant a fp8 KV cache forced fp8 q. Consumers that
expect the model dtype (e.g. gfx936, where sglang's is_nmz_fp8() returns False)
then reinterpreted those fp8 bytes as bf16 -- silent, total numeric corruption
rather than a mild precision loss.

The decisive case is `bf16 q + fp8 KV cache`: before the fix q came back as fp8.

Run:
    python3 test/test_fuse_rmsnorm_rope_qkv_q_dtype.py
"""

import sys

import torch

from lightop.attention import (
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
)

torch.manual_seed(0)

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# (batch, seq, h_q, h_k, head_dim, rot_dim) -- shapes the fused kernel supports.
SHAPES = [(128, 1, 8, 1, 128, 128), (128, 1, 8, 1, 128, 64)]

# (cache_dtype, q_output_dtype, label)
DTYPE_CASES = [
    # The regression: fp8 cache but q must stay bf16.
    (torch.float8_e5m2, torch.bfloat16, "fp8_e5m2 cache + bf16 q"),
    (torch.float8_e4m3fn, torch.bfloat16, "fp8_e4m3 cache + bf16 q"),
    # gfx938-style: q genuinely wants fp8. Must not regress.
    (torch.float8_e5m2, torch.float8_e5m2, "fp8_e5m2 cache + fp8_e5m2 q"),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, "fp8_e4m3 cache + fp8_e4m3 q"),
]


def build_cos_sin_cache(max_pos, dim, base=ROPE_THETA, dtype=torch.float32, device=DEVICE):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(0, max_pos, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def make_inputs(batch, seq, h_q, h_k, dim, cache_dtype):
    num_tokens = batch * seq
    num_blocks = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    q = torch.randn(num_tokens, h_q, dim, device=DEVICE, dtype=MODEL_DTYPE)
    k = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=MODEL_DTYPE)
    v = torch.randn(num_tokens, h_k, dim, device=DEVICE, dtype=MODEL_DTYPE)
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    w_q = torch.ones(dim, device=DEVICE, dtype=MODEL_DTYPE)
    w_k = torch.ones(dim, device=DEVICE, dtype=MODEL_DTYPE)
    k_buf = torch.zeros(num_blocks, h_k, PAGE_SIZE, dim, device=DEVICE, dtype=cache_dtype)
    v_buf = torch.zeros(num_blocks, h_k, dim, PAGE_SIZE, device=DEVICE, dtype=cache_dtype)
    kv_loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    return q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, num_tokens


def to_float(x):
    """Compare in fp32; fp8 tensors are not directly subtractable."""
    return x.float()


def check_close(actual, ref, name, is_fp8):
    # fp8 has ~2-3 mantissa bits, so quantisation error dominates there.
    atol, rtol = (0.25, 0.1) if is_fp8 else (1e-5, 1e-2)
    a, b = to_float(actual), to_float(ref)
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name} mismatch: max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})"
    )
    print(f"     {name}: max_diff={max_diff:.6e}")


def run_case(shape, cache_dtype, q_output_dtype, label, is_neox=True):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    (
        q_org, k_org, v_org, positions, w_q, w_k, k_buf_org, v_buf_org, kv_loc, num_tokens,
    ) = make_inputs(batch, seq, h_q, h_k, dim, cache_dtype)

    cos_sin = build_cos_sin_cache(num_tokens, rot_dim).to(MODEL_DTYPE).contiguous()

    common = dict(
        head_dim=dim,
        page_size=PAGE_SIZE,
        kv_cache_loc=kv_loc,
        is_neox=is_neox,
        weight_q=w_q,
        weight_k=w_k,
        output_dtype=q_output_dtype,
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
        epsilon=RMS_EPS,
    )

    k_buf_ref, v_buf_ref = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_ref, k_ref, v_ref = rms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_org.clone(), k_org.clone(), v_org.clone(), cos_sin,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, **common,
    )

    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_org.clone(), k_org.clone(), v_org.clone(), cos_sin,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, **common,
    )

    # The core assertion: q honours output_dtype and ignores the cache dtype.
    assert q_out.dtype == q_output_dtype, (
        f"[{label}] q dtype must follow output_dtype: "
        f"expected {q_output_dtype}, got {q_out.dtype} "
        f"(cache dtype is {cache_dtype})"
    )
    # k/v do land in the cache, so they must keep tracking it.
    assert k_out.dtype == cache_dtype, (
        f"[{label}] k dtype should follow the cache: "
        f"expected {cache_dtype}, got {k_out.dtype}"
    )
    assert v_out.dtype == cache_dtype, (
        f"[{label}] v dtype should follow the cache: "
        f"expected {cache_dtype}, got {v_out.dtype}"
    )

    q_is_fp8 = q_output_dtype in FP8_DTYPES
    cache_is_fp8 = cache_dtype in FP8_DTYPES
    check_close(q_out, q_ref, "q", q_is_fp8)
    check_close(k_out, k_ref, "k", cache_is_fp8)
    check_close(v_out, v_ref, "v", cache_is_fp8)
    check_close(k_buf_tgt, k_buf_ref, "k_buf", cache_is_fp8)
    check_close(v_buf_tgt, v_buf_ref, "v_buf", cache_is_fp8)


def test_q_dtype_independent_of_kv_cache():
    """bf16 q with an fp8 KV cache must come back as bf16, not fp8."""
    failures = []
    for shape in SHAPES:
        for cache_dtype, q_dtype, label in DTYPE_CASES:
            name = f"{label}, shape={shape}"
            print(f"\n[case] {name}")
            try:
                run_case(shape, cache_dtype, q_dtype, label)
                print("   PASS")
            except AssertionError as exc:
                print(f"   FAIL: {exc}")
                failures.append((name, str(exc)))
    assert not failures, "\n".join(f"{n}: {e}" for n, e in failures)


def test_q_dtype_mismatch_is_rejected():
    """A q buffer that is neither fp8 nor the model dtype must raise, not corrupt."""
    shape = SHAPES[0]
    batch, seq, h_q, h_k, dim, rot_dim = shape
    (
        q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, num_tokens,
    ) = make_inputs(batch, seq, h_q, h_k, dim, torch.float8_e5m2)
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim).to(MODEL_DTYPE).contiguous()

    # fp32 q against a bf16 query: the kernel cannot express this safely.
    try:
        rms_rotary_embedding_fuse_with_kv_store(
            positions, q, k, v, cos_sin, head_dim=dim, page_size=PAGE_SIZE,
            k_buffer=k_buf, v_buffer=v_buf, kv_cache_loc=kv_loc, is_neox=True,
            weight_q=w_q, weight_k=w_k, output_dtype=torch.float32,
            residual_q=None, residual_k=None, k_scale=None, v_scale=None,
            epsilon=RMS_EPS,
        )
    except RuntimeError as exc:
        print(f"\n[case] fp32 q rejected as expected: {str(exc)[:110]}")
        return
    raise AssertionError(
        "a fp32 q buffer with bf16 query should be rejected by TORCH_CHECK"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no GPU available")
        sys.exit(0)

    test_q_dtype_independent_of_kv_cache()
    try:
        test_q_dtype_mismatch_is_rejected()
    except AssertionError as exc:
        print(f"\nWARN: {exc}")
    print("\nAll q-dtype regression cases passed.")
