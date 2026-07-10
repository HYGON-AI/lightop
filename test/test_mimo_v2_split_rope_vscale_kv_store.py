import time

import torch

from lightop import mimo_v2_split_rope_vscale_kv_store


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 192
V_HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
NUM_TOKENS = 128
VALUE_SCALE = 0.5
K_CACHE_SCALE = 0.75
V_CACHE_SCALE = 0.625


def build_cos_sin_cache(max_pos, dim, base=1000000.0):
    half = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=DEVICE) / half)
    )
    t = torch.arange(0, max_pos, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(DTYPE)


def apply_rope_ref(x, positions, cos_sin_cache, is_neox=True):
    x = x.float()
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[:, : HEAD_DIM // 2].unsqueeze(1)
    sin = cos_sin[:, HEAD_DIM // 2 :].unsqueeze(1)
    if is_neox:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        out = out.flatten(-2)
    return out.to(DTYPE)


def dequant_cache(x, dtype):
    if dtype == torch.float8_e4m3fn:
        return x.view(torch.float8_e4m3fn).float()
    if dtype == torch.float8_e5m2:
        return x.view(torch.float8_e5m2).float()
    return x.float()


def set_kv_buffer_ref(
    k_buffer,
    v_buffer,
    cache_loc,
    cache_k,
    cache_v,
    kv_cache_dtype,
    k_scale=None,
    v_scale=None,
):
    expected_k = torch.zeros_like(k_buffer)
    expected_v = torch.zeros_like(v_buffer)
    cache_loc = cache_loc.to(torch.long)

    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if k_scale is not None:
            cache_k = cache_k.float() / k_scale
        if v_scale is not None:
            cache_v = cache_v.float() / v_scale
        expected_k[cache_loc] = cache_k.to(kv_cache_dtype).view(torch.uint8)
        expected_v[cache_loc] = cache_v.to(kv_cache_dtype).view(torch.uint8)
    else:
        expected_k[cache_loc] = cache_k.to(k_buffer.dtype)
        expected_v[cache_loc] = cache_v.to(v_buffer.dtype)
    return expected_k, expected_v


def run_case(use_fp8_cache: bool, loc_dtype: torch.dtype):
    torch.manual_seed(0)
    q_size = NUM_Q_HEADS * HEAD_DIM
    k_size = NUM_KV_HEADS * HEAD_DIM
    v_size = NUM_KV_HEADS * V_HEAD_DIM
    qkv = torch.randn(
        NUM_TOKENS, q_size + k_size + v_size, device=DEVICE, dtype=DTYPE
    )
    positions = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int64)
    cos_sin_cache = build_cos_sin_cache(NUM_TOKENS, HEAD_DIM).contiguous()
    cache_loc = torch.arange(NUM_TOKENS, device=DEVICE, dtype=loc_dtype)

    if use_fp8_cache:
        k_buffer = torch.zeros(
            NUM_TOKENS + 8, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.uint8
        )
        v_buffer = torch.zeros(
            NUM_TOKENS + 8, NUM_KV_HEADS, V_HEAD_DIM, device=DEVICE, dtype=torch.uint8
        )
        kv_cache_dtype = torch.float8_e4m3fn
    else:
        k_buffer = torch.zeros(
            NUM_TOKENS + 8, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        v_buffer = torch.zeros(
            NUM_TOKENS + 8, NUM_KV_HEADS, V_HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        kv_cache_dtype = DTYPE

    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
    q_ref = apply_rope_ref(q.view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), positions, cos_sin_cache)
    k_ref = apply_rope_ref(k.view(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), positions, cos_sin_cache)
    v_ref = (v.view(NUM_TOKENS, NUM_KV_HEADS, V_HEAD_DIM).float() * VALUE_SCALE).to(DTYPE)

    q_out, k_out, v_out = mimo_v2_split_rope_vscale_kv_store(
        positions,
        qkv,
        q_size,
        k_size,
        v_size,
        cos_sin_cache,
        HEAD_DIM,
        V_HEAD_DIM,
        k_buffer,
        v_buffer,
        cache_loc,
        True,
        VALUE_SCALE,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=K_CACHE_SCALE,
        v_cache_scale=V_CACHE_SCALE,
    )

    torch.testing.assert_close(q_out.view_as(q_ref), q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out.view_as(k_ref), k_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v_out.view_as(v_ref), v_ref, atol=2e-2, rtol=2e-2)

    expected_k_buffer, expected_v_buffer = set_kv_buffer_ref(
        k_buffer,
        v_buffer,
        cache_loc,
        k_ref,
        v_ref,
        kv_cache_dtype,
        k_scale=K_CACHE_SCALE,
        v_scale=V_CACHE_SCALE,
    )

    if use_fp8_cache:
        torch.testing.assert_close(
            dequant_cache(k_buffer[:NUM_TOKENS], kv_cache_dtype),
            dequant_cache(expected_k_buffer[:NUM_TOKENS], kv_cache_dtype),
            atol=0.25,
            rtol=0.1,
        )
        torch.testing.assert_close(
            dequant_cache(v_buffer[:NUM_TOKENS], kv_cache_dtype),
            dequant_cache(expected_v_buffer[:NUM_TOKENS], kv_cache_dtype),
            atol=0.25,
            rtol=0.1,
        )
    else:
        torch.testing.assert_close(
            k_buffer[:NUM_TOKENS],
            expected_k_buffer[:NUM_TOKENS],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            v_buffer[:NUM_TOKENS],
            expected_v_buffer[:NUM_TOKENS],
            atol=2e-2,
            rtol=2e-2,
        )


def benchmark():
    q_size = NUM_Q_HEADS * HEAD_DIM
    k_size = NUM_KV_HEADS * HEAD_DIM
    v_size = NUM_KV_HEADS * V_HEAD_DIM
    qkv = torch.randn(
        NUM_TOKENS, q_size + k_size + v_size, device=DEVICE, dtype=DTYPE
    )
    positions = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int64)
    cos_sin_cache = build_cos_sin_cache(NUM_TOKENS, HEAD_DIM).contiguous()
    cache_loc = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int64)
    k_buffer = torch.zeros(
        NUM_TOKENS + 8, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    v_buffer = torch.zeros(
        NUM_TOKENS + 8, NUM_KV_HEADS, V_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    for _ in range(10):
        mimo_v2_split_rope_vscale_kv_store(
            positions, qkv, q_size, k_size, v_size, cos_sin_cache,
            HEAD_DIM, V_HEAD_DIM, k_buffer, v_buffer, cache_loc, True, VALUE_SCALE
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        mimo_v2_split_rope_vscale_kv_store(
            positions, qkv, q_size, k_size, v_size, cos_sin_cache,
            HEAD_DIM, V_HEAD_DIM, k_buffer, v_buffer, cache_loc, True, VALUE_SCALE
        )
    torch.cuda.synchronize()
    mean_us = (time.perf_counter() - start) * 1e6 / 100
    bytes_touched = NUM_TOKENS * (q_size + k_size + v_size) * qkv.element_size() * 2
    bandwidth_gbs = bytes_touched / (mean_us * 1e-6) / 1e9
    print(f"benchmark mean_us={mean_us:.3f}, effective_bandwidth={bandwidth_gbs:.3f} GB/s")


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    run_case(use_fp8_cache=False, loc_dtype=torch.int64)
    run_case(use_fp8_cache=True, loc_dtype=torch.int64)
    run_case(use_fp8_cache=True, loc_dtype=torch.int32)
    benchmark()
    print("All passed.")


if __name__ == "__main__":
    main()
