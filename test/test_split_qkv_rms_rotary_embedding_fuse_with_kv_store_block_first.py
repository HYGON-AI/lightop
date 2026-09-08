"""Detailed acceptance tests for the one-launch block-first QKV writer.

The reference is the pre-existing ``rms_rotary_embedding_fuse`` operator plus
an explicit Python scatter.  This keeps the fused kernel, its layout handling,
and its FP8 conversion as the only behavior under test.
"""

import torch

from lightop import op as lightop_op
from lightop.attention import (
    split_qkv_rms_rotary_embedding_fuse_with_kv_store_block_first,
)


torch.manual_seed(0)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128
Q_HEADS = 16  # TP=4 rank shape: generic 480-thread path, not 64Q/8KV HY3.
KV_HEADS = 2
RMS_EPS = 1e-5
SENTINEL = -3.0

# Covers each required token count, both block sizes, both RoPE modes, unified
# unbind, HND physical storage, and a non-unit final dimension stride.
BF16_CASES = (
    ("nhd", 1, 16, False),
    ("unified_hnd", 2, 16, True),
    ("padded", 3, 16, False),
    ("unified_hnd", 4, 32, True),
    ("padded", 17, 16, False),
    ("unified_hnd", 128, 32, True),
)
FP8_CASES = (
    ("nhd", 17, 16),
    ("padded", 17, 32),
)


def _skip_without_cuda() -> bool:
    if torch.cuda.is_available():
        return False
    print("SKIP: CUDA/HIP is unavailable")
    return True


def _precision_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2


def _compare_tensor(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    atol, rtol = _precision_tolerance(actual.dtype)
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    max_diff = (
        (actual_float[finite] - expected_float[finite]).abs().max().item()
        if finite.any()
        else 0.0
    )
    print(f"      {name}: max_diff={max_diff:.6e}")
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, equal_nan=True
    )


def _compare_fp8_bytes(
    actual: torch.Tensor, expected: torch.Tensor, name: str
) -> None:
    actual_bytes = actual.view(torch.uint8)
    expected_bytes = expected.view(torch.uint8)
    mismatch = actual_bytes != expected_bytes
    different = mismatch.sum().item()
    print(f"      {name}: different_fp8_bytes={different}")
    if different:
        byte_pairs = torch.stack(
            (actual_bytes[mismatch], expected_bytes[mismatch]), dim=1
        )
        unique_pairs, counts = torch.unique(
            byte_pairs, dim=0, return_counts=True
        )
        pair_summary = ", ".join(
            f"0x{actual:02x}->0x{expected:02x}: {count}"
            for (actual, expected), count in zip(
                unique_pairs.cpu().tolist(), counts.cpu().tolist()
            )
        )
        print(f"        actual->expected byte pairs: {pair_summary}")
    torch.testing.assert_close(actual_bytes, expected_bytes)


def _build_cos_sin_cache(
    max_position: int, rotary_dim: int = HEAD_DIM
) -> torch.Tensor:
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(half, device=DEVICE, dtype=torch.float32) / half)
    )
    positions = torch.arange(max_position, device=DEVICE, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE).contiguous()


def _build_inputs(
    tokens: int, blocks: int, block_size: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_size = Q_HEADS * HEAD_DIM
    kv_size = KV_HEADS * HEAD_DIM
    qkv = torch.randn(
        (tokens, q_size + 2 * kv_size), device=DEVICE, dtype=DTYPE
    )
    positions = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    capacity = blocks * block_size
    # Includes first/last positions in two blocks, the last cache position,
    # padding (-1), and two out-of-range values.  Each valid slot is unique:
    # cache writers from different GPU blocks have no ordering guarantee for
    # duplicate slots, so a sequential Python scatter is not their reference.
    required_slots = (
        0,
        block_size - 1,
        block_size,
        block_size + 1,
        capacity - 1,
        -1,
        capacity,
        capacity + 1,
    )
    used_slots = {slot for slot in required_slots if 0 <= slot < capacity}
    remaining_slots = [
        slot for slot in range(capacity) if slot not in used_slots
    ]
    candidates = (*required_slots, *remaining_slots)
    assert tokens <= len(candidates), "test cache capacity is too small"
    slots = torch.tensor(
        candidates[:tokens],
        device=DEVICE,
        dtype=torch.int64,
    )
    weight_q = torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE)
    weight_k = torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return qkv, positions, slots, weight_q, weight_k


def _blocks_for_tokens(tokens: int, block_size: int) -> int:
    # Leave room for two invalid-slot cases while retaining a unique valid
    # slot for every remaining token in the T=128 case.
    return max(3, (tokens + 2 + block_size - 1) // block_size)


def _padded_cache(
    blocks: int,
    block_size: int,
    dtype: torch.dtype,
    dim_stride: int,
) -> torch.Tensor:
    head_stride = HEAD_DIM * dim_stride + 5
    token_stride = KV_HEADS * head_stride + 7
    block_stride = block_size * token_stride + 11
    cache = torch.empty_strided(
        (blocks, block_size, KV_HEADS, HEAD_DIM),
        (block_stride, token_stride, head_stride, dim_stride),
        device=DEVICE,
        dtype=dtype,
    )
    cache.fill_(SENTINEL)
    return cache


def _make_cache_pair(
    layout: str,
    blocks: int,
    block_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == "nhd":
        key_cache = torch.full(
            (blocks, block_size, KV_HEADS, HEAD_DIM),
            SENTINEL,
            device=DEVICE,
            dtype=dtype,
        )
        return key_cache, torch.full_like(key_cache, SENTINEL)

    if layout == "unified_hnd":
        # Emulates [B, 2, N, H, D].unbind(1): data_ptr() starts at the
        # selected view, while stride(0) remains doubled.
        unified = torch.full(
            (blocks, 2, block_size, KV_HEADS, HEAD_DIM),
            SENTINEL,
            device=DEVICE,
            dtype=dtype,
        )
        key_cache = unified.unbind(1)[0]
        # HND physical storage, exposed as the required logical [B, N, H, D].
        value_storage = torch.full(
            (blocks, KV_HEADS, block_size, HEAD_DIM),
            SENTINEL,
            device=DEVICE,
            dtype=dtype,
        )
        value_cache = value_storage.permute(0, 2, 1, 3)
        return key_cache, value_cache

    if layout == "padded":
        # K and V intentionally have different non-unit last-dimension strides.
        return (
            _padded_cache(blocks, block_size, dtype, dim_stride=2),
            _padded_cache(blocks, block_size, dtype, dim_stride=3),
        )

    raise AssertionError(f"unknown cache layout: {layout}")


def _reference_qkv(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_size = Q_HEADS * HEAD_DIM
    kv_size = KV_HEADS * HEAD_DIM
    q = qkv[:, :q_size].contiguous()
    k = qkv[:, q_size:q_size + kv_size].contiguous()
    v = qkv[:, q_size + kv_size:].contiguous()
    lightop_op.rms_rotary_embedding_fuse(
        positions,
        q,
        k,
        HEAD_DIM,
        cos_sin,
        is_neox,
        weight_q,
        weight_k,
        None,
        None,
        RMS_EPS,
    )
    return q, k, v


def _reference_partial_rope_qkv(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference GLM-style RMSNorm plus NeoX partial RoPE on the first 64 dims."""
    q_size = Q_HEADS * HEAD_DIM
    kv_size = KV_HEADS * HEAD_DIM
    rotary_dim = cos_sin.size(1)
    q = qkv[:, :q_size].view(-1, Q_HEADS, HEAD_DIM).float()
    k = qkv[:, q_size:q_size + kv_size].view(-1, KV_HEADS, HEAD_DIM).float()
    v = qkv[:, q_size + kv_size:].contiguous()

    def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + RMS_EPS) * weight.float()

    def partial_rope(x: torch.Tensor) -> torch.Tensor:
        x = rmsnorm(x, weight_q if x.size(1) == Q_HEADS else weight_k)
        half = rotary_dim // 2
        cache = cos_sin[positions].float()
        cos = cache[:, :half].unsqueeze(1)
        sin = cache[:, half:].unsqueeze(1)
        low, high = x[..., :half], x[..., half:rotary_dim]
        rotated = torch.cat((low * cos - high * sin, high * cos + low * sin), dim=-1)
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1).to(DTYPE)

    return partial_rope(q).reshape(qkv.size(0), q_size), partial_rope(k).reshape(qkv.size(0), kv_size), v


def _write_reference(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    block_size = key_cache.size(1)
    capacity = key_cache.size(0) * block_size
    k_heads = k_out.reshape(k_out.size(0), KV_HEADS, HEAD_DIM)
    v_heads = v_out.reshape(v_out.size(0), KV_HEADS, HEAD_DIM)
    for token, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < capacity:
            block, offset = divmod(slot, block_size)
            key_cache[block, offset].copy_(k_heads[token])
            value_cache[block, offset].copy_(v_heads[token])


def _call_operator(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    block_size: int,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    slots: torch.Tensor | None,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    is_neox: bool,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    residual_q: torch.Tensor | None = None,
    residual_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return split_qkv_rms_rotary_embedding_fuse_with_kv_store_block_first(
        positions=positions,
        qkv=qkv,
        q_size=Q_HEADS * HEAD_DIM,
        kv_size=KV_HEADS * HEAD_DIM,
        cos_sin_cache=cos_sin,
        head_dim=HEAD_DIM,
        block_size=block_size,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slots,
        is_neox=is_neox,
        weight_q=weight_q,
        weight_k=weight_k,
        output_dtype=DTYPE,
        residual_q=residual_q,
        residual_k=residual_k,
        k_scale=k_scale,
        v_scale=v_scale,
        epsilon=RMS_EPS,
    )


def _run_bf16_case(
    layout: str, tokens: int, block_size: int, is_neox: bool
) -> None:
    blocks = _blocks_for_tokens(tokens, block_size)
    qkv, positions, slots, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens)
    q_ref, k_ref, v_ref = _reference_qkv(
        positions, qkv.clone(), cos_sin, weight_q, weight_k, is_neox
    )
    key_cache, value_cache = _make_cache_pair(layout, blocks, block_size, DTYPE)
    key_ref = key_cache.clone(memory_format=torch.preserve_format)
    value_ref = value_cache.clone(memory_format=torch.preserve_format)

    q_out, k_out, v_out = _call_operator(
        positions,
        qkv,
        cos_sin,
        block_size,
        key_cache,
        value_cache,
        slots,
        weight_q,
        weight_k,
        is_neox,
    )
    _write_reference(key_ref, value_ref, slots, k_ref, v_ref)
    _compare_tensor(q_out, q_ref, "Q")
    _compare_tensor(k_out, k_ref, "K")
    _compare_tensor(v_out, v_ref, "V")
    _compare_tensor(key_cache, key_ref, "key_cache")
    _compare_tensor(value_cache, value_ref, "value_cache")


def test_block_first_bf16_matrix() -> None:
    if _skip_without_cuda():
        return
    print("BF16 block-first cache matrix")
    for layout, tokens, block_size, is_neox in BF16_CASES:
        print(
            f"  layout={layout:11s} tokens={tokens:3d} "
            f"block_size={block_size:2d} is_neox={is_neox}"
        )
        _run_bf16_case(layout, tokens, block_size, is_neox)
        print("    OK")


def test_block_first_partial_rope_rot_dim_64() -> None:
    if _skip_without_cuda():
        return
    tokens, blocks, block_size = 4, 2, 16
    qkv, positions, slots, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens, rotary_dim=64)
    q_ref, k_ref, v_ref = _reference_partial_rope_qkv(
        positions, qkv.clone(), cos_sin, weight_q, weight_k
    )
    key_cache, value_cache = _make_cache_pair("nhd", blocks, block_size, DTYPE)
    key_ref = key_cache.clone(memory_format=torch.preserve_format)
    value_ref = value_cache.clone(memory_format=torch.preserve_format)
    q_out, k_out, v_out = _call_operator(
        positions, qkv, cos_sin, block_size, key_cache, value_cache, slots,
        weight_q, weight_k, True,
    )
    _write_reference(key_ref, value_ref, slots, k_ref, v_ref)
    print("partial RoPE: head_dim=128 rotary_dim=64")
    _compare_tensor(q_out, q_ref, "Q")
    _compare_tensor(k_out, k_ref, "K")
    _compare_tensor(v_out, v_ref, "V")
    _compare_tensor(key_cache, key_ref, "key_cache")
    _compare_tensor(value_cache, value_ref, "value_cache")
    print("    OK")


def _fp8_boundary_values(tokens: int, scale: float) -> torch.Tensor:
    # E4M3FN boundaries and overflow values.  V is not RMSNorm/RoPE'd, so this
    # provides an exact conversion test in addition to random K coverage.
    values = torch.tensor(
        [
            -1000.0,
            -480.0,
            -464.0,
            -449.0,
            -448.0,
            -0.015625,
            -0.001953125,
            0.0,
            0.001953125,
            0.015625,
            1.0,
            448.0,
            449.0,
            464.0,
            480.0,
            1000.0,
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    values = (values * scale).repeat((tokens * KV_HEADS * HEAD_DIM + 9) // 10)
    return values[:tokens * KV_HEADS * HEAD_DIM].reshape(
        tokens, KV_HEADS * HEAD_DIM
    ).to(DTYPE)


def _fp8_e5m2_boundary_values(tokens: int, scale: float) -> torch.Tensor:
    values = torch.tensor(
        [
            -float("inf"),
            -65536.0,
            -61440.0,
            -57344.0,
            -(2.0**-14),
            -(2.0**-16),
            -0.0,
            0.0,
            2.0**-17,
            2.0**-16,
            2.0**-14,
            1.0,
            57344.0,
            61440.0,
            65536.0,
            float("inf"),
            float("nan"),
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    values = (values * scale).repeat(
        (tokens * KV_HEADS * HEAD_DIM + values.numel() - 1) // values.numel()
    )
    return values[:tokens * KV_HEADS * HEAD_DIM].reshape(
        tokens, KV_HEADS * HEAD_DIM
    ).to(DTYPE)


def _run_fp8_case(
    layout: str,
    tokens: int,
    block_size: int,
    cache_dtype: torch.dtype = torch.float8_e4m3fn,
    rotary_dim: int = HEAD_DIM,
) -> None:
    blocks = _blocks_for_tokens(tokens, block_size)
    qkv, positions, slots, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens, rotary_dim=rotary_dim)
    k_scale = torch.tensor([0.5], device=DEVICE, dtype=torch.float32)
    v_scale = torch.tensor([0.25], device=DEVICE, dtype=torch.float32)
    if cache_dtype == torch.float8_e4m3fn:
        qkv[:, Q_HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM:] = _fp8_boundary_values(
            tokens, 0.25
        )
    else:
        qkv[:, Q_HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM:] = (
            _fp8_e5m2_boundary_values(tokens, 0.25)
        )
    is_neox = rotary_dim != HEAD_DIM
    if is_neox:
        q_ref, k_ref, v_ref = _reference_partial_rope_qkv(
            positions, qkv.clone(), cos_sin, weight_q, weight_k
        )
    else:
        q_ref, k_ref, v_ref = _reference_qkv(
            positions, qkv.clone(), cos_sin, weight_q, weight_k, False
        )
    key_cache, value_cache = _make_cache_pair(
        layout, blocks, block_size, cache_dtype
    )
    key_ref = key_cache.clone(memory_format=torch.preserve_format)
    value_ref = value_cache.clone(memory_format=torch.preserve_format)

    q_out, k_out, v_out = _call_operator(
        positions,
        qkv,
        cos_sin,
        block_size,
        key_cache,
        value_cache,
        slots,
        weight_q,
        weight_k,
        is_neox,
        k_scale,
        v_scale,
    )
    _write_reference(
        key_ref,
        value_ref,
        slots,
        k_ref.float() / k_scale,
        v_ref.float() / v_scale,
    )
    _compare_tensor(q_out, q_ref, "Q")
    _compare_tensor(k_out, k_ref, "K")
    _compare_tensor(v_out, v_ref, "V")
    _compare_fp8_bytes(key_cache, key_ref, "key_cache")
    _compare_fp8_bytes(value_cache, value_ref, "value_cache")


def test_block_first_fp8_e4m3_scales_and_boundaries() -> None:
    if _skip_without_cuda():
        return
    print("FP8 E4M3 block-first cache matrix")
    for layout, tokens, block_size in FP8_CASES:
        print(
            f"  layout={layout:11s} tokens={tokens:3d} "
            f"block_size={block_size:2d} k_scale=0.5 v_scale=0.25"
        )
        _run_fp8_case(layout, tokens, block_size)
        print("    OK")


def test_block_first_fp8_e5m2_scales() -> None:
    if _skip_without_cuda():
        return
    print("FP8 E5M2 block-first cache matrix")
    for layout, tokens, block_size in FP8_CASES:
        print(
            f"  layout={layout:11s} tokens={tokens:3d} "
            f"block_size={block_size:2d} k_scale=0.5 v_scale=0.25"
        )
        _run_fp8_case(layout, tokens, block_size, torch.float8_e5m2)
        print("    OK")
    print("  partial RoPE rotary_dim=64")
    _run_fp8_case(
        "nhd", 4, 16, torch.float8_e5m2, rotary_dim=64
    )
    print("    OK")


def test_block_first_compute_only_does_not_require_cache_or_scales() -> None:
    if _skip_without_cuda():
        return
    tokens, block_size = 17, 16
    blocks = _blocks_for_tokens(tokens, block_size)
    qkv, positions, _, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens)
    q_ref, k_ref, v_ref = _reference_qkv(
        positions, qkv.clone(), cos_sin, weight_q, weight_k, True
    )
    print("compute-only: slot_mapping=None")
    q_out, k_out, v_out = _call_operator(
        positions,
        qkv,
        cos_sin,
        block_size,
        None,
        None,
        None,
        weight_q,
        weight_k,
        True,
    )
    _compare_tensor(q_out, q_ref, "Q")
    _compare_tensor(k_out, k_ref, "K")
    _compare_tensor(v_out, v_ref, "V")
    print("    OK")


def test_block_first_validation_errors() -> None:
    if _skip_without_cuda():
        return
    tokens, blocks, block_size = 4, 2, 16
    qkv, positions, slots, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens)
    key_cache, value_cache = _make_cache_pair("nhd", blocks, block_size, DTYPE)
    print("native validation: residual and missing FP8 scale")
    try:
        _call_operator(
            positions,
            qkv,
            cos_sin,
            block_size,
            key_cache,
            value_cache,
            slots,
            weight_q,
            weight_k,
            False,
            residual_q=torch.zeros(
                (tokens, Q_HEADS * HEAD_DIM), device=DEVICE, dtype=DTYPE
            ),
        )
    except RuntimeError as error:
        assert "does not support residual" in str(error)
    else:
        raise AssertionError("residual_q must be rejected")

    fp8_key, fp8_value = _make_cache_pair(
        "nhd", blocks, block_size, torch.float8_e4m3fn
    )
    try:
        _call_operator(
            positions,
            qkv,
            cos_sin,
            block_size,
            fp8_key,
            fp8_value,
            slots,
            weight_q,
            weight_k,
            False,
        )
    except RuntimeError as error:
        assert "requires k_scale and v_scale" in str(error)
    else:
        raise AssertionError("FP8 cache without scales must be rejected")
    print("    OK")


def test_block_first_cuda_graph_replay_uses_device_slots_and_scales() -> None:
    if _skip_without_cuda() or not hasattr(torch.cuda, "CUDAGraph"):
        return
    tokens, blocks, block_size = 7, 3, 16
    qkv, positions, initial_slots, weight_q, weight_k = _build_inputs(
        tokens, blocks, block_size
    )
    cos_sin = _build_cos_sin_cache(tokens)
    old_k_scale, old_v_scale = 0.5, 0.25
    new_k_scale, new_v_scale = 0.25, 0.5
    k_scale = torch.tensor([old_k_scale], device=DEVICE, dtype=torch.float32)
    v_scale = torch.tensor([old_v_scale], device=DEVICE, dtype=torch.float32)
    key_cache, value_cache = _make_cache_pair(
        "padded", blocks, block_size, torch.float8_e4m3fn
    )
    q_ref, k_ref, v_ref = _reference_qkv(
        positions, qkv.clone(), cos_sin, weight_q, weight_k, False
    )

    # Warm-up performs one normal launch.  Reset cache contents so that only
    # capture and replay writes contribute to the byte-for-byte reference.
    _call_operator(
        positions,
        qkv,
        cos_sin,
        block_size,
        key_cache,
        value_cache,
        initial_slots,
        weight_q,
        weight_k,
        False,
        k_scale,
        v_scale,
    )
    key_cache.fill_(SENTINEL)
    value_cache.fill_(SENTINEL)
    expected_key = key_cache.clone(memory_format=torch.preserve_format)
    expected_value = value_cache.clone(memory_format=torch.preserve_format)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call_operator(
            positions,
            qkv,
            cos_sin,
            block_size,
            key_cache,
            value_cache,
            initial_slots,
            weight_q,
            weight_k,
            False,
            k_scale,
            v_scale,
        )

    replay_slots = torch.tensor(
        [1, block_size - 2, block_size + 2, block_size + 3,
         blocks * block_size - 2, -1, blocks * block_size],
        device=DEVICE,
        dtype=torch.int64,
    )
    initial_slots.copy_(replay_slots)
    k_scale.fill_(new_k_scale)
    v_scale.fill_(new_v_scale)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    # Stream capture records the launch; it is not a cache write.  Both
    # replays use the in-place updated slot/scale tensors, so one reference
    # scatter with the new values represents their final identical state.
    _write_reference(
        expected_key,
        expected_value,
        replay_slots,
        k_ref.float() / new_k_scale,
        v_ref.float() / new_v_scale,
    )
    print("CUDA Graph replay: in-place slot and scale updates")
    _compare_tensor(captured[0], q_ref, "Q")
    _compare_tensor(captured[1], k_ref, "K")
    _compare_tensor(captured[2], v_ref, "V")
    _compare_fp8_bytes(key_cache, expected_key, "key_cache")
    _compare_fp8_bytes(value_cache, expected_value, "value_cache")
    print("    OK")


def main() -> None:
    if _skip_without_cuda():
        return
    print(
        "split_qkv_rms_rotary_embedding_fuse_with_kv_store_block_first "
        "vs rms_rotary_embedding_fuse + block-first scatter"
    )
    test_block_first_bf16_matrix()
    test_block_first_partial_rope_rot_dim_64()
    test_block_first_fp8_e4m3_scales_and_boundaries()
    test_block_first_fp8_e5m2_scales()
    test_block_first_compute_only_does_not_require_cache_or_scales()
    test_block_first_validation_errors()
    test_block_first_cuda_graph_replay_uses_device_slots_and_scales()
    print("All block-first split QKV tests passed.")


if __name__ == "__main__":
    main()
