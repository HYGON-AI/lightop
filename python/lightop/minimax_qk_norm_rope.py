"""MiniMax-M3.1 BF16 Q/K/index Gemma RMSNorm + RoPE HIP fast path."""



def _native():
    from . import op

    return op if callable(getattr(op, "minimax_qk_norm_rope", None)) else None


def supports_minimax_qk_norm_rope():
    return _native() is not None


def minimax_qk_norm_rope(
    q,
    k,
    idx_q,
    idx_k,
    q_weight,
    k_weight,
    idx_q_weight,
    idx_k_weight,
    positions,
    cos_sin_cache,
    eps,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    if head_dim != 128:
        raise ValueError("MiniMax HIP fast path requires head_dim=128")
    native = _native()
    if native is None:
        raise RuntimeError("LightOp QK HIP native extension is not built")
    # Native dispatch checks vector alignment/strides and retains the
    # generic fallback. Preserve the previous rotary128 selection.
    variant = 6 if rotary_dim == 64 else 5
    return tuple(
        native.minimax_qk_norm_rope(
            q,
            k,
            idx_q,
            idx_k,
            q_weight,
            k_weight,
            idx_q_weight,
            idx_k_weight,
            positions,
            cos_sin_cache,
            eps,
            rotary_dim,
            is_neox_style,
            variant,
        )
    )


def minimax_qk_norm_rope_fp8_cache(
    qkv, weights, positions, cos_sin_cache, eps, locations, main_mapping,
    key_cache, value_cache, index_cache, *, heads_per_block=None,
):
    """Fused rotary64 QK/index norm with unit-scale E4M3 queries and KV stores.

    QKV accepts aligned row-major storage or dense transposed storage with
    a multiple of 16 rows and heads_per_block=16.

    Locations are logical for index K and translated through main_mapping for
    main K/V. Live locations must be distinct within one launch. Position IDs
    must address cos_sin_cache, and all tensors must be on the same device.
    """
    from . import op

    args = (qkv, weights, positions, cos_sin_cache, eps, locations, main_mapping,
            key_cache, value_cache, index_cache)
    if heads_per_block is None:
        return tuple(op.minimax_qk_norm_rope_fp8_cache(*args))
    return tuple(op.minimax_qk_norm_rope_fp8_cache(*args, heads_per_block))
