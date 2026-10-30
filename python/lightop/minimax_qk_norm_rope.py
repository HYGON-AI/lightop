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
