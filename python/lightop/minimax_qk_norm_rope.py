"""MiniMax-M3.1 BF16 Q/K/index Gemma RMSNorm + RoPE HIP fast path."""

import importlib
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _native():
    from . import op

    if hasattr(op, "minimax_qk_norm_rope"):
        return op
    try:
        return importlib.import_module("._minimax_qk_rope", __package__)
    except ImportError:
        return None


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
    variant = int(os.environ.get("LIGHTOP_MINIMAX_QK_VARIANT", "5"))
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
