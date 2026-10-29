"""Exact FP8 K/V and BF16 index-K CP transport packet (gfx938)."""

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def _native():
    from . import op

    if hasattr(op, "minimax_cp_fp8_pack"):
        return op
    return importlib.import_module("._minimax_cp_fp8_pack", __package__)


def minimax_cp_fp8_pack(k, v, index_k, k_scale=None, v_scale=None):
    return _native().minimax_cp_fp8_pack(
        k,
        v,
        index_k,
        1.0 if k_scale is None else k_scale,
        1.0 if v_scale is None else v_scale,
        k_scale is not None,
        v_scale is not None,
    )
