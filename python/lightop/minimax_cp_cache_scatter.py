"""Fuse rank-major CP row reorder and exact-byte K/V/index-K cache writes."""

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def _native():
    from . import op

    if hasattr(op, "minimax_cp_cache_scatter"):
        return op
    return importlib.import_module("._minimax_cp_cache_scatter", __package__)


def minimax_cp_cache_scatter(raw, reorder, slots, k_cache, v_cache, index_k_cache):
    _native().minimax_cp_cache_scatter(
        raw, reorder, slots, k_cache, v_cache, index_k_cache
    )
