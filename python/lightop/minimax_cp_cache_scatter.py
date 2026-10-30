"""Fuse rank-major CP row reorder and exact-byte K/V/index-K cache writes."""



def _native():
    from . import op

    if not callable(getattr(op, "minimax_cp_cache_scatter", None)):
        raise RuntimeError("LightOp native extension is missing minimax_cp_cache_scatter; rebuild LightOp")
    return op


def minimax_cp_cache_scatter(raw, reorder, slots, k_cache, v_cache, index_k_cache):
    _native().minimax_cp_cache_scatter(
        raw, reorder, slots, k_cache, v_cache, index_k_cache
    )
