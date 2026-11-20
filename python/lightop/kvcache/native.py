# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for native KV-cache operators."""

from .. import op as lightop_native

fuse_act_quant_and_store_index_k_cache = (
    lightop_native.fuse_act_quant_and_store_index_k_cache
)
fuse_qk_quant_and_store_index_k_cache = (
    lightop_native.fuse_qk_quant_and_store_index_k_cache
)
fused_quantize_and_store_mla_kv_cache = (
    lightop_native.fused_quantize_and_store_mla_kv_cache
)
quantize_nope_fp8_rope_bf16_pack_store = (
    lightop_native.quantize_nope_fp8_rope_bf16_pack_store
)
fused_concat_and_store_mla_kv_cache = (
    lightop_native.fused_concat_and_store_mla_kv_cache
)

__all__ = [
    "fuse_act_quant_and_store_index_k_cache",
    "fuse_qk_quant_and_store_index_k_cache",
    "fused_quantize_and_store_mla_kv_cache",
    "quantize_nope_fp8_rope_bf16_pack_store",
    "fused_concat_and_store_mla_kv_cache",
]

if hasattr(lightop_native, "fused_quantize_and_store_mla_kv_cache_dual"):
    fused_quantize_and_store_mla_kv_cache_dual = lightop_native.fused_quantize_and_store_mla_kv_cache_dual
    __all__.append("fused_quantize_and_store_mla_kv_cache_dual")

if hasattr(lightop_native, "main_kv_loc_to_compact"):
    main_kv_loc_to_compact = lightop_native.main_kv_loc_to_compact
    __all__.append("main_kv_loc_to_compact")

for _name in ('indexer_k_norm_rope_hadamard_store', 'fill_swa_ring_mapping'):
    if hasattr(lightop_native, _name):
        globals()[_name] = getattr(lightop_native, _name)
        __all__.append(_name)

def dsv4_gather_upconvert_compact_k_cache_paged(
    swa_cache, swa_indices, swa_lengths, extra_cache, extra_indices,
    extra_lengths, out, compact_indices, extra_page_size, *, reuse_group=1,
):
    """Gather packed DPSKV4 KV into one BF16 cache, writing compact indices.

    Caches are UINT8 [pages,page_size,1,584], with compact inner strides.
    SWA page_size is 256. Indices are contiguous INT32 [Q,K], lengths INT32
    [Q], out is contiguous BF16 [Q,Ka+Kb,1,512], compact_indices INT32
    [Q,1,Ka+Kb]. Outputs must not alias inputs. Valid lengths are bounded
    by their corresponding K; negative/out-of-range indices are masked.
    Extra page size is 2/64/128/256. reuse_group 5/6 requires complete query
    groups; group 6 requires at least five SWA columns. Sharing only occurs after comparing the actual valid KV indices.
    Empty extra inputs permit the single-cache draft reuse path. No hidden
    allocations, synchronization or compilation are performed.
    """
    if reuse_group == 1:
        return lightop_native.dsv4_gather_upconvert_combined_k_cache_paged(
            swa_cache, swa_indices, swa_lengths, extra_cache, extra_indices,
            extra_lengths, out, compact_indices, extra_page_size)
    if reuse_group not in (5, 6):
        raise ValueError("reuse_group must be 1, 5 or 6")
    return lightop_native.dsv4_gather_upconvert_query_reuse_k_cache_paged(
        swa_cache, swa_indices, swa_lengths, extra_cache, extra_indices,
        extra_lengths, out, compact_indices, extra_page_size, reuse_group)


__all__.append("dsv4_gather_upconvert_compact_k_cache_paged")
