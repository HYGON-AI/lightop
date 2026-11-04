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
