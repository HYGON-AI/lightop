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

__all__ = [
    "fuse_act_quant_and_store_index_k_cache",
    "fuse_qk_quant_and_store_index_k_cache",
    "fused_quantize_and_store_mla_kv_cache",
    "quantize_nope_fp8_rope_bf16_pack_store",
]
