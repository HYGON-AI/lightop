# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache and sequence-metadata APIs."""

from ..fused_metadata import fused_metadata_kernel_general
from ..gather_and_maybe_dequant_cache import gather_and_maybe_dequant_cache
from ..kvcacheio import hcu_alloc_extend_kernel
from ..reshape_and_cache_cuda import reshape_and_cache_cuda
from .dequantize_k_cache_paged import (
    dequantize_k_cache_paged_hcu_counted,
    dsv4_dequantize_k_cache_paged_out,
)
from .topk_workspace import (
    dsv4_dequantize_sparse_topk_out,
    dsv4_pack_c4_topk_workspace,
    dsv4_pack_sparse_topk_workspace,
)
from .native import (
    fuse_act_quant_and_store_index_k_cache,
    fuse_qk_quant_and_store_index_k_cache,
    fused_quantize_and_store_mla_kv_cache,
    quantize_nope_fp8_rope_bf16_pack_store,
    fused_concat_and_store_mla_kv_cache,
)

__all__ = [
    "dsv4_pack_c4_topk_workspace",
    "dsv4_pack_sparse_topk_workspace",
    "dsv4_dequantize_sparse_topk_out",
    "dsv4_dequantize_k_cache_paged_out",
    "dequantize_k_cache_paged_hcu_counted",
    "hcu_alloc_extend_kernel",
    "fuse_act_quant_and_store_index_k_cache",
    "fuse_qk_quant_and_store_index_k_cache",
    "fused_metadata_kernel_general",
    "fused_quantize_and_store_mla_kv_cache",
    "quantize_nope_fp8_rope_bf16_pack_store",
    "reshape_and_cache_cuda",
    "fused_concat_and_store_mla_kv_cache",
    "gather_and_maybe_dequant_cache",
]
