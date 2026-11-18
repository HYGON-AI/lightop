// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Phase-1 pilot: split the "MQA logits" family bindings into a dedicated
// extension module (lightop._ext.op_mqa_logits), mirroring AITER's "one
// operator backend = one module_xxx.so" layout and following the Phase-0
// norm pilot (csrc/kernels/rmsnorm/binding_norm.cpp).
//
// Scope (all definitions live inside the MQA family directories of this tree):
//   csrc/mqa_logits/                  (incl. bf16/, int8/, fp8/)
//   csrc/mqa_logits/fp8_fp4/
//   csrc/mqa_logits/paged/
//   csrc/mqa_logits/sparse/
//   csrc/kernels/index_k_int8_dequant/
//
// The m.def / m.attr blocks below are copied verbatim from
// csrc/pybind/export.cpp (main-module binding region, plus bind_paged_sparse_mqa
// and bind_prefill_sparse_fusion). Symbols whose definitions live in root
// csrc/kernels/dsv41_*.cu (dsv41_build_swa_token_ids,
// dsv41_remap_indices_to_staging, dsv41_ragged_topk_postprocess,
// dsv41_pair_partners) are intentionally NOT migrated: they belong to the
// dsv4 prefill family and stay bound by the monolith.
//
// Dual-track pilot: while LIGHTOP_ENABLE_MQA_MODULE=1 the monolith
// (lightop.op_full) still compiles these sources and keeps its bindings;
// python/lightop/op.py routes MQA symbols to this module first. After the
// pilot is verified, Step-2 removes the MQA globs from the monolith and
// guards the corresponding export.cpp region.

#include <torch/extension.h>

#include <optional>
#include <vector>

// Same family headers as csrc/pybind/export.cpp's main branch. Paths are
// relative to this file's directory (csrc/mqa_logits/), matching the style of
// export.cpp; the 1118 tree regrouped the family under mqa_logits/{fp8_fp4,
// paged,sparse}/ and moved index_k_int8_dequant under csrc/kernels/.
#include "fp8_fp4/fp8_fp4_mqa_logits.h"
#include "fp8_fp4/dsv41_mqa_prep.h"
#include "paged/paged_mqa_logits.h"
#include "paged/paged_mqa_logits_persisten.h"
#include "paged/paged_mqa_logits_fp4.h"
#include "sparse/sparse_page_mqa.h"
#include "sparse/paged_sparse_mqa_logits_metadata.h"
#include "sparse/paged_sparse_mqa_logits.h"
#include "sparse/prefill_sparse_sort.h"
#include "sparse/prefill_sparse_fusion.h"
#include "../kernels/index_k_int8_dequant/index_k_int8_dequant.h"

namespace at::native {
// Declaration mirrored verbatim from csrc/pybind/export.cpp; defined in
// csrc/mqa_logits/mqa_logits_asm.cu (dispatch stays inside the family).
torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor> &KV_scale =  torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor> &D_out = torch::nullopt);
}  // namespace at::native

namespace {
// Copied verbatim from csrc/pybind/export.cpp (bind_paged_sparse_mqa).
void bind_paged_sparse_mqa(pybind11::module_& m) {
  namespace py = pybind11;
#ifdef PREFILL_SPARSE_MQA_BUILD_ID
  m.attr("PAGED_SPARSE_MQA_BUILD_ID") = PREFILL_SPARSE_MQA_BUILD_ID;
#endif
#ifdef PREFILL_SPARSE_MQA_CONSUMER_SHA256
  m.attr("PAGED_SPARSE_MQA_CONSUMER_SHA256") = PREFILL_SPARSE_MQA_CONSUMER_SHA256;
#endif
  m.attr("PAGED_SPARSE_MQA_LOGITS_ABI") = 3;
  m.attr("PAGED_SPARSE_MQA_METADATA_FORMAT") = "deepgemm_paged_sparse_v1";
  m.def("fp8_fp4_paged_sparse_mqa_logits",
        &lightop::deep_gemm::fp8_fp4_paged_sparse_mqa_logits,
        "MXFP4/MXFP8 paged sparse logits, FP32 weights/output, Prefill/Decode dispatch",
        py::arg("q"), py::arg("q_scales"), py::arg("fused_kv_cache"),
        py::arg("weights"), py::arg("metadata"), py::arg("num_max_sparse_blocks"),
        py::arg("sparse_block_kv"), py::arg("context_lens") = std::nullopt,
        py::arg("sparse_kv_block_indices") = std::nullopt,
        py::arg("output") = std::nullopt, py::arg("num_sms") = std::nullopt,
        py::arg("is_prefill") = false);
  m.attr("sparse_fp8_fp4_paged_mqa_logits") = m.attr("fp8_fp4_paged_sparse_mqa_logits");
  m.def("get_paged_sparse_mqa_logits_metadata",
        &lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata,
        "DeepGEMM get_paged_sparse_mqa_logits_metadata: merged sparse KV splits and persistent schedule",
        py::arg("context_lens"), py::arg("block_table"), py::arg("indices"),
        py::arg("page_kv"), py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"),
        py::arg("sparse_block_kv"), py::arg("num_sms") = std::nullopt);
  m.def("get_paged_sparse_mqa_logits_metadata_bytes",
        &lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata_bytes,
        "Size of get_paged_sparse_mqa_logits_metadata's output",
        py::arg("num_q_tokens"), py::arg("num_max_sparse_blocks"),
        py::arg("sparse_block_kv"), py::arg("split_kv"), py::arg("num_sms"));
    m.def("prefill_sparse_fp8_fp4_paged_mqa_logits", &lightop::deep_gemm::prefill_sparse_fp8_fp4_paged_mqa_logits,
        py::arg("q"),py::arg("scales"),py::arg("kv"),py::arg("weights"),py::arg("lens"),py::arg("table"),
        py::arg("indices"),py::arg("candidates"),py::arg("capacity"),py::arg("block"),py::arg("num_sms")=std::nullopt,
        py::arg("paired_rows")=false,py::arg("threads")=0);
}

// Copied verbatim from csrc/pybind/export.cpp (bind_prefill_sparse_fusion).
// The monolith calls this at export.cpp:1994; definitions live in
// csrc/sparse_page_mqa/prefill_sparse_fusion.cu and prefill_sparse_sort.cu.
void bind_prefill_sparse_fusion(pybind11::module_& m) {
#ifdef PREFILL_FUSION_BUILD_ID
  m.attr("PREFILL_SPARSE_FUSION_BUILD_ID") = PREFILL_FUSION_BUILD_ID;
#endif
  m.attr("PREFILL_SPARSE_SORT_ABI") = 1;
  m.attr("PREFILL_SPARSE_FUSION_ABI")=1;
  m.def("prefill_sparse_native_prepare_query", &prefill_sparse_native_prepare_query);
  m.def("prefill_sparse_native_prepare_candidates", &prefill_sparse_native_prepare_candidates);
  m.def("prefill_sparse_native_finalize", &prefill_sparse_native_finalize);
  m.def("prefill_sparse_sort_candidates", &prefill_sparse_sort_candidates);
  m.def("prefill_sparse_finalize_indices", &prefill_sparse_finalize_indices);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp MQA logits operators (Phase-1 module split pilot)";
  // Marker proving the aggregator (lightop/op.py) routed to this module.
  m.attr("LIGHTOP_MQA_MODULE") = 1;

  // --- verbatim from csrc/pybind/export.cpp ---
  m.def("mqa_logits", &at::native::mqa_logits, "mqa_logits",
      py::arg("Q"),
      py::arg("K"),
      py::arg("Weights"),
      py::arg("KS"),
      py::arg("KE"),
      py::arg("seq_len"),
      py::arg("kv_seq_len"),
      py::arg("num_heads"),
      py::arg("head_dim"),
      py::arg("KV_scale") = torch::nullopt, 
      py::arg("clean_logits") = true,
      py::arg("D_out") = torch::nullopt);
  // --- verbatim from csrc/pybind/export.cpp ---
  m.def("get_paged_mqa_logits_metadata", &lightop::deep_gemm::get_paged_mqa_logits_metadata,"get_paged_mqa_logits_metadata");
  m.def("dsv41_mqa_query_lens", &lightop::fp4_mqa::dsv41_mqa_query_lens,
        py::arg("extend_start_loc"), py::arg("extend_seq_lens"),
        py::arg("cp_size"), py::arg("cp_rank"));
  m.def("dsv41_mqa_prefill_metadata", &lightop::fp4_mqa::dsv41_mqa_prefill_metadata,
        py::arg("seq_lens"), py::arg("q_lens"), py::arg("positions"),
        py::arg("req_pool_indices"), py::arg("req_to_token"), py::arg("ratio"),
        py::arg("num_k_slots"), py::arg("max_request_work"));
  m.def("dsv41_gather_index_k_fp4", &lightop::fp4_mqa::dsv41_gather_index_k_fp4,
        py::arg("cache"), py::arg("slots"), py::arg("page_size"));
  m.def("fp8_fp4_mqa_logits", &lightop::fp4_mqa::fp8_fp4_mqa_logits,
        py::arg("q"), py::arg("kv"), py::arg("weights"),
        py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
        py::arg("clean_logits"), py::arg("max_seqlen_k"),
        py::arg("candidate_blocks") = std::nullopt,
        py::arg("candidate_query_lens") = std::nullopt);
  m.attr("_fp8_fp4_mqa_logits_supports_candidates") = true;
  m.def("paged_mqa_logits", &lightop::deep_gemm::paged_mqa_logits,"paged_mqa_logits");
  m.def("paged_mqa_logits_persisten", &lightop::deep_gemm::paged_mqa_logits_persisten,
        "dense persistent paged MQA logits with per-request KV reuse",
        py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
        py::arg("max_context_len"), py::arg("clean_logits") = true,
        py::arg("num_warps") = 4, py::arg("persistent_ctas") = 0,
        py::arg("group_size") = 1);
  m.def("paged_mqa_logits_fp4", &lightop::deep_gemm::paged_mqa_logits_fp4,
        "DeepGEMM fp8_fp4_paged_mqa_logits (clean_logits=False) on the fused FP4 index-K pool",
        py::arg("q"), py::arg("q_sf"), py::arg("kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
        py::arg("max_context_len"), py::arg("table_block_size") = 64,
        py::arg("out") = std::nullopt, py::arg("rows_per_request") = 1);
  m.attr("PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS") = lightop::deep_gemm::kPagedMqaLogitsFp4PairMinRequests;
  m.attr("PAGED_MQA_LOGITS_FP4_MAX_SCHEDULE_ROWS") = lightop::deep_gemm::kPagedMqaLogitsFp4MaxScheduleRows;
  m.def("get_paged_mqa_logits_fp4_metadata",
        &lightop::deep_gemm::get_paged_mqa_logits_fp4_metadata,
        "FP4 paged MQA split-KV schedule planner",
        py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"),
        py::arg("rows_per_request") = 1);
  // --- verbatim from csrc/pybind/export.cpp (sparse paged MQA group) ---
  bind_paged_sparse_mqa(m);
  // --- verbatim from csrc/pybind/export.cpp (defined in paged_mqa_logits/) ---
  m.def("dsv41_candidate_block_table", &lightop::deep_gemm::dsv41_candidate_block_table,
        "DSV4.1 candidate indexer level one: top blocks of 8 by block max as pool slots / 8",
        py::arg("logits"), py::arg("seq_lens"), py::arg("page_table"), py::arg("page_size"),
        py::arg("block_size"), py::arg("out_phys_blocks"), py::arg("out_valid_lens"));
  m.def("dsv41_select_candidate_blocks", &lightop::deep_gemm::dsv41_select_candidate_blocks,
        "DSV4.1 candidate block selection and dense mask expansion",
        py::arg("logits"), py::arg("seq_lens"), py::arg("block_size"),
        py::arg("out_mask"), py::arg("out_block_ids"));
  // NOTE: export.cpp:1573-1582 (dsv41_build_swa_token_ids,
  // dsv41_remap_indices_to_staging, dsv41_ragged_topk_postprocess,
  // dsv41_pair_partners) are defined in root csrc/dsv41_*.cu /
  // dsv4_topk_workspace.cu and intentionally stay in the monolith.
  // --- verbatim from csrc/pybind/export.cpp ---
  m.def(
      "dequantize_index_k_int8_paged",
      &lightop::deep_gemm::dequantize_index_k_int8_paged,
      "dequantize_index_k_int8_paged",
      py::arg("packed_cache"),
      py::arg("block_tables"),
      py::arg("context_lens"),
      py::arg("workspace"),
      py::arg("page_claims"),
      py::arg("waves_per_page") = 1);
  m.def(
      "page_mqa_logits_sparse_mask",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask,
      "page_mqa_logits_sparse_mask",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0);
  m.def(
      "page_mqa_logits_sparse_mask_grouped",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped,
      "page_mqa_logits_sparse_mask_grouped",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
  m.def(
      "page_mqa_logits_sparse_mask_grouped_int8",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped_int8,
      "page_mqa_logits_sparse_mask_grouped_int8",
      py::arg("q"),
      py::arg("packed_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
  // --- verbatim from csrc/pybind/export.cpp (prefill sparse fusion group) ---
  bind_prefill_sparse_fusion(m);
}