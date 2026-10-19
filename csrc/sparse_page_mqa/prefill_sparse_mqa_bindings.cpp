#include <torch/extension.h>
#include "paged_sparse_mqa_logits.h"
#include "paged_sparse_mqa_logits_metadata.h"
namespace py=pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.attr("PAGED_SPARSE_MQA_BUILD_ID")=PREFILL_SPARSE_MQA_BUILD_ID;
  m.attr("PAGED_SPARSE_MQA_CONSUMER_SHA256")=PREFILL_SPARSE_MQA_CONSUMER_SHA256;
  m.attr("PAGED_SPARSE_MQA_LOGITS_ABI")=3;
  m.attr("PAGED_SPARSE_MQA_METADATA_FORMAT")="deepgemm_paged_sparse_v1";
  m.def("fp8_fp4_paged_sparse_mqa_logits",&lightop::deep_gemm::fp8_fp4_paged_sparse_mqa_logits,
    py::arg("q"),py::arg("q_scales"),py::arg("fused_kv_cache"),py::arg("weights"),
    py::arg("metadata"),py::arg("num_max_sparse_blocks"),py::arg("sparse_block_kv"),
    py::arg("context_lens")=std::nullopt,py::arg("sparse_kv_block_indices")=std::nullopt,
    py::arg("output")=std::nullopt,py::arg("num_sms")=std::nullopt,py::arg("is_prefill")=false);
  m.def("prefill_sparse_fp8_fp4_paged_mqa_logits",&lightop::deep_gemm::prefill_sparse_fp8_fp4_paged_mqa_logits,
    py::arg("q"),py::arg("scales"),py::arg("kv"),py::arg("weights"),py::arg("lens"),py::arg("table"),
    py::arg("indices"),py::arg("candidates"),py::arg("capacity"),py::arg("block"),py::arg("num_sms")=std::nullopt,
    py::arg("paired_rows")=false,py::arg("threads")=0);
  m.def("get_paged_sparse_mqa_logits_metadata",&lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata,
    py::arg("context_lens"),py::arg("block_table"),py::arg("indices"),py::arg("page_kv"),
    py::arg("sparse_kv_block_indices"),py::arg("qk_dtype"),py::arg("sparse_block_kv"),py::arg("num_sms")=std::nullopt);
  m.def("get_paged_sparse_mqa_logits_metadata_bytes",&lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata_bytes,
    py::arg("num_q_tokens"),py::arg("num_max_sparse_blocks"),py::arg("sparse_block_kv"),py::arg("split_kv"),py::arg("num_sms"));
  m.attr("sparse_fp8_fp4_paged_mqa_logits")=m.attr("fp8_fp4_paged_sparse_mqa_logits");
}
