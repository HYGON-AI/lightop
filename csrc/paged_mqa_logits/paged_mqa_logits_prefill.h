#pragma once

#include "paged_mqa_logits.h"

namespace lightop::deep_gemm {

torch::Tensor paged_mqa_logits_prefill(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, const std::optional<torch::Tensor>& schedule_meta,
    int64_t max_context_len, bool clean_logits);

}  // namespace lightop::deep_gemm
