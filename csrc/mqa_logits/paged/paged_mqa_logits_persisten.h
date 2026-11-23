// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace lightop::deep_gemm {

// Dense logits, with -inf outside each row's valid context. Group sizes 1..5
// permit partial tails and independent page tables; identical pages reuse KV.
torch::Tensor paged_mqa_logits_persisten(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size);

torch::Tensor paged_mqa_topk_sampled_int8(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& weights, const torch::Tensor& lengths,
    const torch::Tensor& table, const torch::Tensor& pages,
    const torch::Tensor& cu, int64_t width);

}  // namespace lightop::deep_gemm
