// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <optional>
#include <tuple>
#include <cstdint>
#include <vector>
#include <torch/extension.h>

namespace lightop::fp4_mqa {

torch::Tensor fp8_fp4_mqa_logits(
    const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::tuple<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    bool clean_logits,
    int64_t max_seqlen_k,
    const std::optional<std::vector<torch::Tensor>>& candidate_blocks,
    const std::optional<std::vector<int64_t>>& candidate_query_lens);

}  // namespace lightop::fp4_mqa
