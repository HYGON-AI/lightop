// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <torch/extension.h>

namespace lightop::deep_gemm {
torch::Tensor paged_mqa_logits_fp4(
    const torch::Tensor& q, const torch::Tensor& weights,
    const torch::Tensor& slots, const torch::Tensor& lens,
    const torch::Tensor& table, int64_t page_size,
    std::optional<torch::Tensor> out = std::nullopt);
}
