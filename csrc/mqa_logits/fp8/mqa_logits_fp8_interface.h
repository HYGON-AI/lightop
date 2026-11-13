// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <torch/extension.h>

namespace at {
namespace native {
torch::Tensor mqa_logits_fp8(const torch::Tensor &Q, const torch::Tensor &K,
                             const torch::Tensor &weights, const torch::Tensor &ks,
                             const torch::Tensor &ke, int m, int n, int h, int d,
                             const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                             const torch::optional<torch::Tensor> &out);
}
} // namespace at
