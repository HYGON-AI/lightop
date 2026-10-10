// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <torch/extension.h>

namespace lightop::mqa_beta {
// Called by the existing native entry after its Q/K dtype check. A disengaged
// result keeps the original implementation; an enabled beta path returns its
// result or raises its own validation error.
torch::optional<torch::Tensor> try_dispatch(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
    const torch::optional<torch::Tensor> &out);
} // namespace lightop::mqa_beta

namespace lightop::mqa_int8 {
torch::Tensor run(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
    const torch::optional<torch::Tensor> &out);
} // namespace lightop::mqa_int8
