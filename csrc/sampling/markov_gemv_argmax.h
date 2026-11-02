// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/types.h>
#include <tuple>

namespace lightop::sampling {

std::tuple<torch::Tensor, torch::Tensor> markov_gemv_add_argmax(
    const torch::Tensor& latent,
    const torch::Tensor& weight,
    const torch::Tensor& base_logits);

}  // namespace lightop::sampling
