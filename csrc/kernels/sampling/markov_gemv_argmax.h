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


void dsv41_markov_fp8_partials(
    const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& embedding_scale,
    const torch::Tensor& weight, const torch::Tensor& weight_scale,
    int64_t vocab_start, int64_t vocab_size, torch::Tensor out, int64_t tile_rows = 128);
void dsv41_markov_fp8_partials(
    const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& embedding_scale,
    const torch::Tensor& weight, const torch::Tensor& weight_scale,
    int64_t vocab_start, int64_t vocab_size, torch::Tensor out, int64_t tile_rows,
    bool packed_argmax);
void dsv41_markov_fp8_partials(
    const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& embedding_scale,
    const torch::Tensor& weight, const torch::Tensor& weight_scale,
    int64_t vocab_start, int64_t vocab_size, torch::Tensor out, int64_t tile_rows,
    bool packed_argmax, int64_t k_stage);
void dsv41_markov_compact_partials(const torch::Tensor& input, torch::Tensor out);
void dsv41_markov_merge_partials(
    const torch::Tensor& input, int64_t rows, int64_t world_size, torch::Tensor out);
void dsv41_markov_merge_partials(
    const torch::Tensor& input, int64_t rows, int64_t world_size, torch::Tensor out,
    bool packed_argmax);

}  // namespace lightop::sampling
