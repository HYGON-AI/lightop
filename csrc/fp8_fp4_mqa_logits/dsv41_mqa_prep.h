// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <cstdint>
#include <tuple>
#include <torch/extension.h>

namespace lightop::fp4_mqa {

torch::Tensor dsv41_mqa_query_lens(
    const torch::Tensor& extend_start_loc,
    const torch::Tensor& extend_seq_lens,
    int64_t cp_size,
    int64_t cp_rank);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dsv41_mqa_prefill_metadata(
    const torch::Tensor& seq_lens,
    const torch::Tensor& q_lens,
    const torch::Tensor& positions,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& req_to_token,
    int64_t ratio,
    int64_t num_k_slots,
    int64_t max_request_work);

std::tuple<torch::Tensor, torch::Tensor> dsv41_gather_index_k_fp4(
    const torch::Tensor& cache,
    const torch::Tensor& slots,
    int64_t page_size);

}  // namespace lightop::fp4_mqa
