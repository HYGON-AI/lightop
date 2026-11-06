// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace lightop::dsv4 {
std::tuple<at::Tensor, at::Tensor, at::Tensor> build_cp_query_metadata(
    const at::Tensor& extend, const at::Tensor& starts,
    const at::Tensor& positions, int64_t cp_size, int64_t cp_rank);

std::tuple<at::Tensor, at::Tensor> pack_compact_token_ids(
    const at::Tensor& source, const at::Tensor& seq_lens,
    int64_t ratio, int64_t page_size, int64_t total,
    const c10::optional<at::Tensor>& out_ids,
    const c10::optional<at::Tensor>& out_base);
}  // namespace lightop::dsv4

namespace lightop::dsv41 {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_swa_token_ids(
    const at::Tensor& seq, const at::Tensor& extend, const at::Tensor& req,
    const at::Tensor& table, const at::Tensor& mapping, int64_t window, int64_t total,
    const c10::optional<at::Tensor>& query_lens,
    const c10::optional<at::Tensor>& query_starts);
}

namespace at::native {
void combine_topk_swa_indices_hcu(
    const at::Tensor& indices, const at::Tensor& starts,
    const c10::optional<at::Tensor>& positions, const at::Tensor& seq_lens,
    const at::Tensor& gather_lens, const at::Tensor& compressed_base,
    const at::Tensor& swa_base, at::Tensor& output, at::Tensor& lengths,
    int64_t window, int64_t ratio, int64_t topk, bool initialize_padding);
}
