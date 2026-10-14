// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <torch/extension.h>

namespace lightop::deep_gemm {

/// \brief DeepGEMM get_paged_sparse_mqa_logits_metadata on HCU, byte-compatible metadata.
/// \param context_lens [Q] int32 visible positions per query token
/// \param block_table [Q,pages] int32 with unit column stride; paired queries share a row
/// \param indices [Q] int32 request id per query token; a request's tokens are consecutive
/// \param page_kv positions per block_table entry, a multiple of sparse_block_kv
/// \param sparse_kv_block_indices [Q,M] int32 logical block ids, M % 4 == 0, M <= 4096; each
///        row starts with min(M, ceil(len / sparse_block_kv)) unique ascending blocks
/// \param qk_dtype packed FP4 (int8/uint8, split 640) or FP8 E4M3 (split 512)
/// \param sparse_block_kv 8 or 16
/// \param num_sms persistent CTAs of the consumer (schedule wave width), default the CU count
/// \return uint8 metadata: MetadataHeader, KV splits, then num_waves * num_sms ScheduleEntry
torch::Tensor get_paged_sparse_mqa_logits_metadata(
    const torch::Tensor& context_lens, const torch::Tensor& block_table, const torch::Tensor& indices,
    int64_t page_kv, const torch::Tensor& sparse_kv_block_indices, at::ScalarType qk_dtype,
    int64_t sparse_block_kv, std::optional<int64_t> num_sms = std::nullopt);

/// Size in bytes of the metadata tensor returned above (DeepGEMM get_num_metadata_bytes, paged).
int64_t get_paged_sparse_mqa_logits_metadata_bytes(
    int64_t num_q_tokens, int64_t num_max_sparse_blocks, int64_t sparse_block_kv,
    int64_t split_kv, int64_t num_sms);

}  // namespace lightop::deep_gemm
