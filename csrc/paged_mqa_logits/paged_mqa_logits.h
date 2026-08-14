#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/extension.h>

namespace lightop {
namespace deep_gemm {

torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms);

torch::Tensor paged_mqa_logits(const torch::Tensor& q,
                               const torch::Tensor& fused_kv_cache,
                               const torch::Tensor& weights,
                               const torch::Tensor& context_lens,
                               const torch::Tensor& block_table,
                               std::optional<torch::Tensor>& schedule_meta,
                               const int& max_context_len,
                               const bool& clean_logits);

// Prefix-only sparse-output variant. Non-zero logits are stored in the dense
// values tensor and the accompanying int16 mask records one bit per token.
// Zero logits are left unwritten and must be reconstructed by a mask-aware
// consumer.
std::vector<torch::Tensor> page_mqa_logits_sparse_mask(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas);

// Grouped MQA evaluates 3/4/5 rows belonging to the same request while sharing
// each paged KV tile. It supports MTP2/3/4, 8/16/32/64 heads and BF16 or packed
// FP8 KV. The output mask is length-clamped: every bit at or beyond context_len
// is zero.
std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped(
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

}
}
