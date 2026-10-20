#pragma once
#include <torch/extension.h>
#include <optional>

namespace lightop::deep_gemm {
// Core arguments match DeepGEMM, except that Q and its packed UE8M0 scales
// are separate C++ tensors and weights/output are FP32. The two optional
// tail tensors add per-query context masking without changing the metadata.
// Prefill flattens query tokens into Q rows; tail tensors are optional.
// The same request-aligned two-row metadata is consumed on gfx936 and gfx938.
// is_prefill selects the multi-CTA Prefill consumer; false preserves Decode.
torch::Tensor fp8_fp4_paged_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_scales,
    const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
    const torch::Tensor& metadata, int64_t num_max_sparse_blocks,
    int64_t sparse_block_kv,
    const std::optional<torch::Tensor>& context_lens = std::nullopt,
    const std::optional<torch::Tensor>& sparse_kv_block_indices = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    std::optional<int64_t> num_sms = std::nullopt,
    bool is_prefill = false);
torch::Tensor prefill_sparse_fp8_fp4_paged_mqa_logits(
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,int64_t,std::optional<int64_t>,bool,int);
}  // namespace lightop::deep_gemm
