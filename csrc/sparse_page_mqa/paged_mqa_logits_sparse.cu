#include "sparse_page_mqa.h"
#include "paged_mqa_logits_sparse_impl.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace lightop {
namespace deep_gemm {

namespace {

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& value,
                       const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(
        value.device() == reference.device(),
        name,
        " must be on ",
        reference.device());
}

template <int kNumWarps, bool kGroupScheduled = false>
std::vector<torch::Tensor> launch_bf16_page_mqa_logits_sparse_mask(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int max_context_len,
    int persistent_ctas) {
    constexpr int kNumHeads = 64;
    constexpr int kHeadDim = 128;
    constexpr int kBlockKV = 64;
    constexpr int kTokensPerWarp = 64;
    constexpr int kMaskTile = 16;
    constexpr int kThreads = kNumWarps * 64;

    const auto batch_size = static_cast<int>(q.size(0));
    const auto aligned_context_len =
        (max_context_len + kBlockKV - 1) / kBlockKV * kBlockKV;
    const auto logits_stride = static_cast<uint64_t>(aligned_context_len);
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));

    auto logits_storage = torch::empty(
        {batch_size, aligned_context_len},
        q.options().dtype(torch::kFloat));
    auto logits = logits_storage.slice(-1, 0, max_context_len);
    auto nonzero_mask = torch::empty(
        {batch_size, (max_context_len + kMaskTile - 1) / kMaskTile},
        q.options().dtype(torch::kInt16));

    constexpr int smem_size = kNumWarps * 16 * kHeadDim *
                              static_cast<int>(sizeof(at::BFloat16));
    dim3 grid(
        static_cast<uint32_t>(
            kGroupScheduled ? persistent_ctas * batch_size
                            : persistent_ctas),
        static_cast<uint32_t>(kGroupScheduled ? 1 : batch_size),
        1);
    dim3 block(kThreads, 1, 1);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(q.get_device());

    paged_mqa_logits_bf16_sparse_mask_persistent<
        at::BFloat16,
        1,
        kNumHeads,
        kHeadDim,
        kBlockKV,
        kNumWarps,
        kTokensPerWarp,
        kGroupScheduled><<<grid, block, smem_size, stream>>>(
        q.data_ptr<at::BFloat16>(),
        kv_cache.data_ptr<at::BFloat16>(),
        weights.data_ptr<float>(),
        logits_stride,
        block_table_stride,
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(),
        block_table.data_ptr<int>(),
        nonzero_mask.data_ptr<int16_t>());

    const auto error = hipGetLastError();
    TORCH_CHECK(
        error == hipSuccess,
        "BF16 page_mqa_logits_sparse_mask launch failed: ",
        hipGetErrorString(error));
    return {logits, nonzero_mask};
}

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_impl(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas) {
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    check_same_device(q, fused_kv_cache, "fused_kv_cache");
    check_same_device(q, weights, "weights");
    check_same_device(q, context_lens, "context_lens");
    check_same_device(q, block_table, "block_table");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    TORCH_CHECK(
        !schedule_meta.has_value(),
        "page_mqa_logits_sparse_mask requires schedule_meta=None");
    TORCH_CHECK(
        clean_logits,
        "page_mqa_logits_sparse_mask requires clean_logits=True");

    hipDeviceProp_t properties;
    const auto prop_error =
        hipGetDeviceProperties(&properties, q.get_device());
    TORCH_CHECK(
        prop_error == hipSuccess,
        "hipGetDeviceProperties failed: ",
        hipGetErrorString(prop_error));
    TORCH_CHECK(
        std::strncmp(properties.gcnArchName, "gfx936", 6) == 0 &&
            properties.multiProcessorCount == 80,
        "page_mqa_logits_sparse_mask requires gfx936/80CU, got ",
        properties.gcnArchName,
        "/",
        properties.multiProcessorCount,
        "CU");

    TORCH_CHECK(
        q.scalar_type() == torch::kBFloat16 && q.dim() == 4 &&
            q.size(0) > 0 && q.size(1) == 1 && q.size(2) == 64 &&
            q.size(3) == 128 && q.is_contiguous(),
        "q must be contiguous BF16 [batch, 1, 64, 128]");
    TORCH_CHECK(
        fused_kv_cache.scalar_type() == torch::kBFloat16 &&
            fused_kv_cache.dim() == 4 && fused_kv_cache.size(0) > 0 &&
            fused_kv_cache.size(1) == 64 &&
            fused_kv_cache.size(2) == 1 &&
            fused_kv_cache.size(3) == 128 &&
            fused_kv_cache.stride(0) == 64 * 128 &&
            fused_kv_cache.stride(1) == 128 &&
            fused_kv_cache.stride(2) == 128 &&
            fused_kv_cache.stride(3) == 1,
        "fused_kv_cache must be packed BF16 [blocks, 64, 1, 128]");
    const auto batch_size = q.size(0);
    TORCH_CHECK(
        batch_size == 8 || batch_size == 40,
        "page_mqa_logits_sparse_mask requires batch 8 or 40");
    TORCH_CHECK(
        weights.scalar_type() == torch::kFloat && weights.dim() == 2 &&
            weights.size(0) == batch_size && weights.size(1) == 64 &&
            weights.is_contiguous(),
        "weights must be contiguous float32 [batch, 64]");
    TORCH_CHECK(
        context_lens.scalar_type() == torch::kInt &&
            context_lens.dim() == 1 &&
            context_lens.size(0) == batch_size &&
            context_lens.is_contiguous(),
        "context_lens must be contiguous int32 [batch]");
    TORCH_CHECK(
        block_table.scalar_type() == torch::kInt &&
            block_table.dim() == 2 &&
            block_table.size(0) == batch_size &&
            block_table.size(1) > 0 && block_table.stride(1) == 1,
        "block_table must be int32 [batch, pages] with stride(1)=1");
    TORCH_CHECK(
        max_context_len > 0 &&
            max_context_len == block_table.size(1) * 64,
        "max_context_len must equal block_table capacity");
    TORCH_CHECK(
        num_warps == 4,
        "page_mqa_logits_sparse_mask requires num_warps=4");
    TORCH_CHECK(
        persistent_ctas >= 0 && persistent_ctas <= 4096,
        "persistent_ctas must be 0 (auto) or in [1, 4096]");

    const auto max_chunks =
        (static_cast<int64_t>(max_context_len) + 255) / 256;
    if (batch_size == 40) {
        const int resolved = persistent_ctas == 0
            ? static_cast<int>(std::min<int64_t>(max_chunks, 32))
            : static_cast<int>(
                  std::min<int64_t>(persistent_ctas, max_chunks));
        return launch_bf16_page_mqa_logits_sparse_mask<4, true>(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            max_context_len,
            resolved);
    }

    const int resolved = persistent_ctas == 0
        ? static_cast<int>(std::min<int64_t>(max_chunks, 20))
        : static_cast<int>(
              std::min<int64_t>(persistent_ctas, max_chunks));
    return launch_bf16_page_mqa_logits_sparse_mask<4, false>(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        resolved);
}

}  // namespace

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
    const int& persistent_ctas) {
    return page_mqa_logits_sparse_mask_impl(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas);
}

}  // namespace deep_gemm
}  // namespace lightop
