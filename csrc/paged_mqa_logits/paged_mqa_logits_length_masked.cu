#include "paged_mqa_logits.h"
#include "paged_mqa_logits_length_masked_impl.hpp"

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

template <int kNumWarps, bool kClearTail>
torch::Tensor launch_fp8_length_masked(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int max_context_len,
    int persistent_ctas) {
    constexpr int kNextN = 1;
    constexpr int kNumHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kBlockKV = 64;
    constexpr int kTokensPerWarp = 64;
    constexpr int kBatchSplit = kBlockKV * 4;
    constexpr int kThreads = kNumWarps * 64;

    const auto batch_size = static_cast<int>(q.size(0));
    const auto num_kv_blocks =
        static_cast<int>(fused_kv_cache.size(0));
    const auto kv_cache_stride_bytes =
        static_cast<uint64_t>(fused_kv_cache.stride(0));
    const auto aligned_context_len =
        (max_context_len + kBlockKV - 1) / kBlockKV * kBlockKV;
    const auto logits_stride =
        static_cast<uint64_t>(aligned_context_len);
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));

    auto logits_storage = torch::empty(
        {batch_size, aligned_context_len},
        q.options().dtype(torch::kFloat));
    auto logits = logits_storage.slice(-1, 0, max_context_len);

    const int smem_kv_pipe_size =
        kNumWarps * 32 * kHeadDim;
    const int smem_q_cache_size = kNumHeads * kHeadDim;
    const int smem_weight_cache_size =
        kNumHeads * static_cast<int>(sizeof(float));
    const int smem_scale_cache_size =
        kNumWarps * 64 * static_cast<int>(sizeof(float));
    const int smem_cache_size =
        smem_q_cache_size + smem_weight_cache_size +
        smem_scale_cache_size;
    const int smem_size =
        smem_kv_pipe_size > smem_cache_size ? smem_kv_pipe_size
                                             : smem_cache_size;

    dim3 grid(
        static_cast<uint32_t>(persistent_ctas),
        static_cast<uint32_t>(batch_size),
        1);
    dim3 block(kThreads, 1, 1);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(q.get_device());

    auto* kv_bytes =
        static_cast<uint8_t*>(fused_kv_cache.data_ptr());
    auto* kv_scales = reinterpret_cast<float*>(
        kv_bytes + kBlockKV * kHeadDim);
    paged_mqa_logits_fp8_length_masked_persistent<
        at::Float8_e4m3fn,
        kNextN,
        kNumHeads,
        kHeadDim,
        kBlockKV,
        kBatchSplit,
        kNumWarps,
        kTokensPerWarp,
        kClearTail><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const at::Float8_e4m3fn*>(q.data_ptr()),
        reinterpret_cast<const at::Float8_e4m3fn*>(
            fused_kv_cache.data_ptr()),
        kv_scales,
        weights.data_ptr<float>(),
        batch_size,
        num_kv_blocks,
        kv_cache_stride_bytes,
        logits_stride,
        block_table_stride,
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(),
        block_table.data_ptr<int>(),
        nullptr);

    const auto error = hipGetLastError();
    TORCH_CHECK(
        error == hipSuccess,
        "length-masked paged MQA launch failed: ",
        hipGetErrorString(error));
    return logits;
}

template <int kNumWarps, bool kGroupScheduled = false,
          bool kClearTail = false>
torch::Tensor launch_bf16_page_mqa_logits_mask(
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

    paged_mqa_logits_bf16_length_masked_persistent<
        at::BFloat16,
        1,
        kNumHeads,
        kHeadDim,
        kBlockKV,
        kNumWarps,
        kTokensPerWarp,
        kGroupScheduled,
        kClearTail><<<grid, block, smem_size, stream>>>(
        q.data_ptr<at::BFloat16>(),
        kv_cache.data_ptr<at::BFloat16>(),
        weights.data_ptr<float>(),
        logits_stride,
        block_table_stride,
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(),
        block_table.data_ptr<int>());

    const auto error = hipGetLastError();
    TORCH_CHECK(
        error == hipSuccess,
        "BF16 page_mqa_logits_mask launch failed: ",
        hipGetErrorString(error));
    return logits;
}

}  // namespace

torch::Tensor paged_mqa_logits_length_masked(
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
    // Compatibility entry: preserve the baseline gfx938/64CU FP8/H32
    // contract and its prefix-only output semantics.  The new Spark BF16 path
    // is exposed only through page_mqa_logits_mask.
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    check_same_device(q, fused_kv_cache, "fused_kv_cache");
    check_same_device(q, weights, "weights");
    check_same_device(q, context_lens, "context_lens");
    check_same_device(q, block_table, "block_table");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    TORCH_CHECK(
        !schedule_meta.has_value(),
        "length-masked paged MQA requires schedule_meta=None");
    TORCH_CHECK(
        clean_logits,
        "length-masked paged MQA requires clean_logits=True at the call "
        "site; only the valid prefix is guaranteed and the consumer must "
        "receive the same context_lens tensor");

    hipDeviceProp_t properties;
    const auto prop_error =
        hipGetDeviceProperties(&properties, q.get_device());
    TORCH_CHECK(
        prop_error == hipSuccess,
        "hipGetDeviceProperties failed: ",
        hipGetErrorString(prop_error));
    TORCH_CHECK(
        std::strncmp(properties.gcnArchName, "gfx938", 6) == 0,
        "length-masked paged MQA requires gfx938, got ",
        properties.gcnArchName);
    TORCH_CHECK(
        properties.multiProcessorCount == 64,
        "length-masked paged MQA requires 64 CUs, got ",
        properties.multiProcessorCount);

    TORCH_CHECK(
        q.scalar_type() == torch::kFloat8_e4m3fn,
        "q must be float8_e4m3fn");
    TORCH_CHECK(
        q.dim() == 4 && q.size(0) > 0 && q.size(1) == 1 &&
            q.size(2) == 32 && q.size(3) == 128,
        "q must have shape [batch, 1, 32, 128]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");

    TORCH_CHECK(
        fused_kv_cache.scalar_type() == torch::kUInt8,
        "fused_kv_cache must be uint8");
    TORCH_CHECK(
        fused_kv_cache.dim() == 4 &&
            fused_kv_cache.size(0) > 0 &&
            fused_kv_cache.size(1) == 64 &&
            fused_kv_cache.size(2) == 1 &&
            fused_kv_cache.size(3) == 132,
        "fused_kv_cache must have shape [num_blocks, 64, 1, 132]");
    TORCH_CHECK(
        fused_kv_cache.stride(0) == 64 * 132 &&
            fused_kv_cache.stride(1) == 132 &&
            fused_kv_cache.stride(2) == 132 &&
            fused_kv_cache.stride(3) == 1,
        "fused_kv_cache must have packed stride [8448, 132, 132, 1]");

    const auto batch_size = q.size(0);
    TORCH_CHECK(
        weights.scalar_type() == torch::kFloat &&
            weights.dim() == 2 && weights.size(0) == batch_size &&
            weights.size(1) == 32 && weights.is_contiguous(),
        "weights must be contiguous float32 [batch, 32]");
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
        "length-masked paged MQA requires num_warps=4");
    TORCH_CHECK(
        persistent_ctas >= 0 && persistent_ctas <= 4096,
        "persistent_ctas must be 0 (auto) or in [1, 4096]");

    int resolved_persistent_ctas = persistent_ctas;
    if (resolved_persistent_ctas == 0) {
        const auto max_chunks =
            (static_cast<int64_t>(max_context_len) + 255) / 256;
        const auto batch_aware_floor =
            std::max<int64_t>(128, (2048 + batch_size - 1) / batch_size);
        resolved_persistent_ctas = static_cast<int>(
            std::min<int64_t>(max_chunks, batch_aware_floor));
    }

    return launch_fp8_length_masked<4, false>(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        resolved_persistent_ctas);
}

template <bool kClearTail>
torch::Tensor page_mqa_logits_mask_dispatch(
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
        "length-masked paged MQA requires schedule_meta=None");
    TORCH_CHECK(
        clean_logits,
        "page_mqa_logits_mask requires clean_logits=True at the call site; "
        "the selected native entry controls whether the tail is cleared");

    hipDeviceProp_t properties;
    const auto prop_error =
        hipGetDeviceProperties(&properties, q.get_device());
    TORCH_CHECK(
        prop_error == hipSuccess,
        "hipGetDeviceProperties failed: ",
        hipGetErrorString(prop_error));
    const bool is_fp8_64cu =
        std::strncmp(properties.gcnArchName, "gfx938", 6) == 0 &&
        properties.multiProcessorCount == 64 &&
        q.scalar_type() == torch::kFloat8_e4m3fn;
    const bool is_bf16_80cu =
        std::strncmp(properties.gcnArchName, "gfx936", 6) == 0 &&
        properties.multiProcessorCount == 80 &&
        q.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(
        is_fp8_64cu || is_bf16_80cu,
        "page_mqa_logits_mask supports only gfx938/64CU FP8 or "
        "gfx936/80CU BF16, got ",
        properties.gcnArchName,
        "/",
        properties.multiProcessorCount,
        "CU with q dtype ",
        q.scalar_type());

    const int expected_heads = is_bf16_80cu ? 64 : 32;
    TORCH_CHECK(
        q.dim() == 4 && q.size(0) > 0 && q.size(1) == 1 &&
            q.size(2) == expected_heads && q.size(3) == 128,
        "q must have shape [batch, 1, ", expected_heads, ", 128]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");

    const int expected_kv_width = is_bf16_80cu ? 128 : 132;
    const auto expected_kv_dtype =
        is_bf16_80cu ? torch::kBFloat16 : torch::kUInt8;
    TORCH_CHECK(
        fused_kv_cache.scalar_type() == expected_kv_dtype &&
        fused_kv_cache.dim() == 4 &&
            fused_kv_cache.size(0) > 0 &&
            fused_kv_cache.size(1) == 64 &&
            fused_kv_cache.size(2) == 1 &&
            fused_kv_cache.size(3) == expected_kv_width,
        "fused_kv_cache has an invalid dtype or shape");
    TORCH_CHECK(
        fused_kv_cache.stride(0) == 64 * expected_kv_width &&
            fused_kv_cache.stride(1) == expected_kv_width &&
            fused_kv_cache.stride(2) == expected_kv_width &&
            fused_kv_cache.stride(3) == 1,
        "fused_kv_cache must be packed and contiguous");

    const auto batch_size = q.size(0);
    TORCH_CHECK(
        weights.scalar_type() == torch::kFloat &&
            weights.dim() == 2 && weights.size(0) == batch_size &&
            weights.size(1) == expected_heads && weights.is_contiguous(),
        "weights must be contiguous float32 [batch, heads]");
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
        "length-masked paged MQA requires num_warps=4");
    TORCH_CHECK(
        persistent_ctas >= 0 && persistent_ctas <= 4096,
        "persistent_ctas must be 0 (auto) or in [1, 4096]");

    // Preserve one independent row per CTA for B40. Only the linear launch
    // order changes: at each persistent-CTA seed, the five request-major
    // draft rows are adjacent so repeated physical KV pages have short-lived
    // L2 reuse. Arbitrary block tables remain correct without a special
    // fallback because no data is shared across CTAs.
    if (is_bf16_80cu && batch_size == 40) {
        const auto max_chunks =
            (static_cast<int64_t>(max_context_len) + 255) / 256;
        const int grouped_persistent_ctas = persistent_ctas == 0
            ? static_cast<int>(std::min<int64_t>(max_chunks, 32))
            : static_cast<int>(
                  std::min<int64_t>(persistent_ctas, max_chunks));
        return launch_bf16_page_mqa_logits_mask<4, true, kClearTail>(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            max_context_len,
            grouped_persistent_ctas);
    }

    int resolved_persistent_ctas = persistent_ctas;
    if (resolved_persistent_ctas == 0) {
        const auto max_chunks =
            (static_cast<int64_t>(max_context_len) + 255) / 256;
        const auto batch_aware_floor = is_bf16_80cu
            ? std::max<int64_t>(1, (160 + batch_size - 1) / batch_size)
            : std::max<int64_t>(
                  128, (2048 + batch_size - 1) / batch_size);
        resolved_persistent_ctas = static_cast<int>(
            std::min<int64_t>(max_chunks, batch_aware_floor));
    } else {
        const auto max_chunks =
            (static_cast<int64_t>(max_context_len) + 255) / 256;
        resolved_persistent_ctas = static_cast<int>(
            std::min<int64_t>(resolved_persistent_ctas, max_chunks));
    }

    if (is_bf16_80cu) {
        return launch_bf16_page_mqa_logits_mask<4, false, kClearTail>(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            max_context_len,
            resolved_persistent_ctas);
    }
    return launch_fp8_length_masked<4, kClearTail>(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        resolved_persistent_ctas);
}

torch::Tensor page_mqa_logits_mask(
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
    // Public new API: exactly the validated length-aware, prefix-only path.
    return page_mqa_logits_mask_dispatch<false>(
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

torch::Tensor page_mqa_logits_mask_clean_tail(
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
    // Private binding used only by paged_mqa_logits(clean_logits=True).
    return page_mqa_logits_mask_dispatch<true>(
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
