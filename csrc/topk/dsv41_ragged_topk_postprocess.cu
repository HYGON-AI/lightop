// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cmath>
#include <cstdint>

namespace lightop::dsv41 {
namespace {
constexpr int kThreads = 128;

template <int Items>
__global__ void ragged_topk_postprocess_kernel(
    const float* scores, const int32_t* selected, const int32_t* offsets,
    const void* slots, bool wide_slots, int32_t* pages, int32_t* raw,
    int64_t k, int64_t score_width, int64_t num_slots, bool mask_scores,
    int64_t score_row_stride, int64_t score_col_stride, int64_t selected_row_stride,
    int64_t selected_col_stride, int64_t offset_stride, int64_t slot_stride,
    int64_t page_row_stride, int64_t page_col_stride, int64_t raw_row_stride,
    int64_t raw_col_stride) {
    using Sort = hipcub::BlockRadixSort<int32_t, kThreads, Items>;
    __shared__ typename Sort::TempStorage storage;
    const int64_t row = blockIdx.x;
    const int64_t offset = offsets[row * offset_stride];
    int32_t keys[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int64_t col = threadIdx.x + i * kThreads;
        int32_t key = col < k ? selected[row * selected_row_stride + col * selected_col_stride]
                              : INT32_MAX;
        if (key < 0) key = INT32_MAX;
        if (mask_scores && key != INT32_MAX) {
            const int64_t local = static_cast<int64_t>(key) - offset;
            if (local < 0 || local >= score_width ||
                !(scores[row * score_row_stride + local * score_col_stride] > -INFINITY)) {
                key = INT32_MAX;
            }
        }
        keys[i] = key;
    }
    // Only the key multiset matters; striped loads keep global accesses coalesced.
    // hipCUB redistributes the sorted keys to striped order for coalesced writes.
    Sort(storage).SortBlockedToStriped(keys);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int64_t col = threadIdx.x + i * kThreads;
        if (col >= k) continue;
        const int32_t key = keys[i];
        int32_t physical = -1, relative = -1;
        if (key != INT32_MAX) {
            const int64_t clamped = key < num_slots ? key : num_slots - 1;
            physical = static_cast<int32_t>(wide_slots
                ? static_cast<const int64_t*>(slots)[clamped * slot_stride]
                : static_cast<const int32_t*>(slots)[clamped * slot_stride]);
            relative = static_cast<int32_t>(static_cast<int64_t>(key) - offset);
        }
        pages[row * page_row_stride + col * page_col_stride] = physical;
        if (raw) raw[row * raw_row_stride + col * raw_col_stride] = relative;
    }
}

template <int Items>
void launch(const torch::Tensor& scores, const torch::Tensor& selected,
    const torch::Tensor& offsets, const torch::Tensor& slots, torch::Tensor& pages,
    const c10::optional<torch::Tensor>& raw, bool mask_scores) {
    ragged_topk_postprocess_kernel<Items><<<selected.size(0), kThreads, 0,
        at::hip::getCurrentHIPStream()>>>(scores.data_ptr<float>(), selected.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(), slots.data_ptr(), slots.scalar_type() == torch::kInt64,
        pages.data_ptr<int32_t>(), raw ? raw->data_ptr<int32_t>() : nullptr,
        selected.size(1), scores.size(1), slots.numel(), mask_scores,
        scores.stride(0), scores.stride(1), selected.stride(0), selected.stride(1),
        offsets.stride(0), slots.stride(0), pages.stride(0), pages.stride(1),
        raw ? raw->stride(0) : 0, raw ? raw->stride(1) : 0);
}
}  // namespace

void ragged_topk_postprocess(const torch::Tensor& scores, const torch::Tensor& selected,
    const torch::Tensor& offsets, const torch::Tensor& slots, torch::Tensor& pages,
    const c10::optional<torch::Tensor>& raw, bool mask_scores) {
    TORCH_CHECK(selected.is_cuda() && selected.dim() == 2 && selected.scalar_type() == torch::kInt32,
                "ragged_topk_postprocess: selected must be a device int32 matrix");
    TORCH_CHECK(scores.device() == selected.device() && scores.dim() == 2 &&
                scores.size(0) == selected.size(0) && scores.scalar_type() == torch::kFloat32,
                "ragged_topk_postprocess: scores must be a matching device float32 matrix");
    TORCH_CHECK(offsets.device() == selected.device() && offsets.dim() == 1 &&
                offsets.numel() == selected.size(0) && offsets.scalar_type() == torch::kInt32,
                "ragged_topk_postprocess: offsets must match rows and be device int32");
    TORCH_CHECK(slots.device() == selected.device() && slots.dim() == 1 &&
                (slots.scalar_type() == torch::kInt32 || slots.scalar_type() == torch::kInt64) &&
                (selected.numel() == 0 || slots.numel() > 0),
                "ragged_topk_postprocess: slots must be a nonempty device integer vector");
    for (const auto& out : {pages, raw ? *raw : pages}) {
        TORCH_CHECK(out.device() == selected.device() && out.sizes() == selected.sizes() &&
                    out.scalar_type() == torch::kInt32,
                    "ragged_topk_postprocess: outputs must match selected and be device int32");
    }
    TORCH_CHECK(selected.size(1) <= 2048, "ragged_topk_postprocess: top-k exceeds 2048");
    if (selected.numel() == 0) return;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(selected.device());
    const auto k = selected.size(1);
    if (k <= 128) launch<1>(scores, selected, offsets, slots, pages, raw, mask_scores);
    else if (k <= 256) launch<2>(scores, selected, offsets, slots, pages, raw, mask_scores);
    else if (k <= 512) launch<4>(scores, selected, offsets, slots, pages, raw, mask_scores);
    else if (k <= 1024) launch<8>(scores, selected, offsets, slots, pages, raw, mask_scores);
    else launch<16>(scores, selected, offsets, slots, pages, raw, mask_scores);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "ragged_topk_postprocess: kernel launch failed");
}
}  // namespace lightop::dsv41
