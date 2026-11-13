// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace lightop::dsv41 {
namespace {
constexpr int kThreads = 256;

__global__ void initialize_page_map(int32_t* map, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < count) map[i] = -1;
}

__device__ int64_t load_page(const void* ptr, bool wide, int64_t offset) {
    return wide ? static_cast<const int64_t*>(ptr)[offset]
                : static_cast<const int32_t*>(ptr)[offset];
}

__global__ void invert_selected_pages(const void* pages, int64_t stride, bool wide,
    int64_t count, int64_t max_pages, bool static_pages, int32_t* map) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= count) return;
    int64_t page = load_page(pages, wide, i * stride);
    if (page < 0) {
        if (static_pages) return;
        page += max_pages;
    }
    map[page] = static_cast<int32_t>(i);
}

template <typename T>
__global__ void remap_indices_kernel(const T* indices, const int32_t* map, T* output,
    int64_t count, int64_t width, int64_t row_stride, int64_t col_stride, int64_t page_size) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= count) return;
    const T token = indices[(i / width) * row_stride + (i % width) * col_stride];
    if (token < 0) {
        output[i] = token;
        return;
    }
    const int64_t page = static_cast<int64_t>(token) / page_size;
    const int64_t offset = static_cast<int64_t>(token) - page * page_size;
    output[i] = static_cast<T>(static_cast<int64_t>(map[page]) * page_size + offset);
}
}  // namespace

torch::Tensor remap_indices_to_staging(const torch::Tensor& indices,
    const torch::Tensor& pages, int64_t page_size, int64_t max_pages, bool static_pages) {
    TORCH_CHECK(indices.is_cuda() &&
                (indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt64),
                "remap_indices_to_staging: indices must be device int32 or int64");
    TORCH_CHECK(pages.device() == indices.device() && pages.dim() == 1 &&
                (pages.scalar_type() == torch::kInt32 || pages.scalar_type() == torch::kInt64),
                "remap_indices_to_staging: pages must be an integer vector on the same device");
    TORCH_CHECK(page_size > 0 && max_pages >= 0 && pages.numel() <= INT32_MAX &&
                (indices.numel() == 0 || max_pages > 0),
                "remap_indices_to_staging: invalid page geometry");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(indices.device());
    auto output = torch::empty(indices.sizes(), indices.options());
    if (indices.numel() == 0) return output;
    // Common 1D/2D views are read directly. Only uncommon higher-rank strided
    // tensors need materialization; contiguous tensors of any rank are direct.
    auto input = indices.dim() <= 2 || indices.is_contiguous() ? indices : indices.contiguous();
    int64_t width = input.numel(), row_stride = 0, col_stride = 1;
    if (input.dim() == 1) col_stride = input.stride(0);
    if (input.dim() == 2) {
        width = input.size(1);
        row_stride = input.stride(0);
        col_stride = input.stride(1);
    }
    auto map = torch::empty({max_pages}, indices.options().dtype(torch::kInt32));
    auto stream = at::hip::getCurrentHIPStream();
    // These launches deliberately establish global ordering. A CTA barrier
    // cannot make initialization, scatter and dependent gathers race-free.
    initialize_page_map<<<(max_pages + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        map.data_ptr<int32_t>(), max_pages);
    if (pages.numel()) {
        invert_selected_pages<<<(pages.numel() + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            pages.data_ptr(), pages.stride(0), pages.scalar_type() == torch::kInt64,
            pages.numel(), max_pages, static_pages, map.data_ptr<int32_t>());
    }
    const int64_t blocks = (input.numel() + kThreads - 1) / kThreads;
    if (input.scalar_type() == torch::kInt32) {
        remap_indices_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(input.data_ptr<int32_t>(),
            map.data_ptr<int32_t>(), output.data_ptr<int32_t>(), input.numel(), width,
            row_stride, col_stride, page_size);
    } else {
        remap_indices_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(input.data_ptr<int64_t>(),
            map.data_ptr<int32_t>(), output.data_ptr<int64_t>(), input.numel(), width,
            row_stride, col_stride, page_size);
    }
    TORCH_CHECK(hipGetLastError() == hipSuccess, "remap_indices_to_staging: kernel launch failed");
    return output;
}
}  // namespace lightop::dsv41
