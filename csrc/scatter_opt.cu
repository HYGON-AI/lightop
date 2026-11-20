// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_scatter_hip.hip for lightop P1.
#include <algorithm>
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using int4_vec = int4;
constexpr int VEC_BYTES = 16;

template <typename scalar_t>
struct VecConfig {
    static constexpr int kElements = VEC_BYTES / sizeof(scalar_t);
};

template <typename scalar_t>
__device__ __forceinline__ void copy_128bit(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int64_t count) {
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    int64_t vec_count = count / kElems;
    auto* dst_vec = reinterpret_cast<int4_vec*>(dst);
    const auto* src_vec = reinterpret_cast<const int4_vec*>(src);
    for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
    int64_t remaining_start = vec_count * kElems;
    for (int64_t i = remaining_start + threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void copy_scalar(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int64_t count) {
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <typename scalar_t>
__global__ void scatter_special_small_dim(
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ index,
    const int64_t index_stride0,
    const scalar_t* __restrict__ src,
    int64_t src_rows, int64_t dst_rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= src_rows) return;
    int64_t dst_row = index[row * index_stride0];
    if (dst_row < 0 || dst_row >= dst_rows) return;
    const scalar_t* src_ptr = src + row * cols;
    scalar_t* dst_ptr = dst + dst_row * cols;
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
    if (aligned && cols >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, cols);
    } else {
        copy_scalar<scalar_t>(dst_ptr, src_ptr, cols);
    }
}

template <typename scalar_t>
__global__ void scatter_special_large_batch(
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ index,
    const int64_t index_stride0,
    const scalar_t* __restrict__ src,
    int64_t src_rows, int64_t dst_rows, int64_t cols,
    int64_t rows_per_block) {
    int64_t base_row = blockIdx.x * rows_per_block;
    if (base_row >= src_rows) return;
    int64_t end_row = std::min(base_row + rows_per_block, src_rows);
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    for (int64_t r = base_row; r < end_row; ++r) {
        int64_t dst_row = index[r * index_stride0];
        if (dst_row < 0 || dst_row >= dst_rows) continue;
        const scalar_t* src_ptr = src + r * cols;
        scalar_t* dst_ptr = dst + dst_row * cols;
        bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                       (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
        if (aligned && cols >= kElems) {
            copy_128bit<scalar_t>(dst_ptr, src_ptr, cols);
        } else {
            copy_scalar<scalar_t>(dst_ptr, src_ptr, cols);
        }
    }
}

template <typename scalar_t>
__global__ void scatter_special_large_dim(
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ index,
    const int64_t index_stride0,
    const scalar_t* __restrict__ src,
    int64_t src_rows, int64_t dst_rows, int64_t cols,
    int64_t blocks_per_row) {
    int64_t row = blockIdx.x / blocks_per_row;
    if (row >= src_rows) return;
    int64_t sub_block = blockIdx.x % blocks_per_row;
    int64_t dst_row = index[row * index_stride0];
    if (dst_row < 0 || dst_row >= dst_rows) return;
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    int64_t chunk_elems = ((cols / blocks_per_row) / kElems) * kElems;
    if (chunk_elems < kElems) chunk_elems = kElems;
    int64_t offset = sub_block * chunk_elems;
    // blocks_per_row and chunk_elems both round down, so their product can cover
    // fewer than `cols` elements (float cols=1025 → 1 block × 1024). Let the last
    // sub-block absorb the remainder so the row tail is always written; earlier
    // sub-blocks keep their own chunk, so ranges never overlap.
    const int64_t end = (sub_block + 1 == blocks_per_row)
                            ? cols
                            : std::min((sub_block + 1) * chunk_elems, cols);
    int64_t actual_chunk = end - offset;
    if (actual_chunk <= 0) return;
    const scalar_t* src_ptr = src + row * cols + offset;
    scalar_t* dst_ptr = dst + dst_row * cols + offset;
    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
    if (aligned && actual_chunk >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, actual_chunk);
    } else {
        copy_scalar<scalar_t>(dst_ptr, src_ptr, actual_chunk);
    }
}

template <typename scalar_t>
__global__ void scatter_kernel_random(
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ index,
    const scalar_t* __restrict__ src,
    int64_t total_elements,
    int64_t dst_rows,
    int64_t cols) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    int64_t dst_row = index[tid];
    if (dst_row < 0 || dst_row >= dst_rows) return;
    dst[dst_row * cols + (tid % cols)] = src[tid];
}

template <typename scalar_t>
void scatter_launch(
    torch::Tensor& dst,
    const torch::Tensor& index,
    const torch::Tensor& src,
    bool special) {

    const int64_t src_rows = src.size(0);
    const int64_t dst_rows = dst.size(0);
    const int64_t cols = src.size(1);
    const int64_t total = src_rows * cols;
    if (total == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK = 256;

    const bool is_row_uniform = (index.dim() == 2 && index.stride(1) == 0);
    const bool use_special = special || is_row_uniform;

    if (use_special) {
        const int64_t idx_stride0 = index.stride(0);
        const int64_t row_bytes = cols * sizeof(scalar_t);
        constexpr int64_t LARGE_DIM_BYTE_THRESHOLD = 4096;
        constexpr int64_t LARGE_BATCH_THRESHOLD = 2048;

        if (row_bytes > LARGE_DIM_BYTE_THRESHOLD) {
            int64_t blocks_per_row = row_bytes / LARGE_DIM_BYTE_THRESHOLD;
            if (blocks_per_row < 1) blocks_per_row = 1;
            int64_t total_blocks = src_rows * blocks_per_row;
            scatter_special_large_dim<scalar_t><<<total_blocks, BLOCK, 0, stream>>>(
                dst.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), idx_stride0,
                src.data_ptr<scalar_t>(), src_rows, dst_rows, cols, blocks_per_row);
        } else if (src_rows > LARGE_BATCH_THRESHOLD) {
            int64_t rows_per_block = std::max(int64_t(1), int64_t(32768) / row_bytes);
            rows_per_block = std::min(rows_per_block, src_rows);
            int64_t total_blocks = (src_rows + rows_per_block - 1) / rows_per_block;
            scatter_special_large_batch<scalar_t><<<total_blocks, BLOCK, 0, stream>>>(
                dst.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), idx_stride0,
                src.data_ptr<scalar_t>(), src_rows, dst_rows, cols, rows_per_block);
        } else {
            scatter_special_small_dim<scalar_t><<<src_rows, BLOCK, 0, stream>>>(
                dst.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), idx_stride0,
                src.data_ptr<scalar_t>(), src_rows, dst_rows, cols);
        }
    } else {
        const unsigned int grid_size = static_cast<unsigned int>(
            (total + BLOCK - 1) / BLOCK);
        scatter_kernel_random<scalar_t><<<grid_size, BLOCK, 0, stream>>>(
            dst.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(),
            src.data_ptr<scalar_t>(),
            total, dst_rows, cols);
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

namespace at {
namespace native {

torch::Tensor scatter_opt(
    torch::Tensor& dst,
    torch::Tensor& index,
    torch::Tensor& src,
    bool special) {
    TORCH_CHECK(dst.is_cuda() && index.is_cuda() && src.is_cuda(), "Must be HIP tensors");
    TORCH_CHECK(dst.dim() == 2 && src.dim() == 2, "dst and src must be 2D");
    TORCH_CHECK(index.dim() == 2, "index must be 2D");
    TORCH_CHECK(dst.size(1) == src.size(1), "Column count mismatch");
    TORCH_CHECK(index.dtype() == torch::kInt64, "index must be int64");
    TORCH_CHECK(dst.dtype() == src.dtype(), "dtype mismatch");
    TORCH_CHECK(index.size(0) == src.size(0) && index.size(1) == src.size(1),
                 "index shape must match src shape");
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dst.scalar_type(), "scatter_opt",
        ([&] { scatter_launch<scalar_t>(dst, index, src, special); }));
    return dst;
}

}  // namespace native
}  // namespace at
