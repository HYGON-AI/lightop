// Ported from custom_ops/custom_embedding_hip.hip for lightop P1.
#include <algorithm>
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

using int4_vec = int4;
constexpr int VEC_BYTES = 16;

template <typename scalar_t>
struct VecConfig {
    static constexpr int kElements = VEC_BYTES / sizeof(scalar_t);
};

template <typename scalar_t>
__device__ __forceinline__ void copy_128bit_aligned(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int64_t vec_count) {

    auto* dst_vec = reinterpret_cast<int4_vec*>(dst);
    const auto* src_vec = reinterpret_cast<const int4_vec*>(src);

    #pragma unroll 4
    for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void copy_128bit(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int64_t count) {

    constexpr int kElems = VecConfig<scalar_t>::kElements;
    int64_t vec_count = count / kElems;

    copy_128bit_aligned<scalar_t>(dst, src, vec_count);

    int64_t remaining_start = vec_count * kElems;
    int64_t remaining = count - remaining_start;
    if (remaining > 0 && threadIdx.x < remaining) {
        dst[remaining_start + threadIdx.x] = src[remaining_start + threadIdx.x];
    }
}

template <typename scalar_t>
__global__ void embedding_kernel_one_block_per_row(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_indices) return;

    int64_t src_row = index[row_idx];
    const scalar_t* src_ptr = weight + src_row * embedding_dim;
    scalar_t* dst_ptr = dst + row_idx * embedding_dim;

    constexpr int kElems = VecConfig<scalar_t>::kElements;
    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);

    if (aligned && embedding_dim >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, embedding_dim);
    } else {
        for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

template <typename scalar_t>
__global__ void embedding_kernel_multi_row(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t rows_per_block) {

    int64_t base_row = blockIdx.x * rows_per_block;
    if (base_row >= num_indices) return;

    int64_t end_row = std::min(base_row + rows_per_block, num_indices);
    constexpr int kElems = VecConfig<scalar_t>::kElements;

    for (int64_t r = base_row; r < end_row; ++r) {
        int64_t src_row = index[r];
        const scalar_t* src_ptr = weight + src_row * embedding_dim;
        scalar_t* dst_ptr = dst + r * embedding_dim;

        bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                       (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);

        if (aligned && embedding_dim >= kElems) {
            copy_128bit<scalar_t>(dst_ptr, src_ptr, embedding_dim);
        } else {
            for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
                dst_ptr[i] = src_ptr[i];
            }
        }
    }
}

template <typename scalar_t>
__global__ void embedding_kernel_large_dim_gather_style(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t blocks_per_row) {

    int64_t row_idx = blockIdx.x / blocks_per_row;
    if (row_idx >= num_indices) return;

    int64_t sub_block = blockIdx.x % blocks_per_row;

    constexpr int64_t CHUNK_BYTE_THRESHOLD = 8192;
    int64_t chunk_bytes = CHUNK_BYTE_THRESHOLD;
    int64_t offset_bytes = sub_block * chunk_bytes;

    int64_t row_total_bytes = embedding_dim * sizeof(scalar_t);
    if (offset_bytes >= row_total_bytes) return;

    int64_t actual_chunk_bytes = std::min(chunk_bytes, row_total_bytes - offset_bytes);
    int64_t elem_offset = offset_bytes / sizeof(scalar_t);
    int64_t actual_elems = actual_chunk_bytes / sizeof(scalar_t);

    if (actual_elems <= 0) return;

    int64_t src_row = index[row_idx];
    const scalar_t* src_ptr = weight + src_row * embedding_dim + elem_offset;
    scalar_t* dst_ptr = dst + row_idx * embedding_dim + elem_offset;

    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);

    constexpr int kElems = VecConfig<scalar_t>::kElements;
    if (aligned && actual_elems >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, actual_elems);
    } else {
        for (int64_t i = threadIdx.x; i < actual_elems; i += blockDim.x) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

template <typename scalar_t>
void launch_embedding_kernel(
    scalar_t* dst, const scalar_t* weight, const int64_t* index,
    int64_t num_indices, int64_t embedding_dim, cudaStream_t stream) {

    const int threads = 256;
    // Raise the large-dimension threshold and round blocks per row up.
    constexpr int64_t LARGE_DIM_BYTE_THRESHOLD = 8192;
    constexpr int64_t MULTI_ROW_BATCH_THRESHOLD = 8192;
    constexpr int64_t GATHER_STYLE_DIM_THRESHOLD = 65536;
    constexpr int64_t GATHER_CHUNK_BYTE_THRESHOLD = 8192;

    int64_t row_bytes = embedding_dim * sizeof(scalar_t);

    constexpr int kElems = VecConfig<scalar_t>::kElements;
    int64_t vecs_per_row = embedding_dim / kElems;
    bool row_fills_block = (vecs_per_row >= threads);

    if (embedding_dim >= GATHER_STYLE_DIM_THRESHOLD) {
        int64_t blocks_per_row = (row_bytes + GATHER_CHUNK_BYTE_THRESHOLD - 1) / GATHER_CHUNK_BYTE_THRESHOLD;
        if (blocks_per_row < 1) blocks_per_row = 1;
        int64_t total_blocks = num_indices * blocks_per_row;

        embedding_kernel_large_dim_gather_style<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        blocks_per_row);

    } else if (row_bytes > LARGE_DIM_BYTE_THRESHOLD) {
        int64_t blocks_per_row = (row_bytes + LARGE_DIM_BYTE_THRESHOLD - 1) / LARGE_DIM_BYTE_THRESHOLD;
        if (blocks_per_row < 1) blocks_per_row = 1;
        int64_t total_blocks = num_indices * blocks_per_row;

        embedding_kernel_large_dim_gather_style<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        blocks_per_row);

    } else if (num_indices > MULTI_ROW_BATCH_THRESHOLD && !row_fills_block) {
        int64_t rows_per_block = std::max(int64_t(1), int64_t(32768) / row_bytes);
        rows_per_block = std::min(rows_per_block, num_indices);
        int64_t total_blocks = (num_indices + rows_per_block - 1) / rows_per_block;

        embedding_kernel_multi_row<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        rows_per_block);

    } else {
        embedding_kernel_one_block_per_row<scalar_t><<<dim3(num_indices), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim);
    }
}

namespace at {
namespace native {

torch::Tensor embedding_opt(
    torch::Tensor& dst,
    torch::Tensor& src,
    torch::Tensor& weight) {

    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(src.dim() == 1 && src.scalar_type() == at::ScalarType::Long,
                "index (src) must be a 1D int64 tensor");
    TORCH_CHECK(dst.dim() == 2 && dst.size(1) == weight.size(1),
                "dst shape mismatch");
    TORCH_CHECK(dst.scalar_type() == weight.scalar_type(),
                "dst and weight must have the same dtype");

    int64_t num_indices = src.size(0);
    int64_t embedding_dim = weight.size(1);

    if (num_indices == 0) return dst;

    auto scalar_type = weight.scalar_type();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (scalar_type == at::ScalarType::Float) {
        launch_embedding_kernel<float>(
            dst.data_ptr<float>(), weight.data_ptr<float>(),
            src.data_ptr<int64_t>(), num_indices, embedding_dim, stream);
    } else if (scalar_type == at::ScalarType::Half) {
        launch_embedding_kernel<at::Half>(
            dst.data_ptr<at::Half>(), weight.data_ptr<at::Half>(),
            src.data_ptr<int64_t>(), num_indices, embedding_dim, stream);
    } else if (scalar_type == at::ScalarType::BFloat16) {
        launch_embedding_kernel<at::BFloat16>(
            dst.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
            src.data_ptr<int64_t>(), num_indices, embedding_dim, stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for embedding_opt: ", scalar_type);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

}  // namespace native
}  // namespace at
