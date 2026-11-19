// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_gather_hip.hip for lightop P1.
#include <algorithm>
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void gather_kernel_general(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ dst,
    int dst_batch_size, int dim_size) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int vec_col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kElementsPerVec = sizeof(scalar_t) == 4 ? 4 : 8;
    int col_start = vec_col * kElementsPerVec;
    if (row < dst_batch_size && col_start < dim_size - kElementsPerVec + 1) {
        const longlong2* idx_vec_ptr = reinterpret_cast<const longlong2*>(
            __builtin_assume_aligned(&index[row * dim_size + col_start], 16));
        longlong2 idx_vec[4];
        #pragma unroll
        for (int i = 0; i < kElementsPerVec / 2; ++i) idx_vec[i] = idx_vec_ptr[i];
        float4 out_vec; scalar_t* out_ptr = reinterpret_cast<scalar_t*>(&out_vec);
        if constexpr (sizeof(scalar_t) == 4) {
            int idx0 = idx_vec[0].x, idx1 = idx_vec[0].y, idx2 = idx_vec[1].x, idx3 = idx_vec[1].y;
            out_ptr[0] = src[idx0*dim_size+col_start]; out_ptr[1] = src[idx1*dim_size+col_start+1];
            out_ptr[2] = src[idx2*dim_size+col_start+2]; out_ptr[3] = src[idx3*dim_size+col_start+3];
        } else {
            int idx0=idx_vec[0].x, idx1=idx_vec[0].y, idx2=idx_vec[1].x, idx3=idx_vec[1].y;
            int idx4=idx_vec[2].x, idx5=idx_vec[2].y, idx6=idx_vec[3].x, idx7=idx_vec[3].y;
            out_ptr[0]=src[idx0*dim_size+col_start]; out_ptr[1]=src[idx1*dim_size+col_start+1];
            out_ptr[2]=src[idx2*dim_size+col_start+2]; out_ptr[3]=src[idx3*dim_size+col_start+3];
            out_ptr[4]=src[idx4*dim_size+col_start+4]; out_ptr[5]=src[idx5*dim_size+col_start+5];
            out_ptr[6]=src[idx6*dim_size+col_start+6]; out_ptr[7]=src[idx7*dim_size+col_start+7];
        }
        float4* dst_vec_ptr = reinterpret_cast<float4*>(
            __builtin_assume_aligned(&dst[row * dim_size + col_start], 16));
        *dst_vec_ptr = out_vec;
    }
}

template <typename scalar_t>
__global__ void gather_kernel_wide(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ dst,
    int dst_batch_size, int dim_size) 
{
    int row = blockIdx.y;
    int vec_col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kElementsPerVec = sizeof(scalar_t) == 4 ? 4 : 8;
    int col_start = vec_col * kElementsPerVec;
    if (row < dst_batch_size && col_start < dim_size - kElementsPerVec + 1) {
        const longlong4* idx_base = reinterpret_cast<const longlong4*>(
            __builtin_assume_aligned(&index[row * dim_size + col_start], 32));
        float4 out_vec; scalar_t* out_ptr = reinterpret_cast<scalar_t*>(&out_vec);
        if constexpr (sizeof(scalar_t) == 4) {
            longlong4 iv = idx_base[0];
            const scalar_t* b0=&src[iv.x*dim_size], *b1=&src[iv.y*dim_size];
            const scalar_t* b2=&src[iv.z*dim_size], *b3=&src[iv.w*dim_size];
            out_ptr[0]=b0[col_start]; out_ptr[1]=b1[col_start+1];
            out_ptr[2]=b2[col_start+2]; out_ptr[3]=b3[col_start+3];
        } else {
            longlong4 iv0=idx_base[0], iv1=idx_base[1];
            const scalar_t* b0=&src[iv0.x*dim_size], *b1=&src[iv0.y*dim_size];
            const scalar_t* b2=&src[iv0.z*dim_size], *b3=&src[iv0.w*dim_size];
            const scalar_t* b4=&src[iv1.x*dim_size], *b5=&src[iv1.y*dim_size];
            const scalar_t* b6=&src[iv1.z*dim_size], *b7=&src[iv1.w*dim_size];
            out_ptr[0]=b0[col_start]; out_ptr[1]=b1[col_start+1];
            out_ptr[2]=b2[col_start+2]; out_ptr[3]=b3[col_start+3];
            out_ptr[4]=b4[col_start+4]; out_ptr[5]=b5[col_start+5];
            out_ptr[6]=b6[col_start+6]; out_ptr[7]=b7[col_start+7];
        }
        float4* dst_vec_ptr = reinterpret_cast<float4*>(
            __builtin_assume_aligned(&dst[row * dim_size + col_start], 16));
        *dst_vec_ptr = out_vec;
    }
}

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
    int64_t count) 
{
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
    int64_t count) 
{
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <typename scalar_t>
__global__ void special_embedding_small_dim(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t index_stride) 
{
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_indices) return;
    int64_t src_row = index[row_idx * index_stride];
    const scalar_t* src_ptr = weight + src_row * embedding_dim;
    scalar_t* dst_ptr = dst + row_idx * embedding_dim;
    
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
    if (aligned && embedding_dim >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, embedding_dim);
    } else {
        copy_scalar<scalar_t>(dst_ptr, src_ptr, embedding_dim);
    }
}

template <typename scalar_t>
__global__ void special_embedding_large_batch(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t index_stride,
    int64_t rows_per_block) 
{
    int64_t base_row = blockIdx.x * rows_per_block;
    if (base_row >= num_indices) return;
    int64_t end_row = std::min(base_row + rows_per_block, num_indices);
    constexpr int kElems = VecConfig<scalar_t>::kElements;
    
    for (int64_t r = base_row; r < end_row; ++r) {
        int64_t src_row = index[r * index_stride];
        const scalar_t* src_ptr = weight + src_row * embedding_dim;
        scalar_t* dst_ptr = dst + r * embedding_dim;
        bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                       (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
        if (aligned && embedding_dim >= kElems) {
            copy_128bit<scalar_t>(dst_ptr, src_ptr, embedding_dim);
        } else {
            copy_scalar<scalar_t>(dst_ptr, src_ptr, embedding_dim);
        }
    }
}

template <typename scalar_t>
__global__ void special_embedding_large_dim(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ index,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t index_stride,
    int64_t blocks_per_row) 
{
    int64_t row_idx = blockIdx.x / blocks_per_row;
    if (row_idx >= num_indices) return;
    int64_t sub_block = blockIdx.x % blocks_per_row;
    constexpr int kElems = VecConfig<scalar_t>::kElements;

    // Balance vector chunks across blocks, including the remainder.
    int64_t total_vecs = embedding_dim / kElems;
    int64_t base_vecs_per_block = total_vecs / blocks_per_row;
    int64_t remainder = total_vecs % blocks_per_row;

    int64_t my_vecs = base_vecs_per_block + (sub_block < remainder ? 1 : 0);
    if (my_vecs <= 0) return;

    int64_t offset;
    if (sub_block < remainder) {
        offset = sub_block * (base_vecs_per_block + 1) * kElems;
    } else {
        offset = (remainder * (base_vecs_per_block + 1) +
                  (sub_block - remainder) * base_vecs_per_block) * kElems;
    }
    int64_t actual_chunk = my_vecs * kElems;

    int64_t src_row = index[row_idx * index_stride];
    const scalar_t* src_ptr = weight + src_row * embedding_dim + offset;
    scalar_t* dst_ptr = dst + row_idx * embedding_dim + offset;

    bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % VEC_BYTES == 0) &&
                   (reinterpret_cast<uintptr_t>(dst_ptr) % VEC_BYTES == 0);
    if (aligned && actual_chunk >= kElems) {
        copy_128bit<scalar_t>(dst_ptr, src_ptr, actual_chunk);
    } else {
        copy_scalar<scalar_t>(dst_ptr, src_ptr, actual_chunk);
    }
}

template <typename scalar_t>
void launch_special_embedding(
    scalar_t* dst, const scalar_t* weight, const int64_t* index,
    int64_t num_indices, int64_t embedding_dim, int64_t index_stride, cudaStream_t stream)
{
    if (num_indices == 0) return;

    const int threads = 256;
    constexpr int64_t LARGE_BATCH_THRESHOLD = 4096;
    constexpr int64_t ROWS_PER_BLOCK_TARGET_BYTES = 65536;
    constexpr int64_t MAX_ROW_BYTES_FOR_LARGE_BATCH = 512;

    int64_t row_bytes = embedding_dim * sizeof(scalar_t);

    bool use_large_dim = false;
    int64_t blocks_per_row = 1;

    if (row_bytes > 65536) {
        use_large_dim = true;
        blocks_per_row = row_bytes / 8192;
        if (blocks_per_row < 2) blocks_per_row = 2;
    } else if (row_bytes > 16384) {
        use_large_dim = true;
        blocks_per_row = 2;
    }

    if (use_large_dim) {
        int64_t total_blocks = num_indices * blocks_per_row;
        special_embedding_large_dim<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        index_stride,
        blocks_per_row);
    } else if (sizeof(scalar_t) == 2
             && num_indices > LARGE_BATCH_THRESHOLD
             && row_bytes > MAX_ROW_BYTES_FOR_LARGE_BATCH
             && row_bytes <= 4096) {
        constexpr int64_t MEDIUM_BATCH_TARGET_BYTES = 8192;
        int64_t rows_per_block = std::max(int64_t(1), MEDIUM_BATCH_TARGET_BYTES / row_bytes);
        rows_per_block = std::min(rows_per_block, num_indices);
        int64_t total_blocks = (num_indices + rows_per_block - 1) / rows_per_block;
        special_embedding_large_batch<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        index_stride,
        rows_per_block);
    } else if (num_indices > LARGE_BATCH_THRESHOLD
             && row_bytes <= MAX_ROW_BYTES_FOR_LARGE_BATCH) {
        int64_t rows_per_block = std::max(int64_t(1), ROWS_PER_BLOCK_TARGET_BYTES / row_bytes);
        rows_per_block = std::min(rows_per_block, num_indices);
        int64_t total_blocks = (num_indices + rows_per_block - 1) / rows_per_block;
        special_embedding_large_batch<scalar_t><<<dim3(total_blocks), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        index_stride,
        rows_per_block);
    } else {
        special_embedding_small_dim<scalar_t><<<dim3(num_indices), dim3(threads), 0, stream>>>(
        dst,
        weight,
        index,
        num_indices,
        embedding_dim,
        index_stride);
    }
}

namespace at {
namespace native {

torch::Tensor gather_opt(
    torch::Tensor& dst,
    torch::Tensor& index,
    torch::Tensor& src,
    bool special
) {
    const int dst_batch_size = static_cast<int>(dst.size(0));
    const int dim_size = static_cast<int>(dst.size(1));

    int kElementsPerVec = (src.scalar_type() == at::ScalarType::Float || 
                           src.scalar_type() == at::ScalarType::Double) ? 4 : 8;
    TORCH_CHECK(dim_size % kElementsPerVec == 0, 
                "dim_size must be multiple of ", kElementsPerVec);

    const bool is_row_uniform = (index.dim() == 2 && index.stride(1) == 0);
    const bool use_special = special || is_row_uniform;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (use_special) {
        int64_t num_indices = dst_batch_size;
        int64_t embedding_dim = dim_size;
        int64_t index_stride = index.stride(0);

        auto scalar_type = src.scalar_type();
        if (scalar_type == at::ScalarType::Float) {
            launch_special_embedding<float>(
                dst.data_ptr<float>(), src.data_ptr<float>(),
                index.data_ptr<int64_t>(), num_indices, embedding_dim, index_stride, stream);
        } else if (scalar_type == at::ScalarType::Half) {
            launch_special_embedding<at::Half>(
                dst.data_ptr<at::Half>(), src.data_ptr<at::Half>(),
                index.data_ptr<int64_t>(), num_indices, embedding_dim, index_stride, stream);
        } else if (scalar_type == at::ScalarType::BFloat16) {
            launch_special_embedding<at::BFloat16>(
                dst.data_ptr<at::BFloat16>(), src.data_ptr<at::BFloat16>(),
                index.data_ptr<int64_t>(), num_indices, embedding_dim, index_stride, stream);
        } else {
            TORCH_CHECK(false, "Unsupported dtype for special=True: ", scalar_type);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return dst;
    }

    int vec_dim = dim_size / kElementsPerVec;
    bool use_wide_kernel = (dim_size >= 65536);
    dim3 block, grid;
    if (use_wide_kernel) {
        block = dim3(256, 1, 1);
        grid = dim3((vec_dim + 255) / 256, dst_batch_size, 1);
    } else {
        block = dim3(std::min(16, vec_dim), std::min(16, dst_batch_size), 1);
        while (block.x * block.y < 64 && block.x < vec_dim) block.x *= 2;
        while (block.x * block.y < 64 && block.y < dst_batch_size) block.y *= 2;
        grid = dim3((vec_dim + block.x - 1) / block.x,
                    (dst_batch_size + block.y - 1) / block.y, 1);
    }
    if (use_wide_kernel) {
        if (src.scalar_type() == at::ScalarType::Float)
            gather_kernel_wide<float><<<grid, block, 0, stream>>>(src.data_ptr<float>(), index.data_ptr<int64_t>(), dst.data_ptr<float>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::Half)
            gather_kernel_wide<at::Half><<<grid, block, 0, stream>>>(src.data_ptr<at::Half>(), index.data_ptr<int64_t>(), dst.data_ptr<at::Half>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::BFloat16)
            gather_kernel_wide<at::BFloat16><<<grid, block, 0, stream>>>(src.data_ptr<at::BFloat16>(), index.data_ptr<int64_t>(), dst.data_ptr<at::BFloat16>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::Double)
            gather_kernel_wide<double><<<grid, block, 0, stream>>>(src.data_ptr<double>(), index.data_ptr<int64_t>(), dst.data_ptr<double>(), dst_batch_size, dim_size);
    } else {
        if (src.scalar_type() == at::ScalarType::Float)
            gather_kernel_general<float><<<grid, block, 0, stream>>>(src.data_ptr<float>(), index.data_ptr<int64_t>(), dst.data_ptr<float>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::Half)
            gather_kernel_general<at::Half><<<grid, block, 0, stream>>>(src.data_ptr<at::Half>(), index.data_ptr<int64_t>(), dst.data_ptr<at::Half>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::BFloat16)
            gather_kernel_general<at::BFloat16><<<grid, block, 0, stream>>>(src.data_ptr<at::BFloat16>(), index.data_ptr<int64_t>(), dst.data_ptr<at::BFloat16>(), dst_batch_size, dim_size);
        else if (src.scalar_type() == at::ScalarType::Double)
            gather_kernel_general<double><<<grid, block, 0, stream>>>(src.data_ptr<double>(), index.data_ptr<int64_t>(), dst.data_ptr<double>(), dst_batch_size, dim_size);
    }
    return dst;
}

}  // namespace native
}  // namespace at
