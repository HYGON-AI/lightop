// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "hip/hip_runtime.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/core/ScalarType.h>
#include <hip/hip_bf16.h>
#include <algorithm> // For std::min

namespace at {
namespace native {

// 用于向量化内存访问的对齐向量类型
template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
    T val[N];
};


enum class KVCacheDataType {
    kAuto, // 直接复制，不量化
    kInt8, // INT8 量化
};



// 用于处理 batch_size > 1 的情况。每个块处理一个请求。
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_batch_kernel(
    const scalar_t* __restrict__ packed_qkv, cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ q_lens, const int32_t* __restrict__ accum_q_lens,
    const int32_t* __restrict__ cache_lens, const int32_t* __restrict__ cache_slot_ids, 
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int q_head_num, int kv_head_num, int head_dim, int max_kv_len
) {
    const int batch_idx = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_len = q_lens[batch_idx];
    if (q_len == 0) return;

    const int q_token_offset = accum_q_lens[batch_idx];
    const int cache_start_pos = cache_lens[batch_idx];
    const int cache_slot_id = cache_slot_ids[batch_idx];
    const int total_head_num = q_head_num + 2 * kv_head_num;
    const int total_elements_to_process = q_len * kv_head_num * head_dim;

    for (int i = threadIdx.x * Vec_Size; i < total_elements_to_process; i += blockDim.x * Vec_Size) {
        const int token_idx_in_q = i / (kv_head_num * head_dim);
        const int remainder = i % (kv_head_num * head_dim);
        const int head_idx_in_kv = remainder / head_dim;
        const int val_idx_in_head = remainder % head_dim;

        const int64_t abs_token_idx = q_token_offset + token_idx_in_q;
        const int64_t k_head_idx = q_head_num + head_idx_in_kv;
        const int64_t v_head_idx = q_head_num + kv_head_num + head_idx_in_kv;
        const int64_t k_src_offset = abs_token_idx * total_head_num * head_dim + k_head_idx * head_dim + val_idx_in_head;
        const int64_t v_src_offset = abs_token_idx * total_head_num * head_dim + v_head_idx * head_dim + val_idx_in_head;

        const int64_t dst_token_idx = cache_start_pos + token_idx_in_q;
        const int64_t dst_offset = (int64_t)cache_slot_id * kv_head_num * max_kv_len * head_dim +
                                     (int64_t)head_idx_in_kv * max_kv_len * head_dim +
                                     (int64_t)dst_token_idx * head_dim +
                                     val_idx_in_head;

        const scalar_vector_t  k_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + k_src_offset);
        const scalar_vector_t  v_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + v_src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);

        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out; *v_dst_vec_ptr = v_vec_out;
        }
    }
}


// 用于处理 batch_size=1, q_len=1 的解码阶段。每个块处理一个 KV 头。
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_head_centric_kernel(
    const scalar_t* __restrict__ packed_qkv, cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ q_lens, const int32_t* __restrict__ accum_q_lens,
    const int32_t* __restrict__ cache_lens, const int32_t* __restrict__ cache_slot_ids, 
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int q_head_num, int kv_head_num, int head_dim, int max_kv_len
) {
    const int head_idx_in_kv = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_len = 1; // 此 Kernel 假定 q_len == 1
    const int q_token_offset = accum_q_lens[0];
    const int cache_start_pos = cache_lens[0];
    const int cache_slot_id = cache_slot_ids[0];
    const int total_head_num = q_head_num + 2 * kv_head_num;
    //一个block处理一个head_dim
    for (int i = threadIdx.x * Vec_Size; i < head_dim; i += blockDim.x * Vec_Size) {
        const int val_idx_in_head = i;
        const int64_t k_head_idx = q_head_num + head_idx_in_kv;
        const int64_t v_head_idx = q_head_num + kv_head_num + head_idx_in_kv;
        const int64_t k_src_offset = (int64_t)q_token_offset * total_head_num * head_dim + k_head_idx * head_dim + val_idx_in_head;
        const int64_t v_src_offset = (int64_t)q_token_offset * total_head_num * head_dim + v_head_idx * head_dim + val_idx_in_head;
        const int64_t dst_offset = (int64_t)cache_slot_id * kv_head_num * max_kv_len * head_dim +
                                     (int64_t)head_idx_in_kv * max_kv_len * head_dim +
                                     (int64_t)cache_start_pos * head_dim +
                                     val_idx_in_head;
        const scalar_vector_t k_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + k_src_offset);
        const scalar_vector_t v_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + v_src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);
        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out; *v_dst_vec_ptr = v_vec_out;
        }
    }
}

//Kernel 3: 单请求预填充优化 (q_len>1, Token-Centric)
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_token_centric_kernel(
    const scalar_t* __restrict__ packed_qkv, cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ q_lens, const int32_t* __restrict__ accum_q_lens,
    const int32_t* __restrict__ cache_lens, const int32_t* __restrict__ cache_slot_ids, 
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int q_head_num, int kv_head_num, int head_dim, int max_kv_len
) {
    const int token_idx_in_q = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_token_offset = accum_q_lens[0];
    const int cache_start_pos = cache_lens[0];
    const int cache_slot_id = cache_slot_ids[0];
    const int total_head_num = q_head_num + 2 * kv_head_num;
    const int total_elements_per_token = kv_head_num * head_dim;
    //一个block处理一个token
    for (int i = threadIdx.x * Vec_Size; i < total_elements_per_token; i += blockDim.x * Vec_Size) {
        const int head_idx_in_kv = i / head_dim;
        const int val_idx_in_head = i % head_dim;

        const int64_t abs_token_idx = q_token_offset + token_idx_in_q;
        const int64_t k_head_idx = q_head_num + head_idx_in_kv;
        const int64_t v_head_idx = q_head_num + kv_head_num + head_idx_in_kv;
        const int64_t k_src_offset = abs_token_idx * total_head_num * head_dim + k_head_idx * head_dim + val_idx_in_head;
        const int64_t v_src_offset = abs_token_idx * total_head_num * head_dim + v_head_idx * head_dim + val_idx_in_head;

        const int64_t dst_token_idx = cache_start_pos + token_idx_in_q;
        const int64_t dst_offset = (int64_t)cache_slot_id * kv_head_num * max_kv_len * head_dim +
                                     (int64_t)head_idx_in_kv * max_kv_len * head_dim +
                                     (int64_t)dst_token_idx * head_dim +
                                     val_idx_in_head;
        const scalar_vector_t k_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + k_src_offset);
        const scalar_vector_t v_vec_in = *reinterpret_cast<const scalar_vector_t*>(packed_qkv + v_src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);
        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out; *v_dst_vec_ptr = v_vec_out;
        }
    }
}

void store_kv_cache(
    torch::Tensor packed_qkv, torch::Tensor& K_cache, torch::Tensor& V_cache,
    torch::Tensor q_lens, torch::Tensor accum_q_lens, torch::Tensor cache_lens, torch::Tensor cache_slot_ids,
    torch::Tensor k_scale, torch::Tensor v_scale, const std::string& kv_cache_dtype_str,
    int q_head_num, int kv_head_num
) {
    const int batch_size = q_lens.size(0);
    const int head_dim = packed_qkv.size(2);
    const int max_kv_len = K_cache.size(2);

    TORCH_CHECK(batch_size > 0, "Batch size must be positive.");
    TORCH_CHECK(packed_qkv.dim() == 3, "packed_qkv must be [num_tokens, total_heads, head_dim]");
    TORCH_CHECK(K_cache.dim() == 4, "K_cache must be [batch_size, kv_head_num, max_kv_len, head_dim]");
    TORCH_CHECK(k_scale.dim() == 2, "k_scale must be [kv_head_num, head_dim]");

    int vec_size = 1;
    if (kv_cache_dtype_str != "auto" && head_dim > 0 && head_dim % 16 == 0) { vec_size = 16; }
    else if (head_dim > 0 && head_dim % 8 == 0) { vec_size = 8; }
    else if (head_dim > 0 && head_dim % 4 == 0) { vec_size = 4; }
    else if (head_dim > 0 && head_dim % 2 == 0) { vec_size = 2; }

    #define LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, VEC_SIZE) \
        kernel_name<scalar_t, cache_t, kv_dt, VEC_SIZE><<<grid, block>>>( \
            packed_qkv.data_ptr<scalar_t>(), K_cache.data_ptr<cache_t>(), V_cache.data_ptr<cache_t>(), \
            q_lens.data_ptr<int32_t>(), accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>(), cache_slot_ids.data_ptr<int32_t>(), \
            k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), \
            q_head_num, kv_head_num, head_dim, max_kv_len)

    #define LAUNCH_KERNEL(kernel_name, grid, block, scalar_t, cache_t, kv_dt) \
    do { \
        if (vec_size == 16) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 16); } \
        else if (vec_size == 8) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 8); } \
        else if (vec_size == 4) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 4); } \
        else if (vec_size == 2) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 2); } \
        else { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 1); } \
    } while(0)

    if (batch_size == 1) {
        const int q_len_val = q_lens.item<int32_t>();
        if (q_len_val == 1) { // 单请求解码 (q_len=1)
            const dim3 grid(kv_head_num);
            const int threads_needed = (head_dim + vec_size - 1) / vec_size;
            const dim3 block(std::min(1024, threads_needed));
            if (kv_cache_dtype_str == "auto") {
                if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_head_centric_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto); }
                else { TORCH_CHECK(false, "Unsupported dtype for auto mode"); }
            } else if (kv_cache_dtype_str == "int8") {
                if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_head_centric_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8); }
                else { TORCH_CHECK(false, "Unsupported input dtype for int8 mode"); }
            } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
        } else { // 单请求预填充 (q_len>1)
            const dim3 grid(q_len_val);
            const int elements_per_token = kv_head_num * head_dim;
            const int threads_needed = (elements_per_token + vec_size - 1) / vec_size;
            const dim3 block(std::min(1024, threads_needed));
            if (kv_cache_dtype_str == "auto") {
                if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_token_centric_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto); }
                else { TORCH_CHECK(false, "Unsupported dtype for auto mode"); }
            } else if (kv_cache_dtype_str == "int8") {
                if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_token_centric_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8); }
                else { TORCH_CHECK(false, "Unsupported input dtype for int8 mode"); }
            } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
        }
    } else { // 批处理 (batch_size > 1)
        const dim3 grid(batch_size);
        const dim3 block(1024);
        if (kv_cache_dtype_str == "auto") {
            if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_batch_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto); }
            else { TORCH_CHECK(false, "Unsupported dtype for auto mode"); }
        } else if (kv_cache_dtype_str == "int8") {
            if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_batch_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8); }
            else { TORCH_CHECK(false, "Unsupported input dtype for int8 mode"); }
        } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
    }

    #undef LAUNCH_KERNEL_VEC
    #undef LAUNCH_KERNEL
}

// Kernel 1: 用于处理批处理 (batch_size > 1) 的情况。每个块处理一个请求。
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_paged_batch_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ q_lens, const int32_t* __restrict__ accum_q_lens,
    const int32_t* __restrict__ cache_lens, const int32_t* __restrict__ block_table,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int kv_head_num, int head_dim, int block_size, int max_blocks_per_seq
) {
    const int batch_idx = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_len = q_lens[batch_idx];
    if (q_len == 0) return;

    const int q_token_offset = accum_q_lens[batch_idx];
    const int cache_start_pos = cache_lens[batch_idx];
    const int total_elements_to_process = q_len * kv_head_num * head_dim;


    for (int i = threadIdx.x * Vec_Size; i < total_elements_to_process; i += blockDim.x * Vec_Size) {
       
        const int token_idx_in_q = i / (kv_head_num * head_dim);
        const int remainder = i % (kv_head_num * head_dim);
        const int head_idx_in_kv = remainder / head_dim;
        const int val_idx_in_head = remainder % head_dim;

 
        const int64_t abs_token_idx = q_token_offset + token_idx_in_q;
        const int64_t src_offset = abs_token_idx * kv_head_num * head_dim +
                                   head_idx_in_kv * head_dim +
                                   val_idx_in_head;


        const int64_t dst_token_idx = cache_start_pos + token_idx_in_q;

        const int logical_block_idx = dst_token_idx / block_size;
        const int offset_in_block = dst_token_idx % block_size;

        const int physical_block_id = block_table[batch_idx * max_blocks_per_seq + logical_block_idx];

 
        const int64_t dst_offset = (int64_t)physical_block_id * kv_head_num * block_size * head_dim +
                                   (int64_t)head_idx_in_kv * block_size * head_dim +
                                   (int64_t)offset_in_block * head_dim +
                                   val_idx_in_head;

        const scalar_vector_t k_vec_in = *reinterpret_cast<const scalar_vector_t*>(key + src_offset);
        const scalar_vector_t v_vec_in = *reinterpret_cast<const scalar_vector_t*>(value + src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);

        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        }
    }
}

// Kernel 2: 用于处理单请求解码阶段 (batch_size=1, q_len=1)。每个块处理一个KV头。
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_paged_head_centric_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ accum_q_lens, const int32_t* __restrict__ cache_lens,
    const int32_t* __restrict__ block_table,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int kv_head_num, int head_dim, int block_size, int max_blocks_per_seq
) {
    const int head_idx_in_kv = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_token_offset = accum_q_lens[0];
    const int cache_start_pos = cache_lens[0]; 

    const int logical_block_idx = cache_start_pos / block_size;
    const int offset_in_block = cache_start_pos % block_size;

    const int physical_block_id = block_table[logical_block_idx];

    // 每个块处理一个KV头的所有维度
    for (int i = threadIdx.x * Vec_Size; i < head_dim; i += blockDim.x * Vec_Size) {
        const int val_idx_in_head = i;

        
        const int64_t src_offset = (int64_t)q_token_offset * kv_head_num * head_dim +
                                   (int64_t)head_idx_in_kv * head_dim +
                                   val_idx_in_head;

   
        const int64_t dst_offset = (int64_t)physical_block_id * kv_head_num * block_size * head_dim +
                                   (int64_t)head_idx_in_kv * block_size * head_dim +
                                   (int64_t)offset_in_block * head_dim +
                                   val_idx_in_head;

       
        const scalar_vector_t k_vec_in = *reinterpret_cast<const scalar_vector_t*>(key + src_offset);
        const scalar_vector_t v_vec_in = *reinterpret_cast<const scalar_vector_t*>(value + src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);

        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        }
    }
}

// Kernel 3: 用于处理单请求预填充 
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_paged_token_centric_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ accum_q_lens, const int32_t* __restrict__ cache_lens,
    const int32_t* __restrict__ block_table,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int kv_head_num, int head_dim, int block_size, int max_blocks_per_seq
) {
    const int token_idx_in_q = blockIdx.x;
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t =  Vector<cache_t, Vec_Size>;

    const int q_token_offset = accum_q_lens[0];
    const int cache_start_pos = cache_lens[0];
    const int total_elements_per_token = kv_head_num * head_dim;

    const int64_t dst_token_idx = cache_start_pos + token_idx_in_q;
    const int logical_block_idx = dst_token_idx / block_size;
    const int offset_in_block = dst_token_idx % block_size;
    const int physical_block_id = block_table[logical_block_idx];


    for (int i = threadIdx.x * Vec_Size; i < total_elements_per_token; i += blockDim.x * Vec_Size) {
        const int head_idx_in_kv = i / head_dim;
        const int val_idx_in_head = i % head_dim;


        const int64_t abs_token_idx = q_token_offset + token_idx_in_q;
        const int64_t src_offset = abs_token_idx * kv_head_num * head_dim +
                                   head_idx_in_kv * head_dim +
                                   val_idx_in_head;

    
        const int64_t dst_offset = (int64_t)physical_block_id * kv_head_num * block_size * head_dim +
                                   (int64_t)head_idx_in_kv * block_size * head_dim +
                                   (int64_t)offset_in_block * head_dim +
                                   val_idx_in_head;

       
        const scalar_vector_t k_vec_in = *reinterpret_cast<const scalar_vector_t*>(key + src_offset);
        const scalar_vector_t v_vec_in = *reinterpret_cast<const scalar_vector_t*>(value + src_offset);
        auto* k_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(k_cache + dst_offset);
        auto* v_dst_vec_ptr = reinterpret_cast<cache_vector_t*>(v_cache + dst_offset);

        if constexpr (kv_dt == KVCacheDataType::kAuto) {
            *k_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&k_vec_in);
            *v_dst_vec_ptr = *reinterpret_cast<const cache_vector_t*>(&v_vec_in);
        } else if constexpr (kv_dt == KVCacheDataType::kInt8) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const int32_t k_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
                const int32_t v_quantized_int = static_cast<int32_t>(roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
                k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
                v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        }
    }
}


void store_kv_cache_paged(
    const torch::Tensor& key, const torch::Tensor& value,
    torch::Tensor& K_cache, torch::Tensor& V_cache,
    const torch::Tensor& q_lens, const torch::Tensor& accum_q_lens,
    const torch::Tensor& cache_lens, const torch::Tensor& block_table,
    const torch::Tensor& k_scale, const torch::Tensor& v_scale,
    const std::string& kv_cache_dtype_str
) {
    const int batch_size = q_lens.size(0);
    const int kv_head_num = K_cache.size(1);
    const int block_size = K_cache.size(2);
    const int head_dim = K_cache.size(3);
    const int max_blocks_per_seq = block_table.size(1);

    TORCH_CHECK(batch_size > 0, "Batch size must be positive.");
    TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, kv_head_num, head_dim]");
    TORCH_CHECK(key.size(1) == kv_head_num && key.size(2) == head_dim, "key shape mismatch");
    TORCH_CHECK(value.sizes() == key.sizes(), "key and value must have the same shape");
    TORCH_CHECK(K_cache.dim() == 4, "K_cache must be [max_block_num, kv_head_num, block_size, head_dim]");
    TORCH_CHECK(V_cache.sizes() == K_cache.sizes(), "K_cache and V_cache must have the same shape");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch_size, "block_table must be [batch_size, max_blocks_per_seq]");
    TORCH_CHECK(k_scale.dim() == 2, "k_scale must be [kv_head_num, head_dim]");

    int vec_size = 1;
    if (head_dim > 0 && head_dim % 16 == 0) { vec_size = 16; }
    else if (head_dim > 0 && head_dim % 8 == 0) { vec_size = 8; }
    else if (head_dim > 0 && head_dim % 4 == 0) { vec_size = 4; }
    else if (head_dim > 0 && head_dim % 2 == 0) { vec_size = 2; }

    #define LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, VEC_SIZE, ...) \
        kernel_name<scalar_t, cache_t, kv_dt, VEC_SIZE><<<grid, block>>>( \
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), \
            K_cache.data_ptr<cache_t>(), V_cache.data_ptr<cache_t>(), \
            __VA_ARGS__, \
            block_table.data_ptr<int32_t>(), \
            k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), \
            kv_head_num, head_dim, block_size, max_blocks_per_seq)

    #define LAUNCH_KERNEL(kernel_name, grid, block, scalar_t, cache_t, kv_dt, ...) \
    do { \
        if (vec_size == 16) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 16, __VA_ARGS__); } \
        else if (vec_size == 8) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 8, __VA_ARGS__); } \
        else if (vec_size == 4) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 4, __VA_ARGS__); } \
        else if (vec_size == 2) { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 2, __VA_ARGS__); } \
        else { LAUNCH_KERNEL_VEC(kernel_name, grid, block, scalar_t, cache_t, kv_dt, 1, __VA_ARGS__); } \
    } while(0)

    if (batch_size == 1) {
        const int q_len_val = q_lens.item<int32_t>();
        const auto common_args = { accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>() };
        if (q_len_val == 1) {
            const dim3 grid(kv_head_num);
            const int threads_needed = (head_dim + vec_size - 1) / vec_size;
            const dim3 block(std::min(1024, threads_needed));
            if (kv_cache_dtype_str == "auto") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported dtype for auto mode");
                LAUNCH_KERNEL(store_kv_cache_paged_head_centric_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
            } else if (kv_cache_dtype_str == "int8") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for int8 mode");
                LAUNCH_KERNEL(store_kv_cache_paged_head_centric_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
            } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
        } else { 
            const dim3 grid(q_len_val);
            const int elements_per_token = kv_head_num * head_dim;
            const int threads_needed = (elements_per_token + vec_size - 1) / vec_size;
            const dim3 block(std::min(1024, threads_needed));
            if (kv_cache_dtype_str == "auto") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported dtype for auto mode");
                LAUNCH_KERNEL(store_kv_cache_paged_token_centric_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
            } else if (kv_cache_dtype_str == "int8") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for int8 mode");
                LAUNCH_KERNEL(store_kv_cache_paged_token_centric_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
            } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
        }
    } else { 
        const dim3 grid(batch_size);
        const dim3 block(1024); 
        if (kv_cache_dtype_str == "auto") {
            TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported dtype for auto mode");
            LAUNCH_KERNEL(store_kv_cache_paged_batch_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto, q_lens.data_ptr<int32_t>(), accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
        } else if (kv_cache_dtype_str == "int8") {
            TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for int8 mode");
            LAUNCH_KERNEL(store_kv_cache_paged_batch_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8, q_lens.data_ptr<int32_t>(), accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
        } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
    }

    #undef LAUNCH_KERNEL_VEC
    #undef LAUNCH_KERNEL
}

} // namespace native
} // namespace at
