// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "hip/hip_runtime.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/core/ScalarType.h>
#include <hip/hip_bf16.h>
#include <algorithm> // For std::min
#include <cstdint>

#if defined(USE_ROCM) && __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#define LIGHTOP_HAS_HIP_FP8_HW_CVT 1
#endif

namespace at {
namespace native {

// 用于向量化内存访问的对齐向量类型
template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
    T val[N];
};



enum class KVCacheDataType {
    kAuto,    // 直接复制，不量化
    kInt8,    // INT8 量化
    kFp8E4M3, // float8_e4m3fn 静态量化（对齐 custom_ops）
};

__device__ __forceinline__ uint32_t fp32_bits_u32(float f) {
    return __builtin_bit_cast(uint32_t, f);
}

// FP32 -> FP8 E4M3, round nearest-even, satfinite (same numeric style as CUTLASS / custom_ops).
__device__ __forceinline__ uint8_t float_to_fp8_e4m3_rne(float flt) {
    uint32_t s = fp32_bits_u32(flt);
    uint8_t sign = uint8_t((s >> 24) & 0x80u);
    int32_t exp = int32_t((s >> 23) & 0xffu) - 127;
    int32_t mantissa = int32_t(s & 0x7fffffu);
    uint8_t u = 0;

    constexpr uint8_t kF8_NaN = 0x7f;
    constexpr uint8_t FP8_MAX_FLT = 0x7e;
    constexpr int FP8_MIN_EXPONENT = -6;
    constexpr int FP8_MAX_EXPONENT = 7;
    constexpr int FP8_EXPONENT_BIAS = 7;
    constexpr int FP8_NUM_MANTISSA_BITS = 3;
    constexpr int FP8_NUM_EXPONENT_BITS = 4;
    constexpr uint8_t FP8_EXPONENT_MASK = (1u << FP8_NUM_EXPONENT_BITS) - 1u;
    constexpr uint8_t FP8_MANTISSA_MASK = (1u << FP8_NUM_MANTISSA_BITS) - 1u;
    constexpr int FP32_NUM_MANTISSA_BITS = 23;

    if (flt != flt) {
        return uint8_t(sign | kF8_NaN);
    }
    uint32_t abs_bits = s & 0x7fffffffu;
    if (abs_bits == 0x7f800000u) {
        return uint8_t(sign | FP8_MAX_FLT);
    }

    bool skip_sign = false;
    bool may_be_nan = false;
    int sticky_bit = 0;

    if ((exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT)) {
        exp = exp + FP8_EXPONENT_BIAS;
        u = uint8_t((uint32_t(exp) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
        u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
    } else if (exp < FP8_MIN_EXPONENT) {
        int rshift = (FP8_MIN_EXPONENT - exp);
        if (rshift < 32) {
            mantissa |= (1 << FP32_NUM_MANTISSA_BITS);
            sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0) ? 1 : 0;
            mantissa = (mantissa >> rshift);
            u = uint8_t((mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
        } else {
            mantissa = 0;
            u = 0;
        }
    } else {
        if (exp == (FP8_MAX_EXPONENT + 1)) {
            uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            if (mantissa_tmp < FP8_MANTISSA_MASK) {
                exp = exp + FP8_EXPONENT_BIAS;
                u = uint8_t(uint32_t(exp) << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
                may_be_nan = (mantissa_tmp == (FP8_MANTISSA_MASK - 1));
            } else {
                return uint8_t(sign | FP8_MAX_FLT);
            }
        } else {
            return uint8_t(sign | FP8_MAX_FLT);
        }
    }

    const int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
    int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
    sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0) ? 1 : 0;

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
        u = uint8_t(u + 1);
        if (may_be_nan) {
            skip_sign = true;
        }
    }

    if (u > FP8_MAX_FLT) {
        u = uint8_t(sign | FP8_MAX_FLT);
    }

    if (!skip_sign) {
        u |= sign;
    }

    return u;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3_scaled_fast(float q) {
#if defined(LIGHTOP_HAS_HIP_FP8_HW_CVT)
    const __hip_fp8_storage_t bits = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
    return static_cast<uint8_t>(bits);
#else
    return float_to_fp8_e4m3_rne(q);
#endif
}


// Vec_Size is always 1/2/4/8/16 in launches: multiply -> shift at compile time.
template <int Vec_Size>
__device__ __forceinline__ constexpr int store_kv_vec_shift() {
    static_assert(Vec_Size >= 1 && (Vec_Size & (Vec_Size - 1)) == 0,
                  "Vec_Size must be a power of 2");
    if constexpr (Vec_Size == 1) return 0;
    else if constexpr (Vec_Size == 2) return 1;
    else if constexpr (Vec_Size == 4) return 2;
    else if constexpr (Vec_Size == 8) return 3;
    else if constexpr (Vec_Size == 16) return 4;
    else return 0;
}

// Runtime positive dim: if power-of-2, / and % become >> and & (safe & exact).
__device__ __forceinline__ void store_kv_div_mod_dim(int idx, int dim, int& q, int& r) {
    if ((dim & (dim - 1)) == 0) {
        const int shift = __builtin_ctz(static_cast<unsigned>(dim));
        q = idx >> shift;
        r = idx & (dim - 1);
    } else {
        q = idx / dim;
        r = idx % dim;
    }
}

__device__ __forceinline__ void store_kv_div_mod_dim64(int64_t idx, int dim, int& q, int& r) {
    if ((dim & (dim - 1)) == 0) {
        const int shift = __builtin_ctz(static_cast<unsigned>(dim));
        q = static_cast<int>(idx >> shift);
        r = static_cast<int>(idx & static_cast<int64_t>(dim - 1));
    } else {
        q = static_cast<int>(idx / dim);
        r = static_cast<int>(idx % dim);
    }
}


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
        const int elems_per_tok = kv_head_num * head_dim;
        int token_idx_in_q, remainder;
        store_kv_div_mod_dim(i, elems_per_tok, token_idx_in_q, remainder);
        int head_idx_in_kv, val_idx_in_head;
        store_kv_div_mod_dim(remainder, head_dim, head_idx_in_kv, val_idx_in_head);

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
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
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
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
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
        int head_idx_in_kv, val_idx_in_head;
        store_kv_div_mod_dim(i, head_dim, head_idx_in_kv, val_idx_in_head);

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
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
            }
            *k_dst_vec_ptr = k_vec_out;
            *v_dst_vec_ptr = v_vec_out;
        }
    }
}


// Prefill flat fused (batch=1): 1D grid over q_len*kvh*(hd/Vec).
// Used when token-centric would launch a tiny block (TP-sharded kvh=1/2).
// Keeps lightop dispatch; no fixed-nt template explosion.
template <typename scalar_t, typename cache_t, KVCacheDataType kv_dt, int Vec_Size>
__global__ void store_kv_cache_flat_prefill_kernel(
    const scalar_t* __restrict__ packed_qkv, cache_t* __restrict__ k_cache, cache_t* __restrict__ v_cache,
    const int32_t* __restrict__ accum_q_lens,
    const int32_t* __restrict__ cache_lens, const int32_t* __restrict__ cache_slot_ids,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int q_head_num, int kv_head_num, int head_dim, int max_kv_len, int q_len
) {
    using scalar_vector_t = Vector<scalar_t, Vec_Size>;
    using cache_vector_t = Vector<cache_t, Vec_Size>;

    const int work_per_head = head_dim / Vec_Size;
    const int64_t total_work = (int64_t)q_len * kv_head_num * work_per_head;
    const int64_t linear_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_id >= total_work) return;

    const int q_token_offset = accum_q_lens[0];
    const int cache_start_pos = cache_lens[0];
    const int cache_slot_id = cache_slot_ids[0];
    const int total_head_num = q_head_num + 2 * kv_head_num;

    int token_idx_in_q, head_idx_in_kv, d_chunk;
    {
        int rem;
        store_kv_div_mod_dim64(linear_id, work_per_head, rem, d_chunk);
        store_kv_div_mod_dim(rem, kv_head_num, token_idx_in_q, head_idx_in_kv);
    }
    const int val_idx_in_head = d_chunk * Vec_Size;

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
            const int32_t k_quantized_int = static_cast<int32_t>(
                roundf(static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx]));
            const int32_t v_quantized_int = static_cast<int32_t>(
                roundf(static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx]));
            k_vec_out.val[j] = static_cast<cache_t>(k_quantized_int);
            v_vec_out.val[j] = static_cast<cache_t>(v_quantized_int);
        }
        *k_dst_vec_ptr = k_vec_out;
        *v_dst_vec_ptr = v_vec_out;
    } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
        cache_vector_t k_vec_out, v_vec_out;
        #pragma unroll
        for (int j = 0; j < Vec_Size; ++j) {
            const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
            const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
            const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
            k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
            v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
        }
        *k_dst_vec_ptr = k_vec_out;
        *v_dst_vec_ptr = v_vec_out;
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
            packed_qkv.data_ptr<scalar_t>(), reinterpret_cast<cache_t*>(K_cache.data_ptr()), reinterpret_cast<cache_t*>(V_cache.data_ptr()), \
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
        // batch==1: packed_qkv.size(0) == q_len; avoid q_lens.item() DtoH sync
        const int q_len_val = static_cast<int>(packed_qkv.size(0));
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
            } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
                if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_head_centric_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3); }
                else { TORCH_CHECK(false, "Unsupported input dtype for fp8 mode"); }
            } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
        } else { // 单请求预填充 (q_len>1)
            const int elements_per_token = kv_head_num * head_dim;
            const int threads_needed = (elements_per_token + vec_size - 1) / vec_size;
            // token-centric under-occupies when kvh is small (TP shard kvh=1/2).
            const bool use_flat_prefill = (threads_needed < 64);
            if (use_flat_prefill) {
                // Quant: vec4 matches custom fused packing; bf16: vec8.
                int flat_vec = vec_size;
                if (kv_cache_dtype_str != "auto" && head_dim % 4 == 0) { flat_vec = 4; }
                else if (kv_cache_dtype_str == "auto" && head_dim % 8 == 0) { flat_vec = 8; }
                const int work_per_head = head_dim / flat_vec;
                const int64_t total_work = (int64_t)q_len_val * kv_head_num * work_per_head;
                int block_size = (q_len_val >= 4096) ? 512 : 256;
                const int num_blocks = (int)((total_work + block_size - 1) / block_size);
                const dim3 grid(num_blocks);
                const dim3 block(block_size);
                #define LAUNCH_FLAT(scalar_t, cache_t, kv_dt, VEC_SIZE) \
                    store_kv_cache_flat_prefill_kernel<scalar_t, cache_t, kv_dt, VEC_SIZE><<<grid, block>>>( \
                        packed_qkv.data_ptr<scalar_t>(), reinterpret_cast<cache_t*>(K_cache.data_ptr()), reinterpret_cast<cache_t*>(V_cache.data_ptr()), \
                        accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>(), cache_slot_ids.data_ptr<int32_t>(), \
                        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), \
                        q_head_num, kv_head_num, head_dim, max_kv_len, q_len_val)
                #define LAUNCH_FLAT_VEC(scalar_t, cache_t, kv_dt) \
                do { \
                    if (flat_vec == 16) { LAUNCH_FLAT(scalar_t, cache_t, kv_dt, 16); } \
                    else if (flat_vec == 8) { LAUNCH_FLAT(scalar_t, cache_t, kv_dt, 8); } \
                    else if (flat_vec == 4) { LAUNCH_FLAT(scalar_t, cache_t, kv_dt, 4); } \
                    else if (flat_vec == 2) { LAUNCH_FLAT(scalar_t, cache_t, kv_dt, 2); } \
                    else { LAUNCH_FLAT(scalar_t, cache_t, kv_dt, 1); } \
                } while (0)
                if (kv_cache_dtype_str == "auto") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_FLAT_VEC(at::BFloat16, at::BFloat16, KVCacheDataType::kAuto); }
                    else { TORCH_CHECK(false, "Unsupported dtype for auto mode"); }
                } else if (kv_cache_dtype_str == "int8") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_FLAT_VEC(at::BFloat16, int8_t, KVCacheDataType::kInt8); }
                    else { TORCH_CHECK(false, "Unsupported input dtype for int8 mode"); }
                } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_FLAT_VEC(at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3); }
                    else { TORCH_CHECK(false, "Unsupported input dtype for fp8 mode"); }
                } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
                #undef LAUNCH_FLAT_VEC
                #undef LAUNCH_FLAT
            } else {
                const dim3 grid(q_len_val);
                const dim3 block(std::min(1024, threads_needed));
                if (kv_cache_dtype_str == "auto") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_token_centric_kernel, grid, block, at::BFloat16, at::BFloat16, KVCacheDataType::kAuto); }
                    else { TORCH_CHECK(false, "Unsupported dtype for auto mode"); }
                } else if (kv_cache_dtype_str == "int8") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_token_centric_kernel, grid, block, at::BFloat16, int8_t, KVCacheDataType::kInt8); }
                    else { TORCH_CHECK(false, "Unsupported input dtype for int8 mode"); }
                } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
                    if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_token_centric_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3); }
                    else { TORCH_CHECK(false, "Unsupported input dtype for fp8 mode"); }
                } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
            }
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
        } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
            if (packed_qkv.scalar_type() == at::ScalarType::BFloat16) { LAUNCH_KERNEL(store_kv_cache_batch_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3); }
            else { TORCH_CHECK(false, "Unsupported input dtype for fp8 mode"); }
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
       
        const int elems_per_tok = kv_head_num * head_dim;
        int token_idx_in_q, remainder;
        store_kv_div_mod_dim(i, elems_per_tok, token_idx_in_q, remainder);
        int head_idx_in_kv, val_idx_in_head;
        store_kv_div_mod_dim(remainder, head_dim, head_idx_in_kv, val_idx_in_head);

 
        const int64_t abs_token_idx = q_token_offset + token_idx_in_q;
        const int64_t src_offset = abs_token_idx * kv_head_num * head_dim +
                                   head_idx_in_kv * head_dim +
                                   val_idx_in_head;


        const int64_t dst_token_idx = cache_start_pos + token_idx_in_q;

        int logical_block_idx, offset_in_block;
        store_kv_div_mod_dim64(dst_token_idx, block_size, logical_block_idx, offset_in_block);

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
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
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

    int logical_block_idx, offset_in_block;
    store_kv_div_mod_dim(cache_start_pos, block_size, logical_block_idx, offset_in_block);

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
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
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
    int logical_block_idx, offset_in_block;
    store_kv_div_mod_dim64(dst_token_idx, block_size, logical_block_idx, offset_in_block);
    const int physical_block_id = block_table[logical_block_idx];


    for (int i = threadIdx.x * Vec_Size; i < total_elements_per_token; i += blockDim.x * Vec_Size) {
        int head_idx_in_kv, val_idx_in_head;
        store_kv_div_mod_dim(i, head_dim, head_idx_in_kv, val_idx_in_head);


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
        } else if constexpr (kv_dt == KVCacheDataType::kFp8E4M3) {
            cache_vector_t k_vec_out, v_vec_out;
            #pragma unroll
            for (int j = 0; j < Vec_Size; ++j) {
                const int scale_idx = head_idx_in_kv * head_dim + val_idx_in_head + j;
                const float kq = static_cast<float>(k_vec_in.val[j]) * k_scale[scale_idx];
                const float vq = static_cast<float>(v_vec_in.val[j]) * v_scale[scale_idx];
                k_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(kq));
                v_vec_out.val[j] = static_cast<cache_t>(float_to_fp8_e4m3_scaled_fast(vq));
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
            reinterpret_cast<cache_t*>(K_cache.data_ptr()), reinterpret_cast<cache_t*>(V_cache.data_ptr()), \
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
        // batch==1: key.size(0) == q_len; avoid q_lens.item() DtoH sync
        const int q_len_val = static_cast<int>(key.size(0));
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
            } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for fp8 mode");
                LAUNCH_KERNEL(store_kv_cache_paged_head_centric_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
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
            } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
                TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for fp8 mode");
                LAUNCH_KERNEL(store_kv_cache_paged_token_centric_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3, accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
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
        } else if (kv_cache_dtype_str == "fp8" || kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "float8" || kv_cache_dtype_str == "float8_e4m3fn") {
            TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16, "Unsupported input dtype for fp8 mode");
            LAUNCH_KERNEL(store_kv_cache_paged_batch_kernel, grid, block, at::BFloat16, uint8_t, KVCacheDataType::kFp8E4M3, q_lens.data_ptr<int32_t>(), accum_q_lens.data_ptr<int32_t>(), cache_lens.data_ptr<int32_t>());
        } else { TORCH_CHECK(false, "Unsupported kv_cache_dtype"); }
    }

    #undef LAUNCH_KERNEL_VEC
    #undef LAUNCH_KERNEL
}

} // namespace native
} // namespace at
