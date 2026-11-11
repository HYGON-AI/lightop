// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <type_traits>
#include <vector>

#define DISPATCH_BOOL(VAL, NAME, ...) \
  if (VAL) {                          \
    constexpr bool NAME = true;       \
    __VA_ARGS__();                    \
  } else {                            \
    constexpr bool NAME = false;      \
    __VA_ARGS__();                    \
  }

#define DISPATCH_CUSTOM_TYPE(TYPE, NAME, ...) \
    switch (TYPE) { \
        case at::ScalarType::Float: { \
            using NAME = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::Half: { \
            using NAME = at::Half; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::BFloat16: { \
            using NAME = at::BFloat16; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            TORCH_CHECK(false, "Unsupported fused parameter type"); \
    }

namespace at {
namespace native {

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

template<typename T, int WIDTH>
__device__ __forceinline__ T WarpReduceSum_Local(T val) {
    #pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor(val, mask, WIDTH);
    }
    return val;
}

template<typename T, int WIDTH>
__device__ __forceinline__ T WarpReduceMax_Local(T val) {
    #pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor(val, mask, WIDTH));
    }
    return val;
}

inline __device__ uint8_t float_to_fp8e4m3(float f) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      result = 0x7f;  // NaN/Max
    } else {
      if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = c10::detail::fp32_to_bits(
          c10::detail::fp32_from_bits(f_bits) + 
          c10::detail::fp32_from_bits(denorm_mask)
        );
        result = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t, typename norm_t, 
          bool IS_NEOX, bool IS_RMSNORM, int VEC_SIZE, int THREADS_PER_HEAD,int BLOCK_SIZE>
__device__ __forceinline__ void process_k_block_reg(
    const int k_block_id, const int tid,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ key,            
    const cos_sin_t* __restrict__ cos_sin_cache,
    scalar_t* __restrict__ key_out,          
    const int rot_dim,
    const int head_dim,                    
    const int64_t key_stride,              
    const int64_t head_stride_k,          
    const norm_t* __restrict__ gamma_k,    
    const norm_t* __restrict__ beta_k,     
    const scalar_t eps,
    const int num_tokens,
    const int num_kv_heads
) {
    constexpr int MAX_ITER = 4;
    constexpr int HEADS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_HEAD;
    
    const int global_idx = k_block_id * HEADS_PER_BLOCK + (tid / THREADS_PER_HEAD);
    const int token_idx = global_idx / num_kv_heads;
    const int head_idx = global_idx % num_kv_heads;
    const int lane_in_head = tid % THREADS_PER_HEAD;

    if (token_idx >= num_tokens) return;

    const int64_t pos = positions[token_idx];
    const scalar_t* head_ptr = key + token_idx * key_stride + head_idx * head_stride_k;
    scalar_t* k_out_ptr = key_out + token_idx * key_stride + head_idx * head_stride_k;

    T_ACC local_sum = 0;
    T_ACC local_sq_sum = 0;
    const int HALF_ROT = rot_dim / 2;
    
    constexpr int HALF_VEC = VEC_SIZE / 2; 
    using LoadT = AlignedVector<scalar_t, VEC_SIZE>;
    using LoadHalfT = AlignedVector<scalar_t, HALF_VEC>;

    if constexpr (IS_NEOX) {
        LoadHalfT reg_L[MAX_ITER];
        LoadHalfT reg_H[MAX_ITER];
        LoadT reg_rem[MAX_ITER];

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * HALF_VEC + lane_in_head * HALF_VEC;
            if (i < HALF_ROT) {
                reg_L[iter] = *(LoadHalfT*)(head_ptr + i);
                reg_H[iter] = *(LoadHalfT*)(head_ptr + i + HALF_ROT);
                #pragma unroll
                for(int j=0; j<HALF_VEC; ++j) {
                    T_ACC val_L = static_cast<T_ACC>(reg_L[iter].val[j]);
                    T_ACC val_H = static_cast<T_ACC>(reg_H[iter].val[j]);
                    local_sum += val_L + val_H;
                    local_sq_sum += val_L * val_L + val_H * val_H;
                }
            }
        }
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = rot_dim + iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                reg_rem[iter] = *(LoadT*)(head_ptr + i);
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    T_ACC val = static_cast<T_ACC>(reg_rem[iter].val[j]);
                    local_sum += val;
                    local_sq_sum += val * val;
                }
            }
        }

        T_ACC head_sum = WarpReduceSum_Local<T_ACC, THREADS_PER_HEAD>(local_sum);
        T_ACC head_sq_sum = WarpReduceSum_Local<T_ACC, THREADS_PER_HEAD>(local_sq_sum);
        T_ACC mean = IS_RMSNORM ? 0 : head_sum / head_dim;
        T_ACC var = (head_sq_sum / head_dim) - (mean * mean);
        T_ACC inv_std = c10::cuda::compat::rsqrt(var + static_cast<T_ACC>(eps));

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * HALF_VEC + lane_in_head * HALF_VEC;
            if (i < HALF_ROT) {
                #pragma unroll
                for(int j=0; j<HALF_VEC; ++j) {
                    int global_i_L = i + j;
                    int global_i_H = i + HALF_ROT + j;
                    
                    T_ACC n_L = (static_cast<T_ACC>(reg_L[iter].val[j]) - mean) * inv_std;
                    T_ACC n_H = (static_cast<T_ACC>(reg_H[iter].val[j]) - mean) * inv_std;
                    T_ACC g_L = gamma_k ? static_cast<T_ACC>(gamma_k[global_i_L]) : 1.0f;
                    T_ACC g_H = gamma_k ? static_cast<T_ACC>(gamma_k[global_i_H]) : 1.0f;
                    T_ACC b_L = beta_k ? static_cast<T_ACC>(beta_k[global_i_L]) : 0.0f;
                    T_ACC b_H = beta_k ? static_cast<T_ACC>(beta_k[global_i_H]) : 0.0f;

                    scalar_t ql = n_L * g_L + b_L;
                    scalar_t qh = n_H * g_H + b_H;

                    int cs_idx_L = pos * rot_dim + global_i_L;
                    int cs_idx_H = pos * rot_dim + global_i_H;
                    scalar_t cl = static_cast<scalar_t>(cos_sin_cache[cs_idx_L]);
                    scalar_t sl = static_cast<scalar_t>(cos_sin_cache[cs_idx_H]);

                    reg_L[iter].val[j] = ql * cl - qh * sl;
                    reg_H[iter].val[j] = ql * sl + qh * cl;
                }
            }
        }
        
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = rot_dim + iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    int global_i = i + j;
                    T_ACC n = (static_cast<T_ACC>(reg_rem[iter].val[j]) - mean) * inv_std;
                    T_ACC g = gamma_k ? static_cast<T_ACC>(gamma_k[global_i]) : 1.0f;
                    T_ACC b = beta_k ? static_cast<T_ACC>(beta_k[global_i]) : 0.0f;
                    reg_rem[iter].val[j] = n * g + b;
                }
            }
        }

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * HALF_VEC + lane_in_head * HALF_VEC;
            if (i < HALF_ROT) {
                *(LoadHalfT*)(k_out_ptr + i) = reg_L[iter];
                *(LoadHalfT*)(k_out_ptr + i + HALF_ROT) = reg_H[iter];
            }
        }
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = rot_dim + iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                *(LoadT*)(k_out_ptr + i) = reg_rem[iter];
            }
        }
    } else {
        LoadT reg[MAX_ITER];

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                reg[iter] = *(LoadT*)(head_ptr + i);
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    T_ACC val = static_cast<T_ACC>(reg[iter].val[j]);
                    local_sum += val;
                    local_sq_sum += val * val;
                }
            }
        }

        T_ACC head_sum = WarpReduceSum_Local<T_ACC, THREADS_PER_HEAD>(local_sum);
        T_ACC head_sq_sum = WarpReduceSum_Local<T_ACC, THREADS_PER_HEAD>(local_sq_sum);
        T_ACC mean = IS_RMSNORM ? 0 : head_sum / head_dim;
        T_ACC var = (head_sq_sum / head_dim) - (mean * mean);
        T_ACC inv_std = c10::cuda::compat::rsqrt(var + static_cast<T_ACC>(eps));

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    int global_i = i + j;
                    T_ACC n = (static_cast<T_ACC>(reg[iter].val[j]) - mean) * inv_std;
                    T_ACC g = gamma_k ? static_cast<T_ACC>(gamma_k[global_i]) : 1.0f;
                    T_ACC b = beta_k ? static_cast<T_ACC>(beta_k[global_i]) : 0.0f;
                    reg[iter].val[j] = n * g + b;
                }

                if (i < rot_dim) {
                    #pragma unroll
                    for(int j=0; j<VEC_SIZE; j+=2) {
                        scalar_t q0 = reg[iter].val[j];
                        scalar_t q1 = reg[iter].val[j+1];
                        int c_idx = i/2 + j/2;
                        scalar_t cc = static_cast<scalar_t>(cos_sin_cache[pos * rot_dim + c_idx]);
                        scalar_t ss = static_cast<scalar_t>(cos_sin_cache[pos * rot_dim + HALF_ROT + c_idx]);
                        reg[iter].val[j]   = q0 * cc - q1 * ss;
                        reg[iter].val[j+1] = q1 * cc + q0 * ss;
                    }
                }
            }
        }

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                *(LoadT*)(k_out_ptr + i) = reg[iter];
            }
        }
    }
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t,
          bool IS_NEOX, int VEC_SIZE, int THREADS_PER_HEAD,int BLOCK_SIZE>
__device__ __forceinline__ void process_q_block_reg(
    const int q_block_id, const int tid,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ query,          
    const cos_sin_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ query_fp8_out,     
    float* __restrict__ query_scale_out,   
    const int rot_dim,
    const int head_dim,                    
    const int64_t query_stride,            
    const int64_t head_stride_q,           
    const int num_tokens,
    const int num_heads
) {
    constexpr int MAX_ITER = 4;
    constexpr int HEADS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_HEAD; 
    
    const int global_idx = q_block_id * HEADS_PER_BLOCK + (tid / THREADS_PER_HEAD);
    const int token_idx = global_idx / num_heads;
    const int head_idx = global_idx % num_heads;
    const int lane_in_head = tid % THREADS_PER_HEAD;

    if (token_idx >= num_tokens) return;

    const int64_t pos = positions[token_idx];
    const scalar_t* head_ptr = query + token_idx * query_stride + head_idx * head_stride_q;
    uint8_t* q_fp8_ptr = query_fp8_out + token_idx * (num_heads * head_dim) + head_idx * head_dim;

    const int HALF_ROT = rot_dim / 2;
    constexpr int HALF_VEC = VEC_SIZE / 2; 
    using LoadT = AlignedVector<scalar_t, VEC_SIZE>;
    using LoadHalfT = AlignedVector<scalar_t, HALF_VEC>;

    T_ACC local_max = 0;
    if constexpr (IS_NEOX) {
        LoadHalfT reg_L[MAX_ITER]; 
        LoadHalfT reg_H[MAX_ITER]; 
        LoadT reg_rem[MAX_ITER];
        
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * HALF_VEC + lane_in_head * HALF_VEC;
            if (i < HALF_ROT) {
                reg_L[iter] = *(LoadHalfT*)(head_ptr + i);
                reg_H[iter] = *(LoadHalfT*)(head_ptr + i + HALF_ROT);
                #pragma unroll
                for(int j=0; j<HALF_VEC; ++j) {
                    scalar_t ql = reg_L[iter].val[j];
                    scalar_t qh = reg_H[iter].val[j];
                    
                    int cs_idx_L = pos * rot_dim + i + j;
                    int cs_idx_H = pos * rot_dim + HALF_ROT + i + j;
                    scalar_t cl = static_cast<scalar_t>(cos_sin_cache[cs_idx_L]);
                    scalar_t sl = static_cast<scalar_t>(cos_sin_cache[cs_idx_H]);
                    
                    scalar_t new_ql = ql * cl - qh * sl;
                    scalar_t new_qh = ql * sl + qh * cl;
                    
                    reg_L[iter].val[j] = new_ql;
                    reg_H[iter].val[j] = new_qh;
                    
                    local_max = fmaxf(local_max, fabsf((float)new_ql));
                    local_max = fmaxf(local_max, fabsf((float)new_qh));
                }
            }
        }

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = rot_dim + iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                reg_rem[iter] = *(LoadT*)(head_ptr + i);
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    local_max = fmaxf(local_max, fabsf((float)reg_rem[iter].val[j]));
                }
            }
        }

        T_ACC head_max = WarpReduceMax_Local<T_ACC, THREADS_PER_HEAD>(local_max);
        float scale = fmaxf(head_max / 448.0f, 1e-12f);
        float inv_scale = 1.0f / scale;

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i_L = iter * THREADS_PER_HEAD * HALF_VEC + lane_in_head * HALF_VEC;
            if (i_L < HALF_ROT) {
                AlignedVector<uint8_t, HALF_VEC> out_L, out_H;
                #pragma unroll
                for(int j=0; j<HALF_VEC; ++j) {
                    out_L.val[j] = float_to_fp8e4m3((float)reg_L[iter].val[j] * inv_scale);
                    out_H.val[j] = float_to_fp8e4m3((float)reg_H[iter].val[j] * inv_scale);
                }
                *(AlignedVector<uint8_t, HALF_VEC>*)(q_fp8_ptr + i_L) = out_L;
                *(AlignedVector<uint8_t, HALF_VEC>*)(q_fp8_ptr + i_L + HALF_ROT) = out_H;
            }
        }
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i_rem = rot_dim + iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i_rem < head_dim) {
                AlignedVector<uint8_t, VEC_SIZE> out_rem;
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    out_rem.val[j] = float_to_fp8e4m3((float)reg_rem[iter].val[j] * inv_scale);
                }
                *(AlignedVector<uint8_t, VEC_SIZE>*)(q_fp8_ptr + i_rem) = out_rem;
            }
        }
        if (lane_in_head == 0) query_scale_out[token_idx * num_heads + head_idx] = scale;
    } else {
        LoadT reg[MAX_ITER]; 
        
        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                reg[iter] = *(LoadT*)(head_ptr + i);
                if (i < rot_dim) {
                    #pragma unroll
                    for(int j=0; j<VEC_SIZE; j+=2) {
                        scalar_t q0 = reg[iter].val[j];
                        scalar_t q1 = reg[iter].val[j+1];
                        int c_idx = i/2 + j/2;
                        scalar_t cc = static_cast<scalar_t>(cos_sin_cache[pos * rot_dim + c_idx]);
                        scalar_t ss = static_cast<scalar_t>(cos_sin_cache[pos * rot_dim + HALF_ROT + c_idx]);
                        
                        scalar_t new_q0 = q0 * cc - q1 * ss;
                        scalar_t new_q1 = q1 * cc + q0 * ss;
                        reg[iter].val[j]   = new_q0;
                        reg[iter].val[j+1] = new_q1;
                        local_max = fmaxf(local_max, fabsf((float)new_q0));
                        local_max = fmaxf(local_max, fabsf((float)new_q1));
                    }
                } else {
                    #pragma unroll
                    for(int j=0; j<VEC_SIZE; ++j) {
                        local_max = fmaxf(local_max, fabsf((float)reg[iter].val[j]));
                    }
                }
            }
        }

        T_ACC head_max = WarpReduceMax_Local<T_ACC, THREADS_PER_HEAD>(local_max);
        float scale = fmaxf(head_max / 448.0f, 1e-12f);
        float inv_scale = 1.0f / scale;

        #pragma unroll
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            int i = iter * THREADS_PER_HEAD * VEC_SIZE + lane_in_head * VEC_SIZE;
            if (i < head_dim) {
                AlignedVector<uint8_t, VEC_SIZE> out_vec;
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    out_vec.val[j] = float_to_fp8e4m3((float)reg[iter].val[j] * inv_scale);
                }
                *(AlignedVector<uint8_t, VEC_SIZE>*)(q_fp8_ptr + i) = out_vec;
            }
        }
        if (lane_in_head == 0) query_scale_out[token_idx * num_heads + head_idx] = scale;
    }
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t, typename norm_t,
          bool IS_NEOX, bool IS_RMSNORM, int VEC_SIZE, int THREADS_Q, int THREADS_K,int BLOCK_SIZE>
__global__ void fused_layer_norm_rope_quant_fp8_kernel(
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const cos_sin_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ query_fp8_out,
    float* __restrict__ query_scale_out,
    scalar_t* __restrict__ key_out,
    const int rot_dim,
    const int head_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride_q,
    const int64_t head_stride_k,
    const norm_t* __restrict__ gamma_k, 
    const norm_t* __restrict__ beta_k,  
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int total_q_blocks
) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    if (block_id < total_q_blocks) {
        process_q_block_reg<T_ACC, scalar_t, cos_sin_t, IS_NEOX, VEC_SIZE, THREADS_Q,BLOCK_SIZE>(
            block_id, tid, positions, query, cos_sin_cache, 
            query_fp8_out, query_scale_out,
            rot_dim, head_dim, query_stride, head_stride_q, 
            num_tokens, num_heads
        );
    } else {
        process_k_block_reg<T_ACC, scalar_t, cos_sin_t, norm_t, IS_NEOX, IS_RMSNORM, VEC_SIZE, THREADS_K,BLOCK_SIZE>(
            block_id - total_q_blocks, tid, positions, key, cos_sin_cache, 
            key_out, rot_dim, head_dim, key_stride, head_stride_k, 
            gamma_k, beta_k, eps, num_tokens, num_kv_heads
        );
    }
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t, typename norm_t>
void launch_fused_norm_rope_quant_fp8_unified(
    const int64_t* positions,
    const scalar_t* query,
    const scalar_t* key,
    const cos_sin_t* cos_sin_cache,
    uint8_t* query_fp8_out,
    float* query_scale_out,
    scalar_t* key_out,
    int rot_dim,
    int head_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride_q,
    int64_t head_stride_k,
    const norm_t* gamma_k, 
    const norm_t* beta_k,  
    scalar_t eps,
    int num_tokens,
    bool is_neox,
    bool is_rmsnorm,
    const int num_heads,
    const int num_kv_heads,
    hipStream_t stream) {

    constexpr int VEC = 8;
    constexpr int BLOCK_SIZE = 128;
    constexpr int THREADS_Q = 8;
    constexpr int THREADS_K = 16;
    
    int heads_per_block_q = BLOCK_SIZE / THREADS_Q;
    int total_q_items = num_tokens * num_heads;
    int total_q_blocks = (total_q_items + heads_per_block_q - 1) / heads_per_block_q;
    
    int heads_per_block_k = BLOCK_SIZE / THREADS_K;
    int total_k_items = num_tokens * num_kv_heads;
    int total_k_blocks = (total_k_items + heads_per_block_k - 1) / heads_per_block_k;
    
    int total_blocks = total_q_blocks + total_k_blocks;

    if (total_blocks == 0) return;

    dim3 grid(total_blocks);
    dim3 block(BLOCK_SIZE);

    DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
        DISPATCH_BOOL(is_rmsnorm, IS_RMSNORM_CONST, [&] {
            fused_layer_norm_rope_quant_fp8_kernel<T_ACC, scalar_t, cos_sin_t, norm_t, IS_NEOX_CONST, IS_RMSNORM_CONST, VEC, THREADS_Q, THREADS_K,BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                positions, query, key, cos_sin_cache, 
                query_fp8_out, query_scale_out, key_out,
                rot_dim, head_dim, query_stride, key_stride, head_stride_q, head_stride_k, 
                gamma_k, beta_k, eps, num_tokens, num_heads, num_kv_heads, total_q_blocks
            );
        });
    });
}

std::vector<at::Tensor> fuse_norm_rope_quant_fp8(
    at::Tensor positions, 
    at::Tensor query, 
    at::Tensor key,
    int64_t head_size, 
    at::Tensor cos_sin_cache, 
    bool is_neox, 
    bool is_rmsnorm,
    std::optional<at::Tensor> weight_k, 
    std::optional<at::Tensor> bias_k,   
    double epsilon) {

    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    
    int rot_dim = cos_sin_cache.size(1);
    int head_dim = head_size;   
    TORCH_CHECK(head_dim == 128, "FusedNormRopeQuantFP8 kernel currently only supports head_dim == 128, but got ", head_dim);
    int seq_dim_idx = positions_ndim - 1;
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int64_t head_stride_q = (query.dim() >= 3) ? query.stride(-2) : head_size;
    int64_t head_stride_k = (key.dim() >= 3) ? key.stride(-2) : head_size;      

    auto out_q_fp8 = at::empty({num_tokens, num_heads, head_dim}, query.options().dtype(at::kFloat8_e4m3fn));
    auto out_q_scale = at::empty({num_tokens, num_heads, 1}, query.options().dtype(at::kFloat));
    auto out_k = at::empty_like(key);

    const c10::OptionalDeviceGuard device_guard(device_of(query));
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    at::ScalarType q_type = query.scalar_type();
    at::ScalarType cos_sin_type = cos_sin_cache.scalar_type();
    at::ScalarType norm_type = weight_k.has_value() && weight_k->defined() ? weight_k->scalar_type() : q_type;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q_type, "fuse_norm_rope_quant_fp8", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        DISPATCH_CUSTOM_TYPE(cos_sin_type, cos_sin_t, [&] {
            DISPATCH_CUSTOM_TYPE(norm_type, norm_t, [&] {
                const norm_t* w_k_ptr = weight_k.has_value() && weight_k->defined() ? weight_k->data_ptr<norm_t>() : nullptr;
                const norm_t* b_k_ptr = bias_k.has_value() && bias_k->defined() ? bias_k->data_ptr<norm_t>() : nullptr;
                
                launch_fused_norm_rope_quant_fp8_unified<T_ACC, scalar_t, cos_sin_t, norm_t>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<cos_sin_t>(),
                    reinterpret_cast<uint8_t*>(out_q_fp8.data_ptr()),
                    out_q_scale.data_ptr<float>(),
                    out_k.data_ptr<scalar_t>(),
                    rot_dim, head_dim,
                    query_stride, key_stride, head_stride_q, head_stride_k,
                    w_k_ptr, b_k_ptr,
                    static_cast<scalar_t>(epsilon),
                    (int)num_tokens,
                    is_neox,
                    is_rmsnorm,
                    num_heads, num_kv_heads,
                    stream
                );
            });
        });
    });
    return {out_k, out_q_fp8, out_q_scale};
}

} // namespace native
} // namespace at