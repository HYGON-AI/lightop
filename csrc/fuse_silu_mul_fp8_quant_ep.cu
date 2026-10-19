// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
#include "reduce.h"


namespace at {
namespace native {



// float8_e4m3fn + float8_e5m2
#ifndef USE_ROCM
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e5m2.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <ATen/hip/HIPContext.h>
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e4m3fnuz.h>
  #include <c10/util/Float8_e5m2.h>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||    
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||    
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};
// float8_e4m3fn + float8_e5m2

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)  \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif


    inline __device__ uint8_t float_to_fp8e5m2(float f) {
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    } else {
        if (f_bits < (UINT32_C(113) << 23)) {
        f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
                    + c10::detail::fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
        uint32_t mant_odd = (f_bits >> 21) & 1;
        f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 21);
        }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
    }


    inline __device__ uint8_t float_to_fp8e4m3(float f) {
    // ===== 常量定义 =====
    // FP8 E4M3最大值: 1087 << 20
    // 1087 = 2^(4+1) - 1 + 2^4 = 31 + 16 = 47? 等等，这里需要重新计算
    // 实际上是基于指数和尾数的位模式
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

    // 非正规化数处理掩码
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

    // ===== 步骤1: 提取float的位表示 =====
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;

    // ===== 步骤2: 提取符号位并移除 =====
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;

    // ===== 步骤3: 处理不同数值范围 =====

    // 3a. 溢出处理
    if (f_bits >= fp8_max) {
        result = 0x7f;  // 01111111 (NaN/最大值)
    } else {
        
        // 3b. 非正规化数处理
        if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = c10::detail::fp32_to_bits(
            c10::detail::fp32_from_bits(f_bits) + 
            c10::detail::fp32_from_bits(denorm_mask)
        );
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        
        // 3c. 正规化数处理
        } else {
        // 检查尾数第20位是否为1（对应FP8 E4M3 3位尾数的LSB）
        uint8_t mant_odd = (f_bits >> 20) & 1;
        
        // 指数偏置调整: FP32偏置127 → FP8 E4M3偏置7
        // 调整量 = 7 - 127 = -120
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        
        // 应用奇数舍入
        f_bits += mant_odd;
        
        // 右移20位得到8位结果
        result = static_cast<uint8_t>(f_bits >> 20);
        }
    }

    // ===== 步骤4: 加上符号位 =====
    result |= static_cast<uint8_t>(sign >> 24);

    return result;
    }


    inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d)
    {
    #if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
    return res;
    #else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
    return out;
    #endif
    }


    inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d)
    {
    #if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
    return res;
    #else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return out;
    #endif
    }


template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  // return (T)(((float)x) / (1.0f + __expf((float)-x)));
  constexpr float LOG2E = 1.44269504088896340736f;
  return (T)(((float)x) / (1.0f + __builtin_amdgcn_exp2f(-((float)x) * LOG2E)));
}

template <typename scalar_t,
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? silu_kernel(x) * y : x * silu_kernel(y);
}

// template <typename scalar_t, bool act_first>
// __device__ __forceinline__ scalar_t compute_with_swiglu_limit(
//     const scalar_t& x, const scalar_t& y, const float limit) {
//   const scalar_t gate = act_first ? x : y;
//   const scalar_t up = act_first ? y : x;

//   float gate_f = static_cast<float>(gate);
//   float up_f = static_cast<float>(up);
//   gate_f = gate_f > limit ? limit : gate_f;
//   up_f = up_f < -limit ? -limit : (up_f > limit ? limit : up_f);

//   const scalar_t gate_clamped = static_cast<scalar_t>(gate_f);
//   const scalar_t up_clamped = static_cast<scalar_t>(up_f);
//   return static_cast<scalar_t>(silu_kernel(gate_clamped) * up_clamped);
// }

template <typename scalar_t, bool act_first>
__device__ __forceinline__ scalar_t compute_with_swiglu_limit(
    const scalar_t& x,
    const scalar_t& y,
    const float limit) {
  const scalar_t gate = act_first ? x : y;
  const scalar_t up = act_first ? y : x;

  const scalar_t limit_t = static_cast<scalar_t>(limit);
  const scalar_t neg_limit_t = -limit_t;

  const scalar_t gate_clamped =
      gate > limit_t ? limit_t : gate;

  const scalar_t up_clamped =
      up < neg_limit_t
          ? neg_limit_t
          : (up > limit_t ? limit_t : up);

  return silu_kernel(gate_clamped) * up_clamped;
}

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_ep_gernel(
    int64_t num_tokens,
    uint8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int fp8type,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert
    ) {
    int64_t token_idx = blockIdx.x * T_dim + blockIdx.y;
    const int64_t num_total_tokens = num_tokens;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];
    __shared__ float s_token_scale; 

    for(; token_idx < num_total_tokens ; token_idx += gridDim.y){
        if (tokens_per_expert != nullptr) {
            const int32_t e = token_idx / T_dim; 
            const int32_t t = token_idx % T_dim; 
            
            if (t >= tokens_per_expert[e]) {
                break; 
            }
        }

        const int64_t input_offset = token_idx * 2 * d;
        const int64_t output_offset = token_idx * d;

        float row_max = 0.0f;
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            scalar_t val = compute<scalar_t, act_first>(x, y);
            row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
        }
        row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
        float scale = 0.0f; 
        if(threadIdx.x == 0){
            scale = row_max;
            s_token_scale = scale; 
            if(fp8type==0)
                scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e4m3fn>, min_scaling_factor<c10::Float8_e4m3fn>::val()); 
            else
                scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e5m2>, min_scaling_factor<c10::Float8_e5m2>::val()); 
        }
        __syncthreads(); 

        float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            float val = static_cast<float>(compute<scalar_t, act_first>(x, y));
            uint8_t q_val;
            if(fp8type == 0)
                q_val = float_to_fp8e4m3(val * inv_s);
            else
                q_val = float_to_fp8e5m2(val * inv_s);
            out[output_offset + idx] = static_cast<uint8_t>(q_val);
        }
    }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1_one_ep(
    int32_t num_tokens,                 
    uint8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int fp8type,
    const int d,
    const int32_t T,
    const int32_t *tokens_per_expert
    ) {
    int32_t token_idx= blockIdx.x;

    const int32_t e = token_idx / T; // 专家索引
    const int32_t t = token_idx % T; // 专家内的 Token 索引
    
    if (tokens_per_expert != nullptr && t >= tokens_per_expert[e]) {
        return; 
    }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;
    const int32_t x_index = token_idx * 2 * d + idx;
    const int32_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecInt8Type* y = (VecInt8Type*)(out + y_index);
    
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_y[VEC];
    if (idx < d) {
        *(VecType*)r_x1 = *x1;
        *(VecType*)r_x2 = *x2;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
        }
    }
    float row_max = 0.f;
    if (idx < d) {
        #pragma unroll
        for(int ii = 0;ii < VEC;ii++){
            row_max = fmaxf(row_max,fabsf(r_y[ii]));
        }
    }

    row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);
    __shared__ float s_token_scale; 
    float scale = 0.0f;     
    if(tidx == 0){
        scale = row_max;
        s_token_scale = scale; 
        if(fp8type==0)
            scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e4m3fn>, min_scaling_factor<c10::Float8_e4m3fn>::val()); 
        else
            scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e5m2>, min_scaling_factor<c10::Float8_e5m2>::val()); 
    }
    __syncthreads();

    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

    uint8_t out_vec[VEC];
    int loop = VEC / 4;
    if (idx < d) {
        #pragma unroll
        for(int ii = 0;ii < loop;ii++){
            if(fp8type==0)
                out_vec[ii] = float_to_fp8e4m3_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
            else
                out_vec[ii] = float_to_fp8e5m2_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
        }
        *y =*(VecInt8Type*)out_vec;
    }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_ep(
    int64_t num_tokens,
    uint8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int fp8type,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert
    ) {
    int64_t token_idx = blockIdx.x * T_dim + blockIdx.y;
    int64_t num_total_tokens = num_tokens;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    __shared__ float s_token_scale; 

    for(; token_idx < num_total_tokens ; token_idx += gridDim.y)
    {
        if (tokens_per_expert != nullptr) {
            const int32_t e = token_idx / T_dim; 
            const int32_t t = token_idx % T_dim; 
            
            if (t >= tokens_per_expert[e]) {
                break; 
            }
        }

        using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
        using VecInt8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

        int idx = threadIdx.x * VEC;
        int tidx = threadIdx.x;
        const int64_t x_index = token_idx * 2 * d + idx;
        const int64_t y_index = token_idx * d + idx;
        
        VecType* x1 = (VecType*)(input + x_index);
        VecType* x2 = (VecType*)(input + x_index + d);
        VecInt8Type* y = (VecInt8Type*)(out + y_index);
        scalar_t r_x1[VEC];
        scalar_t r_x2[VEC];
        float r_y[VEC];

        if (idx < d) { 
            *(VecType*)r_x1 = *x1;
            *(VecType*)r_x2 = *x2;
            #pragma unroll
            for (int i = 0; i < VEC; i++) {
                r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
            }
        } 

        float row_max = 0.f;
        if (idx < d) {
            #pragma unroll
            for(int ii = 0;ii < VEC;ii++){
                row_max = fmaxf(row_max,fabsf(r_y[ii]));
            }
        }

        row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);

        float fp8type_max_v;     
        float min_scaling;    
        
        if (fp8type == 0) { // E4M3
            fp8type_max_v = quant_type_max_v<c10::Float8_e4m3fn>; 
            min_scaling = min_scaling_factor<c10::Float8_e4m3fn>::val();
        } else { // E5M2
            fp8type_max_v = quant_type_max_v<c10::Float8_e5m2>;   
            min_scaling = min_scaling_factor<c10::Float8_e5m2>::val();
        }

        float scale = 0.0f;
        if(tidx == 0){
            scale = fmaxf(row_max / fp8type_max_v, min_scaling); 
            s_token_scale = scale; 
            scales[token_idx] = scale; 
        }
        __syncthreads();

        float inv_s = 1.0f / s_token_scale;

        uint8_t out_vec[VEC];
        int loop = VEC / 4;
        if (idx < d) {
            #pragma unroll
            for(int ii = 0;ii < loop;ii++){
                if(fp8type==0)
                    ((uint32_t*)out_vec)[ii] = float_to_fp8e4m3_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
                else
                    ((uint32_t*)out_vec)[ii] = float_to_fp8e5m2_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
            }
            *y =*(VecInt8Type*)out_vec;
        }

    } // 结束 Grid-Stride Loop
}


template <typename scalar_t, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_ep_gernel_with_limit(
    int64_t num_tokens,
    uint8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int fp8type,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert,
    const float limit
    ) {
    (void)num_tokens;
    (void)expert_num;
    const int32_t expert_idx = blockIdx.x;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];
    __shared__ float s_token_scale; 

    for (int32_t token_in_expert = blockIdx.y;
         token_in_expert < T_dim;
         token_in_expert += gridDim.y) {
        if (tokens_per_expert != nullptr &&
            token_in_expert >= tokens_per_expert[expert_idx]) {
            break;
        }

        const int64_t token_idx =
            static_cast<int64_t>(expert_idx) * T_dim + token_in_expert;

        const int64_t input_offset = token_idx * 2 * d;
        const int64_t output_offset = token_idx * d;

        float row_max = 0.0f;
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            const scalar_t val =
                static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
            row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
        }
        row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
        float scale = 0.0f; 
        if(threadIdx.x == 0){
            scale = row_max;
            s_token_scale = scale; 
            if(fp8type==0)
                scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e4m3fn>, min_scaling_factor<c10::Float8_e4m3fn>::val()); 
            else
                scales[token_idx] = fmaxf(s_token_scale / quant_type_max_v<c10::Float8_e5m2>, min_scaling_factor<c10::Float8_e5m2>::val()); 
        }
        __syncthreads(); 

        float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            const float val = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
            uint8_t q_val;
            if(fp8type == 0)
                q_val = float_to_fp8e4m3(val * inv_s);
            else
                q_val = float_to_fp8e5m2(val * inv_s);
            out[output_offset + idx] = static_cast<uint8_t>(q_val);
        }
    }
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_ep_with_limit(
    int64_t num_tokens,
    uint8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int fp8type,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert,
    const float limit
    ) {
    int64_t token_idx = blockIdx.x * T_dim + blockIdx.y;
    int64_t num_total_tokens = num_tokens;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    __shared__ float s_token_scale; 

    for(; token_idx < num_total_tokens ; token_idx += gridDim.y)
    {
        if (tokens_per_expert != nullptr) {
            const int32_t e = token_idx / T_dim; 
            const int32_t t = token_idx % T_dim; 
            
            if (t >= tokens_per_expert[e]) {
                break; 
            }
        }

        using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
        using VecInt8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

        int idx = threadIdx.x * VEC;
        int tidx = threadIdx.x;
        const int64_t x_index = token_idx * 2 * d + idx;
        const int64_t y_index = token_idx * d + idx;
        
        VecType* x1 = (VecType*)(input + x_index);
        VecType* x2 = (VecType*)(input + x_index + d);
        VecInt8Type* y = (VecInt8Type*)(out + y_index);
        scalar_t r_x1[VEC];
        scalar_t r_x2[VEC];
        float r_y[VEC];

        if (idx < d) { 
            *(VecType*)r_x1 = *x1;
            *(VecType*)r_x2 = *x2;
            #pragma unroll
            for (int i = 0; i < VEC; i++) {
                r_y[i] = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(r_x1[i], r_x2[i], limit));
            }
        } 

        float row_max = 0.f;
        if (idx < d) {
            #pragma unroll
            for(int ii = 0;ii < VEC;ii++){
                row_max = fmaxf(row_max,fabsf(r_y[ii]));
            }
        }

        row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);

        float fp8type_max_v;     
        float min_scaling;    
        
        if (fp8type == 0) { // E4M3
            fp8type_max_v = quant_type_max_v<c10::Float8_e4m3fn>; 
            min_scaling = min_scaling_factor<c10::Float8_e4m3fn>::val();
        } else { // E5M2
            fp8type_max_v = quant_type_max_v<c10::Float8_e5m2>;   
            min_scaling = min_scaling_factor<c10::Float8_e5m2>::val();
        }

        float scale = 0.0f;
        if(tidx == 0){
            scale = fmaxf(row_max / fp8type_max_v, min_scaling); 
            s_token_scale = scale; 
            scales[token_idx] = scale; 
        }
        __syncthreads();

        float inv_s = 1.0f / s_token_scale;

        uint8_t out_vec[VEC];
        int loop = VEC / 4;
        if (idx < d) {
            #pragma unroll
            for(int ii = 0;ii < loop;ii++){
                if(fp8type==0)
                    ((uint32_t*)out_vec)[ii] = float_to_fp8e4m3_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
                else
                    ((uint32_t*)out_vec)[ii] = float_to_fp8e5m2_x4(r_y[ii*4] * inv_s, r_y[ii*4 + 1] * inv_s, r_y[ii*4 + 2] * inv_s, r_y[ii*4 + 3] * inv_s);
            }
            *y =*(VecInt8Type*)out_vec;
        }

    } // 结束 Grid-Stride Loop
}

void fuse_silu_mul_fp8_quant_ep(torch::Tensor& input, 
                                 torch::Tensor& out, 
                                 torch::Tensor& scales, 
                                 const int fp8type,
                                 torch::optional<torch::Tensor> &tokens_per_expert, 
                                 torch::optional<torch::Tensor> &num_local_tokens_tensor, 
                                 int topK,   
                                 int expert_m) 
{
  TORCH_CHECK(input.dim() == 3, "Input tensor must be 3-dimensional [E, T, H]"); 
  int32_t E = input.size(0);
  int32_t T = input.size(1); 
  int32_t H = input.size(2);
  int32_t d = H / 2;
  int32_t num_tokens = E * T; 
  
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  const int32_t* tokens_per_expert_ptr = 
        tokens_per_expert.has_value() 
            ? tokens_per_expert->data_ptr<int32_t>() 
            : nullptr;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "act_and_mul_kernel_ep", [&] {
        dim3 grid(E, 128);
        if (d <= 512) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 4, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T, 
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 1024) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 8, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E, 
                                        tokens_per_expert_ptr); 
        } else if (d <= 2048) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 8, true, 256>
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 4096) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 8, true, 512>
                <<<grid, 512, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 8192) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 8, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else if (d <= 16384) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 16, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else if (d <= 32768) {
            act_and_mul_fp8_quant_kernel_opt1_ep<scalar_t, 32, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else {
            act_and_mul_fp8_quant_kernel_opt1_ep_gernel<scalar_t, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        }
      });
}

#ifdef USE_ROCM
// The column count is dynamic; template parameters only control row tiling.
template <int ThreadsPerRow, int RowsPerBlock, int Segments, int FP8Type>
__global__ void act_and_mul_fp8_quant_ep_masked(
    uint8_t* __restrict__ out,
    const c10::BFloat16* __restrict__ input,
    float* __restrict__ scales,
    int d,
    int capacity,
    const int32_t* __restrict__ tokens_per_expert,
    float limit) {
  constexpr int Vec = 8;
  constexpr int Wave = C10_WARP_SIZE;
  static_assert(RowsPerBlock == 1 || ThreadsPerRow == Wave);
  using InputVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutputVec = at::native::memory::aligned_vector<uint8_t, Vec>;
  const int expert = blockIdx.x;
  const int row_in_block = threadIdx.x / ThreadsPerRow;
  const int lane = threadIdx.x % ThreadsPerRow;
  const int count = tokens_per_expert[expert];
  const int valid_rows = count < 0 ? 0 : (count > capacity ? capacity : count);
  const bool vector_aligned = d % Vec == 0;
  __shared__ float warp_max[ThreadsPerRow / Wave];
  __shared__ float inverse_scale;

  // Each row belongs to this expert, even at a full-capacity masked count.
  for (int row = blockIdx.y * RowsPerBlock + row_in_block;
       row < valid_rows; row += gridDim.y * RowsPerBlock) {
    const int64_t token = static_cast<int64_t>(expert) * capacity + row;
    float values[Segments][Vec];
    float max_abs = 0.0f;
    #pragma unroll
    for (int segment = 0; segment < Segments; ++segment) {
      const int col = (lane + segment * ThreadsPerRow) * Vec;
      if (col < d) {
        InputVec gate, up;
        if (vector_aligned) {
          gate = *reinterpret_cast<const InputVec*>(input + token * 2 * d + col);
          up = *reinterpret_cast<const InputVec*>(input + token * 2 * d + d + col);
        } else {
          #pragma unroll
          for (int i = 0; i < Vec; ++i) {
            gate.val[i] = col + i < d ? input[token * 2 * d + col + i] : c10::BFloat16(0);
            up.val[i] = col + i < d ? input[token * 2 * d + d + col + i] : c10::BFloat16(0);
          }
        }
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
          // Preserve BF16 rounding after SiLU and after multiplication.
          values[segment][i] = static_cast<float>(
              compute_with_swiglu_limit<c10::BFloat16, true>(gate.val[i], up.val[i], limit));
          max_abs = fmaxf(max_abs, fabsf(values[segment][i]));
        }
      }
    }

    constexpr float max_fp8 = FP8Type == 0 ? 448.0f : 57344.0f;
    constexpr float min_scale = 1.0f / (max_fp8 * 512.0f);
    float inverse;
    if constexpr (ThreadsPerRow == Wave) {
      max_abs = WarpReduceMax_ROW<float>(max_abs);
      float lane_inverse = 0.0f;
      if (lane == 0) {
        const float scale = fmaxf(max_abs / max_fp8, min_scale);
        scales[token] = scale;
        lane_inverse = 1.0f / scale;
      }
      inverse = __shfl(lane_inverse, 0, Wave);
    } else {
      max_abs = BlockReduceMax_ROW<float, ThreadsPerRow>(max_abs, warp_max);
      if (lane == 0) {
        const float scale = fmaxf(max_abs / max_fp8, min_scale);
        scales[token] = scale;
        inverse_scale = 1.0f / scale;
      }
      __syncthreads();
      inverse = inverse_scale;
    }

    #pragma unroll
    for (int segment = 0; segment < Segments; ++segment) {
      const int col = (lane + segment * ThreadsPerRow) * Vec;
      if (col < d) {
        OutputVec bytes;
        #pragma unroll
        for (int i = 0; i < Vec; i += 4) {
          uint32_t packed;
          if constexpr (FP8Type == 0) {
            packed = float_to_fp8e4m3_x4(
                values[segment][i] * inverse, values[segment][i + 1] * inverse,
                values[segment][i + 2] * inverse, values[segment][i + 3] * inverse);
          } else {
            packed = float_to_fp8e5m2_x4(
                values[segment][i] * inverse, values[segment][i + 1] * inverse,
                values[segment][i + 2] * inverse, values[segment][i + 3] * inverse);
          }
          *reinterpret_cast<uint32_t*>(&bytes.val[i]) = packed;
        }
        if (vector_aligned) {
          *reinterpret_cast<OutputVec*>(out + token * d + col) = bytes;
        } else {
          #pragma unroll
          for (int i = 0; i < Vec; ++i) {
            if (col + i < d) out[token * d + col + i] = bytes.val[i];
          }
        }
      }
    }
    if constexpr (ThreadsPerRow != Wave) {
      __syncthreads();
    }
  }
}

template <int ThreadsPerRow, int RowsPerBlock, int Segments>
static void launch_act_and_mul_fp8_quant_ep_masked(
    torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales,
    const int32_t* counts, int fp8type, float limit, cudaStream_t stream) {
  const int capacity = input.size(1);
  const int d = input.size(2) / 2;
  // A smaller expert count needs more row blocks to keep the device busy.
  // Multi-wave rows also need more blocks to avoid serializing dense routing.
  const int target_blocks = ThreadsPerRow == C10_WARP_SIZE ? 384 : 1536;
  int row_blocks = 8;
  while (row_blocks < 32 && row_blocks * input.size(0) < target_blocks) {
    row_blocks *= 2;
  }
  row_blocks = std::min(row_blocks, (capacity + RowsPerBlock - 1) / RowsPerBlock);
  const dim3 grid(input.size(0), row_blocks);
  const dim3 block(ThreadsPerRow * RowsPerBlock);
  if (fp8type == 0) {
    act_and_mul_fp8_quant_ep_masked<ThreadsPerRow, RowsPerBlock, Segments, 0>
        <<<grid, block, 0, stream>>>(out.data_ptr<uint8_t>(), input.data_ptr<c10::BFloat16>(),
                                    scales.data_ptr<float>(), d, capacity, counts, limit);
  } else {
    act_and_mul_fp8_quant_ep_masked<ThreadsPerRow, RowsPerBlock, Segments, 1>
        <<<grid, block, 0, stream>>>(out.data_ptr<uint8_t>(), input.data_ptr<c10::BFloat16>(),
                                    scales.data_ptr<float>(), d, capacity, counts, limit);
  }
}
#endif

static void fuse_silu_mul_fp8_quant_ep_with_limit(torch::Tensor& input, 
                                 torch::Tensor& out, 
                                 torch::Tensor& scales, 
                                 const int fp8type,
                                 torch::optional<torch::Tensor> &tokens_per_expert, 
                                 torch::optional<torch::Tensor> &num_local_tokens_tensor, 
                                 int topK,   
                                 int expert_m,
                                 float limit) 
{
  TORCH_CHECK(input.dim() == 3, "Input tensor must be 3-dimensional [E, T, H]"); 
  int32_t E = input.size(0);
  int32_t T = input.size(1); 
  int32_t H = input.size(2);
  int32_t d = H / 2;
  int32_t num_tokens = E * T; 
  
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  const int32_t* tokens_per_expert_ptr = 
        tokens_per_expert.has_value() 
            ? tokens_per_expert->data_ptr<int32_t>() 
            : nullptr;

#ifdef USE_ROCM
  // Use a small masked grid for narrow rows; wider rows retain the existing path.
  // Each row loop remains bounded by its own expert's valid count.
  if (d > 0 && d <= 4096 && input.scalar_type() == at::ScalarType::BFloat16 &&
      tokens_per_expert_ptr != nullptr && (fp8type == 0 || fp8type == 1) &&
      input.is_contiguous() && out.is_contiguous() && scales.is_contiguous() &&
      tokens_per_expert->is_contiguous()) {
    if (d <= 1024) {
      launch_act_and_mul_fp8_quant_ep_masked<64, 4, 2>(
          input, out, scales, tokens_per_expert_ptr, fp8type, limit, stream);
    } else if (d <= 2048) {
      launch_act_and_mul_fp8_quant_ep_masked<64, 4, 4>(
          input, out, scales, tokens_per_expert_ptr, fp8type, limit, stream);
    } else {
      launch_act_and_mul_fp8_quant_ep_masked<64, 4, 8>(
          input, out, scales, tokens_per_expert_ptr, fp8type, limit, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "act_and_mul_kernel_ep", [&] {
        dim3 grid(E, 128);
        if (d <= 512) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 4, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T, 
                                        E,
                                        tokens_per_expert_ptr,
                                        limit); 
        } else if (d <= 1024) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 8, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E, 
                                        tokens_per_expert_ptr,
                                        limit); 
        } else if (d <= 2048) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 8, true, 256>
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr,
                                        limit); 
        } else if (d <= 4096) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 8, true, 512>
                <<<grid, 512, 0, stream>>>(num_tokens ,
                                        out.data_ptr<uint8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        fp8type,
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr,
                                        limit); 
        } else if (d <= 8192) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 8, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr,
                                            limit); 
        } else if (d <= 16384) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 16, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr,
                                            limit); 
        } else if (d <= 32768) {
            act_and_mul_fp8_quant_kernel_opt1_ep_with_limit<scalar_t, 32, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr,
                                            limit); 
        } else {
            act_and_mul_fp8_quant_kernel_opt1_ep_gernel_with_limit<scalar_t, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<uint8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            fp8type,
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr,
                                            limit); 
        }
      });
}

void fuse_silu_mul_fp8_quant_ep(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& scales,
    const int fp8type,
    torch::optional<torch::Tensor> &tokens_per_expert,
    torch::optional<torch::Tensor> &num_local_tokens_tensor,
    int topK,
    int expert_m,
    float limit)
{
  TORCH_CHECK(
      limit == -1.0f || limit > 0.0f,
      "limit must be None (encoded as -1.0) or a positive value");

  if (limit > 0.0f) {
    fuse_silu_mul_fp8_quant_ep_with_limit(
        input, out, scales, fp8type, tokens_per_expert,
        num_local_tokens_tensor, topK, expert_m, limit);
    return;
  }

  fuse_silu_mul_fp8_quant_ep(
      input, out, scales, fp8type, tokens_per_expert,
      num_local_tokens_tensor, topK, expert_m);
}

}
}
