#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
#include "reduce.h"

// #include "cuda_compat.h"
// #include "../dispatch_utils.h"

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

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

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

  // 只进行一次 float -> scalar_t 转换
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

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){
  // for(; token_idx < num_tokens ;token_idx += gridDim.x){
    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
    __shared__ float s_token_scale; 
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
  }
}

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_gernel(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
  int64_t token_idx= blockIdx.x;
  int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){
    int tidx = threadIdx.x;
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      scalar_t val = compute<scalar_t, act_first>(x, y);
      row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
    }
    row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
    __shared__ float s_token_scale; 
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

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y));
      uint8_t q_val;
      if(fp8type==0)
        q_val = float_to_fp8e4m3(val * inv_s);
      else
        q_val = float_to_fp8e5m2(val * inv_s);
      out[output_offset + idx] = static_cast<uint8_t>(q_val);
    }
  }
}


template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_new(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    int32_t *expert_ids
    ) {

    int64_t token_idx= blockIdx.x;
    if(expert_ids != nullptr){
      if(expert_ids[token_idx] == -1)  return;
    }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
      __shared__ float s_token_scale;
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
}

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_new_gernel(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    int32_t *expert_ids
    ) {
    
    int64_t token_idx= blockIdx.x;
    if(expert_ids != nullptr){
      if(expert_ids[token_idx] == -1)  return;
    }
    int tidx = threadIdx.x;
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      scalar_t val = compute<scalar_t, act_first>(x, y);
      row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
    }
    row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
    __shared__ float s_token_scale; 
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

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y));
      uint8_t q_val;
      if(fp8type==0)
        q_val = float_to_fp8e4m3(val * inv_s);
      else
        q_val = float_to_fp8e5m2(val * inv_s);
      out[output_offset + idx] = static_cast<uint8_t>(q_val);
    }
  
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_one(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(token_idx >= num_local_tokens_tensor[0] * topK)  return;
  }
    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
    __shared__ float s_token_scale; 
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

}

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_one_gernel(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
  const int64_t token_idx = blockIdx.x;
  int tidx = threadIdx.x;
  const int64_t input_offset = token_idx * 2 * d;
  const int64_t output_offset = token_idx * d;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float shared_mem[share_size];

  float row_max = 0.0f;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
    const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
    scalar_t val = compute<scalar_t, act_first>(x, y);
    row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
  }
  row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
  __shared__ float s_token_scale; 

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

  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
    const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
    float val = static_cast<float>(compute<scalar_t, act_first>(x, y));
    uint8_t q_val;
    if(fp8type==0)
      q_val = float_to_fp8e4m3(val * inv_s);
    else
      q_val = float_to_fp8e5m2(val * inv_s);
    out[output_offset + idx] = static_cast<uint8_t>(q_val);
  }
}



template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor,
    const float limit
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){
  // for(; token_idx < num_tokens ;token_idx += gridDim.x){
    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
    __shared__ float s_token_scale; 
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
  }
}

template <typename scalar_t, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_gernel_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor,
    const float limit
    ) {
  int64_t token_idx= blockIdx.x;
  int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){
    int tidx = threadIdx.x;
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const scalar_t val =
          static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
      row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
    }
    row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
    __shared__ float s_token_scale; 
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

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float val = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
      uint8_t q_val;
      if(fp8type==0)
        q_val = float_to_fp8e4m3(val * inv_s);
      else
        q_val = float_to_fp8e5m2(val * inv_s);
      out[output_offset + idx] = static_cast<uint8_t>(q_val);
    }
  }
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_new_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    int32_t *expert_ids,
    const float limit
    ) {

    int64_t token_idx= blockIdx.x;
    if(expert_ids != nullptr){
      if(expert_ids[token_idx] == -1)  return;
    }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
      __shared__ float s_token_scale;
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
}

template <typename scalar_t, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_new_gernel_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    int32_t *expert_ids,
    const float limit
    ) {
    
    int64_t token_idx= blockIdx.x;
    if(expert_ids != nullptr){
      if(expert_ids[token_idx] == -1)  return;
    }
    int tidx = threadIdx.x;
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const scalar_t val =
          static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
      row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
    }
    row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
    __shared__ float s_token_scale; 
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

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float val = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
      uint8_t q_val;
      if(fp8type==0)
        q_val = float_to_fp8e4m3(val * inv_s);
      else
        q_val = float_to_fp8e5m2(val * inv_s);
      out[output_offset + idx] = static_cast<uint8_t>(q_val);
    }
  
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_one_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor,
    const float limit
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(token_idx >= num_local_tokens_tensor[0] * topK)  return;
  }
    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
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
    __shared__ float s_token_scale; 
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

}

template <typename scalar_t, bool act_first, int block_size>
__global__ void act_and_mul_fp8_quant_kernel_opt1_one_gernel_with_limit(
    int64_t num_tokens,
    const int fp8type,
    uint8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor,
    const float limit
    ) {
  const int64_t token_idx = blockIdx.x;
  int tidx = threadIdx.x;
  const int64_t input_offset = token_idx * 2 * d;
  const int64_t output_offset = token_idx * d;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float shared_mem[share_size];

  float row_max = 0.0f;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
    const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
    const scalar_t val = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
    row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
  }
  row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
  __shared__ float s_token_scale; 

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

  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
    const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
    const float val = static_cast<float>(compute_with_swiglu_limit<scalar_t, act_first>(x, y, limit));
    uint8_t q_val;
    if(fp8type==0)
      q_val = float_to_fp8e4m3(val * inv_s);
    else
      q_val = float_to_fp8e5m2(val * inv_s);
    out[output_offset + idx] = static_cast<uint8_t>(q_val);
  }
}

void fuse_silu_mul_fp8_quant(torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids)
{
  int d = input.size(-1) / 2;                                                  
  int64_t num_tokens = input.numel() / input.size(-1);                         
  int grid_opt = num_tokens;                                             
  if(num_local_tokens_tensor.has_value()){
    if(expert_m != -1){
      grid_opt = std::max(1, expert_m * topK);
    }
    else{
      if(num_tokens < 8192){                                                
        grid_opt = std::max(1, num_tokens);                                   
      }else if(num_tokens < 16384){                                            
        grid_opt = std::max(1, num_tokens/4);                                  
      }else{                                                                 
        grid_opt = std::max(1, num_tokens/8);                                      
      }                                                                          
    }
  }                                                                        
  dim3 grid(grid_opt);                                                   
  dim3 block(std::min(d, 1024));                                               
  if (num_tokens == 0) {                                                       
    return;                                                                    
  }                                                                            
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                
  VLLM_DISPATCH_FLOATING_TYPES(                                                
      input.scalar_type(), "act_and_mul_kernel", [&] {                         
      if(expert_ids.has_value()){                                      
          if (d <= 512) {                                                      
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 4, true, 128> 
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),          
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d, 
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     
          } else if (d <= 1024) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 8, true, 128> 
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d,
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     
          } else if (d <= 2048) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 8, true, 256> 
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d,
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     
          } else if (d <= 4096) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 8, true, 512> 
                <<<grid, 512, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(),      
                                           scales.data_ptr<float>(),d, 
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     
          } else if (d <= 8192) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 8, true, 1024> 
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);    
          } else if (d <= 16384) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 16, true, 1024> 
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);    
          } else if (d <= 32768) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new<scalar_t, 32, true, 1024> 
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);    
          } else{
            act_and_mul_fp8_quant_kernel_opt1_new_gernel<scalar_t, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);  
          }                                                                
 
      }
      else{
        if(num_local_tokens_tensor.has_value()){
          if(num_tokens < 8192 && expert_m == -1){                                       
              if (d <= 512) {                                                      
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 4, true, 128> 
                    <<<grid, 256, 0, stream>>>(num_tokens , 
                                              fp8type,                 
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 1024) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 128> 
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 2048) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 256> 
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 4096) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 512> 
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 8192) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);
                                                    
              } else if (d <= 16384) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 16, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else if (d <= 32768) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 32, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else{
                act_and_mul_fp8_quant_kernel_opt1_one_gernel<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr); 
              }                                                                    

          }                                                                     
            else{                                       
              if (d <= 512) {                                                      
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 4, true, 128> 
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 1024) {                                              
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 8, true, 128> 
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 2048) {                                              
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 8, true, 256> 
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 4096) {                                              
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 8, true, 512> 
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 8192) {                                                             
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 8, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);
                                                    
              } else if (d <= 16384) {                                                             
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 16, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else if (d <= 32768) {                                                             
                act_and_mul_fp8_quant_kernel_opt1<scalar_t, 32, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else{
                act_and_mul_fp8_quant_kernel_opt1_gernel<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              }                                                                     
            }
        }else{
              if (d <= 512) {                                                      
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 4, true, 128> 
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,                  
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 1024) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 128> 
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type, 
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 2048) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 256> 
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 4096) {                                              
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 512> 
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     
              } else if (d <= 8192) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 8, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);
                                                    
              } else if (d <= 16384) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 16, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else if (d <= 32768) {                                                             
                act_and_mul_fp8_quant_kernel_opt1_one<scalar_t, 32, true, 1024> 
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);   
              } else{
                act_and_mul_fp8_quant_kernel_opt1_one_gernel<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr); 
              } 
          }
      }
      });
}

static void fuse_silu_mul_fp8_quant_with_limit(torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids, float limit)
{
  int d = input.size(-1) / 2;                                                  
  int64_t num_tokens = input.numel() / input.size(-1);                         
  int grid_opt = num_tokens;                                             
  if(num_local_tokens_tensor.has_value()){
    if(expert_m != -1){
      grid_opt = std::max(1, expert_m * topK);
    }
    else{
      if(num_tokens < 8192){                                                
        grid_opt = std::max(1, num_tokens);                                   
      }else if(num_tokens < 16384){                                            
        grid_opt = std::max(1, num_tokens/4);                                  
      }else{                                                                 
        grid_opt = std::max(1, num_tokens/8);                                      
      }                                                                          
    }
  }                                                                        
  dim3 grid(grid_opt);                                                   
  dim3 block(std::min(d, 1024));                                               
  if (num_tokens == 0) {                                                       
    return;                                                                    
  }                                                                            
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                
  VLLM_DISPATCH_FLOATING_TYPES(                                                
      input.scalar_type(), "act_and_mul_kernel", [&] {                         
      if(expert_ids.has_value()){                                      
          if (d <= 512) {                                                      
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 4, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),          
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d, 
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);     
          } else if (d <= 1024) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 8, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d,
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);     
          } else if (d <= 2048) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 8, true, 256>
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(), 
                                           scales.data_ptr<float>(),d,
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);     
          } else if (d <= 4096) {                                              
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 8, true, 512>
                <<<grid, 512, 0, stream>>>(num_tokens ,
                                           fp8type,
                                           out.data_ptr<uint8_t>(),           
                                           input.data_ptr<scalar_t>(),      
                                           scales.data_ptr<float>(),d, 
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);     
          } else if (d <= 8192) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 8, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);    
          } else if (d <= 16384) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 16, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);    
          } else if (d <= 32768) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_new_with_limit<scalar_t, 32, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);    
          } else{
            act_and_mul_fp8_quant_kernel_opt1_new_gernel_with_limit<scalar_t, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            fp8type,
                                            out.data_ptr<uint8_t>(),          
                                            input.data_ptr<scalar_t>(),     
                                            scales.data_ptr<float>(),d, 
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr, limit);  
          }                                                                
 
      }
      else{
        if(num_local_tokens_tensor.has_value()){
          if(num_tokens < 8192 && expert_m == -1){                                       
              if (d <= 512) {                                                      
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 4, true, 128>
                    <<<grid, 256, 0, stream>>>(num_tokens , 
                                              fp8type,                 
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 1024) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 128>
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 2048) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 256>
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 4096) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 512>
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 8192) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);
                                                    
              } else if (d <= 16384) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 16, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else if (d <= 32768) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 32, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else{
            act_and_mul_fp8_quant_kernel_opt1_one_gernel_with_limit<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit); 
              }                                                                    

          }                                                                     
            else{                                       
              if (d <= 512) {                                                      
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 4, true, 128>
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 1024) {                                              
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 8, true, 128>
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 2048) {                                              
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 8, true, 256>
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 4096) {                                              
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 8, true, 512>
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 8192) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 8, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);
                                                    
              } else if (d <= 16384) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 16, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else if (d <= 32768) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_with_limit<scalar_t, 32, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else{
            act_and_mul_fp8_quant_kernel_opt1_gernel_with_limit<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              }                                                                     
            }
        }else{
              if (d <= 512) {                                                      
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 4, true, 128>
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type,                  
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 1024) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 128>
                    <<<grid, 128, 0, stream>>>(num_tokens ,
                                              fp8type, 
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 2048) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 256>
                    <<<grid, 256, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(), 
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 4096) {                                              
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 512>
                    <<<grid, 512, 0, stream>>>(num_tokens ,
                                              fp8type,
                                              out.data_ptr<uint8_t>(),           
                                              input.data_ptr<scalar_t>(),      
                                              scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);     
              } else if (d <= 8192) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 8, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);
                                                    
              } else if (d <= 16384) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 16, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else if (d <= 32768) {                                                             
            act_and_mul_fp8_quant_kernel_opt1_one_with_limit<scalar_t, 32, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit);   
              } else{
            act_and_mul_fp8_quant_kernel_opt1_one_gernel_with_limit<scalar_t, true, 1024>
                    <<<grid, 1024, 0, stream>>>(num_tokens ,
                                                fp8type,
                                                out.data_ptr<uint8_t>(),          
                                                input.data_ptr<scalar_t>(),     
                                                scales.data_ptr<float>(), d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr, limit); 
              } 
          }
      }
      });
}

void fuse_silu_mul_fp8_quant(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& scales,
    const int fp8type,
    torch::optional<torch::Tensor> &num_local_tokens_tensor,
    int topK,
    int expert_m,
    torch::optional<torch::Tensor> &expert_ids,
    float limit)
{
  TORCH_CHECK(
      limit == -1.0f || limit > 0.0f,
      "limit must be None (encoded as -1.0) or a positive value");

  if (limit > 0.0f) {
    fuse_silu_mul_fp8_quant_with_limit(
        input, out, scales, fp8type, num_local_tokens_tensor,
        topK, expert_m, expert_ids, limit);
    return;
  }

  fuse_silu_mul_fp8_quant(
      input, out, scales, fp8type, num_local_tokens_tensor,
      topK, expert_m, expert_ids);
}


}
}
