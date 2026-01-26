#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "reduce.h"
#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #include <cub/util_type.cuh>
#else
  #include <hipcub/hipcub.hpp>
  #include <hipcub/util_type.hpp>
#endif

namespace at{
namespace native{

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}


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
  // lj
  // FP32 Constants
  constexpr int fp32_bits = 32;
  constexpr int fp32_mant_bits = 23;
  constexpr int fp32_bias = 127;

  // FP8 E4M3 Constants (通常用于 forward/activation)
  constexpr int e4m3_mant_bits = 3;
  constexpr int e4m3_bias = 7; 

  // FP8 E5M2 Constants (通常用于 backward/gradient)
  constexpr int e5m2_bits = 8;
  inline __device__ uint8_t float_to_fp8e4m3(float src) {
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint32_t __src = *(unsigned int*)&src;
    // fetch sign bits
    uint8_t sign_bits = (__src & 0x80000000) >> (fp32_bits - e5m2_bits);
    // fetch exponent bitss
    uint32_t exp_bits  = __src & 0x7f800000;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x007fffff;
    // fetch absolute value
    uint32_t data_scale = __src & 0x7fffffff;
    // categorical discussions
    /* NAN */
    uint8_t result = 0x0;
    if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
        // result = sign_bits | 0x7f; // output qNAN
        result = 0x7f; // for NV's __nv_cvt_float_to_fp8:cvt.rn.satfinite.e4m3x2.f32, output are all 0x7f
    }
    /* inf or greater than MAX value of E5M2 */
    else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000)) {
        result = sign_bits | 0x7e; // output MAX
    }
    /* less than MIN of denorm */
    else if (exp_bits <= 0x3a800000) {
        result = (exp_bits == 0x3a800000 and mant_bits > 0x0) ? sign_bits | 0x1: sign_bits;
    }
    /* others */
    else {
        /* norm fp32 can be represented as denorm fp8 / norm */
        mant_bits = exp_bits < 0x3c800000 ? (0x800000 | mant_bits) >> ((0x3c800000 - exp_bits) >> fp32_mant_bits): mant_bits;
        exp_bits  = exp_bits < 0x3c800000 ? 0x0: ((exp_bits >> fp32_mant_bits) - (fp32_bias - e4m3_bias)) << e4m3_mant_bits;
        // get discard bits
        uint32_t discard_bits = mant_bits & 0xfffff;
        // rounding
        bool carry_a_bit = discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000));
        mant_bits = (mant_bits & 0x700000) >> (fp32_mant_bits - e4m3_mant_bits);
        mant_bits = carry_a_bit ? mant_bits + 1: mant_bits;
        result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
    }
    return result;
}
  // lj

    // inline __device__ uint8_t float_to_fp8e4m3(float f) {
    // // ===== 常量定义 =====
    // // FP8 E4M3最大值: 1087 << 20
    // // 1087 = 2^(4+1) - 1 + 2^4 = 31 + 16 = 47? 等等，这里需要重新计算
    // // 实际上是基于指数和尾数的位模式
    // constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

    // // 非正规化数处理掩码
    // constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

    // // ===== 步骤1: 提取float的位表示 =====
    // uint32_t f_bits = c10::detail::fp32_to_bits(f);
    // uint8_t result = 0u;

    // // ===== 步骤2: 提取符号位并移除 =====
    // const uint32_t sign = f_bits & UINT32_C(0x80000000);
    // f_bits ^= sign;

    // // ===== 步骤3: 处理不同数值范围 =====

    // // 3a. 溢出处理
    // if (f_bits >= fp8_max) {
    //     result = 0x7f;  // 01111111 (NaN/最大值)
    // } else {
        
    //     // 3b. 非正规化数处理
    //     if (f_bits < (UINT32_C(121) << 23)) {
    //     f_bits = c10::detail::fp32_to_bits(
    //         c10::detail::fp32_from_bits(f_bits) + 
    //         c10::detail::fp32_from_bits(denorm_mask)
    //     );
    //     result = static_cast<uint8_t>(f_bits - denorm_mask);
        
    //     // 3c. 正规化数处理
    //     } else {
    //     // 检查尾数第20位是否为1（对应FP8 E4M3 3位尾数的LSB）
    //     uint8_t mant_odd = (f_bits >> 20) & 1;
        
    //     // 指数偏置调整: FP32偏置127 → FP8 E4M3偏置7
    //     // 调整量 = 7 - 127 = -120
    //     f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        
    //     // 应用奇数舍入
    //     f_bits += mant_odd;
        
    //     // 右移20位得到8位结果
    //     result = static_cast<uint8_t>(f_bits >> 20);
    //     }
    // }

    // // ===== 步骤4: 加上符号位 =====
    // result |= static_cast<uint8_t>(sign >> 24);

    // return result;
    // }
//lj

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

template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512, bool inpit_update=false>
__global__ void rms_norm_fp8_quant_(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps,scalar_out_t* out,float* scales, int stride, int fp8type)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  scalar_t intput_vec[Vec];
  T_ACC trstd;
  int64_t idx = bid * tcol + tid;
  idx *= Vec;
  int64_t index = bid * stride + tid * Vec;
  if (tid < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+index);
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
          val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
      }
  }
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd=s_rstd;

  float block_absmax_val_maybe = 0.f;
  scalar_t elem[Vec];
  if (tid < tcol) {
      #pragma unroll
      for(int ii = 0;ii < Vec;ii++){
        int jj=tid * Vec + ii;
        elem[ii] = static_cast<T_ACC>(intput_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
      }      
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
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
  if(tid == 0){
      scale = fmaxf(block_absmax_val_maybe / fp8type_max_v, min_scaling); 
      s_token_scale = scale; 
      scales[bid] = scale; 
  }
  __syncthreads();

  float inv_s = 1.0f / s_token_scale;

  scalar_out_t out_vec[Vec];
  int loop = Vec / 4;
  if (tid < tcol) {
    #pragma unroll
    for(int ii = 0;ii < loop;ii++){
      if(fp8type==0)
          ((uint32_t*)out_vec)[ii] = float_to_fp8e4m3_x4(elem[ii*4] * inv_s, elem[ii*4 + 1] * inv_s, elem[ii*4 + 2] * inv_s, elem[ii*4 + 3] * inv_s);
      else
          ((uint32_t*)out_vec)[ii] = float_to_fp8e5m2_x4(elem[ii*4] * inv_s, elem[ii*4 + 1] * inv_s, elem[ii*4 + 2] * inv_s, elem[ii*4 + 3] * inv_s);
    }
    *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;
    if constexpr (inpit_update){
      *(LoadT*)(input+index)=*(LoadT*)elem;
    }
  }
}

template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec,int block_size>
__global__ void rms_norm_fp8_quant_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps, scalar_out_t* out,float* scales, int stride, int fp8type)
{
  constexpr int share_size=block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0.0f;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  T_ACC trstd;
  int64_t rows = bid * stride;

  for(int i = tid;i < cols;i += block_size)
  {
    val += static_cast<T_ACC>(input[rows + i])*static_cast<T_ACC>(input[rows + i]);
  }
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd = c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd = s_rstd;
  float block_absmax_val_maybe = 0.0f;
  for(int i = tid;i < cols;i += block_size){
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(input[i + rows]) * trstd * static_cast<T_ACC>(gamma[i])));
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe,val_shared);

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
  if(tid == 0){
      scale = fmaxf(block_absmax_val_maybe / fp8type_max_v, min_scaling); 
      s_token_scale = scale; 
      scales[bid] = scale; 
  }
  __syncthreads();
  float inv_s = 1.0f / s_token_scale;
  scalar_out_t out_vec[Vec];
  for(int i = tid; i < cols; i += block_size)
  {
    out[rows + i] = float_to_fp8e4m3(static_cast<T_ACC>(input[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]) * inv_s);
    input[rows + i] = static_cast<T_ACC>(input[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]);
  }
}


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512,bool inpit_update=false>
__global__ void rms_norm_fp8_quant_residual(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride, int fp8type)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  scalar_t intput_vec[Vec];
  scalar_t residual_vec[Vec];
  T_ACC trstd;
  int64_t idx = bid * tcol + tid;
  idx *= Vec;
  if (tid < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+idx);
      *(LoadT*)residual_vec = *(LoadT*)(residual+idx);

      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
         residual_vec[ii] += (intput_vec[ii]);  
          val += static_cast<T_ACC>(residual_vec[ii])*static_cast<T_ACC>(residual_vec[ii]);
      }
  }

  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd=s_rstd;

  float block_absmax_val_maybe = 0.f;
  scalar_t elem[Vec];
  if (tid < tcol) {
      #pragma unroll
      for(int ii = 0;ii < Vec;ii++){
        int jj=tid * Vec + ii;
        elem[ii] = static_cast<T_ACC>(residual_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
      }      
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
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
  if(tid == 0){
      scale = fmaxf(block_absmax_val_maybe / fp8type_max_v, min_scaling); 
      s_token_scale = scale; 
      scales[bid] = scale; 
  }
  __syncthreads();
  float inv_s = 1.0f / s_token_scale;
  
  scalar_out_t out_vec[Vec];
  int loop = Vec / 4;
  if (tid < tcol) {
  #pragma unroll
  for(int ii = 0;ii < loop;ii++){
    if(fp8type==0)
        ((uint32_t*)out_vec)[ii] = float_to_fp8e4m3_x4(elem[ii*4] * inv_s, elem[ii*4 + 1] * inv_s, elem[ii*4 + 2] * inv_s, elem[ii*4 + 3] * inv_s);
    else
        ((uint32_t*)out_vec)[ii] = float_to_fp8e5m2_x4(elem[ii*4] * inv_s, elem[ii*4 + 1] * inv_s, elem[ii*4 + 2] * inv_s, elem[ii*4 + 3] * inv_s);
  }
  
    *(LoadT*)(residual+idx) = *(LoadT*)residual_vec;
    *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;
  if constexpr (inpit_update){
    *(LoadT*)(input+idx)=*(LoadT*)elem;  
  }
    
  }  
}


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=1,int block_size=512>
__global__ void rms_norm_fp8_quant_residual_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride, int fp8type)
{
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0.0f;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  T_ACC trstd;
  int64_t rows = bid * cols;

  for(int i = tid;i < cols;i += block_size)
  {
    float x = static_cast<T_ACC>(residual[rows + i]) + static_cast<T_ACC>(input[rows + i]);
        val += x * x;
  }
    
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd = s_rstd;
  float block_absmax_val_maybe = 0.0f;

  for(int i = tid;i < cols;i += block_size){
    T_ACC x = static_cast<T_ACC>(residual[i + rows]) + static_cast<T_ACC>(input[i + rows]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(static_cast<T_ACC>(x * trstd) * gamma[i])));
  }

  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);

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
  if(tid == 0){
      scale = fmaxf(block_absmax_val_maybe / fp8type_max_v, min_scaling); 
      s_token_scale = scale; 
      scales[bid] = scale; 
  }
  __syncthreads();
  float inv_s = 1.0f / s_token_scale;

  scalar_out_t out_vec[Vec];
  for(int i = tid;i < cols;i += block_size)
  {
    residual[rows + i] = static_cast<T_ACC>(residual[i + rows]) + static_cast<T_ACC>(input[i + rows]);
    out[rows + i] = float_to_fp8e4m3(static_cast<T_ACC>(residual[rows + i] ) * trstd * static_cast<T_ACC>(gamma[i]) * inv_s);
    input[rows + i] = static_cast<T_ACC>(residual[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]);
  }

}


template <typename scalar_in_t>
void fuse_rms_norm_fp8_quant(
  torch::Tensor& out,           // [..., hidden_size]
  torch::Tensor& input,   // [..., hidden_size]
  torch::Tensor const& weight,  // [hidden_size]
  torch::Tensor& scales,        // [num_tokens]
  double const epsilon,     // Variance epsilon used in norm calculation
  std::optional<at::Tensor> residual,
  bool input_flag,
  int fp8type) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0 && wt_ptr % 16 == 0;
  int input_ndim = input.dim();
  int stride;
  if(input_ndim > 1){
    stride = input.stride(-2);
  }else{
    stride = input.stride(-1);
  }
  if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned && input_flag){
        using T_ACC = at::acc_type<scalar_in_t, true>;
        
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();

        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  
          "fused_add_rms_norm_kernel_out_type",
          [&] {
             using scalar_out_t = scalar_t;
        scalar_out_t* out_data = out.data_ptr<scalar_out_t>();
        float* scaless = scales.data_ptr<float>();
        if(residual_flag){
          if (hidden_size <= 512){
              if(num_tokens <= 1024){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else if(num_tokens <= 3072){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else{
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                  
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 1000){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  }
              else if(num_tokens < 8000){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }  
              else{
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                else{
                    rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }
            else{
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }
          else if (hidden_size <= 1024){
                rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                else{
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                }
                else{
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
            else{
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
        }
        
      });

  }
  else  if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned && input_flag==false){
        using T_ACC = at::acc_type<scalar_in_t, true>;
        
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();

        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  
          "fused_add_rms_norm_kernel_out_type",
          [&] {
             using scalar_out_t = scalar_t;
        scalar_out_t* out_data = out.data_ptr<scalar_out_t>();
        float* scaless = scales.data_ptr<float>();
        if(residual_flag){
          if (hidden_size <= 512){
              if(num_tokens <= 1024){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else if(num_tokens < 4096){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else{
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 512){
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,false><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  } 
              else{
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                else{
                    rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
            }
            else if(hidden_size <= 8192){
              if(num_tokens <= 4){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else if(num_tokens < 64){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else if(num_tokens < 290){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else if(num_tokens < 2000){
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else{
                rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                  
          }
            else{
                  rms_norm_fp8_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }
          else if (hidden_size <= 1024){
                rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
                else{
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,256,false><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                }
                else{
                    rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
            else{
                  rms_norm_fp8_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
        }
        
      });

  }
  else{
        using T_ACC = at::acc_type<scalar_in_t, true>;
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();
        float* scaless = scales.data_ptr<float>();
        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  // scalar_out_t 跟 output 对应
          "fused_add_rms_norm_kernel_out_type",
          [&] {
          scalar_t* out_data = out.data_ptr<scalar_t>();
        if(residual_flag){
            if (hidden_size <= 1024){
                rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
            else if(hidden_size <= 2048){
              if(num_tokens >= 1024){
                  rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
              else{
                  rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }  
              
            }
            else if(hidden_size <= 4096){
              if(num_tokens > 1200){
                  rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  
              }
              else{
                  rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
              }
            }
            else {
                rms_norm_fp8_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }

        }else{
            if (hidden_size <= 1024){
                rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                    rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                }
                else{
                    rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                }                      

            }
            else if(hidden_size <= 4096){
                if(num_tokens > 1200){
                    rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                  }
                else{
                    rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
                }
            }
            else{
                  rms_norm_fp8_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,fp8type);
            }
        }

      });
  }
    }


void rms_norm_fp8_quant(
  Tensor &out,
  Tensor& input,
  const Tensor& weight,
  Tensor scales, 
  double epsilon,
  std::optional<Tensor> residual,
  bool update_input, 
  int fp8type
) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_fp8_quant_kernel",
    [&] {
      fuse_rms_norm_fp8_quant<scalar_t>(out,input,weight,scales,epsilon,residual,update_input,fp8type);
    });
}
}}
