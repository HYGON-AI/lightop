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


#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {  
namespace native {

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

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
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  
  if (f_bits >= fp8_max) {
    result = 0x7f;  
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

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val =  fmaxf(val,  __shfl_down(val, offset));
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceMax_ROW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceMax_ROW<T>(val);
  if constexpr(block_size==C10_WARP_SIZE)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0&&lid<share_size) {
      val = WarpReduceMax_ROW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

template <int THREADS_PER_GROUP>
__inline__ __device__ float WarpGroupReduceMax(float val) {
    #pragma unroll
    for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor(val, mask));
    }
    return val;
}

template <typename scalar_t, int THREADS_PER_GROUP, bool is_e4m3>
__global__ void per_token_group_quant_fp8_small_kernel(
    const scalar_t* __restrict__ input, 
    uint8_t* __restrict__ output,
    float* scale_out, 
    const int group_size,
    const float eps,
    const bool use_ue8m0,
    const int64_t total_groups) {

    constexpr int VEC = 16;
    
  
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
   
    int global_group_id = global_tid / THREADS_PER_GROUP;
    int lane_in_group = global_tid % THREADS_PER_GROUP;

    
    if (global_group_id >= total_groups) return;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    
    int64_t idx = (int64_t)global_group_id * group_size + lane_in_group * VEC;

    scalar_t input_vec[VEC];
    *(VecType*)input_vec = *(VecType*)(input + idx);

    float input_float_vec[VEC];
    float max_val = 0.f;
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        input_float_vec[ii] = static_cast<float>(input_vec[ii]);
        max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
    }

    max_val = WarpGroupReduceMax<THREADS_PER_GROUP>(max_val);

    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
    max_val = fmaxf(max_val, eps);
    float scale_raw = max_val / FP8_MAX_VAL;
    
    float s_group_scale;
    if (use_ue8m0) {
        s_group_scale = exp2f(ceilf(log2f(scale_raw)));
    } else {
        s_group_scale = scale_raw;
    }

    if (lane_in_group == 0) {
        scale_out[global_group_id] = s_group_scale;
    }

    float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;

    uint8_t out_vec[VEC];
    #pragma unroll
    for(int ii = 0; ii < VEC; ii+=4){
        uint32_t packed;
        if constexpr (is_e4m3) {
            packed = float_to_fp8e4m3_x4(
                input_float_vec[ii] * inv_s,
                input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s,
                input_float_vec[ii+3] * inv_s);
        } else {
            packed = float_to_fp8e5m2_x4(
                input_float_vec[ii] * inv_s,
                input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s,
                input_float_vec[ii+3] * inv_s);
        }
        *(uint32_t*)(&out_vec[ii]) = packed;
    }

    *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_vec;
}

template <typename scalar_t, int block_size, bool is_e4m3>
__global__ void per_token_group_quant_fp8_kernel(
    const scalar_t* __restrict__ input, 
    uint8_t* __restrict__ output,
    float* scale_out, 
    const int group_size,   
    const float eps,        
    const bool use_ue8m0) { 

    constexpr int VEC = 16;
    int bid = blockIdx.x;  
    
    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    __shared__ float s_group_scale;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int tid = threadIdx.x;
    int tcol = group_size / VEC;
    
    int64_t idx = (int64_t)bid * tcol + tid; 
    idx *= VEC;

    float input_float_vec[VEC];
    float max_val = 0.f;
    if (tid < tcol) {
        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(VecType*)(input+idx);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input_float_vec[ii] = static_cast<float>(input_vec[ii]);
            max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
        }
    }
    
    max_val = BlockReduceMax_ROW<float, block_size>(max_val, val_shared);
    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;

    if(tid == 0){
        max_val = fmaxf(max_val, eps);
        float scale_raw = max_val / FP8_MAX_VAL;
        if (use_ue8m0) {
            s_group_scale = exp2f(ceilf(log2f(scale_raw)));
        } else {
            s_group_scale = scale_raw;
        }
        scale_out[bid] = s_group_scale;
    }
    __syncthreads();

    float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;

    uint8_t out_vec[VEC];
    if (tid < tcol) {
        #pragma unroll
        for(int ii = 0; ii < VEC; ii+=4){
            uint32_t packed;
            if constexpr (is_e4m3) {
                packed = float_to_fp8e4m3_x4(
                    input_float_vec[ii] * inv_s,
                    input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s,
                    input_float_vec[ii+3] * inv_s);
            } else {
                packed = float_to_fp8e5m2_x4(
                    input_float_vec[ii] * inv_s,
                    input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s,
                    input_float_vec[ii+3] * inv_s);
            }
            *(uint32_t*)(&out_vec[ii]) = packed;
        }
        *(VecFp8Type*)(output+idx) = *(VecFp8Type*)out_vec;
    }
}

void per_token_group_quant_fp8(
    torch::Tensor& out,        
    torch::Tensor  input, 
    torch::Tensor& scales,
    int group_size,            
    float eps = 1e-5f,         
    bool use_ue8m0 = false)    
{
    input = input.is_contiguous() ? input : input.contiguous();
    TORCH_CHECK(out.scalar_type() == torch::kFloat8_e4m3fn || out.scalar_type() == torch::kFloat8_e5m2, 
                "Output tensor must be of type float8_e4m3fn or float8_e5m2");
 
    bool is_e4m3 = (out.scalar_type() == torch::kFloat8_e4m3fn);
    int const hidden_size = input.size(-1);
    
    TORCH_CHECK(hidden_size % group_size == 0, "hidden_size must be a multiple of group_size");
    TORCH_CHECK(group_size <= 16384, "group_size exceeds maximum supported size of 16384");
    TORCH_CHECK(group_size >= 16 && group_size % 16 == 0, "group_size must be >=16 and a multiple of 16 for vectorized VEC=16 memory access");
    TORCH_CHECK((group_size & (group_size - 1)) == 0, "group_size must be a power of 2 for optimized reductions");

    int64_t num_groups = input.numel() / group_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_fp8_kernel", [&] {
            
            if (group_size < 1024) {
                int threads_per_group = group_size / 16;
                constexpr int BLOCK_SIZE = 256; 
                int groups_per_block = BLOCK_SIZE / threads_per_group;
                int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;

                #define LAUNCH_SMALL_KERNEL(THREADS, IS_E4M3_BOOL)                     \
                    per_token_group_quant_fp8_small_kernel<scalar_t, THREADS, IS_E4M3_BOOL> \
                        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(                        \
                            input.data_ptr<scalar_t>(), out_ptr, scales.data_ptr<float>(), \
                            group_size, eps, use_ue8m0, num_groups)

                if (is_e4m3) {
                    if (threads_per_group == 1) { LAUNCH_SMALL_KERNEL(1, true); }
                    else if (threads_per_group == 2) { LAUNCH_SMALL_KERNEL(2, true); }
                    else if (threads_per_group == 4) { LAUNCH_SMALL_KERNEL(4, true); }
                    else if (threads_per_group == 8) { LAUNCH_SMALL_KERNEL(8, true); }
                    else if (threads_per_group == 16) { LAUNCH_SMALL_KERNEL(16, true); }
                    else if (threads_per_group == 32) { LAUNCH_SMALL_KERNEL(32, true); }
                } else {
                    if (threads_per_group == 1) { LAUNCH_SMALL_KERNEL(1, false); }
                    else if (threads_per_group == 2) { LAUNCH_SMALL_KERNEL(2, false); }
                    else if (threads_per_group == 4) { LAUNCH_SMALL_KERNEL(4, false); }
                    else if (threads_per_group == 8) { LAUNCH_SMALL_KERNEL(8, false); }
                    else if (threads_per_group == 16) { LAUNCH_SMALL_KERNEL(16, false); }
                    else if (threads_per_group == 32) { LAUNCH_SMALL_KERNEL(32, false); }
                }

            } else {
                int threads_per_group = group_size / 16;
                int block_size = threads_per_group;

                #define LAUNCH_LARGE_KERNEL(BLOCK_SIZE, IS_E4M3_BOOL)                    \
                    per_token_group_quant_fp8_kernel<scalar_t, BLOCK_SIZE, IS_E4M3_BOOL> \
                        <<<num_groups, BLOCK_SIZE, 0, stream>>>(                         \
                            input.data_ptr<scalar_t>(), out_ptr, scales.data_ptr<float>(), \
                            group_size, eps, use_ue8m0)

                if (is_e4m3) {
                    if (block_size == 64) { LAUNCH_LARGE_KERNEL(64, true); }
                    else if (block_size == 128) { LAUNCH_LARGE_KERNEL(128, true); }
                    else if (block_size == 256) { LAUNCH_LARGE_KERNEL(256, true); }
                    else if (block_size == 512) { LAUNCH_LARGE_KERNEL(512, true); }
                    else if (block_size == 1024) { LAUNCH_LARGE_KERNEL(1024, true); }
                } else {
                    if (block_size == 64) { LAUNCH_LARGE_KERNEL(64, false); }
                    else if (block_size == 128) { LAUNCH_LARGE_KERNEL(128, false); }
                    else if (block_size == 256) { LAUNCH_LARGE_KERNEL(256, false); }
                    else if (block_size == 512) { LAUNCH_LARGE_KERNEL(512, false); }
                    else if (block_size == 1024) { LAUNCH_LARGE_KERNEL(1024, false); }
                }
            }
        }
    );
}
} // namespace native
} // namespace at