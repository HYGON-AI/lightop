// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// #include "type_convert.cuh"
// #include "dispatch_utils.h"

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/Dispatch.h>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif


#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at{
namespace native{

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)


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

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min = static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max = static_cast<float>(std::numeric_limits<int8_t>::max());
  float dst = std::nearbyint(x);
  dst = fminf(fmaxf(dst, i8_min), i8_max);
  return static_cast<int8_t>(dst);
#else
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
#if defined(USE_ROCM)
    val = fmaxf(val, __shfl_down(val, offset));
#else
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
#endif
  }
  return val;
}

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceSum_NEW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceSum_NEW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceSum_NEW<T>(val);
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
      val = WarpReduceSum_NEW<T,share_size>(shared[lid]);
    }
    return val;
  }
}
// =========================================



template <typename scalar_t,typename T_ACC, int Vec=8, int num_warp_per_block=4, int rot_dim=64, int kv_c_dim=512>
__global__ void rms_rope_int8quant_kernel(
    scalar_t* __restrict__ input_q,  // [bs, num_head, q_c_dim + rot_dim]
    scalar_t* __restrict__ input_k,  // [bs, kv_c_dim + rot_dim]
    scalar_t* __restrict__ input_kv_c,  // [bs, kv_c_dim]
    scalar_t* __restrict__ output_kv_c_normed,  // [bs, kv_c_dim] 
    int8_t* __restrict__ output_kv_c_normed_int8,  // [bs, kv_c_dim]
    float* __restrict__ output_kv_c_normed_scale,  // [bs]
    scalar_t* __restrict__ weight,  // [kv_c_dim]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int64_t* __restrict__ positions,  // [bs]
    const int64_t* __restrict__ slot_mapping,  // [bs]
    void* __restrict__ kv_cache,  // [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]
    const int kv_cache_blocks_size,
    const int q_c_dim,
    const int num_head,
    const int q_stride_head,
    const int k_stride_head,
    const int q_grid_x,
    const int k_grid_x,
    const int k_rms_grid_x,
    const int bs,
    const float epsilon,
    const int kv_dtype,
    const int max_slot_id,
    const int k_bs_stride) 
  {
    if(blockIdx.x < q_grid_x){
      const int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;
      const int num_head_per_block = num_head_per_warp * num_warp_per_block;
      const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      const int embed_dim = rot_dim / 2;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int thread_per_head = rot_dim / Vec;
      const int token_idx = blockIdx.x / block_per_token;
      if (token_idx >= bs) return;
      
      const int head_block_idx = blockIdx.x % block_per_token;     // 0-3
      const int head_start = head_block_idx * num_head_per_block;  // 0, 32, 64, 96
      const int lane_of_head = laneid / thread_per_head;  // 0, 1, 2, 3
      const int head_in_block = warpid * num_head_per_warp + lane_of_head;  // 0, 8, 16, 24
      const int64_t pos = positions[token_idx];
      const int head_offset_in_token = head_start + head_in_block;
      
      const int element_in_head = (threadIdx.x % thread_per_head)*Vec; // 0 8 16 24 32 40 48 56 
      const int q_pe_offset = token_idx * num_head * (q_stride_head) + head_offset_in_token * (q_stride_head) + element_in_head;
      scalar_t* q_pe_ptr = input_q + q_pe_offset;
      
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const int cos_index_in_cache = element_in_head /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      if(head_offset_in_token >= num_head) return;
      *(LoadT*)(input_vec) = *(LoadT*)(q_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);
      for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
        scalar_t x = input_vec[ii];
        scalar_t y = input_vec[ii+1];
        scalar_t cos_val = cos_vec[rot_dim_idx];
        scalar_t sin_val = sin_vec[rot_dim_idx];
        input_vec[ii] = x * cos_val - y * sin_val;
        input_vec[ii+1] = y * cos_val + x * sin_val;
        rot_dim_idx += 1;
      }
      *(LoadT*)q_pe_ptr = *(LoadT*)input_vec;
    }
    else if(blockIdx.x < q_grid_x+k_grid_x){
      constexpr int thread_per_head = rot_dim / Vec;
      constexpr int element_per_warp = C10_WARP_SIZE * Vec;
      constexpr int num_token_per_warp = (element_per_warp) / rot_dim;
      constexpr int num_token_per_block = num_token_per_warp * num_warp_per_block;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      using OutputT = at::native::memory::aligned_vector<uint8_t, Vec>;
      constexpr int embed_dim = rot_dim / 2;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int token_idx =  (blockIdx.x - q_grid_x)*num_token_per_block + warpid*num_token_per_warp + laneid/thread_per_head;
      if(token_idx >= bs) return;  
      
      const int thread_offset_in_token = (laneid % thread_per_head)*Vec;
      const int k_pe_offset = token_idx * k_bs_stride + thread_offset_in_token;
      scalar_t* k_pe_ptr = input_k + k_pe_offset;
      const int64_t pos = positions[token_idx];
      const int cos_index_in_cache = thread_offset_in_token /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      
      const int64_t slot = slot_mapping[token_idx];
      if(slot >= max_slot_id || slot < 0) return;

      const int64_t kv_cache_offset = slot * 576 + thread_offset_in_token;
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      uint8_t output_vec[Vec];
      float float_vec[Vec];
      *(LoadT*)(input_vec) = *(LoadT*)(k_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);

      if (kv_dtype != 2) 
      {
        for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
          scalar_t x = input_vec[ii];
          scalar_t y = input_vec[ii+1];
          scalar_t cos_val = cos_vec[rot_dim_idx];
          scalar_t sin_val = sin_vec[rot_dim_idx];
          float_vec[ii] = x * cos_val - y * sin_val;
          float_vec[ii+1] = y * cos_val + x * sin_val;
          input_vec[ii] = static_cast<scalar_t>(float_vec[ii]);
          input_vec[ii+1] = static_cast<scalar_t>(float_vec[ii+1]);
          rot_dim_idx += 1;
        }
        if(kv_dtype == 0)
        {
          ((uint32_t*)output_vec)[0] = float_to_fp8e5m2_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_fp8e5m2_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        else
        {
          ((uint32_t*)output_vec)[0] = float_to_fp8e4m3_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_fp8e4m3_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        uint8_t* output_ptr = (uint8_t*)kv_cache + kv_cache_offset + 512;
        *(OutputT*)output_ptr = *(OutputT*)output_vec;
        *(LoadT*)(k_pe_ptr) = *(LoadT*)input_vec;
      }
      else{
        for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
          scalar_t x = input_vec[ii];
          scalar_t y = input_vec[ii+1];
          scalar_t cos_val = cos_vec[rot_dim_idx];
          scalar_t sin_val = sin_vec[rot_dim_idx];
          input_vec[ii] = static_cast<scalar_t>(x * cos_val - y * sin_val);
          input_vec[ii+1] = static_cast<scalar_t>(y * cos_val + x * sin_val);
          rot_dim_idx += 1;
        }
        scalar_t* output_ptr = (scalar_t*)kv_cache + kv_cache_offset + 512;
        *(LoadT*)output_ptr = *(LoadT*)input_vec;
        *(LoadT*)(k_pe_ptr) = *(LoadT*)input_vec;
      }
    }
    else
    {
      constexpr int element_per_warp = C10_WARP_SIZE * Vec; 
      constexpr int token_per_warp = element_per_warp / (kv_c_dim); 
      constexpr int token_per_block = token_per_warp * num_warp_per_block; 
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using OutputT = at::native::memory::aligned_vector<uint8_t, Vec>;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int token_id = (blockIdx.x - q_grid_x - k_grid_x) * token_per_block + (warpid * token_per_warp);
      if(token_id >= bs) return;
      
      const int thread_offset_in_token = laneid*Vec;
      if(thread_offset_in_token >= kv_c_dim) return;
      const int kv_c_offset = token_id * k_bs_stride + thread_offset_in_token;
      scalar_t* kv_c_ptr = input_kv_c + kv_c_offset;
      
      const int64_t slot = slot_mapping[token_id];
      if(slot >= max_slot_id || slot < 0) return;
      
      const int64_t kv_cache_offset = slot * 576 + thread_offset_in_token;
      const int64_t kv_c_normed_offset = token_id * kv_c_dim + thread_offset_in_token;
      scalar_t* kv_c_normed_ptr = output_kv_c_normed + kv_c_normed_offset;
      scalar_t* weight_ptr = weight + thread_offset_in_token;
      
      scalar_t input_vec[Vec];
      scalar_t weight_vec[Vec];
      uint8_t output_vec[Vec];
      float float_vec[Vec];
      
      *(LoadT*)(input_vec) = *(LoadT*)(kv_c_ptr);
      *(LoadT*)(weight_vec) = *(LoadT*)(weight_ptr);
      
      T_ACC val = 0;
      T_ACC s_rstd;
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        val += static_cast<T_ACC>(input_vec[ii]) * static_cast<T_ACC>(input_vec[ii]);
      }
      val = BlockReduceSum_NEW<T_ACC,C10_WARP_SIZE>(val,nullptr);
      if(laneid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(val / kv_c_dim + epsilon);
      }
#if defined(USE_ROCM)
      s_rstd = __shfl(s_rstd, 0);
#else
      s_rstd = __shfl_sync(0xFFFFFFFF, s_rstd, 0);
#endif
      if (kv_dtype != 2) 
      {
        #pragma unroll
        for(int ii = 0; ii < Vec; ii+=1){
          float_vec[ii] = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
          input_vec[ii] = static_cast<scalar_t>(float_vec[ii]);
        }
        if(kv_dtype == 0) {
          ((uint32_t*)output_vec)[0] = float_to_fp8e5m2_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_fp8e5m2_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        } else {
          ((uint32_t*)output_vec)[0] = float_to_fp8e4m3_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_fp8e4m3_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        uint8_t* output_ptr = (uint8_t*)kv_cache + kv_cache_offset;
        *(OutputT*)output_ptr = *(OutputT*)output_vec;
        
        if (output_kv_c_normed != nullptr) {
            *(LoadT*)kv_c_normed_ptr = *(LoadT*)input_vec;
        }
      }
      else{
        for(int ii = 0; ii < Vec; ii+=1){
          input_vec[ii] = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
        }
        scalar_t* output_ptr = (scalar_t*)kv_cache + kv_cache_offset;
        *(LoadT*)(output_ptr) = *(LoadT*)(input_vec);
        
        if (output_kv_c_normed != nullptr) {
            *(LoadT*)kv_c_normed_ptr = *(LoadT*)input_vec;
        }
      }
      float local_max = 0.f;
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        local_max = fmaxf(local_max, fabs(static_cast<float>(input_vec[ii])));
      }
      
      float warp_max = WarpReduceMax_ROW<float, C10_WARP_SIZE>(local_max);
#if defined(USE_ROCM)
      warp_max = __shfl(warp_max, 0);
#else
      warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);
#endif

      float inv_s = (warp_max == 0.f) ? 0.f : 127.f / warp_max;

      uint8_t out_vec_int8[Vec];
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        out_vec_int8[ii] = static_cast<uint8_t>(float_to_int8_rn(static_cast<float>(input_vec[ii]) * inv_s));
      }

      if (output_kv_c_normed_int8 != nullptr) {
        uint8_t* kv_c_normed_int8_ptr = reinterpret_cast<uint8_t*>(output_kv_c_normed_int8) + kv_c_normed_offset;
        *(OutputT*)kv_c_normed_int8_ptr = *(OutputT*)out_vec_int8;
      }

      if (laneid == 0 && output_kv_c_normed_scale != nullptr) {
        output_kv_c_normed_scale[token_id] = warp_max / 127.f;
      }
    }
  }
  void fused_rms_norm_rope_int8quant_contiguous(
    torch::Tensor& positions, 
    torch::Tensor& q_pe, 
    torch::Tensor& k_pe, 
    torch::Tensor& kv_c, 
    torch::Tensor& kv_c_normed,      
    torch::Tensor& kv_c_normed_int8, 
    torch::Tensor& kv_c_normed_scale, 
    torch::Tensor& weight, 
    torch::Tensor& cos_sin_cache, 
    torch::Tensor& slot_mapping, 
    torch::Tensor& kv_cache, 
    std::string& kv_cache_dtype_str, 
    float kv_cache_scale, 
    bool is_neox_style, 
    float epsilon) {

    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D tensor [bs, num_head, rot_dim]");
    const int bs = q_pe.size(0);
    const int num_head = q_pe.size(1);
    constexpr int q_c_dim = 128;
    constexpr int rot_dim = 64;
    const int q_stride_head = q_pe.stride(1);
    TORCH_CHECK(q_stride_head == q_c_dim+rot_dim, "q_stride_head must be q_c_dim + rot_dim");
    TORCH_CHECK(q_pe.size(2) ==  rot_dim, "q_pe.size(2) must be rot_dim");
    TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D tensor [max_position, rot_dim]");
    TORCH_CHECK(cos_sin_cache.size(1) == rot_dim, "cos_sin_cache.size(1) must equal rot_dim");
    
    // 计算ROPE(Q)
    constexpr int num_warp_per_block = 4;
    constexpr int Vec = 8;
    constexpr int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;  
    constexpr int num_head_per_block = num_head_per_warp * num_warp_per_block;  
    const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block; 
    const int q_grid_x = block_per_token * bs;  
    const int block_x = num_warp_per_block * C10_WARP_SIZE;  
    
    // 计算ROPE(K)
    constexpr int kv_c_dim = 512;
    constexpr int k_stride_head = kv_c_dim + rot_dim;
    TORCH_CHECK(k_pe.dim() == 2, "k_pe must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(k_pe.size(1) ==  rot_dim, "k_pe.size(1) must be rot_dim");
    TORCH_CHECK(k_pe.size(0) == bs, "k_pe.size(0) must be bs");
    const int k_bs_stride = k_pe.stride(0);
    
    TORCH_CHECK(kv_c_normed.dim() == 2, "kv_c_normed must be 2D tensor [bs, kv_c_dim]");
    TORCH_CHECK(kv_c_normed.size(1) == kv_c_dim, "kv_c_normed.size(1) must be kv_c_dim");
    
    // 检查新增的量化 Tensor
    TORCH_CHECK(kv_c_normed_int8.dim() == 2, "kv_c_normed_int8 must be 2D tensor [bs, kv_c_dim]");
    TORCH_CHECK(kv_c_normed_int8.size(1) == kv_c_dim, "kv_c_normed_int8.size(1) must be kv_c_dim");
    TORCH_CHECK(kv_c_normed_scale.dim() == 1 || kv_c_normed_scale.dim() == 2, "kv_c_normed_scale must be [bs] or [bs, 1]");
    
    constexpr int element_per_warp = C10_WARP_SIZE * Vec;
    constexpr int k_token_per_warp = element_per_warp / (rot_dim);  
    constexpr int k_token_per_block = k_token_per_warp * num_warp_per_block; 
    const int k_grid_x = (bs + k_token_per_block - 1) / k_token_per_block; 
    
    // 计算RMS(K)
    TORCH_CHECK(kv_c.dim() == 2, "kv_c must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(kv_c.size(1) == kv_c_dim, "kv_c.size(1) must be kv_c_dim");
    constexpr int k_rms_token_per_warp = element_per_warp / (kv_c_dim);
    constexpr int k_rms_token_per_block = k_rms_token_per_warp * num_warp_per_block; 
    const int k_rms_grid_x = (bs + k_rms_token_per_block - 1) / k_rms_token_per_block; 
    
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D tensor [bs]");
    TORCH_CHECK(kv_cache.dim() == 3, "kv_cache must be 3D tensor [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]");
    const int num_kv_cache_blocks = kv_cache.size(0);
    const int kv_cache_blocks_size = kv_cache.size(1);
    const int max_slot_id = num_kv_cache_blocks * kv_cache_blocks_size;

    int kv_dtype = -1;
    if(kv_cache_dtype_str == "fp8_e5m2"){ kv_dtype = 0; }
    else if(kv_cache_dtype_str == "fp8_e4m3"){ kv_dtype = 1; }
    else { kv_dtype = 2; }

    const int grid_x = q_grid_x + k_grid_x + k_rms_grid_x;
    dim3 grid(grid_x);
    dim3 block(block_x);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_pe));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_pe.scalar_type(),
      "fused_rms_norm_rope_int8quant_kernel",
      [&]{
        using T_ACC = at::acc_type<scalar_t, true>;
        scalar_t* q_pe_ptr  = q_pe.data_ptr<scalar_t>();
        scalar_t* k_pe_ptr  = k_pe.data_ptr<scalar_t>();
        scalar_t* kv_c_ptr = kv_c.data_ptr<scalar_t>();
        scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
        
        scalar_t* kv_c_normed_ptr = kv_c_normed.data_ptr<scalar_t>();
        int8_t* kv_c_normed_int8_ptr = kv_c_normed_int8.data_ptr<int8_t>();
        float* kv_c_normed_scale_ptr = kv_c_normed_scale.data_ptr<float>();

        void* kv_cache_ptr;
        if (kv_cache.scalar_type() == at::ScalarType::Byte) {
            kv_cache_ptr = kv_cache.data_ptr<uint8_t>();
        } else {
            if (kv_cache.scalar_type() == at::ScalarType::Half) {
                kv_cache_ptr = kv_cache.data_ptr<at::Half>();
            } else if (kv_cache.scalar_type() == at::ScalarType::BFloat16) {
                kv_cache_ptr = kv_cache.data_ptr<at::BFloat16>();
            }
        }
        int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();
        const scalar_t* cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        const int64_t* positions_ptr = positions.data_ptr<int64_t>();

        rms_rope_int8quant_kernel<scalar_t,T_ACC,Vec,num_warp_per_block,rot_dim,kv_c_dim><<<grid, block, 0, stream>>>(
            q_pe_ptr, k_pe_ptr, kv_c_ptr, 
            kv_c_normed_ptr, kv_c_normed_int8_ptr, kv_c_normed_scale_ptr, 
            weight_ptr, cos_sin_cache_ptr,
            positions_ptr, slot_mapping_ptr, kv_cache_ptr,
            kv_cache_blocks_size, q_c_dim, num_head, q_stride_head,
            k_stride_head, q_grid_x, k_grid_x, k_rms_grid_x, bs, epsilon, kv_dtype, max_slot_id, k_bs_stride);
      });
}

}
}