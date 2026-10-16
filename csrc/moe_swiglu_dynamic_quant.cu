// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
// #include "reduce.h"

// #include "cuda_compat.h"
// #include "../dispatch_utils.h"


#include "lightop_warp_size.h"

namespace at {
namespace native {

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

    float dst = std::nearbyint(x);

    dst = fminf(fmaxf(dst, i8_min), i8_max);
    return static_cast<int8_t>(dst);
#else
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
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

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
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

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel_gernel(
    int64_t num_tokens,
    int8_t* __restrict__ out,         
    const scalar_t* __restrict__ input,  
    float* __restrict__ scales,
    const float* __restrict__ smooth,
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts 
    ) {
  int64_t token_idx = blockIdx.x;
  const int64_t input_offset = token_idx * 2 * d;
  const int64_t output_offset = token_idx * d;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float shared_mem[share_size];
  __shared__ int s_expert_index; 
  
  if (threadIdx.x == 0) {
      int expert_idx = -1;
      for (int i = 0; i < num_experts; ++i) {
          int start = experts_tokens_start[i];
          int count = experts_tokens_count[i];
          if (token_idx >= start && token_idx < start + count) {
              expert_idx = i;
              break;
          }
      }
      s_expert_index = expert_idx;
  }
  __syncthreads(); 

  for(; token_idx < num_tokens; token_idx += gridDim.x){
  int expert_index = s_expert_index;
    if (expert_index == -1) {
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            out[output_offset + idx] = 0;
        }
        if (threadIdx.x == 0) {
            scales[token_idx] = 0.0f;
        }
        return; 
    }

    const int64_t smooth_offset = expert_index * d;

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float smooth_val = VLLM_LDG(&smooth[smooth_offset + idx]); 
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y)) * smooth_val;
      row_max = fmaxf(row_max, fabsf(val));
    }
    
    row_max = BlockReduceMax_ROW<float, block_size>(row_max, shared_mem);
    
    __shared__ float s_token_scale; 
    if (threadIdx.x == 0) {
        s_token_scale = row_max;  
        scales[token_idx] = s_token_scale / 127.f; 
    }
    __syncthreads(); 

    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float smooth_val = VLLM_LDG(&smooth[smooth_offset + idx]); 
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y)) * smooth_val;
      
      int8_t q_val = float_to_int8_rn(val * inv_s);
      out[output_offset + idx] = static_cast<int8_t>(q_val);
    }
  }
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel_one_warp(
    int64_t num_tokens,
    int8_t* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    float* __restrict__ scales,
    const float* __restrict__ smooth,      
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts                 
) {
    int64_t token_idx = blockIdx.x;
    int tidx = threadIdx.x;
    int idx = threadIdx.x * VEC;
    constexpr int MAX_EXPERTS = 64;
    __shared__ int sec[MAX_EXPERTS];
    __shared__ int ses[MAX_EXPERTS]; 
    
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    using VecFloatType = at::native::memory::aligned_vector<float, VEC>;

    if (tidx < num_experts && tidx < MAX_EXPERTS) {
        sec[tidx] = experts_tokens_count[tidx];
        ses[tidx] = experts_tokens_start[tidx];   
    }
    __syncthreads(); 

    for(; token_idx < num_tokens; token_idx += gridDim.x){

      int expert_index = -1; 
      if (tidx == 0) {
        int left = 0, right = num_experts - 1, res = -1;
        while (left <= right) {
            int mid = (left + right) >> 1;
            int start = ses[mid];
            if (start <= token_idx) { 
                res = mid;       
                left = mid + 1;  
            } else { 
                right = mid - 1; 
            }
        }
        
        if (res != -1) {
            int start = ses[res];
            int count = sec[res];
            if (token_idx >= start && token_idx < start + count) {
                expert_index = res;
            } 
        } 
      }
      expert_index = __shfl(expert_index, 0, C10_WARP_SIZE);
      if (expert_index == -1) return;

      const int64_t y_index = token_idx * d + idx;
      VecInt8Type* y = (VecInt8Type*)(out + y_index);

      const int64_t x_index = token_idx * 2 * d + idx;
      VecType* x1 = (VecType*)(input + x_index);
      VecType* x2 = (VecType*)(input + x_index + d);
      VecFloatType* smooth_vec = (VecFloatType*)(smooth + expert_index * d + idx);

      scalar_t r_x1[VEC];
      scalar_t r_x2[VEC];
      float r_smooth[VEC];

      float r_y[VEC];

      if (idx < d) {
        *(VecType*)r_x1 = *x1;
        *(VecType*)r_x2 = *x2;
        *(VecFloatType*)r_smooth = *smooth_vec;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
          // r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]) * r_smooth[i];
          float silu1 = static_cast<float>(silu_kernel(r_x1[i]));
          float silu2 = static_cast<float>(r_x2[i]);
          r_y[i] = silu1 * silu2 * r_smooth[i];

        }
      }

      float row_max = 0.f;
      if (idx < d) {
        #pragma unroll
        for(int ii = 0; ii < VEC; ii++){
          row_max = fmaxf(row_max, fabsf(r_y[ii]));
        }
      }

      row_max = WarpReduceMax_ROW<float>(row_max);
      float quant_scale = 1.0f;
      if(tidx == 0){
        quant_scale = 127.0f / row_max;  
        scales[token_idx] = row_max / 127.f;  
      }

      quant_scale = __shfl(quant_scale, 0, C10_WARP_SIZE);
      int8_t out_vec[VEC];
      
      if (idx < d) {
        #pragma unroll
        for(int ii = 0; ii < VEC; ii++){
            out_vec[ii] = float_to_int8_rn(r_y[ii] * quant_scale);
        }
        *y = *(VecInt8Type*)out_vec;
      }
  }
}


template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel(
    int64_t num_tokens,
    int8_t* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    float* __restrict__ scales,
    const float* __restrict__ smooth,      
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts                 
) {
    int64_t token_idx = blockIdx.x;
    int tidx = threadIdx.x;
    int idx = threadIdx.x * VEC;
    constexpr int MAX_EXPERTS = 64;
    __shared__ int sec[MAX_EXPERTS];
    __shared__ int ses[MAX_EXPERTS]; 
    
    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];

    __shared__ int s_expert_index;
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    using VecFloatType = at::native::memory::aligned_vector<float, VEC>;

    if (tidx < num_experts && tidx < MAX_EXPERTS) {
        sec[tidx] = experts_tokens_count[tidx];
        ses[tidx] = experts_tokens_start[tidx];   
    }
    __syncthreads(); 

    for(; token_idx < num_tokens; token_idx += gridDim.x){
      int local_expert_index = -1;
      if (tidx == 0) {
          int left = 0, right = num_experts - 1, res = -1;
          while (left <= right) {
              int mid = (left + right) >> 1;
              int start = ses[mid];
              if (start <= token_idx) { 
                  res = mid;       
                  left = mid + 1;  
              } else { 
                  right = mid - 1; 
              }
          }
          
          if (res != -1) {
              int start = ses[res];
              int count = sec[res];
              if (token_idx >= start && token_idx < start + count) {
                  local_expert_index = res;
              } 
          }
          s_expert_index = local_expert_index;
      }
      __syncthreads();
      int expert_index = s_expert_index;
      if (expert_index == -1) return;

      const int64_t y_index = token_idx * d + idx;
      VecInt8Type* y = (VecInt8Type*)(out + y_index);

      const int64_t x_index = token_idx * 2 * d + idx;
      VecType* x1 = (VecType*)(input + x_index);
      VecType* x2 = (VecType*)(input + x_index + d);
      VecFloatType* smooth_vec = (VecFloatType*)(smooth + expert_index * d + idx);

      scalar_t r_x1[VEC];
      scalar_t r_x2[VEC];
      float r_smooth[VEC];

      float r_y[VEC];

      if (idx < d) {
        *(VecType*)r_x1 = *x1;
        *(VecType*)r_x2 = *x2;
        *(VecFloatType*)r_smooth = *smooth_vec;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
          // r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]) * r_smooth[i];
          float silu1 = static_cast<float>(silu_kernel(r_x1[i]));
          float silu2 = static_cast<float>(r_x2[i]);
          r_y[i] = silu1 * silu2 * r_smooth[i];
        }
      }

      float row_max = 0.f;
      if (idx < d) {
        #pragma unroll
        for(int ii = 0; ii < VEC; ii++){
          row_max = fmaxf(row_max, fabsf(r_y[ii]));
        }
      }


      row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
      __shared__ float s_token_scale; 
      
      if(tidx == 0){
        s_token_scale = row_max;  
        scales[token_idx] = s_token_scale / 127.f;  
      }
      __syncthreads();

      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
      int8_t out_vec[VEC];
      
      if (idx < d) {
        #pragma unroll
        for(int ii = 0; ii < VEC; ii++){
            out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
        }
        *y = *(VecInt8Type*)out_vec;
      }
  }
}

void moe_swiglu_dynamic_quant(torch::Tensor& scatter_tokens, torch::Tensor& smooth, torch::Tensor& experts_tokens_count, torch::Tensor& experts_tokens_start, torch::Tensor& output, torch::Tensor& scales, float beta)
{
  int d = scatter_tokens.size(-1) / 2;                                                  
  int64_t num_tokens = scatter_tokens.numel() / scatter_tokens.size(-1); 
  int num_experts = experts_tokens_count.size(0);                        
  int grid_opt = num_tokens;  
  if(num_tokens == 9216 || num_tokens == 10240 || num_tokens == 11264 || num_tokens == 12288 ||
     num_tokens == 13312 || num_tokens == 14336){
    grid_opt = 8192;
  } else if(num_tokens == 3072 || num_tokens == 4096 || 
    num_tokens == 5120 || num_tokens == 6144 || num_tokens == 7168){
    grid_opt = 2048;
  } else if(num_tokens <= 2048 && num_tokens >= 1024){
    grid_opt = 1024;
  } else{
    grid_opt = num_tokens;
  }
                                                                                                     
  dim3 grid(grid_opt);                                                                                                 
  if (num_tokens == 0) {                                                       
    return;                                                                    
  }                                                                            
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scatter_tokens));            
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                
  VLLM_DISPATCH_FLOATING_TYPES(                                                
    scatter_tokens.scalar_type(), "moe_swiglu_dynamic_quant_kernel", [&] {                         
        if (d <= 512) {                                                      
        moe_swiglu_dynamic_quant_kernel<scalar_t, 2, true, 256> 
            <<<grid, 256, 0, stream>>>(num_tokens ,                  
                                        output.data_ptr<int8_t>(),           
                                        scatter_tokens.data_ptr<scalar_t>(), 
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);     
        } else if (d <= 1024) {  
              moe_swiglu_dynamic_quant_kernel_one_warp<scalar_t, 16, true, 64> 
              <<<grid, 64, 0, stream>>>(num_tokens ,
                                          output.data_ptr<int8_t>(),           
                                          scatter_tokens.data_ptr<scalar_t>(), 
                                          scales.data_ptr<float>(),
                                          smooth.data_ptr<float>(),
                                          experts_tokens_count.data_ptr<int>(),
                                          experts_tokens_start.data_ptr<int>(),
                                          d,num_experts);                                           
        } else if (d <= 2048) {                                              
        moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 128> 
            <<<grid, 128, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),           
                                        scatter_tokens.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(), 
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);     
        } else if (d <= 4096) {                                              
        moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 256> 
            <<<grid, 256, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),           
                                        scatter_tokens.data_ptr<scalar_t>(), 
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),   
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);     
        } else if (d <= 8192) {                                                             
        moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 512> 
            <<<grid, 512, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),          
                                        scatter_tokens.data_ptr<scalar_t>(), 
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),    
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);
                                            
        } else if (d <= 16384) {                                                             
        moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 1024> 
            <<<grid, 1024, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),          
                                        scatter_tokens.data_ptr<scalar_t>(), 
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),    
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);   
        } else if (d <= 32768) {                                                             
        moe_swiglu_dynamic_quant_kernel<scalar_t, 32, true, 1024> 
            <<<grid, 1024, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),          
                                        scatter_tokens.data_ptr<scalar_t>(),  
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),   
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts);   
        } else{
        moe_swiglu_dynamic_quant_kernel_gernel<scalar_t, true, 1024>
            <<<grid, 1024, 0, stream>>>(num_tokens ,
                                        output.data_ptr<int8_t>(),          
                                        scatter_tokens.data_ptr<scalar_t>(),   
                                        scales.data_ptr<float>(), 
                                        smooth.data_ptr<float>(), 
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d,num_experts); 
        } 
       
      
      });
}

}
}

