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

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
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

template <typename scalar_t,typename T_ACC,int Vec=4,int block_size=512>
__global__ void fused_rms_kernel_contiguous(scalar_t* input,scalar_t* output,scalar_t* gamma,int cols,T_ACC eps,int stride)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val=0;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  scalar_t intput_vec[Vec];
  T_ACC trstd;
  int64_t row_start_offset = i * stride;
  int64_t col_offset = j * Vec;
  int64_t idx = row_start_offset + col_offset;
  int64_t out_idx = i * cols + col_offset;
  
  if (j < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+idx);
    #pragma unroll
    for (int ii = 0; ii < Vec; ii++) {
      val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
    }
  }
  val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
  if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
  __syncthreads();
  trstd=s_rstd;
  if (j < tcol) {
    #pragma unroll
    for(int ii=0;ii<Vec;ii++){
      int jj=j*Vec+ii;
      intput_vec[ii] = static_cast<T_ACC>(intput_vec[ii]) *trstd* static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(output+out_idx)=*(LoadT*)intput_vec;
  }
}

void fused_rms_norm_contiguous(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  int hidden_size = input.size(-1);
  int stride = input.stride(0);
  int num_tokens = input.numel() / hidden_size;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =inp_ptr % 16 == 0  && wt_ptr % 16 == 0;
  if(hidden_size%16==0&&hidden_size<=16384&&ptrs_are_aligned){
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_rms_norm_contiguous_kernel",
    [&] {
      using T_ACC = at::acc_type<scalar_t, true>;
      T_ACC eps = epsilon;
      // scalar_t* self_data  = input.expect_contiguous()->data_ptr<scalar_t>();
      scalar_t* self_data  = input.data_ptr<scalar_t>();
      scalar_t* out_data   =  out.expect_contiguous()->data_ptr<scalar_t>();
      scalar_t* weight_data= weight.expect_contiguous()->data_ptr<scalar_t>();
      if (hidden_size<=1024){
          fused_rms_kernel_contiguous<scalar_t,T_ACC,8,128><<<num_tokens,  128, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
      }
      else if(hidden_size<=2048){
          fused_rms_kernel_contiguous<scalar_t,T_ACC,8,256><<<num_tokens,  256, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
      }
      else if(hidden_size<=4096){
          if(num_tokens>1200){
            fused_rms_kernel_contiguous<scalar_t,T_ACC,8,512><<<num_tokens,  512, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
          }
          else{
            fused_rms_kernel_contiguous<scalar_t,T_ACC,4,1024><<<num_tokens,  1024, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
          }
      }
      else if(hidden_size<=8192){
           fused_rms_kernel_contiguous<scalar_t,T_ACC,8,1024><<<num_tokens,  1024, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
      }
      else{
          fused_rms_kernel_contiguous<scalar_t,T_ACC,16,1024><<<num_tokens,  1024, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,stride);
      } 
    });
  }
  else{
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
  
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    });
  }
}

}
}