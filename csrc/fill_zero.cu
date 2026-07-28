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
template <typename scalar_t, int vec=4>
__global__ void fill_zeros_kernel(scalar_t* out, int col) {
    int tid = threadIdx.x;
    int tcol = (col + vec - 1)/vec;
    int64_t idx = blockIdx.x * tcol + tid;
    idx *= vec;
    using LoadT = at::native::memory::aligned_vector<scalar_t, vec>;
    scalar_t output_vec[vec];
    if (tid < tcol) {

      #pragma unroll
      for (int i = 0; i < vec; i++) {
        output_vec[i] = scalar_t(0);
    }
    *(LoadT*)(out+idx)=*(LoadT*)output_vec;
}
}


template <typename scalar_t, int vec=4>
__global__ void fill_zeros_loop_kernel(scalar_t* out, int col) {
    int tid = threadIdx.x;
    int tcol = (col + vec - 1)/vec;
    int64_t idx = blockIdx.x * tcol + tid;
    int num_loop = col / blockDim.x / vec;
    idx *= vec;
    using LoadT = at::native::memory::aligned_vector<scalar_t, vec>;
    scalar_t output_vec[vec];
    if (tid < tcol) {
        #pragma unroll
        for (int i = 0; i < vec; i++) {
            output_vec[i] = scalar_t(0);
        }
        for(int loop = 0;loop < num_loop; loop++)
        {
            
            *(LoadT*)(out+idx+loop*tid*vec)=*(LoadT*)output_vec;
        }
      
}
}

template <typename scalar_t, int vec=4>
__global__ void fill_zeros_kernel_linear(scalar_t* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    using LoadT = at::native::memory::aligned_vector<scalar_t, vec>;
    scalar_t zeros[vec];
    #pragma unroll
    for(int i=0; i<vec; i++) zeros[i] = scalar_t(0);

    // 用 vec 大小步长循环写
    #pragma unroll
    for(int64_t i=idx*vec; i<size; i+=stride*vec) {
        *(LoadT*)(out + i) = *(LoadT*)zeros;
    }
}

template <typename scalar_t, int VEC = 4>
__global__ void fill_zeros_vectorized(scalar_t* __restrict__ out, int64_t numel) {
    using VecT = at::native::memory::aligned_vector<scalar_t, VEC>;
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (idx >= numel) return;

    scalar_t zeros[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i)
        zeros[i] = 0;

    *reinterpret_cast<VecT*>(out + idx) = *reinterpret_cast<VecT*>(zeros);
}


void fill_zeros(
  Tensor &out,
  const int bs,
  const int hidden_size
) {

    int64_t numel = out.numel();
    if (numel == 0) return;
    bool is_fp32 = (out.scalar_type() == at::kFloat);
    if(is_fp32)
    {
        const int block = 1024;
        const int grid = (numel + block * 8 - 1) / (block * 8);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            out.scalar_type(),
            "fill_zeros_vectorized",
            [&] {
                fill_zeros_vectorized<scalar_t, 8>
                    <<<grid, block, 0, at::hip::getCurrentHIPStream()>>>(
                        out.data_ptr<scalar_t>(), numel);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });

    }else{
            int grid=bs;
            int block;
            int size = bs * hidden_size;
            if(hidden_size<=2048){
                block = 256;
            }else if(hidden_size<=4096)
            {
                block = 512;
            }else if(hidden_size <= 8192){
                block = 1024;
            }else{
                block = 1024;
            }
            
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            out.scalar_type(),
            "fill_zeros_kernel",
            [&] {
                        if(hidden_size <= 8192){
                    fill_zeros_kernel<scalar_t, 8><<<grid, block, 0, at::hip::getCurrentHIPStream()>>>(
                        out.data_ptr<scalar_t>(), hidden_size
                    );
                } else{
                    fill_zeros_kernel<scalar_t, 16><<<grid, block, 0, at::hip::getCurrentHIPStream()>>>(
                        out.data_ptr<scalar_t>(), hidden_size
                    );
                }

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
    }
            }
    
}
}