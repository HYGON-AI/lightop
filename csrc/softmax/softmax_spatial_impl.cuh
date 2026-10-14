// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>

#include <ATen/CUDAFunctions.h>
#include "cuda_meta.h"

using namespace std;

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif

// #include "softmax_math.cuh"


namespace at {
namespace native {

namespace softmax_spatial_impl {
const int max_threads = 1024;

inline dim3 SpatialSoftMax_getBlockSize(int64_t dim_size, int64_t inner_size) {
    int inner_threads = inner_size;
    inner_threads = std::min(inner_threads, static_cast<int>(max_threads));
    int dim_threads = 1;
    if (inner_threads <= 64 && dim_size >= 64) {
        while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
            dim_threads *= 2;
        dim_threads /= 2;
    }
    return dim3(dim_threads, inner_threads);
}

// Note that it's not a complete block-wide reduction.
// Only threads that share threadIdx.y reduce values.
template<typename acc_t, typename ReduceOp>
__forceinline__ __device__
acc_t spatialBlockReduceX(acc_t *shared, acc_t val,
                          ReduceOp &r) {
  shared += threadIdx.y * blockDim.x;

  __syncthreads();

  shared[threadIdx.x] = val;

  // NOTE: loop starts with __syncthreads()
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset)
      shared[threadIdx.x] = r(shared[threadIdx.x], shared[threadIdx.x + offset]);
    offset /= 2;
  }

  __syncthreads();

  return shared[0];
}

// Note that it's not a complete block_dims-wide reduction.
// Only threads that share threadIdx.y reduce values.
template<typename acc_t, typename ReduceOp>
__forceinline__ __device__
void spatialBlockReduceX_onlinesoftmax(acc_t *shared_max, acc_t &max_val, 
                                       acc_t *shared_sum, acc_t &sum_val, 
                                       ReduceOp &r) {
    shared_max += threadIdx.y * blockDim.x;
    shared_sum += threadIdx.y * blockDim.x;

    __syncthreads();

    shared_max[threadIdx.x] = max_val;
    shared_sum[threadIdx.x] = sum_val;

    // NOTE: loop starts with __syncthreads()
    int offset = blockDim.x / 2;
    while (offset > 0) {
        __syncthreads();
        if (threadIdx.x < offset) {
            r(shared_max[threadIdx.x + offset], shared_sum[threadIdx.x + offset], 
              shared_max[threadIdx.x], shared_sum[threadIdx.x]);
        }
        offset /= 2;
    }
    __syncthreads();

    max_val = shared_max[0];
    sum_val = shared_sum[0];
}

template <typename input_t, 
    typename acc_t, 
    typename output_t, 
    bool is_log_softmax,
    bool is_multi_warp>
__global__ void softmax_spatial_fwd(
    output_t *output, const input_t *input,
    int outer_size, int dim_size, int inner_size)
{
    const int outer_stride = inner_size * dim_size;        // 每次计算的总Stride
    const int dim_stride = inner_size;                     // 不计算的Stride

    const int outer_index = blockIdx.x;
    const int outer_offset = outer_index * outer_stride;

    const auto online_softmax_updater = softmax_op::OnlineSoftmaxUpdate<input_t, acc_t>();
    const auto online_softmax_combine_updater = softmax_op::OnlineSoftmaxCombineUpdate<input_t, acc_t>();

    if constexpr(is_multi_warp) {
        extern __shared__ unsigned char smem[];
        const int smem_single_size = blockDim.x * blockDim.y;
        auto sdata_max = reinterpret_cast<acc_t*>(smem);
        auto sdata_sum = sdata_max + smem_single_size;

        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const int data_offset = outer_offset + inner_index;
            ////////////////////////////////////////////////////////////
            // These two blocks are really equivalent, but specializing on
            // blockDim.x == 1 makes the kernel faster when it's unused.
            // I didn't want to thread an extra template parameter, and nvcc
            // seems to be smart enough to hoist the if outside of the loops.
            ////////////////////////////////////////////////////////////
            acc_t max_value = at::numeric_limits<acc_t>::lowest();
            acc_t sum_value = 0;
            for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                online_softmax_updater(input[data_offset + d * dim_stride], max_value, sum_value);
            }
            spatialBlockReduceX_onlinesoftmax<acc_t>(sdata_max, max_value, sdata_sum, sum_value, 
                                                     online_softmax_combine_updater);

            if constexpr(is_log_softmax) {
                softmax_op::LogSoftMaxForwardEpilogue<input_t, acc_t, output_t> epilogue(max_value, sum_value);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
                }
            } else {
                softmax_op::SoftMaxForwardEpilogue<input_t, acc_t, output_t> epilogue(max_value, sum_value);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
                }
            }
        }
    } else {
        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const int data_offset = outer_offset + inner_index;
            ////////////////////////////////////////////////////////////
            // These two blocks are really equivalent, but specializing on
            // blockDim.x == 1 makes the kernel faster when it's unused.
            // I didn't want to thread an extra template parameter, and nvcc
            // seems to be smart enough to hoist the if outside of the loops.
            ////////////////////////////////////////////////////////////

            acc_t max_value = at::numeric_limits<acc_t>::lowest();
            acc_t sum_value = 0;
            for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                online_softmax_updater(input[data_offset + d * dim_stride], max_value, sum_value);
            }

            if constexpr(is_log_softmax) {
                softmax_op::LogSoftMaxForwardEpilogue<input_t, acc_t, output_t> epilogue(max_value, sum_value);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
                }
            } else {
                softmax_op::SoftMaxForwardEpilogue<input_t, acc_t, output_t> epilogue(max_value, sum_value);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
                }
            }
        }
    }
}


template <typename input_t, 
    typename acc_t, 
    typename output_t, 
    bool is_log_softmax,
    bool is_multi_warp>
__global__ void softmax_spatial_bwd(input_t *gradInput, const output_t *gradOutput, const output_t *output,
                                    int outer_size, int dim_size, int inner_size)
{
    const int outer_stride = inner_size * dim_size;
    const int dim_stride = inner_size;

    const int outer_index = blockIdx.x;
    const int outer_offset = outer_index * outer_stride;

    if constexpr(is_multi_warp) {
        extern __shared__ unsigned char smem[];
        auto sdata = reinterpret_cast<acc_t*>(smem);

        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const int data_offset = outer_offset + inner_index;
            // See the comment in forward kernel
            acc_t sum = 0;
            for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                sum += gradOutput[data_offset + d * dim_stride];
            }
            auto reduce_r = softmax_math::Add<acc_t>();
            sum = spatialBlockReduceX<acc_t>(sdata, sum, reduce_r);

            if constexpr(is_log_softmax) {
                softmax_op::LogSoftMaxBackwardEpilogue<input_t, acc_t, output_t> epilogue(sum);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    gradInput[data_offset + d * dim_stride] =
                        epilogue(gradOutput[data_offset + d * dim_stride], output[data_offset + d * dim_stride]);
                }
            } else {
                softmax_op::SoftMaxBackwardEpilogue<input_t, acc_t, output_t> epilogue(sum);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    gradInput[data_offset + d * dim_stride] =
                        epilogue(gradOutput[data_offset + d * dim_stride], output[data_offset + d * dim_stride]);
                }
            }
        }
    } else {
        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const int data_offset = outer_offset + inner_index;
            // See the comment in forward kernel
            acc_t sum = 0;
            for (int d = 0; d < dim_size; d++) {
                sum += gradOutput[data_offset + d * dim_stride];
            }

            if constexpr(is_log_softmax) {
                softmax_op::LogSoftMaxBackwardEpilogue<input_t, acc_t, output_t> epilogue(sum);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    gradInput[data_offset + d * dim_stride] =
                        epilogue(gradOutput[data_offset + d * dim_stride], output[data_offset + d * dim_stride]);
                }
            } else {
                softmax_op::SoftMaxBackwardEpilogue<input_t, acc_t, output_t> epilogue(sum);
                for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    gradInput[data_offset + d * dim_stride] =
                        epilogue(gradOutput[data_offset + d * dim_stride], output[data_offset + d * dim_stride]);
                }
            }
        }
    }
}

} // namespace softmax_spatial_impl

namespace softmax_spatial{

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_spatial_forward(output_t *dst, const input_t *src, 
                             const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size, 
                             cudaStream_t stream) {
    dim3 block_dims = softmax_spatial_impl::SpatialSoftMax_getBlockSize(dim_size, inner_size);
    int inner_blocks = (inner_size + block_dims.y - 1) / block_dims.y;

    dim3 grid_dims(outer_size, inner_blocks);
    // printf("block_dims.x=%d, block_dims.y=%d, outer_size=%d, inner_blocks=%d, \n", 
    // block_dims.x, block_dims.y, outer_size, inner_blocks);
    const bool is_multi_warp = block_dims.x > 1;

    if (is_multi_warp) {
        int block_threads = block_dims.x * block_dims.y;
        int smem_size = block_threads * sizeof(acc_t) * 2;
        softmax_spatial_impl::softmax_spatial_fwd<input_t, acc_t, output_t, is_log_softmax, true>
            <<<grid_dims, block_dims, smem_size, stream>>>(dst, src, outer_size, dim_size, inner_size);
    } else {
        softmax_spatial_impl::softmax_spatial_fwd<input_t, acc_t, output_t, is_log_softmax, false>
            <<<grid_dims, block_dims, 0, stream>>>(dst, src, outer_size, dim_size, inner_size);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_spatial_backward(input_t *gradInput, const output_t *gradOutput, const output_t *output,
                              const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size, 
                              cudaStream_t stream) {
    dim3 block_dims = softmax_spatial_impl::SpatialSoftMax_getBlockSize(dim_size, inner_size);
    int inner_blocks = (inner_size + block_dims.y - 1) / block_dims.y;

    dim3 grid_dims(outer_size, inner_blocks);
    // printf("block_dims.x=%d, block_dims.y=%d, outer_size=%d, inner_blocks=%d, \n", 
    // block_dims.x, block_dims.y, outer_size, inner_blocks);
    const bool is_multi_warp = block_dims.x > 1;

    if (is_multi_warp) {
        int block_threads = block_dims.x * block_dims.y;
        int smem_size = block_threads * sizeof(acc_t);
        softmax_spatial_impl::softmax_spatial_bwd<input_t, acc_t, output_t, is_log_softmax, true>
            <<<grid_dims, block_dims, smem_size, stream>>>(gradInput, gradOutput, output, outer_size, dim_size, inner_size);
    } else {
        softmax_spatial_impl::softmax_spatial_bwd<input_t, acc_t, output_t, is_log_softmax, false>
            <<<grid_dims, block_dims, 0, stream>>>(gradInput, gradOutput, output, outer_size, dim_size, inner_size);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}   // namespace softmax_fwd



}   // namespace native
}   // namespace at
