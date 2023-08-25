#pragma once

#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <type_traits>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/UpSample.cuh>

#include <ATen/NumericUtils.h>



#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

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
#include <oneflow_softmax.cuh>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif

namespace at {
namespace native {

namespace {



#if defined(__HIP_PLATFORM_HCC__)
#define C10_WARP_SIZE 64
#else
#define C10_WARP_SIZE 32
#endif

template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
    return __int_as_float(0x7f800000U);
}

template<>
__inline__ __device__ double Inf<double>() {
    return __longlong_as_double(0x7ff0000000000000ULL);
}

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
};

constexpr int ALIGN_BYTES = 16;

constexpr double inf = INFINITY;

template <typename T>
struct numeric_limits {
};

template <>
struct numeric_limits<float> {
    static inline __host__ __device__ float lowest() { return -FLT_MAX; }
    static inline __host__ __device__ float max() { return FLT_MAX; }
    static inline __host__ __device__ float lower_bound() { return -static_cast<float>(inf); }
    static inline __host__ __device__ float upper_bound() { return static_cast<float>(inf); }
};

template <>
struct numeric_limits<double> {
    static inline __host__ __device__ double lowest() { return -DBL_MAX; }
    static inline __host__ __device__ double max() { return DBL_MAX; }
    static inline __host__ __device__ double lower_bound() { return -inf; }
    static inline __host__ __device__ double upper_bound() { return inf; }
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
    __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
            : max_input(max_input),  logsum(log(sum)) {}

    __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(input - max_input - logsum);
    }

    const AccumT max_input;
    const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
    __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
            : sum(sum) {}

    __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
        return static_cast<T>(gradOutput - exp(static_cast<AccumT>(output)) * sum);
    }

    const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
    __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
            : max_input(max_input)
            , sum(sum) {}

    __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(exp(input - max_input) / sum);
    }

    const AccumT max_input;
    const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
    __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
            : sum(sum) {}

    // XXX: gradOutput that we get here is really gradOutput * output
    // Look for cmul in SoftMax_updateGradInput
    __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
        return static_cast<T>(gradOutput - output * sum);
    }

    const AccumT sum;
};

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

template<typename T>
struct Max {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? b : a;
    }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = __shfl_xor(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "C10_WARPS_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architectures, but there is no guarantee this will not change for future arch.
// ROCm warp size is 64 for all currently ROCm-supported GPU architectures, but this may change for future archs.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.
template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * stride + local_idx;
    dst += first_batch * stride + local_idx;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_log_softmax) {
              sum[i] += std::exp(elements[i][it] - max_value[i]);
            } else {
              elements[i][it] = std::exp(elements[i][it] - max_value[i]);
              sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}



template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE];
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    acc_t sum[WARP_BATCH];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0];
#pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if (is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                    \
            softmax_warp_forward<input_t, output_t, acc_t, L2E, is_log_softmax>   \
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                    src, batch_count, softmax_elements_stride, softmax_elements); \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            break;

            LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
            LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
            LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
            LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
            LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
            LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
            LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
            LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
            LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
            LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
            LAUNCH_SOFTMAX_WARP_FORWARD(10); ; // 1024
            default:
                break;
        }
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    // TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        // std::cout << "torch grad_warp_impl grid_size: " << blocks << " block_x: " << threads.x << " block_y: " << threads.y << " block_z: " << threads.z << std::endl;

        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_BACKWARD(L2E) case L2E:                      \
            softmax_warp_backward<input_t, output_t, acc_t, L2E, is_log_softmax> \
                <<<blocks, threads, 0, 0>>>       \
                (grad_input, grad, output, batch_count, softmax_elements_stride, \
                softmax_elements);                                              \
            break;

            LAUNCH_SOFTMAX_WARP_BACKWARD(0); // 1
            LAUNCH_SOFTMAX_WARP_BACKWARD(1); // 2
            LAUNCH_SOFTMAX_WARP_BACKWARD(2); // 4
            LAUNCH_SOFTMAX_WARP_BACKWARD(3); // 8
            LAUNCH_SOFTMAX_WARP_BACKWARD(4); // 16
            LAUNCH_SOFTMAX_WARP_BACKWARD(5); // 32
            LAUNCH_SOFTMAX_WARP_BACKWARD(6); // 64
            LAUNCH_SOFTMAX_WARP_BACKWARD(7); // 128
            LAUNCH_SOFTMAX_WARP_BACKWARD(8); // 256
            LAUNCH_SOFTMAX_WARP_BACKWARD(9); // 512
            LAUNCH_SOFTMAX_WARP_BACKWARD(10); // 1024
            default:
                break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Spatial kernel (fast with large inner_size and small dim_size)
////////////////////////////////////////////////////////////////////////////////
// Let's assume that our input has been flattened to have only three dimension:
//     outer x dim x inner
// The spatial algorithm tries to parallelize along all of them.
// Within a 2d block threadIdx.y paralelizes over dim slices, and threads that
// share it will speed up reductions over dim (along axis x).
// The 2d grid is used to parallelize inner dimension over y axis and outer over x.
inline dim3 SpatialSoftMax_getGridSize(
        dim3 block, uint32_t max_active_blocks,
        uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
    // First, tile as many blocks as we can over the y axis
    uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
    if (inner_blocks > max_active_blocks)
        inner_blocks = max_active_blocks;
    // Fill the x axis with as many blocks as we can fit (a little more is ok too)
    uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
    if (outer_blocks > outer_size)
        outer_blocks = outer_size;
    return dim3(outer_blocks, inner_blocks);
}

const int max_threads = 1024;

inline dim3 SpatialSoftMax_getBlockSize(
        uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
    uint32_t inner_threads = inner_size;
    inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
    uint32_t dim_threads = 1;
    if (inner_threads <= 64 && dim_size >= 64) {
        while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
            dim_threads *= 2;
        dim_threads /= 2;
    }
    return dim3(dim_threads, inner_threads);
}


template<typename accscalar_t, typename Kernel>
void SpatialSoftMax_getLaunchSizes(
        Kernel k,
        uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
        dim3& grid, dim3& block, uint32_t& smem_size) {
    block = SpatialSoftMax_getBlockSize(outer_size, dim_size, inner_size);
    uint32_t block_threads = block.x * block.y;
    smem_size = block.x == 1 ? 0 : block_threads * sizeof(accscalar_t);
    int max_active_blocks;
    int dev;
    int sm_count;
#if defined(__HIP_PLATFORM_HCC__)
    // HIP function signature is not compatible yet.
    int max_blocks;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                 k, block_threads, smem_size);
    max_active_blocks = max_blocks;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, dev);
#else
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  k, block_threads, smem_size);
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
#endif
    max_active_blocks *= sm_count;
    grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, dim_size, inner_size);
}

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
    uint64_t block_size = 1;
    uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));

    // In the vectorized case we want to trade off allowing more of the buffers to be accessed
    // in a vectorized way against wanting a larger block size to get better utilisation.
    // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
    // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
    // allowing a larger block size.
    if (ILP > 1) {
        max_block_size /= 2;
    }

    while (block_size < (max_block_size)) block_size *= 2;
    // Launch at least a single warp - the kernel assumes that.
    block_size = std::max(block_size, static_cast<uint64_t>(C10_WARP_SIZE));
    return dim3(block_size);
}

// template<typename T>
// struct Add {
//   __device__ __forceinline__ T operator()(T a, T b) const {
//     return a + b;
//   }
// };

// template<typename T>
// struct Max {
//   __device__ __forceinline__ T operator()(T a, T b) const {
//     return a < b ? b : a;
//   }
// };

// Note that it's not a complete block-wide reduction.
// Only threads that share threadIdx.y reduce values.
template<typename T, template<typename> class ReduceOp>
__forceinline__ __device__
T spatialBlockReduceX(T *shared, T val) {
    ReduceOp<T> r;
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

template <typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxForward(
        outscalar_t *output, scalar_t *input,
        uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<accscalar_t*>(smem);
    const uint32_t outer_stride = inner_size * dim_size;
    const uint32_t dim_stride = inner_size;

    for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
        const uint32_t outer_offset = outer_index * outer_stride;
        for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const uint32_t data_offset = outer_offset + inner_index;
            ////////////////////////////////////////////////////////////
            // These two blocks are really equivalent, but specializing on
            // blockDim.x == 1 makes the kernel faster when it's unused.
            // I didn't want to thread an extra template parameter, and nvcc
            // seems to be smart enough to hoist the if outside of the loops.
            ////////////////////////////////////////////////////////////

            if (blockDim.x > 1) {
                accscalar_t max_input = numeric_limits<accscalar_t>::lowest();
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    const accscalar_t value = static_cast<accscalar_t>(input[data_offset + d * dim_stride]);
                    max_input = Max<accscalar_t>()(max_input, value);
                }
                max_input = spatialBlockReduceX<accscalar_t, Max>(sdata,max_input);

                accscalar_t sum = 0;
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
                    sum += exp(static_cast<accscalar_t>(input[data_offset + d * dim_stride])
                               - max_input);
                sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

                Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_input, sum);
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
            } else {
                accscalar_t max_input = numeric_limits<accscalar_t>::lowest();
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    const accscalar_t value = static_cast<accscalar_t>(input[data_offset + d * dim_stride]);
                    max_input = Max<accscalar_t>()(max_input, value);
                }
                accscalar_t sum = 0;
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
                    sum += exp(static_cast<accscalar_t>(input[data_offset + d * dim_stride])
                               - max_input);
                Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_input, sum);
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
                    output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
            }
        }
    }
}



template <typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxBackward(
        scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput,
        uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<accscalar_t*>(smem);
    const uint32_t outer_stride = inner_size * dim_size;
    const uint32_t dim_stride = inner_size;

    for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
        const uint32_t outer_offset = outer_index * outer_stride;
        for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
            const uint32_t data_offset = outer_offset + inner_index;
            // See the comment in forward kernel
            if (blockDim.x > 1) {
                accscalar_t sum = 0;
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
                    sum += gradOutput[data_offset + d * dim_stride];
                sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

                Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
                for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
                    gradInput[data_offset + d * dim_stride] =
                            epilogue(gradOutput[data_offset + d * dim_stride],
                                     output[data_offset + d * dim_stride]);
                }
            } else {
                accscalar_t sum = 0;
                for (uint32_t d = 0; d < dim_size; d++)
                    sum += gradOutput[data_offset + d * dim_stride];

                Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
                for (uint32_t d = 0; d < dim_size; d++) {
                    gradInput[data_offset + d * dim_stride] =
                            epilogue(gradOutput[data_offset + d * dim_stride],
                                     output[data_offset + d * dim_stride]);
                }
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + exp(v - max_k);
  }

  const AccumT max_k;
};


template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
    // To avoid RaW races from chaining blockReduce calls together, we need a sync here
    __syncthreads();

    smem[threadIdx.x] = val;

    __syncthreads();

    AccumT warpVal = defaultVal;

    // First warp will perform per-warp reductions for the remaining warps
    uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
    if (threadIdx.x < C10_WARP_SIZE) {
        int lane = threadIdx.x % C10_WARP_SIZE;
        if (lane < blockDim.x / C10_WARP_SIZE) {
#pragma unroll
            for (int i = 0; i < C10_WARP_SIZE; ++i) {
                warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
            }
#ifndef __HIP_PLATFORM_HCC__
            __syncwarp(mask);
#endif
            smem[lane] = warpVal;
        }
    }

    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    AccumT blockVal = defaultVal;

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
            blockVal = r(blockVal, smem[i]);
        }
        smem[0] = blockVal;
    }

    // Sync and broadcast
    __syncthreads();
    return smem[0];
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(int shift,
          T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
    using LoadT = aligned_vector<T, ILP>;
    AccumT threadVal = defaultVal;
    int offset = threadIdx.x;

    // shift and do 1
    if(shift > 0){
        data -= shift;
        size += shift;
        if(threadIdx.x >= shift){
            threadVal = r(threadVal, data[offset]);
        }
        size -= blockDim.x;
        data += blockDim.x;
    }
    int last = size % (ILP * blockDim.x);

    T v[ILP];
    LoadT* value = reinterpret_cast<LoadT*>(&v);

    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *value = reinterpret_cast<LoadT*>(data)[offset];

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            threadVal = r(threadVal, v[j]);
        }
    }

    offset = size - last + threadIdx.x;
    // Epilogue
    for (; offset < size; offset += blockDim.x)
        threadVal = r(threadVal, data[offset]);

    return threadVal;
}

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResultsVectorized(
        int size,
        const int shift,
        scalar_t *input,
        outscalar_t *output,
        Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
    using LoadT = aligned_vector<scalar_t, ILP>;
    using StoreT = aligned_vector<outscalar_t, ILP>;

    int offset = threadIdx.x;

    // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
    if (shift > 0) {
        input -= shift;
        output -= shift;
        size += shift;

        if (threadIdx.x >= shift) {
            output[offset] = epilogue(input[offset]);
        }
        size -= blockDim.x;
        input += blockDim.x;
        output += blockDim.x;
    }

    const int last = size % (ILP * blockDim.x);

    scalar_t in_v[ILP];
    LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);

    outscalar_t out_v[ILP];
    StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);

    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *in_value = reinterpret_cast<LoadT*>(input)[offset];

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            out_v[j] = epilogue(in_v[j]);
        }

        reinterpret_cast<StoreT*>(output)[offset] = *out_value;
    }

    offset = size - last + threadIdx.x;
    // handle the tail
    for (; offset < size; offset += blockDim.x) {
        output[offset] = epilogue(input[offset]);
    }
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteBpropResultsVectorized(
        int size,
        const int shift,
        scalar_t *gradInput,
        outscalar_t *output,
        outscalar_t *gradOutput,
        Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
    using gradInputT = aligned_vector<scalar_t, ILP>;
    using outputT = aligned_vector<outscalar_t, ILP>;

    int offset = threadIdx.x;

    // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
    if (shift > 0) {
        gradInput -= shift;
        output -= shift;
        gradOutput -= shift;
        size += shift;

        if (threadIdx.x >= shift) {
            gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
        }
        size -= blockDim.x;
        gradInput += blockDim.x;
        output += blockDim.x;
        gradOutput += blockDim.x;
    }

    const int last = size % (ILP * blockDim.x);

    scalar_t dX[ILP];
    gradInputT *dX_v = reinterpret_cast<gradInputT*>(&dX);

    outscalar_t Y[ILP];
    outputT *Y_v = reinterpret_cast<outputT*>(&Y);

    outscalar_t dY[ILP];
    outputT *dY_v = reinterpret_cast<outputT*>(&dY);

    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *Y_v = reinterpret_cast<outputT*>(output)[offset];
        *dY_v = reinterpret_cast<outputT*>(gradOutput)[offset];

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            dX[j] = epilogue(dY[j], Y[j]);
        }

        reinterpret_cast<gradInputT*>(gradInput)[offset] = *dX_v;
    }

    offset = size - last + threadIdx.x;
    for (; offset < size; offset += blockDim.x) {
        gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
    }
}

/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResults(
        int classes,
        scalar_t *input,
        outscalar_t *output,
        Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
    int offset = threadIdx.x;

    int last = classes % (ILP * blockDim.x);

    // Main bulk of loop with ILP
    for (; offset < classes - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            tmp[j] = input[offset + j * blockDim.x];
        }
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            output[offset + j * blockDim.x] = epilogue(tmp[j]);
        }
    }

    // Remainder - no ILP
    for (; offset < classes; offset += blockDim.x) {
        output[offset] = epilogue(input[offset]);
    }
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteBpropResults(
        int classes,
        scalar_t *gradInput,
        outscalar_t *output,
        outscalar_t *gradOutput,
        Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {

    int offset = threadIdx.x;

    int last = classes % (ILP * blockDim.x);

    for (; offset < classes - last; offset += blockDim.x * ILP) {
        outscalar_t tmpOutput[ILP];
        outscalar_t tmpGradOutput[ILP];

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            tmpOutput[j] = output[offset + j * blockDim.x];
            tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
        }

#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            gradInput[offset + j * blockDim.x] = epilogue(tmpGradOutput[j], tmpOutput[j]);
        }
    }

    // Remainder - no ILP
    for (; offset < classes; offset += blockDim.x) {
        gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
    }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}


template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxBackward(scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput, int classes)
{
    using LoadT = aligned_vector<scalar_t, ILP>;
    using StoreT = aligned_vector<outscalar_t, ILP>;

    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<accscalar_t*>(smem);
    gradInput += blockIdx.x * classes;
    output += blockIdx.x * classes;
    gradOutput += blockIdx.x * classes;

    const int shift = ((uint64_t)gradInput) % ALIGN_BYTES / sizeof(scalar_t);
    const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);
    const int grad_output_shift = ((uint64_t)gradOutput) % ALIGN_BYTES / sizeof(outscalar_t);

    accscalar_t threadSum = ilpReduce<AddFloat, ILP, outscalar_t, accscalar_t>(
            grad_output_shift, gradOutput, classes, AddFloat<outscalar_t, accscalar_t>(), accscalar_t(0));
    accscalar_t sum_k = blockReduce<Add, accscalar_t>(
            sdata, threadSum, Add<accscalar_t>(), accscalar_t(0));

    Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum_k);

    if (shift == output_shift && shift == grad_output_shift) {
        WriteBpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, gradInput, output, gradOutput, epilogue);
    } else {
        WriteBpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, gradInput, output, gradOutput, epilogue);
    }
}

// #if defined(__HIP_PLATFORM_HCC__)
// #define STREAM hipStream_t
// #else
// #define STREAM cudaStream_t
// #endif

//aiss add for of kernel softmax 20221020


// template<template<typename, typename, typename> class Epilogue, bool is_log_softmax, typename T>
// template<template<typename, typename, typename> class Epilogue, bool is_log_softmax, typename T>
// T* torch_softmax(T* input_, const int64_t dim_, const bool half_to_float, T* output){
template<template<typename, typename, typename> class Epilogue, bool is_log_softmax>
Tensor torch_softmax(const Tensor & input_, const int64_t dim_, const bool half_to_float, const Tensor& output){
  if (half_to_float) {
    TORCH_CHECK(input_.scalar_type() == ScalarType::Half, "conversion is supported for Half type only");
  }
  auto input = input_.contiguous();
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1
//aiss
    //printf("dim=%d, outer_size=%d, dim_size=%d, inner_size=%d\n", dim, outer_size, dim_size, inner_size);
    if (inner_size == 1) {
        dim3 grid(outer_size);
        // AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "torch_softmax", [&] {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "torch_softmax", [&] {
        
            using accscalar_t = acc_type<scalar_t, true>;
            if (!half_to_float) {
                if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
                    auto output_ptr = output.data_ptr<scalar_t>();
                    auto input_ptr = input.data_ptr<scalar_t>();
                    int64_t remaining = outer_size;
                    int64_t chunk_size = (1L << 30L) / dim_size;
                    while(remaining > 0) {
                    dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                        output_ptr, input_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
                    input_ptr += chunk_size * dim_size;
                    output_ptr += chunk_size * dim_size;
                    remaining -= chunk_size;
                    }
                }
                else if (1024 < dim_size && dim_size <= 4096) {     //aiss add of kernel
                    //aiss
                    //printf("input kernel 2!!!!!!\n");
                    // using accscalar_t = acc_type<scalar_t, true>;
                    auto input_ptr = input.data_ptr<scalar_t>();
                    auto output_ptr = output.data_ptr<scalar_t>();
                    using ComputeType = typename oneflow::cuda::softmax::DefaultComputeType<scalar_t>::type;
                    oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load(input_ptr, dim_size);
                    oneflow::cuda::softmax::DirectStore<ComputeType, scalar_t> store(output_ptr, dim_size);
                    
                    if (!is_log_softmax) {
                        //printf("start softmax!!!");
                        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
                            stream, load, store, outer_size, dim_size);
                    } 
                    else {
                        oneflow::cuda::softmax::DispatchLogSoftmax<decltype(load), decltype(store), ComputeType>(
                            stream, load, store, outer_size, dim_size);
                    }
                }
                else {//aiss add pf kernel end
                    // constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
                    // dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                    // cunn_SoftMaxForward<ILP, T, T, T, Epilogue>
                    // <<<grid, block, block.x * sizeof(T), stream>>>(
                    //         output, input, dim_size);
                    constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
                    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                    cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                    <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), dim_size);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            }
            else {
                if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
                    auto output_ptr = output.data_ptr<accscalar_t>();
                    auto input_ptr = input.data_ptr<scalar_t>();
                    int64_t remaining = outer_size;
                    int64_t chunk_size = (1<<30) / dim_size;
                    while(remaining > 0) {
                    dispatch_softmax_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                        output_ptr, input_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
                    input_ptr += chunk_size * dim_size;
                    output_ptr += chunk_size * dim_size;
                    remaining -= chunk_size;
                    }
                }
                else if (1024 < dim_size <= 4096 ) {     //aiss add of kernel
                    auto input_ptr = input.data_ptr<scalar_t>();
                    auto output_ptr = output.data_ptr<scalar_t>();
                    using ComputeType = typename oneflow::cuda::softmax::DefaultComputeType<scalar_t>::type;
                    oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load(input_ptr, dim_size);
                    oneflow::cuda::softmax::DirectStore<ComputeType, scalar_t> store(output_ptr, dim_size);
                    if (!is_log_softmax) {
                        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
                            stream, load, store, outer_size, dim_size);
                    } 
                    else {
                        oneflow::cuda::softmax::DispatchLogSoftmax<decltype(load), decltype(store), ComputeType>(
                            stream, load, store, outer_size, dim_size);
                    }


                }
                else {
                    //aiss add pf kernel end
                    // constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
                    // dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                    // cunn_SoftMaxForward<ILP, T, T, T, Epilogue>
                    // <<<grid, block, block.x * sizeof(T), stream>>>(
                    //         output, input, dim_size);
                    constexpr int ILP = sizeof(float4) / sizeof(accscalar_t);
                    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                    cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
                    <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                        output.data_ptr<accscalar_t>(), input.data_ptr<scalar_t>(), dim_size);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                }
        }
    }); 
    }
        // This kernel runs in a 2D grid, where each application along y dimension has a fixed
        // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
        // Reductions over dim are done in a single-threaded manner.
    else {
        uint32_t smem_size;
        dim3 grid, block;
        // AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "torch_softmax", [&] {
            using accscalar_t = acc_type<scalar_t, true>;
        // using accscalar_t = acc_type<scalar_t, true>;
            if (!half_to_float) {
                SpatialSoftMax_getLaunchSizes<accscalar_t>(
                    &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>,
                    outer_size, dim_size, inner_size,
                    grid, block, smem_size);
                cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_size, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                SpatialSoftMax_getLaunchSizes<accscalar_t>(
                    &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
                    outer_size, dim_size, inner_size,
                    grid, block, smem_size);
                cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>
                <<<grid, block, smem_size, stream>>>(
                output.data_ptr<accscalar_t>(), input.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
  }
    return output;
}


template<template<typename, typename, typename> class Epilogue, bool is_log_softmax>
void torch_softmax_backward(const Tensor &grad_, const Tensor &output_, int64_t dim_, bool half_to_float,const Tensor &gI){
    int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
    if (grad_.numel() == 0) {
    return;
    }
    auto grad = grad_.contiguous();
    static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
    if (grad.dim() == 0) grad = grad.view(1);
    TORCH_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
    auto output = output_.contiguous();
    if (output.dim() == 0) output = output.view(1);
    int64_t outer_size = 1;
    int64_t dim_size = output.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
    for (int64_t i = dim + 1; i < output.dim(); ++i)
    inner_size *= output.size(i);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (inner_size == 1) {
        dim3 grid(outer_size);
        AT_DISPATCH_FLOATING_TYPES(gI.scalar_type(), "torch_softmax_backward", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
                auto gI_ptr = gI.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<scalar_t>();
                auto output_ptr = output.data_ptr<scalar_t>();
                int64_t remaining = outer_size;
                int64_t chunk_size = (1<<30) / dim_size;
                while(remaining > 0) {
                dispatch_softmax_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    gI_ptr, grad_ptr, output_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
                gI_ptr += chunk_size * dim_size;
                grad_ptr += chunk_size * dim_size;
                output_ptr += chunk_size * dim_size;
                remaining -= chunk_size;
                }
            } else if(1024 < dim_size <= 4096 && 4096 < dim_size*sizeof(float) <= 16384) {     //aiss add of kernel
                using ComputeType = typename oneflow::cuda::softmax::DefaultComputeType<scalar_t>::type;
                auto gI_ptr = gI.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<scalar_t>();
                auto output_ptr = output.data_ptr<scalar_t>();
                oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load_y(output_ptr, dim_size);
                oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load_dy(grad_ptr, dim_size);
                oneflow::cuda::softmax::DirectStore<ComputeType, scalar_t> store(gI_ptr, dim_size);


                if (!is_log_softmax) {
                    oneflow::cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                                    decltype(store), ComputeType>(
                        stream, load_y, load_dy, store, outer_size, dim_size);
                } else {
                    oneflow::cuda::softmax::DispatchLogSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                                        decltype(store), ComputeType>(
                        stream, load_y, load_dy, store, outer_size, dim_size);
                } 


            } else {//aiss add pf kernel end
                constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
                dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                    gI.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), dim_size
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();

            }
        } else {
            if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
                auto gI_ptr = gI.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<accscalar_t>();
                auto output_ptr = output.data_ptr<accscalar_t>();
                int64_t remaining = outer_size;
                int64_t chunk_size = (1<<30) / dim_size;
                while(remaining > 0) {
                dispatch_softmax_backward<accscalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    gI_ptr, grad_ptr, output_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
                gI_ptr += chunk_size * dim_size;
                grad_ptr += chunk_size * dim_size;
                output_ptr += chunk_size * dim_size;
                remaining -= chunk_size;
                }
            } else if(1024 < dim_size <= 4096) {     //aiss add of kernel
                using ComputeType = typename oneflow::cuda::softmax::DefaultComputeType<scalar_t>::type;
                auto gI_ptr = gI.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<scalar_t>();
                auto output_ptr = output.data_ptr<scalar_t>();
                oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load_y(output_ptr, dim_size);
                oneflow::cuda::softmax::DirectLoad<scalar_t, ComputeType> load_dy(grad_ptr, dim_size);
                oneflow::cuda::softmax::DirectStore<ComputeType, scalar_t> store(gI_ptr, dim_size);

                if (!is_log_softmax) {
                    oneflow::cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                                    decltype(store), ComputeType>(
                        stream, load_y, load_dy, store, outer_size, dim_size);
                } else {
                    oneflow::cuda::softmax::DispatchLogSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                                        decltype(store), ComputeType>(
                        stream, load_y, load_dy, store, outer_size, dim_size);
                } 


            } else {//aiss add pf kernel end                
                constexpr int ILP = sizeof(float4) / sizeof(accscalar_t);
                dim3 block = SoftMax_getBlockSize(ILP, dim_size);
                cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
                <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                    gI.data_ptr<scalar_t>(), output.data_ptr<accscalar_t>(), grad.data_ptr<accscalar_t>(), dim_size
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();

            }
        }
        });
    } else {
        uint32_t smem_size;
        dim3 grid, block;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gI.scalar_type(), "torch_softmax_backward", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            SpatialSoftMax_getLaunchSizes<accscalar_t>(
                &cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, scalar_t, Epilogue>,
                outer_size, dim_size, inner_size,
                grid, block, smem_size);

            cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_size, stream>>>(
                gI.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                outer_size, dim_size, inner_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            SpatialSoftMax_getLaunchSizes<accscalar_t>(
                &cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
                outer_size, dim_size, inner_size,
                grid, block, smem_size);

            cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>
                <<<grid, block, smem_size, stream>>>(
                gI.data_ptr<scalar_t>(), output.data_ptr<accscalar_t>(), grad.data_ptr<accscalar_t>(),
                outer_size, dim_size, inner_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        });
    }

}

struct TORCH_API structured__fused_softmax : public at::impl::MetaBase {  
    void meta(const Tensor& input, const int64_t dim, const bool half_to_float) {
        int64_t dim_ = maybe_wrap_dim(dim, input.dim());

        auto output_options =
          input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        if (half_to_float) {
        output_options = output_options.dtype(ScalarType::Float);
        }

        int64_t input_dim = input.dim() > 0 ? input.dim() : 1; 
        TORCH_CHECK(
          dim_ >= 0 && dim_ < input_dim,
          "dim must be non-negative and less than input dimensions");

        set_output(input.sizes(), output_options);
    }
    void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out){
        torch_softmax<SoftMaxForwardEpilogue,false>(self, dim, half_to_float, out);
    }
    
};
TORCH_META_FUNC_FINAL(_fused_softmax)

struct TORCH_API structured__fused_softmax_backward : public at::impl::MetaBase {  
    void meta(const Tensor& grad,const Tensor& output,int64_t dim,const Tensor& input) {
      TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
      checkSameSize("softmax_backward", grad_arg, output_arg);

      int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

      auto grad_input_options =
          grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      bool half_to_float = grad.scalar_type() != input.scalar_type();
      if (half_to_float) {
        // The code below is only valid for the CUDA implementation. It's "okay"
        // to put it here because half-to-float conversion is not supported by
        // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
        // implementation that should ideally go here as well, but there is at least
        // one test in which the grad and input dtypes do not match for the CPU
        // implementation of this kernel and it is not true that the grad type is
        // float and the input dtype is half (see #63057).
        if (grad.scalar_type() == ScalarType::Float &&
             input.scalar_type() == ScalarType::Half) {
          grad_input_options = grad_input_options.dtype(ScalarType::Half);
        }
      }

      int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
      TORCH_CHECK(
          dim_ >= 0 && dim_ < grad_dim,
          "dim must be non-negative and less than input dimensions");

      set_output(grad.sizes(), grad_input_options);
    }
    void impl(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, bool half_to_float, const at::Tensor &grad_input){
        torch_softmax_backward<SoftMaxBackwardEpilogue,false>(grad_output, output, dim, half_to_float, grad_input);          
    }
};
TORCH_META_FUNC_FINAL(_fused_softmax_backward)

struct TORCH_API structured__fused_log_softmax_cuda_out : public at::impl::MetaBase {  
    void meta(const Tensor& input, const int64_t dim, const bool half_to_float) {
        int64_t dim_ = maybe_wrap_dim(dim, input.dim());

        auto output_options =
            input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        if (half_to_float) {
        output_options = output_options.dtype(ScalarType::Float);
        }

        int64_t input_dim = input.dim() > 0 ? input.dim() : 1; 
        TORCH_CHECK(
            dim_ >= 0 && dim_ < input_dim,
            "dim must be non-negative and less than input dimensions");

        set_output(input.sizes(), output_options);
    }
    void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out){
        torch_softmax<LogSoftMaxForwardEpilogue,true>(self, dim, half_to_float, out);
    }
};
TORCH_META_FUNC_FINAL(_fused_log_softmax_cuda_out)

struct TORCH_API structured__fused_log_softmax_backward_cuda_out : public at::impl::MetaBase {
    void meta(const Tensor& grad,const Tensor& output,int64_t dim,const Tensor& input) {
      TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
      checkSameSize("softmax_backward", grad_arg, output_arg);

      int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

      auto grad_input_options =
          grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      bool half_to_float = grad.scalar_type() != input.scalar_type();
      if (half_to_float) {
        // The code below is only valid for the CUDA implementation. It's "okay"
        // to put it here because half-to-float conversion is not supported by
        // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
        // implementation that should ideally go here as well, but there is at least
        // one test in which the grad and input dtypes do not match for the CPU
        // implementation of this kernel and it is not true that the grad type is
        // float and the input dtype is half (see #63057).
        if (grad.scalar_type() == ScalarType::Float &&
             input.scalar_type() == ScalarType::Half) {
          grad_input_options = grad_input_options.dtype(ScalarType::Half);
        }
      }

      int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
      TORCH_CHECK(
          dim_ >= 0 && dim_ < grad_dim,
          "dim must be non-negative and less than input dimensions");

      set_output(grad.sizes(), grad_input_options);
    }
    void impl(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, bool half_to_float, const at::Tensor &grad_input){
        torch_softmax_backward<LogSoftMaxBackwardEpilogue,true>(grad_output, output, dim, half_to_float, grad_input);          
    }
};
TORCH_META_FUNC_FINAL(_fused_log_softmax_backward_cuda_out)
}


Tensor fuse_log_softmax_forward(const at::Tensor & self, int64_t dim, bool half_to_float) {
    c10::optional<Device> common_device = nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, self, "wrapper__log_softmax", "self");
    structured__fused_log_softmax_cuda_out_functional op;
    op.meta(self, dim, half_to_float);
    op.impl(self, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor fuse_log_softmax_backward(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    bool half_to_float = grad_output.scalar_type() != self.scalar_type();
    if (half_to_float) {
    TORCH_CHECK(
        (grad_output.scalar_type() == ScalarType::Float &&
         self.scalar_type() == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
    }
    c10::optional<Device> common_device = nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__log_softmax_backward_data", "grad_output");
    c10::impl::check_and_update_common_device(common_device, output, "wrapper__log_softmax_backward_data", "output");
    c10::impl::check_and_update_common_device(common_device, self, "wrapper__log_softmax_backward_data", "self");
    structured__fused_log_softmax_backward_cuda_out_functional op;
    op.meta(grad_output, output, dim, self);
    op.impl(grad_output, output, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}


Tensor fuse_softmax_forward(const Tensor &input, const int64_t dim, const bool half_to_float) {
    c10::optional<Device> common_device = nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, input, "fuse_softmax_forward", "input");
    structured__fused_softmax_functional op;
    op.meta(input, dim, half_to_float);
    op.impl(input, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor fuse_softmax_backward(const Tensor& grad, const Tensor& output, int64_t dim, const Tensor& input){
    bool half_to_float = grad.scalar_type() != input.scalar_type();
    if (half_to_float) {
    TORCH_CHECK(
        (grad.scalar_type() == ScalarType::Float &&
         input.scalar_type() == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
    }
    c10::optional<Device> common_device = nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, grad, "wrapper__softmax_backward_data", "grad");
    c10::impl::check_and_update_common_device(common_device, output, "wrapper__softmax_backward_data", "output");
    c10::impl::check_and_update_common_device(common_device, input, "wrapper__softmax_backward_data", "input");
    structured__fused_softmax_backward_functional op;
    op.meta(grad, output, dim, input);
    Tensor tmp = grad * output;
    op.impl(tmp, output, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

}
}

