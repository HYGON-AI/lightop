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

#include "softmax_math.cuh"

namespace at {
namespace native {

namespace softmax_warp_impl {
int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template <typename input_t, 
    typename output_t, 
    typename acc_t, 
    bool is_log_softmax,
    int log2_elements, 
    int threads_per_block>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int outer_size, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    // use 512 threads per block to maximimize gpu utilization
    constexpr int WARP_BATCH = (threads_per_block + next_power_of_two - 1) / next_power_of_two;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // outer_size might not be a multiple of WARP_BATCH. 设置此WARP中需要计算的batch数.
    int local_batches = outer_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    int idx_offset = first_batch * element_count + local_idx;
    src += idx_offset;
    dst += idx_offset;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
 
        for (int it=0; it < WARP_ITERATIONS; it++) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = static_cast<acc_t>(src[i*element_count+it*WARP_SIZE]);
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute online softmax
    const auto online_softmax_updater = softmax_op::OnlineSoftmaxUpdate<input_t, acc_t>();
    const auto online_softmax_combine_updater = softmax_op::OnlineSoftmaxCombineUpdate<input_t, acc_t>();

    acc_t max_value[WARP_BATCH];
    acc_t sum_value[WARP_BATCH] = { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            online_softmax_updater(elements[i][it], max_value[i], sum_value[i]);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t max_v = WARP_SHFL_XOR(max_value[i], offset, WARP_SIZE);
            acc_t sum_v = WARP_SHFL_XOR(sum_value[i], offset, WARP_SIZE);
            online_softmax_combine_updater(max_v, sum_v, max_value[i], sum_value[i]);
        }
    }

    // store result
    #pragma unroll
    for (int i=0; i < WARP_BATCH; ++i) {
        if (i >= local_batches)
            break;

        auto tmp_sum = softmax_math::log(sum_value[i]);
        int base_idx = i*element_count;

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;

            if (element_index < element_count) {
                if (sum_value[i] == 0) {
                    dst[base_idx + it * WARP_SIZE] = static_cast<output_t>(std::numeric_limits<acc_t>::quiet_NaN());
                } else {
                    if constexpr(is_log_softmax) {
                        auto tmp_element = static_cast<acc_t>(elements[i][it]);
                        dst[base_idx + it * WARP_SIZE] = static_cast<output_t>(tmp_element - max_value[i] - tmp_sum);
                    } else  {
                        auto tmp_element = static_cast<acc_t>(elements[i][it]);
                        dst[base_idx + it * WARP_SIZE] = static_cast<output_t>(softmax_math::exp(tmp_element - max_value[i] - tmp_sum));
                    }
                }
            } else {
                break;
            }
        }
    }
}

template <typename input_t, 
    typename output_t, 
    typename acc_t, 
    bool is_log_softmax,
    int log2_elements, 
    int threads_per_block>
__global__ void softmax_warp_backward(input_t *gradInput, const output_t *grad_output, const output_t *output, 
                                      int outer_size, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (threads_per_block + next_power_of_two - 1) / next_power_of_two;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // outer_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = outer_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * element_count + local_idx;
    grad_output += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = static_cast<acc_t>(grad_output[i*element_count+it*WARP_SIZE]);
                output_reg[i][it] = static_cast<acc_t>(output[i*element_count+it*WARP_SIZE]);
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    softmax_op::warp_batch_reduce<acc_t, WARP_BATCH, WARP_SIZE, softmax_math::Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;

        auto logsum_ = softmax_math::log(sum[i]);

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if constexpr(is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = static_cast<acc_t>(grad_reg[i][it] - softmax_math::exp(output_reg[i][it] + logsum_));
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = static_cast<acc_t>(grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}
    
} // namespace softmax_warp_impl
    

namespace softmax_block_impl {
const int max_threads = 1024;
constexpr int ALIGN_BYTES = 16;

inline int64_t SoftMax_getBlockSize(int ILP, int64_t dim_size) {
    int64_t block_size = 1;
    int64_t max_block_size = std::min(dim_size / ILP, static_cast<int64_t>(max_threads));
    
    // 在矢量化的情况下，我们希望在允许以矢量化的方式访问更多缓冲区与希望更大的块大小以获得更好的利用率之间进行权衡。
    // 一般来说，使用ILP，可以将访问的缓冲区的（ILP-1）/ILP矢量化，但有可能具有非常小的块大小。
    // 我们选择保持 >=1/2 的缓冲区矢量化，同时允许更大的块大小。
    if (ILP > 1) {
        max_block_size /= 2;
    }
    
    while (block_size < (max_block_size)) block_size *= 2;
    // Launch at least a single warp - the kernel assumes that.
    block_size = std::max(block_size, static_cast<int64_t>(at::cuda::warp_size()));
    return block_size;
}

template <int ILP, typename input_t, typename output_t, typename Epilogue>
__device__ __forceinline__ void
WriteFpropResultsVectorized(int size,
                            const input_t *input,
                            output_t *output,
                            const Epilogue &epilogue) {
    using LoadT = at::native::memory::aligned_vector<input_t, ILP>;
    using StoreT = at::native::memory::aligned_vector<output_t, ILP>;
    
    int offset = threadIdx.x;
    
    const int last = size % (ILP * blockDim.x);
    
    input_t in_v[ILP];
    LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);
    
    output_t out_v[ILP];
    StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);
    
    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *in_value = reinterpret_cast<const LoadT*>(input)[offset];
    
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

template <int ILP, typename input_t, typename acc_t, typename Reduction>
__device__ __forceinline__ void softmax_warp_online_reduce(const input_t* data, int size,
                                                           const Reduction& r,
                                                           acc_t &thread_max, acc_t &thread_sum) 
{
    using LoadT = at::native::memory::aligned_vector<input_t, ILP>;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int last = size % (ILP * block_size);

    input_t v[ILP];
    auto value = reinterpret_cast<LoadT*>(&v);
    for (; tid * ILP < (size - last); tid += block_size) {
        *value = reinterpret_cast<const LoadT*>(data)[tid];

        #pragma unroll
        for (int j = 0; j < ILP; ++j) {
            r(v[j], thread_max, thread_sum);
        }
    }

    // 对于不够向量化读取的长度进行收尾
    tid = size - last + tid;
    // Epilogue
    for (; tid < size; tid += block_size) {
        r(data[tid], thread_max, thread_sum);
    }
}

template <typename acc_t, typename ReductionCombine>
__device__ __forceinline__ void softmax_block_online_reduce_v1(
    acc_t* smem_max, const acc_t &thread_max, 
    acc_t* smem_sum, const acc_t &thread_sum, 
    const ReductionCombine& r,
    acc_t &block_max, acc_t &block_sum) 
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // To avoid RaW races from chaining blockReduce calls together, we need a sync here
    __syncthreads();

    smem_max[tid] = thread_max;
    smem_sum[tid] = thread_sum;
    __syncthreads();

    acc_t warp_max = block_max;
    acc_t warp_sum = block_sum;
    // First warp will perform per-warp reductions for the remaining warps
    if (tid < C10_WARP_SIZE) {
        int lane = tid % C10_WARP_SIZE;
        if (lane < block_size / C10_WARP_SIZE) {
            #pragma unroll
            for (int i = 0; i < C10_WARP_SIZE; ++i) {
                auto max_v = smem_max[lane * C10_WARP_SIZE + i];
                auto sum_v = smem_sum[lane * C10_WARP_SIZE + i];
                r(max_v, sum_v, warp_max, warp_sum);
            }
            smem_max[lane] = warp_max;
            smem_sum[lane] = warp_sum;
        }
    }

    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    if (tid == 0) {
        for (int i = 0; i < block_size / C10_WARP_SIZE; ++i) {
            auto max_v = smem_max[i];
            auto sum_v = smem_sum[i];
            r(max_v, sum_v, block_max, block_sum);
        }

        smem_max[0] = block_max;
        smem_sum[0] = block_sum;
    }

    // Sync and broadcast
    __syncthreads();

    block_max = smem_max[0];
    block_sum = smem_sum[0];
}

template <typename acc_t, typename ReductionCombine>
__device__ __forceinline__ void softmax_block_online_reduce_v2(
    acc_t* smem_max, const acc_t &thread_max, 
    acc_t* smem_sum, const acc_t &thread_sum, 
    const ReductionCombine& r,
    acc_t &block_max, acc_t &block_sum) 
{
    __syncthreads();

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int lane_id = tid % C10_WARP_SIZE;
    int warp_id = tid / C10_WARP_SIZE;

    acc_t warp_max = thread_max;
    acc_t warp_sum = thread_sum;

    #pragma unroll
    for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc_t max_v = WARP_SHFL_XOR(warp_max, offset, C10_WARP_SIZE);
        acc_t sum_v = WARP_SHFL_XOR(warp_sum, offset, C10_WARP_SIZE);
        r(max_v, sum_v, warp_max, warp_sum);
    }

    if(lane_id == 0) {
        smem_max[warp_id] = warp_max;
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    if (tid == 0) {
        for (int i = 0; i < block_size / C10_WARP_SIZE; ++i) {
            auto max_v = smem_max[i];
            auto sum_v = smem_sum[i];
            r(max_v, sum_v, block_max, block_sum);
        }

        smem_max[0] = block_max;
        smem_sum[0] = block_sum;
    }

    // Sync and broadcast
    __syncthreads();

    block_max = smem_max[0];
    block_sum = smem_sum[0];
}

template <int ILP, 
    typename input_t, 
    typename output_t, 
    typename acc_t, 
    bool is_log_softmax,
    bool using_smem_blockreduce>
__global__ void softmax_block_forward(output_t *output, const input_t *input, int element_count)
{
    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<acc_t*>(smem);

    using LoadT = at::native::memory::aligned_vector<input_t, ILP>;
    using StoreT = at::native::memory::aligned_vector<output_t, ILP>;

    // forward pointers to batch[blockIdx.x]，每个block计算一行
    // each block handles a sample in the mini-batch
    const int64_t batch_offset = static_cast<int64_t>(blockIdx.x) * element_count;
    input += batch_offset;
    output += batch_offset;

    //// 1.执行warp reduce
    acc_t thread_max = at::numeric_limits<acc_t>::lowest();
    acc_t thread_sum = static_cast<acc_t>(0);
    const auto online_softmax_updater = softmax_op::OnlineSoftmaxUpdate<input_t, acc_t>();
    softmax_warp_online_reduce<ILP, input_t, acc_t>(input, element_count, 
                                                    online_softmax_updater, thread_max, thread_sum);

    //// 2.执行block reduce
    int block_size = blockDim.x;
    // 设置lds起点
    auto sdata_max = sdata;
    auto sdata_sum = sdata + block_size;
    acc_t block_max = at::numeric_limits<acc_t>::lowest();
    acc_t block_sum = static_cast<acc_t>(0);
    const auto online_softmax_combine_updater = softmax_op::OnlineSoftmaxCombineUpdate<input_t, acc_t>();
    if constexpr(using_smem_blockreduce) {
        softmax_block_online_reduce_v1<acc_t>(
            sdata_max, thread_max, sdata_sum, thread_sum, online_softmax_combine_updater,
            block_max, block_sum);
    } else {
        softmax_block_online_reduce_v2<acc_t>(
            sdata_max, thread_max, sdata_sum, thread_sum, online_softmax_combine_updater,
            block_max, block_sum);
    }

    //// 3.将结果写入global中
    if constexpr(is_log_softmax) {
        auto softmax_epilogue = softmax_op::LogSoftMaxForwardEpilogue<input_t, acc_t, output_t>(block_max, block_sum);
        WriteFpropResultsVectorized<ILP, input_t, output_t>(element_count, input, output, softmax_epilogue);
    } else {
        auto softmax_epilogue = softmax_op::SoftMaxForwardEpilogue<input_t, acc_t, output_t>(block_max, block_sum);
        WriteFpropResultsVectorized<ILP, input_t, output_t>(element_count, input, output, softmax_epilogue);
    }
}


template <int ILP, typename T, typename acc_t, typename Reduction>
__device__ __forceinline__ acc_t
ilpReduce(const T* data,
          const int &size,
          const Reduction& r,
          const acc_t &defaultVal)
{
    using LoadT = at::native::memory::aligned_vector<T, ILP>;
    acc_t threadVal = defaultVal;
    int offset = threadIdx.x;
    int last = size % (ILP * blockDim.x);

    T v[ILP];
    LoadT* value = reinterpret_cast<LoadT*>(&v);

    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *value = reinterpret_cast<const LoadT*>(data)[offset];

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


template <typename acc_t, typename Reduction>
__device__ __forceinline__ acc_t blockReduce(acc_t* smem, acc_t val,
                                             const Reduction& r,
                                             const acc_t &defaultVal)
{
    // To avoid RaW races from chaining blockReduce calls together, we need a sync here
    __syncthreads();

    smem[threadIdx.x] = val;

    __syncthreads();

    acc_t warpVal = defaultVal;

    // First warp will perform per-warp reductions for the remaining warps
    if (threadIdx.x < C10_WARP_SIZE) {
        int lane = threadIdx.x % C10_WARP_SIZE;
        if (lane < blockDim.x / C10_WARP_SIZE) {
        #pragma unroll
            for (int i = 0; i < C10_WARP_SIZE; ++i) {
                warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
            }
            smem[lane] = warpVal;
        }
    }

    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    acc_t blockVal = defaultVal;
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


template <int ILP, 
    typename input_t, 
    typename acc_t, 
    typename output_t, 
    typename Epilogue>
__device__ __forceinline__ void
WriteBpropResultsVectorized(int size,
                            input_t *gradInput,
                            const output_t *output,
                            const output_t *gradOutput,
                            const Epilogue &epilogue) {
    using gradInputT = at::native::memory::aligned_vector<input_t, ILP>;
    using outputT = at::native::memory::aligned_vector<output_t, ILP>;

    int offset = threadIdx.x;
    const int last = size % (ILP * blockDim.x);

    input_t dX[ILP];
    gradInputT *dX_v = reinterpret_cast<gradInputT*>(&dX);

    output_t Y[ILP];
    outputT *Y_v = reinterpret_cast<outputT*>(&Y);

    output_t dY[ILP];
    outputT *dY_v = reinterpret_cast<outputT*>(&dY);

    for (; offset * ILP < (size - last); offset += blockDim.x) {
        *Y_v = reinterpret_cast<const outputT*>(output)[offset];
        *dY_v = reinterpret_cast<const outputT*>(gradOutput)[offset];

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


template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, typename Epilogue>
__device__ __forceinline__ void
WriteBpropResults(int classes,
                scalar_t *gradInput,
                const outscalar_t *output,
                const outscalar_t *gradOutput,
                const Epilogue& epilogue) {
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


template <int ILP, 
    typename input_t, 
    typename output_t, 
    typename acc_t, 
    bool is_log_softmax>
__global__ void softmax_block_backward(input_t *gradInput, const output_t *output, const output_t *gradOutput, 
                                       int element_count)
{
    using LoadT = at::native::memory::aligned_vector<input_t, ILP>;
    using StoreT = at::native::memory::aligned_vector<output_t, ILP>;
  
    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<acc_t*>(smem);

    const int64_t batch_offset = static_cast<int64_t>(blockIdx.x) * element_count;
    gradInput += batch_offset;
    output += batch_offset;
    gradOutput += batch_offset;
  
    auto thread_op = softmax_math::AddFloat<output_t, acc_t>();
    acc_t thread_sum = ilpReduce<ILP, output_t, acc_t>(gradOutput, element_count, thread_op, acc_t(0));
    auto block_op = softmax_math::Add<acc_t>();
    acc_t block_sum = blockReduce<acc_t>(sdata, thread_sum, block_op, acc_t(0));

    //// 3.将结果写入global中
    if constexpr(is_log_softmax) {
        auto softmax_epilogue = softmax_op::LogSoftMaxBackwardEpilogue<input_t, acc_t, output_t>(block_sum);
        WriteBpropResultsVectorized<ILP, input_t, acc_t, output_t>(element_count, gradInput, output, gradOutput, softmax_epilogue);
        // WriteBpropResults<ILP, input_t, acc_t, output_t>(element_count, gradInput, output, gradOutput, softmax_epilogue);
    } else {
        auto softmax_epilogue = softmax_op::SoftMaxBackwardEpilogue<input_t, acc_t, output_t>(block_sum);
        WriteBpropResultsVectorized<ILP, input_t, acc_t, output_t>(element_count, gradInput, output, gradOutput, softmax_epilogue);
        // WriteBpropResults<ILP, input_t, acc_t, output_t>(element_count, gradInput, output, gradOutput, softmax_epilogue);
    }
}

} // namespace softmax_block_impl


namespace softmax_lastdim {
    
template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_lastdim_smallsize_forward(output_t *dst, const input_t *src, 
                                       const int64_t &dim_size, const int64_t &outer_size, 
                                       const cudaStream_t &stream) {
    if (dim_size == 0) { return; } 
    TORCH_INTERNAL_ASSERT( dim_size >= 0 && dim_size <= 8192);

    int log2_elements = softmax_warp_impl::log2_ceil(dim_size);
    const int next_power_of_two = 1 << log2_elements;

    // use 512 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 512;

    int warp_size = at::cuda::warp_size();
    warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;

    int batches_per_warp = (threads_per_block + next_power_of_two - 1) / next_power_of_two;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int grid_size = (outer_size + batches_per_block - 1) / batches_per_block;

    dim3 block_dims(warp_size, warps_per_block, 1);
                                        
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
        #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                                                              \
        softmax_warp_impl::softmax_warp_forward<input_t, output_t, acc_t, is_log_softmax, L2E, threads_per_block>       \
            <<<grid_size, block_dims, 0, stream>>>(dst, src, outer_size, dim_size);                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                 \
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
        LAUNCH_SOFTMAX_WARP_FORWARD(11); ; // 2048
        LAUNCH_SOFTMAX_WARP_FORWARD(12); ; // 4096
        LAUNCH_SOFTMAX_WARP_FORWARD(13); ; // 8192
        default:
            break;
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_lastdim_largesize_forward(output_t *dst, const input_t *src, 
                                       const int64_t &dim_size, const int64_t &outer_size,
                                       const cudaStream_t &stream) {
    constexpr int ILP = sizeof(float4) / sizeof(input_t);
    int64_t block_size = softmax_block_impl::SoftMax_getBlockSize(ILP, dim_size);
    int64_t grid_size = outer_size;
    int lds_size = block_size * sizeof(acc_t) * 2;

    softmax_block_impl::softmax_block_forward<ILP, input_t, output_t, acc_t, is_log_softmax, true>
        <<<grid_size, block_size, lds_size, stream>>>(dst, src, dim_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_lastdim_smallsize_backward(input_t *grad_input, const output_t *grad_output, const output_t *output, 
                                        const int &dim_size, const int &outer_size, 
                                        const cudaStream_t &stream)
{    
    if (dim_size == 0) { return; }
    TORCH_INTERNAL_ASSERT( dim_size >= 0 && dim_size <= 8192);

    int log2_elements = softmax_warp_impl::log2_ceil(dim_size);
    const int next_power_of_two = 1 << log2_elements;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 512;

    int warp_size = at::cuda::warp_size();
    warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;

    int batches_per_warp = (threads_per_block + next_power_of_two - 1) / next_power_of_two;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int grid_size = (outer_size + batches_per_block - 1) / batches_per_block;
    dim3 block_dims(warp_size, warps_per_block, 1);

    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
        #define LAUNCH_SOFTMAX_WARP_BACKWARD(L2E) case L2E:                         \
        softmax_warp_impl::softmax_warp_backward<input_t, output_t, acc_t, is_log_softmax, L2E, threads_per_block>        \
            <<<grid_size, block_dims, 0, stream>>>                                  \
            (grad_input, grad_output, output, outer_size, dim_size);               \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                             \
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
        LAUNCH_SOFTMAX_WARP_BACKWARD(11); ; // 2048
        LAUNCH_SOFTMAX_WARP_BACKWARD(12); ; // 4096
        LAUNCH_SOFTMAX_WARP_BACKWARD(13); ; // 8192
        default:
            break;
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void softmax_lastdim_largesize_backward(input_t *gradInput, const output_t *gradOutput, const output_t *output, 
                                        const int64_t &dim_size, const int64_t &outer_size,
                                        const cudaStream_t &stream) {
    constexpr int ILP = sizeof(float4) / sizeof(input_t);
    int64_t block_size = softmax_block_impl::SoftMax_getBlockSize(ILP, dim_size);
    int64_t grid_size = outer_size;
    int lds_size = block_size * sizeof(acc_t);

    softmax_block_impl::softmax_block_backward<ILP, input_t, output_t, acc_t, is_log_softmax>
        <<<grid_size, block_size, lds_size, stream>>>(gradInput, output, gradOutput, dim_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}   // namespace softmax_lastdim

}   // namespace native
}   // namespace at
