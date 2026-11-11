// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "softmax.h"
#include <torch/autograd.h>
#include <limits>
#include <stdint.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/autograd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_fp16.h>
#include "reduce.h"

namespace at {
namespace native {

class fuseSoftmaxFunction : public torch::autograd::Function<fuseSoftmaxFunction> {
public:
    static torch::autograd::variable_list forward(
    // static Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::Variable& input,
        const int64_t &dim, 
        const bool &half_to_float) {
        ctx->saved_data["dim"] = dim;
        // at::AutoDispatchBelowADInplaceOrView g;
        at::AutoNonVariableTypeMode g;

        if (input.device().type() == DeviceType::CUDA) {
            auto result = fuse_softmax_forward(input, dim, half_to_float);
            ctx->save_for_backward({input, result});
            return {result};
        } 
        else {
            TORCH_CHECK(0, "Unsupported device ", input.device().type());
        }
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_output) {
        // Use data saved in forward
        auto saved = ctx->get_saved_variables();
        auto save_input = saved[0];
        auto save_result = saved[1];

        if (grad_output[0].device().type() == DeviceType::CUDA) {
        auto grad_in = fuse_softmax_backward(
            grad_output[0],
            save_result,
            ctx->saved_data["dim"].toInt(),
            save_input);
        return {
            grad_in,
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()};
        } else {
            TORCH_CHECK(0, "Unsupported device ", grad_output[0].device().type());
        }
    }
};


class fuseLogSoftmaxFunction : public torch::autograd::Function<fuseLogSoftmaxFunction> {
public:
    static torch::autograd::variable_list forward(
    // static Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::Variable& input,
        const int64_t &dim, 
        const bool &half_to_float) {
        ctx->saved_data["dim"] = dim;
        // at::AutoDispatchBelowADInplaceOrView g;
        at::AutoNonVariableTypeMode g;
        
        if (input.device().type() == DeviceType::CUDA) {
            auto result = fuse_log_softmax_forward(input, dim, half_to_float);
            ctx->save_for_backward({input, result});
            return {result};
        }  else {
            TORCH_CHECK(0, "Unsupported device ", input.device().type());
        }
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_output) {
        // Use data saved in forward
        auto saved = ctx->get_saved_variables();
        auto save_input = saved[0];
        auto save_result = saved[1];
        if (grad_output[0].device().type() == DeviceType::CUDA) {
            auto grad_in = fuse_log_softmax_backward(
                grad_output[0],
                save_result,
                ctx->saved_data["dim"].toInt(),
                save_input);
            return {
                grad_in,
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable()};
        } else {
            TORCH_CHECK(0, "Unsupported device ", grad_output[0].device().type());
        }
    }
};
//新增代码
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    u += 0x7fff + ((u >> 16) & 1);
    out = u>>16;
  }
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}
template<typename scalar_t>
static __device__ __forceinline__ void from_float_large(scalar_t &out, float f) {
  if constexpr (std::is_same<scalar_t, uint16_t>::value) {
#if defined(__gfx938__)
    out = static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(f, false, false));
#else
    from_float(out, f);
#endif
  } else {
    out = f;
  }
}

template<typename scalar_t>
static __device__ __forceinline__ float to_float_large(scalar_t in) {
  if constexpr (std::is_same<scalar_t, uint16_t>::value) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_f32_bf16(static_cast<short>(in), false, 0, false);
#else
    return to_float(in);
#endif
  } else {
    return in;
  }
}
    constexpr int WARP_SIZE = 64;

    template<typename T>
    __inline__ __device__ T warpReduceMax(T val) {
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            val = std::max(val, __shfl_xor(val, mask));
        }
        return val;
    }

    template<typename T>
    __inline__ __device__ T warpReduceSum(T val) {
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            val += __shfl_xor(val,mask);
        }
        return val;
    }

    template<typename T>
    __inline__ __device__ T blockReduceMax(T val, T* shared_mem, int block_size) {
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        val = warpReduceMax(val);

        if (lane == 0) shared_mem[wid] = val;
        __syncthreads();

        T final_val = (threadIdx.x < (block_size / WARP_SIZE)) ? shared_mem[lane] : -FLT_MAX;
        if (wid == 0) final_val = warpReduceMax(final_val);

        __syncthreads(); 
        if (threadIdx.x == 0) shared_mem[0] = final_val;
        __syncthreads();
        
        return shared_mem[0];
    }

    template<typename T>
    __inline__ __device__ T blockReduceSum(T val, T* shared_mem, int block_size) {
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        val = warpReduceSum(val);
        if (lane == 0) shared_mem[wid] = val;
        __syncthreads();

        T final_val = (threadIdx.x < (block_size / WARP_SIZE)) ? shared_mem[lane] : static_cast<T>(0.0f);
        if (wid == 0) final_val = warpReduceSum(final_val);

        __syncthreads();
        if (threadIdx.x == 0) shared_mem[0] = final_val;

        __syncthreads();

        return shared_mem[0];
    }
 
template<typename scalar_t, int VEC_SIZE, int BLOCK_SIZE, int VEC_NUM_PER_THREAD>
__global__ void softmax_fused_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    float* __restrict__ global_max, 
    float* __restrict__ global_sum, 
    int* __restrict__ sync_counters, 
    int rows,
    int cols,
    int elem_per_block 
) {
   
    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_val;

    int blocks_per_row = (cols + elem_per_block - 1) / elem_per_block;
    int bid = blockIdx.x;
    int row = bid / blocks_per_row;
    int block_idx_in_row = bid % blocks_per_row;

    if (row >= rows) return;

    using VecType = AlignedVector<scalar_t, VEC_SIZE>;
    
    int block_offset = row * cols + block_idx_in_row * elem_per_block;
    const VecType* vec_input = reinterpret_cast<const VecType*>(input + block_offset);
    VecType* vec_output = reinterpret_cast<VecType*>(output + block_offset);

  
    VecType loaded_vecs[VEC_NUM_PER_THREAD];
    #pragma unroll 
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        int local_vec_idx = threadIdx.x + i * BLOCK_SIZE;
    
        if (local_vec_idx < elem_per_block / VEC_SIZE) {
             int global_offset = block_idx_in_row * elem_per_block + local_vec_idx * VEC_SIZE;
             
             if (global_offset + VEC_SIZE <= cols) {
                 loaded_vecs[i] = vec_input[local_vec_idx];
             } else {
                 #pragma unroll
                 for (int k = 0; k < VEC_SIZE; ++k) {
                     if (global_offset + k < cols) {
                         loaded_vecs[i].val[k] = input[row * cols + global_offset + k];
                     } else {
                         from_float_large(loaded_vecs[i].val[k], -std::numeric_limits<float>::infinity());
                     }
                 }
             }
        }
    }
    float local_max = -FLT_MAX;
    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val = to_float_large(loaded_vecs[i].val[k]);
            local_max = fmaxf(local_max, val);
        }
    }

    float block_max = BlockReduce(temp_storage).Reduce(local_max, hipcub::Max());
    
 
    if (threadIdx.x == 0) {
        atomicMax(&global_max[row], block_max);
        __threadfence(); 
        atomicAdd(&sync_counters[row], 1);
        while(*(volatile int*)&sync_counters[row] < blocks_per_row);
    }
    __syncthreads(); 


    if (threadIdx.x == 0) shared_val = global_max[row];
    __syncthreads();
    float final_global_max = shared_val;


    float local_sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val = to_float_large(loaded_vecs[i].val[k]);
            float res = __expf(val - final_global_max);
            local_sum += res;
            from_float_large(loaded_vecs[i].val[k], res);
        }
    }
    
    float block_sum = BlockReduce(temp_storage).Sum(local_sum);


    if (threadIdx.x == 0) {
        atomicAdd(&global_sum[row], block_sum);
        __threadfence();
        atomicAdd(&sync_counters[row], 1);
        while(*(volatile int*)&sync_counters[row] < 2 * blocks_per_row);
    }
    __syncthreads();


    if (threadIdx.x == 0) shared_val = global_sum[row];
    __syncthreads();
    float final_global_sum = shared_val;
    float inv_global_sum = __fdividef(1.0f, final_global_sum);

    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        VecType out_vec;
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val_exp = to_float_large(loaded_vecs[i].val[k]);
            float res = val_exp * inv_global_sum;
            from_float_large(out_vec.val[k], res);
        }
        
        if (threadIdx.x + i * BLOCK_SIZE < elem_per_block / VEC_SIZE) {
            int local_vec_idx = threadIdx.x + i * BLOCK_SIZE;
            int global_offset = block_idx_in_row * elem_per_block + local_vec_idx * VEC_SIZE;
            
            if (global_offset + VEC_SIZE <= cols) {
                 vec_output[local_vec_idx] = out_vec;
            } else {
                 #pragma unroll
                 for (int k = 0; k < VEC_SIZE; ++k) {
                     if (global_offset + k < cols) {
                         output[row * cols + global_offset + k] = out_vec.val[k];
                     }
                 }
            }
        }
    }
}   
template<typename scalar_t, int VEC_SIZE, int BLOCK_SIZE, int VEC_NUM_PER_THREAD>
__global__ void softmax_fused_kernel_block_align(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    float* __restrict__ global_max, 
    float* __restrict__ global_sum, 
    int* __restrict__ sync_counters, 
    int rows,
    int cols,
    int elem_per_block 
) {
    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_val;

    int blocks_per_row = cols / elem_per_block;
    int bid = blockIdx.x;
    int row = bid / blocks_per_row;
    int block_idx_in_row = bid % blocks_per_row;

    if (row >= rows) return;

    using VecType = AlignedVector<scalar_t, VEC_SIZE>;
    
    int block_offset = row * cols + block_idx_in_row * elem_per_block;
    const VecType* vec_input = reinterpret_cast<const VecType*>(input + block_offset);
    VecType* vec_output = reinterpret_cast<VecType*>(output + block_offset);


    VecType loaded_vecs[VEC_NUM_PER_THREAD];
    #pragma unroll 
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        if (threadIdx.x + i * BLOCK_SIZE < elem_per_block / VEC_SIZE) {
             loaded_vecs[i] = vec_input[threadIdx.x + i * BLOCK_SIZE];
        }
    }

    float local_max = -FLT_MAX;
    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val = to_float_large(loaded_vecs[i].val[k]);
            local_max = fmaxf(local_max, val);
        }
    }

    float block_max = BlockReduce(temp_storage).Reduce(local_max, hipcub::Max());
    

    if (threadIdx.x == 0) {
        atomicMax(&global_max[row], block_max);
        __threadfence(); 
        
        atomicAdd(&sync_counters[row], 1);
        
        while(atomicAdd(&sync_counters[row], 0) < blocks_per_row);
    }
    __syncthreads(); 


    if (threadIdx.x == 0) shared_val = global_max[row];
    __syncthreads();
    float final_global_max = shared_val;

    float local_sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val = to_float_large(loaded_vecs[i].val[k]);
            float res = __expf(val - final_global_max);
            local_sum += res;
            from_float_large(loaded_vecs[i].val[k], res);
        }
    }
    
    float block_sum = BlockReduce(temp_storage).Sum(local_sum);

    if (threadIdx.x == 0) {
      
        atomicAdd(&global_sum[row], block_sum);

     
        __threadfence();

   
        atomicAdd(&sync_counters[row], 1);

  
        while(atomicAdd(&sync_counters[row], 0) < 2 * blocks_per_row);
    }
    __syncthreads();


    if (threadIdx.x == 0) shared_val = global_sum[row];
    __syncthreads();
    float final_global_sum = shared_val;
    float inv_global_sum = __fdividef(1.0f, final_global_sum);
    #pragma unroll
    for(int i = 0; i < VEC_NUM_PER_THREAD; ++i) {
        VecType out_vec;
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; ++k) {
            float val_exp = to_float_large(loaded_vecs[i].val[k]);
            float res = val_exp * inv_global_sum;
            from_float_large(out_vec.val[k], res);
        }
        
        if (threadIdx.x + i * BLOCK_SIZE < elem_per_block / VEC_SIZE) {
            vec_output[threadIdx.x + i * BLOCK_SIZE] = out_vec;
        }
    }
}
    template<typename scalar_t, int VEC, int BLOCK_SIZE>
    __global__ void softmax_opt_kernel(
        scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        int rows,
        int cols
    ) {
        int row = blockIdx.x;
        if (row >= rows) return;

        using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;

        int offset = row * cols;
        scalar_t* row_input_ptr = input + offset;
        scalar_t* row_output_ptr = output + offset;

        const VecType* vec_input = reinterpret_cast<const VecType*>(row_input_ptr);
        VecType* vec_output = reinterpret_cast<VecType*>(row_output_ptr);

        extern __shared__ float smem[];
        float* shared_buf = smem;
        int num_vecs = cols / VEC;
        int tid = threadIdx.x;

        float local_max = -FLT_MAX;
        VecType vec_reg; 
        if (tid < num_vecs) {
            vec_reg = vec_input[tid]; 
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                local_max = fmaxf(local_max, to_float(vec_reg.val[k])); 
            }
        }
        float global_max = blockReduceMax(local_max, shared_buf, BLOCK_SIZE);
        float local_sum = 0.0f;
        float cached_exp[VEC]; 

        if (tid < num_vecs) {
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float val = to_float(vec_reg.val[k]);
                float res = __expf(val - global_max);
                cached_exp[k] = res; 
                local_sum += res;
            }
        }
        
        float global_sum = blockReduceSum(local_sum, shared_buf, BLOCK_SIZE);
        float inv_sum = 1.0f / global_sum;
        if (tid < num_vecs) {
            VecType out_vec;
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                from_float(out_vec.val[k],cached_exp[k] * inv_sum);
            }
            vec_output[tid] = out_vec;
        }
    }

    template<typename T>
    void launch_softmax_opt_dispatch(T* input, T* output, int rows, int cols, cudaStream_t stream) {
        
        size_t smem_size = 32 * sizeof(float);
        if (cols <= 128) {
            softmax_opt_kernel<T, 1, 128><<<rows, 128, smem_size, stream>>>(input, output, rows, cols);
        }
        else if (cols <= 1024) {
            softmax_opt_kernel<T, 8, 128><<<rows, 128, smem_size, stream>>>(input, output, rows, cols);
        }
        else if (cols <= 2048) {
            softmax_opt_kernel<T, 8, 256><<<rows, 256, smem_size, stream>>>(input, output, rows, cols);
        }
        else if (cols <= 4096) {
            softmax_opt_kernel<T, 8, 512><<<rows, 512, smem_size, stream>>>(input, output, rows, cols);
        }
        else if (cols <= 8192) {
            softmax_opt_kernel<T, 8, 1024><<<rows, 1024, smem_size, stream>>>(input, output, rows, cols);
        }
        else {
            softmax_opt_kernel<T, 16, 1024><<<rows, 1024, smem_size, stream>>>(input, output, rows, cols);
        }
    }
template<typename T,int VEC_SIZE,int BLOCK_SIZE,int VEC_NUM_PER_THREAD,int elem_per_block>
void launch_softmax_lager_dispatch_opt(
    T* input, T* output, 
    float* global_max, 
    float* global_sum, 
    int* sync_counters, 
    int rows, int cols, cudaStream_t stream
) {
   
    int blocks_per_row = (cols + elem_per_block - 1) / elem_per_block;
    int total_blocks = rows * blocks_per_row;
    bool is_block_align = (cols % elem_per_block == 0);
    if(is_block_align){
          softmax_fused_kernel_block_align<T, VEC_SIZE, BLOCK_SIZE, VEC_NUM_PER_THREAD><<<total_blocks, BLOCK_SIZE, 0, stream>>>(
                output, input, global_max, global_sum, sync_counters, rows, cols, elem_per_block
        );
    }else{
        softmax_fused_kernel<T, VEC_SIZE, BLOCK_SIZE, VEC_NUM_PER_THREAD><<<total_blocks, BLOCK_SIZE, 0, stream>>>(
                output, input, global_max, global_sum, sync_counters, rows, cols, elem_per_block
        );
    }
  
}

template<typename T>
void launch_softmax_large_dispatch(
    T* input, T* output,
    float* global_max,
    float* global_sum,
    int* sync_counters,
    int rows, int cols, cudaStream_t stream
) {
    constexpr int VEC_SIZE = (sizeof(T) == 4) ? 4 : 8;
    constexpr int BLOCK_SIZE = 256;
    constexpr int VEC_NUM_PER_THREAD = 8;
    constexpr int elem_per_block = BLOCK_SIZE * VEC_SIZE * VEC_NUM_PER_THREAD;

    launch_softmax_lager_dispatch_opt<
        T, VEC_SIZE, BLOCK_SIZE, VEC_NUM_PER_THREAD, elem_per_block>(
            input,
            output,
            global_max,
            global_sum,
            sync_counters,
            rows, cols, stream);
}

Tensor fuse_softmax_forward_autograd(const Tensor &input, const int64_t dim, const bool half_to_float, const bool training) {
    if(training) {
        auto result = fuseSoftmaxFunction::apply(input, dim, half_to_float);
        return result[0];
    } else {
        int64_t dim_ = at::maybe_wrap_dim(dim, input.dim());
        if (dim_ != input.dim() - 1) {
             return fuse_softmax_forward(input, dim, half_to_float);
        }
        Tensor input_ = input.is_contiguous() ? input : input.contiguous();
        int cols = input_.size(dim_);
        int rows = input_.numel() / cols;

        bool is_supported_type = (input_.scalar_type() == at::ScalarType::Half || 
                                  input_.scalar_type() == at::ScalarType::BFloat16 ||
                                  input_.scalar_type() == at::ScalarType::Float);
        
  
        int element_size = input_.element_size(); 
        int vec_size_conf = (element_size == 4) ? 4 : 8;

        if (cols > 16384 && cols % vec_size_conf == 0 && is_supported_type) {
             auto output = at::empty_like(input_);
    
             auto stream = at::cuda::getCurrentCUDAStream().stream();

             auto options = input_.options().dtype(at::kFloat); 
             auto int_options = input_.options().dtype(at::kInt); 
             
           
             auto global_max = at::full({rows}, -std::numeric_limits<float>::infinity(), options);
             auto global_sum = at::zeros({rows}, options);
             auto sync_counters = at::zeros({rows}, int_options);
             
             float* d_max = global_max.data_ptr<float>();
             float* d_sum = global_sum.data_ptr<float>();
             int* d_counters = sync_counters.data_ptr<int>();
             
             if (input_.scalar_type() == at::ScalarType::BFloat16) {
                 launch_softmax_large_dispatch<uint16_t>(
                     static_cast<uint16_t*>(input_.data_ptr()),
                     static_cast<uint16_t*>(output.data_ptr()),
                     d_max,
                     d_sum,
                     d_counters,
                     rows, cols, stream);
             } else {
                 AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input_.scalar_type(), "softmax_hip_opt_large", [&] {
                     launch_softmax_large_dispatch<scalar_t>(
                         static_cast<scalar_t*>(input_.data_ptr()),
                         static_cast<scalar_t*>(output.data_ptr()),
                         d_max,
                         d_sum,
                         d_counters,
                         rows, cols, stream);
                 });
             }
             return output;
        }else if(cols <= 16384 && is_supported_type && cols % 8 == 0){
             auto output = at::empty_like(input_);
             auto stream = at::cuda::getCurrentCUDAStream().stream();

             if (input_.scalar_type() == at::ScalarType::BFloat16) {
                 launch_softmax_opt_dispatch<uint16_t>(
                    static_cast<uint16_t*>(input_.data_ptr()),
                    static_cast<uint16_t*>(output.data_ptr()),
                    rows,
                    cols,
                    stream);
             } else {
                 AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input_.scalar_type(), "softmax_hip_opt_small", [&] {
                     launch_softmax_opt_dispatch<scalar_t>(
                        static_cast<scalar_t*>(input_.data_ptr()),
                        static_cast<scalar_t*>(output.data_ptr()),
                        rows,
                        cols,
                        stream);
                 });
             }
             return output;                       
        } 
        else {
            return fuse_softmax_forward(input, dim, half_to_float);
        }
    }
}

Tensor fuse_logsoftmax_forward_autograd(const Tensor &input, const int64_t dim, const bool half_to_float, const bool training) {
    if(training) {
        auto result = fuseLogSoftmaxFunction::apply(input, dim, half_to_float);
        return result[0];
    } else {
        return fuse_log_softmax_forward(input, dim, half_to_float);
    }
}
  
} // namespace native
} // namespace at
