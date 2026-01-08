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

template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1); 
    u += 0x8000; 
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


Tensor fuse_softmax_forward_autograd(const Tensor &input, const int64_t dim, const bool half_to_float, const bool training) {
    if(training) {
        auto result = fuseSoftmaxFunction::apply(input, dim, half_to_float);
        return result[0];
    } else {
        bool is_fp16_bf16 = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16);
        auto output = at::empty_like(input);
        if(input.size(1) <= 16384 && is_fp16_bf16){
             auto stream = at::cuda::getCurrentCUDAStream().stream();
             if (input.scalar_type() == at::ScalarType::Half) {
                 using scalar_t = at::Half;
                 launch_softmax_opt_dispatch<scalar_t>(
                    static_cast<scalar_t*>(input.data_ptr()),
                    static_cast<scalar_t*>(output.data_ptr()),
                    static_cast<int>(input.size(0)),
                    static_cast<int>(input.size(1)),
                    stream
                 );
             } else if (input.scalar_type() == at::ScalarType::BFloat16) {
                 using scalar_t = uint16_t;
                 launch_softmax_opt_dispatch<scalar_t>(
                    static_cast<scalar_t*>(input.data_ptr()),
                    static_cast<scalar_t*>(output.data_ptr()),
                    static_cast<int>(input.size(0)),
                    static_cast<int>(input.size(1)),
                    stream
                 );
             } else {
                 TORCH_CHECK(false, "softmax_opt_autograd_inference: Only Half and BFloat16 are supported.");
             }
            return output;                       
            
        }else{
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