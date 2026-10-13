/* coding=utf-8
 * Copyright (c) 2023, SUGON CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

namespace at{
namespace native{

static constexpr float one = 1.0F;

constexpr int VEC_n = 8;

template <typename acc_t>
__device__ acc_t silu_mul(acc_t x, acc_t m) {
    return m * x / (one + expf(-x));
}

template <typename scalar_t, typename acc_t, int VEC>
__global__ void FusedBiasSwiGLU_gpu_kernel(int elem_cnt, scalar_t* out, const scalar_t* in, const scalar_t* bias, int dim) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_in[VEC];
    scalar_t v_multiplier[VEC];
    scalar_t v_bias_in[VEC];
    scalar_t v_bias_multiplier[VEC];
    scalar_t v_out[VEC];
    int idx = ((int)blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if(idx < elem_cnt / 2){
        int row_id = idx / (dim / 2);
        int col_id = idx % (dim / 2);
        *(LoadT*)(v_in) = *(LoadT*)(in + row_id * dim + col_id);
        *(LoadT*)(v_bias_in) = *(LoadT*)(bias + col_id);
        *(LoadT*)(v_multiplier) = *(LoadT*)(in + row_id * dim + col_id + dim / 2);
        *(LoadT*)(v_bias_multiplier) = *(LoadT*)(bias + col_id + dim / 2);
        #pragma unroll
        for(int i = 0; i < VEC; i++){
            v_out[i] = static_cast<scalar_t>(silu_mul<acc_t>(static_cast<acc_t>(v_in[i] + v_bias_in[i]), 
                                             static_cast<acc_t>(v_multiplier[i] + v_bias_multiplier[i])));
        }
        *(LoadT*)(out + row_id * dim / 2 + col_id) = *(LoadT*)v_out;
    }
}

template <typename scalar_t>
void host_apply_FusedBiasSwiGLU_gpu(int elem_cnt, scalar_t* out, const scalar_t* in, const scalar_t* bias, int dim) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int ThreadsPerBlock = REDUCE_BLOCK_SIZE;
    int gridsize=(elem_cnt/2/VEC_n-1)/ThreadsPerBlock+1;
    using T_ACC = at::acc_type<scalar_t, true>;
    FusedBiasSwiGLU_gpu_kernel<scalar_t, T_ACC, VEC_n><<<gridsize, ThreadsPerBlock, 0, stream>>>(elem_cnt, out, in, bias, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void FusedBiasSwiGLU_gpu_launch(int elem_cnt, at::Tensor* in, at::Tensor* out, at::Tensor* bias) {
    int dim = bias->numel();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in->scalar_type(), "FusedBiasSwiGLU_forward_gpu",[&] {
            host_apply_FusedBiasSwiGLU_gpu<scalar_t>(elem_cnt,
                                       out->data_ptr<scalar_t>(), in->data_ptr<scalar_t>(), bias->data_ptr<scalar_t>(), dim);
        });
}

template <typename acc_t>
__device__ void silu_mul_backward(acc_t& x_diff, acc_t& m_diff, acc_t dy, acc_t x, acc_t m) {
    m_diff = dy * x / (one + expf(-x));
    acc_t s = one / (one + expf(-x));
    x_diff = dy * m * s * (one + x * (one - s));
}

template <typename scalar_t, typename acc_t, int VEC>
__global__ void FusedBiasSwiGLU_backward_gpu_kernel(int elem_cnt, scalar_t* in_grad, const scalar_t* out_grad, const scalar_t* in, const scalar_t* bias, int dim) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_in_grad[VEC];
    scalar_t v_multiplier_grad[VEC];
    scalar_t v_out_grad[VEC];
    scalar_t v_in[VEC];
    scalar_t v_multiplier[VEC];
    scalar_t v_bias_in[VEC];
    scalar_t v_bias_multiplier[VEC];
    int idx = ((int)blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if(idx < elem_cnt / 2){
        int row_id = idx / (dim / 2);
        int col_id = idx % (dim / 2);
        *(LoadT*)(v_in) = *(LoadT*)(in + row_id * dim + col_id);
        *(LoadT*)(v_bias_in) = *(LoadT*)(bias + col_id);
        *(LoadT*)(v_multiplier) = *(LoadT*)(in + row_id * dim + col_id + dim / 2);
        *(LoadT*)(v_bias_multiplier) = *(LoadT*)(bias + col_id + dim / 2);
        *(LoadT*)(v_out_grad) = *(LoadT*)(out_grad + row_id * dim / 2 + col_id);
        #pragma unroll
        for(int i = 0; i < VEC; i++){
            acc_t x_diff;
            acc_t m_diff;
            silu_mul_backward<acc_t>(x_diff, m_diff, static_cast<acc_t>(v_out_grad[i]), static_cast<acc_t>(v_in[i] + v_bias_in[i]), 
                                          static_cast<acc_t>(v_multiplier[i] + v_bias_multiplier[i]));
            v_in_grad[i] = static_cast<scalar_t>(x_diff);
            v_multiplier_grad[i] = static_cast<scalar_t>(m_diff);
        }
        *(LoadT*)(in_grad + row_id * dim + col_id) = *(LoadT*)v_in_grad;
        *(LoadT*)(in_grad + row_id * dim + col_id + dim / 2) = *(LoadT*)v_multiplier_grad;
    }
}

template <typename scalar_t>
void host_apply_FusedBiasSwiGLU_backward_gpu(cudaStream_t stream, int elem_cnt, scalar_t* in_grad, scalar_t* bias_grad, const scalar_t* out_grad, const scalar_t* in, const scalar_t* bias, int dim) {
    const int ThreadsPerBlock = REDUCE_BLOCK_SIZE;
    int gridsize=(elem_cnt/2/VEC_n-1)/ThreadsPerBlock+1;
    int M = elem_cnt / dim;
    int B =(dim - 1) / kColwiseReduceTileSize + 1;
    using T_ACC = at::acc_type<scalar_t, true>;
    FusedBiasSwiGLU_backward_gpu_kernel<scalar_t, T_ACC, VEC_n><<<gridsize, ThreadsPerBlock, 0, stream>>>(elem_cnt, in_grad, out_grad, in, bias, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bias_reduce_sum(bias_grad, in_grad, elem_cnt, dim);
}

void FusedBiasSwiGLU_backward_gpu_launch(int elem_cnt, int dim, at::Tensor* out_grad, at::Tensor* in, at::Tensor* bias, at::Tensor* in_grad, at::Tensor* bias_grad) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in->scalar_type(), "FusedBiasSwiGLU_backward_gpu",
        [&] {
            host_apply_FusedBiasSwiGLU_backward_gpu<scalar_t>(stream, elem_cnt, in_grad->data_ptr<scalar_t>(), bias_grad->data_ptr<scalar_t>(),
                out_grad->data_ptr<scalar_t>(), in->data_ptr<scalar_t>(), bias->data_ptr<scalar_t>(), dim);});
}

at::Tensor FusedBiasSwiGLU_forward_gpu(at::Tensor in, at::Tensor bias) {
    int elem_cnt = in.numel();
    at::Tensor out =
        at::empty({elem_cnt / 2},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));    
               
    FusedBiasSwiGLU_gpu_launch(elem_cnt, &in, &out, &bias);
    return out;
}

std::vector<at::Tensor> FusedBiasSwiGLU_backward_gpu(at::Tensor out_grad, at::Tensor in, at::Tensor bias) {
    int elem_cnt = in.numel();
    int dim = bias.numel();
    at::Tensor in_grad =
        at::empty({elem_cnt},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));
    at::Tensor bias_grad =
        at::empty({dim},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));
    FusedBiasSwiGLU_backward_gpu_launch(elem_cnt, dim, &out_grad, &in, &bias, &in_grad, &bias_grad);
    return {in_grad, bias_grad};
}

}//namespace native
}//namespace at
