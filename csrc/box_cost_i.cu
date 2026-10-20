// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <random>
#include "hip/hip_runtime.h"
#include <hip/hip_hcc.h>
#include <torch/extension.h>
#include <string>
#include <sstream>
#include <ATen/dtk_macros.h>
#include "hip/hip_runtime.h"
#include <c10/util/BFloat16.h>
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include <ATen/hip/HIPContext.h>
#include <hip/hip_hcc.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>
#include <filesystem>
#include <torch/all.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"

namespace at {
    namespace native {

        __global__ __launch_bounds__(256)void box_cost_kernel(
                const float *pred,
                const float *target,
                const float *inst_weights,
                const float *dim_weights,
                float *cost_matrix,
                const float cost_weight,
                const int N, const int M, const int D
        ) {

            // j 对应 N 维度, k 对应 M 维度
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            const int k = blockIdx.y * blockDim.y + threadIdx.y;
            
            extern __shared__ float s_dim_weights[];

            int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
            int block_size = blockDim.x * blockDim.y;

            for (int i = tid_in_block; i < D; i += block_size) {
                s_dim_weights[i] = dim_weights[i];
            }
            __syncthreads();

            if (j >= N || k >= M) {
                return;
            }
            float cost = 0.0f;
            const float *pred_base = pred + j * D;
            const float *target_base = target + k * D;
            const float *inst_w_base = inst_weights + k * D;
            // int D_vec = D / 2;
            // for (int i = 0; i < D_vec; ++i) {
            //     int d_offset = i * 2;
            //     float2 p_vec = *((const float2 *) (pred_base + d_offset));
            //     float2 t_vec = *((const float2 *) (target_base + d_offset));
            //     float2 iw_vec = *((const float2 *) (inst_w_base + d_offset));
            //     float2 dw_vec = ((float2 *) s_dim_weights)[i];

            //     cost += fabsf(p_vec.x - t_vec.x) * iw_vec.x * dw_vec.x;
            //     cost += fabsf(p_vec.y - t_vec.y) * iw_vec.y * dw_vec.y;
            // }

            // // 处理剩余的维度
            // for (int d = D_vec * 2; d < D; ++d) {
            //     float diff = pred_base[d] - target_base[d];
            //     cost += fabsf(diff) * inst_w_base[d] * s_dim_weights[d];
            // }
            for (int d = 0; d < D; ++d) {
                float diff = pred_base[d] - target_base[d];
                cost += fabsf(diff) * inst_w_base[d] * s_dim_weights[d];
            }
            cost_matrix[j * M + k] = cost * cost_weight;
        }

       torch::Tensor box_cost(
                const torch::Tensor &pred,        
                const torch::Tensor &target,      
                const torch::Tensor &inst_weights,  
                const torch::Tensor &dim_weights,   
                const float cost_weight) {      

            auto ensure_fp32_contiguous = [](const torch::Tensor &t) {
                if (t.dtype() == torch::kFloat32 && t.is_contiguous()) {
                    return t; 
                } else {
                    return t.to(torch::kFloat32).contiguous(); 
                }
            };
            auto pred_c = ensure_fp32_contiguous(pred);
            auto target_c = ensure_fp32_contiguous(target);
            auto inst_weights_c = ensure_fp32_contiguous(inst_weights);
            auto dim_weights_c = ensure_fp32_contiguous(dim_weights);

            const int N = pred_c.size(0);
            const int D = pred_c.size(1);
            const int M = target_c.size(0);
            
            auto options = torch::TensorOptions()
                    .device(pred_c.device())
                    .dtype(torch::kFloat32);
            torch::Tensor cost_matrix = torch::empty({N, M}, options);
            
            const dim3 blockDim(16, 16, 1);
            
            const dim3 gridDim(
                    (N + blockDim.x - 1) / blockDim.x,
                    (M + blockDim.y - 1) / blockDim.y,
                    1 
            );
            
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            const int seme_size = D * sizeof(float);
            box_cost_kernel<<<gridDim, blockDim, seme_size, stream>>>(
                    pred_c.data_ptr<float>(),
                    target_c.data_ptr<float>(),
                    inst_weights_c.data_ptr<float>(),
                    dim_weights_c.data_ptr<float>(),
                    cost_matrix.data_ptr<float>(),
                    cost_weight,
                    N, M, D 
            );
            return cost_matrix;
        }
    }
}