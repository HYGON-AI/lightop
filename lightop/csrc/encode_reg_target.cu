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
        __global__ void encode_reg_target_kernel(
            const float* input,      // [N, D]
            const int64_t len,       // 实际行数
            float* output,           // [N, Dout]
            const int N,
            const int D,
            const int Dout
        ) {
            const float eps = 1e-6f;
            const int X=0, Y=1, Z=2, W=3, L=4, H=5, YAW=6;

            for (int n = threadIdx.x + blockIdx.x * blockDim.x;
                n < N;
                n += blockDim.x * gridDim.x) {
                if (n >= len) continue;

                const float* in_row = input + n * D;
                float* out_row = output + n * Dout;

                // 1. copy X,Y,Z
                out_row[0] = in_row[X];
                out_row[1] = in_row[Y];
                out_row[2] = in_row[Z];

                // 2. log(W,L,H)
                out_row[3] = logf(fmaxf(in_row[W], eps));
                out_row[4] = logf(fmaxf(in_row[L], eps));
                out_row[5] = logf(fmaxf(in_row[H], eps));

                // 3. sin/cos(YAW)
                out_row[6] = sinf(in_row[YAW]);
                out_row[7] = cosf(in_row[YAW]);

                // 4. rest
                for (int i = 7; i < D; ++i) {
                    out_row[8 + i - 7] = in_row[i];
                }
            }
        }

        torch::Tensor encode_reg_t(
            const torch::Tensor& input,   // [N, D]
            const int64_t len             // 实际行数
        ) {
            int N = input.size(0);
            int D = input.size(1);
            std::vector<int> dims = {3, 3, 2, std::max(0, D-7)};
            int Dout = std::accumulate(dims.begin(), dims.end(), 0);

            auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
            torch::Tensor output = torch::empty({N, Dout}, options);

            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            dim3 block(threads);
            dim3 grid(blocks);

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            encode_reg_target_kernel<<<grid, block, 0, stream>>>(
                input.data_ptr<float>(),
                len,
                output.data_ptr<float>(),
                N,
                D,
                Dout
            );

            return output;
        }

    } 
} 
