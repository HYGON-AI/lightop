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
            const float* input,        // [B, N, D]
            const int64_t* lengths,    // [B]
            float* output,             // [B, N, Dout]
            const int B,
            const int N,
            const int D,
            const int Dout
        ) {
            int b = blockIdx.y; 
            if (b >= B) return;

            int64_t len = lengths[b];
            const float eps = 1e-6f;

            for (int n = threadIdx.x + blockIdx.x * blockDim.x;
                n < N;
                n += blockDim.x * gridDim.x) {

                if (n >= len) continue;

                const int X=0, Y=1, Z=2, W=3, L=4, H=5, YAW=6;

                const float* in_row = input + b*N*D + n*D;
                float* out_row = output + b*N*Dout + n*Dout;

                out_row[0] = in_row[X];
                out_row[1] = in_row[Y];
                out_row[2] = in_row[Z];

                out_row[3] = logf(fmaxf(in_row[W], eps));
                out_row[4] = logf(fmaxf(in_row[L], eps));
                out_row[5] = logf(fmaxf(in_row[H], eps));

                out_row[6] = sinf(in_row[YAW]);
                out_row[7] = cosf(in_row[YAW]);

                for (int i = 7; i < D; ++i) {
                    out_row[8 + i - 7] = in_row[i];
                }
            }
        }

        torch::Tensor encode_reg_t(
            const torch::Tensor& input,   // [B, N, D]
            const torch::Tensor& lengths  // [B]
        ) {
            int B = input.size(0);
            int N = input.size(1);
            int D = input.size(2);
            std::vector<int> dims = {3, 3, 2, std::max(0, D-7)};
            int Dout = std::accumulate(dims.begin(), dims.end(), 0);

            auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
            torch::Tensor output = torch::empty({B, N, Dout}, options);

            int threads = 256;
            int blocks_x = (N + threads - 1) / threads;
            dim3 block(threads);
            dim3 grid(blocks_x, B);

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            encode_reg_target_kernel<<<grid, block, 0, stream>>>(
                input.data_ptr<float>(),
                lengths.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                B, N, D, Dout
            );

            return output;
        }

    } 
}
