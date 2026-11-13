// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <cmath>

namespace at {
namespace native {
namespace {

__device__ __forceinline__ float pool_pair(
    float previous, float current, float previous_score, float score) {
#pragma clang fp contract(off)
  const float maximum = fmaxf(previous_score, score);
  const float a = expf(previous_score - maximum);
  const float b = expf(score - maximum);
  const float denominator = a + b;
  // Keep the reference's two products and sum before its BF16 cast.
  return static_cast<float>(at::BFloat16(previous * (a / denominator) + current * (b / denominator)));
}

template <typename Weight>
__global__ void dsv41_pool_pairs_norm_kernel(
    const float* __restrict__ previous, const float* __restrict__ current,
    const float* __restrict__ previous_score, const float* __restrict__ score,
    const Weight* __restrict__ weight, at::BFloat16* __restrict__ output,
    int64_t previous_stride, int64_t current_stride,
    int64_t previous_score_stride, int64_t score_stride, float eps) {
#pragma clang fp contract(off)
  const int64_t row = blockIdx.x;
  const int column = threadIdx.x * 2;
  const float x = pool_pair(previous[row * previous_stride + column],
                            current[row * current_stride + column],
                            previous_score[row * previous_score_stride + column],
                            score[row * score_stride + column]);
  const float y = pool_pair(previous[row * previous_stride + column + 1],
                            current[row * current_stride + column + 1],
                            previous_score[row * previous_score_stride + column + 1],
                            score[row * score_stride + column + 1]);
  float sum = x * x + y * y;
#pragma unroll
  for (int delta = 8; delta; delta >>= 1) sum += __shfl_xor(sum, delta, 16);
  sum += __shfl_xor(sum, 16, 64);
  sum += __shfl_xor(sum, 32, 64);
  __shared__ float wave_sums[4];
  __shared__ float inverse_rms;
  if (threadIdx.x % 64 == 0) wave_sums[threadIdx.x / 64] = sum;
  __syncthreads();
  if (column == 0) {
    // Match the reference's sizePerThread=2 / wave64 / four-wave reduction.
    const float total = (wave_sums[0] + wave_sums[2]) + (wave_sums[1] + wave_sums[3]);
    inverse_rms = rsqrtf(total * (1.f / 512.f) + eps);
  }
  __syncthreads();
  output[row * 512 + column] = at::BFloat16((x * inverse_rms) * static_cast<float>(weight[column]));
  output[row * 512 + column + 1] =
      at::BFloat16((y * inverse_rms) * static_cast<float>(weight[column + 1]));
}

}  // namespace

at::Tensor dsv41_pool_pairs_norm(
    const at::Tensor& previous, const at::Tensor& current,
    const at::Tensor& previous_score, const at::Tensor& score,
    const at::Tensor& weight, double eps) {
  for (const auto* tensor : {&previous, &current, &previous_score, &score}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == current.device() &&
                    tensor->scalar_type() == at::kFloat && tensor->dim() == 2 &&
                    tensor->size(0) == current.size(0) && tensor->size(1) == 512 &&
                    tensor->stride(1) == 1 && tensor->stride(0) >= 512 && !tensor->is_neg(),
                "pair inputs must be FP32 [N,512] on one GPU with contiguous columns");
  }
  TORCH_CHECK(weight.is_cuda() && weight.device() == current.device() && weight.dim() == 1 &&
                  weight.numel() == 512 && weight.is_contiguous() && !weight.is_neg() &&
                  (weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16),
              "norm weight must be contiguous FP32/BF16 [512] on the input GPU");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(static_cast<float>(eps)) &&
                  static_cast<float>(eps) > 0, "eps must be positive finite FP32");
  TORCH_CHECK(current.size(0) <= 2147483647, "too many token rows");
  const c10::cuda::CUDAGuard guard(current.device());
  auto output = at::empty({current.size(0), 512}, current.options().dtype(at::kBFloat16));
  if (current.size(0) == 0) return output;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (weight.scalar_type() == at::kFloat) {
    dsv41_pool_pairs_norm_kernel<float><<<current.size(0), 256, 0, stream>>>(
        previous.data_ptr<float>(), current.data_ptr<float>(), previous_score.data_ptr<float>(),
        score.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<at::BFloat16>(),
        previous.stride(0), current.stride(0), previous_score.stride(0), score.stride(0), eps);
  } else {
    dsv41_pool_pairs_norm_kernel<at::BFloat16><<<current.size(0), 256, 0, stream>>>(
        previous.data_ptr<float>(), current.data_ptr<float>(), previous_score.data_ptr<float>(),
        score.data_ptr<float>(), weight.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(),
        previous.stride(0), current.stride(0), previous_score.stride(0), score.stride(0), eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace native
}  // namespace at
