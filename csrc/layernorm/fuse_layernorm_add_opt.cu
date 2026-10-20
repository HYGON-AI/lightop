// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <THC/THCDeviceUtils.cuh>
#include <torch/cuda.h>

#include <cmath>
#include <cstdint>

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {
namespace {

template <typename T>
__device__ __forceinline__ T warp_sum(T x) {
#pragma unroll
  for (int offset = C10_WARP_SIZE / 2; offset; offset >>= 1) {
    x += WARP_SHFL_DOWN(x, offset);
  }
  return x;
}

template <typename T, int BLOCK>
__device__ __forceinline__ void block_sum2(T& sum, T& sum_sq, T* shared) {
  constexpr int WARPS = BLOCK / C10_WARP_SIZE;
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int warp = threadIdx.x / C10_WARP_SIZE;
  sum = warp_sum(sum);
  sum_sq = warp_sum(sum_sq);
  if (lane == 0) {
    shared[warp] = sum;
    shared[WARPS + warp] = sum_sq;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < WARPS ? shared[lane] : T(0);
    sum_sq = lane < WARPS ? shared[WARPS + lane] : T(0);
    sum = warp_sum(sum);
    sum_sq = warp_sum(sum_sq);
  }
}

// Target-shape path: one thread owns one vector and keeps it in registers
// across the sum/sum_sq reduction. No Welford state is used.
template <typename scalar_t, typename acc_t, int VEC, int BLOCK,
          bool FUSED_ADD, bool APPLY_CLAMP>
__global__ void layer_norm_target_kernel(
    scalar_t* output_or_input, const scalar_t* input_or_null,
    scalar_t* residual_or_null, const scalar_t* weight,
    const scalar_t* bias,
    int hidden_size, acc_t eps, int num_tokens,
    acc_t limit) {
  constexpr int WARPS = BLOCK / C10_WARP_SIZE;
  __shared__ acc_t reduce_shared[WARPS * 2];
  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;
  using VecT = memory::aligned_vector<scalar_t, VEC>;
  const int vecs = hidden_size / VEC;
  const int j = threadIdx.x;

  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    alignas(16) scalar_t values[VEC];
    acc_t sum = 0;
    acc_t sum_sq = 0;
    const int64_t offset =
        (static_cast<int64_t>(token) * vecs + j) * VEC;
    if (j < vecs) {
      if constexpr (FUSED_ADD) {
        alignas(16) scalar_t input_vec[VEC];
        alignas(16) scalar_t residual_vec[VEC];
        *reinterpret_cast<VecT*>(input_vec) =
            *reinterpret_cast<const VecT*>(output_or_input + offset);
        *reinterpret_cast<VecT*>(residual_vec) =
            *reinterpret_cast<const VecT*>(residual_or_null + offset);
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          // Match vLLM fused norm semantics: round add before normalization.
          values[k] = static_cast<scalar_t>(input_vec[k] + residual_vec[k]);
        }
      } else {
        *reinterpret_cast<VecT*>(values) =
            *reinterpret_cast<const VecT*>(input_or_null + offset);
      }
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const acc_t x = static_cast<acc_t>(values[k]);
        sum += x;
        sum_sq += x * x;
      }
    }

    block_sum2<acc_t, BLOCK>(sum, sum_sq, reduce_shared);
    if (threadIdx.x == 0) {
      const acc_t inv_n = acc_t(1) / static_cast<acc_t>(hidden_size);
      s_mean = sum * inv_n;
      acc_t variance = sum_sq * inv_n - s_mean * s_mean;
      variance = variance > acc_t(0) ? variance : acc_t(0);
      s_rstd = c10::cuda::compat::rsqrt(variance + eps);
    }
    __syncthreads();

    if (j < vecs) {
      alignas(16) scalar_t bias_vec[VEC];
      alignas(16) scalar_t weight_vec[VEC];
      alignas(16) scalar_t result_vec[VEC];
      *reinterpret_cast<VecT*>(weight_vec) =
          *reinterpret_cast<const VecT*>(weight + j * VEC);
      *reinterpret_cast<VecT*>(bias_vec) =
          *reinterpret_cast<const VecT*>(bias + j * VEC);
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        acc_t y = (static_cast<acc_t>(values[k]) - s_mean) * s_rstd;
        y *= static_cast<acc_t>(weight_vec[k]);
        y += static_cast<acc_t>(bias_vec[k]);
        if constexpr (APPLY_CLAMP) {
          y = y > limit ? limit : (y < -limit ? -limit : y);
        }
        result_vec[k] = static_cast<scalar_t>(y);
      }
      if constexpr (FUSED_ADD) {
        *reinterpret_cast<VecT*>(residual_or_null + offset) =
            *reinterpret_cast<VecT*>(values);
      }
      *reinterpret_cast<VecT*>(output_or_input + offset) =
          *reinterpret_cast<VecT*>(result_vec);
    }
    // Some blocks process a second token when num_tokens==288 and grid==256.
    __syncthreads();
  }
}

template <typename scalar_t, typename acc_t, int BLOCK,
          bool FUSED_ADD, bool APPLY_CLAMP>
__global__ void layer_norm_generic_kernel(
    scalar_t* output_or_input, const scalar_t* input_or_null,
    scalar_t* residual_or_null, const scalar_t* weight,
    const scalar_t* bias,
    int hidden_size, acc_t eps, int num_tokens,
    acc_t limit) {
  constexpr int WARPS = BLOCK / C10_WARP_SIZE;
  __shared__ acc_t reduce_shared[WARPS * 2];
  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;

  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const int64_t base = static_cast<int64_t>(token) * hidden_size;
    acc_t sum = 0;
    acc_t sum_sq = 0;
    for (int j = threadIdx.x; j < hidden_size; j += BLOCK) {
      scalar_t value;
      if constexpr (FUSED_ADD) {
        value = static_cast<scalar_t>(
            output_or_input[base + j] + residual_or_null[base + j]);
        residual_or_null[base + j] = value;
      } else {
        value = input_or_null[base + j];
      }
      const acc_t x = static_cast<acc_t>(value);
      sum += x;
      sum_sq += x * x;
    }
    block_sum2<acc_t, BLOCK>(sum, sum_sq, reduce_shared);
    if (threadIdx.x == 0) {
      const acc_t inv_n = acc_t(1) / static_cast<acc_t>(hidden_size);
      s_mean = sum * inv_n;
      acc_t variance = sum_sq * inv_n - s_mean * s_mean;
      variance = variance > acc_t(0) ? variance : acc_t(0);
      s_rstd = c10::cuda::compat::rsqrt(variance + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < hidden_size; j += BLOCK) {
      const scalar_t value = FUSED_ADD ? residual_or_null[base + j]
                                       : input_or_null[base + j];
      acc_t y = (static_cast<acc_t>(value) - s_mean) * s_rstd;
      y *= static_cast<acc_t>(weight[j]);
      y += static_cast<acc_t>(bias[j]);
      if constexpr (APPLY_CLAMP) {
        y = y > limit ? limit : (y < -limit ? -limit : y);
      }
      output_or_input[base + j] = static_cast<scalar_t>(y);
    }
    __syncthreads();
  }
}

void check_common(const Tensor& input, const Tensor& weight,
                  const Tensor& bias,
                  double epsilon, double limit) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
              "input must be a contiguous CUDA tensor");
  TORCH_CHECK(input.dim() > 0 && input.size(-1) > 0,
              "input must have a non-empty last dimension");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
              "weight must be a contiguous CUDA tensor");
  TORCH_CHECK(weight.device() == input.device(),
              "input and weight must be on the same device");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(weight.numel() == input.size(-1),
              "weight numel must equal hidden_size");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(),
              "bias must be a contiguous CUDA tensor");
  TORCH_CHECK(bias.device() == input.device(),
              "input and bias must be on the same device");
  TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
              "input and bias must have the same dtype");
  TORCH_CHECK(bias.numel() == input.size(-1),
              "bias numel must equal hidden_size");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
              "epsilon must be finite and non-negative");
  TORCH_CHECK(limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
              "limit must be -1 (disabled) or a finite non-negative value");
}

template <bool FUSED_ADD, bool APPLY_CLAMP>
void launch_layer_norm_opt(Tensor& output_or_input, Tensor* input,
                           Tensor* residual, Tensor& weight, Tensor& bias,
                           double epsilon, double limit) {
  const Tensor& source = FUSED_ADD ? output_or_input : *input;
  check_common(source, weight, bias, epsilon, limit);
  if constexpr (FUSED_ADD) {
    TORCH_CHECK(residual->is_cuda() && residual->is_contiguous() &&
                    residual->device() == source.device() &&
                    residual->scalar_type() == source.scalar_type() &&
                    residual->sizes() == source.sizes(),
                "residual must match input device, dtype, shape, and be contiguous");
    TORCH_CHECK(source.data_ptr() != residual->data_ptr(),
                "input and residual must not alias each other");
  } else {
    TORCH_CHECK(output_or_input.is_cuda() && output_or_input.is_contiguous() &&
                    output_or_input.device() == source.device() &&
                    output_or_input.scalar_type() == source.scalar_type() &&
                    output_or_input.sizes() == source.sizes(),
                "out must match input device, dtype, shape, and be contiguous");
  }

  const int hidden_size = source.size(-1);
  const int num_tokens = source.numel() / hidden_size;
  if (num_tokens == 0) return;
  const int grid = num_tokens == 288 ? 256 : num_tokens;
  const at::cuda::OptionalCUDAGuard guard(device_of(source));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, source.scalar_type(),
      "layer_norm_opt", [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        scalar_t* out_data = output_or_input.data_ptr<scalar_t>();
        const scalar_t* input_data = FUSED_ADD ? nullptr : input->data_ptr<scalar_t>();
        scalar_t* residual_data = FUSED_ADD ? residual->data_ptr<scalar_t>() : nullptr;
        const scalar_t* weight_data = weight.data_ptr<scalar_t>();
        const scalar_t* bias_data = bias.data_ptr<scalar_t>();
        auto aligned16 = [](const void* p) {
          return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
        };
        bool vectorized = sizeof(scalar_t) == 2 && aligned16(out_data) &&
            aligned16(weight_data) && aligned16(bias_data);
        if constexpr (FUSED_ADD) {
          vectorized = vectorized && aligned16(residual_data);
        } else {
          vectorized = vectorized && aligned16(input_data);
        }

        if (vectorized && hidden_size == 1024) {
          layer_norm_target_kernel<scalar_t, acc_t, 8, 128,
                                   FUSED_ADD, APPLY_CLAMP>
              <<<grid, 128, 0, stream>>>(
                  out_data, input_data, residual_data, weight_data, bias_data,
                  hidden_size, static_cast<acc_t>(epsilon), num_tokens,
                  static_cast<acc_t>(limit));
        } else if (vectorized && hidden_size == 1152) {
          layer_norm_target_kernel<scalar_t, acc_t, 8, 256,
                                   FUSED_ADD, APPLY_CLAMP>
              <<<grid, 256, 0, stream>>>(
                  out_data, input_data, residual_data, weight_data, bias_data,
                  hidden_size, static_cast<acc_t>(epsilon), num_tokens,
                  static_cast<acc_t>(limit));
        } else {
          layer_norm_generic_kernel<scalar_t, acc_t, 256,
                                    FUSED_ADD, APPLY_CLAMP>
              <<<grid, 256, 0, stream>>>(
                  out_data, input_data, residual_data, weight_data, bias_data,
                  hidden_size, static_cast<acc_t>(epsilon), num_tokens,
                  static_cast<acc_t>(limit));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void layer_norm_opt(Tensor& out, Tensor& input, Tensor& weight, Tensor& bias,
                    double epsilon, double limit) {
  if (limit < 0.0) {
    launch_layer_norm_opt<false, false>(out, &input, nullptr, weight, bias,
                                        epsilon, limit);
  } else {
    launch_layer_norm_opt<false, true>(out, &input, nullptr, weight, bias,
                                       epsilon, limit);
  }
}

void fused_add_layer_norm_opt(Tensor& input, Tensor& residual, Tensor& weight,
                              Tensor& bias,
                              double epsilon, double limit) {
  if (limit < 0.0) {
    launch_layer_norm_opt<true, false>(input, nullptr, &residual, weight, bias,
                                       epsilon, limit);
  } else {
    launch_layer_norm_opt<true, true>(input, nullptr, &residual, weight, bias,
                                      epsilon, limit);
  }
}

}  // namespace native
}  // namespace at
