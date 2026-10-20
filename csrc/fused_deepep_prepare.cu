// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <tuple>

namespace at {
namespace native {
namespace {

__global__ void fused_deepep_prepare_kernel(
    const __hip_bfloat16* hidden,
    const float* smooth,
    const int32_t* ids,
    __hip_bfloat16* hidden_out,
    int64_t* ids_out,
    int64_t hidden_numel,
    int64_t hidden_size,
    int64_t ids_numel,
    float min_limit,
    float max_limit) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < hidden_numel) {
    float value = __bfloat162float(hidden[index]);
    // Comparisons preserve NaNs.  fmin/fmax would replace them with a bound.
    if (value > max_limit) {
      value = max_limit;
    } else if (value < min_limit) {
      value = min_limit;
    }
    // Preserve the old three operator boundaries: BF16 clamp output, FP32 to
    // BF16 scale cast, then BF16 multiplication rounded to BF16.
    const __hip_bfloat16 clamped = __float2bfloat16(value);
    const __hip_bfloat16 scale =
        __float2bfloat16(smooth[index % hidden_size]);
    hidden_out[index] = __float2bfloat16(
        __bfloat162float(clamped) * __bfloat162float(scale));
  }
  if (index < ids_numel) {
    ids_out[index] = static_cast<int64_t>(ids[index]);
  }
}

}  // namespace

std::tuple<Tensor, Tensor> fused_deepep_prepare(
    const Tensor& hidden,
    const Tensor& smooth,
    const Tensor& ids,
    double min_limit,
    double max_limit) {
  TORCH_CHECK(hidden.is_cuda() && smooth.is_cuda() && ids.is_cuda(),
              "fused_deepep_prepare expects CUDA/HIP tensors");
  TORCH_CHECK(hidden.scalar_type() == kBFloat16,
              "hidden must be bfloat16");
  TORCH_CHECK(smooth.scalar_type() == kFloat, "smooth must be float32");
  TORCH_CHECK(ids.scalar_type() == kInt, "ids must be int32");
  TORCH_CHECK(hidden.dim() == 2 && smooth.dim() == 1 && ids.dim() == 2,
              "expected hidden[M,H], smooth[H], ids[M,K]");
  TORCH_CHECK(hidden.is_contiguous() && smooth.is_contiguous() &&
                  ids.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(hidden.size(0) == ids.size(0), "row count mismatch");
  TORCH_CHECK(hidden.size(1) == smooth.numel(), "hidden size mismatch");
  TORCH_CHECK(ids.size(1) > 0 && ids.size(1) <= 32,
              "top-k width must be in [1, 32]");
  TORCH_CHECK(min_limit <= max_limit, "invalid clamp limits");

  Tensor hidden_out = torch::empty_like(hidden);
  Tensor ids_out = torch::empty_like(ids, ids.options().dtype(kLong));
  constexpr int threads = 256;
  const int64_t hidden_numel = hidden.numel();
  const int blocks = static_cast<int>((hidden_numel + threads - 1) / threads);
  fused_deepep_prepare_kernel<<<
      blocks, threads, 0, cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __hip_bfloat16*>(hidden.data_ptr()),
      smooth.data_ptr<float>(),
      ids.data_ptr<int32_t>(),
      reinterpret_cast<__hip_bfloat16*>(hidden_out.data_ptr()),
      ids_out.data_ptr<int64_t>(),
      hidden_numel,
      hidden.size(1),
      ids.numel(),
      static_cast<float>(min_limit),
      static_cast<float>(max_limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hidden_out, ids_out};
}

}  // namespace native
}  // namespace at
