// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "reduce.h"

namespace at {
namespace native {

#define LIGHTOP_DISPATCH_CASE_FLOATING_TYPES(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define LIGHTOP_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                    \
      TYPE, NAME, LIGHTOP_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
#define LIGHTOP_LDG(arg) __ldg(arg)
#else
#define LIGHTOP_LDG(arg) *(arg)
#endif

static inline __device__ int8_t swiglu_clamp_float_to_int8_rn(float x) {
#ifdef USE_ROCM
  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  float dst = std::nearbyint(x);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename scalar_t, bool use_gelu>
__device__ __forceinline__ scalar_t swiglu_clamp_value(
    scalar_t gate, scalar_t up, float limit) {
  float gate_f = static_cast<float>(gate);
  float up_f = static_cast<float>(up);

  if constexpr (use_gelu) {
    // GELU(BF16) -> multiply(BF16) -> clamp(BF16) -> dynamic INT8 quant.
    const scalar_t gate_value = static_cast<scalar_t>(gate_f);
    const scalar_t up_value = static_cast<scalar_t>(up_f);
    const float f = static_cast<float>(gate_value);
    constexpr float ALPHA = M_SQRT1_2;
    const scalar_t activated =
        static_cast<scalar_t>(f * 0.5f * (1.0f + std::erf(f * ALPHA)));
    const scalar_t multiplied =
        static_cast<scalar_t>(activated * up_value);
    float value = static_cast<float>(multiplied);
    value = value < -limit ? -limit : (value > limit ? limit : value);
    return static_cast<scalar_t>(value);
  }
  // clamp(BF16) -> SiLU(BF16) -> multiply(BF16) -> dynamic INT8 quant.
  gate_f = gate_f > limit ? limit : gate_f;
  up_f = up_f < -limit ? -limit : (up_f > limit ? limit : up_f);
  const scalar_t gate_clamped = static_cast<scalar_t>(gate_f);
  const scalar_t up_clamped = static_cast<scalar_t>(up_f);
  constexpr float log2e = 1.44269504088896340736f;
#ifdef USE_ROCM
  const float exp_value =
      __builtin_amdgcn_exp2f(-static_cast<float>(gate_clamped) * log2e);
#else
  const float exp_value = __expf(-static_cast<float>(gate_clamped));
#endif
  const scalar_t activated = static_cast<scalar_t>(
      static_cast<float>(gate_clamped) / (1.0f + exp_value));
  return static_cast<scalar_t>(activated * up_clamped);
}

template <typename scalar_t, bool apply_smooth>
__device__ __forceinline__ scalar_t swiglu_apply_smooth(
    scalar_t value, const float* __restrict__ smooth_scale, int idx) {
  if constexpr (apply_smooth) {
    const scalar_t scale_value = static_cast<scalar_t>(smooth_scale[idx]);
    return static_cast<scalar_t>(
        static_cast<float>(value) * static_cast<float>(scale_value));
  }
  return value;
}

template <typename scalar_t,
          int vec,
          int block_size,
          bool ep,
          bool use_gelu,
          bool apply_smooth>
__global__ void swiglu_clamp_quant_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* __restrict__ mask_m) {
  int64_t token_idx = blockIdx.x;
  int64_t token_end = token_idx + 1;
  int64_t token_stride = 1;
  if constexpr (ep) {
    const int64_t expert_idx = blockIdx.x;
    int64_t valid_m = mask_m[expert_idx];
    valid_m =
        valid_m < 0 ? 0 : (valid_m > tokens_per_expert ? tokens_per_expert
                                                        : valid_m);
    token_idx = expert_idx * tokens_per_expert + blockIdx.y;
    token_end = expert_idx * tokens_per_expert + valid_m;
    token_stride = gridDim.y;
  } else if (mask_m != nullptr) {
    // 2D path with a device-side row count: the grid is launched at a
    // host-side upper bound; each block clamps itself to the real total.
    int64_t valid = mask_m[0];
    valid = valid < 0 ? 0 : valid;
    token_end = valid;
    token_stride = gridDim.x;
  }

  using InputVec = at::native::memory::aligned_vector<scalar_t, vec>;
  using OutputVec = at::native::memory::aligned_vector<int8_t, vec>;

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  __shared__ float row_absmax;

  for (; token_idx < token_end; token_idx += token_stride) {
    const int idx = threadIdx.x * vec;
    const int64_t input_offset = token_idx * 2 * d + idx;
    const int64_t output_offset = token_idx * d + idx;
    scalar_t gate_values[vec];
    scalar_t up_values[vec];
    scalar_t activated[vec];
    float row_max = 0.0f;

    if (idx < d) {
      *reinterpret_cast<InputVec*>(gate_values) =
          *reinterpret_cast<const InputVec*>(input + input_offset);
      *reinterpret_cast<InputVec*>(up_values) =
          *reinterpret_cast<const InputVec*>(input + input_offset + d);
#pragma unroll
      for (int i = 0; i < vec; ++i) {
        activated[i] = swiglu_apply_smooth<scalar_t, apply_smooth>(
            swiglu_clamp_value<scalar_t, use_gelu>(
                gate_values[i], up_values[i], limit),
            smooth_scale,
            idx + i);
        row_max = fmaxf(row_max, fabsf(static_cast<float>(activated[i])));
      }
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

    if (threadIdx.x == 0) {
      // Match per_token_quant_int8's non-zero minimum scale for an all-zero
      // row.
      row_absmax = fmaxf(row_max, 1.0e-10f);
      scales[token_idx] = row_absmax / 127.0f;
    }
    __syncthreads();

    if (idx < d) {
      int8_t quantized[vec];
      const float inv_scale = 127.0f / row_absmax;
#pragma unroll
      for (int i = 0; i < vec; ++i) {
        quantized[i] = swiglu_clamp_float_to_int8_rn(
            static_cast<float>(activated[i]) * inv_scale);
      }
      *reinterpret_cast<OutputVec*>(output + output_offset) =
          *reinterpret_cast<OutputVec*>(quantized);
    }
    if constexpr (ep) {
      __syncthreads();
    } else if (mask_m != nullptr) {
      // grid-stride reuse of reduce_scratch/row_absmax requires a barrier
      __syncthreads();
    }
  }
}

template <typename scalar_t,
          int block_size,
          bool ep,
          bool use_gelu,
          bool apply_smooth>
__global__ void swiglu_clamp_quant_scalar_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* __restrict__ mask_m) {
  int64_t token_idx = blockIdx.x;
  int64_t token_end = token_idx + 1;
  int64_t token_stride = 1;
  if constexpr (ep) {
    const int64_t expert_idx = blockIdx.x;
    int64_t valid_m = mask_m[expert_idx];
    valid_m =
        valid_m < 0 ? 0 : (valid_m > tokens_per_expert ? tokens_per_expert
                                                        : valid_m);
    token_idx = expert_idx * tokens_per_expert + blockIdx.y;
    token_end = expert_idx * tokens_per_expert + valid_m;
    token_stride = gridDim.y;
  } else if (mask_m != nullptr) {
    // 2D path with a device-side row count: the grid is launched at a
    // host-side upper bound; each block clamps itself to the real total.
    int64_t valid = mask_m[0];
    valid = valid < 0 ? 0 : valid;
    token_end = valid;
    token_stride = gridDim.x;
  }

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  __shared__ float row_absmax;

  for (; token_idx < token_end; token_idx += token_stride) {
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;
    float row_max = 0.0f;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t value = swiglu_apply_smooth<scalar_t, apply_smooth>(
          swiglu_clamp_value<scalar_t, use_gelu>(
              LIGHTOP_LDG(input + input_offset + idx),
              LIGHTOP_LDG(input + input_offset + d + idx),
              limit),
          smooth_scale,
          idx);
      row_max = fmaxf(row_max, fabsf(static_cast<float>(value)));
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

    if (threadIdx.x == 0) {
      row_absmax = fmaxf(row_max, 1.0e-10f);
      scales[token_idx] = row_absmax / 127.0f;
    }
    __syncthreads();

    const float inv_scale = 127.0f / row_absmax;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t value = swiglu_apply_smooth<scalar_t, apply_smooth>(
          swiglu_clamp_value<scalar_t, use_gelu>(
              LIGHTOP_LDG(input + input_offset + idx),
              LIGHTOP_LDG(input + input_offset + d + idx),
              limit),
          smooth_scale,
          idx);
      output[output_offset + idx] = swiglu_clamp_float_to_int8_rn(
          static_cast<float>(value) * inv_scale);
    }
    if constexpr (ep) {
      __syncthreads();
    } else if (mask_m != nullptr) {
      // grid-stride reuse of reduce_scratch/row_absmax requires a barrier
      __syncthreads();
    }
  }
}

template <typename scalar_t, bool ep, bool use_gelu, bool apply_smooth>
void launch_swiglu_clamp_quant(
    const torch::Tensor& input,
    const float* smooth_scale,
    torch::Tensor& output,
    torch::Tensor& scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* mask_m,
    dim3 grid,
    cudaStream_t stream) {
  if (d <= 512 && d % 2 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 2, 256, ep, use_gelu, apply_smooth>
        <<<grid, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 1024 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 128, ep, use_gelu, apply_smooth>
        <<<grid, 128, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 2048 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 256, ep, use_gelu, apply_smooth>
        <<<grid, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 4096 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 512, ep, use_gelu, apply_smooth>
        <<<grid, 512, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 8192 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 16384 && d % 16 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 16, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else {
    swiglu_clamp_quant_scalar_kernel<
        scalar_t, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  }
}

static void fuse_silu_mul_clamp_quant_impl(
    torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    double limit,
    const torch::Tensor* mask_m,
    int expect_m,
    bool use_ep,
    bool use_gelu,
    const c10::optional<torch::Tensor>& smooth_scale) {
  TORCH_CHECK(
      input.dim() == (use_ep ? 3 : 2),
      use_ep ? "EP input must have shape [experts, tokens, 2 * hidden]"
             : "input must have shape [tokens, 2 * hidden]");
  TORCH_CHECK(input.is_cuda(), "input must be on a CUDA/HIP device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "input's last dimension must be even");
  TORCH_CHECK(limit > 0.0, "limit must be positive");

  const int64_t num_tokens =
      use_ep ? input.size(0) * input.size(1) : input.size(0);
  const int d = input.size(-1) / 2;
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Char,
              "output must be INT8");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
              "scales must be FP32");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(input.device() == output.device() &&
                  input.device() == scales.device(),
              "input, output, and scales must be on the same device");

  const float* smooth_scale_ptr = nullptr;
  if (smooth_scale.has_value()) {
    const torch::Tensor& smooth = smooth_scale.value();
    TORCH_CHECK(smooth.is_cuda(),
                "smooth_scale must be on a CUDA/HIP device");
    TORCH_CHECK(smooth.device() == input.device(),
                "input and smooth_scale must be on the same device");
    TORCH_CHECK(smooth.is_contiguous(),
                "smooth_scale must be contiguous");
    TORCH_CHECK(smooth.scalar_type() == at::ScalarType::Float,
                "smooth_scale must have torch.float32 dtype");
    TORCH_CHECK(smooth.dim() == 1 && smooth.numel() == d,
                "smooth_scale must have shape [hidden]");
    smooth_scale_ptr = smooth.data_ptr<float>();
  }

  int64_t tokens_per_expert = 0;
  const int32_t* mask_ptr = nullptr;
  dim3 grid;
  if (use_ep) {
    const int64_t num_experts = input.size(0);
    tokens_per_expert = input.size(1);
    TORCH_CHECK(
        output.dim() == 3 && output.size(0) == num_experts &&
            output.size(1) == tokens_per_expert && output.size(2) == d,
        "EP output must have shape [experts, tokens, hidden]");
    TORCH_CHECK(
        scales.dim() == 3 && scales.size(0) == num_experts &&
            scales.size(1) == tokens_per_expert && scales.size(2) == 1,
        "EP scales must have shape [experts, tokens, 1]");
    TORCH_CHECK(mask_m != nullptr, "mask_m is required for the EP path");
    const torch::Tensor& mask = *mask_m;
    TORCH_CHECK(mask.is_cuda(), "mask_m must be on a CUDA/HIP device");
    TORCH_CHECK(mask.device() == input.device(),
                "input and mask_m must be on the same device");
    TORCH_CHECK(mask.is_contiguous(), "mask_m must be contiguous");
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int,
                "mask_m must have torch.int32 dtype");
    TORCH_CHECK(mask.dim() == 1 && mask.numel() == num_experts,
                "mask_m must have shape [experts]");
    mask_ptr = mask.data_ptr<int32_t>();

    if (num_experts != 0 && tokens_per_expert != 0) {
      const int64_t expected_blocks =
          expect_m >= 0 ? int64_t(expect_m) : tokens_per_expert;
      const int grid_y = static_cast<int>(std::max<int64_t>(
          1, std::min(tokens_per_expert, expected_blocks)));
      grid = dim3(static_cast<uint32_t>(num_experts),
                  static_cast<uint32_t>(grid_y));
    }
  } else {
    TORCH_CHECK(output.dim() == 2 && output.size(0) == num_tokens &&
                    output.size(1) == d,
                "output must have shape [tokens, hidden]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == num_tokens &&
                    scales.size(1) == 1,
                "scales must have shape [tokens, 1]");
    if (mask_m != nullptr) {
      // Optional device-side valid-row count: allocate/launch at a host-side
      // upper bound, no D2H sync.
      const torch::Tensor& mask = *mask_m;
      TORCH_CHECK(mask.is_cuda(),
                  "num_local_tokens_tensor must be on a CUDA/HIP device");
      TORCH_CHECK(
          mask.device() == input.device(),
          "input and num_local_tokens_tensor must be on the same device");
      TORCH_CHECK(mask.is_contiguous(),
                  "num_local_tokens_tensor must be contiguous");
      TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int,
                  "num_local_tokens_tensor must have torch.int32 dtype");
      TORCH_CHECK(mask.numel() >= 1,
                  "num_local_tokens_tensor must have at least one element");
      mask_ptr = mask.data_ptr<int32_t>();
    }
    if (num_tokens != 0) {
      // With a device-side row count the kernel is grid-stride: cap the grid
      // (expect_m overrides the 4096 default). Otherwise one block per row.
      int64_t blocks = num_tokens;
      if (mask_ptr != nullptr) {
        const int64_t cap = expect_m > 0 ? static_cast<int64_t>(expect_m) : 4096;
        blocks = std::min<int64_t>(num_tokens, cap);
      }
      grid = dim3(static_cast<uint32_t>(std::max<int64_t>(1, blocks)));
    }
  }

  if (num_tokens == 0 || d == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float limit_f = static_cast<float>(limit);

  LIGHTOP_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fuse_silu_mul_clamp_quant", [&] {
        if (use_ep && use_gelu && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, true, true, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep && use_gelu) {
          launch_swiglu_clamp_quant<scalar_t, true, true, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, true, false, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep) {
          launch_swiglu_clamp_quant<scalar_t, true, false, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_gelu && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, false, true, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              mask_ptr,
              grid,
              stream);
        } else if (use_gelu) {
          launch_swiglu_clamp_quant<scalar_t, false, true, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              mask_ptr,
              grid,
              stream);
        } else if (smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, false, false, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              mask_ptr,
              grid,
              stream);
        } else {
          launch_swiglu_clamp_quant<scalar_t, false, false, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              mask_ptr,
              grid,
              stream);
        }
      });
}

void fuse_silu_mul_clamp_quant(torch::Tensor& input,
                               torch::Tensor& output,
                               torch::Tensor& scales,
                               double limit,
                               bool use_gelu,
                               const c10::optional<torch::Tensor>& smooth_scale,
                               const c10::optional<torch::Tensor>& num_local_tokens_tensor,
                               int expect_m) {
  const torch::Tensor* mask = num_local_tokens_tensor.has_value()
                                  ? &num_local_tokens_tensor.value()
                                  : nullptr;
  fuse_silu_mul_clamp_quant_impl(
      input, output, scales, limit, mask, expect_m, false, use_gelu, smooth_scale);
}

void fuse_silu_mul_clamp_quant_ep(torch::Tensor& input,
                                  torch::Tensor& output,
                                  torch::Tensor& scales,
                                  double limit,
                                  const torch::Tensor& mask_m,
                                  int expect_m,
                                  bool use_gelu,
                                  const c10::optional<torch::Tensor>& smooth_scale) {
  fuse_silu_mul_clamp_quant_impl(
      input,
      output,
      scales,
      limit,
      &mask_m,
      expect_m,
      true,
      use_gelu,
      smooth_scale);
}

}  // namespace native
}  // namespace at
