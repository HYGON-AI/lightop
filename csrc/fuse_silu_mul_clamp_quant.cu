#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

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

template <typename scalar_t>
__device__ __forceinline__ scalar_t swiglu_clamp_value(
    scalar_t gate, scalar_t up, float limit) {
  float gate_f = static_cast<float>(gate);
  float up_f = static_cast<float>(up);

  // Match the native GLM path, including a cast after each operation:
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
  const scalar_t silu = static_cast<scalar_t>(
      static_cast<float>(gate_clamped) / (1.0f + exp_value));
  return static_cast<scalar_t>(silu * up_clamped);
}

template <typename scalar_t, int vec, int block_size>
__global__ void swiglu_clamp_quant_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit) {
  const int64_t token_idx = blockIdx.x;
  const int idx = threadIdx.x * vec;
  const int64_t input_offset = token_idx * 2 * d + idx;
  const int64_t output_offset = token_idx * d + idx;

  using InputVec = at::native::memory::aligned_vector<scalar_t, vec>;
  using OutputVec = at::native::memory::aligned_vector<int8_t, vec>;

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
      activated[i] =
          swiglu_clamp_value(gate_values[i], up_values[i], limit);
      row_max = fmaxf(row_max, fabsf(static_cast<float>(activated[i])));
    }
  }

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

  __shared__ float row_absmax;
  if (threadIdx.x == 0) {
    // Match per_token_quant_int8's non-zero minimum scale for an all-zero row.
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
}

template <typename scalar_t, int block_size>
__global__ void swiglu_clamp_quant_scalar_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit) {
  const int64_t token_idx = blockIdx.x;
  const int64_t input_offset = token_idx * 2 * d;
  const int64_t output_offset = token_idx * d;

  float row_max = 0.0f;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t value = swiglu_clamp_value(
        LIGHTOP_LDG(input + input_offset + idx),
        LIGHTOP_LDG(input + input_offset + d + idx),
        limit);
    row_max = fmaxf(row_max, fabsf(static_cast<float>(value)));
  }

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

  __shared__ float row_absmax;
  if (threadIdx.x == 0) {
    row_absmax = fmaxf(row_max, 1.0e-10f);
    scales[token_idx] = row_absmax / 127.0f;
  }
  __syncthreads();

  const float inv_scale = 127.0f / row_absmax;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t value = swiglu_clamp_value(
        LIGHTOP_LDG(input + input_offset + idx),
        LIGHTOP_LDG(input + input_offset + d + idx),
        limit);
    output[output_offset + idx] = swiglu_clamp_float_to_int8_rn(
        static_cast<float>(value) * inv_scale);
  }
}

void fuse_silu_mul_clamp_quant(torch::Tensor& input,
                               torch::Tensor& output,
                               torch::Tensor& scales,
                               double limit) {
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, 2 * hidden]");
  TORCH_CHECK(input.is_cuda(), "input must be on a CUDA/HIP device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.size(1) % 2 == 0, "input's last dimension must be even");
  TORCH_CHECK(limit > 0.0, "limit must be positive");

  const int64_t num_tokens = input.size(0);
  const int d = input.size(1) / 2;
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Char,
              "output must be INT8");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == num_tokens &&
                  output.size(1) == d,
              "output must have shape [tokens, hidden]");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
              "scales must be FP32");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == num_tokens &&
                  scales.size(1) == 1,
              "scales must have shape [tokens, 1]");
  TORCH_CHECK(input.device() == output.device() &&
                  input.device() == scales.device(),
              "input, output, and scales must be on the same device");

  if (num_tokens == 0 || d == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_tokens);
  const float limit_f = static_cast<float>(limit);

  LIGHTOP_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fuse_silu_mul_clamp_quant", [&] {
        if (d <= 512 && d % 2 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 2, 256>
              <<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         output.data_ptr<int8_t>(),
                                         scales.data_ptr<float>(), d, limit_f);
        } else if (d <= 1024 && d % 8 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 8, 128>
              <<<grid, 128, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         output.data_ptr<int8_t>(),
                                         scales.data_ptr<float>(), d, limit_f);
        } else if (d <= 2048 && d % 8 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 8, 256>
              <<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         output.data_ptr<int8_t>(),
                                         scales.data_ptr<float>(), d, limit_f);
        } else if (d <= 4096 && d % 8 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 8, 512>
              <<<grid, 512, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         output.data_ptr<int8_t>(),
                                         scales.data_ptr<float>(), d, limit_f);
        } else if (d <= 8192 && d % 8 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 8, 1024>
              <<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),
                                          output.data_ptr<int8_t>(),
                                          scales.data_ptr<float>(), d, limit_f);
        } else if (d <= 16384 && d % 16 == 0) {
          swiglu_clamp_quant_kernel<scalar_t, 16, 1024>
              <<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),
                                          output.data_ptr<int8_t>(),
                                          scales.data_ptr<float>(), d, limit_f);
        } else {
          swiglu_clamp_quant_scalar_kernel<scalar_t, 1024>
              <<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),
                                          output.data_ptr<int8_t>(),
                                          scales.data_ptr<float>(), d, limit_f);
        }
      });
}

}  // namespace native
}  // namespace at
