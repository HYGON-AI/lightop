// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace at {
namespace native {
namespace {

// One wave owns a row: keep the BF16-rounded activation in registers until
// its scale is known. No intermediate allocation, second input pass or LDS.
template <int Capacity, int Vec, int RowsPerBlock>
__global__ __launch_bounds__(64 * RowsPerBlock)
void swiglu_oai_fp8_quant_contiguous_kernel(
    const c10::BFloat16* __restrict__ input,
    uint8_t* __restrict__ output, float* __restrict__ scales,
    int64_t rows, int cols, float alpha, float beta, float limit) {
  const int lane = threadIdx.x & 63;
  const int64_t row = int64_t(blockIdx.x) * RowsPerBlock + threadIdx.x / 64;
  if (row >= rows) return;  // Wave-uniform, no block-wide barriers.
  constexpr int Iterations = (Capacity + 64 * Vec - 1) / (64 * Vec);
  float values[Iterations * Vec];
  float amax = 0.0f;
  const auto* gate_row = input + row * (2 * cols);
  const auto* up_row = gate_row + cols;
#pragma unroll
  for (int it = 0; it < Iterations; ++it) {
    const int col = (it * 64 + lane) * Vec;
    if (col < cols) {
      using InputVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
      const InputVec gates = *reinterpret_cast<const InputVec*>(gate_row + col);
      const InputVec ups = *reinterpret_cast<const InputVec*>(up_row + col);
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        const float gate = fminf(float(gates.val[v]), limit);
        const float up = fminf(fmaxf(float(ups.val[v]), -limit), limit);
        const float sigmoid = 1.0f / (1.0f + __expf(-alpha * gate));
        // Match the contiguous SGLang activation -> BF16 -> native FP8 chain.
        const float value = float(c10::BFloat16((gate * sigmoid) * (up + beta)));
        values[it * Vec + v] = value;
        amax = fmaxf(amax, fabsf(value));
      }
    }
  }
#pragma unroll
  for (int delta = 32; delta; delta >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, delta, 64));
  const float scale = amax / 448.0f;
  if (lane == 0) scales[row] = scale;
  // Preserve the native scale=0/output=0 contract for all-zero rows.
  const float inv_scale = scale == 0.0f ? 0.0f : 1.0f / scale;
#pragma unroll
  for (int it = 0; it < Iterations; ++it) {
    const int col = (it * 64 + lane) * Vec;
    if (col < cols) {
#pragma unroll
      for (int v = 0; v < Vec; v += (Vec >= 4 ? 4 : 1)) {
        if constexpr (Vec >= 4) {
          float q[4];
#pragma unroll
          for (int k = 0; k < 4; ++k)
            q[k] = fminf(448.0f, fmaxf(-448.0f, values[it * Vec + v + k] * inv_scale));
          uint32_t packed = 0;
#if defined(__gfx938__)
          packed = __builtin_hcu_cvt_pk_fp8_f32(q[0], q[1], packed, false);
          packed = __builtin_hcu_cvt_pk_fp8_f32(q[2], q[3], packed, true);
#else
#pragma unroll
          for (int k = 0; k < 4; ++k)
            reinterpret_cast<uint8_t*>(&packed)[k] = c10::Float8_e4m3fn(q[k]).x;
#endif
          *reinterpret_cast<uint32_t*>(output + row * cols + col + v) = packed;
        } else {
          const float q = fminf(448.0f, fmaxf(-448.0f, values[it] * inv_scale));
          output[row * cols + col] = c10::Float8_e4m3fn(q).x;
        }
      }
    }
  }
}

template <int Capacity, int Vec>
void launch(const torch::Tensor& input, torch::Tensor& output,
            torch::Tensor& scales, float alpha, float beta, float limit) {
  constexpr int RowsPerBlock = 4;
  const int64_t rows = input.size(0);
  const int cols = int(input.size(1) / 2);
  swiglu_oai_fp8_quant_contiguous_kernel<Capacity, Vec, RowsPerBlock>
      <<<((rows + RowsPerBlock - 1) / RowsPerBlock), 64 * RowsPerBlock,
         0, at::cuda::getCurrentCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          reinterpret_cast<uint8_t*>(output.data_ptr<c10::Float8_e4m3fn>()),
          scales.data_ptr<float>(), rows, cols, alpha, beta, limit);
}

template <int Vec>
void dispatch(const torch::Tensor& input, torch::Tensor& output,
              torch::Tensor& scales, float alpha, float beta, float limit) {
  const int cols = int(input.size(1) / 2);
  if (cols <= 512) launch<512, Vec>(input, output, scales, alpha, beta, limit);
  else if (cols <= 1024) launch<1024, Vec>(input, output, scales, alpha, beta, limit);
  else if (cols <= 2048) launch<2048, Vec>(input, output, scales, alpha, beta, limit);
  else if (cols <= 3072) launch<3072, Vec>(input, output, scales, alpha, beta, limit);
  else if (cols <= 4096) launch<4096, Vec>(input, output, scales, alpha, beta, limit);
  else launch<8192, Vec>(input, output, scales, alpha, beta, limit);
}
}  // namespace

std::vector<torch::Tensor> swiglu_oai_fp8_quant_contiguous(
    const torch::Tensor& input, double alpha, double beta, double limit) {
  TORCH_CHECK(input.is_cuda(), "input must be a HIP tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "input must be BF16; the intermediate rounding is part of this API");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "expected contiguous [rows, gate_cols + up_cols] input");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % 2 == 0 && input.size(1) <= 16384,
              "expected an even width with 1 <= output columns <= 8192");
  TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta) && std::isfinite(limit) &&
              alpha > 0.0 && limit > 0.0 && alpha <= std::numeric_limits<float>::max() &&
              std::abs(beta) <= std::numeric_limits<float>::max() &&
              limit <= std::numeric_limits<float>::max(), "invalid OAI parameters");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto rows = input.size(0);
  const auto cols = input.size(1) / 2;
  auto output = torch::empty({rows, cols}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = torch::empty({rows, 1}, input.options().dtype(at::kFloat));
  if (rows == 0) return {output, scales};
  TORCH_CHECK((rows + 3) / 4 <= std::numeric_limits<uint32_t>::max(), "too many rows");
  const bool aligned = reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0;
  if (cols % 8 == 0 && aligned)
    dispatch<8>(input, output, scales, float(alpha), float(beta), float(limit));
  else
    dispatch<1>(input, output, scales, float(alpha), float(beta), float(limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}
}  // namespace native
}  // namespace at
