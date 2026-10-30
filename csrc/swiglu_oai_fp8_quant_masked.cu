// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>

#include <ATen/native/cuda/MemoryAccess.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace at::native {
namespace {
// This is the OAI activation with FP32 intermediates, not BF16 SiLU * up.
__device__ __forceinline__ float activate(float g, float u, float alpha,
                                          float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  constexpr float log2e = 1.4426950408889634f;
  return (g * (1.0f / (1.0f + __builtin_amdgcn_exp2f(-alpha * g * log2e)))) *
         (u + 1.0f);
}

__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
#if defined(__gfx938__)
  uint32_t bits = 0;
  bits = __builtin_hcu_cvt_pk_fp8_f32(a, b, bits, false);
  return __builtin_hcu_cvt_pk_fp8_f32(c, d, bits, true);
#else
  uint32_t bits = 0;
  auto* bytes = reinterpret_cast<uint8_t*>(&bits);
  bytes[0] = c10::Float8_e4m3fn(a).x;
  bytes[1] = c10::Float8_e4m3fn(b).x;
  bytes[2] = c10::Float8_e4m3fn(c).x;
  bytes[3] = c10::Float8_e4m3fn(d).x;
  return bits;
#endif
}

template <int Threads, int Segments>
__global__ void swiglu_oai_fp8_quant_masked_kernel(
    const c10::BFloat16* __restrict__ x, uint8_t* __restrict__ q,
    float* __restrict__ scales, const int32_t* __restrict__ counts,
    int capacity, int width, float alpha, float limit) {
  constexpr int Wave = 64, Vec = 8;
  constexpr int Waves = Threads / Wave;
  const int expert = blockIdx.x, tid = threadIdx.x;
  const int raw_count = counts[expert];
  const int count =
      raw_count < 0 ? 0 : (raw_count > capacity ? capacity : raw_count);
  __shared__ float wave_max[Waves];
  using InputVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutputVec = at::native::memory::aligned_vector<uint8_t, Vec>;
  for (int row = blockIdx.y; row < count; row += gridDim.y) {
    const int64_t token = int64_t(expert) * capacity + row;
    float values[Segments][Vec];
    float maximum = 0.0f;
#pragma unroll
    for (int s = 0; s < Segments; ++s) {
      const int col = (tid + s * Threads) * Vec;
      if (col < width) {
        InputVec g =
            *reinterpret_cast<const InputVec*>(x + token * 2 * width + col);
        InputVec u = *reinterpret_cast<const InputVec*>(x + token * 2 * width +
                                                        width + col);
#pragma unroll
        for (int j = 0; j < Vec; ++j) {
          values[s][j] =
              activate(float(g.val[j]), float(u.val[j]), alpha, limit);
          maximum = fmaxf(maximum, fabsf(values[s][j]));
        }
      }
    }
#pragma unroll
    for (int offset = Wave / 2; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor(maximum, offset, Wave));
    if constexpr (Waves > 1) {
      if ((tid & (Wave - 1)) == 0) wave_max[tid / Wave] = maximum;
      __syncthreads();
#pragma unroll
      for (int w = 0; w < Waves; ++w) maximum = fmaxf(maximum, wave_max[w]);
    }
    const float scale = fmaxf(maximum, 1.0e-10f) / 448.0f;
    if (tid == 0) scales[token] = scale;
#pragma unroll
    for (int s = 0; s < Segments; ++s) {
      const int col = (tid + s * Threads) * Vec;
      if (col < width) {
        OutputVec out;
#pragma unroll
        for (int j = 0; j < Vec; j += 4) {
          float a[4];
#pragma unroll
          for (int k = 0; k < 4; ++k)
            a[k] = fminf(fmaxf(values[s][j + k] / scale, -448.0f), 448.0f);
          *reinterpret_cast<uint32_t*>(&out.val[j]) =
              pack4(a[0], a[1], a[2], a[3]);
        }
        *reinterpret_cast<OutputVec*>(q + token * width + col) = out;
      }
    }
    // All waves must finish reading wave_max before the next row overwrites it.
    if constexpr (Waves > 1) __syncthreads();
  }
}

template <int Threads>
void launch(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
            const torch::Tensor& counts, float alpha, float limit,
            int row_blocks, cudaStream_t stream) {
  const dim3 grid(x.size(0), std::min<int64_t>(row_blocks, x.size(1)));
  // All vector loads are valid: the public entry requires width % 8 == 0.
  swiglu_oai_fp8_quant_masked_kernel<Threads,
                                     (4096 + Threads * 8 - 1) / (Threads * 8)>
      <<<grid, Threads, 0, stream>>>(
          x.data_ptr<c10::BFloat16>(), reinterpret_cast<uint8_t*>(q.data_ptr()),
          s.data_ptr<float>(), counts.data_ptr<int32_t>(), x.size(1),
          x.size(2) / 2, alpha, limit);
}
}  // namespace

void swiglu_oai_fp8_quant_masked(const torch::Tensor& x, torch::Tensor& q,
                                 torch::Tensor& scales,
                                 const torch::Tensor& counts, double alpha,
                                 double limit, int64_t row_blocks,
                                 int64_t threads) {
  TORCH_CHECK(x.is_cuda() && q.device() == x.device() &&
                  scales.device() == x.device() &&
                  counts.device() == x.device(),
              "All tensors must share a GPU");
  TORCH_CHECK(
      x.dim() == 3 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
      "Expected contiguous BF16 [E, capacity, 2*width]");
  const int64_t e = x.size(0), m = x.size(1), n = x.size(2) / 2;
  TORCH_CHECK(x.size(2) % 2 == 0 && n > 0 && n <= 4096 && n % 8 == 0,
              "Supported logical width: 8..4096, divisible by 8");
  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && q.is_contiguous() &&
                  q.sizes() == c10::IntArrayRef({e, m, n}),
              "Invalid FP8 output");
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() &&
                  scales.numel() == e * m,
              "Expected one FP32 scale per row");
  TORCH_CHECK(counts.scalar_type() == at::kInt && counts.dim() == 1 &&
                  counts.numel() == e && counts.is_contiguous(),
              "Invalid counts");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0,
              "Vectorized input/output pointers require 16/8-byte alignment");
  TORCH_CHECK(!x.is_alias_of(q) && !x.is_alias_of(scales) &&
                  !x.is_alias_of(counts) && !q.is_alias_of(scales) &&
                  !q.is_alias_of(counts) && !scales.is_alias_of(counts),
              "Input/output/count buffers must not share storage");
  TORCH_CHECK(m <= std::numeric_limits<int32_t>::max() &&
                  e <= std::numeric_limits<int32_t>::max(),
              "Expert/capacity axis exceeds int32");
  TORCH_CHECK(
      alpha > 0 && std::isfinite(alpha) && limit > 0 && std::isfinite(limit),
      "alpha and limit must be positive finite values");
  TORCH_CHECK(row_blocks > 0 && row_blocks <= 128 &&
                  (threads == 64 || threads == 128 || threads == 256),
              "Invalid launch configuration");
  if (e == 0 || m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 64)
    launch<64>(x, q, scales, counts, alpha, limit, row_blocks, stream);
  else if (threads == 128)
    launch<128>(x, q, scales, counts, alpha, limit, row_blocks, stream);
  else
    launch<256>(x, q, scales, counts, alpha, limit, row_blocks, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace at::native
