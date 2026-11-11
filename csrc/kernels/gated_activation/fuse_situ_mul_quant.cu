// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
#include "reduce.h"

namespace at {
namespace native {

namespace situ_quant_detail {

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
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

#ifndef USE_ROCM
  #define SITU_LDG(arg) __ldg(arg)
#else
  #define SITU_LDG(arg) *(arg)
#endif

__device__ __forceinline__ float fast_sigmoid(float x) {
  constexpr float LOG2E = 1.44269504088896340736f;
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x * LOG2E));
}

// SiTU / SoftCap-GLU used by Kimi K3. The host precomputes `two_inv_beta`,
// `two_inv_linear_beta`, and `beta_prod = situ_beta * situ_linear_beta`, so
// the kernel path uses only mults/exp2 (no divisions). Numerical parity with
// the sglang-das Triton reference is preserved via the tanh-via-sigmoid
// identity `tanh(y) = 2*sigmoid(2y) - 1`.
template <typename scalar_t>
__device__ __forceinline__ float situ_activate(scalar_t gate_v,
                                               scalar_t up_v,
                                               float two_inv_beta,
                                               float two_inv_linear_beta,
                                               float beta_prod) {
  const float g = static_cast<float>(gate_v);
  const float u = static_cast<float>(up_v);
  const float gate_tanh = 2.0f * fast_sigmoid(g * two_inv_beta) - 1.0f;
  const float up_tanh   = 2.0f * fast_sigmoid(u * two_inv_linear_beta) - 1.0f;
  const float sig_g     = fast_sigmoid(g);
  return beta_prod * gate_tanh * sig_g * up_tanh;
}

// Generic (non-vectorized) fallback for very large hidden dims.
template <typename scalar_t, int block_size>
__global__ void situ_mul_quant_kernel_generic(
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int hidden,
    const float two_inv_beta,
    const float two_inv_linear_beta,
    const float beta_prod) {
  const int64_t token_idx = blockIdx.x;
  const int64_t input_offset  = token_idx * 2 * hidden;
  const int64_t output_offset = token_idx * hidden;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float shared_mem[share_size];
  __shared__ float s_token_scale;

  float row_max = 0.0f;
  for (int idx = threadIdx.x; idx < hidden; idx += blockDim.x) {
    const scalar_t g = SITU_LDG(&input[input_offset + idx]);
    const scalar_t u = SITU_LDG(&input[input_offset + hidden + idx]);
    const float val =
        situ_activate(g, u, two_inv_beta, two_inv_linear_beta, beta_prod);
    row_max = fmaxf(row_max, fabsf(val));
  }
  row_max = BlockReduceMax_ROW<float, block_size>(row_max, shared_mem);

  if (threadIdx.x == 0) {
    s_token_scale = row_max;
    scales[token_idx] = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
  }
  __syncthreads();

  const float inv_s =
      (s_token_scale == 0.0f) ? 0.0f : 127.0f / s_token_scale;

  for (int idx = threadIdx.x; idx < hidden; idx += blockDim.x) {
    const scalar_t g = SITU_LDG(&input[input_offset + idx]);
    const scalar_t u = SITU_LDG(&input[input_offset + hidden + idx]);
    const float val =
        situ_activate(g, u, two_inv_beta, two_inv_linear_beta, beta_prod);
    out[output_offset + idx] = float_to_int8_rn(val * inv_s);
  }
}

// Vectorized single-tile kernel: one block per token, one VEC-wide pass.
// Materializes r_y[VEC] in registers between the amax reduce and the int8
// quant, so the activation math runs exactly once per element.
template <typename scalar_t, int VEC, int block_size>
__global__ void situ_mul_quant_kernel_vec(
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int hidden,
    const float two_inv_beta,
    const float two_inv_linear_beta,
    const float beta_prod) {
  const int64_t token_idx = blockIdx.x;

  using VecType     = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

  const int tidx = threadIdx.x;
  const int idx  = tidx * VEC;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  __shared__ float s_token_scale;

  const int64_t x_index = token_idx * 2 * hidden + idx;
  const int64_t y_index = token_idx * hidden + idx;

  const VecType* x1 = reinterpret_cast<const VecType*>(input + x_index);
  const VecType* x2 = reinterpret_cast<const VecType*>(input + x_index + hidden);
  VecInt8Type*   y  = reinterpret_cast<VecInt8Type*>(out + y_index);

  scalar_t r_g[VEC];
  scalar_t r_u[VEC];
  float    r_y[VEC];

  const bool in_range = (idx < hidden);

  if (in_range) {
    *reinterpret_cast<VecType*>(r_g) = *x1;
    *reinterpret_cast<VecType*>(r_u) = *x2;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
      r_y[i] = situ_activate(r_g[i], r_u[i],
                             two_inv_beta, two_inv_linear_beta, beta_prod);
    }
  }

  float row_max = 0.0f;
  if (in_range) {
    #pragma unroll
    for (int i = 0; i < VEC; ++i) row_max = fmaxf(row_max, fabsf(r_y[i]));
  }
  row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);

  if (tidx == 0) {
    s_token_scale = row_max;
    scales[token_idx] = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
  }
  __syncthreads();

  const float inv_s = (s_token_scale == 0.0f) ? 0.0f : 127.0f / s_token_scale;

  if (in_range) {
    int8_t out_vec[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out_vec[i] = float_to_int8_rn(r_y[i] * inv_s);
    }
    *y = *reinterpret_cast<const VecInt8Type*>(out_vec);
  }
}

}  // namespace situ_quant_detail

#ifndef SITU_DISPATCH_FLOATING_TYPES
#define SITU_DISPATCH_CASE_FLOATING_TYPES(...)          \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define SITU_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                 \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                       \
                     SITU_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#endif

void fuse_situ_mul_quant_contiguous(torch::Tensor& input,
                                    torch::Tensor& out,
                                    torch::Tensor& scales,
                                    double situ_beta,
                                    double situ_linear_beta) {
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, 2*hidden]");
  TORCH_CHECK(input.size(-1) % 2 == 0, "input last dim must be even");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(situ_beta > 0.0, "situ_beta must be positive");
  TORCH_CHECK(situ_linear_beta > 0.0, "situ_linear_beta must be positive");

  const int64_t tokens = input.size(0);
  const int hidden = static_cast<int>(input.size(-1) / 2);

  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens && out.size(1) == hidden,
              "out shape must be [tokens, hidden]");
  TORCH_CHECK(scales.numel() == tokens,
              "scales must have `tokens` elements");

  if (tokens == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Host-side precompute keeps the device path div-free.
  const float two_inv_beta        = 2.0f / static_cast<float>(situ_beta);
  const float two_inv_linear_beta = 2.0f / static_cast<float>(situ_linear_beta);
  const float beta_prod           =
      static_cast<float>(situ_beta) * static_cast<float>(situ_linear_beta);

  dim3 grid(tokens);

  SITU_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_mul_quant_contiguous", [&] {
    if (hidden <= 512) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 2, 256>
          <<<grid, 256, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      hidden, two_inv_beta,
                                      two_inv_linear_beta, beta_prod);
    } else if (hidden <= 1024) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 8, 128>
          <<<grid, 128, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      hidden, two_inv_beta,
                                      two_inv_linear_beta, beta_prod);
    } else if (hidden <= 2048) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 8, 256>
          <<<grid, 256, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      hidden, two_inv_beta,
                                      two_inv_linear_beta, beta_prod);
    } else if (hidden <= 4096) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 8, 512>
          <<<grid, 512, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      hidden, two_inv_beta,
                                      two_inv_linear_beta, beta_prod);
    } else if (hidden <= 8192) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 8, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       hidden, two_inv_beta,
                                       two_inv_linear_beta, beta_prod);
    } else if (hidden <= 16384) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 16, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       hidden, two_inv_beta,
                                       two_inv_linear_beta, beta_prod);
    } else if (hidden <= 32768) {
      situ_quant_detail::situ_mul_quant_kernel_vec<scalar_t, 32, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       hidden, two_inv_beta,
                                       two_inv_linear_beta, beta_prod);
    } else {
      situ_quant_detail::situ_mul_quant_kernel_generic<scalar_t, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       hidden, two_inv_beta,
                                       two_inv_linear_beta, beta_prod);
    }
  });
}

}  // namespace native
}  // namespace at
