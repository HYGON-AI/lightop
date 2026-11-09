// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "reduce.h"
#include <cmath>
#include <cstdlib>


#ifndef USE_ROCM
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
#include <ATen/hip/HIPContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>

#define MAYBE_HOST_DEVICE
#endif
namespace lightop_glm53_activation {
using at::native::BlockReduceMax_ROW;

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

template <> struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <> struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};

inline __device__ uint8_t float_to_fp8e5m2(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      f_bits =
          c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits) +
                                    c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint32_t mant_odd = (f_bits >> 21) & 1;
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

inline __device__ uint8_t float_to_fp8e4m3(float f) {

  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;

  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = 0x7f;
  } else {

    if (f_bits < (UINT32_C(121) << 23)) {
      f_bits =
          c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits) +
                                    c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);

    } else {

      uint8_t mant_odd = (f_bits >> 20) & 1;

      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

      f_bits += mant_odd;

      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);

  return result;
}

inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c,
                                               float d) {
#if defined(__gfx938__)
  int32_t res;
  res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
  res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
  return res;
#else
  int32_t out;
  reinterpret_cast<uint8_t *>(&out)[0] = float_to_fp8e4m3(a);
  reinterpret_cast<uint8_t *>(&out)[1] = float_to_fp8e4m3(b);
  reinterpret_cast<uint8_t *>(&out)[2] = float_to_fp8e4m3(c);
  reinterpret_cast<uint8_t *>(&out)[3] = float_to_fp8e4m3(d);
  return out;
#endif
}

inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c,
                                               float d) {
#if defined(__gfx938__)
  int32_t res;
  res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
  res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
  return res;
#else
  int32_t out;
  reinterpret_cast<uint8_t *>(&out)[0] = float_to_fp8e5m2(a);
  reinterpret_cast<uint8_t *>(&out)[1] = float_to_fp8e5m2(b);
  reinterpret_cast<uint8_t *>(&out)[2] = float_to_fp8e5m2(c);
  reinterpret_cast<uint8_t *>(&out)[3] = float_to_fp8e5m2(d);
  return out;
#endif
}

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
#define VLLM_LDG(arg) __ldg(arg)
#else
#define VLLM_LDG(arg) *(arg)
#endif

template <typename T> __device__ __forceinline__ T silu_kernel(const T &x) {

  constexpr float LOG2E = 1.44269504088896340736f;
  return (T)(((float)x) / (1.0f + __builtin_amdgcn_exp2f(-((float)x) * LOG2E)));
}

template <typename scalar_t, bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t &x,
                                            const scalar_t &y) {
  return act_first ? silu_kernel(x) * y : x * silu_kernel(y);
}

template <typename scalar_t, bool act_first>
__device__ __forceinline__ float
compute_with_swiglu_limit_f(const scalar_t &x, const scalar_t &y,
                            const float limit) {

  float gate = act_first ? static_cast<float>(x) : static_cast<float>(y);
  float up = act_first ? static_cast<float>(y) : static_cast<float>(x);
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  constexpr float LOG2E = 1.44269504088896340736f;

  const float silu = gate * __builtin_amdgcn_rcpf(
                                1.0f + __builtin_amdgcn_exp2f(-gate * LOG2E));
  return silu * up;
}

template <typename scalar_t, bool act_first>
__device__ __forceinline__ scalar_t compute_with_swiglu_limit(
    const scalar_t &x, const scalar_t &y, const float limit) {
  return static_cast<scalar_t>(
      compute_with_swiglu_limit_f<scalar_t, act_first>(x, y, limit));
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void r09_activation_control(int64_t num_tokens, const int fp8type,
                                       uint8_t *__restrict__ out,
                                       const scalar_t *__restrict__ input,
                                       float *__restrict__ scales, const int d,
                                       const int topK,
                                       int32_t *num_local_tokens_tensor) {
  int64_t token_idx = blockIdx.x;
  int num_local_tokens = num_tokens;
  if (num_local_tokens_tensor != nullptr) {
    if (token_idx >= num_local_tokens_tensor[0] * topK)
      return;
  }
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

  int idx = threadIdx.x * VEC;
  int tidx = threadIdx.x;
  const int64_t x_index = token_idx * 2 * d + idx;
  const int64_t y_index = token_idx * d + idx;
  VecType *x1 = (VecType *)(input + x_index);
  VecType *x2 = (VecType *)(input + x_index + d);
  VecInt8Type *y = (VecInt8Type *)(out + y_index);
  scalar_t r_x1[VEC];
  scalar_t r_x2[VEC];
  float r_y[VEC];
  if (idx < d) {
    *(VecType *)r_x1 = *x1;
    *(VecType *)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
    }
  }
  float row_max = 0.f;
  if (idx < d) {
#pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      row_max = fmaxf(row_max, fabsf(r_y[ii]));
    }
  }

  row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
  __shared__ float s_token_scale;
  float fp8type_max_v;
  float min_scaling;

  if (fp8type == 0) {
    fp8type_max_v = quant_type_max_v<c10::Float8_e4m3fn>;
    min_scaling = min_scaling_factor<c10::Float8_e4m3fn>::val();
  } else {
    fp8type_max_v = quant_type_max_v<c10::Float8_e5m2>;
    min_scaling = min_scaling_factor<c10::Float8_e5m2>::val();
  }

  float scale = 0.0f;
  if (tidx == 0) {
    scale = fmaxf(row_max / fp8type_max_v, min_scaling);
    s_token_scale = scale;
    scales[token_idx] = scale;
  }
  __syncthreads();

  float inv_s = 1.0f / s_token_scale;

  uint8_t out_vec[VEC];
  int loop = VEC / 4;
  if (idx < d) {
#pragma unroll
    for (int ii = 0; ii < loop; ii++) {
      if (fp8type == 0)
        ((uint32_t *)out_vec)[ii] = float_to_fp8e4m3_x4(
            r_y[ii * 4] * inv_s, r_y[ii * 4 + 1] * inv_s,
            r_y[ii * 4 + 2] * inv_s, r_y[ii * 4 + 3] * inv_s);
      else
        ((uint32_t *)out_vec)[ii] = float_to_fp8e5m2_x4(
            r_y[ii * 4] * inv_s, r_y[ii * 4 + 1] * inv_s,
            r_y[ii * 4 + 2] * inv_s, r_y[ii * 4 + 3] * inv_s);
    }
    *y = *(VecInt8Type *)out_vec;
  }
}

template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void r09_activation_inverse_broadcast(
    int64_t num_tokens, const int fp8type, uint8_t *__restrict__ out,
    const scalar_t *__restrict__ input, float *__restrict__ scales, const int d,
    const int topK, int32_t *num_local_tokens_tensor) {
  int64_t token_idx = blockIdx.x;
  int num_local_tokens = num_tokens;
  if (num_local_tokens_tensor != nullptr) {
    if (token_idx >= num_local_tokens_tensor[0] * topK)
      return;
  }
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

  int idx = threadIdx.x * VEC;
  int tidx = threadIdx.x;
  const int64_t x_index = token_idx * 2 * d + idx;
  const int64_t y_index = token_idx * d + idx;
  VecType *x1 = (VecType *)(input + x_index);
  VecType *x2 = (VecType *)(input + x_index + d);
  VecInt8Type *y = (VecInt8Type *)(out + y_index);
  scalar_t r_x1[VEC];
  scalar_t r_x2[VEC];
  float r_y[VEC];
  if (idx < d) {
    *(VecType *)r_x1 = *x1;
    *(VecType *)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
    }
  }
  float row_max = 0.f;
  if (idx < d) {
#pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      row_max = fmaxf(row_max, fabsf(r_y[ii]));
    }
  }

  row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
  __shared__ float s_token_scale;
  float fp8type_max_v;
  float min_scaling;

  if (fp8type == 0) {
    fp8type_max_v = quant_type_max_v<c10::Float8_e4m3fn>;
    min_scaling = min_scaling_factor<c10::Float8_e4m3fn>::val();
  } else {
    fp8type_max_v = quant_type_max_v<c10::Float8_e5m2>;
    min_scaling = min_scaling_factor<c10::Float8_e5m2>::val();
  }

  float scale = 0.0f;
  if (tidx == 0) {
    scale = fmaxf(row_max / fp8type_max_v, min_scaling);
    s_token_scale = 1.0f / scale;
    scales[token_idx] = scale;
  }
  __syncthreads();

  float inv_s = s_token_scale;

  uint8_t out_vec[VEC];
  int loop = VEC / 4;
  if (idx < d) {
#pragma unroll
    for (int ii = 0; ii < loop; ii++) {
      if (fp8type == 0)
        ((uint32_t *)out_vec)[ii] = float_to_fp8e4m3_x4(
            r_y[ii * 4] * inv_s, r_y[ii * 4 + 1] * inv_s,
            r_y[ii * 4 + 2] * inv_s, r_y[ii * 4 + 3] * inv_s);
      else
        ((uint32_t *)out_vec)[ii] = float_to_fp8e5m2_x4(
            r_y[ii * 4] * inv_s, r_y[ii * 4 + 1] * inv_s,
            r_y[ii * 4 + 2] * inv_s, r_y[ii * 4 + 3] * inv_s);
    }
    *y = *(VecInt8Type *)out_vec;
  }
}

} // namespace lightop_glm53_activation

void glm53_activation(torch::Tensor x, torch::Tensor out, torch::Tensor scales, int mode) {
  TORCH_CHECK(mode == 0 || mode == 1);
  TORCH_CHECK(x.is_cuda() && out.is_cuda() && scales.is_cuda());
  TORCH_CHECK(x.device() == out.device() && x.device() == scales.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kByte &&
              scales.scalar_type() == at::kFloat);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 4096);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) &&
              out.size(1) == 2048);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == x.size(0) &&
              scales.size(1) == 1);
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() &&
              scales.is_contiguous());
  TORCH_CHECK(x.data_ptr() != out.data_ptr() &&
              x.data_ptr() != scales.data_ptr() &&
              out.data_ptr() != scales.data_ptr());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = x.size(0);
  TORCH_CHECK(rows >= 0 && rows <= 2147483647);
  if (!rows)
    return;
  if (mode == 0)
    lightop_glm53_activation::r09_activation_control<c10::BFloat16, 8, true, 256>
        <<<rows, 256, 0, stream>>>(rows, 0, out.data_ptr<uint8_t>(),
                                   x.data_ptr<c10::BFloat16>(),
                                   scales.data_ptr<float>(), 2048, 1, nullptr);
  else
    lightop_glm53_activation::r09_activation_inverse_broadcast<c10::BFloat16, 8, true, 256>
        <<<rows, 256, 0, stream>>>(rows, 0, out.data_ptr<uint8_t>(),
                                   x.data_ptr<c10::BFloat16>(),
                                   scales.data_ptr<float>(), 2048, 1, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
