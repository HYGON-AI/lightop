// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <c10/util/floating_point_utils.h>
#include <cstdint>
namespace lightop::quant {
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
  int32_t res = 0;
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
  int32_t res = 0;
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

inline __device__ uint16_t float_to_fp8e4m3_x2(float a, float b) {
#if defined(__gfx938__)
  return static_cast<uint16_t>(__builtin_hcu_cvt_pk_fp8_f32(a, b, 0, false));
#else
  return static_cast<uint16_t>(float_to_fp8e4m3(a)) |
         (static_cast<uint16_t>(float_to_fp8e4m3(b)) << 8);
#endif
}
} // namespace lightop::quant
