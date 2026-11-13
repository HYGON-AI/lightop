// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hcu_mmac.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>

#if defined(__gfx938__)
constexpr float LOG2_E_VAL = 1.44269504089f;
__device__ inline float exp_func(float x) {
  return __builtin_amdgcn_exp2f(x * LOG2_E_VAL);
}
#else
__device__ inline float exp_func(float x) { return __expf(x); }
#endif

namespace at {
namespace native {

// Helpers used by the BF16 mega-gate implementation.
// Keep the standalone sqrtsoftplus gate kernel in its original source file.
// BF16/FP16 matrix instructions are shared with GEMM.
using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
struct half4x2 {
  half4_t data[2];
};

struct half2x4 {
  half2_t data[4];
};

template <bool is_half = true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a,
                                           const half4_t& reg_b,
                                           float4_t& reg_c) {
#if defined(__gfx936__) || (defined(__gfx928__) || defined(__gfx92a__))
  if constexpr (is_half)
    reg_c = HCU_MMAC_F32_F16(reg_a, reg_b, reg_c);
  else
    reg_c = HCU_MMAC_F32_BF16(*(v4bh*)&reg_a,
                                                   *(v4bh*)&reg_b, reg_c);
#elif defined(__gfx938__)
  if constexpr (is_half)
    reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c,
                                                        false, false);
  else
    reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(
        *(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
#endif
}

#ifndef USE_ROCM
#define LIGHTOP_SHFL_XOR_WIDTH(var, lane_mask, width) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#define LIGHTOP_SHFL_BCAST_WIDTH(var, src_lane, width) \
  __shfl_sync(uint32_t(-1), var, src_lane, width)
#else
#define LIGHTOP_SHFL_XOR_WIDTH(var, lane_mask, width) \
  __shfl_xor(var, lane_mask, width)
#define LIGHTOP_SHFL_BCAST_WIDTH(var, src_lane, width) \
  __shfl(var, src_lane, width)
#endif

}  // namespace native
}  // namespace at

namespace at::native::moe_gate {

// Four-element sorting network with a packed permutation. The algorithm is
// adapted to HCU wave widths and LightOp's NaN, tie and normalization contract.
__device__ __forceinline__ void order_pair(float& a, int& ia, float& b,
                                           int& ib) {
  const bool swap = b > a || (b == a && ib < ia);
  const float old_a = a;
  const int old_ia = ia;
  a = swap ? b : a;
  ia = swap ? ib : ia;
  b = swap ? old_a : b;
  ib = swap ? old_ia : ib;
}

__device__ __forceinline__ unsigned sort_four(float4& v) {
  int a = 0, b = 1, c = 2, d = 3;
  order_pair(v.x, a, v.y, b);
  order_pair(v.z, c, v.w, d);
  order_pair(v.x, a, v.z, c);
  order_pair(v.y, b, v.w, d);
  order_pair(v.y, b, v.z, c);
  return unsigned(a | b << 2 | c << 4 | d << 6);
}

template <int EXPERT_CAPACITY, int TOP_K, bool PADDED>
__device__ __forceinline__ void topk_row(
    const float* __restrict__ input, const float* __restrict__ bias,
    const float* __restrict__ logits, const bool* __restrict__ mask,
    float* __restrict__ output, int64_t* __restrict__ ids, int row, int rows,
    int num_experts, bool renormalize, float scale, bool apply_scale) {
  constexpr int WIDTH = 32;
  static_assert(TOP_K > 0 && TOP_K <= WIDTH);
  static_assert(EXPERT_CAPACITY % (WIDTH * 4) == 0 && EXPERT_CAPACITY <= 512);
  const int experts = PADDED ? num_experts : EXPERT_CAPACITY;
  if (row >= rows) return;
  constexpr int GROUPS = EXPERT_CAPACITY / (WIDTH * 4);
  const int lane = threadIdx.x % WIDTH;
  float4 ranked[GROUPS];
  unsigned permutations[GROUPS];
#pragma unroll
  for (int group = 0; group < GROUPS; ++group) {
    const int first = group * WIDTH * 4 + lane * 4;
    float v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float4 b = {0, 0, 0, 0};
    if (first < experts) {
      // Without a bias, rank raw logits as DeepGEMM does. The activation is
      // monotonic, but rounding its output can otherwise introduce new ties.
      const float* ranking = bias != nullptr ? input : logits;
      const float4 r =
          *reinterpret_cast<const float4*>(ranking + row * experts + first);
      v[0] = r.x;
      v[1] = r.y;
      v[2] = r.z;
      v[3] = r.w;
      if (bias != nullptr) b = *reinterpret_cast<const float4*>(bias + first);
    }
    const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] += bv[i];
      if (v[i] != v[i]) v[i] = INFINITY;
    }
    ranked[group] = {v[0], v[1], v[2], v[3]};
    permutations[group] = sort_four(ranked[group]);
  }
  unsigned cursors = 0;
  int selected = 0;
#pragma unroll
  for (int rank = 0; rank < TOP_K; ++rank) {
    float best = -INFINITY;
    int expert = EXPERT_CAPACITY, local_group = 0;
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
      const unsigned cursor = (cursors >> (3 * group)) & 7;
      const int first = group * WIDTH * 4 + lane * 4;
      const float4 v = ranked[group];
      const float value = (cursor & 2) ? ((cursor & 1) ? v.w : v.z)
                                       : ((cursor & 1) ? v.y : v.x);
      const int candidate = first + ((permutations[group] >> (2 * cursor)) & 3);
      if (cursor < 4 && first < experts &&
          (value > best || (value == best && candidate < expert))) {
        best = value;
        expert = candidate;
        local_group = group;
      }
    }
    const int local_expert = expert;
    {
      // Canonicalize signed zero so the integer order preserves float ties.
      // NaNs have already become +Inf, matching the existing gate contract.
      unsigned bits = __float_as_uint(best == 0.0f ? 0.0f : best);
      const unsigned ordered =
          (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
      unsigned long long key =
          (static_cast<unsigned long long>(ordered) << 32) |
          (0xffffffffu - static_cast<unsigned>(expert));
#pragma unroll
      for (int delta = WIDTH / 2; delta > 0; delta /= 2) {
        const auto other = LIGHTOP_SHFL_XOR_WIDTH(key, delta, WIDTH);
        key = other > key ? other : key;
      }
      expert = int(0xffffffffu - static_cast<unsigned>(key));
    }
    if (local_expert == expert) cursors += 1u << (3 * local_group);
    if (lane == rank) selected = expert;
  }
  // Re-read only the selected values. Retain the baseline's serial sum
  // order instead of changing it to a tree reduction or adding an epsilon.
  float value = 0.0f;
  if (lane < TOP_K) {
    value = input[row * experts + selected];
  }
  float sum = 0.0f;
#pragma unroll
  for (int rank = 0; rank < TOP_K; ++rank)
    sum += LIGHTOP_SHFL_BCAST_WIDTH(value, rank, WIDTH);
  if (lane < TOP_K) {
    const float norm = renormalize && sum > 0.0f ? sum : 1.0f;
    const bool valid = mask == nullptr || mask[row];
    output[row * TOP_K + lane] =
        valid ? value / norm * (apply_scale ? scale : 1.0f) : 0.0f;
    ids[row * TOP_K + lane] = valid ? selected : -1;
  }
}
}  // namespace at::native::moe_gate
