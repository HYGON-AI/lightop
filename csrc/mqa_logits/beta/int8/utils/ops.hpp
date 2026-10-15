// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../arch.h"
#include "../../window.h"
#include "../../epilogue.h"

namespace lightop::mqa_int8 {
using mqa_beta::classify_window;
using mqa_beta::F4;
using mqa_beta::fill_empty_tile;
using mqa_beta::I2;
using mqa_beta::I4;
using mqa_beta::load_window;
using mqa_beta::output_vector_aligned;
using mqa_beta::small_window;
using mqa_beta::store1;
using mqa_beta::store_logits4;
using mqa_beta::tail_rows;
using mqa_beta::TileWindow;
using mqa_beta::Window;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Public row views guarantee only four-byte alignment, as in FP8.
// memcpy avoids asserting the stronger C++ vector-pointer alignment.
template <class Vector, class Scalar>
__device__ __forceinline__ Vector load_aligned4(const Scalar *src) {
    Vector value;
    const auto *aligned = static_cast<const Scalar *>(__builtin_assume_aligned(src, 4));
    __builtin_memcpy(&value, aligned, sizeof(value));
    return value;
}

// Rebase at the tile so every lane offset fits in 32 bits even for a large KV.
__device__ __forceinline__ I4 make_kv_resource(const void *tile) {
    const uint64_t address = reinterpret_cast<uint64_t>(tile);
    return {static_cast<int>(address), static_cast<int>(address >> 32), -1, 0x00020000};
}

// Only KV uses SLC. The builtin preserves compiler-managed load dependencies;
// GLC stays off and Q/scales retain their ordinary four-byte-aligned loads.
__device__ __forceinline__ I2 load_kv_slc(I4 resource, int offset) {
    return __builtin_amdgcn_buffer_load_dwordx2(resource, 0, offset, false, true);
}

// Signed INT8 MMAC shares the non-LIT layout on gfx936 and gfx938.
__device__ __forceinline__ void mmac_i8(const I2 &a, const I2 &b, I4 &c) {
#if defined(__gfx936__) || defined(__gfx938__)
    c = __builtin_hcu_mmac_i32_16x16x32_i8(a, b, c);
#endif
}

// For signed INT8 and D=128, every partial dot is in [-2^21, 2^21].
// Adding its integer value to these bits keeps the FP32 exponent unchanged,
// so bit_cast<float>(biased_dot) - 1.5*2^23 is exact. Clamp before subtracting.
constexpr int kDotBias = 0x4b400000;
constexpr float kDotBiasValue = 12582912.0f;
static_assert(128 * 128 * 128 < (1 << 22));

__device__ __forceinline__ I4 dot_init() {
    return I4{kDotBias, kDotBias, kDotBias, kDotBias};
}

__device__ __forceinline__ void relu_weighted_sum(const I4 &dot, float w, F4 &accum) {
    using pair_t = float __attribute__((ext_vector_type(2)));
    const pair_t weight = {w, w};
    const pair_t bias = {kDotBiasValue, kDotBiasValue};
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        pair_t value = {__builtin_bit_cast(float, max(dot[i], kDotBias)),
                        __builtin_bit_cast(float, max(dot[i + 1], kDotBias))};
        value -= bias;
        pair_t sum = {accum[i], accum[i + 1]};
        sum = value * weight + sum;
        accum[i] = sum[0];
        accum[i + 1] = sum[1];
    }
}

__device__ __forceinline__ void prefetch4(I4 *dst, const int8_t *ptr, int offset, bool valid) {
    using resource_t = uint32_t __attribute__((ext_vector_type(4)));
    const uint64_t address = reinterpret_cast<uint64_t>(ptr);
    resource_t resource = {
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32)) + 0x41000000u,
        0x80000000u, 0x00020000u};
    if (valid) {
        I4 value;
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen"
                     : "=v"(value)
                     : "v"(offset), "s"(resource)
                     : "memory");
        *dst = value;
    }
}

#endif
} // namespace lightop::mqa_int8
