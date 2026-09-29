// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../window.h"
#include "../../epilogue.h"
#include "../../params.h"

namespace lightop {
namespace mqa_fp8 {
using mqa_beta::tail_rows;
using mqa_beta::F2;
using mqa_beta::F4;
using mqa_beta::store1;
using mqa_beta::store4;
using mqa_beta::output_vector_aligned;
using mqa_beta::store_logits4;

using mqa_beta::Window;
using mqa_beta::TileWindow;
using mqa_beta::load_window;

template <bool EvenM, bool CheckFull = false>
__device__ __forceinline__ TileWindow classify_window(const MqaFp8Params &p, int64_t m0, int64_t n0,
                                                      int valid_m, int valid_n, int lane) {
    bool overlaps = false;
    bool covered = true;
    // All four waves examine the SAME M64 rows, hence take a CTA-uniform exit.
    if (EvenM || lane < valid_m) {
        const Window w = load_window(p.ks, p.ke, p.n, m0 + lane);
        overlaps = w.start < w.end && w.start < n0 + valid_n && w.end > n0;
        if constexpr (CheckFull)
            covered = w.start <= n0 && w.end >= n0 + valid_n;
    }
    // The conditional load must reconverge before this full-wave collective.
    const bool any = __ballot(overlaps) != 0;
    bool full = false;
    if constexpr (CheckFull)
        full = __ballot(!covered) == 0;
    return {any, full};
}

// Empty tiles still own all their logical output when fused clean is enabled.
// No Q/KV/scale access, LDS or cross-wave synchronization is needed here.
template <int AlignN>
__device__ __forceinline__ void fill_empty_tile(const MqaFp8Params &p, int64_t m0, int64_t n0,
                                                int valid_m, int valid_n, int wave, int lane) {
    const float ni = -__builtin_inff();
    const bool vector_out = output_vector_aligned<AlignN>(p.out, p.out_stride_m);
    for (int row = wave; row < valid_m; row += 4) {
        for (int col = AlignN * lane; col < valid_n; col += AlignN * 64) {
            float *dst = p.out + (m0 + row) * p.out_stride_m + n0 + col;
            if constexpr (AlignN == 4)
                store_logits4(dst, F4{ni, ni, ni, ni}, vector_out);
            else if constexpr (AlignN == 2) {
                const F2 value = {ni, ni};
                __builtin_memcpy(dst, &value, sizeof(value));
            } else
                store1(dst, ni);
        }
    }
}

template <bool Clean>
__device__ __forceinline__ TileWindow small_window(const MqaFp8Params &p, int64_t m0, int64_t n0,
                                                   int vm, int vn, int lane) {
    if (!p.ks && !p.ke)
        return {true, true};
    bool any = false, covered = true;
    if (lane < vm) {
        const Window w = load_window(p.ks, p.ke, p.n, m0 + lane);
        any = w.start < w.end && w.start < n0 + vn && w.end > n0;
        if constexpr (Clean)
            covered = w.start <= n0 && w.end >= n0 + vn;
    }
    // Every wave examines the same logical query rows, even when BM < 64.
    const bool active = __ballot(any) != 0;
    bool full = false;
    if constexpr (Clean)
        full = __ballot(!covered) == 0;
    return {active, full};
}

__device__ __forceinline__ void empty_output(const MqaFp8Params &p, int64_t m0, int64_t n0, int vm,
                                             int vn) {
    for (int i = threadIdx.x; i < vm * vn; i += 256)
        p.out[(m0 + i / vn) * p.out_stride_m + n0 + i % vn] = -__builtin_inff();
}

} // namespace mqa_fp8
} // namespace lightop
