// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace lightop::mqa_beta {
template <int Rows> __device__ __forceinline__ int tail_rows(int64_t remaining) {
    return remaining <= 0 ? 0 : remaining >= Rows ? Rows : int(remaining);
}

struct Window {
    int64_t start, end;
};

__device__ __forceinline__ Window load_window(const int32_t *ks, const int32_t *ke, int64_t n,
                                              int64_t row) {
    // Widen signed int32 before comparing with the int64 logical extent.
    int64_t s = ks ? int64_t(ks[row]) : 0;
    int64_t e = ke ? int64_t(ke[row]) : n;
    return {s < 0 ? 0 : (s > n ? n : s), e < 0 ? 0 : (e > n ? n : e)};
}

struct TileWindow {
    bool any, full;
};
template <bool EvenM, bool CheckFull = false, class Params>
__device__ __forceinline__ TileWindow classify_window(const Params &p, int64_t m0, int64_t n0,
                                                      int valid_m, int valid_n, int lane) {
    bool overlaps = false;
    bool covered = true;
    // Every wave examines the same logical rows, so a CTA takes a uniform exit.
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

template <bool Clean, class Params>
__device__ __forceinline__ TileWindow small_window(const Params &p, int64_t m0, int64_t n0, int vm,
                                                   int vn, int lane) {
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

} // namespace lightop::mqa_beta
