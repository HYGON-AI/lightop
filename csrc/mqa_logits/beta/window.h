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
} // namespace lightop::mqa_beta
