// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../params.h"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace lightop::mqa_int8 {
enum class KernelKind : uint32_t { SmallKV, Tiled };

struct KernelConfig {
    KernelKind kind = KernelKind::Tiled;
    int block_m = 16;
    int block_n = 64;
    int group_heads = 1;
    int waves = 4;
    bool paired_queries = false;
};

struct LaunchGrid {
    uint32_t x = 0, y = 0;
};

inline bool make_grid(const MqaInt8Params &p, const KernelConfig &c, uint32_t max_x, uint32_t max_y,
                      LaunchGrid &grid) {
    grid = {};
    if (p.m < 0 || p.n < 0 || p.d != 128 || p.h < 1 || p.h > 128 || (p.h & (p.h - 1)))
        return false;
    if (c.kind == KernelKind::SmallKV) {
        if (p.m < 16 || p.m > 128 || p.n > 128 || (p.h != 8 && p.h != 16 && p.h != 32 && p.h != 64 && p.h != 128) ||
            (c.block_n != 16 && c.block_n != 32) || (c.waves != 1 && c.waves != 2) ||
            (c.paired_queries && p.h != 8) || c.block_m != c.waves * (c.paired_queries ? 2 : 1))
            return false;
    } else if (c.kind == KernelKind::Tiled) {
        if ((c.block_m != 16 && c.block_m != 32) ||
            (c.block_n != 64 && c.block_n != 128 && c.block_n != 256) ||
            (c.group_heads != 1 && c.group_heads != 2 && c.group_heads != 4) ||
            p.h % c.group_heads || c.waves != 4 || c.paired_queries)
            return false;
    } else {
        return false;
    }
    if (!p.m || !p.n)
        return true;
    const uint64_t rows = uint64_t(p.m) / c.block_m + (p.m % c.block_m != 0);
    const uint64_t cols = uint64_t(p.n) / c.block_n + (p.n % c.block_n != 0);
    const uint64_t x = c.kind == KernelKind::SmallKV ? rows : cols;
    const uint64_t y = c.kind == KernelKind::SmallKV ? cols : rows;
    if (x > max_x || y > max_y)
        return false;
    grid = {uint32_t(x), uint32_t(y)};
    return true;
}

hipError_t launch(const MqaInt8Params &p, const KernelConfig &c, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y);
} // namespace lightop::mqa_int8
