// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../params.h"
#include <cstdint>

namespace lightop::mqa_fp8 {
enum class KernelKind : uint32_t { M64 = 1, M16 = 2, QH = 3 };

struct KernelConfig {
    KernelKind kind = KernelKind::M64;
    uint32_t block_n = 256;
};

struct LaunchGrid {
    uint32_t x = 0;
    uint32_t y = 0;
};

// Used once by dispatch, for both device-limit validation and the actual launch.
inline bool make_grid(const MqaFp8Params &p, const KernelConfig &c, uint32_t max_x, uint32_t max_y,
                      LaunchGrid &grid) {
    grid = {};
    if (p.m < 0 || p.n < 0 || p.d != 128 || p.h < 1 || p.h > 128 || (p.h & (p.h - 1)) ||
        p.clean_logits > 1) {
        return false;
    }
    if (c.block_n != 64 && c.block_n != 128 && c.block_n != 256)
        return false;
    uint32_t block_m;
    switch (c.kind) {
    case KernelKind::M64:
        if (c.block_n != 256)
            return false;
        block_m = 64;
        break;
    case KernelKind::M16:
        block_m = 16;
        break;
    case KernelKind::QH:
        if (p.h != 8 && p.h != 16 && p.h != 32 && p.h != 64)
            return false;
        block_m = 64 / p.h;
        break;
    default:
        return false;
    }
    if (p.m == 0 || p.n == 0)
        return true;
    // Division before addition avoids overflow near INT64_MAX.
    const uint64_t x = p.n / c.block_n + (p.n % c.block_n != 0);
    const uint64_t y = p.m / block_m + (p.m % block_m != 0);
    if (x > max_x || y > max_y)
        return false;
    grid = {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
    return true;
}
} // namespace lightop::mqa_fp8
