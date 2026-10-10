// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"

namespace lightop::mqa_int8 {
// Select among the existing short-KV and tiled schedules on gfx936.
inline KernelConfig resolve(const MqaInt8Params &p) {
    KernelConfig c;
    if (p.m >= 16 && p.m <= 128 && p.n <= 128 && (p.h == 8 || p.h == 64)) {
        c.kind = KernelKind::SmallKV;
        c.block_n = p.h == 64 && p.m > 32 ? 32 : 16;
        c.waves = p.h == 8 && p.m > 16 && p.m <= 32 ? 2 : 1;
        c.paired_queries = p.h == 8 && p.m > 32;
        c.block_m = c.waves * (c.paired_queries ? 2 : 1);
        return c;
    }
    if (p.m >= 16 && p.m <= 128 && p.n >= 16 && p.n <= 128 &&
        (p.h == 32 || p.h == 128 || (p.h == 16 && (p.m <= 96 || p.n <= 96)))) {
        // Short-KV head-parallel MMAC avoids the underfilled tiled grid.
        // H16 keeps the upper M/N corner tiled for full-window performance.
        c.kind = KernelKind::SmallKV;
        c.block_m = 1;
        c.block_n = p.n <= 32 || (p.m <= 32 && p.n <= 96) ? 16 : 32;
        c.waves = 1;
        return c;
    }
    const int gh4 = p.h < 4 ? p.h : 4;
    const int gh2 = p.h < 2 ? p.h : 2;
    const int64_t area = int64_t(p.m) * p.n;
    if (p.h <= 8 && p.m >= 32 && p.n >= 4096) {
        // Count the padded rows: a short final M tile can favor BM16.
        const int64_t area32 = ((int64_t(p.m) + 31) / 32) * 32 * p.n;
        const int64_t area16 = ((int64_t(p.m) + 15) / 16) * 16 * p.n;
        if (area32 >= 524288) {
            c.block_m = 32;
            c.block_n = 128;
            c.group_heads = gh2;
            if (p.h == 8 && area32 <= 524288 && p.n % 128 == 0) {
                c.block_m = 16;
                c.block_n = 64;
                c.group_heads = gh4;
            } else if (p.h == 8 && p.m % 32 > 0 && p.m % 32 <= 16 && area16 <= 1048576) {
                c.block_m = 16;
                c.group_heads = gh4;
            } else if (area32 <= 1048576 ||
                       (p.h == 8 && p.n % 128 != 0 && area32 <= 1310720)) {
                // A partial N tile favors BN64 over a wider, less populated grid.
                c.block_n = 64;
            } else if ((p.n >= 131072 && p.m >= 64) ||
                       (p.m >= 1024 && p.n >= 65536 && p.n % 256 == 0)) {
                // Reuse Q across more columns once the N grid is sufficiently large.
                c.block_n = 256;
            }
            return c;
        }
    }
    c.group_heads = gh4;
    if (p.n <= 256) {
        c.block_n = 64;
    } else if (p.m >= 1024) {
        c.block_m = 32;
        c.group_heads = gh2;
        // Ordinary tiled output handles partial N tiles, including N%4 tails.
        c.block_n = (p.h == 32 && p.n >= 131072) || (p.n % 256 == 0 && p.n >= 8192) ? 256 : 128;
    } else if (p.h <= 8) {
        if (p.m >= 32 && area >= 524288 && (p.n % 128 == 0 || p.h <= 4)) {
            c.block_m = 32;
            c.block_n = 128;
            c.group_heads = gh2;
        } else {
            c.block_n = (p.m < 32 && p.n > 32768) || (p.n % 128 != 0 && p.n >= 1024) ? 128 : 64;
        }
    } else {
        if (area <= 524288 && p.n < 65536) {
            c.block_n = 64;
        } else if (area <= 4194304 || p.m < 32) {
            c.block_n = 128;
        } else {
            c.block_m = 32;
            c.block_n = 256;
            c.group_heads = gh2;
        }
    }
    if (p.m >= 24 && p.m <= 128 && area >= 3145728 && area <= 8388608) {
        // With a short M grid, BM16 avoids redundant rows in a partial BM32 tile.
        if (p.h >= 64) {
            c.block_m = 16;
            c.block_n = 256;
            c.group_heads = gh4;
        } else if (p.h >= 16 && area > 4194304 && p.m % 32 > 0 && p.m % 32 <= 16 &&
                   (p.h >= 32 || p.m <= 80)) {
            // At H16, require enough saved M padding to offset the smaller tile.
            c.block_m = 16;
            c.block_n = 128;
            c.group_heads = gh4;
        }
    }
    return c;
}
} // namespace lightop::mqa_int8
