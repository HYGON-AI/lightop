// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"
#include <limits>

namespace lightop::mqa_int8 {

enum class AutoArea : uint8_t { Logical, PaddedM16, PaddedM32, PaddedM64 };
enum AutoFlag : uint32_t {
    N128Aligned = 1 << 0,
    N128Tail = 1 << 1,
    N256Aligned = 1 << 2,
    ShortMTail = 1 << 3,
    LargeLowHeadGrid = 1 << 4,
    // Padding M to 64 rows adds no rows beyond padding it to 32.
    M64NoExtraPad = 1 << 5,
};

inline constexpr int32_t kMaxDim = std::numeric_limits<int32_t>::max();
inline constexpr int64_t kMaxArea = std::numeric_limits<int64_t>::max();

// M/N/H and area bounds are inclusive. As in FP8, the first matching row wins.
// Area and alignment flags preserve the measured INT8 tail-tile choices.
// Tiled group_heads is capped by H after matching; small-KV configs are exact.
// Both cleaning modes use the same rules. KS/KE values are not read on the host.
struct AutoRule {
    int32_t h_min, h_max, m_min, m_max, n_min, n_max;
    KernelConfig config;
    int64_t area_min = 0, area_max = kMaxArea;
    AutoArea area = AutoArea::Logical;
    uint32_t flags = 0;
};

inline constexpr AutoRule kAutoRules[] = {
    // Small KV: H8 uses one wave, two waves, or paired queries as M grows.
    {8, 8, 16, 16, 0, 128, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {8, 8, 17, 32, 0, 128, {KernelKind::SmallKV, 2, 16, 1, 2, false}},
    {8, 8, 33, 128, 0, 128, {KernelKind::SmallKV, 2, 16, 1, 1, true}},
    {64, 64, 16, 32, 0, 128, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {64, 64, 33, 128, 0, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    // H16 leaves M>96 and N>96 on the tiled path.
    {16, 16, 16, 128, 16, 32, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {16, 16, 16, 32, 33, 96, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {16, 16, 16, 96, 33, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    {16, 16, 97, 128, 33, 96, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    // H64 was handled above; these rows select the H32/H128 schedules.
    {32, 128, 16, 128, 16, 32, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {32, 128, 16, 32, 33, 96, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {32, 128, 16, 128, 33, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},

    // Tiled 64x256 needs at least ~1.5 waves of CTAs (480 at 4 CTAs/CU). H<=8
    // gains 6-43% for N in [8K, 64K]; at N=128K (K > L2) it loses up to 22%.
    // H>=16 gains 2-19%, which a partial last 64-row tile can erase (M=321: -3%).
    // H8 keeps M=65..96 on BM32: BM64 rounds 96 rows up to 128.
    {1, 4, 0, kMaxDim, 8192, 65536, {KernelKind::Tiled, 64, 256, 1}, 7500000, kMaxArea,
     AutoArea::PaddedM64},
    {8, 8, 97, kMaxDim, 8192, 65536, {KernelKind::Tiled, 64, 256, 1}, 7500000, kMaxArea,
     AutoArea::PaddedM64},
    {16, 128, 0, kMaxDim, 8192, kMaxDim, {KernelKind::Tiled, 64, 256, 1}, 11000000, kMaxArea,
     AutoArea::PaddedM64, M64NoExtraPad},
    // H<=8, M>=32, N>=4096, and padded-M32 area>=524288.
    // Match the narrow-grid and short-M-tail exceptions before wider tiles.
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 16, 64, 4},
     524288, 524288, AutoArea::PaddedM32, N128Aligned},
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     0, 1048576, AutoArea::PaddedM16, LargeLowHeadGrid | ShortMTail},
    {1, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 64, 2},
     524288, 1048576, AutoArea::PaddedM32},
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 64, 2},
     524288, 1310720, AutoArea::PaddedM32, N128Tail},
    {1, 8, 64, kMaxDim, 131072, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
    {1, 8, 1024, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 32, 256, 2},
     0, kMaxArea, AutoArea::Logical, N256Aligned},
    {1, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 128, 2},
     524288, kMaxArea, AutoArea::PaddedM32},

    // Short M grids: avoid redundant rows in a partially filled BM32 tile.
    {64, 128, 24, 128, 0, kMaxDim, {KernelKind::Tiled, 16, 256, 4}, 3145728, 8388608},
    {32, 32, 24, 128, 0, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     4194305, 8388608, AutoArea::Logical, ShortMTail},
    {16, 16, 24, 80, 0, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     4194305, 8388608, AutoArea::Logical, ShortMTail},

    // General tiled rules, after all small-KV and padded-grid exceptions.
    {1, 128, 0, kMaxDim, 0, 256, {KernelKind::Tiled, 16, 64, 4}},
    {32, 32, 1024, kMaxDim, 131072, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
    {1, 128, 1024, kMaxDim, 8192, kMaxDim, {KernelKind::Tiled, 32, 256, 2},
     0, kMaxArea, AutoArea::Logical, N256Aligned},
    {1, 128, 1024, kMaxDim, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2}},
    {1, 4, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2}, 524288},
    {8, 8, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2},
     524288, kMaxArea, AutoArea::Logical, N128Aligned},
    {1, 8, 0, 31, 32769, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    {1, 8, 0, 1023, 1024, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     0, kMaxArea, AutoArea::Logical, N128Tail},
    {16, 128, 0, 1023, 257, 65535, {KernelKind::Tiled, 16, 64, 4}, 0, 524288},
    {16, 128, 0, 1023, 257, kMaxDim, {KernelKind::Tiled, 16, 128, 4}, 0, 4194304},
    {16, 128, 0, 31, 257, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    {16, 128, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
};

inline void mqa_auto_config(const MqaInt8Params& p, KernelConfig& c) {
    c = {KernelKind::Tiled, 16, 64, 4};
    const int64_t areas[] = {p.m * p.n, ((p.m + 15) / 16) * 16 * p.n,
                             ((p.m + 31) / 32) * 32 * p.n, ((p.m + 63) / 64) * 64 * p.n};
    const uint32_t flags = (p.n % 128 == 0 ? N128Aligned : N128Tail) |
        (p.n % 256 == 0 ? N256Aligned : 0u) |
        (p.m % 32 > 0 && p.m % 32 <= 16 ? ShortMTail : 0u) |
        (p.h <= 8 && p.m >= 32 && p.n >= 4096 && areas[2] >= 524288 ? LargeLowHeadGrid : 0u) |
        (areas[3] == areas[2] ? M64NoExtraPad : 0u);
    // make_grid accepts BM=64 only when a 64-row tile's weight offsets fit in 32 bits.
    const bool m64_ok = p.weights_stride_m >= p.h && p.weights_stride_m <= (int64_t(1) << 23);
    for (const AutoRule &r : kAutoRules) {
        if (p.h < r.h_min || p.h > r.h_max || p.m < r.m_min || p.m > r.m_max ||
            p.n < r.n_min || p.n > r.n_max)
            continue;
        const int64_t area = areas[static_cast<unsigned>(r.area)];
        if (area < r.area_min || area > r.area_max || (flags & r.flags) != r.flags)
            continue;
        if (r.config.block_m == 64 && !m64_ok)
            continue;
        c = r.config;
        break;
    }
    if (c.kind == KernelKind::Tiled) {
        // Tiled kernels never pair queries; keep this explicit after rule selection.
        c.paired_queries = false;
        if (p.h < c.group_heads)
            c.group_heads = static_cast<int>(p.h);
    }
    return;
}

inline KernelConfig resolve(const MqaInt8Params &p) {
    KernelConfig config;
    mqa_auto_config(p, config);
    return config;
}
} // namespace lightop::mqa_int8
