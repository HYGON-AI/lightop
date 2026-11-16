// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"
#include <limits>

namespace lightop::mqa_bf16 {

enum class AutoGrid : uint8_t { None, M16N64, M16N128, M16N256, M32N128 };

inline constexpr int32_t kMaxDim = std::numeric_limits<int32_t>::max();
inline constexpr int64_t kMaxGrid = std::numeric_limits<int64_t>::max();

// M/N/H and CTA-count bounds are inclusive; the first matching row wins, as in
// INT8. Grid counts use padded tiles, not M*N, so partial tiles keep their route.
// Refined with the 2026-10-19 gfx936 M/N sweep; stable existing schedules are
// retained outside those intervals. These ranges describe tile utilization,
// CTA occupancy and K reuse, rather than an exact-shape benchmark lookup.
// KS/KE values and the cleaning mode do not affect host-side rule selection.
struct AutoRule {
    int32_t h_min, h_max, m_min, m_max, n_min, n_max;
    KernelConfig config;
    int64_t grid_min = 0, grid_max = kMaxGrid;
    AutoGrid grid = AutoGrid::None;
};

// Measured on gfx936 with the 2026-10-20 9,745-shape M/N/H sweep using
// causal and full windows. Rule order and thresholds preserve that measured table.
inline constexpr AutoRule kAutoRules[] = {
    // Short H8 windows: grouped BM16 amortizes cooperative Q staging.
    {8, 8, 24, 512, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Larger H8 batches: BN128 reduces the number of short-window CTAs.
    {8, 8, 513, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 128, 2}},
    // Short H16 windows: grouped BM16 covers the moderate-M interval.
    {16, 16, 32, 1023, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Larger H16 batches use BN128 with two heads per group.
    {16, 16, 1024, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 128, 2}},
    // High H and enough rows amortize the grouped tiled schedule.
    {32, 128, 128, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // High H with fewer rows benefits from partitioning heads within each CTA.
    {32, 128, 64, 127, 384, 512, {KernelKind::SplitHead, 16, 64, 1, 8}},
    // Very short queries use one-wave small-KV tiles.
    {1, 16, 0, 4, 513, 768, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // Tiny low-H queries use narrower BN64 tiles for this KV interval.
    {1, 8, 2, 7, 4096, 8192, {KernelKind::Tiled, 16, 64, 4}},
    // Longer KV intervals amortize BN128 even for tiny query batches.
    {1, 16, 2, 7, 8193, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Tiny H32 queries use BN128 once KV is sufficiently long.
    {32, 32, 2, 7, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // H64/H128 tiny batches use BN128 in the long-KV interval.
    {64, 128, 4, 7, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Single-row low-H batches retain the measured BN128 schedule.
    {1, 4, 0, 1, 8192, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Single-row H8 batches need a longer KV interval to amortize BN128.
    {8, 8, 0, 1, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // H8 moderate batches use BM32/BN128 in this KV interval.
    {8, 8, 64, 256, 16384, 32768, {KernelKind::Tiled, 32, 128, 1}},
    // Large H8 batches reuse K across BM64 rows.
    {8, 8, 1024, kMaxDim, 16384, 65535, {KernelKind::Tiled, 64, 256, 1}},
    // H8 short batches use BM32 in the measured long-KV interval.
    {8, 8, 32, 63, 49152, 98304, {KernelKind::Tiled, 32, 128, 1}},
    // Large H16 batches and long KV amortize BM64/BN256.
    {16, 16, 1536, kMaxDim, 32768, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Moderate H16 batches use BM32/BN256 for long KV.
    {16, 16, 256, 1535, 16384, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // High H moderate batches use BM32/BN256 for long KV.
    {32, 128, 256, 1023, 32768, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // Short high-H query batches use BN128 in this KV interval.
    {16, 128, 8, 16, 8193, 16384, {KernelKind::Tiled, 16, 128, 4}},
    // High-H moderate batches retain the measured BN128 schedule.
    {16, 128, 49, 96, 2049, 4096, {KernelKind::Tiled, 16, 128, 4}},
    // Low H with 17-32 rows uses BM32 to reuse K.
    {1, 4, 17, 32, 32768, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Low H with 33-63 rows amortizes BM64 when KV is long.
    {1, 4, 33, 63, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Remaining queries below eight rows use one-wave small-KV tiles.
    {1, 128, 0, 7, 513, kMaxDim, {KernelKind::SmallKV, 1, 64, 1, 1}},
    // Remaining high-H short windows use wider small-KV tiles.
    {32, 128, 0, kMaxDim, 0, 512, {KernelKind::SmallKV, 1, 32, 1, 1}},
    // H8 pairs queries when the short-window batch has enough rows.
    {8, 8, 32, kMaxDim, 0, 512, {KernelKind::SmallKV, 2, 16, 1, 1, true}},
    // Remaining H8/H16 short windows use one query per wave.
    {8, 16, 0, kMaxDim, 0, 512, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // Tiny low-H short windows use one query per wave.
    {1, 4, 0, 15, 0, 512, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // At most 80 CTAs on gfx936: partition heads within each CTA.
    {32, 128, 0, kMaxDim, 0, kMaxDim,
     {KernelKind::SplitHead, 16, 64, 1, 8}, 0, 80, AutoGrid::M16N64},
    // H8 long KV reuses K across ordinary BM64 rows.
    {8, 8, 64, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Large high-H batches with short KV use grouped BM16/BN128.
    {16, 128, 1024, kMaxDim, 0, 736, {KernelKind::Tiled, 16, 128, 2}},
    // The remaining large high-H batches use BM32/BN256.
    {16, 128, 1024, kMaxDim, 737, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // Remaining H8 long-window batches use BM32/BN128.
    {8, 8, 64, kMaxDim, 16384, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Remaining short windows use compact BM16/BN64 tiles.
    {1, 128, 0, kMaxDim, 0, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Low-H long KV amortizes ordinary BM64/BN256.
    {1, 4, 64, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Shrink BN128 to BN64 when the larger tile would leave fewer than 80 CTAs.
    {1, 8, 64, kMaxDim, 513, kMaxDim, {KernelKind::Tiled, 32, 64, 1}, 0, 79, AutoGrid::M32N128},
    // Otherwise low-H batches of at least 64 rows use BM32/BN128.
    {1, 8, 64, kMaxDim, 513, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Low-H batches below 32 rows retain GH2 for long KV.
    {1, 8, 0, 31, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Short low-H batches shrink BN when BN128 would leave fewer than 80 CTAs.
    {1, 8, 0, 63, 513, kMaxDim, {KernelKind::Tiled, 16, 64, 4}, 0, 79, AutoGrid::M16N128},
    // Otherwise short low-H batches use BM16/BN128 with grouped heads.
    {1, 8, 0, 63, 513, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    // High H: shrink BN128 below 40 CTAs.
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 64, 4}, 0, 39, AutoGrid::M16N128},
    // High H: shrink BN256 below 64 CTAs.
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 128, 4}, 0, 63, AutoGrid::M16N256},
    // Up to 128 BN256 CTAs retain GH4.
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 256, 4}, 0, 128, AutoGrid::M16N256},
    // Larger BN256 grids use GH2 to reduce register and LDS pressure.
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 256, 2}},
};

inline void mqa_auto_config(const MqaBf16Params &p, KernelConfig &c) {
    c = {KernelKind::Tiled, 16, 64, 4};
    // Shared params store int64 dimensions; the public BF16 entry is int32.
    // Reject larger internal inputs in make_grid before any kernel launch.
    if (p.m < 0 || p.n < 0 || p.m > kMaxDim || p.n > kMaxDim)
        return;
    // The checked range keeps rounded CTA counts and their products in int64.
    const int64_t m16 = (int64_t(p.m) + 15) / 16;
    const int64_t m32 = (int64_t(p.m) + 31) / 32;
    const int64_t n64 = (int64_t(p.n) + 63) / 64;
    const int64_t n128 = (int64_t(p.n) + 127) / 128;
    const int64_t n256 = (int64_t(p.n) + 255) / 256;
    const int64_t grids[] = {0, m16 * n64, m16 * n128, m16 * n256, m32 * n128};
    for (const AutoRule &r : kAutoRules) {
        if (p.h < r.h_min || p.h > r.h_max || p.m < r.m_min || p.m > r.m_max ||
            p.n < r.n_min || p.n > r.n_max)
            continue;
        const int64_t grid = grids[static_cast<unsigned>(r.grid)];
        if (grid < r.grid_min || grid > r.grid_max)
            continue;
        c = r.config;
        break;
    }
    if (c.kind == KernelKind::Tiled && p.h < c.group_heads)
        c.group_heads = p.h;
    return;
}

inline KernelConfig resolve(const MqaBf16Params &p) {
    KernelConfig config;
    mqa_auto_config(p, config);
    return config;
}
} // namespace lightop::mqa_bf16
