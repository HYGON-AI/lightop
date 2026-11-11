// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"

namespace lightop::mqa_fp8 {

// Frozen selection after the 2026-09-27 causal-mask measurements. M/N bounds
// are inclusive. The first matching row within an H/clean group wins. Untuned
// regions retain the previous selection or the M64:256 fallback.
// A true layout flag requires D=128, N divisible by 256, and tightly packed
// Q, K, weights, and output rows. Grouped M16 rules also need K scale and a
// 16-byte-aligned output base, so their selected BN actually reaches the
// grouped kernel. KS/KE values are never read on the host.
struct AutoRule {
    int32_t m_min, m_max, n_min, n_max;
    KernelKind kind;
    uint16_t block_n;
    bool aligned_contiguous;
    bool requires_grouped_m16 = false;
};

inline constexpr AutoRule kAutoRules[] = {
    // H=8, clean_logits=0
    {16, 16, 32768, 1048576, KernelKind::M16, 256, true},
    {16, 16, 12288, 32767, KernelKind::M16, 128, true},
    {16, 16, 4096, 12287, KernelKind::M16, 64, true},
    {16, 16, 2048, 4095, KernelKind::QH, 64, true},
    {32, 32, 12288, 1048576, KernelKind::M16, 256, true},
    {32, 32, 6144, 12287, KernelKind::M16, 128, true},
    {32, 32, 5888, 6143, KernelKind::M16, 64, true},
    {32, 32, 2048, 4095, KernelKind::QH, 64, true},
    {64, 64, 16384, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 16383, KernelKind::M16, 256, true},
    {64, 64, 3072, 8191, KernelKind::M16, 128, true},
    {64, 64, 2048, 3071, KernelKind::M16, 64, true},
    {33, 63, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // H=8, clean_logits=1
    {16, 16, 32768, 32768, KernelKind::M16, 128, true, true},
    {32, 32, 16384, 24576, KernelKind::M16, 128, true, true},
    {16, 16, 3072, 4095, KernelKind::M16, 64, true},
    {16, 16, 2048, 3071, KernelKind::QH, 64, true},
    {64, 64, 8192, 32767, KernelKind::M16, 256, true},
    {64, 64, 3072, 8191, KernelKind::M16, 128, true},
    {64, 64, 2048, 3071, KernelKind::M16, 64, true},
    {16, 16, 32768, 1048576, KernelKind::M16, 256, true},
    {16, 16, 12288, 32767, KernelKind::M16, 128, true},
    {16, 16, 4096, 12287, KernelKind::M16, 64, true},
    {32, 32, 12288, 1048576, KernelKind::M16, 256, true},
    {32, 32, 6144, 12287, KernelKind::M16, 128, true},
    {32, 32, 4608, 6143, KernelKind::M16, 64, true},
    {32, 32, 2048, 4095, KernelKind::QH, 64, true},
    {64, 64, 16384, 1048576, KernelKind::M64, 256, true},
    {33, 63, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 64, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // H=16, clean_logits=0
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {33, 64, 1449, 2048, KernelKind::QH, 128, false},
    {33, 64, 2049, 4096, KernelKind::QH, 256, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // H=16, clean_logits=1
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 64, 2049, 8192, KernelKind::M16, 256, false},
    {65, 256, 1449, 8192, KernelKind::M16, 256, false},
    {33, 64, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    // H=32, clean_logits=0
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {1, 4, 1, 4, KernelKind::QH, 64, false},
    {33, 64, 1449, 2048, KernelKind::QH, 256, false},
    {33, 48, 2049, 4096, KernelKind::QH, 256, false},
    {1, 64, 64, 1448, KernelKind::QH, 64, false},
    {65, 512, 64, 181, KernelKind::QH, 64, false},
    {65, 512, 182, 1448, KernelKind::QH, 256, false},
    {1, 4, 1449, 8192, KernelKind::QH, 64, false},
    {5, 32, 1449, 8192, KernelKind::QH, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    {1, 4, 8193, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::QH, 256, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    // H=32, clean_logits=1
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {1, 4, 1, 4, KernelKind::QH, 64, false},
    {33, 64, 1449, 2048, KernelKind::QH, 128, false},
    {33, 48, 2049, 4096, KernelKind::QH, 128, false},
    {1, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::QH, 256, false},
    {1, 4, 1449, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 1449, 8192, KernelKind::QH, 128, false},
    {5, 32, 8193, 65536, KernelKind::M16, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // H=64, clean_logits=0
    {16, 16, 40960, 1048576, KernelKind::M16, 256, true},
    {16, 16, 4096, 40959, KernelKind::QH, 256, true},
    {16, 16, 2048, 4095, KernelKind::QH, 128, true},
    {32, 32, 14336, 1048576, KernelKind::M16, 256, true},
    {32, 32, 2048, 14335, KernelKind::QH, 256, true},
    {64, 64, 24576, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 24575, KernelKind::M16, 256, true},
    {64, 64, 2048, 8191, KernelKind::QH, 256, true},
    {33, 64, 1449, 4096, KernelKind::QH, 256, false},
    {1, 4, 64, 8192, KernelKind::QH, 64, false},
    {1, 1, 8193, 65536, KernelKind::QH, 128, false},
    {2, 4, 8193, 65536, KernelKind::QH, 256, false},
    {5, 32, 64, 1448, KernelKind::QH, 64, false},
    {5, 32, 1449, 32768, KernelKind::QH, 256, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 512, 64, 181, KernelKind::QH, 64, false},
    {33, 512, 182, 1448, KernelKind::QH, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // H=64, clean_logits=1
    {16, 16, 32768, 45056, KernelKind::M16, 128, true, true},
    {32, 32, 12288, 12544, KernelKind::M16, 128, true, true},
    {16, 16, 40960, 1048576, KernelKind::M16, 256, true},
    {16, 16, 4096, 40959, KernelKind::QH, 256, true},
    {16, 16, 2048, 4095, KernelKind::QH, 128, true},
    {32, 32, 14336, 1048576, KernelKind::M16, 256, true},
    {32, 32, 2048, 14335, KernelKind::QH, 256, true},
    {64, 64, 24576, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 24575, KernelKind::M16, 256, true},
    {64, 64, 2048, 8191, KernelKind::QH, 256, true},
    {33, 64, 1449, 4096, KernelKind::QH, 256, false},
    {1, 4, 64, 8192, KernelKind::QH, 64, false},
    {1, 1, 8193, 65536, KernelKind::QH, 128, false},
    {2, 4, 8193, 65536, KernelKind::QH, 256, false},
    {5, 32, 64, 1448, KernelKind::QH, 64, false},
    {5, 32, 1449, 32768, KernelKind::QH, 256, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 512, 64, 181, KernelKind::QH, 64, false},
    {33, 512, 182, 1448, KernelKind::QH, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
};

inline constexpr uint16_t kAutoRuleOffsets[] = {0, 21, 46, 59, 72, 86, 99, 117, 137};
static_assert(kAutoRuleOffsets[8] == sizeof(kAutoRules) / sizeof(kAutoRules[0]));

inline void mqa_auto_config(const MqaFp8Params& p, KernelConfig& c) {
    c = {KernelKind::M64, 256};
    if (p.clean_logits > 1) return;
    unsigned h_group;
    switch (p.h) {
    case 8: h_group = 0; break;
    case 16: h_group = 1; break;
    case 32: h_group = 2; break;
    case 64: h_group = 3; break;
    default: return;
    }
    const unsigned group = 2 * h_group + p.clean_logits;
    const bool aligned_contiguous = p.d == 128 && p.n % 256 == 0 &&
        p.q_stride_m == p.h * 128 && p.q_stride_h == 128 &&
        p.kv_stride_n == 128 && p.weights_stride_m == p.h &&
        p.out_stride_m == p.n;
    for (uint16_t i = kAutoRuleOffsets[group]; i < kAutoRuleOffsets[group + 1]; ++i) {
        const AutoRule& r = kAutoRules[i];
        if (p.m < r.m_min || p.m > r.m_max || p.n < r.n_min || p.n > r.n_max)
            continue;
        if (r.aligned_contiguous && !aligned_contiguous) continue;
        if (r.requires_grouped_m16 &&
            (!p.kv_scale || (reinterpret_cast<uintptr_t>(p.out) & 15) != 0)) continue;
        c = {r.kind, r.block_n};
        return;
    }
}

} // namespace lightop::mqa_fp8
