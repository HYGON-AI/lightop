// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"

namespace lightop::mqa_fp8 {

// Distinct types make an accidental M/N interval swap a compile error.
struct MRange { int64_t min, max; }; // inclusive
struct NRange { int64_t min, max; }; // inclusive
struct Rule {
    Group groups;
    MRange m;
    NRange n;
    KernelId id;
    Layout layout; // performance gates may be stricter than kernel requirements
};

// First match wins: the row's group table, then kShared, then M64. Rows were
// measured on 2026-09-27, 2026-10-12 and 2026-10-13. KS/KE values are never read on the host.
// H=8, clean_logits=0
inline constexpr Rule kH8C0[] = {
    {Group::H8C0, MRange{16, 16}, NRange{32768, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H8C0, MRange{16, 16}, NRange{4096, 8192}, KernelId::M16_64_H8Q, Layout::Contig | Layout::Full256},
    {Group::H8C0, MRange{16, 16}, NRange{4096, 12287}, KernelId::M16_64, Layout::Contig},
    {Group::H8C0, MRange{32, 32}, NRange{12288, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H8C0, MRange{32, 32}, NRange{6144, 12287}, KernelId::M16_128_K64, Layout::Contig | Layout::Full256},
    {Group::H8C0, MRange{32, 32}, NRange{6144, 12287}, KernelId::M16_128, Layout::Contig},
    {Group::H8C0, MRange{32, 32}, NRange{5888, 6143}, KernelId::M16_64, Layout::Contig},
    {Group::H8C0, MRange{64, 64}, NRange{8192, 16383}, KernelId::M16_256, Layout::Contig},
    {Group::H8C0, MRange{64, 64}, NRange{4096, 8191}, KernelId::M16_128_K64, Layout::Contig | Layout::Full256},
    {Group::H8C0, MRange{64, 64}, NRange{3072, 8191}, KernelId::M16_128, Layout::Contig},
    {Group::H8C0, MRange{64, 64}, NRange{2048, 3071}, KernelId::M16_64, Layout::Contig},
    {Group::H8C0, MRange{33, 63}, NRange{1449, 2048}, KernelId::QH_64, Layout::Any},
    {Group::H8C0, MRange{1, 32}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H8C0, MRange{1, 4}, NRange{8193, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H8C0, MRange{5, 32}, NRange{8193, 16384}, KernelId::M16_128_K64, Layout::Full128},
    {Group::H8C0, MRange{5, 32}, NRange{8193, 32768}, KernelId::M16_128, Layout::Any},
    {Group::H8C0, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H8C0, MRange{33, 128}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H8C0, MRange{129, 512}, NRange{182, 1448}, KernelId::M16_128, Layout::Any},
    {Group::H8C0, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
};
// H=8, clean_logits=1
inline constexpr Rule kH8C1[] = {
    {Group::H8C1, MRange{16, 16}, NRange{32768, 32768}, KernelId::M16G_W8, Layout::Grouped | Layout::Nx256},
    {Group::H8C1, MRange{32, 32}, NRange{16384, 16384}, KernelId::M16G_W8_K64, Layout::Grouped | Layout::Nx256},
    {Group::H8C1, MRange{32, 32}, NRange{16385, 24576}, KernelId::M16G_W8, Layout::Grouped | Layout::Nx256},
    {Group::H8C1, MRange{16, 16}, NRange{3072, 4095}, KernelId::M16_64_H8Q, Layout::Contig | Layout::Full256},
    {Group::H8C1, MRange{16, 16}, NRange{3072, 4095}, KernelId::M16_64, Layout::Contig},
    {Group::H8C1, MRange{64, 64}, NRange{8192, 32767}, KernelId::M16_256, Layout::Contig},
    {Group::H8C1, MRange{64, 64}, NRange{4096, 8191}, KernelId::M16G_W8_K64, Layout::Grouped | Layout::Nx256},
    {Group::H8C1, MRange{64, 64}, NRange{4096, 8191}, KernelId::M16_128_K64, Layout::Contig | Layout::Full256},
    {Group::H8C1, MRange{64, 64}, NRange{3072, 8191}, KernelId::M16_128, Layout::Contig},
    {Group::H8C1, MRange{64, 64}, NRange{2048, 3071}, KernelId::M16_64, Layout::Contig},
    {Group::H8C1, MRange{16, 16}, NRange{32768, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H8C1, MRange{16, 16}, NRange{4096, 8192}, KernelId::M16_64_H8Q, Layout::Contig | Layout::Full256},
    {Group::H8C1, MRange{16, 16}, NRange{4096, 12287}, KernelId::M16_64, Layout::Contig},
    {Group::H8C1, MRange{32, 32}, NRange{12288, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H8C1, MRange{32, 32}, NRange{8192, 12287}, KernelId::M16G_W8_K64, Layout::Grouped | Layout::Nx256},
    {Group::H8C1, MRange{32, 32}, NRange{6144, 12287}, KernelId::M16_128_K64, Layout::Contig | Layout::Full256},
    {Group::H8C1, MRange{32, 32}, NRange{6144, 12287}, KernelId::M16_128, Layout::Contig},
    {Group::H8C1, MRange{32, 32}, NRange{4608, 6143}, KernelId::M16_64, Layout::Contig},
    {Group::H8C1, MRange{33, 63}, NRange{1449, 2048}, KernelId::QH_64, Layout::Any},
    {Group::H8C1, MRange{1, 32}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H8C1, MRange{1, 4}, NRange{8193, 32768}, KernelId::QH_64, Layout::Any},
    {Group::H8C1, MRange{1, 4}, NRange{32769, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H8C1, MRange{16, 32}, NRange{8193, 16384}, KernelId::M16G_W8_K64, Layout::Grouped},
    {Group::H8C1, MRange{16, 32}, NRange{16385, 32768}, KernelId::M16G_W8, Layout::Grouped},
    {Group::H8C1, MRange{5, 32}, NRange{8193, 16384}, KernelId::M16_128_K64, Layout::Full128},
    {Group::H8C1, MRange{5, 32}, NRange{8193, 32768}, KernelId::M16_128, Layout::Any},
    {Group::H8C1, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H8C1, MRange{33, 128}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H8C1, MRange{129, 512}, NRange{182, 1448}, KernelId::M16_64, Layout::Any},
    {Group::H8C1, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
};
// H=16, clean_logits=0
inline constexpr Rule kH16C0[] = {
    {Group::H16C0, MRange{32, 32}, NRange{65536, 262144}, KernelId::M16_256, Layout::Contig},
    {Group::H16C0, MRange{33, 64}, NRange{1449, 2048}, KernelId::QH_128, Layout::Any},
    {Group::H16C0, MRange{33, 64}, NRange{2049, 4096}, KernelId::QH_256, Layout::Any},
    {Group::H16C0, MRange{1, 32}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H16C0, MRange{1, 4}, NRange{8193, 32768}, KernelId::QH_64, Layout::Any},
    {Group::H16C0, MRange{1, 4}, NRange{32769, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H16C0, MRange{5, 32}, NRange{8193, 32768}, KernelId::M16_128, Layout::Any},
    {Group::H16C0, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H16C0, MRange{33, 128}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H16C0, MRange{129, 512}, NRange{182, 1448}, KernelId::M16_128, Layout::Any},
    {Group::H16C0, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
    // H16/C0 benefits at aligned N; N4095/4097 and larger M can favor M64.
    {Group::H16C0, MRange{129, 192}, NRange{2049, 4096}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Nx256},
};
// H=16, clean_logits=1
inline constexpr Rule kH16C1[] = {
    {Group::H16C1, MRange{32, 32}, NRange{65536, 262144}, KernelId::M16_256, Layout::Contig},
    {Group::H16C1, MRange{129, 512}, NRange{182, 1448}, KernelId::M16_128, Layout::Any},
    {Group::H16C1, MRange{33, 64}, NRange{2049, 8192}, KernelId::M16_256, Layout::Any},
    {Group::H16C1, MRange{65, 256}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
    {Group::H16C1, MRange{33, 64}, NRange{1449, 2048}, KernelId::QH_64, Layout::Any},
    {Group::H16C1, MRange{1, 32}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H16C1, MRange{1, 4}, NRange{8193, 32768}, KernelId::QH_64, Layout::Any},
    {Group::H16C1, MRange{1, 4}, NRange{32769, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H16C1, MRange{5, 32}, NRange{8193, 32768}, KernelId::M16_128, Layout::Any},
    {Group::H16C1, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H16C1, MRange{33, 128}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
};
// H=32, clean_logits=0
inline constexpr Rule kH32C0[] = {
    // Aligned N tiles favor M16 over the QH_256 row below from M24 onward.
    {Group::H32C0, MRange{24, 32}, NRange{16384, 32768}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Nx256},
    {Group::H32C0, MRange{33, 64}, NRange{1449, 2048}, KernelId::QH_256, Layout::Any},
    {Group::H32C0, MRange{33, 48}, NRange{2049, 4096}, KernelId::QH_256, Layout::Any},
    {Group::H32C0, MRange{1, 64}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H32C0, MRange{65, 512}, NRange{182, 1448}, KernelId::QH_256, Layout::Any},
    {Group::H32C0, MRange{1, 4}, NRange{1449, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H32C0, MRange{5, 32}, NRange{1449, 8192}, KernelId::QH_128, Layout::Any},
    {Group::H32C0, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
    {Group::H32C0, MRange{1, 4}, NRange{8193, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H32C0, MRange{5, 32}, NRange{8193, 32768}, KernelId::QH_256, Layout::Any},
    {Group::H32C0, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
};
// H=32, clean_logits=1
inline constexpr Rule kH32C1[] = {
    {Group::H32C1, MRange{33, 64}, NRange{1449, 2048}, KernelId::QH_128, Layout::Any},
    {Group::H32C1, MRange{33, 48}, NRange{2049, 4096}, KernelId::QH_128, Layout::Any},
    {Group::H32C1, MRange{1, 128}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H32C1, MRange{129, 512}, NRange{182, 1448}, KernelId::QH_256, Layout::Any},
    {Group::H32C1, MRange{1, 4}, NRange{1449, 32768}, KernelId::QH_64, Layout::Any},
    {Group::H32C1, MRange{1, 4}, NRange{32769, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H32C1, MRange{5, 32}, NRange{1449, 8192}, KernelId::QH_128, Layout::Any},
    {Group::H32C1, MRange{5, 32}, NRange{8193, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H32C1, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
};
// H=64, clean_logits=0
inline constexpr Rule kH64C0[] = {
    {Group::H64C0, MRange{16, 16}, NRange{40960, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H64C0, MRange{16, 16}, NRange{4096, 40959}, KernelId::QH_256, Layout::Contig},
    {Group::H64C0, MRange{16, 16}, NRange{2048, 4095}, KernelId::QH_128, Layout::Contig},
    {Group::H64C0, MRange{32, 32}, NRange{14336, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H64C0, MRange{64, 64}, NRange{8192, 24575}, KernelId::M16_256, Layout::Contig},
    {Group::H64C0, MRange{64, 64}, NRange{2048, 8191}, KernelId::QH_256, Layout::Contig},
    {Group::H64C0, MRange{33, 64}, NRange{1449, 4096}, KernelId::QH_256, Layout::Any},
    {Group::H64C0, MRange{1, 4}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H64C0, MRange{1, 1}, NRange{8193, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H64C0, MRange{2, 4}, NRange{8193, 65536}, KernelId::QH_256, Layout::Any},
    {Group::H64C0, MRange{5, 32}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H64C0, MRange{5, 32}, NRange{1449, 32768}, KernelId::QH_256, Layout::Any},
    {Group::H64C0, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H64C0, MRange{33, 512}, NRange{182, 1448}, KernelId::QH_256, Layout::Any},
    {Group::H64C0, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
};
// H=64, clean_logits=1
inline constexpr Rule kH64C1[] = {
    {Group::H64C1, MRange{16, 16}, NRange{32768, 45056}, KernelId::M16G_W16, Layout::Grouped | Layout::Nx256},
    {Group::H64C1, MRange{32, 32}, NRange{12288, 12544}, KernelId::M16G_W16, Layout::Grouped | Layout::Nx256},
    {Group::H64C1, MRange{16, 16}, NRange{40960, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H64C1, MRange{16, 16}, NRange{4096, 40959}, KernelId::QH_256, Layout::Contig},
    {Group::H64C1, MRange{16, 16}, NRange{2048, 4095}, KernelId::QH_128, Layout::Contig},
    {Group::H64C1, MRange{32, 32}, NRange{14336, 1048576}, KernelId::M16_256, Layout::Contig},
    {Group::H64C1, MRange{64, 64}, NRange{8192, 24575}, KernelId::M16_256, Layout::Contig},
    {Group::H64C1, MRange{64, 64}, NRange{2048, 8191}, KernelId::QH_256, Layout::Contig},
    {Group::H64C1, MRange{33, 64}, NRange{1449, 4096}, KernelId::QH_256, Layout::Any},
    {Group::H64C1, MRange{1, 4}, NRange{64, 8192}, KernelId::QH_64, Layout::Any},
    {Group::H64C1, MRange{1, 1}, NRange{8193, 65536}, KernelId::QH_128, Layout::Any},
    {Group::H64C1, MRange{2, 4}, NRange{8193, 65536}, KernelId::QH_256, Layout::Any},
    {Group::H64C1, MRange{5, 32}, NRange{64, 1448}, KernelId::QH_64, Layout::Any},
    {Group::H64C1, MRange{5, 32}, NRange{1449, 32768}, KernelId::QH_256, Layout::Any},
    {Group::H64C1, MRange{5, 32}, NRange{32769, 65536}, KernelId::M16_256, Layout::Any},
    {Group::H64C1, MRange{33, 512}, NRange{182, 1448}, KernelId::QH_256, Layout::Any},
    {Group::H64C1, MRange{33, 128}, NRange{1449, 8192}, KernelId::M16_256, Layout::Any},
};
// Shared intervals fill per-group gaps without overriding tuned rows.
inline constexpr Rule kShared[] = {
    // The M33..63 gap favors BN256 at aligned N up to 12288. Only M48
    // extends to N16384; larger M tails and N tails can favor M64.
    {Group::TunedHeads, MRange{33, 63}, NRange{8193, 12288}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Nx256},
    {Group::TunedHeads, MRange{48, 48}, NRange{8193, 12288}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Full128},
    {Group::TunedHeads, MRange{48, 48}, NRange{12289, 16384}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Full256},
    // Measured with scale, including padded rows. Unit-scale keeps the pre-existing
    // N524288 limit until independently benchmarked; Scaled is a performance gate.
    {Group::H16, MRange{5, 16}, NRange{524289, 1048576}, KernelId::M16_256, Layout::Scaled},
    // At large M, QH wins for small N but loses on full windows at N181.
    {Group::TunedHeads, MRange{1025, 2048}, NRange{1, 128}, KernelId::QH_64, Layout::Packed | Layout::Scaled},
    {Group::TunedHeads, MRange{1025, 2048}, NRange{129, 181}, KernelId::M16_256, Layout::Packed | Layout::Scaled},
    // Keep complete M80 tiles separate and stop before the larger N tails.
    {Group::TunedHeads, MRange{65, 79}, NRange{8449, 12288}, KernelId::M16_128, Layout::Packed | Layout::Scaled},
    // Only M80/N8320 changes selection here; N8448 already uses BN256 below.
    {Group::TunedHeads, MRange{80, 80}, NRange{8320, 8320}, KernelId::M16_256, Layout::Packed | Layout::Scaled | Layout::Full128},
    {Group::TunedHeads, MRange{1, 1024}, NRange{1, 181}, KernelId::QH_64, Layout::Any},
    {Group::H8, MRange{1, 4}, NRange{65537, 524288}, KernelId::M16_256, Layout::Any},
    {Group::H16 | Group::H32, MRange{1, 4}, NRange{65537, 524288}, KernelId::QH_128, Layout::Any},
    {Group::H64, MRange{1, 1}, NRange{65537, 524288}, KernelId::QH_128, Layout::Any},
    {Group::H64, MRange{2, 4}, NRange{65537, 524288}, KernelId::QH_256, Layout::Any},
    {Group::TunedHeads, MRange{5, 16}, NRange{65537, 524288}, KernelId::M16_256, Layout::Any},
    // H8 M17..32 tails did not consistently beat M64. H16 crosses back
    // toward M64 above M24; its aligned M32 priority rows still win.
    {Group::H16, MRange{17, 24}, NRange{65537, 262144}, KernelId::M16_256, Layout::Any},
    {Group::H32 | Group::H64, MRange{17, 32}, NRange{65537, 262144}, KernelId::M16_256, Layout::Any},
    // Stop at N2048: nearby N tails (e.g. M192/N4095) can favor M64.
    {Group::H8, MRange{129, 224}, NRange{1449, 2048}, KernelId::M16_256, Layout::Any},
    // H16C1 is already covered by its group table.
    {Group::H16C0 | Group::H32 | Group::H64, MRange{129, 256}, NRange{1449, 2048}, KernelId::M16_256, Layout::Any},
    // Complete tiles with vector stores favor BN256; otherwise BN128 wins.
    {Group::TunedHeads, MRange{80, 80}, NRange{8448, 8448}, KernelId::M16_256, Layout::Full256},
    {Group::H8, MRange{80, 80}, NRange{8320, 8320}, KernelId::M16_128_K64, Layout::Full128},
    {Group::TunedHeads, MRange{65, 80}, NRange{8193, 8448}, KernelId::M16_128, Layout::Any},
};

struct RuleSpan {
    const Rule *first, *last;
    constexpr const Rule *begin() const { return first; }
    constexpr const Rule *end() const { return last; }
};
template <size_t N> constexpr RuleSpan rules(const Rule (&r)[N]) { return {r, r + N}; }

// Indexed by group_of(); keep the group identity next to its rows.
struct GroupRules { Group group; RuleSpan rows; };
inline constexpr GroupRules kGroupRules[] = {
    {Group::H8C0, rules(kH8C0)}, {Group::H8C1, rules(kH8C1)},
    {Group::H16C0, rules(kH16C0)}, {Group::H16C1, rules(kH16C1)},
    {Group::H32C0, rules(kH32C0)}, {Group::H32C1, rules(kH32C1)},
    {Group::H64C0, rules(kH64C0)}, {Group::H64C1, rules(kH64C1)},
};
inline constexpr size_t kGroupCount = sizeof(kGroupRules) / sizeof(kGroupRules[0]);

// Automatic routes must stay inside the measured groups and meet the kernel's hard requirements.
constexpr bool rules_valid(RuleSpan rs, Group group = Group::None) {
    for (const Rule &r : rs) {
        if (!valid_kernel_id(r.id))
            return false;
        const KernelTraits &t = traits(r.id);
        if ((group != Group::None && r.groups != group) || r.groups == Group::None ||
            !contains(t.verified_groups, r.groups) || !contains(r.layout, t.required.layout) ||
            r.m.min < 1 || r.n.min < 1 || r.m.min > r.m.max || r.n.min > r.n.max)
            return false;
    }
    return true;
}
constexpr bool table_valid() {
    if (kGroupCount > sizeof(uint16_t) * 8)
        return false;
    Group groups = Group::None;
    for (size_t g = 0; g < kGroupCount; ++g) {
        const auto &entry = kGroupRules[g];
        if (entry.group != Group(1u << g) || !rules_valid(entry.rows, entry.group))
            return false;
        groups = groups | entry.group;
    }
    return groups == Group::TunedHeads && rules_valid(rules(kShared));
}
static_assert(table_valid());

// -1: H without tuned rows or clean_logits outside {0, 1}; make_grid validates.
inline int group_of(const MqaFp8Params &p) {
    if (p.clean_logits > 1)
        return -1;
    switch (p.h) {
    case 8: return 0 + int(p.clean_logits);
    case 16: return 2 + int(p.clean_logits);
    case 32: return 4 + int(p.clean_logits);
    case 64: return 6 + int(p.clean_logits);
    default: return -1;
    }
}

inline bool matches(const Rule &r, const MqaFp8Params &p, Layout layout) {
    return p.m >= r.m.min && p.m <= r.m.max && p.n >= r.n.min && p.n <= r.n.max &&
           contains(layout, r.layout);
}

inline KernelId select_kernel(const MqaFp8Params &p) {
    const int g = group_of(p);
    if (g < 0)
        return KernelId::M64;
    const Layout layout = layout_of(p);
    for (const Rule &r : kGroupRules[g].rows)
        if (matches(r, p, layout))
            return r.id;
    for (const Rule &r : kShared)
        if (contains(r.groups, kGroupRules[g].group) && matches(r, p, layout))
            return r.id;
    return KernelId::M64;
}

} // namespace lightop::mqa_fp8
