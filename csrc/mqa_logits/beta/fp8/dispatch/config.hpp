// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../params.h"
#include <cstddef>
#include <cstdint>

namespace lightop::mqa_fp8 {

// One bit per tuned H/clean_logits pair.
enum class Group : uint16_t {
    None = 0,
    H8C0 = 1u << 0, H8C1 = 1u << 1,
    H16C0 = 1u << 2, H16C1 = 1u << 3,
    H32C0 = 1u << 4, H32C1 = 1u << 5,
    H64C0 = 1u << 6, H64C1 = 1u << 7,
    H8 = H8C0 | H8C1, H16 = H16C0 | H16C1,
    H32 = H32C0 | H32C1, H64 = H64C0 | H64C1,
    TunedHeads = H8 | H16 | H32 | H64,
};

// Layout facts derived from the params; rules and kernels list the bits they need.
enum class Layout : uint8_t {
    Any = 0,
    Packed = 1u << 0,          // D128, tightly packed Q/K/weights/output
    Nx64 = 1u << 6,
    Nx128 = (1u << 1) | Nx64,
    Nx256 = (1u << 2) | Nx128,
    Mx16 = 1u << 3,
    OutVec = 1u << 4,          // output base and row stride are 16-byte aligned
    Scaled = 1u << 5,          // kv_scale != nullptr
    Contig = Packed | Nx256,
    Full64 = Mx16 | Nx64 | OutVec,
    Full128 = Mx16 | Nx128 | OutVec,
    Full256 = Mx16 | Nx256 | OutVec,
    Grouped = Packed | Scaled | Full128,
};

constexpr Group operator|(Group a, Group b) { return Group(uint16_t(a) | uint16_t(b)); }
constexpr Layout operator|(Layout a, Layout b) { return Layout(uint8_t(a) | uint8_t(b)); }
constexpr bool contains(Group facts, Group required) {
    return (uint16_t(facts) & uint16_t(required)) == uint16_t(required);
}
constexpr bool contains(Layout facts, Layout required) {
    return (uint8_t(facts) & uint8_t(required)) == uint8_t(required);
}

enum class Family : uint8_t { M64, M16, M16G, QH };

// Every launchable variant. Values are the index into kKernelTraits.
enum class KernelId : uint8_t {
    M64,
    M16_64, M16_128, M16_256,
    M16_64_H8Q, M16_128_K64,
    M16G_W8, M16G_W8_K64, M16G_W16,
    QH_64, QH_128, QH_256,
};

// Head counts are powers of two, so each supported count occupies its own bit.
inline constexpr uint16_t kAllHeads = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128;
inline constexpr uint16_t kQhHeads = 8 | 16 | 32 | 64;
inline constexpr uint8_t kBothCleanModes = 0b11;
inline constexpr uint8_t kCleanOnly = 0b10;

struct KernelRequirements {
    uint16_t heads;
    uint8_t clean_modes; // bit 0: clean_logits=0, bit 1: clean_logits=1
    Layout layout;
};

struct KernelTraits {
    KernelId id;
    const char *name;
    Family family;
    uint16_t block_n;
    Group verified_groups; // allowed in automatic tuning rules, not a runtime support limit
    KernelRequirements required;
};

inline constexpr KernelTraits kKernelTraits[] = {
    {KernelId::M64, "m64", Family::M64, 256, Group::TunedHeads,
     {kAllHeads, kBothCleanModes, Layout::Any}},
    {KernelId::M16_64, "m16_64", Family::M16, 64, Group::TunedHeads,
     {kAllHeads, kBothCleanModes, Layout::Any}},
    {KernelId::M16_128, "m16_128", Family::M16, 128, Group::TunedHeads,
     {kAllHeads, kBothCleanModes, Layout::Any}},
    {KernelId::M16_256, "m16_256", Family::M16, 256, Group::TunedHeads,
     {kAllHeads, kBothCleanModes, Layout::Any}},
    {KernelId::M16_64_H8Q, "m16_64_h8q", Family::M16, 64, Group::H8,
     {8, kBothCleanModes, Layout::Full64}},
    // K64 changes KV loading only; the head loop remains dynamic. Auto tuning is verified on H8.
    {KernelId::M16_128_K64, "m16_128_k64", Family::M16, 128, Group::H8,
     {kAllHeads, kBothCleanModes, Layout::Full128}},
    {KernelId::M16G_W8, "m16g_w8", Family::M16G, 128, Group::H8C1,
     {8, kCleanOnly, Layout::Grouped}},
    {KernelId::M16G_W8_K64, "m16g_w8_k64", Family::M16G, 128, Group::H8C1,
     {8, kCleanOnly, Layout::Grouped}},
    // The kernel handles any H%16; only H64 with clean logits has been verified on GPU.
    {KernelId::M16G_W16, "m16g_w16", Family::M16G, 128, Group::H64C1,
     {16 | 32 | 64 | 128, kCleanOnly, Layout::Grouped}},
    {KernelId::QH_64, "qh_64", Family::QH, 64, Group::TunedHeads,
     {kQhHeads, kBothCleanModes, Layout::Any}},
    {KernelId::QH_128, "qh_128", Family::QH, 128, Group::TunedHeads,
     {kQhHeads, kBothCleanModes, Layout::Any}},
    {KernelId::QH_256, "qh_256", Family::QH, 256, Group::TunedHeads,
     {kQhHeads, kBothCleanModes, Layout::Any}},
};
inline constexpr size_t kKernelCount = sizeof(kKernelTraits) / sizeof(kKernelTraits[0]);

constexpr bool valid_kernel_id(KernelId id) { return size_t(id) < kKernelCount; }
// Callers with runtime ids must validate them before indexing.
constexpr const KernelTraits &traits(KernelId id) { return kKernelTraits[size_t(id)]; }
constexpr const char *kernel_name(KernelId id) {
    return valid_kernel_id(id) ? traits(id).name : "invalid";
}

constexpr bool traits_ordered() {
    for (size_t i = 0; i < kKernelCount; ++i)
        if (size_t(kKernelTraits[i].id) != i)
            return false;
    return true;
}
static_assert(traits_ordered());

inline bool packed(const MqaFp8Params &p) {
    return p.d == 128 && p.q_stride_m == p.h * 128 && p.q_stride_h == 128 && p.kv_stride_n == 128 &&
           p.weights_stride_m == p.h && p.out_stride_m == p.n;
}

inline bool out_vec(const MqaFp8Params &p) {
    return ((reinterpret_cast<uintptr_t>(p.out) | (uint64_t(p.out_stride_m) * sizeof(float))) & 15) == 0;
}

inline Layout layout_of(const MqaFp8Params &p) {
    Layout l = Layout::Any;
    if (packed(p)) l = l | Layout::Packed;
    if (p.n % 64 == 0) l = l | Layout::Nx64;
    if (p.n % 128 == 0) l = l | Layout::Nx128;
    if (p.n % 256 == 0) l = l | Layout::Nx256;
    if (p.m % 16 == 0) l = l | Layout::Mx16;
    if (out_vec(p)) l = l | Layout::OutVec;
    if (p.kv_scale) l = l | Layout::Scaled;
    return l;
}

// Host launch/selector requirements. Tensor memory and Q/K strides are validated by the adapter.
// verified_groups deliberately does not restrict forced calls or M64's untuned-head fallback.
inline bool kernel_supported(const MqaFp8Params &p, KernelId id) {
    if (!valid_kernel_id(id) || p.m < 0 || p.n < 0 || p.d != 128 || p.h < 1 || p.h > 128 ||
        (p.h & (p.h - 1)) || p.clean_logits > 1)
        return false;
    const auto &r = traits(id).required;
    return (r.heads & uint16_t(p.h)) && (r.clean_modes & (1u << p.clean_logits)) &&
           contains(layout_of(p), r.layout);
}

inline bool kernel_supported(const MqaFp8Params &p, KernelId id, Family family) {
    return valid_kernel_id(id) && traits(id).family == family && kernel_supported(p, id);
}

struct LaunchGrid {
    uint32_t x = 0;
    uint32_t y = 0;
};

// Used once by dispatch, for both device-limit validation and the actual launch.
inline bool make_grid(const MqaFp8Params &p, KernelId id, uint32_t max_x, uint32_t max_y,
                      LaunchGrid &grid) {
    grid = {};
    if (!kernel_supported(p, id)) {
        return false;
    }
    const KernelTraits &t = traits(id);
    uint32_t block_m;
    switch (t.family) {
    case Family::M64:
        block_m = 64;
        break;
    case Family::M16:
    case Family::M16G:
        block_m = 16;
        break;
    case Family::QH:
        block_m = 64 / p.h;
        break;
    default:
        return false;
    }
    if (p.m == 0 || p.n == 0)
        return true;
    // Division before addition avoids overflow near INT64_MAX.
    const uint64_t x = p.n / t.block_n + (p.n % t.block_n != 0);
    const uint64_t y = p.m / block_m + (p.m % block_m != 0);
    if (x > max_x || y > max_y)
        return false;
    grid = {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
    return true;
}
} // namespace lightop::mqa_fp8
