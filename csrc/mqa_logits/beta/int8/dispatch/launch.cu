// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "config.hpp"
#include "../kernels/prefill.hpp"
#include "../kernels/small_kv.hpp"

namespace lightop::mqa_int8 {
namespace {
using Kernel = void (*)(MqaInt8Params);

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
template <int Heads, int BM, int BN, int GroupHeads, bool HasBounds, bool CleanLogits>
Kernel select_tiled_tile(const MqaInt8Params &p) {
    if (p.m % BM == 0) {
        if (p.n % BN == 0)
            return tiled<Heads, BM, BN, GroupHeads, true, true, HasBounds, CleanLogits>;
        return tiled<Heads, BM, BN, GroupHeads, true, false, HasBounds, CleanLogits>;
    }
    if (p.n % BN == 0)
        return tiled<Heads, BM, BN, GroupHeads, false, true, HasBounds, CleanLogits>;
    return tiled<Heads, BM, BN, GroupHeads, false, false, HasBounds, CleanLogits>;
}

template <int Heads, int BM, int GroupHeads, bool HasBounds, bool CleanLogits>
Kernel select_tiled_n(const MqaInt8Params &p, int bn) {
    switch (bn) {
    case 64:
        return select_tiled_tile<Heads, BM, 64, GroupHeads, HasBounds, CleanLogits>(p);
    case 128:
        return select_tiled_tile<Heads, BM, 128, GroupHeads, HasBounds, CleanLogits>(p);
    case 256:
        return select_tiled_tile<Heads, BM, 256, GroupHeads, HasBounds, CleanLogits>(p);
    default:
        return nullptr;
    }
}

template <int H, int TN, int Waves, bool Pair, bool HasBounds, bool CleanLogits>
Kernel select_small(const MqaInt8Params &p) {
    constexpr int BM = Waves * (Pair ? 2 : 1);
    if (p.m % BM == 0) {
        if (p.n % TN == 0)
            return small_kv<H, TN, Waves, Pair, true, true, HasBounds, CleanLogits>;
        return small_kv<H, TN, Waves, Pair, true, false, HasBounds, CleanLogits>;
    }
    if (p.n % TN == 0)
        return small_kv<H, TN, Waves, Pair, false, true, HasBounds, CleanLogits>;
    return small_kv<H, TN, Waves, Pair, false, false, HasBounds, CleanLogits>;
}

template <int Heads, bool HasBounds, bool CleanLogits>
Kernel select_tiled(const MqaInt8Params &p, const KernelConfig &c) {
    constexpr int GH4 = Heads < 4 ? Heads : 4;
    constexpr int GH2 = Heads < 2 ? Heads : 2;
    if (c.block_m == 16 && c.group_heads == GH4)
        return select_tiled_n<Heads, 16, GH4, HasBounds, CleanLogits>(p, c.block_n);
    if (c.block_m == 32 && c.group_heads == GH2)
        return select_tiled_n<Heads, 32, GH2, HasBounds, CleanLogits>(p, c.block_n);
    return nullptr;
}

template <bool HasBounds, bool CleanLogits>
Kernel select_kernel(const MqaInt8Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV) {
        if (p.h == 8 && c.block_n == 16) {
            if (c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, true, HasBounds, CleanLogits>(p);
            if (!c.paired_queries && c.waves == 2)
                return select_small<8, 16, 2, false, HasBounds, CleanLogits>(p);
            if (!c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, false, HasBounds, CleanLogits>(p);
        }
        if (p.h == 64 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<64, 16, 1, false, HasBounds, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<64, 32, 1, false, HasBounds, CleanLogits>(p);
        }
        if (p.h == 16 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<16, 16, 1, false, HasBounds, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<16, 32, 1, false, HasBounds, CleanLogits>(p);
        }
        if (p.h == 32 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<32, 16, 1, false, HasBounds, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<32, 32, 1, false, HasBounds, CleanLogits>(p);
        }
        if (p.h == 128 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<128, 16, 1, false, HasBounds, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<128, 32, 1, false, HasBounds, CleanLogits>(p);
        }
        return nullptr;
    }
    if (c.kind != KernelKind::Tiled)
        return nullptr;
    switch (p.h) {
    case 1:
        return select_tiled<1, HasBounds, CleanLogits>(p, c);
    case 2:
        return select_tiled<2, HasBounds, CleanLogits>(p, c);
    case 4:
        return select_tiled<4, HasBounds, CleanLogits>(p, c);
    case 8:
        return select_tiled<8, HasBounds, CleanLogits>(p, c);
    case 16:
        return select_tiled<16, HasBounds, CleanLogits>(p, c);
    case 32:
        return select_tiled<32, HasBounds, CleanLogits>(p, c);
    case 64:
        return select_tiled<64, HasBounds, CleanLogits>(p, c);
    case 128:
        return select_tiled<128, HasBounds, CleanLogits>(p, c);
    default:
        return nullptr;
    }
}
#endif
} // namespace

hipError_t launch(const MqaInt8Params &p, const KernelConfig &c, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y) {
    LaunchGrid grid;
    if (!make_grid(p, c, max_grid_x, max_grid_y, grid))
        return hipErrorInvalidValue;
    if (!grid.x || !grid.y)
        return hipSuccess;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
    Kernel kernel;
    if (!p.ks && !p.ke) {
        // With no bounds, both cleaning settings have the same all-valid output.
        kernel = select_kernel<false, false>(p, c);
    } else {
        kernel =
            p.clean_logits ? select_kernel<true, true>(p, c) : select_kernel<true, false>(p, c);
    }
    if (!kernel)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(kernel, dim3(grid.x, grid.y), dim3(c.waves * 64), 0, stream, p);
    return hipGetLastError();
#else
    return hipErrorInvalidDeviceFunction;
#endif
}
} // namespace lightop::mqa_int8
