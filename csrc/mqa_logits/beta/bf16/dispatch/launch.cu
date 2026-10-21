// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "config.hpp"
#include "../kernels/small_kv.hpp"
#include "../kernels/m16.hpp"
#include "../kernels/m32.hpp"
#include "../kernels/m64.hpp"

namespace lightop::mqa_bf16 {
namespace {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
using Kernel = void (*)(MqaBf16Params);

template <int H, int TN, int Waves, bool Pair, bool HasBounds, bool CleanLogits>
Kernel select_small(const MqaBf16Params &p) {
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

template <int H, bool HasBounds, bool CleanLogits>
Kernel select_small_n(const MqaBf16Params &p, const KernelConfig &c) {
    if constexpr (H == 8) {
        if (c.paired_queries)
            return select_small<8, 16, 1, true, HasBounds, CleanLogits>(p);
    }
    if (c.block_n == 16)
        return select_small<H, 16, 1, false, HasBounds, CleanLogits>(p);
    if (c.block_n == 32)
        return select_small<H, 32, 1, false, HasBounds, CleanLogits>(p);
    return select_small<H, 64, 1, false, HasBounds, CleanLogits>(p);
}

template <bool HasBounds, bool CleanLogits>
Kernel select_small_kv_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    switch (p.h) {
    case 1:
        return select_small_n<1, HasBounds, CleanLogits>(p, c);
    case 2:
        return select_small_n<2, HasBounds, CleanLogits>(p, c);
    case 4:
        return select_small_n<4, HasBounds, CleanLogits>(p, c);
    case 8:
        return select_small_n<8, HasBounds, CleanLogits>(p, c);
    case 16:
        return select_small_n<16, HasBounds, CleanLogits>(p, c);
    case 32:
        return select_small_n<32, HasBounds, CleanLogits>(p, c);
    case 64:
        return select_small_n<64, HasBounds, CleanLogits>(p, c);
    case 128:
        return select_small_n<128, HasBounds, CleanLogits>(p, c);
    }
    return nullptr;
}

// Choose a host-side BM entry without adding a device-call layer.
template <int BM, int BN, int GH, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
Kernel select_tiled_kernel() {
    static_assert(BM == 16 || BM == 32 || BM == 64);
    if constexpr (BM == 16)
        return select_m16_kernel<BN, GH, EvenM, EvenN, HasBounds, CleanLogits>();
    else if constexpr (BM == 32)
        return select_m32_kernel<BN, GH, EvenM, EvenN, HasBounds, CleanLogits>();
    else
        return select_m64_kernel<BN, GH, EvenM, EvenN, HasBounds, CleanLogits>();
}

template <int BM, int BN, int GH, bool HasBounds, bool CleanLogits>
Kernel select_tiled_tile(const MqaBf16Params &p) {
    if (p.m % BM == 0) {
        if (p.n % BN == 0)
            return select_tiled_kernel<BM, BN, GH, true, true, HasBounds, CleanLogits>();
        return select_tiled_kernel<BM, BN, GH, true, false, HasBounds, CleanLogits>();
    }
    if (p.n % BN == 0)
        return select_tiled_kernel<BM, BN, GH, false, true, HasBounds, CleanLogits>();
    return select_tiled_kernel<BM, BN, GH, false, false, HasBounds, CleanLogits>();
}

template <int BM, int GH, bool HasBounds, bool CleanLogits>
Kernel select_tiled_n(const MqaBf16Params &p, int bn) {
    switch (bn) {
    case 64:
        return select_tiled_tile<BM, 64, GH, HasBounds, CleanLogits>(p);
    case 128:
        return select_tiled_tile<BM, 128, GH, HasBounds, CleanLogits>(p);
    case 256:
        if constexpr (BM == 64 && GH == 2)
            return nullptr;
        else
            return select_tiled_tile<BM, 256, GH, HasBounds, CleanLogits>(p);
    default:
        return nullptr;
    }
}

template <int BM, bool HasBounds, bool CleanLogits>
Kernel select_tiled_group(const MqaBf16Params &p, const KernelConfig &c) {
    switch (c.group_heads) {
    case 1:
        return select_tiled_n<BM, 1, HasBounds, CleanLogits>(p, c.block_n);
    case 2:
        return select_tiled_n<BM, 2, HasBounds, CleanLogits>(p, c.block_n);
    case 4:
        if constexpr (BM <= 32)
            return select_tiled_n<BM, 4, HasBounds, CleanLogits>(p, c.block_n);
    }
    return nullptr;
}

template <bool HasBounds, bool CleanLogits>
Kernel select_split_head_tile(const MqaBf16Params &p) {
    if (p.m % 16 == 0) {
        if (p.n % 64 == 0)
            return split_head<16, 64, 4, true, true, HasBounds, CleanLogits>;
        return split_head<16, 64, 4, true, false, HasBounds, CleanLogits>;
    }
    if (p.n % 64 == 0)
        return split_head<16, 64, 4, false, true, HasBounds, CleanLogits>;
    return split_head<16, 64, 4, false, false, HasBounds, CleanLogits>;
}

template <bool HasBounds, bool CleanLogits>
Kernel select_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV)
        return select_small_kv_kernel<HasBounds, CleanLogits>(p, c);
    if (c.kind == KernelKind::SplitHead)
        return select_split_head_tile<HasBounds, CleanLogits>(p);
    switch (c.block_m) {
    case 16:
        return select_tiled_group<16, HasBounds, CleanLogits>(p, c);
    case 32:
        return select_tiled_group<32, HasBounds, CleanLogits>(p, c);
    case 64:
        return select_tiled_group<64, HasBounds, CleanLogits>(p, c);
    }
    return nullptr;
}
#endif
} // namespace

hipError_t launch(const MqaBf16Params &p, const KernelConfig &c, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y) {
    LaunchGrid grid;
    if (!make_grid(p, c, max_grid_x, max_grid_y, grid))
        return hipErrorInvalidValue;
    if (!grid.x || !grid.y)
        return hipSuccess;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
    Kernel kernel;
    if (!p.ks && !p.ke)
        kernel = select_kernel<false, false>(p, c);
    else if (p.clean_logits)
        kernel = select_kernel<true, true>(p, c);
    else
        kernel = select_kernel<true, false>(p, c);
    // Auto has a default tiled config. A null kernel here means an unsupported
    // explicit config; report it instead of silently changing backend behavior.
    if (!kernel)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(kernel, dim3(grid.x, grid.y), dim3(c.waves * 64), 0, stream, p);
    return hipGetLastError();
#else
    return hipErrorInvalidDeviceFunction;
#endif
}
} // namespace lightop::mqa_bf16
