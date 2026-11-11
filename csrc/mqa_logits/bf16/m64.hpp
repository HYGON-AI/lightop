// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "m32.hpp"

namespace lightop::mqa_bf16 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Host-side BM64 entry into the common tiled kernel in m32.hpp.
template <int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
inline auto select_m64_kernel() {
    return tiled<64, BN, GroupHeads, EvenM, EvenN, HasBounds, CleanLogits>;
}
#endif
} // namespace lightop::mqa_bf16
