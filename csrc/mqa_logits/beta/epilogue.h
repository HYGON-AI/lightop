// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "arch.h"

namespace lightop::mqa_beta {
__device__ __forceinline__ void store4(float *destination, F4 value) {
    // DCC must see the store to protect its source registers from early reuse.
    // An opaque global_store ASM miscompiled the last M16's aligned writeback.
    *reinterpret_cast<F4 *>(destination) = value;
}

__device__ __forceinline__ void store1(float *destination, float value) { *destination = value; }

// AlignN is the logical N divisibility. An arbitrary caller-owned output may
// have weaker base/row alignment; retain scalar fallback without touching padding.
template <int AlignN>
__device__ __forceinline__ bool output_vector_aligned(const float *out, int64_t stride) {
    static_assert(AlignN == 4 || AlignN == 2 || AlignN == 1);
    return ((uintptr_t(out) | (uint64_t(stride) * sizeof(float))) & (AlignN * sizeof(float) - 1)) ==
           0;
}

__device__ __forceinline__ void store_logits4(float *out, F4 value, bool vector_out) {
    // The caller guards the packet's first column; N%4 guarantees all four.
    if (vector_out)
        store4(out, value);
    else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            store1(out + j, value[j]);
    }
}

} // namespace lightop::mqa_beta
