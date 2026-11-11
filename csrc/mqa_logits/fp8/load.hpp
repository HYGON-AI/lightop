// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../arch.h"

namespace lightop::mqa_fp8 {
using ScalarBufferLoad = mqa_beta::BufferLoadAccessor<4, mqa_beta::BufferEmitMode::Builtin>;
// Keep lane voffset fixed across heads and put the head stride in soffset.
// A masked lane uses kOOBOffset; finish_weight must consume each loaded VGPR.
using WeightBufferLoad = mqa_beta::BufferLoadAccessor<4, mqa_beta::BufferEmitMode::InlineAsm>;

// Assemble four adjacent scales using the widest fully valid buffer requests.
// Each issued request is either entirely valid or uses the OOB sentinel.
template <int AlignN, bool EvenN>
__device__ __forceinline__ mqa_beta::F4 load_scale_buffer(mqa_beta::I4 descriptor, int col,
                                                           int valid_n) {
    static_assert(AlignN == 4 || AlignN == 2 || AlignN == 1);
    mqa_beta::F4 value;
#pragma unroll
    for (int j = 0; j < 4; j += AlignN) {
        uint32_t offset = uint32_t(col + j) * 4;
        if constexpr (!EvenN)
            offset = col + j < valid_n ? offset : ScalarBufferLoad::kOOBOffset;
        if constexpr (AlignN == 4) {
            value = mqa_beta::BufferLoadAccessor<16, mqa_beta::BufferEmitMode::Builtin>::load(
                descriptor, offset);
        } else if constexpr (AlignN == 2) {
            const mqa_beta::F2 pair =
                mqa_beta::BufferLoadAccessor<8, mqa_beta::BufferEmitMode::Builtin>::load(
                    descriptor, offset);
            value[j] = pair[0];
            value[j + 1] = pair[1];
        } else {
            value[j] = ScalarBufferLoad::load(descriptor, offset);
        }
    }
    return value;
}
} // namespace lightop::mqa_fp8
