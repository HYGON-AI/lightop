// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../arch.h"

namespace lightop::mqa_fp8 {
using mqa_beta::I2;
using mqa_beta::I4;
using mqa_beta::F2;
using mqa_beta::F4;
using mqa_beta::wait_vm;
using mqa_beta::barrier;
using mqa_beta::finish_kv_half;
using mqa_beta::finish_weight;
using mqa_beta::finish_scales;
using MLS64x32B8 = mqa_beta::MLSAccessor<mqa_beta::MLSKind::B8_64x32_T1_F1>;
using MLS128x16B8 = mqa_beta::MLSAccessor<mqa_beta::MLSKind::B8_128x16_T1_F0>;
using MLS64x16B8 = mqa_beta::MLSAccessor<mqa_beta::MLSKind::B8_64x16_T1_F0>;
using mqa_beta::shuffle_weight;
using mqa_beta::relu;

// Keep both FP32 pairs live across ReLU so LLVM can allocate adjacent VGPRs.
// gfx938 packed FMA has no op_sel broadcast: both weight lanes are explicit.
__device__ __forceinline__ F4 accumulate_relu_ready(F4 sum, F4 dot, F2 weight) {
    // The paired-dot caller issues four independent MMACs before consuming
    // its first dot, then four independent maxes before consuming the second.
    // The actual binary audit verifies this latency cover for every dot.
    __builtin_hcu_sched_barrier(0);
    asm volatile("; MQA_RELU_PAIRED" ::"v"(dot) : "memory");
    __builtin_hcu_sched_barrier(0);
    F4 activated;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        activated[j] = relu(dot[j]);
    sum.xy = activated.xy * weight + sum.xy;
    sum.zw = activated.zw * weight + sum.zw;
    return sum;
}

template <int Pair> __device__ __forceinline__ F4 mmac(I4 a, I4 b, F4 c) {
    static_assert(Pair == 0 || Pair == 1);
    const I2 av = {a[2 * Pair], a[2 * Pair + 1]};
    const I2 bv = {b[2 * Pair], b[2 * Pair + 1]};
    return mqa_beta::mmac<mqa_beta::MMADataType::FP8E4M3FN, true, false,
                          mqa_beta::MMAEmitMode::InlineAsm>(av, bv, c);
}

__device__ __forceinline__ F4 dot128(I4 a0, I4 a1, I4 b0, I4 b1, F4 zero) {
    F4 d = mmac<0>(a0, b0, zero);
    d = mmac<1>(a0, b0, d);
    d = mmac<0>(a1, b1, d);
    return mmac<1>(a1, b1, d);
}
} // namespace lightop::mqa_fp8
