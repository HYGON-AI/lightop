// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../epilogue.h"
#include "../../params.h"

namespace lightop {
namespace mqa_fp8 {
using mqa_beta::F2;
using mqa_beta::F4;
using mqa_beta::store1;
using mqa_beta::store4;
using mqa_beta::output_vector_aligned;
using mqa_beta::store_logits4;

// Read the same XOR-swizzled LDS tile with a lane grid matching store width.
// Align2: two rows per wave, 32 lanes/row. Align1: one row, 64 lanes/row.
// This avoids issuing four sparse scalar store passes over an Align4 lane grid.
template <int AlignN, bool EvenM>
__device__ __forceinline__ void store_logits_narrow(const MqaFp8Params &p, uint32_t shared_base,
                                                    int64_t m0, int64_t n0, int m16, int valid_m,
                                                    int valid_n, int wave, int lane) {
    static_assert(AlignN == 2 || AlignN == 1);
    using LdsF2 = F2 __attribute__((address_space(3)));
    using LdsF1 = float __attribute__((address_space(3)));
#pragma unroll
    for (int r = 0; r < 16 / AlignN; ++r) {
        const int row = AlignN * r + lane / (64 / AlignN);
        const int col = AlignN * (lane % (64 / AlignN));
        const int lds_index = wave * 1024 + row * 64 + (col ^ (row << 2));
        bool valid = wave * 64 + col < valid_n;
        if constexpr (!EvenM)
            valid = valid && m16 * 16 + row < valid_m;
        if (valid) {
            float *out = p.out + (m0 + m16 * 16 + row) * p.out_stride_m + n0 + wave * 64 + col;
            if constexpr (AlignN == 2) {
                const F2 value = reinterpret_cast<LdsF2 *>(uintptr_t(shared_base))[lds_index / 2];
                // Only 4-byte destination alignment is guaranteed for a
                // caller-owned view. LLVM can emit dwordx2 without asserting
                // an 8-byte-aligned C++ vector pointer or duplicating this path.
                __builtin_memcpy(out, &value, sizeof(value));
            } else {
                const float value = reinterpret_cast<LdsF1 *>(uintptr_t(shared_base))[lds_index];
                store1(out, value);
            }
        }
    }
}
} // namespace mqa_fp8
} // namespace lightop
