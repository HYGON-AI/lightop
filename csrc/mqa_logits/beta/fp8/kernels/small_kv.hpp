// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../utils/compute.hpp"
#include "../utils/load.hpp"
#include "../utils/window.hpp"
#include "../../params.h"

namespace lightop {
namespace mqa_fp8 {
template <int BN, bool FullTile = false>
__device__ __forceinline__ void small_kv(const MqaFp8Params &p, uint32_t lds, int64_t n0, int vn,
                                         int wave, I4 (&kv)[BN / 64][2]) {
    constexpr int NF = BN / 64; // N16 fragments per wave.
    constexpr int WN = BN / 4;
#pragma unroll
    for (int phase = 0; phase < (NF + 1) / 2; ++phase) {
        constexpr int Loads = NF == 1 ? 1 : 2;
#pragma unroll
        for (int j = 0; j < Loads; ++j) {
            const int nr = wave * WN + (2 * phase + j) * 16;
            const int vr = FullTile ? 16 : tail_rows<16>(vn - nr);
            const int64_t row = vr ? n0 + nr : n0;
            if (vr > 0) {
                const I4 descriptor =
                    MLS128x16B8::make_descriptor<!FullTile>(p.kv + row * 128, 128, vr);
                MLS128x16B8::matrix_load(descriptor, lds + wave * 4096 + j * 2048);
            }
        }
        wait_vm();
#pragma unroll
        for (int j = 0; j < Loads; ++j) {
            kv[2 * phase + j][0] =
                MLS128x16B8::ds_read_format<>(lds + wave * 4096 + j * 2048);
            kv[2 * phase + j][1] =
                MLS128x16B8::ds_read_format<1024>(lds + wave * 4096 + j * 2048);
        }
        if constexpr (NF == 1)
            finish_kv_half(kv[0][0], kv[0][1], kv[0][0], kv[0][1]);
        else
            finish_kv_half(kv[2 * phase][0], kv[2 * phase][1], kv[2 * phase + 1][0],
                           kv[2 * phase + 1][1]);
    }
    barrier(); // Publish completion before another wave overwrites KV LDS.
}

} // namespace mqa_fp8
} // namespace lightop
