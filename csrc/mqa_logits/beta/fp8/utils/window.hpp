// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../window.h"
#include "../../epilogue.h"
#include "../../params.h"

namespace lightop {
namespace mqa_fp8 {
using mqa_beta::F2;
using mqa_beta::F4;
using mqa_beta::output_vector_aligned;
using mqa_beta::store1;
using mqa_beta::store4;
using mqa_beta::store_logits4;
using mqa_beta::tail_rows;

using mqa_beta::load_window;
using mqa_beta::TileWindow;
using mqa_beta::Window;

using mqa_beta::classify_window;
using mqa_beta::small_window;

template <int AlignN>
__device__ __forceinline__ void fill_empty_tile(const MqaFp8Params &p, int64_t m0, int64_t n0,
                                                int valid_m, int valid_n, int wave, int lane) {
    mqa_beta::fill_empty_tile<AlignN>(p.out, p.out_stride_m, m0, n0, valid_m, valid_n, wave, lane);
}

__device__ __forceinline__ void empty_output(const MqaFp8Params &p, int64_t m0, int64_t n0, int vm,
                                             int vn) {
    for (int i = threadIdx.x; i < vm * vn; i += 256)
        p.out[(m0 + i / vn) * p.out_stride_m + n0 + i % vn] = -__builtin_inff();
}

} // namespace mqa_fp8
} // namespace lightop
