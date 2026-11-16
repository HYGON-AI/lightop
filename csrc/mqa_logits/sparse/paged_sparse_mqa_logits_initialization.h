// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace lightop::deep_gemm::sparse_mqa_detail {

// The gfx938 fill implementation is shared by both architectures. This only
// initializes memory; it contains no architecture-specific matrix instructions.
// Some gfx936 routes allocate two index flags instead of four. Preserve that
// allocation contract instead of writing the gfx938 four-flag layout blindly.
template<bool ConditionalRouteFlags, typename Args>
__global__ void sparse_fp8_fp4_paged_mqa_logits__initialize_prefill_output_and_scratch(
    Args a, int64_t nvec, int64_t union_words) {
    using floatx4 = float __attribute__((ext_vector_type(4)));
    using uintx4 = unsigned __attribute__((ext_vector_type(4)));
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    if (tid == 0) {
        *a.exceptional = 0;
        *a.fp8_invalid = 0;
        *a.cache_count = 0;
        if (a.index_invalid) {
            a.index_invalid[0] = 0;
            a.index_invalid[1] = 0;
            if (!ConditionalRouteFlags || a.size_auto) {
                a.index_invalid[2] = 0;
                a.index_invalid[3] = 0;
            }
        }
    }
    const floatx4 negative = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int64_t i = tid; i < nvec; i += stride)
        reinterpret_cast<floatx4*>(a.out)[i] = negative;
    const uintx4 zeros = {0, 0, 0, 0};
    const uintx4 missing = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    for (int64_t i = tid; i < a.num_pages / 4; i += stride)
        reinterpret_cast<uintx4*>(a.cache_pages)[i] = missing;
    const int64_t tail = (a.num_pages / 4) * 4 + tid;
    if (tail < a.num_pages) a.cache_pages[tail] = -1;
    for (int64_t i = tid; i < a.num_pages * 2; i += stride)
        reinterpret_cast<uintx4*>(a.selected_blocks)[i] = zeros;
    if (a.union_mask)
        for (int64_t i = tid; i < union_words / 4; i += stride)
            reinterpret_cast<uintx4*>(a.union_mask)[i] = zeros;
}

}  // namespace lightop::deep_gemm::sparse_mqa_detail
