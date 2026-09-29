// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lightop::mqa_fp8 {
struct MqaFp8Params {
    // Preserve the audited device argument offsets. No public/versioned ABI.
    uint32_t reserved_header[4]{};
    const uint8_t *q;
    const uint8_t *kv;
    const float *weights;
    const float *kv_scale; /* per-KV-token scale; nullptr selects unit scale */
    float *out;
    int64_t m, n, h, d;
    int64_t q_stride_m, q_stride_h, kv_stride_n;
    int64_t weights_stride_m, out_stride_m;
    /* Optional contiguous signed int32 [M] bounds, clamped to [0,N] on
     * device. Null ks/ke mean 0/N. ke <= ks denotes an empty window. */
    const int32_t *ks;
    const int32_t *ke;
    /* 0: only in-window output is defined; 1: write -inf outside it.
     * Both values permit skipping tiles with no window intersection. */
    uint32_t clean_logits;
    uint32_t bounds_reserved;
};

static_assert(std::is_standard_layout_v<MqaFp8Params>);
static_assert(offsetof(MqaFp8Params, q) == 16);
static_assert(offsetof(MqaFp8Params, ks) == 128);
static_assert(sizeof(MqaFp8Params) == 152);
} // namespace lightop::mqa_fp8
