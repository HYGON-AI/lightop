// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

namespace lightop::mqa_beta {
// A sorted union of nonoverlapping byte intervals, one per tensor row.
struct ByteRows {
    uint64_t begin;
    uint64_t rows;
    uint64_t stride;
    uint64_t width;
};

inline bool valid_rows(const ByteRows &r) {
    if (!r.rows || !r.width)
        return true;
    if (r.stride < r.width)
        return false;
    const auto limit = std::numeric_limits<uint64_t>::max();
    return r.width <= limit - r.begin && r.rows - 1 <= (limit - r.begin - r.width) / r.stride;
}

// Exact for the supported row layouts, including disjoint views sharing storage.
// Call only after valid_rows. Dense tensors collapse to one interval. Overlap
// is checked by addresses, not Storage identity (from_blob may wrap aliases).
inline bool rows_overlap(ByteRows a, ByteRows b) {
    if (!a.rows || !b.rows || !a.width || !b.width)
        return false;
    const auto collapse = [](ByteRows &r) {
        if (r.width == r.stride) {
            r.width *= r.rows;
            r.stride = r.width;
            r.rows = 1;
        }
    };
    collapse(a);
    collapse(b);
    const uint64_t a_end = a.begin + (a.rows - 1) * a.stride + a.width;
    const uint64_t b_end = b.begin + (b.rows - 1) * b.stride + b.width;
    if (a_end <= b.begin || b_end <= a.begin)
        return false;
    uint64_t i = 0, j = 0;
    while (i < a.rows && j < b.rows) {
        const uint64_t x = a.begin + i * a.stride;
        const uint64_t y = b.begin + j * b.stride;
        if (x < y + b.width && y < x + a.width)
            return true;
        if (x + a.width <= y) {
            const auto skip = (y - x - a.width) / a.stride + 1;
            i += std::min(skip, a.rows - i);
        } else {
            const auto skip = (x - y - b.width) / b.stride + 1;
            j += std::min(skip, b.rows - j);
        }
        // With equal strides the relative placement repeats from here. One
        // more pair proves whether all remaining rows overlap or are disjoint.
        if (a.stride == b.stride && i < a.rows && j < b.rows) {
            const uint64_t x1 = a.begin + i * a.stride;
            const uint64_t y1 = b.begin + j * b.stride;
            return x1 < y1 + b.width && y1 < x1 + a.width;
        }
    }
    return false;
}
} // namespace lightop::mqa_beta
