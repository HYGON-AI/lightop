// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../utils/ops.hpp"
#include "../../epilogue.h"
#include "../../params.h"

namespace lightop::mqa_int8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
// Four waves share Q fragments while retaining K and head accumulators in
// registers. The producer mapping follows lightop_i8's contiguous-Q schedule.
template <int Heads, int BM, int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
__global__ __launch_bounds__(256) void tiled(MqaInt8Params p) {
    static_assert(BM == 16 || BM == 32);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    static_assert(Heads % GroupHeads == 0 && BM * GroupHeads <= 256);
    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int bx = blockIdx.y;
    const int by = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int row_begin = bx * BM;
    const int col_begin = by * BN;
    constexpr float neg_inf = -__builtin_inff();
    constexpr int WN = BN / 4;
    // Final Q/weight reads complete before the same storage becomes output scratch.
    union SharedStorage {
        struct {
            alignas(16) int8_t q[GroupHeads][BM][136];
            float weights[GroupHeads][BM];
        } inputs;
        alignas(16) float output[4][16][WN];
    };
    __shared__ SharedStorage shared;
    auto &q_lds = shared.inputs.q;
    auto &weights_lds = shared.inputs.weights;
    const int valid_m = EvenM ? BM : tail_rows<BM>(int64_t(p.m) - row_begin);
    const int valid_n = EvenN ? BN : tail_rows<BN>(int64_t(p.n) - col_begin);
    TileWindow tile{true, true};
    if constexpr (HasBounds) {
        // BM is 16/32, so restrict classifier lanes even for a complete M tile.
        // With EvenM=true, valid_m is a compile-time constant.
        tile = classify_window<false, CleanLogits>(p, row_begin, col_begin, valid_m, valid_n, lane);
    }
    const bool tile_full = tile.full;
    if (!tile.any) {
        const int vm = valid_m;
        const int vn = valid_n;
        if constexpr (HasBounds && CleanLogits) {
            if (EvenN || p.n % 4 == 0)
                fill_empty_tile<4>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
            else if (p.n % 2 == 0)
                fill_empty_tile<2>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
            else
                fill_empty_tile<1>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
        }
        return;
    }
    constexpr int NT = BN / 64, MT = BM / 16;
    I2 kr[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = col_begin + wave * (BN / 4) + t * 16 + lane % 16;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            kr[t][d] = {0, 0};
            if (EvenN || col < p.n)
                kr[t][d] = load_aligned4<I2>(p.kv + int64_t(col) * 128 + lane / 16 * 8 + d * 32);
        }
    }
    F4 accum[MT][NT] = {};
    constexpr int Copies = (BM * GroupHeads * 8 + 255) / 256;
    I4 next_q[Copies];
    const int8_t *q_block = p.q + int64_t(row_begin) * Heads * 128;
#pragma unroll
    for (int c = 0; c < Copies; ++c) {
        const int i = tid + c * 256;
        const int r = i / 8 % BM, h = i / (BM * 8), d = i % 8 * 16;
        next_q[c] = {0, 0, 0, 0};
        prefetch4(&next_q[c], q_block, (r * Heads + h) * 128 + d,
                  i < BM * GroupHeads * 8 && (EvenM || row_begin + r < p.m));
    }
    // Reuse a register-valued C operand for the first MMAC of every head.
    // gfx936 MMAC does not accept an inline constant for C.
    I4 dot_bias = dot_init();
    asm volatile("" : "+v"(dot_bias));
    float next_weight = 0.f;
    if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
        next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m + tid / BM];
#pragma unroll 1
    for (int group = 0; group < Heads; group += GroupHeads) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
        for (int c = 0; c < Copies; ++c) {
            const int i = tid + c * 256;
            if (i < BM * GroupHeads * 8) {
                const int r = i / 8 % BM, h = i / (BM * 8), d = i % 8 * 16;
                *reinterpret_cast<I4 *>(&q_lds[h][r][d]) = next_q[c];
            }
        }
        if (tid < BM * GroupHeads)
            weights_lds[tid / BM][tid % BM] = next_weight;
        __syncthreads();
        if (group + GroupHeads < Heads) {
#pragma unroll
            for (int c = 0; c < Copies; ++c) {
                const int i = tid + c * 256;
                const int r = i / 8 % BM, h = i / (BM * 8), d = i % 8 * 16;
                prefetch4(&next_q[c], q_block, (r * Heads + group + GroupHeads + h) * 128 + d,
                          i < BM * GroupHeads * 8 && (EvenM || row_begin + r < p.m));
            }
            if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
                next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m + group +
                                        GroupHeads + tid / BM];
        }
#pragma unroll
        for (int h = 0; h < GroupHeads; ++h) {
            I4 dot[MT][NT];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const I2 qr = *reinterpret_cast<const I2 *>(
                        &q_lds[h][lane % 16 + r * 16][lane / 16 * 8 + d * 32]);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        if (d == 0) {
                            // Distinct D/C avoids copying the bias into each accumulator.
                            // Inline MMAC scheduling is checked by check_asm_hazard.py.
                            asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %3"
                                         : "=&v"(dot[r][t])
                                         : "v"(qr), "v"(kr[t][d]), "v"(dot_bias));
                        } else {
                            mmac_i8(qr, kr[t][d], dot[r][t]);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < MT; ++r) {
                const float weight = weights_lds[h][lane % 16 + r * 16];
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    relu_weighted_sum(dot[r][t], weight, accum[r][t]);
            }
        }
        // Publish all final Q reads before the next producer reuses LDS.
        __syncthreads();
    }
    const bool vector_out = output_vector_aligned<4>(p.out, p.out_stride_m);
    const bool pair_out = !EvenN && p.n % 4 == 2;
    // gfx936's non-LIT accumulators own columns a+4*e, a=lane/16.
    // Scatter them into an XOR-swizzled row, then gather consecutive F4 packets.
    // Each wave owns a disjoint scratch region; no inter-wave exchange is needed.
    constexpr int RowShift = BN == 64 ? 2 : BN == 128 ? 1 : 0;
    constexpr int PacketsPerRow = WN / 4;
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        const int producer_row = lane % 16;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int producer_col = t * 16 + lane / 16 + e * 4;
                const float result = accum[r][t][e];
                const int swizzled_col = producer_col ^ ((producer_row >> RowShift) << 2);
                shared.output[wave][producer_row][swizzled_col] = result;
            }
        }
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
#pragma unroll
        for (int packet = 0; packet < NT; ++packet) {
            const int consumer_row = packet * (64 / PacketsPerRow) + lane / PacketsPerRow;
            const int consumer_col = 4 * (lane % PacketsPerRow);
            const int swizzled_col = consumer_col ^ ((consumer_row >> RowShift) << 2);
            F4 result =
                *reinterpret_cast<const F4 *>(&shared.output[wave][consumer_row][swizzled_col]);
            const int out_row = row_begin + r * 16 + consumer_row;
            const int out_col = col_begin + wave * WN + consumer_col;
            if ((EvenM || out_row < p.m) && (EvenN || out_col < p.n)) {
                Window bounds{};
                if constexpr (HasBounds && CleanLogits)
                    if (!tile_full)
                        bounds = load_window(p.ks, p.ke, p.n, out_row);
                if (EvenN || out_col + 3 < p.n)
                    result *= load_aligned4<F4>(p.kv_scale + out_col);
                else {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (out_col + e < p.n)
                            result[e] *= p.kv_scale[out_col + e];
                }
                if constexpr (HasBounds && CleanLogits) {
                    if (!tile_full) {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            if (!(bounds.start <= out_col + e && out_col + e < bounds.end))
                                result[e] = neg_inf;
                    }
                }
                float *dst = p.out + int64_t(out_row) * p.out_stride_m + out_col;
                if (EvenN || out_col + 3 < p.n) {
                    if (pair_out) {
                        // Keep the F4 LDS mapping and emit two contiguous F2 stores.
                        // memcpy retains support for four-byte-aligned output views.
                        const mqa_beta::F2 first = {result[0], result[1]};
                        const mqa_beta::F2 second = {result[2], result[3]};
                        __builtin_memcpy(dst, &first, sizeof(first));
                        __builtin_memcpy(dst + 2, &second, sizeof(second));
                    } else
                        store_logits4(dst, result, vector_out);
                } else if (pair_out) {
                    // N % 4 == 2 leaves exactly two columns in the final packet.
                    const mqa_beta::F2 last = {result[0], result[1]};
                    __builtin_memcpy(dst, &last, sizeof(last));
                } else {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (out_col + e < p.n)
                            store1(dst + e, result[e]);
                }
            }
        }
        // Drain reads before the next M16 overwrites this wave's scratch.
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
#endif
} // namespace lightop::mqa_int8
