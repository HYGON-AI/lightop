// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../utils/ops.hpp"
#include "../../epilogue.h"
#include "../../params.h"

namespace lightop::mqa_bf16 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Common BM16/BM32/BM64 computation; m16.hpp and m64.hpp include this file.
// Four waves reuse register-resident BF16 K across head groups.
// Q is staged cooperatively; output is masked and coalesced in this kernel.
template <int BM, int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
__global__ __launch_bounds__(256) void tiled(MqaBf16Params p) {
    // make_grid enforces the int32 dimension range used by these kernels.
    p.m = int(p.m);
    p.n = int(p.n);
    static_assert(BM == 16 || BM == 32 || BM == 64);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    static_assert(BM * GroupHeads <= 128);
    const int Heads = __builtin_amdgcn_readfirstlane(int(p.h));
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
            alignas(16) I4 q[GroupHeads][BM / 16][4][64];
            float weights[GroupHeads][BM];
        } inputs;
        alignas(16) float output[4][16][WN];
    };
    __shared__ SharedStorage shared;
    auto &q_lds = shared.inputs.q;
    auto &weights_lds = shared.inputs.weights;
    // Four 16-byte fragments per lane cover a 256-byte BF16 row.
    // The low-bit permutation separates adjacent producer chunks while
    // preserving the bank groups of the vector LDS readers.
    const auto q_slot = [](int l, int fragment) {
        return l ^ (2 * fragment + ((l / 16) & 1));
    };
    const int valid_m = EvenM ? BM : tail_rows<BM>(int64_t(p.m) - row_begin);
    const int valid_n = EvenN ? BN : tail_rows<BN>(int64_t(p.n) - col_begin);
    TileWindow tile{true, true};
    if constexpr (HasBounds) {
        // BM is 16/32/64, so restrict classifier lanes even for a complete M tile.
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
    const I4 kv_resource = make_kv_resource(
        reinterpret_cast<const uint16_t *>(p.kv) + int64_t(col_begin) * 128);
    B4 kr[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = col_begin + wave * (BN / 4) + t * 16 + (lane % 4 * 4 + lane % 16 / 4);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            I4 value = {};
            if (EvenN || col < p.n)
                value = load_kv4(kv_resource, (col - col_begin) * 256 + lane / 16 * 64 + d * 16);
            // DTK miscompiles bit_cast of a vector swizzle; copy exact byte ranges.
            __builtin_memcpy(&kr[t][d * 2], &value, 8);
            __builtin_memcpy(&kr[t][d * 2 + 1],
                             reinterpret_cast<const char *>(&value) + 8, 8);
        }
    }
    F4 accum[MT][NT] = {};
    F4 zero = {};
    // Keep one shared zero-C tuple resident across the head loop.
    if constexpr (GroupHeads == 1)
        asm volatile("" : "+v"(zero));
    constexpr int Copies = (BM * GroupHeads * 16 + 255) / 256;
    I4 next_q[Copies];
    const uint16_t *q_block = reinterpret_cast<const uint16_t *>(p.q) + int64_t(row_begin) * Heads * 128;
#pragma unroll
    for (int c = 0; c < Copies; ++c) {
        const int i = tid + c * 256;
        const int r = i / 16 % BM, h = i / (BM * 16), d = i % 16 * 8;
        next_q[c] = {0, 0, 0, 0};
        prefetch4(&next_q[c], q_block, (r * Heads + h) * 128 + d,
                  i < BM * GroupHeads * 16 && (EvenM || row_begin + r < p.m));
    }
    float next_weight = 0.f;
    if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
        next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m + tid / BM];
#pragma unroll 1
    for (int group = 0; group < Heads; group += GroupHeads) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
        for (int c = 0; c < Copies; ++c) {
            const int i = tid + c * 256;
            if (i < BM * GroupHeads * 16) {
                const int r = i / 16 % BM, h = i / (BM * 16);
                const int chunk = i % 16, fragment = chunk % 4;
                const int qlane = chunk / 4 * 16 + r % 16;
                q_lds[h][r / 16][fragment][q_slot(qlane, fragment)] = next_q[c];
            }
        }
        if (tid < BM * GroupHeads)
            weights_lds[tid / BM][tid % BM] = next_weight;
        __syncthreads();
        if (group + GroupHeads < Heads) {
#pragma unroll
            for (int c = 0; c < Copies; ++c) {
                const int i = tid + c * 256;
                const int r = i / 16 % BM, h = i / (BM * 16), d = i % 16 * 8;
                prefetch4(&next_q[c], q_block, (r * Heads + group + GroupHeads + h) * 128 + d,
                          i < BM * GroupHeads * 16 && (EvenM || row_begin + r < p.m));
            }
            if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
                next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m + group +
                                        GroupHeads + tid / BM];
        }
#pragma unroll
        for (int h = 0; h < GroupHeads; ++h) {
            F4 dot[MT][NT];
            // Grouped short-row tiles are latency-sensitive. Their builtin
            // zero-C form schedules better than an additional resident tuple.
            if constexpr (GroupHeads > 1) {
#pragma unroll
                for (int r = 0; r < MT; ++r)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        dot[r][t] = {};
            }
#pragma unroll
            for (int fragment = 0; fragment < 4; ++fragment) {
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const I4 packet = q_lds[h][r][fragment][q_slot(lane, fragment)];
                    B4 q[2];
                    // Copy byte ranges: DTK miscompiles bit_cast(vector.xy/zw).
                    __builtin_memcpy(&q[0], &packet, 8);
                    __builtin_memcpy(&q[1], reinterpret_cast<const char *>(&packet) + 8, 8);
#pragma unroll
                    for (int step = 0; step < 2; ++step) {
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            if (GroupHeads == 1 && fragment == 0 && step == 0)
                                dot[r][t] = mmac_first(q[step], kr[t][fragment * 2 + step], zero);
                            else
                                mmac_bf16(q[step], kr[t][fragment * 2 + step], dot[r][t]);
                        }
                    }
                }
            }
            // BN256 amortizes the detector; excluding BM64 avoids register spills.
            if constexpr (BM <= 32 && BN == 256 && GroupHeads == 1) {
                // The detector never contributes to logits. False positives
                // from finite overflow or opposite infinities take the slow path.
                F4 detect = dot[0][0];
#pragma unroll
                for (int r = 0; r < MT; ++r)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        if (r != 0 || t != 0)
                            detect += dot[r][t];
                using F2 = float __attribute__((ext_vector_type(2)));
                const F2 pair = detect.xy + detect.zw;
                const float total = pair[0] + pair[1];
                const bool exceptional = __builtin_amdgcn_ballot_w64(__builtin_isnan(total)) != 0;
                if (__builtin_expect(exceptional, false)) {
#pragma unroll
                    for (int r = 0; r < MT; ++r)
#pragma unroll
                        for (int t = 0; t < NT; ++t)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                dot[r][t][e] = relu(dot[r][t][e]);
                } else {
#pragma unroll
                    for (int r = 0; r < MT; ++r)
#pragma unroll
                        for (int t = 0; t < NT; ++t)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                dot[r][t][e] = relu_non_nan(dot[r][t][e]);
                }
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const float weight = weights_lds[h][lane % 16 + r * 16];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        weighted_sum_only(dot[r][t], weight, accum[r][t]);
                    }
                }
            } else {
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const float weight = weights_lds[h][lane % 16 + r * 16];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        relu_weighted_sum(dot[r][t], weight, accum[r][t]);
                    }
                }
            }
        }
        // Publish all final Q reads before the next producer reuses LDS.
        __syncthreads();
    }
    const bool vector_out = output_vector_aligned<4>(p.out, p.out_stride_m);
    const bool pair_out = !EvenN && p.n % 4 == 2;
    // Permute the key lanes as in INT8 so each accumulator owns four
    // consecutive keys. Both LDS phases transfer aligned F4 packets.
    constexpr int PacketsPerRow = WN / 4;
    const auto swizzle = [](int row) {
        return PacketsPerRow == 4 ? (row >> 1) & 3 : row & (PacketsPerRow - 1);
    };
    using LdsF4 = F4 __attribute__((address_space(3)));
    const uint32_t lds_base = uint32_t(reinterpret_cast<uintptr_t>(&shared.output[0][0][0]));
    auto *output = reinterpret_cast<LdsF4 *>(uintptr_t(lds_base)) +
                   wave * 16 * PacketsPerRow;
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        const int producer_row = lane % 16;
#pragma unroll
        for (int t = 0; t < NT; ++t)
            output[producer_row * PacketsPerRow +
                   ((t * 4 + lane / 16) ^ swizzle(producer_row))] = accum[r][t];
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
#pragma unroll
        for (int packet = 0; packet < NT; ++packet) {
            const int consumer_row = packet * (64 / PacketsPerRow) + lane / PacketsPerRow;
            const int consumer_col = 4 * (lane % PacketsPerRow);
            F4 result = output[consumer_row * PacketsPerRow +
                               ((lane % PacketsPerRow) ^ swizzle(consumer_row))];
            const int out_row = row_begin + r * 16 + consumer_row;
            const int out_col = col_begin + wave * WN + consumer_col;
            if ((EvenM || out_row < p.m) && (EvenN || out_col < p.n)) {
                Window bounds{};
                if constexpr (HasBounds && CleanLogits)
                    if (!tile_full)
                        bounds = load_window(p.ks, p.ke, p.n, out_row);
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
    return;
}

// Host-side BM32 entry into the common tiled kernel in m32.hpp.
template <int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
inline auto select_m32_kernel() {
    return tiled<32, BN, GroupHeads, EvenM, EvenN, HasBounds, CleanLogits>;
}
#endif
} // namespace lightop::mqa_bf16
