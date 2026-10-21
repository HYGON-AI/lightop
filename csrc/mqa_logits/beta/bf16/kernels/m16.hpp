// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "m32.hpp"

namespace lightop::mqa_bf16 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Host-side BM16 entry into the common tiled kernel in m32.hpp.
template <int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
inline auto select_m16_kernel() {
    return tiled<16, BN, GroupHeads, EvenM, EvenN, HasBounds, CleanLogits>;
}

// Two four-wave partitions process disjoint head ranges. A fixed-order LDS
// reduction combines them within the CTA, without atomics or extra launches.
template <int BM, int BN, int GroupHeads, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
__global__ __launch_bounds__(512) void split_head(MqaBf16Params p) {
    // make_grid enforces the int32 dimension range used by these kernels.
    p.m = int(p.m);
    p.n = int(p.n);
    constexpr int Parts = 2, WavesPerPart = 4;
    static_assert(BM == 16 || BM == 32 || BM == 64);
    static_assert(BN % (WavesPerPart * 16) == 0);
    constexpr int PartThreads = WavesPerPart * 64;
    static_assert(BM * GroupHeads <= 128);
    const int Heads = __builtin_amdgcn_readfirstlane(int(p.h));
    const int part = __builtin_amdgcn_readfirstlane(threadIdx.x / PartThreads);
    const int LocalHeads = Heads / Parts;
    const int head_base = part * LocalHeads;
    const int tid = threadIdx.x % PartThreads;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int bx = blockIdx.y;
    const int by = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int row_begin = bx * BM;
    const int col_begin = by * BN;
    constexpr float neg_inf = -__builtin_inff();
    constexpr int WN = BN / WavesPerPart;
    // Final Q/weight reads complete before the same storage becomes output scratch.
    union SharedStorage {
        struct {
            alignas(16) I4 q[Parts][GroupHeads][BM / 16][4][64];
            float weights[Parts][GroupHeads][BM];
        } inputs;
        alignas(16) float output[WavesPerPart][16][WN];
        F4 partial[Parts - 1][BM / 16][BN / (WavesPerPart * 16)][WavesPerPart][64];
    };
    __shared__ SharedStorage shared;
    auto &q_lds = shared.inputs.q[part];
    auto &weights_lds = shared.inputs.weights[part];
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
        if constexpr (HasBounds && CleanLogits) {
            for (int i = threadIdx.x; i < valid_m * valid_n;
                 i += Parts * PartThreads) {
                const int row = row_begin + i / valid_n;
                const int col = col_begin + i % valid_n;
                store1(p.out + int64_t(row) * p.out_stride_m + col, neg_inf);
            }
        }
        return;
    }
    constexpr int NT = BN / (WavesPerPart * 16), MT = BM / 16;
    const I4 kv_resource = make_kv_resource(
        reinterpret_cast<const uint16_t *>(p.kv) + int64_t(col_begin) * 128);
    B4 kr[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = col_begin + wave * (BN / WavesPerPart) + t * 16 + (lane % 4 * 4 + lane % 16 / 4);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            I4 value = {};
            if (EvenN || col < p.n)
                value = load_kv4(kv_resource, (col - col_begin) * 256 + lane / 16 * 64 + d * 16);
            __builtin_memcpy(&kr[t][d * 2], &value, 8);
            __builtin_memcpy(&kr[t][d * 2 + 1],
                             reinterpret_cast<const char *>(&value) + 8, 8);
        }
    }
    F4 accum[MT][NT] = {};
    F4 zero = {};
    // Keep one shared zero-C tuple resident across the head loop.
    asm volatile("" : "+v"(zero));
    constexpr int Copies = (BM * GroupHeads * 16 + PartThreads - 1) / PartThreads;
    I4 next_q[Copies];
    const uint16_t *q_block = reinterpret_cast<const uint16_t *>(p.q) + int64_t(row_begin) * Heads * 128;
#pragma unroll
    for (int c = 0; c < Copies; ++c) {
        const int i = tid + c * PartThreads;
        const int r = i / 16 % BM, h = i / (BM * 16), d = i % 16 * 8;
        next_q[c] = {0, 0, 0, 0};
        prefetch4(&next_q[c], q_block, (r * Heads + head_base + h) * 128 + d,
                  i < BM * GroupHeads * 16 && (EvenM || row_begin + r < p.m));
    }
    float next_weight = 0.f;
    if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
        next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m +
                                head_base + tid / BM];
#pragma unroll 1
    for (int group = 0; group < LocalHeads; group += GroupHeads) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
        for (int c = 0; c < Copies; ++c) {
            const int i = tid + c * PartThreads;
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
        if (group + GroupHeads < LocalHeads) {
#pragma unroll
            for (int c = 0; c < Copies; ++c) {
                const int i = tid + c * PartThreads;
                const int r = i / 16 % BM, h = i / (BM * 16), d = i % 16 * 8;
                prefetch4(&next_q[c], q_block,
                          (r * Heads + head_base + group + GroupHeads + h) * 128 + d,
                          i < BM * GroupHeads * 16 && (EvenM || row_begin + r < p.m));
            }
            if (tid < BM * GroupHeads && (EvenM || row_begin + tid % BM < p.m))
                next_weight = p.weights[int64_t(row_begin + tid % BM) * p.weights_stride_m +
                                        head_base + group + GroupHeads + tid / BM];
        }
#pragma unroll
        for (int h = 0; h < GroupHeads; ++h) {
            F4 dot[MT][NT];
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
                            if (fragment == 0 && step == 0)
                                dot[r][t] = mmac_first(q[step], kr[t][fragment * 2 + step], zero);
                            else
                                mmac_bf16(q[step], kr[t][fragment * 2 + step], dot[r][t]);
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
    // Each partition produces a partial sum in a fixed head order.
    // Both barriers include the whole CTA; output reuse begins only after
    // every partial sum has been consumed. There are no floating atomics.
    if (part > 0) {
#pragma unroll
        for (int r = 0; r < MT; ++r) {
#pragma unroll
            for (int t = 0; t < NT; ++t)
                shared.partial[part - 1][r][t][wave][lane] = accum[r][t];
        }
    }
    __syncthreads();
    if (part == 0) {
#pragma unroll
        for (int r = 0; r < MT; ++r) {
#pragma unroll
            for (int t = 0; t < NT; ++t)
                for (int part_index = 0; part_index < Parts - 1; ++part_index)
                    accum[r][t] += shared.partial[part_index][r][t][wave][lane];
        }
    }
    __syncthreads();
    if (part > 0)
        return;
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
#endif
} // namespace lightop::mqa_bf16
