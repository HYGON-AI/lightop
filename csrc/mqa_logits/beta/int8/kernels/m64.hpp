// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../utils/ops.hpp"
#include "../../epilogue.h"
#include "../../params.h"
#include <type_traits>

namespace lightop::mqa_int8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Four waves share Q fragments while retaining K and head accumulators in
// registers. Wave w owns columns [w*BN/4, (w+1)*BN/4) of all BM rows.
//
// D permutation (as in FP8): in d-step d, MMAC lane l = 16g + j holds bytes
// [32g + 8d, 32g + 8d + 8) of its row/column j, so a lane needs 32 contiguous bytes per
// row: two 16-byte K loads and two 16-byte Q reads per M16 tile. Q is stored unpadded in
// fragment order [stage][M16 tile][half][slot]; lane l reads slot l ^ (2g + half). Eight
// consecutive readers stay in one aligned group of eight slots, and the eight 16-byte
// chunks of one row land in eight distinct 16-byte bank groups.
//
// This 64x256 kernel follows FP8's M64 schedule: two Q stages fill the
// 16 KiB output scratch, so one barrier per head both publishes the next head and releases
// the previous one; weights are lane-per-row buffer loads broadcast by ds_bpermute; and
// the head loop runs M16-row by M16-row with row r+1's Q read ahead. Keeping all sixteen
// 16x16 dot tiles live (the BM<=32 order) would exceed 192 VGPRs, i.e. four CTAs per CU.
template <int Heads, bool EvenM, bool EvenN, bool HasBounds, bool CleanLogits,
          int FusedMode = 0, int ScoreBits = 32>
__global__ __launch_bounds__(256) void m64(
    std::conditional_t<FusedMode == 0, MqaInt8Params, MqaTopKParams> p) {
    constexpr int BM = 64, BN = 256;
    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int bx = blockIdx.y;
    const int by = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int row_begin = bx * BM;
    const int col_begin = FusedMode == 1 ? (by * ((p.n + BN - 1) / BN) / 8) * BN : by * BN;
    if constexpr (FusedMode == 3) {
        p.m = *p.fallback_count;
        if (row_begin >= p.m) return;
    }
    constexpr float neg_inf = -__builtin_inff();
    constexpr int WN = BN / 4, NT = BN / 64, MT = BM / 16;
    // Two unpadded Q stages, with weights held in registers.
    constexpr int Rows = BM;
    constexpr int StageBytes = Rows * 128;
    constexpr int InputBytes = 2 * StageBytes;
    constexpr int OutputBytes = 4 * 16 * WN * 4;
    static_assert(InputBytes == OutputBytes, "two stages exactly fill the scratch");
    // Final Q reads complete before the same storage becomes output scratch.
    __shared__ __align__(16) int8_t shared[InputBytes > OutputBytes ? InputBytes : OutputBytes];
    const int valid_m = EvenM ? BM : tail_rows<BM>(int64_t(p.m) - row_begin);
    const int valid_n = EvenN ? BN : tail_rows<BN>(int64_t(p.n) - col_begin);
    TileWindow tile{true, true};
    if constexpr (HasBounds) {
        tile = classify_window<EvenM, CleanLogits>(p, row_begin, col_begin, valid_m, valid_n, lane);
    }
    const bool tile_full = tile.full;
    if (!tile.any) {
        const int vm = valid_m;
        const int vn = valid_n;
        if constexpr (HasBounds && CleanLogits && (FusedMode == 0 || FusedMode == 3)) {
            if (EvenN || p.n % 4 == 0)
                fill_empty_tile<4>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
            else if (p.n % 2 == 0)
                fill_empty_tile<2>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
            else
                fill_empty_tile<1>(p.out, p.out_stride_m, row_begin, col_begin, vm, vn, wave, lane);
        }
        return;
    }
    const I4 kv_resource = make_kv_resource(p.kv + int64_t(col_begin) * 128);
    // F4 epilogue: B column j of N16 tile t holds key t*16 + 4*(j%4) + j/4, so a lane's
    // accumulator holds four consecutive keys.
    const int key = lane % 4 * 4 + lane % 16 / 4;
    I4 kr[NT][2];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = col_begin + wave * WN + t * 16 + key;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            kr[t][half] = I4{0, 0, 0, 0};
            if (EvenN || col < p.n)
                kr[t][half] = __builtin_bit_cast(
                    I4, __builtin_amdgcn_buffer_load_dwordx4(
                            kv_resource, 0, (col - col_begin) * 128 + lane / 16 * 32 + half * 16,
                            false, true));
        }
    }
    // F4 epilogue: every packet a lane stores has the same columns, so kv_scale is read once.
    constexpr int Slots = WN / 4; // F4 packets per scratch row
    const int out_col = col_begin + wave * WN + 4 * (lane % Slots);
    F4 scale = {0.f, 0.f, 0.f, 0.f};
    if constexpr (FusedMode != 7) {
    if (EvenN || out_col + 3 < p.n)
        scale = load_aligned4<F4>(p.kv_scale + out_col);
    else {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (out_col + e < p.n)
                scale[e] = p.kv_scale[out_col + e];
    }
    }
    F4 accum[MT][NT] = {};
    // Thread i = tid + 256c copies 16-byte chunk i%8 of row i/8 = (head i/8/BM, query
    // i/8%BM): eight lanes read one complete 128-byte row. Rows past M repeat the last
    // valid query; their outputs are never stored.
    constexpr int Copies = (Rows * 8 + 255) / 256;
    const int8_t *q_block =
        reinterpret_cast<const int8_t *>(p.q) + int64_t(row_begin) * Heads * 128;
    auto q_slot = [](int l, int half) { return l ^ (2 * (l / 16) + half); };
    int q_src[Copies], q_dst[Copies];
#pragma unroll
    for (int c = 0; c < Copies; ++c) {
        const int i = tid + c * 256, chunk = i % 8, h = i / 8 / BM, r = i / 8 % BM;
        const int src_row = EvenM ? r : min(r, valid_m - 1);
        q_src[c] = (src_row * Heads + h) * 128 + chunk * 16;
        q_dst[c] = h * (BM * 128) +
                   ((r / 16 * 2 + chunk % 2) * 64 + q_slot(chunk / 2 * 16 + r % 16, chunk % 2)) *
                       16;
    }
    auto load_q = [&](int group, I4 (&q)[Copies]) {
#pragma unroll
        for (int c = 0; c < Copies; ++c)
            q[c] = __builtin_bit_cast(I4, __builtin_amdgcn_buffer_load_dwordx4(
                                              make_kv_resource(q_block + group * 128), 0,
                                              q_src[c], false, false));
    };
    auto store_q = [&](int stage, const I4 (&q)[Copies]) {
#pragma unroll
        for (int c = 0; c < Copies; ++c)
            *reinterpret_cast<I4 *>(shared + stage * StageBytes + q_dst[c]) = q[c];
    };
    // Reuse a register-valued C operand for the first MMAC of every head.
    // gfx936 MMAC does not accept an inline constant for C.
    I4 dot_bias = dot_init();
    asm volatile("" : "+v"(dot_bias));
    // One d-step of the (M16, N16) tile t; d-step 0 starts from the bias register.
    auto mma = [&](int d, const I4 &qd, int t, I4 &dot) {
        const I2 qr = d % 2 ? I2{qd[2], qd[3]} : I2{qd[0], qd[1]};
        const I4 &kd = kr[t][d / 2];
        const I2 k = d % 2 ? I2{kd[2], kd[3]} : I2{kd[0], kd[1]};
        if (d == 0) {
            // Distinct D/C avoids copying the bias into each accumulator.
            // Inline MMAC scheduling is checked by check_asm_hazard.py.
            asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %3"
                         : "=&v"(dot)
                         : "v"(qr), "v"(k), "v"(dot_bias));
        } else {
            mmac_i8(qr, k, dot);
        }
    };
    // Lane l loads the weight of row l; ds_bpermute hands row 16r + l%16 to M16 tile r.
    // make_grid bounds weights_stride_m so the lane offset fits in 31 bits.
    const int w_row = EvenM ? lane : min(lane, valid_m - 1);
    const float *w_block = p.weights + int64_t(row_begin) * p.weights_stride_m;
    const int w_src = w_row * int(p.weights_stride_m) * 4;
    auto load_weight = [&](int h) {
        return __builtin_bit_cast(float, __builtin_amdgcn_buffer_load_dword(
                                             make_kv_resource(w_block + h), 0, w_src, false,
                                             false));
    };
    {
        I4 q[Copies] = {};
        load_q(0, q);
        store_q(0, q);
    }
    float weight = load_weight(0);
    __syncthreads();
    // A runtime trip count (p.h == Heads) keeps one loop shape for every H: with the
    // constant, H=2/4 were restructured to 181-199 VGPRs (3 CTAs/CU above 192).
    const int heads = int(p.h);
    auto head_step = [&](auto stage_tag, int h) {
        constexpr int Stage = decltype(stage_tag)::value;
        // Prefetch the next head into the other stage; the last head reloads itself.
        // A uniform branch here splits the block and costs ~38 VGPRs (206 vs 168).
        const int next = h + 1 < heads ? h + 1 : h;
        I4 next_q[Copies] = {};
        load_q(next, next_q);
        const float next_weight = load_weight(next);
        const int8_t *stage = shared + Stage * StageBytes;
        auto read_q = [&](int r, I4 (&q)[2]) {
#pragma unroll
            for (int half = 0; half < 2; ++half)
                q[half] = *reinterpret_cast<const I4 *>(
                    stage + (r * 2 + half) * 1024 + q_slot(lane, half) * 16);
        };
        auto read_w = [&](int r) {
            return __builtin_bit_cast(
                float, __builtin_amdgcn_ds_bpermute((r * 16 + lane % 16) * 4,
                                                    __builtin_bit_cast(int, weight)));
        };
        // Software pipeline: row r + 1's Q and weight are in flight while row r computes.
        I4 q[2][2];
        float w[2];
        read_q(0, q[0]);
        w[0] = read_w(0);
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int cur = r & 1;
            if (r + 1 < MT) {
                read_q(r + 1, q[cur ^ 1]);
                w[cur ^ 1] = read_w(r + 1);
            }
            I4 dot[NT];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    mma(d, q[cur][d / 2], t, dot[t]);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t)
                relu_weighted_sum(dot[t], w[cur], accum[r][t]);
        }
        // All waves finished reading the other stage before the previous barrier.
        store_q(Stage ^ 1, next_q);
        // Publishes the next head; after the last head it releases both stages.
        __syncthreads();
        weight = next_weight;
    };
#pragma unroll 1
    for (int h = 0; h < heads; h += 2) {
        head_step(std::integral_constant<int, 0>{}, h);
        if (h + 1 < heads)
            head_step(std::integral_constant<int, 1>{}, h + 1);
    }
    if constexpr (FusedMode == 7) {
        F4 direct_scale[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int col = col_begin + wave * WN + t * 16 + (lane / 16) * 4;
            direct_scale[t] = F4{0.f, 0.f, 0.f, 0.f};
            if (EvenN || col + 3 < p.n)
                direct_scale[t] = load_aligned4<F4>(p.kv_scale + col);
            else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (col + e < p.n) direct_scale[t][e] = p.kv_scale[col + e];
            }
        }
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int row = row_begin + r * 16 + lane % 16;
            if (EvenM || row < p.m) {
                const float gate = reinterpret_cast<const float *>(p.thresholds)[row];
                if (isfinite(gate)) {
                    const auto window = tile_full ? Window{0, p.n} : load_window(p.ks, p.ke, p.n, row);
                    const uint32_t raw_pivot = __float_as_uint(gate);
                    const uint32_t pivot = raw_pivot & 0x80000000u ? ~raw_pivot : raw_pivot | 0x80000000u;
                    uint32_t mask = 0;
                    int local = 0;
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const F4 value = accum[r][t] * direct_scale[t];
                        accum[r][t] = value;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int col = col_begin + wave * WN + t * 16 + (lane / 16) * 4 + e;
                            const bool valid = window.start <= col && col < window.end;
                            if (valid && !isfinite(value[e])) atomicOr(p.counts + row, 1 << 28);
                            const uint32_t bits = __float_as_uint(value[e]);
                            const uint32_t ordered = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
                            const bool keep = valid && ordered >= pivot;
                            mask |= uint32_t(keep) << (t * 4 + e);
                            local += keep;
                        }
                    }
                    int prefix = local;
                    int previous = __shfl_up(prefix, 16, 64);
                    if (lane >= 16) prefix += previous;
                    previous = __shfl_up(prefix, 32, 64);
                    if (lane >= 32) prefix += previous;
                    const int total = __shfl(prefix, 48 + lane % 16, 64);
                    int offset = 0;
                    if (lane < 16 && total) offset = atomicAdd(p.counts + row, total);
                    offset = __shfl(offset, lane % 16, 64) + prefix - local;
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int col = col_begin + wave * WN + t * 16 + (lane / 16) * 4;
                        if (((mask >> (t * 4)) & 15u) == 15u && (offset & 3) == 0) {
                            if (offset + 3 < 16384) {
                                const int64_t target = int64_t(row) * 16384 + offset;
                                *reinterpret_cast<F4 *>(static_cast<uint32_t *>(p.candidate_keys) + target) = accum[r][t];
                                *reinterpret_cast<I4 *>(p.candidate_ids + target) = I4{col, col + 1, col + 2, col + 3};
                            }
                            offset += 4;
                        } else {
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                if (mask & (1u << (t * 4 + e))) {
                                    const int slot = offset++;
                                    if (slot < 16384) {
                                        const int64_t target = int64_t(row) * 16384 + slot;
                                        static_cast<uint32_t *>(p.candidate_keys)[target] = __float_as_uint(accum[r][t][e]);
                                        p.candidate_ids[target] = col + e;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        return;
    }
    const bool vector_out = output_vector_aligned<4>(p.out, p.out_stride_m);
    const bool pair_out = !EvenN && p.n % 4 == 2;
    // Masks and writes one scaled F4 packet: row out_row, keys [packet_col, packet_col + 4).
    auto emit = [&](int out_row, int packet_col, F4 result, const Window &bounds) {
        if constexpr (HasBounds && CleanLogits) {
            if (!tile_full) {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (!(bounds.start <= packet_col + e && packet_col + e < bounds.end))
                        result[e] = neg_inf;
            }
        }
        if constexpr (FusedMode == 2 || FusedMode == 4 || FusedMode == 6) {
            const auto window = FusedMode == 6 && tile_full ? Window{0, p.n} : load_window(p.ks, p.ke, p.n, out_row);
            const float previous = reinterpret_cast<const float*>(p.thresholds)[out_row];
            if (!isfinite(previous)) return;
            const uint32_t raw_pivot = __float_as_uint(previous);
            const uint32_t pivot = raw_pivot & 0x80000000u ? ~raw_pivot : raw_pivot | 0x80000000u;
            uint32_t codes[4];
            bool keep[4];
            int local = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (window.start <= packet_col + e && packet_col + e < window.end && !isfinite(result[e]))
                    atomicOr(p.counts + out_row, 1 << 28);
                const uint32_t bits = __float_as_uint(result[e]);
                const uint32_t ordered = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
                codes[e] = ScoreBits == 24 ? ordered >> 8 : ordered;
                const int col = packet_col + e;
                keep[e] = window.start <= col && col < window.end && codes[e] >= pivot;
                local += keep[e];
            }
            int prefix = local;
#pragma unroll
            for (int delta = 1; delta < 16; delta *= 2) {
                const int previous = __shfl_up(prefix, delta, 16);
                if (lane % 16 >= delta) prefix += previous;
            }
            const int total = __shfl(prefix, 15, 16);
            int base = 0;
            if (lane % 16 == 0 && total) base = atomicAdd(p.counts + out_row, total);
            base = __shfl(base, 0, 16) + prefix - local;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (keep[e]) {
                    const int slot = base++;
                    if (slot < ((FusedMode == 4 || FusedMode == 6) ? 16384 : 4096)) {
                        const int64_t offset = int64_t(out_row) * ((FusedMode == 4 || FusedMode == 6) ? 16384 : 4096) + slot;
                        const uint32_t id = packet_col + e;
                        if constexpr (ScoreBits == 24) {
                            static_cast<uint16_t *>(p.candidate_keys)[offset] = codes[e];
                            p.candidate_ids[offset] = int32_t(id | ((codes[e] >> 16) << 20));
                        } else {
                            static_cast<uint32_t *>(p.candidate_keys)[offset] = (FusedMode == 4 || FusedMode == 6) ? __float_as_uint(result[e]) : codes[e];
                            p.candidate_ids[offset] = int32_t(id);
                        }
                    }
                }
            }
            return;
        }
        const int dst_col = FusedMode == 1 ? by * BN + packet_col - col_begin : packet_col;
        float *dst = p.out + int64_t(out_row) * p.out_stride_m + dst_col;
        if (EvenN || packet_col + 3 < p.n) {
            if (pair_out) {
                // Keep the F4 packet and emit two contiguous F2 stores.
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
                if (packet_col + e < p.n)
                    store1(dst + e, result[e]);
        }
    };
    // An F4-aligned XOR swizzle turns the MMAC row ownership into consecutive-lane row
    // segments. For eight consecutive lanes both phases touch eight distinct 16-byte
    // bank groups.
    auto swizzle = [](int row) { return row & (Slots - 1); };
    using LdsF4 = F4 __attribute__((address_space(3)));
    const uint32_t lds_base = uint32_t(reinterpret_cast<uintptr_t>(shared));
    auto *output = reinterpret_cast<LdsF4 *>(uintptr_t(lds_base)) + wave * 16 * Slots;
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        const int producer_row = lane % 16;
#pragma unroll
        for (int t = 0; t < NT; ++t)
            output[producer_row * Slots + ((t * 4 + lane / 16) ^ swizzle(producer_row))] =
                accum[r][t];
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
#pragma unroll
        for (int packet = 0; packet < NT; ++packet) {
            const int consumer_row = packet * (64 / Slots) + lane / Slots;
            const F4 value =
                output[consumer_row * Slots + ((lane % Slots) ^ swizzle(consumer_row))];
            const int out_row = row_begin + r * 16 + consumer_row;
            if ((EvenM || out_row < p.m) && (FusedMode == 2 || FusedMode == 4 || FusedMode == 6 || EvenN || out_col < p.n)) {
                const F4 result = value * scale;
                Window bounds{};
                if constexpr (HasBounds && CleanLogits)
                    if (!tile_full)
                        bounds = load_window(p.ks, p.ke, p.n, out_row);
                emit(out_row, out_col, result, bounds);
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
