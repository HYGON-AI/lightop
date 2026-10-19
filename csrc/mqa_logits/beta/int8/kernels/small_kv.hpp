// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../utils/ops.hpp"
#include "../../params.h"
#include <limits>

namespace lightop::mqa_int8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <bool PACK> __device__ __forceinline__ void weighted(const I4 &d, const F4 &w, F4 &a) {
    // Preserve the per-head FMA and the existing head reduction order.
#pragma unroll
    for (int i = 0; i < 4; ++i)
        a[i] += (__builtin_bit_cast(float, max(d[i], kDotBias)) - kDotBiasValue) * w[i];
}
// Heads form MMAC columns. H8 can place two adjacent queries in 16 columns.
template <int H, int TN, int WAVES, bool PAIR, bool PACK, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
__device__ __forceinline__ void run(const int8_t *__restrict__ q, const int8_t *__restrict__ k,
                                    const float *__restrict__ weights,
                                    const int32_t *__restrict__ ks, const int32_t *__restrict__ ke,
                                    const float *__restrict__ scales, float *__restrict__ out,
                                    int m, int n, int64_t weights_stride_m, int64_t out_stride_m) {
    static_assert(!PAIR || H == 8);
    constexpr int Q = PAIR ? 2 : 1, NT = TN / 16;
    const int lane = threadIdx.x % 64, row = (blockIdx.x * WAVES + threadIdx.x / 64) * Q;
    const int begin = blockIdx.y * TN;
    if (!EvenM && row >= m)
        return;
    int lo[Q], hi[Q];
    const int vm = EvenM ? Q : tail_rows<Q>(int64_t(m) - row);
    const int vn = EvenN ? TN : tail_rows<TN>(int64_t(n) - begin);
    MqaInt8Params window_params{};
    window_params.ks = ks;
    window_params.ke = ke;
    window_params.n = n;
    TileWindow tile{true, true};
    if constexpr (HasBounds)
        tile = small_window<CleanLogits>(window_params, row, begin, vm, vn, lane);
    const bool valid = tile.any, tile_full = tile.full;
#pragma unroll
    for (int j = 0; j < Q; j++) {
        lo[j] = hi[j] = 0;
        if constexpr (HasBounds && CleanLogits)
            if ((EvenM || row + j < m) && !tile_full) {
                const Window w = load_window(ks, ke, n, row + j);
                lo[j] = int(w.start);
                hi[j] = int(w.end);
            }
    }
    if (!valid) {
        if constexpr (HasBounds && CleanLogits) {
#pragma unroll
            for (int j = 0; j < Q; ++j)
                if (EvenM || row + j < m)
                    for (int c = lane; c < TN; c += 64)
                        if (EvenN || begin + c < n)
                            store1(out + int64_t(row + j) * out_stride_m + begin + c,
                                   -__builtin_inff());
        }
        return;
    }
    const I4 kv_resource = make_kv_resource(k + int64_t(begin) * 128);
    I2 kr[NT][4];
    F4 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        acc[t] = {0, 0, 0, 0};
        const int col = begin + t * 16 + lane % 16;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            kr[t][d] = {0, 0};
            if (EvenN || col < n)
                kr[t][d] = load_kv_slc(
                    kv_resource, (col - begin) * 128 + lane / 16 * 8 + d * 32);
        }
    }
#pragma unroll
    for (int h = 0; h < H; h += 16) {
        I2 qr[4];
        const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                  head = h + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
        for (int d = 0; d < 4; d++) {
            qr[d] = {0, 0};
            if ((EvenM || qrrow < m) && head < H)
                qr[d] = load_aligned4<I2>(q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 8 +
                                          d * 32);
        }
        F4 w;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int colh = h + lane / 16 + e * 4, wr = row + (PAIR ? colh / 8 : 0),
                      wh = PAIR ? colh % 8 : colh;
            w[e] = ((EvenM || wr < m) && wh < H) ? weights[int64_t(wr) * weights_stride_m + wh] : 0;
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            I4 dot = dot_init();
#pragma unroll
            for (int d = 0; d < 4; d++)
                mmac_i8(kr[t][d], qr[d], dot);
            weighted<PACK>(dot, w, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int col = begin + t * 16 + lane % 16;
#pragma unroll
        for (int j = 0; j < Q; j++) {
            float v = PAIR ? (acc[t][j * 2] + acc[t][j * 2 + 1])
                           : ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3]));
            v += __shfl_xor(v, 16, 64);
            v += __shfl_xor(v, 32, 64);
            if (lane < 16 && (EvenN || col < n) && (EvenM || row + j < m)) {
                float value;
                if constexpr (HasBounds && CleanLogits)
                    value = (tile_full || (lo[j] <= col && col < hi[j])) ? v * scales[col]
                                                                         : -__builtin_inff();
                else
                    value = v * scales[col];
                store1(out + int64_t(row + j) * out_stride_m + col, value);
            }
        }
    }
}

template <int H, int TN, int WAVES, bool PAIR, bool EvenM, bool EvenN, bool HasBounds,
          bool CleanLogits>
__global__ __launch_bounds__(WAVES * 64) void small_kv(MqaInt8Params p) {
    run<H, TN, WAVES, PAIR, false, EvenM, EvenN, HasBounds, CleanLogits>(
        reinterpret_cast<const int8_t *>(p.q), reinterpret_cast<const int8_t *>(p.kv),
        p.weights, p.ks, p.ke, p.kv_scale, p.out, int(p.m), int(p.n), p.weights_stride_m,
        p.out_stride_m);
}
#endif
} // namespace lightop::mqa_int8
