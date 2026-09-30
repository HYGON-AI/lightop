// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include "small_kv.hpp"
#include "../dispatch/selectors.hpp"
#include "../dispatch/static_switch.hpp"
namespace lightop {
namespace mqa_fp8 {

// Swap the MMAC free dimensions: A is N, B is flattened (query, head).
// lane = p + 16*g owns N=p and QH=4*g+j in each 16-row fragment.
__device__ __forceinline__ F4 qh_accumulate(F4 sum, F4 dot, F4 weight) {
    __builtin_hcu_sched_barrier(0);
    asm volatile("; MQA_QH_RELU_PAIRED" ::"v"(dot) : "memory");
    __builtin_hcu_sched_barrier(0);
    F4 activated;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        activated[j] = relu(dot[j]);
    sum.xy = activated.xy * weight.xy + sum.xy;
    sum.zw = activated.zw * weight.zw + sum.zw;
    return sum;
}

template <int H, int BN, bool Clean, bool UnitScale = false, bool FullTile = false>
__global__ __launch_bounds__(256) void mqa_qh(MqaFp8Params p) {
    static_assert(H == 8 || H == 16 || H == 32 || H == 64);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    constexpr int BM = 64 / H, NF = BN / 64, WN = BN / 4, Groups = H <= 16 ? 4 : 64 / H;
    __shared__ __align__(256) uint8_t shared[16384];
    const uint32_t lds = uint32_t(reinterpret_cast<uintptr_t>(shared));
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64), lane = threadIdx.x % 64;
    const int g = lane / 16, np = lane % 16;
    const int64_t m0 = int64_t(blockIdx.y) * BM, n0 = int64_t(blockIdx.x) * BN;
    const int vm = FullTile ? BM : tail_rows<BM>(p.m - m0);
    const int vn = FullTile ? BN : tail_rows<BN>(p.n - n0);
    const TileWindow tile = small_window<Clean>(p, m0, n0, vm, vn, lane);
    if (!tile.any) {
        if constexpr (Clean)
            empty_output(p, m0, n0, vm, vn);
        return;
    }
    I4 kv[NF][2];
    small_kv<BN, FullTile>(p, lds, n0, vn, wave, kv);
    const int qr = FullTile ? 16 : tail_rows<16>(vm * H - wave * 16);
    // A fully filtered MLS leaves LDS untouched. Invalid QH groups are never
    // reduced into a valid query; explicitly skip their producers.
    if (qr)
        MLS128x16B8::matrix_load(
            MLS128x16B8::make_descriptor<!FullTile>(p.q + (m0 * H + wave * 16) * 128,
                                                     128, qr),
            lds + wave * 2048);
    wait_vm();
    barrier();
    I4 q[4][2];
    F4 weights[4];
    const I4 wd = ScalarBufferLoad::make_descriptor(p.weights + m0 * p.weights_stride_m);
#pragma unroll
    for (int rf = 0; rf < 4; ++rf) {
        q[rf][0] = MLS128x16B8::ds_read_format<>(lds + rf * 2048);
        q[rf][1] = MLS128x16B8::ds_read_format<1024>(lds + rf * 2048);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = rf * 16 + g * 4 + j;
            const uint32_t offset =
                (FullTile || r / H < vm) ? uint32_t((r / H) * p.weights_stride_m + r % H) * 4 : UINT32_MAX;
            weights[rf][j] = ScalarBufferLoad::load(wd, offset);
        }
    }
    F4 zero = {};
    asm volatile("s_nop 2" : "+v"(zero));
#pragma unroll
    for (int nf = 0; nf < NF; ++nf) {
        F4 sums[Groups] = {};
#pragma unroll
        for (int rf = 0; rf < 4; rf += 2) {
            F4 d0 = dot128(kv[nf][0], kv[nf][1], q[rf][0], q[rf][1], zero);
            F4 d1 = dot128(kv[nf][0], kv[nf][1], q[rf + 1][0], q[rf + 1][1], zero);
            constexpr int FragmentsPerGroup = H <= 16 ? 1 : H / 16;
            sums[rf / FragmentsPerGroup] =
                qh_accumulate(sums[rf / FragmentsPerGroup], d0, weights[rf]);
            sums[(rf + 1) / FragmentsPerGroup] =
                qh_accumulate(sums[(rf + 1) / FragmentsPerGroup], d1, weights[rf + 1]);
        }
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const F4 s = sums[group];
            float value = (s[0] + s[1]) + (s[2] + s[3]);
            value += __shfl_xor(value, 16, 64);
            if constexpr (H >= 16)
                value += __shfl_xor(value, 32, 64);
            const int mr = H == 8 ? group * 2 + g / 2 : group;
            const int nc = wave * WN + nf * 16 + np;
            const bool leader = H == 8 ? (g % 2 == 0) : (g == 0);
            if (leader && (FullTile || (mr < vm && nc < vn))) {
                if constexpr (!UnitScale)
                    value *= ScalarBufferLoad::load(
                        ScalarBufferLoad::make_descriptor(p.kv_scale + n0), uint32_t(nc) * 4);
                if constexpr (Clean) {
                    if (!tile.full) {
                        const Window w = load_window(p.ks, p.ke, p.n, m0 + mr);
                        if (!(w.start <= n0 + nc && n0 + nc < w.end))
                            value = -__builtin_inff();
                    }
                }
                store1(p.out + (m0 + mr) * p.out_stride_m + n0 + nc, value);
            }
        }
    }
}

Kernel mqa_select_qh(const MqaFp8Params &p, int bn) {
    return MQA_SIZE_SWITCH_4(p.h, H, 8, 16, 32, 64, [&] {
        return MQA_SIZE_SWITCH_3(bn, BN, 64, 128, 256, [&] {
            const bool full_tile = p.m % (64 / H) == 0 && p.n % BN == 0;
            return MQA_BOOL_SWITCH(full_tile, FullTile, [&] {
                return MQA_BOOL_SWITCH(p.clean_logits, Clean, [&] {
                    return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                        return static_cast<Kernel>(mqa_qh<H, BN, Clean, UnitScale, FullTile>);
                    });
                });
            });
        });
    });
}

} // namespace mqa_fp8
} // namespace lightop
#endif
