// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_mqa_logits_impl.hpp"

namespace lightop {
namespace deep_gemm {

typedef uint8_t fp8_t;
typedef fp8_t fp8x2 __attribute__((ext_vector_type(2)));
typedef fp8_t fp8x4 __attribute__((ext_vector_type(4)));
typedef fp8_t fp8x8 __attribute__((ext_vector_type(8)));
typedef fp8_t fp8x16 __attribute__((ext_vector_type(16)));

union f8x16_t {
    fp8x16 data;
    struct {
        fp8x8 front;
        fp8x8 rear;
    };
};

typedef uint32_t v4u __attribute__((ext_vector_type(4)));

#define PAGED_MQA_FP8_WAITCNT(INSTRUCTION)                                                                            \
    do {                                                                                                               \
        __builtin_amdgcn_sched_barrier(0);                                                                             \
        asm volatile(INSTRUCTION ::: "memory");                                                                       \
        __builtin_amdgcn_sched_barrier(0);                                                                             \
    } while (false)

#define PAGED_MQA_FP8_WAIT_VMCNT_0() PAGED_MQA_FP8_WAITCNT("s_waitcnt vmcnt(0)\n\t")
#define PAGED_MQA_FP8_WAIT_LGKMCNT_0() PAGED_MQA_FP8_WAITCNT("s_waitcnt lgkmcnt(0)\n\t")
#define PAGED_MQA_FP8_CTA_BARRIER()                                                                                    \
    do {                                                                                                               \
        __builtin_amdgcn_sched_barrier(0);                                                                             \
        asm volatile("s_barrier\n\t" ::: "memory");                                                                  \
        __builtin_amdgcn_sched_barrier(0);                                                                             \
    } while (false)

inline __device__ __forceinline__ v4u make_mls_128x16_b8_srsrc(const uint8_t* ptr, const uint32_t row_stride_bytes,
                                                               const uint32_t filtered_rows) {
    v4u srsrc = {};
    *reinterpret_cast<uint64_t*>(&srsrc) = reinterpret_cast<uint64_t>(ptr) & 0x0000ffffffffffffULL;
    srsrc[2] = row_stride_bytes;
    srsrc[3] = filtered_rows << 8;
    return srsrc;
}

inline __device__ __forceinline__ void matrix_load_128x16_b8_t(uint8_t* shared_addr, const v4u& srsrc) {
    #if defined(__gfx938__)
    const v4u uniform_srsrc = {static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(srsrc[0])),
                               static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(srsrc[1])),
                               static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(srsrc[2])),
                               static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(srsrc[3]))};
    const int lds_addr = __builtin_amdgcn_readfirstlane(static_cast<int>(reinterpret_cast<uintptr_t>(shared_addr)));
    const int soffset = lds_addr + static_cast<int>(0x80000000u);

    // readfirstlane is a VALU write to the SGPR descriptor immediately consumed by the VMEM-class MLS instruction.
    // AMD Vega 7nm and CDNA4 ISA section 4.5 require five wait states for this RAW hazard; section 12.5 defines
    // s_nop 4 as a five-cycle delay.  The reference flash-attention MLS wrapper uses the same sequence.
    asm volatile("s_nop 4\n\t"
                 "matrix_load_128x16_b8 %0, %1, moffset:0 t lds\n\t"
                 :
                 : "s"(uniform_srsrc), "s"(soffset)
                 : "memory");
    #else
    (void)shared_addr;
    (void)srsrc;
    #endif
}

inline __device__ __forceinline__ v4i ds_read_matrix_trans_b8(uint8_t* ptr) {
    v4i value = {};
    #if defined(__gfx938__)
    const int offset = __builtin_amdgcn_readfirstlane(static_cast<int>(reinterpret_cast<uintptr_t>(ptr)));
    asm volatile("s_add_u32 m0, %1, 0x80000000\n\t"
                 "s_nop 0\n\t"
                 "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 "
                 "row:0x3 col:0x1 alt:0x0\n\t"
                 : "=v"(value)
                 : "s"(offset)
                 : "memory");
    #else
    (void)ptr;
    #endif
    return value;
}

template<bool is_e4m3 = true>
inline __device__ __forceinline__ void builtin_fp8_mmac(const fp8x8& reg_a, const fp8x8& reg_b, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_e4m3) {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
            *reinterpret_cast<const intx2*>(&reg_a), *reinterpret_cast<const intx2*>(&reg_b), reg_c, false, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(
            *reinterpret_cast<const intx2*>(&reg_a), *reinterpret_cast<const intx2*>(&reg_b), reg_c, false, false);
    }
    #else
    (void)reg_a;
    (void)reg_b;
    (void)reg_c;
    #endif
}

template<bool is_e4m3 = true>
inline __device__ __forceinline__ void builtin_fp8_mmac_mls(const fp8x8& reg_q, const fp8x8& reg_k, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_e4m3) {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
            *reinterpret_cast<const intx2*>(&reg_q), *reinterpret_cast<const intx2*>(&reg_k), reg_c, true, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(
            *reinterpret_cast<const intx2*>(&reg_q), *reinterpret_cast<const intx2*>(&reg_k), reg_c, true, false);
    }
    #else
    (void)reg_q;
    (void)reg_k;
    (void)reg_c;
    #endif
}

template<int kDppCtrl>
inline __device__ __forceinline__ float fp32_mov_dpp(const float value) {
    union {
        float fp32;
        uint32_t bits;
    } data = {value};
    data.bits = static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(data.bits, kDppCtrl, 0xf, 0xf, true));
    return data.fp32;
}

// gfx938 FP8 regular-page kernel. MLS and ds_read_matrix_trans load the Q/KV operands, while DPP moves handle three
// of the four 16-lane reduction stages. Only xor-4 remains on the LDS/DS-permute pipe.
template <typename Element, int kNextN = 1, int kNumHeads = 64, int kHeadDim = 128, int BLOCK_KV = 64, int kBatchSplit,
          int kNumWarps = 4, int kTokensPerWarp = BLOCK_KV>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_fp8(const Element* q, const Element* kv_block, const float* kv_block_scales,
                                     const float* weights, const int batch_size, const int num_kv_blocks,
                                     const uint64_t kv_cache_stride_bytes, const uint64_t logits_stride,
                                     const uint64_t block_table_stride, const int* context_lens, float* logits,
                                     const int* block_table, const int* schedule_meta) {
    constexpr int Q_TILE = 16;
    constexpr int KV_TILE = 16;
    constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    constexpr int PAGES_PER_WARP = kTokensPerWarp / BLOCK_KV;
    constexpr int PAGES_PER_CTA = kNumWarps * PAGES_PER_WARP;
    constexpr int kTokensPerCta = kNumWarps * kTokensPerWarp;
    constexpr int kMlsTileBytes = Q_TILE * kHeadDim;
    constexpr int kKvLdsBytes = kNumWarps * kMlsTileBytes;
    constexpr int kWeightLdsOffset = kKvLdsBytes;
    constexpr int kResultLdsOffset = kKvLdsBytes;
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    constexpr bool is_e4m3 = std::is_same<Element, at::Float8_e4m3fn>::value;

    LIGHTOP_STATIC_ASSERT(kHeadDim == 128, "FP8 MLS requires D128");
    LIGHTOP_STATIC_ASSERT(BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64, "FP8 MLS requires regular pages");
    LIGHTOP_STATIC_ASSERT(kNumHeads == 8 || kNumHeads == 16 || kNumHeads == 32 || kNumHeads == 64,
                          "Invalid MLS head count");
    LIGHTOP_STATIC_ASSERT(kTokensPerWarp == 64 && kNumWarps == 4, "Invalid MLS CTA tiling");

    (void)batch_size;
    (void)num_kv_blocks;
    (void)schedule_meta;

    // LDS offset/lifetime map for the 9216-byte launch allocation:
    //   Q/weight phase: Q [0,8192), weights [8192,8448).
    //   metadata phase: block IDs [0,64), scales [0,1024).
    //   KV/result phase: four wave-private KV tiles [0,8192), result scratch [8192,9216).
    //
    // S_BARRIER synchronizes waves but does not drain wait counters.  Before every cross-wave phase transition below,
    // each producer therefore drains exactly the counters that can still own data in the reused LDS range. A global
    // value feeding an LDS store has a true dependency, so the compiler inserts vmcnt(0) at that ds_write consumer;
    // the explicit producer-side wait then drains only LGKM_CNT. Q MLS-only waves still drain VM_CNT explicitly.
    // Q/weight, block-ID, and scale LDS readers drain only LGKM_CNT before their corresponding reuse barriers.
    // The KV phase has only wave-private producers and consumers, so it requires waitcnt but no CTA barrier.

    const int warp_idx = threadIdx.x / WARP_SIZE_GPU;
    const int lane_idx = threadIdx.x % WARP_SIZE_GPU;
    const int t_id = threadIdx.x;
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    const int next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;
    const int q_idx = blockIdx.y;
    const int cta_idx = blockIdx.x / kNextN;
    const int context_len = context_lens[q_idx];
    const int num_kv = ceil_div(context_len, BLOCK_KV);
    const int cta_seq_kv_offset = cta_idx * kTokensPerCta;
    const uint64_t base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;

    if (cta_seq_kv_offset >= num_kv * BLOCK_KV) {
        const int seq_kv_offset = cta_seq_kv_offset + t_id;
        if (seq_kv_offset < logits_stride) {
            logits[base_logits_offset + seq_kv_offset] = -INFINITY;
        }
        return;
    }

    f8x16_t reg_q_operand[Q_ITER][2];
    float reg_weights[Q_ITER];
    const auto* gQ =
        reinterpret_cast<const uint8_t*>(q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim);
    constexpr int kPaddedHeads = Q_ITER * Q_TILE;
    auto* smem_weights = reinterpret_cast<float*>(smem_buffer + kWeightLdsOffset);
    const auto* gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;

    // VMEM issue order before this wait:
    //   wave 0: one Q MLS, then one weight load in each valid head lane;
    //   waves [1,Q_ITER): one Q MLS and no weight load;
    //   remaining waves: no request.
    // H8 wave 0's Q MLS uses nm_filter=8 and only lanes [0,8) load weights; padded lanes write an LDS zero without
    // issuing a dummy VMEM request. The weight global-to-LDS dependency drains wave 0's VM requests at ds_write;
    // each remaining Q producer wave uses vmcnt(0). Counts differ by wave and predicate, so vmcnt(N) is not legal.
    if (warp_idx < Q_ITER) {
        const auto q_srsrc =
            make_mls_128x16_b8_srsrc(gQ + warp_idx * kMlsTileBytes, kHeadDim, kNumHeads == 8 ? 8u : 0u);
        matrix_load_128x16_b8_t(smem_buffer + warp_idx * kMlsTileBytes, q_srsrc);
    }
    if (warp_idx == 0 && lane_idx < kPaddedHeads) {
        smem_weights[lane_idx] = lane_idx < kNumHeads ? gW[lane_idx] : 0.0f;
    }
    if (warp_idx == 0) {
        PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    } else if (warp_idx < Q_ITER) {
        PAGED_MQA_FP8_WAIT_VMCNT_0();
    }
    PAGED_MQA_FP8_CTA_BARRIER();

    #pragma unroll
    for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
        auto* q_tile = smem_buffer + q_iter * kMlsTileBytes;
        *reinterpret_cast<v4i*>(&reg_q_operand[q_iter][0]) = ds_read_matrix_trans_b8(q_tile);
        *reinterpret_cast<v4i*>(&reg_q_operand[q_iter][1]) = ds_read_matrix_trans_b8(q_tile + 1024);
        reg_weights[q_iter] = smem_weights[q_iter * Q_TILE + lane_idx % Q_TILE];
    }
    PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    PAGED_MQA_FP8_CTA_BARRIER();

    // A single wave reads the CTA's contiguous block-table span. Lane i in wave 0 issues request i iff the logical
    // page exists; other waves issue none. The global value is consumed by the following LDS store, whose true data
    // dependency forces vmcnt(0); only wave 0 then drains the LDS store before CTA propagation.
    auto* smem_block_ids = reinterpret_cast<int*>(smem_buffer);
    const int cta_first_page = cta_seq_kv_offset / BLOCK_KV;
    if (warp_idx == 0 && lane_idx < PAGES_PER_CTA) {
        const int page_idx = cta_first_page + lane_idx;
        smem_block_ids[lane_idx] = page_idx < num_kv ? block_table[q_idx * block_table_stride + page_idx] : -1;
    }
    if (warp_idx == 0) {
        PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    }
    PAGED_MQA_FP8_CTA_BARRIER();

    int kv_block_idx[PAGES_PER_WARP];
    #pragma unroll
    for (int page = 0; page < PAGES_PER_WARP; ++page) {
        kv_block_idx[page] = __builtin_amdgcn_readfirstlane(smem_block_ids[warp_idx * PAGES_PER_WARP + page]);
    }
    PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    PAGED_MQA_FP8_CTA_BARRIER();

    // Each thread owns one of the CTA's 256 token scales. Valid pages issue one VMEM load per lane; invalid pages issue
    // none and write zero locally. The global value is consumed by the LDS store, which forces vmcnt(0) for a valid
    // wave; LGKM_CNT is then the only outstanding counter required before CTA propagation.
    auto* smem_kv_scales = reinterpret_cast<float*>(smem_buffer);
    const int lane_page_delta = lane_idx / BLOCK_KV;
    const int lane_row_in_page = lane_idx % BLOCK_KV;
    const int lane_block_idx = kv_block_idx[lane_page_delta];
    float lane_scale = 0.0f;
    if (lane_block_idx != -1) {
        const int kv_scale_stride = kv_cache_stride_bytes / sizeof(float);
        const auto* page_scales = kv_block_scales + static_cast<int64_t>(lane_block_idx) * kv_scale_stride;
        lane_scale = page_scales[lane_row_in_page];
    }
    smem_kv_scales[t_id] = lane_scale;
    PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    PAGED_MQA_FP8_CTA_BARRIER();

    float reg_kv_scales[KV_ITER][4];
    const int token_group = lane_idx / Q_TILE;
    #pragma unroll
    for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
        #pragma unroll
        for (int component = 0; component < 4; ++component) {
            reg_kv_scales[kv_iter][component] =
                smem_kv_scales[warp_idx * kTokensPerWarp + kv_iter * KV_TILE + token_group * 4 + component];
        }
    }
    PAGED_MQA_FP8_WAIT_LGKMCNT_0();
    PAGED_MQA_FP8_CTA_BARRIER();

    auto* smem_results = reinterpret_cast<float*>(smem_buffer + kResultLdsOffset);

    // Prime KV stage 0. Each valid wave issues one MLS and an invalid wave issues none. Since validity comes from the
    // block table, vmcnt(0) is the only legal wait at the first DS consumer. Every wave consumes only its own tile.
    const int first_block_idx = kv_block_idx[0];
    if (first_block_idx != -1) {
        const auto* first_page =
            reinterpret_cast<const uint8_t*>(kv_block) + static_cast<int64_t>(first_block_idx) * kv_cache_stride_bytes;
        const auto first_kv_srsrc = make_mls_128x16_b8_srsrc(first_page, kHeadDim, 0);
        matrix_load_128x16_b8_t(smem_buffer + warp_idx * kMlsTileBytes, first_kv_srsrc);
    }
    PAGED_MQA_FP8_WAIT_VMCNT_0();

    #pragma unroll
    for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
        const int page_delta = kv_iter * KV_TILE / BLOCK_KV;
        const int current_block_idx = kv_block_idx[page_delta];
        const bool valid = current_block_idx != -1;
        f8x16_t reg_kv_operand[2] = {};
        if (valid) {
            auto* kv_tile = smem_buffer + warp_idx * kMlsTileBytes;
            *reinterpret_cast<v4i*>(&reg_kv_operand[0]) = ds_read_matrix_trans_b8(kv_tile);
            *reinterpret_cast<v4i*>(&reg_kv_operand[1]) = ds_read_matrix_trans_b8(kv_tile + 1024);
        }
        PAGED_MQA_FP8_WAIT_LGKMCNT_0();

        // Stage (kv_iter + 1) reuses this wave's 2048-byte tile after both current DS reads complete. Each wave issues
        // one MLS iff the next logical page is valid; the data-dependent request overlaps the current arithmetic and
        // is drained with vmcnt(0) immediately before the next iteration's DS consumer.
        const int next_kv_iter = kv_iter + 1;
        if (next_kv_iter < KV_ITER) {
            const int next_page_delta = next_kv_iter * KV_TILE / BLOCK_KV;
            const int next_row_in_page = next_kv_iter * KV_TILE % BLOCK_KV;
            const int next_block_idx = kv_block_idx[next_page_delta];
            if (next_block_idx != -1) {
                const auto* next_page = reinterpret_cast<const uint8_t*>(kv_block) +
                                        static_cast<int64_t>(next_block_idx) * kv_cache_stride_bytes;
                const auto next_kv_srsrc =
                    make_mls_128x16_b8_srsrc(next_page + next_row_in_page * kHeadDim, kHeadDim, 0);
                matrix_load_128x16_b8_t(smem_buffer + warp_idx * kMlsTileBytes, next_kv_srsrc);
            }
        }

        fp32x4 acc[Q_ITER];
        CLEAR_ACC(acc, Q_ITER);
        #pragma unroll
        for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
            builtin_fp8_mmac_mls<is_e4m3>(reg_q_operand[q_iter][0].front, reg_kv_operand[0].front, acc[q_iter]);
            builtin_fp8_mmac_mls<is_e4m3>(reg_q_operand[q_iter][0].rear, reg_kv_operand[0].rear, acc[q_iter]);
            builtin_fp8_mmac_mls<is_e4m3>(reg_q_operand[q_iter][1].front, reg_kv_operand[1].front, acc[q_iter]);
            builtin_fp8_mmac_mls<is_e4m3>(reg_q_operand[q_iter][1].rear, reg_kv_operand[1].rear, acc[q_iter]);
        }

        fp32x4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
            #pragma unroll
            for (int component = 0; component < 4; ++component) {
                sums[component] += fmaxf(acc[q_iter][component], 0.0f) * reg_weights[q_iter];
            }
        }

        #pragma unroll
        for (int component = 0; component < 4; ++component) {
            sums[component] += fp32_mov_dpp<0xb1>(sums[component]);
            sums[component] += fp32_mov_dpp<0x4e>(sums[component]);
            sums[component] += __shfl_xor(sums[component], 4, Q_TILE);
            sums[component] += fp32_mov_dpp<0x128>(sums[component]);
        }

        if (lane_idx % Q_TILE == 0) {
            #pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int token_in_warp = token_group * 4 + component;
                smem_results[warp_idx * kTokensPerWarp + kv_iter * KV_TILE + token_in_warp] =
                    sums[component] * reg_kv_scales[kv_iter][component];
            }
        }

        // The next MLS has overlapped the current arithmetic. Its variable per-wave request count is drained at the
        // next true DS consumer. Result LDS is disjoint from every KV tile and is drained at its final consumer.
        if (next_kv_iter < KV_ITER) {
            PAGED_MQA_FP8_WAIT_VMCNT_0();
        }
    }

    const float result = smem_results[t_id];
    PAGED_MQA_FP8_WAIT_LGKMCNT_0();

    const int seq_kv_offset = cta_seq_kv_offset + t_id;
    if (seq_kv_offset < context_len - (kNextN - next_n) + 1) {
        logits[base_logits_offset + seq_kv_offset] = result;
    } else if (seq_kv_offset < logits_stride) {
        logits[base_logits_offset + seq_kv_offset] = -INFINITY;
    }
}

#undef PAGED_MQA_FP8_CTA_BARRIER
#undef PAGED_MQA_FP8_WAIT_LGKMCNT_0
#undef PAGED_MQA_FP8_WAIT_VMCNT_0
#undef PAGED_MQA_FP8_WAITCNT

template <typename Element, int kNextN = 1, int kNumHeads = 64, int kHeadDim = 128, int BLOCK_KV = 4,
          int kTokensPerWarp = 64, int kNumWarps = 4>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_fp8_small_block(
    const Element* q, const Element* kv_block, const float* kv_block_scales, const float* weights, const int batch_size,
    const int num_kv_blocks, const uint64_t kv_cache_stride_bytes, const uint64_t logits_stride,
    const uint64_t block_table_stride, const int* context_lens, float* logits, const int* block_table,
    const int* schedule_meta) {
    const int& warp_idx = threadIdx.x / 64;
    const int& lane_idx = threadIdx.x % 64;
    const int& t_id = threadIdx.x;

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    static constexpr int K_TILE = 64;
    static constexpr int Q_TILE = 16;
    static constexpr int KV_TILE = 16;
    static constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    static constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    static constexpr int PAGES_PER_WARP = kTokensPerWarp / BLOCK_KV;
    static constexpr int SUBPAGES_PER_TILE = KV_TILE / BLOCK_KV;
    static constexpr int Stages = kHeadDim / K_TILE;
    LIGHTOP_STATIC_ASSERT((BLOCK_KV == 4 || BLOCK_KV == 8) && kTokensPerWarp == 64, "Invalid small BLOCK_KV");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr bool is_e4m3 = std::is_same<Element, at::Float8_e4m3fn>::value;

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    int q_idx = blockIdx.y;
    int start_kv_idx = blockIdx.x / kNextN * kNumWarps * PAGES_PER_WARP;
    int warp_kv_idx = start_kv_idx + warp_idx * PAGES_PER_WARP;
    int context_len = context_lens[q_idx];
    int num_kv = ceil_div(context_len, BLOCK_KV);
    const bool full_warp = warp_kv_idx + PAGES_PER_WARP <= num_kv;

    auto base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;
    auto base_seq_kv_offset = warp_kv_idx * BLOCK_KV;

    if (start_kv_idx >= num_kv) {
        #pragma unroll
        for (int offset = 0; offset < kTokensPerWarp; offset += WARP_SIZE_GPU) {
            auto seq_kv_offset = base_seq_kv_offset + offset + lane_idx;
            if (seq_kv_offset < logits_stride) {
                logits[base_logits_offset + seq_kv_offset] = -INFINITY;
            }
        }
        return;
    }

    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(uint8_t);
    constexpr int q_required_warps =
        constexpr_ceil_div(kNumHeads * kHeadDim, WARP_SIZE_GPU * kAlignmentQ);
    constexpr int q_load_warps = q_required_warps < kNumWarps ? q_required_warps : kNumWarps;
    constexpr int fetch_q_cnt =
        constexpr_ceil_div(kNumHeads * kHeadDim, q_load_warps * WARP_SIZE_GPU * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * q_load_warps * WARP_SIZE_GPU;

    if (warp_idx < q_load_warps) {
        const int q_t_id = warp_idx * WARP_SIZE_GPU + lane_idx;
        #pragma unroll
        for (int i = 0; i < fetch_q_cnt; ++i) {
            auto pos = q_t_id * kAlignmentQ + i * fetch_q_stride;
            if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
                *reinterpret_cast<f8x16_t*>(&smem_kv_block[pos]) = *reinterpret_cast<const f8x16_t*>(&gQ[pos]);
            }
        }
    }

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    int kv_block_idx[PAGES_PER_WARP];
    uint64_t block_table_offset = q_idx * block_table_stride + warp_kv_idx;
    if (full_warp) {
        #pragma unroll
        for (int i = 0; i < PAGES_PER_WARP; ++i) {
            kv_block_idx[i] = block_table[block_table_offset + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < PAGES_PER_WARP; ++i) {
            kv_block_idx[i] = warp_kv_idx + i < num_kv ? block_table[block_table_offset + i] : -1;
        }
    }

    const int kv_scale_stride = kv_cache_stride_bytes / sizeof(float);
    auto* smem_kv_scales =
        reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim + kNumHeads * sizeof(float));
    const int lane_page_delta = lane_idx / BLOCK_KV;
    const int lane_row_in_page = lane_idx - lane_page_delta * BLOCK_KV;
    if (full_warp) {
        auto gKv_scales = kv_block_scales + kv_block_idx[lane_page_delta] * kv_scale_stride;
        smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
    } else if (lane_page_delta < PAGES_PER_WARP && kv_block_idx[lane_page_delta] != -1) {
        auto gKv_scales = kv_block_scales + kv_block_idx[lane_page_delta] * kv_scale_stride;
        smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
    } else {
        smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = 0.0f;
    }

    constexpr int heads_cnt = Q_ITER;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();

    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            if constexpr (kNumHeads < Q_TILE) {
                reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, f8x16_t>(
                    smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
            } else {
                reg_q_operand[i][j] = *reinterpret_cast<f8x16_t*>(&smem_kv_block[offset]);
            }
        }
    }

    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < Q_ITER; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (kNumHeads < Q_TILE) {
                reg_weights[j * 4 + k] = load_padded_weight<kNumHeads, Q_TILE>(
                    smem_weight, j * Q_TILE + k * 4 + lane_idx / Q_TILE);
            } else {
                reg_weights[j * 4 + k] = smem_weight[j * 16 + k * 4 + lane_idx / 16];
            }
        }
    }

    float kv_scale[KV_ITER];
    const int logits_t_offset = warp_idx * kTokensPerWarp + lane_idx % KV_TILE;
    #pragma unroll
    for (int i = 0; i < KV_ITER; ++i) {
        kv_scale[i] = smem_kv_scales[logits_t_offset + i * KV_TILE];
    }

    __syncthreads();

    if (warp_kv_idx >= num_kv) {
        #pragma unroll
        for (int offset = 0; offset < kTokensPerWarp; offset += WARP_SIZE_GPU) {
            auto seq_kv_offset = base_seq_kv_offset + offset + lane_idx;
            if (seq_kv_offset < logits_stride) {
                logits[base_logits_offset + seq_kv_offset] = -INFINITY;
            }
        }
        return;
    }

    constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
    constexpr int prefetch_kv_stage = KV_ITER >= 2 ? 2 : 1;
    constexpr int smem_kv_block_per_warp = prefetch_kv_stage * KV_TILE * kHeadDim;
    auto smem_warp_kv_start = smem_kv_block + warp_idx * smem_kv_block_per_warp;
    auto base_fetch_lds_KV = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;

    fp32x4 acc[Q_ITER][KV_ITER];
    CLEAR_ACC_2D(acc, Q_ITER, KV_ITER);
    float sum_result[KV_ITER] = {0.f};
    f8x16_t reg_kv_operand[K_BLOCK_MAX];

    const auto& load_kv_tile_full = [&](const uint32_t& load_kv_iter, const uint32_t& smem_stage) {
        constexpr long stride = 0x0;
        auto smem_stage_start = smem_warp_kv_start + smem_stage * KV_TILE * kHeadDim;
        #pragma unroll
        for (int subpage = 0; subpage < SUBPAGES_PER_TILE; ++subpage) {
            const int page_delta = load_kv_iter * SUBPAGES_PER_TILE + subpage;
            const int current_block_idx = kv_block_idx[page_delta];
            auto gKv = kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4);

            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
            glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
            auto* src = reinterpret_cast<v4i*>(glob_kv_desc);

            constexpr int PAGE_LOAD_BYTES = BLOCK_KV == 4 ? 8 : 16;
            const int page_linear = lane_idx * PAGE_LOAD_BYTES;
            const int row_in_page = page_linear / kHeadDim;
            const int col_in_page = page_linear - row_in_page * kHeadDim;
            const int row_in_tile = subpage * BLOCK_KV + row_in_page;
            auto smem_ptr = smem_stage_start + row_in_tile * kHeadDim + col_in_page;
            if constexpr (BLOCK_KV == 4) {
                #if defined(__gfx936__) || defined(__gfx938__)
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x2(src, lds_ptr, page_linear);
                #else
                fp8x8 fetched = *reinterpret_cast<const fp8x8*>(gKv + page_linear);
                *reinterpret_cast<fp8x8*>(smem_ptr) = fetched;
                #endif
            } else {
                #if defined(__gfx936__) || defined(__gfx938__)
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x4(src, lds_ptr, page_linear);
                #else
                f8x16_t fetched = *reinterpret_cast<const f8x16_t*>(gKv + page_linear);
                *reinterpret_cast<f8x16_t*>(smem_ptr) = fetched;
                #endif
            }
        }
    };

    const auto& load_kv_tile_tail = [&](const uint32_t& load_kv_iter, const uint32_t& smem_stage) {
        constexpr long stride = 0x0;
        auto smem_stage_start = smem_warp_kv_start + smem_stage * KV_TILE * kHeadDim;
        #pragma unroll
        for (int subpage = 0; subpage < SUBPAGES_PER_TILE; ++subpage) {
            const int page_delta = load_kv_iter * SUBPAGES_PER_TILE + subpage;
            const int current_block_idx = kv_block_idx[page_delta];
            auto gKv = current_block_idx != -1
                           ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4)
                           : kv_block;

            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
            glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
            auto* src = reinterpret_cast<v4i*>(glob_kv_desc);

            constexpr int PAGE_LOAD_BYTES = BLOCK_KV == 4 ? 8 : 16;
            const int page_linear = lane_idx * PAGE_LOAD_BYTES;
            const int row_in_page = page_linear / kHeadDim;
            const int col_in_page = page_linear - row_in_page * kHeadDim;
            const int row_in_tile = subpage * BLOCK_KV + row_in_page;
            auto smem_ptr = smem_stage_start + row_in_tile * kHeadDim + col_in_page;
            if constexpr (BLOCK_KV == 4) {
                #if defined(__gfx936__) || defined(__gfx938__)
                int inner_warp_offset = current_block_idx != -1 ? page_linear : -1;
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x2(src, lds_ptr, inner_warp_offset);
                #else
                fp8x8 fetched = {0};
                if (current_block_idx != -1) {
                    fetched = *reinterpret_cast<const fp8x8*>(gKv + page_linear);
                }
                *reinterpret_cast<fp8x8*>(smem_ptr) = fetched;
                #endif
            } else {
                #if defined(__gfx936__) || defined(__gfx938__)
                int inner_warp_offset = current_block_idx != -1 ? page_linear : -1;
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x4(src, lds_ptr, inner_warp_offset);
                #else
                f8x16_t fetched;
                fetched.data = {0};
                if (current_block_idx != -1) {
                    fetched = *reinterpret_cast<const f8x16_t*>(gKv + page_linear);
                }
                *reinterpret_cast<f8x16_t*>(smem_ptr) = fetched;
                #endif
            }
        }
    };

    #pragma unroll
    for (uint32_t prefetch_iter = 0; prefetch_iter < prefetch_kv_stage; ++prefetch_iter) {
        if (full_warp) {
            load_kv_tile_full(prefetch_iter, prefetch_iter);
        } else {
            load_kv_tile_tail(prefetch_iter, prefetch_iter);
        }
    }

    #pragma unroll
    for (uint32_t kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
        auto smem_stage_start = smem_warp_kv_start + (kv_iter % prefetch_kv_stage) * KV_TILE * kHeadDim;
        __builtin_amdgcn_sched_barrier(0);
        if (kv_iter + 1 < KV_ITER) {
            asm volatile("s_waitcnt vmcnt(%0)\n\t" :: "I"(SUBPAGES_PER_TILE));
        } else {
            asm volatile("s_waitcnt vmcnt(0)\n\t" :::);
        }
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * K_TILE;
            int lds_addr = reinterpret_cast<size_t>(smem_stage_start + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("\n ds_read_b128 %0 ,%1\n\t"
                         "s_waitcnt lgkmcnt(0) \n\t"
                         : "=v"(reg_kv_operand[k_block])
                         : "v"(lds_addr)
                         :);
            __builtin_amdgcn_sched_barrier(0);
        }

        if (kv_iter + 1 < KV_ITER) {
            const uint32_t load_kv_iter = kv_iter + prefetch_kv_stage;
            const uint32_t load_stage = kv_iter % prefetch_kv_stage;
            if (load_kv_iter < KV_ITER) {
                if (full_warp) {
                    load_kv_tile_full(load_kv_iter, load_stage);
                } else {
                    load_kv_tile_tail(load_kv_iter, load_stage);
                }
            }
        }

        #pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            #pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_fp8_mmac<is_e4m3>(
                    reg_kv_operand[k_block].front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_fp8_mmac<is_e4m3>(
                    reg_kv_operand[k_block].rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const float& f) {
                    return fmaxf(f, 0) * reg_weights[j * 4 + k];
                };

                auto& sum = sum_result[kv_iter];
                for (int j = 0; j < Q_ITER; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sum += transform(j, k, acc[j][kv_iter][k]);
                    }
                }

                sum *= kv_scale[kv_iter];

                #pragma unroll
                for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
                    sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
                }
            }
        }
    }

    auto* smem_write_back = reinterpret_cast<float*>(smem_warp_kv_start);
    if (lane_idx < KV_TILE) {
        #pragma unroll
        for (int i = 0; i < KV_ITER; ++i) {
            smem_write_back[i * KV_TILE + lane_idx] = sum_result[i];
        }
    }
    __syncthreads();

    float permuted_result = smem_write_back[lane_idx];

    auto seq_kv_offset = base_seq_kv_offset + lane_idx;
    if (seq_kv_offset < context_len - (kNextN - next_n) + 1) {
        logits[base_logits_offset + seq_kv_offset] = permuted_result;
    } else {
        logits[base_logits_offset + seq_kv_offset] = -INFINITY;
    }
}

} // namespace deep_gemm
} // namespace lightop
