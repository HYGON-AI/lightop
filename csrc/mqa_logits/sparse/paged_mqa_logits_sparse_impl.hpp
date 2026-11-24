// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../paged/paged_mqa_logits_impl.hpp"
#include "../paged/paged_mqa_logits_fp8_impl.hpp"

#include <algorithm>

namespace lightop {
namespace deep_gemm {

#if defined(__gfx938__)
__device__ __forceinline__ void v1_wait_vm() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ void v1_wait_lds() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ void v1_mmac_boundary() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 0" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ float v1_relu(float x) {
    float v;
    asm("v_max3_f32 %0, %1, %1, 0" : "=v"(v) : "v"(x));
    return v;
}

using v1_fp32x2 = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void v1_load_k(const uint8_t* page, int row, int lane, uint8_t* dst) {
    uintptr_t address = reinterpret_cast<uintptr_t>(page);
    v4i desc = {int(__builtin_amdgcn_readfirstlane(uint32_t(address))),
                int(__builtin_amdgcn_readfirstlane(uint32_t(address >> 32))),
                -2,
                0x20000};
    int col = (lane % 8) ^ (lane / 8);
#pragma unroll
    for (int it = 0; it < 2; it++) {
        auto* p = dst + it * 1024;
        __builtin_amdgcn_raw_buffer_load_lds(
            desc, *(__attribute__((address_space(3))) int**)(&p), 16,
            (lane / 8) * 128 + col * 16, (row + it * 8) * 128, 0, 0);
    }
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ float v1_xor_head_sum(float x, int lane, int mask) {
    return __uint_as_float(__builtin_amdgcn_ds_bpermute((lane ^ mask) * 4, __float_as_uint(x)));
}
#endif

template <int kGroupedRows>
__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void paged_mqa_logits_v1_fp8_kernel(
    const uint8_t* q,
    const uint8_t* kv_block,
    uint64_t kv_stride,
    const float* weights,
    uint64_t logits_stride,
    uint64_t table_stride,
    const int* lens,
    float* logits,
    const int* table,
    int16_t* masks,
    int persistent) {
#if defined(__gfx938__)
    static_assert(kGroupedRows == 1 || (kGroupedRows >= 3 && kGroupedRows <= 6));
    constexpr int Rows = 2, Groups = (kGroupedRows + Rows - 1) / Rows;
    int lane = threadIdx.x, req = blockIdx.x / (persistent * Groups), local = blockIdx.x % (persistent * Groups);
    int group = local % Groups;
    int first = req * kGroupedRows + group * Rows, seed = local / Groups;
    int lengths[Rows], max_len = 0;
#pragma unroll
    for (int r = 0; r < Rows; r++) {
        lengths[r] = group * Rows + r < kGroupedRows
            ? max(0, min(int(logits_stride), lens[first + r])) : 0;
        max_len = max(max_len, lengths[r]);
    }
    max_len = __builtin_amdgcn_readfirstlane(max_len);
    if (seed * 64 >= max_len) return;
    __shared__ __align__(1024) uint8_t sk[2048];
    f8x16_t rq[Rows][2][2] = {};
    float rw[Rows][8] = {};
#pragma unroll
    for (int r = 0; r < Rows; r++) {
        if (group * Rows + r >= kGroupedRows) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
            for (int d = 0; d < 2; d++)
                rq[r][h][d] = *reinterpret_cast<const f8x16_t*>(q + uint64_t(first + r) * 4096 +
                                                                (lane % 16 + h * 16) * 128 + (lane / 16) * 16 + d * 64);
#pragma unroll
            for (int c = 0; c < 4; c++) rw[r][h * 4 + c] = weights[(first + r) * 32 + h * 16 + c * 4 + lane / 16];
        }
    }
    const uint64_t table_base = uint64_t(req * kGroupedRows) * table_stride;
    int physical = __builtin_amdgcn_readfirstlane(table[table_base + seed]);
    const uint8_t* page = kv_block + uint64_t(physical) * kv_stride;
    float scale = reinterpret_cast<const float*>(page + 8192)[lane];
    v1_wait_vm();
    v1_load_k(page, 0, lane, sk);
    const uint64_t mask_stride = (logits_stride + 15) / 16;
    for (int pg = seed; pg * 64 < max_len; pg += persistent) {
        int next_pg = pg + persistent;
        bool has_next = next_pg * 64 < max_len;
        int next_physical = 0;
        if (has_next) next_physical = table[table_base + next_pg];
        const uint8_t* next_page = nullptr;
        float next_scale = 0.f;
        float page_sums[Rows][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v1_wait_vm();
            f8x16_t rk[2];
#pragma unroll
            for (int d = 0; d < 2; d++)
                rk[d] = *reinterpret_cast<const f8x16_t*>(sk + (lane % 16) * 128 +
                                                          (((lane / 16 + d * 4) ^ ((lane % 16) & 7)) * 16));
            v1_wait_lds();
            if (i < 3) v1_load_k(page, (i + 1) * 16, lane, sk);
            if (i == 1 && has_next) {
                next_physical = __builtin_amdgcn_readfirstlane(next_physical);
                next_page = kv_block + uint64_t(next_physical) * kv_stride;
                next_scale = reinterpret_cast<const float*>(next_page + 8192)[lane];
            }
            if (i == 3 && has_next) v1_load_k(next_page, 0, lane, sk);
            fp32x4 acc[Rows][2] = {};
#pragma unroll
            for (int d = 0; d < 2; d++) {
#pragma unroll
                for (int r = 0; r < Rows; r++) {
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        builtin_fp8_mmac(rk[d].front, rq[r][h][d].front, acc[r][h]);
                        builtin_fp8_mmac(rk[d].rear, rq[r][h][d].rear, acc[r][h]);
                    }
                }
            }
            v1_mmac_boundary();
            v1_fp32x2 pair_sum = {0.f, 0.f};
#pragma unroll
            for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    v1_fp32x2 values = {v1_relu(acc[0][h][c]), v1_relu(acc[1][h][c])};
                    v1_fp32x2 weight = {rw[0][h * 4 + c], rw[1][h * 4 + c]};
                    pair_sum = __builtin_elementwise_fma(values, weight, pair_sum);
                }
            }
            pair_sum += v1_fp32x2{v1_xor_head_sum(pair_sum[0], lane, 16), v1_xor_head_sum(pair_sum[1], lane, 16)};
            pair_sum += v1_fp32x2{v1_xor_head_sum(pair_sum[0], lane, 32), v1_xor_head_sum(pair_sum[1], lane, 32)};
            page_sums[0][i] = pair_sum[0];
            page_sums[1][i] = pair_sum[1];
        }
#pragma unroll
        for (int r = 0; r < Rows; r++) {
            float sum = (lane < 16 ? page_sums[r][0] : lane < 32 ? page_sums[r][1]
                         : lane < 48 ? page_sums[r][2] : page_sums[r][3]) * scale;
            int token = pg * 64 + lane;
            bool valid = token < lengths[r] && __float_as_uint(sum) != 0u;
            uint64_t bits = __ballot(valid);
            if (valid) logits[uint64_t(first + r) * logits_stride + token] = sum;
            if (group * Rows + r < kGroupedRows && lane == 0) {
                auto* mask_row = masks + uint64_t(first + r) * mask_stride;
                if ((mask_stride & 3) == 0 && pg * 4 + 3 < mask_stride)
                    *reinterpret_cast<uint64_t*>(mask_row + pg * 4) = bits;
                else {
#pragma unroll
                    for (int i = 0; i < 4; i++)
                        if (pg * 4 + i < mask_stride)
                            mask_row[pg * 4 + i] = int16_t(bits >> (16 * i));
                }
            }
        }
        if (has_next) {
            page = next_page;
            scale = next_scale;
        }
    }
#endif
}

template <int kGroupedRows>
inline void launch_paged_mqa_logits_v1_fp8(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    const int persistent_ctas,
    cudaStream_t stream) {
    const auto logits_stride = static_cast<uint64_t>(logits.stride(0));
    const auto kv_stride = static_cast<uint64_t>(kv_cache.stride(0) * kv_cache.element_size());
    const auto table_stride = static_cast<uint64_t>(block_table.stride(0));
    const int64_t pages = (logits_stride + 63) / 64;
    const int persistent = static_cast<int>(std::min<int64_t>(persistent_ctas == 0 ? 256 : persistent_ctas, pages));
    constexpr int groups = (kGroupedRows + 1) / 2;
    const int64_t blocks = (q.size(0) / kGroupedRows) * groups * persistent;
    TORCH_CHECK(blocks <= INT32_MAX, "V1 FP8 sparse MQA grid is too large");
    paged_mqa_logits_v1_fp8_kernel<kGroupedRows><<<static_cast<unsigned int>(blocks), 64, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr<at::Float8_e4m3fn>()),
        kv_cache.data_ptr<uint8_t>(), kv_stride, weights.data_ptr<float>(),
        logits_stride, table_stride, context_lens.data_ptr<int>(), logits.data_ptr<float>(),
        block_table.data_ptr<int>(), nonzero_mask.data_ptr<int16_t>(), persistent);
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "V1 FP8 sparse MQA launch failed: ", hipGetErrorString(error));
}

inline void dispatch_paged_mqa_logits_v1_fp8(
    const int group_size,
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    const int persistent_ctas,
    cudaStream_t stream) {
    switch (group_size) {
        case 1:
            launch_paged_mqa_logits_v1_fp8<1>(q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 3:
            launch_paged_mqa_logits_v1_fp8<3>(q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 4:
            launch_paged_mqa_logits_v1_fp8<4>(q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 5:
            launch_paged_mqa_logits_v1_fp8<5>(q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 6:
            launch_paged_mqa_logits_v1_fp8<6>(q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        default:
            TORCH_CHECK(false, "V1 FP8 sparse MQA supports group_size 1/3/4/5/6");
    }
}


inline __device__ __forceinline__ bool bf16x8_all_finite(
    const f16x8_t& values) {
    const v4i raw = *reinterpret_cast<const v4i*>(&values);
    bool all_finite = true;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = static_cast<uint32_t>(raw[i]);
        all_finite &= (word & 0x00007f80u) != 0x00007f80u;
        all_finite &= (word & 0x7f800000u) != 0x7f800000u;
    }
    return all_finite;
}

// BF16 sparse paged-MQA keeps one query row and its head weights in
// registers while a fixed CTA pool walks all useful 256-token segments.
// It writes only nonzero logits together with a 16-token occupancy mask.
template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kNumWarps = 4, int kTokensPerWarp = BLOCK_KV,
          bool kGroupScheduled = false>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_bf16_sparse_mask_persistent(
    const Element* q,
    const Element* kv_block,
    const float* weights,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table,
    int16_t* sparse_mask) {
    LIGHTOP_STATIC_ASSERT(
        std::is_same<Element, at::BFloat16>::value,
        "Spark persistent paged MQA supports BF16 only");
    LIGHTOP_STATIC_ASSERT(
        kNextN == 1 && kNumHeads == 64 && kHeadDim == 128 &&
            BLOCK_KV == 64 && kNumWarps == 4 && kTokensPerWarp == 64,
        "Invalid Spark persistent paged MQA specialization");

    constexpr int K_TILE = 32;
    constexpr int Q_TILE = 16;
    constexpr int KV_TILE = 16;
    constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    constexpr int kTokensPerCta = kTokensPerWarp * kNumWarps;
    constexpr int kAlignmentQ = 16 / sizeof(Element);
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    const int warp_idx = threadIdx.x / WARP_SIZE_GPU;
    const int lane_idx = threadIdx.x % WARP_SIZE_GPU;
    const int t_id = threadIdx.x;
    // The B40 target-verify specialization keeps the exact row-independent
    // math/resource shape, but launches the five draft rows for one request
    // next to each other at every persistent-CTA seed. This improves temporal
    // L2 reuse of their repeated physical KV pages without introducing CTA
    // synchronization or changing the generic block-table semantics.
    int ctas_per_row;
    int cta_seed;
    int q_idx;
    if constexpr (kGroupScheduled) {
        ctas_per_row = static_cast<int>(gridDim.x) / 40;
        const int blocks_per_request = ctas_per_row * 5;
        const int request_idx =
            static_cast<int>(blockIdx.x) / blocks_per_request;
        const int request_block =
            static_cast<int>(blockIdx.x) - request_idx * blocks_per_request;
        cta_seed = request_block / 5;
        const int draft_idx = request_block - cta_seed * 5;
        q_idx = request_idx * 5 + draft_idx;
    } else {
        ctas_per_row = static_cast<int>(gridDim.x);
        cta_seed = static_cast<int>(blockIdx.x);
        q_idx = static_cast<int>(blockIdx.y);
    }
    const int logits_limit = static_cast<int>(logits_stride);
    const int raw_context_len = context_lens[q_idx];
    const int context_len =
        raw_context_len < 0
            ? 0
            : (raw_context_len > logits_limit ? logits_limit
                                               : raw_context_len);
    const int num_ctas = ceil_div(context_len, kTokensPerCta);
    const uint64_t base_logits_offset = q_idx * logits_stride;
    const uint64_t sparse_mask_stride =
        (logits_stride + KV_TILE - 1) / KV_TILE;
    const uint64_t base_sparse_mask_offset = q_idx * sparse_mask_stride;
    const uint64_t block_table_offset = q_idx * block_table_stride;

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    auto* smem_kv_block = reinterpret_cast<Element*>(smem_buffer);

    // Cache Q in registers once for all segments assigned to this CTA.
    const auto* gQ = q + q_idx * kNumHeads * kHeadDim;
    constexpr int fetch_q_cnt = constexpr_ceil_div(
        kNumHeads * kHeadDim,
        kNumWarps * WARP_SIZE_GPU * kAlignmentQ);
    constexpr int fetch_q_stride =
        kAlignmentQ * kNumWarps * WARP_SIZE_GPU;
    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        const int pos = t_id * kAlignmentQ + i * fetch_q_stride;
        if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
            *reinterpret_cast<f16x8_t*>(&smem_kv_block[pos]) =
                *reinterpret_cast<const f16x8_t*>(&gQ[pos]);
        }
    }

    f16x8_t reg_q_operand[Q_ITER][K_BLOCK_MAX];
    const int base_fetch_lds_q =
        lane_idx % Q_TILE * kHeadDim + lane_idx / Q_TILE * kAlignmentQ;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q_ITER; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            const int offset = base_fetch_lds_q +
                               i * Q_TILE * kHeadDim + j * K_TILE;
            reg_q_operand[i][j] =
                *reinterpret_cast<f16x8_t*>(&smem_kv_block[offset]);
        }
    }

    // The 64-head production case can reuse the Q LDS allocation for weights
    // after all Q fragments have moved to registers.
    auto* smem_weight = reinterpret_cast<float*>(smem_kv_block);
    const auto* gW = weights + q_idx * kNumHeads;
    __syncthreads();
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }
    __syncthreads();

    float reg_weights[Q_ITER * 4];
    #pragma unroll
    for (int j = 0; j < Q_ITER; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_weights[j * 4 + k] =
                smem_weight[j * Q_TILE + k * 4 + lane_idx / Q_TILE];
        }
    }
    constexpr int smem_kv_block_per_warp = KV_TILE * kHeadDim;
    auto* smem_warp_kv_start =
        smem_kv_block + warp_idx * smem_kv_block_per_warp;
    for (int cta_idx = cta_seed; cta_idx < num_ctas;
         cta_idx += ctas_per_row) {
        const int cta_seq_kv_offset = cta_idx * kTokensPerCta;
        float sum_result[KV_ITER] = {0.0f};

        const auto prefetch_tile = [&](const int kv_iter) {
            const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
            const int seq_kv_tile_offset =
                cta_seq_kv_offset + cta_tile_idx * KV_TILE;
            const int page_idx = seq_kv_tile_offset / BLOCK_KV;
            if (seq_kv_tile_offset < context_len) {
                const int block_idx =
                    block_table[block_table_offset + page_idx];
                const auto* page =
                    kv_block + static_cast<int64_t>(block_idx) *
                                   BLOCK_KV * kHeadDim;
                buffer_load_page_tile_lds<Element, kHeadDim, KV_TILE>(
                    page,
                    seq_kv_tile_offset % BLOCK_KV,
                    lane_idx,
                    smem_warp_kv_start);
            }
        };

        prefetch_tile(0);
        #pragma unroll
        for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
            const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
            const int seq_kv_tile_offset =
                cta_seq_kv_offset + cta_tile_idx * KV_TILE;
            const bool valid = seq_kv_tile_offset < context_len;

            #if defined(__gfx936__) || defined(__gfx938__)
            // Every wave owns a disjoint KV LDS slice.  Waiting for this
            // wave's direct-to-LDS loads is sufficient; a CTA-wide barrier
            // only couples unrelated random page loads across the four waves.
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "s_waitcnt vmcnt(0)\n\t");
            __builtin_amdgcn_sched_barrier(0);
            #else
            __syncthreads();
            #endif

            fp32x4 acc[Q_ITER];
            if (valid) {
                CLEAR_ACC(acc, Q_ITER);
            }

            const int base_fetch = lane_idx % KV_TILE * kHeadDim +
                                   lane_idx / KV_TILE * kAlignmentQ;
            #pragma unroll
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                f16x8_t reg_kv_operand;
                if (valid) {
                    const int fetch_lds = base_fetch + k_block * K_TILE;
                    const int lds_addr = reinterpret_cast<size_t>(
                        smem_warp_kv_start + fetch_lds);
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile(
                        "\n ds_read_b128 %0, %1\n\t"
                        "s_waitcnt lgkmcnt(0)\n\t"
                        : "=v"(reg_kv_operand)
                        : "v"(lds_addr)
                        :);
                    __builtin_amdgcn_sched_barrier(0);
                }

                if (k_block == K_BLOCK_MAX - 1 &&
                    kv_iter + 1 < KV_ITER) {
                    prefetch_tile(kv_iter + 1);
                }

                if (valid) {
                    #pragma unroll
                    for (int i = 0; i < Q_ITER; ++i) {
                        builtin_b16_mmac<false>(
                            reg_kv_operand.front,
                            reg_q_operand[i][k_block].front,
                            acc[i]);
                        builtin_b16_mmac<false>(
                            reg_kv_operand.rear,
                            reg_q_operand[i][k_block].rear,
                            acc[i]);
                    }
                }
            }

            float sum = 0.0f;
            if (valid) {
                #pragma unroll
                for (int j = 0; j < Q_ITER; ++j) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        sum += fmaxf(acc[j][k], 0.0f) *
                               reg_weights[j * 4 + k];
                    }
                }
                #pragma unroll
                for (uint32_t shfl_idx = 16; shfl_idx <= 32;
                     shfl_idx <<= 1) {
                    sum += __shfl_down_sync(
                        0xffffffffffffffff, sum, shfl_idx);
                }
                sum_result[kv_iter] = sum;
            }
        }

        #pragma unroll
        for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
            const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
            const int seq_kv_tile_offset =
                cta_seq_kv_offset + cta_tile_idx * KV_TILE;
            const int seq_kv_offset = seq_kv_tile_offset + lane_idx;
            const bool in_range =
                lane_idx < KV_TILE && seq_kv_offset < context_len;
            const float value = in_range ? sum_result[kv_iter] : 0.0f;

            const auto nonzero_bits = __ballot(
                in_range && __float_as_uint(value) != 0u);
            if (lane_idx == 0 &&
                seq_kv_tile_offset < logits_limit) {
                sparse_mask[
                    base_sparse_mask_offset +
                    seq_kv_tile_offset / KV_TILE] =
                    static_cast<int16_t>(nonzero_bits & 0xffffu);
            }
            if (in_range && __float_as_uint(value) != 0u) {
                logits[base_logits_offset + seq_kv_offset] = value;
            }
        }

        // KV LDS and output ownership stay wave-local across persistent
        // iterations, so no CTA-wide synchronization is needed here.
    }

}

}  // namespace deep_gemm
}  // namespace lightop
