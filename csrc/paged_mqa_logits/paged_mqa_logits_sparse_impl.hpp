#pragma once

#include "paged_mqa_logits_impl.hpp"

namespace lightop {
namespace deep_gemm {

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
