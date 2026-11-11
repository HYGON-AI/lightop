// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <cstring>

#include "sparse_page_mqa.h"
#include "../paged_mqa_logits/paged_mqa_logits_fp8_impl.hpp"

namespace lightop {
namespace deep_gemm {
namespace grouped_sparse_native_fp8 {

constexpr int kHeadDim = 128;
constexpr int kHeads = 32;
constexpr int kPageSize = 64;
constexpr int kTokenTile = 16;
constexpr int kTokenWarps = 4;
constexpr int kTokensPerCta = kTokenWarps * kPageSize;

// Extend the row-pair FP8 pipeline from 6cd1dae to group_size 1/3/4/5/6.
// Each CTA caches rows of the same request and loops over its KV chunks
// using the existing persistent CTA count. Each wave owns one page and
// reuses its FP8 KV fragments for every cached Q row.
template <int kGroupedRows, int kRowsPerCta = (kGroupedRows == 1 ? 1 : 2)>
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void paged_mqa_logits_grouped_native_fp8_kernel(
    const at::Float8_e4m3fn* q,
    const uint8_t* kv_block,
    const uint64_t kv_cache_stride_bytes,
    const float* weights,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table,
    int16_t* sparse_mask,
    const int persistent_ctas_per_request) {
    static_assert(kGroupedRows == 1 ||
                  (kGroupedRows >= 3 && kGroupedRows <= 6));
#if defined(__gfx938__)
    static_assert(kRowsPerCta >= 1 && kRowsPerCta <= kGroupedRows);
    constexpr int kPairsPerRequest =
        (kGroupedRows + kRowsPerCta - 1) / kRowsPerCta;
    constexpr int kTileK = 64;
    constexpr int kTileQ = 16;
    constexpr int kQIterations = kHeads / kTileQ;
    constexpr int kKBlocks = kHeadDim / kTileK;
    constexpr int kAlignment = 16;
    constexpr int kQElements = kHeads * kHeadDim;
    constexpr int kGroupQBytes = kRowsPerCta * kQElements;
    constexpr int kGroupWeightElements = kRowsPerCta * kHeads;
    constexpr int kKvStageBytes = kTokenWarps * kTokenTile * kHeadDim;

    const int tid = threadIdx.x;
    const int token_warp = tid / WARP_SIZE_GPU;
    const int lane = tid % WARP_SIZE_GPU;
    const int blocks_per_request =
        persistent_ctas_per_request * kPairsPerRequest;
    const int request = static_cast<int>(blockIdx.x) / blocks_per_request;
    const int request_block =
        static_cast<int>(blockIdx.x) - request * blocks_per_request;
    const int cta_seed = request_block / kPairsPerRequest;
    const int pair = request_block - cta_seed * kPairsPerRequest;
    const int group_first_row = request * kGroupedRows;
    const int pair_first_row = group_first_row + pair * kRowsPerCta;
    const int logits_limit = static_cast<int>(logits_stride);

    int row_lengths[kRowsPerCta];
    int pair_max_length = 0;
    #pragma unroll
    for (int row = 0; row < kRowsPerCta; ++row) {
        const bool row_active = pair * kRowsPerCta + row < kGroupedRows;
        const int raw_length = row_active
            ? context_lens[pair_first_row + row] : 0;
        const int length = raw_length < 0
            ? 0 : (raw_length > logits_limit ? logits_limit : raw_length);
        row_lengths[row] = length;
        pair_max_length = length > pair_max_length ? length : pair_max_length;
    }
    // Uniform across the CTA, before any staging/barriers. Empty requests and
    // graph-padding rows do not read Q, weights or unused block-table entries.
    if (cta_seed * kTokensPerCta >= pair_max_length) {
        return;
    }

    extern __shared__ __align__(kHeadDim * 8) uint8_t smem_bytes[];
    auto* smem_q = smem_bytes;
    auto* smem_weights = reinterpret_cast<float*>(smem_q + kGroupQBytes);
    const auto* group_q = q + pair_first_row * kQElements;
    for (int pos = tid * kAlignment; pos < kGroupQBytes;
         pos += blockDim.x * kAlignment) {
        f8x16_t value{};
        if (pair * kRowsPerCta + pos / kQElements < kGroupedRows) {
            value = *reinterpret_cast<const f8x16_t*>(group_q + pos);
        }
        *reinterpret_cast<f8x16_t*>(smem_q + pos) = value;
    }
    if (tid < kGroupWeightElements) {
        smem_weights[tid] =
            pair * kRowsPerCta + tid / kHeads < kGroupedRows
            ? weights[pair_first_row * kHeads + tid] : 0.0f;
    }
    __syncthreads();

    const int base_q_lane =
        lane % kTileQ * kHeadDim + lane / kTileQ * kAlignment;
    const int base_k_lane =
        lane % kTokenTile * kHeadDim + lane / kTokenTile * kAlignment;
    f8x16_t reg_q[kRowsPerCta][kQIterations][kKBlocks];
    float reg_weights[kRowsPerCta][kQIterations * 4];
    #pragma unroll
    for (int row = 0; row < kRowsPerCta; ++row) {
        #pragma unroll
        for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                const int offset = row * kQElements + base_q_lane +
                    q_iter * kTileQ * kHeadDim + k_block * kTileK;
                reg_q[row][q_iter][k_block] =
                    *reinterpret_cast<const f8x16_t*>(smem_q + offset);
            }
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int head = q_iter * kTileQ + item * 4 + lane / kTileQ;
                // The producer already folds the per-Q scale into weights.
                reg_weights[row][q_iter * 4 + item] =
                    smem_weights[row * kHeads + head];
            }
        }
    }
    __syncthreads();

    // After the last CTA barrier, each wave exclusively owns its LDS slices.
    // Q/weight staging can now be reused for double-buffered FP8 KV tiles.
    const int num_compute_ctas = ceil_div(pair_max_length, kTokensPerCta);
    for (int chunk = cta_seed; chunk < num_compute_ctas;
         chunk += persistent_ctas_per_request) {
        const int cta_token_offset = chunk * kTokensPerCta;
        const int wave_token_offset = cta_token_offset + token_warp * kPageSize;
        if (wave_token_offset >= pair_max_length) {
            return;
        }
        auto* smem_kv = smem_bytes;
        const uint64_t shared_block_table_offset =
            group_first_row * block_table_stride;
        const uint64_t mask_stride = (logits_stride + kTokenTile - 1) / kTokenTile;
        const int physical_page = block_table[
            shared_block_table_offset + wave_token_offset / kPageSize];
        const auto* page = kv_block +
            static_cast<int64_t>(physical_page) * kv_cache_stride_bytes;
        const auto* page_scales =
            reinterpret_cast<const float*>(page + kPageSize * kHeadDim);
        float kv_scales[kPageSize / kTokenTile];
        #pragma unroll
        for (int kv_iter = 0; kv_iter < kPageSize / kTokenTile; ++kv_iter) {
            kv_scales[kv_iter] =
                page_scales[kv_iter * kTokenTile + lane % kTokenTile];
        }
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            auto* smem_warp_stage = smem_kv + stage * kKvStageBytes +
                token_warp * kTokenTile * kHeadDim;
            buffer_load_page_tile_lds<uint8_t, kHeadDim, kTokenTile>(
                page, stage * kTokenTile, lane, smem_warp_stage);
        }

        #pragma unroll
        for (int kv_iter = 0; kv_iter < kPageSize / kTokenTile; ++kv_iter) {
            const int stage = kv_iter & 1;
            auto* smem_warp_stage = smem_kv + stage * kKvStageBytes +
                token_warp * kTokenTile * kHeadDim;
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);

            f8x16_t reg_kv[kKBlocks];
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                reg_kv[k_block] = *reinterpret_cast<const f8x16_t*>(
                    smem_warp_stage + base_k_lane + k_block * kTileK);
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);

            // Refill the consumed LDS stage while the resident fragments are used
            // by both rows. Q and KV remain FP8 through the MMAC instructions.
            if (kv_iter + 2 < kPageSize / kTokenTile) {
                buffer_load_page_tile_lds<uint8_t, kHeadDim, kTokenTile>(
                    page, (kv_iter + 2) * kTokenTile, lane, smem_warp_stage);
            }
            fp32x4 accumulators[kRowsPerCta][kQIterations];
            #pragma unroll
            for (int row = 0; row < kRowsPerCta; ++row) {
                CLEAR_ACC(accumulators[row], kQIterations);
            }
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                #pragma unroll
                for (int row = 0; row < kRowsPerCta; ++row) {
                    #pragma unroll
                    for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                        builtin_fp8_mmac<true>(
                            reg_kv[k_block].front,
                            reg_q[row][q_iter][k_block].front,
                            accumulators[row][q_iter]);
                        builtin_fp8_mmac<true>(
                            reg_kv[k_block].rear,
                            reg_q[row][q_iter][k_block].rear,
                            accumulators[row][q_iter]);
                    }
                }
            }

            float sums[kRowsPerCta] = {};
            #pragma unroll
            for (int row = 0; row < kRowsPerCta; ++row) {
                #pragma unroll
                for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                    #pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        sums[row] +=
                            fmaxf(accumulators[row][q_iter][item], 0.0f) *
                            reg_weights[row][q_iter * 4 + item];
                    }
                }
                #pragma unroll
                for (uint32_t delta = 16; delta <= 32; delta <<= 1) {
                    sums[row] += __shfl_down_sync(
                        0xffffffffffffffff, sums[row], delta);
                }
                sums[row] *= kv_scales[kv_iter];
            }

            const int tile_token_offset = wave_token_offset + kv_iter * kTokenTile;
            #pragma unroll
            for (int row = 0; row < kRowsPerCta; ++row) {
                const bool row_active = pair * kRowsPerCta + row < kGroupedRows;
                const int token = tile_token_offset + lane;
                const bool in_range = lane < kTokenTile && token < row_lengths[row];
                // Match the sparse consumer's raw-bit contract, including -0 and
                // non-finite values. Masked-off logits are never initialized/read.
                const uint64_t nonzero_bits = __ballot(
                    in_range && __float_as_uint(sums[row]) != 0u);
                const int q_idx = pair_first_row + row;
                if (row_active && lane == 0 && tile_token_offset < logits_limit) {
                    sparse_mask[q_idx * mask_stride + tile_token_offset / kTokenTile] =
                        static_cast<int16_t>(nonzero_bits & 0xffffu);
                }
                if (in_range && __float_as_uint(sums[row]) != 0u) {
                    logits[q_idx * logits_stride + token] = sums[row];
                }
            }
        }
    }
#endif
}

template <int kGroupedRows, int kRowsPerCta = (kGroupedRows == 1 ? 1 : 2)>
void launch_grouped_kernel(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    const int persistent_ctas,
    cudaStream_t stream) {
    static_assert(kRowsPerCta >= 1 && kRowsPerCta <= kGroupedRows);
    constexpr int kPairsPerRequest =
        (kGroupedRows + kRowsPerCta - 1) / kRowsPerCta;
    constexpr int kThreads = kTokenWarps * WARP_SIZE_GPU;
    constexpr int kQStagingBytes = kRowsPerCta * kHeads * (kHeadDim + sizeof(float));
    constexpr int kKvStagingBytes = 2 * kTokenWarps * kTokenTile * kHeadDim;
    constexpr int kSmemBytes =
        kQStagingBytes > kKvStagingBytes ? kQStagingBytes : kKvStagingBytes;
    const auto logits_stride = static_cast<uint64_t>(logits.stride(0));
    const auto kv_cache_stride_bytes = static_cast<uint64_t>(
        kv_cache.stride(0) * kv_cache.element_size());
    const auto block_table_stride = static_cast<uint64_t>(block_table.stride(0));
    const int64_t requests = q.size(0) / kGroupedRows;
    const int64_t blocks = persistent_ctas * requests * kPairsPerRequest;
    TORCH_CHECK(blocks <= INT32_MAX, "native FP8 sparse MQA grid is too large");
    const dim3 grid(static_cast<unsigned int>(blocks), 1, 1);
    const dim3 block(kThreads, 1, 1);

    paged_mqa_logits_grouped_native_fp8_kernel<kGroupedRows, kRowsPerCta><<<
        grid, block, kSmemBytes, stream>>>(
        q.data_ptr<at::Float8_e4m3fn>(),
        kv_cache.data_ptr<uint8_t>(),
        kv_cache_stride_bytes,
        weights.data_ptr<float>(),
        logits_stride,
        block_table_stride,
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(),
        block_table.data_ptr<int>(),
        nonzero_mask.data_ptr<int16_t>(),
        persistent_ctas);
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "native FP8 sparse MQA launch failed: ",
                hipGetErrorString(error));
}

inline void dispatch_grouped_rows(
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
            launch_grouped_kernel<1>(q, kv_cache, weights, context_lens,
                block_table, logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 3:
            launch_grouped_kernel<3>(q, kv_cache, weights, context_lens,
                block_table, logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 4:
            launch_grouped_kernel<4>(q, kv_cache, weights, context_lens,
                block_table, logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 5:
            launch_grouped_kernel<5>(q, kv_cache, weights, context_lens,
                block_table, logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 6: {
            const char* rows = std::getenv("LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA");
            if (rows == nullptr || std::strcmp(rows, "2") == 0) {
                launch_grouped_kernel<6, 2>(q, kv_cache, weights, context_lens,
                    block_table, logits, nonzero_mask, persistent_ctas, stream);
            } else if (std::strcmp(rows, "3") == 0) {
                launch_grouped_kernel<6, 3>(q, kv_cache, weights, context_lens,
                    block_table, logits, nonzero_mask, persistent_ctas, stream);
            } else if (std::strcmp(rows, "6") == 0) {
                launch_grouped_kernel<6, 6>(q, kv_cache, weights, context_lens,
                    block_table, logits, nonzero_mask, persistent_ctas, stream);
            } else {
                TORCH_CHECK(false,
                    "LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA must be 2, 3 or 6");
            }
            return;
        }
        default:
            TORCH_CHECK(false, "native FP8 sparse MQA supports group_size 1/3/4/5/6");
    }
}

}  // namespace grouped_sparse_native_fp8
}  // namespace deep_gemm
}  // namespace lightop
