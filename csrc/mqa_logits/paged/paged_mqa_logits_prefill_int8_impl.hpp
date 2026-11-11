// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_mqa_logits_int8_impl.hpp"

namespace lightop::deep_gemm {

// 32 query tokens x 128 KV positions, native INT8 MMAC. Each wave
// retains 32 KV rows in registers and reuses them across all query heads.
// The two Q buffers permit one CTA barrier per head without overwriting a
// buffer still consumed by another wave. No speculative load overlap is used.
template <int NumHeads, int BlockKV>
__global__ __launch_bounds__(256) void paged_mqa_logits_prefill_int8_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ packed_cache,
    const float* __restrict__ weights, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ block_table, float* __restrict__ logits,
    int tokens, int max_context_len, int num_physical_pages) {
    constexpr int Rows = 32, Cols = 128, Dim = 128, Threads = 256;
    constexpr int Pages = Cols / BlockKV;
    static_assert(BlockKV == 4 || BlockKV == 8 || BlockKV == 16 || BlockKV == 32 || BlockKV == 64);
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    const int row_start = blockIdx.y * Rows, col_start = blockIdx.x * Cols;
    __shared__ int lengths[Rows], page_ids[Pages], tile_max_len;
    __shared__ float scales[Cols];
    __shared__ __align__(16) union {
        struct {
            int8_t q[2][Rows][136];
            // Only two groups of eight heads are resident. Within a group,
            // a head's row weights remain adjacent as in the V2 layout.
            float weights[2][8][Rows];
        } input;
        // Transpose two 64-column halves so output does not force a 16-KiB
        // union after shrinking the weight working set.
        float output[Rows][68];
    } shared;

    int length = 0;
    if (wave == 0 && lane < Rows && row_start + lane < tokens) {
        length = context_lens[row_start + lane];
    }
    // VMEM order: wave 0 issues one context load iff lane < 32 and row <
    // tokens; waves 1..3 issue none. Predicate-dependent count => N=0.
    // Consumers: lengths publication and wave-0 maximum reduction below.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (wave == 0) {
        if (lane < Rows) lengths[lane] = length;
        #pragma unroll
        for (int delta = 32; delta > 0; delta /= 2) {
            length = max(length, __shfl_down(length, delta));
        }
        if (lane == 0) tile_max_len = length;
    }
    __syncthreads();

    // This branch is CTA-uniform and derived from all 32 endpoints, so it
    // remains valid for graph replay with nonmonotonic or changed lengths.
    if (col_start >= tile_max_len) {
        for (int index = tid; index < Rows * Cols; index += Threads) {
            const int row = row_start + index / Cols, col = col_start + index % Cols;
            if (row < tokens && col < max_context_len) {
                logits[static_cast<int64_t>(row) * max_context_len + col] = -INFINITY;
            }
        }
        return;
    }

    int page_id = -1;
    if (wave == 0 && lane < Pages && col_start + lane * BlockKV < max_context_len &&
        col_start + lane * BlockKV < tile_max_len) {
        // One wave reads consecutive logical pages of row zero. Other waves
        // obtain these IDs exclusively through CTA shared memory.
        page_id = block_table[col_start / BlockKV + lane];
    }
    // VMEM order since prior barrier: wave 0 issues one page-table load
    // under the bounds/endpoint predicate above; waves 1..3 issue zero.
    // N=0 is required before page-ID LDS publication; no dummy requests.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (wave == 0 && lane < Pages) page_ids[lane] = page_id;
    __syncthreads();

    int8x8_t key[2][4];
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        const int local_col = wave * 32 + n * 16 + lane % 16;
        const int physical = page_ids[local_col / BlockKV];
        const bool valid = col_start + local_col < max_context_len &&
                           col_start + local_col < tile_max_len &&
                           physical >= 0 && physical < num_physical_pages;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            key[n][k] = {};
            if (valid) {
                const int64_t offset = static_cast<int64_t>(physical) * BlockKV * (Dim + 4) +
                                       (local_col % BlockKV) * Dim + (lane / 16) * 8 + k * 32;
                key[n][k] = *reinterpret_cast<const int8x8_t*>(packed_cache + offset);
            }
        }
    }
    // Each wave issues up to eight K loads, n-major then k-major, with a
    // data-dependent valid predicate for each 16-column fragment. N=0;
    // key[][] is retained for MMAC and no later VMEM issue is counted here.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");

    const auto load_weight_group = [&](int first_head, int buffer) {
        const int row = tid / 8, head = tid % 8;
        float value = 0.0f;
        if (row_start + row < tokens) {
            value = weights[static_cast<int64_t>(row_start + row) * NumHeads + first_head + head];
        }
        // Every wave issues one scalar weight load under row < tokens.
        // The group is requested only when first_head + 8 <= NumHeads;
        // the initial invocation follows the K drain, later invocations
        // follow the current Q drain/barrier. Branch-dependent count => N=0.
        // The real consumer is the LDS write immediately below.
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        shared.input.weights[buffer][head][row] = value;
    };
    load_weight_group(0, 0);
    if (wave == 0) {
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int col = lane + n * 64, physical = page_ids[col / BlockKV];
            float scale = 0.0f;
            if (col_start + col < max_context_len && col_start + col < tile_max_len &&
                physical >= 0 && physical < num_physical_pages) {
                const auto* page_scales = reinterpret_cast<const float*>(
                    packed_cache + static_cast<int64_t>(physical) * BlockKV * (Dim + 4) + BlockKV * Dim);
                scale = page_scales[col % BlockKV];
            }
            scales[col] = scale;
        }
    }
    // Weight loads have already drained. Wave 0 then loads two scale
    // elements per lane, under page/endpoint predicates; other waves issue
    // none. N=0 before the CTA barrier publishes initial weights/scales.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    __syncthreads();

    fp32x4 result[2][2] = {};
    const int load_row = tid / 8, load_col = (tid % 8) * 16;
    #pragma unroll 1
    for (int head = 0; head < NumHeads; ++head) {
        i8x16_t query = {};
        if (row_start + load_row < tokens) {
            query = *reinterpret_cast<const i8x16_t*>(
                q + (static_cast<int64_t>(row_start + load_row) * NumHeads + head) * Dim + load_col);
        }
        // Per head, each wave issues one Q vector load iff row < tokens.
        // Previous VMEM is drained. N=0 before the real consumer, the LDS
        // write below. Any earlier weight-group load has already drained.
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        *reinterpret_cast<i8x16_t*>(&shared.input.q[head % 2][load_row][load_col]) = query;
        __syncthreads();

        if (head % 8 == 0 && head + 8 < NumHeads) {
            // The current Q barrier retires all readers of the group that
            // occupied this slot two groups ago. The next Q barrier will
            // publish this write, seven heads before its first consumer.
            // This version drains at the LDS write; it claims no overlap.
            load_weight_group(head + 8, 1 - (head / 8) % 2);
        }

        int32x4_t dot[2][2] = {};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const auto operand = *reinterpret_cast<const int8x8_t*>(
                    &shared.input.q[head % 2][lane % 16 + m * 16][lane / 16 * 8 + k * 32]);
                builtin_i8_mmac(operand, key[0][k], dot[m][0]);
                builtin_i8_mmac(operand, key[1][k], dot[m][1]);
            }
        }
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float weight = shared.input.weights[(head / 8) % 2][head % 8][lane % 16 + m * 16];
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    result[m][n][j] += fmaxf(static_cast<float>(dot[m][n][j]), 0.0f) * weight;
                }
            }
        }
    }

    // All waves finish reading Q/weights before the union becomes output.
    __syncthreads();
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        if (wave / 2 == half) {
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        shared.output[lane % 16 + m * 16][(wave % 2) * 32 + n * 16 + lane / 16 + j * 4] = result[m][n][j];
                    }
                }
            }
        }
        __syncthreads();
        for (int index = tid * 4; index < Rows * 64; index += Threads * 4) {
            const int row = index / 64, local_col = index % 64;
            const int col = half * 64 + local_col;
            if (row_start + row < tokens && col_start + col < max_context_len) {
                fp32x4 value = *reinterpret_cast<const fp32x4*>(&shared.output[row][local_col]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    value[j] = col_start + col + j < lengths[row] ? value[j] * scales[col + j] : -INFINITY;
                }
                auto* output = logits + static_cast<int64_t>(row_start + row) * max_context_len + col_start + col;
                if (max_context_len % 4 == 0 && col_start + col + 3 < max_context_len) {
                    *reinterpret_cast<fp32x4*>(output) = value;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (col_start + col + j < max_context_len) output[j] = value[j];
                    }
                }
            }
        }
        // Every wave finishes reading the first half before another wave
        // overwrites the shared transpose buffer with the second half.
        if (half == 0) __syncthreads();
    }
}

}  // namespace lightop::deep_gemm
