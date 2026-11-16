// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "paged_mqa_logits_persisten.h"
#include "paged_mqa_logits.h"
#include "paged_mqa_logits_impl.hpp"
#include "paged_mqa_logits_fp8_impl.hpp"
#include "paged_mqa_logits_int8_impl.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace lightop {
namespace deep_gemm {

template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4,
          int kTokensPerWarp = BLOCK_KV, bool kClearTail = false>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_fp8_length_masked_persistent(
    const Element* q,
    const Element* kv_block,
    const float* kv_block_scales,
    const float* weights,
    const int batch_size,
    const int num_kv_blocks,
    const uint64_t kv_cache_stride_bytes,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table,
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
    static constexpr int Stages = kHeadDim / K_TILE;
    LIGHTOP_STATIC_ASSERT(
        BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64,
        "Invalid BLOCK_KV");
    LIGHTOP_STATIC_ASSERT(
        kTokensPerWarp % BLOCK_KV == 0 &&
            kTokensPerWarp <= WARP_SIZE_GPU,
        "Invalid tokens per warp");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr bool is_e4m3 =
        std::is_same<Element, at::Float8_e4m3fn>::value;

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    LIGHTOP_STATIC_ASSERT(
        kNextN == 1, "persistent FP8 MQA supports kNextN=1 only");
    constexpr int next_n = 0;

    const int q_idx = blockIdx.y;
    const int raw_context_len = context_lens[q_idx];
    const int logits_limit = static_cast<int>(logits_stride);
    // Keep malformed metadata from making this kernel access outside its
    // storage. Callers must still satisfy the documented value-range contract
    // so a downstream length-aware consumer sees the same effective lengths.
    const int context_len =
        raw_context_len < 0
            ? 0
            : (raw_context_len > logits_limit ? logits_limit
                                               : raw_context_len);
    const int num_kv = ceil_div(context_len, BLOCK_KV);
    const int num_ctas = ceil_div(num_kv, kNumWarps * PAGES_PER_WARP);
    constexpr int kTokensPerCta = kNumWarps * kTokensPerWarp;
    const int num_output_ctas = ceil_div(logits_limit, kTokensPerCta);

    // Each resident CTA walks a strided list of useful 256-token segments, so
    // the launch grid is independent of the padded logits allocation width.
    const auto base_logits_offset = q_idx * logits_stride;
    for (int cta_idx = blockIdx.x; cta_idx < num_ctas;
         cta_idx += gridDim.x) {
        const int start_kv_idx =
            cta_idx * kNumWarps * PAGES_PER_WARP;
        const int kv_idx = start_kv_idx + warp_idx * PAGES_PER_WARP;
        const auto base_seq_kv_offset = kv_idx * BLOCK_KV;

        // Fetch Q and Q weights.
        auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim +
                  next_n * kNumHeads * kHeadDim;
        constexpr int kAlignmentQ = 16 / sizeof(uint8_t);
        constexpr int fetch_q_cnt = constexpr_ceil_div(
            kNumHeads * kHeadDim,
            kNumWarps * WARP_SIZE_GPU * kAlignmentQ);
        constexpr int fetch_q_stride =
            kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

        #pragma unroll
        for (int i = 0; i < fetch_q_cnt; ++i) {
            auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
            if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
                *reinterpret_cast<f8x16_t*>(&smem_kv_block[pos]) =
                    *reinterpret_cast<const f8x16_t*>(&gQ[pos]);
            }
        }

        auto gW = weights + q_idx * kNextN * kNumHeads +
                  next_n * kNumHeads;
        auto smem_weight = reinterpret_cast<float*>(
            smem_kv_block + kNumHeads * kHeadDim);
        if (warp_idx == 0 && lane_idx < kNumHeads) {
            smem_weight[lane_idx] = gW[lane_idx];
        }

        // Fetch KV block indices. Invalid lanes use the first valid page in
        // this CTA; their results are discarded by the length-masked store.
        int kv_block_idx[PAGES_PER_WARP];
        uint64_t block_table_offset =
            q_idx * block_table_stride + kv_idx;
        const int fallback_block_idx =
            block_table[q_idx * block_table_stride + start_kv_idx];
        #pragma unroll
        for (int i = 0; i < PAGES_PER_WARP; ++i) {
            kv_block_idx[i] = kv_idx + i < num_kv
                                  ? block_table[block_table_offset + i]
                                  : -1;
        }

        // Fetch KV block scales.
        const int64_t kv_scale_stride =
            static_cast<int64_t>(kv_cache_stride_bytes) /
            sizeof(float);
        auto* smem_kv_scales = reinterpret_cast<float*>(
            smem_kv_block + kNumHeads * kHeadDim +
            kNumHeads * sizeof(float));
        if (lane_idx < kTokensPerWarp) {
            const int lane_page_delta = lane_idx / BLOCK_KV;
            const int lane_row_in_page = lane_idx % BLOCK_KV;
            const int lane_block_idx =
                kv_block_idx[lane_page_delta] != -1
                    ? kv_block_idx[lane_page_delta]
                    : fallback_block_idx;
            auto gKv_scales =
                kv_block_scales +
                static_cast<int64_t>(lane_block_idx) *
                    kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] =
                gKv_scales[lane_row_in_page];
        }

        constexpr int heads_cnt = Q_ITER;
        constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
        f8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

        __syncthreads();
        // Pre-fetch Q from smem.
        auto base_fetch_lds_Q =
            lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
        #pragma unroll
        for (int i = 0; i < heads_cnt; ++i) {
            #pragma unroll
            for (int j = 0; j < K_BLOCK_MAX; ++j) {
                auto offset = base_fetch_lds_Q +
                              i * Q_TILE * kHeadDim + j * K_TILE;
                if constexpr (kNumHeads < Q_TILE) {
                    reg_q_operand[i][j] = load_padded_q_operand<
                        kNumHeads, Q_TILE, f8x16_t>(
                        smem_kv_block,
                        offset,
                        i * Q_TILE + lane_idx % Q_TILE);
                } else {
                    reg_q_operand[i][j] =
                        *reinterpret_cast<f8x16_t*>(
                            &smem_kv_block[offset]);
                }
            }
        }

        // Pre-fetch weights.
        float reg_weights[heads_cnt * 4];
        #pragma unroll
        for (int j = 0; j < Q_ITER; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (kNumHeads < Q_TILE) {
                    reg_weights[j * 4 + k] = load_padded_weight<
                        kNumHeads, Q_TILE>(
                        smem_weight,
                        j * Q_TILE + k * 4 + lane_idx / Q_TILE);
                } else {
                    reg_weights[j * 4 + k] =
                        smem_weight[j * Q_TILE + k * 4 +
                                    lane_idx / Q_TILE];
                }
            }
        }

        // Pre-fetch KV scales.
        float kv_scale[KV_ITER];
        const int logits_t_offset =
            warp_idx * kTokensPerWarp + lane_idx % 16;
        #pragma unroll
        for (int i = 0; i < KV_ITER; ++i) {
            kv_scale[i] =
                smem_kv_scales[logits_t_offset + i * 16];
        }

        __syncthreads();

        // Fetch KV blocks.
        constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
        constexpr long stride = 0x0;

        constexpr int prefetch_kv_stage = KV_ITER >= 2 ? 2 : 1;
        constexpr int smem_kv_block_per_warp =
            prefetch_kv_stage * KV_TILE * kHeadDim;

        auto smem_warp_kv_start =
            smem_kv_block + warp_idx * smem_kv_block_per_warp;

        #pragma unroll
        for (int i = 0; i < prefetch_kv_stage; ++i) {
            const int page_delta = i * KV_TILE / BLOCK_KV;
            const int row_in_page = i * KV_TILE % BLOCK_KV;
            const int current_block_idx =
                kv_block_idx[page_delta] != -1
                    ? kv_block_idx[page_delta]
                    : fallback_block_idx;
            auto gKv = kv_block +
                       static_cast<int64_t>(current_block_idx) *
                           BLOCK_KV * (kHeadDim + 4);
            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
            glob_kv_desc[1] =
                static_cast<long>(0x20000) << 32 | 0xFFFFFFFE;
            auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
            auto glob_stage_offset = row_in_page * kHeadDim;
            auto smem_per_stage =
                smem_warp_kv_start + i * KV_TILE * kHeadDim;
            #pragma unroll
            for (int j = 0; j < Stages; ++j) {
                auto smem_ptr =
                    smem_per_stage + j * KV_TILE * K_TILE;
                auto inner_warp_offset =
                    glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                    lane_idx / 4 * kHeadDim + j * K_TILE;
                buffer_load_lds_x4(src, smem_ptr, inner_warp_offset);
            }
        }

        // Pre-fetch KV from smem.
        f8x16_t reg_kv_operand;

        auto base_fetch_lds_KV =
            lane_idx % 16 * K_TILE + lane_idx / 16 * 16;

        fp32x4 acc[Q_ITER][KV_ITER];
        CLEAR_ACC_2D(acc, Q_ITER, KV_ITER);
        float sum_result[KV_ITER] = {0.f};
        constexpr int num_issue_kv_prefetch =
            Stages * prefetch_kv_stage;

        #pragma unroll
        for (uint32_t kv_iter = 0;
             kv_iter < KV_ITER - prefetch_kv_stage;
             ++kv_iter) {
            const uint32_t load_kv_iter = kv_iter + prefetch_kv_stage;
            const int page_delta = load_kv_iter * KV_TILE / BLOCK_KV;
            const int row_in_page = load_kv_iter * KV_TILE % BLOCK_KV;
            const int current_block_idx =
                kv_block_idx[page_delta] != -1
                    ? kv_block_idx[page_delta]
                    : fallback_block_idx;
            auto gKv = kv_block +
                       static_cast<int64_t>(current_block_idx) *
                           BLOCK_KV * (kHeadDim + 4);
            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
            glob_kv_desc[1] =
                static_cast<long>(0x20000) << 32 | 0xFFFFFFFE;
            auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
            auto glob_stage_offset = row_in_page * kHeadDim;
            auto smem_per_stage = smem_warp_kv_start +
                                  kv_iter % prefetch_kv_stage *
                                      KV_TILE * kHeadDim;
            #pragma unroll
            for (uint32_t k_block = 0;
                 k_block < K_BLOCK_MAX;
                 ++k_block) {
                auto fetch_lds_kv =
                    base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
                WAIT_VMCNT_LDS(num_issue_kv_prefetch - 1);

                int lds_addr = reinterpret_cast<size_t>(
                    smem_per_stage + fetch_lds_kv);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile(
                    "\n ds_read_b128 %0 ,%1\n\t"
                    "s_waitcnt lgkmcnt(0) \n\t"
                    "s_barrier \n\t"
                    : "=v"(reg_kv_operand)
                    : "v"(lds_addr)
                    :);
                __builtin_amdgcn_sched_barrier(0);

                auto smem_ptr =
                    smem_per_stage + k_block * KV_TILE * K_TILE;
                int inner_warp_offset =
                    glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                    lane_idx / 4 * kHeadDim + k_block * K_TILE;
                buffer_load_lds_x4(
                    src, smem_ptr, inner_warp_offset);

                #pragma unroll
                for (int i = 0; i < Q_ITER; ++i) {
                    builtin_fp8_mmac<is_e4m3>(
                        reg_kv_operand.front,
                        reg_q_operand[i][k_block].front,
                        acc[i][kv_iter]);
                    builtin_fp8_mmac<is_e4m3>(
                        reg_kv_operand.rear,
                        reg_q_operand[i][k_block].rear,
                        acc[i][kv_iter]);
                }

                if (k_block == K_BLOCK_MAX - 1) {
                    const auto& transform = [&](
                                                const uint32_t& j,
                                                const uint32_t& k,
                                                const float& f) {
                        return fmaxf(f, 0) * reg_weights[j * 4 + k];
                    };

                    auto& sum = sum_result[kv_iter];
                    // Intra-thread reduction.
                    for (int j = 0; j < Q_ITER; ++j) {
                        for (int k = 0; k < 4; ++k) {
                            sum += transform(j, k, acc[j][kv_iter][k]);
                        }
                    }

                    sum *= kv_scale[kv_iter];

                    // Inter-thread reduction.
                    #pragma unroll
                    for (uint32_t shfl_idx = 16, j = 0;
                         j < 2;
                         ++j, shfl_idx <<= 1) {
                        sum += __shfl_down_sync(
                            0xffffffffffffffff, sum, shfl_idx);
                    }
                }
            }
        }

        #pragma unroll
        for (uint32_t kv_iter = KV_ITER - prefetch_kv_stage;
             kv_iter < KV_ITER;
             ++kv_iter) {
            auto smem_per_stage = smem_warp_kv_start +
                                  kv_iter % prefetch_kv_stage *
                                      KV_TILE * kHeadDim;
            #pragma unroll
            for (uint32_t k_block = 0; k_block < Stages; ++k_block) {
                auto fetch_lds_kv =
                    base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
                WAIT_VMCNT_LDS(
                    num_issue_kv_prefetch -
                    (kv_iter - (KV_ITER - prefetch_kv_stage)) * Stages -
                    k_block - 1);

                int lds_addr = reinterpret_cast<size_t>(
                    smem_per_stage + fetch_lds_kv);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile(
                    "\n ds_read_b128 %0 ,%1\n\t"
                    "s_waitcnt lgkmcnt(0) \n\t"
                    : "=v"(reg_kv_operand)
                    : "v"(lds_addr)
                    :);
                __builtin_amdgcn_sched_barrier(0);

                #pragma unroll
                for (int i = 0; i < Q_ITER; ++i) {
                    builtin_fp8_mmac<is_e4m3>(
                        reg_kv_operand.front,
                        reg_q_operand[i][k_block].front,
                        acc[i][kv_iter]);
                    builtin_fp8_mmac<is_e4m3>(
                        reg_kv_operand.rear,
                        reg_q_operand[i][k_block].rear,
                        acc[i][kv_iter]);
                }

                if (k_block == K_BLOCK_MAX - 1) {
                    const auto& transform = [&](
                                                const uint32_t& j,
                                                const uint32_t& k,
                                                const float& f) {
                        return fmaxf(f, 0) * reg_weights[j * 4 + k];
                    };

                    auto& sum = sum_result[kv_iter];
                    // Intra-thread reduction.
                    for (int j = 0; j < Q_ITER; ++j) {
                        for (int k = 0; k < 4; ++k) {
                            sum += transform(j, k, acc[j][kv_iter][k]);
                        }
                    }

                    sum *= kv_scale[kv_iter];

                    // Inter-thread reduction.
                    #pragma unroll
                    for (uint32_t shfl_idx = 16, j = 0;
                         j < 2;
                         ++j, shfl_idx <<= 1) {
                        sum += __shfl_down_sync(
                            0xffffffffffffffff, sum, shfl_idx);
                    }
                }
            }
        }

        __syncthreads();
        auto* smem_write_back =
            reinterpret_cast<float*>(smem_kv_block);
        if (lane_idx < 16) {
            #pragma unroll
            for (int i = 0; i < KV_ITER; ++i) {
                smem_write_back[
                    warp_idx * kTokensPerWarp + i * KV_TILE +
                    lane_idx % 16] = sum_result[i];
            }
        }
        __syncthreads();

        float permuted_result =
            lane_idx < kTokensPerWarp
                ? smem_write_back[
                      warp_idx * kTokensPerWarp + lane_idx]
                : 0.0f;

        // Store only the valid prefix. The padded tail remains uninitialized.
        auto seq_kv_offset = base_seq_kv_offset + lane_idx;
        if (lane_idx < kTokensPerWarp &&
            seq_kv_offset < context_len - (kNextN - next_n) + 1) {
            logits[base_logits_offset + seq_kv_offset] =
                permuted_result;
        } else if (lane_idx < kTokensPerWarp &&
                   seq_kv_offset < logits_stride) {
            logits[base_logits_offset + seq_kv_offset] = -INFINITY;
        }

        // The next persistent iteration reuses the same LDS bytes for Q,
        // weights, scales, KV tiles, and writeback.  All waves must finish
        // reading the current writeback before any wave starts overwriting it.
        if (cta_idx + gridDim.x < num_ctas) {
            __syncthreads();
        }
    }

    if constexpr (kClearTail) {
        // The last computed CTA already writes -infinity after context_len.
        // Cover all later 256-token output segments without loading Q or KV.
        int first_tail_cta = static_cast<int>(blockIdx.x);
        if (first_tail_cta < num_ctas) {
            first_tail_cta +=
                ceil_div(num_ctas - first_tail_cta,
                         static_cast<int>(gridDim.x)) *
                static_cast<int>(gridDim.x);
        }
        for (int cta_idx = first_tail_cta; cta_idx < num_output_ctas;
             cta_idx += gridDim.x) {
            const int cta_seq_kv_offset = cta_idx * kTokensPerCta;
            for (int i = t_id; i < kTokensPerCta; i += blockDim.x) {
                const int seq_kv_offset = cta_seq_kv_offset + i;
                if (seq_kv_offset < logits_limit) {
                    logits[base_logits_offset + seq_kv_offset] = -INFINITY;
                }
            }
        }
    }
}

namespace {

template <int kNumWarps, bool kClearTail>
torch::Tensor launch_fp8_persisten(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int max_context_len,
    int persistent_ctas) {
    constexpr int kNextN = 1;
    constexpr int kNumHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kBlockKV = 64;
    constexpr int kTokensPerWarp = 64;
    constexpr int kBatchSplit = kBlockKV * 4;
    constexpr int kThreads = kNumWarps * 64;

    const auto batch_size = static_cast<int>(q.size(0));
    const auto num_kv_blocks =
        static_cast<int>(fused_kv_cache.size(0));
    const auto kv_cache_stride_bytes =
        static_cast<uint64_t>(fused_kv_cache.stride(0));
    const auto aligned_context_len =
        (max_context_len + kBlockKV - 1) / kBlockKV * kBlockKV;
    const auto logits_stride =
        static_cast<uint64_t>(aligned_context_len);
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));

    auto logits_storage = torch::empty(
        {batch_size, aligned_context_len},
        q.options().dtype(torch::kFloat));
    auto logits = logits_storage.slice(-1, 0, max_context_len);

    const int smem_kv_pipe_size =
        kNumWarps * 32 * kHeadDim;
    const int smem_q_cache_size = kNumHeads * kHeadDim;
    const int smem_weight_cache_size =
        kNumHeads * static_cast<int>(sizeof(float));
    const int smem_scale_cache_size =
        kNumWarps * 64 * static_cast<int>(sizeof(float));
    const int smem_cache_size =
        smem_q_cache_size + smem_weight_cache_size +
        smem_scale_cache_size;
    const int smem_size =
        smem_kv_pipe_size > smem_cache_size ? smem_kv_pipe_size
                                             : smem_cache_size;

    dim3 grid(
        static_cast<uint32_t>(persistent_ctas),
        static_cast<uint32_t>(batch_size),
        1);
    dim3 block(kThreads, 1, 1);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(q.get_device());

    auto* kv_bytes =
        static_cast<uint8_t*>(fused_kv_cache.data_ptr());
    auto* kv_scales = reinterpret_cast<float*>(
        kv_bytes + kBlockKV * kHeadDim);
    paged_mqa_logits_fp8_length_masked_persistent<
        at::Float8_e4m3fn,
        kNextN,
        kNumHeads,
        kHeadDim,
        kBlockKV,
        kBatchSplit,
        kNumWarps,
        kTokensPerWarp,
        kClearTail><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const at::Float8_e4m3fn*>(q.data_ptr()),
        reinterpret_cast<const at::Float8_e4m3fn*>(
            fused_kv_cache.data_ptr()),
        kv_scales,
        weights.data_ptr<float>(),
        batch_size,
        num_kv_blocks,
        kv_cache_stride_bytes,
        logits_stride,
        block_table_stride,
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(),
        block_table.data_ptr<int>(),
        nullptr);

    const auto error = hipGetLastError();
    TORCH_CHECK(
        error == hipSuccess,
        "length-masked paged MQA launch failed: ",
        hipGetErrorString(error));
    return logits;
}

constexpr int kHeadDim = 128;
constexpr int kPageSize = 64;
constexpr int kTokensPerCta = 256;

void check_grouped_same_device(const torch::Tensor& reference,
                               const torch::Tensor& value,
                               const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(
        value.device() == reference.device(),
        name,
        " must be on ",
        reference.device());
}

template <int kHeads, bool kFp8, int kGroupRows,
          bool kLastQLds, bool kReduce = true>
__device__ __forceinline__ float grouped_mqa_row_sum(
    const std::conditional_t<kFp8, f8x16_t, i8x16_t>
        (&reg_q)[kLastQLds ? kGroupRows - 1 : kGroupRows]
                [(kHeads + 15) / 16][2],
    const uint8_t* shared_q,
    const std::conditional_t<kFp8, f8x16_t, i8x16_t> (&reg_kv)[2],
    const float* shared_weights, int row, int lane, float scale) {
    using accumulator_t = std::conditional_t<kFp8, fp32x4, int32x4_t>;
    constexpr int kQIterations = (kHeads + 15) / 16;
    accumulator_t acc[kQIterations];
    #pragma unroll
    for (int qi = 0; qi < kQIterations; ++qi) {
        acc[qi] = {0, 0, 0, 0};
    }
    #pragma unroll
    for (int block = 0; block < 2; ++block) {
        #pragma unroll
        for (int qi = 0; qi < kQIterations; ++qi) {
            std::conditional_t<kFp8, f8x16_t, i8x16_t> row_q;
            if constexpr (kLastQLds) {
                if (row == kGroupRows - 1) {
                    const int q_offset = (lane & 15) * 128 +
                        (lane >> 4) * 16 + qi * 16 * 128 + block * 64;
                    row_q = *reinterpret_cast<const i8x16_t*>(
                        shared_q + q_offset);
                } else {
                    row_q = reg_q[row][qi][block];
                }
            } else {
                row_q = reg_q[row][qi][block];
            }
            if constexpr (kFp8) {
                builtin_fp8_mmac<true>(
                    reg_kv[block].front, row_q.front, acc[qi]);
                builtin_fp8_mmac<true>(
                    reg_kv[block].rear, row_q.rear, acc[qi]);
            } else {
                builtin_i8_mmac(reg_kv[block].front, row_q.front, acc[qi]);
                builtin_i8_mmac(reg_kv[block].rear, row_q.rear, acc[qi]);
            }
        }
    }
    const float* row_weights = shared_weights + row * kHeads;
    float sum = 0.0f;
    #pragma unroll
    for (int qi = 0; qi < kQIterations; ++qi) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int head = qi * 16 + item * 4 + (lane >> 4);
            const float weight = head < kHeads ? row_weights[head] : 0.0f;
            sum += fmaxf(static_cast<float>(acc[qi][item]), 0.0f) * weight;
        }
    }
    sum *= scale;
    if constexpr (kReduce) {
        #pragma unroll
        for (int delta = 16; delta <= 32; delta <<= 1) {
            sum += __shfl_down_sync(0xffffffffffffffffull, sum, delta);
        }
    }
    return sum;
}

__device__ __forceinline__ float select_four(
    float a, float b, float c, float d, int index) {
    return index == 0 ? a : index == 1 ? b : index == 2 ? c : d;
}

// One wave owns a 16-token tile. Consecutive rows reuse a KV operand only
// when their physical page is equal; unrelated requests remain independent.
template <int kHeads, bool kFp8, int kGroupRows, bool kLastQLds>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_persisten_grouped(
    const uint8_t* q,
    const uint8_t* packed_kv,
    const uint64_t kv_stride_bytes,
    const float* weights,
    const uint64_t logits_stride,
    const uint64_t table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table,
    const int batch_size,
    const int ctas_per_group) {
    constexpr int kHeadsPerMma = 16;
    constexpr int kTokenTile = 16;
    constexpr int kQIterations = (kHeads + kHeadsPerMma - 1) / kHeadsPerMma;
    constexpr int kKBlocks = 2;
    constexpr int kTokenChunk = 256;
    constexpr int kPageBytes = 64 * (128 + 4);
    static_assert(kHeads == 8 || kHeads == 16 || kHeads == 32 || kHeads == 64);
    using operand_t = std::conditional_t<kFp8, f8x16_t, i8x16_t>;

    const int group = blockIdx.y;
    const int group_first = group * kGroupRows;
    const int valid_rows = min(kGroupRows, batch_size - group_first);
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int q_base = (lane & 15) * 128 + (lane >> 4) * 16;
    const int kv_base = q_base;
    int row_length[kGroupRows];
    int group_max = 0;
    #pragma unroll
    for (int row = 0; row < kGroupRows; ++row) {
        const int raw = row < valid_rows ? context_lens[group_first + row] : 0;
        const int length = raw < 0 ? 0 :
            (raw > logits_stride ? static_cast<int>(logits_stride) : raw);
        row_length[row] = length;
        group_max = max(group_max, length);
    }
    operand_t reg_q[kLastQLds ? kGroupRows - 1 : kGroupRows]
                   [kQIterations][kKBlocks];
    #pragma unroll
    for (int row = 0;
         row < (kLastQLds ? kGroupRows - 1 : kGroupRows); ++row) {
        const uint8_t* row_q = q +
            (group_first + min(row, valid_rows - 1)) * kHeads * 128;
        #pragma unroll
        for (int qi = 0; qi < kQIterations; ++qi) {
            #pragma unroll
            for (int block = 0; block < kKBlocks; ++block) {
                const int q_offset = q_base +
                    qi * kHeadsPerMma * 128 + block * 64;
                // H8 fills only half of the 16-head MMAC tile. The other
                // operand lanes must be zero and must not read past this row.
                reg_q[row][qi][block] =
                    qi * kHeadsPerMma + (lane & 15) < kHeads
                    ? *reinterpret_cast<const operand_t*>(row_q + q_offset)
                    : operand_t{};
            }
        }
    }
    __shared__ __align__(16) uint8_t shared_q[
        kLastQLds ? kHeads * 128 : 1];
    if constexpr (kLastQLds) {
        if (valid_rows == kGroupRows) {
            const uint8_t* last_q = q +
                (group_first + kGroupRows - 1) * kHeads * 128;
            for (int byte = threadIdx.x * 16; byte < kHeads * 128;
                 byte += blockDim.x * 16) {
                *reinterpret_cast<i8x16_t*>(shared_q + byte) =
                    *reinterpret_cast<const i8x16_t*>(last_q + byte);
            }
        }
    }
    __shared__ float shared_weights[kGroupRows * kHeads];
    for (int item = threadIdx.x; item < valid_rows * kHeads;
         item += blockDim.x) {
        shared_weights[item] = weights[group_first * kHeads + item];
    }
    __syncthreads();
    const int chunks = ceil_div(static_cast<int>(logits_stride), kTokenChunk);

    for (int chunk = blockIdx.x; chunk < chunks; chunk += ctas_per_group) {
        const int chunk_start = chunk * kTokenChunk;
        if (chunk_start >= group_max) {
            const int token = chunk_start + threadIdx.x;
            if (token < logits_stride) {
                #pragma unroll
                for (int row = 0; row < kGroupRows; ++row) {
                    if (row < valid_rows) {
                        logits[(group_first + row) * logits_stride + token] =
                            -INFINITY;
                    }
                }
            }
            continue;
        }
        // A wave owns one 64-token page within this 256-token chunk. Reuse
        // its page-table entries for all four 16-token MMAC tiles.
        const int page_start = chunk_start + wave * kPageSize;
        int row_page[kGroupRows];
        #pragma unroll
        for (int row = 0; row < kGroupRows; ++row) {
            int page = -1;
            if (row < valid_rows && page_start < row_length[row]) {
                // All lanes address the same table entry. A wave-uniform
                // load can use the L1 broadcast path without a shuffle.
                page = block_table[
                    (group_first + row) * table_stride + page_start / kPageSize];
            }
            row_page[row] = page;
        }
        bool same_page = row_page[0] >= 0;
        #pragma unroll
        for (int row = 1; row < kGroupRows; ++row) {
            if (row < valid_rows && row_page[row] != row_page[0]) {
                same_page = false;
            }
        }
        #pragma unroll
        for (int tile_iter = 0; tile_iter < 4; ++tile_iter) {
            const int token_start = page_start + tile_iter * kTokenTile;
            int cached_page = -2;
            operand_t reg_kv[kKBlocks] = {};
            float scale = 0.0f;
            if (same_page) {
                const uint8_t* page = packed_kv +
                    static_cast<int64_t>(row_page[0]) * kv_stride_bytes;
                const int page_token = token_start & 63;
                #pragma unroll
                for (int block = 0; block < kKBlocks; ++block) {
                    reg_kv[block] = *reinterpret_cast<const operand_t*>(
                        page + page_token * 128 + kv_base + block * 64);
                }
                scale = reinterpret_cast<const float*>(
                    page + kPageBytes - 64 * 4)[page_token + (lane & 15)];
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_sched_barrier(0);

                if constexpr (kHeads == 32 && !kFp8 &&
                              kGroupRows == 4) {
                    const float p0 = valid_rows > 0
                        ? grouped_mqa_row_sum<kHeads, kFp8, kGroupRows, kLastQLds, false>(
                              reg_q, shared_q, reg_kv, shared_weights, 0, lane, scale)
                        : 0.0f;
                    const float p1 = valid_rows > 1
                        ? grouped_mqa_row_sum<kHeads, kFp8, kGroupRows, kLastQLds, false>(
                              reg_q, shared_q, reg_kv, shared_weights, 1, lane, scale)
                        : 0.0f;
                    const float p2 = valid_rows > 2
                        ? grouped_mqa_row_sum<kHeads, kFp8, kGroupRows, kLastQLds, false>(
                              reg_q, shared_q, reg_kv, shared_weights, 2, lane, scale)
                        : 0.0f;
                    float p3 = 0.0f;
                    if constexpr (kGroupRows >= 4) {
                        if (valid_rows > 3) {
                            p3 = grouped_mqa_row_sum<
                                kHeads, kFp8, kGroupRows, kLastQLds, false>(
                                    reg_q, shared_q, reg_kv, shared_weights,
                                    3, lane, scale);
                        }
                    }
                    const int slot = lane >> 4;
                    float sum = select_four(p0, p1, p2, p3, slot);
                    const float peer1 = select_four(
                        p0, p1, p2, p3, slot ^ 1);
                    const float peer2 = select_four(
                        p0, p1, p2, p3, slot ^ 2);
                    const float peer3 = select_four(
                        p0, p1, p2, p3, slot ^ 3);
                    sum += __shfl_xor_sync(
                        0xffffffffffffffffull, peer1, 16);
                    sum += __shfl_xor_sync(
                        0xffffffffffffffffull, peer2, 32);
                    sum += __shfl_xor_sync(
                        0xffffffffffffffffull, peer3, 48);
                    int length = row_length[0];
                    if (slot == 1) length = row_length[1];
                    if (slot == 2) length = row_length[2];
                    if constexpr (kGroupRows >= 4) {
                        if (slot == 3) length = row_length[3];
                    }
                    const int token = token_start + (lane & 15);
                    if (slot < valid_rows && slot < kGroupRows &&
                        token < logits_stride) {
                        logits[(group_first + slot) * logits_stride + token] =
                            token < length ? sum : -INFINITY;
                    }
                } else {
                    #pragma unroll
                    for (int row = 0; row < kGroupRows; ++row) {
                        if (row >= valid_rows) continue;
                        const float sum =
                            grouped_mqa_row_sum<kHeads, kFp8, kGroupRows, kLastQLds>(
                                reg_q, shared_q, reg_kv, shared_weights,
                                row, lane, scale);
                        if (lane < kTokenTile) {
                            const int token = token_start + lane;
                            if (token < logits_stride) {
                                logits[(group_first + row) * logits_stride + token] =
                                    token < row_length[row] ? sum : -INFINITY;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int row = 0; row < kGroupRows; ++row) {
                    if (row >= valid_rows) {
                        continue;
                    }
                    const int physical_page = row_page[row];
                    const bool kv_valid = physical_page >= 0;
                    if (physical_page != cached_page) {
                        cached_page = physical_page;
                        scale = 0.0f;
                        if (kv_valid) {
                            const uint8_t* page = packed_kv +
                                static_cast<int64_t>(physical_page) * kv_stride_bytes;
                            const int page_token = token_start & 63;
                            #pragma unroll
                            for (int block = 0; block < kKBlocks; ++block) {
                                reg_kv[block] = *reinterpret_cast<const operand_t*>(
                                    page + page_token * 128 + kv_base + block * 64);
                            }
                            scale = reinterpret_cast<const float*>(
                                page + kPageBytes - 64 * 4)[page_token + (lane & 15)];
                        }
                        __builtin_amdgcn_sched_barrier(0);
                        __builtin_amdgcn_sched_barrier(0);

                    }
                    float sum = 0.0f;
                    if (kv_valid) {
                        sum = grouped_mqa_row_sum<kHeads, kFp8, kGroupRows, kLastQLds>(
                            reg_q, shared_q, reg_kv, shared_weights, row, lane, scale);
                    }
                    if (lane < kTokenTile) {
                        const int token = token_start + lane;
                        if (token < logits_stride) {
                            logits[(group_first + row) * logits_stride + token] =
                                kv_valid && token < row_length[row]
                                ? sum : -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

template <int kHeads, bool kFp8, int kGroupRows>
void launch_native_persisten_grouped(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& logits,
    int ctas_per_group,
    hipStream_t stream) {
    const dim3 grid(ctas_per_group,
                    (q.size(0) + kGroupRows - 1) / kGroupRows);
    constexpr bool kLastQLds = !kFp8 && kHeads == 32 && kGroupRows == 5;
    paged_mqa_logits_persisten_grouped<
        kHeads, kFp8, kGroupRows, kLastQLds>
        <<<grid, 256, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(fused_kv_cache.data_ptr()),
            fused_kv_cache.stride(0),
            weights.data_ptr<float>(),
            logits.stride(0),
            block_table.stride(0),
            context_lens.data_ptr<int>(),
            logits.data_ptr<float>(),
            block_table.data_ptr<int>(),
            static_cast<int>(q.size(0)),
            ctas_per_group);
}

template <int kHeads, bool kFp8>
void launch_native_persisten(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& logits,
    int ctas_per_group,
    int group_size,
    hipStream_t stream) {
#define LAUNCH_GROUP(G) \
    launch_native_persisten_grouped<kHeads, kFp8, G>( \
        q, fused_kv_cache, weights, context_lens, block_table, logits, \
        ctas_per_group, stream)
    switch (group_size) {
        case 1: LAUNCH_GROUP(1); break;
        case 2: LAUNCH_GROUP(2); break;
        case 3: LAUNCH_GROUP(3); break;
        case 4: LAUNCH_GROUP(4); break;
        case 5: LAUNCH_GROUP(5); break;
    }
#undef LAUNCH_GROUP
}

torch::Tensor paged_mqa_logits_persisten_impl(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size) {
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    check_grouped_same_device(q, fused_kv_cache, "fused_kv_cache");
    check_grouped_same_device(q, weights, "weights");
    check_grouped_same_device(q, context_lens, "context_lens");
    check_grouped_same_device(q, block_table, "block_table");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    TORCH_CHECK(
        !schedule_meta.has_value(),
        "grouped persistent MQA requires schedule_meta=None");
    TORCH_CHECK(
        clean_logits,
        "grouped persistent MQA requires clean_logits=True");

    hipDeviceProp_t properties;
    const auto prop_error =
        hipGetDeviceProperties(&properties, q.get_device());
    TORCH_CHECK(
        prop_error == hipSuccess,
        "hipGetDeviceProperties failed: ",
        hipGetErrorString(prop_error));
    const bool is_supported_gfx938 =
        std::strncmp(properties.gcnArchName, "gfx938", 6) == 0 &&
        (properties.multiProcessorCount == 64 ||
         properties.multiProcessorCount == 72);
    const bool supported_device =
        (std::strncmp(properties.gcnArchName, "gfx936", 6) == 0 &&
         properties.multiProcessorCount == 80) ||
        is_supported_gfx938;
    TORCH_CHECK(
        supported_device,
        "grouped persistent MQA requires gfx936/80CU or gfx938/64CU/72CU, got ",
        properties.gcnArchName,
        "/",
        properties.multiProcessorCount,
        "CU");

    TORCH_CHECK(
        group_size >= 1 && group_size <= 5,
        "group_size must be in [1, 5]");
    const bool supported_batch = q.dim() == 4 && q.size(0) > 0;
    TORCH_CHECK(
        (q.scalar_type() == torch::kInt8 ||
         q.scalar_type() == torch::kFloat8_e4m3fn) && q.dim() == 4 &&
            supported_batch && q.size(1) == 1 &&
            (q.size(2) == 8 || q.size(2) == 16 ||
             q.size(2) == 32 || q.size(2) == 64) &&
            q.size(3) == kHeadDim && q.is_contiguous(),
        "q must be contiguous INT8 or FP8 [batch, 1, 8/16/32/64, 128]");
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const bool int8_kv = fused_kv_cache.scalar_type() == torch::kInt8;
    const bool fp8_kv = fused_kv_cache.scalar_type() == torch::kUInt8;
    TORCH_CHECK(
        ((q.scalar_type() == torch::kInt8 && int8_kv) ||
         (q.scalar_type() == torch::kFloat8_e4m3fn && fp8_kv &&
          is_supported_gfx938)) &&
            fused_kv_cache.dim() == 4 && fused_kv_cache.size(0) > 0 &&
            fused_kv_cache.size(1) == kPageSize &&
            fused_kv_cache.size(2) == 1 &&
            fused_kv_cache.size(3) == 132 &&
            fused_kv_cache.stride(1) == 132 &&
            fused_kv_cache.stride(2) == 132 &&
            fused_kv_cache.stride(3) == 1,
        "INT8 q requires packed INT8 KV; FP8 q requires gfx938 and packed FP8 KV");
    TORCH_CHECK(
        weights.scalar_type() == torch::kFloat && weights.dim() == 2 &&
            weights.size(0) == batch_size && weights.size(1) == num_heads &&
            weights.is_contiguous(),
        "weights must be contiguous float32 [batch, heads]");
    TORCH_CHECK(
        context_lens.scalar_type() == torch::kInt &&
            context_lens.dim() == 1 && context_lens.size(0) == batch_size &&
            context_lens.is_contiguous(),
        "context_lens must be contiguous int32 [batch]");
    TORCH_CHECK(
        block_table.scalar_type() == torch::kInt && block_table.dim() == 2 &&
            block_table.size(0) == batch_size && block_table.size(1) > 0 &&
            block_table.stride(1) == 1,
        "block_table must be row-contiguous int32 [batch, pages]");
    TORCH_CHECK(
        max_context_len > 0 &&
            max_context_len <= block_table.size(1) * kPageSize,
        "max_context_len must fit in block_table capacity");
    TORCH_CHECK(num_warps == 4, "grouped persistent MQA requires 4 warps per row");
    TORCH_CHECK(
        persistent_ctas >= 0 && persistent_ctas <= 4096,
        "persistent_ctas must be 0 (auto) or in [1, 4096]");

    // Preserve the historical H32 FP8 MMAC implementation and its persistent
    // CTA scheduling. The existing paged_mqa_logits and TopK remain untouched.
    if (q.scalar_type() == torch::kFloat8_e4m3fn && group_size == 1 &&
        num_heads == 32 && properties.multiProcessorCount == 64) {
        TORCH_CHECK(
            std::strncmp(properties.gcnArchName, "gfx938", 6) == 0 &&
                properties.multiProcessorCount == 64 && num_heads == 32 &&
                fp8_kv && fused_kv_cache.stride(0) == 64 * 132,
            "persistent FP8 MQA requires gfx938/64CU, 32 heads, and packed FP8 KV");
        const auto max_chunks =
            (static_cast<int64_t>(max_context_len) + 255) / 256;
        const auto batch_aware_floor =
            std::max<int64_t>(128, (2048 + batch_size - 1) / batch_size);
        const int resolved = persistent_ctas == 0
            ? static_cast<int>(std::min<int64_t>(max_chunks, batch_aware_floor))
            : static_cast<int>(std::min<int64_t>(max_chunks, persistent_ctas));
        return launch_fp8_persisten<4, true>(
            q, fused_kv_cache, weights, context_lens, block_table,
            max_context_len, resolved);
    }
    const int aligned_context_len =
        (max_context_len + kPageSize - 1) / kPageSize * kPageSize;
    auto logits_storage = torch::empty(
        {batch_size, aligned_context_len}, q.options().dtype(torch::kFloat));
    auto logits = logits_storage.slice(-1, 0, max_context_len);

    const auto max_chunks =
        (static_cast<int64_t>(max_context_len) + kTokensPerCta - 1) /
        kTokensPerCta;
    const int resolved = persistent_ctas == 0
        ? static_cast<int>(std::min<int64_t>(max_chunks, 128))
        : static_cast<int>(std::min<int64_t>(persistent_ctas, max_chunks));
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
#define LAUNCH_HEADS(HEADS, FP8) \
    launch_native_persisten<HEADS, FP8>( \
        q, fused_kv_cache, weights, context_lens, block_table, logits, \
        resolved, group_size, stream)
    if (int8_kv) {
        switch (num_heads) {
            case 8: LAUNCH_HEADS(8, false); break;
            case 16: LAUNCH_HEADS(16, false); break;
            case 32: LAUNCH_HEADS(32, false); break;
            case 64: LAUNCH_HEADS(64, false); break;
        }
    } else {
        switch (num_heads) {
            case 8: LAUNCH_HEADS(8, true); break;
            case 16: LAUNCH_HEADS(16, true); break;
            case 32: LAUNCH_HEADS(32, true); break;
            case 64: LAUNCH_HEADS(64, true); break;
        }
    }
#undef LAUNCH_HEADS
    return logits;
}

}  // namespace

torch::Tensor paged_mqa_logits_persisten(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size) {
    return paged_mqa_logits_persisten_impl(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas,
        group_size);
}

}  // namespace deep_gemm
}  // namespace lightop
