#pragma once

#include "paged_mqa_logits_impl.hpp"

namespace lightop {
namespace deep_gemm {

// The public length-masked entries instantiate kClearTail=false and preserve
// their prefix-only contract.  The private compatibility route from
// paged_mqa_logits(clean_logits=True) instantiates kClearTail=true so the old
// generic API keeps its full -infinity tail guarantee.  Every context_lens
// value is still required to be in [0, logits_stride], and every referenced
// block-table entry must be in [0, num_kv_blocks).  The defensive clamp
// protects score buffer accesses; it does not relax the caller's value-range
// contract.
// Unlike paged_mqa_logits_fp8, the fixed CTA pool walks all useful 256-token
// segments instead of launching against the padded logits width.
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

// The SparkV1 BF16 specialization keeps one query row and its head weights in
// registers while a fixed CTA pool walks all useful 256-token segments.  The
// legacy paged_mqa_logits kernel launches one CTA for every padded segment and
// reloads Q/weights for every CTA.  Reusing them is especially important for
// the production shape [local_bs=8, heads=64, head_dim=128, page=64].
template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kNumWarps = 4, int kTokensPerWarp = BLOCK_KV,
          bool kGroupScheduled = false, bool kClearTail = false>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_bf16_length_masked_persistent(
    const Element* q,
    const Element* kv_block,
    const float* weights,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table) {
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
    const int num_output_ctas = ceil_div(logits_limit, kTokensPerCta);
    const int num_kv = ceil_div(context_len, BLOCK_KV);
    const uint64_t base_logits_offset = q_idx * logits_stride;
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
    __syncthreads();

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
            if (page_idx < num_kv) {
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
            if (lane_idx < KV_TILE) {
                const int seq_kv_offset = seq_kv_tile_offset + lane_idx;
                if (seq_kv_offset < context_len) {
                    logits[base_logits_offset + seq_kv_offset] =
                        sum_result[kv_iter];
                } else if constexpr (kClearTail) {
                    if (seq_kv_offset < logits_limit) {
                        logits[base_logits_offset + seq_kv_offset] =
                            -INFINITY;
                    }
                }
            }
        }

        // KV LDS and output ownership stay wave-local across persistent
        // iterations, so no CTA-wide synchronization is needed here.
    }

    if constexpr (kClearTail) {
        // Preserve paged_mqa_logits(clean_logits=True) in the same dispatch.
        // The partial final compute CTA is handled by the store above; these
        // iterations cover only later full tail CTAs.  With context_len==0,
        // num_ctas is zero and every CTA starts directly in this loop.
        int first_tail_cta = cta_seed;
        if (first_tail_cta < num_ctas) {
            first_tail_cta +=
                ceil_div(num_ctas - first_tail_cta, ctas_per_row) *
                ctas_per_row;
        }
        for (int cta_idx = first_tail_cta; cta_idx < num_output_ctas;
             cta_idx += ctas_per_row) {
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

}  // namespace deep_gemm
}  // namespace lightop
