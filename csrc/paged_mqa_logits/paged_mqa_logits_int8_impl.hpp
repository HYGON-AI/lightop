#pragma once

#include "paged_mqa_logits_impl.hpp"

namespace lightop {
namespace deep_gemm {

using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using int8x16_t = __attribute__((__vector_size__(16 * sizeof(int8_t)))) int8_t;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;

union i8x16_t {
    int8x16_t data;
    struct {
        int8x8_t front;
        int8x8_t rear;
    };
};

inline __device__ __forceinline__ void builtin_i8_mmac(const int8x8_t& reg_a, const int8x8_t& reg_b, int32x4_t& reg_c) {
#if defined(__gfx936__) || defined(__gfx928__)
    const auto* a = reinterpret_cast<const long*>(&reg_a);
    const auto* b = reinterpret_cast<const long*>(&reg_b);
    reg_c = __builtin_amdgcn_mmac_i32_16x16x32i8(*a, *b, reg_c);
#elif defined(__gfx938__)
    const auto a = *reinterpret_cast<const intx2*>(&reg_a);
    const auto b = *reinterpret_cast<const intx2*>(&reg_b);
    reg_c = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(a, b, reg_c, false, false, false);
#endif
}

template <typename Element, int kNextN = 1, int kNumHeads = 64, int kHeadDim = 128, int BLOCK_KV = 64, int kBatchSplit,
          int kNumWarps = 4, int kTokensPerWarp = BLOCK_KV>
__attribute__((amdgpu_flat_work_group_size(1, 512))) __global__ void
paged_mqa_logits_int8(const Element* q, const Element* kv_block, const float* kv_block_scales, const float* weights,
                      const int, const int, const uint64_t kv_cache_stride_bytes, const uint64_t logits_stride,
                      const uint64_t block_table_stride, const int* context_lens, float* logits, const int* block_table,
                      const int*) {
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
    LIGHTOP_STATIC_ASSERT(std::is_same<Element, int8_t>::value, "Invalid INT8 element type");
    LIGHTOP_STATIC_ASSERT(kBatchSplit == BLOCK_KV * kNumWarps, "Invalid INT8 batch split");
    LIGHTOP_STATIC_ASSERT(BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64, "Invalid BLOCK_KV");
    LIGHTOP_STATIC_ASSERT(kTokensPerWarp % BLOCK_KV == 0 && kTokensPerWarp <= WARP_SIZE_GPU, "Invalid tokens per warp");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    int q_idx = blockIdx.y;
    int start_kv_idx = blockIdx.x / kNextN * kNumWarps * PAGES_PER_WARP;
    int kv_idx = start_kv_idx + warp_idx * PAGES_PER_WARP;
    int context_len = context_lens[q_idx];
    int num_kv = ceil_div(context_len, BLOCK_KV);

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;

    if (start_kv_idx >= num_kv) {
        uint32_t fill_offset = base_logits_offset + base_seq_kv_offset + lane_idx;
        if (lane_idx < kTokensPerWarp && base_seq_kv_offset + lane_idx < logits_stride) {
            logits[fill_offset] = -INFINITY;
        }
        return;
    }

    // fetch Q && Q weights
    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(uint8_t);
    constexpr int fetch_q_cnt = constexpr_ceil_div(kNumHeads * kHeadDim, kNumWarps * WARP_SIZE_GPU * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

#pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
            *reinterpret_cast<i8x16_t*>(&smem_kv_block[pos]) = *reinterpret_cast<const i8x16_t*>(&gQ[pos]);
        }
    }

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    // fetch kv block idx
    int kv_block_idx[PAGES_PER_WARP];
    uint64_t block_table_offset = q_idx * block_table_stride + kv_idx;
#pragma unroll
    for (int i = 0; i < PAGES_PER_WARP; ++i) {
        kv_block_idx[i] = kv_idx + i < num_kv ? block_table[block_table_offset + i] : -1;
    }

    // fetch kv block scales
    const int kv_scale_stride = kv_cache_stride_bytes / sizeof(float);
    auto* smem_kv_scales = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim + kNumHeads * sizeof(float));
    if (lane_idx < kTokensPerWarp) {
        const int lane_page_delta = lane_idx / BLOCK_KV;
        const int lane_row_in_page = lane_idx % BLOCK_KV;
        const int lane_block_idx = kv_block_idx[lane_page_delta];
        if (lane_block_idx != -1) {
            auto gKv_scales = kv_block_scales + lane_block_idx * kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
        } else {
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = 0.0f;
        }
    }

    constexpr int heads_cnt = Q_ITER;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    i8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
#pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
#pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            if constexpr (kNumHeads < Q_TILE) {
                reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, i8x16_t>(
                    smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
            } else {
                reg_q_operand[i][j] = *reinterpret_cast<i8x16_t*>(&smem_kv_block[offset]);
            }
        }
    }

    // pre-fetch weights
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

    // pre-fetch kv_scale
    float kv_scale[KV_ITER];
    const int logits_t_offset = warp_idx * kTokensPerWarp + lane_idx % 16;
#pragma unroll
    for (int i = 0; i < KV_ITER; ++i) {
        kv_scale[i] = smem_kv_scales[logits_t_offset + i * 16];
    }

    __syncthreads();

    // shared data loading complete, invalid warps return in advance
    if (kv_idx >= num_kv) {
        uint32_t fill_offset = base_logits_offset + base_seq_kv_offset + lane_idx;
        if (lane_idx < kTokensPerWarp && base_seq_kv_offset + lane_idx < logits_stride) {
            logits[fill_offset] = -INFINITY;
        }
        return;
    }

    // fetch kv block
    constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
    constexpr int prefetch_kv_stage = KV_ITER >= 2 ? 2 : 1;
    constexpr int smem_kv_block_per_warp = prefetch_kv_stage * KV_TILE * kHeadDim;

    auto smem_warp_kv_start = smem_kv_block + warp_idx * smem_kv_block_per_warp;

#pragma unroll
    for (int i = 0; i < prefetch_kv_stage; ++i) {
        const int page_delta = i * KV_TILE / BLOCK_KV;
        const int row_in_page = i * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[page_delta];
        auto gKv =
            current_block_idx != -1 ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
        glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
        auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + i * KV_TILE * kHeadDim;
#pragma unroll
        for (int j = 0; j < Stages; ++j) {
            auto smem_ptr = smem_per_stage + j * KV_TILE * K_TILE;
            auto inner_warp_offset = current_block_idx != -1 ? glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                                                                   lane_idx / 4 * kHeadDim + j * K_TILE
                                                             : -1;
            buffer_load_lds_x4(src, smem_ptr, inner_warp_offset);
        }
    }

    // pre-fetch KV from smem
    i8x16_t reg_kv_operand;

    auto base_fetch_lds_KV = lane_idx % 16 * K_TILE + lane_idx / 16 * 16;

    int32x4_t acc[Q_ITER][KV_ITER];
#pragma unroll
    for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
#pragma unroll
        for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
            acc[q_iter][kv_iter] = {0, 0, 0, 0};
        }
    }
    float sum_result[KV_ITER] = {0.f};
    constexpr int num_issue_kv_prefetch = Stages * prefetch_kv_stage;

#pragma unroll
    for (uint32_t kv_iter = 0; kv_iter < KV_ITER - prefetch_kv_stage; ++kv_iter) {
        const uint32_t load_kv_iter = kv_iter + prefetch_kv_stage;
        const int page_delta = load_kv_iter * KV_TILE / BLOCK_KV;
        const int row_in_page = load_kv_iter * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[page_delta];
        auto gKv =
            current_block_idx != -1 ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
        glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
        auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            WAIT_VMCNT_LDS(num_issue_kv_prefetch - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("\n ds_read_b128 %0 ,%1\n\t"
                         "s_waitcnt lgkmcnt(0) \n\t"
                         "s_barrier \n\t"
                         : "=v"(reg_kv_operand)
                         : "v"(lds_addr)
                         :);
            __builtin_amdgcn_sched_barrier(0);

            auto smem_ptr = smem_per_stage + k_block * KV_TILE * K_TILE;
            int inner_warp_offset = current_block_idx != -1 ? glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                                                                  lane_idx / 4 * kHeadDim + k_block * K_TILE
                                                            : -1;
            buffer_load_lds_x4(src, smem_ptr, inner_warp_offset);

#pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_i8_mmac(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_i8_mmac(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const int32_t& value) {
                    return fmaxf(static_cast<float>(value), 0.0f) * reg_weights[j * 4 + k];
                };

                auto& sum = sum_result[kv_iter];
                // Intra-thread reduction
                for (int j = 0; j < Q_ITER; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sum += transform(j, k, acc[j][kv_iter][k]);
                    }
                }

                sum *= kv_scale[kv_iter];

// Inter-thread reduction
#pragma unroll
                for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
                    sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
                }
            }
        }
    }

#pragma unroll
    for (uint32_t kv_iter = KV_ITER - prefetch_kv_stage; kv_iter < KV_ITER; ++kv_iter) {
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (uint32_t k_block = 0; k_block < Stages; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            WAIT_VMCNT_LDS(num_issue_kv_prefetch - (kv_iter - (KV_ITER - prefetch_kv_stage)) * Stages - k_block - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("\n ds_read_b128 %0 ,%1\n\t"
                         "s_waitcnt lgkmcnt(0) \n\t"
                         : "=v"(reg_kv_operand)
                         : "v"(lds_addr)
                         :);
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_i8_mmac(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_i8_mmac(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const int32_t& value) {
                    return fmaxf(static_cast<float>(value), 0.0f) * reg_weights[j * 4 + k];
                };

                auto& sum = sum_result[kv_iter];
                // Intra-thread reduction
                for (int j = 0; j < Q_ITER; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sum += transform(j, k, acc[j][kv_iter][k]);
                    }
                }

                sum *= kv_scale[kv_iter];

// Inter-thread reduction
#pragma unroll
                for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
                    sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
                }
            }
        }
    }

    __syncthreads();
    auto* smem_write_back = reinterpret_cast<float*>(smem_kv_block);
    if (lane_idx < 16) {
#pragma unroll
        for (int i = 0; i < KV_ITER; ++i) {
            smem_write_back[warp_idx * kTokensPerWarp + i * KV_TILE + lane_idx % 16] = sum_result[i];
        }
    }
    __syncthreads();

    float permuted_result = lane_idx < kTokensPerWarp ? smem_write_back[warp_idx * kTokensPerWarp + lane_idx] : 0.0f;

    // Store into the global memory
    auto seq_kv_offset = base_seq_kv_offset + lane_idx;
    if (lane_idx < kTokensPerWarp && seq_kv_offset < context_len - (kNextN - next_n) + 1) {
        logits[base_logits_offset + seq_kv_offset] = permuted_result;
    } else if (lane_idx < kTokensPerWarp && seq_kv_offset < logits_stride) {
        logits[base_logits_offset + seq_kv_offset] = -INFINITY;
    }
}

template <typename Element, int kNextN = 1, int kNumHeads = 64, int kHeadDim = 128, int BLOCK_KV = 4,
          int kTokensPerWarp = 64, int kNumWarps = 4>
__attribute__((amdgpu_flat_work_group_size(1, 512))) __global__ void
paged_mqa_logits_int8_small_block(const Element* q, const Element* kv_block, const float* kv_block_scales,
                                  const float* weights, const int, const int, const uint64_t kv_cache_stride_bytes,
                                  const uint64_t logits_stride, const uint64_t block_table_stride,
                                  const int* context_lens, float* logits, const int* block_table, const int*) {
    const int& warp_idx = threadIdx.x / 64;
    const int& lane_idx = threadIdx.x % 64;
    const int& t_id = threadIdx.x;

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    static constexpr int K_TILE = 64;
    static constexpr int Q_TILE = 16;
    static constexpr int KV_TILE = 16;
    // A second zero-padded tile hides small-page memory latency for H8 without reading extra Q or weights.
    static constexpr int Q_ITER = kNumHeads == 8 ? 2 : constexpr_ceil_div(kNumHeads, Q_TILE);
    static constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    static constexpr int PAGES_PER_WARP = kTokensPerWarp / BLOCK_KV;
    static constexpr int SUBPAGES_PER_TILE = KV_TILE / BLOCK_KV;
    static constexpr int Stages = kHeadDim / K_TILE;
    LIGHTOP_STATIC_ASSERT(std::is_same<Element, int8_t>::value, "Invalid INT8 element type");
    LIGHTOP_STATIC_ASSERT((BLOCK_KV == 4 || BLOCK_KV == 8) && kTokensPerWarp == 64, "Invalid small BLOCK_KV");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

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
    constexpr int q_required_warps = constexpr_ceil_div(kNumHeads * kHeadDim, WARP_SIZE_GPU * kAlignmentQ);
    constexpr int q_load_warps = q_required_warps < kNumWarps ? q_required_warps : kNumWarps;
    constexpr int fetch_q_cnt = constexpr_ceil_div(kNumHeads * kHeadDim, q_load_warps * WARP_SIZE_GPU * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * q_load_warps * WARP_SIZE_GPU;

    if (warp_idx < q_load_warps) {
        const int q_t_id = warp_idx * WARP_SIZE_GPU + lane_idx;
#pragma unroll
        for (int i = 0; i < fetch_q_cnt; ++i) {
            auto pos = q_t_id * kAlignmentQ + i * fetch_q_stride;
            if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
                *reinterpret_cast<i8x16_t*>(&smem_kv_block[pos]) = *reinterpret_cast<const i8x16_t*>(&gQ[pos]);
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
    auto* smem_kv_scales = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim + kNumHeads * sizeof(float));
    const int lane_page_delta = lane_idx / BLOCK_KV;
    const int lane_row_in_page = lane_idx - lane_page_delta * BLOCK_KV;
    if constexpr (kNumHeads < Q_TILE) {
        const bool valid_page = full_warp || warp_kv_idx + lane_page_delta < num_kv;
        if (valid_page) {
            const int current_block_idx = block_table[block_table_offset + lane_page_delta];
            auto gKv_scales = kv_block_scales + current_block_idx * kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
        } else {
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = 0.0f;
        }
    } else {
        if (full_warp) {
            auto gKv_scales = kv_block_scales + kv_block_idx[lane_page_delta] * kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
        } else if (lane_page_delta < PAGES_PER_WARP && kv_block_idx[lane_page_delta] != -1) {
            auto gKv_scales = kv_block_scales + kv_block_idx[lane_page_delta] * kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
        } else {
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = 0.0f;
        }
    }

    constexpr int heads_cnt = Q_ITER;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    i8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();

    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
#pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
#pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            if constexpr (kNumHeads < Q_TILE) {
                reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, i8x16_t>(
                    smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
            } else {
                reg_q_operand[i][j] = *reinterpret_cast<i8x16_t*>(&smem_kv_block[offset]);
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

    int32x4_t acc[Q_ITER][KV_ITER];
#pragma unroll
    for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
#pragma unroll
        for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
            acc[q_iter][kv_iter] = {0, 0, 0, 0};
        }
    }
    float sum_result[KV_ITER] = {0.f};
    i8x16_t reg_kv_operand[K_BLOCK_MAX];

    const auto& load_kv_tile_full = [&](const uint32_t& load_kv_iter, const uint32_t& smem_stage) {
        auto smem_stage_start = smem_warp_kv_start + smem_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (int subpage = 0; subpage < SUBPAGES_PER_TILE; ++subpage) {
            const int page_delta = load_kv_iter * SUBPAGES_PER_TILE + subpage;
            const int current_block_idx = kv_block_idx[page_delta];
            auto gKv = kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4);

            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
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
                int8x8_t fetched = *reinterpret_cast<const int8x8_t*>(gKv + page_linear);
                *reinterpret_cast<int8x8_t*>(smem_ptr) = fetched;
#endif
            } else {
#if defined(__gfx936__) || defined(__gfx938__)
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x4(src, lds_ptr, page_linear);
#else
                i8x16_t fetched = *reinterpret_cast<const i8x16_t*>(gKv + page_linear);
                *reinterpret_cast<i8x16_t*>(smem_ptr) = fetched;
#endif
            }
        }
    };

    const auto& load_kv_tile_tail = [&](const uint32_t& load_kv_iter, const uint32_t& smem_stage) {
        auto smem_stage_start = smem_warp_kv_start + smem_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (int subpage = 0; subpage < SUBPAGES_PER_TILE; ++subpage) {
            const int page_delta = load_kv_iter * SUBPAGES_PER_TILE + subpage;
            const int current_block_idx = kv_block_idx[page_delta];
            auto gKv =
                current_block_idx != -1 ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4) : kv_block;

            long glob_kv_desc[2];
            glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
            glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
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
                int8x8_t fetched = {0};
                if (current_block_idx != -1) {
                    fetched = *reinterpret_cast<const int8x8_t*>(gKv + page_linear);
                }
                *reinterpret_cast<int8x8_t*>(smem_ptr) = fetched;
#endif
            } else {
#if defined(__gfx936__) || defined(__gfx938__)
                int inner_warp_offset = current_block_idx != -1 ? page_linear : -1;
                auto lds_ptr = smem_stage_start + subpage * BLOCK_KV * kHeadDim;
                buffer_load_lds_x4(src, lds_ptr, inner_warp_offset);
#else
                i8x16_t fetched;
                fetched.data = {0};
                if (current_block_idx != -1) {
                    fetched = *reinterpret_cast<const i8x16_t*>(gKv + page_linear);
                }
                *reinterpret_cast<i8x16_t*>(smem_ptr) = fetched;
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
            asm volatile("s_waitcnt vmcnt(%0)\n\t" ::"I"(SUBPAGES_PER_TILE));
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
                builtin_i8_mmac(reg_kv_operand[k_block].front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_i8_mmac(reg_kv_operand[k_block].rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const int32_t& value) {
                    return fmaxf(static_cast<float>(value), 0.0f) * reg_weights[j * 4 + k];
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

}
}
