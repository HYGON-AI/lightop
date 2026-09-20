#include "sparse_page_mqa.h"
#include "paged_mqa_logits_grouped_native_fp8_impl.hpp"
#include "paged_mqa_logits_sparse_impl.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace lightop {
namespace deep_gemm {

namespace {

constexpr int kHeadDim = 128;
constexpr int kPageSize = 64;
constexpr int kTokenTile = 16;
constexpr int kTokenWarps = 4;
constexpr int kTokensPerCta = 256;

struct alignas(16) GroupedBFloat16x8 {
    __hip_bfloat16 value[8];
};

inline __device__ __forceinline__ f16x8_t
dequantize_int8x8_to_bf16(const intx2& raw, const float scale) {
    const auto* values = reinterpret_cast<const int8_t*>(&raw);
    GroupedBFloat16x8 converted{};
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        converted.value[item] = __float2bfloat16(
            static_cast<float>(values[item]) * scale);
    }
    return *reinterpret_cast<const f16x8_t*>(&converted);
}

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

// One 1024-thread CTA computes one independent row (kGroupedRows=1) or one MTP
// request group. Up to four rows run in parallel; for MTP4 row slot 0 evaluates
// the fifth row after its primary row. Grouped rows keep independent
// Q/weights/lengths while sharing each paged KV tile.
template <int kGroupedRows, int kHeads, bool kFp8Kv, bool kInt8Kv>
__attribute__((amdgpu_flat_work_group_size(1024, 1024)))
__global__ void paged_mqa_logits_grouped_sparse_kernel(
    const at::BFloat16* q,
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
    static_assert(
        kGroupedRows == 1 || (kGroupedRows >= 3 && kGroupedRows <= 6));
    static_assert(
        kHeads == 8 || kHeads == 16 || kHeads == 32 || kHeads == 64);
    static_assert(!(kFp8Kv && kInt8Kv));
    constexpr int kPrimaryRows = kGroupedRows < 4 ? kGroupedRows : 4;
    constexpr bool kHasSecondary = kGroupedRows >= 5;
    constexpr int kSecondaryRow = 4;
    constexpr int kTileK = 32;
    constexpr int kTileQ = 16;
    constexpr int kQIterations = (kHeads + kTileQ - 1) / kTileQ;
    constexpr int kPaddedHeads = kQIterations * kTileQ;
    constexpr int kKBlocks = kHeadDim / kTileK;
    constexpr int kKvIterations = kPageSize / kTokenTile;
    constexpr int kAlignmentQ = 16 / sizeof(at::BFloat16);
    constexpr int kQElements = kPaddedHeads * kHeadDim;
    constexpr int kQSourceElements = kHeads * kHeadDim;
    constexpr int kThreadsPerRow = kTokenWarps * WARP_SIZE_GPU;
    constexpr int kThreads = kPrimaryRows * kThreadsPerRow;
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    __shared__ int tile_nonzero[2][kTokenWarps];

    const int tid = threadIdx.x;
    const int wave = tid / WARP_SIZE_GPU;
    const int lane = tid % WARP_SIZE_GPU;
    const int row_slot = wave / kTokenWarps;
    const bool row_active = row_slot < kPrimaryRows;
    const int token_warp = wave % kTokenWarps;
    const int request = static_cast<int>(blockIdx.x) /
                        persistent_ctas_per_request;
    const int cta_seed = static_cast<int>(blockIdx.x) -
                         request * persistent_ctas_per_request;
    const int group_first_row = request * kGroupedRows;
    const int q_idx = group_first_row + (row_active ? row_slot : 0);
    const int secondary_row =
        kSecondaryRow + ((kGroupedRows == 6 && row_slot == 1) ? 1 : 0);
    const int secondary_q_idx =
        group_first_row + (kHasSecondary ? secondary_row : 0);
    const int logits_limit = static_cast<int>(logits_stride);

    int row_lengths[kGroupedRows];
    int group_max_length = 0;
    #pragma unroll
    for (int row = 0; row < kGroupedRows; ++row) {
        const int raw_length = context_lens[group_first_row + row];
        const int length =
            raw_length < 0
                ? 0
                : (raw_length > logits_limit ? logits_limit : raw_length);
        row_lengths[row] = length;
        group_max_length =
            length > group_max_length ? length : group_max_length;
    }
    const int context_len = row_active ? row_lengths[row_slot] : 0;
    const int secondary_context_len =
        kHasSecondary ? row_lengths[secondary_row] : 0;
    const int num_compute_ctas = ceil_div(group_max_length, kTokensPerCta);

    constexpr int kKvBufferElements =
        kTokenWarps * kTokenTile * kHeadDim;
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_bytes[];
    auto* smem_kv = reinterpret_cast<at::BFloat16*>(smem_bytes);
    auto* smem_secondary_q = smem_kv + 2 * kKvBufferElements;
    auto* smem_secondary_weights = reinterpret_cast<float*>(
        smem_secondary_q + kQElements);
    auto* smem_kv_scales = smem_secondary_weights + kPaddedHeads;

    // Keep exactly one row's Q in LDS at a time.  Its four token waves move
    // their fragments to registers before the next row overwrites the buffer.
    f16x8_t reg_q[kQIterations][kKBlocks];
    const int base_q_lane =
        lane % kTileQ * kHeadDim + lane / kTileQ * kAlignmentQ;
    #pragma unroll
    for (int load_row = 0; load_row < kPrimaryRows; ++load_row) {
        const auto* row_q =
            q + (group_first_row + load_row) * kQSourceElements;
        for (int pos = tid * kAlignmentQ; pos < kQElements;
             pos += kThreads * kAlignmentQ) {
            f16x8_t value{};
            if (pos + kAlignmentQ <= kQSourceElements) {
                value = *reinterpret_cast<const f16x8_t*>(&row_q[pos]);
            }
            *reinterpret_cast<f16x8_t*>(&smem_kv[pos]) = value;
        }
        __syncthreads();
        if (row_slot == load_row) {
            #pragma unroll
            for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                #pragma unroll
                for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                    const int offset = base_q_lane +
                                       q_iter * kTileQ * kHeadDim +
                                       k_block * kTileK;
                    reg_q[q_iter][k_block] =
                        *reinterpret_cast<f16x8_t*>(&smem_kv[offset]);
                }
            }
        }
        __syncthreads();
    }

    // Row 4 remains in LDS. Row slot 0 reloads one fragment at a time after
    // finishing row 0, avoiding a second full set of cached-Q VGPRs.
    const auto* secondary_q =
        q + (group_first_row + (kHasSecondary ? kSecondaryRow : 0)) *
                kQSourceElements;
    for (int pos = tid * kAlignmentQ; pos < kQElements;
         pos += kThreads * kAlignmentQ) {
        f16x8_t value{};
        if constexpr (kHasSecondary) {
            if (pos + kAlignmentQ <= kQSourceElements) {
                value = *reinterpret_cast<const f16x8_t*>(&secondary_q[pos]);
            }
        }
        *reinterpret_cast<f16x8_t*>(&smem_secondary_q[pos]) = value;
    }
    if (tid < kPaddedHeads) {
        smem_secondary_weights[tid] =
            kHasSecondary && tid < kHeads
            ? weights[(group_first_row + kSecondaryRow) * kHeads + tid]
            : 0.0f;
    }
    __syncthreads();

    const auto load_secondary_q = [&](int offset) {
        if constexpr (kGroupedRows == 6) {
            if (row_slot == 1) {
                f16x8_t value{};
                if (offset + kAlignmentQ <= kQSourceElements) {
                    value = *reinterpret_cast<const f16x8_t*>(
                        q + secondary_q_idx * kQSourceElements + offset);
                }
                return value;
            }
        }
        return *reinterpret_cast<const f16x8_t*>(smem_secondary_q + offset);
    };
    const auto load_secondary_weight = [&](int head) {
        if constexpr (kGroupedRows == 6) {
            if (row_slot == 1) {
                return head < kHeads
                    ? weights[secondary_q_idx * kHeads + head] : 0.0f;
            }
        }
        return smem_secondary_weights[head];
    };

    float reg_weights[kQIterations * 4];
    const auto* row_weights = weights + q_idx * kHeads;
    #pragma unroll
    for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int head =
                q_iter * kTileQ + item * 4 + lane / kTileQ;
            reg_weights[q_iter * 4 + item] =
                row_active && head < kHeads ? row_weights[head] : 0.0f;
        }
    }
    bool lane_q_all_finite = true;
    bool secondary_lane_q_all_finite = true;
    if (row_active) {
        #pragma unroll
        for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                lane_q_all_finite &= bf16x8_all_finite(
                    reg_q[q_iter][k_block]);
            }
        }
    }
    if constexpr (kHasSecondary) {
        #pragma unroll
        for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
            const int q_offset = base_q_lane +
                                 q_iter * kTileQ * kHeadDim +
                                 k_block * kTileK;
            const f16x8_t secondary_q_fragment =
                load_secondary_q(q_offset);
            secondary_lane_q_all_finite &=
                bf16x8_all_finite(secondary_q_fragment);
            }
        }
    }
    bool lane_weights_all_finite = true;
    #pragma unroll
    for (int item = 0; item < kQIterations * 4; ++item) {
        lane_weights_all_finite &=
            (__float_as_uint(reg_weights[item]) & 0x7f800000u) !=
            0x7f800000u;
    }
    const bool primary_inputs_all_finite =
        __ballot(lane_q_all_finite && lane_weights_all_finite) ==
        0xffffffffffffffffull;
    bool secondary_lane_weight_finite = true;
    if constexpr (kHasSecondary) {
        if (lane < kHeads) {
            secondary_lane_weight_finite =
                (__float_as_uint(load_secondary_weight(lane)) &
                 0x7f800000u) != 0x7f800000u;
        }
    }
    const bool secondary_inputs_all_finite =
        __ballot(
            secondary_lane_q_all_finite && secondary_lane_weight_finite) ==
        0xffffffffffffffffull;
    const uint64_t logits_offset = q_idx * logits_stride;
    const uint64_t secondary_logits_offset =
        secondary_q_idx * logits_stride;
    const uint64_t mask_stride = (logits_stride + kTokenTile - 1) /
                                 kTokenTile;
    const uint64_t mask_offset = q_idx * mask_stride;
    const uint64_t secondary_mask_offset =
        secondary_q_idx * mask_stride;
    const uint64_t shared_block_table_offset =
        group_first_row * block_table_stride;

    // The first 32 KiB of LDS is a double-buffered KV staging area. While all
    // primary rows and row 4 consume one tile, row slot 0 fetches the next.

    const auto prefetch_group_tile = [&](const int tile_token_offset,
                                         const int buffer_slot) {
        const bool group_tile_valid = tile_token_offset < group_max_length;
        const bool owns_prefetch = kInt8Kv || row_slot == 0;
        if (owns_prefetch && group_tile_valid) {
            const int logical_page = tile_token_offset / kPageSize;
            const int physical_page = block_table[
                shared_block_table_offset + logical_page];
            const auto* page = kv_block +
                static_cast<int64_t>(physical_page) * kv_cache_stride_bytes;
            auto* smem_warp_kv =
                smem_kv + buffer_slot * kKvBufferElements +
                token_warp * kTokenTile * kHeadDim;
            const int row_in_page = tile_token_offset % kPageSize;
            if constexpr (kFp8Kv) {
                constexpr int kFp8Alignment = 8;
                constexpr int kFp8Elements = kTokenTile * kHeadDim;
                #pragma unroll
                for (int pos = lane * kFp8Alignment;
                     pos < kFp8Elements;
                     pos += WARP_SIZE_GPU * kFp8Alignment) {
                    const intx2 raw = *reinterpret_cast<const intx2*>(
                        page + row_in_page * kHeadDim + pos);
                    *reinterpret_cast<f16x8_t*>(smem_warp_kv + pos) =
                        convert_fp8x8_e4m3fn_to_bf16_flush_subnormal(raw);
                }
                if (lane < kTokenTile) {
                    const auto* page_scales = reinterpret_cast<const float*>(
                        page + kPageSize * kHeadDim);
                    smem_kv_scales[
                        (buffer_slot * kTokenWarps + token_warp) *
                            kTokenTile + lane] =
                        page_scales[row_in_page + lane];
                }
            } else if constexpr (kInt8Kv) {
                const auto* page_scales = reinterpret_cast<const float*>(
                    page + kPageSize * kHeadDim);
                auto* tile_scales = smem_kv_scales +
                    (buffer_slot * kTokenWarps + token_warp) * kTokenTile;
                if (row_slot == 0 && lane < kTokenTile) {
                    tile_scales[lane] = page_scales[row_in_page + lane];
                }
                // All four row slots cooperate on the one shared KV tile.
                // Each of the 16 waves owns four tokens (one aligned int8x8
                // vector per lane), matching the standalone kernel's
                // per-wave conversion width without materializing a global
                // BF16 workspace.
                constexpr int kInt8Alignment = 8;
                const int vector_slot = row_slot * WARP_SIZE_GPU + lane;
                const int pos = vector_slot * kInt8Alignment;
                const int token_in_tile = pos / kHeadDim;
                float scale = (lane % 16) == 0
                    ? page_scales[row_in_page + token_in_tile]
                    : 0.0f;
                scale = __shfl(scale, 0, 16);
                const intx2 raw = *reinterpret_cast<const intx2*>(
                    page + row_in_page * kHeadDim + pos);
                *reinterpret_cast<f16x8_t*>(smem_warp_kv + pos) =
                    dequantize_int8x8_to_bf16(raw, scale);
            } else {
                buffer_load_page_tile_lds<
                    at::BFloat16, kHeadDim, kTokenTile>(
                    reinterpret_cast<const at::BFloat16*>(page),
                    row_in_page,
                    lane,
                    smem_warp_kv);
            }
            return true;
        }
        return false;
    };

    const auto classify_buffer_tile = [&](const int buffer_slot,
                                           const bool tile_loaded) {
        if (row_slot != 0) {
            return;
        }
        if (!tile_loaded) {
            if (lane == 0) {
                tile_nonzero[buffer_slot][token_warp] = 0;
            }
            return;
        }
        const int base_k_lane =
            lane % kTokenTile * kHeadDim +
            lane / kTokenTile * kAlignmentQ;
        const auto* smem_warp_kv =
            smem_kv + buffer_slot * kKvBufferElements +
            token_warp * kTokenTile * kHeadDim;
        uint32_t lane_bits = 0u;
        #pragma unroll
        for (int k_block = 0; k_block < kKBlocks; ++k_block) {
            const v4i raw = *reinterpret_cast<const v4i*>(
                smem_warp_kv + base_k_lane + k_block * kTileK);
            lane_bits |= static_cast<uint32_t>(raw[0]);
            lane_bits |= static_cast<uint32_t>(raw[1]);
            lane_bits |= static_cast<uint32_t>(raw[2]);
            lane_bits |= static_cast<uint32_t>(raw[3]);
        }
        const uint64_t nonzero_lanes = __ballot(lane_bits != 0u);
        bool scale_finite = true;
        if constexpr (kFp8Kv || kInt8Kv) {
            if (lane < kTokenTile) {
                const float scale = smem_kv_scales[
                    (buffer_slot * kTokenWarps + token_warp) *
                        kTokenTile + lane];
                scale_finite =
                    (__float_as_uint(scale) & 0x7f800000u) != 0x7f800000u;
            }
        }
        const bool scales_all_finite =
            __ballot(scale_finite) == 0xffffffffffffffffull;
        if (lane == 0) {
            tile_nonzero[buffer_slot][token_warp] =
                nonzero_lanes != 0u || !scales_all_finite;
        }
    };

    for (int cta_idx = cta_seed; cta_idx < num_compute_ctas;
         cta_idx += persistent_ctas_per_request) {
        const int cta_token_offset = cta_idx * kTokensPerCta;
        bool tile_loaded[2] = {false, false};

        tile_loaded[0] = prefetch_group_tile(
            cta_token_offset + token_warp * kTokenTile, 0);
        if constexpr (kInt8Kv) {
            __syncthreads();
        }
        if (row_slot == 0) {
            #if defined(__gfx936__) || defined(__gfx938__)
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)\n\t");
            __builtin_amdgcn_sched_barrier(0);
            #endif
            classify_buffer_tile(0, tile_loaded[0]);
        }
        __syncthreads();

        #pragma unroll
        for (int kv_iter = 0; kv_iter < kKvIterations; ++kv_iter) {
            const int cta_tile = kv_iter * kTokenWarps + token_warp;
            const int tile_token_offset =
                cta_token_offset + cta_tile * kTokenTile;
            const int buffer_slot = kv_iter & 1;
            auto* smem_warp_kv =
                smem_kv + buffer_slot * kKvBufferElements +
                token_warp * kTokenTile * kHeadDim;

            const bool has_next_tile = kv_iter + 1 < kKvIterations;
            if (has_next_tile) {
                const int next_tile_token_offset =
                    cta_token_offset +
                    ((kv_iter + 1) * kTokenWarps + token_warp) * kTokenTile;
                tile_loaded[buffer_slot ^ 1] = prefetch_group_tile(
                    next_tile_token_offset, buffer_slot ^ 1);
            }

            const bool row_tile_valid =
                row_active && tile_token_offset < context_len;
            const bool kv_tile_nonzero =
                tile_nonzero[buffer_slot][token_warp] != 0;
            const bool compute_primary =
                row_tile_valid &&
                (kv_tile_nonzero || !primary_inputs_all_finite);
            fp32x4 accumulators[kQIterations];
            if (compute_primary) {
                CLEAR_ACC(accumulators, kQIterations);
                const int base_k_lane =
                    lane % kTokenTile * kHeadDim +
                    lane / kTokenTile * kAlignmentQ;
                #pragma unroll
                for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                    const f16x8_t reg_kv =
                        *reinterpret_cast<f16x8_t*>(
                            smem_warp_kv + base_k_lane +
                            k_block * kTileK);
                    #pragma unroll
                    for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                        builtin_b16_mmac<false>(
                            reg_kv.front,
                            reg_q[q_iter][k_block].front,
                            accumulators[q_iter]);
                        builtin_b16_mmac<false>(
                            reg_kv.rear,
                            reg_q[q_iter][k_block].rear,
                            accumulators[q_iter]);
                    }
                }
            }

            float sum = 0.0f;
            if (compute_primary) {
                #pragma unroll
                for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                    #pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        sum += fmaxf(accumulators[q_iter][item], 0.0f) *
                               reg_weights[q_iter * 4 + item];
                    }
                }
                #pragma unroll
                for (uint32_t delta = 16; delta <= 32; delta <<= 1) {
                    sum += __shfl_down_sync(
                        0xffffffffffffffff, sum, delta);
                }
                if constexpr (kFp8Kv) {
                    sum *= smem_kv_scales[
                        (buffer_slot * kTokenWarps + token_warp) *
                            kTokenTile + lane % kTokenTile];
                }
            }

            const int token = tile_token_offset + lane;
            const bool in_range = lane < kTokenTile && token < context_len;
            const auto nonzero_bits = __ballot(
                in_range && __float_as_uint(sum) != 0u);
            if (row_active && lane == 0 &&
                tile_token_offset < logits_limit) {
                sparse_mask[mask_offset + tile_token_offset / kTokenTile] =
                    static_cast<int16_t>(nonzero_bits & 0xffffu);
            }
            if (in_range && __float_as_uint(sum) != 0u) {
                logits[logits_offset + token] = sum;
            }

            if constexpr (kHasSecondary) {
              if (row_slot < kGroupedRows - kPrimaryRows) {
                const bool secondary_tile_valid =
                    tile_token_offset < secondary_context_len;
                const bool compute_secondary =
                    secondary_tile_valid &&
                    (kv_tile_nonzero || !secondary_inputs_all_finite);
                if (compute_secondary) {
                    CLEAR_ACC(accumulators, kQIterations);
                    const int base_k_lane =
                        lane % kTokenTile * kHeadDim +
                        lane / kTokenTile * kAlignmentQ;
                    #pragma unroll
                    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                        const f16x8_t reg_kv =
                            *reinterpret_cast<f16x8_t*>(
                                smem_warp_kv + base_k_lane +
                                k_block * kTileK);
                        #pragma unroll
                        for (int q_iter = 0; q_iter < kQIterations;
                             ++q_iter) {
                            const int q_offset =
                                base_q_lane +
                                q_iter * kTileQ * kHeadDim +
                                k_block * kTileK;
                            const f16x8_t reg_secondary_q =
                                load_secondary_q(q_offset);
                            builtin_b16_mmac<false>(
                                reg_kv.front,
                                reg_secondary_q.front,
                                accumulators[q_iter]);
                            builtin_b16_mmac<false>(
                                reg_kv.rear,
                                reg_secondary_q.rear,
                                accumulators[q_iter]);
                        }
                    }
                }

                float secondary_sum = 0.0f;
                if (compute_secondary) {
                    #pragma unroll
                    for (int q_iter = 0; q_iter < kQIterations;
                         ++q_iter) {
                        #pragma unroll
                        for (int item = 0; item < 4; ++item) {
                            const int head =
                                q_iter * kTileQ + item * 4 +
                                lane / kTileQ;
                            secondary_sum +=
                                fmaxf(accumulators[q_iter][item], 0.0f) *
                                load_secondary_weight(head);
                        }
                    }
                    #pragma unroll
                    for (uint32_t delta = 16; delta <= 32;
                         delta <<= 1) {
                        secondary_sum += __shfl_down_sync(
                            0xffffffffffffffff, secondary_sum, delta);
                    }
                    if constexpr (kFp8Kv) {
                        secondary_sum *= smem_kv_scales[
                            (buffer_slot * kTokenWarps + token_warp) *
                                kTokenTile + lane % kTokenTile];
                    }
                }

                const bool secondary_in_range =
                    lane < kTokenTile && token < secondary_context_len;
                const auto secondary_nonzero_bits = __ballot(
                    secondary_in_range &&
                    __float_as_uint(secondary_sum) != 0u);
                if (lane == 0 && tile_token_offset < logits_limit) {
                    sparse_mask[
                        secondary_mask_offset +
                        tile_token_offset / kTokenTile] =
                        static_cast<int16_t>(
                            secondary_nonzero_bits & 0xffffu);
                }
                if (secondary_in_range &&
                    __float_as_uint(secondary_sum) != 0u) {
                    logits[secondary_logits_offset + token] =
                        secondary_sum;
                }
              }
            }

            if (has_next_tile) {
                if constexpr (kInt8Kv) {
                    __syncthreads();
                }
                if (row_slot == 0) {
                    #if defined(__gfx936__) || defined(__gfx938__)
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt vmcnt(0)\n\t");
                    __builtin_amdgcn_sched_barrier(0);
                    #endif
                    classify_buffer_tile(
                        buffer_slot ^ 1,
                        tile_loaded[buffer_slot ^ 1]);
                }
                __syncthreads();
            }
        }
    }
}

template <int kGroupedRows, int kHeads, bool kFp8Kv, bool kInt8Kv>
void launch_grouped_kernel(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    int persistent_ctas,
    cudaStream_t stream) {
    constexpr int kThreads = 4 * kTokenWarps * WARP_SIZE_GPU;
    constexpr int kPaddedHeads = ((kHeads + 15) / 16) * 16;
    constexpr int kQElements = kPaddedHeads * kHeadDim;
    constexpr int kKvBufferElements =
        kTokenWarps * kTokenTile * kHeadDim;
    constexpr int kSmemBytes =
        (2 * kKvBufferElements + kQElements) *
            sizeof(at::BFloat16) +
        kPaddedHeads * sizeof(float) +
        ((kFp8Kv || kInt8Kv)
             ? 2 * kTokenWarps * kTokenTile * sizeof(float)
             : 0);
    const auto logits_stride = static_cast<uint64_t>(logits.stride(0));
    const auto kv_cache_stride_bytes =
        static_cast<uint64_t>(kv_cache.stride(0) * kv_cache.element_size());
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));
    const int requests = static_cast<int>(q.size(0)) / kGroupedRows;
    const dim3 grid(persistent_ctas * requests, 1, 1);
    const dim3 block(kThreads, 1, 1);

    paged_mqa_logits_grouped_sparse_kernel<
        kGroupedRows, kHeads, kFp8Kv, kInt8Kv><<<
        grid, block, kSmemBytes, stream>>>(
        q.data_ptr<at::BFloat16>(),
        reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
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
    TORCH_CHECK(
        error == hipSuccess,
        "grouped sparse MQA launch failed: ",
        hipGetErrorString(error));
}

template <int kGroupedRows, bool kFp8Kv, bool kInt8Kv>
void dispatch_grouped_heads(
    int num_heads,
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    int persistent_ctas,
    cudaStream_t stream) {
    switch (num_heads) {
        case 8:
            launch_grouped_kernel<kGroupedRows, 8, kFp8Kv, kInt8Kv>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 16:
            launch_grouped_kernel<kGroupedRows, 16, kFp8Kv, kInt8Kv>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 32:
            launch_grouped_kernel<kGroupedRows, 32, kFp8Kv, kInt8Kv>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 64:
            launch_grouped_kernel<kGroupedRows, 64, kFp8Kv, kInt8Kv>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        default:
            TORCH_CHECK(false, "grouped sparse MQA supports 8/16/32/64 heads");
    }
}

template <bool kFp8Kv, bool kInt8Kv>
void dispatch_grouped_rows(
    int group_size,
    int num_heads,
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    int persistent_ctas,
    cudaStream_t stream) {
    switch (group_size) {
        case 1:
            dispatch_grouped_heads<1, kFp8Kv, kInt8Kv>(
                num_heads, q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 3:
            dispatch_grouped_heads<3, kFp8Kv, kInt8Kv>(
                num_heads, q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 4:
            dispatch_grouped_heads<4, kFp8Kv, kInt8Kv>(
                num_heads, q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 5:
            dispatch_grouped_heads<5, kFp8Kv, kInt8Kv>(
                num_heads, q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        case 6:
            dispatch_grouped_heads<6, kFp8Kv, kInt8Kv>(
                num_heads, q, kv_cache, weights, context_lens, block_table,
                logits, nonzero_mask, persistent_ctas, stream);
            return;
        default:
            TORCH_CHECK(false, "sparse MQA supports group_size 1/3/4/5/6");
    }
}

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_impl(
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
    const int& group_size,
    const bool int8_kv) {
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    check_grouped_same_device(q, fused_kv_cache, "fused_kv_cache");
    check_grouped_same_device(q, weights, "weights");
    check_grouped_same_device(q, context_lens, "context_lens");
    check_grouped_same_device(q, block_table, "block_table");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    TORCH_CHECK(
        !schedule_meta.has_value(),
        "grouped-wave sparse MQA requires schedule_meta=None");
    TORCH_CHECK(
        clean_logits,
        "grouped-wave sparse MQA requires clean_logits=True");

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
        "grouped-wave sparse MQA requires gfx936/80CU or gfx938/64CU/72CU, got ",
        properties.gcnArchName,
        "/",
        properties.multiProcessorCount,
        "CU");

    TORCH_CHECK(
        group_size == 1 || (group_size >= 3 && group_size <= 6),
        "group_size must be 1 (independent rows) or 3/4/5/6 (MTP2/3/4/5)");
    const bool supported_batch = q.dim() == 4 && q.size(0) > 0 &&
        (group_size == 1 || q.size(0) % group_size == 0);
    const bool native_fp8_path = is_supported_gfx938 && !int8_kv &&
        q.scalar_type() == torch::kFloat8_e4m3fn &&
        fused_kv_cache.scalar_type() == torch::kUInt8 &&
        q.dim() == 4 && q.size(2) == grouped_sparse_native_fp8::kHeads;
    TORCH_CHECK(
        (q.scalar_type() == torch::kBFloat16 || native_fp8_path) && q.dim() == 4 &&
            supported_batch && q.size(1) == 1 &&
            (q.size(2) == 8 || q.size(2) == 16 || q.size(2) == 32 ||
             q.size(2) == 64) &&
            q.size(3) == kHeadDim && q.is_contiguous(),
        "q must be contiguous BF16 [batch, 1, heads, 128], or native "
        "E4M3FN [batch, 1, 32, 128] with packed FP8 KV on gfx938/64CU/72CU, with positive "
        "batch, batch divisible by group_size for group_size 3/4/5/6, and "
        "heads in 8/16/32/64");
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const bool fp8_kv =
        fused_kv_cache.scalar_type() == torch::kUInt8 && !int8_kv;
    const bool quantized_kv = fp8_kv || int8_kv;
    const int kv_width = quantized_kv
        ? kHeadDim + static_cast<int>(sizeof(float))
        : kHeadDim;
    TORCH_CHECK(
        ((quantized_kv &&
          fused_kv_cache.scalar_type() == torch::kUInt8) ||
         (!quantized_kv &&
          fused_kv_cache.scalar_type() == torch::kBFloat16)) &&
            fused_kv_cache.dim() == 4 && fused_kv_cache.size(0) > 0 &&
            fused_kv_cache.size(1) == kPageSize &&
            fused_kv_cache.size(2) == 1 &&
            fused_kv_cache.size(3) == kv_width &&
            fused_kv_cache.stride(1) == kv_width &&
            fused_kv_cache.stride(2) == kv_width &&
            fused_kv_cache.stride(3) == 1,
        "fused_kv_cache must be packed BF16 [blocks,64,1,128] or "
        "packed FP8/INT8+scale uint8 [blocks,64,1,132]");
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
            max_context_len == block_table.size(1) * kPageSize,
        "max_context_len must equal block_table capacity");
    TORCH_CHECK(num_warps == 4, "grouped-wave sparse MQA requires 4 warps per row");
    TORCH_CHECK(
        persistent_ctas >= 0 && persistent_ctas <= 4096,
        "persistent_ctas must be 0 (auto) or in [1, 4096]");

    const int aligned_context_len =
        (max_context_len + kPageSize - 1) / kPageSize * kPageSize;
    auto logits_storage = torch::empty(
        {batch_size, aligned_context_len}, q.options().dtype(torch::kFloat));
    auto logits = logits_storage.slice(-1, 0, max_context_len);
    auto nonzero_mask = torch::zeros(
        {batch_size, (max_context_len + kTokenTile - 1) / kTokenTile},
        q.options().dtype(torch::kInt16));

    const auto max_chunks =
        (static_cast<int64_t>(max_context_len) + kTokensPerCta - 1) /
        kTokensPerCta;
    const int resolved = persistent_ctas == 0
        ? static_cast<int>(std::min<int64_t>(max_chunks, 32))
        : static_cast<int>(std::min<int64_t>(persistent_ctas, max_chunks));
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    if (native_fp8_path) {
        grouped_sparse_native_fp8::dispatch_grouped_rows(
            group_size, q, fused_kv_cache, weights, context_lens, block_table,
            logits, nonzero_mask, resolved, stream);
    } else if (int8_kv) {
        dispatch_grouped_rows<false, true>(
            group_size, num_heads, q, fused_kv_cache, weights, context_lens,
            block_table, logits, nonzero_mask, resolved, stream);
    } else if (fp8_kv) {
        dispatch_grouped_rows<true, false>(
            group_size, num_heads, q, fused_kv_cache, weights, context_lens,
            block_table, logits, nonzero_mask, resolved, stream);
    } else {
        dispatch_grouped_rows<false, false>(
            group_size, num_heads, q, fused_kv_cache, weights, context_lens,
            block_table, logits, nonzero_mask, resolved, stream);
    }
    return {logits, nonzero_mask};
}

}  // namespace

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped(
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
    return page_mqa_logits_sparse_mask_grouped_impl(
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
        group_size,
        false);
}

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_int8(
    const torch::Tensor& q,
    const torch::Tensor& packed_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    std::optional<torch::Tensor>& schedule_meta,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size) {
    return page_mqa_logits_sparse_mask_grouped_impl(
        q,
        packed_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas,
        group_size,
        true);
}

}  // namespace deep_gemm
}  // namespace lightop
