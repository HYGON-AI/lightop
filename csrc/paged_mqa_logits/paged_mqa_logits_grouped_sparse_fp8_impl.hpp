#pragma once

#include "paged_mqa_logits_impl.hpp"

// The grouped BF16 kernel intentionally stays in
// paged_mqa_logits_grouped_sparse.cu.  This is a separate gfx938-only FP8
// specialization: keeping the operand type, LDS layout and MMAC instruction
// stream independent prevents the BF16 fallback from silently becoming the
// implementation of the FP8 fast path.
namespace lightop {
namespace deep_gemm {
namespace grouped_sparse_fp8 {

constexpr int kHeadDim = 128;
constexpr int kPageSize = 64;
constexpr int kTokenTile = 16;
constexpr int kTokenWarps = 4;
constexpr int kTokensPerCta = 256;
constexpr int kHeads = 32;
constexpr int kGroup4TokensPerCta = kTokenWarps * kPageSize;

inline __device__ __forceinline__ bool fp8x16_all_finite_e4m3fn(
    const f8x16_t& value) {
    // OCP E4M3FN reserves exponent=0xf/mantissa=0x7 (both signs) for NaN.
    // All other bit patterns, including signed zero and subnormals, are
    // finite operands.  Keep this test raw so the zero-tile fast path does
    // not hide NaNs by converting FP8 through BF16 first.
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if ((bytes[i] & 0x7fu) == 0x7fu) {
            return false;
        }
    }
    return true;
}

// The generic grouped kernel below mirrors the gfx936 BF16 implementation:
// four rows occupy four independent four-wave partitions in one 1024-thread
// CTA.  That layout gives the rows a shared KV tile, but it leaves twelve
// waves waiting while the first partition performs each direct-to-LDS load
// and permits only one CTA to reside on a CU.
//
// GLM Target Verify always has four rows per request.  Fuse its pairwise KV
// reuse with the proven 5c61f24 persistent FP8 pipeline: every wave owns one
// 64-token page, double-buffers its four 16-token tiles, and evaluates two Q
// rows with interleaved MMACs while each KV fragment is resident in VGPRs.
// Two adjacent CTAs cover the two row pairs.  Q/weight staging aliases the KV
// buffers after a CTA barrier, keeping the persistent LDS footprint at 16 KiB.
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void paged_mqa_logits_group4_sparse_fp8_four_wave_kernel(
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
    constexpr int kGroupedRows = 4;
    constexpr int kRowsPerCta = 2;
    constexpr int kPairsPerRequest = kGroupedRows / kRowsPerCta;
    constexpr int kTileK = 64;
    constexpr int kTileQ = 16;
    constexpr int kQIterations = kHeads / kTileQ;
    constexpr int kKBlocks = kHeadDim / kTileK;
    constexpr int kAlignment = 16;
    constexpr int kQElements = kHeads * kHeadDim;
    constexpr int kGroupQBytes = kRowsPerCta * kQElements;
    constexpr int kGroupWeightElements = kRowsPerCta * kHeads;
    constexpr int kKvStageBytes =
        kTokenWarps * kTokenTile * kHeadDim;

    const int tid = threadIdx.x;
    const int token_warp = tid / WARP_SIZE_GPU;
    const int lane = tid % WARP_SIZE_GPU;
    const int blocks_per_request =
        persistent_ctas_per_request * kPairsPerRequest;
    const int request = static_cast<int>(blockIdx.x) /
                        blocks_per_request;
    const int request_block = static_cast<int>(blockIdx.x) -
                              request * blocks_per_request;
    const int cta_seed = request_block / kPairsPerRequest;
    const int pair = request_block - cta_seed * kPairsPerRequest;
    const int group_first_row = request * kGroupedRows;
    const int pair_first_row = group_first_row + pair * kRowsPerCta;
    const int logits_limit = static_cast<int>(logits_stride);

    int row_lengths[kRowsPerCta];
    int group_max_length = 0;
    #pragma unroll
    for (int row = 0; row < kRowsPerCta; ++row) {
        const int raw_length = context_lens[pair_first_row + row];
        const int length = raw_length < 0
            ? 0
            : (raw_length > logits_limit ? logits_limit : raw_length);
        row_lengths[row] = length;
        group_max_length = length > group_max_length
            ? length
            : group_max_length;
    }
    const int num_compute_ctas =
        ceil_div(group_max_length, kGroup4TokensPerCta);

    extern __shared__ __align__(kHeadDim * 8) uint8_t smem_bytes[];
    auto* smem_q = smem_bytes;
    auto* smem_weights = reinterpret_cast<float*>(
        smem_q + kGroupQBytes);
    const auto* group_q = q + pair_first_row * kQElements;
    for (int pos = tid * kAlignment; pos < kGroupQBytes;
         pos += blockDim.x * kAlignment) {
        *reinterpret_cast<f8x16_t*>(smem_q + pos) =
            *reinterpret_cast<const f8x16_t*>(group_q + pos);
    }
    if (tid < kGroupWeightElements) {
        smem_weights[tid] =
            weights[pair_first_row * kHeads + tid];
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
                const int q_offset = row * kQElements + base_q_lane +
                    q_iter * kTileQ * kHeadDim + k_block * kTileK;
                reg_q[row][q_iter][k_block] =
                    *reinterpret_cast<const f8x16_t*>(smem_q + q_offset);
            }
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int head = q_iter * kTileQ +
                    item * 4 + lane / kTileQ;
                reg_weights[row][q_iter * 4 + item] =
                    smem_weights[row * kHeads + head];
            }
        }
    }
    __syncthreads();

    // Q and weights are live in registers from this point onward, so their
    // staging allocation can be reused as two direct-to-LDS KV buffers.
    auto* smem_kv = smem_bytes;
    const uint64_t shared_block_table_offset =
        group_first_row * block_table_stride;
    const uint64_t mask_stride =
        (logits_stride + kTokenTile - 1) / kTokenTile;

    for (int cta_idx = cta_seed; cta_idx < num_compute_ctas;
         cta_idx += persistent_ctas_per_request) {
        const int cta_token_offset =
            cta_idx * kGroup4TokensPerCta;
        const int wave_token_offset =
            cta_token_offset + token_warp * kPageSize;
        const bool wave_valid = wave_token_offset < group_max_length;
        const int logical_page = wave_valid
            ? wave_token_offset / kPageSize
            : 0;
        const int physical_page = block_table[
            shared_block_table_offset + logical_page];
        const auto* page = kv_block +
            static_cast<int64_t>(physical_page) * kv_cache_stride_bytes;
        const auto* page_scales = reinterpret_cast<const float*>(
            page + kPageSize * kHeadDim);
        float kv_scales[kPageSize / kTokenTile];
        #pragma unroll
        for (int kv_iter = 0; kv_iter < kPageSize / kTokenTile;
             ++kv_iter) {
            kv_scales[kv_iter] = page_scales[
                kv_iter * kTokenTile + lane % kTokenTile];
        }

        // Prime both LDS stages.  Each wave has a disjoint slice, so the
        // subsequent VMEM waits do not require CTA-wide synchronization.
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            auto* smem_warp_stage = smem_kv + stage * kKvStageBytes +
                token_warp * kTokenTile * kHeadDim;
            buffer_load_page_tile_lds<uint8_t, kHeadDim, kTokenTile>(
                page, stage * kTokenTile, lane, smem_warp_stage);
        }

        #pragma unroll
        for (int kv_iter = 0; kv_iter < kPageSize / kTokenTile;
             ++kv_iter) {
            const int stage = kv_iter & 1;
            auto* smem_warp_stage = smem_kv + stage * kKvStageBytes +
                token_warp * kTokenTile * kHeadDim;
            #if defined(__gfx936__) || defined(__gfx938__)
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)\n\t");
            __builtin_amdgcn_sched_barrier(0);
            #else
            __syncthreads();
            #endif

            f8x16_t reg_kv[kKBlocks];
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                reg_kv[k_block] =
                    *reinterpret_cast<const f8x16_t*>(
                        smem_warp_stage + base_k_lane + k_block * kTileK);
            }
            #if defined(__gfx936__) || defined(__gfx938__)
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n\t");
            __builtin_amdgcn_sched_barrier(0);
            #endif

            // Once the current LDS fragment is in VGPRs, refill its stage
            // with the tile two iterations ahead while MMACs execute.
            if (kv_iter + 2 < kPageSize / kTokenTile) {
                buffer_load_page_tile_lds<uint8_t, kHeadDim, kTokenTile>(
                    page,
                    (kv_iter + 2) * kTokenTile,
                    lane,
                    smem_warp_stage);
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

            float sums[kRowsPerCta] = {0.0f, 0.0f};
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

            const int tile_token_offset =
                wave_token_offset + kv_iter * kTokenTile;
            #pragma unroll
            for (int row = 0; row < kRowsPerCta; ++row) {
                const int token = tile_token_offset + lane;
                const bool in_range = lane < kTokenTile &&
                    token < row_lengths[row];
                const uint64_t nonzero_bits = __ballot(
                    in_range && __float_as_uint(sums[row]) != 0u);
                const int q_idx = pair_first_row + row;
                if (lane == 0 && tile_token_offset < logits_limit) {
                    sparse_mask[
                        q_idx * mask_stride +
                        tile_token_offset / kTokenTile] =
                        static_cast<int16_t>(nonzero_bits & 0xffffu);
                }
                if (in_range && __float_as_uint(sums[row]) != 0u) {
                    logits[q_idx * logits_stride + token] = sums[row];
                }
            }
        }
    }
}

template <int kGroupedRows>
__attribute__((amdgpu_flat_work_group_size(1024, 1024)))
__global__ void paged_mqa_logits_grouped_sparse_fp8_kernel(
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
                  (kGroupedRows >= 3 && kGroupedRows <= 5));

    constexpr int kPrimaryRows = kGroupedRows < 4 ? kGroupedRows : 4;
    constexpr bool kHasSecondary = kGroupedRows == 5;
    constexpr int kSecondaryRow = 4;
    constexpr int kTileK = 64;
    constexpr int kTileQ = 16;
    constexpr int kQIterations = kHeads / kTileQ;
    constexpr int kKBlocks = kHeadDim / kTileK;
    constexpr int kKvIterations = kPageSize / kTokenTile;
    constexpr int kAlignmentQ = 16;
    constexpr int kQElements = kHeads * kHeadDim;
    constexpr int kKvBufferBytes =
        kTokenWarps * kTokenTile * kHeadDim * sizeof(uint8_t);
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
    const int secondary_q_idx =
        group_first_row + (kHasSecondary ? kSecondaryRow : 0);
    const int logits_limit = static_cast<int>(logits_stride);

    int row_lengths[kGroupedRows];
    int group_max_length = 0;
    #pragma unroll
    for (int row = 0; row < kGroupedRows; ++row) {
        const int raw_length = context_lens[group_first_row + row];
        const int length = raw_length < 0
            ? 0
            : (raw_length > logits_limit ? logits_limit : raw_length);
        row_lengths[row] = length;
        group_max_length = length > group_max_length ? length : group_max_length;
    }
    const int context_len = row_active ? row_lengths[row_slot] : 0;
    const int secondary_context_len =
        kHasSecondary ? row_lengths[kSecondaryRow] : 0;
    const int num_compute_ctas = ceil_div(group_max_length, kTokensPerCta);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_bytes[];
    auto* smem_kv = smem_bytes;
    auto* smem_secondary_q = smem_kv + 2 * kKvBufferBytes;
    auto* smem_secondary_weights = reinterpret_cast<float*>(
        smem_secondary_q + kQElements);
    auto* smem_kv_scales = smem_secondary_weights + kHeads;

    // Primary rows cache their two FP8 K=64 fragments in registers.  All
    // rows in the MTP group later consume the same FP8 LDS KV tile.
    f8x16_t reg_q[kQIterations][kKBlocks];
    const int base_q_lane =
        lane % kTileQ * kHeadDim + lane / kTileQ * kAlignmentQ;
    #pragma unroll
    for (int load_row = 0; load_row < kPrimaryRows; ++load_row) {
        const auto* row_q = q +
            (group_first_row + load_row) * kQElements;
        for (int pos = tid * kAlignmentQ; pos < kQElements;
             pos += blockDim.x * kAlignmentQ) {
            *reinterpret_cast<f8x16_t*>(&smem_kv[pos]) =
                *reinterpret_cast<const f8x16_t*>(&row_q[pos]);
        }
        __syncthreads();
        if (row_slot == load_row) {
            #pragma unroll
            for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                #pragma unroll
                for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                    const int offset = base_q_lane +
                        q_iter * kTileQ * kHeadDim + k_block * kTileK;
                    reg_q[q_iter][k_block] =
                        *reinterpret_cast<const f8x16_t*>(&smem_kv[offset]);
                }
            }
        }
        __syncthreads();
    }

    // For MTP4, row 4 uses LDS rather than keeping another full set of Q
    // fragments in VGPRs.  Row slot 0 evaluates it after its primary row.
    const auto* secondary_q = q + secondary_q_idx * kQElements;
    for (int pos = tid * kAlignmentQ; pos < kQElements;
         pos += blockDim.x * kAlignmentQ) {
        f8x16_t value{};
        if constexpr (kHasSecondary) {
            value = *reinterpret_cast<const f8x16_t*>(&secondary_q[pos]);
        }
        *reinterpret_cast<f8x16_t*>(&smem_secondary_q[pos]) = value;
    }
    if (tid < kHeads) {
        smem_secondary_weights[tid] = kHasSecondary
            ? weights[secondary_q_idx * kHeads + tid]
            : 0.0f;
    }
    __syncthreads();

    // The quantization producer folds the per-Q FP8 scale into weights.  This
    // kernel intentionally consumes only those FP32 weights, matching the
    // dense FP8 Page-MQA math and avoiding a second Q dequantization scale.
    float reg_weights[kQIterations * 4];
    const auto* row_weights = weights + q_idx * kHeads;
    #pragma unroll
    for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int head = q_iter * kTileQ + item * 4 + lane / kTileQ;
            reg_weights[q_iter * 4 + item] = row_active
                ? row_weights[head]
                : 0.0f;
        }
    }

    bool lane_q_all_finite = true;
    if (row_active) {
        #pragma unroll
        for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                lane_q_all_finite &=
                    fp8x16_all_finite_e4m3fn(reg_q[q_iter][k_block]);
            }
        }
    }
    bool secondary_lane_q_all_finite = true;
    if constexpr (kHasSecondary) {
        #pragma unroll
        for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
            #pragma unroll
            for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                const int q_offset = base_q_lane +
                    q_iter * kTileQ * kHeadDim + k_block * kTileK;
                const auto secondary_fragment =
                    *reinterpret_cast<const f8x16_t*>(
                        smem_secondary_q + q_offset);
                secondary_lane_q_all_finite &=
                    fp8x16_all_finite_e4m3fn(secondary_fragment);
            }
        }
    }
    // Only skip a raw +0 KV tile when multiplication cannot turn +0 into
    // -0.  Sparse-mask TopK distinguishes those raw output bits.
    bool lane_weights_safe_for_zero_tile = true;
    #pragma unroll
    for (int item = 0; item < kQIterations * 4; ++item) {
        const uint32_t bits = __float_as_uint(reg_weights[item]);
        lane_weights_safe_for_zero_tile &=
            (bits & 0x7f800000u) != 0x7f800000u &&
            (bits & 0x80000000u) == 0u;
    }
    const bool primary_inputs_safe_for_zero_tile =
        __ballot(lane_q_all_finite && lane_weights_safe_for_zero_tile) ==
        0xffffffffffffffffull;
    bool secondary_lane_weight_safe_for_zero_tile = true;
    if constexpr (kHasSecondary) {
        if (lane < kHeads) {
            const uint32_t bits =
                __float_as_uint(smem_secondary_weights[lane]);
            secondary_lane_weight_safe_for_zero_tile =
                (bits & 0x7f800000u) != 0x7f800000u &&
                (bits & 0x80000000u) == 0u;
        }
    }
    const bool secondary_inputs_safe_for_zero_tile =
        __ballot(secondary_lane_q_all_finite &&
                 secondary_lane_weight_safe_for_zero_tile) ==
        0xffffffffffffffffull;

    const uint64_t logits_offset = q_idx * logits_stride;
    const uint64_t secondary_logits_offset = secondary_q_idx * logits_stride;
    const uint64_t mask_stride = (logits_stride + kTokenTile - 1) /
                                 kTokenTile;
    const uint64_t mask_offset = q_idx * mask_stride;
    const uint64_t secondary_mask_offset = secondary_q_idx * mask_stride;
    const uint64_t shared_block_table_offset =
        group_first_row * block_table_stride;

    const auto prefetch_group_tile = [&](const int tile_token_offset,
                                         const int buffer_slot) {
        const bool group_tile_valid = tile_token_offset < group_max_length;
        if (row_slot == 0 && group_tile_valid) {
            const int logical_page = tile_token_offset / kPageSize;
            const int physical_page = block_table[
                shared_block_table_offset + logical_page];
            const auto* page = kv_block +
                static_cast<int64_t>(physical_page) * kv_cache_stride_bytes;
            auto* smem_warp_kv = smem_kv +
                buffer_slot * kKvBufferBytes +
                token_warp * kTokenTile * kHeadDim;
            const int row_in_page = tile_token_offset % kPageSize;
            // This is a direct global-to-LDS FP8 load.  In particular, do
            // not materialize a BF16 KV tile or use the BF16 conversion path
            // that the gfx936 kernel needs.
            buffer_load_page_tile_lds<uint8_t, kHeadDim, kTokenTile>(
                page, row_in_page, lane, smem_warp_kv);
            if (lane < kTokenTile) {
                const auto* page_scales = reinterpret_cast<const float*>(
                    page + kPageSize * kHeadDim);
                smem_kv_scales[
                    (buffer_slot * kTokenWarps + token_warp) *
                        kTokenTile + lane] =
                    page_scales[row_in_page + lane];
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
        const auto* smem_warp_kv = smem_kv +
            buffer_slot * kKvBufferBytes +
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
        bool scale_safe_for_zero_tile = true;
        if (lane < kTokenTile) {
            const float scale = smem_kv_scales[
                (buffer_slot * kTokenWarps + token_warp) *
                    kTokenTile + lane];
            const uint32_t bits = __float_as_uint(scale);
            scale_safe_for_zero_tile =
                (bits & 0x7f800000u) != 0x7f800000u &&
                (bits & 0x80000000u) == 0u;
        }
        const bool scales_safe_for_zero_tile =
            __ballot(scale_safe_for_zero_tile) == 0xffffffffffffffffull;
        if (lane == 0) {
            tile_nonzero[buffer_slot][token_warp] =
                nonzero_lanes != 0u || !scales_safe_for_zero_tile;
        }
    };

    for (int cta_idx = cta_seed; cta_idx < num_compute_ctas;
         cta_idx += persistent_ctas_per_request) {
        const int cta_token_offset = cta_idx * kTokensPerCta;
        bool tile_loaded[2] = {false, false};
        tile_loaded[0] = prefetch_group_tile(
            cta_token_offset + token_warp * kTokenTile, 0);
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
            const auto* smem_warp_kv = smem_kv +
                buffer_slot * kKvBufferBytes +
                token_warp * kTokenTile * kHeadDim;

            const bool has_next_tile = kv_iter + 1 < kKvIterations;
            if (has_next_tile) {
                const int next_tile_token_offset = cta_token_offset +
                    ((kv_iter + 1) * kTokenWarps + token_warp) *
                    kTokenTile;
                tile_loaded[buffer_slot ^ 1] = prefetch_group_tile(
                    next_tile_token_offset, buffer_slot ^ 1);
            }

            const bool row_tile_valid =
                row_active && tile_token_offset < context_len;
            const bool kv_tile_nonzero =
                tile_nonzero[buffer_slot][token_warp] != 0;
            const bool compute_primary = row_tile_valid &&
                (kv_tile_nonzero || !primary_inputs_safe_for_zero_tile);
            fp32x4 accumulators[kQIterations];
            if (compute_primary) {
                CLEAR_ACC(accumulators, kQIterations);
                const int base_k_lane =
                    lane % kTokenTile * kHeadDim +
                    lane / kTokenTile * kAlignmentQ;
                #pragma unroll
                for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                    const auto reg_kv = *reinterpret_cast<const f8x16_t*>(
                        smem_warp_kv + base_k_lane + k_block * kTileK);
                    #pragma unroll
                    for (int q_iter = 0; q_iter < kQIterations; ++q_iter) {
                        builtin_fp8_mmac<true>(
                            reg_kv.front,
                            reg_q[q_iter][k_block].front,
                            accumulators[q_iter]);
                        builtin_fp8_mmac<true>(
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
                sum *= smem_kv_scales[
                    (buffer_slot * kTokenWarps + token_warp) *
                        kTokenTile + lane % kTokenTile];
            }

            const int token = tile_token_offset + lane;
            const bool in_range = lane < kTokenTile && token < context_len;
            const auto nonzero_bits = __ballot(
                in_range && __float_as_uint(sum) != 0u);
            if (row_active && lane == 0 && tile_token_offset < logits_limit) {
                sparse_mask[mask_offset + tile_token_offset / kTokenTile] =
                    static_cast<int16_t>(nonzero_bits & 0xffffu);
            }
            if (in_range && __float_as_uint(sum) != 0u) {
                logits[logits_offset + token] = sum;
            }

            if constexpr (kHasSecondary) {
                if (row_slot == 0) {
                    const bool secondary_tile_valid =
                        tile_token_offset < secondary_context_len;
                    const bool compute_secondary = secondary_tile_valid &&
                        (kv_tile_nonzero ||
                         !secondary_inputs_safe_for_zero_tile);
                    if (compute_secondary) {
                        CLEAR_ACC(accumulators, kQIterations);
                        const int base_k_lane =
                            lane % kTokenTile * kHeadDim +
                            lane / kTokenTile * kAlignmentQ;
                        #pragma unroll
                        for (int k_block = 0; k_block < kKBlocks; ++k_block) {
                            const auto reg_kv =
                                *reinterpret_cast<const f8x16_t*>(
                                    smem_warp_kv + base_k_lane +
                                    k_block * kTileK);
                            #pragma unroll
                            for (int q_iter = 0; q_iter < kQIterations;
                                 ++q_iter) {
                                const int q_offset = base_q_lane +
                                    q_iter * kTileQ * kHeadDim +
                                    k_block * kTileK;
                                const auto reg_secondary_q =
                                    *reinterpret_cast<const f8x16_t*>(
                                        smem_secondary_q + q_offset);
                                builtin_fp8_mmac<true>(
                                    reg_kv.front,
                                    reg_secondary_q.front,
                                    accumulators[q_iter]);
                                builtin_fp8_mmac<true>(
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
                                const int head = q_iter * kTileQ +
                                    item * 4 + lane / kTileQ;
                                secondary_sum +=
                                    fmaxf(accumulators[q_iter][item], 0.0f) *
                                    smem_secondary_weights[head];
                            }
                        }
                        #pragma unroll
                        for (uint32_t delta = 16; delta <= 32;
                             delta <<= 1) {
                            secondary_sum += __shfl_down_sync(
                                0xffffffffffffffff, secondary_sum, delta);
                        }
                        secondary_sum *= smem_kv_scales[
                            (buffer_slot * kTokenWarps + token_warp) *
                                kTokenTile + lane % kTokenTile];
                    }

                    const bool secondary_in_range =
                        lane < kTokenTile && token < secondary_context_len;
                    const auto secondary_nonzero_bits = __ballot(
                        secondary_in_range &&
                        __float_as_uint(secondary_sum) != 0u);
                    if (lane == 0 && tile_token_offset < logits_limit) {
                        sparse_mask[secondary_mask_offset +
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

template <int kGroupedRows>
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
    constexpr int kThreads = 4 * kTokenWarps * WARP_SIZE_GPU;
    constexpr int kKvBufferBytes =
        kTokenWarps * kTokenTile * kHeadDim * sizeof(uint8_t);
    constexpr int kQBytes = kHeads * kHeadDim * sizeof(uint8_t);
    constexpr int kSmemBytes = 2 * kKvBufferBytes + kQBytes +
        kHeads * sizeof(float) +
        2 * kTokenWarps * kTokenTile * sizeof(float);
    const auto logits_stride = static_cast<uint64_t>(logits.stride(0));
    const auto kv_cache_stride_bytes = static_cast<uint64_t>(
        kv_cache.stride(0) * kv_cache.element_size());
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));
    const int requests = static_cast<int>(q.size(0)) / kGroupedRows;
    const dim3 grid(persistent_ctas * requests, 1, 1);
    const dim3 block(kThreads, 1, 1);

    paged_mqa_logits_grouped_sparse_fp8_kernel<kGroupedRows><<<
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
    TORCH_CHECK(
        error == hipSuccess,
        "grouped sparse FP8 MQA launch failed: ",
        hipGetErrorString(error));
}

inline void launch_group4_four_wave_kernel(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    torch::Tensor& nonzero_mask,
    const int persistent_ctas,
    cudaStream_t stream) {
    constexpr int kGroupedRows = 4;
    constexpr int kRowsPerCta = 2;
    constexpr int kPairsPerRequest = kGroupedRows / kRowsPerCta;
    constexpr int kThreads = kTokenWarps * WARP_SIZE_GPU;
    constexpr int kGroupQBytes =
        kRowsPerCta * kHeads * kHeadDim * sizeof(uint8_t);
    constexpr int kGroupWeightBytes =
        kRowsPerCta * kHeads * sizeof(float);
    constexpr int kKvBufferBytes =
        kTokenWarps * kTokenTile * kHeadDim * sizeof(uint8_t);
    constexpr int kQStagingBytes = kGroupQBytes + kGroupWeightBytes;
    constexpr int kKvStagingBytes = 2 * kKvBufferBytes;
    constexpr int kSmemBytes =
        kQStagingBytes > kKvStagingBytes
        ? kQStagingBytes
        : kKvStagingBytes;
    const auto logits_stride = static_cast<uint64_t>(logits.stride(0));
    const auto kv_cache_stride_bytes = static_cast<uint64_t>(
        kv_cache.stride(0) * kv_cache.element_size());
    const auto block_table_stride =
        static_cast<uint64_t>(block_table.stride(0));
    const int requests = static_cast<int>(q.size(0)) / kGroupedRows;
    const dim3 grid(
        persistent_ctas * requests * kPairsPerRequest, 1, 1);
    const dim3 block(kThreads, 1, 1);

    paged_mqa_logits_group4_sparse_fp8_four_wave_kernel<<<
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
    TORCH_CHECK(
        error == hipSuccess,
        "group4 four-wave sparse FP8 MQA launch failed: ",
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
            launch_grouped_kernel<1>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 3:
            launch_grouped_kernel<3>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 4:
            launch_group4_four_wave_kernel(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        case 5:
            launch_grouped_kernel<5>(
                q, kv_cache, weights, context_lens, block_table, logits,
                nonzero_mask, persistent_ctas, stream);
            return;
        default:
            TORCH_CHECK(false, "sparse FP8 MQA supports group_size 1/3/4/5");
    }
}

}  // namespace grouped_sparse_fp8
}  // namespace deep_gemm
}  // namespace lightop
