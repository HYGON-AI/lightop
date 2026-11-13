// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// DeepGEMM get_paged_sparse_mqa_logits_metadata on HCU (wave64). The metadata is
// byte-compatible with DeepGEMM's layout::sparse_mqa_logits for the paged case:
//
//   [MetadataHeader {num_kv_splits, num_waves, use_unaligned_ks = 0}]
//   [KVSplit x num_kv_splits]   KVSplitHeader {q_token_base, num_kv_blocks, q0_slot_base,
//                               q1_slot_base or ~0} + KVBlockInfo {physical block,
//                               q0 | q1 << 16 split-local slot, 0xffff if absent} x split
//   [ScheduleEntry x num_waves * num_sms] {kv_split_begin, kv_split_end, q_token_base,
//                               num_q_tokens}, unused slots zero
//
// A Q block is a request-aligned pair of query tokens (or a request's last odd one).
// Its rows' first min(M, ceil(len / sparse_block_kv)) sparse blocks are merged into
// one ascending, de-duplicated list, cut into splits of split_kv / sparse_block_kv
// blocks (the last padded with absent slots) and mapped through the leader's
// block_table row. Every Q block's splits are cut into ceil(splits / 8) near-equal
// schedule entries in Q order, num_sms per wave; with more than one wave each wave
// is counting-sorted by split count and rotated across SMs.
//
// As in DeepGEMM, Q blocks claim their split range with a global atomic, so split
// order varies between runs; the merge, slots, splits and schedule shape do not.
// Ties in the wave sort are broken by SM index (DeepGEMM leaves them to atomics).
//
// Structure follows DeepGEMM: CTAs take Q tokens by ticket, merge with a merge path
// in LDS, and the last CTA plans the schedule. Work per Q block is bounded by M <= 4096
// whatever the context length (up to 1M tokens), so the kernel is latency bound; the
// HCU-specific choices, all measured on gfx936 for decode batches (Q <= 96):
//  - the request-start scan for a Q token is one wave-wide load + ballot instead of
//    a serial walk by one thread,
//  - the CTA size follows M so each thread merges about 4 blocks (8 for M = 4096 at
//    the 1024-thread limit, and for M = 2048 with 2..63 tokens, where it is faster),
//  - when there are more CTAs than CUs, even tokens (the Q-block leaders of
//    even-sized requests) go to the first half of the CTAs so working CTAs do not
//    share a CU (a template parameter: reading a runtime flag first cost 0.15-0.5 us).
#include "paged_sparse_mqa_logits_metadata.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace lightop::deep_gemm {
namespace {

using uintx2 = unsigned __attribute__((ext_vector_type(2)));
using uintx4 = unsigned __attribute__((ext_vector_type(4)));

constexpr uint32_t kBlockQ = 2;
constexpr uint32_t kNumSparseSlotBits = 16;
constexpr uint32_t kInvalidSparseSlot = (1u << kNumSparseSlotBits) - 1;
constexpr uint32_t kPresentBit = 1u << (kNumSparseSlotBits - 1);
constexpr uint32_t kSlotIndexMask = kPresentBit - 1;
constexpr uint32_t kNumKVSplitsPerEntry = 8;
constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kMaxSparseBlocks = 4096;
constexpr uint32_t kMaxSMs = 256;
constexpr uint32_t kMaxSMsPerLane = kMaxSMs / kWaveSize;
constexpr int64_t kNumMaxQTokens = 1 << 20;
constexpr int kCtasPerCU = 4;

// DeepGEMM layout::sparse_mqa_logits
struct alignas(16) MetadataHeader {
    uint32_t num_kv_splits;
    uint32_t num_waves;
    uint32_t use_unaligned_ks;
};
struct alignas(16) KVSplitHeader {
    uint32_t q_token_base;
    uint32_t packed_num_kv_blocks;
    uint32_t q0_slot_base;
    uint32_t q1_slot_base;
};
struct KVBlockInfo {
    uint32_t physical_kv_block_idx;
    uint32_t packed_slot_offsets;
};
struct alignas(16) ScheduleEntry {
    uint32_t kv_split_begin;
    uint32_t kv_split_end;
    uint32_t q_token_base;
    uint32_t num_q_tokens;
};
struct alignas(128) WorkspaceState {
    uint32_t num_kv_splits;
    alignas(128) uint32_t next_q_offset;
    alignas(128) uint32_t num_finished_ctas;
};
struct alignas(8) QBlockInfo {
    uint32_t kv_split_base;
    uint32_t num_kv_splits;
};
static_assert(sizeof(MetadataHeader) == 16 && sizeof(KVSplitHeader) == 16 && sizeof(KVBlockInfo) == 8 &&
              sizeof(ScheduleEntry) == 16 && sizeof(WorkspaceState) == 384 && sizeof(QBlockInfo) == 8,
              "DeepGEMM sparse MQA metadata layout");

struct Params {
    uint32_t num_q_tokens;
    uint32_t num_max_sparse_blocks;
    uint32_t num_kv_blocks_per_page;
    uint32_t page_shift;  // log2(num_kv_blocks_per_page), or 32 if not a power of two
    uint32_t num_sms;
    uint32_t block_table_stride;
    const uint32_t* context_lens;
    const uint32_t* block_table;
    const uint32_t* indices;
    const uint32_t* sparse_kv_block_indices;
    uint8_t* metadata;
    uint8_t* workspace;
    bool paired_rows;
    PrefillSparseMQASelection selection;
};

__device__ __forceinline__ uint32_t ceil_div_u32(const uint32_t a, const uint32_t b) {
    return (a + b - 1) / b;
}

// Exclusive CTA prefix sum; `total` gets the CTA sum. Callers separate reuses of
// wave_sums with a barrier.
template <uint32_t kNumThreads>
__device__ __forceinline__ uint32_t cta_exclusive_sum(const uint32_t value, uint32_t* wave_sums, uint32_t& total) {
    constexpr uint32_t kNumWaves = kNumThreads / kWaveSize;
    const uint32_t lane = threadIdx.x % kWaveSize;
    const uint32_t wave = threadIdx.x / kWaveSize;
    uint32_t inclusive = value;
#pragma unroll
    for (uint32_t offset = 1; offset < kWaveSize; offset <<= 1) {
        const uint32_t other = __shfl_up(inclusive, offset, kWaveSize);
        inclusive += lane >= offset ? other : 0;
    }
    if (lane == kWaveSize - 1)
        wave_sums[wave] = inclusive;
    __syncthreads();
    uint32_t base = 0, sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumWaves; ++i) {
        const uint32_t wave_sum = wave_sums[i];
        base += i < wave ? wave_sum : 0;
        sum += wave_sum;
    }
    total = sum;
    return base + inclusive - value;
}

template <uint32_t SPARSE_BLOCK_KV, uint32_t kNumKVBlocksPerSplit, uint32_t kNumKVBlocksPerThread, uint32_t kNumThreads,
          bool kInterleaveCtas>
__global__ __launch_bounds__(kNumThreads) void sparse_fp8_fp4_paged_mqa_logits__prepare_metadata(const Params p) {
    constexpr uint32_t kSplitBytes = sizeof(KVSplitHeader) + kNumKVBlocksPerSplit * sizeof(KVBlockInfo);
    constexpr uint32_t kNumVecsPerThread = (kNumKVBlocksPerThread + 3) / 4;
    constexpr uint32_t kNumWaves = kNumThreads / kWaveSize;
    static_assert(kNumKVBlocksPerSplit <= kInvalidSparseSlot, "Sparse slot offset overflow");

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % kWaveSize;
    const uint32_t num_q_tokens = p.num_q_tokens;
    const uint32_t num_max_sparse_blocks = p.num_max_sparse_blocks;
    const uint32_t* __restrict__ indices = p.indices;
    auto* state = reinterpret_cast<WorkspaceState*>(p.workspace);
    auto* q_block_infos = reinterpret_cast<QBlockInfo*>(p.workspace + sizeof(WorkspaceState));
    uint8_t* kv_splits = p.metadata + sizeof(MetadataHeader);

    extern __shared__ uintx4 dynamic_smem[];
    uint32_t* logical_kv_block_indices = reinterpret_cast<uint32_t*>(dynamic_smem);  // [2][M]
    uint32_t* packed_slot_bases_by_kv_split = logical_kv_block_indices + 2 * num_max_sparse_blocks;
    __shared__ uint32_t wave_sums[kNumWaves];
    __shared__ uint32_t s_q_token_base, s_num_q_tokens, s_num_kv_blocks[kBlockQ], s_kv_split_base, s_is_last;

    // Wave 0 owns Q-block selection; `q_token_idx` is uniform in it, `next_ticket` lives in lane 0.
    // Interleaving is a template parameter: reading a runtime flag first delays the first load
    uint32_t q_token_idx = blockIdx.x;
    if constexpr (kInterleaveCtas) {
        const uint32_t num_even_ctas = (gridDim.x + 1) / 2;
        q_token_idx = blockIdx.x < num_even_ctas ? 2 * blockIdx.x : 2 * (blockIdx.x - num_even_ctas) + 1;
    }
    if (p.paired_rows) q_token_idx = blockIdx.x * 2;
    uint32_t selected_slots = 0;
    uint32_t next_ticket = 0;
    const bool page_is_pow2 = p.page_shift < 32;

    while (true) {
        if (tid < kWaveSize) {
            uint32_t num_q_block_tokens = 0, num_kv_blocks_q0 = 0, num_kv_blocks_q1 = 0;
            if (p.paired_rows) {
                if (q_token_idx < num_q_tokens) {
                    num_q_block_tokens = min(2u, num_q_tokens-q_token_idx);
                    num_kv_blocks_q0 = min(num_max_sparse_blocks,ceil_div_u32(p.context_lens[q_token_idx],SPARSE_BLOCK_KV));
                    num_kv_blocks_q1 = num_q_block_tokens==2 ? min(num_max_sparse_blocks,ceil_div_u32(p.context_lens[q_token_idx+1],SPARSE_BLOCK_KV)) : 0;
                    if (lane==0) next_ticket=2*(gridDim.x+atomicAdd(&state->next_q_offset,1u));
                }
            } else while (q_token_idx < num_q_tokens) {
                // Lanes 0..61 see the 62 preceding tokens, lane 62 this token, lane 63 the next
                const int64_t pos = static_cast<int64_t>(q_token_idx) - (kWaveSize - 2) + lane;
                const bool valid = pos >= 0 and pos < num_q_tokens;
                const uint32_t request = valid ? indices[pos] : 0;
                const uint32_t context_len = valid and lane >= kWaveSize - 2 ? p.context_lens[pos] : 0;
                const uint32_t request_idx = __shfl(request, kWaveSize - 2, kWaveSize);
                const uint64_t others = __ballot(lane < kWaveSize - 2 and not(valid and request == request_idx));
                uint32_t offset_in_request;
                if (others != 0) {
                    offset_in_request = (kWaveSize - 3) - (63 - __clzll(others));
                } else {
                    // Requests contain few Q tokens; walk the rest serially
                    uint32_t request_q_token_base = q_token_idx - (kWaveSize - 2);
                    if (num_q_tokens>=64) {
                        // Cooperatively find the nearest boundary. Unlike an
                        // exponential/binary search this also preserves the
                        // behavior for repeated nonadjacent request IDs.
                        while(request_q_token_base>0) {
                            const int64_t scan_pos=int64_t(request_q_token_base)-64+lane;
                            const uint64_t boundary=__ballot(scan_pos<0 || indices[scan_pos]!=request_idx);
                            if(boundary) {
                                request_q_token_base=uint32_t(int64_t(request_q_token_base)-64+(63-__clzll(boundary))+1);
                                break;
                            }
                            request_q_token_base=request_q_token_base>=64?request_q_token_base-64:0;
                        }
                    } else if (lane == 0) {
                        while (request_q_token_base > 0 and indices[request_q_token_base - 1] == request_idx)
                            --request_q_token_base;
                    }
                    offset_in_request = q_token_idx - __shfl(request_q_token_base, 0, kWaveSize);
                }
                if (offset_in_request % kBlockQ != 0) {
                    uint32_t ticket = 0;
                    if (lane == 0)
                        ticket = gridDim.x + atomicAdd(&state->next_q_offset, 1u);
                    q_token_idx = __shfl(ticket, 0, kWaveSize);
                    continue;
                }
                const bool has_next = __shfl(static_cast<int>(valid and request == request_idx), kWaveSize - 1,
                                             kWaveSize) != 0;
                num_q_block_tokens = has_next ? kBlockQ : 1;
                num_kv_blocks_q0 = min(num_max_sparse_blocks,
                                       ceil_div_u32(__shfl(context_len, kWaveSize - 2, kWaveSize), SPARSE_BLOCK_KV));
                const uint32_t num_kv_blocks_next = min(
                    num_max_sparse_blocks, ceil_div_u32(__shfl(context_len, kWaveSize - 1, kWaveSize), SPARSE_BLOCK_KV));
                num_kv_blocks_q1 = has_next ? num_kv_blocks_next : 0;
                // Prefetch the next ticket; it is always consumed below
                if (lane == 0)
                    next_ticket = gridDim.x + atomicAdd(&state->next_q_offset, 1u);
                break;
            }
            if (lane == 0) {
                s_q_token_base = q_token_idx;
                s_num_q_tokens = num_q_block_tokens;
                s_num_kv_blocks[0] = num_kv_blocks_q0;
                s_num_kv_blocks[1] = num_kv_blocks_q1;
            }
        }
        __syncthreads();

        const uint32_t num_q_block_tokens = s_num_q_tokens;
        if (num_q_block_tokens == 0)
            break;
        const uint32_t q_token_base = s_q_token_base;
        const uint32_t num_kv_blocks_in_q0 = s_num_kv_blocks[0];
        const uint32_t num_kv_blocks_in_q1 = s_num_kv_blocks[1];

        // Load both rows' sparse indices (16-byte vectors, read once)
        {
            const uint32_t num_vecs_q0 = (num_kv_blocks_in_q0 + 3) / 4;
            const uint32_t num_vecs = num_vecs_q0 + (num_kv_blocks_in_q1 + 3) / 4;
            const uint32_t* src = p.sparse_kv_block_indices + static_cast<uint64_t>(q_token_base) * num_max_sparse_blocks;
            uintx4 blocks[kNumVecsPerThread];
#pragma unroll
            for (uint32_t i = 0; i < kNumVecsPerThread; ++i) {
                const uint32_t vec_idx = tid + i * kNumThreads;
                if (vec_idx < num_vecs) {
                    const uint32_t offset = vec_idx < num_vecs_q0 ? 4 * vec_idx
                                                                  : num_max_sparse_blocks + 4 * (vec_idx - num_vecs_q0);
                    blocks[i] = __builtin_nontemporal_load(reinterpret_cast<const uintx4*>(src + offset));
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < kNumVecsPerThread; ++i) {
                const uint32_t vec_idx = tid + i * kNumThreads;
                if (vec_idx < num_vecs) {
                    const uint32_t offset = vec_idx < num_vecs_q0 ? 4 * vec_idx
                                                                  : num_max_sparse_blocks + 4 * (vec_idx - num_vecs_q0);
                    *reinterpret_cast<uintx4*>(logical_kv_block_indices + offset) = blocks[i];
                }
            }
        }
        __syncthreads();
        const uint32_t* logical_q0 = logical_kv_block_indices;
        const uint32_t* logical_q1 = logical_kv_block_indices + num_max_sparse_blocks;

        // Merge-path partition
        const uint32_t num_input_kv_blocks = num_kv_blocks_in_q0 + num_kv_blocks_in_q1;
        const uint32_t merge_begin = tid * num_input_kv_blocks / kNumThreads;
        const uint32_t merge_end = (tid + 1) * num_input_kv_blocks / kNumThreads;
        uint32_t lo = max(merge_begin, num_kv_blocks_in_q1) - num_kv_blocks_in_q1;
        uint32_t hi = min(merge_begin, num_kv_blocks_in_q0);
        while (lo < hi) {
            const uint32_t q0_slot_idx = (lo + hi) / 2;
            const uint32_t q1_slot_idx = merge_begin - q0_slot_idx;
            if (q1_slot_idx > 0 and q0_slot_idx < num_kv_blocks_in_q0 and
                logical_q1[q1_slot_idx - 1] >= logical_q0[q0_slot_idx])
                lo = q0_slot_idx + 1;
            else
                hi = q0_slot_idx;
        }
        uint32_t q0_slot_idx = lo;
        uint32_t q1_slot_idx = merge_begin - lo;
        uint32_t num_remaining_inputs = merge_end - merge_begin;
        uint32_t num_merged_kv_blocks_in_thread = 0;

        // Drop a duplicate carried across merge partitions
        if (num_remaining_inputs > 0 and q0_slot_idx > 0 and q1_slot_idx < num_kv_blocks_in_q1 and
            logical_q0[q0_slot_idx - 1] == logical_q1[q1_slot_idx]) {
            ++q1_slot_idx;
            --num_remaining_inputs;
        }

        // Merge, pack each Q's slot and presence bit, and gather from the block table
        const uint64_t block_table_row = static_cast<uint64_t>(q_token_base) * p.block_table_stride;
        uint32_t packed_slots_in_thread[kNumKVBlocksPerThread];
        uint32_t physical_kv_block_indices[kNumKVBlocksPerThread];
#pragma unroll
        for (uint32_t i = 0; i < kNumKVBlocksPerThread; ++i) {
            if (num_remaining_inputs == 0)
                continue;
            const uint32_t q0_logical_kv_block_idx = q0_slot_idx < num_kv_blocks_in_q0 ? logical_q0[q0_slot_idx] : ~0u;
            const uint32_t q1_logical_kv_block_idx = q1_slot_idx < num_kv_blocks_in_q1 ? logical_q1[q1_slot_idx] : ~0u;
            const bool in_q0 = q0_logical_kv_block_idx <= q1_logical_kv_block_idx;
            const bool in_q1 = q1_logical_kv_block_idx <= q0_logical_kv_block_idx;
            // Carry a final duplicate into the next partition
            const bool consume_q1 = in_q1 and num_remaining_inputs > static_cast<uint32_t>(in_q0);
            packed_slots_in_thread[i] = (q0_slot_idx | (in_q0 ? kPresentBit : 0u)) |
                                        ((q1_slot_idx | (in_q1 ? kPresentBit : 0u)) << kNumSparseSlotBits);
            const uint32_t logical_kv_block_idx = min(q0_logical_kv_block_idx, q1_logical_kv_block_idx);
            const uint32_t logical_page_idx = page_is_pow2 ? logical_kv_block_idx >> p.page_shift
                                                           : logical_kv_block_idx / p.num_kv_blocks_per_page;
            const uint32_t kv_block_idx_in_page = logical_kv_block_idx - logical_page_idx * p.num_kv_blocks_per_page;
            physical_kv_block_indices[i] =
                p.block_table[block_table_row + logical_page_idx] * p.num_kv_blocks_per_page + kv_block_idx_in_page;
            if (p.selection.selected_blocks) {
                const uint32_t physical=physical_kv_block_indices[i];
                if (!(physical < static_cast<uint64_t>(p.selection.num_pages)*8)) __builtin_trap();
                selected_slots += in_q0;
                selected_slots += num_q_block_tokens==2 && in_q1;
                // Only a newly selected physical block claims its page. Across
                // all Q pairs this bounds page CAS traffic by eight per page.
                if (atomicExch(p.selection.selected_blocks+physical,1)==0) {
                    const uint32_t page=physical>>3;
                    if (atomicCAS(p.selection.cache_pages+page,-1,-2)==-1) {
                        const int slot=atomicAdd(p.selection.cache_count,1);
                        atomicExch(p.selection.cache_pages+page,slot);
                        if(slot>=p.selection.cache_capacity && !p.selection.long_packed_cache)
                            atomicOr(p.selection.invalid,2);
                        if(slot<(p.selection.long_packed_cache?p.selection.num_pages:p.selection.cache_capacity))
                            p.selection.cache_physical_pages[slot]=page;
                    }
                }
            }
            ++num_merged_kv_blocks_in_thread;
            q0_slot_idx += in_q0;
            q1_slot_idx += consume_q1;
            num_remaining_inputs -= static_cast<uint32_t>(in_q0) + static_cast<uint32_t>(consume_q1);
        }

        // Compact merged blocks and reserve their KV-split range
        uint32_t num_merged_kv_blocks;
        const uint32_t merged_kv_block_base =
            cta_exclusive_sum<kNumThreads>(num_merged_kv_blocks_in_thread, wave_sums, num_merged_kv_blocks);
        const uint32_t num_kv_splits_in_q_block = ceil_div_u32(num_merged_kv_blocks, kNumKVBlocksPerSplit);
        if (tid == 0) {
            const uint32_t kv_split_base =
                num_kv_splits_in_q_block == 0 ? 0 : atomicAdd(&state->num_kv_splits, num_kv_splits_in_q_block);
            s_kv_split_base = kv_split_base;
            *reinterpret_cast<uintx2*>(q_block_infos + q_token_base) = uintx2{kv_split_base, num_kv_splits_in_q_block};
            if (num_q_block_tokens == kBlockQ)
                *reinterpret_cast<uintx2*>(q_block_infos + q_token_base + 1) = uintx2{0u, 0u};
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumKVBlocksPerThread; ++i) {
            const uint32_t merged_kv_block_idx = merged_kv_block_base + i;
            if (i < num_merged_kv_blocks_in_thread and merged_kv_block_idx % kNumKVBlocksPerSplit == 0)
                packed_slot_bases_by_kv_split[merged_kv_block_idx / kNumKVBlocksPerSplit] =
                    packed_slots_in_thread[i] & (kSlotIndexMask | (kSlotIndexMask << kNumSparseSlotBits));
        }
        __syncthreads();

        // Write KV split metadata
        const uint32_t kv_split_base = s_kv_split_base;
#pragma unroll
        for (uint32_t i = 0; i < kNumKVBlocksPerThread; ++i) {
            if (i >= num_merged_kv_blocks_in_thread)
                continue;
            const uint32_t merged_kv_block_idx = merged_kv_block_base + i;
            const uint32_t kv_split_offset = merged_kv_block_idx / kNumKVBlocksPerSplit;
            const uint32_t kv_block_idx_in_split = merged_kv_block_idx % kNumKVBlocksPerSplit;
            const uint32_t packed_slots = packed_slots_in_thread[i];
            const uint32_t packed_slot_bases = packed_slot_bases_by_kv_split[kv_split_offset];
            const uint32_t q0_slot_base = packed_slot_bases & kSlotIndexMask;
            const uint32_t q1_slot_base = (packed_slot_bases >> kNumSparseSlotBits) & kSlotIndexMask;
            const bool in_q0 = (packed_slots & kPresentBit) != 0;
            const bool in_q1 = ((packed_slots >> kNumSparseSlotBits) & kPresentBit) != 0;
            const uint32_t q0_slot_offset = in_q0 ? (packed_slots & kSlotIndexMask) - q0_slot_base : kInvalidSparseSlot;
            const uint32_t q1_slot_offset =
                in_q1 ? ((packed_slots >> kNumSparseSlotBits) & kSlotIndexMask) - q1_slot_base : kInvalidSparseSlot;
            uint8_t* kv_split = kv_splits + static_cast<uint64_t>(kv_split_base + kv_split_offset) * kSplitBytes;
            if (kv_block_idx_in_split == 0) {
                const uint32_t num_kv_blocks_in_split =
                    min(kNumKVBlocksPerSplit, num_merged_kv_blocks - merged_kv_block_idx);
                *reinterpret_cast<uintx4*>(kv_split) = uintx4{
                    q_token_base, num_kv_blocks_in_split, q0_slot_base,
                    num_q_block_tokens == kBlockQ ? q1_slot_base : ~0u};
            }
            *reinterpret_cast<uintx2*>(kv_split + sizeof(KVSplitHeader) + kv_block_idx_in_split * sizeof(KVBlockInfo)) =
                uintx2{physical_kv_block_indices[i], q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits)};
        }

        // Pad the last split to keep the consumer's copy loop branch-free
        for (uint32_t padded_slot = num_merged_kv_blocks + tid;
             padded_slot < num_kv_splits_in_q_block * kNumKVBlocksPerSplit; padded_slot += kNumThreads) {
            uint8_t* kv_split =
                kv_splits + static_cast<uint64_t>(kv_split_base + padded_slot / kNumKVBlocksPerSplit) * kSplitBytes;
            *reinterpret_cast<uintx2*>(kv_split + sizeof(KVSplitHeader) +
                                       (padded_slot % kNumKVBlocksPerSplit) * sizeof(KVBlockInfo)) =
                uintx2{0u, kInvalidSparseSlot | (kInvalidSparseSlot << kNumSparseSlotBits)};
        }
        if (tid < kWaveSize)
            q_token_idx = __shfl(next_ticket, 0, kWaveSize);
    }

    if (p.selection.selected_slots) {
        for(int offset=32;offset;offset/=2) selected_slots+=__shfl_down(selected_slots,offset,64);
        if(lane==0) atomicAdd(reinterpret_cast<unsigned*>(p.selection.selected_slots),selected_slots);
    }
    // The last CTA acquires all producer writes and builds the schedule
    __threadfence();
    __syncthreads();
    if (tid == 0)
        s_is_last = atomicAdd(&state->num_finished_ctas, 1u) + 1 == gridDim.x;
    __syncthreads();
    if (s_is_last == 0)
        return;
    __threadfence();

    const uint32_t total_kv_splits = __atomic_load_n(&state->num_kv_splits, __ATOMIC_RELAXED);
    auto* schedule_entries =
        reinterpret_cast<ScheduleEntry*>(kv_splits + static_cast<uint64_t>(total_kv_splits) * kSplitBytes);

    // Split each Q block into bounded entries in Q order: thread t owns a contiguous Q range
    const uint32_t num_q_tokens_per_thread = ceil_div_u32(num_q_tokens, kNumThreads);
    const uint32_t q_token_begin = min(num_q_tokens, tid * num_q_tokens_per_thread);
    const uint32_t q_token_end = min(num_q_tokens, q_token_begin + num_q_tokens_per_thread);
    uint32_t num_entries_in_thread = 0;
    for (uint32_t q = q_token_begin; q < q_token_end; ++q)
        num_entries_in_thread += ceil_div_u32(q_block_infos[q].num_kv_splits, kNumKVSplitsPerEntry);
    uint32_t num_entries;
    uint32_t entry_idx = cta_exclusive_sum<kNumThreads>(num_entries_in_thread, wave_sums, num_entries);
    for (uint32_t q = q_token_begin; q < q_token_end; ++q) {
        const uintx2 q_block_info = *reinterpret_cast<const uintx2*>(q_block_infos + q);
        const uint32_t num_kv_splits = q_block_info.y;
        if (num_kv_splits == 0)
            continue;
        const uint32_t num_q_entries = ceil_div_u32(num_kv_splits, kNumKVSplitsPerEntry);
        const uint32_t kv_splits_per_entry = num_kv_splits / num_q_entries;
        const uint32_t num_larger_entries = num_kv_splits % num_q_entries;
        const uint32_t num_q_block_tokens = q + 1 < num_q_tokens and indices[q + 1] == indices[q] ? kBlockQ : 1;
        uint32_t kv_split_begin = q_block_info.x;
        for (uint32_t q_entry_idx = 0; q_entry_idx < num_q_entries; ++q_entry_idx) {
            const uint32_t kv_split_end = kv_split_begin + kv_splits_per_entry + (q_entry_idx < num_larger_entries);
            *reinterpret_cast<uintx4*>(schedule_entries + entry_idx++) =
                uintx4{kv_split_begin, kv_split_end, q, num_q_block_tokens};
            kv_split_begin = kv_split_end;
        }
    }
    const uint32_t num_sms = p.num_sms;
    const uint32_t num_waves = max(1u, ceil_div_u32(num_entries, num_sms));
    for (uint32_t i = num_entries + tid; i < num_waves * num_sms; i += kNumThreads)
        *reinterpret_cast<uintx4*>(schedule_entries + i) = uintx4{0u, 0u, 0u, 0u};
    __syncthreads();

    // Sort each wave by KV-split count (stable in SM order) and rotate heavy entries across SMs
    if (num_waves > 1) {
        const uint64_t lanes_below = (1ull << lane) - 1;
        for (uint32_t wave_idx = tid / kWaveSize; wave_idx < num_waves; wave_idx += kNumWaves) {
            auto* wave_entries = schedule_entries + static_cast<uint64_t>(wave_idx) * num_sms;
            uintx4 entries[kMaxSMsPerLane];
            uint32_t counts[kMaxSMsPerLane], ranks[kMaxSMsPerLane];
#pragma unroll
            for (uint32_t j = 0; j < kMaxSMsPerLane; ++j) {
                const uint32_t sm_idx = lane + j * kWaveSize;
                counts[j] = ~0u;
                ranks[j] = 0;
                if (sm_idx < num_sms) {
                    entries[j] = *reinterpret_cast<const uintx4*>(wave_entries + sm_idx);
                    counts[j] = entries[j].y - entries[j].x;
                }
            }
            uint32_t rank_begin = 0;
#pragma unroll
            for (uint32_t count = 0; count <= kNumKVSplitsPerEntry; ++count) {
#pragma unroll
                for (uint32_t j = 0; j < kMaxSMsPerLane; ++j) {
                    const uint64_t mask = __ballot(counts[j] == count);
                    if (counts[j] == count)
                        ranks[j] = rank_begin + __popcll(mask & lanes_below);
                    rank_begin += __popcll(mask);
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < kMaxSMsPerLane; ++j) {
                if (lane + j * kWaveSize >= num_sms)
                    continue;
                const uint32_t rank = ranks[j];
                // Reverse the two-wave tail to make active SM sets complementary
                const uint32_t dst_sm_idx = num_waves == 2 and wave_idx == 1
                                                ? num_sms - 1 - rank
                                                : (rank + num_sms - wave_idx * num_sms / num_waves) % num_sms;
                *reinterpret_cast<uintx4*>(wave_entries + dst_sm_idx) = entries[j];
            }
        }
    }

    if (tid == 0) {
        *reinterpret_cast<uintx4*>(p.metadata) = uintx4{total_kv_splits, num_waves, 0u, 0u};
        state->num_kv_splits = 0;
        state->next_q_offset = 0;
        state->num_finished_ctas = 0;
    }
}

int get_num_cus(const int device) {
    static std::mutex mutex;
    static std::map<int, int> num_cus;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = num_cus.find(device);
    if (it == num_cus.end()) {
        int value = 0;
        TORCH_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMultiprocessorCount, device) == hipSuccess,
                    "get_paged_sparse_mqa_logits_metadata: failed to query the CU count");
        it = num_cus.emplace(device, value).first;
    }
    return it->second;
}

// One zeroed workspace per (device, stream); the kernel leaves its counters zeroed
const torch::Tensor& get_workspace(const torch::Tensor& like, const hipStream_t stream) {
    constexpr int64_t kNumWorkspaceBytes = sizeof(WorkspaceState) + kNumMaxQTokens * sizeof(QBlockInfo);
    static std::mutex mutex;
    static std::map<std::pair<int, hipStream_t>, torch::Tensor> workspaces;
    std::lock_guard<std::mutex> lock(mutex);
    auto& workspace = workspaces[{like.get_device(), stream}];
    if (not workspace.defined()) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        TORCH_CHECK(hipStreamIsCapturing(stream, &status) == hipSuccess and status == hipStreamCaptureStatusNone,
                    "get_paged_sparse_mqa_logits_metadata: call once on this stream before graph capture");
        workspace = torch::zeros({kNumWorkspaceBytes}, like.options().dtype(torch::kByte));
    }
    return workspace;
}

int64_t get_split_kv(const at::ScalarType qk_dtype) {
    if (qk_dtype == torch::kInt8 or qk_dtype == torch::kByte)
        return 640;
    TORCH_CHECK(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == torch::kFloat8_e4m3fnuz,
                "get_paged_sparse_mqa_logits_metadata: qk_dtype must be packed FP4 (int8/uint8) or FP8 E4M3");
    return 512;
}

int64_t ceil_div_i64(const int64_t a, const int64_t b) {
    return (a + b - 1) / b;
}

}  // namespace

int64_t get_paged_sparse_mqa_logits_metadata_bytes(
    const int64_t num_q_tokens, const int64_t num_max_sparse_blocks, const int64_t sparse_block_kv,
    const int64_t split_kv, const int64_t num_sms) {
    TORCH_CHECK(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= kMaxSparseBlocks,
                "get_paged_sparse_mqa_logits_metadata: num_max_sparse_blocks must be a positive multiple of 4 <= 4096");
    TORCH_CHECK(sparse_block_kv == 8 or sparse_block_kv == 16,
                "get_paged_sparse_mqa_logits_metadata: sparse_block_kv must be 8 or 16");
    TORCH_CHECK(num_q_tokens > 0 and num_sms > 0 and split_kv > 0 and split_kv % sparse_block_kv == 0,
                "get_paged_sparse_mqa_logits_metadata: invalid metadata shape");
    const int64_t num_kv_blocks_per_split = split_kv / sparse_block_kv;
    const int64_t num_kv_split_bytes = sizeof(KVSplitHeader) + num_kv_blocks_per_split * sizeof(KVBlockInfo);
    const int64_t num_max_kv_splits = num_q_tokens * ceil_div_i64(num_max_sparse_blocks, num_kv_blocks_per_split);
    TORCH_CHECK(num_max_kv_splits <= std::numeric_limits<uint32_t>::max(),
                "get_paged_sparse_mqa_logits_metadata: too many KV splits");
    const int64_t num_max_schedule_entries = ceil_div_i64(num_max_kv_splits, num_sms) * num_sms;
    TORCH_CHECK(num_max_schedule_entries <= std::numeric_limits<uint32_t>::max(),
                "get_paged_sparse_mqa_logits_metadata: too many schedule entries");
    return static_cast<int64_t>(sizeof(MetadataHeader)) + num_max_kv_splits * num_kv_split_bytes +
           num_max_schedule_entries * static_cast<int64_t>(sizeof(ScheduleEntry));
}

static torch::Tensor make_paged_sparse_mqa_logits_metadata(
    const torch::Tensor& context_lens, const torch::Tensor& block_table, const torch::Tensor& indices,
    const int64_t page_kv, const torch::Tensor& sparse_kv_block_indices, const at::ScalarType qk_dtype,
    const int64_t sparse_block_kv, const std::optional<int64_t> num_sms_opt,
    const std::optional<torch::Tensor>& destination,
    const PrefillSparseMQASelection& selection, bool paired_rows, int thread_hint) {
    TORCH_CHECK(sparse_kv_block_indices.is_cuda() and sparse_kv_block_indices.dim() == 2,
                "get_paged_sparse_mqa_logits_metadata: sparse_kv_block_indices must be a 2D CUDA/HIP tensor");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(sparse_kv_block_indices.device());
    const int64_t num_q_tokens = sparse_kv_block_indices.size(0);
    const int64_t num_max_sparse_blocks = sparse_kv_block_indices.size(1);
    TORCH_CHECK(num_q_tokens > 0 and num_q_tokens <= kNumMaxQTokens,
                "get_paged_sparse_mqa_logits_metadata: num_q_tokens must be in [1, 2^20]");
    for (const torch::Tensor* tensor : {&context_lens, &block_table, &indices})
        TORCH_CHECK(tensor->device() == sparse_kv_block_indices.device(),
                    "get_paged_sparse_mqa_logits_metadata: all tensors must be on one device");
    TORCH_CHECK(context_lens.numel() == num_q_tokens and context_lens.scalar_type() == torch::kInt and
                    context_lens.is_contiguous(),
                "get_paged_sparse_mqa_logits_metadata: context_lens must be a contiguous int32 [num_q_tokens]");
    TORCH_CHECK(block_table.dim() == 2 and block_table.size(0) == num_q_tokens and block_table.size(1) > 0 and
                    block_table.scalar_type() == torch::kInt and block_table.stride(1) == 1 and
                    block_table.stride(0) <= std::numeric_limits<uint32_t>::max(),
                "get_paged_sparse_mqa_logits_metadata: block_table must be int32 [num_q_tokens, pages] "
                "with unit column stride");
    TORCH_CHECK(indices.dim() == 1 and indices.size(0) == num_q_tokens and indices.scalar_type() == torch::kInt and
                    indices.is_contiguous(),
                "get_paged_sparse_mqa_logits_metadata: indices must be a contiguous int32 [num_q_tokens]");
    TORCH_CHECK(sparse_block_kv == 8 or sparse_block_kv == 16,
                "get_paged_sparse_mqa_logits_metadata: sparse_block_kv must be 8 or 16");
    TORCH_CHECK(page_kv > 0 and page_kv % sparse_block_kv == 0 and page_kv <= std::numeric_limits<uint32_t>::max(),
                "get_paged_sparse_mqa_logits_metadata: page_kv must be a positive multiple of sparse_block_kv");
    TORCH_CHECK(sparse_kv_block_indices.scalar_type() == torch::kInt and sparse_kv_block_indices.is_contiguous() and
                    reinterpret_cast<uintptr_t>(sparse_kv_block_indices.data_ptr()) % 16 == 0,
                "get_paged_sparse_mqa_logits_metadata: sparse_kv_block_indices must be contiguous 16-byte aligned int32");

    const int64_t split_kv = get_split_kv(qk_dtype);
    const int64_t num_cus = get_num_cus(sparse_kv_block_indices.get_device());
    const int64_t num_sms = num_sms_opt.value_or(num_cus);
    TORCH_CHECK(num_sms > 0 and num_sms <= kMaxSMs, "get_paged_sparse_mqa_logits_metadata: num_sms must be in [1, 256]");
    auto metadata = destination ? *destination : torch::empty({get_paged_sparse_mqa_logits_metadata_bytes(num_q_tokens, num_max_sparse_blocks,
                                                                             sparse_block_kv, split_kv, num_sms)},
                                 sparse_kv_block_indices.options().dtype(torch::kByte));

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    const auto& workspace = get_workspace(metadata, stream);

    const int64_t num_kv_blocks_per_split = split_kv / sparse_block_kv;
    const uint32_t num_kv_blocks_per_page = static_cast<uint32_t>(page_kv / sparse_block_kv);
    const int64_t num_max_merged_kv_blocks = kBlockQ * num_max_sparse_blocks;
    const int num_ctas = static_cast<int>(std::min<int64_t>(paired_rows ? (num_q_tokens+1)/2 : num_q_tokens, num_cus * kCtasPerCU));
    const Params params{
        static_cast<uint32_t>(num_q_tokens), static_cast<uint32_t>(num_max_sparse_blocks), num_kv_blocks_per_page,
        (num_kv_blocks_per_page & (num_kv_blocks_per_page - 1)) == 0
            ? static_cast<uint32_t>(__builtin_ctz(num_kv_blocks_per_page)) : 32u,
        static_cast<uint32_t>(num_sms), static_cast<uint32_t>(block_table.stride(0)),
        reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(block_table.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(indices.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(sparse_kv_block_indices.data_ptr<int>()),
        metadata.data_ptr<uint8_t>(), workspace.data_ptr<uint8_t>(), paired_rows, selection};

    // About 4 merged blocks per thread, at most 1024 threads; M = 2048 with 2..63 tokens
    // is faster at 512 threads and 8 blocks each
    int64_t num_threads = 64;
    while (num_threads < 1024 and num_threads * 4 < num_max_merged_kv_blocks)
        num_threads *= 2;
    if (num_threads == 1024 and num_max_merged_kv_blocks <= 4096 and num_q_tokens > 1 and num_q_tokens < 64)
        num_threads = 512;
    if (num_q_tokens>=64 && num_max_sparse_blocks==2048) num_threads=512;
    if (thread_hint) {
        TORCH_CHECK(thread_hint==256 || thread_hint==512 || thread_hint==1024,"invalid metadata thread hint");
        num_threads=thread_hint;
    }
    const int64_t num_kv_blocks_per_thread = ceil_div_i64(num_max_merged_kv_blocks, num_threads);
    const size_t num_smem_bytes = static_cast<size_t>(
        ceil_div_i64(num_max_merged_kv_blocks + ceil_div_i64(num_max_merged_kv_blocks, num_kv_blocks_per_split), 4) * 16);

#define PSM_LAUNCH(SBK, BPS, BPT, THREADS)                                                                 \
    do {                                                                                                   \
        if (num_ctas > num_cus)                                                                            \
            sparse_fp8_fp4_paged_mqa_logits__prepare_metadata<SBK, BPS, BPT, THREADS, true>                          \
                <<<num_ctas, THREADS, num_smem_bytes, stream>>>(params);                                   \
        else                                                                                               \
            sparse_fp8_fp4_paged_mqa_logits__prepare_metadata<SBK, BPS, BPT, THREADS, false>                         \
                <<<num_ctas, THREADS, num_smem_bytes, stream>>>(params);                                   \
    } while (0)
#define PSM_DISPATCH_BPT(SBK, BPS, THREADS)                                                                \
    do {                                                                                                   \
        if (num_kv_blocks_per_thread <= 1) PSM_LAUNCH(SBK, BPS, 1, THREADS);                               \
        else if (num_kv_blocks_per_thread <= 2) PSM_LAUNCH(SBK, BPS, 2, THREADS);                          \
        else if (num_kv_blocks_per_thread <= 4) PSM_LAUNCH(SBK, BPS, 4, THREADS);                          \
        else if (num_kv_blocks_per_thread <= 8) PSM_LAUNCH(SBK, BPS, 8, THREADS); \
        else if (num_kv_blocks_per_thread <= 16) PSM_LAUNCH(SBK, BPS, 16, THREADS); \
        else PSM_LAUNCH(SBK, BPS, 32, THREADS);                                                             \
    } while (0)
#define PSM_DISPATCH_THREADS(SBK, BPS)                                                                     \
    do {                                                                                                   \
        switch (num_threads) {                                                                             \
            case 64: PSM_DISPATCH_BPT(SBK, BPS, 64); break;                                                \
            case 128: PSM_DISPATCH_BPT(SBK, BPS, 128); break;                                              \
            case 256: PSM_DISPATCH_BPT(SBK, BPS, 256); break;                                              \
            case 512: PSM_DISPATCH_BPT(SBK, BPS, 512); break;                                              \
            default: PSM_DISPATCH_BPT(SBK, BPS, 1024); break;                                              \
        }                                                                                                  \
    } while (0)
    if (sparse_block_kv == 8) {
        if (split_kv == 640) PSM_DISPATCH_THREADS(8, 80); else PSM_DISPATCH_THREADS(8, 64);
    } else {
        if (split_kv == 640) PSM_DISPATCH_THREADS(16, 40); else PSM_DISPATCH_THREADS(16, 32);
    }
#undef PSM_DISPATCH_THREADS
#undef PSM_DISPATCH_BPT
#undef PSM_LAUNCH
    TORCH_CHECK(hipGetLastError() == hipSuccess, "get_paged_sparse_mqa_logits_metadata: launch failed");
    return metadata;
}

torch::Tensor get_paged_sparse_mqa_logits_metadata(
    const torch::Tensor& lens,const torch::Tensor& table,const torch::Tensor& indices,
    int64_t page,const torch::Tensor& candidates,at::ScalarType dtype,
    int64_t block,std::optional<int64_t> sms) {
    return make_paged_sparse_mqa_logits_metadata(lens,table,indices,page,candidates,dtype,block,sms,
        std::nullopt,PrefillSparseMQASelection{},false,0);
}
void prepare_prefill_sparse_mqa_logits_metadata(
    const torch::Tensor& metadata,const PrefillSparseMQAMetadataPlan& p,
    int64_t page,at::ScalarType dtype,int64_t block,std::optional<int64_t> sms,
    const PrefillSparseMQASelection& selection) {
    make_paged_sparse_mqa_logits_metadata(p.context_lens,p.block_table,p.indices,page,p.candidates,
        dtype,block,sms,metadata,selection,p.paired_rows,p.threads);
}
}  // namespace lightop::deep_gemm
