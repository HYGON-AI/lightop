#pragma once

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#include "device_runtime.h"
#include "exception.h"
#include "utils.h"

namespace lightop {
namespace deep_gemm {

typedef __fp16 fp16VecType;
typedef fp16VecType fp16x2 __attribute__((ext_vector_type(2)));
typedef fp16VecType fp16x4 __attribute__((ext_vector_type(4)));
typedef fp16VecType fp16x8 __attribute__((ext_vector_type(8)));

typedef short v4bh __attribute__((ext_vector_type(4)));

typedef float fp32x2 __attribute__((ext_vector_type(2)));
typedef float fp32x4 __attribute__((ext_vector_type(4)));
typedef float fp32x8 __attribute__((ext_vector_type(8)));

typedef union
{
    fp16x8 data;
    struct
    {
        fp16x4 front;
        fp16x4 rear;
    };
} f16x8_t;

#define CLEAR_ACC(acc, x)               \
    for (int i = 0; i < x; ++i) {       \
        acc[i] = {0.0f};                \
    }

template<bool is_half = true>
inline __device__ void builtin_b16_mmac(const fp16x4& reg_a, const fp16x4& reg_b, fp32x4& reg_c) {
    #if defined(__gfx936__) || defined(__gfx928__)
    if constexpr (is_half) {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
    } else {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*) &reg_a, *(v4bh*) &reg_b, reg_c);
    }
    #endif
}

template <int kAlignedBatchSize, int SPLIT_KV, int kNumSMs>
__global__ void paged_mqa_logits_metadata(const int batch_size,
                                          const int* context_lens,
                                          int* schedule_metadata) {
    LIGHTOP_STATIC_ASSERT(kAlignedBatchSize % 64 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = get_lane_idx();

    uint32_t num_segs[kAlignedBatchSize / 64];
#pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 64; ++ k) {
        const int& context_len = (k * 64 + lane_idx < batch_size ? *(context_lens + k * 64 + lane_idx) : 0);
        num_segs[k] = ceil_div(context_len, SPLIT_KV);
    }

    __shared__ uint32_t prefix_sum[kAlignedBatchSize];
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 64; ++ k) {
        uint32_t x = num_segs[k];
#pragma unroll
        for (uint32_t offset = 1; offset < 64; offset <<= 1) {
            const uint32_t& y = __shfl_up_sync(0xffffffffffffffff, x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 64 + lane_idx] = x;
        sum = __shfl_sync(0xffffffffffffffff, x, 63);
    }

    const uint32_t& q = sum / kNumSMs, r = sum % kNumSMs;
    for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 64) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
        int q_idx = 0;
        while (q_idx < batch_size and prefix_sum[q_idx] <= seg_starts)
            ++ q_idx;
        const int& kv_split_idx = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1]);
        __syncthreads();

        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

template <int BLOCK_KV, int kNumWarps>
struct PagedMQALogitsScheduler {
    int batch_size;
    const int* context_lens;

    int current_q_idx, current_kv_idx;
    int end_q_idx, end_kv_idx;
    int current_num_kv;
    int current_context_len;

    __device__ __forceinline__ explicit PagedMQALogitsScheduler(const int& batch_size, const int& sm_idx,
                                                                const int* context_lens, const int* schedule_meta) {
        this->batch_size = batch_size;
        this->context_lens = context_lens;

        const auto& current_pack = *(reinterpret_cast<const int2*>(schedule_meta) + sm_idx);
        const auto& end_pack = *(reinterpret_cast<const int2*>(schedule_meta) + sm_idx + 1);
        current_q_idx = current_pack.x, current_kv_idx = current_pack.y * kNumWarps;
        end_q_idx = end_pack.x, end_kv_idx = end_pack.y * kNumWarps;
        current_context_len = current_q_idx < batch_size ? *(this->context_lens + current_q_idx) : 0;
        current_num_kv = current_context_len ? ceil_div(current_context_len, BLOCK_KV) : 0;
    }

    __device__ __forceinline__ bool fetch_next_task(int &q_idx, int &kv_idx, int &num_kv, int &context_len) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;
        context_len = current_context_len;

        if (q_idx == end_q_idx and kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumWarps;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            current_context_len = current_q_idx < batch_size ? *(this->context_lens + current_q_idx) : 0;
            current_num_kv = current_context_len ? ceil_div(current_context_len, BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const int& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }
};

template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits(const Element* q,
                                 const Element* kv_block,
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

    static constexpr int K_TILE = 32;
    static constexpr int Q_TILE = 16;
    static constexpr int Stages = kHeadDim / K_TILE;

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr bool is_half = std::is_same<Element, at::Half>::value;

    auto* smem_kv_block = reinterpret_cast<Element*>(smem_buffer);

    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, 1>(batch_size, blockIdx.x / kNextN, context_lens, schedule_meta);
    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
    int q_idx = batch_size, kv_idx, num_kv;
    int next_q_idx, next_kv_idx, next_num_kv;
    int context_len, next_context_len;
    bool fetched_next_task;

    // Prefetch the first Q
    fetched_next_task =
             scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv, next_context_len);

    if (!fetched_next_task) {
        return;
    }

    // fetch Q && Q weights...
    auto gQ = q + next_q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(Element);

    constexpr int heads_cnt = kNumHeads / Q_TILE;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f16x8_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 8;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            reg_q_operand[i][j] = *reinterpret_cast<const f16x8_t *>(&gQ[offset]);
        }
    }

    auto gW = weights + next_q_idx * kNextN * kNumHeads + next_n * kNumHeads;

    // pre-fetch weights...
    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_weights[j * 4 + k] = gW[j * 16 + k * 4 + lane_idx / 16];
        }
    }

    q_idx = next_q_idx;
    kv_idx = next_kv_idx;
    num_kv = next_num_kv;
    context_len = next_context_len;

    // fetch kv block idx...
    uint64_t kv_block_idx;

    uint32_t block_table_offset = q_idx * block_table_stride + kv_idx;
    kv_block_idx = kv_idx < num_kv ? *(block_table + block_table_offset) : 0;

    // fetch kv block...
    auto gKv = kv_block + kv_block_idx * BLOCK_KV * kHeadDim;
    constexpr int kAlignmentKV = 16 / sizeof(Element);
    constexpr int fetch_kv_block_cnt = BLOCK_KV * K_TILE / kNumWarps / WARP_SIZE_GPU / kAlignmentKV;
    constexpr int fetch_kv_block_stride = kNumWarps * WARP_SIZE_GPU / (K_TILE / kAlignmentKV) * kHeadDim;

    #pragma unroll
    for (int i = 0; i < Stages; ++i) {
        #pragma unroll
        for (int j = 0; j < fetch_kv_block_cnt; ++j) {
            auto pos = t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + j * fetch_kv_block_stride + i * K_TILE;
            *reinterpret_cast<fp16x8*>(&smem_kv_block[pos]) = *reinterpret_cast<const fp16x8*>(&gKv[pos]);
        }
    }

    __syncthreads();

    // pre-fetch KV from smem...
    f16x8_t reg_kv_operand;
    auto base_fetch_lds_KV = warp_idx * 16 * kHeadDim + lane_idx % 16 * kHeadDim + lane_idx / 16 * 8;

    fp32x4 acc[kNumHeads / Q_TILE];
    CLEAR_ACC(acc, kNumHeads / Q_TILE);
    const int logits_t_offset = warp_idx * 16 + lane_idx % 16;

    while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv, next_context_len)) {
        // fetch next kv block idx...
        block_table_offset = next_q_idx * block_table_stride + next_kv_idx;
        kv_block_idx = next_kv_idx < num_kv ? *(block_table + block_table_offset) : 0;
        gKv = kv_block + kv_block_idx * BLOCK_KV * kHeadDim;

        // Calculate logits KV block offset in advance
        auto base_logits_offset = q_idx * kNextN * logits_stride;
        auto base_seq_kv_offset = kv_idx * BLOCK_KV;

        #pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            // pre-fetch KV from smem...
            reg_kv_operand = *reinterpret_cast<f16x8_t *>(&smem_kv_block[base_fetch_lds_KV + k_block * K_TILE]);

            __syncthreads();

            // pre-fetch next kv block per stage
            #pragma unroll
            for (int j = 0; j < fetch_kv_block_cnt; ++j) {
                auto pos = t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + j * fetch_kv_block_stride + k_block * K_TILE;
                *reinterpret_cast<fp16x8*>(&smem_kv_block[pos]) = *reinterpret_cast<const fp16x8*>(&gKv[pos]);
            }

            #pragma unroll
            for (int i = 0; i < kHeadDim / Q_TILE; ++i) {
                builtin_b16_mmac<is_half>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i]);
                builtin_b16_mmac<is_half>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i]);
            }
        }

        const auto& transform = [&](const uint32_t& j, const uint32_t& k, const float& f) {
            return fmaxf(f, 0) * reg_weights[j * 4 + k];
        };
        float sum = 0.0f;
        // Intra-thread reduction
        for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
            for (int k = 0; k < 4; ++k) {
                sum += transform(j, k, acc[j][k]);
            }
        }

        // Inter-thread reduction
        #pragma unroll
        for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
            sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
        }

        // Store into the global memory
        // NOTES: we have redundant writes here, consider more carefully

        auto seq_kv_offset = base_seq_kv_offset + logits_t_offset;
        if (lane_idx < 16 && seq_kv_offset < context_len - (kNextN - next_n) + 1) {
            logits[base_logits_offset + next_n * logits_stride + seq_kv_offset] = sum;
        }

        // Prefetch next Q when current Q changes
        if (q_idx != next_q_idx) {
            if (scheduler.exist_q_idx(next_q_idx)) {
                gQ = q + next_q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
                #pragma unroll
                for (int i = 0; i < heads_cnt; ++i) {
                    #pragma unroll
                    for (int j = 0; j < K_BLOCK_MAX; ++j) {
                        auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
                        reg_q_operand[i][j] = *reinterpret_cast<const f16x8_t *>(&gQ[offset]);
                    }
                }

                gW = weights + next_q_idx * kNextN * kNumHeads + next_n * kNumHeads;
                // pre-fetch weights...
                #pragma unroll
                for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        reg_weights[j * 4 + k] = gW[j * 16 + k * 4 + lane_idx / 16];
                    }
                }
            }
        }

        // Fetch next task, reload idx to launch next mmac
        q_idx = next_q_idx;
        kv_idx = next_kv_idx;
        num_kv = next_num_kv;
        context_len = next_context_len;

        CLEAR_ACC(acc, kHeadDim / Q_TILE);
    }

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;

    #pragma unroll
    for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        // pre-fetch KV from smem...
        reg_kv_operand = *reinterpret_cast<f16x8_t *>(&smem_kv_block[base_fetch_lds_KV + k_block * K_TILE]);

        #pragma unroll
        for (int i = 0; i < kHeadDim / Q_TILE; ++i) {
            builtin_b16_mmac<is_half>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i]);
            builtin_b16_mmac<is_half>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i]);
        }
    }

    const auto& transform = [&](const uint32_t& j, const uint32_t& k, const float& f) {
        return fmaxf(f, 0) * reg_weights[j * 4 + k];
    };
    float sum = 0.0f;
    // Intra-thread reduction
    for (int j = 0; j < kNumHeads / 16; ++j) {
        for (int k = 0; k < 4; ++k) {
            sum += transform(j, k, acc[j][k]);
        }
    }

    // Inter-thread reduction
    for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
        sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
    }

    auto seq_kv_offset = base_seq_kv_offset + logits_t_offset;
    if (lane_idx < 16 && seq_kv_offset < context_len - (kNextN - next_n) + 1) {
        logits[base_logits_offset + next_n * logits_stride + seq_kv_offset] = sum;
    }
}

template<int kNextN, int kNumWarps>
__global__ __launch_bounds__(kNumWarps * 64, 1)
void internal_clean_logits(const int seq_len, const int seq_len_kv, const uint64_t stride_kv,
                           const int* cu_seq_len_k_start, const int* cu_seq_len_k_end, float* logits) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const fp32x4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
    const fp32x2 neg_infx2 = {neg_inf, neg_inf};

    const int num_threads_per_block = kNumWarps * 64;
    const int tid = threadIdx.x;

    for (int i = blockIdx.x; i < seq_len; i += gridDim.x) {
        const int row_group = i / kNextN;
        const int ks = (cu_seq_len_k_start == nullptr) ? 0 : cu_seq_len_k_start[row_group];
        const int ke = cu_seq_len_k_end[row_group] - kNextN + (i % kNextN) + 1;
        const uint64_t row_base_addr = i * stride_kv;
        for (int j = tid; j < ks; j += num_threads_per_block) {
            logits[row_base_addr + j] = neg_inf;
        }
        const int num_elements_to_clean = seq_len_kv - ke;
        if (ke % 4 == 0 && num_elements_to_clean % 4 == 0) {
            const int num_f4_chunks = num_elements_to_clean / 4;
            fp32x4* start_ptr_f4 = reinterpret_cast<fp32x4*>(&logits[row_base_addr + ke]);
            for (int j_f4 = tid; j_f4 < num_f4_chunks; j_f4 += num_threads_per_block) {
                start_ptr_f4[j_f4] = neg_infx4;
            }
        } else if (ke % 2 == 0 && num_elements_to_clean % 2 == 0) {
            const int num_f2_chunks = num_elements_to_clean / 2;
            fp32x2* start_ptr_f2 = reinterpret_cast<fp32x2*>(&logits[row_base_addr + ke]);
            for (int j_f2 = tid; j_f2 < num_f2_chunks; j_f2 += num_threads_per_block) {
                start_ptr_f2[j_f2] = neg_infx2;
            }
        } else {
            for (int j = ke + tid; j < seq_len_kv; j += num_threads_per_block) {
                logits[row_base_addr + j] = neg_inf;
            }
        }
    }
}

}
}