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

using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;

typedef uint8_t fp8_t;
typedef fp8_t fp8x2 __attribute__((ext_vector_type(2)));   
typedef fp8_t fp8x4 __attribute__((ext_vector_type(4)));   
typedef fp8_t fp8x8 __attribute__((ext_vector_type(8)));   
typedef fp8_t fp8x16 __attribute__((ext_vector_type(16))); 

typedef short v4bh __attribute__((ext_vector_type(4)));

typedef float fp32x2 __attribute__((ext_vector_type(2)));
typedef float fp32x4 __attribute__((ext_vector_type(4)));
typedef float fp32x8 __attribute__((ext_vector_type(8)));

typedef int v4i __attribute__((ext_vector_type(4)));

typedef union
{
    fp16x8 data;
    struct
    {
        fp16x4 front;
        fp16x4 rear;
    };
} f16x8_t;

typedef union
{
    fp8x16 data;
    struct
    {
        fp8x8 front;
        fp8x8 rear;
    };
} f8x16_t;

#define DIRECT_LDS_WORDx4 16
#define DIRECT_LDS_WORD 4

#define WAIT_VMCNT_LDS(X)               \
    __builtin_amdgcn_sched_barrier(0);  \
    asm volatile(                       \
    "s_waitcnt vmcnt(%0)\n\t"           \
    "s_barrier\n"                       \
    :: "I"(X)                           \
    :);                                 \
    __builtin_amdgcn_sched_barrier(0);

#define CLEAR_ACC(acc, x)               \
    for (int i = 0; i < x; ++i) {       \
        acc[i] = {0.0f};                \
    }

template<bool is_half = true>
inline __device__ void builtin_b16_mmac(const fp16x4& reg_a, const fp16x4& reg_b, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_half) {
        reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
    }
    #elif defined(__gfx936__) || (__gfx928__)
    if constexpr (is_half) {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
    } else {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*) &reg_a, *(v4bh*) &reg_b, reg_c);
    }
    #endif
}

template<bool is_e4m3 = true>
inline __device__ void builtin_fp8_mmac(const fp8x8& reg_a, const fp8x8& reg_b, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_e4m3) {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(*(intx2*) &reg_a,*(intx2*) &reg_b, reg_c, false, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(*(intx2*) &reg_a,*(intx2*) &reg_b, reg_c, false, false);
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

        if ((q_idx == end_q_idx and kv_idx == end_kv_idx) || q_idx >= this->batch_size)
            return false;

        current_kv_idx += kNumWarps;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            // current_context_len = current_q_idx < batch_size ? *(this->context_lens + current_q_idx) : 0;
            current_context_len = 0;
            if (current_q_idx < batch_size) {
                const auto* context_len_ptr = this->context_lens + current_q_idx;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(context_len_ptr),
                    "=s"(current_context_len));
            }
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

    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    int q_idx = blockIdx.y;
    int kv_idx = blockIdx.x / kNextN;
    int context_len = context_lens[q_idx];
    int num_kv = ceil_div(context_len, BLOCK_KV);

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;

    if (kv_idx >= num_kv) {
        if (warp_idx == 0) {
            logits[base_logits_offset + base_seq_kv_offset + lane_idx] = -INFINITY;
        }
        return;
    }

    // fetch Q && Q weights...
    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(Element);
    constexpr int fetch_q_cnt = kNumHeads * kHeadDim / kNumWarps / WARP_SIZE_GPU / kAlignmentQ;
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        *reinterpret_cast<f16x8_t *>(&smem_kv_block[pos]) = *reinterpret_cast<const f16x8_t *>(&gQ[pos]);
    }

    constexpr int heads_cnt = kNumHeads / Q_TILE;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f16x8_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem...
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 8;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            reg_q_operand[i][j] = *reinterpret_cast<f16x8_t *>(&smem_kv_block[offset]);
        }
    }
    __syncthreads();

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float *>(smem_kv_block);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    __syncthreads();
    // pre-fetch weights...
    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_weights[j * 4 + k] = smem_weight[j * 16 + k * 4 + lane_idx / 16];
        }
    }
    __syncthreads();

    // fetch kv block idx...
    uint32_t kv_block_idx;

    uint32_t block_table_offset = q_idx * block_table_stride + kv_idx;
    kv_block_idx = kv_idx < num_kv ? *(block_table + block_table_offset) : 0;

    // fetch kv block...
    auto gKv = kv_block + kv_block_idx * BLOCK_KV * kHeadDim;
    constexpr int kAlignmentKV = 16 / sizeof(Element);
    constexpr int fetch_kv_block_cnt = BLOCK_KV * K_TILE / kNumWarps / WARP_SIZE_GPU / kAlignmentKV;
    constexpr int fetch_kv_block_stride = kNumWarps * WARP_SIZE_GPU / (K_TILE / kAlignmentKV) * kHeadDim;

    long glob_kv_desc[2];

    glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
    // Use offset enable mode, it's unnecessary to set stride args.
    long stride = 0x0;
    glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
    glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
    auto *src = reinterpret_cast<v4i *>(glob_kv_desc);

    #pragma unroll
    for (int i = 0; i < Stages; ++i) {
        auto smem_kv_stage = smem_kv_block + i * BLOCK_KV * K_TILE + warp_idx * kAlignmentKV * WARP_SIZE_GPU;
        #pragma unroll
        for (int j = 0; j < fetch_kv_block_cnt; ++j) {
            auto inner_warp_offset =
                (t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + i * K_TILE) * sizeof(__fp16);
            #if defined(__gfx936__) || defined(__gfx938__)
            __builtin_amdgcn_raw_buffer_load_lds(*src,
                                                 // Cast address type to which compiler can recognize as lds type.
                                                 *(__attribute__((address_space(3))) int**) (&smem_kv_stage),
                                                 DIRECT_LDS_WORDx4,
                                                 inner_warp_offset,
                                                 0,
                                                 0,
                                                 0);
            #endif
        }
    }

    // pre-fetch KV from smem...
    f16x8_t reg_kv_operand;
    auto base_fetch_lds_KV = warp_idx * 16 * K_TILE + lane_idx % 16 * K_TILE + lane_idx / 16 * 8;

    fp32x4 acc[kNumHeads / Q_TILE];
    CLEAR_ACC(acc, kNumHeads / Q_TILE);
    const int logits_t_offset = warp_idx * 16 + lane_idx % 16;
    uint32_t s_kv_block_idx;

    #pragma unroll
    for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        auto fetch_lds_kv = base_fetch_lds_KV + k_block * BLOCK_KV * K_TILE;
        WAIT_VMCNT_LDS(Stages - k_block - 1);

        // pre-fetch KV from smem...
        int ldsAddr = reinterpret_cast<size_t>(&smem_kv_block[fetch_lds_kv]) ;
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "\n ds_read_b128 %0 ,%1\n\t"
            "s_waitcnt lgkmcnt(0) \n\t"
            "s_barrier \n\t"
            : "=v"(reg_kv_operand)
            : "v"(ldsAddr)
            :);
        __builtin_amdgcn_sched_barrier(0);

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

    auto seq_kv_offset = base_seq_kv_offset + logits_t_offset;
    if (lane_idx < 16) {
        if (seq_kv_offset < context_len - (kNextN - next_n) + 1) {
            logits[base_logits_offset + seq_kv_offset] = sum;
        } else {
            logits[base_logits_offset + seq_kv_offset] = -INFINITY;
        }
    }
}

template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_persistent_schedule(const Element* q,
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
    constexpr int fetch_q_cnt = kNumHeads * kHeadDim / kNumWarps / WARP_SIZE_GPU / kAlignmentQ;
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        *reinterpret_cast<f16x8_t *>(&smem_kv_block[pos]) = *reinterpret_cast<const f16x8_t *>(&gQ[pos]);
    }

    constexpr int heads_cnt = kNumHeads / Q_TILE;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f16x8_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem...
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 8;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            reg_q_operand[i][j] = *reinterpret_cast<f16x8_t *>(&smem_kv_block[offset]);
        }
    }
    __syncthreads();

    auto gW = weights + next_q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float *>(smem_kv_block);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    __syncthreads();
    // pre-fetch weights...
    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_weights[j * 4 + k] = smem_weight[j * 16 + k * 4 + lane_idx / 16];
        }
    }
    __syncthreads();

    q_idx = next_q_idx;
    kv_idx = next_kv_idx;
    num_kv = next_num_kv;
    context_len = next_context_len;

    // fetch kv block idx...
    uint32_t kv_block_idx;

    uint32_t block_table_offset = q_idx * block_table_stride + kv_idx;
    kv_block_idx = kv_idx < num_kv ? *(block_table + block_table_offset) : 0;

    // fetch kv block...
    auto gKv = kv_block + kv_block_idx * BLOCK_KV * kHeadDim;
    constexpr int kAlignmentKV = 16 / sizeof(Element);
    constexpr int fetch_kv_block_cnt = BLOCK_KV * K_TILE / kNumWarps / WARP_SIZE_GPU / kAlignmentKV;
    constexpr int fetch_kv_block_stride = kNumWarps * WARP_SIZE_GPU / (K_TILE / kAlignmentKV) * kHeadDim;

    long glob_kv_desc[2];

    glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
    // Use offset enable mode, it's unnecessary to set stride args.
    long stride = 0x0;
    glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
    glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
    auto *src = reinterpret_cast<v4i *>(glob_kv_desc);

    #pragma unroll
    for (int i = 0; i < Stages; ++i) {
        auto smem_kv_stage = smem_kv_block + i * BLOCK_KV * K_TILE + warp_idx * kAlignmentKV * WARP_SIZE_GPU;
        #pragma unroll
        for (int j = 0; j < fetch_kv_block_cnt; ++j) {
            auto inner_warp_offset =
                (t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + i * K_TILE) * sizeof(__fp16);
            #if defined(__gfx936__) || defined(__gfx938__)
            __builtin_amdgcn_raw_buffer_load_lds(*src,
                                                 // Cast address type to which compiler can recognize as lds type.
                                                 *(__attribute__((address_space(3))) int**) (&smem_kv_stage),
                                                 DIRECT_LDS_WORDx4,
                                                 inner_warp_offset,
                                                 0,
                                                 0,
                                                 0);
            #endif
        }
    }

    // pre-fetch KV from smem...
    f16x8_t reg_kv_operand;
    auto base_fetch_lds_KV = warp_idx * 16 * K_TILE + lane_idx % 16 * K_TILE + lane_idx / 16 * 8;

    fp32x4 acc[kNumHeads / Q_TILE];
    CLEAR_ACC(acc, kNumHeads / Q_TILE);
    const int logits_t_offset = warp_idx * 16 + lane_idx % 16;
    uint32_t s_kv_block_idx;

    while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv, next_context_len)) {
        // fetch next kv block idx...
        block_table_offset = next_q_idx * block_table_stride + next_kv_idx;

        s_kv_block_idx = 0;
        if (next_kv_idx < num_kv) {
            const auto* block_id_ptr = block_table + block_table_offset;
            asm volatile("s_load_dword %1, %0, 0x0\n\t"
                "s_waitcnt lgkmcnt(0)\n\t":
                "+s"(block_id_ptr),
                "=s"(s_kv_block_idx));
        }
        gKv = kv_block + s_kv_block_idx * BLOCK_KV * kHeadDim;

        glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
        glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
        glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
        src = reinterpret_cast<v4i *>(glob_kv_desc);

        // Calculate logits KV block offset in advance
        auto base_logits_offset = q_idx * kNextN * logits_stride;
        auto base_seq_kv_offset = kv_idx * BLOCK_KV;

        #pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * BLOCK_KV * K_TILE;
            WAIT_VMCNT_LDS(Stages - 1);

            // pre-fetch KV from smem...
            int ldsAddr = reinterpret_cast<size_t>(&smem_kv_block[fetch_lds_kv]) ;
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "\n ds_read_b128 %0 ,%1\n\t"
                "s_waitcnt lgkmcnt(0) \n\t"
                "s_barrier \n\t"
                : "=v"(reg_kv_operand)
                : "v"(ldsAddr)
                :);
            __builtin_amdgcn_sched_barrier(0);

            auto smem_kv_stage = smem_kv_block + k_block * BLOCK_KV * K_TILE + warp_idx * kAlignmentKV * WARP_SIZE_GPU;
            #pragma unroll
            for (int j = 0; j < fetch_kv_block_cnt; ++j) {
                auto inner_warp_offset =
                        (t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + k_block * K_TILE) * sizeof(__fp16);
                #if defined(__gfx936__) || defined(__gfx938__)
                __builtin_amdgcn_raw_buffer_load_lds(*src,
                                                     // Cast address type to which compiler can recognize as lds type.
                                                     *(__attribute__((address_space(3))) int**) (&smem_kv_stage),
                                                     DIRECT_LDS_WORDx4,
                                                     inner_warp_offset,
                                                     0,
                                                     0,
                                                     0);
                #endif
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

            __syncthreads();
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
        auto fetch_lds_kv = base_fetch_lds_KV + k_block * BLOCK_KV * K_TILE;
        WAIT_VMCNT_LDS(0);
        // pre-fetch KV from smem...
        reg_kv_operand = *reinterpret_cast<f16x8_t *>(&smem_kv_block[fetch_lds_kv]);

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

template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4>
__attribute__((amdgpu_flat_work_group_size(1, 512)))
__global__ void paged_mqa_logits_fp8(const Element* q,
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
    static constexpr int Stages = kHeadDim / K_TILE;

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr bool is_e4m3 = std::is_same<Element, at::Float8_e4m3fn>::value;

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    int q_idx = blockIdx.y;
    int kv_idx = blockIdx.x / kNextN;
    int context_len = context_lens[q_idx];
    int num_kv = ceil_div(context_len, BLOCK_KV);

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;

    if (kv_idx >= num_kv) {
        if (warp_idx == 0) {
            logits[base_logits_offset + base_seq_kv_offset + lane_idx] = -INFINITY;
        }
        return;
    }

    // fetch Q && Q weights
    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(uint8_t);

    constexpr int fetch_q_cnt = kNumHeads * kHeadDim / kNumWarps / WARP_SIZE_GPU / kAlignmentQ;
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        *reinterpret_cast<f8x16_t *>(&smem_kv_block[pos]) = *reinterpret_cast<const f8x16_t *>(&gQ[pos]);
    }

    constexpr int heads_cnt = kNumHeads / Q_TILE;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem fix
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = (base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE) * sizeof(uint8_t);
            reg_q_operand[i][j] = *reinterpret_cast<f8x16_t *>(&smem_kv_block[offset]);
        }
    }
    __syncthreads();

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float *>(smem_kv_block);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    __syncthreads();
    // pre-fetch weights
    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < kNumHeads / Q_TILE; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_weights[j * 4 + k] = smem_weight[j * 16 + k * 4 + lane_idx / 16];
        }
    }
    __syncthreads();

    // fetch kv block idx
    uint32_t kv_block_idx;

    uint32_t block_table_offset = q_idx * block_table_stride + kv_idx;
    kv_block_idx = kv_idx < num_kv ? *(block_table + block_table_offset) : 0;

    // fetch kv block
    auto gKv = kv_block + kv_block_idx * BLOCK_KV * (kHeadDim + 4);
    constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
    constexpr int fetch_kv_block_cnt = BLOCK_KV * K_TILE / kNumWarps / WARP_SIZE_GPU / kAlignmentKV;
    constexpr int fetch_kv_block_stride = kNumWarps * WARP_SIZE_GPU / (K_TILE / kAlignmentKV) * kHeadDim;

    long glob_kv_desc[2];

    glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
    // Use offset enable mode, it's unnecessary to set stride args.
    long stride = 0x0;
    glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
    glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
    auto *src = reinterpret_cast<v4i *>(glob_kv_desc);

    #pragma unroll
    for (int i = 0; i < Stages; ++i) {
        auto smem_kv_stage = smem_kv_block + i * BLOCK_KV * K_TILE + warp_idx * kAlignmentKV * WARP_SIZE_GPU;
        #pragma unroll
        for (int j = 0; j < fetch_kv_block_cnt; ++j) {
            auto inner_warp_offset =
                (t_id % 4 * kAlignmentKV + t_id / 4 * kHeadDim + i * K_TILE) * sizeof(uint8_t);
            #if defined(__gfx938__)
            __builtin_amdgcn_raw_buffer_load_lds(*src,
                                                 // Cast address type to which compiler can recognize as lds type.
                                                 *(__attribute__((address_space(3))) int**) (&smem_kv_stage),
                                                 DIRECT_LDS_WORDx4,
                                                 inner_warp_offset,
                                                 0,
                                                 0,
                                                 0);
            #endif
        }
    }

    // pre-fetch KV from smem
    f8x16_t reg_kv_operand;

    auto base_fetch_lds_KV = warp_idx * 16 * K_TILE + lane_idx % 16 * K_TILE + lane_idx / 16 * 16;

    fp32x4 acc[kNumHeads / Q_TILE];
    CLEAR_ACC(acc, kNumHeads / Q_TILE);
    const int logits_t_offset = warp_idx * 16 + lane_idx % 16;
    uint32_t s_kv_block_idx;

    // fetch kv block scales
    const int kv_scale_stride = kv_cache_stride_bytes / sizeof(float);
    auto gKv_scales = kv_block_scales + kv_block_idx * kv_scale_stride;

    auto KV_BYTES = Stages * BLOCK_KV * K_TILE * sizeof(uint8_t);
    auto* smem_kv_scales = reinterpret_cast<float*>(smem_kv_block + KV_BYTES);

    #if defined(__gfx938__)
    long glob_kv_scale_desc[2];
    glob_kv_scale_desc[0] = *reinterpret_cast<const long *>(&gKv_scales);
    long stride_scale = 0x0;
    glob_kv_scale_desc[0] = glob_kv_scale_desc[0] | stride_scale << 48;
    glob_kv_scale_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
    auto *src_scale = reinterpret_cast<v4i *>(glob_kv_scale_desc);
    auto inner_warp_offset = lane_idx * sizeof(float);
    if (warp_idx == 0) {
          __builtin_amdgcn_raw_buffer_load_lds(*src_scale,
                                          // Cast address type to which compiler can recognize as lds type.
                                          *(__attribute__((address_space(3))) int**) (&smem_kv_scales),
                                          DIRECT_LDS_WORD,
                                          inner_warp_offset,
                                          0,
                                          0,
                                          0);

    }
    #endif

    float kv_scale;

#pragma unroll
    for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

        auto fetch_lds_kv = (base_fetch_lds_KV + k_block * BLOCK_KV * K_TILE) * sizeof(uint8_t);
        WAIT_VMCNT_LDS(Stages - k_block - 1);

        // pre-fetch KV from smem
        int ldsAddr = reinterpret_cast<size_t>(&smem_kv_block[fetch_lds_kv]) ;
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "\n ds_read_b128 %0 ,%1\n\t"
            "s_waitcnt lgkmcnt(0) \n\t"
            "s_barrier \n\t"
            : "=v"(reg_kv_operand)
            : "v"(ldsAddr)
            :);
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (int i = 0; i < kHeadDim / Q_TILE; ++i) {
            builtin_fp8_mmac<is_e4m3>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i]);
            builtin_fp8_mmac<is_e4m3>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i]);
        }
    }

    int ldsaddr_kv_scale = reinterpret_cast<size_t>(&smem_kv_scales[logits_t_offset]);
    asm volatile(
        "ds_read_b32 %0 ,%1"
        : "=v"(kv_scale)
        : "v"(ldsaddr_kv_scale)
        :);

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

    asm volatile("s_waitcnt lgkmcnt(0); s_barrier");

    sum *= kv_scale;

    // Inter-thread reduction
    #pragma unroll
    for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
        sum += __shfl_down_sync(0xffffffffffffffff, sum, shfl_idx);
    }

    // Store into the global memory
    auto seq_kv_offset = base_seq_kv_offset + logits_t_offset;
    if (lane_idx < 16) {
        if (seq_kv_offset < context_len - (kNextN - next_n) + 1) {
            logits[base_logits_offset + seq_kv_offset] = sum;
        } else {
            logits[base_logits_offset + seq_kv_offset] = -INFINITY;
        }
    }
}

}
}