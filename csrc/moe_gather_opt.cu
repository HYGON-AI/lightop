// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops custom_moe_gather.hip + custom_moe_gather_v2_hip.hip for lightop.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(condition)                                                                \
    {                                                                                       \
        cudaError_t error = condition;                                                       \
        if (error != cudaSuccess) {                                                          \
            fprintf(stderr, "HIP error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, \
                    __LINE__);                                                              \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }
#endif

using bfloat16 = __hip_bfloat16;
struct Bf16Vec8 {
    bfloat16 val[8];
};

__device__ __forceinline__ float bf16_to_float(bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ Bf16Vec8 load_bf16_vec8(const bfloat16* ptr) {
    Bf16Vec8 v;
    *reinterpret_cast<uint4*>(v.val) = *reinterpret_cast<const uint4*>(ptr);
    return v;
}

__device__ __forceinline__ void store_bf16_vec8(bfloat16* ptr, const Bf16Vec8& v) {
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(v.val);
}

__device__ __forceinline__ bfloat16 weighted_bf16_contrib(bfloat16 v, float weight) {
    return float_to_bf16(fmaf(bf16_to_float(v), weight, 0.f));
}

__device__ __forceinline__ void vec8_load_fp32_from_bf16(float acc[8], const bfloat16* ptr) {
    const Bf16Vec8 v = load_bf16_vec8(ptr);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc[k] = bf16_to_float(v.val[k]);
    }
}

__device__ __forceinline__ void vec8_store_bf16_from_fp32(bfloat16* ptr, const float acc[8]) {
    Bf16Vec8 out;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        out.val[k] = float_to_bf16(acc[k]);
    }
    store_bf16_vec8(ptr, out);
}

__device__ __forceinline__ void vec8_golden_accumulate_fp32(
    float acc[8],
    const Bf16Vec8& src,
    float weight) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const bfloat16 contrib = weighted_bf16_contrib(src.val[k], weight);
        acc[k] = bf16_to_float(float_to_bf16(acc[k] + bf16_to_float(contrib)));
    }
}

__device__ __forceinline__ void vec8_store_weighted_bf16(
    bfloat16* dst,
    const Bf16Vec8& src,
    float weight) {
    Bf16Vec8 out;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        out.val[k] = weighted_bf16_contrib(src.val[k], weight);
    }
    store_bf16_vec8(dst, out);
}

__global__ void moe_gather_residual_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ residual,
    int res_row_offset,
    int hidden_size,
    int num_out_rows,
    int residual_rows,
    float res_scale) {
    const int local_row = blockIdx.x;
    if (local_row >= num_out_rows) {
        return;
    }

    const int64_t global_row = static_cast<int64_t>(res_row_offset) + local_row;
    bfloat16* out_row = convergent + global_row * static_cast<int64_t>(hidden_size);
    const bfloat16* res_row =
        (residual_rows == 1)
            ? residual
            : (residual + static_cast<int64_t>(local_row) * hidden_size);

    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        const Bf16Vec8 dv = load_bf16_vec8(out_row + i);
        const Bf16Vec8 rv = load_bf16_vec8(res_row + i);
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] = fmaf(res_scale, bf16_to_float(rv.val[k]), bf16_to_float(dv.val[k]));
        }
        vec8_store_bf16_from_fp32(out_row + i, acc);
    }
}

// Static, hyparam config
static constexpr int kEpRowsPerBlock = 32;      // 每个 block 处理的 scatter 行数
static constexpr int kEpPingPongRows = 2;
static constexpr int kInvPermMaxTopk = 8;
static constexpr int kInvPermTokensPerBlock = 4;
static constexpr int kMaxGatherThreads = 512;
static constexpr int kWavefrontSize = 64;
#define GRID_RESONANCE_STRIDE 4096
#define GRID_RESONANCE_REM 2048
#define GRID_REMAP_MULT 4097

__device__ __forceinline__ int32_t moe_gather_map_block_idx(int32_t block_idx, int32_t block_count) {
    if (block_count <= 0 || (block_count % GRID_RESONANCE_STRIDE) != GRID_RESONANCE_REM) {
        return block_idx;
    }
    return static_cast<int32_t>(
        (static_cast<uint64_t>(block_idx) * GRID_REMAP_MULT) % static_cast<uint32_t>(block_count));
}

static int moe_gather_round_up_wavefront(int threads) {
    threads = (threads + kWavefrontSize - 1) / kWavefrontSize * kWavefrontSize;
    return threads < kWavefrontSize ? kWavefrontSize : threads;
}

static size_t moe_gather_ep_smem_bytes(int hidden_size) {
    const size_t meta_bytes =
        static_cast<size_t>(kEpRowsPerBlock) * (sizeof(int) + sizeof(float));
    const size_t row_bytes = static_cast<size_t>(kEpPingPongRows) *
                             static_cast<size_t>(hidden_size) * sizeof(bfloat16);
    return meta_bytes + row_bytes;
}

struct MoeGatherEpSmem {
    int* token_id;
    float* weight;
    bfloat16* row_ping;
    bfloat16* row_pong;

    __device__ static MoeGatherEpSmem from_raw(unsigned char* raw, int hidden_size) {
        const int64_t h = static_cast<int64_t>(hidden_size);
        int* ids = reinterpret_cast<int*>(raw);
        float* w = reinterpret_cast<float*>(ids + kEpRowsPerBlock);
        bfloat16* ping = reinterpret_cast<bfloat16*>(w + kEpRowsPerBlock);
        return MoeGatherEpSmem{ids, w, ping, ping + h};
    }

    __device__ bfloat16* row_slot(int slot) const {
        return (slot & 1) ? row_pong : row_ping;
    }
};

// Set func for smem sanity check
static void moe_gather_set_dynamic_smem(const void* kernel_func, size_t smem_bytes) {
    if (smem_bytes <= 49152) {
        return;
    }
    CUDA_CHECK(hipFuncSetAttribute(
        kernel_func, hipFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
}

__device__ __forceinline__ void moe_gather_vec8_copy_row_to_smem(
    bfloat16* __restrict__ dst,
    const bfloat16* __restrict__ src,
    int hidden_size) {
    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        store_bf16_vec8(dst + i, load_bf16_vec8(src + i));
    }
}

__device__ __forceinline__ void moe_gather_ep_stage_batch_metadata(
    int* __restrict__ smem_ids,
    float* __restrict__ smem_weights,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    int batch_base,
    int batch_rows) 
    {
        for (int r = threadIdx.x; r < batch_rows; r += blockDim.x) {
            smem_ids[r] = scatter_token_id[batch_base + r];
            smem_weights[r] = scatter_token_weight[batch_base + r];
        }
    }

__device__ __forceinline__ void moe_gather_ep_row_assign(
    bfloat16* __restrict__ dst_row,
    const bfloat16* __restrict__ src_row,
    float weight,
    int hidden_size) {
    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        const Bf16Vec8 sv = load_bf16_vec8(src_row + i);
        vec8_store_weighted_bf16(dst_row + i, sv, weight);
    }
}

__device__ __forceinline__ void moe_gather_ep_row_accumulate(
    bfloat16* __restrict__ dst_row,
    const bfloat16* __restrict__ src_row,
    float weight,
    int hidden_size) {
    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        const Bf16Vec8 sv = load_bf16_vec8(src_row + i);
        float acc[8];
        
        vec8_load_fp32_from_bf16(acc, dst_row + i);
        vec8_golden_accumulate_fp32(acc, sv, weight);
        vec8_store_bf16_from_fp32(dst_row + i, acc);
    }
}

__device__ __forceinline__ void moe_gather_ep_process_row(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ src_row,
    int token_id,
    float weight,
    int num_tokens,
    int hidden_size,
    int assign_mode) {
    if (token_id < 0 || token_id >= num_tokens) {
        return;
    }
    bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size;
    if (assign_mode) {
        moe_gather_ep_row_assign(dst_row, src_row, weight, hidden_size);
    } else {
        moe_gather_ep_row_accumulate(dst_row, src_row, weight, hidden_size);
    }
}

__device__ __forceinline__ void moe_gather_ep_fused_process_row(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ src_row,
    int token_id,
    float weight,
    const bfloat16* __restrict__ residual,
    int num_tokens,
    int hidden_size,
    int res_token_start,
    int res_token_end,
    int residual_rows,
    float res_scale,
    int load_dst_first) {
    if (token_id < 0 || token_id >= num_tokens) {
        return;
    }

    bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size;

    const bool in_res = (token_id >= res_token_start && token_id < res_token_end);
    const bfloat16* res_row = nullptr;
    if (in_res) {
        res_row = (residual_rows == 1)
                      ? residual
                      : (residual + static_cast<int64_t>(token_id - res_token_start) * hidden_size);
    }

    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        const Bf16Vec8 sv = load_bf16_vec8(src_row + i);
        float acc[8];
        if (load_dst_first) {
            vec8_load_fp32_from_bf16(acc, dst_row + i);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] = 0.f;
            }
        }
        if (in_res) {
            const Bf16Vec8 rv = load_bf16_vec8(res_row + i);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] = fmaf(res_scale, bf16_to_float(rv.val[k]), acc[k]);
            }
        }
        vec8_golden_accumulate_fp32(acc, sv, weight);
        vec8_store_bf16_from_fp32(dst_row + i, acc);
    }
}

__device__ __forceinline__ void moe_gather_ep_run_batch_smem(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    int batch_base,
    int batch_rows,
    int hidden_size,
    int assign_mode,
    const bfloat16* __restrict__ residual,
    int num_tokens,
    int res_token_start,
    int res_token_end,
    int residual_rows,
    float res_scale,
    int load_dst_first,
    unsigned char* smem_raw) {
    const MoeGatherEpSmem smem = MoeGatherEpSmem::from_raw(smem_raw, hidden_size);
    const int64_t h = static_cast<int64_t>(hidden_size);
    const bfloat16* gmem_batch = scatter_tokens + static_cast<int64_t>(batch_base) * h;

    moe_gather_ep_stage_batch_metadata(
        smem.token_id, smem.weight, scatter_token_id, scatter_token_weight, batch_base,
        batch_rows);
    __syncthreads();

    if (batch_rows > 0) {
        moe_gather_vec8_copy_row_to_smem(smem.row_ping, gmem_batch, hidden_size);
    }
    __syncthreads();

    const bool fused = (residual != nullptr);
    for (int r = 0; r < batch_rows; ++r) {
        const bfloat16* src_row = smem.row_slot(r);
        const int token_id = smem.token_id[r];
        const float weight = smem.weight[r];
            
        if (fused) {
            moe_gather_ep_fused_process_row(
                convergent, src_row, token_id, weight, residual, num_tokens, hidden_size,
                res_token_start, res_token_end, residual_rows, res_scale, load_dst_first);
        } else {
            moe_gather_ep_process_row(
                convergent, src_row, token_id, weight, num_tokens, hidden_size, assign_mode);
        }

        if (r + 1 < batch_rows) {
            moe_gather_vec8_copy_row_to_smem(
                smem.row_slot(r + 1), gmem_batch + static_cast<int64_t>(r + 1) * h, hidden_size);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(512)
void moe_gather_ep_direct_row_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    int dispatch_tokens,
    int num_tokens,
    int hidden_size) {
    const int row = moe_gather_map_block_idx(blockIdx.x, dispatch_tokens);
    if (row >= dispatch_tokens) {
        return;
    }
    const int token_id = scatter_token_id[row];
    const float weight = scatter_token_weight[row];
    const bfloat16* src_row =
        scatter_tokens + static_cast<int64_t>(row) * static_cast<int64_t>(hidden_size);
    moe_gather_ep_process_row(
        convergent, src_row, token_id, weight, num_tokens, hidden_size, 1);
}

__global__ __launch_bounds__(512)
void moe_gather_ep_direct_fused_row_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    const bfloat16* __restrict__ residual,
    int dispatch_tokens,
    int num_tokens,
    int hidden_size,
    int res_token_start,
    int res_token_end,
    int residual_rows,
    float res_scale,
    int load_dst_first) {
    const int row = moe_gather_map_block_idx(blockIdx.x, dispatch_tokens);
    if (row >= dispatch_tokens) {
        return;
    }
    const int token_id = scatter_token_id[row];
    const float weight = scatter_token_weight[row];
    const bfloat16* src_row =
        scatter_tokens + static_cast<int64_t>(row) * static_cast<int64_t>(hidden_size);
    moe_gather_ep_fused_process_row(
        convergent, src_row, token_id, weight, residual, num_tokens, hidden_size,
        res_token_start, res_token_end, residual_rows, res_scale, load_dst_first);
}

__global__ __launch_bounds__(512)
void moe_gather_ep_batch_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    const bfloat16* __restrict__ residual,
    int dispatch_tokens,
    int num_tokens,
    int hidden_size,
    int assign_mode,
    int res_token_start,
    int res_token_end,
    int residual_rows,
    float res_scale,
    int load_dst_first) 
    {
        const int batch_blocks =
            (dispatch_tokens + kEpRowsPerBlock - 1) / kEpRowsPerBlock;
        const int batch_block = moe_gather_map_block_idx(blockIdx.x, batch_blocks);
        const int batch_base = batch_block * kEpRowsPerBlock;
        if (batch_base >= dispatch_tokens) {
            return;
        }
        const int batch_rows = min(kEpRowsPerBlock, dispatch_tokens - batch_base);

        extern __shared__ __align__(16) unsigned char smem_raw[];
        moe_gather_ep_run_batch_smem(
            convergent, scatter_tokens, scatter_token_id, scatter_token_weight, batch_base, batch_rows,
            hidden_size, assign_mode, residual, num_tokens, res_token_start, res_token_end,
            residual_rows, res_scale, load_dst_first, smem_raw);
}

__device__ __forceinline__ bool moe_gather_topk_rows_are_consecutive(
    const int* rows,
    int topk) {

    const int base = rows[0];
    for (int k = 1; k < topk; ++k) {
        if (rows[k] != base + k) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void moe_gather_prepare_inv_perm_smem(
    int* __restrict__ smem_rows,
    float* __restrict__ smem_weights,
    int* __restrict__ smem_rows_consecutive,
    const int* __restrict__ inv_perm_token,
    const float* __restrict__ scatter_token_weight,
    int topk) {
    if (threadIdx.x == 0) {
        for (int k = 0; k < kInvPermMaxTopk; ++k) {
            if (k < topk) {
                const int row = inv_perm_token[k];
                smem_rows[k] = row;
                smem_weights[k] = (row >= 0) ? scatter_token_weight[row] : 0.f;
            } else {
                smem_rows[k] = 0;
                smem_weights[k] = 0.f;
            }
        }
        *smem_rows_consecutive =
            moe_gather_topk_rows_are_consecutive(smem_rows, topk) ? 1 : 0;
    }
    __syncthreads();
}

__device__ __forceinline__ void moe_gather_accumulate_token_topk_vec8(
    bfloat16* __restrict__ dst_row,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ smem_rows,
    const float* __restrict__ smem_weights,
    int topk,
    int base_row,
    int rows_consecutive,
    int hidden_size,
    int has_prior_values) {
    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);
    const int vec8_limit = hidden_size >> 3;

    if (blockDim.x >= vec8_limit && threadIdx.x < vec8_limit) {
        const int vi = threadIdx.x;
        const int i = vi << 3;
        float acc[8];
        if (has_prior_values) {
            vec8_load_fp32_from_bf16(acc, dst_row + i);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] = 0.f;
            }
        }

#pragma unroll
        for (int rk = 0; rk < kInvPermMaxTopk; ++rk) {
            if (rk >= topk) {
                break;
            }
            const int64_t row_off =
                rows_consecutive
                    ? static_cast<int64_t>(base_row + rk) * hidden_size64
                    : static_cast<int64_t>(smem_rows[rk]) * hidden_size64;
            const float weight = smem_weights[rk];
            const Bf16Vec8 sv = load_bf16_vec8(scatter_tokens + row_off + i);
            vec8_golden_accumulate_fp32(acc, sv, weight);
        }

        vec8_store_bf16_from_fp32(dst_row + i, acc);
        return;
    }

    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        float acc[8];
        if (has_prior_values) {
            vec8_load_fp32_from_bf16(acc, dst_row + i);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) 
            {
                acc[k] = 0.f;
            }
        }
#pragma unroll
        for (int rk = 0; rk < kInvPermMaxTopk; ++rk) {
            if (rk >= topk) {
                break;
            }
            const int64_t row_off =
                rows_consecutive
                    ? static_cast<int64_t>(base_row + rk) * hidden_size64
                    : static_cast<int64_t>(smem_rows[rk]) * hidden_size64;
            const float weight = smem_weights[rk];
            const Bf16Vec8 sv = load_bf16_vec8(scatter_tokens + row_off + i);
            vec8_golden_accumulate_fp32(acc, sv, weight);
        }
        vec8_store_bf16_from_fp32(dst_row + i, acc);
    }
}

__device__ __forceinline__ int moe_gather_segment_start(const int* sorted_token, int i) {
    const int tok = sorted_token[i];
    int lo = 0;
    int hi = i;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (sorted_token[mid] < tok) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ __launch_bounds__(256)
void moe_gather_fill_inv_perm_from_sorted_kernel(
    const int* __restrict__ sorted_token,
    const int* __restrict__ sorted_row,
    int* __restrict__ inv_perm,
    int topk,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int tok = sorted_token[i];
    const int row = sorted_row[i];
    const int k = i - moe_gather_segment_start(sorted_token, i);
    if (k < topk) {
        inv_perm[tok * topk + k] = row;
    }
}

__global__ __launch_bounds__(512)
void moe_gather_token_dense_topk_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const float* __restrict__ scatter_token_weight,
    int num_tokens,
    int topk,
    int hidden_size,
    int has_prior_values) {
    const int token_blocks =
        (num_tokens + kInvPermTokensPerBlock - 1) / kInvPermTokensPerBlock;
    const int batch_block = moe_gather_map_block_idx(blockIdx.x, token_blocks);
    const int batch_base = batch_block * kInvPermTokensPerBlock;
    if (batch_base >= num_tokens) {
        return;
    }

    const int batch_tokens = min(kInvPermTokensPerBlock, num_tokens - batch_base);

    __shared__ int smem_rows[kInvPermMaxTopk];
    __shared__ float smem_weights[kInvPermMaxTopk];
    __shared__ int smem_rows_consecutive;

    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);

    for (int local_t = 0; local_t < batch_tokens; ++local_t) {
        const int token_id = batch_base + local_t;
        const int base_row = token_id * topk;

        if (threadIdx.x == 0) {
            for (int k = 0; k < kInvPermMaxTopk; ++k) {
                if (k < topk) {
                    smem_rows[k] = base_row + k;
                    smem_weights[k] = scatter_token_weight[base_row + k];
                } else {
                    smem_rows[k] = 0;
                    smem_weights[k] = 0.f;
                }
            }
            smem_rows_consecutive = 1;
        }
        __syncthreads();

        bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size64;
        moe_gather_accumulate_token_topk_vec8(
            dst_row, scatter_tokens, smem_rows, smem_weights, topk, base_row,
            smem_rows_consecutive, hidden_size, has_prior_values);
        __syncthreads();
    }
}

__global__ __launch_bounds__(512)
void moe_gather_token_inv_perm_direct_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const float* __restrict__ scatter_token_weight,
    const int* __restrict__ inv_perm,
    int num_tokens,
    int topk,
    int hidden_size,
    int has_prior_values) {
    const int token_blocks =
        (num_tokens + kInvPermTokensPerBlock - 1) / kInvPermTokensPerBlock;
    const int batch_block = moe_gather_map_block_idx(blockIdx.x, token_blocks);
    const int batch_base = batch_block * kInvPermTokensPerBlock;
    if (batch_base >= num_tokens) {
        return;
    }
    const int batch_tokens = min(kInvPermTokensPerBlock, num_tokens - batch_base);

    __shared__ int smem_rows[kInvPermMaxTopk];
    __shared__ float smem_weights[kInvPermMaxTopk];
    __shared__ int smem_rows_consecutive;

    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);

    for (int local_t = 0; local_t < batch_tokens; ++local_t) {
        const int token_id = batch_base + local_t;
        moe_gather_prepare_inv_perm_smem(
            smem_rows, smem_weights, &smem_rows_consecutive,
            inv_perm + static_cast<int64_t>(token_id) * topk, scatter_token_weight, topk);

        bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size64;
        const int base_row = smem_rows[0];
        moe_gather_accumulate_token_topk_vec8(
            dst_row, scatter_tokens, smem_rows, smem_weights, topk, base_row,
            smem_rows_consecutive, hidden_size, has_prior_values);
        __syncthreads();
    }
}

static bool moe_gather_optional_tensor_provided(const torch::Tensor& t) {
    return t.defined() && t.numel() > 0 && t.is_cuda();
}

static bool moe_gather_residual_active(
    const torch::Tensor& residual_tokens,
    int64_t res_rows,
    double res_scale) {
    // Primary gate is ops.py moe_gather_should_apply_residual (strips inactive residual).
    // Keep res_scale / slice checks here for direct torch.ops.custom_ops.moe_gather_forward calls.
    return moe_gather_optional_tensor_provided(residual_tokens) && res_rows > 0 &&
           res_scale != 0.0;
}

static bool moe_gather_env_flag_1(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] == '1' && env[1] == '\0';
}

static bool moe_gather_use_aten_path() {
    return moe_gather_env_flag_1("MOE_GATHER_USE_ATEN");
}

static void moe_gather_validate_token_id_if_enabled(
    const torch::Tensor& scatter_token_id,
    int64_t num_tokens) {
    if (!moe_gather_env_flag_1("MOE_GATHER_CHECK_TOKEN_ID")) {
        return;
    }
    const int64_t id_min = scatter_token_id.amin().item<int64_t>();
    const int64_t id_max = scatter_token_id.amax().item<int64_t>();
    TORCH_CHECK(id_min >= 0 && id_max < num_tokens,
                "scatter_token_id out of range: min=", id_min, " max=", id_max,
                " num_tokens=", num_tokens);
}

static int moe_gather_threads_for_hidden(int64_t hidden_size) {
    const int vec8_cols = static_cast<int>(hidden_size >> 3);
    if (vec8_cols >= 256 && vec8_cols <= 512) {
        return 256;
    }
    if (vec8_cols <= kMaxGatherThreads) {
        return moe_gather_round_up_wavefront(vec8_cols);
    }
    return moe_gather_round_up_wavefront(kMaxGatherThreads);
}

static void moe_gather_launch_residual_hip(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& residual_tokens,
    int64_t res_token_start,
    int64_t res_rows,
    int64_t hidden_size,
    double res_scale) {
    const int threads = moe_gather_threads_for_hidden(hidden_size);
    const int blocks = static_cast<int>(res_rows);
   moe_gather_residual_vec8_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(residual_tokens.data_ptr<at::BFloat16>()),
        static_cast<int>(res_token_start),
        static_cast<int>(hidden_size),
        static_cast<int>(res_rows),
        static_cast<int>(residual_tokens.size(0)),
        static_cast<float>(res_scale));;
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_launch_ep_direct(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_id,
    const torch::Tensor& scatter_token_weight,
    int64_t dispatch_tokens,
    int64_t num_tokens,
    int64_t hidden_size,
    bool store_only) {
    const int threads = moe_gather_threads_for_hidden(hidden_size);
    if (store_only) {
        // ep_size>1 typical case: 1 scatter row -> 1 token, no smem staging.
        const int blocks = static_cast<int>(dispatch_tokens);
       moe_gather_ep_direct_row_vec8_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_id.data_ptr<int>(),
        scatter_token_weight.data_ptr<float>(),
        static_cast<int>(dispatch_tokens),
        static_cast<int>(num_tokens),
        static_cast<int>(hidden_size));;
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int blocks =
        (static_cast<int>(dispatch_tokens) + kEpRowsPerBlock - 1) / kEpRowsPerBlock;
    const size_t smem_bytes = moe_gather_ep_smem_bytes(static_cast<int>(hidden_size));
    moe_gather_set_dynamic_smem(
        reinterpret_cast<const void*>(moe_gather_ep_batch_vec8_kernel), smem_bytes);
   moe_gather_ep_batch_vec8_kernel<<<dim3(blocks), dim3(threads), smem_bytes, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_id.data_ptr<int>(),
        scatter_token_weight.data_ptr<float>(),
        nullptr,
        static_cast<int>(dispatch_tokens),
        static_cast<int>(num_tokens),
        static_cast<int>(hidden_size),
        0,
        0,
        0,
        0,
        0.f,
        0);;
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_launch_ep_direct_fused(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_id,
    const torch::Tensor& scatter_token_weight,
    const torch::Tensor& residual_tokens,
    int64_t dispatch_tokens,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t res_token_start,
    int64_t res_token_end,
    double res_scale,
    bool load_dst_first) {
    const int threads = moe_gather_threads_for_hidden(hidden_size);
    const int blocks = static_cast<int>(dispatch_tokens);
    moe_gather_ep_direct_fused_row_vec8_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_id.data_ptr<int>(),
        scatter_token_weight.data_ptr<float>(),
        reinterpret_cast<const bfloat16*>(residual_tokens.data_ptr<at::BFloat16>()),
        static_cast<int>(dispatch_tokens),
        static_cast<int>(num_tokens),
        static_cast<int>(hidden_size),
        static_cast<int>(res_token_start),
        static_cast<int>(res_token_end),
        static_cast<int>(residual_tokens.size(0)),
        static_cast<float>(res_scale),
        load_dst_first ? 1 : 0);;
    CUDA_CHECK(cudaGetLastError());
}

static bool moe_gather_use_dense_layout() {
    return moe_gather_env_flag_1("MOE_GATHER_DENSE_LAYOUT");
}

static bool moe_gather_can_use_dense_layout(int64_t dispatch_tokens, int64_t num_tokens) {
    if (dispatch_tokens <= num_tokens) {
        return false;
    }
    if (dispatch_tokens % num_tokens != 0) {
        return false;
    }
    const int64_t topk = dispatch_tokens / num_tokens;
    if (topk <= 0 || topk > kInvPermMaxTopk) {
        return false;
    }
    return moe_gather_use_dense_layout();
}

static void moe_gather_build_inv_perm_on_device(
    const cudaStream_t stream,
    const torch::Tensor& scatter_token_id,
    torch::Tensor& inv_perm,
    int64_t num_tokens,
    int64_t topk) {
    (void)num_tokens;
    const int64_t dispatch_tokens = scatter_token_id.size(0);
    auto opts_i64 = scatter_token_id.options().dtype(torch::kInt64);
    auto row_idx = torch::arange(dispatch_tokens, opts_i64);
    const int64_t row_shift = static_cast<int64_t>(1) << 32;
    auto packed = scatter_token_id.to(torch::kInt64) * row_shift + row_idx;
    auto sorted_packed = std::get<0>(packed.sort(/*dim=*/0, /*descending=*/false));
    auto sorted_token = (sorted_packed / row_shift).to(torch::kInt32);
    auto sorted_row = (sorted_packed % row_shift).to(torch::kInt32);

    const int threads = 256;
    const int blocks =
        static_cast<int>((dispatch_tokens + threads - 1) / threads);
    moe_gather_fill_inv_perm_from_sorted_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        sorted_token.data_ptr<int>(),
        sorted_row.data_ptr<int>(),
        inv_perm.data_ptr<int>(),
        static_cast<int>(topk),
        static_cast<int>(dispatch_tokens));;
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_launch_dense_topk(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_weight,
    int64_t num_tokens,
    int64_t topk,
    int64_t hidden_size,
    bool has_prior_values) {
    const int threads = moe_gather_threads_for_hidden(hidden_size);
    const int blocks =
        (static_cast<int>(num_tokens) + kInvPermTokensPerBlock - 1) / kInvPermTokensPerBlock;
   moe_gather_token_dense_topk_vec8_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_weight.data_ptr<float>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        static_cast<int>(hidden_size),
        has_prior_values ? 1 : 0);;
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_launch_inv_perm(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_weight,
    const torch::Tensor& inv_perm,
    int64_t num_tokens,
    int64_t topk,
    int64_t hidden_size,
    bool has_prior_values) {
    const int threads = moe_gather_threads_for_hidden(hidden_size);
    const int blocks =
        (static_cast<int>(num_tokens) + kInvPermTokensPerBlock - 1) / kInvPermTokensPerBlock;
   moe_gather_token_inv_perm_direct_vec8_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_weight.data_ptr<float>(),
        inv_perm.data_ptr<int>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        static_cast<int>(hidden_size),
        has_prior_values ? 1 : 0);;
    CUDA_CHECK(cudaGetLastError());
}

torch::Tensor moe_gather_build_inv_perm_hip(
    torch::Tensor scatter_token_id,
    int64_t num_tokens,
    int64_t topk) {
    TORCH_CHECK(scatter_token_id.is_contiguous(), "scatter_token_id must be contiguous");
    TORCH_CHECK(scatter_token_id.scalar_type() == torch::kInt32,
                "scatter_token_id must be int32");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(topk > 0 && topk <= kInvPermMaxTopk, "topk must be in (0, ", kInvPermMaxTopk, "]");
    TORCH_CHECK(scatter_token_id.size(0) == num_tokens * topk,
                "scatter_token_id length must be num_tokens * topk");

    auto inv_perm = torch::full(
        {num_tokens, topk}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(scatter_token_id.device()));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    moe_gather_build_inv_perm_on_device(stream, scatter_token_id, inv_perm, num_tokens, topk);
    return inv_perm;
}

static void moe_gather_index_add_aten(
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_id,
    const torch::Tensor& scatter_token_weight) {
    auto weight_col = scatter_token_weight.unsqueeze(1);
    auto weighted = (scatter_tokens * weight_col).to(torch::kBFloat16);
    auto index = scatter_token_id.to(torch::kLong);
    output.index_add_(0, index, weighted);
}

torch::Tensor moe_gather_forward_hip(
    torch::Tensor scatter_tokens,
    torch::Tensor scatter_token_id,
    torch::Tensor scatter_token_weight,
    torch::Tensor convergent_tokens,
    int64_t num_tokens,
    torch::Tensor residual_tokens,
    int64_t res_token_start,
    int64_t res_token_end,
    double res_scale,
    torch::Tensor inv_perm) {
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    TORCH_CHECK(scatter_token_id.is_contiguous(), "scatter_token_id must be contiguous");
    TORCH_CHECK(scatter_token_weight.is_contiguous(), "scatter_token_weight must be contiguous");
    TORCH_CHECK(scatter_tokens.scalar_type() == torch::kBFloat16,
                "scatter_tokens must be bfloat16");
    TORCH_CHECK(scatter_token_id.scalar_type() == torch::kInt32,
                "scatter_token_id must be int32");
    TORCH_CHECK(scatter_token_weight.scalar_type() == torch::kFloat32,
                "scatter_token_weight must be float32");

    const int64_t dispatch_tokens = scatter_tokens.size(0);
    const int64_t hidden_size = scatter_tokens.size(1);

    TORCH_CHECK(scatter_token_id.dim() == 1 && scatter_token_id.size(0) == dispatch_tokens,
                "scatter_token_id shape must be [dispatch_tokens]");
    TORCH_CHECK(scatter_token_weight.dim() == 1 &&
                    scatter_token_weight.size(0) == dispatch_tokens,
                "scatter_token_weight shape must be [dispatch_tokens]");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(res_token_start >= 0 && res_token_start <= num_tokens,
                "res_token_start out of range");
    TORCH_CHECK(res_token_end >= res_token_start && res_token_end <= num_tokens,
                "res_token_end out of range");
    TORCH_CHECK(hidden_size % 8 == 0 || moe_gather_use_aten_path(),
                "HIP gather requires hidden_size %% 8 == 0 (or set MOE_GATHER_USE_ATEN=1)");

    auto options = scatter_tokens.options();
    const bool convergent_provided = moe_gather_optional_tensor_provided(convergent_tokens);
    torch::Tensor output;
    if (!convergent_provided) {
        output = torch::empty({num_tokens, hidden_size}, options);
    } else {
        TORCH_CHECK(convergent_tokens.scalar_type() == torch::kBFloat16,
                    "convergent_tokens must be bfloat16");
        TORCH_CHECK(convergent_tokens.sizes() == torch::IntArrayRef({num_tokens, hidden_size}),
                    "convergent_tokens shape must be [num_tokens, hidden_size]");
        output = convergent_tokens;
    }

    const int64_t res_rows = res_token_end - res_token_start;
    const bool has_residual =
        moe_gather_residual_active(residual_tokens, res_rows, res_scale);

    if (dispatch_tokens == 0 && !has_residual) {
        return output;
    }
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (hidden_size % 8 != 0 || moe_gather_use_aten_path()) {
        if (!convergent_provided) {
            output.zero_();
        }
        if (has_residual) {
            auto out_slice = output.narrow(0, res_token_start, res_rows);
            out_slice.add_(residual_tokens, res_scale);
        }
        if (dispatch_tokens > 0) {
            moe_gather_validate_token_id_if_enabled(scatter_token_id, num_tokens);
            moe_gather_index_add_aten(output, scatter_tokens, scatter_token_id, scatter_token_weight);
        }
        return output;
    }

    moe_gather_validate_token_id_if_enabled(scatter_token_id, num_tokens);

    if (dispatch_tokens == 0) {
        if (has_residual) {
            moe_gather_launch_residual_hip(
                stream, output, residual_tokens, res_token_start, res_rows, hidden_size, res_scale);
        }
        return output;
    }

    if (dispatch_tokens == num_tokens) {
        if (has_residual) {
            const bool load_dst_first = false;
            moe_gather_launch_ep_direct_fused(
                stream, output, scatter_tokens, scatter_token_id, scatter_token_weight,
                residual_tokens, dispatch_tokens, num_tokens, hidden_size,
                res_token_start, res_token_end, res_scale, load_dst_first);
        } else {
            moe_gather_launch_ep_direct(
                stream, output, scatter_tokens, scatter_token_id, scatter_token_weight,
                dispatch_tokens, num_tokens, hidden_size, true);
        }
        return output;
    }

    const bool has_prior_values = has_residual || convergent_provided;
    const int64_t topk = dispatch_tokens / num_tokens;
    const bool inv_perm_provided = moe_gather_optional_tensor_provided(inv_perm);

    if (has_residual) {
        moe_gather_launch_residual_hip(
            stream, output, residual_tokens, res_token_start, res_rows, hidden_size, res_scale);
    }

    if (moe_gather_can_use_dense_layout(dispatch_tokens, num_tokens)) {
        moe_gather_launch_dense_topk(
            stream, output, scatter_tokens, scatter_token_weight, num_tokens, topk, hidden_size,
            has_prior_values);
        return output;
    }

    torch::Tensor inv_perm_tensor;
    if (inv_perm_provided) {
        inv_perm_tensor = inv_perm;
    } else {
        inv_perm_tensor = torch::full(
            {num_tokens, topk}, -1,
            torch::TensorOptions().dtype(torch::kInt32).device(scatter_tokens.device()));
        moe_gather_build_inv_perm_on_device(
            stream, scatter_token_id, inv_perm_tensor, num_tokens, topk);
    }

    TORCH_CHECK(inv_perm_tensor.scalar_type() == torch::kInt32, "inv_perm must be int32");
    TORCH_CHECK(inv_perm_tensor.dim() == 2 && inv_perm_tensor.size(0) == num_tokens,
                "inv_perm shape must be [num_tokens, topk]");
    TORCH_CHECK(inv_perm_tensor.size(1) > 0 && inv_perm_tensor.size(1) <= kInvPermMaxTopk,
                "inv_perm topk must be in (0, ", kInvPermMaxTopk, "]");

    moe_gather_launch_inv_perm(
        stream, output, scatter_tokens, scatter_token_weight, inv_perm_tensor, num_tokens, topk,
        hidden_size, has_prior_values);

    return output;
}

// ---- v2 ----

static constexpr int kV2Topk = 8;
static constexpr int kV2Wavefront = 64;
static constexpr int kV2MaxThreads = 512;
static constexpr int kV2TokensPerBlock = 4;    

// EP direct tiling hyperparameters (edit here, recompile to tune).
static constexpr bool kV2EpUseTiledKernel = true;
static constexpr int kV2EpTileBlocks = 512;
static constexpr int kV2EpRowsPerBlock = 20;









__device__ __forceinline__ float moe_gather_v2_warp_broadcast_float(float value) {
    return __shfl(value, 0, kV2Wavefront);
}











static int moe_gather_v2_round_up_wavefront(int threads) {
    threads = (threads + 63) & 0xFFFFFFC0;      // Compilation time confirmed @ wavefront-1
    return threads < kV2Wavefront ? kV2Wavefront : threads;
}

static __forceinline__ int moe_gather_v2_threads_for_hidden(int64_t hidden_size) {
    const int vec8_cols = static_cast<int>(hidden_size >> 3);
    return moe_gather_v2_round_up_wavefront(min(vec8_cols, kV2MaxThreads));
}

__device__ __forceinline__ void moe_gather_v2_ep_row_assign(
    bfloat16* __restrict__ dst_row,
    const bfloat16* __restrict__ src_row,
    float weight,
    int hidden_size) {
    const int vec8_limit = hidden_size >> 3;
    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        vec8_store_weighted_bf16(dst_row + i, load_bf16_vec8(src_row + i), weight);
    }
}

__device__ __forceinline__ void moe_gather_v2_ep_process_row(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ src_row,
    int token_id,
    float weight,
    int num_tokens,
    int hidden_size) {
    // if (token_id < 0 || token_id >= num_tokens) {
    //     return;
    // }
    bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size;
    moe_gather_v2_ep_row_assign(dst_row, src_row, weight, hidden_size);
}

// EP direct: 1 scatter row
__global__ __launch_bounds__(512)
void moe_gather_v2_ep_direct_row_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    int dispatch_tokens,
    int num_tokens,
    int hidden_size) {
    const int row = blockIdx.x;
    if (row >= dispatch_tokens) {
        return;
    }

    const int token_id = scatter_token_id[row];
    const float weight = scatter_token_weight[row];
    const bfloat16* src_row =
        scatter_tokens + static_cast<int64_t>(row) * static_cast<int64_t>(hidden_size);
    moe_gather_v2_ep_process_row(
        convergent, src_row, token_id, weight, num_tokens, hidden_size);
}

// EP direct tiling: stage token_id/weight metadata in smem once per tile, then vec8 gather.
__global__ __launch_bounds__(512)
void moe_gather_v2_ep_tiled_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ scatter_token_id,
    const float* __restrict__ scatter_token_weight,
    int dispatch_tokens,
    int num_tokens,
    int hidden_size,
    int rows_per_block) {
    __shared__ int smem_token_id[kV2EpRowsPerBlock];
    __shared__ float smem_weight[kV2EpRowsPerBlock];

    const int batch_base = blockIdx.x * rows_per_block;
    if (batch_base >= dispatch_tokens) {
        return;
    }

    const int batch_rows = min(rows_per_block, dispatch_tokens - batch_base);
    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);

    for (int local = threadIdx.x; local < batch_rows; local += blockDim.x) {
        const int row = batch_base + local;
        smem_token_id[local] = scatter_token_id[row];
        smem_weight[local] = scatter_token_weight[row];
    }
    __syncthreads();

    // rows_per_block is kV2EpRowsPerBlock at launch; unroll removes loop ctrl overhead.
    // Rows still run serially: every thread collaborates on one row, then __syncthreads.
#pragma unroll
    for (int local = 0; local < kV2EpRowsPerBlock; ++local) {
        if (local >= batch_rows) {
            continue;
        }
        const int row = batch_base + local;
        const int token_id = smem_token_id[local];
        const int lane = threadIdx.x & (kV2Wavefront - 1);
        float weight = 0.f;
        if (lane == 0) {
            weight = smem_weight[local];
        }
        weight = moe_gather_v2_warp_broadcast_float(weight);
        const bfloat16* src_row = scatter_tokens + static_cast<int64_t>(row) * hidden_size64;
        moe_gather_v2_ep_process_row(
            convergent, src_row, token_id, weight, num_tokens, hidden_size);
        if (local + 1 < batch_rows) {
            __syncthreads();
        }
    }
}

__device__ __forceinline__ bool moe_gather_v2_topk_rows_are_consecutive(const int* rows, int topk) {
    if (topk <= 1) {
        return true;
    }
    const int base = rows[0];
    for (int k = 1; k < topk; ++k) {
        if (rows[k] != base + k) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void moe_gather_v2_accumulate_token_topk_vec8(
    bfloat16* __restrict__ dst_row,
    const bfloat16* __restrict__ scatter_tokens,
    const int* __restrict__ smem_rows,
    const float* __restrict__ smem_weights,
    int topk,
    int base_row,
    int rows_consecutive,
    int hidden_size) {
    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);
    const int vec8_limit = hidden_size >> 3;

    if (blockDim.x >= vec8_limit && threadIdx.x < vec8_limit) {
        const int vi = threadIdx.x;
        const int i = vi << 3;
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] = 0.f;
        }

#pragma unroll
        for (int rk = 0; rk < kV2Topk; ++rk) {
            if (rk >= topk) {
                break;
            }
            const int64_t row_off = rows_consecutive
                                        ? static_cast<int64_t>(base_row + rk) * hidden_size64
                                        : static_cast<int64_t>(smem_rows[rk]) * hidden_size64;
            vec8_golden_accumulate_fp32(
                acc, load_bf16_vec8(scatter_tokens + row_off + i), smem_weights[rk]);
        }
        vec8_store_bf16_from_fp32(dst_row + i, acc);
        return;
    }

    for (int vi = threadIdx.x; vi < vec8_limit; vi += blockDim.x) {
        const int i = vi << 3;
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] = 0.f;
        }
#pragma unroll
        for (int rk = 0; rk < kV2Topk; ++rk) {
            if (rk >= topk) {
                break;
            }
            const int64_t row_off = rows_consecutive
                                        ? static_cast<int64_t>(base_row + rk) * hidden_size64
                                        : static_cast<int64_t>(smem_rows[rk]) * hidden_size64;
            vec8_golden_accumulate_fp32(
                acc, load_bf16_vec8(scatter_tokens + row_off + i), smem_weights[rk]);
        }
        vec8_store_bf16_from_fp32(dst_row + i, acc);
    }
}

// =====
// TODO
// ======
// Possible optim point: random random access
// may entail scatter read transaction
// But the perf reward might be marginal
__device__ __forceinline__ void moe_gather_v2_prepare_inv_perm_smem(
    int* __restrict__ smem_rows,
    float* __restrict__ smem_weights,
    int* __restrict__ smem_rows_consecutive,
    const int* __restrict__ inv_perm_token,
    const float* __restrict__ scatter_token_weight,
    int topk) {
    if (threadIdx.x == 0)
    {
        for (int k = 0; k < kV2Topk; ++k) {
            if (k < topk) {
                const int row = inv_perm_token[k];
                smem_rows[k] = row;
                smem_weights[k] = (row >= 0) ? scatter_token_weight[row] : 0.f;
            } else {
                smem_rows[k] = 0;
                smem_weights[k] = 0.f;
            }
        }
        *smem_rows_consecutive = moe_gather_v2_topk_rows_are_consecutive(smem_rows, topk) ? 1 : 0;
    }
    __syncthreads();
}

 
__global__ __launch_bounds__(512)
void moe_gather_v2_token_dense_topk_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const float* __restrict__ scatter_token_weight,
    int num_tokens,
    int topk,
    int hidden_size) {
    const int batch_base = blockIdx.x * kV2TokensPerBlock;
    if (batch_base >= num_tokens) {
        return;
    }

    const int batch_tokens = min(kV2TokensPerBlock, num_tokens - batch_base);
    __shared__ int smem_rows[kV2Topk];
    __shared__ float smem_weights[kV2Topk];
    __shared__ int smem_rows_consecutive;
    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);

    for (int local_batch = 0; local_batch < batch_tokens; ++local_batch) {
        const int token_id = batch_base + local_batch;
        const int base_row = token_id * topk;
        if (threadIdx.x == 0) {
            for (int k = 0; k < kV2Topk; ++k) {
                if (k < topk) {
                    smem_rows[k] = base_row + k;
                    smem_weights[k] = scatter_token_weight[base_row + k];
                } else {
                    smem_rows[k] = 0;
                    smem_weights[k] = 0.f;
                }
            }
            smem_rows_consecutive = 1;
        }
        __syncthreads();

        bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size64;
        moe_gather_v2_accumulate_token_topk_vec8(
            dst_row, scatter_tokens, smem_rows, smem_weights, topk, base_row,
            smem_rows_consecutive, hidden_size);
        __syncthreads();
    }
}

// Multi-row general layout: 
__global__ __launch_bounds__(512)
void moe_gather_v2_token_inv_perm_vec8_kernel(
    bfloat16* __restrict__ convergent,
    const bfloat16* __restrict__ scatter_tokens,
    const float* __restrict__ scatter_token_weight,
    const int* __restrict__ inv_perm,
    int num_tokens,
    int topk,
    int hidden_size) {
    const int batch_base = blockIdx.x * kV2TokensPerBlock;
    if (batch_base >= num_tokens) {
        return;
    }

    const int batch_tokens = min(kV2TokensPerBlock, num_tokens - batch_base);
    __shared__ int smem_rows[kV2Topk];
    __shared__ float smem_weights[kV2Topk];
    __shared__ int smem_rows_consecutive;
    const int64_t hidden_size64 = static_cast<int64_t>(hidden_size);

    for (int local_batch = 0; local_batch < batch_tokens; ++local_batch) {
        const int token_id = batch_base + local_batch;
        moe_gather_v2_prepare_inv_perm_smem(
            smem_rows, smem_weights, &smem_rows_consecutive,
            inv_perm + static_cast<int64_t>(token_id) * topk, scatter_token_weight, topk);

        bfloat16* dst_row = convergent + static_cast<int64_t>(token_id) * hidden_size64;
        moe_gather_v2_accumulate_token_topk_vec8(
            dst_row, scatter_tokens, smem_rows, smem_weights, topk, smem_rows[0],
            smem_rows_consecutive, hidden_size);
        __syncthreads();
    }
}

// Only launched when inv_perm is not provided by caller (fallback, <<<1,1>>>).
__global__ void moe_gather_v2_build_inv_perm_seq_kernel(
    const int* __restrict__ scatter_token_id,
    int* __restrict__ inv_perm,
    int* __restrict__ slot,
    int dispatch_tokens,
    int topk) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int row = 0; row < dispatch_tokens; ++row) {
        const int tok = scatter_token_id[row];
        const int k = slot[tok];
        if (k < topk) {
            inv_perm[tok * topk + k] = row;
        }
        slot[tok] = k + 1;
    }
}

static __forceinline__ bool moe_gather_v2_optional_tensor_provided(const torch::Tensor& t) {
    return t.defined() && t.numel() > 0 && t.is_cuda();
}

static __forceinline__ bool moe_gather_v2_residual_active(
    const torch::Tensor& residual_tokens,
    int64_t res_rows,
    double res_scale) {
    return moe_gather_v2_optional_tensor_provided(residual_tokens) && res_rows > 0 &&
           res_scale != 0.0;
}

static bool moe_gather_v2_env_flag_1(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] == '1' && env[1] == '\0';
}

static __forceinline__ bool moe_gather_v2_use_aten_path() {
    return moe_gather_v2_env_flag_1("MOE_GATHER_USE_ATEN");
}

static __forceinline__ bool moe_gather_v2_use_dense_layout() {
    return moe_gather_v2_env_flag_1("MOE_GATHER_DENSE_LAYOUT");
}

static __forceinline__ bool moe_gather_v2_can_use_dense_layout(int64_t dispatch_tokens, int64_t num_tokens) {
    if (dispatch_tokens <= num_tokens || dispatch_tokens % num_tokens != 0) {
        return false;
    }
    const int64_t topk = dispatch_tokens / num_tokens;
    return topk > 0 && topk <= kV2Topk && moe_gather_v2_use_dense_layout();
}

static void moe_gather_v2_build_inv_perm_on_device(
    const cudaStream_t stream,
    const torch::Tensor& scatter_token_id,
    torch::Tensor& inv_perm,
    int64_t num_tokens,
    int64_t topk) {
    auto slot = torch::zeros({num_tokens}, scatter_token_id.options().dtype(torch::kInt32));
    moe_gather_v2_build_inv_perm_seq_kernel<<<1, 1, 0, stream>>>(
        scatter_token_id.data_ptr<int>(),
        inv_perm.data_ptr<int>(),
        slot.data_ptr<int>(),
        static_cast<int>(scatter_token_id.size(0)),
        static_cast<int>(topk));
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_v2_launch_ep_direct(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_id,
    const torch::Tensor& scatter_token_weight,
    int64_t dispatch_tokens,
    int64_t num_tokens,
    int64_t hidden_size) {
    const int threads = moe_gather_v2_threads_for_hidden(hidden_size);
    bfloat16* out_ptr = reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>());
    const bfloat16* scatter_ptr =
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>());
    const int* token_id_ptr = scatter_token_id.data_ptr<int>();
    const float* weight_ptr = scatter_token_weight.data_ptr<float>();
    const int dispatch = static_cast<int>(dispatch_tokens);
    const int n_tokens = static_cast<int>(num_tokens);
    const int h = static_cast<int>(hidden_size);

    if (kV2EpUseTiledKernel) {
        TORCH_CHECK(
            static_cast<int64_t>(kV2EpTileBlocks) * static_cast<int64_t>(kV2EpRowsPerBlock) >=
                dispatch_tokens,
            "EP tile coverage insufficient: kV2EpTileBlocks*kV2EpRowsPerBlock (",
            static_cast<int64_t>(kV2EpTileBlocks) * kV2EpRowsPerBlock,
            ") < dispatch_tokens (", dispatch_tokens,
            "). Increase kV2EpTileBlocks or kV2EpRowsPerBlock in custom_moe_gather_v2_hip.hip.");

        moe_gather_v2_ep_tiled_vec8_kernel<<<kV2EpTileBlocks, threads, 0, stream>>>(
            out_ptr, scatter_ptr, token_id_ptr, weight_ptr,
            dispatch, n_tokens, h, kV2EpRowsPerBlock);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    moe_gather_v2_ep_direct_row_vec8_kernel<<<dispatch, threads, 0, stream>>>(
        out_ptr, scatter_ptr, token_id_ptr, weight_ptr,
        dispatch, n_tokens, h);
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_v2_launch_dense_topk(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_weight,
    int64_t num_tokens,
    int64_t topk,
    int64_t hidden_size) {
    const int threads = moe_gather_v2_threads_for_hidden(hidden_size);
    const int blocks =
        (static_cast<int>(num_tokens) + kV2TokensPerBlock - 1) / kV2TokensPerBlock;
    moe_gather_v2_token_dense_topk_vec8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_weight.data_ptr<float>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        static_cast<int>(hidden_size));
    CUDA_CHECK(cudaGetLastError());
}

static void moe_gather_v2_launch_inv_perm(
    const cudaStream_t stream,
    torch::Tensor& output,
    const torch::Tensor& scatter_tokens,
    const torch::Tensor& scatter_token_weight,
    const torch::Tensor& inv_perm,
    int64_t num_tokens,
    int64_t topk,
    int64_t hidden_size) {
    const int threads = moe_gather_v2_threads_for_hidden(hidden_size);
    const int blocks =
        (static_cast<int>(num_tokens) + kV2TokensPerBlock - 1) / kV2TokensPerBlock;
    moe_gather_v2_token_inv_perm_vec8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_weight.data_ptr<float>(),
        inv_perm.data_ptr<int>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        static_cast<int>(hidden_size));
    CUDA_CHECK(cudaGetLastError());
}

torch::Tensor moe_gather_forward_v2_hip(
    torch::Tensor scatter_tokens,
    torch::Tensor scatter_token_id,
    torch::Tensor scatter_token_weight,
    torch::Tensor convergent_tokens,
    int64_t num_tokens,
    torch::Tensor residual_tokens,
    int64_t res_token_start,
    int64_t res_token_end,
    double res_scale,
    torch::Tensor inv_perm) {
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    TORCH_CHECK(scatter_token_id.is_contiguous(), "scatter_token_id must be contiguous");
    TORCH_CHECK(scatter_token_weight.is_contiguous(), "scatter_token_weight must be contiguous");
    TORCH_CHECK(scatter_tokens.scalar_type() == torch::kBFloat16,
                "scatter_tokens must be bfloat16");
    TORCH_CHECK(scatter_token_id.scalar_type() == torch::kInt32,
                "scatter_token_id must be int32");
    TORCH_CHECK(scatter_token_weight.scalar_type() == torch::kFloat32,
                "scatter_token_weight must be float32");

    const int64_t dispatch_tokens = scatter_tokens.size(0);
    const int64_t hidden_size = scatter_tokens.size(1);

    TORCH_CHECK(scatter_token_id.dim() == 1 && scatter_token_id.size(0) == dispatch_tokens,
                "scatter_token_id shape must be [dispatch_tokens]");
    TORCH_CHECK(scatter_token_weight.dim() == 1 &&
                    scatter_token_weight.size(0) == dispatch_tokens,
                "scatter_token_weight shape must be [dispatch_tokens]");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");

    const int64_t res_rows = res_token_end - res_token_start;
    const bool has_residual = moe_gather_v2_residual_active(residual_tokens, res_rows, res_scale);



    if (hidden_size % 8 != 0 || moe_gather_v2_use_aten_path() || has_residual) {
        return moe_gather_forward_hip(
            scatter_tokens, scatter_token_id, scatter_token_weight, convergent_tokens,
            num_tokens, residual_tokens, res_token_start, res_token_end, res_scale, inv_perm);
    }

    TORCH_CHECK(
        moe_gather_v2_optional_tensor_provided(convergent_tokens),
        "convergent_tokens must be a pre-allocated CUDA/HIP [num_tokens, hidden_size] tensor");
    TORCH_CHECK(convergent_tokens.scalar_type() == torch::kBFloat16,
                "convergent_tokens must be bfloat16");
    TORCH_CHECK(convergent_tokens.sizes() == torch::IntArrayRef({num_tokens, hidden_size}),
                "convergent_tokens shape must be [num_tokens, hidden_size]");
    torch::Tensor output = convergent_tokens;

    if (dispatch_tokens == 0) {
        return output;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // EP direct: single kernel 
    if (dispatch_tokens == num_tokens) {
        moe_gather_v2_launch_ep_direct(
            stream, output, scatter_tokens, scatter_token_id, scatter_token_weight,
            dispatch_tokens, num_tokens, hidden_size);
        return output;
    }

    const int64_t topk = dispatch_tokens / num_tokens;
    TORCH_CHECK(topk > 0 && topk <= kV2Topk, "v2 supports topk in (0, ", kV2Topk, "]");

    // Dense layout: single kernel.
    if (moe_gather_v2_can_use_dense_layout(dispatch_tokens, num_tokens)) {
        moe_gather_v2_launch_dense_topk(
            stream, output, scatter_tokens, scatter_token_weight,
            num_tokens, topk, hidden_size);
        return output;
    }

    // General multi-row: inv_perm lookup, single gather kernel.
    torch::Tensor inv_perm_tensor;
    if (moe_gather_v2_optional_tensor_provided(inv_perm)) {
        inv_perm_tensor = inv_perm;
    } else {
        inv_perm_tensor = torch::full(
            {num_tokens, topk}, -1,
            torch::TensorOptions().dtype(torch::kInt32).device(scatter_tokens.device()));
        moe_gather_v2_build_inv_perm_on_device(
            stream, scatter_token_id, inv_perm_tensor, num_tokens, topk);
    }

    TORCH_CHECK(inv_perm_tensor.scalar_type() == torch::kInt32, "inv_perm must be int32");
    TORCH_CHECK(inv_perm_tensor.dim() == 2 && inv_perm_tensor.size(0) == num_tokens,
                "inv_perm shape must be [num_tokens, topk]");

    moe_gather_v2_launch_inv_perm(
        stream, output, scatter_tokens, scatter_token_weight, inv_perm_tensor,
        num_tokens, topk, hidden_size);

    return output;
}

namespace at {
namespace native {

namespace {
bool moe_gather_optional_tensor_provided(const torch::Tensor& t) {
  return t.defined() && t.numel() > 0 && t.is_cuda();
}
}  // namespace

torch::Tensor moe_gather_build_inv_perm(
    torch::Tensor scatter_token_id,
    int64_t num_tokens,
    int64_t topk) {
  TORCH_CHECK(scatter_token_id.is_cuda(), "moe_gather_build_inv_perm: CUDA/HIP required");
  return moe_gather_build_inv_perm_hip(scatter_token_id, num_tokens, topk);
}

torch::Tensor moe_gather(
    torch::Tensor scatter_tokens,
    torch::Tensor scatter_token_id,
    torch::Tensor scatter_token_weight,
    torch::Tensor convergent_tokens,
    int64_t num_tokens,
    torch::Tensor residual_tokens,
    int64_t res_token_start,
    int64_t res_token_end,
    double res_scale,
    torch::Tensor inv_perm) {
  TORCH_CHECK(scatter_tokens.is_cuda(), "moe_gather: scatter_tokens must be CUDA/HIP");
  TORCH_CHECK(scatter_token_id.is_cuda(), "moe_gather: scatter_token_id must be CUDA/HIP");
  TORCH_CHECK(scatter_token_weight.is_cuda(), "moe_gather: scatter_token_weight must be CUDA/HIP");
  if (moe_gather_optional_tensor_provided(convergent_tokens)) {
    TORCH_CHECK(convergent_tokens.is_cuda(), "moe_gather: convergent_tokens must be CUDA/HIP");
  }
  if (moe_gather_optional_tensor_provided(residual_tokens)) {
    TORCH_CHECK(residual_tokens.is_cuda(), "moe_gather: residual_tokens must be CUDA/HIP");
  }
  if (moe_gather_optional_tensor_provided(inv_perm)) {
    TORCH_CHECK(inv_perm.is_cuda(), "moe_gather: inv_perm must be CUDA/HIP");
  }
  return moe_gather_forward_v2_hip(
      scatter_tokens, scatter_token_id, scatter_token_weight, convergent_tokens,
      num_tokens, residual_tokens, res_token_start, res_token_end, res_scale, inv_perm);
}

}  // namespace native
}  // namespace at
