// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <random>
#include <limits>
#include <string>
#include <sstream>
#include <filesystem>
#include <cstdint>
#include <cstdlib>

#include "hip/hip_runtime.h"
#include <hip/hip_hcc.h>
#include "rocblas.h"
#include "internal/rocblas-auxiliary.h"
#include "mqa_logits_asm.h"
#include "beta/mqa_logits_api.h"

#include <torch/extension.h>
#include <ATen/dtk_macros.h>
#include <c10/util/BFloat16.h>
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>
#include <torch/all.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"

#define fp8 uint8_t
using half_t = __half;
using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using half16_t = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using floatx2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;
using fp8x8 = __attribute__( (__vector_size__(8 * sizeof(fp8)) )) fp8;

#define FETCH_HALF8(pointer) (reinterpret_cast<half8_t*>(&(pointer))[0])
#define FETCH_HALF4(pointer) (reinterpret_cast<half4_t*>(&(pointer))[0])
#define DIV(x, y) ((x) + (y) - 1) / (y)

namespace at {
namespace native {

struct half4x8 {
    half4_t data[8];
};

struct half4x16 {
    half4_t data[16];
};

#define Input_Type_SWITCH(SRC_DTYPE, ...)      \
    [&] {                                      \
        if (SRC_DTYPE == at::ScalarType::Half) { \
            using scalar_t = _Float16;         \
            return __VA_ARGS__();              \
        } else {                               \
            using scalar_t = uint16_t;         \
            return __VA_ARGS__();              \
        }                                      \
    }()

// Load the per-row KE values used by the existing store masks and decide
// whether this output tile contains any valid [KS[row], KE[row]) element.
// A last-valid-row probe may prove that the tile must be computed, but never
// that it may be skipped. A possible skip is confirmed by balloting every
// valid row. For 32-row (and larger) tiles every wave repeats the same scan,
// avoiding cross-wave LDS communication. The 16-row kernels use a first-wave
// scan plus LDS broadcast because duplicating their scan regresses the masked
// fast path on the target hardware.
template <int BLOCK_M, int BLOCK_N, bool is_align_m>
__device__ __forceinline__ bool mqa_load_bounds_and_check_tile(
    const uint32_t* __restrict__ KS,
    const uint32_t* __restrict__ KE,
    int M,
    int N,
    int block_x,
    int block_y,
    uint32_t* boundary_smem,
    uint32_t* tile_has_work_smem) {
    const int tid = threadIdx.x;
    const int lane = tid % C10_WARP_SIZE;
    const int row_start = block_y * BLOCK_M;
    const int remaining_rows = M - row_start;
    const int valid_rows = is_align_m
        ? BLOCK_M
        : (remaining_rows < BLOCK_M ? remaining_rows : BLOCK_M);
    const int probe_local_row = valid_rows - 1;
    const int tile_start = block_x * BLOCK_N;
    const int unclamped_tile_end = tile_start + BLOCK_N;
    const int tile_end = unclamped_tile_end < N ? unclamped_tile_end : N;
    if constexpr (BLOCK_M > C10_WARP_SIZE) {
        const bool first_wave = tid < C10_WARP_SIZE;
        constexpr int rows_per_lane =
            (BLOCK_M + C10_WARP_SIZE - 1) / C10_WARP_SIZE;

        // Keep the existing BM>=32 strategy: every wave scans the complete
        // row tile, while the first wave publishes all row boundaries. Each
        // lane handles lane + k * wave_size, so rows beyond the first wave
        // are covered without another LDS reduction buffer.
        if (first_wave) {
#pragma unroll
            for (int k = 0; k < rows_per_lane; ++k) {
                const int local_row = lane + k * C10_WARP_SIZE;
                if (local_row < BLOCK_M) {
                    boundary_smem[local_row] = local_row < valid_rows
                        ? KE[row_start + local_row]
                        : 0;
                }
            }
        }

        bool row_overlaps = false;
#pragma unroll
        for (int k = 0; k < rows_per_lane; ++k) {
            const int local_row = lane + k * C10_WARP_SIZE;
            if (local_row < valid_rows) {
                const int row = row_start + local_row;
                const int ks = static_cast<int>(KS[row]);
                const int ke = static_cast<int>(KE[row]);
                row_overlaps |= ks < ke && ke > tile_start && ks < tile_end;
            }
        }
        const uint64_t overlap_mask = __ballot(row_overlaps);
        return overlap_mask != 0;
    } else if constexpr (BLOCK_M >= 32) {
        const bool first_wave = tid < C10_WARP_SIZE;
        const bool valid_row = lane < valid_rows;
        uint32_t ke_raw = 0;
        int ks = 0;
        int ke = 0;

        // The first wave publishes KE for the existing store masks. Their
        // later compute barriers make these writes visible before
        // boundary_smem is consumed. Other waves defer per-row KE loads until
        // a full scan is actually needed.
        if (first_wave && valid_row) {
            const int row = row_start + lane;
            ke_raw = KE[row];
            ke = static_cast<int>(ke_raw);
        }
        if (tid < BLOCK_M) {
            boundary_smem[tid] = ke_raw;
        }

        // This probe is only a positive witness. If it misses, the full scan
        // below is still mandatory because KS/KE need not be monotonic.
        bool probe_overlaps = false;
        if (lane == probe_local_row) {
            const int row = row_start + lane;
            if (!first_wave) {
                ke = static_cast<int>(KE[row]);
            }
            ks = static_cast<int>(KS[row]);
            probe_overlaps =
                ks < ke && ke > tile_start && ks < tile_end;
        }
        const uint64_t probe_mask = __ballot(probe_overlaps);
        if (probe_mask != 0) {
            return true;
        }

        bool row_overlaps = false;
        if (valid_row) {
            const int row = row_start + lane;
            if (!first_wave && lane != probe_local_row) {
                ke = static_cast<int>(KE[row]);
            }
            if (lane != probe_local_row) {
                ks = static_cast<int>(KS[row]);
            }
            // KS/KE arrive as int32 tensors through uint32_t containers.
            // Signed comparisons handle negative and reversed intervals.
            row_overlaps =
                ks < ke && ke > tile_start && ks < tile_end;
        }
        const uint64_t overlap_mask = __ballot(row_overlaps);
        return overlap_mask != 0;
    } else {
        const bool valid_row = tid < valid_rows;
        int ks = 0;
        int ke = 0;

        if (tid < BLOCK_M) {
            if (valid_row) {
                const uint32_t ke_raw = KE[row_start + tid];
                boundary_smem[tid] = ke_raw;
                ke = static_cast<int>(ke_raw);
            } else {
                boundary_smem[tid] = 0;
            }
        }

        bool probe_overlaps = false;
        if (tid == probe_local_row) {
            ks = static_cast<int>(KS[row_start + tid]);
            probe_overlaps =
                ks < ke && ke > tile_start && ks < tile_end;
        }
        const uint64_t probe_mask = __ballot(probe_overlaps);
        if (tid == 0) {
            *tile_has_work_smem = probe_mask != 0;
        }
        __syncthreads();
        if (*tile_has_work_smem != 0) {
            return true;
        }

        bool row_overlaps = false;
        if (valid_row) {
            if (tid != probe_local_row) {
                ks = static_cast<int>(KS[row_start + tid]);
            }
            row_overlaps =
                ks < ke && ke > tile_start && ks < tile_end;
        }
        const uint64_t overlap_mask = __ballot(row_overlaps);
        if (tid == 0) {
            *tile_has_work_smem = overlap_mask != 0;
        }
        __syncthreads();
        return *tile_has_work_smem != 0;
    }
}

template <bool is_half = true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, floatx4& reg_c) {
#if defined(__gfx936__) || defined(__gfx928__)
    if constexpr (is_half)
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
    else
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c);

#elif defined(__gfx938__)
    if constexpr (is_half)
        reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false); 
    else
        reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
#endif
}

template <bool is_e4m3 = true>
inline __device__ void builtin_amdgcn_mmac_fp8(const intx2& reg_a, const intx2& reg_b, floatx4& reg_c) {
    #if defined(__gfx936__) || defined(__gfx928__)
        return;
    #elif defined(__gfx938__)
        if constexpr (is_e4m3){
            reg_c =__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(reg_a, reg_b, reg_c, false, false); 
        }else{
            reg_c =__builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(reg_a, reg_b, reg_c, false, false); 
        }
    #endif
}

inline __device__ void builtin_amdgcn_mmac_i8(const intx2& reg_a, const intx2& reg_b, intx4& reg_c) {
#if defined(__gfx936__) || defined(__gfx928__)
    auto a = reinterpret_cast<const long*>(&reg_a);
    auto b = reinterpret_cast<const long*>(&reg_b);
    reg_c = __builtin_amdgcn_mmac_i32_16x16x32i8(*a, *b, reg_c);
#elif defined(__gfx938__)
    reg_c = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(
        reg_a, reg_b, reg_c, false, false, false);
#endif
}

template <uint32_t kNextN, uint32_t kNumWarps>
__global__ __launch_bounds__(kNumWarps * 64, 1) void clean_logits(
    const uint32_t seq_len,
    const uint32_t seq_len_kv,
    const uint32_t stride_kv,
    const uint32_t* cu_seq_len_k_start,
    const uint32_t* cu_seq_len_k_end,
    float* logits) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
    const floatx2 neg_infx2 = {neg_inf, neg_inf};

    const uint32_t num_threads_per_block = kNumWarps * 64;
    const uint32_t tid = threadIdx.x;

    for (uint32_t i = blockIdx.x; i < seq_len; i += gridDim.x) {
        const uint32_t row_group = i / kNextN;
        const uint32_t ks = (cu_seq_len_k_start == nullptr) ? 0 : cu_seq_len_k_start[row_group];
        const uint32_t ke = cu_seq_len_k_end[row_group] - kNextN + (i % kNextN) + 1;
        const uint64_t row_base_addr = i * stride_kv;
        for (uint32_t j = tid; j < ks; j += num_threads_per_block) {
            logits[row_base_addr + j] = neg_inf;
        }
        const uint32_t num_elements_to_clean = seq_len_kv - ke;
        if (ke % 4 == 0 && num_elements_to_clean % 4 == 0) {
            const uint32_t num_f4_chunks = num_elements_to_clean / 4;
            floatx4* start_ptr_f4 = reinterpret_cast<floatx4*>(&logits[row_base_addr + ke]);
            for (uint32_t j_f4 = tid; j_f4 < num_f4_chunks; j_f4 += num_threads_per_block) {
                start_ptr_f4[j_f4] = neg_infx4;
            }

        } else if (ke % 2 == 0 && num_elements_to_clean % 2 == 0) {
            const uint32_t num_f2_chunks = num_elements_to_clean / 2;
            floatx2* start_ptr_f2 = reinterpret_cast<floatx2*>(&logits[row_base_addr + ke]);
            for (uint32_t j_f2 = tid; j_f2 < num_f2_chunks; j_f2 += num_threads_per_block) {
                start_ptr_f2[j_f2] = neg_infx2;
            }
        } else {
            for (uint32_t j = ke + tid; j < seq_len_kv; j += num_threads_per_block) {
                logits[row_base_addr + j] = neg_inf;
            }
        }
    }
}

// One block per row (grid-stride over rows). Threads cooperatively write -inf to
// [0, ks) and [ke, N). Vectorized stores are used when row and edge alignment
// make them safe.
template <int kNumWarps>
__device__ __forceinline__ void clean_logits_mqa_core(
    int M,
    int N,
    int stride_d,
    const uint32_t* __restrict__ KS,
    const uint32_t* __restrict__ KE,
    float* __restrict__ D) {
    using cl_floatx4 = float __attribute__((ext_vector_type(4)));
    using cl_floatx2 = float __attribute__((ext_vector_type(2)));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const cl_floatx4 ninf4 = {neg_inf, neg_inf, neg_inf, neg_inf};
    const cl_floatx2 ninf2 = {neg_inf, neg_inf};
    constexpr int nthreads = kNumWarps * 64;
    const int tid = threadIdx.x;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        // KS/KE are torch::kInt tensors passed through uint32_t* containers.
        // Interpret the 32-bit values as signed before clamping.
        int ks = (KS == nullptr) ? 0 : static_cast<int>(KS[row]);
        int ke = static_cast<int>(KE[row]);
        ks = ks < 0 ? 0 : (ks > N ? N : ks);
        ke = ke < 0 ? 0 : (ke > N ? N : ke);
        const size_t base = static_cast<size_t>(row) * stride_d;

        for (int j = tid; j < ks; j += nthreads) {
            D[base + j] = neg_inf;
        }

        if (ke >= N) {
            continue;
        }
        const int nclean = N - ke;
        if ((N % 4 == 0) && (ke % 4 == 0)) {
            cl_floatx4* p = reinterpret_cast<cl_floatx4*>(&D[base + ke]);
            const int chunks = nclean / 4;
            for (int c = tid; c < chunks; c += nthreads) {
                p[c] = ninf4;
            }
        } else if ((N % 2 == 0) && (ke % 2 == 0)) {
            cl_floatx2* p = reinterpret_cast<cl_floatx2*>(&D[base + ke]);
            const int chunks = nclean / 2;
            for (int c = tid; c < chunks; c += nthreads) {
                p[c] = ninf2;
            }
        } else {
            for (int j = ke + tid; j < N; j += nthreads) {
                D[base + j] = neg_inf;
            }
        }
    }
}

template <int kNumWarps>
__global__ __launch_bounds__(kNumWarps * 64, 1) void clean_logits_mqa(
    int M,
    int N,
    int stride_d,
    const uint32_t* KS,
    const uint32_t* KE,
    float* D) {
    clean_logits_mqa_core<kNumWarps>(M, N, stride_d, KS, KE, D);
}

template <uint32_t kNumWarps>
__global__ __launch_bounds__(kNumWarps * 64, 1) void clean_logits_padding(
    const uint32_t seq_len,
    const uint32_t seq_len_kv,
    const uint32_t stride_kv,
    const uint32_t* cu_seq_len_k_start,
    const uint32_t* cu_seq_len_k_end,
    float* logits,
    const uint32_t seq_len_padding) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
    const uint32_t num_threads_per_block = kNumWarps * 64;
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = blockIdx.x; i < seq_len_padding; i += gridDim.x) {

        const uint32_t row_group = i;
        const bool is_over_boundry = row_group >= seq_len;
        uint32_t ks = is_over_boundry ? 0 : cu_seq_len_k_start[row_group];
        const uint32_t ke = is_over_boundry ? 0  : cu_seq_len_k_end[row_group];
        const uint64_t row_base_addr = i * stride_kv;
        if (ks > seq_len_kv) {
            ks = seq_len_kv;
        }

        for (uint32_t j = tid; j < ks; j += num_threads_per_block) {
            logits[row_base_addr + j] = neg_inf;
        }

        const uint32_t start_idx = ke;
        const uint32_t end_idx = seq_len_kv;

        if (start_idx >= end_idx) {
            continue;
        }

        const uint32_t aligned_start = std::min((start_idx + 3) & (~3U), end_idx);

        for (uint32_t j = start_idx + tid; j < aligned_start; j += num_threads_per_block) {
            logits[row_base_addr + j] = neg_inf;
        }

        const uint32_t aligned_end = end_idx & (~3U);

        if (aligned_start < aligned_end) {

            floatx4* start_ptr_f4 = reinterpret_cast<floatx4*>(&logits[row_base_addr + aligned_start]);
            const uint32_t num_f4_chunks = (aligned_end - aligned_start) / 4;
            for (uint32_t j_f4 = tid; j_f4 < num_f4_chunks; j_f4 += num_threads_per_block) {
                start_ptr_f4[j_f4] = neg_infx4;
            }
        }
        for (uint32_t j = aligned_end + tid; j < end_idx; j += num_threads_per_block) {
            logits[row_base_addr + j] = neg_inf;
        }
    }
}
template<typename scalar_t, bool is_align_n>
inline __device__ void
loadtileB2reg(half4x8 &B_reg, scalar_t *B_data, const int row, const int col, const int head_dim, const int max_rows) {
    if constexpr(is_align_n) {
#pragma unroll 8
        for (int i = 0; i < 8; i++) {
            B_reg.data[i] = FETCH_HALF4(B_data[row * head_dim + col + i * 16]);
        }
    } else {
        if (row >= max_rows) {
            for (int i = 0; i < 8; i++) {
                *reinterpret_cast<uint64_t*>(&B_reg.data[i]) = 0;
            }
        }else{
            for (int i = 0; i < 8; i++) {
                B_reg.data[i] = FETCH_HALF4(B_data[row * head_dim + col + i * 16]);
            }
        }
    }
}
//FP8 Loadtile B
template<typename scalar_t, bool is_align_n>
inline __device__ void
loadtileB2reg_fp8(intx2 B_reg[4], scalar_t *B_data, const int row, const int col, const int head_dim, const int max_rows) {
    uint64_t* B_ptr = (uint64_t*)B_reg;
    if constexpr(is_align_n) {
#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            //load 8个
            B_ptr[i] = *(uint64_t*)(&B_data[row * head_dim + col + i * 32]);
        }
    } else {
        if (row >= max_rows) {
            for (int i = 0; i < 4; i++) {
                B_ptr[i] = 0;
            }
        } else {
            for (int i = 0; i < 4; i++) {
                B_ptr[i] = *(uint64_t*)(&B_data[row * head_dim + col + i * 32]);
            }
        }
    }
}
template <typename scalar_t>
__device__ __forceinline__ void cp_async4_vgpr(intx4* dst_vgpr, const scalar_t* glob_ptr, int offset_v, bool pred) {
    if (pred) {
        intx4 res;
        typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
        uint32x4_t global_addr = {0};
        *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
        global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
        global_addr[2] = 0x80000000;
        global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
        //  000
        asm volatile(
            "buffer_load_dwordx4 %0, %1, %2 ,0 offen slc\n"
            " \n\t"
            : "=v"(res),
            "+v"(offset_v), "+s"(global_addr));

        *reinterpret_cast<intx4*>(dst_vgpr) = res;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void cp_async2_vgpr(intx2* dst_vgpr, const scalar_t* glob_ptr, int offset_v, bool pred) {
    if (pred) {
        intx2 res;
        typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
        uint32x4_t global_addr = {0};
        *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
        global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
        global_addr[2] = 0x80000000;
        global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
        //  000
        asm volatile(
            "buffer_load_dwordx2 %0, %1, %2 ,0 offen slc\n"
            " \n\t"
            : "=v"(res),
            "+v"(offset_v), "+s"(global_addr));

        *reinterpret_cast<intx2*>(dst_vgpr) = res;
    }
}
/*fp8, KV_scale_data*/
template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x64_TN_impl_with_weight_prefetch_fp8(
    fp8* A_data,
    fp8* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {

    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;

    __align__(16) __shared__ fp8 A_Seme[2][16 * 136];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ float W_Seme[2][BlOCK_M];

    __shared__ float C_smem[BlOCK_M][BLOCK_N + 4];

    // Split-K
    const int num_split = gridDim.z;
    const int split_idx = blockIdx.z;
    const int heads_per_split = (num_heads + num_split - 1) / num_split;
    const int start_h = split_idx * heads_per_split;
    const int end_h = (start_h + heads_per_split) < num_heads ? (start_h + heads_per_split) : num_heads;

    if (start_h >= end_h) return;

    if (tid < BlOCK_M) {
        if constexpr (is_align_m) {
            boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
        } else {
            if (blockIdx_y * BlOCK_M + tid < M) {
                boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
            } else {
                boundary_Seme[tid] = 0;
            }
        }
    }

    floatx4 C_reg_out = {0.0, 0.0, 0.0, 0.0};

    intx2 B_reg[4];
    intx2 A_reg_move = {0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;

    int ldg_row_B = (lane / 16) * 8;
    int ldg_col_B = blockIdx_x * BLOCK_N + warp_idx * 16 + (lane % 16);

    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8;
    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;

    const int stg_row = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;

    float scale0 = 1.0f, scale1 = 1.0f, scale2 = 1.0f, scale3 = 1.0f;
    if (KV_scale_data != nullptr) {
        if (stg_col < N) scale0 = KV_scale_data[stg_col];
        if (stg_col + 4 < N) scale1 = KV_scale_data[stg_col + 4];
        if (stg_col + 8 < N) scale2 = KV_scale_data[stg_col + 8];
        if (stg_col + 12 < N) scale3 = KV_scale_data[stg_col + 12];
    }

    loadtileB2reg_fp8<fp8, is_align_n>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);

    ldg_col_A += start_h * 128;

   // A->reg
    if (ldg_row_A < M) {
        *(uint64_t*)&A_reg_move = *(uint64_t*)&A_data[ldg_row_A * K + ldg_col_A];
    }

    *(uint64_t*)&A_Seme[0][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
    if constexpr (is_align_m) {
        if (tid < 16) {
            W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
        }
    } else {
        if (tid < 16) {
            if (stg_row < M) {
                W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128; // 滑动到下一个head;

    const int num_iters = end_h - start_h;
    for (int iter = 0; iter < num_iters; iter++) {
        int i = start_h + iter;

        int offset;
        if constexpr (is_align_m) {
            offset = (ldg_row_A * K + ldg_col_A) * 1;
        } else {
            if (ldg_row_A < M) {
                offset = (ldg_row_A * K + ldg_col_A) * 1;
            } else {
                offset = -1;
            }
        }

        bool is_load = (iter + 1 < num_iters);
        cp_async2_vgpr(&A_reg_move, A_data, offset, is_load);

        floatx4 C_reg = {0.0, 0.0, 0.0, 0.0};
        asm volatile("s_waitcnt lgkmcnt(0)");

        const int buf_idx = iter % 2;

        intx2 A_reg[4];
        //mma
        for (int k = 0; k < 4; k++) {
            A_reg[k] = *(intx2*)&A_Seme[buf_idx][lds_row_A * 136 + lds_col_A + k * 32];
            builtin_amdgcn_mmac_fp8(A_reg[k], B_reg[k], C_reg);
        }

        float w = W_Seme[buf_idx][lane % 16];

        // relu 和 w
        C_reg[0] = fmaxf(C_reg[0], 0.0f); C_reg_out[0] += C_reg[0] * w;
        C_reg[1] = fmaxf(C_reg[1], 0.0f); C_reg_out[1] += C_reg[1] * w;
        C_reg[2] = fmaxf(C_reg[2], 0.0f); C_reg_out[2] += C_reg[2] * w;
        C_reg[3] = fmaxf(C_reg[3], 0.0f); C_reg_out[3] += C_reg[3] * w;

        if (iter + 1 < num_iters) {
            if (tid < 16) {
                const int next_head_idx = i + 1;
                const int next_buf_idx = 1 - buf_idx;
                if constexpr(is_align_m) {
                    W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                } else {
                    if (stg_row < M) {
                        W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                    } else {
                        W_Seme[next_buf_idx][tid] = 0.0f;
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)");
            *(uint64_t*)&A_Seme[1 - buf_idx][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
            ldg_col_A += 128;
        }
        asm volatile("s_barrier");
    }

    // 反量化
    C_reg_out[0] *= scale0;
    C_reg_out[1] *= scale1;
    C_reg_out[2] *= scale2;
    C_reg_out[3] *= scale3;


    int local_row = lane % 16;
    int local_col = lane / 16 + warp_idx * 16;

    C_smem[local_row][local_col]      = C_reg_out[0];
    C_smem[local_row][local_col + 4]  = C_reg_out[1];
    C_smem[local_row][local_col + 8]  = C_reg_out[2];
    C_smem[local_row][local_col + 12] = C_reg_out[3];

    __syncthreads();

    bool use_atomic = (num_split > 1);
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const int total_elements = BlOCK_M * BLOCK_N;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int e_row = i / BLOCK_N;
        int e_col = i % BLOCK_N;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col;

        if (global_row < M && global_col < N) {
            int boundary = boundary_Seme[e_row];
            if (global_col < boundary) {
                float val = C_smem[e_row][e_col];
                if (use_atomic) {
                    atomicAdd(&D_data[global_row * N + global_col], val);
                } else {
                    D_data[global_row * N + global_col] = val;
                }
            } else {
                if (!use_atomic || split_idx == 0) {
                    D_data[global_row * N + global_col] = neg_inf;
                }
            }
        }
    }
}

template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x128_TN_impl_fp8(
    fp8* A_data,
    fp8* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {

    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;
    const int tid = threadIdx.x;

    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;

    __align__(16) __shared__ fp8 A_Seme[2][16 * 136];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ float W_Seme[2][BlOCK_M];

    if (tid < BlOCK_M) {
        if constexpr (is_align_m) {
            boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
        } else {
            if (blockIdx_y * BlOCK_M + tid < M) {
                boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
            } else {
                boundary_Seme[tid] = 0; // 越界行对应的边界设为 0
            }
        }
    }

    /* block mask */
    if (M > 0) {
        const int ke_max_idx_in_block = blockIdx_y * BlOCK_M + BlOCK_M - 1;
        const int clamped_ke_idx = (ke_max_idx_in_block < M) ? ke_max_idx_in_block : (M - 1);
        if (blockIdx_x * BLOCK_N >= KE[clamped_ke_idx]) {
            const float neg_inf = -std::numeric_limits<float>::infinity();
            const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
            const int total_vec4 = (BlOCK_M * BLOCK_N) / 4;
            for (int i = tid; i < total_vec4; i += blockDim.x) {
                int r = i / (BLOCK_N / 4);
                int c = (i % (BLOCK_N / 4)) * 4;
                int global_row = blockIdx_y * BlOCK_M + r;
                int global_col = blockIdx_x * BLOCK_N + c;
                if (global_row < M) {
                    if (is_align_n && global_col + 3 < N) {
                        *(floatx4*)(&D_data[global_row * N + global_col]) = neg_infx4;
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            if (global_col + k < N) {
                                D_data[global_row * N + global_col + k] = neg_inf;
                            }
                        }
                    }
                }
            }
            return;
        }
    }

    floatx4 C_reg_out[2] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};


    intx2 A_reg_move = {0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;


    int ldg_row_B = (lane / 16) * 8;
    int ldg_col_B[2];
    ldg_col_B[0] = blockIdx_x * BLOCK_N + warp_idx * 32 + (lane % 16);
    ldg_col_B[1] = blockIdx_x * BLOCK_N + warp_idx * 32 + 16 + (lane % 16);

    intx2 B_reg[2][4];

    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8;

    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;


    loadtileB2reg_fp8<fp8, is_align_n>(B_reg[0], B_data, ldg_col_B[0], ldg_row_B, head_dim, N);
    loadtileB2reg_fp8<fp8, is_align_n>(B_reg[1], B_data, ldg_col_B[1], ldg_row_B, head_dim, N);


    if (ldg_row_A < M) {
        *(uint64_t*)&A_reg_move = *(uint64_t*)&A_data[ldg_row_A * K + ldg_col_A];
    }
    *(uint64_t*)&A_Seme[0][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;

    if (tid < 16) {
        int w_row = blockIdx_y * BlOCK_M + tid;
        if constexpr (is_align_m) {
            W_Seme[0][tid] = Weights_data[w_row * num_heads];
        } else {
            if (w_row < M) {
                W_Seme[0][tid] = Weights_data[w_row * num_heads];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128;

    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        asm volatile("s_barrier");

        bool is_load_next = (i + 1 < num_heads);

        int offset_bytes = -1;
        if constexpr (is_align_m) {
            offset_bytes = (ldg_row_A * K + ldg_col_A);
        } else {
            if (ldg_row_A < M) {
                offset_bytes = (ldg_row_A * K + ldg_col_A);
            }
        }


        cp_async2_vgpr(&A_reg_move, A_data, offset_bytes, is_load_next);


        float w_reg_next = 0.0f;
        if (is_load_next) {
            if (tid < 16) {
                int w_row = blockIdx_y * BlOCK_M + tid;
                if constexpr (is_align_m) {
                    w_reg_next = Weights_data[w_row * num_heads + (i + 1)];
                } else {
                    w_reg_next = (w_row < M) ? Weights_data[w_row * num_heads + (i + 1)] : 0.0f;
                }
            }
        }

        floatx4 C_reg[2] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
        asm volatile("s_waitcnt lgkmcnt(0)");

        intx2 A_reg[4];
        for (int k = 0; k < 4; k++) {
            A_reg[k] = *(intx2*)&A_Seme[i % 2][lds_row_A * 136 + lds_col_A + k * 32];
            builtin_amdgcn_mmac_fp8(A_reg[k], B_reg[0][k], C_reg[0]);
            builtin_amdgcn_mmac_fp8(A_reg[k], B_reg[1][k], C_reg[1]);
        }


        float w = W_Seme[i % 2][lane % 16];


        #pragma unroll
        for(int g = 0; g < 2; ++g) {
            C_reg_out[g][0] += fmaxf(C_reg[g][0], 0.0f) * w;
            C_reg_out[g][1] += fmaxf(C_reg[g][1], 0.0f) * w;
            C_reg_out[g][2] += fmaxf(C_reg[g][2], 0.0f) * w;
            C_reg_out[g][3] += fmaxf(C_reg[g][3], 0.0f) * w;
        }
        if (is_load_next) {
            asm volatile("s_waitcnt vmcnt(0)");

            *(uint64_t*)&A_Seme[1 - i % 2][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
            ldg_col_A += 128;

            if (tid < 16) {
                W_Seme[1 - i % 2][tid] = w_reg_next;
            }
        }
    }

    float* C_smem_ptr = reinterpret_cast<float*>(A_Seme);
    int local_row = lane % 16;


    int local_col_0 = warp_idx * 32 + lane / 16;
    int local_col_1 = warp_idx * 32 + 16 + lane / 16;

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    if (warp_idx < 2) {
        C_smem_ptr[local_row * 68 + local_col_0]      = C_reg_out[0][0];
        C_smem_ptr[local_row * 68 + local_col_0 + 4]  = C_reg_out[0][1];
        C_smem_ptr[local_row * 68 + local_col_0 + 8]  = C_reg_out[0][2];
        C_smem_ptr[local_row * 68 + local_col_0 + 12] = C_reg_out[0][3];

        C_smem_ptr[local_row * 68 + local_col_1]      = C_reg_out[1][0];
        C_smem_ptr[local_row * 68 + local_col_1 + 4]  = C_reg_out[1][1];
        C_smem_ptr[local_row * 68 + local_col_1 + 8]  = C_reg_out[1][2];
        C_smem_ptr[local_row * 68 + local_col_1 + 12] = C_reg_out[1][3];
    }

    __syncthreads();

    {
        int e_row = tid / 16;
        int e_col = (tid % 16) * 4;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col; // Offset 0

        if (global_row < M && global_col < N) {
            floatx4 val = *(floatx4*)(&C_smem_ptr[e_row * 68 + e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[global_row * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[global_row * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[global_row * N + global_col + k] = neg_inf;
                }
            }
        }
    }

    __syncthreads();

    if (warp_idx >= 2) {
        int smem_col_0 = local_col_0 - 64;
        int smem_col_1 = local_col_1 - 64;

        C_smem_ptr[local_row * 68 + smem_col_0]      = C_reg_out[0][0];
        C_smem_ptr[local_row * 68 + smem_col_0 + 4]  = C_reg_out[0][1];
        C_smem_ptr[local_row * 68 + smem_col_0 + 8]  = C_reg_out[0][2];
        C_smem_ptr[local_row * 68 + smem_col_0 + 12] = C_reg_out[0][3];

        C_smem_ptr[local_row * 68 + smem_col_1]      = C_reg_out[1][0];
        C_smem_ptr[local_row * 68 + smem_col_1 + 4]  = C_reg_out[1][1];
        C_smem_ptr[local_row * 68 + smem_col_1 + 8]  = C_reg_out[1][2];
        C_smem_ptr[local_row * 68 + smem_col_1 + 12] = C_reg_out[1][3];
    }

    __syncthreads();

    {
        int e_row = tid / 16;
        int e_col = (tid % 16) * 4;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col + 64; // Offset 64

        if (global_row < M && global_col < N) {
            floatx4 val = *(floatx4*)(&C_smem_ptr[e_row * 68 + e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[global_row * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[global_row * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[global_row * N + global_col + k] = neg_inf;
                }
            }
        }
    }
}

template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_32x128x128_TN_impl_fp8(
    fp8* A_data,
    fp8* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {

    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;
    const int tid = threadIdx.x;

    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;

    __align__(16) __shared__ union {
        fp8 A_Seme[2][32 * 136];
        float C_smem[32][68];
    } smem;
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ float W_Seme_Cache[2][32][8];

    if (tid < BlOCK_M) {
        if constexpr (is_align_m) {
            boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
        } else {
            if (blockIdx_y * BlOCK_M + tid < M) {
                boundary_Seme[tid] = KE[blockIdx_y * BlOCK_M + tid];
            } else {
                boundary_Seme[tid] = 0;
            }
        }
    }

    /* block mask */
    if (M > 0) {
        const int ke_max_idx_in_block = blockIdx_y * BlOCK_M + BlOCK_M - 1;
        const int clamped_ke_idx = (ke_max_idx_in_block < M) ? ke_max_idx_in_block : (M - 1);
        if (blockIdx_x * BLOCK_N >= KE[clamped_ke_idx]) {
            constexpr float neg_inf = -std::numeric_limits<float>::infinity();
            constexpr floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
            const int total_vec4 = (BlOCK_M * BLOCK_N) / 4;
            for (int i = tid; i < total_vec4; i += blockDim.x) {
                int r = i / (BLOCK_N / 4);
                int c = (i % (BLOCK_N / 4)) * 4;
                int global_row = blockIdx_y * BlOCK_M + r;
                int global_col = blockIdx_x * BLOCK_N + c;
                if (global_row < M) {
                    if (is_align_n && global_col + 3 < N) {
                        *(floatx4*)(&D_data[(int64_t)global_row * N + global_col]) = neg_infx4;
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            if (global_col + k < N) {
                                D_data[(int64_t)global_row * N + global_col + k] = neg_inf;
                            }
                        }
                    }
                }
            }
            return;
        }
    }

    floatx4 C_reg_out[2][2];
    C_reg_out[0][0] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[0][1] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[1][0] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[1][1] = {0.0, 0.0, 0.0, 0.0};

    intx4 A_reg_move = {0, 0, 0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 8);
    int ldg_col_A = (tid % 8) * 16;

    int ldg_row_B = (lane / 16) * 8;
    int ldg_col_B[2];
    ldg_col_B[0] = blockIdx_x * BLOCK_N + warp_idx * 32 + (lane % 16);
    ldg_col_B[1] = blockIdx_x * BLOCK_N + warp_idx * 32 + 16 + (lane % 16);

    intx2 B_reg[2][4];

    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8;

    int sts_row_A = (tid / 8);
    int sts_col_A = (tid % 8) * 16;

    loadtileB2reg_fp8<fp8, is_align_n>(B_reg[0], B_data, ldg_col_B[0], ldg_row_B, head_dim, N);
    loadtileB2reg_fp8<fp8, is_align_n>(B_reg[1], B_data, ldg_col_B[1], ldg_row_B, head_dim, N);

    if (ldg_row_A < M) {
        A_reg_move = *(intx4*)&A_data[(int64_t)ldg_row_A * K + ldg_col_A];
    }
    *(intx4*)&smem.A_Seme[0][sts_row_A * 136 + sts_col_A] = A_reg_move;
    ldg_col_A += 128;

    if (1 < num_heads) {
        int64_t offset_bytes = (ldg_row_A < M) ? ((int64_t)ldg_row_A * K + ldg_col_A) : -1;
        cp_async4_vgpr(&A_reg_move, A_data, offset_bytes, true);
    }

    if (tid < 256) {
        int r = tid / 8;
        int c = tid % 8;
        if (c < num_heads) {
            int w_row = blockIdx_y * BlOCK_M + r;
            W_Seme_Cache[0][r][c] = (w_row < M) ? Weights_data[(int64_t)w_row * num_heads + c] : 0.f;
        }
    }

    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        if (i % 8 == 0 && (i + 8) < num_heads) {
            int next_buf = 1 - (i / 8) % 2;
            int next_i = i + 8;
            if (tid < 256) {
                int r = tid / 8;
                int c = tid % 8;
                if (next_i + c < num_heads) {
                    int w_row = blockIdx_y * BlOCK_M + r;
                    W_Seme_Cache[next_buf][r][c] = (w_row < M) ? Weights_data[(int64_t)w_row * num_heads + next_i + c] : 0.f;
                }
            }
        }

        floatx4 C_reg[2][2];
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                C_reg[k][j] = {0.0, 0.0, 0.0, 0.0};
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0)");

        intx2 A_reg[2][4];
        for (int k = 0; k < 4; k++) {
            A_reg[0][k] = *(intx2*)&smem.A_Seme[i % 2][lds_row_A * 136 + lds_col_A + k * 32];
            A_reg[1][k] = *(intx2*)&smem.A_Seme[i % 2][(lds_row_A + 16) * 136 + lds_col_A + k * 32];

            builtin_amdgcn_mmac_fp8(A_reg[0][k], B_reg[0][k], C_reg[0][0]);
            builtin_amdgcn_mmac_fp8(A_reg[0][k], B_reg[1][k], C_reg[0][1]);
            builtin_amdgcn_mmac_fp8(A_reg[1][k], B_reg[0][k], C_reg[1][0]);
            builtin_amdgcn_mmac_fp8(A_reg[1][k], B_reg[1][k], C_reg[1][1]);
        }
        float w1 = W_Seme_Cache[(i / 8) % 2][lane % 16][i % 8];
        float w2 = W_Seme_Cache[(i / 8) % 2][lane % 16 + 16][i % 8];

        #pragma unroll
        for(int g = 0; g < 2; ++g) {
            C_reg_out[0][g][0] += fmaxf(C_reg[0][g][0], 0.0f) * w1;
            C_reg_out[0][g][1] += fmaxf(C_reg[0][g][1], 0.0f) * w1;
            C_reg_out[0][g][2] += fmaxf(C_reg[0][g][2], 0.0f) * w1;
            C_reg_out[0][g][3] += fmaxf(C_reg[0][g][3], 0.0f) * w1;

            C_reg_out[1][g][0] += fmaxf(C_reg[1][g][0], 0.0f) * w2;
            C_reg_out[1][g][1] += fmaxf(C_reg[1][g][1], 0.0f) * w2;
            C_reg_out[1][g][2] += fmaxf(C_reg[1][g][2], 0.0f) * w2;
            C_reg_out[1][g][3] += fmaxf(C_reg[1][g][3], 0.0f) * w2;
        }

        if (i + 1 < num_heads) {
            asm volatile("s_waitcnt vmcnt(0)");

            *(intx4*)&smem.A_Seme[1 - i % 2][sts_row_A * 136 + sts_col_A] = A_reg_move;
            ldg_col_A += 128;
            if (i + 2 < num_heads) {
                int64_t offset_bytes = (ldg_row_A < M) ? ((int64_t)ldg_row_A * K + ldg_col_A) : -1;
                cp_async4_vgpr(&A_reg_move, A_data, offset_bytes, true);
            }
        }
    }

    int local_row = lane % 16;

    int local_col_0 = warp_idx * 32 + lane / 16;
    int local_col_1 = warp_idx * 32 + 16 + lane / 16;

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    if (warp_idx < 2) {
        smem.C_smem[local_row][local_col_0]      = C_reg_out[0][0][0];
        smem.C_smem[local_row][local_col_0 + 4]  = C_reg_out[0][0][1];
        smem.C_smem[local_row][local_col_0 + 8]  = C_reg_out[0][0][2];
        smem.C_smem[local_row][local_col_0 + 12] = C_reg_out[0][0][3];

        smem.C_smem[local_row][local_col_1]      = C_reg_out[0][1][0];
        smem.C_smem[local_row][local_col_1 + 4]  = C_reg_out[0][1][1];
        smem.C_smem[local_row][local_col_1 + 8]  = C_reg_out[0][1][2];
        smem.C_smem[local_row][local_col_1 + 12] = C_reg_out[0][1][3];

        smem.C_smem[local_row + 16][local_col_0]      = C_reg_out[1][0][0];
        smem.C_smem[local_row + 16][local_col_0 + 4]  = C_reg_out[1][0][1];
        smem.C_smem[local_row + 16][local_col_0 + 8]  = C_reg_out[1][0][2];
        smem.C_smem[local_row + 16][local_col_0 + 12] = C_reg_out[1][0][3];

        smem.C_smem[local_row + 16][local_col_1]      = C_reg_out[1][1][0];
        smem.C_smem[local_row + 16][local_col_1 + 4]  = C_reg_out[1][1][1];
        smem.C_smem[local_row + 16][local_col_1 + 8]  = C_reg_out[1][1][2];
        smem.C_smem[local_row + 16][local_col_1 + 12] = C_reg_out[1][1][3];
    }

    __syncthreads();

    {
        int e_row_0 = tid / 16;
        int e_row_1 = tid / 16 + 16;
        int e_col = (tid % 16) * 4;
        int global_row_0 = blockIdx_y * BlOCK_M + e_row_0;
        int global_row_1 = blockIdx_y * BlOCK_M + e_row_1;
        int global_col = blockIdx_x * BLOCK_N + e_col; // Offset 0

        if (global_row_0 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_0][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_0];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_0 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_0 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_0 * N + global_col + k] = neg_inf;
                }
            }
        }

        if (global_row_1 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_1][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_1];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_1 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_1 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_1 * N + global_col + k] = neg_inf;
                }
            }
        }
    }

    __syncthreads();

    if (warp_idx >= 2) {
        int smem_col_0 = local_col_0 - 64;
        int smem_col_1 = local_col_1 - 64;

        smem.C_smem[local_row][smem_col_0]      = C_reg_out[0][0][0];
        smem.C_smem[local_row][smem_col_0 + 4]  = C_reg_out[0][0][1];
        smem.C_smem[local_row][smem_col_0 + 8]  = C_reg_out[0][0][2];
        smem.C_smem[local_row][smem_col_0 + 12] = C_reg_out[0][0][3];

        smem.C_smem[local_row][smem_col_1]      = C_reg_out[0][1][0];
        smem.C_smem[local_row][smem_col_1 + 4]  = C_reg_out[0][1][1];
        smem.C_smem[local_row][smem_col_1 + 8]  = C_reg_out[0][1][2];
        smem.C_smem[local_row][smem_col_1 + 12] = C_reg_out[0][1][3];

        smem.C_smem[local_row + 16][smem_col_0]      = C_reg_out[1][0][0];
        smem.C_smem[local_row + 16][smem_col_0 + 4]  = C_reg_out[1][0][1];
        smem.C_smem[local_row + 16][smem_col_0 + 8]  = C_reg_out[1][0][2];
        smem.C_smem[local_row + 16][smem_col_0 + 12] = C_reg_out[1][0][3];

        smem.C_smem[local_row + 16][smem_col_1]      = C_reg_out[1][1][0];
        smem.C_smem[local_row + 16][smem_col_1 + 4]  = C_reg_out[1][1][1];
        smem.C_smem[local_row + 16][smem_col_1 + 8]  = C_reg_out[1][1][2];
        smem.C_smem[local_row + 16][smem_col_1 + 12] = C_reg_out[1][1][3];
    }

    __syncthreads();

    {
        int e_row_0 = tid / 16;
        int e_row_1 = tid / 16 + 16;
        int e_col = (tid % 16) * 4;
        int global_row_0 = blockIdx_y * BlOCK_M + e_row_0;
        int global_row_1 = blockIdx_y * BlOCK_M + e_row_1;
        int global_col = blockIdx_x * BLOCK_N + e_col + 64; // Offset 64

        if (global_row_0 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_0][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_0];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_0 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_0 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_0 * N + global_col + k] = neg_inf;
                }
            }
        }

        if (global_row_1 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_1][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_1];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_1 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_1 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_1 * N + global_col + k] = neg_inf;
                }
            }
        }
    }
}

/*int8, KV_scale_data (gfx936 native i8 mmac)*/
template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x64_TN_impl_with_weight_prefetch_i8(
    int8_t* A_data,
    int8_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {
    
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x; 
    const int blockIdx_y = blockIdx.y; 
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    
    __align__(16) __shared__ int8_t A_Seme[2][16 * 136];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme[2][BlOCK_M];
    
    __shared__ float C_smem[BlOCK_M][BLOCK_N + 4]; 
    
    // Split-K
    const int num_split = gridDim.z;
    const int split_idx = blockIdx.z;
    const int heads_per_split = (num_heads + num_split - 1) / num_split;
    const int start_h = split_idx * heads_per_split;
    const int end_h = (start_h + heads_per_split) < num_heads ? (start_h + heads_per_split) : num_heads;

    if (start_h >= end_h) return;

    if (!mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme)) {
        return;
    }

    floatx4 C_reg_out = {0.0, 0.0, 0.0, 0.0};
   
    intx2 B_reg[4];
    intx2 A_reg_move = {0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;

    int ldg_row_B = (lane / 16) * 8;  
    int ldg_col_B = blockIdx_x * BLOCK_N + warp_idx * 16 + (lane % 16);
    
    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8;  
    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;
    
    const int stg_row = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;
    
    float scale0 = 1.0f, scale1 = 1.0f, scale2 = 1.0f, scale3 = 1.0f;
    if (KV_scale_data != nullptr) {
        if (stg_col < N) scale0 = KV_scale_data[stg_col];
        if (stg_col + 4 < N) scale1 = KV_scale_data[stg_col + 4];
        if (stg_col + 8 < N) scale2 = KV_scale_data[stg_col + 8];
        if (stg_col + 12 < N) scale3 = KV_scale_data[stg_col + 12];
    }

    loadtileB2reg_fp8<int8_t, is_align_n>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);

    ldg_col_A += start_h * 128;

   // A->reg
    if (ldg_row_A < M) {
        *(uint64_t*)&A_reg_move = *(uint64_t*)&A_data[ldg_row_A * K + ldg_col_A];
    }
    
    *(uint64_t*)&A_Seme[0][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
    if constexpr (is_align_m) {
        if (tid < 16) {
            W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
        }
    } else {
        if (tid < 16) {
            if (stg_row < M) {
                W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128; // 滑动到下一个head;

    const int num_iters = end_h - start_h;
    for (int iter = 0; iter < num_iters; iter++) {
        int i = start_h + iter;
        
        int offset;
        if constexpr (is_align_m) {
            offset = (ldg_row_A * K + ldg_col_A) * 1; 
        } else {
            if (ldg_row_A < M) {
                offset = (ldg_row_A * K + ldg_col_A) * 1;
            } else {
                offset = -1; 
            }
        }
        
        bool is_load = (iter + 1 < num_iters); 
        cp_async2_vgpr(&A_reg_move, A_data, offset, is_load); 

        intx4 C_reg = {0, 0, 0, 0};
        asm volatile("s_waitcnt lgkmcnt(0)");
        
        const int buf_idx = iter % 2;
        
        intx2 A_reg[4];
        //mma
        for (int k = 0; k < 4; k++) {
            A_reg[k] = *(intx2*)&A_Seme[buf_idx][lds_row_A * 136 + lds_col_A + k * 32];
            builtin_amdgcn_mmac_i8(A_reg[k], B_reg[k], C_reg);
        }

        float w = W_Seme[buf_idx][lane % 16];
        
        // relu 和 w
        C_reg_out[0] += fmaxf((float)C_reg[0], 0.0f) * w;
        C_reg_out[1] += fmaxf((float)C_reg[1], 0.0f) * w;
        C_reg_out[2] += fmaxf((float)C_reg[2], 0.0f) * w;
        C_reg_out[3] += fmaxf((float)C_reg[3], 0.0f) * w;

        if (iter + 1 < num_iters) {
            if (tid < 16) {
                const int next_head_idx = i + 1;
                const int next_buf_idx = 1 - buf_idx;
                if constexpr(is_align_m) {
                    W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                } else {
                    if (stg_row < M) {
                        W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                    } else {
                        W_Seme[next_buf_idx][tid] = 0.0f;
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)");
            *(uint64_t*)&A_Seme[1 - buf_idx][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
            ldg_col_A += 128;
        }
        asm volatile("s_barrier");
    }
    
    // 反量化
    C_reg_out[0] *= scale0;
    C_reg_out[1] *= scale1;
    C_reg_out[2] *= scale2;
    C_reg_out[3] *= scale3;

 
    int local_row = lane % 16;
    int local_col = lane / 16 + warp_idx * 16;
    
    C_smem[local_row][local_col]      = C_reg_out[0];
    C_smem[local_row][local_col + 4]  = C_reg_out[1];
    C_smem[local_row][local_col + 8]  = C_reg_out[2];
    C_smem[local_row][local_col + 12] = C_reg_out[3];

    __syncthreads(); 

    bool use_atomic = (num_split > 1);
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const int total_elements = BlOCK_M * BLOCK_N;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int e_row = i / BLOCK_N;
        int e_col = i % BLOCK_N;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col;

        if (global_row < M && global_col < N) {
            int boundary = boundary_Seme[e_row]; 
            if (global_col < boundary) {
                float val = C_smem[e_row][e_col];
                if (use_atomic) {
                    atomicAdd(&D_data[global_row * N + global_col], val);
                } else {
                    D_data[global_row * N + global_col] = val;
                }
            } else {
                if (!use_atomic || split_idx == 0) {
                    D_data[global_row * N + global_col] = neg_inf;
                }
            }
        }
    }
}

template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x128_TN_impl_i8(
    int8_t* A_data,
    int8_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {
    
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x; 
    const int blockIdx_y = blockIdx.y; 
    const int tid = threadIdx.x;
    
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    
    __align__(16) __shared__ int8_t A_Seme[2][16 * 136];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme[2][BlOCK_M];

    if (!mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme)) {
        return;
    }
    
    floatx4 C_reg_out[2] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    
  
    intx2 A_reg_move = {0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;

   
    int ldg_row_B = (lane / 16) * 8;  
    int ldg_col_B[2];
    ldg_col_B[0] = blockIdx_x * BLOCK_N + warp_idx * 32 + (lane % 16);
    ldg_col_B[1] = blockIdx_x * BLOCK_N + warp_idx * 32 + 16 + (lane % 16);

    intx2 B_reg[2][4];

    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8; 
    
    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;


    loadtileB2reg_fp8<int8_t, is_align_n>(B_reg[0], B_data, ldg_col_B[0], ldg_row_B, head_dim, N);
    loadtileB2reg_fp8<int8_t, is_align_n>(B_reg[1], B_data, ldg_col_B[1], ldg_row_B, head_dim, N);

 
    if (ldg_row_A < M) {
        *(uint64_t*)&A_reg_move = *(uint64_t*)&A_data[ldg_row_A * K + ldg_col_A];
    }
    *(uint64_t*)&A_Seme[0][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;

    if (tid < 16) {
        int w_row = blockIdx_y * BlOCK_M + tid;
        if constexpr (is_align_m) {
            W_Seme[0][tid] = Weights_data[w_row * num_heads];
        } else {
            if (w_row < M) {
                W_Seme[0][tid] = Weights_data[w_row * num_heads];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128; 

    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        asm volatile("s_barrier");

        bool is_load_next = (i + 1 < num_heads);
        
        int offset_bytes = -1;
        if constexpr (is_align_m) {
            offset_bytes = (ldg_row_A * K + ldg_col_A); 
        } else {
            if (ldg_row_A < M) {
                offset_bytes = (ldg_row_A * K + ldg_col_A);
            }
        }
        
    
        cp_async2_vgpr(&A_reg_move, A_data, offset_bytes, is_load_next);

    
        float w_reg_next = 0.0f;
        if (is_load_next) {
            if (tid < 16) {
                int w_row = blockIdx_y * BlOCK_M + tid;
                if constexpr (is_align_m) {
                    w_reg_next = Weights_data[w_row * num_heads + (i + 1)];
                } else {
                    w_reg_next = (w_row < M) ? Weights_data[w_row * num_heads + (i + 1)] : 0.0f;
                }
            }
        }

        intx4 C_reg[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        asm volatile("s_waitcnt lgkmcnt(0)");
        
        intx2 A_reg[4];
        for (int k = 0; k < 4; k++) {
            A_reg[k] = *(intx2*)&A_Seme[i % 2][lds_row_A * 136 + lds_col_A + k * 32];
            builtin_amdgcn_mmac_i8(A_reg[k], B_reg[0][k], C_reg[0]);
            builtin_amdgcn_mmac_i8(A_reg[k], B_reg[1][k], C_reg[1]);
        }


        float w = W_Seme[i % 2][lane % 16];
        
  
        #pragma unroll
        for(int g = 0; g < 2; ++g) {
            C_reg_out[g][0] += fmaxf((float)C_reg[g][0], 0.0f) * w;
            C_reg_out[g][1] += fmaxf((float)C_reg[g][1], 0.0f) * w;
            C_reg_out[g][2] += fmaxf((float)C_reg[g][2], 0.0f) * w;
            C_reg_out[g][3] += fmaxf((float)C_reg[g][3], 0.0f) * w;
        }
        if (is_load_next) {
            asm volatile("s_waitcnt vmcnt(0)");
            
            *(uint64_t*)&A_Seme[1 - i % 2][sts_row_A * 136 + sts_col_A] = *(uint64_t*)&A_reg_move;
            ldg_col_A += 128;
            
            if (tid < 16) {
                W_Seme[1 - i % 2][tid] = w_reg_next;
            }
        }
    }

    float* C_smem_ptr = reinterpret_cast<float*>(A_Seme);
    int local_row = lane % 16;
    

    int local_col_0 = warp_idx * 32 + lane / 16;
    int local_col_1 = warp_idx * 32 + 16 + lane / 16;

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
 
    if (warp_idx < 2) {
        C_smem_ptr[local_row * 68 + local_col_0]      = C_reg_out[0][0];
        C_smem_ptr[local_row * 68 + local_col_0 + 4]  = C_reg_out[0][1];
        C_smem_ptr[local_row * 68 + local_col_0 + 8]  = C_reg_out[0][2];
        C_smem_ptr[local_row * 68 + local_col_0 + 12] = C_reg_out[0][3];

        C_smem_ptr[local_row * 68 + local_col_1]      = C_reg_out[1][0];
        C_smem_ptr[local_row * 68 + local_col_1 + 4]  = C_reg_out[1][1];
        C_smem_ptr[local_row * 68 + local_col_1 + 8]  = C_reg_out[1][2];
        C_smem_ptr[local_row * 68 + local_col_1 + 12] = C_reg_out[1][3];
    }
    
    __syncthreads();
    
    {
        int e_row = tid / 16;
        int e_col = (tid % 16) * 4;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col; // Offset 0

        if (global_row < M && global_col < N) {
            floatx4 val = *(floatx4*)(&C_smem_ptr[e_row * 68 + e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[global_row * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[global_row * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[global_row * N + global_col + k] = neg_inf;
                }
            }
        }
    }
    
    __syncthreads(); 
    
    if (warp_idx >= 2) {
        int smem_col_0 = local_col_0 - 64;
        int smem_col_1 = local_col_1 - 64;
        
        C_smem_ptr[local_row * 68 + smem_col_0]      = C_reg_out[0][0];
        C_smem_ptr[local_row * 68 + smem_col_0 + 4]  = C_reg_out[0][1];
        C_smem_ptr[local_row * 68 + smem_col_0 + 8]  = C_reg_out[0][2];
        C_smem_ptr[local_row * 68 + smem_col_0 + 12] = C_reg_out[0][3];

        C_smem_ptr[local_row * 68 + smem_col_1]      = C_reg_out[1][0];
        C_smem_ptr[local_row * 68 + smem_col_1 + 4]  = C_reg_out[1][1];
        C_smem_ptr[local_row * 68 + smem_col_1 + 8]  = C_reg_out[1][2];
        C_smem_ptr[local_row * 68 + smem_col_1 + 12] = C_reg_out[1][3];
    }
    
    __syncthreads();
    
    {
        int e_row = tid / 16;
        int e_col = (tid % 16) * 4;
        int global_row = blockIdx_y * BlOCK_M + e_row;
        int global_col = blockIdx_x * BLOCK_N + e_col + 64; // Offset 64

        if (global_row < M && global_col < N) {
            floatx4 val = *(floatx4*)(&C_smem_ptr[e_row * 68 + e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[global_row * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[global_row * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[global_row * N + global_col + k] = neg_inf;
                }
            }
        }
    }
}

template <int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_32x128x128_TN_impl_i8(
    int8_t* A_data,
    int8_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const float* KV_scale_data) {
    
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x; 
    const int blockIdx_y = blockIdx.y; 
    const int tid = threadIdx.x;
    
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    
    __align__(16) __shared__ union {
        int8_t A_Seme[2][32 * 136];
        float C_smem[32][68];
    } smem;
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme_Cache[2][32][8];

    if (!mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme)) {
        return;
    }
    
    floatx4 C_reg_out[2][2];
    C_reg_out[0][0] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[0][1] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[1][0] = {0.0, 0.0, 0.0, 0.0};
    C_reg_out[1][1] = {0.0, 0.0, 0.0, 0.0};
    
    intx4 A_reg_move = {0, 0, 0, 0};

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 8);
    int ldg_col_A = (tid % 8) * 16;
    
    int ldg_row_B = (lane / 16) * 8;  
    int ldg_col_B[2];
    ldg_col_B[0] = blockIdx_x * BLOCK_N + warp_idx * 32 + (lane % 16);
    ldg_col_B[1] = blockIdx_x * BLOCK_N + warp_idx * 32 + 16 + (lane % 16);

    intx2 B_reg[2][4];

    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 8; 
    
    int sts_row_A = (tid / 8);
    int sts_col_A = (tid % 8) * 16;

    loadtileB2reg_fp8<int8_t, is_align_n>(B_reg[0], B_data, ldg_col_B[0], ldg_row_B, head_dim, N);
    loadtileB2reg_fp8<int8_t, is_align_n>(B_reg[1], B_data, ldg_col_B[1], ldg_row_B, head_dim, N);

    if (ldg_row_A < M) {
        A_reg_move = *(intx4*)&A_data[(int64_t)ldg_row_A * K + ldg_col_A];
    }
    *(intx4*)&smem.A_Seme[0][sts_row_A * 136 + sts_col_A] = A_reg_move;
    ldg_col_A += 128; 

    if (1 < num_heads) {
        int64_t offset_bytes = (ldg_row_A < M) ? ((int64_t)ldg_row_A * K + ldg_col_A) : -1;
        cp_async4_vgpr(&A_reg_move, A_data, offset_bytes, true);
    }

    if (tid < 256) {
        int r = tid / 8;
        int c = tid % 8;
        if (c < num_heads) {
            int w_row = blockIdx_y * BlOCK_M + r;
            W_Seme_Cache[0][r][c] = (w_row < M) ? Weights_data[(int64_t)w_row * num_heads + c] : 0.f;
        }
    }

    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        if (i % 8 == 0 && (i + 8) < num_heads) {
            int next_buf = 1 - (i / 8) % 2;
            int next_i = i + 8;
            if (tid < 256) {
                int r = tid / 8;
                int c = tid % 8;
                if (next_i + c < num_heads) {
                    int w_row = blockIdx_y * BlOCK_M + r;
                    W_Seme_Cache[next_buf][r][c] = (w_row < M) ? Weights_data[(int64_t)w_row * num_heads + next_i + c] : 0.f;
                }
            }
        }
        
        intx4 C_reg[2][2];
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                C_reg[k][j] = {0, 0, 0, 0};
            }
        }
        
        asm volatile("s_waitcnt lgkmcnt(0)");
        
        intx2 A_reg[2][4];
        for (int k = 0; k < 4; k++) {
            A_reg[0][k] = *(intx2*)&smem.A_Seme[i % 2][lds_row_A * 136 + lds_col_A + k * 32];
            A_reg[1][k] = *(intx2*)&smem.A_Seme[i % 2][(lds_row_A + 16) * 136 + lds_col_A + k * 32];
            
            builtin_amdgcn_mmac_i8(A_reg[0][k], B_reg[0][k], C_reg[0][0]);
            builtin_amdgcn_mmac_i8(A_reg[0][k], B_reg[1][k], C_reg[0][1]);
            builtin_amdgcn_mmac_i8(A_reg[1][k], B_reg[0][k], C_reg[1][0]);
            builtin_amdgcn_mmac_i8(A_reg[1][k], B_reg[1][k], C_reg[1][1]);
        }
        float w1 = W_Seme_Cache[(i / 8) % 2][lane % 16][i % 8];
        float w2 = W_Seme_Cache[(i / 8) % 2][lane % 16 + 16][i % 8];
        
        #pragma unroll
        for(int g = 0; g < 2; ++g) {
            C_reg_out[0][g][0] += fmaxf((float)C_reg[0][g][0], 0.0f) * w1;
            C_reg_out[0][g][1] += fmaxf((float)C_reg[0][g][1], 0.0f) * w1;
            C_reg_out[0][g][2] += fmaxf((float)C_reg[0][g][2], 0.0f) * w1;
            C_reg_out[0][g][3] += fmaxf((float)C_reg[0][g][3], 0.0f) * w1;

            C_reg_out[1][g][0] += fmaxf((float)C_reg[1][g][0], 0.0f) * w2;
            C_reg_out[1][g][1] += fmaxf((float)C_reg[1][g][1], 0.0f) * w2;
            C_reg_out[1][g][2] += fmaxf((float)C_reg[1][g][2], 0.0f) * w2;
            C_reg_out[1][g][3] += fmaxf((float)C_reg[1][g][3], 0.0f) * w2;
        }

        if (i + 1 < num_heads) {
            asm volatile("s_waitcnt vmcnt(0)");
            
            *(intx4*)&smem.A_Seme[1 - i % 2][sts_row_A * 136 + sts_col_A] = A_reg_move;
            ldg_col_A += 128;
            if (i + 2 < num_heads) {
                int64_t offset_bytes = (ldg_row_A < M) ? ((int64_t)ldg_row_A * K + ldg_col_A) : -1;
                cp_async4_vgpr(&A_reg_move, A_data, offset_bytes, true);
            }
        }
    }

    int local_row = lane % 16;
    
    int local_col_0 = warp_idx * 32 + lane / 16;
    int local_col_1 = warp_idx * 32 + 16 + lane / 16;

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
 
    if (warp_idx < 2) {
        smem.C_smem[local_row][local_col_0]      = C_reg_out[0][0][0];
        smem.C_smem[local_row][local_col_0 + 4]  = C_reg_out[0][0][1];
        smem.C_smem[local_row][local_col_0 + 8]  = C_reg_out[0][0][2];
        smem.C_smem[local_row][local_col_0 + 12] = C_reg_out[0][0][3];

        smem.C_smem[local_row][local_col_1]      = C_reg_out[0][1][0];
        smem.C_smem[local_row][local_col_1 + 4]  = C_reg_out[0][1][1];
        smem.C_smem[local_row][local_col_1 + 8]  = C_reg_out[0][1][2];
        smem.C_smem[local_row][local_col_1 + 12] = C_reg_out[0][1][3];

        smem.C_smem[local_row + 16][local_col_0]      = C_reg_out[1][0][0];
        smem.C_smem[local_row + 16][local_col_0 + 4]  = C_reg_out[1][0][1];
        smem.C_smem[local_row + 16][local_col_0 + 8]  = C_reg_out[1][0][2];
        smem.C_smem[local_row + 16][local_col_0 + 12] = C_reg_out[1][0][3];

        smem.C_smem[local_row + 16][local_col_1]      = C_reg_out[1][1][0];
        smem.C_smem[local_row + 16][local_col_1 + 4]  = C_reg_out[1][1][1];
        smem.C_smem[local_row + 16][local_col_1 + 8]  = C_reg_out[1][1][2];
        smem.C_smem[local_row + 16][local_col_1 + 12] = C_reg_out[1][1][3];
    }
    
    __syncthreads();
    
    {
        int e_row_0 = tid / 16;
        int e_row_1 = tid / 16 + 16;
        int e_col = (tid % 16) * 4;
        int global_row_0 = blockIdx_y * BlOCK_M + e_row_0;
        int global_row_1 = blockIdx_y * BlOCK_M + e_row_1;
        int global_col = blockIdx_x * BLOCK_N + e_col; // Offset 0

        if (global_row_0 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_0][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_0];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_0 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_0 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_0 * N + global_col + k] = neg_inf;
                }
            }
        }
        
        if (global_row_1 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_1][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_1];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_1 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_1 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_1 * N + global_col + k] = neg_inf;
                }
            }
        }
    }
    
    __syncthreads(); 
    
    if (warp_idx >= 2) {
        int smem_col_0 = local_col_0 - 64;
        int smem_col_1 = local_col_1 - 64;
        
        smem.C_smem[local_row][smem_col_0]      = C_reg_out[0][0][0];
        smem.C_smem[local_row][smem_col_0 + 4]  = C_reg_out[0][0][1];
        smem.C_smem[local_row][smem_col_0 + 8]  = C_reg_out[0][0][2];
        smem.C_smem[local_row][smem_col_0 + 12] = C_reg_out[0][0][3];

        smem.C_smem[local_row][smem_col_1]      = C_reg_out[0][1][0];
        smem.C_smem[local_row][smem_col_1 + 4]  = C_reg_out[0][1][1];
        smem.C_smem[local_row][smem_col_1 + 8]  = C_reg_out[0][1][2];
        smem.C_smem[local_row][smem_col_1 + 12] = C_reg_out[0][1][3];

        smem.C_smem[local_row + 16][smem_col_0]      = C_reg_out[1][0][0];
        smem.C_smem[local_row + 16][smem_col_0 + 4]  = C_reg_out[1][0][1];
        smem.C_smem[local_row + 16][smem_col_0 + 8]  = C_reg_out[1][0][2];
        smem.C_smem[local_row + 16][smem_col_0 + 12] = C_reg_out[1][0][3];

        smem.C_smem[local_row + 16][smem_col_1]      = C_reg_out[1][1][0];
        smem.C_smem[local_row + 16][smem_col_1 + 4]  = C_reg_out[1][1][1];
        smem.C_smem[local_row + 16][smem_col_1 + 8]  = C_reg_out[1][1][2];
        smem.C_smem[local_row + 16][smem_col_1 + 12] = C_reg_out[1][1][3];
    }
    
    __syncthreads();
    
    {
        int e_row_0 = tid / 16;
        int e_row_1 = tid / 16 + 16;
        int e_col = (tid % 16) * 4;
        int global_row_0 = blockIdx_y * BlOCK_M + e_row_0;
        int global_row_1 = blockIdx_y * BlOCK_M + e_row_1;
        int global_col = blockIdx_x * BLOCK_N + e_col + 64; // Offset 64

        if (global_row_0 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_0][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_0];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_0 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_0 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_0 * N + global_col + k] = neg_inf;
                }
            }
        }
        
        if (global_row_1 < M && global_col < N) {
            floatx4 val = *(floatx4*)(&smem.C_smem[e_row_1][e_col]);
            if (KV_scale_data != nullptr) {
                floatx4 scale;
                if constexpr (is_align_n) scale = *(floatx4*)(&KV_scale_data[global_col]);
                else {
                    if (global_col + 3 < N) scale = *(floatx4*)(&KV_scale_data[global_col]);
                    else {
                        scale[0] = (global_col < N) ? KV_scale_data[global_col] : 1.0f;
                        scale[1] = (global_col + 1 < N) ? KV_scale_data[global_col+1] : 1.0f;
                        scale[2] = (global_col + 2 < N) ? KV_scale_data[global_col+2] : 1.0f;
                        scale[3] = (global_col + 3 < N) ? KV_scale_data[global_col+3] : 1.0f;
                    }
                }
                val[0] *= scale[0]; val[1] *= scale[1]; val[2] *= scale[2]; val[3] *= scale[3];
            }
            int boundary = boundary_Seme[e_row_1];
            if (global_col + 3 < boundary && is_align_n) {
                *(floatx4*)(&D_data[(int64_t)global_row_1 * N + global_col]) = val;
            } else {
                #pragma unroll
                for(int k = 0; k < 4; ++k) {
                    if (global_col + k < boundary) D_data[(int64_t)global_row_1 * N + global_col + k] = val[k];
                    else if (global_col + k < N) D_data[(int64_t)global_row_1 * N + global_col + k] = neg_inf;
                }
            }
        }
    }
}


/**
M = q_seq_len
K = head_dim
N = kv_seq_len
shape C(M,N) , A(M,K) , B(N,k)
compute C = A * BT
**/
template <typename scalar_t, int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x64_TN_impl_with_weight_prefetch(
    scalar_t* A_data,
    scalar_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len) {
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x; // (blockIdx.x << logtile) + ((blockIdx.y) & ((1 << (logtile)) - 1));
    const int blockIdx_y = blockIdx.y; // >> logtile;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
    __shared__ scalar_t A_Seme[2][16 * 128 + 16 * 4];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme[2][BlOCK_M];
    
    // Split-K
    const int num_split = gridDim.z;
    const int split_idx = blockIdx.z;
    const int heads_per_split = (num_heads + num_split - 1) / num_split;
    const int start_h = split_idx * heads_per_split;
    const int end_h = (start_h + heads_per_split) < num_heads ? (start_h + heads_per_split) : num_heads;

    // 如果此 block 没有分到 work，直接退出
    if (start_h >= end_h) return;

    if (!mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme)) {
        return;
    }

    floatx4 C_reg_out = {0.0, 0.0, 0.0, 0.0};
    half4x8 A_reg;
    half4x8 B_reg;
    scalar_t A_reg_move[8] = {0};

    /** ldg offset**/
    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;
    int ldg_row_B = (lane / 16) * 4;
    int ldg_col_B = blockIdx_x * BLOCK_N + warp_idx * 16 + (lane % 16);
    /** lds offset**/
    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 4;
    /** sts offset**/
    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;
    /* stg offset */
    const int stg_row = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;


    /* B转置了 */
    loadtileB2reg<scalar_t, is_align_n>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);


    ldg_col_A += start_h * 128;

    // A->reg
    if (ldg_row_A < M) {
        FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
    }
    
    FETCH_HALF8(A_Seme[0][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
    if constexpr (is_align_m) {
        if (tid < 16) {
            //  
            W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
        }
    } else {
        if (tid < 16) {
            if (stg_row < M) {
                W_Seme[0][tid] = Weights_data[stg_row * num_heads + start_h];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128; // 滑动到下一个head;


    const int num_iters = end_h - start_h;
    for (int iter = 0; iter < num_iters; iter++) {
        int i = start_h + iter;
        
        int offset;
        if constexpr (is_align_m) {
            offset = (ldg_row_A * K + ldg_col_A) * 2;
        } else {
            if (ldg_row_A < M) {
                offset = (ldg_row_A * K + ldg_col_A) * 2;
            } else {
                offset = -1; // 越界取0
            }
        }
        
        bool is_load = (iter + 1 < num_iters); 
        cp_async4_vgpr(reinterpret_cast<intx4*>(&A_reg_move[0]), A_data, offset, is_load);

        floatx4 C_reg = {0.0, 0.0, 0.0, 0.0};
        asm volatile("s_waitcnt lgkmcnt(0)");
        
        const int buf_idx = iter % 2;
        
        FETCH_HALF4(A_reg.data[0]) = FETCH_HALF4(A_Seme[buf_idx][lds_row_A * 132 + lds_col_A + 0 * 16]);
        for (int k = 0; k < 7; k++) {
            FETCH_HALF4(A_reg.data[k + 1]) = FETCH_HALF4(A_Seme[buf_idx][lds_row_A * 132 + lds_col_A + (k + 1) * 16]);
            builtin_amdgcn_mmac<is_half>(A_reg.data[k], B_reg.data[k], C_reg);
        }
        builtin_amdgcn_mmac<is_half>(A_reg.data[7], B_reg.data[7], C_reg);

        // asm volatile("s_waitcnt lgkmcnt(0)");
        float w = W_Seme[buf_idx][lane % 16];
        for (int j = 0; j < 4; j++) {
            C_reg[j] = fmaxf(C_reg[j], 0.0f);
            C_reg_out[j] += C_reg[j] * w;
        }
        //  __syncthreads();
        if (iter + 1 < num_iters) {
            if (tid < 16) {
                const int next_head_idx = i + 1;
                const int next_buf_idx = 1 - buf_idx;
                if constexpr(is_align_m) {
                    W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                } else {
                    if (stg_row < M) {
                        W_Seme[next_buf_idx][tid] = Weights_data[stg_row * num_heads + next_head_idx];
                    } else {
                        W_Seme[next_buf_idx][tid] = 0.0f;
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)");
            FETCH_HALF8(A_Seme[1 - buf_idx][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
            ldg_col_A += 128;
        }
        asm volatile("s_barrier");
    }
    const int boundary = boundary_Seme[lane % 16];
    

    bool use_atomic = (num_split > 1);
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    if (stg_row < M) {
        if (stg_col < N) {
            if (stg_col < boundary) {
                if(use_atomic) atomicAdd(&D_data[stg_row * N + stg_col], C_reg_out[0]);
                else D_data[stg_row * N + stg_col] = C_reg_out[0];
            } else {
                if ( split_idx == 0) {
                    D_data[stg_row * N + stg_col] = neg_inf;
                }
            }
        }
        if (stg_col + 4 < N) {
            if (stg_col + 4 < boundary) {
                if(use_atomic) atomicAdd(&D_data[stg_row * N + stg_col + 4], C_reg_out[1]);
                else D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
            } else {
                if (split_idx == 0) {
                    D_data[stg_row * N + stg_col + 4] = neg_inf;
                }
            }
        }
        if (stg_col + 8 < N) {
            if (stg_col + 8 < boundary) {
                if(use_atomic) atomicAdd(&D_data[stg_row * N + stg_col + 8], C_reg_out[2]);
                else D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
            } else {
                if (split_idx == 0) {
                    D_data[stg_row * N + stg_col + 8] = neg_inf;
                }
            }
        }
        if (stg_col + 12 < N) {
            if (stg_col + 12 < boundary) {
                if(use_atomic) atomicAdd(&D_data[stg_row * N + stg_col + 12], C_reg_out[3]);
                else D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
            } else {
                if (!use_atomic || split_idx == 0) {
                    D_data[stg_row * N + stg_col + 12] = neg_inf;
                }
            }
        }
    }
}

template <typename scalar_t, int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ __launch_bounds__(512, 1) void _mqa_logits_16x128x128_TN_impl(
    scalar_t* A_data,
    scalar_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len) {
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int SWIZZLE_Y_GROUP_SIZE = 4;
    const int blockIdx_x = blockIdx.x; // (blockIdx.x << logtile) + ((blockIdx.y) & ((1 << (logtile)) - 1));
    const int blockIdx_y = blockIdx.y; // >> logtile;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
    __shared__ scalar_t A_Seme[2][16 * 128 + 16 * 4];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme[2][BlOCK_M];

    const bool tile_has_work =
        mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme);

    if (!tile_has_work) {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
        const int total_vec4 = (BlOCK_M * BLOCK_N) / 4;
        for (int i = tid; i < total_vec4; i += blockDim.x) {
            int r = i / (BLOCK_N / 4);
            int c = (i % (BLOCK_N / 4)) * 4;
            int global_row = blockIdx_y * BlOCK_M + r;
            int global_col = blockIdx_x * BLOCK_N + c;
            if (global_row < M) {
                if (is_align_n && global_col + 3 < N) {
                    *(floatx4*)(&D_data[global_row * N + global_col]) = neg_infx4;
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (global_col + k < N) {
                            D_data[global_row * N + global_col + k] = neg_inf;
                        }
                    }
                }
            }
        }
        return;
    }

    floatx4 C_reg_out = {0.0, 0.0, 0.0, 0.0};
    half4x8 A_reg;
    half4x8 B_reg;
    scalar_t A_reg_move[4] = {0};

    /** ldg offset 改为一个线程load4个**/
    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 32);
    int ldg_col_A = (tid % 32) * 4;

    int ldg_row_B = (lane / 16) * 4;
    int ldg_col_B = blockIdx_x * BLOCK_N + warp_idx * 16 + (lane % 16);
    /** lds offset**/
    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 4;
    /** sts offset 为一个线程store4个**/
    int sts_row_A = (tid / 32);
    int sts_col_A = (tid % 32) * 4;
    /* stg offset */
    const int stg_row = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;

    // TODO buffer_load
    /* B转置了 */
    loadtileB2reg<scalar_t, is_align_n>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);

    // A->reg
    if (ldg_row_A < M) {
        FETCH_HALF4(A_reg_move[0]) = FETCH_HALF4(A_data[ldg_row_A * K + ldg_col_A]);
    }
    // A_reg to lds
    FETCH_HALF4(A_Seme[0][sts_row_A * 132 + sts_col_A]) = FETCH_HALF4(A_reg_move[0]);
    if (tid < BlOCK_M) {
        if constexpr (is_align_m) {
            W_Seme[0][tid] = Weights_data[stg_row * num_heads];
        } else {
            if (stg_row < M) {
                W_Seme[0][tid] = Weights_data[stg_row * num_heads];
            } else {
                W_Seme[0][tid] = 0.f;
            }
        }
    }
    __syncthreads();
    ldg_col_A += 128; // 滑动到下一个head;
    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        asm volatile("s_barrier");
        int offset_bytes;
        bool is_load_next = (i + 1 < num_heads);

        if constexpr (is_align_m) {
            offset_bytes = (ldg_row_A * K + ldg_col_A) * 2;
        } else {
            if (ldg_row_A < M) {
                offset_bytes = (ldg_row_A * K + ldg_col_A) * 2;
            } else {
                offset_bytes = -1; // 越界取 -1
            }
        }
        // 只有当有下一个 head 且 offset 有效时才 load
        cp_async2_vgpr(reinterpret_cast<intx2*>(&A_reg_move[0]), A_data, offset_bytes, is_load_next);
        // FETCH_HALF4(A_reg_move[0]) =  FETCH_HALF4(A_data[ldg_row_A*K + ldg_col_A]);
        floatx4 C_reg = {0.0, 0.0, 0.0, 0.0};
        asm volatile("s_waitcnt lgkmcnt(0)");
        for (int k = 0; k < 8; k++) {
            FETCH_HALF4(A_reg.data[k]) = FETCH_HALF4(A_Seme[i % 2][lds_row_A * 132 + lds_col_A + k * 16]);
            builtin_amdgcn_mmac<is_half>(A_reg.data[k], B_reg.data[k], C_reg);
        }
        // asm volatile("s_waitcnt lgkmcnt(0)");
        float w = W_Seme[i % 2][lane % 16];
        for (int j = 0; j < 4; j++) {
            C_reg[j] = fmaxf(C_reg[j], 0.0f); // C_reg[j] > 0.0f ? C_reg[j] : 0.0f;
            C_reg_out[j] += C_reg[j] * w;
        }

        if (i + 1 < num_heads) {
            //  __syncthreads();
            if (tid < 16) {
                if constexpr (is_align_m) {
                    W_Seme[1 - i % 2][tid] = Weights_data[stg_row * num_heads + (i + 1)];
                } else {
                    if (stg_row < M) {
                        W_Seme[1 - i % 2][tid] = Weights_data[stg_row * num_heads + (i + 1)];
                    } else {
                        W_Seme[1 - i % 2][tid] = 0.f;
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)");
            FETCH_HALF4(A_Seme[1 - i % 2][sts_row_A * 132 + sts_col_A]) = FETCH_HALF4(A_reg_move[0]);
            ldg_col_A += 128;
        }
    }

    const int boundary = boundary_Seme[lane % 16];
    if (stg_row < M) {
        if (stg_col + 12 < boundary && stg_col + 12 < N) {
            D_data[stg_row * N + stg_col] = C_reg_out[0];
            D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
            D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
            D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
        } else {
            if (stg_col < boundary && stg_col < N) {
                D_data[stg_row * N + stg_col] = C_reg_out[0];
                if (stg_col + 4 < boundary && stg_col + 4 < N) {
                    D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
                    if (stg_col + 8 < boundary && stg_col + 8 < N) {
                        D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
                        if (stg_col + 12 < boundary && stg_col + 12 < N) {
                            D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
                        }
                    }
                }
            }
        }
    }
}

template <typename scalar_t, int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N>
__global__ void _mqa_logits_32x128x64_NT_impl(
    scalar_t* A_data,
    scalar_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len) {
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;
    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;

    __shared__ scalar_t A_Seme[2][32 * 132];

    __shared__ uint32_t boundary_Seme[BlOCK_M]; // BLOCK_M = 32
    __shared__ uint32_t tile_has_work_Seme;
    __shared__ float W_Seme[2][BlOCK_M]; // BLOCK_M = 32

    const bool tile_has_work =
        mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, false>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme);

    if (!tile_has_work) {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
        const int total_vec4 = (BlOCK_M * BLOCK_N) / 4;
        for (int i = tid; i < total_vec4; i += blockDim.x) {
            int r = i / (BLOCK_N / 4);
            int c = (i % (BLOCK_N / 4)) * 4;
            int global_row = blockIdx_y * BlOCK_M + r;
            int global_col = blockIdx_x * BLOCK_N + c;
            if (global_row < M) {
                if ((N % 4 == 0) && global_col + 3 < N) {
                    *(floatx4*)(&D_data[global_row * N + global_col]) = neg_infx4;
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (global_col + k < N) {
                            D_data[global_row * N + global_col + k] = neg_inf;
                        }
                    }
                }
            }
        }
        return;
    }

    floatx4 C_reg_out1 = {0.0, 0.0, 0.0, 0.0};
    floatx4 C_reg_out2 = {0.0, 0.0, 0.0, 0.0};
    half4x16 A_reg;
    half4x8 B_reg;
    scalar_t A_reg_move[16];

    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;
    int ldg_row_B = (lane / 16) * 4;
    int ldg_col_B = blockIdx_x * BLOCK_N + warp_idx * 16 + (lane % 16);
    /** lds offset**/
    int lds_row_A = (lane % 16); // 对应 C_reg1
    int lds_col_A = (lane / 16) * 4;
    // lds_row_A + 16 对应 C_reg2
    /** sts offset**/
    int sts_row_A = (tid / 16); // 0
    int sts_col_A = (tid % 16) * 8;

    // TODO buffer_load
    /* B转置了 */
    loadtileB2reg<scalar_t, true>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);

    FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
    FETCH_HALF8(A_reg_move[8]) = FETCH_HALF8(A_data[(ldg_row_A + 16) * K + ldg_col_A]);
    FETCH_HALF8(A_Seme[0][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
    FETCH_HALF8(A_Seme[0][(sts_row_A + 16) * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[8]);

    if (tid < BlOCK_M) {
        W_Seme[0][tid] = Weights_data[(blockIdx_y * BlOCK_M + tid) * num_heads];
    }
    __syncthreads();
    ldg_col_A += 128;

    for (int i = 0; i < num_heads; i++) {
        bool is_load = i + 1 < num_heads;
        cp_async4_vgpr(reinterpret_cast<intx4*>(&A_reg_move[0]), A_data, (ldg_row_A * K + ldg_col_A) * 2,
            is_load);
        cp_async4_vgpr(reinterpret_cast<intx4*>(&A_reg_move[8]), A_data,
            ((ldg_row_A + 16) * K + ldg_col_A) * 2, is_load);

        // float W_reg_move = 0.0f;
        // if((i + 1 < num_heads) && (tid < BlOCK_M)) {
        //     W_reg_move = Weights_data[(blockIdx_y * BlOCK_M + tid) * num_heads + (i+1)];
        // }

        floatx4 C_reg1 = {0.0, 0.0, 0.0, 0.0};
        floatx4 C_reg2 = {0.0, 0.0, 0.0, 0.0};

        FETCH_HALF4(A_reg.data[0]) = FETCH_HALF4(A_Seme[i % 2][lds_row_A * 132 + lds_col_A + 0 * 16]);
        FETCH_HALF4(A_reg.data[8]) = FETCH_HALF4(A_Seme[i % 2][(lds_row_A + 16) * 132 + lds_col_A + 0 * 16]);
        for (int k = 0; k < 7; k++) {
            FETCH_HALF4(A_reg.data[k + 1]) = FETCH_HALF4(
                A_Seme[i % 2][lds_row_A * 132 + lds_col_A + (k + 1) * 16]);
            builtin_amdgcn_mmac<is_half>(A_reg.data[k], B_reg.data[k], C_reg1);

            FETCH_HALF4(A_reg.data[k + 9]) = FETCH_HALF4(
                A_Seme[i % 2][(lds_row_A + 16) * 132 + lds_col_A + (k + 1) * 16]);
            builtin_amdgcn_mmac<is_half>(A_reg.data[k + 8], B_reg.data[k], C_reg2);
        }
        builtin_amdgcn_mmac<is_half>(A_reg.data[7], B_reg.data[7], C_reg1);
        builtin_amdgcn_mmac<is_half>(A_reg.data[15], B_reg.data[7], C_reg2);

        float w1 = W_Seme[i % 2][lane % 16];
        float w2 = W_Seme[i % 2][lane % 16 + 16];
        for (int j = 0; j < 4; j++) {
            C_reg1[j] = fmaxf(C_reg1[j], 0.0f);
            C_reg_out1[j] += C_reg1[j] * w1;
            C_reg2[j] = fmaxf(C_reg2[j], 0.0f);
            C_reg_out2[j] += C_reg2[j] * w2;
        }

        if (i + 1 < num_heads) {
            asm volatile("s_waitcnt vmcnt(1)");
            FETCH_HALF8(A_Seme[1 - i % 2][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
            asm volatile("s_waitcnt vmcnt(0)");
            FETCH_HALF8(A_Seme[1 - i % 2][(sts_row_A + 16) * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[8]);

            if (tid < BlOCK_M) {
                W_Seme[1 - i % 2][tid] = Weights_data[(blockIdx_y * BlOCK_M + tid) * num_heads + (i + 1)];
            }
            ldg_col_A += 128;
        }
        asm volatile("s_barrier");
    }
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;
    const int stg_row1 = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_row2 = lane % 16 + 16 + blockIdx_y * BlOCK_M;
    const int boundary1 = boundary_Seme[lane % 16];
    const int boundary2 = boundary_Seme[lane % 16 + 16];
    if (stg_col + 12 < boundary1) {
        D_data[stg_row1 * N + stg_col] = C_reg_out1[0];
        D_data[stg_row1 * N + stg_col + 4] = C_reg_out1[1];
        D_data[stg_row1 * N + stg_col + 8] = C_reg_out1[2];
        D_data[stg_row1 * N + stg_col + 12] = C_reg_out1[3];
    } else {
        if (stg_col < boundary1) {
            D_data[stg_row1 * N + stg_col] = C_reg_out1[0];
            if (stg_col + 4 < boundary1) {
                D_data[stg_row1 * N + stg_col + 4] = C_reg_out1[1];
                if (stg_col + 8 < boundary1) {
                    D_data[stg_row1 * N + stg_col + 8] = C_reg_out1[2];
                    if (stg_col + 12 < boundary1) {
                        D_data[stg_row1 * N + stg_col + 12] = C_reg_out1[3];
                    }
                }
            }
        }
    }
    if (stg_col + 12 < boundary2) {
        D_data[stg_row2 * N + stg_col] = C_reg_out2[0];
        D_data[stg_row2 * N + stg_col + 4] = C_reg_out2[1];
        D_data[stg_row2 * N + stg_col + 8] = C_reg_out2[2];
        D_data[stg_row2 * N + stg_col + 12] = C_reg_out2[3];
    } else {
        if (stg_col < boundary2) {
            D_data[stg_row2 * N + stg_col] = C_reg_out2[0];
            if (stg_col + 4 < boundary2) {
                D_data[stg_row2 * N + stg_col + 4] = C_reg_out2[1];
                if (stg_col + 8 < boundary2) {
                    D_data[stg_row2 * N + stg_col + 8] = C_reg_out2[2];
                    if (stg_col + 12 < boundary2) {
                        D_data[stg_row2 * N + stg_col + 12] = C_reg_out2[3];
                    }
                }
            }
        }
    }
}

template <typename scalar_t, int num_heads, int head_dim, int BlOCK_M, int BLOCK_K, int BLOCK_N, bool is_align_m, bool is_align_n>
__global__ void _mqa_logits_16x128x64_NT_impl(
    scalar_t* A_data,
    scalar_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len) {
    const int M = q_seq_len;
    const int N = kv_seq_len;
    const int K = head_dim * num_heads;

    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
    __shared__ scalar_t A_Seme[2][16 * 128 + 16 * 4];
    __shared__ uint32_t boundary_Seme[BlOCK_M];
    __shared__ uint32_t tile_has_work_Seme;

    const bool tile_has_work =
        mqa_load_bounds_and_check_tile<BlOCK_M, BLOCK_N, is_align_m>(
            KS, KE, M, N, blockIdx_x, blockIdx_y,
            boundary_Seme, &tile_has_work_Seme);
    if (!tile_has_work) {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const floatx4 neg_infx4 = {neg_inf, neg_inf, neg_inf, neg_inf};
        const int total_vec4 = (BlOCK_M * BLOCK_N) / 4;
        for (int i = tid; i < total_vec4; i += blockDim.x) {
            int r = i / (BLOCK_N / 4);
            int c = (i % (BLOCK_N / 4)) * 4;
            int global_row = blockIdx_y * BlOCK_M + r;
            int global_col = blockIdx_x * BLOCK_N + c;
            if (global_row < M) {
                if (is_align_n && global_col + 3 < N) {
                    *(floatx4*)(&D_data[global_row * N + global_col]) = neg_infx4;
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (global_col + k < N) {
                            D_data[global_row * N + global_col + k] = neg_inf;
                        }
                    }
                }
            }
        }
        return;
    }

    floatx4 C_reg_out = {0.0, 0.0, 0.0, 0.0};
    half4x8 A_reg;
    half4x8 B_reg;
    // 确保 A_reg_move 被初始化，以防越界后残留值影响 LDS
    scalar_t A_reg_move[8] = {0};

    /** ldg offset**/
    int ldg_row_A = blockIdx_y * BlOCK_M + (tid / 16);
    int ldg_col_A = (tid % 16) * 8;
    int ldg_row_B = (lane / 16) * 4;
    int ldg_col_B = blockIdx.x * BLOCK_N + warp_idx * 16 + (lane % 16);
    /** lds offset**/
    int lds_row_A = (lane % 16);
    int lds_col_A = (lane / 16) * 4;
    /** sts offset**/
    int sts_row_A = (tid / 16);
    int sts_col_A = (tid % 16) * 8;
    /* stg offset */
    const int stg_row = lane % 16 + blockIdx_y * BlOCK_M;
    const int stg_col = lane / 16 + warp_idx * 16 + blockIdx_x * BLOCK_N;

    // TODO buffer_load
    /* B转置了 */
    loadtileB2reg<scalar_t, is_align_n>(B_reg, B_data, ldg_col_B, ldg_row_B, head_dim, N);

    // A->reg 增加 M 边界检查
    if constexpr (is_align_m) {
        FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
    } else {
        if (ldg_row_A < M) {
            FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
        }
    }

    // A_reg to lds
    FETCH_HALF8(A_Seme[0][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
    ldg_col_A += 128; // 滑动到下一个head;
    for (int i = 0; i < num_heads; i++) {
        __syncthreads();
        if (i + 1 < num_heads) {
            if constexpr (is_align_m) {
                FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
            } else {
                if (ldg_row_A < M) {
                    FETCH_HALF8(A_reg_move[0]) = FETCH_HALF8(A_data[ldg_row_A * K + ldg_col_A]);
                }
            }
            // *reinterpret_cast<intx4*>(&A_reg_move[0]) = builtin_amdgcn_buffer_load_reg_dwordx4(A_data,0,ldg_row_A*K + ldg_col_A);
        }
        floatx4 C_reg = {0.0, 0.0, 0.0, 0.0};
        // TODO lds预取
        for (int k = 0; k < 8; k++) {
            FETCH_HALF4(A_reg.data[k]) = FETCH_HALF4(A_Seme[i % 2][lds_row_A * 132 + lds_col_A + k * 16]);
            builtin_amdgcn_mmac<is_half>(A_reg.data[k], B_reg.data[k], C_reg);
        }
        float w;
        if constexpr (is_align_m) {
            w = Weights_data[stg_row * num_heads + i];
        } else {
            if (stg_row < M) {
                w = Weights_data[stg_row * num_heads + i];
            } else {
                w = 0.0f; // 越界权重设为 0
            }
        }
        for (int j = 0; j < 4; j++) {
            C_reg[j] = C_reg[j] > 0.0f ? C_reg[j] : 0.0f;
            C_reg_out[j] += C_reg[j] * w;
        }
        //  __syncthreads();
        if (i + 1 < num_heads) {
            // asm volatile("s_waitcnt vmcnt(1)");
            FETCH_HALF8(A_Seme[1 - i % 2][sts_row_A * 132 + sts_col_A]) = FETCH_HALF8(A_reg_move[0]);
            ldg_col_A += 128;
        }
    }
    // __syncthreads();
    const int boundary = boundary_Seme[lane % BlOCK_M];
    if constexpr (is_align_n && is_align_m) {
        // if(stg_row < M){
        if (stg_col < boundary) {
            D_data[stg_row * N + stg_col] = C_reg_out[0];
        }
        if (stg_col + 4 < boundary) {
            D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
        }
        if (stg_col + 8 < boundary) {
            D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
        }
        if (stg_col + 12 < boundary) {
            D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
        }
        //  }
    } else if (is_align_n) {
        if (stg_row < M) {
            if (stg_col < boundary) {
                D_data[stg_row * N + stg_col] = C_reg_out[0];
            }
            if (stg_col + 4 < boundary) {
                D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
            }
            if (stg_col + 8 < boundary) {
                D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
            }
            if (stg_col + 12 < boundary) {
                D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
            }
        }
    } else {
        if (stg_row < M) {
            if (stg_col < boundary && stg_col < N) {
                D_data[stg_row * N + stg_col] = C_reg_out[0];
            }
            if (stg_col + 4 < boundary && stg_col + 4 < N) {
                D_data[stg_row * N + stg_col + 4] = C_reg_out[1];
            }
            if (stg_col + 8 < boundary && stg_col + 8 < N) {
                D_data[stg_row * N + stg_col + 8] = C_reg_out[2];
            }
            if (stg_col + 12 < boundary && stg_col + 12 < N) {
                D_data[stg_row * N + stg_col + 12] = C_reg_out[3];
            }
        }
    }
}

void _mqa_logits_128x128x32_TN_impl(
    half_t* A_data,
    half_t* B_data,
    float* D_data,
    float* Weights_data,
    uint32_t* KS,
    uint32_t* KE,
    const int q_seq_len,
    const int kv_seq_len,
    const int q_seq_len_aligned,
    const int kv_seq_len_aligned,
    const int num_heads,
    const int head_dim,
    const bool use_fp16) {
    const int size_m = kv_seq_len;
    const int size_n = q_seq_len;
    const int size_k = head_dim * num_heads;
    const int size_a_k = head_dim; // 用于kv
    const int size_b_k = head_dim * num_heads; // 用于q
    // printf("hello ! mqa_logits_asm: size_m=%d, size_n=%d, size_k=%d\n", size_m, size_n, size_k);

    hipFunctionArgs_mqa_logits<half_t> hipFunctionArgs;

    _Float16* a_device = reinterpret_cast<_Float16*>(A_data); // PS, 由于是AWQ，我们这里传入4bit，但还是当成16bit来传入，以适应现有的接口
    _Float16* b_device = reinterpret_cast<_Float16*>(B_data);
    _Float16* c_device = reinterpret_cast<_Float16*>(Weights_data); //  这里视为 16bit来传入，以适应现有的接口。但实际是float。
    _Float16* d_device = reinterpret_cast<_Float16*>(D_data);
    _Float16 alpha = 1.0;
    _Float16 beta = 0.0;
    const size_t lda = size_a_k;
    const size_t ldb = size_b_k;
    const size_t ldc = size_m;
    const size_t stridea = lda * size_m;
    const size_t strideb = ldb * size_n;
    const size_t stridec = ldc * size_n;
    const int workgroupmapping = 8;
    const int depth = 32;
    const int SU = 32;
    int lds = 0;
    const size_t strideC1J = ldc; //
    const size_t strideC2K = stridec; //
    const size_t strideA1L = lda;
    const size_t strideA2K = stridea;
    const size_t strideB1L = ldb;
    const size_t strideB2K = strideb;
    const size_t sizeA = lda * size_m;
    const size_t sizeB = ldb * size_n;
    unsigned int staggerUIter = SU;
    const int unrollLoopIters = size_b_k / 32 / 1; // 待修改
    const unsigned int workDim = 3;
    const unsigned int threadTile[2] = {8, 8};
    const unsigned int groupSize[2] = {16, 16};
    size_t localWorkSize[3] = {256, 1, 1};
    size_t globalWorkSize[1][3];
    globalWorkSize[0][2] = 1;
    const unsigned int sizeOfC0 = size_m; // C矩阵的0维度大小
    const unsigned int sizeOfC1 = size_n; // C矩阵的1维度大小
    unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]); // 宏瓦片0大小
    unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]); // 宏瓦片1大小
    unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0; // 0维度的总工作组数
    unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1; // 1维度的总工作组数
    if (totalWorkGroups0 * macroTile0 < sizeOfC0) {
        totalWorkGroups0++;
    }
    if (totalWorkGroups1 * macroTile1 < sizeOfC1) {
        totalWorkGroups1++;
    }
    unsigned int problemNumGroupTiles0 = totalWorkGroups0;
    unsigned int problemNumGroupTiles1 = totalWorkGroups1;
    const unsigned smallNumMagicShift = 31; // 小数字魔数移位
    unsigned magicNumberProblemNumGroupTiles0 = (1L << smallNumMagicShift) / problemNumGroupTiles0 + 1;

    unsigned numFullBlocks = problemNumGroupTiles1 / workgroupmapping; // 完整块数量
    unsigned wgmRemainder1 = problemNumGroupTiles1 % workgroupmapping; // 余数
    if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping; // 处理余数为0的情况
    unsigned magicNumberWgmRemainder1 = ((1L << smallNumMagicShift) / wgmRemainder1 + 1); // 余数魔数

    globalWorkSize[0][0] = totalWorkGroups0; // gridDim.x = size_m / 256
    globalWorkSize[0][1] = totalWorkGroups1; // gridDim.y = size_n / 256

    const uint64_t tensor2dSizeC = 1 * std::max(size_m, strideC1J) * std::max(size_n, strideC2K);
    uint64_t tensor2dSizeA = 1;
    uint64_t tensor2dSizeAStride = 0;
    uint64_t tensor2dSizeAOffset = 0;
    tensor2dSizeAStride = std::max(tensor2dSizeA * size_a_k, (uint64_t)strideA1L);
    tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA * size_a_k;
    tensor2dSizeA = tensor2dSizeAStride;
    tensor2dSizeAStride = std::max(tensor2dSizeA * size_m, (uint64_t)strideA2K);
    tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA * size_m;
    tensor2dSizeA = tensor2dSizeAStride;
    tensor2dSizeA -= tensor2dSizeAOffset;

    uint64_t tensor2dSizeB = 1;
    uint64_t tensor2dSizeBStride = 0;
    uint64_t tensor2dSizeBOffset = 0;
    tensor2dSizeBStride = std::max(tensor2dSizeB * size_b_k, (uint64_t)strideB1L);
    tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB * size_b_k;
    tensor2dSizeB = tensor2dSizeBStride;
    tensor2dSizeBStride = std::max(tensor2dSizeB * size_n, (uint64_t)strideB2K);
    tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB * size_n;
    tensor2dSizeB = tensor2dSizeBStride;
    tensor2dSizeB -= tensor2dSizeBOffset;

    hipFunctionArgs.tensor2dSizeC = tensor2dSizeC;
    hipFunctionArgs.tensor2dSizeB = tensor2dSizeB;
    hipFunctionArgs.tensor2dSizeA = tensor2dSizeA;
    hipFunctionArgs.dataD = d_device;
    hipFunctionArgs.dataC = c_device;
    hipFunctionArgs.dataA = a_device;
    hipFunctionArgs.dataB = b_device;
    hipFunctionArgs.offseta = 0;
    hipFunctionArgs.offsetb = 0;
    hipFunctionArgs.offsetc = 0;
    hipFunctionArgs.offsetd = 0;
    hipFunctionArgs.alpha[0] = alpha;
    hipFunctionArgs.alpha[1] = alpha;
    hipFunctionArgs.beta[0] = beta;
    hipFunctionArgs.beta[1] = beta;
    // hipFunctionArgs.strideD1J = strideC1J;
    hipFunctionArgs.strideD1J = kv_seq_len_aligned;
    hipFunctionArgs.strideD2K = strideC2K;
    hipFunctionArgs.strideC1J = strideC1J;
    hipFunctionArgs.strideC2K = strideC2K;
    hipFunctionArgs.strideA1L = strideA1L;
    hipFunctionArgs.strideA2K = strideA2K;
    hipFunctionArgs.strideB1L = strideB1L;
    hipFunctionArgs.strideB2K = strideB2K;
    hipFunctionArgs.sizeI = size_m;
    hipFunctionArgs.sizeJ = size_n;
    hipFunctionArgs.sizeK = 1;
    hipFunctionArgs.sizeL = size_k;

    hipFunctionArgs.staggerUIter = staggerUIter;
    hipFunctionArgs.problemNumGroupTiles0 = problemNumGroupTiles0;
    hipFunctionArgs.problemNumGroupTiles1 = problemNumGroupTiles1;
    hipFunctionArgs.numFullBlocks = numFullBlocks;
    hipFunctionArgs.wgmRemainder1 = wgmRemainder1;
    hipFunctionArgs.use_fp16 = use_fp16;
    hipFunctionArgs.cu_seq_len_k_start = KS;
    hipFunctionArgs.cu_seq_len_k_end = KE;
    hipFunctionArgs.num_heads = num_heads;
    const char* LIGHTOP_ASM_DIR = std::getenv("LIGHTOP_ASM_DIR");
    auto co_file_path = std::string(LIGHTOP_ASM_DIR) + "mqa_logits/test_mqa_logits.co";
    static hipModule_t Module = nullptr; // 静态模块句柄，只加载一次
    static hipFunction_t Function = nullptr; // 静态函数句柄，只获取一次
    static bool module_loaded = false;
    if (!module_loaded) {
        printf("load kernel module from %s\n", co_file_path.c_str());
        HIP_CALL(hipModuleLoad(&Module, co_file_path.c_str())); // 加载kernel文件
        HIP_CALL(hipModuleGetFunction(&Function, Module, "mqa_logits_128x128x32")); // 获取kernel函数
        module_loaded = true;
    }

    // 准备kernel启动配置
    size_t size = sizeof(hipFunctionArgs);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &hipFunctionArgs, // 设置参数缓冲区指针
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END}; // 设置参数缓冲区大小
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    HIP_CALL(hipExtModuleLaunchKernel(
        Function,
        globalWorkSize[0][0] * localWorkSize[0],
        globalWorkSize[0][1] * localWorkSize[1],
        globalWorkSize[0][2] * localWorkSize[2],
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        0, stream, NULL, (void**)&config, 0, 0, 0));
    // printf("finish launch hipExtModuleLaunchKernel!\n");

    return;
}

#define NUM_HEAD_SWITCH(num_heads, ...)          \
  [&] {                                          \
    if (num_heads == 1) {                        \
      constexpr static int NumHeads = 1;         \
      return __VA_ARGS__();                      \
    } else if (num_heads == 2) {                 \
      constexpr static int NumHeads = 2;         \
      return __VA_ARGS__();                      \
    } else if (num_heads == 4) {                 \
      constexpr static int NumHeads = 4;         \
      return __VA_ARGS__();                      \
    } else if (num_heads == 8) {                 \
      constexpr static int NumHeads = 8;         \
      return __VA_ARGS__();                      \
    } else if (num_heads == 16) {                \
      constexpr static int NumHeads = 16;        \
      return __VA_ARGS__();                      \
    } else if (num_heads == 32) {                \
      constexpr static int NumHeads = 32;        \
      return __VA_ARGS__();                      \
    } else if (num_heads == 64) {                \
      constexpr static int NumHeads = 64;        \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr static int NumHeads = 128;       \
      return __VA_ARGS__();                      \
    }                                            \
  }()


torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor> &KV_scale =  torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor> &D_out = torch::nullopt) {
    const int M = q_seq_len;
    const int N = kv_seq_len;
    auto qtype = Q.scalar_type();
    auto ktype = K.scalar_type();
    if (!((qtype == torch::kHalf && ktype == torch::kHalf) || 
          (qtype == torch::kBFloat16 && ktype == torch::kBFloat16) || 
          (qtype == torch::kFloat8_e4m3fn && ktype == torch::kFloat8_e4m3fn) ||
          (qtype == torch::kInt8 && ktype == torch::kInt8))) {
            TORCH_CHECK(false, "Q和K的类型必须同时为FP16、同时为BF16、同时为FP8_E4M3, 或同时为INT8!");
    }
    if (auto beta = lightop::mqa_beta::try_dispatch(
            Q, K, Weights, KS, KE, q_seq_len, kv_seq_len, num_heads, head_dim,
            KV_scale, clean_logits, D_out))
        return *beta;
    const bool is_fp8 = (qtype == torch::kFloat8_e4m3fn);
    const bool is_i8 = (qtype == torch::kInt8);
    auto Weightstype = Weights.scalar_type();
    if(Weightstype != torch::kFloat32){
         TORCH_CHECK(false, "Weights的数据类型必须为fp32!");
    }
    const  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
       TORCH_CHECK(N >= 0, "kv_seq_len 必须大于等于 0!");
    TORCH_CHECK(M >= 0, "q_seq_len 必须大于等于 0!");

    if (KV_scale.has_value() && KV_scale.value().defined()) {
        auto kv_scale_tensor = KV_scale.value();
        
        // 1. 检查连续性：避免 slice/transpose 导致的步长非 1 问题
        TORCH_CHECK(kv_scale_tensor.is_contiguous(), "KV_scale 必须是连续的 (contiguous)!");
        
        // 2. 检查长度：防止底层按 N 越界读取
        TORCH_CHECK(kv_scale_tensor.numel() >= N, "KV_scale 的元素数量必须大于等于 kv_seq_len!");
        
        // 3. 检查 16 字节对齐：这是 *(floatx4*) 向量化读取的硬件强制要求
        uintptr_t ptr_address = reinterpret_cast<uintptr_t>(kv_scale_tensor.data_ptr<float>());
        TORCH_CHECK(ptr_address % 16 == 0, 
                    "KV_scale 的首地址必须是 16 字节对齐的，以防止底层 floatx4 读取引发段错误!");
    }
    bool has_d_out = D_out.has_value() && D_out.value().defined();

    if (M >= 128 && !is_fp8 && !is_i8) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    
        const bool use_fp16 = (qtype == torch::kHalf);
        // 将维度对齐到128的整数倍
        const int q_seq_len_aligned = (q_seq_len + 127) / 128 * 128;
        const int kv_seq_len_aligned = (kv_seq_len + 127) / 128 * 128;
        
        torch::Tensor D;
        if (has_d_out) {
            D = D_out.value();
            TORCH_CHECK(D.size(0) >= q_seq_len_aligned && D.size(1) >= kv_seq_len_aligned,
                        "传入的D_out大小不足，ASM kernel要求分配按128对齐的大小(q_seq_len_aligned, kv_seq_len_aligned)");
        } else {
            D = torch::empty({q_seq_len_aligned, kv_seq_len_aligned}, options);
        }
        
        _mqa_logits_128x128x32_TN_impl(
            reinterpret_cast<half_t*>(K.data_ptr()),
            reinterpret_cast<half_t*>(Q.data_ptr()),
            reinterpret_cast<float*>(D.data_ptr()),
            reinterpret_cast<float*>(Weights.data_ptr()),
            reinterpret_cast<uint32_t*>(KS.data_ptr()),
            reinterpret_cast<uint32_t*>(KE.data_ptr()),
            q_seq_len, kv_seq_len, q_seq_len_aligned, kv_seq_len_aligned, num_heads,
            head_dim, use_fp16);
        if (clean_logits) {
            clean_logits_padding<4><<<DIV(M, 4), 256,0,stream>>>(
                q_seq_len, kv_seq_len_aligned, kv_seq_len_aligned,
                reinterpret_cast<uint32_t*>(KS.data_ptr()),
                reinterpret_cast<uint32_t*>(KE.data_ptr()),
                reinterpret_cast<float*>(D.data_ptr()), q_seq_len_aligned);
        }
        return D.slice(0, 0, q_seq_len).slice(1, 0, kv_seq_len);
    } else {
        //fp8
        if (is_fp8) {
            torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
            
            torch::Tensor D;
            bool is_large_fp8 = ((M >= 64 && N >= 1024) || N > 8192);
            if (has_d_out) {
                D = D_out.value();
                if (!is_large_fp8) {
                    D.zero_(); 
                }
            } else {
                if (is_large_fp8) {
                    D = torch::empty({q_seq_len, kv_seq_len}, options);
                } else {
                    D = torch::full({q_seq_len, kv_seq_len}, 0, options);
                }
            }
            
            auto* q_ptr = reinterpret_cast<fp8*>(Q.data_ptr());
            auto* k_ptr = reinterpret_cast<fp8*>(K.data_ptr());
            auto* w_ptr = reinterpret_cast<float*>(Weights.data_ptr());
            auto* ks_ptr = reinterpret_cast<uint32_t*>(KS.data_ptr());
            auto* ke_ptr = reinterpret_cast<uint32_t*>(KE.data_ptr());
            const float* kv_scale_ptr = KV_scale.has_value() ? KV_scale.value().data_ptr<float>() : nullptr;
            auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());

            NUM_HEAD_SWITCH(num_heads, [&] {
                if (is_large_fp8) {
                    int current_block_m = (M >= 256) ? 32 : 16;
                    const int blockx = (N + 127) / 128;
                    const int blocky = (M + current_block_m - 1) / current_block_m;
                    dim3 threadsPerBlock(256);
                    bool is_align_m = (M % current_block_m == 0);
                    bool is_align_n = (N % 128 == 0);
                    dim3 numBlock(blockx, blocky);

                    using kernel_fp8_t = void (*)(
                        fp8* A_data, fp8* B_data, float* D_data, float* Weights_data,
                        uint32_t* KS, uint32_t* KE, const int q_seq_len, const int kv_seq_len, const float* KV_scale_data);
                    
                    auto launch_fp8 = [&](kernel_fp8_t kernel_to_launch) {
                        kernel_to_launch<<<numBlock, threadsPerBlock,0,stream>>>(
                            q_ptr, k_ptr, d_ptr, w_ptr, ks_ptr, ke_ptr, q_seq_len, kv_seq_len, kv_scale_ptr);
                    };

                    if (M >= 256) {
                        if (is_align_m) {
                            if (is_align_n) {
                                launch_fp8(_mqa_logits_32x128x128_TN_impl_fp8<NumHeads, 128, 32, 128, 128, true, true>);
                            } else {
                                launch_fp8(_mqa_logits_32x128x128_TN_impl_fp8<NumHeads, 128, 32, 128, 128, true, false>);
                            }
                        } else {
                            if (is_align_n) {
                                launch_fp8(_mqa_logits_32x128x128_TN_impl_fp8<NumHeads, 128, 32, 128, 128, false, true>);
                            } else {
                                launch_fp8(_mqa_logits_32x128x128_TN_impl_fp8<NumHeads, 128, 32, 128, 128, false, false>);
                            }
                        }
                    } else {
                        if (is_align_m) {
                            if (is_align_n) {
                                launch_fp8(_mqa_logits_16x128x128_TN_impl_fp8<NumHeads, 128, 16, 128, 128, true, true>);
                            } else {
                                launch_fp8(_mqa_logits_16x128x128_TN_impl_fp8<NumHeads, 128, 16, 128, 128, true, false>);
                            }
                        } else {
                            if (is_align_n) {
                                launch_fp8(_mqa_logits_16x128x128_TN_impl_fp8<NumHeads, 128, 16, 128, 128, false, true>);
                            } else {
                                launch_fp8(_mqa_logits_16x128x128_TN_impl_fp8<NumHeads, 128, 16, 128, 128, false, false>);
                            }
                        }
                    }
                } else {
                    const int blockx = DIV(N, 64);
                    const int blocky = DIV(M, 16);
                    dim3 threadsPerBlock(256);
                    dim3 numBlock(blockx, blocky);
                    bool is_align_m = (M % 16 == 0);
                    bool is_align_n = (N % 64 == 0);

                    int split_k = 4; 
                    if (M <= 32 && N <= 128) {
                         split_k = 16; 
                    }
                    if (NumHeads < 32){
                        split_k = 2;
                    }
                    dim3 grid(blockx, blocky, split_k);

                    using kernel_fp8_t = void (*)(
                        fp8* A_data, fp8* B_data, float* D_data, float* Weights_data,
                        uint32_t* KS, uint32_t* KE, const int q_seq_len, const int kv_seq_len, const float* KV_scale_data);
                    
                    auto launch_fp8 = [&](kernel_fp8_t kernel_to_launch) {
                        kernel_to_launch<<<grid, threadsPerBlock,0,stream>>>(
                            q_ptr, k_ptr, d_ptr, w_ptr, ks_ptr, ke_ptr, q_seq_len, kv_seq_len, kv_scale_ptr);
                    };

                    if (is_align_m) {
                        if (is_align_n) {
                            launch_fp8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_fp8<NumHeads, 128, 16, 128, 64, true, true>);
                        } else {
                            launch_fp8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_fp8<NumHeads, 128, 16, 128, 64, true, false>);
                        }
                    } else {
                        if (is_align_n) {
                            launch_fp8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_fp8<NumHeads, 128, 16, 128, 64, false, true>);
                        } else {
                            launch_fp8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_fp8<NumHeads, 128, 16, 128, 64, false, false>);
                        }
                    }
                }
            });
            return D;
        }
        //int8 (gfx936 native i8 mmac; Q_scale fused into Weights by caller)
        if (is_i8) {
            TORCH_CHECK(KV_scale.has_value() && KV_scale.value().defined(),
                        "INT8 mqa_logits 必须提供 KV_scale!");
            torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

            torch::Tensor D;
            // The large INT8 kernels require a complete 128-column tile.
            // Fall back to the stable 16x64 kernel for an unaligned N.
            bool is_large_i8 =
                (((M >= 64 && N >= 1024) || N > 8192) && (N % 128 == 0));
            if (has_d_out) {
                D = D_out.value();
                if (!is_large_i8) {
                    D.zero_();
                }
            } else {
                if (is_large_i8) {
                    D = torch::empty({q_seq_len, kv_seq_len}, options);
                } else {
                    D = torch::full({q_seq_len, kv_seq_len}, 0, options);
                }
            }

            auto* q_ptr = reinterpret_cast<int8_t*>(Q.data_ptr());
            auto* k_ptr = reinterpret_cast<int8_t*>(K.data_ptr());
            auto* w_ptr = reinterpret_cast<float*>(Weights.data_ptr());
            auto* ks_ptr = reinterpret_cast<uint32_t*>(KS.data_ptr());
            auto* ke_ptr = reinterpret_cast<uint32_t*>(KE.data_ptr());
            const float* kv_scale_ptr = KV_scale.value().data_ptr<float>();
            auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());

            NUM_HEAD_SWITCH(num_heads, [&] {
                if (is_large_i8) {
                    int current_block_m = (M >= 256) ? 32 : 16;
                    const int blockx = (N + 127) / 128;
                    const int blocky = (M + current_block_m - 1) / current_block_m;
                    dim3 threadsPerBlock(256);
                    bool is_align_m = (M % current_block_m == 0);
                    bool is_align_n = (N % 128 == 0);
                    dim3 numBlock(blockx, blocky);

                    using kernel_i8_t = void (*)(
                        int8_t* A_data, int8_t* B_data, float* D_data, float* Weights_data,
                        uint32_t* KS, uint32_t* KE, const int q_seq_len, const int kv_seq_len, const float* KV_scale_data);

                    auto launch_i8 = [&](kernel_i8_t kernel_to_launch) {
                        kernel_to_launch<<<numBlock, threadsPerBlock,0,stream>>>(
                            q_ptr, k_ptr, d_ptr, w_ptr, ks_ptr, ke_ptr, q_seq_len, kv_seq_len, kv_scale_ptr);
                    };

                    if (M >= 256) {
                        if (is_align_m) {
                            if (is_align_n) {
                                launch_i8(_mqa_logits_32x128x128_TN_impl_i8<NumHeads, 128, 32, 128, 128, true, true>);
                            } else {
                                launch_i8(_mqa_logits_32x128x128_TN_impl_i8<NumHeads, 128, 32, 128, 128, true, false>);
                            }
                        } else {
                            if (is_align_n) {
                                launch_i8(_mqa_logits_32x128x128_TN_impl_i8<NumHeads, 128, 32, 128, 128, false, true>);
                            } else {
                                launch_i8(_mqa_logits_32x128x128_TN_impl_i8<NumHeads, 128, 32, 128, 128, false, false>);
                            }
                        }
                    } else {
                        if (is_align_m) {
                            if (is_align_n) {
                                launch_i8(_mqa_logits_16x128x128_TN_impl_i8<NumHeads, 128, 16, 128, 128, true, true>);
                            } else {
                                launch_i8(_mqa_logits_16x128x128_TN_impl_i8<NumHeads, 128, 16, 128, 128, true, false>);
                            }
                        } else {
                            if (is_align_n) {
                                launch_i8(_mqa_logits_16x128x128_TN_impl_i8<NumHeads, 128, 16, 128, 128, false, true>);
                            } else {
                                launch_i8(_mqa_logits_16x128x128_TN_impl_i8<NumHeads, 128, 16, 128, 128, false, false>);
                            }
                        }
                    }
                } else {
                    const int blockx = DIV(N, 64);
                    const int blocky = DIV(M, 16);
                    dim3 threadsPerBlock(256);
                    bool is_align_m = (M % 16 == 0);
                    bool is_align_n = (N % 64 == 0);

                    int split_k = 4;
                    if (M <= 32 && N <= 128) {
                         split_k = 16;
                    }
                    if (NumHeads < 32){
                        split_k = 2;
                    }
                    dim3 grid(blockx, blocky, split_k);

                    using kernel_i8_t = void (*)(
                        int8_t* A_data, int8_t* B_data, float* D_data, float* Weights_data,
                        uint32_t* KS, uint32_t* KE, const int q_seq_len, const int kv_seq_len, const float* KV_scale_data);

                    auto launch_i8 = [&](kernel_i8_t kernel_to_launch) {
                        kernel_to_launch<<<grid, threadsPerBlock,0,stream>>>(
                            q_ptr, k_ptr, d_ptr, w_ptr, ks_ptr, ke_ptr, q_seq_len, kv_seq_len, kv_scale_ptr);
                    };

                    if (is_align_m) {
                        if (is_align_n) {
                            launch_i8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_i8<NumHeads, 128, 16, 128, 64, true, true>);
                        } else {
                            launch_i8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_i8<NumHeads, 128, 16, 128, 64, true, false>);
                        }
                    } else {
                        if (is_align_n) {
                            launch_i8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_i8<NumHeads, 128, 16, 128, 64, false, true>);
                        } else {
                            launch_i8(_mqa_logits_16x128x64_TN_impl_with_weight_prefetch_i8<NumHeads, 128, 16, 128, 64, false, false>);
                        }
                    }
                }
            });
            if (clean_logits && M > 0) {
                at::native::clean_logits_mqa<4>
                    <<<DIV(M, 4), 256, 0, stream>>>(
                        M,
                        N,
                        N,
                        ks_ptr,
                        ke_ptr,
                        d_ptr);
            }

            return D;
        }
        //fp16/bf16
        torch::Tensor D;
        if (has_d_out) {
            D = D_out.value(); 
        }

        Input_Type_SWITCH(Q.scalar_type(), [&] {
            auto* q_ptr = reinterpret_cast<scalar_t*>(Q.data_ptr());
            auto* k_ptr = reinterpret_cast<scalar_t*>(K.data_ptr());
            //auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());
            auto* w_ptr = reinterpret_cast<float*>(Weights.data_ptr());
            auto* ks_ptr = reinterpret_cast<uint32_t*>(KS.data_ptr());
            auto* ke_ptr = reinterpret_cast<uint32_t*>(KE.data_ptr());
            using kernel_t = void (*)(
                scalar_t * A_data,
                scalar_t * B_data,
                float* D_data,
                float* Weights_data,
                uint32_t* KS,
                uint32_t* KE,
                const int q_seq_len,
                const int kv_seq_len);
            auto launch_kernel = [&](kernel_t kernel_to_launch, dim3 grid, dim3 block,float *d_ptr_local) {
                kernel_to_launch<<<grid, block,0,stream>>>(
                    q_ptr, k_ptr, d_ptr_local, w_ptr, ks_ptr, ke_ptr, q_seq_len, kv_seq_len);
            };
            NUM_HEAD_SWITCH(num_heads,[&] {
                if (N > 8192) {
                    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
                    if (!D.defined()) { 
                        D = torch::full({q_seq_len, kv_seq_len}, -std::numeric_limits<float>::infinity(), options);
                    } else {
                        D.fill_(-std::numeric_limits<float>::infinity());
                    }
                    auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());
                    const int blockx = (N + 127) / 128;
                    const int blocky = (M + 15) / 16;
                    dim3 threadsPerBlock(512);
                    bool is_align_m = (M % 16 == 0);
                    bool is_align_n = (N % 128 == 0);
                    dim3 numBlock(blockx, blocky);
                    if (is_align_m) {
                        if (is_align_n) {
                            kernel_t kernel_ptr = _mqa_logits_16x128x128_TN_impl<scalar_t, NumHeads, 128, 16, 128, 128, true, true>;
                            launch_kernel(kernel_ptr, numBlock, threadsPerBlock,d_ptr);
                        } else {
                            kernel_t kernel_ptr = _mqa_logits_16x128x128_TN_impl<scalar_t, NumHeads, 128, 16, 128, 128, true, false>;
                            launch_kernel(kernel_ptr, numBlock, threadsPerBlock,d_ptr);
                        }
                    } else {
                        if (is_align_n) {
                            kernel_t kernel_ptr = _mqa_logits_16x128x128_TN_impl<scalar_t, NumHeads, 128, 16, 128, 128, false, true>;
                            launch_kernel(kernel_ptr, numBlock, threadsPerBlock,d_ptr);
                        } else {
                            kernel_t kernel_ptr = _mqa_logits_16x128x128_TN_impl<scalar_t, NumHeads, 128, 16, 128, 128, false, false>;
                            launch_kernel(kernel_ptr, numBlock, threadsPerBlock,d_ptr);
                        }
                    }
                } else {
                    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
                    if (!D.defined()) { 
                        D = torch::full({q_seq_len, kv_seq_len}, 0, options);
                    } else {
                        D.zero_(); 
                    }
                    auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());
                    const int blockx = DIV(N, 64);
                    const int blocky = DIV(M, 16);
                    dim3 threadsPerBlock(256);
                    dim3 numBlock(blockx, blocky);
                    bool is_align_m = (M % 16 == 0);
                    bool is_align_n = (N % 64 == 0);
    
                    int split_k = 4; 
                    if (M <= 32 && N <= 128) {
                         split_k = 16; 
                    }
                    if (NumHeads < 32){
                        split_k = 2;
                    }
                    dim3 grid(blockx, blocky, split_k);
    
                     if (is_align_m) {
                        if (is_align_n) {
                            kernel_t kernel_ptr = _mqa_logits_16x128x64_TN_impl_with_weight_prefetch<scalar_t, NumHeads, 128, 16, 128, 64, true, true>;
                            launch_kernel(kernel_ptr, grid, threadsPerBlock,d_ptr);
                        } else {
                            kernel_t kernel_ptr = _mqa_logits_16x128x64_TN_impl_with_weight_prefetch<scalar_t, NumHeads, 128, 16, 128, 64, true, false>;
                            launch_kernel(kernel_ptr, grid, threadsPerBlock,d_ptr);
                        }
                    } else {
                        if (is_align_n) {
                            kernel_t kernel_ptr = _mqa_logits_16x128x64_TN_impl_with_weight_prefetch<scalar_t, NumHeads, 128, 16, 128, 64, false, true>;
                            launch_kernel(kernel_ptr, grid, threadsPerBlock,d_ptr);
                        } else {
                            kernel_t kernel_ptr = _mqa_logits_16x128x64_TN_impl_with_weight_prefetch<scalar_t, NumHeads, 128, 16, 128, 64, false, false>;
                            launch_kernel(kernel_ptr, grid, threadsPerBlock,d_ptr);
                        }
                    }
                }
            });
            if (clean_logits && M > 0) {
                auto* d_ptr = reinterpret_cast<float*>(D.data_ptr());
                at::native::clean_logits_mqa<4>
                    <<<DIV(M, 4), 256, 0, stream>>>(
                        M,
                        N,
                        N,
                        ks_ptr,
                        ke_ptr,
                        d_ptr);
            }
        });
        return D;
    }
}
}
}
