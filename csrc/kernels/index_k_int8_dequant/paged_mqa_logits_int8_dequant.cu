// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "index_k_int8_dequant.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

namespace lightop {
namespace deep_gemm {

namespace {

constexpr int kPageSize = 64;
constexpr int kHeadDim = 128;
constexpr int kScaleBytes = sizeof(float);
constexpr int kPageBytes = kPageSize * (kHeadDim + kScaleBytes);
constexpr int kWavesPerCta = 4;
constexpr int kThreads = kWavesPerCta * 64;

struct alignas(16) BFloat16x8 {
    __hip_bfloat16 value[8];
};

// One, two, or four cooperating waves process each logical page. A page claim prevents
// duplicate work when radix-cache prefixes or grouped MTP rows reference the
// same physical page.  The winner converts one pair of adjacent K elements per
// lane, preserving the exact FP32 multiply -> BF16 rounding used by the
// unfused reference path.
template <int kWavesPerPage, typename LengthT>
__attribute__((amdgpu_flat_work_group_size(kThreads, kThreads)))
__global__ void dequantize_index_k_int8_paged_kernel(
    const uint8_t* packed_cache,
    const int32_t* block_tables,
    const LengthT* context_lens,
    at::BFloat16* workspace,
    int32_t* page_claims,
    int64_t block_table_stride,
    int64_t context_lens_stride,
    int logical_pages_per_row,
    int parts_per_row,
    int num_physical_pages) {
    static_assert(
        kWavesPerPage == 1 || kWavesPerPage == 2 || kWavesPerPage == 4);
    constexpr int kPagesPerCta = kWavesPerCta / kWavesPerPage;
    constexpr int kTokensInFlight = 4 * kWavesPerPage;
    __shared__ int claimed_pages[kPagesPerCta];
    __shared__ int claim_winners[kPagesPerCta];

    const int wave = static_cast<int>(threadIdx.x) / 64;
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int page_slot = wave / kWavesPerPage;
    const int wave_in_page = wave % kWavesPerPage;
    const int row = static_cast<int>(blockIdx.x) / parts_per_row;
    const int part = static_cast<int>(blockIdx.x) - row * parts_per_row;
    const int logical_page = part * kPagesPerCta + page_slot;

    if (wave_in_page == 0 && lane == 0) {
        int physical_page = -1;
        int won_claim = 0;
        const int context_len =
            static_cast<int>(context_lens[row * context_lens_stride]);
        const int valid_pages =
            (context_len > 0) ? (context_len + kPageSize - 1) / kPageSize : 0;
        if (logical_page < logical_pages_per_row && logical_page < valid_pages) {
            physical_page =
                block_tables[row * block_table_stride + logical_page];
            if (physical_page >= 0 && physical_page < num_physical_pages) {
                won_claim =
                    atomicCAS(page_claims + physical_page, 0, 1) == 0;
            }
        }
        claimed_pages[page_slot] = physical_page;
        claim_winners[page_slot] = won_claim;
    }
    __syncthreads();
    const int physical_page = claimed_pages[page_slot];
    if (!claim_winners[page_slot]) {
        return;
    }

    const uint8_t* page = packed_cache +
                          static_cast<int64_t>(physical_page) * kPageBytes;
    const int8_t* quantized = reinterpret_cast<const int8_t*>(page);
    const float* scales =
        reinterpret_cast<const float*>(page + kPageSize * kHeadDim);
    at::BFloat16* output =
        workspace + static_cast<int64_t>(physical_page) * kPageSize * kHeadDim;
    const int token_lane = wave_in_page * 4 + lane / 16;
    const int vector_lane = lane % 16;
    #pragma unroll
    for (int token_base = 0;
         token_base < kPageSize;
         token_base += kTokensInFlight) {
        const int token = token_base + token_lane;
        float scale = vector_lane == 0 ? scales[token] : 0.0f;
        scale = __shfl(scale, token_lane * 16, 16);
        const int2 raw = *reinterpret_cast<const int2*>(
            quantized + token * kHeadDim + vector_lane * 8);
        const int8_t* values = reinterpret_cast<const int8_t*>(&raw);
        BFloat16x8 converted{};
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            converted.value[item] = __float2bfloat16(
                static_cast<float>(values[item]) * scale);
        }
        *reinterpret_cast<BFloat16x8*>(
            output + token * kHeadDim + vector_lane * 8) = converted;
    }
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& value,
                       const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(
        value.device() == reference.device(),
        name,
        " must be on ",
        reference.device());
}

}  // namespace

torch::Tensor dequantize_index_k_int8_paged(
    const torch::Tensor& packed_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& workspace,
    const torch::Tensor& page_claims,
    const int waves_per_page) {
    TORCH_CHECK(packed_cache.is_cuda(), "packed_cache must be a HIP tensor");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(packed_cache));
    check_same_device(packed_cache, block_tables, "block_tables");
    check_same_device(packed_cache, context_lens, "context_lens");
    check_same_device(packed_cache, workspace, "workspace");
    check_same_device(packed_cache, page_claims, "page_claims");

    TORCH_CHECK(
        packed_cache.scalar_type() == torch::kUInt8 &&
            packed_cache.dim() == 2 && packed_cache.is_contiguous() &&
            packed_cache.size(1) == kPageBytes,
        "packed_cache must be contiguous uint8 [pages, ",
        kPageBytes,
        "]");
    TORCH_CHECK(
        block_tables.scalar_type() == torch::kInt32 &&
            block_tables.dim() == 2 && block_tables.stride(1) == 1,
        "block_tables must be int32 [rows, logical_pages] with stride(1)=1");
    TORCH_CHECK(
        (context_lens.scalar_type() == torch::kInt32 ||
         context_lens.scalar_type() == torch::kInt64) &&
            context_lens.dim() == 1 && context_lens.stride(0) > 0,
        "context_lens must be a strided int32/int64 [rows] tensor");
    TORCH_CHECK(
        block_tables.size(0) == context_lens.size(0),
        "block_tables and context_lens must have matching rows");
    TORCH_CHECK(
        workspace.scalar_type() == torch::kBFloat16 &&
            workspace.dim() == 4 && workspace.is_contiguous() &&
            workspace.size(0) == packed_cache.size(0) &&
            workspace.size(1) == kPageSize && workspace.size(2) == 1 &&
            workspace.size(3) == kHeadDim,
        "workspace must be contiguous BF16 [pages, 64, 1, 128]");
    TORCH_CHECK(
        page_claims.scalar_type() == torch::kInt32 &&
            page_claims.dim() == 1 && page_claims.is_contiguous() &&
            page_claims.numel() == packed_cache.size(0),
        "page_claims must be contiguous int32 [pages]");
    TORCH_CHECK(
        waves_per_page == 1 || waves_per_page == 2 || waves_per_page == 4,
        "waves_per_page must be 1, 2, or 4");

    hipDeviceProp_t properties;
    const auto prop_error =
        hipGetDeviceProperties(&properties, packed_cache.get_device());
    TORCH_CHECK(
        prop_error == hipSuccess,
        "hipGetDeviceProperties failed: ",
        hipGetErrorString(prop_error));
    TORCH_CHECK(
        std::strncmp(properties.gcnArchName, "gfx936", 6) == 0 &&
            properties.multiProcessorCount == 80,
        "INT8 index-K dequant requires gfx936/80CU, got ",
        properties.gcnArchName,
        "/",
        properties.multiProcessorCount,
        "CU");

    if (context_lens.numel() == 0 || block_tables.size(1) == 0) {
        return workspace;
    }
    const int logical_pages_per_row = static_cast<int>(block_tables.size(1));
    const int pages_per_cta = kWavesPerCta / waves_per_page;
    const int parts_per_row =
        (logical_pages_per_row + pages_per_cta - 1) / pages_per_cta;
    const int64_t grid_x = context_lens.numel() * parts_per_row;
    TORCH_CHECK(grid_x <= INT32_MAX, "dequant grid is too large");

    auto stream = at::cuda::getCurrentCUDAStream(packed_cache.get_device());
    #define LAUNCH_DEQUANT(WAVES_PER_PAGE, LENGTH_T)                        \
        dequantize_index_k_int8_paged_kernel<WAVES_PER_PAGE, LENGTH_T><<<   \
            dim3(static_cast<uint32_t>(grid_x), 1, 1),                      \
            dim3(kThreads, 1, 1),                                           \
            0,                                                              \
            stream>>>(                                                      \
            packed_cache.data_ptr<uint8_t>(),                              \
            block_tables.data_ptr<int32_t>(),                              \
            context_lens.data_ptr<LENGTH_T>(),                             \
            workspace.data_ptr<at::BFloat16>(),                            \
            page_claims.data_ptr<int32_t>(),                               \
            block_tables.stride(0),                                        \
            context_lens.stride(0),                                        \
            logical_pages_per_row,                                         \
            parts_per_row,                                                 \
            static_cast<int>(packed_cache.size(0)))
    const bool lengths_are_int64 =
        context_lens.scalar_type() == torch::kInt64;
    if (waves_per_page == 1 && !lengths_are_int64) {
        LAUNCH_DEQUANT(1, int32_t);
    } else if (waves_per_page == 2 && !lengths_are_int64) {
        LAUNCH_DEQUANT(2, int32_t);
    } else if (waves_per_page == 4 && !lengths_are_int64) {
        LAUNCH_DEQUANT(4, int32_t);
    } else if (waves_per_page == 1) {
        LAUNCH_DEQUANT(1, int64_t);
    } else if (waves_per_page == 2) {
        LAUNCH_DEQUANT(2, int64_t);
    } else {
        LAUNCH_DEQUANT(4, int64_t);
    }
    #undef LAUNCH_DEQUANT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return workspace;
}

}  // namespace deep_gemm
}  // namespace lightop
