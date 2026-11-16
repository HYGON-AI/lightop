// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <optional>

#ifndef USE_ROCM
#define USE_ROCM
#endif

#ifdef USE_ROCM
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
using dsv4_bfloat16 = __hip_bfloat16;
#else
#include <cuda_bf16.h>
using dsv4_bfloat16 = __nv_bfloat16;
#endif

#ifndef HIP_FP8_TYPE_OCP
#define HIP_FP8_TYPE_OCP 0
#endif

#if HIP_FP8_TYPE_OCP
using dsv4_fp8_e4m3 = __hip_fp8_e4m3;
#else
using dsv4_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#endif

namespace at {
namespace native {
namespace {

// This file deliberately exposes DPSKV4-only entry points.  It does not
// participate in, or alter, the Spark 656-byte gather/upconvert dispatch.
constexpr int64_t kDsv4NopeDim = 448;
constexpr int64_t kDsv4RopeDim = 64;
constexpr int64_t kDsv4HeadDim = kDsv4NopeDim + kDsv4RopeDim;
constexpr int64_t kDsv4TileSize = 64;
constexpr int64_t kDsv4ScaleTiles = kDsv4NopeDim / kDsv4TileSize;
constexpr int64_t kDsv4TokenDataBytes =
    kDsv4NopeDim + kDsv4RopeDim * sizeof(dsv4_bfloat16);
constexpr int64_t kDsv4ScaleBytesPerToken = 8;
constexpr int kThreadsPerRow = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kThreadsPerBlock = kThreadsPerRow * kRowsPerBlock;
constexpr int kCase13RowsPerBlock = 16;
constexpr int kCase13ThreadsPerBlock =
    kThreadsPerRow * kCase13RowsPerBlock;
constexpr uint32_t kCase13BlocksPerQuery = 524;
constexpr int kDsv4ElementsPerLane = kDsv4HeadDim / kThreadsPerRow;
constexpr int kDsv4NopeLanes = kDsv4NopeDim / kDsv4ElementsPerLane;

template <typename T, int N>
struct alignas(sizeof(T) * N) Dsv4Vector {
  T val[N];
};

static_assert(kDsv4ElementsPerLane == 16);
static_assert(kDsv4NopeLanes == 28);
static_assert(
    sizeof(Dsv4Vector<uint32_t, kDsv4ElementsPerLane / 2>) == 32);

__device__ __forceinline__ float dsv4_fp8_to_float(uint8_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x80u) << 24;
  const uint32_t magnitude = static_cast<uint32_t>(value & 0x7fu);
  const uint32_t exponent = magnitude >> 3;
  const uint32_t mantissa = magnitude & 0x7u;
  if (exponent == 0) {
    const float subnormal = static_cast<float>(mantissa) * 0x1.0p-9f;
    return __uint_as_float(__float_as_uint(subnormal) | sign);
  }
  if (magnitude == 0x7fu) {
    return __uint_as_float(0x7fc00000u | sign);
  }
  return __uint_as_float(
      sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

__device__ __forceinline__ float dsv4_ue8m0_to_float(uint8_t value) {
  // DPSKV4 uses UE8M0 with bias 127.  Byte zero maps below the minimum
  // normal FP32 value and the Triton/reference implementation flushes it.
  return value == 0
      ? 0.0f
      : __uint_as_float(static_cast<uint32_t>(value) << 23);
}

__device__ __forceinline__ uint32_t dsv4_scaled_pair_to_bf16_rne(
    uint8_t first,
    uint8_t second,
    float scale) {
  union {
    float value;
    uint32_t bits;
  } first_value{dsv4_fp8_to_float(first) * scale};
  union {
    float value;
    uint32_t bits;
  } second_value{dsv4_fp8_to_float(second) * scale};
  first_value.bits += 0x7fffu + ((first_value.bits >> 16) & 1u);
  second_value.bits += 0x7fffu + ((second_value.bits >> 16) & 1u);
  return __builtin_amdgcn_perm(
      first_value.bits, second_value.bits, 0x03020706u);
}

// The flat P-path entry has stricter byte-for-byte semantics than the older
// gather entry points above. Keep its conversion private so those existing
// kernels retain their original code, behavior, and resource usage.
__device__ __forceinline__ uint32_t dsv4_flat_scaled_pair_to_bf16_rne(
    uint8_t first,
    uint8_t second,
    float scale) {
  union {
    float value;
    uint32_t bits;
  } first_value{dsv4_fp8_to_float(first) * scale};
  union {
    float value;
    uint32_t bits;
  } second_value{dsv4_fp8_to_float(second) * scale};
  // Triton canonicalizes every NaN, including signed FN NaNs and 0 * +inf,
  // to positive BF16 0x7fff.
  if ((first_value.bits & 0x7f800000u) == 0x7f800000u &&
      (first_value.bits & 0x007fffffu) != 0) {
    first_value.bits = 0x7fff0000u;
  }
  if ((second_value.bits & 0x7f800000u) == 0x7f800000u &&
      (second_value.bits & 0x007fffffu) != 0) {
    second_value.bits = 0x7fff0000u;
  }
  first_value.bits += 0x7fffu + ((first_value.bits >> 16) & 1u);
  second_value.bits += 0x7fffu + ((second_value.bits >> 16) & 1u);
  return __builtin_amdgcn_perm(
      first_value.bits, second_value.bits, 0x03020706u);
}

template <typename index_t, int64_t PageSize>
__device__ __forceinline__ void dsv4_gather_one_row(
    const uint8_t* __restrict__ src_cache,
    const index_t* __restrict__ token_indices,
    const int32_t* __restrict__ valid_lengths,
    dsv4_bfloat16* __restrict__ dst,
    int32_t* __restrict__ compact_indices,
    int64_t query,
    int64_t column,
    int64_t topk,
    int64_t num_pages,
    int64_t page_size,
    int64_t page_stride_bytes,
    int lane) {
  const int64_t valid_length = static_cast<int64_t>(valid_lengths[query]);
  const bool in_valid_prefix = column < valid_length;
  const int64_t index_offset = query * topk + column;
  const int64_t location = in_valid_prefix
      ? static_cast<int64_t>(token_indices[index_offset])
      : 0;
  const int64_t effective_page_size =
      PageSize > 0 ? PageSize : page_size;
  const bool valid_location =
      in_valid_prefix &&
      location >= 0 &&
      location < num_pages * effective_page_size;

  const int64_t safe_location = valid_location ? location : 0;
  int64_t page;
  int64_t in_page;
  if constexpr (PageSize > 0) {
    static_assert(
        (PageSize & (PageSize - 1)) == 0,
        "specialized page size must be a power of two");
    page = safe_location / PageSize;
    in_page = safe_location % PageSize;
  } else {
    page = safe_location / page_size;
    in_page = safe_location - page * page_size;
  }
  const int64_t page_base = page * page_stride_bytes;
  const int64_t token_data_base =
      page_base + in_page * kDsv4TokenDataBytes;
  const int64_t token_scale_base =
      page_base + effective_page_size * kDsv4TokenDataBytes +
      in_page * kDsv4ScaleBytesPerToken;
  dsv4_bfloat16* const output_row =
      dst + index_offset * kDsv4HeadDim;

  using Fp8Vec = Dsv4Vector<uint8_t, kDsv4ElementsPerLane>;
  using PackedBf16Vec = Dsv4Vector<uint32_t, kDsv4ElementsPerLane / 2>;
  PackedBf16Vec output_vec;
  if (valid_location && lane < kDsv4NopeLanes) {
    const Fp8Vec input_vec = *reinterpret_cast<const Fp8Vec*>(
        src_cache + token_data_base + lane * kDsv4ElementsPerLane);
    const float scale = dsv4_ue8m0_to_float(
        src_cache[token_scale_base + lane / (kDsv4TileSize /
                                             kDsv4ElementsPerLane)]);
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = dsv4_scaled_pair_to_bf16_rne(
          input_vec.val[2 * i], input_vec.val[2 * i + 1], scale);
    }
  } else if (valid_location) {
    output_vec = *reinterpret_cast<const PackedBf16Vec*>(
        src_cache + token_data_base + kDsv4NopeDim +
        (lane - kDsv4NopeLanes) * sizeof(PackedBf16Vec));
  } else {
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = 0;
    }
  }
  *reinterpret_cast<PackedBf16Vec*>(
      output_row + lane * kDsv4ElementsPerLane) = output_vec;

  if (lane == 0) {
    // Match the existing Triton contract: an in-prefix but out-of-range source
    // location produces zero KV while retaining its compact destination index.
    compact_indices[index_offset] = in_valid_prefix
        ? static_cast<int32_t>(index_offset)
        : 0;
  }
}

template <int64_t PageSize>
__device__ __forceinline__ void dsv4_gather_one_row_u32(
    const uint8_t* __restrict__ src_cache,
    const int32_t* __restrict__ token_indices,
    const int32_t* __restrict__ valid_lengths,
    dsv4_bfloat16* __restrict__ dst,
    int32_t* __restrict__ compact_indices,
    uint32_t query,
    uint32_t column,
    uint32_t topk,
    uint32_t num_pages,
    uint32_t page_size,
    uint32_t page_stride_bytes,
    int lane) {
  const int32_t valid_length = valid_lengths[query];
  const bool in_valid_prefix =
      valid_length > 0 && column < static_cast<uint32_t>(valid_length);
  const uint32_t index_offset = query * topk + column;
  const int32_t location = in_valid_prefix ? token_indices[index_offset] : 0;
  const uint32_t effective_page_size =
      PageSize > 0 ? static_cast<uint32_t>(PageSize) : page_size;
  const bool valid_location =
      in_valid_prefix && location >= 0 &&
      static_cast<uint32_t>(location) < num_pages * effective_page_size;

  const uint32_t safe_location =
      valid_location ? static_cast<uint32_t>(location) : 0;
  uint32_t page;
  uint32_t in_page;
  if constexpr (PageSize > 0) {
    static_assert(
        (PageSize & (PageSize - 1)) == 0,
        "specialized page size must be a power of two");
    page = safe_location / static_cast<uint32_t>(PageSize);
    in_page = safe_location % static_cast<uint32_t>(PageSize);
  } else {
    page = safe_location / page_size;
    in_page = safe_location - page * page_size;
  }
  const uint32_t page_base = page * page_stride_bytes;
  const uint32_t token_data_base =
      page_base + in_page * static_cast<uint32_t>(kDsv4TokenDataBytes);
  const uint32_t token_scale_base =
      page_base +
      effective_page_size * static_cast<uint32_t>(kDsv4TokenDataBytes) +
      in_page * static_cast<uint32_t>(kDsv4ScaleBytesPerToken);
  const uint32_t output_offset =
      index_offset * static_cast<uint32_t>(kDsv4HeadDim);
  dsv4_bfloat16* const output_row = dst + output_offset;

  using Fp8Vec = Dsv4Vector<uint8_t, kDsv4ElementsPerLane>;
  using PackedBf16Vec = Dsv4Vector<uint32_t, kDsv4ElementsPerLane / 2>;
  PackedBf16Vec output_vec;
  if (valid_location && lane < kDsv4NopeLanes) {
    const Fp8Vec input_vec = *reinterpret_cast<const Fp8Vec*>(
        src_cache + token_data_base + lane * kDsv4ElementsPerLane);
    const float scale = dsv4_ue8m0_to_float(
        src_cache[token_scale_base + lane / (kDsv4TileSize /
                                             kDsv4ElementsPerLane)]);
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = dsv4_scaled_pair_to_bf16_rne(
          input_vec.val[2 * i], input_vec.val[2 * i + 1], scale);
    }
  } else if (valid_location) {
    output_vec = *reinterpret_cast<const PackedBf16Vec*>(
        src_cache + token_data_base + kDsv4NopeDim +
        (lane - kDsv4NopeLanes) * sizeof(PackedBf16Vec));
  } else {
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = 0;
    }
  }
  *reinterpret_cast<PackedBf16Vec*>(
      output_row + lane * kDsv4ElementsPerLane) = output_vec;

  if (lane == 0) {
    compact_indices[index_offset] = in_valid_prefix
        ? static_cast<int32_t>(index_offset)
        : 0;
  }
}

template <typename index_t, int64_t PageSize, bool UseU32Offsets = false>
__device__ __forceinline__ void dsv4_gather_one_row_dispatch(
    const uint8_t* __restrict__ src_cache,
    const index_t* __restrict__ token_indices,
    const int32_t* __restrict__ valid_lengths,
    dsv4_bfloat16* __restrict__ dst,
    int32_t* __restrict__ compact_indices,
    int64_t query,
    int64_t column,
    int64_t topk,
    int64_t num_pages,
    int64_t page_size,
    int64_t page_stride_bytes,
    int lane) {
  if constexpr (UseU32Offsets) {
    dsv4_gather_one_row_u32<PageSize>(
        src_cache,
        reinterpret_cast<const int32_t*>(token_indices),
        valid_lengths,
        dst,
        compact_indices,
        static_cast<uint32_t>(query),
        static_cast<uint32_t>(column),
        static_cast<uint32_t>(topk),
        static_cast<uint32_t>(num_pages),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(page_stride_bytes),
        lane);
  } else {
    dsv4_gather_one_row<index_t, PageSize>(
        src_cache,
        token_indices,
        valid_lengths,
        dst,
        compact_indices,
        query,
        column,
        topk,
        num_pages,
        page_size,
        page_stride_bytes,
        lane);
  }
}

template <typename index_t, int64_t PageSize, bool UseU32Offsets = false>
__global__ void dsv4_gather_upconvert_single_kernel(
    const uint8_t* __restrict__ src_cache,
    const index_t* __restrict__ token_indices,
    const int32_t* __restrict__ valid_lengths,
    dsv4_bfloat16* __restrict__ dst,
    int32_t* __restrict__ compact_indices,
    int64_t topk,
    int64_t num_pages,
    int64_t page_size,
    int64_t page_stride_bytes) {
  const int lane = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  if constexpr (UseU32Offsets) {
    const uint32_t column =
        blockIdx.x * static_cast<uint32_t>(kRowsPerBlock) +
        static_cast<uint32_t>(row_in_block);
    if (column >= static_cast<uint32_t>(topk)) {
      return;
    }
    dsv4_gather_one_row_u32<PageSize>(
        src_cache,
        reinterpret_cast<const int32_t*>(token_indices),
        valid_lengths,
        dst,
        compact_indices,
        blockIdx.y,
        column,
        static_cast<uint32_t>(topk),
        static_cast<uint32_t>(num_pages),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(page_stride_bytes),
        lane);
  } else {
    const int64_t column =
        static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
    if (column >= topk) {
      return;
    }
    const int64_t query = static_cast<int64_t>(blockIdx.y);
    dsv4_gather_one_row<index_t, PageSize>(
        src_cache,
        token_indices,
        valid_lengths,
        dst,
        compact_indices,
        query,
        column,
        topk,
        num_pages,
        page_size,
        page_stride_bytes,
        lane);
  }
}

template <
    typename index_t,
    int64_t SwaPageSize,
    int64_t ExtraPageSize,
    bool UseU32Offsets = false>
__global__ void dsv4_gather_upconvert_dual_kernel(
    const uint8_t* __restrict__ swa_cache,
    const index_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    dsv4_bfloat16* __restrict__ swa_dst,
    int32_t* __restrict__ swa_compact,
    int64_t swa_topk,
    int64_t swa_num_pages,
    int64_t swa_page_size,
    int64_t swa_page_stride_bytes,
    const uint8_t* __restrict__ extra_cache,
    const index_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    dsv4_bfloat16* __restrict__ extra_dst,
    int32_t* __restrict__ extra_compact,
    int64_t extra_topk,
    int64_t extra_num_pages,
    int64_t extra_page_size,
    int64_t extra_page_stride_bytes) {
  const int lane = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  if constexpr (UseU32Offsets) {
    const uint32_t packed_column =
        blockIdx.x * static_cast<uint32_t>(kRowsPerBlock) +
        static_cast<uint32_t>(row_in_block);
    const uint32_t swa_topk32 = static_cast<uint32_t>(swa_topk);
    const uint32_t extra_topk32 = static_cast<uint32_t>(extra_topk);
    if (packed_column >= swa_topk32 + extra_topk32) {
      return;
    }
    if (packed_column < swa_topk32) {
      dsv4_gather_one_row_u32<SwaPageSize>(
          swa_cache,
          reinterpret_cast<const int32_t*>(swa_indices),
          swa_lengths,
          swa_dst,
          swa_compact,
          blockIdx.y,
          packed_column,
          swa_topk32,
          static_cast<uint32_t>(swa_num_pages),
          static_cast<uint32_t>(swa_page_size),
          static_cast<uint32_t>(swa_page_stride_bytes),
          lane);
    } else {
      dsv4_gather_one_row_u32<ExtraPageSize>(
          extra_cache,
          reinterpret_cast<const int32_t*>(extra_indices),
          extra_lengths,
          extra_dst,
          extra_compact,
          blockIdx.y,
          packed_column - swa_topk32,
          extra_topk32,
          static_cast<uint32_t>(extra_num_pages),
          static_cast<uint32_t>(extra_page_size),
          static_cast<uint32_t>(extra_page_stride_bytes),
          lane);
    }
    return;
  }
  const int64_t packed_column =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  const int64_t combined_topk = swa_topk + extra_topk;
  if (packed_column >= combined_topk) {
    return;
  }

  const int64_t query = static_cast<int64_t>(blockIdx.y);
  if (packed_column < swa_topk) {
    dsv4_gather_one_row<index_t, SwaPageSize>(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_dst,
        swa_compact,
        query,
        packed_column,
        swa_topk,
        swa_num_pages,
        swa_page_size,
        swa_page_stride_bytes,
        lane);
  } else {
    dsv4_gather_one_row<index_t, ExtraPageSize>(
        extra_cache,
        extra_indices,
        extra_lengths,
        extra_dst,
        extra_compact,
        query,
        packed_column - swa_topk,
        extra_topk,
        extra_num_pages,
        extra_page_size,
        extra_page_stride_bytes,
        lane);
  }
}

template <
    typename index_t,
    int64_t SwaPageSize,
    int64_t ExtraPageSize,
    bool UseU32Offsets = false>
__global__ void dsv4_gather_upconvert_dual_tiny_kernel(
    const uint8_t* __restrict__ swa_cache,
    const index_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    dsv4_bfloat16* __restrict__ swa_dst,
    int32_t* __restrict__ swa_compact,
    int64_t swa_topk,
    int64_t swa_num_pages,
    int64_t swa_page_size,
    int64_t swa_page_stride_bytes,
    const uint8_t* __restrict__ extra_cache,
    const index_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    dsv4_bfloat16* __restrict__ extra_dst,
    int32_t* __restrict__ extra_compact,
    int64_t extra_topk,
    int64_t extra_num_pages,
    int64_t extra_page_size,
    int64_t extra_page_stride_bytes,
    int64_t num_queries) {
  const int lane = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  if constexpr (UseU32Offsets) {
    const uint32_t row =
        blockIdx.x * static_cast<uint32_t>(kRowsPerBlock) +
        static_cast<uint32_t>(row_in_block);
    const uint32_t swa_topk32 = static_cast<uint32_t>(swa_topk);
    const uint32_t extra_topk32 = static_cast<uint32_t>(extra_topk);
    const uint32_t combined_topk32 = swa_topk32 + extra_topk32;
    const uint32_t total_rows32 =
        static_cast<uint32_t>(num_queries) * combined_topk32;
    if (row >= total_rows32) {
      return;
    }
    const uint32_t query = row / combined_topk32;
    const uint32_t packed_column = row - query * combined_topk32;
    if (packed_column < swa_topk32) {
      dsv4_gather_one_row_u32<SwaPageSize>(
          swa_cache,
          reinterpret_cast<const int32_t*>(swa_indices),
          swa_lengths,
          swa_dst,
          swa_compact,
          query,
          packed_column,
          swa_topk32,
          static_cast<uint32_t>(swa_num_pages),
          static_cast<uint32_t>(swa_page_size),
          static_cast<uint32_t>(swa_page_stride_bytes),
          lane);
    } else {
      dsv4_gather_one_row_u32<ExtraPageSize>(
          extra_cache,
          reinterpret_cast<const int32_t*>(extra_indices),
          extra_lengths,
          extra_dst,
          extra_compact,
          query,
          packed_column - swa_topk32,
          extra_topk32,
          static_cast<uint32_t>(extra_num_pages),
          static_cast<uint32_t>(extra_page_size),
          static_cast<uint32_t>(extra_page_stride_bytes),
          lane);
    }
    return;
  }
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  const int64_t combined_topk = swa_topk + extra_topk;
  const int64_t total_rows = num_queries * combined_topk;
  if (row >= total_rows) {
    return;
  }

  const int64_t query = row / combined_topk;
  const int64_t packed_column = row - query * combined_topk;
  if (packed_column < swa_topk) {
    dsv4_gather_one_row<index_t, SwaPageSize>(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_dst,
        swa_compact,
        query,
        packed_column,
        swa_topk,
        swa_num_pages,
        swa_page_size,
        swa_page_stride_bytes,
        lane);
  } else {
    dsv4_gather_one_row<index_t, ExtraPageSize>(
        extra_cache,
        extra_indices,
        extra_lengths,
        extra_dst,
        extra_compact,
        query,
        packed_column - swa_topk,
        extra_topk,
        extra_num_pages,
        extra_page_size,
        extra_page_stride_bytes,
        lane);
  }
}

template <
    typename index_t,
    int64_t SwaPageSize,
    int64_t ExtraPageSize,
    bool UseU32Offsets = false>
__global__ void dsv4_gather_upconvert_dual_case13_kernel(
    const uint8_t* __restrict__ swa_cache,
    const index_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    dsv4_bfloat16* __restrict__ swa_dst,
    int32_t* __restrict__ swa_compact,
    int64_t swa_topk,
    int64_t swa_num_pages,
    int64_t swa_page_size,
    int64_t swa_page_stride_bytes,
    const uint8_t* __restrict__ extra_cache,
    const index_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    dsv4_bfloat16* __restrict__ extra_dst,
    int32_t* __restrict__ extra_compact,
    int64_t extra_topk,
    int64_t extra_num_pages,
    int64_t extra_page_size,
    int64_t extra_page_stride_bytes) {
  const int lane = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  const uint32_t query = blockIdx.x / kCase13BlocksPerQuery;
  const uint32_t block_in_query =
      blockIdx.x - query * kCase13BlocksPerQuery;
  if constexpr (UseU32Offsets) {
    const uint32_t packed_column =
        block_in_query * static_cast<uint32_t>(kCase13RowsPerBlock) +
        static_cast<uint32_t>(row_in_block);
    const uint32_t swa_topk32 = static_cast<uint32_t>(swa_topk);
    const uint32_t extra_topk32 = static_cast<uint32_t>(extra_topk);
    if (packed_column >= swa_topk32 + extra_topk32) {
      return;
    }
    if (packed_column < swa_topk32) {
      dsv4_gather_one_row_u32<SwaPageSize>(
          swa_cache,
          reinterpret_cast<const int32_t*>(swa_indices),
          swa_lengths,
          swa_dst,
          swa_compact,
          query,
          packed_column,
          swa_topk32,
          static_cast<uint32_t>(swa_num_pages),
          static_cast<uint32_t>(swa_page_size),
          static_cast<uint32_t>(swa_page_stride_bytes),
          lane);
    } else {
      dsv4_gather_one_row_u32<ExtraPageSize>(
          extra_cache,
          reinterpret_cast<const int32_t*>(extra_indices),
          extra_lengths,
          extra_dst,
          extra_compact,
          query,
          packed_column - swa_topk32,
          extra_topk32,
          static_cast<uint32_t>(extra_num_pages),
          static_cast<uint32_t>(extra_page_size),
          static_cast<uint32_t>(extra_page_stride_bytes),
          lane);
    }
    return;
  }
  const int64_t packed_column =
      static_cast<int64_t>(block_in_query) * kCase13RowsPerBlock +
      row_in_block;
  const int64_t combined_topk = swa_topk + extra_topk;
  if (packed_column >= combined_topk) {
    return;
  }

  if (packed_column < swa_topk) {
    dsv4_gather_one_row<index_t, SwaPageSize>(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_dst,
        swa_compact,
        query,
        packed_column,
        swa_topk,
        swa_num_pages,
        swa_page_size,
        swa_page_stride_bytes,
        lane);
  } else {
    dsv4_gather_one_row<index_t, ExtraPageSize>(
        extra_cache,
        extra_indices,
        extra_lengths,
        extra_dst,
        extra_compact,
        query,
        packed_column - swa_topk,
        extra_topk,
        extra_num_pages,
        extra_page_size,
        extra_page_stride_bytes,
        lane);
  }
}

void check_dsv4_cache(
    const torch::Tensor& cache,
    int64_t page_size,
    const char* name) {
  TORCH_CHECK(cache.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(
      cache.scalar_type() == at::ScalarType::Byte,
      name,
      " must be a uint8 byte view");
  TORCH_CHECK(cache.dim() == 2 || cache.dim() == 4, name, " must be 2-D or 4-D");
  TORCH_CHECK(cache.stride(-1) == 1, name, " must be contiguous in byte dimension");
  TORCH_CHECK(page_size > 0, name, " page_size must be positive");
  if (cache.dim() == 4) {
    TORCH_CHECK(
        cache.size(1) == page_size && cache.size(2) == 1 &&
            cache.size(3) == kDsv4TokenDataBytes + kDsv4ScaleBytesPerToken,
        name,
        " 4-D view must have shape [num_pages, page_size, 1, 584]");
  }
  const int64_t page_stride_bytes = cache.stride(0);
  TORCH_CHECK(
      page_stride_bytes >=
          page_size * (kDsv4TokenDataBytes + kDsv4ScaleBytesPerToken),
      name,
      " page stride is too small for the DPSKV4 paged layout");
}

void check_dsv4_io(
    const torch::Tensor& cache,
    const torch::Tensor& indices,
    const torch::Tensor& lengths,
    const torch::Tensor& dst,
    const torch::Tensor& compact,
    int64_t page_size,
    const char* prefix) {
  check_dsv4_cache(cache, page_size, prefix);
  TORCH_CHECK(indices.is_cuda() && lengths.is_cuda(), prefix, " indices/lengths must be on GPU");
  TORCH_CHECK(dst.is_cuda() && compact.is_cuda(), prefix, " outputs must be on GPU");
  TORCH_CHECK(
      cache.device() == indices.device() && cache.device() == lengths.device() &&
          cache.device() == dst.device() && cache.device() == compact.device(),
      prefix,
      " tensors must be on the same device");
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Int ||
          indices.scalar_type() == at::ScalarType::Long,
      prefix,
      " indices must be int32 or int64");
  TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous(), prefix, " indices must be contiguous [Q, topk]");
  TORCH_CHECK(
      lengths.scalar_type() == at::ScalarType::Int && lengths.dim() == 1 &&
          lengths.is_contiguous() && lengths.size(0) == indices.size(0),
      prefix,
      " lengths must be contiguous int32 [Q]");
  TORCH_CHECK(
      dst.scalar_type() == at::ScalarType::BFloat16 && dst.dim() == 4 &&
          dst.is_contiguous() && dst.size(0) == indices.size(0) &&
          dst.size(1) == indices.size(1) && dst.size(2) == 1 &&
          dst.size(3) == kDsv4HeadDim,
      prefix,
      " dst must be contiguous BF16 [Q, topk, 1, 512]");
  TORCH_CHECK(
      compact.scalar_type() == at::ScalarType::Int && compact.dim() == 3 &&
          compact.is_contiguous() && compact.size(0) == indices.size(0) &&
          compact.size(1) == 1 && compact.size(2) == indices.size(1),
      prefix,
      " compact must be contiguous int32 [Q, 1, topk]");
  TORCH_CHECK(
      indices.size(0) == 0 ||
          indices.size(1) <=
              std::numeric_limits<int32_t>::max() / indices.size(0),
      prefix,
      " compact index space exceeds int32");
}

bool dsv4_u32_offsets_fit(
    const torch::Tensor& cache,
    int64_t num_queries,
    int64_t topk) {
  const uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  const uint64_t rows =
      static_cast<uint64_t>(num_queries) * static_cast<uint64_t>(topk);
  const uint64_t max_output_rows =
      max_u32 / static_cast<uint64_t>(kDsv4HeadDim) + 1;
  const uint64_t page_stride = static_cast<uint64_t>(cache.stride(0));
  const uint64_t num_pages = static_cast<uint64_t>(cache.size(0));
  const bool cache_fits =
      page_stride == 0 || num_pages <= max_u32 / page_stride;
  return rows <= max_output_rows && cache_fits;
}

}  // namespace

void dsv4_gather_upconvert_k_cache_paged(
    const torch::Tensor& cache,
    const torch::Tensor& indices,
    const torch::Tensor& lengths,
    const torch::Tensor& dst,
    const torch::Tensor& compact,
    int64_t page_size) {
  check_dsv4_io(cache, indices, lengths, dst, compact, page_size, "dsv4 single");
  const int64_t num_queries = indices.size(0);
  const int64_t topk = indices.size(1);
  if (num_queries == 0 || topk == 0) {
    return;
  }
  const int64_t blocks_per_query64 =
      (topk + kRowsPerBlock - 1) / kRowsPerBlock;
  TORCH_CHECK(
      blocks_per_query64 <= std::numeric_limits<uint32_t>::max() &&
          num_queries <= std::numeric_limits<uint32_t>::max(),
      "DPSKV4 single launch grid is too large");

  at::cuda::OptionalCUDAGuard device_guard(cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<uint32_t>(blocks_per_query64),
      static_cast<uint32_t>(num_queries));
  const dim3 block(kThreadsPerBlock);
  const bool use_u32_offsets =
      indices.scalar_type() == at::ScalarType::Int &&
      dsv4_u32_offsets_fit(cache, num_queries, topk);
#define DSV4_LAUNCH_SINGLE(index_t, page_size_constant, use_u32)              \
  dsv4_gather_upconvert_single_kernel<                                       \
      index_t, page_size_constant, use_u32>                                   \
      <<<grid, block, 0, stream>>>(                                           \
          cache.data_ptr<uint8_t>(),                                          \
          indices.data_ptr<index_t>(),                                        \
          lengths.data_ptr<int32_t>(),                                        \
          reinterpret_cast<dsv4_bfloat16*>(dst.data_ptr()),                   \
          compact.data_ptr<int32_t>(),                                        \
          topk,                                                               \
          cache.size(0),                                                      \
          page_size,                                                          \
          cache.stride(0))
  if (indices.scalar_type() == at::ScalarType::Int) {
    if (use_u32_offsets && page_size == 256) {
      DSV4_LAUNCH_SINGLE(int32_t, 256, true);
    } else if (use_u32_offsets) {
      DSV4_LAUNCH_SINGLE(int32_t, 0, true);
    } else if (page_size == 256) {
      DSV4_LAUNCH_SINGLE(int32_t, 256, false);
    } else {
      DSV4_LAUNCH_SINGLE(int32_t, 0, false);
    }
  } else {
    if (page_size == 256) {
      DSV4_LAUNCH_SINGLE(int64_t, 256, false);
    } else {
      DSV4_LAUNCH_SINGLE(int64_t, 0, false);
    }
  }
#undef DSV4_LAUNCH_SINGLE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_gather_upconvert_dual_k_cache_paged(
    const torch::Tensor& swa_cache,
    const torch::Tensor& swa_indices,
    const torch::Tensor& swa_lengths,
    const torch::Tensor& swa_dst,
    const torch::Tensor& swa_compact,
    int64_t swa_page_size,
    const torch::Tensor& extra_cache,
    const torch::Tensor& extra_indices,
    const torch::Tensor& extra_lengths,
    const torch::Tensor& extra_dst,
    const torch::Tensor& extra_compact,
    int64_t extra_page_size) {
  check_dsv4_io(
      swa_cache, swa_indices, swa_lengths, swa_dst, swa_compact,
      swa_page_size, "dsv4 swa");
  check_dsv4_io(
      extra_cache, extra_indices, extra_lengths, extra_dst, extra_compact,
      extra_page_size, "dsv4 extra");
  TORCH_CHECK(
      swa_indices.size(0) == extra_indices.size(0),
      "DPSKV4 dual caches must have the same query count");
  TORCH_CHECK(
      swa_indices.scalar_type() == extra_indices.scalar_type(),
      "DPSKV4 dual index tensors must have the same dtype");
  TORCH_CHECK(
      swa_cache.device() == extra_cache.device(),
      "DPSKV4 dual caches must be on the same device");

  const int64_t num_queries = swa_indices.size(0);
  const int64_t swa_topk = swa_indices.size(1);
  const int64_t extra_topk = extra_indices.size(1);
  const int64_t combined_topk = swa_topk + extra_topk;
  if (num_queries == 0 || combined_topk == 0) {
    return;
  }
  const bool use_case13_kernel =
      combined_topk == 8384 && swa_page_size == 256 &&
      extra_page_size == 2;
  const bool use_tiny_kernel =
      !use_case13_kernel &&
      num_queries <= 32768 / combined_topk;
  const int rows_per_block =
      use_case13_kernel ? kCase13RowsPerBlock : kRowsPerBlock;
  const int64_t blocks_per_query64 =
      (combined_topk + rows_per_block - 1) / rows_per_block;
  TORCH_CHECK(
      blocks_per_query64 <= std::numeric_limits<uint32_t>::max() &&
          (use_case13_kernel
               ? num_queries <=
                   std::numeric_limits<uint32_t>::max() / blocks_per_query64
               : num_queries <= std::numeric_limits<uint32_t>::max()),
      "DPSKV4 dual launch grid is too large");
  TORCH_CHECK(
      !use_case13_kernel ||
          blocks_per_query64 == kCase13BlocksPerQuery,
      "DPSKV4 case13 launch specialization mismatch");

  at::cuda::OptionalCUDAGuard device_guard(swa_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_rows = num_queries * combined_topk;
  const dim3 grid = use_case13_kernel
      ? dim3(static_cast<uint32_t>(num_queries * blocks_per_query64))
      : (use_tiny_kernel
             ? dim3(static_cast<uint32_t>(
                   (total_rows + kRowsPerBlock - 1) / kRowsPerBlock))
             : dim3(
                   static_cast<uint32_t>(blocks_per_query64),
                   static_cast<uint32_t>(num_queries)));
  const dim3 block(
      use_case13_kernel ? kCase13ThreadsPerBlock : kThreadsPerBlock);
  const bool use_u32_offsets =
      swa_indices.scalar_type() == at::ScalarType::Int &&
      dsv4_u32_offsets_fit(swa_cache, num_queries, swa_topk) &&
      dsv4_u32_offsets_fit(extra_cache, num_queries, extra_topk);
#define DSV4_LAUNCH_DUAL(index_t, swa_page_size_constant,                    \
                         extra_page_size_constant, use_u32)                   \
  dsv4_gather_upconvert_dual_kernel<                                          \
      index_t, swa_page_size_constant, extra_page_size_constant, use_u32>     \
      <<<grid, block, 0, stream>>>(                                           \
          swa_cache.data_ptr<uint8_t>(),                                      \
          swa_indices.data_ptr<index_t>(),                                    \
          swa_lengths.data_ptr<int32_t>(),                                    \
          reinterpret_cast<dsv4_bfloat16*>(swa_dst.data_ptr()),               \
          swa_compact.data_ptr<int32_t>(),                                    \
          swa_topk,                                                           \
          swa_cache.size(0),                                                  \
          swa_page_size,                                                      \
          swa_cache.stride(0),                                                \
          extra_cache.data_ptr<uint8_t>(),                                    \
          extra_indices.data_ptr<index_t>(),                                  \
          extra_lengths.data_ptr<int32_t>(),                                  \
          reinterpret_cast<dsv4_bfloat16*>(extra_dst.data_ptr()),             \
          extra_compact.data_ptr<int32_t>(),                                  \
          extra_topk,                                                         \
          extra_cache.size(0),                                                \
          extra_page_size,                                                    \
          extra_cache.stride(0))
#define DSV4_LAUNCH_DUAL_TINY(                                                \
    index_t, swa_page_size_constant, extra_page_size_constant, use_u32)       \
  dsv4_gather_upconvert_dual_tiny_kernel<                                     \
      index_t, swa_page_size_constant, extra_page_size_constant, use_u32>     \
      <<<grid, block, 0, stream>>>(                                           \
          swa_cache.data_ptr<uint8_t>(),                                      \
          swa_indices.data_ptr<index_t>(),                                    \
          swa_lengths.data_ptr<int32_t>(),                                    \
          reinterpret_cast<dsv4_bfloat16*>(swa_dst.data_ptr()),               \
          swa_compact.data_ptr<int32_t>(),                                    \
          swa_topk,                                                           \
          swa_cache.size(0),                                                  \
          swa_page_size,                                                      \
          swa_cache.stride(0),                                                \
          extra_cache.data_ptr<uint8_t>(),                                    \
          extra_indices.data_ptr<index_t>(),                                  \
          extra_lengths.data_ptr<int32_t>(),                                  \
          reinterpret_cast<dsv4_bfloat16*>(extra_dst.data_ptr()),             \
          extra_compact.data_ptr<int32_t>(),                                  \
          extra_topk,                                                         \
          extra_cache.size(0),                                                \
          extra_page_size,                                                    \
          extra_cache.stride(0),                                              \
          num_queries)
#define DSV4_LAUNCH_DUAL_CASE13(index_t, use_u32)                            \
  dsv4_gather_upconvert_dual_case13_kernel<index_t, 256, 2, use_u32>         \
      <<<grid, block, 0, stream>>>(                                           \
          swa_cache.data_ptr<uint8_t>(),                                      \
          swa_indices.data_ptr<index_t>(),                                    \
          swa_lengths.data_ptr<int32_t>(),                                    \
          reinterpret_cast<dsv4_bfloat16*>(swa_dst.data_ptr()),               \
          swa_compact.data_ptr<int32_t>(),                                    \
          swa_topk,                                                           \
          swa_cache.size(0),                                                  \
          swa_page_size,                                                      \
          swa_cache.stride(0),                                                \
          extra_cache.data_ptr<uint8_t>(),                                    \
          extra_indices.data_ptr<index_t>(),                                  \
          extra_lengths.data_ptr<int32_t>(),                                  \
          reinterpret_cast<dsv4_bfloat16*>(extra_dst.data_ptr()),             \
          extra_compact.data_ptr<int32_t>(),                                  \
          extra_topk,                                                         \
          extra_cache.size(0),                                                \
          extra_page_size,                                                    \
          extra_cache.stride(0))
  if (swa_indices.scalar_type() == at::ScalarType::Int) {
    if (use_case13_kernel) {
      if (use_u32_offsets) {
        DSV4_LAUNCH_DUAL_CASE13(int32_t, true);
      } else {
        DSV4_LAUNCH_DUAL_CASE13(int32_t, false);
      }
    } else if (use_tiny_kernel) {
      if (swa_page_size == 256 && extra_page_size == 2) {
        if (use_u32_offsets) {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 256, 2, true);
        } else {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 256, 2, false);
        }
      } else if (swa_page_size == 256 && extra_page_size == 64) {
        if (use_u32_offsets) {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 256, 64, true);
        } else {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 256, 64, false);
        }
      } else {
        if (use_u32_offsets) {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 0, 0, true);
        } else {
          DSV4_LAUNCH_DUAL_TINY(int32_t, 0, 0, false);
        }
      }
    } else if (swa_page_size == 256 && extra_page_size == 2) {
      if (use_u32_offsets) {
        DSV4_LAUNCH_DUAL(int32_t, 256, 2, true);
      } else {
        DSV4_LAUNCH_DUAL(int32_t, 256, 2, false);
      }
    } else if (swa_page_size == 256 && extra_page_size == 64) {
      if (use_u32_offsets) {
        DSV4_LAUNCH_DUAL(int32_t, 256, 64, true);
      } else {
        DSV4_LAUNCH_DUAL(int32_t, 256, 64, false);
      }
    } else {
      if (use_u32_offsets) {
        DSV4_LAUNCH_DUAL(int32_t, 0, 0, true);
      } else {
        DSV4_LAUNCH_DUAL(int32_t, 0, 0, false);
      }
    }
  } else {
    if (use_case13_kernel) {
      DSV4_LAUNCH_DUAL_CASE13(int64_t, false);
    } else if (use_tiny_kernel) {
      if (swa_page_size == 256 && extra_page_size == 2) {
        DSV4_LAUNCH_DUAL_TINY(int64_t, 256, 2, false);
      } else if (swa_page_size == 256 && extra_page_size == 64) {
        DSV4_LAUNCH_DUAL_TINY(int64_t, 256, 64, false);
      } else {
        DSV4_LAUNCH_DUAL_TINY(int64_t, 0, 0, false);
      }
    } else if (swa_page_size == 256 && extra_page_size == 2) {
      DSV4_LAUNCH_DUAL(int64_t, 256, 2, false);
    } else if (swa_page_size == 256 && extra_page_size == 64) {
      DSV4_LAUNCH_DUAL(int64_t, 256, 64, false);
    } else {
      DSV4_LAUNCH_DUAL(int64_t, 0, 0, false);
    }
  }
#undef DSV4_LAUNCH_DUAL_CASE13
#undef DSV4_LAUNCH_DUAL_TINY
#undef DSV4_LAUNCH_DUAL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace dsv4_dequant_detail {

// Flat physical-token gather. Keep the query/top-k kernels above unchanged.
// This entry point consumes E4M3FN/OCP bytes, not E4M3FNUZ bytes.
template <typename index_t, int64_t PageSize, bool VectorIO, typename offset_t>
__global__ void dsv4_dequantize_k_cache_flat_out_kernel(
    const uint8_t* __restrict__ cache,
    const index_t* __restrict__ page_table,
    uint16_t* __restrict__ out,
    offset_t num_rows,
    offset_t num_locations,
    offset_t page_stride_bytes,
    offset_t output_stride) {
  const offset_t row = static_cast<offset_t>(blockIdx.x) * kRowsPerBlock +
      static_cast<offset_t>(threadIdx.x / kThreadsPerRow);
  if (row >= num_rows) {
    return;
  }
  const int lane = threadIdx.x % kThreadsPerRow;
  const int64_t location = static_cast<int64_t>(page_table[row]);
  const bool valid_location =
      location >= 0 && location < static_cast<int64_t>(num_locations);
  // A masked invalid location must never form an out-of-range source pointer.
  const offset_t safe_location =
      valid_location ? static_cast<offset_t>(location) : 0;
  const offset_t page = safe_location / PageSize;
  const offset_t in_page = safe_location % PageSize;
  const offset_t data_base = page * page_stride_bytes +
      in_page * kDsv4TokenDataBytes;
  const offset_t scale_base = page * page_stride_bytes +
      PageSize * kDsv4TokenDataBytes + in_page * kDsv4ScaleBytesPerToken;
  uint16_t* const output_lane =
      out + row * output_stride + lane * kDsv4ElementsPerLane;

  using Fp8Vec = Dsv4Vector<uint8_t, kDsv4ElementsPerLane>;
  using PackedBf16Vec = Dsv4Vector<uint32_t, kDsv4ElementsPerLane / 2>;
  PackedBf16Vec output_vec;
  if (valid_location && lane < kDsv4NopeLanes) {
    const offset_t input_base = data_base + lane * kDsv4ElementsPerLane;
    const float scale = dsv4_ue8m0_to_float(
        cache[scale_base + lane / (kDsv4TileSize / kDsv4ElementsPerLane)]);
    if constexpr (VectorIO) {
      const Fp8Vec input_vec =
          *reinterpret_cast<const Fp8Vec*>(cache + input_base);
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        output_vec.val[i] = dsv4_flat_scaled_pair_to_bf16_rne(
            input_vec.val[2 * i], input_vec.val[2 * i + 1], scale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        output_vec.val[i] = dsv4_flat_scaled_pair_to_bf16_rne(
            cache[input_base + 2 * i], cache[input_base + 2 * i + 1],
            scale);
      }
    }
  } else if (valid_location) {
    const offset_t input_base = data_base + kDsv4NopeDim +
        (lane - kDsv4NopeLanes) * sizeof(PackedBf16Vec);
    if constexpr (VectorIO) {
      output_vec = *reinterpret_cast<const PackedBf16Vec*>(cache + input_base);
    } else {
      // Byte loads also support a byte-offset cache view without aligned casts.
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        const offset_t byte = input_base + 4 * i;
        output_vec.val[i] = static_cast<uint32_t>(cache[byte]) |
            (static_cast<uint32_t>(cache[byte + 1]) << 8) |
            (static_cast<uint32_t>(cache[byte + 2]) << 16) |
            (static_cast<uint32_t>(cache[byte + 3]) << 24);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = 0;
    }
  }

  if constexpr (VectorIO) {
    *reinterpret_cast<PackedBf16Vec*>(output_lane) = output_vec;
  } else {
    // Preserve the pair helper's raw little-endian BF16 packing; no FP cast.
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_lane[2 * i] = static_cast<uint16_t>(output_vec.val[i]);
      output_lane[2 * i + 1] = static_cast<uint16_t>(output_vec.val[i] >> 16);
    }
  }
}

template <typename index_t, int64_t PageSize, bool VectorIO, typename offset_t>
__global__ void dsv4_dequantize_k_cache_flat_out_counted_kernel(
    const uint8_t* __restrict__ cache,
    const index_t* __restrict__ page_table,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ valid_count,
    offset_t num_rows,
    offset_t num_locations,
    offset_t page_stride_bytes,
    offset_t output_stride) {
  const offset_t row = static_cast<offset_t>(blockIdx.x) * kRowsPerBlock +
      static_cast<offset_t>(threadIdx.x / kThreadsPerRow);
  if (row >= num_rows) {
    return;
  }
  // The count stays on-device. Do not touch inactive IDs, cache or out.
  const int64_t active_rows = static_cast<int64_t>(*valid_count);
  if (static_cast<int64_t>(row) >= active_rows) {
    return;
  }
  const int lane = threadIdx.x % kThreadsPerRow;
  const int64_t location = static_cast<int64_t>(page_table[row]);
  const bool valid_location =
      location >= 0 && location < static_cast<int64_t>(num_locations);
  // A masked invalid location must never form an out-of-range source pointer.
  const offset_t safe_location =
      valid_location ? static_cast<offset_t>(location) : 0;
  const offset_t page = safe_location / PageSize;
  const offset_t in_page = safe_location % PageSize;
  const offset_t data_base = page * page_stride_bytes +
      in_page * kDsv4TokenDataBytes;
  const offset_t scale_base = page * page_stride_bytes +
      PageSize * kDsv4TokenDataBytes + in_page * kDsv4ScaleBytesPerToken;
  uint16_t* const output_lane =
      out + row * output_stride + lane * kDsv4ElementsPerLane;

  using Fp8Vec = Dsv4Vector<uint8_t, kDsv4ElementsPerLane>;
  using PackedBf16Vec = Dsv4Vector<uint32_t, kDsv4ElementsPerLane / 2>;
  PackedBf16Vec output_vec;
  if (valid_location && lane < kDsv4NopeLanes) {
    const offset_t input_base = data_base + lane * kDsv4ElementsPerLane;
    const float scale = dsv4_ue8m0_to_float(
        cache[scale_base + lane / (kDsv4TileSize / kDsv4ElementsPerLane)]);
    if constexpr (VectorIO) {
      const Fp8Vec input_vec =
          *reinterpret_cast<const Fp8Vec*>(cache + input_base);
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        output_vec.val[i] = dsv4_flat_scaled_pair_to_bf16_rne(
            input_vec.val[2 * i], input_vec.val[2 * i + 1], scale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        output_vec.val[i] = dsv4_flat_scaled_pair_to_bf16_rne(
            cache[input_base + 2 * i], cache[input_base + 2 * i + 1],
            scale);
      }
    }
  } else if (valid_location) {
    const offset_t input_base = data_base + kDsv4NopeDim +
        (lane - kDsv4NopeLanes) * sizeof(PackedBf16Vec);
    if constexpr (VectorIO) {
      output_vec = *reinterpret_cast<const PackedBf16Vec*>(cache + input_base);
    } else {
      // Byte loads also support a byte-offset cache view without aligned casts.
#pragma unroll
      for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
        const offset_t byte = input_base + 4 * i;
        output_vec.val[i] = static_cast<uint32_t>(cache[byte]) |
            (static_cast<uint32_t>(cache[byte + 1]) << 8) |
            (static_cast<uint32_t>(cache[byte + 2]) << 16) |
            (static_cast<uint32_t>(cache[byte + 3]) << 24);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_vec.val[i] = 0;
    }
  }

  if constexpr (VectorIO) {
    *reinterpret_cast<PackedBf16Vec*>(output_lane) = output_vec;
  } else {
    // Preserve the pair helper's raw little-endian BF16 packing; no FP cast.
#pragma unroll
    for (int i = 0; i < kDsv4ElementsPerLane / 2; ++i) {
      output_lane[2 * i] = static_cast<uint16_t>(output_vec.val[i]);
      output_lane[2 * i + 1] = static_cast<uint16_t>(output_vec.val[i] >> 16);
    }
  }
}

struct Dsv4FlatByteRange {
  uintptr_t begin;
  uintptr_t end;
};

int64_t dsv4_flat_checked_multiply(int64_t first, int64_t second) {
  TORCH_CHECK(
      first >= 0 && second >= 0 &&
          (first == 0 || second <= std::numeric_limits<int64_t>::max() / first),
      "DSV4 flat dequant: address multiplication overflow");
  return first * second;
}

int64_t dsv4_flat_checked_add(int64_t first, int64_t second) {
  TORCH_CHECK(
      first >= 0 && second >= 0 &&
          second <= std::numeric_limits<int64_t>::max() - first,
      "DSV4 flat dequant: address addition overflow");
  return first + second;
}

Dsv4FlatByteRange dsv4_flat_byte_range(
    const torch::Tensor& tensor, int64_t elements) {
  const int64_t bytes = dsv4_flat_checked_multiply(elements, tensor.element_size());
  const int64_t offset_bytes =
      dsv4_flat_checked_multiply(tensor.storage_offset(), tensor.element_size());
  const uint64_t storage_bytes = static_cast<uint64_t>(tensor.storage().nbytes());
  TORCH_CHECK(
      static_cast<uint64_t>(offset_bytes) <= storage_bytes &&
          static_cast<uint64_t>(bytes) <= storage_bytes - offset_bytes,
      "DSV4 flat dequant: accessed span exceeds tensor storage");
  if (bytes == 0) {
    return {0, 0};
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TORCH_CHECK(
      begin != 0 && static_cast<uint64_t>(bytes) <=
          std::numeric_limits<uintptr_t>::max() - begin,
      "DSV4 flat dequant: invalid storage address range");
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

bool dsv4_flat_ranges_overlap(
    const Dsv4FlatByteRange& first, const Dsv4FlatByteRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

void check_dsv4_flat_io(
    const torch::Tensor& cache,
    const torch::Tensor& page_table,
    const torch::Tensor& out,
    int64_t page_size,
    bool allow_empty_cache = false) {
  const torch::Tensor* tensors[] = {&cache, &page_table, &out};
  for (const auto* tensor : tensors) {
    TORCH_CHECK(
        tensor->defined() && tensor->layout() == c10::Layout::Strided &&
            tensor->is_cuda(),
        "DSV4 flat dequant: defined strided GPU tensors required");
    TORCH_CHECK(
        tensor->device() == cache.device(),
        "DSV4 flat dequant: tensors must be on the same device");
    TORCH_CHECK(
        !tensor->requires_grad() && !tensor->is_conj() && !tensor->is_neg(),
        "DSV4 flat dequant: inference-only tensors without unresolved views required");
  }
  TORCH_CHECK(
      page_size == 2 || page_size == 64 || page_size == 256,
      "DSV4 flat dequant: page_size must be 2, 64 or 256");
  TORCH_CHECK(
      cache.scalar_type() == at::ScalarType::Byte && cache.dim() == 2 &&
          cache.is_contiguous() && cache.stride(1) == 1 &&
          cache.size(1) >= page_size *
              (kDsv4TokenDataBytes + kDsv4ScaleBytesPerToken),
      "DSV4 flat dequant: contiguous uint8 cache[num_pages, B], B >= page_size*584 required");
  TORCH_CHECK(
      (page_table.scalar_type() == at::ScalarType::Int ||
       page_table.scalar_type() == at::ScalarType::Long) &&
          page_table.dim() == 1 && page_table.is_contiguous(),
      "DSV4 flat dequant: contiguous int32/int64 page_table[N] required");
  const int64_t num_rows = page_table.size(0);
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::BFloat16 && out.dim() == 3 &&
          out.size(0) == num_rows && out.size(1) == 1 &&
          out.size(2) == kDsv4HeadDim && out.stride(2) == 1 &&
          out.stride(0) >= 0 &&
          (num_rows <= 1 || out.stride(0) >= kDsv4HeadDim),
      "DSV4 flat dequant: BF16 out[N,1,512] with inner stride 1 and nonoverlapping rows required");
  TORCH_CHECK(
      num_rows == 0 || allow_empty_cache || cache.size(0) > 0,
      "DSV4 flat dequant: nonempty input IDs require at least one cache page");
  dsv4_flat_checked_multiply(cache.size(0), page_size);
  const int64_t cache_elements =
      dsv4_flat_checked_multiply(cache.size(0), cache.size(1));
  const int64_t output_elements = num_rows == 0 ? 0 : dsv4_flat_checked_add(
      dsv4_flat_checked_multiply(num_rows - 1, out.stride(0)), kDsv4HeadDim);
  const auto cache_range = dsv4_flat_byte_range(cache, cache_elements);
  const auto index_range = dsv4_flat_byte_range(page_table, num_rows);
  const auto output_range = dsv4_flat_byte_range(out, output_elements);
  TORCH_CHECK(
      !dsv4_flat_ranges_overlap(output_range, cache_range) &&
          !dsv4_flat_ranges_overlap(output_range, index_range),
      "DSV4 flat dequant: out must not overlap cache or page_table storage spans");
  TORCH_CHECK(
      num_rows == 0 || (output_range.begin % alignof(uint16_t) == 0 &&
          index_range.begin % page_table.element_size() == 0),
      "DSV4 flat dequant: scalar tensor pointer alignment required");
}

void launch(
    const torch::Tensor& cache_u8,
    const torch::Tensor& page_table,
    const torch::Tensor& out,
    int64_t page_size) {
  check_dsv4_flat_io(cache_u8, page_table, out, page_size);
  const int64_t num_rows = page_table.size(0);
  if (num_rows == 0) {
    return;
  }
  const int64_t blocks = num_rows / kRowsPerBlock +
      (num_rows % kRowsPerBlock != 0);
  const at::cuda::OptionalCUDAGuard device_guard(cache_u8.device());
  TORCH_CHECK(
      blocks <= at::cuda::getDeviceProperties(cache_u8.get_device())->maxGridSize[0],
      "DSV4 flat dequant: launch grid is too large");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<uint32_t>(blocks));
  const dim3 block(kThreadsPerBlock);
  // Contiguous cache pages use B bytes even if a singleton page has a dummy stride.
  const int64_t page_stride_bytes = cache_u8.size(1);
  const int64_t num_locations =
      dsv4_flat_checked_multiply(cache_u8.size(0), page_size);
  const int64_t output_span = dsv4_flat_checked_add(
      dsv4_flat_checked_multiply(num_rows - 1, out.stride(0)), kDsv4HeadDim);
  const int64_t uint32_limit = std::numeric_limits<uint32_t>::max();
  const bool use_32bit_offsets =
      cache_u8.numel() <= uint32_limit && output_span <= uint32_limit &&
      num_rows <= uint32_limit && num_locations <= uint32_limit &&
      page_stride_bytes <= uint32_limit && out.stride(0) <= uint32_limit;
  const bool vector_io =
      reinterpret_cast<uintptr_t>(cache_u8.data_ptr()) % 32 == 0 &&
      page_stride_bytes % 32 == 0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 32 == 0 &&
      (num_rows == 1 || out.stride(0) % (32 / sizeof(uint16_t)) == 0);
#define DSV4_LAUNCH_FLAT(                                                     \
    index_t, page_constant, vector_constant, offset_t)                        \
  dsv4_dequantize_k_cache_flat_out_kernel<                                    \
      index_t, page_constant, vector_constant, offset_t>                      \
      <<<grid, block, 0, stream>>>(                                           \
          cache_u8.data_ptr<uint8_t>(),                                      \
          page_table.data_ptr<index_t>(),                                    \
          reinterpret_cast<uint16_t*>(out.data_ptr()),                        \
          static_cast<offset_t>(num_rows),                                   \
          static_cast<offset_t>(num_locations),                              \
          static_cast<offset_t>(page_stride_bytes),                          \
          static_cast<offset_t>(out.stride(0)))
#define DSV4_DISPATCH_FLAT(index_t, vector_constant, offset_t)                \
  do {                                                                        \
    if (page_size == 2) {                                                     \
      DSV4_LAUNCH_FLAT(index_t, 2, vector_constant, offset_t);                \
    } else if (page_size == 64) {                                             \
      DSV4_LAUNCH_FLAT(index_t, 64, vector_constant, offset_t);               \
    } else {                                                                  \
      DSV4_LAUNCH_FLAT(index_t, 256, vector_constant, offset_t);              \
    }                                                                         \
  } while (false)
  if (page_table.scalar_type() == at::ScalarType::Int) {
    if (use_32bit_offsets && vector_io) {
      DSV4_DISPATCH_FLAT(int32_t, true, uint32_t);
    } else if (use_32bit_offsets) {
      DSV4_DISPATCH_FLAT(int32_t, false, uint32_t);
    } else if (vector_io) {
      DSV4_DISPATCH_FLAT(int32_t, true, int64_t);
    } else {
      DSV4_DISPATCH_FLAT(int32_t, false, int64_t);
    }
  } else {
    if (use_32bit_offsets && vector_io) {
      DSV4_DISPATCH_FLAT(int64_t, true, uint32_t);
    } else if (use_32bit_offsets) {
      DSV4_DISPATCH_FLAT(int64_t, false, uint32_t);
    } else if (vector_io) {
      DSV4_DISPATCH_FLAT(int64_t, true, int64_t);
    } else {
      DSV4_DISPATCH_FLAT(int64_t, false, int64_t);
    }
  }
#undef DSV4_DISPATCH_FLAT
#undef DSV4_LAUNCH_FLAT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_dsv4_flat_valid_count(
    const torch::Tensor& cache,
    const torch::Tensor& out,
    const torch::Tensor& valid_count) {
  TORCH_CHECK(
      valid_count.defined() && valid_count.layout() == c10::Layout::Strided &&
          valid_count.is_cuda() && valid_count.device() == cache.device() &&
          valid_count.scalar_type() == at::ScalarType::Int &&
          valid_count.numel() == 1 && valid_count.is_contiguous(),
      "DSV4 flat dequant: valid_count must be one contiguous int32 value on the same GPU");
  TORCH_CHECK(
      !valid_count.requires_grad() && !valid_count.is_conj() && !valid_count.is_neg(),
      "DSV4 flat dequant: valid_count must be an inference tensor without unresolved views");
  const int64_t num_rows = out.size(0);
  const int64_t output_elements = num_rows == 0 ? 0 : dsv4_flat_checked_add(
      dsv4_flat_checked_multiply(num_rows - 1, out.stride(0)), kDsv4HeadDim);
  const auto count_range = dsv4_flat_byte_range(valid_count, 1);
  const auto output_range = dsv4_flat_byte_range(out, output_elements);
  TORCH_CHECK(
      count_range.begin % alignof(int32_t) == 0 &&
          !dsv4_flat_ranges_overlap(output_range, count_range),
      "DSV4 flat dequant: valid_count must be aligned and must not overlap out");
}

void launch_counted(
    const torch::Tensor& cache_u8,
    const torch::Tensor& page_table,
    const torch::Tensor& out,
    int64_t page_size,
    const torch::Tensor& valid_count) {
  check_dsv4_flat_io(cache_u8, page_table, out, page_size, true);
  check_dsv4_flat_valid_count(cache_u8, out, valid_count);
  const int64_t num_rows = page_table.size(0);
  if (num_rows == 0) {
    return;
  }
  const int64_t blocks = num_rows / kRowsPerBlock +
      (num_rows % kRowsPerBlock != 0);
  const at::cuda::OptionalCUDAGuard device_guard(cache_u8.device());
  TORCH_CHECK(
      blocks <= at::cuda::getDeviceProperties(cache_u8.get_device())->maxGridSize[0],
      "DSV4 flat dequant: launch grid is too large");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<uint32_t>(blocks));
  const dim3 block(kThreadsPerBlock);
  // Contiguous cache pages use B bytes even if a singleton page has a dummy stride.
  const int64_t page_stride_bytes = cache_u8.size(1);
  const int64_t num_locations =
      dsv4_flat_checked_multiply(cache_u8.size(0), page_size);
  const int64_t output_span = dsv4_flat_checked_add(
      dsv4_flat_checked_multiply(num_rows - 1, out.stride(0)), kDsv4HeadDim);
  const int64_t uint32_limit = std::numeric_limits<uint32_t>::max();
  const bool use_32bit_offsets =
      cache_u8.numel() <= uint32_limit && output_span <= uint32_limit &&
      num_rows <= uint32_limit && num_locations <= uint32_limit &&
      page_stride_bytes <= uint32_limit && out.stride(0) <= uint32_limit;
  const bool vector_io =
      reinterpret_cast<uintptr_t>(cache_u8.data_ptr()) % 32 == 0 &&
      page_stride_bytes % 32 == 0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 32 == 0 &&
      (num_rows == 1 || out.stride(0) % (32 / sizeof(uint16_t)) == 0);
#define DSV4_LAUNCH_FLAT_COUNTED(                                                     \
    index_t, page_constant, vector_constant, offset_t)                        \
  dsv4_dequantize_k_cache_flat_out_counted_kernel<                                    \
      index_t, page_constant, vector_constant, offset_t>                      \
      <<<grid, block, 0, stream>>>(                                           \
          cache_u8.data_ptr<uint8_t>(),                                      \
          page_table.data_ptr<index_t>(),                                    \
          reinterpret_cast<uint16_t*>(out.data_ptr()),                        \
          valid_count.data_ptr<int32_t>(),                                   \
          static_cast<offset_t>(num_rows),                                   \
          static_cast<offset_t>(num_locations),                              \
          static_cast<offset_t>(page_stride_bytes),                          \
          static_cast<offset_t>(out.stride(0)))
#define DSV4_DISPATCH_FLAT_COUNTED(index_t, vector_constant, offset_t)                \
  do {                                                                        \
    if (page_size == 2) {                                                     \
      DSV4_LAUNCH_FLAT_COUNTED(index_t, 2, vector_constant, offset_t);                \
    } else if (page_size == 64) {                                             \
      DSV4_LAUNCH_FLAT_COUNTED(index_t, 64, vector_constant, offset_t);               \
    } else {                                                                  \
      DSV4_LAUNCH_FLAT_COUNTED(index_t, 256, vector_constant, offset_t);              \
    }                                                                         \
  } while (false)
  if (page_table.scalar_type() == at::ScalarType::Int) {
    if (use_32bit_offsets && vector_io) {
      DSV4_DISPATCH_FLAT_COUNTED(int32_t, true, uint32_t);
    } else if (use_32bit_offsets) {
      DSV4_DISPATCH_FLAT_COUNTED(int32_t, false, uint32_t);
    } else if (vector_io) {
      DSV4_DISPATCH_FLAT_COUNTED(int32_t, true, int64_t);
    } else {
      DSV4_DISPATCH_FLAT_COUNTED(int32_t, false, int64_t);
    }
  } else {
    if (use_32bit_offsets && vector_io) {
      DSV4_DISPATCH_FLAT_COUNTED(int64_t, true, uint32_t);
    } else if (use_32bit_offsets) {
      DSV4_DISPATCH_FLAT_COUNTED(int64_t, false, uint32_t);
    } else if (vector_io) {
      DSV4_DISPATCH_FLAT_COUNTED(int64_t, true, int64_t);
    } else {
      DSV4_DISPATCH_FLAT_COUNTED(int64_t, false, int64_t);
    }
  }
#undef DSV4_DISPATCH_FLAT_COUNTED
#undef DSV4_LAUNCH_FLAT_COUNTED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace dsv4_dequant_detail

void dsv4_dequantize_k_cache_paged_out(
    const torch::Tensor& cache_u8,
    const torch::Tensor& page_table,
    const torch::Tensor& out,
    int64_t page_size,
    std::optional<torch::Tensor> valid_count) {
  if (valid_count.has_value()) {
    dsv4_dequant_detail::launch_counted(
        cache_u8, page_table, out, page_size, *valid_count);
  } else {
    dsv4_dequant_detail::launch(cache_u8, page_table, out, page_size);
  }
}

}  // namespace native
}  // namespace at
