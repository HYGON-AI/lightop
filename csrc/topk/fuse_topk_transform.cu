// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

/**
 * @NOTE: Unified Fast TopK Implementation
 * Merges the highly optimized CUB-based 11-11-10 bit filtering Top-K (Code 2)
 * with the memory layout transformations (Page Table / Ragged Tensor) of TileLang (Code 1).
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #define WARP_SIZE 32
#else
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #include <hipcub/hipcub.hpp>
  #include "adaptive_topk.h"
  #define WARP_SIZE 64
#endif

namespace at {
namespace native {
namespace {

constexpr int kThreadsPerBlock = 1024;
constexpr int kKPool4VgprItemsPerThread = 22;
constexpr int kKPool4VgprMaxLength =
    kThreadsPerBlock * kKPool4VgprItemsPerThread;
constexpr int kKPool4VgprCandidateCapacity = 2048;
constexpr size_t kKPool4VgprSmemBytes =
    kKPool4VgprCandidateCapacity * sizeof(uint32_t);

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f>
void setup_kernel_smem_once(size_t max_dynamic_smem) {
  [[maybe_unused]]
  static const auto result = [&] {
#ifdef USE_ROCM
    return ::hipFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::hipFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}


__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

__device__ __forceinline__ uint32_t kpool_descending_key(float value) {
  return ~convert_to_uint32(value);
}

template <int kItems>
__device__ __forceinline__ void kpool_radix_count_2bit(
    const uint32_t (&keys)[kItems],
    uint32_t desired,
    uint32_t mask,
    int bit_pos,
    int* __restrict__ histogram,
    int (&counts)[4]) {
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  if (tid < 4) histogram[tid] = 0;
  __syncthreads();

  int wave_counts[4] = {0, 0, 0, 0};
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const uint32_t key = keys[item];
    const bool matches = (key & mask) == desired;
    const int digit = (key >> bit_pos) & 0x3;
#pragma unroll
    for (int candidate_digit = 0; candidate_digit < 4;
         ++candidate_digit) {
#ifdef USE_ROCM
      const uint64_t votes =
          __ballot(matches && digit == candidate_digit);
#else
      const uint64_t votes =
          __ballot_sync(0xffffffff, matches && digit == candidate_digit);
#endif
      if (lane == 0) {
        wave_counts[candidate_digit] += __popcll(votes);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int digit = 0; digit < 4; ++digit) {
      atomicAdd(&histogram[digit], wave_counts[digit]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int digit = 0; digit < 4; ++digit) {
    counts[digit] = histogram[digit];
  }
  __syncthreads();
}

template <int kItems>
__device__ __forceinline__ uint32_t kpool_radix_select_descending(
    const uint32_t (&keys)[kItems],
    int rank,
    int* __restrict__ histogram) {
  uint32_t desired = 0;
  uint32_t mask = 0;

#pragma unroll
  for (int bit_pos = 30; bit_pos >= 0; bit_pos -= 2) {
    int counts[4];
    kpool_radix_count_2bit<kItems>(
        keys, desired, mask, bit_pos, histogram, counts);
#pragma unroll
    for (int digit = 3; digit >= 0; --digit) {
      if (counts[digit] >= rank) {
        desired |= static_cast<uint32_t>(digit) << bit_pos;
        mask |= uint32_t{0x3} << bit_pos;
        break;
      }
      rank -= counts[digit];
    }
  }
  return desired;
}

template <int kItems>
__device__ __forceinline__ uint32_t kpool_radix_select_descending_4bit(
    const uint32_t (&keys)[kItems],
    int rank,
    int* __restrict__ histogram,
    int* __restrict__ wave_histogram) {
  constexpr int kRadix = 16;
  constexpr int kNumWaves = kThreadsPerBlock / WARP_SIZE;
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int wave = tid / WARP_SIZE;
  uint32_t desired = 0;
  uint32_t mask = 0;

#pragma unroll
  for (int bit_pos = 28; bit_pos >= 0; bit_pos -= 4) {
    int wave_count = 0;
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const uint32_t key = keys[item];
      const bool matches = (key & mask) == desired;
      const int digit = (key >> bit_pos) & 0xf;
#ifdef USE_ROCM
      const uint64_t active = __ballot(matches);
      const uint64_t bit0 = __ballot(matches && (digit & 0x1));
      const uint64_t bit1 = __ballot(matches && (digit & 0x2));
      const uint64_t bit2 = __ballot(matches && (digit & 0x4));
      const uint64_t bit3 = __ballot(matches && (digit & 0x8));
#else
      const uint64_t active = __ballot_sync(0xffffffff, matches);
      const uint64_t bit0 =
          __ballot_sync(0xffffffff, matches && (digit & 0x1));
      const uint64_t bit1 =
          __ballot_sync(0xffffffff, matches && (digit & 0x2));
      const uint64_t bit2 =
          __ballot_sync(0xffffffff, matches && (digit & 0x4));
      const uint64_t bit3 =
          __ballot_sync(0xffffffff, matches && (digit & 0x8));
#endif
      if (lane < kRadix) {
        uint64_t votes = active;
        votes &= (lane & 0x1) ? bit0 : ~bit0;
        votes &= (lane & 0x2) ? bit1 : ~bit1;
        votes &= (lane & 0x4) ? bit2 : ~bit2;
        votes &= (lane & 0x8) ? bit3 : ~bit3;
        wave_count += __popcll(votes);
      }
    }

    if (lane < kRadix) {
      wave_histogram[wave * kRadix + lane] = wave_count;
    }
    __syncthreads();

    if (tid < kRadix) {
      int total = 0;
#pragma unroll
      for (int wave_idx = 0; wave_idx < kNumWaves; ++wave_idx) {
        total += wave_histogram[wave_idx * kRadix + tid];
      }
      histogram[tid] = total;
    }
    __syncthreads();

#pragma unroll
    for (int digit = kRadix - 1; digit >= 0; --digit) {
      const int count = histogram[digit];
      if (count >= rank) {
        desired |= static_cast<uint32_t>(digit) << bit_pos;
        mask |= uint32_t{0xf} << bit_pos;
        break;
      }
      rank -= count;
    }
  }
  return desired;
}

template <int K, int kItems = kKPool4VgprItemsPerThread>
__device__ void select_kpool_topk_vgpr(
    const float* __restrict__ input,
    int* __restrict__ indices,
    int row_start,
    int length,
    uint32_t* __restrict__ candidate_workspace) {
  static_assert(K == 512);
  static_assert(kItems >= 4);
  constexpr int kSampleItems = 4;
  constexpr int kCompactItems = 2;
  constexpr int kSampleRank =
      (K / 2) * kKPool4VgprItemsPerThread / kItems;
  constexpr int kSampleStride = kItems / kSampleItems;

  __shared__ int radix_histogram[16];
  __shared__ int radix_wave_histogram[
      (kThreadsPerBlock / WARP_SIZE) * 16];
  __shared__ int candidate_count;
  __shared__ int sample_equal_count;
  __shared__ int write_count;
  __shared__ int tie_count;

  const int tid = threadIdx.x;
  uint32_t keys[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kThreadsPerBlock;
    keys[item] = idx < length
        ? kpool_descending_key(input[row_start + idx])
        : uint32_t{0};
  }
  uint32_t sample_keys[kSampleItems] = {
      keys[0],
      keys[kSampleStride],
      keys[2 * kSampleStride],
      keys[3 * kSampleStride]};
  const uint32_t sample_threshold =
      kpool_radix_select_descending_4bit<kSampleItems>(
          sample_keys,
          kSampleRank,
          radix_histogram,
          radix_wave_histogram);

  if (tid == 0) {
    candidate_count = 0;
    sample_equal_count = 0;
  }
  __syncthreads();
  int local_sample_equals = 0;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kThreadsPerBlock;
    if (idx < length && keys[item] > sample_threshold) {
      const int pos = atomicAdd(&candidate_count, 1);
      if (pos < kKPool4VgprCandidateCapacity) {
        candidate_workspace[pos] = keys[item];
      }
    } else if (idx < length && keys[item] == sample_threshold) {
      ++local_sample_equals;
    }
  }
  if (local_sample_equals != 0) {
    atomicAdd(&sample_equal_count, local_sample_equals);
  }
  __syncthreads();

  uint32_t kth_key;
  const int compact_count = candidate_count;
  if (compact_count >= K &&
      compact_count <= kKPool4VgprCandidateCapacity) {
    uint32_t compact_keys[kCompactItems];
#pragma unroll
    for (int item = 0; item < kCompactItems; ++item) {
      const int pos = tid + item * kThreadsPerBlock;
      compact_keys[item] =
          pos < compact_count ? candidate_workspace[pos] : uint32_t{0};
    }
    kth_key = kpool_radix_select_descending_4bit<kCompactItems>(
        compact_keys, K, radix_histogram, radix_wave_histogram);
  } else if (
      compact_count < K &&
      compact_count + sample_equal_count >= K) {
    kth_key = sample_threshold;
  } else {
    kth_key = kpool_radix_select_descending<kItems>(
        keys, K, radix_histogram);
  }

  if (tid == 0) write_count = 0;
  __syncthreads();
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kThreadsPerBlock;
    if (idx < length && keys[item] > kth_key) {
      indices[atomicAdd(&write_count, 1)] = idx;
    }
  }
  __syncthreads();

  const int better_count = write_count;
  const int ties_needed = K - better_count;
  if (tid == 0) tie_count = 0;
  __syncthreads();
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kThreadsPerBlock;
    if (idx < length && keys[item] == kth_key) {
      const int tie_rank = atomicAdd(&tie_count, 1);
      if (tie_rank < ties_needed) {
        indices[better_count + tie_rank] = idx;
      }
    }
  }
  __syncthreads();
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, idxT len, Func f) {
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
    if (skip_cnt > len) skip_cnt = len;
    
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    if (thread_rank < skip_cnt) f(in[thread_rank], thread_rank);
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) f(in[remain_i], remain_i);
  }
}

template <typename T, typename IdxT, typename Func>
__device__ __forceinline__ void vectorized_write(size_t thread_rank, size_t num_threads,
                                                 T* out, IdxT len, Func f) {
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = thread_rank; i < len; i += num_threads) out[i] = f(i);
  } else {
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(out) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(out) % sizeof(WideT)) / sizeof(T))
            : 0;
    if (skip_cnt > len) skip_cnt = len;

    if (thread_rank < skip_cnt) out[thread_rank] = f(thread_rank);

    WideT* out_cast = reinterpret_cast<decltype(out_cast)>(out + skip_cnt);
    const IdxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (IdxT i = thread_rank; i < len_cast; i += num_threads) {
      const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) wide.array[j] = f(real_i + j);
      out_cast[i] = wide.scalar;
    }

    const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) out[remain_i] = f(remain_i);
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks, typename SmemFinalType,
          typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices, const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx,
    int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal, int stride1, int rowStart, int topK) {
  
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }
  __syncthreads();

  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      distributeToBins(logits[idx * stride1], idx);
    }
  }
  __syncthreads();

  int lastValue = smemFoundTopKValues[0];
  constexpr int kItemsPerThread = kNumBins / kNumThreadsPerBlock;
  int binData[kItemsPerThread];

#pragma unroll
  for (int i = 0; i < kItemsPerThread; ++i) {
    binData[i] = smemFinal.histo.data[threadIdx.x * kItemsPerThread + i];
  }

  int prefixSums[kItemsPerThread];
  int totalSum{0};
#ifndef USE_ROCM
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
#else
  using Scan = hipcub::BlockScan<int, kNumThreadsPerBlock>;
#endif
  Scan(smemFinal.histo.scan).ExclusiveSum(binData, prefixSums, totalSum);

#pragma unroll
  for (int i = 0; i < kItemsPerThread; ++i) {
    prefixSums[i] += lastValue;
    smemFinal.histo.data[threadIdx.x * kItemsPerThread + i] = prefixSums[i];
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = threadIdx.x * kItemsPerThread + i;
    int pSum = prefixSums[i];
    if (pSum < topK) {
      int nextPrefixSum = (idx == kNumBins - 1) ? (totalSum + lastValue) : smemFinal.histo.data[idx + 1];
      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - pSum;
      }
    }
  }
  __syncthreads();

  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      processBins(logits[idx * stride1], idx);
    }
  }
  __syncthreads();

  return smemFinalBinSize[0] > kNumFinalItems;
}

template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false,
          int kMaxTopK = 2048,
          typename TransformFunc>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK,
                                     TransformFunc transform_func,
                                     int* outRawIndices = nullptr) {
  static constexpr int kNumFinalItems = kMaxTopK + 512;
  static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
  
#ifndef USE_ROCM
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
#else
  using FinalSort = hipcub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
  using Scan = hipcub::BlockScan<int, kNumThreadsPerBlock>;
#endif

  using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  
  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  extern __shared__ int32_t smemOutput[];

  __shared__ int smemThresholdBinIdx[1];
  __shared__ int smemFinalDstIdx[1];
  __shared__ int smemFinalBinSize[1];
  __shared__ int smemFoundTopKValues[1];
  // The result of each histogram pass is a block-wide decision. Keep it in
  // shared memory and publish it from lane 0 so every lane takes exactly the
  // same path around the barriers in processHistogramStep().
  __shared__ int smemContinueToNextStep[1];

  int rowLen = rowEnd - rowStart;

  // Shortcut if length <= TopK
  if (rowLen <= topK) {
    vectorized_write<int, int>(threadIdx.x, kNumThreadsPerBlock, outIndices, rowLen, [&](int rowIt) {
      int raw_idx = multipleBlocksPerRow ? rowIt + rowStart : rowIt;
      if (outRawIndices != nullptr) {
        outRawIndices[rowIt] = raw_idx;
      }
      return transform_func(raw_idx);
    });

    if constexpr (multipleBlocksPerRow) {
      vectorized_write<float, int>(threadIdx.x, kNumThreadsPerBlock, outLogits, rowLen, [&](int rowIt) {
        return logits[rowIt + rowStart];
      });
    }

    int fillLen = topK - rowLen;
    if (fillLen > 0) {
      vectorized_write<int, int>(threadIdx.x, kNumThreadsPerBlock, outIndices + rowLen, fillLen, [&](int /*i*/) {
        return -1;
      });
      if (outRawIndices != nullptr) {
        vectorized_write<int, int>(threadIdx.x, kNumThreadsPerBlock, outRawIndices + rowLen, fillLen, [&](int /*i*/) {
          return -1;
        });
      }
      if constexpr (multipleBlocksPerRow) {
        vectorized_write<float, int>(threadIdx.x, kNumThreadsPerBlock, outLogits + rowLen, fillLen, [&](int /*i*/) {
          return -FLT_MAX;
        });
      }
    }
    return;
  }

  // The final transform reads every entry in smemOutput. Keep unwritten slots
  // deterministic so an unexpected ranking gap cannot turn stale shared-memory
  // contents into an arbitrary global-memory address.
  vectorized_write<int, int>(
      threadIdx.x, kNumThreadsPerBlock, smemOutput, topK, [&](int /*i*/) {
        return -1;
      });

  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
      indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  // Do not branch on the per-lane return value. Although the pass ends with
  // a barrier, relying on every lane observing the same shared scalar has
  // caused divergent replay paths on HIP. A single lane publishes the
  // decision, followed by a block barrier, making the control flow explicit.
  if (threadIdx.x == 0) {
    smemContinueToNextStep[0] =
        smemFinalBinSize[0] > kNumFinalItems ? 1 : 0;
  }
  __syncthreads();
  bool continueToNextStep = smemContinueToNextStep[0] != 0;

  if (continueToNextStep) {
    // Step 0 uses FP16 bins. Step 1 is a full FP32 rescan, so discard the
    // coarse writes; otherwise high-score entries can be selected twice.
    if (threadIdx.x == 0) {
      smemFoundTopKValues[0] = 0;
      smemFinalDstIdx[0] = 0;
    }
    __syncthreads();

    processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }
  if (threadIdx.x == 0) {
    smemContinueToNextStep[0] =
        continueToNextStep && smemFinalBinSize[0] > kNumFinalItems ? 1 : 0;
  }
  __syncthreads();
  continueToNextStep = smemContinueToNextStep[0] != 0;

  if (continueToNextStep) {
    processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }
  if (threadIdx.x == 0) {
    smemContinueToNextStep[0] =
        continueToNextStep && smemFinalBinSize[0] > kNumFinalItems ? 1 : 0;
  }
  __syncthreads();
  continueToNextStep = smemContinueToNextStep[0] != 0;

  if (continueToNextStep) {
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (!continueToNextStep) {
    if constexpr (useRadixSort) {
      float finalLogits[kNumFinalItemsPerThread];
      int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) finalLogits[ii] = -FLT_MAX;

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (srcIdx < smemFinalDstIdx[0]) {
          finalLogits[ii] = smemFinal.items.logits[srcIdx];
          finalIndices[ii] = smemFinal.items.indices[srcIdx];
        }
      }
      __syncthreads();

      FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

      int baseIdx = smemFoundTopKValues[0];
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;
        if (dstIdx < topK) {
          smemOutput[dstIdx] = finalIndices[ii];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
          }
        }
      }
    } else {
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        auto logitKey = convert_to_uint32(logit);
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          auto otherLogitKey = convert_to_uint32(otherLogit);
          // Compare total-order integer keys instead of floats. Floating-point
          // comparisons do not order NaNs, which can assign several candidates
          // the same outIndex and leave holes in smemOutput.
          if (logitKey > otherLogitKey || (logitKey == otherLogitKey && i > j)) outIndex++;
        }
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
          }
        }
      }
    }
    __syncthreads();
  }

  vectorized_write<int, int>(threadIdx.x, kNumThreadsPerBlock, outIndices, topK, [&](int i) {
    if constexpr (multipleBlocksPerRow) {
      return smemOutput[i]; 
    } else {
      int raw_idx = (stride1 == 1) ? smemOutput[i] : smemOutput[i] - rowStart;
      if (raw_idx < 0 || raw_idx >= rowLen) {
        if (outRawIndices != nullptr) {
          outRawIndices[i] = -1;
        }
        return -1;
      }
      if (outRawIndices != nullptr) {
        outRawIndices[i] = raw_idx;
      }
      return transform_func(raw_idx); 
    }
  });

  if constexpr (multipleBlocksPerRow) {
    vectorized_write<float, int>(threadIdx.x, kNumThreadsPerBlock, outLogits, topK, [&](int i) {
      return reinterpret_cast<float*>(smemOutput + topK)[i];
    });
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
__global__ __launch_bounds__(kNumThreadsPerBlock) 
void fused_topk_kernel(FastTopKParams params, int topK) {
  int bid = blockIdx.x;
  int length = params.lengths[bid];
  int rowStart = params.row_starts ? params.row_starts[bid] : 0;
  int rowEnd = rowStart + length;
  int* outIndices = params.indices + bid * topK;
  const float* logits = params.input + bid * params.input_stride;

  auto transform_func = [&](int raw_idx) { return raw_idx; };
  topKPerRowJob<kNumThreadsPerBlock, 2048, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, 1, topK, transform_func);
}

template <int kNumThreadsPerBlock, bool useRadixSort>
__global__ __launch_bounds__(kNumThreadsPerBlock) 
void fused_topk_transform_decode_kernel(
    FastTopKParams params, int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table, const int64_t src_stride, int topK) {
  int bid = blockIdx.x;
  int length = params.lengths[bid];
  int rowStart = params.row_starts ? params.row_starts[bid] : 0;
  int rowEnd = rowStart + length;
  int* outIndices = dst_page_table + bid * topK;
  const float* logits = params.input + bid * params.input_stride;
  const int32_t* src_page_entry = src_page_table + bid * src_stride;

  auto transform_func = [&](int raw_idx) { return src_page_entry[raw_idx]; };
  topKPerRowJob<kNumThreadsPerBlock, 2048, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, 1, topK, transform_func);
}

template <int kNumThreadsPerBlock, bool useRadixSort>
__global__ __launch_bounds__(kNumThreadsPerBlock) 
void fused_topk_transform_prefill_kernel(
    FastTopKParams params, int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table, const int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q, const int64_t prefill_bs, int topK) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int length = params.lengths[bid];
  int rowStart = params.row_starts ? params.row_starts[bid] : 0;
  int rowEnd = rowStart + length;
  int* outIndices = dst_page_table + bid * topK;
  const float* logits = params.input + bid * params.input_stride;

  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kNumThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kNumThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const int32_t* src_page_entry = s_src_page_entry;

  auto transform_func = [&](int raw_idx) { return src_page_entry[raw_idx]; };
  topKPerRowJob<kNumThreadsPerBlock, 2048, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, 1, topK, transform_func);
}

template <int kNumThreadsPerBlock, bool useRadixSort>
__global__ __launch_bounds__(kNumThreadsPerBlock) 
void fused_topk_transform_ragged_kernel(
    FastTopKParams params, int32_t* __restrict__ topk_indices_ragged,
    const int32_t* __restrict__ topk_indices_offset, int topK) {
  int bid = blockIdx.x;
  int length = params.lengths[bid];
  int rowStart = params.row_starts ? params.row_starts[bid] : 0;
  int rowEnd = rowStart + length;
  int* outIndices = topk_indices_ragged + bid * topK;
  const float* logits = params.input + bid * params.input_stride;
  int offset = topk_indices_offset[bid];

  auto transform_func = [&](int raw_idx) { return raw_idx + offset; };
  topKPerRowJob<kNumThreadsPerBlock, 2048, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, 1, topK, transform_func);
}

constexpr int kTopK2048 = 2048;

#ifdef USE_ROCM
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kRadixSelect8BitSmemBytes =
    static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kRadixSelect8BitSmemBytes = 48 * 1024;
#endif
#else
constexpr size_t kRadixSelect8BitSmemBytes =
    8 * 1024 * sizeof(uint32_t);
#endif

template <int K, bool PhysicalPages = false>
void launch_adaptive_topk(
    const at::Tensor& score, const at::Tensor& lengths, at::Tensor& output,
    const at::Tensor* pages, const at::Tensor* cu, const at::Tensor* starts,
    uint32_t page_bits = 0, const at::Tensor* raw = nullptr,
    const at::Tensor* offsets = nullptr) {
#ifdef USE_ROCM
  TORCH_CHECK(
      score.size(0) <= std::numeric_limits<int32_t>::max() / K &&
      score.size(1) <= std::numeric_limits<int32_t>::max() &&
      (!pages || pages->size(1) <= std::numeric_limits<int32_t>::max()) &&
      page_bits < 31,
      "Adaptive TopK dimensions/page size exceed int32 indexing");
  auto check_metadata = [&](const at::Tensor* tensor) {
    if (!tensor) return;
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt && tensor->device() == score.device(),
        "Adaptive TopK metadata/output must be int32 on the score device");
  };
  for (const auto* tensor : {&lengths, static_cast<const at::Tensor*>(&output),
                             pages, cu, starts, raw, offsets}) check_metadata(tensor);
  TORCH_CHECK(!starts || starts->is_contiguous(), "row_starts must be contiguous");
  TORCH_CHECK(!offsets || offsets->is_contiguous(), "offsets must be contiguous");
  TORCH_CHECK(!raw || raw->data_ptr() != output.data_ptr(),
              "raw and mapped TopK outputs must not alias");
  const c10::cuda::CUDAGuard device_guard(score.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->warpSize == 64,
              "Adaptive TopK requires a wave64 HIP device");
  if (score.size(0) == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  adaptive_topk_kernel<K, PhysicalPages>
      <<<static_cast<uint32_t>(score.size(0)), adaptive_topk_detail::Threads, 0, stream>>>(
          score.data_ptr<float>(), score.stride(0), static_cast<int>(score.size(1)),
          lengths.data_ptr<int32_t>(), starts ? starts->data_ptr<int32_t>() : nullptr,
          pages ? pages->data_ptr<int32_t>() : nullptr, pages ? pages->stride(0) : 0,
          pages ? static_cast<int>(pages->size(1)) : 0,
          cu ? cu->data_ptr<int32_t>() : nullptr,
          static_cast<int>(pages ? pages->size(0) : score.size(0)),
          raw ? raw->data_ptr<int32_t>() : nullptr, output.data_ptr<int32_t>(),
          nullptr, true, page_bits, offsets ? offsets->data_ptr<int32_t>() : nullptr);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "Adaptive TopK failed: ",
              ::cudaGetErrorString(result));
#else
  TORCH_CHECK(false, "Adaptive TopK requires a HIP build");
#endif
}

int get_lightop_topk_backend() {
  static const int backend = [] {
    const char* value = std::getenv("SGL_USE_LIGHTOP_TOPK_BACKAND");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
      return 0;
    }
    if (std::strcmp(value, "1") == 0) {
      return 1;
    }
    if (std::strcmp(value, "2") == 0) {
      return 2;
    }
    TORCH_CHECK(
        false,
        "SGL_USE_LIGHTOP_TOPK_BACKAND must be 0, 1, or 2, but got ",
        value);
    return 0;
  }();
  return backend;
}

bool use_adaptive_topk() {
#ifdef USE_ROCM
  return get_lightop_topk_backend() == 0;
#else
  // The adaptive kernel is HIP-only; preserve CUDA's existing backend 0.
  return false;
#endif
}

__device__ void radix_select_8bit_naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < kTopK2048; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

__device__ void radix_select_8bit_naive_topk_transform_ragged(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ topk_indices_ragged,
    int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < kTopK2048; i += kThreadsPerBlock) {
    topk_indices_ragged[i] =
        (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto radix_select_8bit_convert_to_uint8(float x)
    -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000)
      ? static_cast<uint16_t>(~bits)
      : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto radix_select_8bit_convert_to_uint32(float x)
    -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int kTopK>
__device__ void radix_select_8bit_topk(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length) {
  static_assert(kTopK > 0 && kTopK <= kTopK2048);
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE =
      kRadixSelect8BitSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = radix_select_8bit_convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk &&
      s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(
          radix_select_8bit_convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(
          radix_select_8bit_convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = radix_select_8bit_convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
        ? _raw_num_input
        : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk &&
        s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin =
            (radix_select_8bit_convert_to_uint32(input[idx + row_start]) >>
             offset) &
            0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin =
            (radix_select_8bit_convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = radix_select_8bit_convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_decode_kernel_radix_select_8bit(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * kTopK2048;
  const auto score = input + bid * input_stride;
  if (length <= kTopK2048) {
    return radix_select_8bit_naive_topk_transform(
        score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[kTopK2048];
    radix_select_8bit_topk<kTopK2048>(
        score, s_indices, row_start, length);
    static_assert(kTopK2048 % kThreadsPerBlock == 0);
    static_assert(kTopK2048 / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_prefill_kernel_radix_select_8bit(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q,
    const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * kTopK2048;
  const auto score = input + bid * input_stride;

  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= kTopK2048) {
    return radix_select_8bit_naive_topk_transform(
        score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[kTopK2048];
    radix_select_8bit_topk<kTopK2048>(
        score, s_indices, row_start, length);
    static_assert(kTopK2048 % kThreadsPerBlock == 0);
    static_assert(kTopK2048 / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_ragged_kernel_radix_select_8bit(
    const FastTopKParams params,
    int32_t* __restrict__ topk_indices_ragged,
    const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * kTopK2048;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];
  if (length <= kTopK2048) {
    return radix_select_8bit_naive_topk_transform_ragged(
        score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[kTopK2048];
    radix_select_8bit_topk<kTopK2048>(
        score, s_indices, row_start, length);
    static_assert(kTopK2048 % kThreadsPerBlock == 0);
    static_assert(kTopK2048 / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

template <int kTopK>
__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_kernel_radix_select_8bit(
    const FastTopKParams params,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices,
    int32_t* __restrict__ raw_indices,
    int64_t page_table_stride,
    int64_t page_table_size,
    int64_t score_width,
    uint32_t page_bits) {
  static_assert(kTopK == 512 || kTopK == 1024);
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t page_capacity =
      page_table_size * (int64_t{1} << page_bits);
  const int64_t max_length =
      score_width < page_capacity ? score_width : page_capacity;
  const int raw_length = params.lengths[row];
  const int length = raw_length <= 0
      ? 0
      : static_cast<int>(
            raw_length < max_length ? raw_length : max_length);
  const float* scores = params.input + row * params.input_stride;
  const int32_t* page_table_row = page_table + row * page_table_stride;
  int32_t* out_page_indices = page_indices + row * kTopK;
  int32_t* out_raw_indices =
      raw_indices == nullptr ? nullptr : raw_indices + row * kTopK;
  const uint32_t page_mask = (uint32_t{1} << page_bits) - 1u;

  __shared__ int selected_indices[kTopK];
  if (length > kTopK) {
    radix_select_8bit_topk<kTopK>(scores, selected_indices, 0, length);
  }

  for (int output_idx = tid; output_idx < kTopK;
       output_idx += kThreadsPerBlock) {
    const int raw_idx = length <= kTopK
        ? (output_idx < length ? output_idx : -1)
        : selected_indices[output_idx];
    if (out_raw_indices != nullptr) {
      out_raw_indices[output_idx] = raw_idx;
    }
    if (raw_idx < 0 || raw_idx >= length) {
      out_page_indices[output_idx] = -1;
      continue;
    }

    const uint32_t page_idx =
        static_cast<uint32_t>(raw_idx) >> page_bits;
    if (page_idx >= static_cast<uint64_t>(page_table_size)) {
      out_page_indices[output_idx] = -1;
      continue;
    }
    const int32_t physical_page = page_table_row[page_idx];
    if (physical_page < 0) {
      out_page_indices[output_idx] = -1;
      continue;
    }
    out_page_indices[output_idx] =
        (static_cast<uint32_t>(physical_page) << page_bits) |
        (static_cast<uint32_t>(raw_idx) & page_mask);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_kernel_vgpr_select_512(
    const FastTopKParams params,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices,
    int32_t* __restrict__ raw_indices,
    int64_t page_table_stride,
    int64_t page_table_size,
    int64_t score_width,
    uint32_t page_bits) {
  constexpr int kTopK = 512;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t page_capacity =
      page_table_size * (int64_t{1} << page_bits);
  const int64_t max_length =
      score_width < page_capacity ? score_width : page_capacity;
  const int raw_length = params.lengths[row];
  const int length = raw_length <= 0
      ? 0
      : static_cast<int>(
            raw_length < max_length ? raw_length : max_length);
  const float* scores = params.input + row * params.input_stride;

  __shared__ int selected_indices[kTopK];
  extern __shared__ uint32_t candidate_workspace[];
  if (length > kTopK) {
    if (length <= 4 * kThreadsPerBlock) {
      select_kpool_topk_vgpr<kTopK, 4>(
          scores, selected_indices, 0, length, candidate_workspace);
    } else if (length <= 8 * kThreadsPerBlock) {
      select_kpool_topk_vgpr<kTopK, 8>(
          scores, selected_indices, 0, length, candidate_workspace);
    } else if (length <= 16 * kThreadsPerBlock) {
      select_kpool_topk_vgpr<kTopK, 16>(
          scores, selected_indices, 0, length, candidate_workspace);
    } else if (length <= kKPool4VgprMaxLength) {
      select_kpool_topk_vgpr<kTopK, kKPool4VgprItemsPerThread>(
          scores, selected_indices, 0, length, candidate_workspace);
    } else {
      auto identity = [] __device__(int raw_idx) { return raw_idx; };
      topKPerRowJob<
          kThreadsPerBlock,
          2048,
          false,
          false,
          false,
          512>(
          nullptr,
          scores,
          0,
          length,
          selected_indices,
          nullptr,
          1,
          kTopK,
          identity);
    }
  }
  __syncthreads();

  const int32_t* page_table_row = page_table + row * page_table_stride;
  int32_t* out_page_indices = page_indices + row * kTopK;
  int32_t* out_raw_indices =
      raw_indices == nullptr ? nullptr : raw_indices + row * kTopK;
  const uint32_t page_mask = (uint32_t{1} << page_bits) - 1u;

  for (int output_idx = tid; output_idx < kTopK;
       output_idx += kThreadsPerBlock) {
    const int raw_idx = length <= kTopK
        ? (output_idx < length ? output_idx : -1)
        : selected_indices[output_idx];
    if (out_raw_indices != nullptr) {
      out_raw_indices[output_idx] = raw_idx;
    }
    if (raw_idx < 0 || raw_idx >= length) {
      out_page_indices[output_idx] = -1;
      continue;
    }

    const uint32_t page_idx =
        static_cast<uint32_t>(raw_idx) >> page_bits;
    if (page_idx >= static_cast<uint64_t>(page_table_size)) {
      out_page_indices[output_idx] = -1;
      continue;
    }
    const int32_t physical_page = page_table_row[page_idx];
    if (physical_page < 0) {
      out_page_indices[output_idx] = -1;
      continue;
    }
    out_page_indices[output_idx] =
        (static_cast<uint32_t>(physical_page) << page_bits) |
        (static_cast<uint32_t>(raw_idx) & page_mask);
  }
}

constexpr int kSparseMaskTile = 16;

__device__ __forceinline__ uint32_t sparse_mask_valid_bits(
    int valid_count) {
  return valid_count >= kSparseMaskTile
      ? 0xffffu
      : ((uint32_t{1} << valid_count) - 1u);
}

__device__ void sparse_mask_exact_topk(
    const float* __restrict__ values,
    const int16_t* __restrict__ nonzero_mask,
    int length,
    int* __restrict__ selected_indices) {
  constexpr int kRadix = 256;
  constexpr uint32_t kZeroKey = 0x80000000u;

  alignas(128) __shared__ int histogram_buf[2][kRadix + 1];
  alignas(128) __shared__ int remaining;
  alignas(128) __shared__ int threshold_digit;
  alignas(128) __shared__ int higher_count;
  alignas(128) __shared__ uint32_t prefix;
  alignas(128) __shared__ uint32_t prefix_mask;
  // Use a deterministic per-thread prefix assignment for the final emit.
  // Atomically reserving output slots lets equal-score tokens race across
  // waves, so CUDA Graph replay can select a different subset of a large
  // zero tie.  These scans preserve the exact threshold while making both
  // the selected tie subset and output layout replay-stable.
  alignas(128) __shared__ int above_prefix[kThreadsPerBlock + 1];
  alignas(128) __shared__ int tie_prefix[kThreadsPerBlock + 1];
  alignas(128) __shared__ int sparse_fast_path;
  alignas(128) __shared__ int sparse_fast_positive_count;

  const int tid = threadIdx.x;

  // Production MQA logits are overwhelmingly +0.  If every strictly
  // positive value fits in TopK and +0 supplies the remaining slots, the
  // exact threshold is known without four full radix passes.  Negative values
  // cannot enter the result in this case.  All other distributions continue
  // through the generic exact selector below.
  const int fast_num_mask_tiles =
      (length + kSparseMaskTile - 1) / kSparseMaskTile;
  int local_positive = 0;
  int local_zero = 0;
  for (int tile = tid; tile < fast_num_mask_tiles;
       tile += kThreadsPerBlock) {
    const int tile_start = tile * kSparseMaskTile;
    const int remaining_in_row = length - tile_start;
    const int valid_count = remaining_in_row < kSparseMaskTile
        ? remaining_in_row
        : kSparseMaskTile;
    const uint32_t valid_bits = sparse_mask_valid_bits(valid_count);
    uint32_t stored_bits =
        static_cast<uint16_t>(nonzero_mask[tile]) & valid_bits;
    local_zero += valid_count - __popc(stored_bits);
    while (stored_bits != 0u) {
      const int bit = __ffs(stored_bits) - 1;
      stored_bits &= stored_bits - 1u;
      const uint32_t key = radix_select_8bit_convert_to_uint32(
          values[tile_start + bit]);
      local_positive += key > kZeroKey;
    }
  }

  if (tid == 0) {
    above_prefix[0] = 0;
    tie_prefix[0] = 0;
  }
  above_prefix[tid + 1] = local_positive;
  tie_prefix[tid + 1] = local_zero;
  __syncthreads();

#pragma unroll 10
  for (int offset = 1; offset <= kThreadsPerBlock; offset <<= 1) {
    const int positive_value = above_prefix[tid + 1];
    const int zero_value = tie_prefix[tid + 1];
    const int positive_add =
        tid + 1 >= offset ? above_prefix[tid + 1 - offset] : 0;
    const int zero_add =
        tid + 1 >= offset ? tie_prefix[tid + 1 - offset] : 0;
    __syncthreads();
    above_prefix[tid + 1] = positive_value + positive_add;
    tie_prefix[tid + 1] = zero_value + zero_add;
    __syncthreads();
  }

  if (tid == 0) {
    const int positive_count = above_prefix[kThreadsPerBlock];
    const int zero_count = tie_prefix[kThreadsPerBlock];
    sparse_fast_positive_count = positive_count;
    sparse_fast_path =
        positive_count <= kTopK2048 &&
        positive_count + zero_count >= kTopK2048;
  }
  __syncthreads();

  if (sparse_fast_path != 0) {
    int positive_pos = above_prefix[tid];
    int zero_pos = tie_prefix[tid];
    const int zero_needed = kTopK2048 - sparse_fast_positive_count;
    for (int tile = tid; tile < fast_num_mask_tiles;
         tile += kThreadsPerBlock) {
      const int tile_start = tile * kSparseMaskTile;
      const int remaining_in_row = length - tile_start;
      const int valid_count = remaining_in_row < kSparseMaskTile
          ? remaining_in_row
          : kSparseMaskTile;
      const uint32_t valid_bits = sparse_mask_valid_bits(valid_count);
      uint32_t stored_bits =
          static_cast<uint16_t>(nonzero_mask[tile]) & valid_bits;
      uint32_t zero_bits = valid_bits & ~stored_bits;

      while (stored_bits != 0u) {
        const int bit = __ffs(stored_bits) - 1;
        stored_bits &= stored_bits - 1u;
        const int raw_index = tile_start + bit;
        const uint32_t key = radix_select_8bit_convert_to_uint32(
            values[raw_index]);
        if (key > kZeroKey) {
          selected_indices[positive_pos++] = raw_index;
        }
      }
      while (zero_bits != 0u) {
        const int bit = __ffs(zero_bits) - 1;
        zero_bits &= zero_bits - 1u;
        if (zero_pos < zero_needed) {
          selected_indices[
              sparse_fast_positive_count + zero_pos] = tile_start + bit;
        }
        ++zero_pos;
      }
    }
    __syncthreads();
    return;
  }

  if (tid == 0) {
    remaining = kTopK2048;
    prefix = 0u;
    prefix_mask = 0u;
  }
  __syncthreads();

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    auto& histogram = histogram_buf[0];
    if (tid < kRadix + 1) histogram[tid] = 0;
    __syncthreads();

    const int shift = 24 - round * 8;
    const uint32_t current_prefix = prefix;
    const uint32_t current_prefix_mask = prefix_mask;
    const int num_mask_tiles =
        (length + kSparseMaskTile - 1) / kSparseMaskTile;

    for (int tile = tid; tile < num_mask_tiles;
         tile += kThreadsPerBlock) {
      const int tile_start = tile * kSparseMaskTile;
      const int remaining_in_row = length - tile_start;
      const int valid_count = remaining_in_row < kSparseMaskTile
          ? remaining_in_row
          : kSparseMaskTile;
      const uint32_t valid_bits = sparse_mask_valid_bits(valid_count);
      uint32_t stored_bits =
          static_cast<uint16_t>(nonzero_mask[tile]) & valid_bits;
      const int implicit_zero_count =
          valid_count - __popc(stored_bits);

      if (implicit_zero_count > 0 &&
          (kZeroKey & current_prefix_mask) == current_prefix) {
        atomicAdd(
            &histogram[(kZeroKey >> shift) & 0xffu],
            implicit_zero_count);
      }

      while (stored_bits != 0u) {
        const int bit = __ffs(stored_bits) - 1;
        stored_bits &= stored_bits - 1u;
        const uint32_t key = radix_select_8bit_convert_to_uint32(
            values[tile_start + bit]);
        if ((key & current_prefix_mask) == current_prefix) {
          atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
        }
      }
    }
    __syncthreads();

#pragma unroll 8
    for (int step = 0; step < 8; ++step) {
      if (tid < kRadix) {
        const int offset = 1 << step;
        const int src = step & 1;
        int value = histogram_buf[src][tid];
        if (tid + offset < kRadix) {
          value += histogram_buf[src][tid + offset];
        }
        histogram_buf[src ^ 1][tid] = value;
      }
      __syncthreads();
    }

    if (tid < kRadix && histogram_buf[0][tid] >= remaining &&
        histogram_buf[0][tid + 1] < remaining) {
      threshold_digit = tid;
      higher_count = histogram_buf[0][tid + 1];
    }
    __syncthreads();

    if (tid == 0) {
      remaining -= higher_count;
      prefix |= static_cast<uint32_t>(threshold_digit) << shift;
      prefix_mask |= 0xffu << shift;
    }
    __syncthreads();
  }

  const uint32_t threshold_key = prefix;
  const int tie_base = kTopK2048 - remaining;
  const int num_mask_tiles =
      (length + kSparseMaskTile - 1) / kSparseMaskTile;

  int local_above = 0;
  int local_ties = 0;
  for (int tile = tid; tile < num_mask_tiles;
       tile += kThreadsPerBlock) {
    const int tile_start = tile * kSparseMaskTile;
    const int remaining_in_row = length - tile_start;
    const int valid_count = remaining_in_row < kSparseMaskTile
        ? remaining_in_row
        : kSparseMaskTile;
    const uint32_t valid_bits = sparse_mask_valid_bits(valid_count);
    uint32_t stored_bits =
        static_cast<uint16_t>(nonzero_mask[tile]) & valid_bits;
    const uint32_t zero_bits = valid_bits & ~stored_bits;
    if (kZeroKey > threshold_key) {
      local_above += __popc(zero_bits);
    } else if (kZeroKey == threshold_key) {
      local_ties += __popc(zero_bits);
    }

    while (stored_bits != 0u) {
      const int bit = __ffs(stored_bits) - 1;
      stored_bits &= stored_bits - 1u;
      const int raw_index = tile_start + bit;
      const uint32_t key =
          radix_select_8bit_convert_to_uint32(values[raw_index]);
      if (key > threshold_key) {
        ++local_above;
      } else if (key == threshold_key) {
        ++local_ties;
      }
    }
  }

  if (tid == 0) {
    above_prefix[0] = 0;
    tie_prefix[0] = 0;
  }
  above_prefix[tid + 1] = local_above;
  tie_prefix[tid + 1] = local_ties;
  __syncthreads();

#pragma unroll 10
  for (int offset = 1; offset <= kThreadsPerBlock; offset <<= 1) {
    const int above_value = above_prefix[tid + 1];
    const int tie_value = tie_prefix[tid + 1];
    const int above_add =
        tid + 1 >= offset ? above_prefix[tid + 1 - offset] : 0;
    const int tie_add =
        tid + 1 >= offset ? tie_prefix[tid + 1 - offset] : 0;
    __syncthreads();
    above_prefix[tid + 1] = above_value + above_add;
    tie_prefix[tid + 1] = tie_value + tie_add;
    __syncthreads();
  }

  int above_pos = above_prefix[tid];
  int tie_pos = tie_prefix[tid];
  for (int tile = tid; tile < num_mask_tiles;
       tile += kThreadsPerBlock) {
    const int tile_start = tile * kSparseMaskTile;
    const int remaining_in_row = length - tile_start;
    const int valid_count = remaining_in_row < kSparseMaskTile
        ? remaining_in_row
        : kSparseMaskTile;
    const uint32_t valid_bits = sparse_mask_valid_bits(valid_count);
    uint32_t stored_bits =
        static_cast<uint16_t>(nonzero_mask[tile]) & valid_bits;
    uint32_t zero_bits = valid_bits & ~stored_bits;

    if (kZeroKey > threshold_key) {
      while (zero_bits != 0u) {
        const int bit = __ffs(zero_bits) - 1;
        zero_bits &= zero_bits - 1u;
        if (above_pos < tie_base) {
          selected_indices[above_pos] = tile_start + bit;
        }
        ++above_pos;
      }
    } else if (kZeroKey == threshold_key) {
      while (zero_bits != 0u) {
        const int bit = __ffs(zero_bits) - 1;
        zero_bits &= zero_bits - 1u;
        if (tie_pos < remaining) {
          selected_indices[tie_base + tie_pos] = tile_start + bit;
        }
        ++tie_pos;
      }
    }

    while (stored_bits != 0u) {
      const int bit = __ffs(stored_bits) - 1;
      stored_bits &= stored_bits - 1u;
      const int raw_index = tile_start + bit;
      const uint32_t key =
          radix_select_8bit_convert_to_uint32(values[raw_index]);
      if (key > threshold_key) {
        if (above_pos < tie_base) {
          selected_indices[above_pos] = raw_index;
        }
        ++above_pos;
      } else if (key == threshold_key) {
        if (tie_pos < remaining) {
          selected_indices[tie_base + tie_pos] = raw_index;
        }
        ++tie_pos;
      }
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_decode_kernel_sparse_mask(
    const FastTopKParams params,
    const int16_t* __restrict__ nonzero_mask,
    int64_t mask_stride,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const int64_t row = blockIdx.x;
  const int length = lengths[row];
  const float* values = input + row * input_stride;
  const int16_t* mask = nonzero_mask + row * mask_stride;
  int32_t* dst = dst_page_table + row * kTopK2048;
  const int32_t* src = src_page_table + row * src_stride;

  if (length <= kTopK2048) {
    return radix_select_8bit_naive_topk_transform(
        values, length, dst, src);
  }

  __shared__ int selected_indices[kTopK2048];
  sparse_mask_exact_topk(values, mask, length, selected_indices);
  const int tid = threadIdx.x;
  dst[tid] = src[selected_indices[tid]];
  dst[tid + kThreadsPerBlock] =
      src[selected_indices[tid + kThreadsPerBlock]];
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_prefill_kernel_sparse_mask(
    const FastTopKParams params,
    const int16_t* __restrict__ nonzero_mask,
    int64_t mask_stride,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q,
    int64_t prefill_bs) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int length = lengths[row];
  const float* values = input + row * input_stride;
  const int16_t* mask = nonzero_mask + row * mask_stride;
  int32_t* dst = dst_page_table + row * kTopK2048;

  __shared__ const int32_t* src;
  if (prefill_bs <= kThreadsPerBlock) {
    if (tid < prefill_bs && row >= cu_seqlens_q[tid] &&
        row < cu_seqlens_q[tid + 1]) {
      src = src_page_table + tid * src_stride;
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (row >= cu_seqlens_q[i] && row < cu_seqlens_q[i + 1]) {
        src = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();

  if (length <= kTopK2048) {
    return radix_select_8bit_naive_topk_transform(
        values, length, dst, src);
  }

  __shared__ int selected_indices[kTopK2048];
  sparse_mask_exact_topk(values, mask, length, selected_indices);
  dst[tid] = src[selected_indices[tid]];
  dst[tid + kThreadsPerBlock] =
      src[selected_indices[tid + kThreadsPerBlock]];
}

template <int kTopK, int kFinalSortTopK, int kNumThreadsPerBlock, bool useRadixSort>
__global__ __launch_bounds__(kNumThreadsPerBlock)
void topk_transform_kernel_lightop(
    FastTopKParams params,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices,
    int32_t* __restrict__ raw_indices,
    int64_t page_table_stride,
    int64_t page_table_size,
    int64_t score_width,
    uint32_t page_bits) {
  const int bid = blockIdx.x;
  const int64_t page_capacity = page_table_size * (int64_t{1} << page_bits);
  const int64_t max_length = score_width < page_capacity ? score_width : page_capacity;
  const int raw_length = params.lengths[bid];
  const int length =
      raw_length <= 0 ? 0 : static_cast<int>(raw_length < max_length ? raw_length : max_length);
  int32_t* out_page_indices = page_indices + bid * kTopK;
  int32_t* out_raw_indices =
      raw_indices == nullptr ? nullptr : raw_indices + bid * kTopK;
  const float* logits = params.input + bid * params.input_stride;
  const int32_t* page_table_row = page_table + bid * page_table_stride;
  const uint32_t page_mask = (1u << page_bits) - 1u;

  auto transform_func = [&](int raw_idx) {
    const uint32_t page_idx = static_cast<uint32_t>(raw_idx) >> page_bits;
    if (page_idx >= static_cast<uint64_t>(page_table_size)) {
      return -1;
    }
    const int32_t physical_page = page_table_row[page_idx];
    if (physical_page < 0) {
      return -1;
    }
    const uint32_t transformed =
        (static_cast<uint32_t>(physical_page) << page_bits) |
        (static_cast<uint32_t>(raw_idx) & page_mask);
    return static_cast<int32_t>(transformed);
  };
  topKPerRowJob<
      kNumThreadsPerBlock, 2048, useRadixSort, false, false, kFinalSortTopK>(
      nullptr,
      logits,
      0,
      length,
      out_page_indices,
      nullptr,
      1,
      kTopK,
      transform_func,
      out_raw_indices);
}

constexpr int kKPoolTokenTopK = 2048;

__device__ __forceinline__ int32_t transform_kpool_token_index(
    int raw_idx,
    const int32_t* page_table,
    int64_t page_table_stride,
    int64_t page_table_rows,
    int64_t page_table_cols,
    int page_row,
    int32_t ragged_offset) {
  if (raw_idx < 0) return -1;
  if (page_table) {
    if (page_row < 0 || page_row >= page_table_rows ||
        raw_idx >= page_table_cols) return -1;
    return page_table[
        static_cast<int64_t>(page_row) * page_table_stride + raw_idx];
  }
  return static_cast<int32_t>(raw_idx + ragged_offset);
}

template <int kKPoolSize, bool kUseRadixRefine = false>
__global__ __launch_bounds__(kThreadsPerBlock)
void fused_kpool_topk_transform_kernel(
    FastTopKParams params,
    int32_t* __restrict__ output,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ page_table_row_index,
    int64_t output_stride,
    int64_t page_table_stride,
    int64_t page_table_rows,
    int64_t page_table_cols,
    int score_cols,
    int rows) {
  static_assert(kKPoolSize == 4 || kKPoolSize == 16);
  static_assert(!kUseRadixRefine || kKPoolSize == 4);
  constexpr int kKPoolGroupTopK = kKPoolTokenTopK / kKPoolSize;
  constexpr int kKPoolOutputCols = kKPoolTokenTopK + kKPoolSize - 1;
  const int bid = blockIdx.x;
  int32_t* out_row = output + static_cast<int64_t>(bid) * output_stride;

  if (bid >= rows) {
    vectorized_write<int32_t, int>(
        threadIdx.x,
        kThreadsPerBlock,
        out_row,
        kKPoolOutputCols,
        [] __device__(int) { return -1; });
    return;
  }

  int row_start = params.row_starts ? params.row_starts[bid] : 0;
  row_start = row_start < 0 ? 0 : row_start;
  const int max_length = score_cols > row_start ? score_cols - row_start : 0;
  int length = params.lengths[bid];
  length = length < 0 ? 0 : (length < max_length ? length : max_length);
  const int row_end = row_start + length;
  const float* logits =
      params.input + static_cast<int64_t>(bid) * params.input_stride;

  __shared__ int32_t selected_groups[kKPoolGroupTopK];
  extern __shared__ int32_t kpool_candidate_workspace[];
  if (length <= kKPoolGroupTopK) {
    vectorized_write<int32_t, int>(
        threadIdx.x,
        kThreadsPerBlock,
        selected_groups,
        kKPoolGroupTopK,
        [length] __device__(int i) { return i < length ? i : -1; });
  } else {
    if constexpr (kUseRadixRefine) {
      if (length <= kKPool4VgprMaxLength) {
        select_kpool_topk_vgpr<kKPoolGroupTopK>(
            logits,
            selected_groups,
            row_start,
            length,
            reinterpret_cast<uint32_t*>(kpool_candidate_workspace));
      } else {
        auto identity = [] __device__(int raw_idx) { return raw_idx; };
        topKPerRowJob<
            kThreadsPerBlock,
            2048,
            false,
            false,
            false,
            512>(
            nullptr,
            logits,
            row_start,
            row_end,
            selected_groups,
            nullptr,
            1,
            kKPoolGroupTopK,
            identity);
      }
    } else {
      auto identity = [] __device__(int raw_idx) { return raw_idx; };
      topKPerRowJob<
          kThreadsPerBlock,
          2048,
          false,
          false,
          false,
          512>(
          nullptr,
          logits,
          row_start,
          row_end,
          selected_groups,
          nullptr,
          1,
          kKPoolGroupTopK,
          identity);
    }
  }
  __syncthreads();

  const int history_groups =
      length < kKPoolGroupTopK ? length : kKPoolGroupTopK;
  const int history_len = history_groups * kKPoolSize;
  const int tail_count = seq_lens[bid] % kKPoolSize;
  const int tail_start = length * kKPoolSize;
  const int page_row = page_table_row_index ? page_table_row_index[bid] : bid;
  const int32_t ragged_offset =
      topk_indices_offset ? topk_indices_offset[bid] : 0;
  const int32_t* selected_groups_ptr = selected_groups;

  vectorized_write<int32_t, int>(
      threadIdx.x,
      kThreadsPerBlock,
      out_row,
      kKPoolOutputCols,
      [=] __device__(int col) {
        int raw_idx = -1;
        if (col < history_len) {
          const int group_rank = col / kKPoolSize;
          const int group_offset = col % kKPoolSize;
          raw_idx =
              selected_groups_ptr[group_rank] * kKPoolSize + group_offset;
        } else {
          const int tail_offset = col - history_len;
          if (tail_offset >= 0 && tail_offset < tail_count) {
            raw_idx = tail_start + tail_offset;
          }
        }
        return transform_kpool_token_index(
            raw_idx,
            page_table,
            page_table_stride,
            page_table_rows,
            page_table_cols,
            page_row,
            ragged_offset);
      });
}
} // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) CHECK_CUDA(row_starts_opt.value());
  CHECK_CUDA(lengths);
  
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const int topK = indices.size(1);
  const size_t dynamic_smem = topK * sizeof(int);
  
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  bool use_radix = score.size(1) > 200000;

  if (use_radix) {
    setup_kernel_smem_once<fused_topk_kernel<kThreadsPerBlock, true>>(dynamic_smem);
    fused_topk_kernel<kThreadsPerBlock, true><<<grid, block, dynamic_smem, stream>>>(params, topK);
  } else {
    setup_kernel_smem_once<fused_topk_kernel<kThreadsPerBlock, false>>(dynamic_smem);
    fused_topk_kernel<kThreadsPerBlock, false><<<grid, block, dynamic_smem, stream>>>(params, topK);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) CHECK_CUDA(row_starts_opt.value());
  
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  const int topK = dst_page_table.size(1);
  const size_t dynamic_smem = topK * sizeof(int);

  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);

  if (use_adaptive_topk()) {
    TORCH_CHECK(topK == 2048, "Adaptive paged TopK requires K=2048");
    launch_adaptive_topk<2048>(score, lengths, dst_page_table, &src_page_table,
        &cu_seqlens_q, row_starts_opt ? &row_starts_opt.value() : nullptr);
    return;
  }

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  const int backend = get_lightop_topk_backend();

  TORCH_CHECK(
      backend != 2,
      "LightOp TopK backend 2 is only supported by the non-KPool fixed "
      "TopK512 interface");
  if (backend == 0) {
    TORCH_CHECK(
        topK == kTopK2048,
        "LightOp TopK backend ",
        backend,
        " requires topK=",
        kTopK2048,
        ", but got ",
        topK);
    if (is_decode) {
      setup_kernel_smem_once<topk_transform_decode_kernel_radix_select_8bit>(
          kRadixSelect8BitSmemBytes);
      topk_transform_decode_kernel_radix_select_8bit
          <<<grid, block, kRadixSelect8BitSmemBytes, stream>>>(
              params,
              dst_page_table.data_ptr<int32_t>(),
              src_page_table.data_ptr<int32_t>(),
              src_stride);
    } else {
      setup_kernel_smem_once<topk_transform_prefill_kernel_radix_select_8bit>(
          kRadixSelect8BitSmemBytes);
      topk_transform_prefill_kernel_radix_select_8bit
          <<<grid, block, kRadixSelect8BitSmemBytes, stream>>>(
              params,
              dst_page_table.data_ptr<int32_t>(),
              src_page_table.data_ptr<int32_t>(),
              src_stride,
              cu_seqlens_q.data_ptr<int32_t>(),
              prefill_bs);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess,
        "topk backend ",
        backend,
        " kernel failed:",
        ::cudaGetErrorString(result));
    return;
  }

  bool use_radix = score.size(1) > 2000000;

  if (is_decode) {
    if (use_radix) {
      setup_kernel_smem_once<fused_topk_transform_decode_kernel<kThreadsPerBlock, true>>(dynamic_smem);
      fused_topk_transform_decode_kernel<kThreadsPerBlock, true><<<grid, block, dynamic_smem, stream>>>(
          params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride, topK);
    } else {
      setup_kernel_smem_once<fused_topk_transform_decode_kernel<kThreadsPerBlock, false>>(dynamic_smem);
      fused_topk_transform_decode_kernel<kThreadsPerBlock, false><<<grid, block, dynamic_smem, stream>>>(
          params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride, topK);
    }
  } else {
    if (use_radix) {
      setup_kernel_smem_once<fused_topk_transform_prefill_kernel<kThreadsPerBlock, true>>(dynamic_smem);
      fused_topk_transform_prefill_kernel<kThreadsPerBlock, true><<<grid, block, dynamic_smem, stream>>>(
          params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride, cu_seqlens_q.data_ptr<int32_t>(), prefill_bs, topK);
    } else {
      setup_kernel_smem_once<fused_topk_transform_prefill_kernel<kThreadsPerBlock, false>>(dynamic_smem);
      fused_topk_transform_prefill_kernel<kThreadsPerBlock, false><<<grid, block, dynamic_smem, stream>>>(
          params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride, cu_seqlens_q.data_ptr<int32_t>(), prefill_bs, topK);
    }
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_sparse_mask_interface(
    const at::Tensor& score,
    const at::Tensor& nonzero_mask,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q) {
  CHECK_CUDA(score);
  CHECK_CUDA(nonzero_mask);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);

  const auto batch_size = score.size(0);
  TORCH_CHECK(
      score.scalar_type() == at::kFloat && score.dim() == 2 &&
          score.stride(1) == 1,
      "score must be row-contiguous float32 [rows, width]");
  TORCH_CHECK(
      nonzero_mask.scalar_type() == at::kShort &&
          nonzero_mask.dim() == 2 && nonzero_mask.stride(1) == 1 &&
          nonzero_mask.size(0) == batch_size &&
          nonzero_mask.size(1) ==
              (score.size(1) + kSparseMaskTile - 1) / kSparseMaskTile,
      "nonzero_mask must be row-contiguous int16 [rows, ceil(width/16)]");
  TORCH_CHECK(
      lengths.scalar_type() == at::kInt && lengths.dim() == 1 &&
          lengths.is_contiguous() && lengths.size(0) == batch_size,
      "lengths must be contiguous int32 [rows]");
  TORCH_CHECK(
      dst_page_table.scalar_type() == at::kInt &&
          dst_page_table.dim() == 2 && dst_page_table.is_contiguous() &&
          dst_page_table.size(0) == batch_size &&
          dst_page_table.size(1) == kTopK2048,
      "dst_page_table must be contiguous int32 [rows, 2048]");
  TORCH_CHECK(
      src_page_table.scalar_type() == at::kInt &&
          src_page_table.dim() == 2 && src_page_table.stride(1) == 1,
      "src_page_table must be row-contiguous int32");
  TORCH_CHECK(
      cu_seqlens_q.scalar_type() == at::kInt &&
          cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous(),
      "cu_seqlens_q must be contiguous int32");

  const int64_t prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(prefill_bs > 0 && prefill_bs <= batch_size);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(
      score.device() == nonzero_mask.device() &&
          score.device() == lengths.device() &&
          score.device() == dst_page_table.device() &&
          score.device() == src_page_table.device() &&
          score.device() == cu_seqlens_q.device(),
      "all sparse-mask TopK tensors must be on the same device");

  if (batch_size == 0) return;
  const auto params = get_params(score, lengths);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(batch_size)};
  const auto block = dim3{kThreadsPerBlock};
  const auto mask_stride = nonzero_mask.stride(0);
  const auto src_stride = src_page_table.stride(0);
  const bool is_decode = prefill_bs == batch_size;

  if (is_decode) {
    topk_transform_decode_kernel_sparse_mask<<<grid, block, 0, stream>>>(
        params,
        nonzero_mask.data_ptr<int16_t>(),
        mask_stride,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride);
  } else {
    topk_transform_prefill_kernel_sparse_mask<<<grid, block, 0, stream>>>(
        params,
        nonzero_mask.data_ptr<int16_t>(),
        mask_stride,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(
      result == cudaSuccess,
      "sparse-mask topk kernel failed:",
      ::cudaGetErrorString(result));
}

template <int kTopK>
void topk_transform_impl(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt,
    const char* op_name) {
  CHECK_CUDA(scores);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(page_tables);
  CHECK_CUDA(out_page_indices);
  if (out_raw_indices_opt.has_value()) CHECK_CUDA(out_raw_indices_opt.value());

  const auto batch_size = scores.size(0);
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(page_tables.scalar_type() == at::kInt, "page_tables must be int32");
  TORCH_CHECK(out_page_indices.scalar_type() == at::kInt, "out_page_indices must be int32");
  TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1, "scores must be a 2D row-contiguous tensor");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous(), "seq_lens must be contiguous and 1D");
  TORCH_CHECK(seq_lens.size(0) == batch_size, "seq_lens batch size mismatch");
  TORCH_CHECK(page_tables.dim() == 2 && page_tables.stride(1) == 1, "page_tables must be a 2D row-contiguous tensor");
  TORCH_CHECK(page_tables.size(0) == batch_size, "page_tables batch size mismatch");
  TORCH_CHECK(seq_lens.device() == scores.device(), "seq_lens must be on the same device as scores");
  TORCH_CHECK(page_tables.device() == scores.device(), "page_tables must be on the same device as scores");
  TORCH_CHECK(out_page_indices.device() == scores.device(), "out_page_indices must be on the same device as scores");
  TORCH_CHECK(
      out_page_indices.dim() == 2 && out_page_indices.is_contiguous() &&
          out_page_indices.size(0) == batch_size && out_page_indices.size(1) == kTopK,
      "out_page_indices must be contiguous with shape [batch_size, ", kTopK, "]");
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a positive power of two");

  int32_t* out_raw_indices = nullptr;
  if (out_raw_indices_opt.has_value()) {
    auto& raw_indices = out_raw_indices_opt.value();
    TORCH_CHECK(raw_indices.scalar_type() == at::kInt, "out_raw_indices must be int32");
    TORCH_CHECK(raw_indices.device() == scores.device(), "out_raw_indices must be on the same device as scores");
    TORCH_CHECK(
        raw_indices.dim() == 2 && raw_indices.is_contiguous() &&
            raw_indices.sizes() == out_page_indices.sizes(),
        "out_raw_indices must be contiguous with shape [batch_size, ", kTopK, "]");
    out_raw_indices = raw_indices.data_ptr<int32_t>();
  }

  if (use_adaptive_topk()) {
    const auto page_bits = static_cast<uint32_t>(
        __builtin_ctzll(static_cast<unsigned long long>(page_size)));
    launch_adaptive_topk<kTopK, true>(scores, seq_lens, out_page_indices,
        &page_tables, nullptr, nullptr, page_bits,
        out_raw_indices_opt ? &out_raw_indices_opt.value() : nullptr);
    return;
  }
  if (batch_size == 0) return;

  auto params = get_params(scores, seq_lens);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(batch_size)};
  const auto block = dim3{kThreadsPerBlock};
  const auto dynamic_smem = static_cast<size_t>(kTopK) * sizeof(int32_t);
  const auto page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
  const int backend = get_lightop_topk_backend();

  if (backend == 1) {
    setup_kernel_smem_once<topk_transform_kernel_radix_select_8bit<kTopK>>(
        kRadixSelect8BitSmemBytes);
    topk_transform_kernel_radix_select_8bit<kTopK>
        <<<grid, block, kRadixSelect8BitSmemBytes, stream>>>(
            params,
            page_tables.data_ptr<int32_t>(),
            out_page_indices.data_ptr<int32_t>(),
            out_raw_indices,
            page_tables.stride(0),
            page_tables.size(1),
            scores.size(1),
            page_bits);
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess,
        op_name,
        " backend ",
        backend,
        " kernel failed:",
        ::cudaGetErrorString(result));
    return;
  }

  if (backend == 2) {
    if constexpr (kTopK == 512) {
      setup_kernel_smem_once<topk_transform_kernel_vgpr_select_512>(
          kKPool4VgprSmemBytes);
      topk_transform_kernel_vgpr_select_512
          <<<grid, block, kKPool4VgprSmemBytes, stream>>>(
              params,
              page_tables.data_ptr<int32_t>(),
              out_page_indices.data_ptr<int32_t>(),
              out_raw_indices,
              page_tables.stride(0),
              page_tables.size(1),
              scores.size(1),
              page_bits);
      const auto result = cudaGetLastError();
      TORCH_CHECK(
          result == cudaSuccess,
          op_name,
          " backend 2 kernel failed:",
          ::cudaGetErrorString(result));
      return;
    } else {
      TORCH_CHECK(
          false,
          op_name,
          " backend 2 requires topK=512, but got ",
          kTopK);
    }
  }

  const bool use_radix = scores.size(1) > 2000000;
  constexpr int kFinalSortTopK = kTopK <= 512 ? kTopK : kTopK + 512;

  if (use_radix) {
    setup_kernel_smem_once<topk_transform_kernel_lightop<kTopK, kFinalSortTopK, kThreadsPerBlock, true>>(dynamic_smem);
    topk_transform_kernel_lightop<kTopK, kFinalSortTopK, kThreadsPerBlock, true><<<grid, block, dynamic_smem, stream>>>(
        params,
        page_tables.data_ptr<int32_t>(),
        out_page_indices.data_ptr<int32_t>(),
        out_raw_indices,
        page_tables.stride(0),
        page_tables.size(1),
        scores.size(1),
        page_bits);
  } else {
    setup_kernel_smem_once<topk_transform_kernel_lightop<kTopK, kFinalSortTopK, kThreadsPerBlock, false>>(dynamic_smem);
    topk_transform_kernel_lightop<kTopK, kFinalSortTopK, kThreadsPerBlock, false><<<grid, block, dynamic_smem, stream>>>(
        params,
        page_tables.data_ptr<int32_t>(),
        out_page_indices.data_ptr<int32_t>(),
        out_raw_indices,
        page_tables.stride(0),
        page_tables.size(1),
        scores.size(1),
        page_bits);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, op_name, " kernel failed:", ::cudaGetErrorString(result));
}

void topk_transform_512(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt) {
  topk_transform_impl<512>(
      scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices_opt, "topk_transform_512");
}

void topk_transform_1024(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt) {
  topk_transform_impl<1024>(
      scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices_opt, "topk_transform_1024");
}

void fast_kpool_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    int64_t pool_size,
    const at::Tensor& seq_lens,
    std::optional<at::Tensor> page_table_opt,
    std::optional<at::Tensor> topk_indices_offset_opt,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> page_table_row_index_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(output);
  CHECK_CUDA(seq_lens);
  if (page_table_opt.has_value()) CHECK_CUDA(page_table_opt.value());
  if (topk_indices_offset_opt.has_value()) {
    CHECK_CUDA(topk_indices_offset_opt.value());
  }
  if (row_starts_opt.has_value()) CHECK_CUDA(row_starts_opt.value());
  if (page_table_row_index_opt.has_value()) {
    CHECK_CUDA(page_table_row_index_opt.value());
  }

  const int64_t rows = score.size(0);
  TORCH_CHECK(
      pool_size == 4 || pool_size == 16,
      "kpool fused topk requires pool_size=4 or 16");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32");
  TORCH_CHECK(
      score.dim() == 2 && score.stride(1) == 1,
      "score must be a 2D row-contiguous tensor");
  TORCH_CHECK(
      score.size(1) <= std::numeric_limits<int>::max() / pool_size,
      "score is too wide for the kpool fused topk kernel");
  TORCH_CHECK(
      lengths.scalar_type() == at::kInt && lengths.dim() == 1 &&
          lengths.is_contiguous() && lengths.size(0) == rows,
      "lengths must be contiguous int32 with shape [score.size(0)]");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
          seq_lens.is_contiguous() && seq_lens.size(0) == rows,
      "seq_lens must be contiguous int32 with shape [score.size(0)]");
  TORCH_CHECK(
      output.scalar_type() == at::kInt && output.dim() == 2 &&
          output.is_contiguous() && output.size(0) >= rows &&
          output.size(1) == kKPoolTokenTopK + pool_size - 1,
      "output must be contiguous int32 with shape "
      "[out_rows, 2048 + pool_size - 1]");
  TORCH_CHECK(
      output.device() == score.device() && lengths.device() == score.device() &&
          seq_lens.device() == score.device(),
      "score, lengths, seq_lens, and output must be on the same device");
  TORCH_CHECK(
      !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
      "page_table and topk_indices_offset are mutually exclusive");

  const int32_t* page_table = nullptr;
  int64_t page_table_stride = 0;
  int64_t page_table_rows = 0;
  int64_t page_table_cols = 0;
  if (page_table_opt.has_value()) {
    const auto& tensor = page_table_opt.value();
    TORCH_CHECK(
        tensor.scalar_type() == at::kInt && tensor.dim() == 2 &&
            tensor.stride(1) == 1,
        "page_table must be row-contiguous int32 and 2D");
    TORCH_CHECK(tensor.device() == score.device(), "page_table device mismatch");
    if (!page_table_row_index_opt.has_value()) {
      TORCH_CHECK(
          tensor.size(0) >= rows,
          "page_table needs one row per score row without page_table_row_index");
    }
    page_table = tensor.data_ptr<int32_t>();
    page_table_stride = tensor.stride(0);
    page_table_rows = tensor.size(0);
    page_table_cols = tensor.size(1);
  }

  const int32_t* topk_indices_offset = nullptr;
  if (topk_indices_offset_opt.has_value()) {
    const auto& tensor = topk_indices_offset_opt.value();
    TORCH_CHECK(
        tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
            tensor.is_contiguous() && tensor.size(0) == rows,
        "topk_indices_offset must be contiguous int32 with shape [score.size(0)]");
    TORCH_CHECK(
        tensor.device() == score.device(),
        "topk_indices_offset device mismatch");
    topk_indices_offset = tensor.data_ptr<int32_t>();
  }

  const int32_t* page_table_row_index = nullptr;
  if (page_table_row_index_opt.has_value()) {
    const auto& tensor = page_table_row_index_opt.value();
    TORCH_CHECK(page_table != nullptr, "page_table_row_index requires page_table");
    TORCH_CHECK(
        tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
            tensor.is_contiguous() && tensor.size(0) == rows,
        "page_table_row_index must be contiguous int32 with shape [score.size(0)]");
    TORCH_CHECK(
        tensor.device() == score.device(),
        "page_table_row_index device mismatch");
    page_table_row_index = tensor.data_ptr<int32_t>();
  }

  if (row_starts_opt.has_value()) {
    const auto& tensor = row_starts_opt.value();
    TORCH_CHECK(
        tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
            tensor.is_contiguous() && tensor.size(0) == rows,
        "row_starts must be contiguous int32 with shape [score.size(0)]");
    TORCH_CHECK(tensor.device() == score.device(), "row_starts device mismatch");
  }

  const int64_t out_rows = output.size(0);
  if (out_rows == 0) return;

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(out_rows)};
  const auto block = dim3{kThreadsPerBlock};
  if (pool_size == 4) {
    setup_kernel_smem_once<fused_kpool_topk_transform_kernel<4, true>>(
        kKPool4VgprSmemBytes);
    fused_kpool_topk_transform_kernel<4, true>
        <<<grid, block, kKPool4VgprSmemBytes, stream>>>(
            params, output.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
            page_table, topk_indices_offset, page_table_row_index,
            output.stride(0), page_table_stride, page_table_rows,
            page_table_cols, static_cast<int>(score.size(1)),
            static_cast<int>(rows));
  } else {
    constexpr size_t dynamic_smem =
        (kKPoolTokenTopK / 16) * sizeof(int32_t);
    fused_kpool_topk_transform_kernel<16>
        <<<grid, block, dynamic_smem, stream>>>(
            params, output.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
            page_table, topk_indices_offset, page_table_row_index,
            output.stride(0), page_table_stride, page_table_rows,
            page_table_cols, static_cast<int>(score.size(1)),
            static_cast<int>(rows));
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(
      result == cudaSuccess,
      "fused kpool topk transform kernel failed:",
      ::cudaGetErrorString(result));
}
void topk_transform_ragged_v2_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    const at::Tensor& offsets,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  TORCH_CHECK(score.dim() == 2 && score.scalar_type() == at::kFloat,
              "ragged TopK scores must be float32 [rows, width]");
  TORCH_CHECK(score.stride(1) == 1 && score.stride(0) >= score.size(1),
              "ragged TopK scores must have contiguous, non-overlapping rows");
  // The selection core advances full-width scans in int32 thread-block steps.
  TORCH_CHECK(score.size(1) <= std::numeric_limits<int32_t>::max() - 1023,
              "ragged TopK width exceeds safe int32 scan indexing");
  const auto rows = score.size(0);
  auto check_vector = [&](const at::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == rows &&
                    tensor.scalar_type() == at::kInt && tensor.is_contiguous() &&
                    tensor.device() == score.device(),
                "ragged TopK metadata must be contiguous int32 [rows] "
                "on the score device");
    TORCH_CHECK(!output.is_alias_of(tensor),
                "ragged TopK output must not alias metadata");
  };
  TORCH_CHECK(output.dim() == 2 && output.size(0) == rows &&
                  output.scalar_type() == at::kInt && output.is_contiguous() &&
                  output.device() == score.device(),
              "ragged TopK output must be contiguous int32 [rows, topk] "
              "on the score device");
  TORCH_CHECK(!output.is_alias_of(score),
              "ragged TopK output must not alias scores");
  check_vector(lengths);
  check_vector(offsets);
  if (row_starts_opt) check_vector(row_starts_opt.value());
  const auto* starts = row_starts_opt ? &row_starts_opt.value() : nullptr;
  switch (output.size(1)) {
    case 512:
      return launch_adaptive_topk<512>(
          score, lengths, output, nullptr, nullptr, starts, 0, nullptr, &offsets);
    case 1024:
      return launch_adaptive_topk<1024>(
          score, lengths, output, nullptr, nullptr, starts, 0, nullptr, &offsets);
    case 2048:
      return launch_adaptive_topk<2048>(
          score, lengths, output, nullptr, nullptr, starts, 0, nullptr, &offsets);
    default:
      TORCH_CHECK(false, "ragged TopK requires topk=512, 1024, or 2048");
  }
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) CHECK_CUDA(row_starts_opt.value());

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  const int topK = topk_indices_ragged.size(1);
  const size_t dynamic_smem = topK * sizeof(int);

  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);
  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  if (use_adaptive_topk()) {
    TORCH_CHECK(topK == 2048, "Adaptive ragged TopK requires K=2048");
    launch_adaptive_topk<2048>(score, lengths, topk_indices_ragged,
        nullptr, nullptr, row_starts_opt ? &row_starts_opt.value() : nullptr,
        0, nullptr, &topk_indices_offset);
    return;
  }

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const int backend = get_lightop_topk_backend();

  TORCH_CHECK(
      backend != 2,
      "LightOp TopK backend 2 is only supported by the non-KPool fixed "
      "TopK512 interface");
  if (backend == 1) {
    TORCH_CHECK(
        topK == kTopK2048,
        "LightOp TopK backend ",
        backend,
        " requires topK=",
        kTopK2048,
        ", but got ",
        topK);
    setup_kernel_smem_once<topk_transform_ragged_kernel_radix_select_8bit>(
        kRadixSelect8BitSmemBytes);
    topk_transform_ragged_kernel_radix_select_8bit
        <<<grid, block, kRadixSelect8BitSmemBytes, stream>>>(
            params,
            topk_indices_ragged.data_ptr<int32_t>(),
            topk_indices_offset.data_ptr<int32_t>());
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess,
        "topk backend ",
        backend,
        " kernel failed:",
        ::cudaGetErrorString(result));
    return;
  }

  bool use_radix = score.size(1) > 2000000;

  if (use_radix) {
    setup_kernel_smem_once<fused_topk_transform_ragged_kernel<kThreadsPerBlock, true>>(dynamic_smem);
    fused_topk_transform_ragged_kernel<kThreadsPerBlock, true><<<grid, block, dynamic_smem, stream>>>(
        params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>(), topK);
  } else {
    setup_kernel_smem_once<fused_topk_transform_ragged_kernel<kThreadsPerBlock, false>>(dynamic_smem);
    fused_topk_transform_ragged_kernel<kThreadsPerBlock, false><<<grid, block, dynamic_smem, stream>>>(
        params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>(), topK);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

} // namespace native
} // namespace at
