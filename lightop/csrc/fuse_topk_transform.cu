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
#include <limits>
#include <optional>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #define WARP_SIZE 32
#else
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #include <hipcub/hipcub.hpp>
  #define WARP_SIZE 64
#endif

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

template <int K>
__device__ void select_kpool_topk_vgpr(
    const float* __restrict__ input,
    int* __restrict__ indices,
    int row_start,
    int length,
    uint32_t* __restrict__ candidate_workspace) {
  static_assert(K == 512);
  constexpr int kItems = kKPool4VgprItemsPerThread;
  constexpr int kSampleItems = 4;
  constexpr int kCompactItems = 2;
  constexpr int kSampleRank = K / 2;

  __shared__ int radix_histogram[4];
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
      keys[0], keys[5], keys[10], keys[15]};
  const uint32_t sample_threshold =
      kpool_radix_select_descending<kSampleItems>(
          sample_keys, kSampleRank, radix_histogram);

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
    kth_key = kpool_radix_select_descending<kCompactItems>(
        compact_keys, K, radix_histogram);
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

  bool continueToNextStep = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
      indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    continueToNextStep = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    continueToNextStep = processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

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

namespace at {  
namespace native {

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

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;

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

  if (batch_size == 0) return;

  auto params = get_params(scores, seq_lens);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(batch_size)};
  const auto block = dim3{kThreadsPerBlock};
  const auto dynamic_smem = static_cast<size_t>(kTopK) * sizeof(int32_t);
  const auto page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
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

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

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
