// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstring>
#include <type_traits>
#include <vector>

namespace at {
namespace native {
void dequantize_k_cache_paged_hcu_selected(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, int64_t, bool);
namespace {

constexpr int kWorkspaceThreads = 256;
constexpr int kScanRows = 1024;

__global__ void dsv4_mark_topk_rows(int32_t *, const int32_t *,
                                  const int32_t *, int64_t, int64_t, int64_t);

#if defined(__HIP_PLATFORM_AMD__)
constexpr int kBitmapTileWords = 4096;

__global__ void dsv4_clear_bitmap_gfx936(uint32_t *bitmap, int64_t words, int32_t *count) {
  const int64_t first = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (first + 3 < words)
    reinterpret_cast<uint4 *>(bitmap)[first / 4] = make_uint4(0, 0, 0, 0);
  else
    for (int j = 0; j < 4 && first + j < words; ++j) bitmap[first + j] = 0;
  if (count != nullptr && first == 0) *count = 0;
}

template <bool BuildQueue>
__global__ void dsv4_mark_bitmap_gfx936(
    uint32_t *bitmap, const int32_t *indices, const int32_t *lengths,
    int64_t stride, int64_t width, int32_t n, int32_t *selected_rows,
    int32_t *valid_count) {
  const int64_t query = blockIdx.x;
  const int64_t length = lengths[query];
  const int lane = threadIdx.x % 64;
  if constexpr (BuildQueue) {
    // A complete logical union makes all remaining references redundant.
    // Exit uniformly within each wave; the next kernel still waits for all
    // outstanding queue stores before consuming the count or the queue.
    int complete = lane == 0 ? __hip_atomic_load(valid_count, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == n : 0;
    if (__shfl(complete, 0, 64)) return;
  }
  // All lanes participate in shuffles, including a partial final wave.
  for (int64_t base = 0; base < width; base += blockDim.x) {
    const int64_t column = base + threadIdx.x;
    const uint32_t row = column < width && column < length ? static_cast<uint32_t>(indices[query * stride + column]) : 0xffffffffu;
    const bool valid = row < static_cast<uint32_t>(n);
    const uint32_t word = valid ? row >> 5 : 0xffffffffu;
    uint32_t mask = valid ? 1u << (row & 31) : 0;
    const uint32_t first_word = __shfl(word, lane & 32, 64);
    const bool compatible = !valid || ((row & 31) == (lane & 31) && word == first_word);
    const uint64_t selected = __ballot(valid);
    const bool coalesced = __ballot(!compatible) == 0;
    bool emit = true;
    if (coalesced) {
      // A verified lane-to-bit mapping needs one ballot, rather than a
      // segmented shuffle scan. Arbitrary indices use direct atomic OR.
      mask = static_cast<uint32_t>(selected >> (lane & 32));
      emit = (lane & 31) == 0;
    }
    uint32_t new_bits = 0;
    if (mask && emit) {
      const uint32_t old = __hip_atomic_load(bitmap + word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if ((old & mask) != mask) {
        if constexpr (BuildQueue) new_bits = mask & ~atomicOr(bitmap + word, mask);
        else atomicOr(bitmap + word, mask);
      }
    }
    if constexpr (BuildQueue) {
      if (coalesced) new_bits = __shfl(new_bits, lane & 32, 64);
      const bool won = valid && (coalesced ? (new_bits & (1u << (row & 31))) != 0 : new_bits != 0);
      const uint64_t winners = __ballot(won);
      const int32_t population = __popcll(winners);
      int32_t first = 0;
      if (lane == 0 && population) first = atomicAdd(valid_count, population);
      first = __shfl(first, 0, 64);
      if (won) {
        const int32_t preceding = __popcll(winners & ((1ull << lane) - 1));
        selected_rows[first + preceding] = static_cast<int32_t>(row);
      }
    }
  }
}

template <int Queries>
__global__ void dsv4_mark_bitmap_lds_gfx936(
    uint32_t *bitmap, const int32_t *indices, const int32_t *lengths,
    int64_t stride, int64_t width, int32_t n, int64_t queries,
    int32_t *selected_rows, int32_t *valid_count) {
  extern __shared__ uint32_t local_bits[];
  __shared__ int complete;
  const int words = (n + 31) / 32;
  const int lane = threadIdx.x % 64;
  if (threadIdx.x == 0)
    complete = __hip_atomic_load(valid_count, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) == n;
  for (int word = threadIdx.x; word < words; word += blockDim.x)
    local_bits[word] = 0;
  __syncthreads();
  if (complete) return;
  const int64_t first_query = static_cast<int64_t>(blockIdx.x) * Queries;
  for (int64_t query = first_query;
       query < first_query + Queries && query < queries; ++query) {
    const int32_t length = lengths[query];
    for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
      const uint32_t row = column < length
          ? static_cast<uint32_t>(indices[query * stride + column]) : 0xffffffffu;
      if (row < static_cast<uint32_t>(n))
        atomicOr(local_bits + (row >> 5), 1u << (row & 31));
    }
  }
  __syncthreads();
  // No grid barrier: each bit has exactly one global atomic winner. Queue
  // stores are complete before the following selected-dequant kernel starts.
  for (int first = 0; first < words; first += blockDim.x) {
    const int word = first + threadIdx.x;
    const uint32_t mask = word < words ? local_bits[word] : 0;
    uint32_t bits = 0;
    if (mask) {
      const uint32_t old = __hip_atomic_load(bitmap + word, __ATOMIC_RELAXED,
                                            __HIP_MEMORY_SCOPE_AGENT);
      if ((old & mask) != mask) bits = mask & ~atomicOr(bitmap + word, mask);
    }
    const int population = __popc(bits);
    int scan = population;
#pragma unroll
    for (int shift = 1; shift < 64; shift *= 2) {
      const int previous = __shfl_up(scan, shift, 64);
      if (lane >= shift) scan += previous;
    }
    const int total = __shfl(scan, 63, 64);
    int start = 0;
    if (lane == 0 && total) start = atomicAdd(valid_count, total);
    start = __shfl(start, 0, 64) + scan - population;
    while (bits) {
      const int bit = __ffs(bits) - 1;
      selected_rows[start++] = word * 32 + bit;
      bits &= bits - 1;
    }
  }
}

template <bool SingleTile>
__global__ __launch_bounds__(1024) void dsv4_scan_bitmap_gfx936(
    const uint32_t *bitmap, int32_t *prefix, int32_t *tile_counts,
    int32_t *valid_count, int32_t *dense_prefix, int64_t words) {
  __shared__ int32_t wave_counts[16], wave_last[16];
  const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
  const int waves = blockDim.x / 64;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * kBitmapTileWords + threadIdx.x * 4;
  uint32_t bits[4] = {};
  if (first + 3 < words) {
    const uint4 values = reinterpret_cast<const uint4 *>(bitmap)[first / 4];
    bits[0] = values.x; bits[1] = values.y; bits[2] = values.z; bits[3] = values.w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (first + j < words) bits[j] = bitmap[first + j];
  }
  int32_t populations[4], total = 0, last = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    populations[j] = __popc(bits[j]); total += populations[j];
    if (bits[j]) last = static_cast<int32_t>((first + j) * 32 + 32 - __clz(bits[j]));
  }
  int32_t scan = total, largest = last;
#pragma unroll
  for (int shift = 1; shift < 64; shift *= 2) {
    const int32_t previous = __shfl_up(scan, shift, 64);
    if (lane >= shift) scan += previous;
    const int32_t other = __shfl_down(largest, shift, 64);
    if (other > largest) largest = other;
  }
  if (lane == 63) wave_counts[wave] = scan;
  if (lane == 0) wave_last[wave] = largest;
  __syncthreads();
  int32_t preceding = 0;
#pragma unroll
  for (int w = 0; w < 16; ++w)
    if (w < wave) preceding += wave_counts[w];
  int32_t current = preceding + scan - total;
  int32_t values[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    values[j] = current; current += populations[j];
  }
  if (first + 3 < words)
    reinterpret_cast<int4 *>(prefix)[first / 4] = make_int4(values[0], values[1], values[2], values[3]);
  else
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (first + j < words) prefix[first + j] = values[j];
  if (threadIdx.x == blockDim.x - 1) {
    int32_t maximum = 0;
#pragma unroll
    for (int w = 0; w < 16; ++w)
      if (w < waves && wave_last[w] > maximum) maximum = wave_last[w];
    const int32_t count = preceding + scan;
    if constexpr (SingleTile) {
      *valid_count = count; *dense_prefix = maximum == count;
    } else {
      tile_counts[blockIdx.x * 2] = count;
      tile_counts[blockIdx.x * 2 + 1] = maximum;
    }
  }
}

__global__ void dsv4_scan_bitmap_counts_gfx936(
    const int32_t *counts, int32_t *offsets, int32_t *valid_count,
    int32_t *dense_prefix, int64_t tiles) {
  const int64_t begin = tiles * threadIdx.x / 64, end = tiles * (threadIdx.x + 1) / 64;
  int32_t total = 0, largest = 0;
  for (int64_t i = begin; i < end; ++i) {
    total += counts[i * 2];
    if (counts[i * 2 + 1] > largest) largest = counts[i * 2 + 1];
  }
  int32_t scan = total;
#pragma unroll
  for (int shift = 1; shift < 64; shift *= 2) {
    const int32_t previous = __shfl_up(scan, shift, 64);
    const int32_t other = __shfl_up(largest, shift, 64);
    if (threadIdx.x >= shift) { scan += previous; if (other > largest) largest = other; }
  }
  int32_t offset = scan - total;
  for (int64_t i = begin; i < end; ++i) { offsets[i] = offset; offset += counts[i * 2]; }
  if (threadIdx.x == 63) { *valid_count = scan; *dense_prefix = largest == scan; }
}

template <bool SingleTile>
__device__ __forceinline__ int32_t bitmap_rank_gfx936(
    uint32_t row, const uint32_t *bitmap, const int32_t *prefix,
    const int32_t *offsets) {
  const uint32_t word = row >> 5;
  int32_t rank = prefix[word] + __popc(bitmap[word] & ((1u << (row & 31)) - 1u));
  if constexpr (!SingleTile) rank += offsets[word / kBitmapTileWords];
  return rank;
}

template <typename index_t, bool Vectorized, bool SingleTile>
__global__ void dsv4_pack_remap_bitmap_gfx936(
    const index_t *full_ids, const uint32_t *bitmap, const int32_t *prefix,
    const int32_t *offsets, const int32_t *valid_count, const int32_t *dense_prefix,
    int32_t *packed, int32_t *output, const int32_t *indices, const int32_t *lengths,
    int64_t stride, int64_t width, int32_t n, int32_t capacity, int32_t swa_rows) {
  const int64_t packing_blocks = (static_cast<int64_t>(n) + blockDim.x - 1) / blockDim.x;
  const bool identity = *dense_prefix;
  if (blockIdx.x < packing_blocks) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t active = *valid_count;
    if (identity) {
      if (row < capacity) packed[row] = row < active ? static_cast<int32_t>(full_ids[row]) : -1;
    } else {
      if (row < capacity && row >= active) packed[row] = -1;
      if (row < n && (bitmap[row / 32] & (1u << (row % 32)))) {
        const int32_t rank = bitmap_rank_gfx936<SingleTile>(row, bitmap, prefix, offsets);
        if (rank >= 0 && rank < capacity) packed[rank] = static_cast<int32_t>(full_ids[row]);
      }
    }
    return;
  }
  const int64_t query = blockIdx.x - packing_blocks;
  const int64_t length = lengths[query];
  const int32_t *source = indices + query * stride;
  int32_t *destination = output + query * width;
  if constexpr (Vectorized) {
    for (int32_t v = threadIdx.x; v < width / 4; v += blockDim.x) {
      const int4 input = reinterpret_cast<const int4 *>(source)[v];
      const int32_t keys[4] = {input.x, input.y, input.z, input.w};
      int32_t result[4];
      if (identity && capacity == n) {
        const uint32_t upper = swa_rows < 0 ? 0x80000000u : static_cast<uint32_t>(n) + swa_rows;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          result[j] = v * 4 + j < length && static_cast<uint32_t>(keys[j]) < upper ? keys[j] : -1;
      } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int32_t row = keys[j]; int32_t value = -1;
        if (v * 4 + j < length) {
          if (static_cast<uint32_t>(row) < static_cast<uint32_t>(n)) {
            const int32_t rank = identity ? row : bitmap_rank_gfx936<SingleTile>(row, bitmap, prefix, offsets);
            if (rank >= 0 && rank < capacity) value = rank;
          } else if (row >= n && (swa_rows < 0 || row - n < swa_rows)) {
            // capacity <= n and row >= n imply 0 <= row-n+capacity <= row.
            value = row - n + capacity;
          }
        }
        result[j] = value;
      }
      }
      reinterpret_cast<int4 *>(destination)[v] = make_int4(result[0], result[1], result[2], result[3]);
    }
  } else {
    for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
      int32_t value = -1;
      if (column < length) {
        const int32_t row = source[column];
        if (static_cast<uint32_t>(row) < static_cast<uint32_t>(n)) {
          const int32_t rank = identity ? row : bitmap_rank_gfx936<SingleTile>(row, bitmap, prefix, offsets);
          if (rank >= 0 && rank < capacity) value = rank;
        } else if (row >= n && (swa_rows < 0 || row - n < swa_rows)) {
          value = row - n + capacity;
        }
      }
      destination[column] = value;
    }
  }
}

std::vector<torch::Tensor> pack_workspace_gfx936(
    const torch::Tensor &full_ids, const torch::Tensor &indices,
    const torch::Tensor &lengths, int64_t capacity, int64_t swa_rows) {
  const int64_t n = full_ids.numel(), queries = indices.size(0), width = indices.size(1);
  const int64_t words = (n + 31) / 32, tiles = (words + kBitmapTileWords - 1) / kBitmapTileWords;
  const bool single = tiles == 1;
  const auto options = full_ids.options().dtype(at::kInt);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto bitmap = at::empty({words}, options), prefix = at::empty({words}, options);
  auto counts = single ? torch::Tensor() : at::empty({tiles * 2}, options);
  auto offsets = single ? torch::Tensor() : at::empty({tiles}, options);
  auto count = at::empty({1}, options), dense = at::empty({1}, options);
  auto packed = at::empty({capacity}, options), remapped = at::empty({queries, width}, options);
  auto *bits = reinterpret_cast<uint32_t *>(bitmap.data_ptr<int32_t>());
  dsv4_clear_bitmap_gfx936<<<(words + 1023) / 1024, 256, 0, stream>>>(bits, words, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dsv4_mark_bitmap_gfx936<false><<<queries, 256, 0, stream>>>(bits, indices.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), indices.stride(0), width, n, nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (single) {
    const int threads = words <= 1024 ? 256 : words <= 2048 ? 512 : 1024;
    dsv4_scan_bitmap_gfx936<true><<<1, threads, 0, stream>>>(bits, prefix.data_ptr<int32_t>(), nullptr, count.data_ptr<int32_t>(), dense.data_ptr<int32_t>(), words);
  } else {
    dsv4_scan_bitmap_gfx936<false><<<tiles, 1024, 0, stream>>>(bits, prefix.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), nullptr, nullptr, words);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!single) {
    dsv4_scan_bitmap_counts_gfx936<<<1, 64, 0, stream>>>(counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), count.data_ptr<int32_t>(), dense.data_ptr<int32_t>(), tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const bool vectorized = width % 4 == 0 && indices.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(indices.data_ptr()) % 16 == 0;
  const int64_t blocks = (n + 127) / 128 + queries;
  AT_DISPATCH_INDEX_TYPES(full_ids.scalar_type(), "pack_workspace_gfx936", [&] {
    const int32_t *offsets_ptr = single ? nullptr : offsets.data_ptr<int32_t>();
    auto launch = [&](auto tile_tag, auto vector_tag) {
      constexpr bool Single = decltype(tile_tag)::value, Vectorized = decltype(vector_tag)::value;
      dsv4_pack_remap_bitmap_gfx936<index_t, Vectorized, Single><<<blocks, 128, 0, stream>>>(
          full_ids.data_ptr<index_t>(), bits, prefix.data_ptr<int32_t>(), offsets_ptr,
          count.data_ptr<int32_t>(), dense.data_ptr<int32_t>(), packed.data_ptr<int32_t>(),
          remapped.data_ptr<int32_t>(), indices.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
          indices.stride(0), width, n, capacity, swa_rows);
    };
    if (single) {
      if (vectorized) launch(std::true_type{}, std::true_type{});
      else launch(std::true_type{}, std::false_type{});
    } else {
      if (vectorized) launch(std::false_type{}, std::true_type{});
      else launch(std::false_type{}, std::false_type{});
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, remapped, count};
}
#endif // __HIP_PLATFORM_AMD__


__global__ void dsv4_mark_topk_rows(int32_t *hits, const int32_t *indices,
                                    const int32_t *lengths, int64_t stride,
                                    int64_t width, int64_t num_compressed) {
  const int64_t query = blockIdx.x;
  const int64_t length = lengths[query];
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    if (column < length) {
      const int64_t row = indices[query * stride + column];
      if (row >= 0 && row < num_compressed) {
        atomicOr(reinterpret_cast<unsigned int *>(hits + row), 1u);
      }
    }
  }
}

template <typename index_t>
__global__ void dsv4_pack_topk_rows(const index_t *full_ids,
                                    const int32_t *hits, const int32_t *prefix,
                                    int32_t *packed, int64_t num_compressed,
                                    int64_t capacity) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < num_compressed && hits[row] != 0) {
    const int64_t rank = static_cast<int64_t>(prefix[row]) - 1;
    if (rank >= 0 && rank < capacity) {
      packed[rank] = static_cast<int32_t>(full_ids[row]);
    }
  }
}

__global__ void dsv4_remap_topk_rows(int32_t *output, const int32_t *indices,
                                     const int32_t *lengths,
                                     const int32_t *prefix, int64_t stride,
                                     int64_t width, int64_t num_compressed,
                                     int64_t capacity, int64_t swa_rows) {
  const int64_t query = blockIdx.x;
  const int64_t length = lengths[query];
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    int32_t result = -1;
    if (column < length) {
      const int64_t row = indices[query * stride + column];
      if (row >= 0 && row < num_compressed) {
        const int64_t rank = static_cast<int64_t>(prefix[row]) - 1;
        if (rank >= 0 && rank < capacity)
          result = static_cast<int32_t>(rank);
      } else if (row >= num_compressed &&
                 (swa_rows < 0 || row - num_compressed < swa_rows)) {
        const int64_t rebased = row - num_compressed + capacity;
        if (rebased <= std::numeric_limits<int32_t>::max()) {
          result = static_cast<int32_t>(rebased);
        }
      }
    }
    output[query * width + column] = result;
  }
}

} // namespace

std::vector<torch::Tensor>
dsv4_pack_sparse_topk_workspace(const torch::Tensor &full_ids,
                                const torch::Tensor &indices,
                                const torch::Tensor &lengths, int64_t topk,
                                int64_t capacity, int64_t swa_rows) {
  TORCH_CHECK(full_ids.is_cuda() && indices.is_cuda() && lengths.is_cuda(),
              "C4 TopK workspace inputs must be GPU tensors");
  TORCH_CHECK(full_ids.device() == indices.device() &&
                  full_ids.device() == lengths.device(),
              "C4 TopK workspace inputs must share a device");
  TORCH_CHECK(full_ids.dim() == 1 && full_ids.is_contiguous() &&
                  (full_ids.scalar_type() == at::kInt ||
                   full_ids.scalar_type() == at::kLong),
              "full_ids must be contiguous one-dimensional int32/int64");
  TORCH_CHECK(indices.dim() == 2 && indices.scalar_type() == at::kInt &&
                  indices.stride(1) == 1,
              "indices must be int32 with contiguous rows");
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous() &&
                  lengths.scalar_type() == at::kInt &&
                  lengths.size(0) == indices.size(0),
              "lengths must be contiguous int32 with one entry per query");
  const int64_t n = full_ids.numel();
  const int64_t queries = indices.size(0);
  const int64_t width = indices.size(1);
  const int64_t max_int = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(n > 0 && n <= max_int && queries > 0 && queries <= max_int &&
                  width > 0 && width <= max_int && topk > 0 && topk <= width,
              "C4 TopK workspace geometry is outside the supported range");
  TORCH_CHECK(capacity >= std::min(n, queries * topk) && capacity <= n,
              "C4 TopK workspace capacity is not a safe union bound");
  TORCH_CHECK(swa_rows >= -1 &&
                  (swa_rows == -1 || swa_rows <= max_int - capacity),
              "compressed capacity plus SWA rows exceeds int32 indices");

  const c10::cuda::CUDAGuard device_guard(full_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto options = full_ids.options().dtype(at::kInt);
#if defined(__HIP_PLATFORM_AMD__)
  if ((std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx936", 6) == 0 || std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx938", 6) == 0))
    return pack_workspace_gfx936(full_ids, indices, lengths, capacity, swa_rows);
#endif
  auto hits = at::zeros({n}, options);
  dsv4_mark_topk_rows<<<queries, kWorkspaceThreads, 0, stream>>>(
      hits.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), indices.stride(0), width, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // The union count stays on the current GPU stream. No host readback or
  // data-dependent allocation is needed; capacity is a caller-known bound.
  auto prefix = at::cumsum(hits, 0, at::kInt);
  // A one-element view would retain the entire N-row scan through Attention.
  // Clone only the GPU count; no host readback or synchronization is needed.
  auto count = prefix.slice(0, n - 1, n).clone();
  auto packed = at::full({capacity}, -1, options);
  const int64_t blocks = (n + kWorkspaceThreads - 1) / kWorkspaceThreads;
  if (full_ids.scalar_type() == at::kLong) {
    dsv4_pack_topk_rows<int64_t><<<blocks, kWorkspaceThreads, 0, stream>>>(
        full_ids.data_ptr<int64_t>(), hits.data_ptr<int32_t>(),
        prefix.data_ptr<int32_t>(), packed.data_ptr<int32_t>(), n, capacity);
  } else {
    dsv4_pack_topk_rows<int32_t><<<blocks, kWorkspaceThreads, 0, stream>>>(
        full_ids.data_ptr<int32_t>(), hits.data_ptr<int32_t>(),
        prefix.data_ptr<int32_t>(), packed.data_ptr<int32_t>(), n, capacity);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto remapped = at::empty({queries, width}, options);
  dsv4_remap_topk_rows<<<queries, kWorkspaceThreads, 0, stream>>>(
      remapped.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), prefix.data_ptr<int32_t>(),
      indices.stride(0), width, n, capacity, swa_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, remapped, count};
}

std::vector<torch::Tensor> dsv4_pack_c4_topk_workspace(
    const torch::Tensor &full_ids, const torch::Tensor &indices,
    const torch::Tensor &lengths, int64_t topk, int64_t capacity) {
  return dsv4_pack_sparse_topk_workspace(full_ids, indices, lengths, topk,
                                         capacity, /*swa_rows=*/-1);
}

// Full-capacity alternative: preserve logical row positions and the caller's
// original Attention indices. The queue is internal and need not be sorted.
// Only selected output rows are written. U owns one GPU int32 allocation.
torch::Tensor dsv4_dequantize_sparse_topk_out(
    const torch::Tensor &cache, const torch::Tensor &full_ids,
    const torch::Tensor &indices, const torch::Tensor &lengths,
    torch::Tensor &output, int64_t topk, int64_t page_size, bool fp8_fnuz) {
#if defined(__HIP_PLATFORM_AMD__)
  TORCH_CHECK(full_ids.is_cuda() && indices.is_cuda() && lengths.is_cuda() &&
                  full_ids.device() == indices.device() && full_ids.device() == lengths.device(),
              "selection inputs must share a GPU");
  TORCH_CHECK(full_ids.dim() == 1 && full_ids.is_contiguous() &&
                  (full_ids.scalar_type() == at::kInt || full_ids.scalar_type() == at::kLong),
              "full_ids must be contiguous int32/int64");
  TORCH_CHECK(indices.dim() == 2 && indices.scalar_type() == at::kInt && indices.stride(1) == 1 &&
                  lengths.dim() == 1 && lengths.is_contiguous() && lengths.scalar_type() == at::kInt &&
                  lengths.numel() == indices.size(0), "invalid selection indices/lengths");
  const int64_t n = full_ids.numel(), q = indices.size(0), width = indices.size(1);
  const int64_t max_int = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(n > 0 && n <= max_int && q > 0 && q <= max_int && width > 0 && width <= max_int &&
                  topk > 0 && topk <= width, "invalid selection geometry");
  TORCH_CHECK(n <= q * topk, "selected dequantization requires the full-capacity workspace");
  const c10::cuda::CUDAGuard guard(full_ids.device());
  TORCH_CHECK((std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx936", 6) == 0 || std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx938", 6) == 0),
              "selected dequantization is currently supported on gfx936/gfx938");
  const auto options = full_ids.options().dtype(at::kInt);
  const int64_t words = (n + 31) / 32;
  auto bitmap = at::empty({words}, options), selected = at::empty({n}, options);
  auto count = at::empty({1}, options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto *bits = reinterpret_cast<uint32_t *>(bitmap.data_ptr<int32_t>());
  dsv4_clear_bitmap_gfx936<<<(words + 1023) / 1024, 256, 0, stream>>>(bits, words, count.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (q >= 1024 && n <= 65536) {
    dsv4_mark_bitmap_lds_gfx936<4><<<(q + 3) / 4, 256, words * sizeof(uint32_t), stream>>>(
        bits, indices.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), indices.stride(0), width, n, q,
        selected.data_ptr<int32_t>(), count.data_ptr<int32_t>());
  } else if (q >= 1024 && n <= 114688) {
    dsv4_mark_bitmap_lds_gfx936<16><<<(q + 15) / 16, 256, words * sizeof(uint32_t), stream>>>(
        bits, indices.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), indices.stride(0), width, n, q,
        selected.data_ptr<int32_t>(), count.data_ptr<int32_t>());
  } else {
    dsv4_mark_bitmap_gfx936<true><<<q, 256, 0, stream>>>(bits, indices.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(), indices.stride(0), width, n, selected.data_ptr<int32_t>(), count.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dequantize_k_cache_paged_hcu_selected(cache, full_ids, selected, count, output, page_size, fp8_fnuz);
  return count;
#else
  TORCH_CHECK(false, "selected dequantization is currently supported on gfx936/gfx938");
#endif
}

} // namespace native
} // namespace at
