// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

namespace lightop::cp_output {
namespace {

constexpr int kCycleThreads = 64;
constexpr int kItemsPerThread = 2;
constexpr int kGenericThreads = 256;
constexpr int kGenericItems = 8;
constexpr int64_t kMaxBlocks = 65535;

__global__ void mark_short_cycle_leaders(int32_t *flags, int64_t rows,
                                         int64_t local_rows, int64_t cp_size) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  int64_t next = (row % cp_size) * local_rows + row / cp_size;
  if (next == row) {
    flags[row] = 0;
    return;
  }
  while (next > row)
    next = (next % cp_size) * local_rows + next / cp_size;
  flags[row] = next == row;
}

__global__ void initialize_cycle_labels(int64_t *labels, int64_t *next,
                                        int64_t rows, int64_t local_rows,
                                        int64_t cp_size) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  labels[row] = row;
  next[row] = (row % cp_size) * local_rows + row / cp_size;
}

__global__ void jump_cycle_labels(const int64_t *labels, const int64_t *next,
                                  int64_t *output_labels, int64_t *output_next,
                                  int64_t rows) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const int64_t neighbor = next[row];
  const int64_t a = labels[row], b = labels[neighbor];
  output_labels[row] = a < b ? a : b;
  output_next[row] = next[neighbor];
}

__global__ void mark_cycle_labels(int32_t *flags, const int64_t *labels,
                                  int64_t rows, int64_t local_rows,
                                  int64_t cp_size) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const int64_t next = (row % cp_size) * local_rows + row / cp_size;
  flags[row] = labels[row] == row && next != row;
}

__global__ void pack_cycle_leaders(int64_t *leaders, const int32_t *flags,
                                   const int64_t *prefix, int64_t rows,
                                   int64_t capacity) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows && flags[row]) {
    const int64_t destination = prefix[row] - 1;
    if (destination < capacity)
      leaders[destination] = row;
  }
}

#if defined(__HIP_PLATFORM_AMD__)
using Vector = uint32_t __attribute__((ext_vector_type(4)));
#else
using Vector = uint4;
#endif
static_assert(sizeof(Vector) == 16);

// One (cycle, column tile) owns disjoint bytes for the entire permutation.
// Load all rows into registers before writing; no global scratch or inter-block
// barrier is needed. Power-of-two layouts have a small known cycle period.
template <int Period>
__global__ __launch_bounds__(kCycleThreads) void short_cycles(
    Vector *data, const int64_t *leaders, int64_t cycles, int64_t row_vectors,
    int cp_shift, int local_shift) {
  constexpr int64_t kTile = kCycleThreads * kItemsPerThread;
  const int64_t tiles = (row_vectors + kTile - 1) / kTile;
  for (int64_t task = blockIdx.x; task < cycles * tiles; task += gridDim.x) {
    const int64_t column = (task / cycles) * kTile + threadIdx.x;
    if (column >= row_vectors)
      continue;
    int64_t positions[Period];
    Vector values[Period][kItemsPerThread] = {};
    positions[0] = leaders[task % cycles];
    if (positions[0] < 0 ||
        positions[0] >= (int64_t{1} << (cp_shift + local_shift)))
      continue;
    const int64_t mask = (int64_t{1} << cp_shift) - 1;
#pragma unroll
    for (int index = 1; index < Period; ++index) {
      const int64_t previous = positions[index - 1];
      positions[index] =
          ((previous & mask) << local_shift) + (previous >> cp_shift);
    }
#pragma unroll
    for (int index = 0; index < Period; ++index) {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int64_t offset = column + item * kCycleThreads;
        if (offset < row_vectors) {
          values[index][item] = data[positions[index] * row_vectors + offset];
        }
      }
    }
#pragma unroll
    for (int index = 0; index < Period; ++index) {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int64_t offset = column + item * kCycleThreads;
        if (offset < row_vectors) {
          data[positions[index] * row_vectors + offset] =
              values[(index + 1) % Period][item];
        }
      }
    }
  }
}

// General shapes, arbitrary CP, and unaligned rows use the same row cycles.
// Only one row tile is retained while the other rows rotate into it. The byte
// path preserves odd widths and unaligned storage offsets without typed loads.
template <typename Element, int Items>
__global__ __launch_bounds__(kGenericThreads) void general_cycles(
    Element *data, const int64_t *leaders, int64_t cycles, int64_t row_elements,
    int64_t local_rows, int64_t cp_size) {
  constexpr int64_t kTile = kGenericThreads * Items;
  const int64_t tiles = (row_elements + kTile - 1) / kTile;
  for (int64_t task = blockIdx.x; task < cycles * tiles; task += gridDim.x) {
    const int64_t leader = leaders[task / tiles];
    if (leader < 0 || leader >= local_rows * cp_size)
      continue;
    const int64_t column = (task % tiles) * kTile + threadIdx.x;
    Element first[Items] = {};
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int64_t offset = column + item * kGenericThreads;
      if (offset < row_elements)
        first[item] = data[leader * row_elements + offset];
    }
    int64_t current = leader;
    int64_t next = (current % cp_size) * local_rows + current / cp_size;
    while (next != leader) {
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        const int64_t offset = column + item * kGenericThreads;
        if (offset < row_elements) {
          data[current * row_elements + offset] =
              data[next * row_elements + offset];
        }
      }
      current = next;
      next = (current % cp_size) * local_rows + current / cp_size;
    }
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int64_t offset = column + item * kGenericThreads;
      if (offset < row_elements)
        data[current * row_elements + offset] = first[item];
    }
  }
}

} // namespace

void make_interleave_plan_(at::Tensor leaders, int64_t rows, int64_t cp_size) {
  TORCH_CHECK(leaders.is_cuda() && leaders.is_contiguous() &&
                  leaders.dim() == 1 && leaders.scalar_type() == at::kLong,
              "contiguous int64 GPU plan storage required");
  TORCH_CHECK(rows >= 0 && cp_size > 0 && rows % cp_size == 0,
              "rows must be nonnegative and divisible by positive CP");
  TORCH_CHECK(leaders.numel() <= rows / 2,
              "cycle-plan capacity exceeds the nontrivial-cycle bound");
  const int64_t blocks = (rows + kGenericThreads - 1) / kGenericThreads;
  TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(),
              "too many plan rows");
  if (leaders.numel() == 0)
    return;
  const c10::cuda::CUDAGuard guard(leaders.device());
  const auto stream = at::cuda::getCurrentCUDAStream(leaders.get_device());
  leaders.fill_(-1);
  auto flags = at::empty({rows}, leaders.options().dtype(at::kInt));
  const int64_t local_rows = rows / cp_size;
  if ((rows & (rows - 1)) == 0 && (cp_size & (cp_size - 1)) == 0) {
    mark_short_cycle_leaders<<<blocks, kGenericThreads, 0, stream>>>(
        flags.data_ptr<int32_t>(), rows, local_rows, cp_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // Pointer doubling bounds generic-cycle discovery by O(rows * log(rows)).
    // All labels and successor maps stay on the current GPU stream.
    auto labels = at::empty({rows}, leaders.options());
    auto next = at::empty_like(labels);
    auto output_labels = at::empty_like(labels);
    auto output_next = at::empty_like(labels);
    initialize_cycle_labels<<<blocks, kGenericThreads, 0, stream>>>(
        labels.data_ptr<int64_t>(), next.data_ptr<int64_t>(), rows, local_rows,
        cp_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int64_t covered = 1; covered < rows; covered <<= 1) {
      jump_cycle_labels<<<blocks, kGenericThreads, 0, stream>>>(
          labels.data_ptr<int64_t>(), next.data_ptr<int64_t>(),
          output_labels.data_ptr<int64_t>(), output_next.data_ptr<int64_t>(),
          rows);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      std::swap(labels, output_labels);
      std::swap(next, output_next);
    }
    mark_cycle_labels<<<blocks, kGenericThreads, 0, stream>>>(
        flags.data_ptr<int32_t>(), labels.data_ptr<int64_t>(), rows, local_rows,
        cp_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto prefix = at::cumsum(flags, 0, at::kLong);
  pack_cycle_leaders<<<blocks, kGenericThreads, 0, stream>>>(
      leaders.data_ptr<int64_t>(), flags.data_ptr<int32_t>(),
      prefix.data_ptr<int64_t>(), rows, leaders.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void interleave_rows_(at::Tensor data, int64_t cp_size,
                      const at::Tensor &leaders) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() && data.dim() >= 1,
              "contiguous GPU row tensor required");
  TORCH_CHECK(!data.requires_grad(), "inference-only tensor required");
  TORCH_CHECK(cp_size > 0 && data.size(0) % cp_size == 0,
              "rows must be divisible by positive CP");
  TORCH_CHECK(leaders.is_cuda() && leaders.device() == data.device() &&
                  leaders.scalar_type() == at::kLong && leaders.dim() == 1 &&
                  leaders.is_contiguous(),
              "contiguous int64 cycle plan on the same device required");
  if (data.numel() == 0 || leaders.numel() == 0)
    return;
  const c10::cuda::CUDAGuard guard(data.device());
  const auto stream = at::cuda::getCurrentCUDAStream(data.get_device());
  const int64_t rows = data.size(0);
  const int64_t row_bytes = data.nbytes() / rows;
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(data.data_ptr());
  if (row_bytes % sizeof(Vector) == 0 && pointer % alignof(Vector) == 0 &&
      (rows & (rows - 1)) == 0 && (cp_size & (cp_size - 1)) == 0 &&
      cp_size > 1 && rows > cp_size) {
    const int total_shift = __builtin_ctzll(rows);
    const int cp_shift = __builtin_ctzll(cp_size);
    const int period = total_shift / std::gcd(total_shift, cp_shift);
    const int64_t vectors = row_bytes / sizeof(Vector);
    const int64_t tile = kCycleThreads * kItemsPerThread;
    const int64_t blocks =
        std::min(kMaxBlocks, leaders.numel() * ((vectors + tile - 1) / tile));
#define LAUNCH_PERIOD(N)                                                       \
  case N:                                                                      \
    short_cycles<N><<<blocks, kCycleThreads, 0, stream>>>(                     \
        static_cast<Vector *>(data.data_ptr()), leaders.data_ptr<int64_t>(),   \
        leaders.numel(), vectors, cp_shift, total_shift - cp_shift);           \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                            \
    return
    switch (period) {
      LAUNCH_PERIOD(2);
      LAUNCH_PERIOD(3);
      LAUNCH_PERIOD(4);
      LAUNCH_PERIOD(5);
      LAUNCH_PERIOD(6);
      LAUNCH_PERIOD(7);
      LAUNCH_PERIOD(8);
      LAUNCH_PERIOD(9);
      LAUNCH_PERIOD(10);
      LAUNCH_PERIOD(11);
      LAUNCH_PERIOD(12);
      LAUNCH_PERIOD(13);
      LAUNCH_PERIOD(14);
      LAUNCH_PERIOD(15);
      LAUNCH_PERIOD(16);
    }
#undef LAUNCH_PERIOD
  }
  if (row_bytes % sizeof(uint32_t) == 0 && pointer % alignof(uint32_t) == 0) {
    const int64_t words = row_bytes / sizeof(uint32_t);
    const int64_t tile = kGenericThreads * kGenericItems;
    const int64_t blocks =
        std::min(kMaxBlocks, leaders.numel() * ((words + tile - 1) / tile));
    general_cycles<uint32_t, kGenericItems>
        <<<blocks, kGenericThreads, 0, stream>>>(
            static_cast<uint32_t *>(data.data_ptr()),
            leaders.data_ptr<int64_t>(), leaders.numel(), words, rows / cp_size,
            cp_size);
  } else {
    const int64_t blocks = std::min(
        kMaxBlocks, leaders.numel() *
                        ((row_bytes + kGenericThreads - 1) / kGenericThreads));
    general_cycles<uint8_t, 1><<<blocks, kGenericThreads, 0, stream>>>(
        static_cast<uint8_t *>(data.data_ptr()), leaders.data_ptr<int64_t>(),
        leaders.numel(), row_bytes, rows / cp_size, cp_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace lightop::cp_output
