// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace at {
namespace native {
namespace {

constexpr int kWorkspaceThreads = 256;

__global__ void dsv4_mark_topk_rows(
    int32_t* hits, const int32_t* indices, const int32_t* lengths,
    int64_t stride, int64_t width, int64_t num_compressed) {
  const int64_t query = blockIdx.x;
  const int64_t length = lengths[query];
  for (int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    if (column < length) {
      const int64_t row = indices[query * stride + column];
      if (row >= 0 && row < num_compressed) {
        atomicOr(reinterpret_cast<unsigned int*>(hits + row), 1u);
      }
    }
  }
}

template <typename index_t>
__global__ void dsv4_pack_topk_rows(
    const index_t* full_ids, const int32_t* hits, const int32_t* prefix,
    int32_t* packed, int64_t num_compressed, int64_t capacity) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row < num_compressed && hits[row] != 0) {
    const int64_t rank = static_cast<int64_t>(prefix[row]) - 1;
    if (rank >= 0 && rank < capacity) {
      packed[rank] = static_cast<int32_t>(full_ids[row]);
    }
  }
}

__global__ void dsv4_remap_topk_rows(
    int32_t* output, const int32_t* indices, const int32_t* lengths,
    const int32_t* prefix, int64_t stride, int64_t width,
    int64_t num_compressed, int64_t capacity) {
  const int64_t query = blockIdx.x;
  const int64_t length = lengths[query];
  for (int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    int32_t result = -1;
    if (column < length) {
      const int64_t row = indices[query * stride + column];
      if (row >= 0 && row < num_compressed) {
        const int64_t rank = static_cast<int64_t>(prefix[row]) - 1;
        if (rank >= 0 && rank < capacity) result = static_cast<int32_t>(rank);
      } else if (row >= num_compressed) {
        result = static_cast<int32_t>(row - num_compressed + capacity);
      }
    }
    output[query * width + column] = result;
  }
}

}  // namespace

std::vector<torch::Tensor> dsv4_pack_c4_topk_workspace(
    const torch::Tensor& full_ids, const torch::Tensor& indices,
    const torch::Tensor& lengths, int64_t topk, int64_t capacity) {
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

  const c10::cuda::CUDAGuard device_guard(full_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto options = full_ids.options().dtype(at::kInt);
  auto hits = at::zeros({n}, options);
  dsv4_mark_topk_rows<<<queries, kWorkspaceThreads, 0, stream>>>(
      hits.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), indices.stride(0), width, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // The union count stays on the current GPU stream. No host readback or
  // data-dependent allocation is needed; capacity is a caller-known bound.
  auto prefix = at::cumsum(hits, 0, at::kInt);
  auto count = prefix.slice(0, n - 1, n);
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
      indices.stride(0), width, n, capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, remapped, count};
}

}  // namespace native
}  // namespace at
