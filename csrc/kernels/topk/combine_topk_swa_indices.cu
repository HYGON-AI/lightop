// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// HIP equivalent of dsv4.sparse_prefill_kernels._combine_topk_swa_indices_kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <limits>

namespace at {
namespace native {
namespace {

__global__ void combine_topk_swa_indices_kernel(
    const int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ query_pos,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ gather_lens,
    const int32_t* __restrict__ compressed_base,
    const int32_t* __restrict__ swa_base,
    int32_t* __restrict__ out_indices,
    int32_t* __restrict__ out_lens,
    int64_t topk_stride,
    int64_t out_stride,
    int64_t num_reqs,
    int64_t window_size,
    int64_t compress_ratio,
    int64_t topk) {
  const int64_t token = blockIdx.x;
  // Resolve a chunk-local row to its request without any host readback. upper_bound
  // handles empty requests and query_start_loc's cross-chunk starting offset.
  __shared__ int64_t topk_len, swa_len, compressed_offset, swa_start;
  if (threadIdx.x == 0) {
    const int64_t global_token = token + query_start_loc[0];
    int64_t lo = 0, hi = num_reqs;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (query_start_loc[mid + 1] <= global_token)
        lo = mid + 1;
      else
        hi = mid;
    }
    const int64_t pos = query_pos[token];
    const int64_t compressed_len = (pos + 1) / compress_ratio;
    topk_len = compressed_len < topk ? compressed_len : topk;
    swa_len = pos + 1 < window_size ? pos + 1 : window_size;
    compressed_offset = compressed_base[lo];
    const int64_t gather_start = static_cast<int64_t>(seq_lens[lo]) - gather_lens[lo];
    swa_start = static_cast<int64_t>(swa_base[lo]) + pos - swa_len + 1 - gather_start;
    out_lens[token] = static_cast<int32_t>(topk_len + swa_len);
  }
  __syncthreads();
  for (int64_t col = threadIdx.x; col < topk_len + swa_len; col += blockDim.x) {
    int32_t value;
    if (col < topk_len) {
      const int32_t raw = topk_indices[token * topk_stride + col];
      value = raw >= 0 ? static_cast<int32_t>(raw + compressed_offset) : -1;
    } else {
      value = static_cast<int32_t>(swa_start + col - topk_len);
    }
    out_indices[token * out_stride + col] = value;
  }
}

void check_vector(const at::Tensor& t, const at::Tensor& ref, int64_t size, const char* name) {
  TORCH_CHECK(t.device() == ref.device(), name, " must be on the same device as topk_indices");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(t.dim() == 1 && t.numel() == size && t.is_contiguous(), name, " has invalid shape or stride");
}

}  // namespace

void combine_topk_swa_indices_hcu(
    const at::Tensor& topk_indices,
    const at::Tensor& query_start_loc,
    const at::Tensor& query_pos,
    const at::Tensor& seq_lens,
    const at::Tensor& gather_lens,
    const at::Tensor& compressed_base,
    const at::Tensor& swa_base,
    at::Tensor& out_indices,
    at::Tensor& out_lens,
    int64_t window_size,
    int64_t compress_ratio,
    int64_t topk) {
  TORCH_CHECK(topk_indices.is_cuda() && topk_indices.scalar_type() == at::kInt, "topk_indices must be device int32");
  TORCH_CHECK(topk_indices.dim() == 2 && topk_indices.stride(1) == 1, "topk_indices must have unit column stride");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1), "invalid topk");
  TORCH_CHECK(compress_ratio >= 1 && window_size >= 0, "invalid compress_ratio or window_size");
  TORCH_CHECK(window_size <= std::numeric_limits<int32_t>::max() - topk - 127, "combined width is too large");
  const int64_t rows = topk_indices.size(0);
  const int64_t width = ((topk + window_size + 127) / 128) * 128;
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be one-dimensional");
  const int64_t num_reqs = seq_lens.numel();
  check_vector(seq_lens, topk_indices, num_reqs, "seq_lens");
  check_vector(query_start_loc, topk_indices, num_reqs + 1, "query_start_loc");
  check_vector(query_pos, topk_indices, rows, "query_pos");
  check_vector(gather_lens, topk_indices, num_reqs, "gather_lens");
  check_vector(compressed_base, topk_indices, num_reqs, "compressed_base");
  check_vector(swa_base, topk_indices, num_reqs, "swa_base");
  check_vector(out_lens, topk_indices, rows, "out_lens");
  TORCH_CHECK(out_indices.device() == topk_indices.device() && out_indices.scalar_type() == at::kInt,
              "out_indices must be int32 on the same device as topk_indices");
  TORCH_CHECK(out_indices.dim() == 2 && out_indices.size(0) == rows && out_indices.size(1) == width &&
                  out_indices.stride(1) == 1 && (rows <= 1 || out_indices.stride(0) >= width),
              "out_indices has invalid shape or stride");
  TORCH_CHECK(num_reqs > 0 || rows == 0, "nonempty rows require requests");
  TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(), "too many rows");
  if (rows == 0) return;

  const c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  combine_topk_swa_indices_kernel<<<rows, 256, 0, stream>>>(
      topk_indices.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(), query_pos.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), gather_lens.data_ptr<int32_t>(), compressed_base.data_ptr<int32_t>(),
      swa_base.data_ptr<int32_t>(), out_indices.data_ptr<int32_t>(), out_lens.data_ptr<int32_t>(),
      topk_indices.stride(0), out_indices.stride(0), num_reqs, window_size, compress_ratio, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at

