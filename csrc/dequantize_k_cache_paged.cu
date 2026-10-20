// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

namespace at {
namespace native {
namespace {

template <bool Fnuz>
__device__ __forceinline__ float decode_e4m3(uint32_t bits) {
  const uint32_t magnitude = bits & 0x7f;
  if (Fnuz ? bits == 0x80 : magnitude == 0x7f) {
    return __uint_as_float(0x7fc00000u);
  }
  float value;
  if (magnitude < 8) {
    value = static_cast<float>(magnitude) * (Fnuz ? 0x1p-10f : 0x1p-9f);
  } else {
    value = __uint_as_float((magnitude << 20) + (Fnuz ? 0x3b800000u : 0x3c000000u));
  }
  return __uint_as_float(__float_as_uint(value) | ((bits & 0x80) << 24));
}

// Two tokens per block, four adjacent dimensions per thread. The packed
// data/scales are gathered once, and BF16 output is written without a temporary.
template <typename Index, bool Fnuz>
__global__ void dequantize_k_cache_paged_hcu_kernel(
    const uint8_t* __restrict__ cache,
    const Index* __restrict__ token_ids,
    at::BFloat16* __restrict__ output,
    int64_t num_tokens,
    int64_t page_size,
    int64_t bytes_per_page,
    int64_t output_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * 2 + threadIdx.x / 128;
  if (row >= num_tokens) return;
  const int column = (threadIdx.x % 128) * 4;
  const int64_t loc = token_ids[row];
  const int64_t page = loc / page_size;
  const int64_t offset = loc % page_size;
  const uint8_t* data = cache + page * bytes_per_page + offset * 576;
  at::BFloat16* dst = output + row * output_stride + column;
  if (column < 448) {
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(data + column);
    const uint32_t scale_byte = cache[page * bytes_per_page + page_size * 576 + offset * 8 + column / 64];
    // The V4 Triton path flushes 2^-127 to zero. Scale 255 is +Inf, not
    // the V4.1 E8M0 NaN sentinel. Preserve those layout-specific semantics.
    const float scale = __uint_as_float(scale_byte << 23);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[i] = at::BFloat16(decode_e4m3<Fnuz>((packed >> (8 * i)) & 255) * scale);
    }
  } else {
    const uint16_t* rope = reinterpret_cast<const uint16_t*>(data + 448) + column - 448;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[i] = at::BFloat16(rope[i], at::BFloat16::from_bits());
    }
  }
}

}  // namespace

void dequantize_k_cache_paged_hcu(
    const at::Tensor& cache,
    const at::Tensor& token_ids,
    at::Tensor& output,
    int64_t page_size,
    bool fp8_fnuz) {
  TORCH_CHECK(cache.is_cuda() && token_ids.is_cuda() && output.is_cuda(), "inputs/output must be on GPU");
  TORCH_CHECK(cache.device() == token_ids.device() && cache.device() == output.device(), "devices must match");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() >= 2 && cache.is_contiguous(),
              "cache must be contiguous uint8 with a paged last dimension");
  TORCH_CHECK(page_size > 0 && page_size <= cache.size(-1) / 584 && cache.size(-1) % 4 == 0,
              "invalid page size or packed V4 page stride");
  TORCH_CHECK(token_ids.dim() == 1 && token_ids.is_contiguous() &&
                  (token_ids.scalar_type() == at::kInt || token_ids.scalar_type() == at::kLong),
              "token_ids must be contiguous int32/int64");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 && output.dim() == 3 &&
                  output.size(0) == token_ids.numel() && output.size(1) == 1 && output.size(2) == 512 &&
                  output.stride(2) == 1 && output.stride(0) >= 512,
              "output must be BF16 [num_tokens, 1, 512] with unit column stride and nonoverlapping rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0, "cache must be 4-byte aligned");
  const int64_t n = token_ids.numel();
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (n + 1) / 2;
  TORCH_CHECK(blocks <= 2147483647, "too many token rows");
  AT_DISPATCH_INDEX_TYPES(token_ids.scalar_type(), "dequantize_k_cache_paged_hcu", [&] {
    if (fp8_fnuz) {
      dequantize_k_cache_paged_hcu_kernel<index_t, true><<<blocks, 256, 0, stream>>>(
          cache.data_ptr<uint8_t>(), token_ids.data_ptr<index_t>(), output.data_ptr<at::BFloat16>(),
          n, page_size, cache.size(-1), output.stride(0));
    } else {
      dequantize_k_cache_paged_hcu_kernel<index_t, false><<<blocks, 256, 0, stream>>>(
          cache.data_ptr<uint8_t>(), token_ids.data_ptr<index_t>(), output.data_ptr<at::BFloat16>(),
          n, page_size, cache.size(-1), output.stride(0));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at

