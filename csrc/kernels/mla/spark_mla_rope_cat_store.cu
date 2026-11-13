// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace at::native {
namespace {

constexpr int kNopeDim = 512;
constexpr int kRopeDim = 64;
constexpr int kQueryDim = kNopeDim + kRopeDim;
constexpr int kCacheBytes = kNopeDim + 4 * sizeof(float) +
                            kRopeDim * sizeof(c10::BFloat16);
constexpr int kThreads = 256;
using BFloat16 = c10::BFloat16;

struct TensorStrides {
  int64_t token;
  int64_t head;
};

template <bool NeoX>
__device__ __forceinline__ BFloat16 rotate_bf16(
    const BFloat16* input, const BFloat16* cache, int dimension) {
  const int partner = NeoX ? (dimension + 32) % 64 : dimension ^ 1;
  const int frequency = NeoX ? dimension % 32 : dimension / 2;
  const bool negative = NeoX ? dimension < 32 : dimension % 2 == 0;
  const float x = static_cast<float>(input[dimension]);
  const float y = static_cast<float>(input[partner]);
  const float cosine = static_cast<float>(cache[frequency]);
  const float sine = static_cast<float>(cache[frequency + 32]);
  // Eager PyTorch stores both products in BF16 before the BF16 addition.
  // Keep these conversion boundaries instead of using an FP32 FMA.
  const BFloat16 first(x * cosine);
  const BFloat16 second((negative ? -y : y) * sine);
  return BFloat16(static_cast<float>(first) + static_cast<float>(second));
}

__device__ __forceinline__ float shuffle_xor(float value, int mask) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return __shfl_xor(value, mask);
#else
  return __shfl_xor_sync(0xffffffff, value, mask);
#endif
}

template <bool NeoX, int HeadsPerBlock>
__global__ __launch_bounds__(kThreads) void spark_mla_rope_cat_store_kernel(
    const int64_t* positions, const BFloat16* query_nope,
    const BFloat16* query_rope, const BFloat16* key_nope,
    const BFloat16* key_rope, const BFloat16* cosine_sine,
    const int64_t* locations, BFloat16* query_out, BFloat16* key_rope_out,
    uint8_t* cache, int tokens, int heads, TensorStrides qn_stride,
    TensorStrides qr_stride, int64_t kn_stride, int64_t kr_stride,
    int64_t cosine_stride, int64_t cache_stride, int64_t cache_rows,
    int query_blocks, float quant_epsilon) {
  if (blockIdx.x < query_blocks) {
    // Q blocks copy the absorbed 512 columns and rotate the 64 PE columns
    // directly into their final contiguous attention input. No intermediate
    // rotated Q or subsequent concatenation is materialized.
    const int groups = (heads + HeadsPerBlock - 1) / HeadsPerBlock;
    const int token = blockIdx.x / groups;
    const int first_head = (blockIdx.x % groups) * HeadsPerBlock;
    const int64_t position = positions[token];
    const BFloat16* trig = cosine_sine + position * cosine_stride;
    for (int index = threadIdx.x * 4; index < HeadsPerBlock * kNopeDim;
         index += kThreads * 4) {
      const int head = first_head + index / kNopeDim;
      const int column = index % kNopeDim;
      if (head < heads) {
        const BFloat16* source = query_nope + token * qn_stride.token +
                                 head * qn_stride.head + column;
        BFloat16* target = query_out +
                           (static_cast<int64_t>(token) * heads + head) *
                               kQueryDim + column;
        *reinterpret_cast<uint2*>(target) =
            *reinterpret_cast<const uint2*>(source);
      }
    }
    for (int index = threadIdx.x; index < HeadsPerBlock * kRopeDim;
         index += kThreads) {
      const int head = first_head + index / kRopeDim;
      const int column = index % kRopeDim;
      if (head < heads) {
        const BFloat16* source = query_rope + token * qr_stride.token +
                                 head * qr_stride.head;
        query_out[(static_cast<int64_t>(token) * heads + head) * kQueryDim +
                  kNopeDim + column] = rotate_bf16<NeoX>(source, trig, column);
      }
    }
    return;
  }

  // K blocks use a 64-thread subgroup per token. The first 32 lanes quantize
  // four groups of 128 latent values; the next eight rotate and store RoPE.
  // Subgroups use shuffle reductions only, so there is no LDS/shared-memory
  // bank traffic or inter-block synchronization.
  const int token = (blockIdx.x - query_blocks) * 4 + threadIdx.x / 64;
  const int lane = threadIdx.x % 64;
  if (token >= tokens) {
    return;
  }
  const int64_t location = locations[token];
  const bool valid_location = location >= 0 && location < cache_rows;
  uint8_t* target = cache + (valid_location ? location : 0) * cache_stride;
  if (lane < 32) {
    if (!valid_location) {
      return;
    }
    const int group = lane / 8;
    const int column = group * 128 + (lane % 8) * 16;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = static_cast<float>(key_nope[token * kn_stride +
                                                  column + index]);
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
      maximum = fmaxf(maximum, shuffle_xor(maximum, mask));
    }
    const float scale = fmaxf(maximum, quant_epsilon) / 448.0f;
    if (lane % 8 == 0) {
      reinterpret_cast<float*>(target + kNopeDim)[group] = scale;
    }
    const float inverse_scale = scale == 0.0f ? 0.0f : 1.0f / scale;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      const c10::Float8_e4m3fn value(values[index] * inverse_scale);
      target[column + index] = value.x;
    }
  } else if (lane < 40) {
    const int64_t position = positions[token];
    const BFloat16* trig = cosine_sine + position * cosine_stride;
    const BFloat16* source = key_rope + token * kr_stride;
    BFloat16* stored_rope = reinterpret_cast<BFloat16*>(
        target + kNopeDim + 4 * sizeof(float));
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int column = (lane - 32) * 8 + index;
      const BFloat16 value = rotate_bf16<NeoX>(source, trig, column);
      if (valid_location) {
        stored_rope[column] = value;
      }
      key_rope_out[token * kRopeDim + column] = value;
    }
  }
}

TensorStrides strides(const torch::Tensor& tensor) {
  return {tensor.stride(-3), tensor.stride(-2)};
}

void check_bf16(const torch::Tensor& tensor, const torch::Tensor& reference,
                const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(), name,
              " must be on the query CUDA device");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16 && tensor.stride(-1) == 1,
              name, " requires BF16 with a contiguous innermost dimension");
}

}  // namespace

void spark_mla_rope_cat_store(
    const torch::Tensor& positions, const torch::Tensor& query_nope,
    const torch::Tensor& query_rope, const torch::Tensor& key_nope,
    const torch::Tensor& key_rope, const torch::Tensor& cosine_sine,
    const torch::Tensor& locations, torch::Tensor query_out,
    torch::Tensor key_rope_out, torch::Tensor cache, bool neox,
    double quant_epsilon, int64_t heads_per_block) {
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(query_nope));
  const int64_t tokens = positions.numel();
  TORCH_CHECK(query_nope.dim() == 3 && query_nope.size(0) == tokens &&
                  query_nope.size(2) == kNopeDim,
              "query_nope must be [tokens, heads, 512]");
  const int64_t heads = query_nope.size(1);
  TORCH_CHECK(heads > 0 && heads <= 128 && tokens <= 36,
              "Spark MLA preprocessing supports up to 36 tokens/128 heads");
  check_bf16(query_rope, query_nope, "query_rope");
  check_bf16(key_nope, query_nope, "key_nope");
  check_bf16(key_rope, query_nope, "key_rope");
  check_bf16(cosine_sine, query_nope, "cosine_sine");
  check_bf16(query_out, query_nope, "query_out");
  check_bf16(key_rope_out, query_nope, "key_rope_out");
  check_bf16(query_nope, query_nope, "query_nope");
  TORCH_CHECK(query_rope.sizes() == at::IntArrayRef({tokens, heads, kRopeDim}) &&
                  key_nope.sizes() == at::IntArrayRef({tokens, 1, kNopeDim}) &&
                  key_rope.sizes() == at::IntArrayRef({tokens, 1, kRopeDim}),
              "Unexpected Spark MLA source shape");
  TORCH_CHECK(query_out.sizes() == at::IntArrayRef({tokens, heads, kQueryDim}) &&
                  key_rope_out.sizes() == at::IntArrayRef({tokens, 1, kRopeDim}) &&
                  query_out.is_contiguous() && key_rope_out.is_contiguous(),
              "Unexpected Spark MLA output shape or layout");
  TORCH_CHECK(cosine_sine.dim() == 2 && cosine_sine.size(1) == kRopeDim,
              "cosine_sine must contain 32 BF16 cosine/sine pairs per row");
  TORCH_CHECK(positions.is_cuda() && locations.is_cuda() &&
                  positions.device() == query_nope.device() &&
                  locations.device() == query_nope.device() &&
                  positions.scalar_type() == at::kLong &&
                  locations.scalar_type() == at::kLong &&
                  positions.is_contiguous() && locations.is_contiguous() &&
                  locations.numel() == tokens,
              "positions and locations must be contiguous int64 CUDA tensors");
  TORCH_CHECK(cache.is_cuda() && cache.device() == query_nope.device() &&
                  cache.scalar_type() == at::kByte && cache.dim() == 3 &&
                  cache.size(1) == 1 && cache.size(2) == kCacheBytes &&
                  cache.is_contiguous(),
              "cache must use the Spark FP8 MLA [slots, 1, 656] byte layout");
  TORCH_CHECK(query_nope.stride(0) % 4 == 0 &&
                  query_nope.stride(1) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(query_nope.data_ptr()) % 8 == 0,
              "query_nope must support aligned four-BF16 copies");
  TORCH_CHECK(std::isfinite(quant_epsilon) && quant_epsilon > 0.0,
              "quant_epsilon must be positive and finite");
  TORCH_CHECK(heads_per_block == 1 || heads_per_block == 2 ||
                  heads_per_block == 4 || heads_per_block == 8,
              "heads_per_block must be 1, 2, 4 or 8");
  if (tokens == 0) {
    return;
  }
  const int q_blocks = tokens * ((heads + heads_per_block - 1) / heads_per_block);
  const int blocks = q_blocks + (tokens + 3) / 4;
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(NEOX, HEADS)                                                     \
  spark_mla_rope_cat_store_kernel<NEOX, HEADS><<<blocks, kThreads, 0, stream>>>(  \
      positions.data_ptr<int64_t>(), query_nope.data_ptr<BFloat16>(),           \
      query_rope.data_ptr<BFloat16>(), key_nope.data_ptr<BFloat16>(),           \
      key_rope.data_ptr<BFloat16>(), cosine_sine.data_ptr<BFloat16>(),          \
      locations.data_ptr<int64_t>(), query_out.data_ptr<BFloat16>(),            \
      key_rope_out.data_ptr<BFloat16>(), cache.data_ptr<uint8_t>(), tokens,     \
      heads, strides(query_nope), strides(query_rope), key_nope.stride(0),      \
      key_rope.stride(0), cosine_sine.stride(0), cache.stride(0), cache.size(0), \
      q_blocks, static_cast<float>(quant_epsilon))
#define DISPATCH(NEOX)                   \
  switch (heads_per_block) {             \
    case 1: LAUNCH(NEOX, 1); break;      \
    case 2: LAUNCH(NEOX, 2); break;      \
    case 4: LAUNCH(NEOX, 4); break;      \
    case 8: LAUNCH(NEOX, 8); break;      \
  }
  if (neox) {
    DISPATCH(true)
  } else {
    DISPATCH(false)
  }
#undef DISPATCH
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace at::native
