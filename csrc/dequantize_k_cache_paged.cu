// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <cstring>
#include <type_traits>
#include <torch/all.h>

namespace at {
namespace native {
namespace {

__device__ __forceinline__ int32_t uniform_row(int32_t value) {
#if defined(__HIP_PLATFORM_AMD__)
  return __builtin_amdgcn_readfirstlane(value);
#else
  return value;
#endif
}

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
    value =
        __uint_as_float((magnitude << 20) + (Fnuz ? 0x3b800000u : 0x3c000000u));
  }
  return __uint_as_float(__float_as_uint(value) | ((bits & 0x80) << 24));
}

// E4M3 times an E8M0 power of two is exactly representable in normal BF16.
// Assemble its bits directly; handle BF16 underflow with integer RNE.
template <bool Fnuz>
__device__ __forceinline__ uint16_t decode_scaled_bf16(uint32_t bits, uint32_t scale) {
  const uint32_t magnitude = bits & 0x7f;
  const uint16_t sign = (bits & 0x80) << 8;
  if (Fnuz ? bits == 0x80 : magnitude == 0x7f) return 0x7fc0;
  if (scale == 255) return magnitude == 0 ? 0x7fc0 : sign | 0x7f80;
  if (magnitude == 0 || scale == 0) return sign;
  int exponent;
  uint32_t significand;
  if (magnitude >= 8) {
    exponent = static_cast<int>(magnitude >> 3) + static_cast<int>(scale) - (Fnuz ? 8 : 7);
    significand = 0x80 | ((magnitude & 7) << 4);
  } else {
    const int leading = 31 - __clz(magnitude);
    exponent = static_cast<int>(scale) + leading - (Fnuz ? 10 : 9);
    significand = magnitude << (7 - leading);
  }
  if (exponent >= 255) return sign | 0x7f80;
  if (exponent > 0) return sign | (exponent << 7) | (significand & 0x7f);
  const int shift = 1 - exponent;
  if (shift >= 9) return sign;
  const uint32_t truncated = significand >> shift;
  const uint32_t remainder = significand & ((1u << shift) - 1);
  const uint32_t half = 1u << (shift - 1);
  const uint32_t rounded = truncated + (remainder > half || (remainder == half && (truncated & 1)));
  return sign | rounded;
}

// For scale bytes 11..246 every finite nonzero E4M3 value is normal BF16.
template <bool Fnuz>
__device__ __forceinline__ uint16_t decode_scaled_bf16_normal(uint32_t bits, int scale) {
  const uint32_t magnitude = bits & 0x7f;
  const uint16_t sign = (bits & 0x80) << 8;
  // FP8 subnormals, zeros and NaNs are mixed within a wave. Predicated bit
  // selection avoids serializing the normal/subnormal branches per value.
  // Eight subnormal encodings fit in two immediate registers. A byte
  // permutation replaces CLZ and a variable normalization shift.
#if defined(__HIP_PLATFORM_AMD__)
  const uint32_t subnormal = __builtin_amdgcn_perm(0x4e4c4a48u, 0x44403800u, magnitude & 7u) & 255u;
#else
  const uint32_t subnormal = __byte_perm(0x44403800u, 0x4e4c4a48u, magnitude & 7u) & 255u;
#endif
  const uint32_t normal = magnitude + 72;
  const uint32_t normal_mask = 0u - static_cast<uint32_t>(magnitude >= 8);
  uint32_t value = (normal & normal_mask) | (subnormal & ~normal_mask);
  value = (value << 4) + (scale - (Fnuz ? 17 : 16)) * 128;
  value &= 0u - static_cast<uint32_t>(magnitude != 0);
  value |= sign;
  const uint32_t nan_mask = 0u - static_cast<uint32_t>(Fnuz ? bits == 0x80 : magnitude == 0x7f);
  return (value & ~nan_mask) | (0x7fc0u & nan_mask);
}

template <bool Fnuz>
__device__ __forceinline__ uint32_t decode_scaled_bf16_normal_pair(uint32_t packed, int scale) {
#if defined(__HIP_PLATFORM_AMD__)
  const uint32_t magnitudes = (packed & 127u) | ((packed & 0x7f00u) << 8);
  // Subnormal values, in units of 16 BF16 bits with an exponent bias of 9,
  // fit in two immediate registers. Select two table bytes in one instruction.
  const uint32_t table = __builtin_amdgcn_perm(0x16141210u, 0x0c080000u, packed & 0x0707u) & 65535u;
  const uint32_t subnormal = (table & 255u) | ((table & 0xff00u) << 8);
  const uint32_t normal = magnitudes + 0x00100010u;
  const uint32_t normal_bits = (magnitudes + 0x7ff87ff8u) & 0x80008000u;
  const uint32_t normal_mask = (normal_bits << 1) - (normal_bits >> 15);
  uint32_t value = ((normal & normal_mask) | (subnormal & ~normal_mask)) << 4;
  const uint32_t exponent = static_cast<uint32_t>(scale - (Fnuz ? 10 : 9)) << 7;
  // Both 16-bit sums are nonnegative and below 0x8000 in this scale range,
  // so the low value cannot carry into the high value.
  value += exponent | (exponent << 16);
  const uint32_t nonzero_bits = (magnitudes + 0x7fff7fffu) & 0x80008000u;
  const uint32_t nonzero = (nonzero_bits << 1) - (nonzero_bits >> 15);
  value &= nonzero;
  const uint32_t sign_bits = ((packed & 128u) << 8) | ((packed & 0x8000u) << 16);
  value |= sign_bits;
  const uint32_t nan_bits = Fnuz ? sign_bits & ~nonzero_bits : (magnitudes + 0x7f817f81u) & 0x80008000u;
  const uint32_t nan_mask = (nan_bits << 1) - (nan_bits >> 15);
  return (value & ~nan_mask) | (0x7fc07fc0u & nan_mask);
#else
  return static_cast<uint32_t>(decode_scaled_bf16_normal<Fnuz>(packed & 255u, scale)) |
         (static_cast<uint32_t>(decode_scaled_bf16_normal<Fnuz>((packed >> 8) & 255u, scale)) << 16);
#endif
}

template <bool Fnuz>
__device__ __forceinline__ uint32_t decode_scaled_bf16_hcu_pair(uint32_t packed, int scale) {
#if defined(__gfx938__)
  // HCU's packed instruction decodes OCP E4M3 (FN). The probe and exhaustive
  // byte-pair tests also cover the two FNUZ-specific encodings below.
  const auto values = __builtin_hcu_cvt_pk_f32_fp8(packed, 0, 0, 0);
  const uint32_t first = packed & 255u, second = (packed >> 8) & 255u;
  float lo = values[0], hi = values[1];
  if constexpr (Fnuz) {
    // FNUZ has one extra finite magnitude. Decode it as +/-480 before
    // applying the one-bit exponent adjustment shared by all FNUZ values.
    lo = (first & 127u) == 127u ? __uint_as_float(((first & 128u) << 24) | 0x43f00000u) : lo;
    hi = (second & 127u) == 127u ? __uint_as_float(((second & 128u) << 24) | 0x43f00000u) : hi;
  }
  const float multiplier = __uint_as_float(static_cast<uint32_t>(scale - (Fnuz ? 1 : 0)) << 23);
  // In the guarded scale range, each finite product is exactly BF16-normal.
  // No rounding or underflow handling is needed; exceptional scales use the
  // general integer path in the caller.
  uint32_t result = (__float_as_uint(lo * multiplier) >> 16) |
                    (__float_as_uint(hi * multiplier) & 0xffff0000u);
  const uint32_t nan_mask = ((0u - static_cast<uint32_t>(Fnuz ? first == 128u : (first & 127u) == 127u)) & 65535u) |
                            ((0u - static_cast<uint32_t>(Fnuz ? second == 128u : (second & 127u) == 127u)) & 0xffff0000u);
  return (result & ~nan_mask) | (0x7fc07fc0u & nan_mask);
#else
  return decode_scaled_bf16_normal_pair<Fnuz>(packed, scale);
#endif
}

// One wave owns a token: eight adjacent FP8 values per lane and a 16-byte
// BF16 store. Separate token waves need no block-wide synchronization.
template <typename Index, bool Fnuz, bool Aligned>
__global__ void dequantize_selected_wave_kernel(
    const uint8_t *__restrict__ cache, const Index *__restrict__ token_ids,
    at::BFloat16 *__restrict__ output, int32_t num_tokens, int32_t page_size,
    int64_t bytes_per_page, int64_t output_stride, const int32_t *valid_count,
    const int32_t *selected_rows, int32_t page_shift) {
  const int32_t count = *valid_count;
  const int32_t active = count < num_tokens ? count : num_tokens;
  const int lane = threadIdx.x & 63;
  const int column = lane * 8;
  for (int64_t slot64 = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / 64;
       slot64 < active; slot64 += static_cast<int64_t>(gridDim.x) * 4) {
    const int32_t slot = uniform_row(static_cast<int32_t>(slot64));
    const int32_t logical_row = active == num_tokens ? slot : uniform_row(selected_rows[slot]);
    if (logical_row < 0 || logical_row >= num_tokens) continue;
    const int32_t loc = uniform_row(static_cast<int32_t>(token_ids[logical_row]));
    const int32_t page = page_shift >= 0 ? loc >> page_shift : loc / page_size;
    const int32_t offset = page_shift >= 0 ? loc & (page_size - 1) : loc % page_size;
    const uint8_t *data = cache + static_cast<int64_t>(page) * bytes_per_page + offset * 576;
    auto *dst = reinterpret_cast<uint16_t *>(output + static_cast<int64_t>(logical_row) * output_stride + column);
    uint32_t result[4];
    if (column < 448) {
      const uint32_t packed[2] = {reinterpret_cast<const uint32_t *>(data + column)[0],
                                  reinterpret_cast<const uint32_t *>(data + column)[1]};
      const uint32_t scale = cache[static_cast<int64_t>(page) * bytes_per_page + page_size * 576 + offset * 8 + column / 64];
      if (scale >= 11 && scale <= 246) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair)
          result[pair] = decode_scaled_bf16_hcu_pair<Fnuz>(packed[pair / 2] >> ((pair & 1) * 16), scale);
      } else {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const uint32_t bytes = packed[pair / 2] >> ((pair & 1) * 16);
          result[pair] = static_cast<uint32_t>(decode_scaled_bf16<Fnuz>(bytes & 255u, scale)) |
                         (static_cast<uint32_t>(decode_scaled_bf16<Fnuz>((bytes >> 8) & 255u, scale)) << 16);
        }
      }
    } else {
      const auto *rope = reinterpret_cast<const uint32_t *>(data + 448 + (column - 448) * 2);
#pragma unroll
      for (int j = 0; j < 4; ++j) result[j] = rope[j];
    }
    if constexpr (Aligned) {
      *reinterpret_cast<uint4 *>(dst) = make_uint4(result[0], result[1], result[2], result[3]);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        dst[j * 2] = result[j] & 65535;
        dst[j * 2 + 1] = result[j] >> 16;
      }
    }
  }
}

// Keep the original floating-point path as the general counted API and
// independent reference for the selected-row implementation.
template <typename Index, bool Fnuz>
__global__ void dequantize_k_cache_paged_hcu_kernel(
    const uint8_t *__restrict__ cache, const Index *__restrict__ token_ids,
    at::BFloat16 *__restrict__ output, int64_t num_tokens, int64_t page_size,
    int64_t bytes_per_page, int64_t output_stride, const int32_t *valid_count) {
  const int64_t count = valid_count == nullptr ? num_tokens : *valid_count;
  const int64_t active = count < num_tokens ? count : num_tokens;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * 2 + threadIdx.x / 128;
       row < active; row += static_cast<int64_t>(gridDim.x) * 2) {
    const int column = (threadIdx.x % 128) * 4;
    const int64_t loc = token_ids[row];
    const int64_t page = loc / page_size, offset = loc % page_size;
    const uint8_t *data = cache + page * bytes_per_page + offset * 576;
    at::BFloat16 *dst = output + row * output_stride + column;
    if (column < 448) {
      const uint32_t packed = *reinterpret_cast<const uint32_t *>(data + column);
      const uint32_t scale_byte = cache[page * bytes_per_page + page_size * 576 + offset * 8 + column / 64];
      const float scale = __uint_as_float(scale_byte << 23);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        dst[i] = at::BFloat16(decode_e4m3<Fnuz>((packed >> (8 * i)) & 255) * scale);
    } else {
      const uint16_t *rope = reinterpret_cast<const uint16_t *>(data + 448) + column - 448;
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = at::BFloat16(rope[i], at::BFloat16::from_bits());
    }
  }
}

} // namespace

static void dequantize_k_cache_paged_hcu_impl(
    const at::Tensor &cache, const at::Tensor &token_ids, at::Tensor &output,
    int64_t page_size, bool fp8_fnuz,
    const std::optional<at::Tensor> &valid_count = std::nullopt,
    const std::optional<at::Tensor> &selected_rows = std::nullopt) {
  TORCH_CHECK(cache.is_cuda() && token_ids.is_cuda() && output.is_cuda(),
              "inputs/output must be on GPU");
  TORCH_CHECK(cache.device() == token_ids.device() &&
                  cache.device() == output.device(),
              "devices must match");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() >= 2 &&
                  cache.is_contiguous(),
              "cache must be contiguous uint8 with a paged last dimension");
  TORCH_CHECK(page_size > 0 && page_size <= cache.size(-1) / 584 &&
                  cache.size(-1) % 4 == 0,
              "invalid page size or packed V4 page stride");
  TORCH_CHECK(token_ids.dim() == 1 && token_ids.is_contiguous() &&
                  (token_ids.scalar_type() == at::kInt ||
                   token_ids.scalar_type() == at::kLong),
              "token_ids must be contiguous int32/int64");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 && output.dim() == 3 &&
                  output.size(0) == token_ids.numel() && output.size(1) == 1 &&
                  output.size(2) == 512 && output.stride(2) == 1 &&
                  output.stride(0) >= 512,
              "output must be BF16 [num_tokens, 1, 512] with unit column "
              "stride and nonoverlapping rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0,
              "cache must be 4-byte aligned");
  if (valid_count.has_value()) {
    const auto &count = *valid_count;
    TORCH_CHECK(count.is_cuda() && count.device() == cache.device() &&
                    count.scalar_type() == at::kInt && count.numel() == 1 &&
                    count.is_contiguous(),
                "valid_count must be one contiguous int32 on the cache device");
  }
  if (selected_rows.has_value()) {
    const auto &rows = *selected_rows;
    TORCH_CHECK(valid_count.has_value() && rows.is_cuda() && rows.device() == cache.device() &&
                    rows.dim() == 1 && rows.is_contiguous() && rows.scalar_type() == at::kInt &&
                    rows.numel() == token_ids.numel(), "selected rows must be int32 on-device with full capacity");
    TORCH_CHECK(token_ids.numel() <= 2147483647 && page_size <= 2147483647 && cache.numel() / cache.size(-1) <= 2147483647 / page_size,
                "selected dequantization requires int32-addressable physical rows");
  }
  const int64_t n = token_ids.numel();
  if (n == 0)
    return;
  const c10::cuda::CUDAGuard guard(cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  int32_t page_shift = -1;
  if (selected_rows.has_value() && (page_size & (page_size - 1)) == 0) {
    page_shift = 0;
    for (int64_t value = page_size; value > 1; value >>= 1) ++page_shift;
  }
  const int64_t full_blocks = selected_rows.has_value() ? (n + 3) / 4 : (n + 1) / 2;
  const int64_t blocks =
      valid_count.has_value()
          ? std::min(full_blocks,
                     static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()
                                              ->multiProcessorCount) *
                         (selected_rows.has_value() ? 16 : 8))
          : full_blocks;
  TORCH_CHECK(blocks <= 2147483647, "too many token rows");
  AT_DISPATCH_INDEX_TYPES(
      token_ids.scalar_type(), "dequantize_k_cache_paged_hcu", [&] {
        auto launch = [&](auto selected_tag, auto aligned_tag) {
        constexpr bool Selected = decltype(selected_tag)::value, Aligned = decltype(aligned_tag)::value;
        if constexpr (Selected) {
          auto launch_selected = [&](auto fnuz_tag) {
            constexpr bool Fnuz = decltype(fnuz_tag)::value;
            dequantize_selected_wave_kernel<index_t, Fnuz, Aligned>
                <<<blocks, 256, 0, stream>>>(cache.data_ptr<uint8_t>(), token_ids.data_ptr<index_t>(),
                    output.data_ptr<at::BFloat16>(), static_cast<int32_t>(n), static_cast<int32_t>(page_size),
                    cache.size(-1), output.stride(0), valid_count->data_ptr<int32_t>(),
                    selected_rows->data_ptr<int32_t>(), page_shift);
          };
          if (fp8_fnuz) launch_selected(std::true_type{});
          else launch_selected(std::false_type{});
          return;
        }
        if (fp8_fnuz) {
          dequantize_k_cache_paged_hcu_kernel<index_t, true>
              <<<blocks, 256, 0, stream>>>(
                  cache.data_ptr<uint8_t>(), token_ids.data_ptr<index_t>(),
                  output.data_ptr<at::BFloat16>(), n, page_size, cache.size(-1),
                  output.stride(0),
                  valid_count.has_value() ? valid_count->data_ptr<int32_t>()
                                          : nullptr);
        } else {
          dequantize_k_cache_paged_hcu_kernel<index_t, false>
              <<<blocks, 256, 0, stream>>>(
                  cache.data_ptr<uint8_t>(), token_ids.data_ptr<index_t>(),
                  output.data_ptr<at::BFloat16>(), n, page_size, cache.size(-1),
                  output.stride(0),
                  valid_count.has_value() ? valid_count->data_ptr<int32_t>()
                                          : nullptr);
        }
        };
        if (selected_rows.has_value()) {
          if (output.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0)
            launch(std::true_type{}, std::true_type{});
          else launch(std::true_type{}, std::false_type{});
        } else launch(std::false_type{}, std::false_type{});
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequantize_k_cache_paged_hcu(const at::Tensor &cache,
                                  const at::Tensor &token_ids,
                                  at::Tensor &output, int64_t page_size,
                                  bool fp8_fnuz) {
  dequantize_k_cache_paged_hcu_impl(cache, token_ids, output, page_size,
                                    fp8_fnuz);
}

void dequantize_k_cache_paged_hcu_counted(const at::Tensor &cache,
                                          const at::Tensor &token_ids,
                                          at::Tensor &output, int64_t page_size,
                                          bool fp8_fnuz,
                                          const at::Tensor &valid_count) {
  dequantize_k_cache_paged_hcu_impl(cache, token_ids, output, page_size,
                                    fp8_fnuz, valid_count);
}

void dequantize_k_cache_paged_hcu_selected(
    const at::Tensor &cache, const at::Tensor &token_ids,
    const at::Tensor &selected_rows, const at::Tensor &valid_count,
    at::Tensor &output, int64_t page_size, bool fp8_fnuz) {
  TORCH_CHECK(cache.is_cuda(), "selected dequantization requires a GPU cache");
  const c10::cuda::CUDAGuard guard(cache.device());
#if defined(__HIP_PLATFORM_AMD__)
  TORCH_CHECK((std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx936", 6) == 0 || std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx938", 6) == 0),
              "selected dequantization is currently supported on gfx936/gfx938");
#else
  TORCH_CHECK(false, "selected dequantization is currently supported on gfx936/gfx938");
#endif
  dequantize_k_cache_paged_hcu_impl(cache, token_ids, output, page_size,
                                  fp8_fnuz, valid_count, selected_rows);
}

} // namespace native
} // namespace at
