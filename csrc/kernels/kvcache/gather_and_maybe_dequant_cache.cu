// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

// Migrated from vLLM v0.21.0 (8c979d2e).
//
// The LightOp adaptation is intentionally limited to replacing vLLM-local
// headers/types and exporting through LightOp.  Cache addressing, launch
// configuration, and vector width remain aligned with vLLM's original
// implementation.  LightOp additionally supports E5M2 through the same
// output-type dispatch as the existing E4M3 path.
namespace lightop_gather_cache {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

inline Fp8KVCacheDataType get_fp8_kv_cache_data_type(
    const std::string& dtype_str) {
  if (dtype_str == "auto" || dtype_str == "float16" ||
      dtype_str == "bfloat16") {
    return Fp8KVCacheDataType::kAuto;
  } else if (dtype_str == "fp8" || dtype_str == "fp8_ds_mla" ||
             dtype_str == "fp8_e4m3") {
    return Fp8KVCacheDataType::kFp8E4M3;
  } else if (dtype_str == "fp8_e5m2") {
    return Fp8KVCacheDataType::kFp8E5M2;
  }
  TORCH_CHECK(false, "Unsupported fp8 kv cache data type: ", dtype_str);
}

template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

#ifndef HIP_FP8_TYPE_OCP
#define HIP_FP8_TYPE_OCP 0
#endif

#if HIP_FP8_TYPE_OCP
using fp8_e4m3_type = __hip_fp8_e4m3;
using fp8_e5m2_type = __hip_fp8_e5m2;
#else
using fp8_e4m3_type = __hip_fp8_e4m3_fnuz;
using fp8_e5m2_type = __hip_fp8_e5m2_fnuz;
#endif

template <typename Tout, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_fp8_to_output(const uint8_t& x,
                                                 const float scale) {
  float value;
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    fp8_e4m3_type f8;
    f8.__x = x;
    value = static_cast<float>(f8);
  } else {
    static_assert(kv_dt == Fp8KVCacheDataType::kFp8E5M2);
    fp8_e5m2_type f8;
    f8.__x = x;
    value = static_cast<float>(f8);
  }
  value *= scale;

  if constexpr (std::is_same_v<Tout, float>) {
    return value;
  } else if constexpr (std::is_same_v<Tout, uint16_t>) {
    __half result = __float2half(value);
    return reinterpret_cast<const uint16_t&>(result);
  } else if constexpr (std::is_same_v<Tout, __hip_bfloat16>) {
    return __float2bfloat16(value);
  } else {
    static_assert(!sizeof(Tout), "Unsupported FP8 output type");
    return {};
  }
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3 ||
                kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    static_assert(std::is_same_v<Tin, uint8_t>);
    return scaled_fp8_to_output<Tout, kv_dt>(x, scale);
  }
  assert(false);
  return {};
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vec_n_t<cache_t, vec_size>;
  using stype = vec_n_t<scalar_t, vec_size>;
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }

    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace lightop_gather_cache

#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ)              \
  lightop_gather_cache::gather_and_maybe_dequant_cache<                       \
      SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ, thread_block_size>                \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                    \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                         \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),    \
          token_to_seq.data_ptr<int32_t>(), num_tokens, block_size,            \
          block_table_stride, cache_block_stride, cache_entry_stride,          \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()),  \
          seq_starts_ptr)

#define CALL_GATHER_CACHE_576(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 576)

#define CALL_GATHER_CACHE_320(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 320)

#define DISPATCH_GATHER_CACHE(SRC_DTYPE, KV_DTYPE, FN)                        \
  do {                                                                         \
    lightop_gather_cache::Fp8KVCacheDataType kv_cache_type =                   \
        lightop_gather_cache::get_fp8_kv_cache_data_type(KV_DTYPE);            \
    if (kv_cache_type ==                                                       \
        lightop_gather_cache::Fp8KVCacheDataType::kAuto) {                     \
      if ((SRC_DTYPE) == at::ScalarType::Float) {                              \
        FN(float, float, lightop_gather_cache::Fp8KVCacheDataType::kAuto);     \
      } else if ((SRC_DTYPE) == at::ScalarType::Half) {                        \
        FN(uint16_t, uint16_t,                                                 \
           lightop_gather_cache::Fp8KVCacheDataType::kAuto);                   \
      } else if ((SRC_DTYPE) == at::ScalarType::BFloat16) {                    \
        FN(__hip_bfloat16, __hip_bfloat16,                                     \
           lightop_gather_cache::Fp8KVCacheDataType::kAuto);                   \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ",            \
                    (SRC_DTYPE));                                              \
      }                                                                        \
    } else if (kv_cache_type ==                                                \
               lightop_gather_cache::Fp8KVCacheDataType::kFp8E4M3) {          \
      if ((SRC_DTYPE) == at::ScalarType::Float) {                              \
        FN(float, uint8_t,                                                      \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if ((SRC_DTYPE) == at::ScalarType::Half) {                        \
        FN(uint16_t, uint8_t,                                                   \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if ((SRC_DTYPE) == at::ScalarType::BFloat16) {                    \
        FN(__hip_bfloat16, uint8_t,                                             \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E4M3);                \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ",            \
                    (SRC_DTYPE));                                              \
      }                                                                        \
    } else if (kv_cache_type ==                                                \
               lightop_gather_cache::Fp8KVCacheDataType::kFp8E5M2) {          \
      if ((SRC_DTYPE) == at::ScalarType::Float) {                              \
        FN(float, uint8_t,                                                      \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E5M2);                \
      } else if ((SRC_DTYPE) == at::ScalarType::Half) {                        \
        FN(uint16_t, uint8_t,                                                   \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E5M2);                \
      } else if ((SRC_DTYPE) == at::ScalarType::BFloat16) {                    \
        FN(__hip_bfloat16, uint8_t,                                             \
           lightop_gather_cache::Fp8KVCacheDataType::kFp8E5M2);                \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ",            \
                    (SRC_DTYPE));                                              \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", (KV_DTYPE)); \
    }                                                                          \
  } while (false)

namespace at {
namespace native {

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts offsets the starting position for each batch.
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& dst,
    torch::Tensor const& block_table,
    torch::Tensor const& cu_seq_lens,
    torch::Tensor const& token_to_seq,
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts) {
  TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on a GPU");
  TORCH_CHECK(dst.is_cuda(), "dst must be on a GPU");
  TORCH_CHECK(src_cache.dim() >= 3,
              "src_cache must have shape [num_blocks, block_size, "
              "entries...]");
  TORCH_CHECK(dst.dim() == 2,
              "dst must have shape [num_tokens, head_dim]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1D");
  TORCH_CHECK(token_to_seq.dim() == 1, "token_to_seq must be 1D");

  TORCH_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(),
              "num_tokens exceeds the int32 kernel limit");
  TORCH_CHECK(dst.size(0) >= num_tokens,
              "dst has fewer rows than num_tokens");
  TORCH_CHECK(token_to_seq.numel() >= num_tokens,
              "token_to_seq has fewer entries than num_tokens");
  TORCH_CHECK(cu_seq_lens.numel() >= 2,
              "cu_seq_lens must contain at least two entries");
  const int64_t batch_size = cu_seq_lens.numel() - 1;
  TORCH_CHECK(block_table.size(0) >= batch_size,
              "block_table has fewer rows than the sequence batch");

  const int32_t block_size = src_cache.size(1);
  const int32_t head_dim = dst.size(1);
  TORCH_CHECK(block_size > 0, "src_cache block_size must be positive");

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  TORCH_CHECK(token_to_seq.dtype() == torch::kInt32,
              "token_to_seq must be int32");
  TORCH_CHECK(scale.dtype() == torch::kFloat32,
              "scale must be float32");
  TORCH_CHECK(scale.numel() == 1,
              "scale must contain exactly one element");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
    TORCH_CHECK(seq_starts.value().dim() == 1,
                "seq_starts must be 1D");
    TORCH_CHECK(seq_starts.value().numel() >= batch_size,
                "seq_starts has fewer entries than the sequence batch");
  }
  TORCH_CHECK(
      head_dim == 320 || head_dim == 576,
      "gather_and_maybe_dequant_cache only support the head_dim to 320 or "
      "576 for better performance");

  TORCH_CHECK(dst.dtype() == torch::kFloat32 ||
                  dst.dtype() == torch::kFloat16 ||
                  dst.dtype() == torch::kBFloat16,
              "dst must be float32, float16, or bfloat16");
  const auto kv_cache_type =
      lightop_gather_cache::get_fp8_kv_cache_data_type(kv_cache_dtype);
  if (kv_cache_type == lightop_gather_cache::Fp8KVCacheDataType::kAuto) {
    TORCH_CHECK(src_cache.dtype() == dst.dtype(),
                "auto cache mode requires src_cache and dst to have the "
                "same dtype");
  } else {
    TORCH_CHECK(src_cache.element_size() == 1,
                "FP8 cache modes require one-byte src_cache storage");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(src_cache.device() == token_to_seq.device(),
              "src_cache and token_to_seq must be on the same device");
  TORCH_CHECK(src_cache.device() == scale.device(),
              "src_cache and scale must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  TORCH_CHECK(dst.stride(1) == 1,
              "dst head dimension must be contiguous");
  TORCH_CHECK(block_table.stride(1) == 1,
              "block_table rows must be contiguous");
  TORCH_CHECK(cu_seq_lens.is_contiguous(),
              "cu_seq_lens must be contiguous");
  TORCH_CHECK(token_to_seq.is_contiguous(),
              "token_to_seq must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().is_contiguous(),
                "seq_starts must be contiguous");
  }

  int64_t contiguous_entry_elements = 1;
  for (int64_t dim = src_cache.dim() - 1; dim >= 2; --dim) {
    TORCH_CHECK(src_cache.stride(dim) == contiguous_entry_elements,
                "src_cache entry dimensions must be contiguous");
    contiguous_entry_elements *= src_cache.size(dim);
  }
  TORCH_CHECK(contiguous_entry_elements >= head_dim,
              "src_cache entries are smaller than dst head_dim");

  if (num_tokens == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (head_dim == 576) {
    DISPATCH_GATHER_CACHE(dst.dtype(), kv_cache_dtype, CALL_GATHER_CACHE_576);
  } else {
    DISPATCH_GATHER_CACHE(dst.dtype(), kv_cache_dtype, CALL_GATHER_CACHE_320);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at

#undef DISPATCH_GATHER_CACHE
#undef CALL_GATHER_CACHE_320
#undef CALL_GATHER_CACHE_576
#undef CALL_GATHER_CACHE
