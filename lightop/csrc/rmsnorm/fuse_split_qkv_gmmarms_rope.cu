#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace at {
namespace native {

namespace {

#define DISPATCH_BOOL(VAL, NAME, ...) \
  if (VAL) {                          \
    constexpr bool NAME = true;       \
    __VA_ARGS__();                    \
  } else {                            \
    constexpr bool NAME = false;      \
    __VA_ARGS__();                    \
  }

template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

template <typename T, int WIDTH>
__device__ __forceinline__ T warp_reduce_sum_local(T val) {
#pragma unroll
  for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor(val, mask);
  }
  return val;
}

template <typename T_ACC, typename scalar_t>
__device__ __forceinline__ T_ACC gemma_weight(
    const scalar_t* __restrict__ gamma, const int idx) {
  // GemmaRMSNorm semantics: x * (1 + weight), not x * weight.
  return static_cast<T_ACC>(1.0) + static_cast<T_ACC>(gamma[idx]);
}

template <typename T_ACC, typename scalar_t, int VEC_SIZE,
          bool HAS_RESIDUAL>
__device__ __forceinline__ T_ACC apply_residual_and_calc_sq_2vec(
    scalar_t* v0, scalar_t* v1, scalar_t* __restrict__ residual,
    const int o0, const int o1) {
  T_ACC sum = 0;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if constexpr (HAS_RESIDUAL) {
      v0[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v0[i]) + static_cast<T_ACC>(residual[o0 + i]));
      v1[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v1[i]) + static_cast<T_ACC>(residual[o1 + i]));
      residual[o0 + i] = v0[i];
      residual[o1 + i] = v1[i];
    }
    const T_ACC x0 = static_cast<T_ACC>(v0[i]);
    const T_ACC x1 = static_cast<T_ACC>(v1[i]);
    sum += x0 * x0 + x1 * x1;
  }

  return sum;
}

template <typename T_ACC, typename scalar_t, int VEC_SIZE,
          bool HAS_RESIDUAL>
__device__ __forceinline__ T_ACC apply_residual_and_calc_sq_4vec(
    scalar_t* v0, scalar_t* v1, scalar_t* v2, scalar_t* v3,
    scalar_t* __restrict__ residual, const int o0, const int o1, const int o2,
    const int o3) {
  T_ACC sum = 0;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if constexpr (HAS_RESIDUAL) {
      v0[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v0[i]) + static_cast<T_ACC>(residual[o0 + i]));
      v1[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v1[i]) + static_cast<T_ACC>(residual[o1 + i]));
      v2[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v2[i]) + static_cast<T_ACC>(residual[o2 + i]));
      v3[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v3[i]) + static_cast<T_ACC>(residual[o3 + i]));
      residual[o0 + i] = v0[i];
      residual[o1 + i] = v1[i];
      residual[o2 + i] = v2[i];
      residual[o3 + i] = v3[i];
    }

    const T_ACC x0 = static_cast<T_ACC>(v0[i]);
    const T_ACC x1 = static_cast<T_ACC>(v1[i]);
    const T_ACC x2 = static_cast<T_ACC>(v2[i]);
    const T_ACC x3 = static_cast<T_ACC>(v3[i]);
    sum += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  }

  return sum;
}

template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX,
          int VEC_SIZE, int THREADS_PER_HEAD>
__global__ void gemma_rms_rope_rot_dim_128_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key, const scalar_t* __restrict__ cos_sin_cache,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride_q, const int64_t head_stride_k,
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k, scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k, const scalar_t eps,
    const int num_tokens, const int num_heads, const int num_kv_heads,
    const int threads_per_token, const int tokens_per_block) {
  extern __shared__ char smem_buffer[];
  scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

  constexpr int HEAD_SIZE = 128;
  constexpr int HALF_ROT = 64;
  constexpr int ROT_DIM = 128;

  const int tid = threadIdx.x;
  const int local_token_idx = tid / threads_per_token;
  const int lane = tid % threads_per_token;
  const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
  const bool valid_token = global_token_idx < num_tokens;

  scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * ROT_DIM;
  const int64_t pos = valid_token ? positions[global_token_idx] : 0;

  for (int i = lane; i < ROT_DIM; i += threads_per_token) {
    my_s_cos_sin[i] = valid_token ? cos_sin_cache[pos * ROT_DIM + i]
                                  : static_cast<scalar_t>(0);
  }

  __syncthreads();

  if (!valid_token) {
    return;
  }

  const int q_boundary = num_heads * THREADS_PER_HEAD;
  const int total_threads = (num_heads + num_kv_heads) * THREADS_PER_HEAD;
  if (lane >= total_threads) {
    return;
  }

  const bool is_query = lane < q_boundary;
  const int head_idx = is_query ? lane / THREADS_PER_HEAD
                                : (lane - q_boundary) / THREADS_PER_HEAD;
  const int lane_in_head = is_query ? lane % THREADS_PER_HEAD
                                    : (lane - q_boundary) % THREADS_PER_HEAD;

  scalar_t* head_ptr =
      is_query
          ? query + global_token_idx * query_stride + head_idx * head_stride_q
          : key + global_token_idx * key_stride + head_idx * head_stride_k;

  scalar_t* residual_ptr =
      HAS_RESIDUAL
          ? (is_query ? residual_q + global_token_idx * query_stride +
                            head_idx * head_stride_q
                      : residual_k + global_token_idx * key_stride +
                            head_idx * head_stride_k)
          : nullptr;

  const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

  const int o0 = lane_in_head * VEC_SIZE;
  const int o1 = HALF_ROT + lane_in_head * VEC_SIZE;

  using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
  scalar_t low[VEC_SIZE];
  scalar_t high[VEC_SIZE];

  *reinterpret_cast<LoadT*>(low) =
      *reinterpret_cast<const LoadT*>(head_ptr + o0);
  *reinterpret_cast<LoadT*>(high) =
      *reinterpret_cast<const LoadT*>(head_ptr + o1);

  T_ACC sum_sq =
      apply_residual_and_calc_sq_2vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
          low, high, residual_ptr, o0, o1);
  sum_sq = warp_reduce_sum_local<T_ACC, THREADS_PER_HEAD>(sum_sq);
  const T_ACC inv_rms =
      c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));

  if constexpr (IS_NEOX) {
    scalar_t cos_v[VEC_SIZE];
    scalar_t sin_v[VEC_SIZE];
    *reinterpret_cast<LoadT*>(cos_v) =
        *reinterpret_cast<const LoadT*>(my_s_cos_sin + o0);
    *reinterpret_cast<LoadT*>(sin_v) =
        *reinterpret_cast<const LoadT*>(my_s_cos_sin + HALF_ROT + o0);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const T_ACC x0 =
          static_cast<T_ACC>(low[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i);
      const T_ACC x1 =
          static_cast<T_ACC>(high[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i);
      const T_ACC c = static_cast<T_ACC>(cos_v[i]);
      const T_ACC s = static_cast<T_ACC>(sin_v[i]);
      low[i] = static_cast<scalar_t>(x0 * c - x1 * s);
      high[i] = static_cast<scalar_t>(x0 * s + x1 * c);
    }
  } else {
    using LoadCacheT =
        at::native::memory::aligned_vector<scalar_t, VEC_SIZE / 2>;
    scalar_t cos_low[VEC_SIZE / 2], sin_low[VEC_SIZE / 2];
    scalar_t cos_high[VEC_SIZE / 2], sin_high[VEC_SIZE / 2];

    *reinterpret_cast<LoadCacheT*>(cos_low) =
        *reinterpret_cast<const LoadCacheT*>(my_s_cos_sin + o0 / 2);
    *reinterpret_cast<LoadCacheT*>(sin_low) =
        *reinterpret_cast<const LoadCacheT*>(my_s_cos_sin + HALF_ROT + o0 / 2);
    *reinterpret_cast<LoadCacheT*>(cos_high) =
        *reinterpret_cast<const LoadCacheT*>(my_s_cos_sin + o1 / 2);
    *reinterpret_cast<LoadCacheT*>(sin_high) =
        *reinterpret_cast<const LoadCacheT*>(my_s_cos_sin + HALF_ROT + o1 / 2);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 2) {
      const int ci = i / 2;

      const T_ACC x0 =
          static_cast<T_ACC>(low[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i);
      const T_ACC x1 =
          static_cast<T_ACC>(low[i + 1]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i + 1);
      const T_ACC c0 = static_cast<T_ACC>(cos_low[ci]);
      const T_ACC s0 = static_cast<T_ACC>(sin_low[ci]);
      low[i] = static_cast<scalar_t>(x0 * c0 - x1 * s0);
      low[i + 1] = static_cast<scalar_t>(x1 * c0 + x0 * s0);

      const T_ACC y0 =
          static_cast<T_ACC>(high[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i);
      const T_ACC y1 =
          static_cast<T_ACC>(high[i + 1]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i + 1);
      const T_ACC c1 = static_cast<T_ACC>(cos_high[ci]);
      const T_ACC s1 = static_cast<T_ACC>(sin_high[ci]);
      high[i] = static_cast<scalar_t>(y0 * c1 - y1 * s1);
      high[i + 1] = static_cast<scalar_t>(y1 * c1 + y0 * s1);
    }
  }

  *reinterpret_cast<LoadT*>(head_ptr + o0) = *reinterpret_cast<LoadT*>(low);
  *reinterpret_cast<LoadT*>(head_ptr + o1) = *reinterpret_cast<LoadT*>(high);
}

template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX,
          int VEC_SIZE, int THREADS_PER_HEAD>
__global__ void gemma_rms_rope_rot_dim_64_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key, const scalar_t* __restrict__ cos_sin_cache,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride_q, const int64_t head_stride_k,
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k, scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k, const scalar_t eps,
    const int num_tokens, const int num_heads, const int num_kv_heads,
    const int threads_per_token, const int tokens_per_block) {
  extern __shared__ char smem_buffer[];
  scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

  constexpr int HEAD_SIZE = 128;
  constexpr int ROT_DIM = 64;
  constexpr int HALF_ROT = 32;

  const int tid = threadIdx.x;
  const int local_token_idx = tid / threads_per_token;
  const int lane = tid % threads_per_token;
  const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
  const bool valid_token = global_token_idx < num_tokens;

  scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * ROT_DIM;
  const int64_t pos = valid_token ? positions[global_token_idx] : 0;

  for (int i = lane; i < ROT_DIM; i += threads_per_token) {
    my_s_cos_sin[i] = valid_token ? cos_sin_cache[pos * ROT_DIM + i]
                                  : static_cast<scalar_t>(0);
  }

  __syncthreads();

  if (!valid_token) {
    return;
  }

  const int q_boundary = num_heads * THREADS_PER_HEAD;
  const int total_threads = (num_heads + num_kv_heads) * THREADS_PER_HEAD;
  if (lane >= total_threads) {
    return;
  }

  const bool is_query = lane < q_boundary;
  const int head_idx = is_query ? lane / THREADS_PER_HEAD
                                : (lane - q_boundary) / THREADS_PER_HEAD;
  const int lane_in_head = is_query ? lane % THREADS_PER_HEAD
                                    : (lane - q_boundary) % THREADS_PER_HEAD;

  scalar_t* head_ptr =
      is_query
          ? query + global_token_idx * query_stride + head_idx * head_stride_q
          : key + global_token_idx * key_stride + head_idx * head_stride_k;

  scalar_t* residual_ptr =
      HAS_RESIDUAL
          ? (is_query ? residual_q + global_token_idx * query_stride +
                            head_idx * head_stride_q
                      : residual_k + global_token_idx * key_stride +
                            head_idx * head_stride_k)
          : nullptr;

  const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

  const int o0 = lane_in_head * VEC_SIZE;
  const int o1 = o0 + 32;
  const int o2 = o0 + 64;
  const int o3 = o0 + 96;

  using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
  scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];

  *reinterpret_cast<LoadT*>(v0) =
      *reinterpret_cast<const LoadT*>(head_ptr + o0);
  *reinterpret_cast<LoadT*>(v1) =
      *reinterpret_cast<const LoadT*>(head_ptr + o1);
  *reinterpret_cast<LoadT*>(v2) =
      *reinterpret_cast<const LoadT*>(head_ptr + o2);
  *reinterpret_cast<LoadT*>(v3) =
      *reinterpret_cast<const LoadT*>(head_ptr + o3);

  T_ACC sum_sq =
      apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
          v0, v1, v2, v3, residual_ptr, o0, o1, o2, o3);
  sum_sq = warp_reduce_sum_local<T_ACC, THREADS_PER_HEAD>(sum_sq);
  const T_ACC inv_rms =
      c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));

  if constexpr (IS_NEOX) {
    scalar_t cos_v[VEC_SIZE], sin_v[VEC_SIZE];
    *reinterpret_cast<LoadT*>(cos_v) =
        *reinterpret_cast<const LoadT*>(my_s_cos_sin + o0);
    *reinterpret_cast<LoadT*>(sin_v) =
        *reinterpret_cast<const LoadT*>(my_s_cos_sin + HALF_ROT + o0);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const T_ACC x0 =
          static_cast<T_ACC>(v0[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i);
      const T_ACC x1 =
          static_cast<T_ACC>(v1[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i);
      const T_ACC c = static_cast<T_ACC>(cos_v[i]);
      const T_ACC s = static_cast<T_ACC>(sin_v[i]);
      v0[i] = static_cast<scalar_t>(x0 * c - x1 * s);
      v1[i] = static_cast<scalar_t>(x0 * s + x1 * c);

      v2[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v2[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o2 + i));
      v3[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v3[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o3 + i));
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 2) {
      const int c0_idx = (o0 + i) / 2;
      const T_ACC c0 = static_cast<T_ACC>(my_s_cos_sin[c0_idx]);
      const T_ACC s0 = static_cast<T_ACC>(my_s_cos_sin[HALF_ROT + c0_idx]);
      const T_ACC x0 =
          static_cast<T_ACC>(v0[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i);
      const T_ACC x1 =
          static_cast<T_ACC>(v0[i + 1]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o0 + i + 1);
      v0[i] = static_cast<scalar_t>(x0 * c0 - x1 * s0);
      v0[i + 1] = static_cast<scalar_t>(x1 * c0 + x0 * s0);

      const int c1_idx = (o1 + i) / 2;
      const T_ACC c1 = static_cast<T_ACC>(my_s_cos_sin[c1_idx]);
      const T_ACC s1 = static_cast<T_ACC>(my_s_cos_sin[HALF_ROT + c1_idx]);
      const T_ACC y0 =
          static_cast<T_ACC>(v1[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i);
      const T_ACC y1 =
          static_cast<T_ACC>(v1[i + 1]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o1 + i + 1);
      v1[i] = static_cast<scalar_t>(y0 * c1 - y1 * s1);
      v1[i + 1] = static_cast<scalar_t>(y1 * c1 + y0 * s1);
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      v2[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v2[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o2 + i));
      v3[i] = static_cast<scalar_t>(
          static_cast<T_ACC>(v3[i]) * inv_rms *
          gemma_weight<T_ACC, scalar_t>(gamma_ptr, o3 + i));
    }
  }

  *reinterpret_cast<LoadT*>(head_ptr + o0) = *reinterpret_cast<LoadT*>(v0);
  *reinterpret_cast<LoadT*>(head_ptr + o1) = *reinterpret_cast<LoadT*>(v1);
  *reinterpret_cast<LoadT*>(head_ptr + o2) = *reinterpret_cast<LoadT*>(v2);
  *reinterpret_cast<LoadT*>(head_ptr + o3) = *reinterpret_cast<LoadT*>(v3);
}

template <typename T_ACC, typename scalar_t>
void launch_gemma_rms_rope(
    const int64_t* positions, scalar_t* query, scalar_t* key,
    const scalar_t* cos_sin_cache, const int rot_dim,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride_q, const int64_t head_stride_k,
    const scalar_t* gamma_q, const scalar_t* gamma_k,
    scalar_t* residual_q, scalar_t* residual_k, const scalar_t eps,
    const int num_tokens, const bool is_neox, const int num_heads,
    const int num_kv_heads, cudaStream_t stream) {
  constexpr int THREADS_PER_HEAD = 8;
  constexpr int TARGET_BLOCK_SIZE = 512;

  const int threads_per_token = (num_heads + num_kv_heads) * THREADS_PER_HEAD;
  TORCH_CHECK(threads_per_token > 0 && threads_per_token <= 1024,
              "unsupported head count for fused Gemma rms_rope: num_heads=",
              num_heads, ", num_kv_heads=", num_kv_heads,
              ", threads_per_token=", threads_per_token);

  const int tokens_per_block = std::max(1, TARGET_BLOCK_SIZE / threads_per_token);
  const int actual_block_size = tokens_per_block * threads_per_token;
  const int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
  const bool has_residual = residual_q != nullptr && residual_k != nullptr;

  if (rot_dim == 128) {
    constexpr int VEC_SIZE = 8;
    const size_t smem_size = tokens_per_block * 128 * sizeof(scalar_t);
    DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
      DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
        gemma_rms_rope_rot_dim_128_kernel<
            T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST, VEC_SIZE,
            THREADS_PER_HEAD><<<grid_size, actual_block_size, smem_size,
                                stream>>>(
            positions, query, key, cos_sin_cache, query_stride, key_stride,
            head_stride_q, head_stride_k, gamma_q, gamma_k, residual_q,
            residual_k, eps, num_tokens, num_heads, num_kv_heads,
            threads_per_token, tokens_per_block);
      });
    });
  } else if (rot_dim == 64) {
    constexpr int VEC_SIZE = 4;
    const size_t smem_size = tokens_per_block * 64 * sizeof(scalar_t);
    DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
      DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
        gemma_rms_rope_rot_dim_64_kernel<
            T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST, VEC_SIZE,
            THREADS_PER_HEAD><<<grid_size, actual_block_size, smem_size,
                                stream>>>(
            positions, query, key, cos_sin_cache, query_stride, key_stride,
            head_stride_q, head_stride_k, gamma_q, gamma_k, residual_q,
            residual_k, eps, num_tokens, num_heads, num_kv_heads,
            threads_per_token, tokens_per_block);
      });
    });
  } else {
    TORCH_CHECK(false,
                "unsupported rot_dim for fused Gemma rms_rope: ", rot_dim,
                ". Expected 64 or 128.");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct QKVMeta {
  int64_t num_tokens;
  int64_t query_stride;
  int64_t key_stride;
  int64_t head_stride_q;
  int64_t head_stride_k;
  int num_heads;
  int num_kv_heads;
  int rot_dim;
};

QKVMeta validate_and_get_meta(const Tensor& positions, const Tensor& query,
                              const Tensor& key, const Tensor& cos_sin_cache,
                              const int64_t head_size) {
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64");
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && cos_sin_cache.is_cuda(),
              "query, key and cos_sin_cache must be CUDA/HIP tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == cos_sin_cache.scalar_type(),
              "query, key and cos_sin_cache must have the same dtype");
  TORCH_CHECK(head_size == 128,
              "this fused Gemma kernel currently supports head_size=128, got ",
              head_size);

  const int64_t num_tokens = positions.numel();
  const int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must be [num_tokens] or [batch_size, seq_len]");

  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == num_tokens && key.size(0) == num_tokens,
                "query/key and positions must have the same token count");
  } else {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    key.size(0) == positions.size(0) &&
                    query.size(1) == positions.size(1) &&
                    key.size(1) == positions.size(1),
                "query/key and positions must have the same batch/seq shape");
  }

  const int64_t query_hidden_size = query.numel() / num_tokens;
  const int64_t key_hidden_size = key.numel() / num_tokens;
  TORCH_CHECK(query_hidden_size % head_size == 0,
              "query hidden size must be divisible by head_size");
  TORCH_CHECK(key_hidden_size % head_size == 0,
              "key hidden size must be divisible by head_size");

  const int num_heads = static_cast<int>(query_hidden_size / head_size);
  const int num_kv_heads = static_cast<int>(key_hidden_size / head_size);
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");

  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  TORCH_CHECK(rot_dim == 64 || rot_dim == 128,
              "cos_sin_cache.size(1) must be 64 or 128, got ", rot_dim);

  const int query_ndim = query.dim();
  const int key_ndim = key.dim();
  TORCH_CHECK(query.stride(-1) == 1 && key.stride(-1) == 1,
              "query/key last dimension must be contiguous");

  if (query_ndim == positions_ndim + 2) {
    TORCH_CHECK(query.stride(-2) == head_size,
                "query head dimension must be contiguous");
  }
  if (key_ndim == positions_ndim + 2) {
    TORCH_CHECK(key.stride(-2) == head_size,
                "key head dimension must be contiguous");
  }

  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.stride(seq_dim_idx);
  const int64_t head_stride_q =
      query_ndim == positions_ndim + 2 ? query.stride(-2) : head_size;
  const int64_t head_stride_k =
      key_ndim == positions_ndim + 2 ? key.stride(-2) : head_size;

  return QKVMeta{num_tokens,
                 query_stride,
                 key_stride,
                 head_stride_q,
                 head_stride_k,
                 num_heads,
                 num_kv_heads,
                 rot_dim};
}

}  // namespace

void gemma_rms_rotary_embedding_fuse(Tensor& positions, Tensor& query,
                                     Tensor& key, int64_t head_size,
                                     Tensor& cos_sin_cache, bool is_neox,
                                     Tensor weight_q, Tensor weight_k,
                                     std::optional<Tensor> residual_q,
                                     std::optional<Tensor> residual_k,
                                     double epsilon) {
  TORCH_CHECK(key.defined(), "key must be defined");
  TORCH_CHECK(!residual_q.has_value() == !residual_k.has_value(),
              "residual_q and residual_k must be both set or both unset");

  const QKVMeta meta =
      validate_and_get_meta(positions, query, key, cos_sin_cache, head_size);
  TORCH_CHECK(weight_q.numel() == head_size && weight_k.numel() == head_size,
              "weight_q/weight_k must have head_size elements");

  
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "gemma_rms_rotary_embedding_fuse", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        scalar_t* res_q_ptr =
            residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
        scalar_t* res_k_ptr =
            residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;

        launch_gemma_rms_rope<T_ACC, scalar_t>(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            meta.rot_dim, meta.query_stride, meta.key_stride,
            meta.head_stride_q, meta.head_stride_k,
            weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
            res_q_ptr, res_k_ptr, static_cast<scalar_t>(epsilon),
            static_cast<int>(meta.num_tokens), is_neox, meta.num_heads,
            meta.num_kv_heads, stream);
      });
}

std::tuple<Tensor, Tensor, Tensor> gemma_qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, Tensor& qkv, int64_t q_size, int64_t kv_size,
    int64_t head_size, Tensor& cos_sin_cache, bool is_neox, Tensor weight_q,
    Tensor weight_k, std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k, double epsilon) {
  TORCH_CHECK(qkv.is_cuda() && cos_sin_cache.is_cuda(),
              "qkv and cos_sin_cache must be CUDA/HIP tensors");
  TORCH_CHECK(qkv.size(-1) == q_size + 2 * kv_size,
              "qkv last dimension size mismatch");
  TORCH_CHECK(!residual_q.has_value() == !residual_k.has_value(),
              "residual_q and residual_k must be both set or both unset");

  Tensor query = qkv.narrow(-1, 0, q_size);
  Tensor key = qkv.narrow(-1, q_size, kv_size);
  Tensor value = qkv.narrow(-1, q_size + kv_size, kv_size);

  const QKVMeta meta =
      validate_and_get_meta(positions, query, key, cos_sin_cache, head_size);
  TORCH_CHECK(weight_q.numel() == head_size && weight_k.numel() == head_size,
              "weight_q/weight_k must have head_size elements");

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "gemma_qkv_split_rms_rotary_embedding_fuse", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        scalar_t* res_q_ptr =
            residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
        scalar_t* res_k_ptr =
            residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;

        launch_gemma_rms_rope<T_ACC, scalar_t>(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            meta.rot_dim, meta.query_stride, meta.key_stride,
            meta.head_stride_q, meta.head_stride_k,
            weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
            res_q_ptr, res_k_ptr, static_cast<scalar_t>(epsilon),
            static_cast<int>(meta.num_tokens), is_neox, meta.num_heads,
            meta.num_kv_heads, stream);
      });

  return std::make_tuple(query, key, value);
}

}  // namespace native
}  // namespace at
