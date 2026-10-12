// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <optional>
#include <string>
#include <type_traits>

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {

enum class GateActivation : int {
  kNone = 0,
  kSilu = 1,
  kSigmoid = 2,
};

template <typename scalar_t>
static __device__ __forceinline__ float ln1p_to_float(const scalar_t in) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    return static_cast<float>(in);
  } else {
    union {
      uint32_t int32;
      float fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template <typename scalar_t, typename value_t>
static __device__ __forceinline__ void ln1p_from_float(scalar_t& out,
                                                       const value_t value) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    out = static_cast<scalar_t>(value);
  } else {
    const float f = static_cast<float>(value);
    uint32_t u = *(uint32_t*)(&f);
    u += 0x8000;
    out = u >> 16;
  }
}

static __device__ __forceinline__ float ln1p_sigmoid(const float x) {
  constexpr float LOG2E = 1.44269504088896340736f;
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x * LOG2E));
}

template <GateActivation ACT>
static __device__ __forceinline__ float ln1p_gate(const float z) {
  if constexpr (ACT == GateActivation::kSilu) {
    return z * ln1p_sigmoid(z);
  } else if constexpr (ACT == GateActivation::kSigmoid) {
    return ln1p_sigmoid(z);
  } else {
    return 1.0f;
  }
}

template <typename T, int reducesize = C10_WARP_SIZE>
__inline__ __device__ T ln1p_warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T, int block_size = 512>
__inline__ __device__ T ln1p_block_reduce_sum(T val, T* shared) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  val = ln1p_warp_reduce_sum<T>(val);

  if constexpr (block_size > C10_WARP_SIZE) {
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0 && lid < share_size) {
      val = ln1p_warp_reduce_sum<T, share_size>(shared[lid]);
    }
  }
  return val;
}

template <typename T>
__inline__ __device__ T ln1p_shfl_xor(T val,
                                      const int lane_mask,
                                      const int width) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  return __shfl_xor(val, lane_mask, width);
#else
  return __shfl_xor_sync(0xffffffff, val, lane_mask, width);
#endif
}

template <typename T, int THREADS_PER_ROW>
__inline__ __device__ T ln1p_row_group_reduce_sum(T val) {
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
    val += ln1p_shfl_xor(val, mask, THREADS_PER_ROW);
  }
  return val;
}

template <typename scalar_t, typename T_ACC, int THREADS_PER_ROW,
          int block_size, int ELEMS_PER_THREAD, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_small_head_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps) {
  constexpr int ROWS_PER_BLOCK = block_size / THREADS_PER_ROW;
  constexpr int CACHE_ELEMS = ELEMS_PER_THREAD > 0 ? ELEMS_PER_THREAD : 1;

  const int tid = threadIdx.x;
  const int lane = tid % THREADS_PER_ROW;
  const int row_in_block = tid / THREADS_PER_ROW;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;

  for (int64_t row_base = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
       row_base < M;
       row_base += static_cast<int64_t>(gridDim.x) * ROWS_PER_BLOCK) {
    const int64_t row = row_base + row_in_block;
    T_ACC local_sum = 0;
    T_ACC local_sq = 0;
    T_ACC x_cache[CACHE_ELEMS];

    if (row < M) {
      int item = 0;
      for (int64_t col = lane; col < N; col += THREADS_PER_ROW) {
        const int64_t x_off = row * stride_x_row + group_col + col;
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        if constexpr (!IS_RMS_NORM) {
          local_sum += x;
        }
        local_sq += x * x;
        if constexpr (ELEMS_PER_THREAD > 0) {
          x_cache[item] = x;
        }
        item++;
      }
    }

    local_sq = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sq);
    if constexpr (!IS_RMS_NORM) {
      local_sum = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sum);
    }

    T_ACC mean = 0;
    T_ACC var = local_sq / N;
    if constexpr (!IS_RMS_NORM) {
      mean = local_sum / N;
      var = var - mean * mean;
      var = var > static_cast<T_ACC>(0) ? var : static_cast<T_ACC>(0);
      if (lane == 0 && row < M) {
        Mean[group * M + row] = static_cast<float>(mean);
      }
    }
    const T_ACC rstd = c10::cuda::compat::rsqrt(var + eps);
    if (lane == 0 && row < M) {
      Rstd[group * M + row] = static_cast<float>(rstd);
    }

    if (row < M) {
      int item = 0;
      for (int64_t col = lane; col < N; col += THREADS_PER_ROW) {
        const int64_t y_off = row * stride_y_row + group_col + col;
        const int64_t w_off = group_col + col;

        T_ACC x;
        if constexpr (ELEMS_PER_THREAD > 0) {
          x = x_cache[item];
        } else {
          const int64_t x_off = row * stride_x_row + group_col + col;
          x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
          if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            const float z = ln1p_to_float(Z[z_off]);
            x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
        }

        T_ACC y = IS_RMS_NORM ? x * rstd : (x - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(W[w_off]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(B[w_off]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }

        scalar_t out;
        ln1p_from_float(out, y);
        Y[y_off] = out;
        item++;
      }
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int THREADS_PER_ROW,
          int block_size, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_small_head_vec_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps) {
  constexpr int ROWS_PER_BLOCK = block_size / THREADS_PER_ROW;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  const int tid = threadIdx.x;
  const int lane = tid % THREADS_PER_ROW;
  const int row_in_block = tid / THREADS_PER_ROW;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;

  for (int64_t row_base = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
       row_base < M;
       row_base += static_cast<int64_t>(gridDim.x) * ROWS_PER_BLOCK) {
    const int64_t row = row_base + row_in_block;
    T_ACC x_acc[Vec];
    T_ACC local_sum = 0;
    T_ACC local_sq = 0;

    if (row < M && lane < vec_cols) {
      const int64_t col = lane * Vec;
      const int64_t x_off = row * stride_x_row + group_col + col;
      scalar_t x_vec[Vec];
      *(LoadT*)x_vec = *(const LoadT*)(X + x_off);

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          const float z = ln1p_to_float(z_vec[i]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        x_acc[i] = x;
        if constexpr (!IS_RMS_NORM) {
          local_sum += x;
        }
        local_sq += x * x;
      }
    }

    local_sq = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sq);
    if constexpr (!IS_RMS_NORM) {
      local_sum = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sum);
    }

    T_ACC mean = 0;
    T_ACC var = local_sq / N;
    if constexpr (!IS_RMS_NORM) {
      mean = local_sum / N;
      var = var - mean * mean;
      var = var > static_cast<T_ACC>(0) ? var : static_cast<T_ACC>(0);
      if (lane == 0 && row < M) {
        Mean[group * M + row] = static_cast<float>(mean);
      }
    }
    const T_ACC rstd = c10::cuda::compat::rsqrt(var + eps);
    if (lane == 0 && row < M) {
      Rstd[group * M + row] = static_cast<float>(rstd);
    }

    if (row < M && lane < vec_cols) {
      const int64_t col = lane * Vec;
      const int64_t y_off = row * stride_y_row + group_col + col;
      const int64_t w_off = group_col + col;

      scalar_t w_vec[Vec];
      scalar_t y_vec[Vec];
      *(LoadT*)w_vec = *(const LoadT*)(W + w_off);

      scalar_t b_vec[Vec];
      if constexpr (HAS_BIAS) {
        *(LoadT*)b_vec = *(const LoadT*)(B + w_off);
      }

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC y = IS_RMS_NORM ? x_acc[i] * rstd : (x_acc[i] - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE) {
          const float z = ln1p_to_float(z_vec[i]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        ln1p_from_float(y_vec[i], y);
      }
      *(LoadT*)(Y + y_off) = *(LoadT*)y_vec;
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int block_size,
          bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_vec_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const int tid = threadIdx.x;
    const int64_t vec_col = tid;
    T_ACC x_acc[Vec];
    T_ACC local = 0;

    if (vec_col < vec_cols) {
      const int64_t col = vec_col * Vec;
      const int64_t x_off = row * stride_x_row + group_col + col;
      scalar_t x_vec[Vec];
      *(LoadT*)x_vec = *(const LoadT*)(X + x_off);

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          const float z = ln1p_to_float(z_vec[i]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        x_acc[i] = x;
        if constexpr (IS_RMS_NORM) {
          local += x * x;
        } else {
          local += x;
        }
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (tid == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (tid == 0) {
        s_mean = local / N;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
      if (vec_col < vec_cols) {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          const T_ACC diff = x_acc[i] - mean;
          var += diff * diff;
        }
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (tid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
    if (vec_col < vec_cols) {
      const int64_t col = vec_col * Vec;
      const int64_t y_off = row * stride_y_row + group_col + col;
      const int64_t w_off = group_col + col;
      scalar_t w_vec[Vec];
      scalar_t y_vec[Vec];
      *(LoadT*)w_vec = *(const LoadT*)(W + w_off);

      scalar_t b_vec[Vec];
      if constexpr (HAS_BIAS) {
        *(LoadT*)b_vec = *(const LoadT*)(B + w_off);
      }

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC y = IS_RMS_NORM ? x_acc[i] * rstd : (x_acc[i] - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE) {
          const float z = ln1p_to_float(z_vec[i]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        ln1p_from_float(y_vec[i], y);
      }
      *(LoadT*)(Y + y_off) = *(LoadT*)y_vec;
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int block_size,
          int ITEMS_PER_THREAD, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_vec_multi_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const int tid = threadIdx.x;
    T_ACC x_acc[ITEMS_PER_THREAD][Vec];
    T_ACC local = 0;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int64_t vec_col = tid + item * block_size;
      if (vec_col < vec_cols) {
        const int64_t col = vec_col * Vec;
        const int64_t x_off = row * stride_x_row + group_col + col;
        scalar_t x_vec[Vec];
        *(LoadT*)x_vec = *(const LoadT*)(X + x_off);

        scalar_t z_vec[Vec];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
        }

#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
          if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
            const float z = ln1p_to_float(z_vec[i]);
            x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
          x_acc[item][i] = x;
          if constexpr (IS_RMS_NORM) {
            local += x * x;
          } else {
            local += x;
          }
        }
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (tid == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (tid == 0) {
        s_mean = local / N;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int64_t vec_col = tid + item * block_size;
        if (vec_col < vec_cols) {
#pragma unroll
          for (int i = 0; i < Vec; ++i) {
            const T_ACC diff = x_acc[item][i] - mean;
            var += diff * diff;
          }
        }
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (tid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int64_t vec_col = tid + item * block_size;
      if (vec_col < vec_cols) {
        const int64_t col = vec_col * Vec;
        const int64_t y_off = row * stride_y_row + group_col + col;
        const int64_t w_off = group_col + col;

        scalar_t w_vec[Vec];
        scalar_t y_vec[Vec];
        *(LoadT*)w_vec = *(const LoadT*)(W + w_off);

        scalar_t b_vec[Vec];
        if constexpr (HAS_BIAS) {
          *(LoadT*)b_vec = *(const LoadT*)(B + w_off);
        }

        scalar_t z_vec[Vec];
        if constexpr (HAS_Z && NORM_BEFORE_GATE) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          *(LoadT*)z_vec = *(const LoadT*)(Z + z_off);
        }

#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          T_ACC y =
              IS_RMS_NORM ? x_acc[item][i] * rstd : (x_acc[item][i] - mean) * rstd;
          y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
          if constexpr (HAS_BIAS) {
            y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
          }
          if constexpr (HAS_Z && NORM_BEFORE_GATE) {
            const float z = ln1p_to_float(z_vec[i]);
            y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
          ln1p_from_float(y_vec[i], y);
        }
        *(LoadT*)(Y + y_off) = *(LoadT*)y_vec;
      }
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, typename T_ACC, int block_size,
          bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_scalar_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    T_ACC local = 0;
    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }
      if constexpr (IS_RMS_NORM) {
        local += x * x;
      } else {
        local += x;
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (threadIdx.x == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        s_mean = local / N;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
      for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
        const int64_t x_off = row * stride_x_row + group_col + col;
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        const T_ACC diff = x - mean;
        var += diff * diff;
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (threadIdx.x == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var / N + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      const int64_t y_off = row * stride_y_row + group_col + col;
      const int64_t w_off = group_col + col;

      T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }

      T_ACC y = IS_RMS_NORM ? x * rstd : (x - mean) * rstd;
      y *= static_cast<T_ACC>(ln1p_to_float(W[w_off]));
      if constexpr (HAS_BIAS) {
        y += static_cast<T_ACC>(ln1p_to_float(B[w_off]));
      }
      if constexpr (HAS_Z && NORM_BEFORE_GATE) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }

      scalar_t out;
      ln1p_from_float(out, y);
      Y[y_off] = out;
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, int Vec>
static bool ln1p_can_vectorize(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const torch::Tensor& W,
    const torch::Tensor& B,
    const torch::Tensor& Z,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t N,
    const bool HAS_BIAS,
    const bool HAS_Z) {
  const int64_t bytes = static_cast<int64_t>(sizeof(scalar_t)) * Vec;
  const auto aligned = [bytes](const torch::Tensor& t) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0;
  };

  if (N % Vec != 0 || stride_x_row % Vec != 0 || stride_y_row % Vec != 0) {
    return false;
  }
  if (HAS_Z && stride_z_row % Vec != 0) {
    return false;
  }
  if (!aligned(X) || !aligned(Y) || !aligned(W)) {
    return false;
  }
  if (HAS_BIAS && !aligned(B)) {
    return false;
  }
  if (HAS_Z && !aligned(Z)) {
    return false;
  }
  return true;
}


static int ln1p_small_head_grid_x(const int64_t M,
                                  const int rows_per_block) {
  const int64_t row_blocks = (M + rows_per_block - 1) / rows_per_block;
  TORCH_CHECK(row_blocks > 0, "row_blocks must be positive");
  int cu_count = 1;
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop != nullptr && prop->multiProcessorCount > 0) {
    cu_count = prop->multiProcessorCount;
  }
  int64_t grid_x = row_blocks;
  if (row_blocks >= static_cast<int64_t>(cu_count) * 2) {
    grid_x = (row_blocks / cu_count) * cu_count;
    grid_x = std::max<int64_t>(grid_x, cu_count);
  }
  return static_cast<int>(std::min<int64_t>(grid_x, 65535));
}

template <typename scalar_t, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
static void ln1p_launch_typed(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double epsilon,
    const cudaStream_t stream) {
  using T_ACC = at::acc_type<scalar_t, true>;
  const T_ACC eps = static_cast<T_ACC>(epsilon);

  scalar_t* x_data = X.data_ptr<scalar_t>();
  scalar_t* y_data = Y.data_ptr<scalar_t>();
  scalar_t* w_data = W.data_ptr<scalar_t>();
  scalar_t* b_data = HAS_BIAS ? B.data_ptr<scalar_t>() : nullptr;
  scalar_t* z_data = HAS_Z ? Z.data_ptr<scalar_t>() : nullptr;
  float* mean_data = (!IS_RMS_NORM && Mean.defined()) ? Mean.data_ptr<float>() : nullptr;
  float* rstd_data = Rstd.data_ptr<float>();

  if (N < 512) {
    if (M < 16384) {
      constexpr int small_block = 256;
      if (N <= 64) {
        constexpr int threads_per_row = 8;
        constexpr int rows_per_block = small_block / threads_per_row;
        const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
        dim3 grid(grid_x, static_cast<unsigned int>(groups));
        layer_norm_fwd_1pass_small_head_kernel<
            scalar_t, T_ACC, threads_per_row, small_block, 8, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps);
        return;
      }
      if (N <= 128) {
        constexpr int threads_per_row = 16;
        constexpr int rows_per_block = small_block / threads_per_row;
        const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
        dim3 grid(grid_x, static_cast<unsigned int>(groups));
        if constexpr (sizeof(scalar_t) == 2) {
          if (N == 128 &&
              ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                              stride_y_row, stride_z_row, N,
                                              HAS_BIAS, HAS_Z)) {
            layer_norm_fwd_1pass_small_head_vec_kernel<
                scalar_t, T_ACC, 8, threads_per_row, small_block, HAS_BIAS,
                HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
                <<<grid, small_block, 0, stream>>>(
                    x_data, y_data, w_data, b_data, z_data, mean_data,
                    rstd_data, stride_x_row, stride_y_row, stride_z_row, M, N,
                    eps);
            return;
          }
        }
        layer_norm_fwd_1pass_small_head_kernel<
            scalar_t, T_ACC, threads_per_row, small_block, 8, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps);
        return;
      }

      constexpr int threads_per_row = 32;
      constexpr int rows_per_block = small_block / threads_per_row;
      const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
      dim3 grid(grid_x, static_cast<unsigned int>(groups));
      layer_norm_fwd_1pass_small_head_kernel<
          scalar_t, T_ACC, threads_per_row, small_block, 16, HAS_BIAS, HAS_Z,
          NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
          <<<grid, small_block, 0, stream>>>(
              x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
              stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;

    } else {
      constexpr int small_block = 128;
      if (N <= 64) {
        constexpr int threads_per_row = 8;
        constexpr int rows_per_block = small_block / threads_per_row;
        const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
        dim3 grid(grid_x, static_cast<unsigned int>(groups));
        layer_norm_fwd_1pass_small_head_kernel<
            scalar_t, T_ACC, threads_per_row, small_block, 8, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps);
        return;
      }
      if (N <= 128) {
        constexpr int threads_per_row = 16;
        constexpr int rows_per_block = small_block / threads_per_row;
        const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
        dim3 grid(grid_x, static_cast<unsigned int>(groups));
        if constexpr (sizeof(scalar_t) == 2) {
          if (N == 128 &&
              ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                              stride_y_row, stride_z_row, N,
                                              HAS_BIAS, HAS_Z)) {
            layer_norm_fwd_1pass_small_head_vec_kernel<
                scalar_t, T_ACC, 8, threads_per_row, small_block, HAS_BIAS,
                HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
                <<<grid, small_block, 0, stream>>>(
                    x_data, y_data, w_data, b_data, z_data, mean_data,
                    rstd_data, stride_x_row, stride_y_row, stride_z_row, M, N,
                    eps);
            return;
          }
        }
        layer_norm_fwd_1pass_small_head_kernel<
            scalar_t, T_ACC, threads_per_row, small_block, 8, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps);
        return;
      }

      constexpr int threads_per_row = 32;
      constexpr int rows_per_block = small_block / threads_per_row;
      const int grid_x = ln1p_small_head_grid_x(M, rows_per_block);
      dim3 grid(grid_x, static_cast<unsigned int>(groups));
      layer_norm_fwd_1pass_small_head_kernel<
          scalar_t, T_ACC, threads_per_row, small_block, 16, HAS_BIAS, HAS_Z,
          NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
          <<<grid, small_block, 0, stream>>>(
              x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
              stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
  }


  int grid_x = static_cast<int>(std::min<int64_t>(M, 65535));
  if (M == 288) {
    grid_x = 256;
  }
  dim3 grid(grid_x, static_cast<unsigned int>(groups));

  if constexpr (sizeof(scalar_t) == 2) {
    if (N > 2048 && N <= 4096 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_multi_kernel<
          scalar_t, T_ACC, 8, 256, 2, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
          IS_RMS_NORM, ACT><<<grid, 256, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N > 4096 && N <= 8192 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_multi_kernel<
          scalar_t, T_ACC, 8, 256, 4, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
          IS_RMS_NORM, ACT><<<grid, 256, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N > 8192 && N <= 16384 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_multi_kernel<
          scalar_t, T_ACC, 8, 512, 4, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
          IS_RMS_NORM, ACT><<<grid, 512, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N <= 1024 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 128, HAS_BIAS,
                                      HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                      ACT><<<grid, 128, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N <= 2048 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 256, HAS_BIAS,
                                      HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                      ACT><<<grid, 256, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N <= 4096 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 512, HAS_BIAS,
                                      HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                      ACT><<<grid, 512, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N <= 8192 &&
        ln1p_can_vectorize<scalar_t, 8>(X, Y, W, B, Z, stride_x_row,
                                        stride_y_row, stride_z_row, N,
                                        HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 1024, HAS_BIAS,
                                      HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                      ACT><<<grid, 1024, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
    if (N <= 16384 &&
        ln1p_can_vectorize<scalar_t, 16>(X, Y, W, B, Z, stride_x_row,
                                         stride_y_row, stride_z_row, N,
                                         HAS_BIAS, HAS_Z)) {
      layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 16, 1024, HAS_BIAS,
                                      HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                      ACT><<<grid, 1024, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps);
      return;
    }
  }

  if (N <= 1024) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 128, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 128, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps);
  } else if (N <= 2048) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 256, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 256, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps);
  } else if (N <= 4096) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 512, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 512, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps);
  } else {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 1024, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 1024, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                    mean_data, rstd_data, stride_x_row,
                                    stride_y_row, stride_z_row, M, N, eps);
  }
}

template <bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
static void ln1p_dispatch_dtype(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double eps,
    const cudaStream_t stream) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "layer_norm_fwd_1pass_opt",
      [&] {
        ln1p_launch_typed<scalar_t, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
                          IS_RMS_NORM, ACT>(
            X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row,
            stride_z_row, M, N, groups, eps, stream);
      });
}

template <bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM>
static void ln1p_dispatch_activation(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double eps,
    const GateActivation activation,
    const cudaStream_t stream) {
  if (!HAS_Z || activation == GateActivation::kNone) {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kNone>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  } else if (activation == GateActivation::kSilu) {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kSilu>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  } else {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kSigmoid>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  }
}

static GateActivation ln1p_parse_activation(const std::string& activation) {
  if (activation == "swish" || activation == "silu") {
    return GateActivation::kSilu;
  }
  if (activation == "sigmoid") {
    return GateActivation::kSigmoid;
  }
  if (activation.empty() || activation == "none" || activation == "identity") {
    return GateActivation::kNone;
  }
  TORCH_CHECK(false,
              "Unsupported ACTIVATION for layer_norm_fwd_1pass_opt: ",
              activation,
              ". Expected 'swish'/'silu', 'sigmoid', or 'none'.");
  return GateActivation::kNone;
}

static void ln1p_check_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

static void ln1p_check_same_device(const torch::Tensor& t,
                                   const torch::Tensor& ref,
                                   const char* name) {
  TORCH_CHECK(t.device() == ref.device(),
              name,
              " must be on the same CUDA device as X");
}

void layer_norm_fwd_1pass_opt(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    std::optional<torch::Tensor> B_opt,
    std::optional<torch::Tensor> Z_opt,
    std::optional<torch::Tensor> Mean_opt,
    torch::Tensor& Rstd,
    int64_t stride_x_row,
    int64_t stride_y_row,
    int64_t stride_z_row,
    int64_t M,
    int64_t N,
    double eps,
    int64_t BLOCK_N,
    int64_t ROWS_PER_BLOCK,
    bool HAS_BIAS,
    bool HAS_Z,
    bool NORM_BEFORE_GATE,
    bool IS_RMS_NORM,
    const std::string& ACTIVATION) {
  TORCH_CHECK(M >= 0, "M must be non-negative");
  TORCH_CHECK(N > 0, "N must be positive");
  TORCH_CHECK(BLOCK_N >= N,
              "BLOCK_N must cover N, matching the Triton kernel assumption");
  TORCH_CHECK(ROWS_PER_BLOCK >= 1, "ROWS_PER_BLOCK must be positive");

  if (M == 0) {
    return;
  }

  torch::Tensor B = B_opt.has_value() ? *B_opt : torch::Tensor();
  torch::Tensor Z = Z_opt.has_value() ? *Z_opt : torch::Tensor();
  torch::Tensor Mean = Mean_opt.has_value() ? *Mean_opt : torch::Tensor();

  ln1p_check_tensor(X, "X");
  ln1p_check_tensor(Y, "Y");
  ln1p_check_tensor(W, "W");
  ln1p_check_tensor(Rstd, "Rstd");
  ln1p_check_same_device(Y, X, "Y");
  ln1p_check_same_device(W, X, "W");
  ln1p_check_same_device(Rstd, X, "Rstd");
  if (HAS_BIAS) {
    ln1p_check_tensor(B, "B");
    ln1p_check_same_device(B, X, "B");
  }
  if (HAS_Z) {
    ln1p_check_tensor(Z, "Z");
    ln1p_check_same_device(Z, X, "Z");
  }
  if (!IS_RMS_NORM) {
    ln1p_check_tensor(Mean, "Mean");
    ln1p_check_same_device(Mean, X, "Mean");
  }

  TORCH_CHECK(Y.scalar_type() == X.scalar_type(),
              "Y must have the same dtype as X");
  TORCH_CHECK(W.scalar_type() == X.scalar_type(),
              "W must have the same dtype as X");
  if (HAS_BIAS) {
    TORCH_CHECK(B.scalar_type() == X.scalar_type(),
                "B must have the same dtype as X");
  }
  if (HAS_Z) {
    TORCH_CHECK(Z.scalar_type() == X.scalar_type(),
                "Z must have the same dtype as X");
  }
  TORCH_CHECK(Rstd.scalar_type() == at::ScalarType::Float,
              "Rstd must be float32");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.scalar_type() == at::ScalarType::Float,
                "Mean must be float32");
  }

  TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
  if (HAS_BIAS) {
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  }
  TORCH_CHECK(Rstd.is_contiguous(), "Rstd must be contiguous");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.is_contiguous(), "Mean must be contiguous");
  }

  TORCH_CHECK(stride_x_row > 0, "stride_x_row must be positive");
  TORCH_CHECK(stride_y_row > 0, "stride_y_row must be positive");
  if (HAS_Z) {
    TORCH_CHECK(stride_z_row > 0, "stride_z_row must be positive");
  }

  TORCH_CHECK(W.numel() % N == 0,
              "W.numel() must be divisible by N to infer the group count");
  const int64_t groups = W.numel() / N;
  TORCH_CHECK(groups > 0, "group count inferred from W must be positive");
  TORCH_CHECK(groups <= 65535,
              "group count is too large for CUDA grid.y in this kernel");
  if (HAS_BIAS) {
    TORCH_CHECK(B.numel() >= groups * N,
                "B must contain at least groups * N elements");
  }
  TORCH_CHECK(Rstd.numel() >= groups * M,
              "Rstd must contain at least groups * M elements");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.numel() >= groups * M,
                "Mean must contain at least groups * M elements");
  }

  TORCH_CHECK(stride_x_row >= groups * N,
              "stride_x_row must be at least groups * N");
  TORCH_CHECK(stride_y_row >= groups * N,
              "stride_y_row must be at least groups * N");
  if (HAS_Z) {
    TORCH_CHECK(stride_z_row >= groups * N,
                "stride_z_row must be at least groups * N");
  }

  const int64_t x_required = (M - 1) * stride_x_row + groups * N;
  const int64_t y_required = (M - 1) * stride_y_row + groups * N;
  TORCH_CHECK(X.numel() >= x_required,
              "X does not contain enough elements for M/stride_x_row/groups/N");
  TORCH_CHECK(Y.numel() >= y_required,
              "Y does not contain enough elements for M/stride_y_row/groups/N");
  if (HAS_Z) {
    const int64_t z_required = (M - 1) * stride_z_row + groups * N;
    TORCH_CHECK(Z.numel() >= z_required,
                "Z does not contain enough elements for M/stride_z_row/groups/N");
  }

  const GateActivation activation = ln1p_parse_activation(ACTIVATION);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LN1P_DISPATCH_CASE(HAS_BIAS_V, HAS_Z_V, NORM_BEFORE_GATE_V, RMS_V)    \
  ln1p_dispatch_activation<HAS_BIAS_V, HAS_Z_V, NORM_BEFORE_GATE_V, RMS_V>(   \
      X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,  \
      N, groups, eps, activation, stream)

  if (HAS_BIAS) {
    if (HAS_Z) {
      if (NORM_BEFORE_GATE) {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(true, true, true, true);
        } else {
          LN1P_DISPATCH_CASE(true, true, true, false);
        }
      } else {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(true, true, false, true);
        } else {
          LN1P_DISPATCH_CASE(true, true, false, false);
        }
      }
    } else {
      if (IS_RMS_NORM) {
        LN1P_DISPATCH_CASE(true, false, false, true);
      } else {
        LN1P_DISPATCH_CASE(true, false, false, false);
      }
    }
  } else {
    if (HAS_Z) {
      if (NORM_BEFORE_GATE) {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(false, true, true, true);
        } else {
          LN1P_DISPATCH_CASE(false, true, true, false);
        }
      } else {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(false, true, false, true);
        } else {
          LN1P_DISPATCH_CASE(false, true, false, false);
        }
      }
    } else {
      if (IS_RMS_NORM) {
        LN1P_DISPATCH_CASE(false, false, false, true);
      } else {
        LN1P_DISPATCH_CASE(false, false, false, false);
      }
    }
  }

#undef LN1P_DISPATCH_CASE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
