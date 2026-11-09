// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Retained GLM prefill implementation; mode 1 selects aligned loads.
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/cuda.h>
#include <torch/extension.h>
#if defined(TORCH_VERSION_OVER_2120)
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif
template <typename scalar_t>
static __device__ inline float to_float(scalar_t in) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    return in;
  } else {
    union {
      uint32_t int32;
      float fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template <typename scalar_t>
static __device__ inline void from_float(scalar_t &out, float f) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    out = f;
  } else {
    uint32_t u = *(uint32_t *)(&f);
    // u += 0x7fff + ((u >> 16) & 1);
    u += 0x8000;
    out = u >> 16;
  }
}

static __device__ inline float clamp_with_limit(float value, float limit) {
  return fminf(fmaxf(value, -limit), limit);
}

// Force exactly one 16-byte transaction per aligned input chunk.
using Word4 = uint32_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ Word4 load16(const void *ptr) {
  Word4 value;
  asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)"
               : "=v"(value)
               : "v"(ptr));
  return value;
}
namespace lightop_glm53_add_rms {
template <typename T, int reducesize = C10_WARP_SIZE>
__inline__ __device__ T WarpReduceSum_NEW(T val) {
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T, int block_size = 512>
__inline__ __device__ T BlockReduceSum_NEW(T val, T *shared) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  val = WarpReduceSum_NEW<T>(val);
  if constexpr (block_size == C10_WARP_SIZE) {
    return val;
  } else {
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0 && lid < share_size) {
      val = WarpReduceSum_NEW<T, share_size>(shared[lid]);
    }
    return val;
  }
}

template <typename scalar_t, typename T_ACC, int Vec = 4, int block_size = 512,
          bool IS_GEMMA = false, bool APPLY_CLAMP = false>
__global__ void fused_add_rms_kernel_opt(scalar_t *input, scalar_t *residual,
                                         scalar_t *gamma, int cols, T_ACC eps,
                                         int num_tokens, T_ACC limit) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  int i = blockIdx.x;
  int j = threadIdx.x;
  for (; i < num_tokens; i += gridDim.x) {
    T_ACC val = 0;
    int tcol = cols / Vec;
    scalar_t intput_vec[Vec];
    scalar_t residual_vec[Vec];
    T_ACC summed_vec[Vec];
    T_ACC trstd;
    int64_t idx = i * tcol + j;
    idx *= Vec;
    if (j < tcol) {
      *(LoadT *)intput_vec = *(LoadT *)(input + idx);
      *(LoadT *)residual_vec = *(LoadT *)(residual + idx);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        summed_vec[ii] = static_cast<T_ACC>(to_float(intput_vec[ii])) +
                         static_cast<T_ACC>(to_float(residual_vec[ii]));
        val += summed_vec[ii] * summed_vec[ii];
      }
    }
    val = BlockReduceSum_NEW<T_ACC, block_size>(val, val_shared);
    if (j == 0)
      s_rstd = c10::cuda::compat::rsqrt(val / cols + eps);
    __syncthreads();
    trstd = s_rstd;
    if (j < tcol) {
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        int jj = j * Vec + ii;
        T_ACC w_val = to_float(gamma[jj]);
        if constexpr (IS_GEMMA) {
          w_val += 1.0f;
        }
        T_ACC normalized = summed_vec[ii] * trstd * w_val;
        if constexpr (APPLY_CLAMP) {
          normalized = clamp_with_limit(normalized, limit);
        }
        from_float<scalar_t>(residual_vec[ii], summed_vec[ii]);
        from_float<scalar_t>(intput_vec[ii], normalized);
      }
      *(LoadT *)(residual + idx) = *(LoadT *)residual_vec;
      *(LoadT *)(input + idx) = *(LoadT *)intput_vec;
    }
  }
}
template <typename scalar_t, typename T_ACC, int Vec = 4, int block_size = 512,
          bool IS_GEMMA = false, bool APPLY_CLAMP = false>
__global__ void
fused_add_rms_kernel_align16(scalar_t *input, scalar_t *residual,
                             scalar_t *gamma, int cols, T_ACC eps,
                             int num_tokens, T_ACC limit) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  int i = blockIdx.x;
  int j = threadIdx.x;
  for (; i < num_tokens; i += gridDim.x) {
    T_ACC val = 0;
    int tcol = cols / Vec;
    alignas(16) scalar_t intput_vec[Vec];
    alignas(16) scalar_t residual_vec[Vec];
    T_ACC summed_vec[Vec];
    T_ACC trstd;
    int64_t idx = i * tcol + j;
    idx *= Vec;
    if (j < tcol) {
      *(Word4 *)intput_vec = load16(input + idx);
      *(Word4 *)residual_vec = load16(residual + idx);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        summed_vec[ii] = static_cast<T_ACC>(to_float(intput_vec[ii])) +
                         static_cast<T_ACC>(to_float(residual_vec[ii]));
        val += summed_vec[ii] * summed_vec[ii];
      }
    }
    val = BlockReduceSum_NEW<T_ACC, block_size>(val, val_shared);
    if (j == 0)
      s_rstd = c10::cuda::compat::rsqrt(val / cols + eps);
    __syncthreads();
    trstd = s_rstd;
    if (j < tcol) {
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        int jj = j * Vec + ii;
        T_ACC w_val = to_float(gamma[jj]);
        if constexpr (IS_GEMMA) {
          w_val += 1.0f;
        }
        T_ACC normalized = summed_vec[ii] * trstd * w_val;
        if constexpr (APPLY_CLAMP) {
          normalized = clamp_with_limit(normalized, limit);
        }
        from_float<scalar_t>(residual_vec[ii], summed_vec[ii]);
        from_float<scalar_t>(intput_vec[ii], normalized);
      }
      *(LoadT *)(residual + idx) = *(LoadT *)residual_vec;
      *(LoadT *)(input + idx) = *(LoadT *)intput_vec;
    }
  }
}

} // namespace lightop_glm53_add_rms

void glm53_add_rms(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
         double epsilon, int mode) {
  TORCH_CHECK(mode == 0 || mode == 1);
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda());
  TORCH_CHECK(x.device() == residual.device() && x.device() == weight.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
              residual.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 6144 &&
              residual.sizes() == x.sizes() && weight.numel() == 6144);
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() &&
              weight.is_contiguous());
  TORCH_CHECK(x.data_ptr() != residual.data_ptr() &&
              x.data_ptr() != weight.data_ptr() &&
              residual.data_ptr() != weight.data_ptr());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int tokens = x.size(0), grid = tokens == 288 ? 256 : tokens;
  if (!tokens)
    return;
  if (mode == 0)
    lightop_glm53_add_rms::fused_add_rms_kernel_opt<c10::BFloat16, float, 8, 1024, false,
                                         false><<<grid, 1024, 0, stream>>>(
        x.data_ptr<c10::BFloat16>(), residual.data_ptr<c10::BFloat16>(),
        weight.data_ptr<c10::BFloat16>(), 6144, static_cast<float>(epsilon),
        tokens, -1.f);
  else
    lightop_glm53_add_rms::fused_add_rms_kernel_align16<c10::BFloat16, float, 8, 1024,
                                             false, false>
        <<<grid, 1024, 0, stream>>>(x.data_ptr<c10::BFloat16>(),
                                    residual.data_ptr<c10::BFloat16>(),
                                    weight.data_ptr<c10::BFloat16>(), 6144,
                                    static_cast<float>(epsilon), tokens, -1.f);
}
