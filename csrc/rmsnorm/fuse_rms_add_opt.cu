// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "type_convert.cuh"
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

template<typename scalar_t>
static __device__ inline float to_float(scalar_t in){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template<typename scalar_t>
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1);
    u += 0x8000;
    out = u>>16;
  }
}

static __device__ inline float clamp_with_limit(float value, float limit) {
  return fminf(fmaxf(value, -limit), limit);
}

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, bool IS_GEMMA = false, bool APPLY_CLAMP = false>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size,
    const float limit) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    if constexpr (IS_GEMMA) { w += 1.0f; }

    scalar_t normalized =
        ((scalar_t)(x * s_variance)) * ((scalar_t)w);
    if constexpr (APPLY_CLAMP) {
      normalized = (scalar_t)clamp_with_limit((float)normalized, limit);
    }
    out[blockIdx.x * hidden_size + idx] = normalized;
  }
}

template <typename scalar_t, typename weight_t, bool HAS_WEIGHT = false>
__global__ void rms_norm_optional_weight_kernel(
    scalar_t* out,                         // [..., hidden_size], may alias input
    const scalar_t* input,                 // [..., hidden_size]
    const weight_t* __restrict__ weight,   // [hidden_size] or nullptr
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const int base = blockIdx.x * hidden_size;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[base + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float y = (float)input[base + idx] * s_variance;
    if constexpr (HAS_WEIGHT) {
      y *= (float)weight[idx];
    }
    out[base + idx] = (scalar_t)y;
  }
}

/* Vectorized fallback: keep the FP32 sum for normalization. Re-read the
   original inputs after the reduction instead of storing a rounded sum. */
template <typename scalar_t, int width, bool IS_GEMMA = false, bool APPLY_CLAMP = false>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size,
    const float limit) {
  using Vec = at::native::memory::aligned_vector<scalar_t, width>;
  const int vec_hidden_size = hidden_size / width;
  auto* input_v = reinterpret_cast<Vec*>(input);
  auto* residual_v = reinterpret_cast<Vec*>(residual);
  const auto* weight_v = reinterpret_cast<const Vec*>(weight);
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    const int id = blockIdx.x * vec_hidden_size + idx;
    const Vec x = input_v[id];
    const Vec r = residual_v[id];
#pragma unroll
    for (int k = 0; k < width; ++k) {
      const float sum = (float)x.val[k] + (float)r.val[k];
      variance += sum * sum;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    const int id = blockIdx.x * vec_hidden_size + idx;
    Vec x = input_v[id];
    Vec r = residual_v[id];
    const Vec w = weight_v[idx];
#pragma unroll
    for (int k = 0; k < width; ++k) {
      const float sum = (float)x.val[k] + (float)r.val[k];
      float weight_value = (float)w.val[k];
      if constexpr (IS_GEMMA) { weight_value += 1.0f; }
      float normalized = sum * s_variance * weight_value;
      if constexpr (APPLY_CLAMP) {
        normalized = clamp_with_limit(normalized, limit);
      }
      r.val[k] = (scalar_t)sum;
      x.val[k] = (scalar_t)normalized;
    }
    residual_v[id] = r;
    input_v[id] = x;
  }
}

/* Generic fallback, including unaligned buffers and non-vectorizable rows. */
template <typename scalar_t, int width, bool IS_GEMMA = false, bool APPLY_CLAMP = false>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size,
    const float limit) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const int id = blockIdx.x * hidden_size + idx;
    const float sum = (float)input[id] + (float)residual[id];
    variance += sum * sum;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const int id = blockIdx.x * hidden_size + idx;
    const float sum = (float)input[id] + (float)residual[id];
    float w = (float)weight[idx];
    if constexpr (IS_GEMMA) { w += 1.0f; }
    float normalized = sum * s_variance * w;
    if constexpr (APPLY_CLAMP) {
      normalized = clamp_with_limit(normalized, limit);
    }
    residual[id] = (scalar_t)sum;
    input[id] = (scalar_t)normalized;
  }
}

}  // namespace vllm

namespace at{
namespace native{

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceSum_NEW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceSum_NEW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceSum_NEW<T>(val);
  if constexpr(block_size==C10_WARP_SIZE)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0&&lid<share_size) {
      val = WarpReduceSum_NEW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

#ifdef USE_ROCM
template <typename T, int width>
__inline__ __device__ T SubWarpReduceSumWithWidth(T val) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, width);
  }
  return val;
}

template <typename scalar_t, typename T_ACC, int HiddenSize,
          int ThreadsPerToken, int BlockSize=256, bool IS_GEMMA=false,
          bool APPLY_CLAMP=false>
__global__ void rms_kernel_small_hidden_multi_token(
    scalar_t* input,
    scalar_t* output,
    scalar_t* gamma,
    T_ACC eps,
    int num_tokens,
    T_ACC limit) {
  constexpr int Vec = 8;
  constexpr int TokensPerBlock = BlockSize / ThreadsPerToken;
  static_assert(HiddenSize == ThreadsPerToken * Vec);
  static_assert(ThreadsPerToken <= C10_WARP_SIZE);
  static_assert(C10_WARP_SIZE % ThreadsPerToken == 0);
  static_assert(BlockSize % C10_WARP_SIZE == 0);

  const int lane = threadIdx.x % ThreadsPerToken;
  const int token_in_block = threadIdx.x / ThreadsPerToken;
  const int token = blockIdx.x * TokensPerBlock + token_in_block;
  const bool valid_token = token < num_tokens;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  scalar_t input_vec[Vec];
  T_ACC sum = 0;
  if (valid_token) {
    const int64_t idx =
        static_cast<int64_t>(token) * HiddenSize + lane * Vec;
    *(LoadT*)input_vec = *(const LoadT*)(input + idx);
#pragma unroll
    for (int ii = 0; ii < Vec; ++ii) {
      const T_ACC x = static_cast<T_ACC>(input_vec[ii]);
      sum += x * x;
    }
  }

  sum = SubWarpReduceSumWithWidth<T_ACC, ThreadsPerToken>(sum);
  T_ACC rstd = 0;
  if (lane == 0) {
    rstd = c10::cuda::compat::rsqrt(sum / HiddenSize + eps);
  }
  rstd = __shfl(rstd, 0, ThreadsPerToken);

  if (valid_token) {
    scalar_t weight_vec[Vec];
    *(LoadT*)weight_vec = *(const LoadT*)(gamma + lane * Vec);
#pragma unroll
    for (int ii = 0; ii < Vec; ++ii) {
      T_ACC w_val = static_cast<T_ACC>(weight_vec[ii]);
      if constexpr (IS_GEMMA) { w_val += 1.0f; }
      T_ACC normalized =
          static_cast<T_ACC>(input_vec[ii]) * rstd * w_val;
      if constexpr (APPLY_CLAMP) {
        normalized = clamp_with_limit(normalized, limit);
      }
      input_vec[ii] = normalized;
    }
    const int64_t idx =
        static_cast<int64_t>(token) * HiddenSize + lane * Vec;
    *(LoadT*)(output + idx) = *(LoadT*)input_vec;
  }
}

template <typename scalar_t, typename T_ACC, int HiddenSize,
          int ThreadsPerToken, int BlockSize=256, bool IS_GEMMA=false,
          bool APPLY_CLAMP=false>
__global__ void fused_add_rms_kernel_small_hidden_multi_token(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    T_ACC eps,
    int num_tokens,
    T_ACC limit) {
  constexpr int Vec = 8;
  constexpr int TokensPerBlock = BlockSize / ThreadsPerToken;
  static_assert(HiddenSize == ThreadsPerToken * Vec);
  static_assert(ThreadsPerToken <= C10_WARP_SIZE);
  static_assert(C10_WARP_SIZE % ThreadsPerToken == 0);
  static_assert(BlockSize % C10_WARP_SIZE == 0);

  const int lane = threadIdx.x % ThreadsPerToken;
  const int token_in_block = threadIdx.x / ThreadsPerToken;
  const int token = blockIdx.x * TokensPerBlock + token_in_block;
  const bool valid_token = token < num_tokens;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  scalar_t input_vec[Vec];
  scalar_t residual_vec[Vec];
  T_ACC summed_vec[Vec];
  T_ACC sum = 0;
  if (valid_token) {
    const int64_t idx =
        static_cast<int64_t>(token) * HiddenSize + lane * Vec;
    *(LoadT*)input_vec = *(const LoadT*)(input + idx);
    *(LoadT*)residual_vec = *(const LoadT*)(residual + idx);
#pragma unroll
    for (int ii = 0; ii < Vec; ++ii) {
      const T_ACC x = static_cast<T_ACC>(to_float(input_vec[ii])) +
                      static_cast<T_ACC>(to_float(residual_vec[ii]));
      summed_vec[ii] = x;
      sum += x * x;
    }
  }

  sum = SubWarpReduceSumWithWidth<T_ACC, ThreadsPerToken>(sum);
  T_ACC rstd = 0;
  if (lane == 0) {
    rstd = c10::cuda::compat::rsqrt(sum / HiddenSize + eps);
  }
  rstd = __shfl(rstd, 0, ThreadsPerToken);

  if (valid_token) {
    scalar_t weight_vec[Vec];
    *(LoadT*)weight_vec = *(const LoadT*)(gamma + lane * Vec);
#pragma unroll
    for (int ii = 0; ii < Vec; ++ii) {
      T_ACC w_val = static_cast<T_ACC>(to_float(weight_vec[ii]));
      if constexpr (IS_GEMMA) { w_val += 1.0f; }
      T_ACC normalized = summed_vec[ii] * rstd * w_val;
      if constexpr (APPLY_CLAMP) {
        normalized = clamp_with_limit(normalized, limit);
      }
      from_float<scalar_t>(residual_vec[ii], summed_vec[ii]);
      from_float<scalar_t>(input_vec[ii], normalized);
    }
    const int64_t idx =
        static_cast<int64_t>(token) * HiddenSize + lane * Vec;
    *(LoadT*)(residual + idx) = *(LoadT*)residual_vec;
    *(LoadT*)(input + idx) = *(LoadT*)input_vec;
  }
}

template <typename scalar_t, typename T_ACC, bool IS_GEMMA, bool APPLY_CLAMP>
static inline bool launch_rms_small_hidden_multi_token(
    scalar_t* input,
    scalar_t* output,
    scalar_t* gamma,
    int hidden_size,
    T_ACC eps,
    int num_tokens,
    T_ACC limit,
    cudaStream_t stream) {
  constexpr int BlockSize = 256;
#define LAUNCH_SMALL_HIDDEN_RMS(HIDDEN_SIZE, THREADS_PER_TOKEN)               \
  do {                                                                        \
    constexpr int TokensPerBlock = BlockSize / THREADS_PER_TOKEN;             \
    const int grid = (num_tokens + TokensPerBlock - 1) / TokensPerBlock;       \
    rms_kernel_small_hidden_multi_token<                                      \
        scalar_t, T_ACC, HIDDEN_SIZE, THREADS_PER_TOKEN, BlockSize,            \
        IS_GEMMA, APPLY_CLAMP><<<grid, BlockSize, 0, stream>>>(                             \
        input, output, gamma, eps, num_tokens, limit);                                \
    return true;                                                              \
  } while (0)

  switch (hidden_size) {
    case 512: LAUNCH_SMALL_HIDDEN_RMS(512, 64);
    case 256: LAUNCH_SMALL_HIDDEN_RMS(256, 32);
    case 128: LAUNCH_SMALL_HIDDEN_RMS(128, 16);
    case 64: LAUNCH_SMALL_HIDDEN_RMS(64, 8);
    default: return false;
  }
#undef LAUNCH_SMALL_HIDDEN_RMS
}

template <typename scalar_t, typename T_ACC, bool IS_GEMMA, bool APPLY_CLAMP>
static inline bool launch_fused_add_rms_small_hidden_multi_token(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    int hidden_size,
    T_ACC eps,
    int num_tokens,
    T_ACC limit,
    cudaStream_t stream) {
  constexpr int BlockSize = 256;
#define LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS(HIDDEN_SIZE, THREADS_PER_TOKEN)     \
  do {                                                                        \
    constexpr int TokensPerBlock = BlockSize / THREADS_PER_TOKEN;             \
    const int grid = (num_tokens + TokensPerBlock - 1) / TokensPerBlock;       \
    fused_add_rms_kernel_small_hidden_multi_token<                            \
        scalar_t, T_ACC, HIDDEN_SIZE, THREADS_PER_TOKEN, BlockSize,            \
        IS_GEMMA, APPLY_CLAMP><<<grid, BlockSize, 0, stream>>>(                             \
        input, residual, gamma, eps, num_tokens, limit);                              \
    return true;                                                              \
  } while (0)

  switch (hidden_size) {
    case 512: LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS(512, 64);
    case 256: LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS(256, 32);
    case 128: LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS(128, 16);
    case 64: LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS(64, 8);
    default: return false;
  }
#undef LAUNCH_SMALL_HIDDEN_FUSED_ADD_RMS
}
#endif


#ifdef USE_ROCM
// FP32 residual sums stay in registers; each thread handles vectorized chunks.
template<typename scalar_t, int Hidden, int Threads, bool IsGemma, bool Clamp>
__global__ __launch_bounds__(Threads) void fused_add_rms_kernel_fp32_tiled(
    scalar_t* input, scalar_t* residual, const scalar_t* weight,
    float eps, float limit) {
  constexpr int Vec = 8;
  constexpr int Chunks = Hidden / (Threads * Vec);
  static_assert(Chunks > 0 && Hidden % (Threads * Vec) == 0);
  using Pack = at::native::memory::aligned_vector<scalar_t, Vec>;
  float sums[Chunks][Vec];
  float variance = 0.f;
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int warp = threadIdx.x / C10_WARP_SIZE;
  constexpr int Warps = Threads / C10_WARP_SIZE;
  __shared__ float partial[Warps];
  __shared__ float rstd;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * Hidden;
#pragma unroll
  for (int c = 0; c < Chunks; ++c) {
    const int col = (c * Threads + threadIdx.x) * Vec;
    const Pack x = *reinterpret_cast<const Pack*>(input + base + col);
    const Pack r = *reinterpret_cast<const Pack*>(residual + base + col);
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      const float s = static_cast<float>(x.val[v]) + static_cast<float>(r.val[v]);
      sums[c][v] = s;
      variance += s * s;
    }
  }
  variance = SubWarpReduceSumWithWidth<float, C10_WARP_SIZE>(variance);
  if (lane == 0) partial[warp] = variance;
  __syncthreads();
  if (warp == 0) {
    float total = lane < Warps ? partial[lane] : 0.f;
    total = SubWarpReduceSumWithWidth<float, C10_WARP_SIZE>(total);
    if (lane == 0) rstd = c10::cuda::compat::rsqrt(total / Hidden + eps);
  }
  __syncthreads();
  const float scale = rstd;
#pragma unroll
  for (int c = 0; c < Chunks; ++c) {
    const int col = (c * Threads + threadIdx.x) * Vec;
    const Pack w = *reinterpret_cast<const Pack*>(weight + col);
    Pack out, res;
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      float wt = static_cast<float>(w.val[v]);
      if constexpr (IsGemma) wt += 1.f;
      float y = sums[c][v] * scale * wt;
      if constexpr (Clamp) y = clamp_with_limit(y, limit);
      out.val[v] = static_cast<scalar_t>(y);
      res.val[v] = static_cast<scalar_t>(sums[c][v]);
    }
    *reinterpret_cast<Pack*>(residual + base + col) = res;
    *reinterpret_cast<Pack*>(input + base + col) = out;
  }
}
#endif

template <typename scalar_t,typename T_ACC,int Vec=4,int block_size=512, bool IS_GEMMA=false, bool APPLY_CLAMP=false>
__global__ void fused_add_rms_kernel_opt(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,T_ACC eps, int num_tokens,T_ACC limit)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  int i=blockIdx.x;
  int j=threadIdx.x;
  for(; i < num_tokens; i += gridDim.x){
        T_ACC val=0;
        int tcol=cols/Vec;
        scalar_t intput_vec[Vec];
        scalar_t residual_vec[Vec];
        T_ACC summed_vec[Vec];
        T_ACC trstd;
        int64_t idx = i * tcol + j;
        idx*=Vec;
        if (j < tcol) {
          *(LoadT*)intput_vec = *(LoadT*)(input+idx);
          *(LoadT*)residual_vec = *(LoadT*)(residual+idx);
          #pragma unroll
          for (int ii = 0; ii < Vec; ii++) {
            summed_vec[ii] = static_cast<T_ACC>(to_float(intput_vec[ii])) +
                             static_cast<T_ACC>(to_float(residual_vec[ii]));
            val += summed_vec[ii] * summed_vec[ii];
          }
        }
        val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
        if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
        __syncthreads();
        trstd=s_rstd;
        if (j < tcol) {
          #pragma unroll
          for(int ii=0;ii<Vec;ii++){
            int jj=j*Vec+ii;
            T_ACC w_val = to_float(gamma[jj]);
            if constexpr (IS_GEMMA) { w_val += 1.0f; }
            T_ACC normalized = summed_vec[ii] * trstd * w_val;
            if constexpr (APPLY_CLAMP) {
              normalized = clamp_with_limit(normalized, limit);
            }
            from_float<scalar_t>(residual_vec[ii], summed_vec[ii]);
            from_float<scalar_t>(intput_vec[ii], normalized);
          }
          *(LoadT*)(residual+idx)=*(LoadT*)residual_vec;
          *(LoadT*)(input+idx)=*(LoadT*)intput_vec;
        }
  }
}

template <typename scalar_t,typename T_ACC,int Vec=4,int block_size=512, bool IS_GEMMA=false, bool APPLY_CLAMP=false>
__global__ void fused_rms_kernel_opt(scalar_t* input,scalar_t* output,scalar_t* gamma,int cols,T_ACC eps,int num_tokens,T_ACC limit)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;

  int i=blockIdx.x;
  int j=threadIdx.x;
  for(; i < num_tokens; i += gridDim.x){
      T_ACC val=0;
      int tcol=cols/Vec;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      scalar_t intput_vec[Vec];
      T_ACC trstd;
      int64_t idx = i * tcol + j;
      idx*=Vec;
      if (j < tcol) {
        *(LoadT*)intput_vec = *(LoadT*)(input+idx);
        #pragma unroll
        for (int ii = 0; ii < Vec; ii++) {
          val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
        }
      }
      val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
      if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
      __syncthreads();
      trstd=s_rstd;
      if (j < tcol) {
        #pragma unroll
        for(int ii=0;ii<Vec;ii++){
          int jj=j*Vec+ii;
          T_ACC w_val = static_cast<T_ACC>(gamma[jj]);
          if constexpr (IS_GEMMA) { w_val += 1.0f; }
          T_ACC normalized = static_cast<T_ACC>(intput_vec[ii]) * trstd * w_val;
          if constexpr (APPLY_CLAMP) {
            normalized = clamp_with_limit(normalized, limit);
          }
          intput_vec[ii] = normalized;
        }
        *(LoadT*)(output+idx)=*(LoadT*)intput_vec;
      }
  }
}

template <typename scalar_t, typename weight_t, typename T_ACC, int Vec=4, int block_size=512,
          bool HAS_WEIGHT=false>
__global__ void rms_kernel_opt_optional_weight(
    const scalar_t* input,
    scalar_t* output,
    const weight_t* gamma,
    int cols,
    T_ACC eps,
    int num_tokens) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  int i=blockIdx.x;
  int j=threadIdx.x;
  const int tcol=cols/Vec;

  for(; i < num_tokens; i += gridDim.x){
      T_ACC val=0;
      scalar_t input_vec[Vec];
      int64_t idx = i * tcol + j;
      idx*=Vec;

      if (j < tcol) {
        *(LoadT*)input_vec = *(const LoadT*)(input+idx);
        #pragma unroll
        for (int ii = 0; ii < Vec; ii++) {
          const T_ACC x = static_cast<T_ACC>(to_float(input_vec[ii]));
          val += x * x;
        }
      }

      val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
      if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
      __syncthreads();

      const T_ACC trstd=s_rstd;
      if (j < tcol) {
        #pragma unroll
        for(int ii=0;ii<Vec;ii++){
          const int jj=j*Vec+ii;
          T_ACC y = static_cast<T_ACC>(to_float(input_vec[ii])) * trstd;
          if constexpr (HAS_WEIGHT) {
            y *= static_cast<T_ACC>(to_float(gamma[jj]));
          }
          from_float<scalar_t>(input_vec[ii], static_cast<float>(y));
        }
        *(LoadT*)(output+idx)=*(LoadT*)input_vec;
      }
  }
}

template <bool HAS_WEIGHT, typename scalar_t, typename weight_t, typename T_ACC>
static inline void launch_rms_kernel_opt_optional_weight(
    const scalar_t* input_data,
    scalar_t* out_data,
    const weight_t* weight_data,
    int hidden_size,
    T_ACC eps,
    int num_tokens,
    int grid,
    cudaStream_t stream) {
  if (hidden_size<=1024){
      rms_kernel_opt_optional_weight<scalar_t,weight_t,T_ACC,8,128,HAS_WEIGHT><<<grid, 128, 0, stream>>>(
          input_data,out_data,weight_data,hidden_size,eps,num_tokens);
  }
  else if(hidden_size<=2048){
      rms_kernel_opt_optional_weight<scalar_t,weight_t,T_ACC,8,256,HAS_WEIGHT><<<grid, 256, 0, stream>>>(
          input_data,out_data,weight_data,hidden_size,eps,num_tokens);
  }
  else if(hidden_size<=4096){
      rms_kernel_opt_optional_weight<scalar_t,weight_t,T_ACC,8,512,HAS_WEIGHT><<<grid, 512, 0, stream>>>(
          input_data,out_data,weight_data,hidden_size,eps,num_tokens);
  }
  else if(hidden_size<=8192){
      rms_kernel_opt_optional_weight<scalar_t,weight_t,T_ACC,8,1024,HAS_WEIGHT><<<grid, 1024, 0, stream>>>(
          input_data,out_data,weight_data,hidden_size,eps,num_tokens);
  }
  else{
      rms_kernel_opt_optional_weight<scalar_t,weight_t,T_ACC,16,1024,HAS_WEIGHT><<<grid, 1024, 0, stream>>>(
          input_data,out_data,weight_data,hidden_size,eps,num_tokens);
  }
}

template<bool IS_GEMMA, bool APPLY_CLAMP>
void rms_norm_opt_impl(torch::Tensor& out,    // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon,
              double limit) {
  TORCH_CHECK(
      limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
      "limit must be -1 (disabled) or a finite non-negative value");
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int grid = num_tokens;
  if(num_tokens == 288) grid = 256;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =inp_ptr % 16 == 0  && wt_ptr % 16 == 0;
  bool small_ptrs_are_aligned = ptrs_are_aligned && out_ptr % 16 == 0;
  if(hidden_size%16==0&&hidden_size<=16384&&ptrs_are_aligned){
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {
      using T_ACC = at::acc_type<scalar_t, true>;
      T_ACC eps = epsilon;
      scalar_t* self_data  = input.expect_contiguous()->data_ptr<scalar_t>();
      scalar_t* out_data   =  out.expect_contiguous()->data_ptr<scalar_t>();
      scalar_t* weight_data= weight.expect_contiguous()->data_ptr<scalar_t>();
#ifdef USE_ROCM
      if constexpr (!IS_GEMMA && sizeof(scalar_t) == 2) {
        if (small_ptrs_are_aligned &&
            launch_rms_small_hidden_multi_token<scalar_t, T_ACC, IS_GEMMA, APPLY_CLAMP>(
                self_data, out_data, weight_data, hidden_size, eps,
                num_tokens, static_cast<T_ACC>(limit), stream)) {
          return;
        }
      }
#endif
      if (hidden_size<=1024){
          fused_rms_kernel_opt<scalar_t,T_ACC,8,128, IS_GEMMA, APPLY_CLAMP><<<grid,  128, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
      }
      else if(hidden_size<=2048){
          fused_rms_kernel_opt<scalar_t,T_ACC,8,256, IS_GEMMA, APPLY_CLAMP><<<grid,  256, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
      }
      else if(hidden_size<=4096){
           fused_rms_kernel_opt<scalar_t,T_ACC,8,512, IS_GEMMA, APPLY_CLAMP><<<grid,  512, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
      }
      else if(hidden_size<=8192){
           fused_rms_kernel_opt<scalar_t,T_ACC,8,1024, IS_GEMMA, APPLY_CLAMP><<<grid,  1024, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
      }
      else{
          fused_rms_kernel_opt<scalar_t,T_ACC,16,1024, IS_GEMMA, APPLY_CLAMP><<<grid,  1024, 0, stream>>>(self_data,out_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
      }
    });
  }
  else{
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,input.scalar_type(), "rms_norm_kernel", [&] {
      vllm::rms_norm_kernel<scalar_t, IS_GEMMA, APPLY_CLAMP><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size,
          static_cast<float>(limit));
    });
  }
}

void rms_norm_opt(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon, double limit) {
    if (limit < 0.0) {
      rms_norm_opt_impl<false, false>(out, input, weight, epsilon, limit);
    } else {
      rms_norm_opt_impl<false, true>(out, input, weight, epsilon, limit);
    }
}

void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon) {
    rms_norm_opt_impl<true, false>(out, input, weight, epsilon, -1.0);
}

void rms_norm_no_weight(
    torch::optional<torch::Tensor> out,
    torch::Tensor& input,
    torch::optional<torch::Tensor> weight,
    double epsilon) {
  TORCH_CHECK(input.dim() > 0, "input must have at least 1 dimension");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const bool has_out = out.has_value() && out.value().defined();
  const bool has_weight = weight.has_value() && weight.value().defined();

  torch::Tensor output = has_out ? out.value() : input;

  TORCH_CHECK(output.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(output.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "out dtype must match input dtype");
  TORCH_CHECK(output.sizes() == input.sizes(),
              "out shape must match input shape");

  torch::Tensor weight_tensor;
  if (has_weight) {
    weight_tensor = weight.value();
    TORCH_CHECK(weight_tensor.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight_tensor.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight_tensor.numel() == input.size(-1),
                "weight numel must equal hidden_size");
  }

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  int grid = num_tokens;
  if(num_tokens == 288) grid = 256;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto out_ptr = reinterpret_cast<std::uintptr_t>(output.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0 && out_ptr % 16 == 0;
  if (has_weight) {
    auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight_tensor.data_ptr());
    ptrs_are_aligned = ptrs_are_aligned && wt_ptr % 16 == 0;
  }

  if(hidden_size%16==0&&hidden_size<=16384&&ptrs_are_aligned){
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_no_weight_opt",
      [&] {
        using input_t = scalar_t;
        using T_ACC = at::acc_type<input_t, true>;
        T_ACC eps = epsilon;
        const input_t* self_data = input.expect_contiguous()->data_ptr<input_t>();
        input_t* out_data = output.expect_contiguous()->data_ptr<input_t>();

        if (has_weight) {
          AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            weight_tensor.scalar_type(),
            "rms_norm_no_weight_opt_weight",
            [&] {
              using weight_t = scalar_t;
              const weight_t* weight_data =
                  weight_tensor.expect_contiguous()->data_ptr<weight_t>();
              launch_rms_kernel_opt_optional_weight<true, input_t, weight_t, T_ACC>(
                  self_data, out_data, weight_data, hidden_size, eps, num_tokens, grid, stream);
            });
        } else {
          launch_rms_kernel_opt_optional_weight<false, input_t, input_t, T_ACC>(
              self_data, out_data, nullptr, hidden_size, eps, num_tokens, grid, stream);
        }
      });
  }
  else{
    dim3 grid_dim(num_tokens);
    dim3 block(std::min(hidden_size, 1024));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,input.scalar_type(), "rms_norm_no_weight", [&] {
      using input_t = scalar_t;
      input_t* out_data = output.data_ptr<input_t>();
      const input_t* input_data = input.data_ptr<input_t>();

      if (has_weight) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          weight_tensor.scalar_type(),
          "rms_norm_no_weight_weight",
          [&] {
            using weight_t = scalar_t;
            const weight_t* weight_data = weight_tensor.data_ptr<weight_t>();
            vllm::rms_norm_optional_weight_kernel<input_t, weight_t, true>
                <<<grid_dim, block, 0, stream>>>(
                    out_data, input_data, weight_data, epsilon, num_tokens, hidden_size);
          });
      } else {
        vllm::rms_norm_optional_weight_kernel<input_t, input_t, false>
            <<<grid_dim, block, 0, stream>>>(
            out_data, input_data, nullptr, epsilon, num_tokens, hidden_size);
      }
    });
  }
}


#define LAUNCH_FUSED_ADD_RMS_NORM(width, is_gemma)                             \
     AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,                     \
        at::ScalarType::BFloat16,                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                  \
        vllm::fused_add_rms_norm_kernel<scalar_t, width, is_gemma, APPLY_CLAMP>             \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                         residual.data_ptr<scalar_t>(),        \
                                         weight.data_ptr<scalar_t>(), epsilon, \
                                         num_tokens, hidden_size, limit);      \
      });


template<bool IS_GEMMA, bool APPLY_CLAMP>
void fused_add_rms_norm_opt_impl(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon,
                        double limit) {
  TORCH_CHECK(
      limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
      "limit must be -1 (disabled) or a finite non-negative value");
  int hidden_size = input.size(-1);
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  int num_tokens = input.numel() / hidden_size;
  if (num_tokens == 0) return;
  int grid = num_tokens;
  if(num_tokens == 288) grid = 256;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if(hidden_size%16==0&&hidden_size<=16384&&ptrs_are_aligned){
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fused_add_rms_norm_kernel",
      [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        T_ACC eps = epsilon;
        scalar_t* self_data  = input.expect_contiguous()->data_ptr<scalar_t>();
        scalar_t* other_data = residual.expect_contiguous()->data_ptr<scalar_t>();
        scalar_t* weight_data= weight.expect_contiguous()->data_ptr<scalar_t>();
#ifdef USE_ROCM
        // The tiled path is specialized for aligned BF16/FP16 rows.
        if constexpr (sizeof(scalar_t) == 2) {
          if (hidden_size == 4096) {
            fused_add_rms_kernel_fp32_tiled<scalar_t, 4096, 512, IS_GEMMA, APPLY_CLAMP>
                <<<num_tokens, 512, 0, stream>>>(
                    self_data, other_data, weight_data, eps, static_cast<float>(limit));
            return;
          }
          if (hidden_size == 8192) {
            fused_add_rms_kernel_fp32_tiled<scalar_t, 8192, 512, IS_GEMMA, APPLY_CLAMP>
                <<<num_tokens, 512, 0, stream>>>(
                    self_data, other_data, weight_data, eps, static_cast<float>(limit));
            return;
          }
        }
        if constexpr (!IS_GEMMA && sizeof(scalar_t) == 2) {
          if (launch_fused_add_rms_small_hidden_multi_token<
                  scalar_t, T_ACC, IS_GEMMA, APPLY_CLAMP>(
                  self_data, other_data, weight_data, hidden_size, eps,
                  num_tokens, static_cast<T_ACC>(limit), stream)) {
            return;
          }
        }
#endif
        if (hidden_size<=1024){
            fused_add_rms_kernel_opt<scalar_t,T_ACC,8,128, IS_GEMMA, APPLY_CLAMP><<<grid,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
        }
        else if(hidden_size<=2048){
            fused_add_rms_kernel_opt<scalar_t,T_ACC,8,256, IS_GEMMA, APPLY_CLAMP><<<grid,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
        }
        else if(hidden_size<=4096){
            fused_add_rms_kernel_opt<scalar_t,T_ACC,8,512, IS_GEMMA, APPLY_CLAMP><<<grid,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
        }
        else if(hidden_size<=8192){
            fused_add_rms_kernel_opt<scalar_t,T_ACC,8,1024, IS_GEMMA, APPLY_CLAMP><<<grid,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
        }
        else{
            fused_add_rms_kernel_opt<scalar_t,T_ACC,16,1024, IS_GEMMA, APPLY_CLAMP><<<grid,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps,num_tokens,static_cast<T_ACC>(limit));
        }
      });
  }
  else{
    dim3 grid(num_tokens);
    /* This kernel is memory-latency bound in many scenarios.
      When num_tokens is large, a smaller block size allows
      for increased block occupancy on CUs and better latency
      hiding on global mem ops. */
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 block(std::min(hidden_size, max_block_size));

    if (ptrs_are_aligned && hidden_size % 8 == 0) {
      LAUNCH_FUSED_ADD_RMS_NORM(8, IS_GEMMA);
    } else {
      LAUNCH_FUSED_ADD_RMS_NORM(0, IS_GEMMA);
    }
  }
}

void fused_add_rms_norm_opt(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon, double limit) {
    if (limit < 0.0) {
      fused_add_rms_norm_opt_impl<false, false>(
          input, residual, weight, epsilon, limit);
    } else {
      fused_add_rms_norm_opt_impl<false, true>(
          input, residual, weight, epsilon, limit);
    }
}

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon) {
    fused_add_rms_norm_opt_impl<true, false>(input, residual, weight, epsilon, -1.0);
}

}
}
