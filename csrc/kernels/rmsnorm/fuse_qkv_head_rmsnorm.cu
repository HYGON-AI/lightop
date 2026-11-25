// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif


#if defined(TORCH_VERSION_OVER_2120)
// torch>=2.12 removed C10_WARP_SIZE_STATIC, wavefront size is fixed at 64
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

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
#define DISPATCH_WEIGHT_TYPES(TYPE, NAME, ...) \
    switch(TYPE) { \
        case at::ScalarType::Float: { \
            using weight_t = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::Half: { \
            using weight_t = at::Half; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::BFloat16: { \
            using weight_t = at::BFloat16; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            AT_ERROR(#NAME, " not implemented for weight type '", toString(TYPE), "'"); \
    }
namespace vllm {
template <typename scalar_t, typename weight_t>
__global__ void qk_rms_norm_kernel(
    scalar_t* __restrict__ token_data,    // [num_tokens, total_dim]
    const weight_t* __restrict__ q_weight,// [qk_head_dim]
    const weight_t* __restrict__ k_weight,// [qk_head_dim]
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    int total_dim,
    const float epsilon, 
    const int total_tasks) 
{
    int i = blockIdx.x;
    if (i >= total_tasks) return;

    int token_idx = i / (q_head_num + kv_head_num);
    int head_idx = i % (q_head_num + kv_head_num);
    
    scalar_t* head_data;
    const weight_t* weight;
    
    if (head_idx < q_head_num) {
        weight = q_weight;
        head_data = token_data + token_idx * total_dim + head_idx * qk_head_dim;
    } else {
        weight = k_weight;
        head_data = token_data + token_idx * total_dim + q_head_num * qk_head_dim + (head_idx - q_head_num) * qk_head_dim;
    }

    __shared__ float s_variance;
    float variance = 0.0f;

    for (int idx = threadIdx.x; idx < qk_head_dim; idx += blockDim.x) {
        const float x = (float)head_data[idx];
        variance += x * x;
    }
    
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / qk_head_dim + epsilon);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < qk_head_dim; idx += blockDim.x) {
        float x = (float)head_data[idx];
        head_data[idx] = static_cast<scalar_t>( x * s_variance * static_cast<float>(weight[idx]) );
    }
}

} // namespace vllm


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

template <typename T, int width>
__inline__ __device__ T SubWarpReduceSum(T val) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
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
//小
template <typename scalar_t, typename weight_t, typename T_ACC, int Vec=8, int threads_per_head=16, int block_size=256>
__global__ void fused_qk_rms_kernel_small_opt(
    scalar_t* token_data,
    weight_t* q_weight,
    weight_t* k_weight,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    int total_dim,
    T_ACC eps,
    int total_tasks)
{
  constexpr int heads_per_block = block_size / threads_per_head;

  int tid = threadIdx.x;
  int head_idx_in_block = tid / threads_per_head;
  int tid_in_head = tid % threads_per_head;
  
  int task_id = blockIdx.x * heads_per_block + head_idx_in_block;
  
  bool is_valid = task_id < total_tasks;

  scalar_t* head_data;
  weight_t* weight;
  T_ACC val = 0;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  scalar_t input_vec[Vec];
  float temp_vec[Vec];
  if (is_valid) {
      int token_idx = task_id / (q_head_num + kv_head_num);
      int head_idx = task_id % (q_head_num + kv_head_num);
      
      if (head_idx < q_head_num) {
          weight = q_weight;
          head_data = token_data + token_idx * total_dim + head_idx * qk_head_dim;
      } else {
          weight = k_weight;
          head_data = token_data + token_idx * total_dim + q_head_num * qk_head_dim + (head_idx - q_head_num) * qk_head_dim;
      }

      *(LoadT*)input_vec = *(LoadT*)(head_data + tid_in_head * Vec);
      
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
          temp_vec[ii] = to_float<scalar_t>(input_vec[ii]);
          val += temp_vec[ii] * temp_vec[ii];
      }
  }

  val = SubWarpReduceSum<T_ACC, threads_per_head>(val);

  T_ACC trstd = val;
  if (tid_in_head == 0) {
      trstd = c10::cuda::compat::rsqrt(val / qk_head_dim + eps);
  }
  
  int lane_id = tid % C10_WARP_SIZE;
  int src_lane = lane_id - tid_in_head; 
  
#ifndef USE_ROCM
  trstd = __shfl_sync(0xffffffff, trstd, src_lane);
#else
  trstd = __shfl(trstd, src_lane);
#endif

  if (is_valid) {
      #pragma unroll
      for(int ii=0; ii<Vec; ii++) {
          int jj = tid_in_head * Vec + ii;
          from_float(input_vec[ii], temp_vec[ii] * trstd * static_cast<float>(weight[jj]));
      }
      *(LoadT*)(head_data + tid_in_head * Vec) = *(LoadT*)input_vec;
  }
}

// 大的 head_dim
template <typename scalar_t, typename weight_t, typename T_ACC, int Vec=4, int block_size=32>
__global__ void fused_qk_rms_kernel_opt(
    scalar_t* token_data,
    weight_t* q_weight,
    weight_t* k_weight,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    int total_dim,
    T_ACC eps,
    int total_tasks)
{
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size > 0 ? share_size : 1];
  __shared__ T_ACC s_rstd;

  int i = blockIdx.x;
  int j = threadIdx.x;
  int tcol = qk_head_dim / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  for(; i < total_tasks; i += gridDim.x) {
      T_ACC val = 0;
      scalar_t input_vec[Vec];
      T_ACC trstd;
      
      int token_idx = i / (q_head_num + kv_head_num);
      int head_idx = i % (q_head_num + kv_head_num);
      
      scalar_t* head_data;
      weight_t* weight;
      if (head_idx < q_head_num) {
          weight = q_weight;
          head_data = token_data + token_idx * total_dim + head_idx * qk_head_dim;
      } else {
          weight = k_weight;
          head_data = token_data + token_idx * total_dim + q_head_num * qk_head_dim + (head_idx - q_head_num) * qk_head_dim;
      }

      if (j < tcol) {
        *(LoadT*)input_vec = *(LoadT*)(head_data + j * Vec);
        #pragma unroll
        for (int ii = 0; ii < Vec; ii++) {
          val += static_cast<T_ACC>(input_vec[ii]) * static_cast<T_ACC>(input_vec[ii]);
        }
      }
      
      val = BlockReduceSum_NEW<T_ACC, block_size>(val, val_shared);
      if (j == 0) s_rstd = c10::cuda::compat::rsqrt(val / qk_head_dim + eps);
      __syncthreads();
      
      trstd = s_rstd;
      if (j < tcol) {
        #pragma unroll
        for(int ii=0; ii<Vec; ii++) {
          int jj = j * Vec + ii;
          input_vec[ii] = static_cast<T_ACC>(input_vec[ii]) * trstd * static_cast<T_ACC>(weight[jj]);
        }
        *(LoadT*)(head_data + j * Vec) = *(LoadT*)input_vec;
      }
  }
}

// ============================================================================
// Ported from custom_ops-20260819/csrc/custom_headrmsnorm_hip.hip
// Fast path: bf16 token + fp32 q/k weight, one block per token (wavefront64).
// ============================================================================
__global__ void fuse_qkv_head_rms_kernel_128_opt(
    at::BFloat16* __restrict__ data,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    int num_tokens,
    int total_dim_elements,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    float eps)
{
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 6;
    int lane_id = tid & 63;
    int sub_group_id = lane_id >> 4;
    int local_lane = lane_id & 15;
    int dim_base = local_lane * 8;

    size_t row_offset = (size_t)token_idx * total_dim_elements;

    int total_norm_heads = q_head_num + kv_head_num;
    int num_batches = (total_norm_heads + 15) >> 4;

    for (int batch = 0; batch < num_batches; ++batch) {
        int logical_head = batch * 16 + warp_id * 4 + sub_group_id;
        if (logical_head >= total_norm_heads) continue;

        const float* w_ptr = (logical_head < q_head_num) ? q_weight : k_weight;
        size_t elem_offset = row_offset + (size_t)logical_head * qk_head_dim + dim_base;

        float4 w_lo = *reinterpret_cast<const float4*>(w_ptr + dim_base);
        float4 w_hi = *reinterpret_cast<const float4*>(w_ptr + dim_base + 4);

        uint4 p = *reinterpret_cast<uint4*>(data + elem_offset);

        float v0 = __builtin_bit_cast(float, (p.x & 0xFFFF) << 16);
        float v1 = __builtin_bit_cast(float, (p.x >> 16) << 16);
        float v2 = __builtin_bit_cast(float, (p.y & 0xFFFF) << 16);
        float v3 = __builtin_bit_cast(float, (p.y >> 16) << 16);
        float v4 = __builtin_bit_cast(float, (p.z & 0xFFFF) << 16);
        float v5 = __builtin_bit_cast(float, (p.z >> 16) << 16);
        float v6 = __builtin_bit_cast(float, (p.w & 0xFFFF) << 16);
        float v7 = __builtin_bit_cast(float, (p.w >> 16) << 16);

        float sum_sq = fma(v0, v0, fma(v1, v1, fma(v2, v2, fma(v3, v3,
                     fma(v4, v4, fma(v5, v5, fma(v6, v6, v7 * v7)))))));
        sum_sq += __shfl_down(sum_sq, 8, 16);
        sum_sq += __shfl_down(sum_sq, 4, 16);
        sum_sq += __shfl_down(sum_sq, 2, 16);
        sum_sq += __shfl_down(sum_sq, 1, 16);

        float inv_dim = 1.0f / (float)qk_head_dim;
        float rms = 0.0f;
        if ((lane_id & 15) == 0) rms = rsqrtf(sum_sq * inv_dim + eps);
        rms = __shfl(rms, lane_id & ~15, 16);

        float o0 = v0 * rms * w_lo.x;
        float o1 = v1 * rms * w_lo.y;
        float o2 = v2 * rms * w_lo.z;
        float o3 = v3 * rms * w_lo.w;
        float o4 = v4 * rms * w_hi.x;
        float o5 = v5 * rms * w_hi.y;
        float o6 = v6 * rms * w_hi.z;
        float o7 = v7 * rms * w_hi.w;

        uint4 out_p;
        out_p.x = (__builtin_bit_cast(unsigned int, o1) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o0) >> 16);
        out_p.y = (__builtin_bit_cast(unsigned int, o3) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o2) >> 16);
        out_p.z = (__builtin_bit_cast(unsigned int, o5) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o4) >> 16);
        out_p.w = (__builtin_bit_cast(unsigned int, o7) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o6) >> 16);

        *reinterpret_cast<uint4*>(data + elem_offset) = out_p;
    }
}

__global__ void fuse_qkv_head_rms_kernel_64_opt(
    at::BFloat16* __restrict__ data,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    int num_tokens,
    int total_dim_elements,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    float eps)
{
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 6;
    int lane_id = tid & 63;
    int sub_group_id = lane_id >> 3;
    int local_lane = lane_id & 7;
    int dim_base = local_lane * 8;

    size_t row_offset = (size_t)token_idx * total_dim_elements;

    int total_norm_heads = q_head_num + kv_head_num;
    int num_batches = (total_norm_heads + 31) >> 5;

    for (int batch = 0; batch < num_batches; ++batch) {
        int logical_head = batch * 32 + warp_id * 8 + sub_group_id;
        if (logical_head >= total_norm_heads) continue;

        const float* w_ptr = (logical_head < q_head_num) ? q_weight : k_weight;
        size_t elem_offset = row_offset + (size_t)logical_head * qk_head_dim + dim_base;

        float4 w_lo = *reinterpret_cast<const float4*>(w_ptr + dim_base);
        float4 w_hi = *reinterpret_cast<const float4*>(w_ptr + dim_base + 4);

        uint4 p = *reinterpret_cast<uint4*>(data + elem_offset);

        float v0 = __builtin_bit_cast(float, (p.x & 0xFFFF) << 16);
        float v1 = __builtin_bit_cast(float, (p.x >> 16) << 16);
        float v2 = __builtin_bit_cast(float, (p.y & 0xFFFF) << 16);
        float v3 = __builtin_bit_cast(float, (p.y >> 16) << 16);
        float v4 = __builtin_bit_cast(float, (p.z & 0xFFFF) << 16);
        float v5 = __builtin_bit_cast(float, (p.z >> 16) << 16);
        float v6 = __builtin_bit_cast(float, (p.w & 0xFFFF) << 16);
        float v7 = __builtin_bit_cast(float, (p.w >> 16) << 16);

        float sum_sq = fma(v0, v0, fma(v1, v1, fma(v2, v2, fma(v3, v3,
                     fma(v4, v4, fma(v5, v5, fma(v6, v6, v7 * v7)))))));
        sum_sq += __shfl_down(sum_sq, 4, 8);
        sum_sq += __shfl_down(sum_sq, 2, 8);
        sum_sq += __shfl_down(sum_sq, 1, 8);

        float inv_dim = 1.0f / (float)qk_head_dim;
        float rms = 0.0f;
        if ((lane_id & 7) == 0) rms = rsqrtf(sum_sq * inv_dim + eps);
        rms = __shfl(rms, lane_id & ~7, 8);

        float o0 = v0 * rms * w_lo.x;
        float o1 = v1 * rms * w_lo.y;
        float o2 = v2 * rms * w_lo.z;
        float o3 = v3 * rms * w_lo.w;
        float o4 = v4 * rms * w_hi.x;
        float o5 = v5 * rms * w_hi.y;
        float o6 = v6 * rms * w_hi.z;
        float o7 = v7 * rms * w_hi.w;

        uint4 out_p;
        out_p.x = (__builtin_bit_cast(unsigned int, o1) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o0) >> 16);
        out_p.y = (__builtin_bit_cast(unsigned int, o3) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o2) >> 16);
        out_p.z = (__builtin_bit_cast(unsigned int, o5) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o4) >> 16);
        out_p.w = (__builtin_bit_cast(unsigned int, o7) >> 16) << 16 |
                  (__builtin_bit_cast(unsigned int, o6) >> 16);

        *reinterpret_cast<uint4*>(data + elem_offset) = out_p;
    }
}

__global__ void fuse_qkv_head_rms_kernel_generic(
    at::BFloat16* __restrict__ data,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    int num_tokens,
    int total_dim_elements,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    float eps)
{
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 6;
    int lane_id = tid & 63;

    size_t row_offset = (size_t)token_idx * total_dim_elements;

    int total_norm_heads = q_head_num + kv_head_num;
    constexpr int heads_per_block = 4;
    int heads_per_iter = heads_per_block;
    int num_batches = (total_norm_heads + heads_per_iter - 1) / heads_per_iter;

    for (int batch = 0; batch < num_batches; ++batch) {
        int logical_head = batch * heads_per_iter + warp_id;
        if (logical_head >= total_norm_heads) continue;

        const float* w_ptr = (logical_head < q_head_num) ? q_weight : k_weight;
        size_t head_base = row_offset + (size_t)logical_head * qk_head_dim;

        float sum_sq = 0.0f;
        for (int dim_offset = lane_id * 4;
             dim_offset < qk_head_dim;
             dim_offset += 64 * 4) {
            uint2 p = *reinterpret_cast<const uint2*>(data + head_base + dim_offset);
            float v0 = __builtin_bit_cast(float, (p.x & 0xFFFF) << 16);
            float v1 = __builtin_bit_cast(float, (p.x >> 16) << 16);
            float v2 = __builtin_bit_cast(float, (p.y & 0xFFFF) << 16);
            float v3 = __builtin_bit_cast(float, (p.y >> 16) << 16);
            sum_sq += fma(v0, v0, fma(v1, v1, fma(v2, v2, v3 * v3)));
        }

        sum_sq += __shfl_down(sum_sq, 32, 64);
        sum_sq += __shfl_down(sum_sq, 16, 64);
        sum_sq += __shfl_down(sum_sq, 8, 64);
        sum_sq += __shfl_down(sum_sq, 4, 64);
        sum_sq += __shfl_down(sum_sq, 2, 64);
        sum_sq += __shfl_down(sum_sq, 1, 64);

        float rms = 0.0f;
        if (lane_id == 0) {
            rms = rsqrtf(sum_sq / static_cast<float>(qk_head_dim) + eps);
        }
        rms = __shfl(rms, 0, 64);

        for (int dim_offset = lane_id * 4;
             dim_offset < qk_head_dim;
             dim_offset += 64 * 4) {
            size_t elem_offset = head_base + dim_offset;
            uint2 p = *reinterpret_cast<const uint2*>(data + elem_offset);
            float v0 = __builtin_bit_cast(float, (p.x & 0xFFFF) << 16);
            float v1 = __builtin_bit_cast(float, (p.x >> 16) << 16);
            float v2 = __builtin_bit_cast(float, (p.y & 0xFFFF) << 16);
            float v3 = __builtin_bit_cast(float, (p.y >> 16) << 16);
            float4 w_vec = *reinterpret_cast<const float4*>(w_ptr + dim_offset);

            float o0 = v0 * rms * w_vec.x;
            float o1 = v1 * rms * w_vec.y;
            float o2 = v2 * rms * w_vec.z;
            float o3 = v3 * rms * w_vec.w;
            uint2 out_p;
            out_p.x = (__builtin_bit_cast(unsigned int, o1) >> 16) << 16 |
                      (__builtin_bit_cast(unsigned int, o0) >> 16);
            out_p.y = (__builtin_bit_cast(unsigned int, o3) >> 16) << 16 |
                      (__builtin_bit_cast(unsigned int, o2) >> 16);
            *reinterpret_cast<uint2*>(data + elem_offset) = out_p;
        }
    }
}

static bool launch_fuse_qkv_head_rms_custom_bf16(
    torch::Tensor& token_data,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    float eps,
    cudaStream_t stream)
{
  if (token_data.scalar_type() != at::ScalarType::BFloat16 ||
      q_weight.scalar_type() != at::ScalarType::Float ||
      k_weight.scalar_type() != at::ScalarType::Float ||
      qk_head_dim <= 0 || (qk_head_dim % 16) != 0) {
    return false;
  }

  int num_tokens = token_data.size(0);
  int total_dim = token_data.size(1);
  if (num_tokens <= 0 || (q_head_num <= 0 && kv_head_num <= 0)) {
    return true;
  }

  auto data_ptr_u = reinterpret_cast<std::uintptr_t>(token_data.data_ptr());
  auto q_ptr_u = reinterpret_cast<std::uintptr_t>(q_weight.data_ptr());
  auto k_ptr_u = reinterpret_cast<std::uintptr_t>(k_weight.data_ptr());
  if ((data_ptr_u % 16) != 0 || (q_ptr_u % 16) != 0 || (k_ptr_u % 16) != 0) {
    return false;
  }

  at::BFloat16* data_ptr = token_data.expect_contiguous()->data_ptr<at::BFloat16>();
  const float* q_weight_ptr = q_weight.expect_contiguous()->data_ptr<float>();
  const float* k_weight_ptr = k_weight.expect_contiguous()->data_ptr<float>();

  dim3 grid(static_cast<unsigned int>(num_tokens));
  dim3 block(256);

  if (qk_head_dim == 128) {
    fuse_qkv_head_rms_kernel_128_opt<<<grid, block, 0, stream>>>(
        data_ptr, q_weight_ptr, k_weight_ptr,
        num_tokens, total_dim, q_head_num, kv_head_num, qk_head_dim, eps);
  } else if (qk_head_dim == 64) {
    fuse_qkv_head_rms_kernel_64_opt<<<grid, block, 0, stream>>>(
        data_ptr, q_weight_ptr, k_weight_ptr,
        num_tokens, total_dim, q_head_num, kv_head_num, qk_head_dim, eps);
  } else {
    fuse_qkv_head_rms_kernel_generic<<<grid, block, 0, stream>>>(
        data_ptr, q_weight_ptr, k_weight_ptr,
        num_tokens, total_dim, q_head_num, kv_head_num, qk_head_dim, eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}


void fuse_qkv_head_rms_norm(
    torch::Tensor& token_data,   // in-place [num_tokens, total_dim]
    torch::Tensor& q_weight,     // [qk_head_dim]
    torch::Tensor& k_weight,     // [qk_head_dim]
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    double epsilon) 
{
  int num_tokens = token_data.size(0);
  int total_dim = token_data.size(1);

  int total_tasks = num_tokens * (q_head_num + kv_head_num);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(token_data));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Prefer custom_ops-20260819 token-parallel bf16 kernels when applicable.
  if (launch_fuse_qkv_head_rms_custom_bf16(
          token_data, q_weight, k_weight,
          q_head_num, kv_head_num, qk_head_dim,
          static_cast<float>(epsilon), stream)) {
    return;
  }
  
  auto data_ptr = reinterpret_cast<std::uintptr_t>(token_data.data_ptr());
  bool ptrs_are_aligned = (data_ptr % 16 == 0);

  if(qk_head_dim % 8 == 0 && ptrs_are_aligned) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      token_data.scalar_type(),
      "fused_qk_rms_kernel_opt",
      [&] {
        using token_t = scalar_t;
        using T_ACC = at::acc_type<token_t, true>;
        T_ACC eps = epsilon;
        token_t* token_ptr = token_data.expect_contiguous()->data_ptr<token_t>();

        DISPATCH_WEIGHT_TYPES(q_weight.scalar_type(), "fused_qk_rms_kernel_opt_weight", [&] {
            weight_t* q_wt = q_weight.expect_contiguous()->data_ptr<weight_t>();
            weight_t* k_wt = k_weight.expect_contiguous()->data_ptr<weight_t>();

            if (qk_head_dim == 32) {
                constexpr int block_size = 256;
                constexpr int threads_per_head = 4;
                constexpr int heads_per_block = block_size / threads_per_head;
                int grid_small = (total_tasks + heads_per_block - 1) / heads_per_block;
                fused_qk_rms_kernel_small_opt<token_t, weight_t, T_ACC, 8, 4, 256><<<grid_small, block_size, 0, stream>>>(
                    token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                );
            } else if (qk_head_dim == 64) {
                constexpr int block_size = 256;
                constexpr int threads_per_head = 8;
                constexpr int heads_per_block = block_size / threads_per_head;
                int grid_small = (total_tasks + heads_per_block - 1) / heads_per_block;
                fused_qk_rms_kernel_small_opt<token_t, weight_t, T_ACC, 8, 8, 256><<<grid_small, block_size, 0, stream>>>(
                    token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                );
            } else if (qk_head_dim == 128) {
                constexpr int block_size = 128;
                constexpr int threads_per_head = 16;
                constexpr int heads_per_block = block_size / threads_per_head;
                int grid_small = (total_tasks + heads_per_block - 1) / heads_per_block;
                fused_qk_rms_kernel_small_opt<token_t, weight_t, T_ACC, 8, threads_per_head, block_size><<<grid_small, block_size, 0, stream>>>(
                    token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                );
            } else if (qk_head_dim == 256) {
                constexpr int block_size = 256;
                constexpr int threads_per_head = 32;
                constexpr int heads_per_block = block_size / threads_per_head;
                int grid_small = (total_tasks + heads_per_block - 1) / heads_per_block;
                fused_qk_rms_kernel_small_opt<token_t, weight_t, T_ACC, 8, 32, 256><<<grid_small, block_size, 0, stream>>>(
                    token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                );
            } else {
                int grid = total_tasks; 
                if (qk_head_dim <= 512) {
                    fused_qk_rms_kernel_opt<token_t, weight_t, T_ACC, 8, 64><<<grid, 64, 0, stream>>>(
                        token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                    );
                } else {
                    fused_qk_rms_kernel_opt<token_t, weight_t, T_ACC, 16, 128><<<grid, 128, 0, stream>>>(
                        token_ptr, q_wt, k_wt, q_head_num, kv_head_num, qk_head_dim, total_dim, eps, total_tasks
                    );
                }
            }
        }); 
      }); 
  } else {
    int grid = total_tasks;
    dim3 block(std::min(qk_head_dim, 1024));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        token_data.scalar_type(), 
        "qk_rms_norm_kernel", 
        [&] {
            using token_t = scalar_t;
            DISPATCH_WEIGHT_TYPES(q_weight.scalar_type(), "qk_rms_norm_kernel_weight", [&] {
                vllm::qk_rms_norm_kernel<token_t, weight_t><<<grid, block, 0, stream>>>(
                    token_data.data_ptr<token_t>(), 
                    q_weight.data_ptr<weight_t>(),
                    k_weight.data_ptr<weight_t>(), 
                    q_head_num, kv_head_num, qk_head_dim, total_dim, 
                    epsilon, total_tasks);
            });
    });
  }
}

} // native
} // at
