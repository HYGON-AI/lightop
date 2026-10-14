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


#ifdef TORCH_VERSION_OVER_290
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