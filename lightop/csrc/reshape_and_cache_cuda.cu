#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "reduce.h"
#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #include <cub/util_type.cuh>
#else
  #include <hipcub/hipcub.hpp>
  #include <hipcub/util_type.hpp>
#endif

namespace at{
namespace native{
typedef __hip_bfloat16 __nv_bfloat16;
enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  kInt8 = 3,
};

static inline __device__ uint8_t float_to_int8(float x, const float scale) {
  int8_t fx = roundf(std::max(-128.f, std::min(127.f, x / scale)));
  uint8_t res = fx + 128;

  return res;
}

// Define custom FP32 vector data types.
struct Float4_ {
  float2 x;
  float2 y;
};

// int8x4 -> float4
template <typename float4, typename uint32_t>
__inline__ __device__ float4 scaled_vec_conversion_int8(
    const uint32_t& a, const float scale) {
  Float4_ tmp = float_to_int8(a, scale);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

 // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, __nv_bfloat16, Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_e5m2") {                                     \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, Fp8KVCacheDataType::kFp8E5M2);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E5M2);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E5M2);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
      }                                                                        \
    }

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel_vllm(
    const scalar_t* __restrict__ key,       // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,     // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,        // [num_blocks, num_heads, block_size, head_size]  target layout
    cache_t* __restrict__ value_cache,      // [num_blocks, num_heads, head_size, block_size]  
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, 
    const int head_size, const int block_size, int x,    
    const float* k_scale, const float* v_scale) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      // Padding token that should be ignored.
      return;
    }               

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int n = num_heads * head_size;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int64_t src_key_idx = token_idx * key_stride + i;
      const int64_t src_value_idx = token_idx * value_stride + i;

      const int head_idx  = i / head_size;   
      const int head_offset = i % head_size; 

      // ---------- calculate target index ----------
      // K: [num_blocks, num_heads, block_size, head_size]
      const int64_t tgt_key_idx = 
        block_idx * num_heads * block_size * head_size +
        head_idx * block_size * head_size + block_offset * head_size +
        head_offset;
      // V: [num_blocks, num_heads, head_size, block_size]
      const int64_t tgt_value_idx = 
        block_idx  * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
      scalar_t tgt_key = key[src_key_idx];
      scalar_t tgt_value = value[src_value_idx];
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        key_cache[tgt_key_idx] = tgt_key;
        value_cache[tgt_value_idx] = tgt_value;
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
        key_cache[tgt_key_idx] =
            scaled_vec_conversion_int8<cache_t, scalar_t>(tgt_key, 
                                                                *k_scale);
        value_cache[tgt_value_idx] =
            scaled_vec_conversion_int8<cache_t, scalar_t>(tgt_value, 
                                                                *v_scale);
      } 
      // else {
      //   key_cache[tgt_key_idx] =
      //       scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      //   value_cache[tgt_value_idx] =
      //       scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
      // }
    }
}


template <int token_in_block, int VEC, typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt, int head_size>
__global__ void reshape_and_cache_kernel_cuda(
    const scalar_t* __restrict__ key,       // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,     // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,        // [num_blocks, num_heads, block_size, head_size]  target layout
    cache_t* __restrict__ value_cache,      // [num_blocks, num_heads, head_size, block_size]  
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, 
    const int block_size, int x,    
    const float* k_scale, const float* v_scale) {
    int tid = threadIdx.x;

    const int token_idx = blockIdx.x * token_in_block + tid * VEC / (head_size * num_heads);
    const int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      // Padding token that should be ignored.
      return;
    }               

    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    const int idx = threadIdx.x * VEC;
    scalar_t key_vec[VEC];
    scalar_t value_vec[VEC];

    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    int index = (idx) % (head_size * num_heads);
    const int src_key_idx = token_idx * key_stride + index;
    const int src_value_idx = token_idx * value_stride + index;

    const int head_idx  = index / head_size;   
    const int head_offset = index % head_size; 

    const int tgt_key_idx = 
      block_idx * num_heads * block_size * head_size +
      head_idx * block_size * head_size + block_offset * head_size +
      head_offset;
    *(LoadT*)key_vec = *(LoadT*)(key+src_key_idx);
    *(LoadT*)value_vec = *(LoadT*)(value+src_value_idx);

      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {

          const int head_idx  = (index) / head_size;   
          const int head_offset = (index) % head_size; 
          int tgt_value_idx = 
            block_idx  * num_heads * head_size * block_size +
            head_idx * head_size * block_size + head_offset * block_size +
            block_offset;
          #pragma unroll
          for(int i=0;i<VEC;i++)
          {
            // const int head_idx  = (index + i) / head_size;   
            // const int head_offset = (index + i) % head_size; 
            //   const int64_t tgt_value_idx = 
            // block_idx  * num_heads * head_size * block_size +
            // head_idx * head_size * block_size + head_offset * block_size +
            // block_offset;
            value_cache[tgt_value_idx + i * block_size] = value_vec[i];
          }
          *(LoadT*)(key_cache+tgt_key_idx)=*(LoadT*)key_vec;

          } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
            #pragma unroll
            for(int i=0; i < VEC; i++)
            {


              const int head_idx  = (index + i) / head_size;   
              const int head_offset = (index + i) % head_size; 
                const int64_t tgt_key_idx = 
              block_idx * num_heads * block_size * head_size +
              head_idx * block_size * head_size + block_offset * head_size +
              head_offset;
                const int64_t tgt_value_idx = 
              block_idx  * num_heads * head_size * block_size +
              head_idx * head_size * block_size + head_offset * block_size +
              block_offset;

              key_cache[tgt_key_idx] =
                  scaled_vec_conversion_int8<cache_t, scalar_t>(key_vec[i], 
                                                                      *k_scale);
              value_cache[tgt_value_idx] =
                  scaled_vec_conversion_int8<cache_t, scalar_t>(value_vec[i], 
                                                                      *v_scale);
            }
          } 
  
}

template <int token_in_block, int VEC, typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt, int head_size>
__global__ void reshape_and_cache_kernel_cuda_special(
    const scalar_t* __restrict__ key,       // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,     // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,        // [num_blocks, num_heads, block_size, head_size]  target layout
    cache_t* __restrict__ value_cache,      // [num_blocks, num_heads, head_size, block_size]  
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, 
    const int block_size, int x,    
    const float* k_scale, const float* v_scale, int num_tokens) {
    int tid = threadIdx.x;

    const int token_idx = blockIdx.x * token_in_block + tid * VEC / (head_size * num_heads);
    if(token_idx >= num_tokens)
      return;
    const int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      // Padding token that should be ignored.
      return;
    }               

    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    const int idx = threadIdx.x * VEC;
    scalar_t key_vec[VEC];
    scalar_t value_vec[VEC];

    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    int index = (idx) % (head_size * num_heads);
    const int src_key_idx = token_idx * key_stride + index;
    const int src_value_idx = token_idx * value_stride + index;

    const int head_idx  = index / head_size;   
    const int head_offset = index % head_size; 

    const int tgt_key_idx = 
      block_idx * num_heads * block_size * head_size +
      head_idx * block_size * head_size + block_offset * head_size +
      head_offset;
    *(LoadT*)key_vec = *(LoadT*)(key+src_key_idx);
    *(LoadT*)value_vec = *(LoadT*)(value+src_value_idx);

      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {

          const int head_idx  = (index) / head_size;   
          const int head_offset = (index) % head_size; 
          int tgt_value_idx = 
            block_idx  * num_heads * head_size * block_size +
            head_idx * head_size * block_size + head_offset * block_size +
            block_offset;
          #pragma unroll
          for(int i=0;i<VEC;i++)
          {
            // const int head_idx  = (index + i) / head_size;   
            // const int head_offset = (index + i) % head_size; 
            //   const int64_t tgt_value_idx = 
            // block_idx  * num_heads * head_size * block_size +
            // head_idx * head_size * block_size + head_offset * block_size +
            // block_offset;
            value_cache[tgt_value_idx + i * block_size] = value_vec[i];
          }
          *(LoadT*)(key_cache+tgt_key_idx)=*(LoadT*)key_vec;

          } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
            #pragma unroll
            for(int i=0; i < VEC; i++)
            {


              const int head_idx  = (index + i) / head_size;   
              const int head_offset = (index + i) % head_size; 
                const int64_t tgt_key_idx = 
              block_idx * num_heads * block_size * head_size +
              head_idx * block_size * head_size + block_offset * head_size +
              head_offset;
                const int64_t tgt_value_idx = 
              block_idx  * num_heads * head_size * block_size +
              head_idx * head_size * block_size + head_offset * block_size +
              block_offset;

              key_cache[tgt_key_idx] =
                  scaled_vec_conversion_int8<cache_t, scalar_t>(key_vec[i], 
                                                                      *k_scale);
              value_cache[tgt_value_idx] =
                  scaled_vec_conversion_int8<cache_t, scalar_t>(value_vec[i], 
                                                                      *v_scale);
            }
          } 
  
}

#define CALL_RESHAPE_AND_CACHE_CUDA(KV_T, CACHE_T, KV_DTYPE)               \
  reshape_and_cache_kernel_cuda<token_in_block, VEC, KV_T, CACHE_T, KV_DTYPE, head_sizes>             \
      <<<grid, block, 0, stream>>>(                                        \
          reinterpret_cast<KV_T*>(key.data_ptr()),                         \
          reinterpret_cast<KV_T*>(value.data_ptr()),                       \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,      \
          num_heads, block_size, 1,                             \
          reinterpret_cast<const float*>(k_scale.data_ptr()),              \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

#define CALL_RESHAPE_AND_CACHE_CUDA_SPECIAL(KV_T, CACHE_T, KV_DTYPE)               \
  reshape_and_cache_kernel_cuda_special<token_in_block, VEC, KV_T, CACHE_T, KV_DTYPE, head_sizes>             \
      <<<grid, block, 0, stream>>>(                                        \
          reinterpret_cast<KV_T*>(key.data_ptr()),                         \
          reinterpret_cast<KV_T*>(value.data_ptr()),                       \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,      \
          num_heads, block_size, 1,                             \
          reinterpret_cast<const float*>(k_scale.data_ptr()),              \
          reinterpret_cast<const float*>(v_scale.data_ptr()), num_tokens);

#define CALL_RESHAPE_AND_CACHE_VLLM(KV_T, CACHE_T, KV_DTYPE)               \
  reshape_and_cache_kernel_vllm<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                        \
          reinterpret_cast<KV_T*>(key.data_ptr()),                         \
          reinterpret_cast<KV_T*>(value.data_ptr()),                       \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,      \
          num_heads, head_size, block_size, 1,                             \
          reinterpret_cast<const float*>(k_scale.data_ptr()),              \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache_cuda(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& 
        value_cache, // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must be [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "cache tensor shape mismatch");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
              key_cache.size(1) == value_cache.size(1) &&
              key_cache.size(2) == value_cache.size(3) &&
              key_cache.size(3) == value_cache.size(2),
              "key/value cache dimension mismatch");

  int num_tokens = slot_mapping.size(0);
  int num_heads  = key.size(1);
  int head_size  = key.size(2);
  int block_size = key_cache.size(2);   // k layout: [num_blocks, num_heads, block_size, head_size]

  int key_stride   = key.stride(0);
  int value_stride = value.stride(0);
AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      key.scalar_type(), "reshape_and_cache_cuda", [&] {

  
  if(num_heads <= 2 && head_size == 128){
    constexpr int head_sizes = 128;
    if(num_tokens<=32 && num_tokens % 2 == 0)
    {
      constexpr int VEC = 2;
      constexpr int token_in_block = 1;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }
    else if(num_tokens % 8 == 0)
    {
      constexpr int VEC = 16;
      constexpr int token_in_block = 8;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens % 4 == 0)
    {
      constexpr int VEC = 16;
      constexpr int token_in_block = 4;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens % 2 == 0)
    {
      constexpr int VEC = 8;
      constexpr int token_in_block = 2;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens >= 256 && num_tokens % 3 == 0)
    {
      constexpr int VEC = 8;
      constexpr int token_in_block = 3;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens >= 256 && num_tokens % 5 == 0)
    {
      constexpr int VEC = 16;
      constexpr int token_in_block = 5;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens >= 256 && num_tokens % 7 == 0)
    {
      constexpr int VEC = 16;
      constexpr int token_in_block = 7;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }else if(num_tokens >= 256)
    {
      constexpr int VEC = 16;
      constexpr int token_in_block = 4;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA_SPECIAL);
    }
    else
    {
      constexpr int VEC = 2;
      constexpr int token_in_block = 1;
      dim3 grid((num_tokens + token_in_block - 1) / token_in_block);
      dim3 block(num_heads * head_size * token_in_block / VEC);
      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                                CALL_RESHAPE_AND_CACHE_CUDA);
    }
    
  }else{
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_VLLM);
  }
      }
      );
}
}}
