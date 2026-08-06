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
};

static inline __device__ uint8_t float_to_fp8_e4m3(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = 0x7f;
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      f_bits =
        c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits) + c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint8_t mant_odd = (f_bits >> 20) & 1;
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

static inline __device__ uint8_t float_to_fp8_e5m2(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
               + c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint32_t mant_odd = (f_bits >> 21) & 1;
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

template <bool is_e4m3>
static __device__ uint8_t to_f8_from_f32(float v1) {
  
#if (defined(__gfx938__) )
  union fp8x4{
    int int32;
    uint8_t  u8x4[4];
  };
  fp8x4 val;
  val.int32=0;
  if constexpr(is_e4m3){
    val.int32 = __builtin_hcu_cvt_pk_fp8_f32(v1,v1,val.int32,false);
  }
  else{
    val.int32 = __builtin_hcu_cvt_pk_bf8_f32(v1,v1,val.int32,false);
  }
  return val.u8x4[0];
#else
  if constexpr(is_e4m3){
    return float_to_fp8_e4m3(v1);
  }
  else{
    return float_to_fp8_e5m2(v1);
  }
#endif
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
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

template<typename scalar_t,bool is_e4m3> 
static __device__ inline float to_fp8(scalar_t in,float scale){
  float v = to_float<scalar_t>(in)/scale;
  return to_f8_from_f32<is_e4m3>(v);
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
        FN(_Float16, _Float16, Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);     \
      } else {                                                             \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(_Float16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                             \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_e5m2") {                                     \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, Fp8KVCacheDataType::kFp8E5M2);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(_Float16, uint8_t, Fp8KVCacheDataType::kFp8E5M2);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E5M2);      \
        } else {                                                             \
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
      } else {
        constexpr bool is_e4m3 = (kv_dt == Fp8KVCacheDataType::kFp8E4M3);
        key_cache[tgt_key_idx] = to_fp8<scalar_t,is_e4m3>(tgt_key, *k_scale);
        value_cache[tgt_value_idx] = to_fp8<scalar_t, is_e4m3>(tgt_value, *v_scale);
      } 
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
        value_cache[tgt_value_idx + i * block_size] = value_vec[i];
      }
      *(LoadT*)(key_cache+tgt_key_idx)=*(LoadT*)key_vec;
    } else  {
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
        constexpr bool is_e4m3 = (kv_dt == Fp8KVCacheDataType::kFp8E4M3);
        key_cache[tgt_key_idx] = to_fp8<scalar_t,is_e4m3>(key_vec[i], *k_scale);
        value_cache[tgt_value_idx] = to_fp8<scalar_t, is_e4m3>(value_vec[i], *v_scale);
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

     int token_idx = blockIdx.x * token_in_block + tid * VEC / (head_size * num_heads);



    if(token_idx >= num_tokens)
      return;
    const int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
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
          value_cache[tgt_value_idx + i * block_size] = value_vec[i];
        }
        *(LoadT*)(key_cache+tgt_key_idx)=*(LoadT*)key_vec;

      } else{
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
          constexpr bool is_e4m3 = (kv_dt == Fp8KVCacheDataType::kFp8E4M3);
          key_cache[tgt_key_idx] = to_fp8<scalar_t,is_e4m3>(key_vec[i], *k_scale);
          value_cache[tgt_value_idx] = to_fp8<scalar_t, is_e4m3>(value_vec[i], *v_scale);
        }
      } 
  
}


template <int TOKENS_PER_BLOCK, int VEC, typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void opt_reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,       // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,     // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,        // [num_blocks, num_heads, block_size, head_size]
    cache_t* __restrict__ value_cache,      // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, 
    const int head_size, const int block_size,
    const float* k_scale, const float* v_scale,
    const int num_tokens) { 

    const int num_elements_per_token = num_heads * head_size;
    const int threads_per_token = num_elements_per_token / VEC;
    const int local_token_idx = threadIdx.x / threads_per_token;
    
    if (local_token_idx >= TOKENS_PER_BLOCK) return;
    const int global_token_idx = blockIdx.x * TOKENS_PER_BLOCK + local_token_idx;

    if (global_token_idx >= num_tokens) return;

    const int slot_idx = slot_mapping[global_token_idx];
    if (slot_idx < 0) return;

    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    using StoreT = at::native::memory::aligned_vector<cache_t, VEC>; 
    const int segment_idx = threadIdx.x % threads_per_token;
    const int idx = segment_idx * VEC; 

  
    scalar_t k_local[VEC];
    scalar_t v_local[VEC];
    
    const int64_t src_offset = global_token_idx * key_stride + idx; 
    
   
    *(LoadT*)k_local = *(LoadT*)(key + src_offset);
    *(LoadT*)v_local = *(LoadT*)(value + src_offset);

  
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int head_idx = idx / head_size;
    const int head_offset = idx % head_size;
    constexpr bool is_e4m3 = (kv_dt == Fp8KVCacheDataType::kFp8E4M3);
    //key
    {
        const int64_t tgt_key_idx = 
        (int64_t)block_idx * num_heads * block_size * head_size +
        (int64_t)head_idx * block_size * head_size + 
        (int64_t)block_offset * head_size + 
        (int64_t)head_offset;
        cache_t k_temp[VEC];
        if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
            *(StoreT*)(key_cache + tgt_key_idx) = *(StoreT*)k_local;
        } else {
            #pragma unroll
            for(int i=0; i<VEC; ++i) {
                k_temp[i] = to_fp8<scalar_t,is_e4m3>(k_local[i], *k_scale);
            }
            *(StoreT*)(key_cache + tgt_key_idx) = *(StoreT*)k_temp;
        }
    }
    //val
    {
        const int64_t tgt_val_base = 
        (int64_t)block_idx * num_heads * head_size * block_size +
        (int64_t)head_idx * head_size * block_size + 
        (int64_t)head_offset * block_size + 
        (int64_t)block_offset;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
                value_cache[tgt_val_base + i * block_size] = v_local[i];
            } else  {
                value_cache[tgt_val_base + i * block_size] = to_fp8<scalar_t, is_e4m3>(v_local[i], *v_scale);
            }
        }
    }   
}

// Large-token BF16 path for the native HCU cache layout:
//   K: [block, head, slot, dim]
//   V: [block, head, dim, slot]
//
// A workgroup stages a small input-token tile in LDS. When its slot mappings are
// consecutive and remain in one cache block, threads read the LDS tile in the
// transposed direction and write adjacent V-cache slots with 32-bit stores.
// Arbitrary, padded, or block-crossing slot mappings retain a scalar scatter
// fallback with exactly the same semantics.
//
// For the two-token / 4-KiB specialization, token 1 swaps adjacent 16-element
// LDS segments. The global loads remain aligned Vec16 operations, while
// transposed reads no longer address the same LDS bank as token 0.
template <int kTokensPerBlock>
__device__ __forceinline__ int value_tile_element(
    const int local_token, const int element) {
  if constexpr (kTokensPerBlock == 2) {
    return element ^ (local_token << 4);
  }
  return element;
}

template <int kTokensPerBlock>
__global__ void opt_reshape_and_cache_lds_h8_d128_bf16_kernel(
    const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int key_stride,
    const int value_stride,
    const int num_tokens) {
  constexpr int kNumHeads = 8;
  constexpr int kHeadSize = 128;
  constexpr int kBlockSize = 64;
  constexpr int kElementsPerToken = kNumHeads * kHeadSize;
  constexpr int kLoadVec = 16;

  __shared__ uint16_t value_tile[kTokensPerBlock][kElementsPerToken];
  __shared__ int slot_tile[kTokensPerBlock];
  __shared__ int contiguous_slots;

  const int tid = threadIdx.x;
  const int first_token = blockIdx.x * kTokensPerBlock;

  if (tid < kTokensPerBlock) {
    const int token_idx = first_token + tid;
    slot_tile[tid] =
        token_idx < num_tokens
            ? static_cast<int>(slot_mapping[token_idx])
            : -1;
  }
  __syncthreads();

  using LoadVec = at::native::memory::aligned_vector<uint16_t, kLoadVec>;
  constexpr int kTileLoadVectors =
      kTokensPerBlock * kElementsPerToken / kLoadVec;
  for (int load_index = tid; load_index < kTileLoadVectors;
       load_index += blockDim.x) {
    const int linear = load_index * kLoadVec;
    const int local_token = linear / kElementsPerToken;
    const int element = linear % kElementsPerToken;
    const int token_idx = first_token + local_token;

    uint16_t value_vec[kLoadVec];
    uint16_t key_vec[kLoadVec];
    if (token_idx < num_tokens) {
      const int64_t value_src =
          static_cast<int64_t>(token_idx) * value_stride + element;
      *reinterpret_cast<LoadVec*>(value_vec) =
          *reinterpret_cast<const LoadVec*>(value + value_src);
      const int lds_element =
          value_tile_element<kTokensPerBlock>(local_token, element);
      *reinterpret_cast<LoadVec*>(
          &value_tile[local_token][lds_element]) =
          *reinterpret_cast<LoadVec*>(value_vec);

      const int slot_idx = slot_tile[local_token];
      if (slot_idx >= 0) {
        const int64_t key_src =
            static_cast<int64_t>(token_idx) * key_stride + element;
        *reinterpret_cast<LoadVec*>(key_vec) =
            *reinterpret_cast<const LoadVec*>(key + key_src);

        const int head_idx = element / kHeadSize;
        const int head_offset = element % kHeadSize;
        const int block_idx = slot_idx / kBlockSize;
        const int block_offset = slot_idx % kBlockSize;
        const int64_t key_dst =
            (static_cast<int64_t>(block_idx) * kNumHeads + head_idx) *
                kBlockSize * kHeadSize +
            block_offset * kHeadSize + head_offset;
        *reinterpret_cast<LoadVec*>(key_cache + key_dst) =
            *reinterpret_cast<LoadVec*>(key_vec);
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    const int first_slot = slot_tile[0];
    int is_contiguous = first_slot >= 0;
#pragma unroll
    for (int token = 1; token < kTokensPerBlock; ++token) {
      is_contiguous &= slot_tile[token] == first_slot + token;
    }
    if (is_contiguous) {
      is_contiguous &=
          first_slot / kBlockSize ==
          (first_slot + kTokensPerBlock - 1) / kBlockSize;
    }
    contiguous_slots = is_contiguous;
  }
  __syncthreads();

  for (int element = tid; element < kElementsPerToken;
       element += blockDim.x) {
    const int head_idx = element / kHeadSize;
    const int head_offset = element % kHeadSize;

    if (contiguous_slots) {
      const int first_slot = slot_tile[0];
      const int block_idx = first_slot / kBlockSize;
      const int block_offset = first_slot % kBlockSize;
      uint16_t* value_dst =
          value_cache +
          (static_cast<int64_t>(block_idx) * kNumHeads + head_idx) *
              kHeadSize * kBlockSize +
          head_offset * kBlockSize + block_offset;

      if constexpr (kTokensPerBlock == 2) {
        const uint32_t pair =
            static_cast<uint32_t>(
                value_tile[0][value_tile_element<2>(0, element)]) |
            (static_cast<uint32_t>(
                 value_tile[1][value_tile_element<2>(1, element)])
             << 16);
        // The first slot may be only two-byte aligned. The builtin preserves
        // that alignment contract while allowing the backend to select the
        // optimal four-byte global store.
        __builtin_memcpy(value_dst, &pair, sizeof(pair));
      } else {
        using Store2 =
            at::native::memory::aligned_vector<uint16_t, 2>;
        int token = 0;
        if (block_offset & 1) {
          value_dst[0] =
              value_tile[0][
                  value_tile_element<kTokensPerBlock>(0, element)];
          token = 1;
        }
        for (; token + 1 < kTokensPerBlock; token += 2) {
          uint16_t pair[2] = {
              value_tile[token][value_tile_element<kTokensPerBlock>(
                  token, element)],
              value_tile[token + 1][value_tile_element<kTokensPerBlock>(
                  token + 1, element)],
          };
          *reinterpret_cast<Store2*>(value_dst + token) =
              *reinterpret_cast<Store2*>(pair);
        }
        if (token < kTokensPerBlock) {
          value_dst[token] =
              value_tile[token][value_tile_element<kTokensPerBlock>(
                  token, element)];
        }
      }
    } else {
#pragma unroll
      for (int local_token = 0; local_token < kTokensPerBlock;
           ++local_token) {
        const int slot_idx = slot_tile[local_token];
        if (slot_idx < 0) {
          continue;
        }
        const int block_idx = slot_idx / kBlockSize;
        const int block_offset = slot_idx % kBlockSize;
        const int64_t value_dst =
            (static_cast<int64_t>(block_idx) * kNumHeads + head_idx) *
                kHeadSize * kBlockSize +
            head_offset * kBlockSize + block_offset;
        value_cache[value_dst] =
            value_tile[local_token][
                value_tile_element<kTokensPerBlock>(
                    local_token, element)];
      }
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
          num_heads, block_size, 1,                                        \
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
          num_heads, block_size, 1,                                        \
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

// Helper macro to reduce code duplication in the switch statement
// Modified: Added VEC_VAL as argument
#define LAUNCH_KERNEL_OPT(TOKENS_PER_BLOCK, VEC_VAL, KV_T, CACHE_T, KV_DTYPE) \
    opt_reshape_and_cache_kernel<TOKENS_PER_BLOCK, VEC_VAL, KV_T, CACHE_T, KV_DTYPE> \
    <<<grid, block, 0, stream>>>( \
        reinterpret_cast<KV_T*>(key.data_ptr()), \
        reinterpret_cast<KV_T*>(value.data_ptr()), \
        reinterpret_cast<CACHE_T*>(key_cache.data_ptr()), \
        reinterpret_cast<CACHE_T*>(value_cache.data_ptr()), \
        slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
        num_heads, head_size, block_size, \
        reinterpret_cast<const float*>(k_scale.data_ptr()), \
        reinterpret_cast<const float*>(v_scale.data_ptr()), \
        num_tokens);

#define LAUNCH_VEC_1(S, C, D) LAUNCH_KERNEL_OPT(1, VEC, S, C, D)
#define LAUNCH_VEC_2(S, C, D) LAUNCH_KERNEL_OPT(2, VEC, S, C, D)
#define LAUNCH_VEC_4(S, C, D) LAUNCH_KERNEL_OPT(4, VEC, S, C, D)
#define LAUNCH_VEC_8(S, C, D) LAUNCH_KERNEL_OPT(8, VEC, S, C, D)
#define LAUNCH_VEC_16(S, C, D) LAUNCH_KERNEL_OPT(16, VEC, S, C, D)
#define LAUNCH_VEC_32(S, C, D) LAUNCH_KERNEL_OPT(32, VEC, S, C, D)
#define LAUNCH_VEC_64(S, C, D) LAUNCH_KERNEL_OPT(64, VEC, S, C, D)
#define LAUNCH_VEC_128(S, C, D) LAUNCH_KERNEL_OPT(128, VEC, S, C, D)
#define LAUNCH_VEC_256(S, C, D) LAUNCH_KERNEL_OPT(256, VEC, S, C, D)

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
  TORCH_CHECK(kv_cache_dtype=="auto" || kv_cache_dtype=="fp8" 
             ||kv_cache_dtype=="fp8_e4m3"||kv_cache_dtype=="fp8_e5m2");
  key = key.is_contiguous() ? key : key.contiguous();
  value = value.is_contiguous() ? value : value.contiguous();

  int num_tokens = slot_mapping.size(0);
  int num_heads  = key.size(1);
  int head_size  = key.size(2);
  int block_size = key_cache.size(2);   // k layout: [num_blocks, num_heads, block_size, head_size]
  const int  numel = num_heads*head_size;
  int key_stride   = key.stride(0);
  int value_stride = value.stride(0);

  const bool use_h8_d128_bf16_lds =
      num_heads == 8 && head_size == 128 && block_size == 64 &&
      key.scalar_type() == at::ScalarType::BFloat16 &&
      key_cache.scalar_type() == at::ScalarType::BFloat16 &&
      kv_cache_dtype == "auto";

  // Decode-sized batches use one token per 128-thread workgroup. The resulting
  // 2-KiB Value tile exposes enough workgroups to cover the 120-CU endpoint.
  if (use_h8_d128_bf16_lds && num_tokens > 0 && num_tokens <= 128) {
    constexpr int kTokensPerBlock = 1;
    dim3 block(128);
    dim3 grid((num_tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    opt_reshape_and_cache_lds_h8_d128_bf16_kernel<kTokensPerBlock>
        <<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(key.data_ptr()),
        reinterpret_cast<const uint16_t*>(value.data_ptr()),
        reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, num_tokens);
    return;
  }

  // Larger batches amortize a two-token / 4-KiB tile while halving the number
  // of workgroups and enabling packed stores for adjacent slot pairs.
  if (use_h8_d128_bf16_lds && num_tokens > 128) {
    constexpr int kTokensPerBlock = 2;
    dim3 block(128);
    dim3 grid((num_tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    opt_reshape_and_cache_lds_h8_d128_bf16_kernel<kTokensPerBlock>
        <<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(key.data_ptr()),
        reinterpret_cast<const uint16_t*>(value.data_ptr()),
        reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, num_tokens);
    return;
  }

  auto launch_kernel_opt = [&](auto vec_constant) {

      constexpr int VEC = decltype(vec_constant)::value;
      const int  numel = num_heads*head_size;
      int threads_per_token = numel / VEC;
    

      int target_block_size;
      int tokens_per_block = 1;
      
      if(num_tokens < 128){
          target_block_size = 64;
      }else if(num_tokens < 4096){
        target_block_size = 128;
      } else if(num_tokens < 65536){
          // A 256-thread workgroup keeps more independent blocks resident while
          // the transposed Value-cache stores wait on memory. Larger token
          // counts need more blocks, not more tokens packed into one block.
          target_block_size = 256;
      }else{
          target_block_size = 1024;
      }
      if (threads_per_token < target_block_size) {
          tokens_per_block = target_block_size / threads_per_token;
      }

      int threads_num = threads_per_token * tokens_per_block;
      dim3 block(threads_num);
      dim3 grid((num_tokens + tokens_per_block - 1) / tokens_per_block);

      const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          key.scalar_type(), "reshape_and_cache_vectorized", [&] {
              switch (tokens_per_block) {
                  case 1:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_1);
                        break;
                  case 2:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_2);
                        break;
                  case 4:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_4);
                        break;
                  case 8:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_8);
                        break;
                  case 16:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_16);
                        break;
                  case 32:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_32);
                        break;
                  case 64:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_64);
                        break; 
                  case 128:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_128);
                        break;
                  case 256:
                        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_256);
                        break;                              
                  default:
                        {
                            dim3 grid(num_tokens);
                            dim3 block(threads_per_token);
                            DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, LAUNCH_VEC_1);
                        }
                        break;
              }
      });
  };

  if(numel % 32 == 0){
      if (num_tokens <= 512) {
          launch_kernel_opt(std::integral_constant<int, 16>{});
      } else {
          launch_kernel_opt(std::integral_constant<int, 32>{});
      }
  }else{
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
  
}
}
}
