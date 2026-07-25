
#include <limits>
#include <stdint.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/autograd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "reduce.h"
#include <type_traits> 

inline __device__ uint8_t float_to_fp8e4m3(float f) {
    // ===== 常量定义 =====
    // FP8 E4M3最大值: 1087 << 20
    // 1087 = 2^(4+1) - 1 + 2^4 = 31 + 16 = 47? 等等，这里需要重新计算
    // 实际上是基于指数和尾数的位模式
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    
    // 非正规化数处理掩码
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    
    // ===== 步骤1: 提取float的位表示 =====
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    
    // ===== 步骤2: 提取符号位并移除 =====
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    
    // ===== 步骤3: 处理不同数值范围 =====
    
    // 3a. 溢出处理
    if (f_bits >= fp8_max) {
      result = 0x7f;  // 01111111 (NaN/最大值)
    } else {
      
      // 3b. 非正规化数处理
      if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = c10::detail::fp32_to_bits(
          c10::detail::fp32_from_bits(f_bits) + 
          c10::detail::fp32_from_bits(denorm_mask)
        );
        result = static_cast<uint8_t>(f_bits - denorm_mask);
      
      // 3c. 正规化数处理
      } else {
        // 检查尾数第20位是否为1（对应FP8 E4M3 3位尾数的LSB）
        uint8_t mant_odd = (f_bits >> 20) & 1;
        
        // 指数偏置调整: FP32偏置127 → FP8 E4M3偏置7
        // 调整量 = 7 - 127 = -120
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        
        // 应用奇数舍入
        f_bits += mant_odd;
        
        // 右移20位得到8位结果
        result = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    
    // ===== 步骤4: 加上符号位 =====
    result |= static_cast<uint8_t>(sign >> 24);
    
    return result;
  }


  inline __device__ uint16_t float_to_fp8e4m3_x2(float a, float b)
{
  #if defined(__gfx938__)
    // 对于ROCm，可能需要使用不同的内置函数或组合现有函数
    int32_t temp;
    temp = __builtin_hcu_cvt_pk_fp8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
  #else
    uint16_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    return out;
  #endif
}

inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d)
{
  #if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
    return res;
  #else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
    return out;
  #endif
}

inline __device__ uint8_t float_to_fp8e5m2(float f) {
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

inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d)
{
  #if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
    return res;
  #else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return out;
  #endif
}

inline __device__ uint16_t float_to_fp8e5m2_x2(float a, float b)
{
  #if defined(__gfx938__)
    // 对于ROCm，可能需要使用不同的内置函数或组合现有函数
    int32_t temp;
    temp = __builtin_hcu_cvt_pk_fp8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
  #else
    uint16_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    return out;
  #endif
}

template<int VEC_SIZE, int CACHE_T_CONST>
__inline__ __device__ void convert_vec_to_uint8_t(float* v_vec_data_float, uint8_t* v_vec_data_uint8) {
  if constexpr (VEC_SIZE % 4 == 0) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE / 4; ++i) {
      if constexpr (CACHE_T_CONST == 1) {
        ((uint32_t*)v_vec_data_uint8)[i] = float_to_fp8e4m3_x4(v_vec_data_float[i * 4], v_vec_data_float[i * 4 + 1], v_vec_data_float[i * 4 + 2], v_vec_data_float[i * 4 + 3]);
      } else {
        ((uint32_t*)v_vec_data_uint8)[i] = float_to_fp8e5m2_x4(v_vec_data_float[i * 4], v_vec_data_float[i * 4 + 1], v_vec_data_float[i * 4 + 2], v_vec_data_float[i * 4 + 3]);
      }
    }
  } else if constexpr (VEC_SIZE % 2 == 0) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
      if constexpr (CACHE_T_CONST == 1) {
        ((uint16_t*)v_vec_data_uint8)[i] = float_to_fp8e4m3_x2(v_vec_data_float[i * 2], v_vec_data_float[i * 2 + 1]);
      } else {
        ((uint16_t*)v_vec_data_uint8)[i] = float_to_fp8e5m2_x2(v_vec_data_float[i * 2], v_vec_data_float[i * 2 + 1]);
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      if constexpr (CACHE_T_CONST == 1) {
        v_vec_data_uint8[i] = float_to_fp8e4m3(v_vec_data_float[i]);
      } else {
        v_vec_data_uint8[i] = float_to_fp8e5m2(v_vec_data_float[i]);
      }
    }
  }
}


template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
    T val[N];
};
template <typename Element, size_t len>
struct vec {
    using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
};

#define DISPATCH_BOOL(VAL, NAME, ...) \
  if (VAL) {                          \
    constexpr bool NAME = true;       \
    __VA_ARGS__();                    \
  } else {                            \
    constexpr bool NAME = false;      \
    __VA_ARGS__();                    \
  }


template <int N> using IntConst = std::integral_constant<int, N>;
#define IV(N) IntConst<N>() 

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


template <typename T_ACC, typename scalar_t,int Vec=4,int block_size=512, int NUM_QK_MUL, int num_warp, bool pipeline, bool is_q>
inline __device__ void apply_rmsnorm(scalar_t* input,scalar_t* gamma, int cols,T_ACC eps, scalar_t* intput_vec)
{
    constexpr int share_size=block_size/64;
    __shared__ T_ACC val_shared[share_size];
    __shared__ T_ACC s_rstd[num_warp];
    T_ACC val=0;
    int tid;
    int i=blockIdx.x;
    if(pipeline && is_q){
        tid=threadIdx.x-64;
    }else{
        tid=threadIdx.x;
    }
    int tcol=cols * num_warp/Vec;
    using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
    T_ACC trstd;
    int64_t idx =tid;
    idx*=Vec;
    if (tid < tcol) {
        *(LoadT*)intput_vec = *(LoadT*)(input+idx);
        #pragma unroll
        for (int ii = 0; ii < Vec; ii++) {
            val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
        }
    }

    int tid_in_land = tid % 64;
    int land = tid / 64;
    val = WarpReduceSum_NEW<T_ACC>(val);
    // __syncthreads();
    if (tid_in_land == 0) s_rstd[land]=c10::cuda::compat::rsqrt(val/cols + eps);
     __syncthreads();
    trstd=s_rstd[land];
    if (tid < tcol) {
        #pragma unroll
        for(int ii=0;ii<Vec;ii++){
            int jj=(tid*Vec+ii)%cols;
            intput_vec[ii] = static_cast<T_ACC>(intput_vec[ii]) *trstd* static_cast<T_ACC>(gamma[jj]);
        }
        // *(LoadT*)(input+idx)=*(LoadT*)intput_vec;
    }
}

template <typename T_ACC, typename scalar_t,int Vec=4,int block_size=512, int num_warp, bool pipeline, bool is_q>
inline __device__ void apply_rmsnorm_residual(scalar_t* input,scalar_t* gamma,scalar_t* residual,int cols,T_ACC eps, scalar_t* intput_vec)
{
    constexpr int share_size=block_size/64;
    __shared__ T_ACC val_shared[share_size];
    __shared__ T_ACC s_rstd[num_warp];
    T_ACC val=0;
    int tid;
    int i=blockIdx.x;
    if(pipeline && is_q){
        tid=threadIdx.x-64;
    }else{
        tid=threadIdx.x;
    }
    
    int tcol=cols * num_warp /Vec;
    using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
    scalar_t residual_vec[Vec];
    T_ACC trstd;
    int64_t idx = tid;
    idx*=Vec;
    if (tid < tcol) {
        *(LoadT*)intput_vec = *(LoadT*)(input+idx);
        *(LoadT*)residual_vec = *(LoadT*)(residual+idx);
        #pragma unroll
        for (int ii = 0; ii < Vec; ii++) {
        residual_vec[ii]+=intput_vec[ii];
        val += static_cast<T_ACC>(residual_vec[ii])*static_cast<T_ACC>(residual_vec[ii]);
        }
    }

    int tid_in_land = tid % 64;
    int land = tid / 64;
    val = WarpReduceSum_NEW<T_ACC>(val);
    // __syncthreads();
    if (tid_in_land == 0) s_rstd[land]=c10::cuda::compat::rsqrt(val/cols + eps);
    __syncthreads();
    trstd=s_rstd[land];
  
    if (tid < tcol) {
        #pragma unroll
        for(int ii=0;ii<Vec;ii++){
        int jj=(tid*Vec+ii)%cols;
        intput_vec[ii] = static_cast<T_ACC>(residual_vec[ii]) *trstd* static_cast<T_ACC>(gamma[jj]);
        }
    }
}


//fuse rms_rope
template <typename T_ACC, typename scalar_t, bool IS_NEOX, bool RESIDUAL, int block_size,
          int VEC_SIZE_Q, int VEC_SIZE_K, int Rot_dim, int NUM_QK_MUL, int num_warp>
__global__ void rms_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,      // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,              // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // nullptr or [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int num_tokens, scalar_t* gamma_q, scalar_t* gamma_k, scalar_t* residual_q, scalar_t* residual_k, scalar_t eps) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    int land = tid / 64;
    int stride = land * Rot_dim;
    if (token_idx >= num_tokens) {
        return;
    }
    const int idx_q = tid * VEC_SIZE_Q;
    const int idx_k = tid * VEC_SIZE_K;
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE_K>;
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;
    const int num_heads_size = num_heads * head_size ;
    const int num_kv_heads_size = num_kv_heads * head_size;

    using vector_q = Vector<scalar_t, VEC_SIZE_Q>;
    using vector_k = Vector<scalar_t, VEC_SIZE_K>;
    __shared__ scalar_t cos_sin_seme[Rot_dim];

    if(tid < 64){
        for(int i=0; i<VEC_SIZE_Q; i++)
        {
            cos_sin_seme[idx_q + i] = cache_ptr[idx_q + i];
        }
    }
    
    __syncthreads();
  
    if constexpr(IS_NEOX) {  // gpt-neox style

        for (int head_idx = 0; head_idx < num_heads / num_warp; head_idx ++) {
                scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp ;
                scalar_t q_vec[VEC_SIZE_Q];
                scalar_t q_data[VEC_SIZE_Q];
                if constexpr (RESIDUAL){
                    scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp;
                    apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_Q, block_size, num_warp, false, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                }else{
                    apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_Q, block_size, NUM_QK_MUL, num_warp, false, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                }
                int sign = idx_q % rot_dim >= embed_dim ? 1 : -1;
                __shared__ scalar_t q_smem[Rot_dim * num_warp];
                #pragma unroll
                for (int i = 0; i < VEC_SIZE_Q; i++) {
                    q_smem[idx_q + i] = q_vec[i];
                }
                __syncthreads();
                if (num_warp == 1)
                {
                    land = 0;
                }
                #pragma unroll
                for (int i = 0; i < VEC_SIZE_Q; i++) {
                    if(sign == -1){
                        q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i) % Rot_dim] - q_smem[(idx_q + i + embed_dim) % head_size + stride ] * cos_sin_seme[(idx_q + i + embed_dim) % head_size]);
                    }else{
                        q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i - embed_dim) % Rot_dim] + q_smem[(idx_q + i - embed_dim) % head_size + stride] * cos_sin_seme[(idx_q + i) % Rot_dim ]);
                    }
                }

                *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;
            }
        if (key != nullptr) {
            for (int head_idx = 0; head_idx < num_kv_heads / num_warp; head_idx ++) {
                scalar_t* k_ptr = key + blockIdx.x * key_stride + head_idx * head_stride * num_warp;
                
                scalar_t k_vec[VEC_SIZE_K];
                scalar_t k_data[VEC_SIZE_K];
                if constexpr (RESIDUAL)
                {
                    scalar_t* residual_k_ptr = residual_k + blockIdx.x * key_stride + head_idx * head_stride * num_warp; ;
                    apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp, false, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                }else{
                    apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp, false, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                }
                int sign = idx_k % rot_dim >= embed_dim ? 1 : -1;
                __shared__ scalar_t k_smem[Rot_dim * num_warp];
                #pragma unroll
                for (int i = 0; i < VEC_SIZE_K; i++) {
                    k_smem[idx_k + i] = k_vec[i];
                }
                __syncthreads();
                if constexpr (num_warp==1)
                {
                    land = 0;
                }
                #pragma unroll
                for (int i = 0; i < VEC_SIZE_K; i++) {
                    if(sign == -1){
                        k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i) % Rot_dim] - k_smem[(idx_k + i + embed_dim) % rot_dim + stride] * cos_sin_seme[(idx_k + i + embed_dim) % rot_dim]);
                    }else{
                        k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i - embed_dim) % Rot_dim] + k_smem[(idx_k + i - embed_dim)%rot_dim + stride] * cos_sin_seme[(idx_k + i) % Rot_dim ]);
                    }
                }
                *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;
            }
        }
    } 
    else {  // gpt-j style
            if constexpr(VEC_SIZE_Q == 1){
                for (int head_idx = 0; head_idx < num_heads / num_warp; head_idx ++) {
                    scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp;
                    scalar_t q_vec[VEC_SIZE_Q];
                    scalar_t q_data[VEC_SIZE_Q];
                    if constexpr (RESIDUAL){
                        scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp, false, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp, false, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                    }
                    __shared__ scalar_t q_smem[Rot_dim * num_warp];
                    q_smem[tid] = q_vec[0];
                    __syncthreads();
                    if(tid % 2 ==0)
                    {
                        q_data[0] = (q_vec[0] * cos_sin_seme[tid%rot_dim / 2] - q_smem[tid + 1] * cos_sin_seme[(tid%rot_dim / 2 + embed_dim) % rot_dim]);
                    }
                    else{
                        q_data[0] = (q_vec[0] * cos_sin_seme[tid%rot_dim / 2] + q_smem[tid - 1] * cos_sin_seme[(tid%rot_dim / 2 + embed_dim) % rot_dim]);
                    }
                    
                    *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;
                }
            }else{
                for (int head_idx = 0; head_idx < num_heads / num_warp; head_idx ++) {
                    scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp;
                    scalar_t q_vec[VEC_SIZE_Q];
                    scalar_t q_data[VEC_SIZE_Q];
                    if constexpr (RESIDUAL){
                        scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp, false, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp, false, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                    }
                    __shared__ scalar_t q_smem[Rot_dim * num_warp];
                    #pragma unroll
                    for(int i = 0;i < VEC_SIZE_K; i++)
                    {
                        q_smem[idx_q + i] = q_vec[i];
                    }
                    
                    __syncthreads();
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_Q; i++) {
                        if((idx_q + i) % 2 == 0)
                        {
                            q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i) % rot_dim / 2] - q_smem[(idx_q + i + 1)%rot_dim + stride] * cos_sin_seme[((idx_q + i) % rot_dim / 2 + embed_dim)]);
                        }
                        else{
                            q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i - 1)%rot_dim / 2] + q_smem[(idx_q + i - 1)%rot_dim + stride] * cos_sin_seme[((idx_q + i -1)%rot_dim / 2 + embed_dim)]);
                        }
                    }
                    *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;

                }
            }
        if (key != nullptr) {
            if constexpr(VEC_SIZE_K == 1){

                for (int head_idx = 0; head_idx < num_kv_heads / num_warp; head_idx ++) {
                    scalar_t* k_ptr = key + blockIdx.x * key_stride + head_idx * head_stride * num_warp;
                    scalar_t k_vec[VEC_SIZE_K];
                    scalar_t k_data[VEC_SIZE_K];
                    if constexpr (RESIDUAL)
                    {
                        scalar_t* residual_k_ptr = residual_k + blockIdx.x * key_stride + head_idx * head_stride * num_warp;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp, false, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp, false, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                    }
                    __shared__ scalar_t k_smem[Rot_dim * num_warp];
                    k_smem[tid] = k_vec[0];
                    __syncthreads();
                    if(tid % 2 ==0)
                    {
                        k_data[0] = (k_vec[0] * cos_sin_seme[tid % rot_dim / 2] - k_smem[tid + 1] * cos_sin_seme[(tid % rot_dim / 2 + embed_dim) % rot_dim]);
                    }
                    else{
                        k_data[0] = (k_vec[0] * cos_sin_seme[tid % rot_dim / 2] + k_smem[tid - 1] * cos_sin_seme[(tid % rot_dim / 2 + embed_dim) % rot_dim]);
                    }
                    *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;

                }
            }else{
                for (int head_idx = 0; head_idx < num_kv_heads / num_warp; head_idx ++) {
                    scalar_t* k_ptr = key + blockIdx.x * num_kv_heads_size + head_idx * head_stride * num_warp;
                    scalar_t k_vec[VEC_SIZE_K];
                    scalar_t k_data[VEC_SIZE_K];
                    if constexpr (RESIDUAL)
                    {
                        scalar_t* residual_k_ptr = residual_k + blockIdx.x * num_kv_heads_size + head_idx * head_stride * num_warp;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp, false, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp, false, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                    }
                    __shared__ scalar_t k_smem[Rot_dim * num_warp];
                    #pragma unroll
                    for(int i = 0;i < VEC_SIZE_K; i++)
                    {
                        k_smem[idx_k + i] = k_vec[i];
                    }
                    
                    __syncthreads();
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i++) {
                        if((idx_k + i) % 2 == 0)
                        {
                            k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i) % rot_dim / 2] - k_smem[(idx_k + i + 1) % rot_dim + stride] * cos_sin_seme[((idx_k + i) % rot_dim / 2 + embed_dim)]);
                        }
                        else{
                            k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i - 1)%rot_dim / 2] + k_smem[(idx_k + i - 1) % rot_dim + stride] * cos_sin_seme[((idx_k + i -1) % rot_dim / 2 + embed_dim)]);
                        }
                    }
                    *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;

                }
            }
        }
    }
}

//fuse rms_rope
template <typename T_ACC, typename scalar_t, bool IS_NEOX, bool RESIDUAL, int block_size,
          int VEC_SIZE_Q, int VEC_SIZE_K, int Rot_dim, int NUM_QK_MUL, int num_warp>
__global__ void rms_rotary_embedding_kernel_pipeline(
    const int64_t* __restrict__ positions,      // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,              // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // nullptr or [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int num_tokens, scalar_t* gamma_q, scalar_t* gamma_k, scalar_t* residual_q, scalar_t* residual_k, scalar_t eps) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }
    
    
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE_K>;
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;
    const int num_heads_size = num_heads * head_size ;
    const int num_kv_heads_size = num_kv_heads * head_size;

    using vector_q = Vector<scalar_t, VEC_SIZE_Q>;
    using vector_k = Vector<scalar_t, VEC_SIZE_K>;
    __shared__ scalar_t cos_sin_seme[Rot_dim];
    int idx_k; 
    if(tid < 64){
        idx_k = tid * VEC_SIZE_K;
        for(int i=0; i<VEC_SIZE_Q; i++)
        {
            cos_sin_seme[idx_k + i] = cache_ptr[idx_k + i];
        }
    }
    
    __syncthreads();
    constexpr  int num_warp_q = num_warp-1;
    if constexpr(IS_NEOX) {  // gpt-neox style
        if(tid>=64)
        {
            int land = (tid-64) / 64;
            int stride = land * Rot_dim;
            const int idx_q = (tid-64) * VEC_SIZE_Q;
            for (int head_idx = 0; head_idx < num_heads / num_warp_q; head_idx ++) {
                    scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q ;
                    scalar_t q_vec[VEC_SIZE_Q];
                    scalar_t q_data[VEC_SIZE_Q];
                    if constexpr (RESIDUAL){
                        scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_Q, block_size, num_warp_q, true, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_Q, block_size, NUM_QK_MUL, num_warp_q, true, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                    }
                    int sign = idx_q % rot_dim >= embed_dim ? 1 : -1;
                    __shared__ scalar_t q_smem[Rot_dim * num_warp_q];
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_Q; i++) {
                        q_smem[idx_q + i] = q_vec[i];
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_Q; i++) {
                        if(sign == -1){
                            q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i) % Rot_dim] - q_smem[(idx_q + i + embed_dim) % head_size + stride] * cos_sin_seme[(idx_q + i + embed_dim) % head_size]);
                        }else{
                            q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i - embed_dim) % Rot_dim] + q_smem[(idx_q + i - embed_dim) % head_size+ stride] * cos_sin_seme[(idx_q + i) % Rot_dim ]);
                        }
                    }

                    *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;
                }
        }else{
                if (key != nullptr) {
                    scalar_t* k_ptr = key + blockIdx.x * key_stride;
                    
                    scalar_t k_vec[VEC_SIZE_K];
                    scalar_t k_data[VEC_SIZE_K];
                    if constexpr (RESIDUAL)
                    {
                        scalar_t* residual_k_ptr = residual_k + blockIdx.x * key_stride;
                        apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, 1, true, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                    }else{
                        apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, 1, true, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                    }
                    int sign = idx_k % rot_dim >= embed_dim ? 1 : -1;
                    __shared__ scalar_t k_smem[Rot_dim];
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i++) {
                        k_smem[idx_k + i] = k_vec[i];
                    }
                    __syncthreads();
                    
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i++) {
                        if(sign == -1){
                            k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i) % Rot_dim] - k_smem[(idx_k + i + embed_dim) % rot_dim] * cos_sin_seme[(idx_k + i + embed_dim) % rot_dim]);
                        }else{
                            k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i - embed_dim) % Rot_dim] + k_smem[(idx_k + i - embed_dim)%rot_dim] * cos_sin_seme[(idx_k + i) % Rot_dim ]);
                        }
                    }
                    *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;
                
            }
        }
        
        
    } else {  // gpt-j style
            if(tid >= 64){
                int land = (tid-64) / 64;
                int stride = land * Rot_dim;
                const int idx_q = (tid-64) * VEC_SIZE_Q;
                    if constexpr(VEC_SIZE_Q == 1){
                        for (int head_idx = 0; head_idx < num_heads / num_warp_q; head_idx ++) {
                            scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q;
                            scalar_t q_vec[VEC_SIZE_Q];
                            scalar_t q_data[VEC_SIZE_Q];
                            if constexpr (RESIDUAL){
                                scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q;
                                apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp_q, true, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                            }else{
                                apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp_q, true, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                            }
                            __shared__ scalar_t q_smem[Rot_dim * num_warp_q];
                            q_smem[tid] = q_vec[0];
                            __syncthreads();
                            if(tid % 2 ==0)
                            {
                                q_data[0] = (q_vec[0] * cos_sin_seme[tid%rot_dim / 2] - q_smem[tid + 1] * cos_sin_seme[(tid%rot_dim / 2 + embed_dim) % rot_dim]);
                            }
                            else{
                                q_data[0] = (q_vec[0] * cos_sin_seme[tid%rot_dim / 2] + q_smem[tid - 1] * cos_sin_seme[(tid%rot_dim / 2 + embed_dim) % rot_dim]);
                            }
                            
                            *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;
                        }
                }else{
                    for (int head_idx = 0; head_idx < num_heads / num_warp_q; head_idx ++) {
                            scalar_t* q_ptr = query + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q;
                            scalar_t q_vec[VEC_SIZE_Q];
                            scalar_t q_data[VEC_SIZE_Q];
                            if constexpr (RESIDUAL){
                                scalar_t* residual_q_ptr = residual_q + blockIdx.x * query_stride + head_idx * head_stride * num_warp_q;
                                apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, num_warp_q, true, true>(q_ptr, gamma_q, residual_q_ptr, head_size, eps, q_vec);
                            }else{
                                apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, num_warp_q, true, true>(q_ptr, gamma_q, head_size, eps, q_vec);
                            }
                            __shared__ scalar_t q_smem[Rot_dim * num_warp_q];
                            #pragma unroll
                            for(int i = 0;i < VEC_SIZE_K; i++)
                            {
                                q_smem[idx_q + i] = q_vec[i];
                            }
                            
                            __syncthreads();
                            #pragma unroll
                            for (int i = 0; i < VEC_SIZE_Q; i++) {
                                if((idx_q + i) % 2 == 0)
                                {
                                    q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i) % rot_dim / 2] - q_smem[(idx_q + i + 1)%rot_dim + stride] * cos_sin_seme[((idx_q + i) % rot_dim / 2 + embed_dim)]);
                                }
                                else{
                                    q_data[i] = (q_vec[i] * cos_sin_seme[(idx_q + i - 1)%rot_dim / 2] + q_smem[(idx_q + i - 1)%rot_dim + stride] * cos_sin_seme[((idx_q + i -1)%rot_dim / 2 + embed_dim)]);
                                }
                            }
                            *(LoadT*)(q_ptr + idx_q)=*(LoadT*)q_data;

                    }
                }
            }else{
                if (key != nullptr) {
                if constexpr(VEC_SIZE_K == 1){

              
                        scalar_t* k_ptr = key + blockIdx.x * key_stride;
                        scalar_t k_vec[VEC_SIZE_K];
                        scalar_t k_data[VEC_SIZE_K];
                        if constexpr (RESIDUAL)
                        {
                            scalar_t* residual_k_ptr = residual_k + blockIdx.x * key_stride;
                            apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, 1, true, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                        }else{
                            apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, 1, true, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                        }
                        __shared__ scalar_t k_smem[Rot_dim];
                        k_smem[tid] = k_vec[0];
                        __syncthreads();
                        if(tid % 2 ==0)
                        {
                            k_data[0] = (k_vec[0] * cos_sin_seme[tid % rot_dim / 2] - k_smem[tid + 1] * cos_sin_seme[(tid % rot_dim / 2 + embed_dim) % rot_dim]);
                        }
                        else{
                            k_data[0] = (k_vec[0] * cos_sin_seme[tid % rot_dim / 2] + k_smem[tid - 1] * cos_sin_seme[(tid % rot_dim / 2 + embed_dim) % rot_dim]);
                        }
                        *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;

                    
                }else{
        
                        scalar_t* k_ptr = key + blockIdx.x * num_kv_heads_size;
                        scalar_t k_vec[VEC_SIZE_K];
                        scalar_t k_data[VEC_SIZE_K];
                        if constexpr (RESIDUAL)
                        {
                            scalar_t* residual_k_ptr = residual_k + blockIdx.x * num_kv_heads_size;
                            apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE_K, block_size, 1, true, false>(k_ptr, gamma_k, residual_k_ptr, head_size, eps, k_vec);
                        }else{
                            apply_rmsnorm<T_ACC, scalar_t, VEC_SIZE_K, block_size, NUM_QK_MUL, 1, true, false>(k_ptr, gamma_k, head_size, eps, k_vec);
                        }
                        __shared__ scalar_t k_smem[Rot_dim];
                        #pragma unroll
                        for(int i = 0;i < VEC_SIZE_K; i++)
                        {
                            k_smem[idx_k + i] = k_vec[i];
                        }
                        
                        __syncthreads();
                        #pragma unroll
                        for (int i = 0; i < VEC_SIZE_K; i++) {
                            if((idx_k + i) % 2 == 0)
                            {
                                k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i) % rot_dim / 2] - k_smem[(idx_k + i + 1) % rot_dim] * cos_sin_seme[((idx_k + i) % rot_dim / 2 + embed_dim)]);
                            }
                            else{
                                k_data[i] = (k_vec[i] * cos_sin_seme[(idx_k + i - 1)%rot_dim / 2] + k_smem[(idx_k + i - 1) % rot_dim] * cos_sin_seme[((idx_k + i -1) % rot_dim / 2 + embed_dim)]);
                            }
                        }
                        *(LoadT*)(k_ptr + idx_k)=*(LoadT*)k_data;

                
                }
            }
            }
            
        
    }
}

template<typename T, int WIDTH>
__device__ __forceinline__ T WarpReduceSum_Local(T val) {
    #pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>=1) {
        val += __shfl_xor(val, mask);
    }
    return val;
}
template <typename T_ACC, typename scalar_t, int VEC_SIZE, bool HAS_RESIDUAL>
__device__ __forceinline__ T_ACC apply_residual_and_calc_sq(
    scalar_t* r_data_low,     
    scalar_t* r_data_high,   
    scalar_t* res_head_ptr,   
    int offset_low,            
    int offset_high            
) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
    if constexpr (HAS_RESIDUAL) {
        scalar_t r_res_low[VEC_SIZE];
        scalar_t r_res_high[VEC_SIZE];
        *(LoadT*)r_res_low = *(LoadT*)(res_head_ptr + offset_low);
        *(LoadT*)r_res_high = *(LoadT*)(res_head_ptr + offset_high);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            r_res_low[i] = r_res_low[i] + r_data_low[i];
            r_res_high[i] = r_res_high[i] + r_data_high[i];
            r_data_low[i] = r_res_low[i];
            r_data_high[i] = r_res_high[i];
        }
        *(LoadT*)(res_head_ptr + offset_low) = *(LoadT*)r_res_low;
        *(LoadT*)(res_head_ptr + offset_high) = *(LoadT*)r_res_high;
    }

    T_ACC local_sum_sq = 0;
    #pragma unroll VEC_SIZE
    for (int i = 0; i < VEC_SIZE; ++i) {
        T_ACC low = static_cast<T_ACC>(r_data_low[i]);
        T_ACC high = static_cast<T_ACC>(r_data_high[i]);
        local_sum_sq += low * low;
        local_sum_sq += high * high;
    }

    return local_sum_sq;
}



//8, 8 每个head有8个thread, 每个thread处理8个元素
template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX,int VEC_SIZE, int THEAD_PER_HEAD, bool HAS_NORM>
__global__ void opt_gammarms_rope_qwen3_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,   
    scalar_t* __restrict__ v,      
    const scalar_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ k_buffer,
    uint8_t* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    uint8_t* __restrict__ q_rope_out,
    uint8_t* __restrict__ k_rope_out,
    uint8_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,
    const int64_t query_stride,         
    const int64_t key_stride,           
    const int64_t v_stride,
    const int64_t head_stride_q,       
    const int64_t head_stride_k,  
    const int q_rope_out_stride_n,    
    const int q_rope_out_stride_h,
    const int q_rope_out_stride_p,
    const int k_buffer_stride_n,
    const int k_buffer_stride_h,
    const int k_buffer_stride_p,
    const int v_buffer_stride_n,
    const int v_buffer_stride_h,
    const int v_buffer_stride_p,
    scalar_t* gamma_q,
    scalar_t* gamma_k,
    scalar_t* residual_q,
    scalar_t* residual_k,
    scalar_t eps,
    int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token, 
    const int tokens_per_block,
    const int cache_fp8_type,
    // --- 新增 M-RoPE 参数 ---
    const int mrope_section_t,
    const int mrope_section_h,
    const int mrope_section_w,
    const bool is_interleaved) 
{
    
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 128;
    constexpr int HALF_ROT = 64; 

    const int tid = threadIdx.x;

    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;


    if (local_token_idx >= tokens_per_block) return;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    if (global_token_idx >= num_tokens) return;

    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * HEAD_SIZE;

    // positions: [3, num_tokens]
    const int64_t pos_t = positions[0 * num_tokens + global_token_idx];
    const int64_t pos_h = positions[1 * num_tokens + global_token_idx];
    const int64_t pos_w = positions[2 * num_tokens + global_token_idx];

    const int half_rot_dim = rot_dim / 2;

    // shared memory layout:
    // [0, HALF_ROT)         -> cos_row
    // [HALF_ROT, HEAD_SIZE) -> sin_row
    for (int i = lane; i < HALF_ROT; i += threads_per_token) {
        scalar_t c = scalar_t(1);
        scalar_t s = scalar_t(0);

        if (i < half_rot_dim) {
            bool use_h = false;
            bool use_w = false;
            bool use_t = false;

            if (is_interleaved) {
                use_h = ((i % 3) == 1) && (i <= 3 * mrope_section_h);
                use_w = ((i % 3) == 2) && (i <= 3 * mrope_section_w);
                use_t = !(use_h || use_w);
            } else {
                const int t_end = mrope_section_t;
                const int h_end = t_end + mrope_section_h;
                use_t = (i < t_end);
                use_h = (i >= t_end) && (i < h_end);
                use_w = (i >= h_end) && (i < half_rot_dim);
            }

            const int64_t sel_pos = use_h ? pos_h : (use_w ? pos_w : pos_t);

            c = cos_sin_cache[sel_pos * rot_dim + i];
            s = cos_sin_cache[sel_pos * rot_dim + half_rot_dim + i];
        }

        my_s_cos_sin[i] = c;
        my_s_cos_sin[HALF_ROT + i] = s;
    }

    __syncthreads(); 

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    if (lane < q_boundary) {
        const int q_head_idx = lane / THEAD_PER_HEAD;
        const int q_lane_in_head = lane % THEAD_PER_HEAD;

        uint8_t* q_rope_out_ptr = q_rope_out + global_token_idx * q_rope_out_stride_n + q_head_idx * q_rope_out_stride_h;
        scalar_t* q_head_ptr = query + global_token_idx * query_stride + q_head_idx * head_stride_q;
        scalar_t* res_q_head_ptr = HAS_RESIDUAL ? (residual_q + global_token_idx * query_stride + q_head_idx * head_stride_q) : nullptr;

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        float f_q_low[VEC_SIZE];
        float f_q_high[VEC_SIZE];
        uint8_t u8_q_low[VEC_SIZE];
        uint8_t u8_q_high[VEC_SIZE];
        scalar_t r_q_low[VEC_SIZE];
        scalar_t r_q_high[VEC_SIZE];

        int offset_low = q_lane_in_head * VEC_SIZE;
        int offset_high = HALF_ROT + q_lane_in_head * VEC_SIZE;

        *(LoadT*)r_q_low = *(LoadT*)(q_head_ptr + offset_low);
        *(LoadT*)r_q_high = *(LoadT*)(q_head_ptr + offset_high);

        T_ACC inv_rms = 1.0;
        T_ACC sum_sq = 0.0;
        if constexpr (HAS_NORM) {
            sum_sq = apply_residual_and_calc_sq<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                r_q_low, r_q_high, res_q_head_ptr, offset_low, offset_high
            );

            sum_sq = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq);
            inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        }

        const scalar_t* cache_ptr = my_s_cos_sin;

        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];
            *(LoadT*)r_cos_low = *(LoadT*)(cache_ptr + offset_low);
            *(LoadT*)r_sin_low = *(LoadT*)(cache_ptr + rot_dim / 2 + offset_low);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    r_q_low[i] =
                        static_cast<T_ACC>(r_q_low[i]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_low + i]) + static_cast<T_ACC>(1));
                    r_q_high[i] =
                        static_cast<T_ACC>(r_q_high[i]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_high + i]) + static_cast<T_ACC>(1));
                }

                scalar_t q_l = r_q_low[i];
                scalar_t q_h = r_q_high[i];
                scalar_t c = r_cos_low[i];
                scalar_t s = r_sin_low[i];

                r_q_low[i] = q_l * c - q_h * s;
                r_q_high[i] = q_l * s + q_h * c;
            }
        } else {
            using LoadCacheT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE / 2>;
            scalar_t c_low[VEC_SIZE / 2], s_low[VEC_SIZE / 2];
            scalar_t c_high[VEC_SIZE / 2], s_high[VEC_SIZE / 2];
            int cache_offset_low = offset_low / 2;
            int cache_offset_high = offset_high / 2;

            *(LoadCacheT*)c_low = *(LoadCacheT*)(cache_ptr + cache_offset_low);
            *(LoadCacheT*)s_low = *(LoadCacheT*)(cache_ptr + rot_dim / 2 + cache_offset_low);
            *(LoadCacheT*)c_high = *(LoadCacheT*)(cache_ptr + cache_offset_high);
            *(LoadCacheT*)s_high = *(LoadCacheT*)(cache_ptr + rot_dim / 2 + cache_offset_high);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                int c_idx = i / 2;
                if constexpr (HAS_NORM) {
                    r_q_low[i] =
                        static_cast<T_ACC>(r_q_low[i]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_low + i]) + static_cast<T_ACC>(1));
                    r_q_low[i + 1] =
                        static_cast<T_ACC>(r_q_low[i + 1]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_low + i + 1]) + static_cast<T_ACC>(1));
                    r_q_high[i] =
                        static_cast<T_ACC>(r_q_high[i]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_high + i]) + static_cast<T_ACC>(1));
                    r_q_high[i + 1] =
                        static_cast<T_ACC>(r_q_high[i + 1]) *
                        inv_rms *
                        (static_cast<T_ACC>(gamma_q[offset_high + i + 1]) + static_cast<T_ACC>(1));
                }

                scalar_t q0 = r_q_low[i];
                scalar_t q1 = r_q_low[i + 1];
                scalar_t c = c_low[c_idx];
                scalar_t s = s_low[c_idx];
                r_q_low[i] = q0 * c - q1 * s;
                r_q_low[i + 1] = q1 * c + q0 * s;

                scalar_t qh0 = r_q_high[i];
                scalar_t qh1 = r_q_high[i + 1];
                scalar_t ch = c_high[c_idx];
                scalar_t sh = s_high[c_idx];
                r_q_high[i] = qh0 * ch - qh1 * sh;
                r_q_high[i + 1] = qh1 * ch + qh0 * sh;
            }
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f_q_low[i] = static_cast<float>(r_q_low[i]);
            f_q_high[i] = static_cast<float>(r_q_high[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_q_low, u8_q_low);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_q_high, u8_q_high);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_q_low, u8_q_low);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_q_high, u8_q_high);
        }

        *(LoadT8*)(q_rope_out_ptr + offset_low) = *(LoadT8*)u8_q_low;
        *(LoadT8*)(q_rope_out_ptr + offset_high) = *(LoadT8*)u8_q_high;
    }

    const int total_threads_needed_kv = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads = total_threads_needed_kv + (num_kv_heads * THEAD_PER_HEAD);
    const int64_t cache_loc = kv_cache_loc[global_token_idx];
    const int loc_id = cache_loc / page_size;
    const int loc_offset = cache_loc % page_size;

    if (lane >= q_boundary && lane < total_threads_needed_kv && key != nullptr) {
        const int k_lane_abs = lane - q_boundary;
        const int kv_head_idx = k_lane_abs / THEAD_PER_HEAD;
        const int k_lane_in_head = k_lane_abs % THEAD_PER_HEAD;

        uint8_t* k_rope_out_ptr = k_rope_out + global_token_idx * key_stride + kv_head_idx * head_stride_k;
        uint8_t* k_buffer_ptr = k_buffer + loc_id * k_buffer_stride_n + kv_head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;

        scalar_t* k_head_ptr = key + global_token_idx * key_stride + kv_head_idx * head_stride_k;
        scalar_t* res_k_head_ptr = HAS_RESIDUAL ? (residual_k + global_token_idx * key_stride + kv_head_idx * head_stride_k) : nullptr;

        using LoadTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        float f_k_low[VEC_SIZE];
        float f_k_high[VEC_SIZE];
        uint8_t u8_k_low[VEC_SIZE];
        uint8_t u8_k_high[VEC_SIZE];
        scalar_t r_k_low[VEC_SIZE];
        scalar_t r_k_high[VEC_SIZE];

        int offset_low = k_lane_in_head * VEC_SIZE;
        int offset_high = HALF_ROT + k_lane_in_head * VEC_SIZE;

        *(LoadTK*)r_k_low = *(LoadTK*)(k_head_ptr + offset_low);
        *(LoadTK*)r_k_high = *(LoadTK*)(k_head_ptr + offset_high);

        T_ACC inv_rms_k = 1.0;
        T_ACC sum_sq_k = 0.0;
        if constexpr (HAS_NORM) {
            sum_sq_k = apply_residual_and_calc_sq<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                r_k_low, r_k_high, res_k_head_ptr, offset_low, offset_high
            );

            sum_sq_k = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq_k);
            inv_rms_k = c10::cuda::compat::rsqrt(sum_sq_k / HEAD_SIZE + static_cast<T_ACC>(eps));
        }

        const scalar_t* cache_ptr_k = my_s_cos_sin;

        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];

            *(LoadTK*)r_cos_low = *(LoadTK*)(cache_ptr_k + offset_low);
            *(LoadTK*)r_sin_low = *(LoadTK*)(cache_ptr_k + rot_dim / 2 + offset_low);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    r_k_low[i] =
                        static_cast<T_ACC>(r_k_low[i]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_low + i]) + static_cast<T_ACC>(1));
                    r_k_high[i] =
                        static_cast<T_ACC>(r_k_high[i]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_high + i]) + static_cast<T_ACC>(1));
                }

                scalar_t k_l = r_k_low[i];
                scalar_t k_h = r_k_high[i];
                scalar_t c = r_cos_low[i];
                scalar_t s = r_sin_low[i];

                r_k_low[i] = k_l * c - k_h * s;
                r_k_high[i] = k_l * s + k_h * c;
            }
        } else {
            using LoadCacheTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE / 2>;
            scalar_t r_cos_low[VEC_SIZE / 2], r_sin_low[VEC_SIZE / 2];
            scalar_t r_cos_high[VEC_SIZE / 2], r_sin_high[VEC_SIZE / 2];
            int cache_offset_low = offset_low / 2;
            int cache_offset_high = offset_high / 2;

            *(LoadCacheTK*)r_cos_low = *(LoadCacheTK*)(cache_ptr_k + cache_offset_low);
            *(LoadCacheTK*)r_sin_low = *(LoadCacheTK*)(cache_ptr_k + rot_dim / 2 + cache_offset_low);
            *(LoadCacheTK*)r_cos_high = *(LoadCacheTK*)(cache_ptr_k + cache_offset_high);
            *(LoadCacheTK*)r_sin_high = *(LoadCacheTK*)(cache_ptr_k + rot_dim / 2 + cache_offset_high);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                int c_idx = i / 2;
                if constexpr (HAS_NORM) {
                    r_k_low[i] =
                        static_cast<T_ACC>(r_k_low[i]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_low + i]) + static_cast<T_ACC>(1));
                    r_k_low[i + 1] =
                        static_cast<T_ACC>(r_k_low[i + 1]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_low + i + 1]) + static_cast<T_ACC>(1));
                    r_k_high[i] =
                        static_cast<T_ACC>(r_k_high[i]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_high + i]) + static_cast<T_ACC>(1));
                    r_k_high[i + 1] =
                        static_cast<T_ACC>(r_k_high[i + 1]) *
                        inv_rms_k *
                        (static_cast<T_ACC>(gamma_k[offset_high + i + 1]) + static_cast<T_ACC>(1));
                }

                scalar_t k0 = r_k_low[i];
                scalar_t k1 = r_k_low[i + 1];
                scalar_t c = r_cos_low[c_idx];
                scalar_t s = r_sin_low[c_idx];
                r_k_low[i] = k0 * c - k1 * s;
                r_k_low[i + 1] = k1 * c + k0 * s;

                scalar_t kh0 = r_k_high[i];
                scalar_t kh1 = r_k_high[i + 1];
                scalar_t ch = r_cos_high[c_idx];
                scalar_t sh = r_sin_high[c_idx];
                r_k_high[i] = kh0 * ch - kh1 * sh;
                r_k_high[i + 1] = kh1 * ch + kh0 * sh;
            }
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f_k_low[i] = static_cast<float>(r_k_low[i]);
            f_k_high[i] = static_cast<float>(r_k_high[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_k_low, u8_k_low);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_k_high, u8_k_high);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_k_low, u8_k_low);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_k_high, u8_k_high);
        }

        *(LoadTK8*)(k_rope_out_ptr + offset_low) = *(LoadTK8*)u8_k_low;
        *(LoadTK8*)(k_rope_out_ptr + offset_high) = *(LoadTK8*)u8_k_high;

        *(LoadTK8*)(k_buffer_ptr + offset_low) = *(LoadTK8*)u8_k_low;
        *(LoadTK8*)(k_buffer_ptr + offset_high) = *(LoadTK8*)u8_k_high;
    }

    if (lane >= total_threads_needed_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_needed_kv;
        const int v_head_idx = v_lane_abs / THEAD_PER_HEAD;
        const int v_lane_in_head = v_lane_abs % THEAD_PER_HEAD;

        uint8_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_stride + v_head_idx * head_stride_k;
        uint8_t* v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + v_head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + global_token_idx * v_stride + v_head_idx * head_stride_k;
        using LoadTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        float f_v_low[VEC_SIZE];
        float f_v_high[VEC_SIZE];
        uint8_t u8_v_low[VEC_SIZE];
        uint8_t u8_v_high[VEC_SIZE];
        scalar_t r_v_low[VEC_SIZE];
        scalar_t r_v_high[VEC_SIZE];

        int offset_low = v_lane_in_head * VEC_SIZE;
        int offset_high = HALF_ROT + v_lane_in_head * VEC_SIZE;

        *(LoadTK*)r_v_low = *(LoadTK*)(v_head_ptr + offset_low);
        *(LoadTK*)r_v_high = *(LoadTK*)(v_head_ptr + offset_high);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f_v_low[i] = static_cast<float>(r_v_low[i]);
            f_v_high[i] = static_cast<float>(r_v_high[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_v_low, u8_v_low);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f_v_high, u8_v_high);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_v_low, u8_v_low);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f_v_high, u8_v_high);
        }

        *(LoadTK8*)(v_rope_out_ptr + offset_low) = *(LoadTK8*)u8_v_low;
        *(LoadTK8*)(v_rope_out_ptr + offset_high) = *(LoadTK8*)u8_v_high;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            v_buffer_ptr[offset_low * v_buffer_stride_p + i * v_buffer_stride_p] = u8_v_low[i];
            v_buffer_ptr[offset_high * v_buffer_stride_p + i * v_buffer_stride_p] = u8_v_high[i];
        }
    }
}
template <typename T_ACC, typename scalar_t, int VEC_SIZE, bool HAS_RESIDUAL>
__device__ __forceinline__ T_ACC apply_residual_and_calc_sq_4vec(
    scalar_t* v0, scalar_t* v1, scalar_t* v2, scalar_t* v3,
    scalar_t* res_ptr, int o0, int o1, int o2, int o3) {
    T_ACC local_sum = 0;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if constexpr (HAS_RESIDUAL) {
            v0[i] += res_ptr[o0 + i];
            v1[i] += res_ptr[o1 + i];
            v2[i] += res_ptr[o2 + i];
            v3[i] += res_ptr[o3 + i];
        }
        local_sum += static_cast<T_ACC>(v0[i]) * static_cast<T_ACC>(v0[i]);
        local_sum += static_cast<T_ACC>(v1[i]) * static_cast<T_ACC>(v1[i]);
        local_sum += static_cast<T_ACC>(v2[i]) * static_cast<T_ACC>(v2[i]);
        local_sum += static_cast<T_ACC>(v3[i]) * static_cast<T_ACC>(v3[i]);
    }
    return local_sum;
}
template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, int THEAD_PER_HEAD, bool HAS_NORM>
__global__ void opt_gammarms_rope_qwen3_rotDim64_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ v,
    const scalar_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ k_buffer,
    uint8_t* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    uint8_t* __restrict__ q_rope_out,
    uint8_t* __restrict__ k_rope_out,
    uint8_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t v_stride,
    const int64_t head_stride_q,
    const int64_t head_stride_k,
    const int q_rope_out_stride_n,
    const int q_rope_out_stride_h,
    const int q_rope_out_stride_p,
    const int k_buffer_stride_n,
    const int k_buffer_stride_h,
    const int k_buffer_stride_p,
    const int v_buffer_stride_n,
    const int v_buffer_stride_h,
    const int v_buffer_stride_p,
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k,
    scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k,
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token,
    const int tokens_per_block,
    const int cache_fp8_type,
    const int mrope_section_t,
    const int mrope_section_h,
    const int mrope_section_w,
    const bool is_interleaved)
{
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 128;
    constexpr int HALF_ROT = 32;   // rot_dim = 64 => half_rot = 32

    const int tid = threadIdx.x;
    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    if (local_token_idx >= tokens_per_block) return;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    if (global_token_idx >= num_tokens) return;

    // shared memory layout for rot_dim=64:
    // [0, 32)   -> cos_row
    // [32, 64)  -> sin_row
    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * rot_dim;

    const int64_t pos_t = positions[0 * num_tokens + global_token_idx];
    const int64_t pos_h = positions[1 * num_tokens + global_token_idx];
    const int64_t pos_w = positions[2 * num_tokens + global_token_idx];

    // 只构造 32 个 cos + 32 个 sin
    for (int i = lane; i < HALF_ROT; i += threads_per_token) {
        scalar_t c = static_cast<scalar_t>(1);
        scalar_t s = static_cast<scalar_t>(0);

        bool use_t = false;
        bool use_h = false;
        bool use_w = false;

        if (is_interleaved) {
            // 与 Triton 保持一致：使用 <=
            use_h = ((i % 3) == 1) && (i <= 3 * mrope_section_h);
            use_w = ((i % 3) == 2) && (i <= 3 * mrope_section_w);
            use_t = !(use_h || use_w);
        } else {
            const int t_end = mrope_section_t;
            const int h_end = t_end + mrope_section_h;
            use_t = (i < t_end);
            use_h = (i >= t_end) && (i < h_end);
            use_w = (i >= h_end) && (i < HALF_ROT);
        }

        // 对齐 Triton 的 mask + sum 语义
        c = static_cast<scalar_t>(0);
        s = static_cast<scalar_t>(0);

        if (use_t) {
            c += cos_sin_cache[pos_t * rot_dim + i];
            s += cos_sin_cache[pos_t * rot_dim + HALF_ROT + i];
        }
        if (use_h) {
            c += cos_sin_cache[pos_h * rot_dim + i];
            s += cos_sin_cache[pos_h * rot_dim + HALF_ROT + i];
        }
        if (use_w) {
            c += cos_sin_cache[pos_w * rot_dim + i];
            s += cos_sin_cache[pos_w * rot_dim + HALF_ROT + i];
        }

        my_s_cos_sin[i] = c;
        my_s_cos_sin[HALF_ROT + i] = s;
    }

    __syncthreads();

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    const int total_threads_need_kv = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads = total_threads_need_kv + (num_kv_heads * THEAD_PER_HEAD);

    const int64_t cache_loc = kv_cache_loc[global_token_idx];
    const int loc_id = cache_loc / page_size;
    const int loc_offset = cache_loc % page_size;

    if (lane < total_threads_need_kv) {
        const bool is_query = lane < q_boundary;
        const int head_idx = is_query ? (lane / THEAD_PER_HEAD) : ((lane - q_boundary) / THEAD_PER_HEAD);
        const int lane_in_head = is_query ? (lane % THEAD_PER_HEAD) : ((lane - q_boundary) % THEAD_PER_HEAD);

        uint8_t* rope_out_ptr = is_query
            ? (q_rope_out + global_token_idx * q_rope_out_stride_n + head_idx * q_rope_out_stride_h)
            : (k_rope_out + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* head_ptr = is_query
            ? (query + global_token_idx * query_stride + head_idx * head_stride_q)
            : (key + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* res_head_ptr = HAS_RESIDUAL
            ? (is_query
                ? (residual_q + global_token_idx * query_stride + head_idx * head_stride_q)
                : (residual_k + global_token_idx * key_stride + head_idx * head_stride_k))
            : nullptr;

        const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

        // rot_dim=64: 只对前 64 维做 RoPE
        // 前 64 维分成两个 32-d half：
        // v0/v1 -> 0..31
        // v2/v3 -> 32..63
        // 其中 NeoX 时只旋转前 64 维，后 64 维保持原样
        int o0 = lane_in_head * VEC_SIZE;   // 0..28
        int o1 = o0 + 32;                   // 32..60
        int o2 = o0 + 64;                   // 64..92
        int o3 = o0 + 96;                   // 96..124

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(head_ptr + o3);

        T_ACC inv_rms = 1.0;
        T_ACC sum_sq = 0.0;
        if constexpr (HAS_NORM) {
            sum_sq = apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                v0, v1, v2, v3, res_head_ptr, o0, o1, o2, o3
            );
            sum_sq = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq);
            inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        }

        if constexpr (IS_NEOX) {
            scalar_t r_cos[VEC_SIZE], r_sin[VEC_SIZE];
            *(LoadT*)r_cos = *(LoadT*)(my_s_cos_sin + o0);
            *(LoadT*)r_sin = *(LoadT*)(my_s_cos_sin + HALF_ROT + o0);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                T_ACC s0 = static_cast<T_ACC>(v0[i]);
                T_ACC s1 = static_cast<T_ACC>(v1[i]);
                T_ACC s2 = static_cast<T_ACC>(v2[i]);
                T_ACC s3 = static_cast<T_ACC>(v3[i]);

                if constexpr (HAS_NORM) {
                    s0 = static_cast<T_ACC>(v0[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s1 = static_cast<T_ACC>(v1[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s2 = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1);
                    s3 = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1);
                }

                // 只旋转前 64 维：v0 <-> v1
                v0[i] = s0 * r_cos[i] - s1 * r_sin[i];
                v1[i] = s0 * r_sin[i] + s1 * r_cos[i];

                // 后 64 维不做 rope，只保留 norm/residual 后结果
                v2[i] = static_cast<scalar_t>(s2);
                v3[i] = static_cast<scalar_t>(s3);
            }
        } else {
            // 非 NEOX：前 64 维按相邻偶奇位旋转，后 64 维保持不变
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                int idx_c0 = (o0 + i) / 2;
                int idx_c1 = (o1 + i) / 2;

                scalar_t cos0 = my_s_cos_sin[idx_c0];
                scalar_t sin0 = my_s_cos_sin[HALF_ROT + idx_c0];

                T_ACC s0_0 = static_cast<T_ACC>(v0[i]);
                T_ACC s0_1 = static_cast<T_ACC>(v0[i + 1]);
                T_ACC s1_0 = static_cast<T_ACC>(v1[i]);
                T_ACC s1_1 = static_cast<T_ACC>(v1[i + 1]);

                if constexpr (HAS_NORM) {
                    s0_0 = static_cast<T_ACC>(v0[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s0_1 = static_cast<T_ACC>(v0[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i + 1] + 1);
                    s1_0 = static_cast<T_ACC>(v1[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s1_1 = static_cast<T_ACC>(v1[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i + 1] + 1);
                }

                v0[i]     = s0_0 * cos0 - s0_1 * sin0;
                v0[i + 1] = s0_1 * cos0 + s0_0 * sin0;

                scalar_t cos1 = my_s_cos_sin[idx_c1];
                scalar_t sin1 = my_s_cos_sin[HALF_ROT + idx_c1];

                v1[i]     = s1_0 * cos1 - s1_1 * sin1;
                v1[i + 1] = s1_1 * cos1 + s1_0 * sin1;
            }

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    v2[i] = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1);
                    v3[i] = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1);
                } else {
                    v2[i] = static_cast<T_ACC>(v2[i]);
                    v3[i] = static_cast<T_ACC>(v3[i]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
        }

        *(LoadT8*)(rope_out_ptr + o0) = *(LoadT8*)u8_0;
        *(LoadT8*)(rope_out_ptr + o1) = *(LoadT8*)u8_1;
        *(LoadT8*)(rope_out_ptr + o2) = *(LoadT8*)u8_2;
        *(LoadT8*)(rope_out_ptr + o3) = *(LoadT8*)u8_3;

        if (!is_query) {
            const int k_buffer_offset =
                loc_id * k_buffer_stride_n + head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;
            *(LoadT8*)(k_buffer + k_buffer_offset + o0) = *(LoadT8*)u8_0;
            *(LoadT8*)(k_buffer + k_buffer_offset + o1) = *(LoadT8*)u8_1;
            *(LoadT8*)(k_buffer + k_buffer_offset + o2) = *(LoadT8*)u8_2;
            *(LoadT8*)(k_buffer + k_buffer_offset + o3) = *(LoadT8*)u8_3;
        }
    }

    if (lane >= total_threads_need_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_need_kv;
        const int head_idx = (v_lane_abs / THEAD_PER_HEAD);
        const int lane_in_head = (v_lane_abs % THEAD_PER_HEAD);

        uint8_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_stride + head_idx * head_stride_k;
        uint8_t* v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + global_token_idx * v_stride + head_idx * head_stride_k;

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        int o0 = lane_in_head * VEC_SIZE;
        int o1 = o0 + 32;
        int o2 = o0 + 64;
        int o3 = o0 + 96;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(v_head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(v_head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(v_head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(v_head_ptr + o3);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
        }

        *(LoadTK8*)(v_rope_out_ptr + o0) = *(LoadTK8*)u8_0;
        *(LoadTK8*)(v_rope_out_ptr + o1) = *(LoadTK8*)u8_1;
        *(LoadTK8*)(v_rope_out_ptr + o2) = *(LoadTK8*)u8_2;
        *(LoadTK8*)(v_rope_out_ptr + o3) = *(LoadTK8*)u8_3;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            v_buffer_ptr[o0 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_0[i];
            v_buffer_ptr[o1 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_1[i];
            v_buffer_ptr[o2 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_2[i];
            v_buffer_ptr[o3 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_3[i];
        }
    }
}


template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, int THEAD_PER_HEAD, bool HAS_NORM>
__global__ void opt_gammarms_rope_qwen3_headDim256_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ v,
    const scalar_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ k_buffer,
    uint8_t* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    uint8_t* __restrict__ q_rope_out,
    uint8_t* __restrict__ k_rope_out,
    uint8_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,                  // 固定应为 128
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t v_stride,
    const int64_t head_stride_q,
    const int64_t head_stride_k,
    const int q_rope_out_stride_n,
    const int q_rope_out_stride_h,
    const int q_rope_out_stride_p,
    const int k_buffer_stride_n,
    const int k_buffer_stride_h,
    const int k_buffer_stride_p,
    const int v_buffer_stride_n,
    const int v_buffer_stride_h,
    const int v_buffer_stride_p,
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k,
    scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k,
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token,
    const int tokens_per_block,
    const int cache_fp8_type,
    const int mrope_section_t,
    const int mrope_section_h,
    const int mrope_section_w,
    const bool is_interleaved)
{
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 256;
    constexpr int HALF_ROT = 64;   // rot_dim / 2，固定 64

    const int tid = threadIdx.x;
    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    if (local_token_idx >= tokens_per_block) return;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    if (global_token_idx >= num_tokens) return;

    // shared memory layout for rot_dim=128:
    // [0, 64)    -> cos_row
    // [64, 128)  -> sin_row
    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * rot_dim;

    // positions: [3, num_tokens]
    const int64_t pos_t = positions[0 * num_tokens + global_token_idx];
    const int64_t pos_h = positions[1 * num_tokens + global_token_idx];
    const int64_t pos_w = positions[2 * num_tokens + global_token_idx];

    // 构造 mRoPE cos/sin row
    for (int i = lane; i < HALF_ROT; i += threads_per_token) {
        bool use_t = false;
        bool use_h = false;
        bool use_w = false;

        if (is_interleaved) {
            use_h = ((i % 3) == 1) && (i <= 3 * mrope_section_h);
            use_w = ((i % 3) == 2) && (i <= 3 * mrope_section_w);
            use_t = !(use_h || use_w);
        } else {
            const int t_end = mrope_section_t;
            const int h_end = t_end + mrope_section_h;
            use_t = (i < t_end);
            use_h = (i >= t_end) && (i < h_end);
            use_w = (i >= h_end) && (i < HALF_ROT);
        }

        // Triton 语义：mask + sum
        scalar_t c = static_cast<scalar_t>(0);
        scalar_t s = static_cast<scalar_t>(0);

        if (use_t) {
            c += cos_sin_cache[pos_t * rot_dim + i];
            s += cos_sin_cache[pos_t * rot_dim + HALF_ROT + i];
        }
        if (use_h) {
            c += cos_sin_cache[pos_h * rot_dim + i];
            s += cos_sin_cache[pos_h * rot_dim + HALF_ROT + i];
        }
        if (use_w) {
            c += cos_sin_cache[pos_w * rot_dim + i];
            s += cos_sin_cache[pos_w * rot_dim + HALF_ROT + i];
        }

        my_s_cos_sin[i] = c;
        my_s_cos_sin[HALF_ROT + i] = s;
    }

    __syncthreads();

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    const int total_threads_need_kv = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads = total_threads_need_kv + (num_kv_heads * THEAD_PER_HEAD);

    const int64_t cache_loc = kv_cache_loc[global_token_idx];
    const int loc_id = cache_loc / page_size;
    const int loc_offset = cache_loc % page_size;

    if (lane < total_threads_need_kv) {
        const bool is_query = lane < q_boundary;
        const int head_idx = is_query ? (lane / THEAD_PER_HEAD) : ((lane - q_boundary) / THEAD_PER_HEAD);
        const int lane_in_head = is_query ? (lane % THEAD_PER_HEAD) : ((lane - q_boundary) % THEAD_PER_HEAD);

        uint8_t* rope_out_ptr = is_query
            ? (q_rope_out + global_token_idx * q_rope_out_stride_n + head_idx * q_rope_out_stride_h)
            : (k_rope_out + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* head_ptr = is_query
            ? (query + global_token_idx * query_stride + head_idx * head_stride_q)
            : (key + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* res_head_ptr = HAS_RESIDUAL
            ? (is_query
                ? (residual_q + global_token_idx * query_stride + head_idx * head_stride_q)
                : (residual_k + global_token_idx * key_stride + head_idx * head_stride_k))
            : nullptr;

        const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

        // 8 组向量，每组 32 间隔
        int o0 = lane_in_head * VEC_SIZE;   //   0..31
        int o1 = o0 + 32;                   //  32..63
        int o2 = o0 + 64;                   //  64..95
        int o3 = o0 + 96;                   //  96..127
        int o4 = o0 + 128;                  // 128..159
        int o5 = o0 + 160;                  // 160..191
        int o6 = o0 + 192;                  // 192..223
        int o7 = o0 + 224;                  // 224..255

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        scalar_t v4[VEC_SIZE], v5[VEC_SIZE], v6[VEC_SIZE], v7[VEC_SIZE];

        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        float f4[VEC_SIZE], f5[VEC_SIZE], f6[VEC_SIZE], f7[VEC_SIZE];

        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
        uint8_t u8_4[VEC_SIZE], u8_5[VEC_SIZE], u8_6[VEC_SIZE], u8_7[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(head_ptr + o3);
        *(LoadT*)v4 = *(LoadT*)(head_ptr + o4);
        *(LoadT*)v5 = *(LoadT*)(head_ptr + o5);
        *(LoadT*)v6 = *(LoadT*)(head_ptr + o6);
        *(LoadT*)v7 = *(LoadT*)(head_ptr + o7);

        T_ACC inv_rms = 1.0;
        T_ACC sum_sq = 0.0;
        if constexpr (HAS_NORM) {
            // 复用现有 4vec helper，两次相加
            sum_sq += apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                v0, v1, v2, v3, res_head_ptr, o0, o1, o2, o3
            );
            sum_sq += apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                v4, v5, v6, v7, res_head_ptr, o4, o5, o6, o7
            );
            sum_sq = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq);
            inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        }

        if constexpr (IS_NEOX) {
            scalar_t r_cos0[VEC_SIZE], r_sin0[VEC_SIZE];
            scalar_t r_cos1[VEC_SIZE], r_sin1[VEC_SIZE];

            // 前 64 维 cos/sin 给 v0<->v2
            *(LoadT*)r_cos0 = *(LoadT*)(my_s_cos_sin + o0);
            *(LoadT*)r_sin0 = *(LoadT*)(my_s_cos_sin + HALF_ROT + o0);

            // 后 64 维 cos/sin 给 v1<->v3，对应 cos[32:64], sin[32:64]
            *(LoadT*)r_cos1 = *(LoadT*)(my_s_cos_sin + o1);
            *(LoadT*)r_sin1 = *(LoadT*)(my_s_cos_sin + HALF_ROT + o1);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                T_ACC s0 = static_cast<T_ACC>(v0[i]);
                T_ACC s1 = static_cast<T_ACC>(v1[i]);
                T_ACC s2 = static_cast<T_ACC>(v2[i]);
                T_ACC s3 = static_cast<T_ACC>(v3[i]);
                T_ACC s4 = static_cast<T_ACC>(v4[i]);
                T_ACC s5 = static_cast<T_ACC>(v5[i]);
                T_ACC s6 = static_cast<T_ACC>(v6[i]);
                T_ACC s7 = static_cast<T_ACC>(v7[i]);

                if constexpr (HAS_NORM) {
                    s0 = static_cast<T_ACC>(v0[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s1 = static_cast<T_ACC>(v1[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s2 = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1);
                    s3 = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1);
                    s4 = static_cast<T_ACC>(v4[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o4 + i] + 1);
                    s5 = static_cast<T_ACC>(v5[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o5 + i] + 1);
                    s6 = static_cast<T_ACC>(v6[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o6 + i] + 1);
                    s7 = static_cast<T_ACC>(v7[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o7 + i] + 1);
                }

                // 只旋转前 128 维：
                // v0 (0:32)   <-> v2 (64:96)
                // v1 (32:64)  <-> v3 (96:128)
                v0[i] = static_cast<scalar_t>(s0 * r_cos0[i] - s2 * r_sin0[i]);
                v2[i] = static_cast<scalar_t>(s0 * r_sin0[i] + s2 * r_cos0[i]);

                v1[i] = static_cast<scalar_t>(s1 * r_cos1[i] - s3 * r_sin1[i]);
                v3[i] = static_cast<scalar_t>(s1 * r_sin1[i] + s3 * r_cos1[i]);

                // 后 128 维保持不变（只做 norm/residual）
                v4[i] = static_cast<scalar_t>(s4);
                v5[i] = static_cast<scalar_t>(s5);
                v6[i] = static_cast<scalar_t>(s6);
                v7[i] = static_cast<scalar_t>(s7);
            }
        } else {
            // 非 NEOX：前 128 维按偶奇位旋转，后 128 维保持不变
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                int idx_c0 = (o0 + i) / 2;
                int idx_c1 = (o1 + i) / 2;
                int idx_c2 = (o2 + i) / 2;
                int idx_c3 = (o3 + i) / 2;

                scalar_t cos0 = my_s_cos_sin[idx_c0];
                scalar_t sin0 = my_s_cos_sin[HALF_ROT + idx_c0];
                scalar_t cos1 = my_s_cos_sin[idx_c1];
                scalar_t sin1 = my_s_cos_sin[HALF_ROT + idx_c1];
                scalar_t cos2 = my_s_cos_sin[idx_c2];
                scalar_t sin2 = my_s_cos_sin[HALF_ROT + idx_c2];
                scalar_t cos3 = my_s_cos_sin[idx_c3];
                scalar_t sin3 = my_s_cos_sin[HALF_ROT + idx_c3];

                T_ACC s0_0 = static_cast<T_ACC>(v0[i]);
                T_ACC s0_1 = static_cast<T_ACC>(v0[i + 1]);
                T_ACC s1_0 = static_cast<T_ACC>(v1[i]);
                T_ACC s1_1 = static_cast<T_ACC>(v1[i + 1]);
                T_ACC s2_0 = static_cast<T_ACC>(v2[i]);
                T_ACC s2_1 = static_cast<T_ACC>(v2[i + 1]);
                T_ACC s3_0 = static_cast<T_ACC>(v3[i]);
                T_ACC s3_1 = static_cast<T_ACC>(v3[i + 1]);

                if constexpr (HAS_NORM) {
                    s0_0 = static_cast<T_ACC>(v0[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s0_1 = static_cast<T_ACC>(v0[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i + 1] + 1);
                    s1_0 = static_cast<T_ACC>(v1[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s1_1 = static_cast<T_ACC>(v1[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i + 1] + 1);
                    s2_0 = static_cast<T_ACC>(v2[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1);
                    s2_1 = static_cast<T_ACC>(v2[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i + 1] + 1);
                    s3_0 = static_cast<T_ACC>(v3[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1);
                    s3_1 = static_cast<T_ACC>(v3[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i + 1] + 1);
                }

                v0[i]     = static_cast<scalar_t>(s0_0 * cos0 - s0_1 * sin0);
                v0[i + 1] = static_cast<scalar_t>(s0_1 * cos0 + s0_0 * sin0);

                v1[i]     = static_cast<scalar_t>(s1_0 * cos1 - s1_1 * sin1);
                v1[i + 1] = static_cast<scalar_t>(s1_1 * cos1 + s1_0 * sin1);

                v2[i]     = static_cast<scalar_t>(s2_0 * cos2 - s2_1 * sin2);
                v2[i + 1] = static_cast<scalar_t>(s2_1 * cos2 + s2_0 * sin2);

                v3[i]     = static_cast<scalar_t>(s3_0 * cos3 - s3_1 * sin3);
                v3[i + 1] = static_cast<scalar_t>(s3_1 * cos3 + s3_0 * sin3);
            }

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    v4[i] = static_cast<scalar_t>(static_cast<T_ACC>(v4[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o4 + i] + 1));
                    v5[i] = static_cast<scalar_t>(static_cast<T_ACC>(v5[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o5 + i] + 1));
                    v6[i] = static_cast<scalar_t>(static_cast<T_ACC>(v6[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o6 + i] + 1));
                    v7[i] = static_cast<scalar_t>(static_cast<T_ACC>(v7[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o7 + i] + 1));
                } else {
                    v4[i] = static_cast<scalar_t>(static_cast<T_ACC>(v4[i]));
                    v5[i] = static_cast<scalar_t>(static_cast<T_ACC>(v5[i]));
                    v6[i] = static_cast<scalar_t>(static_cast<T_ACC>(v6[i]));
                    v7[i] = static_cast<scalar_t>(static_cast<T_ACC>(v7[i]));
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
            f4[i] = static_cast<float>(v4[i]);
            f5[i] = static_cast<float>(v5[i]);
            f6[i] = static_cast<float>(v6[i]);
            f7[i] = static_cast<float>(v7[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f7, u8_7);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f7, u8_7);
        }

        *(LoadT8*)(rope_out_ptr + o0) = *(LoadT8*)u8_0;
        *(LoadT8*)(rope_out_ptr + o1) = *(LoadT8*)u8_1;
        *(LoadT8*)(rope_out_ptr + o2) = *(LoadT8*)u8_2;
        *(LoadT8*)(rope_out_ptr + o3) = *(LoadT8*)u8_3;
        *(LoadT8*)(rope_out_ptr + o4) = *(LoadT8*)u8_4;
        *(LoadT8*)(rope_out_ptr + o5) = *(LoadT8*)u8_5;
        *(LoadT8*)(rope_out_ptr + o6) = *(LoadT8*)u8_6;
        *(LoadT8*)(rope_out_ptr + o7) = *(LoadT8*)u8_7;

        if (!is_query) {
            const int k_buffer_offset =
                loc_id * k_buffer_stride_n + head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;

            *(LoadT8*)(k_buffer + k_buffer_offset + o0) = *(LoadT8*)u8_0;
            *(LoadT8*)(k_buffer + k_buffer_offset + o1) = *(LoadT8*)u8_1;
            *(LoadT8*)(k_buffer + k_buffer_offset + o2) = *(LoadT8*)u8_2;
            *(LoadT8*)(k_buffer + k_buffer_offset + o3) = *(LoadT8*)u8_3;
            *(LoadT8*)(k_buffer + k_buffer_offset + o4) = *(LoadT8*)u8_4;
            *(LoadT8*)(k_buffer + k_buffer_offset + o5) = *(LoadT8*)u8_5;
            *(LoadT8*)(k_buffer + k_buffer_offset + o6) = *(LoadT8*)u8_6;
            *(LoadT8*)(k_buffer + k_buffer_offset + o7) = *(LoadT8*)u8_7;
        }
    }

    if (lane >= total_threads_need_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_need_kv;
        const int head_idx = (v_lane_abs / THEAD_PER_HEAD);
        const int lane_in_head = (v_lane_abs % THEAD_PER_HEAD);

        uint8_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_stride + head_idx * head_stride_k;
        uint8_t* v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + global_token_idx * v_stride + head_idx * head_stride_k;

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        int o0 = lane_in_head * VEC_SIZE;
        int o1 = o0 + 32;
        int o2 = o0 + 64;
        int o3 = o0 + 96;
        int o4 = o0 + 128;
        int o5 = o0 + 160;
        int o6 = o0 + 192;
        int o7 = o0 + 224;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        scalar_t v4[VEC_SIZE], v5[VEC_SIZE], v6[VEC_SIZE], v7[VEC_SIZE];
        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        float f4[VEC_SIZE], f5[VEC_SIZE], f6[VEC_SIZE], f7[VEC_SIZE];
        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
        uint8_t u8_4[VEC_SIZE], u8_5[VEC_SIZE], u8_6[VEC_SIZE], u8_7[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(v_head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(v_head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(v_head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(v_head_ptr + o3);
        *(LoadT*)v4 = *(LoadT*)(v_head_ptr + o4);
        *(LoadT*)v5 = *(LoadT*)(v_head_ptr + o5);
        *(LoadT*)v6 = *(LoadT*)(v_head_ptr + o6);
        *(LoadT*)v7 = *(LoadT*)(v_head_ptr + o7);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
            f4[i] = static_cast<float>(v4[i]);
            f5[i] = static_cast<float>(v5[i]);
            f6[i] = static_cast<float>(v6[i]);
            f7[i] = static_cast<float>(v7[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f7, u8_7);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f7, u8_7);
        }

        *(LoadTK8*)(v_rope_out_ptr + o0) = *(LoadTK8*)u8_0;
        *(LoadTK8*)(v_rope_out_ptr + o1) = *(LoadTK8*)u8_1;
        *(LoadTK8*)(v_rope_out_ptr + o2) = *(LoadTK8*)u8_2;
        *(LoadTK8*)(v_rope_out_ptr + o3) = *(LoadTK8*)u8_3;
        *(LoadTK8*)(v_rope_out_ptr + o4) = *(LoadTK8*)u8_4;
        *(LoadTK8*)(v_rope_out_ptr + o5) = *(LoadTK8*)u8_5;
        *(LoadTK8*)(v_rope_out_ptr + o6) = *(LoadTK8*)u8_6;
        *(LoadTK8*)(v_rope_out_ptr + o7) = *(LoadTK8*)u8_7;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            v_buffer_ptr[o0 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_0[i];
            v_buffer_ptr[o1 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_1[i];
            v_buffer_ptr[o2 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_2[i];
            v_buffer_ptr[o3 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_3[i];
            v_buffer_ptr[o4 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_4[i];
            v_buffer_ptr[o5 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_5[i];
            v_buffer_ptr[o6 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_6[i];
            v_buffer_ptr[o7 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_7[i];
        }
    }
}


template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, int THEAD_PER_HEAD, bool HAS_NORM>
__global__ void opt_gammarms_rope_qwen3_headDim256_rotDim64_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ v,
    const scalar_t* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ k_buffer,
    uint8_t* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    uint8_t* __restrict__ q_rope_out,
    uint8_t* __restrict__ k_rope_out,
    uint8_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,                  // 固定应为 64
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t v_stride,
    const int64_t head_stride_q,
    const int64_t head_stride_k,
    const int q_rope_out_stride_n,
    const int q_rope_out_stride_h,
    const int q_rope_out_stride_p,
    const int k_buffer_stride_n,
    const int k_buffer_stride_h,
    const int k_buffer_stride_p,
    const int v_buffer_stride_n,
    const int v_buffer_stride_h,
    const int v_buffer_stride_p,
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k,
    scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k,
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token,
    const int tokens_per_block,
    const int cache_fp8_type,
    const int mrope_section_t,
    const int mrope_section_h,
    const int mrope_section_w,
    const bool is_interleaved)
{
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 256;
    constexpr int HALF_ROT = 32;   // rot_dim / 2

    const int tid = threadIdx.x;
    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    if (local_token_idx >= tokens_per_block) return;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    if (global_token_idx >= num_tokens) return;

    // shared memory layout for rot_dim=64:
    // [0, 32)   -> cos_row
    // [32, 64)  -> sin_row
    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * rot_dim;

    // positions: [3, num_tokens]
    const int64_t pos_t = positions[0 * num_tokens + global_token_idx];
    const int64_t pos_h = positions[1 * num_tokens + global_token_idx];
    const int64_t pos_w = positions[2 * num_tokens + global_token_idx];

    // 构造 mRoPE cos/sin row
    for (int i = lane; i < HALF_ROT; i += threads_per_token) {
        bool use_t = false;
        bool use_h = false;
        bool use_w = false;

        if (is_interleaved) {
            use_h = ((i % 3) == 1) && (i <= 3 * mrope_section_h);
            use_w = ((i % 3) == 2) && (i <= 3 * mrope_section_w);
            use_t = !(use_h || use_w);
        } else {
            const int t_end = mrope_section_t;
            const int h_end = t_end + mrope_section_h;
            use_t = (i < t_end);
            use_h = (i >= t_end) && (i < h_end);
            use_w = (i >= h_end) && (i < HALF_ROT);
        }

        scalar_t c = static_cast<scalar_t>(0);
        scalar_t s = static_cast<scalar_t>(0);

        if (use_t) {
            c += cos_sin_cache[pos_t * rot_dim + i];
            s += cos_sin_cache[pos_t * rot_dim + HALF_ROT + i];
        }
        if (use_h) {
            c += cos_sin_cache[pos_h * rot_dim + i];
            s += cos_sin_cache[pos_h * rot_dim + HALF_ROT + i];
        }
        if (use_w) {
            c += cos_sin_cache[pos_w * rot_dim + i];
            s += cos_sin_cache[pos_w * rot_dim + HALF_ROT + i];
        }

        my_s_cos_sin[i] = c;
        my_s_cos_sin[HALF_ROT + i] = s;
    }

    __syncthreads();

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    const int total_threads_need_kv = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads = total_threads_need_kv + (num_kv_heads * THEAD_PER_HEAD);

    const int64_t cache_loc = kv_cache_loc[global_token_idx];
    const int loc_id = cache_loc / page_size;
    const int loc_offset = cache_loc % page_size;

    if (lane < total_threads_need_kv) {
        const bool is_query = lane < q_boundary;
        const int head_idx = is_query ? (lane / THEAD_PER_HEAD) : ((lane - q_boundary) / THEAD_PER_HEAD);
        const int lane_in_head = is_query ? (lane % THEAD_PER_HEAD) : ((lane - q_boundary) % THEAD_PER_HEAD);

        uint8_t* rope_out_ptr = is_query
            ? (q_rope_out + global_token_idx * q_rope_out_stride_n + head_idx * q_rope_out_stride_h)
            : (k_rope_out + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* head_ptr = is_query
            ? (query + global_token_idx * query_stride + head_idx * head_stride_q)
            : (key + global_token_idx * key_stride + head_idx * head_stride_k);

        scalar_t* res_head_ptr = HAS_RESIDUAL
            ? (is_query
                ? (residual_q + global_token_idx * query_stride + head_idx * head_stride_q)
                : (residual_k + global_token_idx * key_stride + head_idx * head_stride_k))
            : nullptr;

        const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

        // 8 组向量，覆盖 256 维
        int o0 = lane_in_head * VEC_SIZE;   //   0..31
        int o1 = o0 + 32;                   //  32..63
        int o2 = o0 + 64;                   //  64..95
        int o3 = o0 + 96;                   //  96..127
        int o4 = o0 + 128;                  // 128..159
        int o5 = o0 + 160;                  // 160..191
        int o6 = o0 + 192;                  // 192..223
        int o7 = o0 + 224;                  // 224..255

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        scalar_t v4[VEC_SIZE], v5[VEC_SIZE], v6[VEC_SIZE], v7[VEC_SIZE];

        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        float f4[VEC_SIZE], f5[VEC_SIZE], f6[VEC_SIZE], f7[VEC_SIZE];

        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
        uint8_t u8_4[VEC_SIZE], u8_5[VEC_SIZE], u8_6[VEC_SIZE], u8_7[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(head_ptr + o3);
        *(LoadT*)v4 = *(LoadT*)(head_ptr + o4);
        *(LoadT*)v5 = *(LoadT*)(head_ptr + o5);
        *(LoadT*)v6 = *(LoadT*)(head_ptr + o6);
        *(LoadT*)v7 = *(LoadT*)(head_ptr + o7);

        T_ACC inv_rms = 1.0;
        T_ACC sum_sq = 0.0;
        if constexpr (HAS_NORM) {
            sum_sq += apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                v0, v1, v2, v3, res_head_ptr, o0, o1, o2, o3
            );
            sum_sq += apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
                v4, v5, v6, v7, res_head_ptr, o4, o5, o6, o7
            );
            sum_sq = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq);
            inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        }

        if constexpr (IS_NEOX) {
            scalar_t r_cos[VEC_SIZE], r_sin[VEC_SIZE];
            *(LoadT*)r_cos = *(LoadT*)(my_s_cos_sin + o0);              // cos[0:32]
            *(LoadT*)r_sin = *(LoadT*)(my_s_cos_sin + HALF_ROT + o0);   // sin[0:32]

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                T_ACC s0 = static_cast<T_ACC>(v0[i]);
                T_ACC s1 = static_cast<T_ACC>(v1[i]);
                T_ACC s2 = static_cast<T_ACC>(v2[i]);
                T_ACC s3 = static_cast<T_ACC>(v3[i]);
                T_ACC s4 = static_cast<T_ACC>(v4[i]);
                T_ACC s5 = static_cast<T_ACC>(v5[i]);
                T_ACC s6 = static_cast<T_ACC>(v6[i]);
                T_ACC s7 = static_cast<T_ACC>(v7[i]);

                if constexpr (HAS_NORM) {
                    s0 = static_cast<T_ACC>(v0[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s1 = static_cast<T_ACC>(v1[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s2 = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1);
                    s3 = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1);
                    s4 = static_cast<T_ACC>(v4[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o4 + i] + 1);
                    s5 = static_cast<T_ACC>(v5[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o5 + i] + 1);
                    s6 = static_cast<T_ACC>(v6[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o6 + i] + 1);
                    s7 = static_cast<T_ACC>(v7[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o7 + i] + 1);
                }

                // 只旋转前 64 维：v0 (0:32) <-> v1 (32:64)
                v0[i] = static_cast<scalar_t>(s0 * r_cos[i] - s1 * r_sin[i]);
                v1[i] = static_cast<scalar_t>(s0 * r_sin[i] + s1 * r_cos[i]);

                // 其余 64:256 保持不变（只做 norm/residual）
                v2[i] = static_cast<scalar_t>(s2);
                v3[i] = static_cast<scalar_t>(s3);
                v4[i] = static_cast<scalar_t>(s4);
                v5[i] = static_cast<scalar_t>(s5);
                v6[i] = static_cast<scalar_t>(s6);
                v7[i] = static_cast<scalar_t>(s7);
            }
        } else {
            // 非 NEOX：只对前 64 维做相邻偶奇位旋转，后面不动
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                int idx_c0 = (o0 + i) / 2;
                int idx_c1 = (o1 + i) / 2;

                scalar_t cos0 = my_s_cos_sin[idx_c0];
                scalar_t sin0 = my_s_cos_sin[HALF_ROT + idx_c0];
                scalar_t cos1 = my_s_cos_sin[idx_c1];
                scalar_t sin1 = my_s_cos_sin[HALF_ROT + idx_c1];

                T_ACC s0_0 = static_cast<T_ACC>(v0[i]);
                T_ACC s0_1 = static_cast<T_ACC>(v0[i + 1]);
                T_ACC s1_0 = static_cast<T_ACC>(v1[i]);
                T_ACC s1_1 = static_cast<T_ACC>(v1[i + 1]);

                if constexpr (HAS_NORM) {
                    s0_0 = static_cast<T_ACC>(v0[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i] + 1);
                    s0_1 = static_cast<T_ACC>(v0[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i + 1] + 1);
                    s1_0 = static_cast<T_ACC>(v1[i])     * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i] + 1);
                    s1_1 = static_cast<T_ACC>(v1[i + 1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i + 1] + 1);
                }

                v0[i]     = static_cast<scalar_t>(s0_0 * cos0 - s0_1 * sin0);
                v0[i + 1] = static_cast<scalar_t>(s0_1 * cos0 + s0_0 * sin0);

                v1[i]     = static_cast<scalar_t>(s1_0 * cos1 - s1_1 * sin1);
                v1[i + 1] = static_cast<scalar_t>(s1_1 * cos1 + s1_0 * sin1);
            }

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    v2[i] = static_cast<scalar_t>(static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i] + 1));
                    v3[i] = static_cast<scalar_t>(static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i] + 1));
                    v4[i] = static_cast<scalar_t>(static_cast<T_ACC>(v4[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o4 + i] + 1));
                    v5[i] = static_cast<scalar_t>(static_cast<T_ACC>(v5[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o5 + i] + 1));
                    v6[i] = static_cast<scalar_t>(static_cast<T_ACC>(v6[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o6 + i] + 1));
                    v7[i] = static_cast<scalar_t>(static_cast<T_ACC>(v7[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o7 + i] + 1));
                } else {
                    v2[i] = static_cast<scalar_t>(static_cast<T_ACC>(v2[i]));
                    v3[i] = static_cast<scalar_t>(static_cast<T_ACC>(v3[i]));
                    v4[i] = static_cast<scalar_t>(static_cast<T_ACC>(v4[i]));
                    v5[i] = static_cast<scalar_t>(static_cast<T_ACC>(v5[i]));
                    v6[i] = static_cast<scalar_t>(static_cast<T_ACC>(v6[i]));
                    v7[i] = static_cast<scalar_t>(static_cast<T_ACC>(v7[i]));
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
            f4[i] = static_cast<float>(v4[i]);
            f5[i] = static_cast<float>(v5[i]);
            f6[i] = static_cast<float>(v6[i]);
            f7[i] = static_cast<float>(v7[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f7, u8_7);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f7, u8_7);
        }

        *(LoadT8*)(rope_out_ptr + o0) = *(LoadT8*)u8_0;
        *(LoadT8*)(rope_out_ptr + o1) = *(LoadT8*)u8_1;
        *(LoadT8*)(rope_out_ptr + o2) = *(LoadT8*)u8_2;
        *(LoadT8*)(rope_out_ptr + o3) = *(LoadT8*)u8_3;
        *(LoadT8*)(rope_out_ptr + o4) = *(LoadT8*)u8_4;
        *(LoadT8*)(rope_out_ptr + o5) = *(LoadT8*)u8_5;
        *(LoadT8*)(rope_out_ptr + o6) = *(LoadT8*)u8_6;
        *(LoadT8*)(rope_out_ptr + o7) = *(LoadT8*)u8_7;

        if (!is_query) {
            const int k_buffer_offset =
                loc_id * k_buffer_stride_n + head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;

            *(LoadT8*)(k_buffer + k_buffer_offset + o0) = *(LoadT8*)u8_0;
            *(LoadT8*)(k_buffer + k_buffer_offset + o1) = *(LoadT8*)u8_1;
            *(LoadT8*)(k_buffer + k_buffer_offset + o2) = *(LoadT8*)u8_2;
            *(LoadT8*)(k_buffer + k_buffer_offset + o3) = *(LoadT8*)u8_3;
            *(LoadT8*)(k_buffer + k_buffer_offset + o4) = *(LoadT8*)u8_4;
            *(LoadT8*)(k_buffer + k_buffer_offset + o5) = *(LoadT8*)u8_5;
            *(LoadT8*)(k_buffer + k_buffer_offset + o6) = *(LoadT8*)u8_6;
            *(LoadT8*)(k_buffer + k_buffer_offset + o7) = *(LoadT8*)u8_7;
        }
    }

    if (lane >= total_threads_need_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_need_kv;
        const int head_idx = (v_lane_abs / THEAD_PER_HEAD);
        const int lane_in_head = (v_lane_abs % THEAD_PER_HEAD);

        uint8_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_stride + head_idx * head_stride_k;
        uint8_t* v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + global_token_idx * v_stride + head_idx * head_stride_k;

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

        int o0 = lane_in_head * VEC_SIZE;
        int o1 = o0 + 32;
        int o2 = o0 + 64;
        int o3 = o0 + 96;
        int o4 = o0 + 128;
        int o5 = o0 + 160;
        int o6 = o0 + 192;
        int o7 = o0 + 224;

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];
        scalar_t v4[VEC_SIZE], v5[VEC_SIZE], v6[VEC_SIZE], v7[VEC_SIZE];
        float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
        float f4[VEC_SIZE], f5[VEC_SIZE], f6[VEC_SIZE], f7[VEC_SIZE];
        uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
        uint8_t u8_4[VEC_SIZE], u8_5[VEC_SIZE], u8_6[VEC_SIZE], u8_7[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(v_head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(v_head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(v_head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(v_head_ptr + o3);
        *(LoadT*)v4 = *(LoadT*)(v_head_ptr + o4);
        *(LoadT*)v5 = *(LoadT*)(v_head_ptr + o5);
        *(LoadT*)v6 = *(LoadT*)(v_head_ptr + o6);
        *(LoadT*)v7 = *(LoadT*)(v_head_ptr + o7);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            f0[i] = static_cast<float>(v0[i]);
            f1[i] = static_cast<float>(v1[i]);
            f2[i] = static_cast<float>(v2[i]);
            f3[i] = static_cast<float>(v3[i]);
            f4[i] = static_cast<float>(v4[i]);
            f5[i] = static_cast<float>(v5[i]);
            f6[i] = static_cast<float>(v6[i]);
            f7[i] = static_cast<float>(v7[i]);
        }

        if (cache_fp8_type == 1) {
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 1>(f7, u8_7);
        } else {
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f0, u8_0);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f1, u8_1);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f2, u8_2);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f3, u8_3);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f4, u8_4);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f5, u8_5);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f6, u8_6);
            convert_vec_to_uint8_t<VEC_SIZE, 0>(f7, u8_7);
        }

        *(LoadTK8*)(v_rope_out_ptr + o0) = *(LoadTK8*)u8_0;
        *(LoadTK8*)(v_rope_out_ptr + o1) = *(LoadTK8*)u8_1;
        *(LoadTK8*)(v_rope_out_ptr + o2) = *(LoadTK8*)u8_2;
        *(LoadTK8*)(v_rope_out_ptr + o3) = *(LoadTK8*)u8_3;
        *(LoadTK8*)(v_rope_out_ptr + o4) = *(LoadTK8*)u8_4;
        *(LoadTK8*)(v_rope_out_ptr + o5) = *(LoadTK8*)u8_5;
        *(LoadTK8*)(v_rope_out_ptr + o6) = *(LoadTK8*)u8_6;
        *(LoadTK8*)(v_rope_out_ptr + o7) = *(LoadTK8*)u8_7;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            v_buffer_ptr[o0 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_0[i];
            v_buffer_ptr[o1 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_1[i];
            v_buffer_ptr[o2 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_2[i];
            v_buffer_ptr[o3 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_3[i];
            v_buffer_ptr[o4 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_4[i];
            v_buffer_ptr[o5 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_5[i];
            v_buffer_ptr[o6 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_6[i];
            v_buffer_ptr[o7 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_7[i];
        }
    }
}

template <typename T_ACC, typename scalar_t>
void launch_opt_gammarms_rope(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    scalar_t* v,
    const scalar_t* cos_sin_cache,
    uint8_t* k_buffer,
    uint8_t* v_buffer,
    int64_t* kv_cache_loc,
    uint8_t* q_rope_out,
    uint8_t* k_rope_out,
    uint8_t* v_rope_out,
    int page_size,
    int head_size,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t v_stride,
    int64_t head_stride_q,
    int64_t head_stride_k,
    int q_rope_out_stride_n,
    int q_rope_out_stride_h,
    int q_rope_out_stride_p,
    int k_buffer_stride_n,
    int k_buffer_stride_h,
    int k_buffer_stride_p,
    int v_buffer_stride_n,
    int v_buffer_stride_h,
    int v_buffer_stride_p,
    scalar_t* gamma_q,
    scalar_t* gamma_k,
    scalar_t* residual_q_ptr,
    scalar_t* residual_k_ptr,
    scalar_t eps,
    int num_tokens,
    bool is_neox,
    const int num_heads,
    const int num_kv_heads,
    int cache_fp8_type,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bool is_interleaved,
    cudaStream_t stream
) {
    bool has_residual = (residual_q_ptr != nullptr && residual_k_ptr != nullptr);
    bool has_norm = (gamma_q != nullptr && gamma_k != nullptr);

    constexpr int VEC = 8;
    constexpr int THREAD_PER_ROW_128 = 8;
    constexpr int THREAD_PER_ROW_256 = 4;

    DISPATCH_BOOL(has_norm, HAS_NORM_CONST, [&] {
        DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {

                // -------------------------------------------------
                // head_dim = 128, rot_dim = 128
                // -------------------------------------------------
                if (head_size == 128 && rot_dim == 128) {
                    int threads_per_token =
                        (num_heads + num_kv_heads + num_kv_heads) * THREAD_PER_ROW_128;
                    int target_block_size = 512;
                    int tokens_per_block = target_block_size / threads_per_token;
                    if (tokens_per_block < 1) tokens_per_block = 1;

                    int actual_block_size = tokens_per_block * threads_per_token;
                    int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
                    size_t smem_size = tokens_per_block * 128 * sizeof(scalar_t);

                    opt_gammarms_rope_qwen3_with_qkv<
                        T_ACC, scalar_t,
                        HAS_RESIDUAL_CONST, IS_NEOX_CONST,
                        VEC, THREAD_PER_ROW_128, HAS_NORM_CONST
                    ><<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, v,
                        cos_sin_cache,
                        k_buffer, v_buffer, kv_cache_loc,
                        q_rope_out, k_rope_out, v_rope_out,
                        page_size, rot_dim,
                        query_stride, key_stride, v_stride,
                        head_stride_q, head_stride_k,
                        q_rope_out_stride_n, q_rope_out_stride_h, q_rope_out_stride_p,
                        k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                        v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                        gamma_q, gamma_k,
                        residual_q_ptr, residual_k_ptr,
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block, cache_fp8_type,
                        mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved
                    );
                    return;
                }

                // -------------------------------------------------
                // head_dim = 128, rot_dim = 64
                // -------------------------------------------------
                if (head_size == 128 && rot_dim == 64) {
                    int threads_per_token =
                        (num_heads + num_kv_heads + num_kv_heads) * THREAD_PER_ROW_128;
                    int target_block_size = 512;
                    int tokens_per_block = target_block_size / threads_per_token;
                    if (tokens_per_block < 1) tokens_per_block = 1;

                    int actual_block_size = tokens_per_block * threads_per_token;
                    int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
                    size_t smem_size = tokens_per_block * 64 * sizeof(scalar_t);

                    opt_gammarms_rope_qwen3_rotDim64_with_qkv<
                        T_ACC, scalar_t,
                        HAS_RESIDUAL_CONST, IS_NEOX_CONST,
                        4, THREAD_PER_ROW_128, HAS_NORM_CONST
                    ><<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, v,
                        cos_sin_cache,
                        k_buffer, v_buffer, kv_cache_loc,
                        q_rope_out, k_rope_out, v_rope_out,
                        page_size, rot_dim,
                        query_stride, key_stride, v_stride,
                        head_stride_q, head_stride_k,
                        q_rope_out_stride_n, q_rope_out_stride_h, q_rope_out_stride_p,
                        k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                        v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                        gamma_q, gamma_k,
                        residual_q_ptr, residual_k_ptr,
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block, cache_fp8_type,
                        mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved
                    );
                    return;
                }

                // -------------------------------------------------
                // head_dim = 256, rot_dim = 128
                // -------------------------------------------------
                if (head_size == 256 && rot_dim == 128) {
                    int threads_per_token =
                        (num_heads + num_kv_heads + num_kv_heads) * THREAD_PER_ROW_256;
                    int target_block_size = 512;
                    int tokens_per_block = target_block_size / threads_per_token;
                    if (tokens_per_block < 1) tokens_per_block = 1;

                    int actual_block_size = tokens_per_block * threads_per_token;
                    int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
                    size_t smem_size = tokens_per_block * 128 * sizeof(scalar_t);

                    opt_gammarms_rope_qwen3_headDim256_with_qkv<
                        T_ACC, scalar_t,
                        HAS_RESIDUAL_CONST, IS_NEOX_CONST,
                        VEC, THREAD_PER_ROW_256, HAS_NORM_CONST
                    ><<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, v,
                        cos_sin_cache,
                        k_buffer, v_buffer, kv_cache_loc,
                        q_rope_out, k_rope_out, v_rope_out,
                        page_size, rot_dim,
                        query_stride, key_stride, v_stride,
                        head_stride_q, head_stride_k,
                        q_rope_out_stride_n, q_rope_out_stride_h, q_rope_out_stride_p,
                        k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                        v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                        gamma_q, gamma_k,
                        residual_q_ptr, residual_k_ptr,
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block, cache_fp8_type,
                        mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved
                    );
                    return;
                }

                // -------------------------------------------------
                // head_dim = 256, rot_dim = 64
                // -------------------------------------------------
                if (head_size == 256 && rot_dim == 64) {
                    int threads_per_token =
                        (num_heads + num_kv_heads + num_kv_heads) * THREAD_PER_ROW_256;
                    int target_block_size = 512;
                    int tokens_per_block = target_block_size / threads_per_token;
                    if (tokens_per_block < 1) tokens_per_block = 1;

                    int actual_block_size = tokens_per_block * threads_per_token;
                    int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
                    size_t smem_size = tokens_per_block * 64 * sizeof(scalar_t);

                    opt_gammarms_rope_qwen3_headDim256_rotDim64_with_qkv<
                        T_ACC, scalar_t,
                        HAS_RESIDUAL_CONST, IS_NEOX_CONST,
                        VEC, THREAD_PER_ROW_256, HAS_NORM_CONST
                    ><<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, v,
                        cos_sin_cache,
                        k_buffer, v_buffer, kv_cache_loc,
                        q_rope_out, k_rope_out, v_rope_out,
                        page_size, rot_dim,
                        query_stride, key_stride, v_stride,
                        head_stride_q, head_stride_k,
                        q_rope_out_stride_n, q_rope_out_stride_h, q_rope_out_stride_p,
                        k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                        v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                        gamma_q, gamma_k,
                        residual_q_ptr, residual_k_ptr,
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block, cache_fp8_type,
                        mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved
                    );
                    return;
                }

                return;
            });
        });
    });
}

void gammarms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions,
    Tensor& query,
    Tensor& key,
    Tensor& v,
    int64_t head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<Tensor>& weight_q,
    std::optional<Tensor>& weight_k,
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    double epsilon,
    bool use_fp8_e4m3,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bool is_interleaved) {

  const double eps_val = epsilon;
  int64_t num_tokens = query.size(0);
  int positions_ndim = positions.dim();

  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [3, num_tokens]");

  TORCH_CHECK(
      v.dim() == key.dim() && v.dim() == query.dim(),
      "v, key and query must have the same shape");

  TORCH_CHECK(
      v_buffer.size(-2) == head_size,
      "v must have the same head_size as query and key");

  if (positions_ndim == 1) {
    TORCH_CHECK(
        positions.size(0) == num_tokens,
        "positions [num_tokens] must match query.size(0)");
  } else {
    TORCH_CHECK(
        positions.size(0) == 3,
        "mRoPE positions must have shape [3, num_tokens]");
    TORCH_CHECK(
        positions.size(1) == num_tokens,
        "positions [3, num_tokens] must match query.size(0)");
  }

  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.numel() / num_tokens;

  TORCH_CHECK(query_hidden_size % head_size == 0, "query_hidden_size must be divisible by head_size");
  TORCH_CHECK(key_hidden_size % head_size == 0, "key_hidden_size must be divisible by head_size");

  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key_hidden_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

  int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim <= 512, "rot_dim must be <= 512");

  // -----------------------------
  // 正确推导 query/key/v 的 stride
  // 不再依赖 positions_ndim
  // -----------------------------
  int query_ndim = query.dim();
  int key_ndim = key.dim();
  int v_ndim = v.dim();

  TORCH_CHECK(
      query_ndim == 2 || query_ndim == 3,
      "query must be 2D or 3D");
  TORCH_CHECK(
      key_ndim == 2 || key_ndim == 3,
      "key must be 2D or 3D");
  TORCH_CHECK(
      v_ndim == 2 || v_ndim == 3,
      "v must be 2D or 3D");

  // token 维始终是第 0 维
  int64_t query_stride = query.stride(0);
  int64_t key_stride = key.stride(0);
  int64_t v_stride = v.stride(0);

  // 3D: [num_tokens, num_heads, head_dim]
  // 2D: [num_tokens, num_heads * head_dim]
  int64_t head_stride_q = (query_ndim == 3) ? query.stride(-2) : head_size;
  int64_t head_stride_k = (key_ndim == 3) ? key.stride(-2) : head_size;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int q_rope_out_stride_n = q_rope_out.stride(0);
  int q_rope_out_stride_h = q_rope_out.stride(1);
  int q_rope_out_stride_p = q_rope_out.stride(2);

  int k_buffer_stride_n = k_buffer.stride(0);
  int k_buffer_stride_h = k_buffer.stride(1);
  int k_buffer_stride_p = k_buffer.stride(2);

  int v_buffer_stride_n = v_buffer.stride(0);
  int v_buffer_stride_h = v_buffer.stride(1);
  int v_buffer_stride_p = v_buffer.stride(2);

  auto* pos_ptr = positions.data_ptr<int64_t>();
  (void)pos_ptr;

  bool qwen3 =
      (head_size == 128 && (num_heads + num_kv_heads) <= 128 &&
       (rot_dim == 128 || rot_dim == 64));
  bool qwen3_headdim256 =
      (head_size == 256 && (num_heads + num_kv_heads) <= 256 &&
       (rot_dim == 128 || rot_dim == 64));

  int cache_fp8_type = use_fp8_e4m3 ? 1 : 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fuse_rms_rotary_embedding",
      [&] {
        using T_ACC = at::acc_type<scalar_t, true>;

        if (qwen3 || qwen3_headdim256) {
          scalar_t* res_q_ptr =
              residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
          scalar_t* res_k_ptr =
              residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;
          scalar_t* wq_ptr =
              weight_q.has_value() ? weight_q->data_ptr<scalar_t>() : nullptr;
          scalar_t* wk_ptr =
              weight_k.has_value() ? weight_k->data_ptr<scalar_t>() : nullptr;

          launch_opt_gammarms_rope<T_ACC, scalar_t>(
              positions.data_ptr<int64_t>(),
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              v.data_ptr<scalar_t>(),
              cos_sin_cache.data_ptr<scalar_t>(),
              reinterpret_cast<uint8_t*>(k_buffer.data_ptr()),
              reinterpret_cast<uint8_t*>(v_buffer.data_ptr()),
              kv_cache_loc.data_ptr<int64_t>(),
              reinterpret_cast<uint8_t*>(q_rope_out.data_ptr()),
              reinterpret_cast<uint8_t*>(k_rope_out.data_ptr()),
              reinterpret_cast<uint8_t*>(v_rope_out.data_ptr()),
              page_size,
              static_cast<int>(head_size),   // 新增：传给 launch 做 128/256 分发
              rot_dim,
              query_stride,
              key_stride,
              v_stride,
              head_stride_q,
              head_stride_k,
              q_rope_out_stride_n,
              q_rope_out_stride_h,
              q_rope_out_stride_p,
              k_buffer_stride_n,
              k_buffer_stride_h,
              k_buffer_stride_p,
              v_buffer_stride_n,
              v_buffer_stride_h,
              v_buffer_stride_p,
              wq_ptr,
              wk_ptr,
              res_q_ptr,
              res_k_ptr,
              static_cast<scalar_t>(eps_val),
              static_cast<int>(num_tokens),
              is_neox,
              num_heads,
              num_kv_heads,
              cache_fp8_type,
              mrope_section_t,
              mrope_section_h,
              mrope_section_w,
              is_interleaved,
              stream);
          return;
        } else {
          printf("not supported and head_size must = 128/256, rot_dim must = 128 or 64\n");
          return;
        }
      });
}
}
}
