
#include <limits>
#include <algorithm>
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

template <bool IS_E4M3>
inline __device__ uint8_t float_to_fp8_cache_scalar(float value)
{
#if defined(__gfx938__)
    union {
        int32_t packed;
        uint8_t bytes[4];
    } result;
    result.packed = 0;
    if constexpr (IS_E4M3) {
        result.packed =
            __builtin_hcu_cvt_pk_fp8_f32(value, value, result.packed, false);
    } else {
        result.packed =
            __builtin_hcu_cvt_pk_bf8_f32(value, value, result.packed, false);
    }
    return result.bytes[0];
#else
    if constexpr (IS_E4M3) {
        return float_to_fp8e4m3(value);
    } else {
        return float_to_fp8e5m2(value);
    }
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

template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, 
         int THEAD_PER_HEAD, bool HAS_NORM, bool HAS_CACHE, bool USE_FP8_CACHE,
         typename CACHE_TYPE, int NUM_HEADS_CONST = 0,
         int NUM_KV_HEADS_CONST = 0>
__global__ void split_opt_rms_rope_qwen3_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,   
    scalar_t* __restrict__ v,      
    const scalar_t* __restrict__ cos_sin_cache,
    CACHE_TYPE* __restrict__ k_buffer,
    CACHE_TYPE* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    scalar_t* __restrict__ q_rope_out,
    scalar_t* __restrict__ k_rope_out,
    scalar_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,
    const int64_t qkv_stride,         
    const int64_t q_size,           
    const int64_t kv_size,
    const int64_t head_size,
    const int q_rope_out_stride_n,    
    const int k_rope_out_stride_n,
    const int v_rope_out_stride_n,
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
    bool use_fp8_e4m3
) 
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


    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * HEAD_SIZE;  //当前token的cos_sin_cache
    const int64_t pos = positions[global_token_idx];

    for (int i = lane; i < HEAD_SIZE; i += threads_per_token) {
        my_s_cos_sin[i] = cos_sin_cache[pos * HEAD_SIZE + i];
    }

    __syncthreads(); 

    const int q_boundary =
        (NUM_HEADS_CONST > 0 ? NUM_HEADS_CONST : num_heads) *
        THEAD_PER_HEAD;
    if(lane < q_boundary){
        const int q_head_idx = lane / THEAD_PER_HEAD;
        const int q_lane_in_head = lane % THEAD_PER_HEAD;
        ///////////////////////////////////////////////////////
        scalar_t* q_rope_out_ptr = q_rope_out + global_token_idx * q_rope_out_stride_n + q_head_idx * head_size;
        ///////////////////////////////////////////////////////
        scalar_t* q_head_ptr = query + global_token_idx * qkv_stride + q_head_idx * head_size;
        scalar_t* res_q_head_ptr = HAS_RESIDUAL ? (residual_q + global_token_idx * q_size + q_head_idx * head_size) : nullptr;
        
        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
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
        
            sum_sq = WarpReduceSum_Local<T_ACC,THEAD_PER_HEAD>(sum_sq);
            inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        }
        const scalar_t* cache_ptr = my_s_cos_sin; 
       
        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];
            *(LoadT*)r_cos_low = *(LoadT*)(cache_ptr + offset_low);
            *(LoadT*)r_sin_low = *(LoadT*)(cache_ptr + rot_dim/2 + offset_low); 
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    r_q_low[i] = static_cast<T_ACC>(r_q_low[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i]);
                    r_q_high[i] = static_cast<T_ACC>(r_q_high[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i]);
                }
                
                scalar_t q_l = r_q_low[i];
                scalar_t q_h = r_q_high[i];
                scalar_t c = r_cos_low[i];
                scalar_t s = r_sin_low[i];

                r_q_low[i]  = q_l * c - q_h * s;
                r_q_high[i] = q_l * s + q_h * c;
            }
        } else {
            using LoadCacheT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE / 2>;
            scalar_t c_low[VEC_SIZE / 2], s_low[VEC_SIZE / 2];
            scalar_t c_high[VEC_SIZE / 2], s_high[VEC_SIZE / 2];
            int cache_offset_low = offset_low / 2;
            int cache_offset_high = offset_high / 2;
            *(LoadCacheT*)c_low = *(LoadCacheT*)(cache_ptr + cache_offset_low);
            *(LoadCacheT*)s_low = *(LoadCacheT*)(cache_ptr + rot_dim/2 + cache_offset_low);
            *(LoadCacheT*)c_high = *(LoadCacheT*)(cache_ptr + cache_offset_high);
            *(LoadCacheT*)s_high = *(LoadCacheT*)(cache_ptr + rot_dim/2 + cache_offset_high);
            #pragma unroll
            for(int i=0; i<VEC_SIZE; i+=2) {
                int c_idx = i / 2; 
                if constexpr (HAS_NORM) {
                    r_q_low[i]   = static_cast<T_ACC>(r_q_low[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i]);
                    r_q_low[i+1] = static_cast<T_ACC>(r_q_low[i+1]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i + 1]);
                    r_q_high[i]   = static_cast<T_ACC>(r_q_high[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i]);
                    r_q_high[i+1] = static_cast<T_ACC>(r_q_high[i+1]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i + 1]);
                }
                scalar_t q0 = r_q_low[i]; scalar_t q1 = r_q_low[i+1];
                scalar_t c = c_low[c_idx]; scalar_t s = s_low[c_idx];
                r_q_low[i]   = q0 * c - q1 * s;
                r_q_low[i+1] = q1 * c + q0 * s;

                scalar_t qh0 = r_q_high[i]; scalar_t qh1 = r_q_high[i+1];
                scalar_t ch = c_high[c_idx]; scalar_t sh = s_high[c_idx];
                r_q_high[i]   = qh0 * ch - qh1 * sh;
                r_q_high[i+1] = qh1 * ch + qh0 * sh;
            }
        }

        *(LoadT*)(q_rope_out_ptr + offset_low) = *(LoadT*)r_q_low;
        *(LoadT*)(q_rope_out_ptr + offset_high) = *(LoadT*)r_q_high;
    }

   
    const int effective_num_heads =
        NUM_HEADS_CONST > 0 ? NUM_HEADS_CONST : num_heads;
    const int effective_num_kv_heads =
        NUM_KV_HEADS_CONST > 0 ? NUM_KV_HEADS_CONST : num_kv_heads;
    const int total_threads_needed_kv =
        (effective_num_heads + effective_num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads =
        total_threads_needed_kv +
        effective_num_kv_heads * THEAD_PER_HEAD;
    int loc_id = 0;
    int loc_offset = 0;
    if constexpr (HAS_CACHE) {
        const int64_t cache_loc = kv_cache_loc[global_token_idx];
        if (cache_loc < 0) return;
        loc_id = cache_loc / page_size;
        loc_offset = cache_loc % page_size;
    }

    if (lane >= q_boundary && lane < total_threads_needed_kv && key != nullptr) {
        const int k_lane_abs = lane - q_boundary;
        const int kv_head_idx = k_lane_abs / THEAD_PER_HEAD; 
        const int k_lane_in_head = k_lane_abs % THEAD_PER_HEAD;
    
        scalar_t* k_rope_out_ptr = k_rope_out + global_token_idx * k_rope_out_stride_n + kv_head_idx * head_size;
        CACHE_TYPE* k_buffer_ptr = nullptr;
        if constexpr (HAS_CACHE)
            k_buffer_ptr = k_buffer + loc_id * k_buffer_stride_n + kv_head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;

        scalar_t* k_head_ptr = key + (global_token_idx * qkv_stride + kv_head_idx * head_size + q_size);
        scalar_t* res_k_head_ptr = HAS_RESIDUAL ? (residual_k + global_token_idx * kv_size + kv_head_idx * head_size) : nullptr; 

        using LoadTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
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
            
            sum_sq_k = WarpReduceSum_Local<T_ACC,THEAD_PER_HEAD>(sum_sq_k);
            inv_rms_k = c10::cuda::compat::rsqrt(sum_sq_k / HEAD_SIZE + static_cast<T_ACC>(eps));
        }
        const scalar_t* cache_ptr_k = my_s_cos_sin; 
        
        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];

            *(LoadTK*)r_cos_low = *(LoadTK*)(cache_ptr_k + offset_low);
            *(LoadTK*)r_sin_low = *(LoadTK*)(cache_ptr_k + rot_dim/2 + offset_low);

            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    r_k_low[i] = static_cast<T_ACC>(r_k_low[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_low + i]);
                    r_k_high[i] = static_cast<T_ACC>(r_k_high[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_high + i]);
                }
                scalar_t k_l = r_k_low[i];
                scalar_t k_h = r_k_high[i];
                scalar_t c = r_cos_low[i];
                scalar_t s = r_sin_low[i];
                
                r_k_low[i] = k_l * c - k_h * s;
                r_k_high[i] = k_l * s + k_h * c;
            }
        } else {
            // Non-NEOX logic
            using LoadCacheTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE / 2>;
            scalar_t r_cos_low[VEC_SIZE / 2], r_sin_low[VEC_SIZE / 2];
            scalar_t r_cos_high[VEC_SIZE / 2], r_sin_high[VEC_SIZE / 2];
            int cache_offset_low = offset_low / 2;
            int cache_offset_high = offset_high / 2;
            *(LoadCacheTK*)r_cos_low = *(LoadCacheTK*)(cache_ptr_k + cache_offset_low);
            *(LoadCacheTK*)r_sin_low = *(LoadCacheTK*)(cache_ptr_k + rot_dim/2 + cache_offset_low);
            *(LoadCacheTK*)r_cos_high = *(LoadCacheTK*)(cache_ptr_k + cache_offset_high);
            *(LoadCacheTK*)r_sin_high = *(LoadCacheTK*)(cache_ptr_k + rot_dim/2 + cache_offset_high);
            #pragma unroll
            for(int i=0; i<VEC_SIZE; i+=2) {
                int c_idx = i / 2;
                if constexpr (HAS_NORM) {
                    r_k_low[i]   = static_cast<T_ACC>(r_k_low[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_low + i]);
                    r_k_low[i+1] = static_cast<T_ACC>(r_k_low[i+1]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_low + i+1]);
                    r_k_high[i]   = static_cast<T_ACC>(r_k_high[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_high + i]);
                    r_k_high[i+1] = static_cast<T_ACC>(r_k_high[i+1]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_high + i+1]);
                }
                scalar_t k0 = r_k_low[i]; scalar_t k1 = r_k_low[i+1];
                scalar_t c = r_cos_low[c_idx]; scalar_t s = r_sin_low[c_idx];
                r_k_low[i]   = k0 * c - k1 * s;
                r_k_low[i+1] = k1 * c + k0 * s;

                scalar_t kh0 = r_k_high[i]; scalar_t kh1 = r_k_high[i+1];
                scalar_t ch = r_cos_high[c_idx]; scalar_t sh = r_sin_high[c_idx];
                r_k_high[i]   = kh0 * ch - kh1 * sh;
                r_k_high[i+1] = kh1 * ch + kh0 * sh;
            }
        }

        *(LoadTK*)(k_rope_out_ptr + offset_low) = *(LoadTK*)r_k_low;
        *(LoadTK*)(k_rope_out_ptr + offset_high) = *(LoadTK*)r_k_high;

        if constexpr (HAS_CACHE) {
            if constexpr (USE_FP8_CACHE) {
                using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

                float f_k_low[VEC_SIZE];
                float f_k_high[VEC_SIZE];
                uint8_t u8_k_low[VEC_SIZE];
                uint8_t u8_k_high[VEC_SIZE];
                #pragma unroll
                for(int i=0; i<VEC_SIZE; ++i) {
                    f_k_low[i] = static_cast<float>(r_k_low[i]);
                    f_k_high[i] = static_cast<float>(r_k_high[i]);
                }
                if (use_fp8_e4m3) {
                    convert_vec_to_uint8_t<VEC_SIZE, 1>(f_k_low, u8_k_low);
                    convert_vec_to_uint8_t<VEC_SIZE, 1>(f_k_high, u8_k_high);
                } else {
                    convert_vec_to_uint8_t<VEC_SIZE, 0>(f_k_low, u8_k_low);
                    convert_vec_to_uint8_t<VEC_SIZE, 0>(f_k_high, u8_k_high);
                }
                *(LoadTK8*)(k_buffer_ptr + offset_low) = *(LoadTK8*)u8_k_low;
                *(LoadTK8*)(k_buffer_ptr + offset_high) = *(LoadTK8*)u8_k_high;
            } else {
                *(LoadTK*)(k_buffer_ptr + offset_low) = *(LoadTK*)r_k_low;
                *(LoadTK*)(k_buffer_ptr + offset_high) = *(LoadTK*)r_k_high;
            }
        
        }
    } 

    if (lane >= total_threads_needed_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_needed_kv;
        const int v_head_idx = v_lane_abs / THEAD_PER_HEAD;
        const int v_lane_in_head = v_lane_abs % THEAD_PER_HEAD;

        scalar_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_rope_out_stride_n + v_head_idx * head_size;
        CACHE_TYPE* v_buffer_ptr = nullptr;
        if constexpr (HAS_CACHE)
            v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + v_head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + (global_token_idx * qkv_stride + v_head_idx * head_size + q_size + kv_size);
        using LoadTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;

        scalar_t r_v_low[VEC_SIZE];
        scalar_t r_v_high[VEC_SIZE];

        int offset_low = v_lane_in_head * VEC_SIZE;
        int offset_high = HALF_ROT + v_lane_in_head * VEC_SIZE;

        *(LoadTK*)r_v_low = *(LoadTK*)(v_head_ptr + offset_low);
        *(LoadTK*)r_v_high = *(LoadTK*)(v_head_ptr + offset_high);

        *(LoadTK*)(v_rope_out_ptr + offset_low) = *(LoadTK*)r_v_low;
        *(LoadTK*)(v_rope_out_ptr + offset_high) = *(LoadTK*)r_v_high;
        if constexpr (HAS_CACHE) {
        if constexpr (USE_FP8_CACHE) {
            using LoadTK8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;

            float f_v_low[VEC_SIZE];
            float f_v_high[VEC_SIZE];
            uint8_t u8_v_low[VEC_SIZE];
            uint8_t u8_v_high[VEC_SIZE];
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                f_v_low[i] = static_cast<float>(r_v_low[i]);
                f_v_high[i] = static_cast<float>(r_v_high[i]);
            }
            if (use_fp8_e4m3) {
                convert_vec_to_uint8_t<VEC_SIZE, 1>(f_v_low, u8_v_low);
                convert_vec_to_uint8_t<VEC_SIZE, 1>(f_v_high, u8_v_high);
            } else {
                convert_vec_to_uint8_t<VEC_SIZE, 0>(f_v_low, u8_v_low);
                convert_vec_to_uint8_t<VEC_SIZE, 0>(f_v_high, u8_v_high);
            }
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                v_buffer_ptr[offset_low * v_buffer_stride_p + i * v_buffer_stride_p] = u8_v_low[i];
                v_buffer_ptr[offset_high * v_buffer_stride_p + i * v_buffer_stride_p] = u8_v_high[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                v_buffer_ptr[offset_low * v_buffer_stride_p + i * v_buffer_stride_p] = r_v_low[i];
                v_buffer_ptr[offset_high * v_buffer_stride_p + i * v_buffer_stride_p] = r_v_high[i];
            }
        }
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
template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, 
        int THEAD_PER_HEAD, bool HAS_NORM, bool HAS_CACHE, bool USE_FP8_CACHE, typename CACHE_TYPE>
__global__ void split_opt_rms_rope_qwen3_rotDim64_with_qkv(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,      
    scalar_t* __restrict__ v,      
    const scalar_t* __restrict__ cos_sin_cache,
    CACHE_TYPE* __restrict__ k_buffer,
    CACHE_TYPE* __restrict__ v_buffer,
    int64_t* __restrict__ kv_cache_loc,
    scalar_t* __restrict__ q_rope_out,
    scalar_t* __restrict__ k_rope_out,
    scalar_t* __restrict__ v_rope_out,
    int page_size,
    const int rot_dim,                 
    const int64_t qkv_stride,          
    const int64_t q_size,            
    const int64_t kv_size,            
    const int64_t head_size,  
    const int q_rope_out_stride_n,
    const int k_rope_out_stride_n,
    const int v_rope_out_stride_n,
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
    bool use_fp8_e4m3
) 
{
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 128;
    const int tid = threadIdx.x;
    
    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    if (local_token_idx >= tokens_per_block) return;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    if (global_token_idx >= num_tokens) return;

   
    scalar_t* my_s_cos_sin = s_cos_sin_base + local_token_idx * rot_dim;
    const int64_t pos = positions[global_token_idx];


    for (int i = lane; i < rot_dim; i += threads_per_token) {
        my_s_cos_sin[i] = cos_sin_cache[pos * rot_dim + i];
    }
    __syncthreads(); 

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    const int total_threads_need_kv = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    const int total_threads = total_threads_need_kv + (num_kv_heads * THEAD_PER_HEAD);
    int loc_id = 0;
    int loc_offset = 0;
    if constexpr (HAS_CACHE) {
        const int cache_loc = kv_cache_loc[global_token_idx];
        if (cache_loc < 0) return;
        loc_id = cache_loc / page_size;
        loc_offset = cache_loc % page_size;
    }
    if (lane < total_threads_need_kv) {
        const bool is_query = lane < q_boundary;
        const int head_idx = is_query ? (lane / THEAD_PER_HEAD) : ((lane - q_boundary) / THEAD_PER_HEAD);
        const int lane_in_head = is_query ? (lane % THEAD_PER_HEAD) : ((lane - q_boundary) % THEAD_PER_HEAD);
        
        scalar_t* rope_out_ptr = is_query ?
                (q_rope_out + global_token_idx * q_rope_out_stride_n + head_idx * head_size) :
                (k_rope_out + global_token_idx * k_rope_out_stride_n + head_idx * head_size);

        scalar_t* head_ptr = is_query ? 
            (query + global_token_idx * qkv_stride + head_idx * head_size) :
            (key + global_token_idx * qkv_stride + head_idx * head_size + q_size);
        
        scalar_t* res_head_ptr = HAS_RESIDUAL ? (is_query ? 
            (residual_q + global_token_idx * q_size + head_idx * head_size) :
            (residual_k + global_token_idx * kv_size + head_idx * head_size)) : nullptr;

        const scalar_t* gamma_ptr = is_query ? gamma_q : gamma_k;

        //隔32个load 4个，对齐rope
        int o0 = lane_in_head * VEC_SIZE; 
        int o1 = o0 + 32;                 
        int o2 = o0 + 64;                 
        int o3 = o0 + 96;                 

        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];

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
            *(LoadT*)r_sin = *(LoadT*)(my_s_cos_sin + 32 + o0);

            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                T_ACC s0 = static_cast<T_ACC>(v0[i]), s1 = static_cast<T_ACC>(v1[i]), s2 = static_cast<T_ACC>(v2[i]), s3 = static_cast<T_ACC>(v3[i]);
                if constexpr (HAS_NORM) {
                    s0 = static_cast<T_ACC>(v0[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i]);
                    s1 = static_cast<T_ACC>(v1[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i]);
                    s2 = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i]);
                    s3 = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i]);
                }
                v0[i] = s0 * r_cos[i] - s1 * r_sin[i];
                v1[i] = s0 * r_sin[i] + s1 * r_cos[i];
                v2[i] = s2;
                v3[i] = s3;
            }
        } else {
            #pragma unroll
            for(int i=0; i<VEC_SIZE; i+=2) {
            
                int idx_c0 = (o0 + i) / 2;
                int idx_c1 = (o0 + i + 1) / 2;
                
                scalar_t cos0 = my_s_cos_sin[idx_c0];
                scalar_t sin0 = my_s_cos_sin[32 + idx_c0];
                T_ACC s0_0 = static_cast<T_ACC>(v0[i]), s0_1 = static_cast<T_ACC>(v0[i+1]), s1_0 = static_cast<T_ACC>(v1[i]), s1_1 = static_cast<T_ACC>(v1[i+1]);
                if constexpr (HAS_NORM) {
                    s0_0 = static_cast<T_ACC>(v0[i])   * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i]);
                    s0_1 = static_cast<T_ACC>(v0[i+1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i + 1]);
                    s1_0 = static_cast<T_ACC>(v1[i])   * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i]);
                    s1_1 = static_cast<T_ACC>(v1[i+1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i + 1]);
                }
                v0[i]   = s0_0 * cos0 - s0_1 * sin0;
                v0[i+1] = s0_1 * cos0 + s0_0 * sin0;

                int idx_c_v1 = (o1 + i) / 2;
                scalar_t cos1 = my_s_cos_sin[idx_c_v1];
                scalar_t sin1 = my_s_cos_sin[32 + idx_c_v1];

                v1[i]   = s1_0 * cos1 - s1_1 * sin1;
                v1[i+1] = s1_1 * cos1 + s1_0 * sin1;
            }
            
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                if constexpr (HAS_NORM) {
                    v2[i] = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i]);
                    v3[i] = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i]);
                } else {
                    v2[i] = static_cast<T_ACC>(v2[i]);
                    v3[i] = static_cast<T_ACC>(v3[i]);
                }
            }
        }
        *(LoadT*)(rope_out_ptr + o0) = *(LoadT*)v0;
        *(LoadT*)(rope_out_ptr + o1) = *(LoadT*)v1;
        *(LoadT*)(rope_out_ptr + o2) = *(LoadT*)v2;
        *(LoadT*)(rope_out_ptr + o3) = *(LoadT*)v3;

        if constexpr (HAS_CACHE) {
            const int k_buffer_offset = loc_id * k_buffer_stride_n + head_idx * k_buffer_stride_h + loc_offset * k_buffer_stride_p;
            if (!is_query) {
                if constexpr (USE_FP8_CACHE) {
                    using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;
                    float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
                    uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
                    #pragma unroll
                    for(int i=0; i<VEC_SIZE; ++i) {
                        f0[i] = static_cast<float>(v0[i]);
                        f1[i] = static_cast<float>(v1[i]);
                        f2[i] = static_cast<float>(v2[i]);
                        f3[i] = static_cast<float>(v3[i]);
                    }
                    if (use_fp8_e4m3) {
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
                    *(LoadT8*)(k_buffer + k_buffer_offset + o0) = *(LoadT8*)u8_0;
                    *(LoadT8*)(k_buffer + k_buffer_offset + o1) = *(LoadT8*)u8_1;
                    *(LoadT8*)(k_buffer + k_buffer_offset + o2) = *(LoadT8*)u8_2;
                    *(LoadT8*)(k_buffer + k_buffer_offset + o3) = *(LoadT8*)u8_3;
                } else {
                    *(LoadT*)(k_buffer + k_buffer_offset + o0) = *(LoadT*)v0;
                    *(LoadT*)(k_buffer + k_buffer_offset + o1) = *(LoadT*)v1;
                    *(LoadT*)(k_buffer + k_buffer_offset + o2) = *(LoadT*)v2;
                    *(LoadT*)(k_buffer + k_buffer_offset + o3) = *(LoadT*)v3;
                }
            }
        }
    } 
    if (lane >= total_threads_need_kv && lane < total_threads && v != nullptr) {
        const int v_lane_abs = lane - total_threads_need_kv;
        const int head_idx =  (v_lane_abs / THEAD_PER_HEAD);
        const int lane_in_head = (v_lane_abs % THEAD_PER_HEAD);

        scalar_t* v_rope_out_ptr = v_rope_out + global_token_idx * v_rope_out_stride_n + head_idx * head_size;
        CACHE_TYPE* v_buffer_ptr = nullptr;
        if constexpr (HAS_CACHE)
            v_buffer_ptr = v_buffer + loc_id * v_buffer_stride_n + head_idx * v_buffer_stride_h + loc_offset;

        scalar_t* v_head_ptr = v + (global_token_idx * qkv_stride + head_idx * head_size + q_size + kv_size);
        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        
        int o0 = lane_in_head * VEC_SIZE; 
        int o1 = o0 + 32;                 
        int o2 = o0 + 64;                 
        int o3 = o0 + 96;      

        scalar_t v0[VEC_SIZE], v1[VEC_SIZE], v2[VEC_SIZE], v3[VEC_SIZE];

        *(LoadT*)v0 = *(LoadT*)(v_head_ptr + o0);
        *(LoadT*)v1 = *(LoadT*)(v_head_ptr + o1);
        *(LoadT*)v2 = *(LoadT*)(v_head_ptr + o2);
        *(LoadT*)v3 = *(LoadT*)(v_head_ptr + o3);

        *(LoadT*)(v_rope_out_ptr + o0) = *(LoadT*)v0;
        *(LoadT*)(v_rope_out_ptr + o1) = *(LoadT*)v1;
        *(LoadT*)(v_rope_out_ptr + o2) = *(LoadT*)v2;
        *(LoadT*)(v_rope_out_ptr + o3) = *(LoadT*)v3;
        if constexpr (HAS_CACHE) {
            if constexpr (USE_FP8_CACHE) {
                using LoadT8 = at::native::memory::aligned_vector<uint8_t, VEC_SIZE>;
                float f0[VEC_SIZE], f1[VEC_SIZE], f2[VEC_SIZE], f3[VEC_SIZE];
                uint8_t u8_0[VEC_SIZE], u8_1[VEC_SIZE], u8_2[VEC_SIZE], u8_3[VEC_SIZE];
                #pragma unroll
                for(int i=0; i<VEC_SIZE; ++i) {
                  f0[i] = static_cast<float>(v0[i]);
                  f1[i] = static_cast<float>(v1[i]);
                  f2[i] = static_cast<float>(v2[i]);
                  f3[i] = static_cast<float>(v3[i]);
                }
                if (use_fp8_e4m3) {
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
                #pragma unroll
                for(int i=0; i<VEC_SIZE; ++i) {
                  v_buffer_ptr[o0 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_0[i];
                  v_buffer_ptr[o1 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_1[i];
                  v_buffer_ptr[o2 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_2[i];
                  v_buffer_ptr[o3 * v_buffer_stride_p + i * v_buffer_stride_p] = u8_3[i];
                }
            } else {
                #pragma unroll
                for(int i=0; i<VEC_SIZE; ++i) {
                    v_buffer_ptr[o0 * v_buffer_stride_p + i * v_buffer_stride_p] = v0[i];
                    v_buffer_ptr[o1 * v_buffer_stride_p + i * v_buffer_stride_p] = v1[i];
                    v_buffer_ptr[o2 * v_buffer_stride_p + i * v_buffer_stride_p] = v2[i];
                    v_buffer_ptr[o3 * v_buffer_stride_p + i * v_buffer_stride_p] = v3[i];
                }
            }
        }
    }
}

template <typename scalar_t, typename cache_t, bool USE_E4M3>
__global__ void store_split_qkv_cache_hy3_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int64_t cache_capacity,
    int page_size,
    int64_t key_stride,
    int64_t value_stride,
    int64_t k_stride_n,
    int64_t k_stride_h,
    int64_t k_stride_p,
    int64_t v_stride_n,
    int64_t v_stride_h,
    int64_t v_stride_d) {
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_SIZE = 128;
    constexpr int PACK = 4;
    constexpr int THREADS_PER_TOKEN = 64;
    constexpr int TOKENS_PER_BLOCK = 4;
    constexpr int GROUPS_PER_THREAD =
        NUM_KV_HEADS * HEAD_SIZE / (THREADS_PER_TOKEN * PACK);

    const int local_token = threadIdx.x / THREADS_PER_TOKEN;
    const int lane = threadIdx.x % THREADS_PER_TOKEN;
    const int token = blockIdx.x * TOKENS_PER_BLOCK + local_token;

    __shared__ int64_t block_shared[TOKENS_PER_BLOCK];
    __shared__ int64_t block_offset_shared[TOKENS_PER_BLOCK];
    if (lane == 0) {
        const int64_t slot = token < num_tokens ? slot_mapping[token] : -1;
        if (slot >= 0 && slot < cache_capacity) {
            block_shared[local_token] = slot / page_size;
            block_offset_shared[local_token] = slot % page_size;
        } else {
            block_shared[local_token] = -1;
            block_offset_shared[local_token] = 0;
        }
    }
    __syncthreads();
    if (token >= num_tokens || block_shared[local_token] < 0) {
        return;
    }

    const int64_t block = block_shared[local_token];
    const int64_t block_offset = block_offset_shared[local_token];
    volatile uint8_t* key_store =
        reinterpret_cast<volatile uint8_t*>(key_cache);
    #pragma unroll
    for (int group = 0; group < GROUPS_PER_THREAD; ++group) {
        const int linear =
            lane * PACK + group * THREADS_PER_TOKEN * PACK;
        const int head = linear >> 7;
        const int head_offset = linear & (HEAD_SIZE - 1);
        const int64_t src_k =
            static_cast<int64_t>(token) * key_stride + linear;
        const int64_t src_v =
            static_cast<int64_t>(token) * value_stride + linear;
        const int64_t dst_k =
            block * k_stride_n + static_cast<int64_t>(head) * k_stride_h +
            block_offset * k_stride_p + head_offset;
        const int64_t dst_v =
            block * v_stride_n + static_cast<int64_t>(head) * v_stride_h +
            static_cast<int64_t>(head_offset) * v_stride_d + block_offset;

        using LoadT = at::native::memory::aligned_vector<scalar_t, PACK>;
        scalar_t key_vec[PACK];
        scalar_t value_vec[PACK];
        float key_float[PACK];
        float value_float[PACK];
        uint8_t key_fp8[PACK];
        uint8_t value_fp8[PACK];
        *(LoadT*)key_vec = *(LoadT*)(key + src_k);
        *(LoadT*)value_vec = *(LoadT*)(value + src_v);
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            key_float[i] = static_cast<float>(key_vec[i]);
            value_float[i] = static_cast<float>(value_vec[i]);
        }
        convert_vec_to_uint8_t<PACK, USE_E4M3 ? 1 : 0>(
            key_float, key_fp8);
        convert_vec_to_uint8_t<PACK, USE_E4M3 ? 1 : 0>(
            value_float, value_fp8);
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            key_store[dst_k + i] = key_fp8[i];
            value_cache[dst_v + static_cast<int64_t>(i) * v_stride_d] =
                value_fp8[i];
        }
    }
}

template <typename scalar_t, typename cache_t, bool USE_FP8_CACHE>
__global__ void store_split_qkv_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int64_t cache_capacity,
    int page_size,
    int num_heads,
    int head_size,
    int64_t key_stride,
    int64_t value_stride,
    int64_t k_stride_n,
    int64_t k_stride_h,
    int64_t k_stride_p,
    int64_t k_stride_d,
    int64_t v_stride_n,
    int64_t v_stride_h,
    int64_t v_stride_d,
    int64_t v_stride_p,
    bool use_e4m3) {
    constexpr int PACK = 4;
    constexpr int TOKENS_PER_BLOCK = 4;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int local_token = threadIdx.x / threads_per_token;
    if (local_token >= TOKENS_PER_BLOCK) {
        return;
    }

    const int token = blockIdx.x * TOKENS_PER_BLOCK + local_token;
    const int lane = threadIdx.x % threads_per_token;
    __shared__ int64_t slot_shared[TOKENS_PER_BLOCK];
    __shared__ int64_t block_shared[TOKENS_PER_BLOCK];
    __shared__ int64_t block_offset_shared[TOKENS_PER_BLOCK];
    if (lane == 0) {
        const int64_t slot = token < num_tokens ? slot_mapping[token] : -1;
        slot_shared[local_token] = slot;
        if (slot >= 0 && slot < cache_capacity) {
            block_shared[local_token] = slot / page_size;
            block_offset_shared[local_token] = slot % page_size;
        } else {
            block_shared[local_token] = -1;
            block_offset_shared[local_token] = 0;
        }
    }
    __syncthreads();
    if (token >= num_tokens || slot_shared[local_token] < 0) {
        return;
    }

    const int64_t block = block_shared[local_token];
    const int64_t block_offset = block_offset_shared[local_token];
    const int elements_per_token = num_heads * head_size;
    for (int linear = (threadIdx.x % threads_per_token) * PACK;
         linear < elements_per_token;
         linear += threads_per_token * PACK) {
        const int head = linear / head_size;
        const int head_offset = linear - head * head_size;
        const int64_t src_k =
            static_cast<int64_t>(token) * key_stride + linear;
        const int64_t src_v =
            static_cast<int64_t>(token) * value_stride + linear;
        const int64_t dst_k =
            block * k_stride_n + static_cast<int64_t>(head) * k_stride_h +
            block_offset * k_stride_p + head_offset * k_stride_d;
        const int64_t dst_v =
            block * v_stride_n + static_cast<int64_t>(head) * v_stride_h +
            head_offset * v_stride_d + block_offset * v_stride_p;

        if constexpr (USE_FP8_CACHE) {
            if (key_cache != nullptr) {
                uint32_t packed_k = 0;
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const float value =
                        static_cast<float>(key[src_k + i]);
                    const uint8_t converted = use_e4m3
                        ? float_to_fp8_cache_scalar<true>(value)
                        : float_to_fp8_cache_scalar<false>(value);
                    packed_k |=
                        static_cast<uint32_t>(converted) << (i * 8);
                }
                volatile uint8_t* key_store =
                    reinterpret_cast<volatile uint8_t*>(key_cache);
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    key_store[dst_k + i * k_stride_d] =
                        static_cast<uint8_t>(packed_k >> (i * 8));
                }
            }

            if (value_cache != nullptr) {
                uint32_t packed_v = 0;
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const float v_scalar =
                        static_cast<float>(value[src_v + i]);
                    const uint8_t converted = use_e4m3
                        ? float_to_fp8_cache_scalar<true>(v_scalar)
                        : float_to_fp8_cache_scalar<false>(v_scalar);
                    packed_v |=
                        static_cast<uint32_t>(converted) << (i * 8);
                }
                if (v_stride_d == 1) {
                    *reinterpret_cast<uint32_t*>(
                        reinterpret_cast<uint8_t*>(value_cache) + dst_v) =
                        packed_v;
                } else {
                    #pragma unroll
                    for (int i = 0; i < PACK; ++i) {
                        value_cache[dst_v + i * v_stride_d] =
                            static_cast<uint8_t>(packed_v >> (i * 8));
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                if (key_cache != nullptr) {
                    key_cache[dst_k + i * k_stride_d] = key[src_k + i];
                }
                if (value_cache != nullptr) {
                    value_cache[dst_v + i * v_stride_d] = value[src_v + i];
                }
            }
        }
    }
}

template <typename T_ACC, typename scalar_t, typename CACHE_TYPE>
void launch_opt_rms_rope(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    scalar_t* v,
    const scalar_t* cos_sin_cache,
    CACHE_TYPE* k_buffer,
    CACHE_TYPE* v_buffer,
    int64_t* kv_cache_loc,
    scalar_t* q_rope_out,
    scalar_t* k_rope_out,
    scalar_t* v_rope_out,
    int page_size,
    int rot_dim,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    int64_t qkv_stride,
    int q_rope_out_stride_n,
    int k_rope_out_stride_n,
    int v_rope_out_stride_n,
    int64_t k_buffer_stride_n,
    int64_t k_buffer_stride_h,
    int64_t k_buffer_stride_p,
    int64_t k_buffer_stride_d,
    int64_t v_buffer_stride_n,
    int64_t v_buffer_stride_h,
    int64_t v_buffer_stride_d,
    int64_t v_buffer_stride_p,
    scalar_t* gamma_q,
    scalar_t* gamma_k,
    scalar_t* residual_q_ptr, 
    scalar_t* residual_k_ptr,
    scalar_t eps,
    int num_tokens,
    int num_cache_tokens,
    int64_t cache_capacity,
    bool is_neox,
    const int num_heads,
    const int num_kv_heads,
    cudaStream_t stream,
    bool use_fp8_e4m3,
    bool use_fp8_cache
) {

        bool has_residual = (residual_q_ptr != nullptr && residual_k_ptr != nullptr);
        bool has_norm = (gamma_q != nullptr && gamma_k != nullptr);
        bool has_cache = kv_cache_loc != nullptr;

        const bool is_hy3_shape =
            rot_dim == 128 && head_size == 128 &&
            num_heads == 64 && num_kv_heads == 8;
        if (is_hy3_shape) {
            constexpr int HY3_THREAD_PER_ROW = 8;
            constexpr int HY3_VEC = 8;
            constexpr int HY3_THREADS_PER_TOKEN =
                (64 + 8 + 8) * HY3_THREAD_PER_ROW;
            constexpr int HY3_TOKENS_PER_BLOCK = 1;
            const int hy3_grid_size = num_tokens;
            const size_t hy3_smem_size = 128 * sizeof(scalar_t);
            DISPATCH_BOOL(use_fp8_cache, USE_FP8_CACHE_CONST, [&] {
                DISPATCH_BOOL(false, HAS_CACHE_CONST, [&] {
                    DISPATCH_BOOL(has_norm, HAS_NORM_CONST, [&] {
                        DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                                split_opt_rms_rope_qwen3_with_qkv<
                                    T_ACC, scalar_t, HAS_RESIDUAL_CONST,
                                    IS_NEOX_CONST, HY3_VEC,
                                    HY3_THREAD_PER_ROW, HAS_NORM_CONST,
                                    HAS_CACHE_CONST, USE_FP8_CACHE_CONST,
                                    CACHE_TYPE, 64, 8>
                                    <<<hy3_grid_size, HY3_THREADS_PER_TOKEN,
                                       hy3_smem_size, stream>>>(
                                        positions, query, key, v, cos_sin_cache,
                                        k_buffer, v_buffer, kv_cache_loc,
                                        q_rope_out, k_rope_out, v_rope_out,
                                        page_size, rot_dim, qkv_stride, q_size,
                                        kv_size, head_size,
                                        q_rope_out_stride_n,
                                        k_rope_out_stride_n,
                                        v_rope_out_stride_n,
                                        k_buffer_stride_n,
                                        k_buffer_stride_h,
                                        k_buffer_stride_p,
                                        v_buffer_stride_n,
                                        v_buffer_stride_h,
                                        v_buffer_stride_p,
                                        gamma_q, gamma_k,
                                        residual_q_ptr, residual_k_ptr,
                                        eps, num_tokens, num_heads, num_kv_heads,
                                        HY3_THREADS_PER_TOKEN,
                                        HY3_TOKENS_PER_BLOCK,
                                        use_fp8_e4m3);
                            });
                        });
                    });
                });
            });
        } else {
        constexpr int THREAD_PER_ROW = 8;
        constexpr int VEC = 8;
        int threads_per_token = (num_heads + num_kv_heads + num_kv_heads) * THREAD_PER_ROW;
        int target_block_size = 512; 
        int tokens_per_block = target_block_size / threads_per_token;

        if (tokens_per_block < 1) tokens_per_block = 1;
        
        int actual_block_size = tokens_per_block * threads_per_token;
        int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
        if (rot_dim == 128) {
            size_t smem_size = tokens_per_block * 128 * sizeof(scalar_t);
            DISPATCH_BOOL(use_fp8_cache, USE_FP8_CACHE_CONST, [&] {
                DISPATCH_BOOL(false, HAS_CACHE_CONST, [&] {
                    DISPATCH_BOOL(has_norm, HAS_NORM_CONST, [&] {
                        DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                                split_opt_rms_rope_qwen3_with_qkv<T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST,VEC,
                                        THREAD_PER_ROW, HAS_NORM_CONST, HAS_CACHE_CONST, USE_FP8_CACHE_CONST, CACHE_TYPE>
                                    <<<grid_size, actual_block_size, smem_size, stream>>>(
                                    positions, query, key, v, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc, q_rope_out, k_rope_out, v_rope_out,
                                    page_size, rot_dim,
                                    qkv_stride, q_size, kv_size, head_size,
                                    q_rope_out_stride_n, k_rope_out_stride_n, v_rope_out_stride_n,
                                    k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                                    v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                                    gamma_q, gamma_k,
                                    residual_q_ptr, residual_k_ptr,
                                    eps, num_tokens, num_heads, num_kv_heads,
                                    threads_per_token, tokens_per_block, use_fp8_e4m3
                                );
                            });
                        });
                    });
                });
            });
        } else if (rot_dim == 64) {
            size_t smem_size = tokens_per_block * 64 * sizeof(scalar_t);
            DISPATCH_BOOL(use_fp8_cache, USE_FP8_CACHE_CONST, [&] {
                DISPATCH_BOOL(false, HAS_CACHE_CONST, [&] {
                    DISPATCH_BOOL(has_norm, HAS_NORM_CONST, [&] {
                        DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                                split_opt_rms_rope_qwen3_rotDim64_with_qkv<T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST,4,
                                    THREAD_PER_ROW, HAS_NORM_CONST, HAS_CACHE_CONST, USE_FP8_CACHE_CONST, CACHE_TYPE>
                                    <<<grid_size, actual_block_size, smem_size, stream>>>(
                                    positions, query, key, v, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc, q_rope_out, k_rope_out, v_rope_out,
                                    page_size, rot_dim,
                                    qkv_stride, q_size, kv_size, head_size,
                                    q_rope_out_stride_n, k_rope_out_stride_n, v_rope_out_stride_n,
                                    k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p,
                                    v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p,
                                    gamma_q, gamma_k,
                                    residual_q_ptr, residual_k_ptr,
                                    eps, num_tokens, num_heads, num_kv_heads,
                                    threads_per_token, tokens_per_block, use_fp8_e4m3
                                );
                            });
                        });
                    });
                });
            });
        } else {
            return;
        }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (has_cache && num_cache_tokens > 0 &&
            (k_buffer != nullptr || v_buffer != nullptr)) {
            constexpr int STORE_VEC = 16;
            constexpr int STORE_TOKENS_PER_BLOCK = 4;
            const int store_threads =
                num_kv_heads * static_cast<int>(head_size) / STORE_VEC *
                STORE_TOKENS_PER_BLOCK;
            const int store_blocks =
                (num_cache_tokens + STORE_TOKENS_PER_BLOCK - 1) /
                STORE_TOKENS_PER_BLOCK;
            DISPATCH_BOOL(use_fp8_cache, USE_FP8_CACHE_CONST, [&] {
                auto launch_generic_store = [&] {
                    store_split_qkv_cache_kernel<
                        scalar_t, CACHE_TYPE, USE_FP8_CACHE_CONST>
                        <<<store_blocks, store_threads, 0, stream>>>(
                            k_rope_out,
                            v_rope_out,
                            k_buffer,
                            v_buffer,
                            kv_cache_loc,
                            num_cache_tokens,
                            cache_capacity,
                            page_size,
                            num_kv_heads,
                            static_cast<int>(head_size),
                            k_rope_out_stride_n,
                            v_rope_out_stride_n,
                            k_buffer_stride_n,
                            k_buffer_stride_h,
                            k_buffer_stride_p,
                            k_buffer_stride_d,
                            v_buffer_stride_n,
                            v_buffer_stride_h,
                            v_buffer_stride_d,
                            v_buffer_stride_p,
                            use_fp8_e4m3);
                };
                if constexpr (USE_FP8_CACHE_CONST) {
                    const bool use_hy3_store =
                        num_kv_heads == 8 && head_size == 128 &&
                        k_buffer != nullptr && v_buffer != nullptr &&
                        k_buffer_stride_d == 1 && v_buffer_stride_p == 1;
                    if (use_hy3_store) {
                        DISPATCH_BOOL(use_fp8_e4m3, USE_E4M3_CONST, [&] {
                            constexpr int HY3_STORE_THREADS = 64 * 4;
                            store_split_qkv_cache_hy3_kernel<
                                scalar_t, CACHE_TYPE, USE_E4M3_CONST>
                                <<<store_blocks, HY3_STORE_THREADS, 0, stream>>>(
                                    k_rope_out,
                                    v_rope_out,
                                    k_buffer,
                                    v_buffer,
                                    kv_cache_loc,
                                    num_cache_tokens,
                                    cache_capacity,
                                    page_size,
                                    k_rope_out_stride_n,
                                    v_rope_out_stride_n,
                                    k_buffer_stride_n,
                                    k_buffer_stride_h,
                                    k_buffer_stride_p,
                                    v_buffer_stride_n,
                                    v_buffer_stride_h,
                                    v_buffer_stride_d);
                        });
                    } else {
                        launch_generic_store();
                    }
                } else {
                    launch_generic_store();
                }
            });
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
}

void split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant(
    Tensor& positions, 
    Tensor& qkv,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<Tensor>& weight_q, 
    std::optional<Tensor>& weight_k, 
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    ///////////////////////////////////////////////////
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    /////////////////////////////////////////////////
    double epsilon,
    bool use_fp8_e4m3,
    bool use_fp8_cache
) {

  const double eps_val = epsilon;
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");
  TORCH_CHECK(qkv.dim() == 2 && qkv.size(0) == num_tokens,
              "qkv and positions must contain the same number of tokens");

  int num_heads = q_size / head_size;
  int num_kv_heads = kv_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int num_cache_tokens = 0;
  int64_t cache_capacity = 0;
  if (kv_cache_loc.has_value()) {
    TORCH_CHECK(kv_cache_loc->scalar_type() == at::ScalarType::Long &&
                    kv_cache_loc->is_contiguous(),
                "kv_cache_loc must be contiguous int64");
    TORCH_CHECK(kv_cache_loc->numel() <= num_tokens &&
                    kv_cache_loc->numel() <= std::numeric_limits<int>::max(),
                "kv_cache_loc is too large");
    TORCH_CHECK(k_buffer.dim() == 4 && v_buffer.dim() == 4,
                "KV cache buffers must be four-dimensional");
    TORCH_CHECK(k_buffer.size(0) == v_buffer.size(0) &&
                    k_buffer.size(1) == v_buffer.size(1) &&
                    k_buffer.size(2) == v_buffer.size(3) &&
                    k_buffer.size(3) == v_buffer.size(2),
                "expected K [blocks, heads, page, dim] and "
                "V [blocks, heads, dim, page]");
    TORCH_CHECK(k_buffer.size(1) == num_kv_heads &&
                    k_buffer.size(3) == head_size,
                "KV cache shape does not match kv_size/head_size");
    page_size = static_cast<int>(k_buffer.size(2));
    num_cache_tokens = static_cast<int>(kv_cache_loc->numel());
    cache_capacity = k_buffer.size(0) * static_cast<int64_t>(page_size);
  }

  int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim <= 512);
      
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool has_residual = residual_q.has_value() && residual_k.has_value();
  bool has_norm = weight_q.has_value() && weight_k.has_value();

  /////////////////////////////////////////////////////////
  int q_rope_out_stride_n = q_rope_out.stride(0);
  int k_rope_out_stride_n = k_rope_out.stride(0);
  int v_rope_out_stride_n = v_rope_out.stride(0);

  int64_t k_buffer_stride_n = 0;
  int64_t k_buffer_stride_h = 0;
  int64_t k_buffer_stride_p = 0;
  int64_t k_buffer_stride_d = 0;
  int64_t v_buffer_stride_n = 0;
  int64_t v_buffer_stride_h = 0;
  int64_t v_buffer_stride_d = 0;
  int64_t v_buffer_stride_p = 0;
  if (kv_cache_loc.has_value()) {
    k_buffer_stride_n = k_buffer.stride(0);
    k_buffer_stride_h = k_buffer.stride(1);
    k_buffer_stride_p = k_buffer.stride(2);
    k_buffer_stride_d = k_buffer.stride(3);
    v_buffer_stride_n = v_buffer.stride(0);
    v_buffer_stride_h = v_buffer.stride(1);
    v_buffer_stride_d = v_buffer.stride(2);
    v_buffer_stride_p = v_buffer.stride(3);
  }
  /////////////////////////////////////////////////////////
  auto* pos_ptr = positions.data_ptr<int64_t>(); 
  bool qwen3 =  (head_size == 128 && (num_heads + num_kv_heads) <= 128 && (rot_dim == 128 ||  rot_dim == 64));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      qkv.scalar_type(), "fuse_rms_rotary_embedding", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
         if (qwen3) {
                scalar_t* res_q_ptr = residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
                scalar_t* res_k_ptr = residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;
                scalar_t* wq_ptr = weight_q.has_value() ? weight_q->data_ptr<scalar_t>() : nullptr;
                scalar_t* wk_ptr = weight_k.has_value() ? weight_k->data_ptr<scalar_t>() : nullptr;
                int64_t* kv_cache_loc_ptr = kv_cache_loc.has_value() ? kv_cache_loc->data_ptr<int64_t>() : nullptr;
                uint8_t* k_buf_ptr = kv_cache_loc_ptr != nullptr ? reinterpret_cast<uint8_t*>(k_buffer.data_ptr()) : nullptr;
                uint8_t* v_buf_ptr = kv_cache_loc_ptr != nullptr ? reinterpret_cast<uint8_t*>(v_buffer.data_ptr()) : nullptr;
                if (use_fp8_cache) {
                    launch_opt_rms_rope<T_ACC, scalar_t, uint8_t>(
                        positions.data_ptr<int64_t>(),
                        qkv.data_ptr<scalar_t>(),
                        qkv.data_ptr<scalar_t>(),
                        qkv.data_ptr<scalar_t>(),
                        cos_sin_cache.data_ptr<scalar_t>(),
                        k_buf_ptr,
                        v_buf_ptr,
                        kv_cache_loc_ptr,
                        q_rope_out.data_ptr<scalar_t>(),
                        k_rope_out.data_ptr<scalar_t>(),
                        v_rope_out.data_ptr<scalar_t>(),
                        page_size,
                        rot_dim,
                        q_size,
                        kv_size,
                        head_size,
                        q_size+kv_size+kv_size,
                        q_rope_out_stride_n,
                        k_rope_out_stride_n,
                        v_rope_out_stride_n,
                        k_buffer_stride_n,
                        k_buffer_stride_h,
                        k_buffer_stride_p,
                        k_buffer_stride_d,
                        v_buffer_stride_n,
                        v_buffer_stride_h,
                        v_buffer_stride_d,
                        v_buffer_stride_p,
                        wq_ptr,
                        wk_ptr,
                        res_q_ptr,
                        res_k_ptr,
                        static_cast<scalar_t>(eps_val),
                        (int)num_tokens,
                        num_cache_tokens,
                        cache_capacity,
                        is_neox,
                        num_heads,
                        num_kv_heads,
                        stream,
                        use_fp8_e4m3,
                        use_fp8_cache
                    );
                } else {
                    launch_opt_rms_rope<T_ACC, scalar_t, scalar_t>(
                        positions.data_ptr<int64_t>(),
                        qkv.data_ptr<scalar_t>(),
                        qkv.data_ptr<scalar_t>(),
                        qkv.data_ptr<scalar_t>(),
                        cos_sin_cache.data_ptr<scalar_t>(),
                        k_buf_ptr != nullptr ? reinterpret_cast<scalar_t*>(k_buf_ptr) : nullptr,
                        v_buf_ptr != nullptr ? reinterpret_cast<scalar_t*>(v_buf_ptr) : nullptr,
                        kv_cache_loc_ptr,
                        q_rope_out.data_ptr<scalar_t>(),
                        k_rope_out.data_ptr<scalar_t>(),
                        v_rope_out.data_ptr<scalar_t>(),
                        page_size,
                        rot_dim,
                        q_size,
                        kv_size,
                        head_size,
                        q_size+kv_size+kv_size,
                        q_rope_out_stride_n,
                        k_rope_out_stride_n,
                        v_rope_out_stride_n,
                        k_buffer_stride_n,
                        k_buffer_stride_h,
                        k_buffer_stride_p,
                        k_buffer_stride_d,
                        v_buffer_stride_n,
                        v_buffer_stride_h,
                        v_buffer_stride_d,
                        v_buffer_stride_p,
                        wq_ptr,
                        wk_ptr,
                        res_q_ptr,
                        res_k_ptr,
                        static_cast<scalar_t>(eps_val),
                        (int)num_tokens,
                        num_cache_tokens,
                        cache_capacity,
                        is_neox,
                        num_heads,
                        num_kv_heads,
                        stream,
                        use_fp8_e4m3,
                        use_fp8_cache
                    );
                }
                return;
        } else {
          printf("not supported  and head_size must = 128, rot_dim must = 128 or 64\n");
          return;
        }
  });
}

template <typename scalar_t>
__device__ __forceinline__ void mimo_v2_apply_rope_pair_inplace(
    scalar_t* __restrict__ head_ptr,
    const scalar_t* __restrict__ cache_ptr,
    int pair_idx,
    int rot_dim,
    bool is_neox,
    int& idx0,
    int& idx1,
    float& out0,
    float& out1) {
  if (is_neox) {
    const int half = rot_dim / 2;
    idx0 = pair_idx;
    idx1 = pair_idx + half;
    const float x0 = static_cast<float>(head_ptr[idx0]);
    const float x1 = static_cast<float>(head_ptr[idx1]);
    const float c = static_cast<float>(cache_ptr[pair_idx]);
    const float s = static_cast<float>(cache_ptr[half + pair_idx]);
    out0 = x0 * c - x1 * s;
    out1 = x1 * c + x0 * s;
  } else {
    const int half = rot_dim / 2;
    idx0 = pair_idx * 2;
    idx1 = idx0 + 1;
    const float x0 = static_cast<float>(head_ptr[idx0]);
    const float x1 = static_cast<float>(head_ptr[idx1]);
    const float c = static_cast<float>(cache_ptr[pair_idx]);
    const float s = static_cast<float>(cache_ptr[half + pair_idx]);
    out0 = x0 * c - x1 * s;
    out1 = x1 * c + x0 * s;
  }
  head_ptr[idx0] = static_cast<scalar_t>(out0);
  head_ptr[idx1] = static_cast<scalar_t>(out1);
}

__device__ __forceinline__ uint8_t mimo_v2_float_to_fp8(
    float value,
    bool use_fp8_e4m3) {
  return use_fp8_e4m3 ? float_to_fp8e4m3(value) : float_to_fp8e5m2(value);
}

__device__ __forceinline__ uint32_t mimo_v2_float_to_fp8e4m3_x4(
    float a,
    float b,
    float c,
    float d) {
#if defined(__gfx938__)
  int32_t out = 0;
  out = __builtin_hcu_cvt_pk_fp8_f32(a, b, out, false);
  out = __builtin_hcu_cvt_pk_fp8_f32(c, d, out, true);
  return static_cast<uint32_t>(out);
#else
  uint32_t out = 0;
  reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
  reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
  reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
  reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
  return out;
#endif
}

__device__ __forceinline__ uint32_t mimo_v2_float_to_fp8e5m2_x4(
    float a,
    float b,
    float c,
    float d) {
#if defined(__gfx938__)
  int32_t out = 0;
  out = __builtin_hcu_cvt_pk_bf8_f32(a, b, out, false);
  out = __builtin_hcu_cvt_pk_bf8_f32(c, d, out, true);
  return static_cast<uint32_t>(out);
#else
  uint32_t out = 0;
  reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
  reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
  reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
  reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
  return out;
#endif
}

__device__ __forceinline__ uint32_t mimo_v2_float_to_fp8_x4(
    float a,
    float b,
    float c,
    float d,
    bool use_fp8_e4m3) {
  return use_fp8_e4m3
             ? mimo_v2_float_to_fp8e4m3_x4(a, b, c, d)
             : mimo_v2_float_to_fp8e5m2_x4(a, b, c, d);
}

template <int VEC>
__device__ __forceinline__ void mimo_v2_store_fp8_vec(
    uint8_t* __restrict__ ptr,
    int64_t stride,
    const float (&vals)[VEC],
    float inv_scale,
    bool use_fp8_e4m3) {
  static_assert(VEC % 4 == 0, "fp8 packed store expects multiples of 4");
  if (stride == 1) {
#pragma unroll
    for (int i = 0; i < VEC; i += 4) {
      reinterpret_cast<uint32_t*>(ptr + i)[0] = mimo_v2_float_to_fp8_x4(
          vals[i] * inv_scale,
          vals[i + 1] * inv_scale,
          vals[i + 2] * inv_scale,
          vals[i + 3] * inv_scale,
          use_fp8_e4m3);
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      ptr[i * stride] = mimo_v2_float_to_fp8(vals[i] * inv_scale, use_fp8_e4m3);
    }
  }
}

template <typename cache_t, typename scalar_t, int VEC>
__device__ __forceinline__ void mimo_v2_store_cache_vec(
    cache_t* __restrict__ ptr,
    int64_t stride,
    const scalar_t (&vals)[VEC]) {
  if (stride == 1) {
    using StoreT = at::native::memory::aligned_vector<cache_t, VEC>;
    cache_t out[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out[i] = static_cast<cache_t>(vals[i]);
    }
    *reinterpret_cast<StoreT*>(ptr) = *reinterpret_cast<StoreT*>(out);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      ptr[i * stride] = static_cast<cache_t>(vals[i]);
    }
  }
}

template <typename scalar_t, typename cache_t, typename index_t, bool USE_FP8_CACHE>
__global__ void mimo_v2_split_rope_vscale_kv_store_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ cos_sin_cache,
    cache_t* __restrict__ k_buffer,
    cache_t* __restrict__ v_buffer,
    const index_t* __restrict__ kv_cache_loc,
    int64_t num_tokens,
    int64_t q_size,
    int64_t k_size,
    int64_t qkv_stride,
    int64_t k_buffer_stride_n,
    int64_t k_buffer_stride_h,
    int64_t k_buffer_stride_d,
    int64_t v_buffer_stride_n,
    int64_t v_buffer_stride_h,
    int64_t v_buffer_stride_d,
    int head_size,
    int v_head_size,
    int rot_dim,
    int num_q_heads,
    int num_kv_heads,
    int q_threads_per_head,
    int k_threads_per_head,
    int k_tail_threads_per_head,
    int v_threads_per_head,
    int q_units_per_block,
    int k_units_per_block,
    int v_units_per_block,
    int64_t q_block_count,
    int64_t k_block_count,
    float value_scale,
    float k_cache_scale,
    float v_cache_scale,
    bool has_k_cache_scale,
    bool has_v_cache_scale,
    bool is_neox,
    bool has_cache,
    bool use_fp8_e4m3) {
  constexpr int VEC = 16 / sizeof(scalar_t);
  using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;

  const int tid = threadIdx.x;
  const int half_rot = rot_dim / 2;
  int64_t segment_block_idx = static_cast<int64_t>(blockIdx.x);

  if (segment_block_idx < q_block_count) {
    const int unit_lane = tid / q_threads_per_head;
    if (unit_lane >= q_units_per_block) {
      return;
    }
    const int lane_in_head = tid - unit_lane * q_threads_per_head;
    const int64_t unit_idx =
        segment_block_idx * q_units_per_block + unit_lane;
    const int64_t total_units = num_tokens * static_cast<int64_t>(num_q_heads);
    if (unit_idx >= total_units) {
      return;
    }
    const int64_t token_idx = unit_idx / num_q_heads;
    const int head_idx = static_cast<int>(unit_idx - token_idx * num_q_heads);
    const int64_t pos = positions[token_idx];
    const scalar_t* token_cos_sin = cos_sin_cache + pos * rot_dim;
    scalar_t* q_head_ptr =
        qkv + token_idx * qkv_stride + head_idx * head_size;

    if (is_neox) {
      const int offset_low = lane_in_head * VEC;
      const int offset_high = half_rot + offset_low;
      scalar_t q_low[VEC], q_high[VEC], cos_v[VEC], sin_v[VEC];
      *reinterpret_cast<LoadT*>(q_low) =
          *reinterpret_cast<LoadT*>(q_head_ptr + offset_low);
      *reinterpret_cast<LoadT*>(q_high) =
          *reinterpret_cast<LoadT*>(q_head_ptr + offset_high);
      *reinterpret_cast<LoadT*>(cos_v) =
          *reinterpret_cast<const LoadT*>(token_cos_sin + offset_low);
      *reinterpret_cast<LoadT*>(sin_v) =
          *reinterpret_cast<const LoadT*>(token_cos_sin + half_rot + offset_low);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float x0 = static_cast<float>(q_low[i]);
        const float x1 = static_cast<float>(q_high[i]);
        const float c = static_cast<float>(cos_v[i]);
        const float s = static_cast<float>(sin_v[i]);
        q_low[i] = static_cast<scalar_t>(x0 * c - x1 * s);
        q_high[i] = static_cast<scalar_t>(x1 * c + x0 * s);
      }
      *reinterpret_cast<LoadT*>(q_head_ptr + offset_low) =
          *reinterpret_cast<LoadT*>(q_low);
      *reinterpret_cast<LoadT*>(q_head_ptr + offset_high) =
          *reinterpret_cast<LoadT*>(q_high);
    } else {
      const int offset = lane_in_head * VEC;
      const int pair_base = offset / 2;
      using CacheLoadT = at::native::memory::aligned_vector<scalar_t, VEC / 2>;
      scalar_t q_vals[VEC], cos_v[VEC / 2], sin_v[VEC / 2];
      *reinterpret_cast<LoadT*>(q_vals) =
          *reinterpret_cast<LoadT*>(q_head_ptr + offset);
      *reinterpret_cast<CacheLoadT*>(cos_v) =
          *reinterpret_cast<const CacheLoadT*>(token_cos_sin + pair_base);
      *reinterpret_cast<CacheLoadT*>(sin_v) =
          *reinterpret_cast<const CacheLoadT*>(token_cos_sin + half_rot + pair_base);
#pragma unroll
      for (int i = 0; i < VEC; i += 2) {
        const int cache_idx = i / 2;
        const float x0 = static_cast<float>(q_vals[i]);
        const float x1 = static_cast<float>(q_vals[i + 1]);
        const float c = static_cast<float>(cos_v[cache_idx]);
        const float s = static_cast<float>(sin_v[cache_idx]);
        q_vals[i] = static_cast<scalar_t>(x0 * c - x1 * s);
        q_vals[i + 1] = static_cast<scalar_t>(x1 * c + x0 * s);
      }
      *reinterpret_cast<LoadT*>(q_head_ptr + offset) =
          *reinterpret_cast<LoadT*>(q_vals);
    }
    return;
  }

  segment_block_idx -= q_block_count;
  if (segment_block_idx < k_block_count) {
    const int unit_lane = tid / k_threads_per_head;
    if (unit_lane >= k_units_per_block) {
      return;
    }
    const int lane_in_head = tid - unit_lane * k_threads_per_head;
    const int64_t unit_idx =
        segment_block_idx * k_units_per_block + unit_lane;
    const int64_t total_units = num_tokens * static_cast<int64_t>(num_kv_heads);
    if (unit_idx >= total_units) {
      return;
    }
    const int64_t token_idx = unit_idx / num_kv_heads;
    const int head_idx = static_cast<int>(unit_idx - token_idx * num_kv_heads);
    const int64_t pos = positions[token_idx];
    const scalar_t* token_cos_sin = cos_sin_cache + pos * rot_dim;
    scalar_t* k_head_ptr =
        qkv + token_idx * qkv_stride + q_size + head_idx * head_size;

    index_t cache_loc = 0;
    bool write_cache = has_cache;
    if (has_cache) {
      cache_loc = kv_cache_loc[token_idx];
      write_cache = cache_loc >= 0;
    }
    const float inv_k_cache_scale =
        has_k_cache_scale ? 1.0f / k_cache_scale : 1.0f;
    const int64_t k_base_offset =
        static_cast<int64_t>(cache_loc) * k_buffer_stride_n +
        head_idx * k_buffer_stride_h;

    if (lane_in_head < q_threads_per_head) {
      if (is_neox) {
        const int offset_low = lane_in_head * VEC;
        const int offset_high = half_rot + offset_low;
        scalar_t k_low[VEC], k_high[VEC], cos_v[VEC], sin_v[VEC];
        float k_low_f[VEC], k_high_f[VEC];
        *reinterpret_cast<LoadT*>(k_low) =
            *reinterpret_cast<LoadT*>(k_head_ptr + offset_low);
        *reinterpret_cast<LoadT*>(k_high) =
            *reinterpret_cast<LoadT*>(k_head_ptr + offset_high);
        *reinterpret_cast<LoadT*>(cos_v) =
            *reinterpret_cast<const LoadT*>(token_cos_sin + offset_low);
        *reinterpret_cast<LoadT*>(sin_v) =
            *reinterpret_cast<const LoadT*>(token_cos_sin + half_rot + offset_low);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float x0 = static_cast<float>(k_low[i]);
          const float x1 = static_cast<float>(k_high[i]);
          const float c = static_cast<float>(cos_v[i]);
          const float s = static_cast<float>(sin_v[i]);
          k_low_f[i] = x0 * c - x1 * s;
          k_high_f[i] = x1 * c + x0 * s;
          k_low[i] = static_cast<scalar_t>(k_low_f[i]);
          k_high[i] = static_cast<scalar_t>(k_high_f[i]);
        }
        *reinterpret_cast<LoadT*>(k_head_ptr + offset_low) =
            *reinterpret_cast<LoadT*>(k_low);
        *reinterpret_cast<LoadT*>(k_head_ptr + offset_high) =
            *reinterpret_cast<LoadT*>(k_high);

        if (write_cache) {
          if constexpr (USE_FP8_CACHE) {
            mimo_v2_store_fp8_vec<VEC>(
                reinterpret_cast<uint8_t*>(k_buffer) + k_base_offset +
                    offset_low * k_buffer_stride_d,
                k_buffer_stride_d, k_low_f, inv_k_cache_scale, use_fp8_e4m3);
            mimo_v2_store_fp8_vec<VEC>(
                reinterpret_cast<uint8_t*>(k_buffer) + k_base_offset +
                    offset_high * k_buffer_stride_d,
                k_buffer_stride_d, k_high_f, inv_k_cache_scale, use_fp8_e4m3);
          } else {
            mimo_v2_store_cache_vec<cache_t, scalar_t, VEC>(
                k_buffer + k_base_offset + offset_low * k_buffer_stride_d,
                k_buffer_stride_d, k_low);
            mimo_v2_store_cache_vec<cache_t, scalar_t, VEC>(
                k_buffer + k_base_offset + offset_high * k_buffer_stride_d,
                k_buffer_stride_d, k_high);
          }
        }
      } else {
        const int offset = lane_in_head * VEC;
        const int pair_base = offset / 2;
        using CacheLoadT = at::native::memory::aligned_vector<scalar_t, VEC / 2>;
        scalar_t k_vals[VEC], cos_v[VEC / 2], sin_v[VEC / 2];
        float k_vals_f[VEC];
        *reinterpret_cast<LoadT*>(k_vals) =
            *reinterpret_cast<LoadT*>(k_head_ptr + offset);
        *reinterpret_cast<CacheLoadT*>(cos_v) =
            *reinterpret_cast<const CacheLoadT*>(token_cos_sin + pair_base);
        *reinterpret_cast<CacheLoadT*>(sin_v) =
            *reinterpret_cast<const CacheLoadT*>(token_cos_sin + half_rot + pair_base);
#pragma unroll
        for (int i = 0; i < VEC; i += 2) {
          const int cache_idx = i / 2;
          const float x0 = static_cast<float>(k_vals[i]);
          const float x1 = static_cast<float>(k_vals[i + 1]);
          const float c = static_cast<float>(cos_v[cache_idx]);
          const float s = static_cast<float>(sin_v[cache_idx]);
          k_vals_f[i] = x0 * c - x1 * s;
          k_vals_f[i + 1] = x1 * c + x0 * s;
          k_vals[i] = static_cast<scalar_t>(k_vals_f[i]);
          k_vals[i + 1] = static_cast<scalar_t>(k_vals_f[i + 1]);
        }
        *reinterpret_cast<LoadT*>(k_head_ptr + offset) =
            *reinterpret_cast<LoadT*>(k_vals);

        if (write_cache) {
          if constexpr (USE_FP8_CACHE) {
            mimo_v2_store_fp8_vec<VEC>(
                reinterpret_cast<uint8_t*>(k_buffer) + k_base_offset +
                    offset * k_buffer_stride_d,
                k_buffer_stride_d, k_vals_f, inv_k_cache_scale, use_fp8_e4m3);
          } else {
            mimo_v2_store_cache_vec<cache_t, scalar_t, VEC>(
                k_buffer + k_base_offset + offset * k_buffer_stride_d,
                k_buffer_stride_d, k_vals);
          }
        }
      }
    }

    if (lane_in_head < k_tail_threads_per_head) {
      const int dim_offset = rot_dim + lane_in_head * VEC;
      scalar_t k_tail[VEC];
      *reinterpret_cast<LoadT*>(k_tail) =
          *reinterpret_cast<LoadT*>(k_head_ptr + dim_offset);
      if (write_cache) {
        if constexpr (USE_FP8_CACHE) {
          float k_tail_f[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            k_tail_f[i] = static_cast<float>(k_tail[i]);
          }
          mimo_v2_store_fp8_vec<VEC>(
              reinterpret_cast<uint8_t*>(k_buffer) + k_base_offset +
                  dim_offset * k_buffer_stride_d,
              k_buffer_stride_d, k_tail_f, inv_k_cache_scale, use_fp8_e4m3);
        } else {
          mimo_v2_store_cache_vec<cache_t, scalar_t, VEC>(
              k_buffer + k_base_offset + dim_offset * k_buffer_stride_d,
              k_buffer_stride_d, k_tail);
        }
      }
    }
    return;
  }

  segment_block_idx -= k_block_count;
  const int unit_lane = tid / v_threads_per_head;
  if (unit_lane >= v_units_per_block) {
    return;
  }
  const int lane_in_head = tid - unit_lane * v_threads_per_head;
  const int64_t unit_idx = segment_block_idx * v_units_per_block + unit_lane;
  const int64_t total_units = num_tokens * static_cast<int64_t>(num_kv_heads);
  if (unit_idx >= total_units) {
    return;
  }
  const int64_t token_idx = unit_idx / num_kv_heads;
  const int head_idx = static_cast<int>(unit_idx - token_idx * num_kv_heads);

  index_t cache_loc = 0;
  bool write_cache = has_cache;
  if (has_cache) {
    cache_loc = kv_cache_loc[token_idx];
    write_cache = cache_loc >= 0;
  }
  const float inv_v_cache_scale =
      has_v_cache_scale ? 1.0f / v_cache_scale : 1.0f;

  const int dim_offset = lane_in_head * VEC;
  scalar_t* v_head_ptr =
      qkv + token_idx * qkv_stride + q_size + k_size +
      head_idx * v_head_size;
  scalar_t v_vals[VEC];
  float v_vals_f[VEC];
  *reinterpret_cast<LoadT*>(v_vals) =
      *reinterpret_cast<LoadT*>(v_head_ptr + dim_offset);
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    v_vals_f[i] = static_cast<float>(v_vals[i]) * value_scale;
    v_vals[i] = static_cast<scalar_t>(v_vals_f[i]);
  }
  *reinterpret_cast<LoadT*>(v_head_ptr + dim_offset) =
      *reinterpret_cast<LoadT*>(v_vals);

  if (write_cache) {
    const int64_t v_base_offset =
        static_cast<int64_t>(cache_loc) * v_buffer_stride_n +
        head_idx * v_buffer_stride_h + dim_offset * v_buffer_stride_d;
    if constexpr (USE_FP8_CACHE) {
      mimo_v2_store_fp8_vec<VEC>(
          reinterpret_cast<uint8_t*>(v_buffer) + v_base_offset,
          v_buffer_stride_d, v_vals_f, inv_v_cache_scale, use_fp8_e4m3);
    } else {
      mimo_v2_store_cache_vec<cache_t, scalar_t, VEC>(
          v_buffer + v_base_offset, v_buffer_stride_d, v_vals);
    }
  }
}

template <typename scalar_t, typename cache_t, typename index_t, bool USE_FP8_CACHE>
void launch_mimo_v2_split_rope_vscale_kv_store(
    Tensor& positions,
    Tensor& qkv,
    Tensor& cos_sin_cache,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t head_size,
    int64_t v_head_size,
    double value_scale,
    std::optional<double>& k_cache_scale,
    std::optional<double>& v_cache_scale,
    bool is_neox,
    bool use_fp8_e4m3,
    cudaStream_t stream) {
  const int64_t num_tokens = qkv.size(0);
  const int num_q_heads = static_cast<int>(q_size / head_size);
  const int num_kv_heads = static_cast<int>(v_size / v_head_size);
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  constexpr int VEC = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % VEC == 0, "head_size must be 128-bit vector aligned");
  TORCH_CHECK(v_head_size % VEC == 0,
              "v_head_size must be 128-bit vector aligned");
  TORCH_CHECK(rot_dim > 0 && rot_dim <= head_size,
              "rotary dim must be in (0, head_size]");
  TORCH_CHECK(rot_dim % 2 == 0, "rotary dim must be even");
  TORCH_CHECK((head_size - rot_dim) % VEC == 0,
              "unrotated K tail must be 128-bit vector aligned");
  TORCH_CHECK(!is_neox || (rot_dim / 2) % VEC == 0,
              "neox rotary half dimension must be 128-bit vector aligned");
  TORCH_CHECK(is_neox || rot_dim % VEC == 0,
              "rotary dim must be 128-bit vector aligned");

  const int q_threads_per_head =
      is_neox ? static_cast<int>((rot_dim / 2) / VEC)
              : static_cast<int>(rot_dim / VEC);
  const int k_tail_threads_per_head =
      static_cast<int>((head_size - rot_dim) / VEC);
  const int k_threads_per_head =
      std::max(q_threads_per_head, k_tail_threads_per_head);
  const int v_threads_per_head = static_cast<int>(v_head_size / VEC);
  TORCH_CHECK(q_threads_per_head > 0 && k_threads_per_head > 0 &&
                  v_threads_per_head > 0,
              "mimo_v2 fused prepare thread allocation is empty");

  constexpr int threads = 256;
  const int q_units_per_block = std::max(1, threads / q_threads_per_head);
  const int k_units_per_block = std::max(1, threads / k_threads_per_head);
  const int v_units_per_block = std::max(1, threads / v_threads_per_head);
  const auto ceil_div = [](int64_t value, int64_t divisor) -> int64_t {
    return (value + divisor - 1) / divisor;
  };
  const int64_t q_total_units =
      num_tokens * static_cast<int64_t>(num_q_heads);
  const int64_t kv_total_units =
      num_tokens * static_cast<int64_t>(num_kv_heads);
  const int64_t q_block_count =
      ceil_div(q_total_units, static_cast<int64_t>(q_units_per_block));
  const int64_t k_block_count =
      ceil_div(kv_total_units, static_cast<int64_t>(k_units_per_block));
  const int64_t v_block_count =
      ceil_div(kv_total_units, static_cast<int64_t>(v_units_per_block));
  const int64_t total_blocks = q_block_count + k_block_count + v_block_count;
  if (total_blocks == 0) {
    return;
  }
  TORCH_CHECK(total_blocks <= std::numeric_limits<int>::max(),
              "mimo_v2 fused prepare grid is too large");
  const int blocks = static_cast<int>(total_blocks);

  const bool has_cache = kv_cache_loc.has_value();
  const index_t* loc_ptr =
      has_cache ? kv_cache_loc->data_ptr<index_t>() : nullptr;

  cache_t* k_buf_ptr = nullptr;
  cache_t* v_buf_ptr = nullptr;
  int64_t k_stride_n = 0;
  int64_t k_stride_h = 0;
  int64_t k_stride_d = 0;
  int64_t v_stride_n = 0;
  int64_t v_stride_h = 0;
  int64_t v_stride_d = 0;
  if (has_cache) {
    k_buf_ptr = reinterpret_cast<cache_t*>(k_buffer.data_ptr());
    v_buf_ptr = reinterpret_cast<cache_t*>(v_buffer.data_ptr());
    k_stride_n = k_buffer.stride(0);
    k_stride_h = k_buffer.stride(1);
    k_stride_d = k_buffer.stride(2);
    v_stride_n = v_buffer.stride(0);
    v_stride_h = v_buffer.stride(1);
    v_stride_d = v_buffer.stride(2);
  }

  mimo_v2_split_rope_vscale_kv_store_kernel<scalar_t, cache_t, index_t, USE_FP8_CACHE>
      <<<blocks, threads, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          qkv.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          k_buf_ptr,
          v_buf_ptr,
          loc_ptr,
          num_tokens,
          q_size,
          k_size,
          qkv.stride(0),
          k_stride_n,
          k_stride_h,
          k_stride_d,
          v_stride_n,
          v_stride_h,
          v_stride_d,
          static_cast<int>(head_size),
          static_cast<int>(v_head_size),
          rot_dim,
          num_q_heads,
          num_kv_heads,
          q_threads_per_head,
          k_threads_per_head,
          k_tail_threads_per_head,
          v_threads_per_head,
          q_units_per_block,
          k_units_per_block,
          v_units_per_block,
          q_block_count,
          k_block_count,
          static_cast<float>(value_scale),
          static_cast<float>(k_cache_scale.value_or(1.0)),
          static_cast<float>(v_cache_scale.value_or(1.0)),
          k_cache_scale.has_value(),
          v_cache_scale.has_value(),
          is_neox,
          has_cache,
          use_fp8_e4m3);
}

void mimo_v2_split_rope_vscale_kv_store(
    Tensor& positions,
    Tensor& qkv,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t head_size,
    int64_t v_head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_out,
    Tensor& k_out,
    Tensor& v_out,
    double value_scale,
    std::optional<double>& k_cache_scale,
    std::optional<double>& v_cache_scale,
    bool use_fp8_e4m3,
    bool use_fp8_cache) {
  TORCH_CHECK(qkv.dim() == 2, "qkv must be [num_tokens, q_size + k_size + v_size]");
  TORCH_CHECK(positions.dim() == 1, "positions must be a contiguous 1-D tensor");
  TORCH_CHECK(qkv.size(1) == q_size + k_size + v_size, "qkv last dimension mismatch");
  TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == qkv.size(0) &&
                  q_out.size(1) == q_size,
              "q_out shape mismatch");
  TORCH_CHECK(k_out.dim() == 2 && k_out.size(0) == qkv.size(0) &&
                  k_out.size(1) == k_size,
              "k_out shape mismatch");
  TORCH_CHECK(v_out.dim() == 2 && v_out.size(0) == qkv.size(0) &&
                  v_out.size(1) == v_size,
              "v_out shape mismatch");
  TORCH_CHECK(q_size % head_size == 0, "q_size must be divisible by head_size");
  TORCH_CHECK(k_size % head_size == 0, "k_size must be divisible by head_size");
  TORCH_CHECK(v_size % v_head_size == 0, "v_size must be divisible by v_head_size");
  TORCH_CHECK(k_size / head_size == v_size / v_head_size,
              "K and V must have the same number of KV heads");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2-D");
  TORCH_CHECK(cos_sin_cache.size(1) <= head_size,
              "rotary dim must be <= head_size");
  TORCH_CHECK(cos_sin_cache.size(1) % 2 == 0, "rotary dim must be even");
  if (kv_cache_loc.has_value()) {
    TORCH_CHECK(k_buffer.dim() == 3 && v_buffer.dim() == 3,
                "mimo_v2 fused kv_store expects normal MHA buffers [slots, heads, dim]");
    TORCH_CHECK(k_buffer.size(1) == k_size / head_size &&
                    k_buffer.size(2) == head_size,
                "k_buffer shape mismatch");
    TORCH_CHECK(v_buffer.size(1) == v_size / v_head_size &&
                    v_buffer.size(2) == v_head_size,
                "v_buffer shape mismatch");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const size_t elem_size = qkv.element_size();
  const char* qkv_base = reinterpret_cast<const char*>(qkv.data_ptr());
  const bool outputs_alias_qkv =
      reinterpret_cast<const char*>(q_out.data_ptr()) == qkv_base &&
      reinterpret_cast<const char*>(k_out.data_ptr()) ==
          qkv_base + q_size * elem_size &&
      reinterpret_cast<const char*>(v_out.data_ptr()) ==
          qkv_base + (q_size + k_size) * elem_size;
  TORCH_CHECK(outputs_alias_qkv,
              "mimo_v2 fused prepare expects q_out/k_out/v_out to alias qkv.split");
  TORCH_CHECK(q_out.stride(0) == qkv.stride(0) && q_out.stride(1) == 1 &&
                  k_out.stride(0) == qkv.stride(0) && k_out.stride(1) == 1 &&
                  v_out.stride(0) == qkv.stride(0) && v_out.stride(1) == 1,
              "mimo_v2 fused prepare expects q_out/k_out/v_out split strides");

  AT_DISPATCH_SWITCH(
      qkv.scalar_type(),
      "mimo_v2_split_rope_vscale_kv_store",
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        if (use_fp8_cache && kv_cache_loc.has_value()) {
          if (kv_cache_loc->scalar_type() == at::ScalarType::Long) {
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, uint8_t, int64_t, true>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          } else {
            TORCH_CHECK(kv_cache_loc->scalar_type() == at::ScalarType::Int,
                        "kv_cache_loc must be int64 or int32");
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, uint8_t, int32_t, true>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          }
        } else {
          TORCH_CHECK(!kv_cache_loc.has_value() || k_buffer.scalar_type() == qkv.scalar_type(),
                      "non-fp8 k_buffer dtype must match qkv dtype");
          TORCH_CHECK(!kv_cache_loc.has_value() || v_buffer.scalar_type() == qkv.scalar_type(),
                      "non-fp8 v_buffer dtype must match qkv dtype");
          if (kv_cache_loc.has_value() &&
              kv_cache_loc->scalar_type() == at::ScalarType::Int) {
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, scalar_t, int32_t, false>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          } else {
            if (kv_cache_loc.has_value()) {
              TORCH_CHECK(kv_cache_loc->scalar_type() == at::ScalarType::Long,
                          "kv_cache_loc must be int64 or int32");
            }
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, scalar_t, int64_t, false>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          }
        }
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        if (use_fp8_cache && kv_cache_loc.has_value()) {
          if (kv_cache_loc->scalar_type() == at::ScalarType::Long) {
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, uint8_t, int64_t, true>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          } else {
            TORCH_CHECK(kv_cache_loc->scalar_type() == at::ScalarType::Int,
                        "kv_cache_loc must be int64 or int32");
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, uint8_t, int32_t, true>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          }
        } else {
          TORCH_CHECK(!kv_cache_loc.has_value() || k_buffer.scalar_type() == qkv.scalar_type(),
                      "non-fp8 k_buffer dtype must match qkv dtype");
          TORCH_CHECK(!kv_cache_loc.has_value() || v_buffer.scalar_type() == qkv.scalar_type(),
                      "non-fp8 v_buffer dtype must match qkv dtype");
          if (kv_cache_loc.has_value() &&
              kv_cache_loc->scalar_type() == at::ScalarType::Int) {
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, scalar_t, int32_t, false>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          } else {
            if (kv_cache_loc.has_value()) {
              TORCH_CHECK(kv_cache_loc->scalar_type() == at::ScalarType::Long,
                          "kv_cache_loc must be int64 or int32");
            }
            launch_mimo_v2_split_rope_vscale_kv_store<scalar_t, scalar_t, int64_t, false>(
                positions, qkv, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
                q_size, k_size, v_size, head_size, v_head_size,
                value_scale, k_cache_scale, v_cache_scale, is_neox, use_fp8_e4m3, stream);
          }
        }
      }));

}
}
}
