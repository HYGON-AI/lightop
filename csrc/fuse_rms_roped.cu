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
#include <tuple> 

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
    const int64_t max_position,
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
    if (pos < 0 || pos >= max_position) {
        return;
    }
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
    const int64_t max_position,
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
    if (pos < 0 || pos >= max_position) {
        return;
    }
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
        
        
    } 
    else {  // gpt-j style
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


template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX,int VEC_SIZE, int THEAD_PER_HEAD>
__global__ void opt_rms_rope_qwen3(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,        
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t max_position,
    const int rot_dim,
    const int64_t query_stride,         
    const int64_t key_stride,           
    const int64_t head_stride_q,       
    const int64_t head_stride_k,      
    scalar_t* gamma_q,
    scalar_t* gamma_k,
    scalar_t* residual_q,
    scalar_t* residual_k,
    scalar_t eps,
    int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token, 
    const int tokens_per_block   
) 
{
    
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 128;
    constexpr int HALF_ROT = 64; 

    const int tid = threadIdx.x;

    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    const bool valid_token =
        local_token_idx < tokens_per_block && global_token_idx < num_tokens;

    scalar_t* my_s_cos_sin = nullptr;
    bool valid_position = false;
    if (valid_token) {
        const int64_t pos = positions[global_token_idx];
        valid_position = pos >= 0 && pos < max_position;
        if (valid_position) {
            my_s_cos_sin = s_cos_sin_base + local_token_idx * HEAD_SIZE;
            for (int i = lane; i < HEAD_SIZE; i += threads_per_token) {
                my_s_cos_sin[i] = cos_sin_cache[pos * HEAD_SIZE + i];
            }
        }
    }
    __syncthreads();
    if (!valid_token || !valid_position) return;

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    if(lane < q_boundary){
        const int q_head_idx = lane / THEAD_PER_HEAD;
        const int q_lane_in_head = lane % THEAD_PER_HEAD;
        
        scalar_t* q_head_ptr = query + global_token_idx * query_stride + q_head_idx * head_stride_q;
        scalar_t* res_q_head_ptr = HAS_RESIDUAL ? (residual_q + global_token_idx * query_stride + q_head_idx * head_stride_q) : nullptr;
        
        using LoadT = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        scalar_t r_q_low[VEC_SIZE];
        scalar_t r_q_high[VEC_SIZE];

        int offset_low = q_lane_in_head * VEC_SIZE;
        int offset_high = HALF_ROT + q_lane_in_head * VEC_SIZE;

        *(LoadT*)r_q_low = *(LoadT*)(q_head_ptr + offset_low);
        *(LoadT*)r_q_high = *(LoadT*)(q_head_ptr + offset_high);

        T_ACC sum_sq = apply_residual_and_calc_sq<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
            r_q_low, r_q_high, res_q_head_ptr, offset_low, offset_high
        );
    
        sum_sq = WarpReduceSum_Local<T_ACC,THEAD_PER_HEAD>(sum_sq);
        T_ACC inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));
        
        const scalar_t* cache_ptr = my_s_cos_sin; 
       
        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];
            *(LoadT*)r_cos_low = *(LoadT*)(cache_ptr + offset_low);
            *(LoadT*)r_sin_low = *(LoadT*)(cache_ptr + rot_dim/2 + offset_low); 
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                r_q_low[i] = static_cast<T_ACC>(r_q_low[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i]);
                r_q_high[i] = static_cast<T_ACC>(r_q_high[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i]);
                
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
                r_q_low[i]   = static_cast<T_ACC>(r_q_low[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i]);
                r_q_low[i+1] = static_cast<T_ACC>(r_q_low[i+1]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_low + i + 1]);
                scalar_t q0 = r_q_low[i]; scalar_t q1 = r_q_low[i+1];
                scalar_t c = c_low[c_idx]; scalar_t s = s_low[c_idx];
                r_q_low[i]   = q0 * c - q1 * s;
                r_q_low[i+1] = q1 * c + q0 * s;

                r_q_high[i]   = static_cast<T_ACC>(r_q_high[i]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i]);
                r_q_high[i+1] = static_cast<T_ACC>(r_q_high[i+1]) * inv_rms * static_cast<T_ACC>(gamma_q[offset_high + i + 1]);
                scalar_t qh0 = r_q_high[i]; scalar_t qh1 = r_q_high[i+1];
                scalar_t ch = c_high[c_idx]; scalar_t sh = s_high[c_idx];
                r_q_high[i]   = qh0 * ch - qh1 * sh;
                r_q_high[i+1] = qh1 * ch + qh0 * sh;
            }
        }
        *(LoadT*)(q_head_ptr + offset_low) = *(LoadT*)r_q_low;
        *(LoadT*)(q_head_ptr + offset_high) = *(LoadT*)r_q_high;
    }

   
    const int total_threads_needed = (num_heads + num_kv_heads) * THEAD_PER_HEAD;
    
    if (lane >= q_boundary && lane < total_threads_needed && key != nullptr) {
        const int k_lane_abs = lane - q_boundary;
        const int kv_head_idx = k_lane_abs / THEAD_PER_HEAD; 
        const int k_lane_in_head = k_lane_abs % THEAD_PER_HEAD;
    
        
        scalar_t* k_head_ptr = key + global_token_idx * key_stride + kv_head_idx * head_stride_k;
        scalar_t* res_k_head_ptr = HAS_RESIDUAL ? (residual_k + global_token_idx * key_stride + kv_head_idx * head_stride_k) : nullptr; 

        using LoadTK = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
        scalar_t r_k_low[VEC_SIZE];
        scalar_t r_k_high[VEC_SIZE];
        
        int offset_low = k_lane_in_head * VEC_SIZE;          
        int offset_high = HALF_ROT + k_lane_in_head * VEC_SIZE; 
        
        *(LoadTK*)r_k_low = *(LoadTK*)(k_head_ptr + offset_low);
        *(LoadTK*)r_k_high = *(LoadTK*)(k_head_ptr + offset_high);
        
        T_ACC sum_sq_k = apply_residual_and_calc_sq<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
            r_k_low, r_k_high, res_k_head_ptr, offset_low, offset_high
        );
        
        sum_sq_k = WarpReduceSum_Local<T_ACC,THEAD_PER_HEAD>(sum_sq_k);
        T_ACC inv_rms_k = c10::cuda::compat::rsqrt(sum_sq_k / HEAD_SIZE + static_cast<T_ACC>(eps));
        
        const scalar_t* cache_ptr_k = my_s_cos_sin; 
        
        if constexpr (IS_NEOX) {
            scalar_t r_cos_low[VEC_SIZE], r_sin_low[VEC_SIZE];
            scalar_t r_gamma_k_low[VEC_SIZE], r_gamma_k_high[VEC_SIZE];

            *(LoadTK*)r_cos_low = *(LoadTK*)(cache_ptr_k + offset_low);
            *(LoadTK*)r_sin_low = *(LoadTK*)(cache_ptr_k + rot_dim/2 + offset_low);
            *(LoadTK*)r_gamma_k_low = *(LoadTK*)(gamma_k + offset_low);
            *(LoadTK*)r_gamma_k_high = *(LoadTK*)(gamma_k + offset_high);

            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                r_k_low[i] = static_cast<T_ACC>(r_k_low[i]) * inv_rms_k * static_cast<T_ACC>(r_gamma_k_low[i]);
                r_k_high[i] = static_cast<T_ACC>(r_k_high[i]) * inv_rms_k * static_cast<T_ACC>(r_gamma_k_high[i ]);
                
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
                r_k_low[i]   = static_cast<T_ACC>(r_k_low[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_low + i]);
                r_k_low[i+1] = static_cast<T_ACC>(r_k_low[i+1]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_low + i+1]);
                scalar_t k0 = r_k_low[i]; scalar_t k1 = r_k_low[i+1];
                scalar_t c = r_cos_low[c_idx]; scalar_t s = r_sin_low[c_idx];
                r_k_low[i]   = k0 * c - k1 * s;
                r_k_low[i+1] = k1 * c + k0 * s;

                r_k_high[i]   = static_cast<T_ACC>(r_k_high[i]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_high + i]);
                r_k_high[i+1] = static_cast<T_ACC>(r_k_high[i+1]) * inv_rms_k * static_cast<T_ACC>(gamma_k[offset_high + i+1]);
                scalar_t kh0 = r_k_high[i]; scalar_t kh1 = r_k_high[i+1];
                scalar_t ch = r_cos_high[c_idx]; scalar_t sh = r_sin_high[c_idx];
                r_k_high[i]   = kh0 * ch - kh1 * sh;
                r_k_high[i+1] = kh1 * ch + kh0 * sh;
            }
        }
        *(LoadTK*)(k_head_ptr + offset_low) = *(LoadTK*)r_k_low;
        *(LoadTK*)(k_head_ptr + offset_high) = *(LoadTK*)r_k_high;
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
template <typename T_ACC, typename scalar_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, int THEAD_PER_HEAD>
__global__ void opt_rms_rope_qwen3_rotDim64(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,        
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t max_position,
    const int rot_dim,                 
    const int64_t query_stride,          
    const int64_t key_stride,            
    const int64_t head_stride_q,        
    const int64_t head_stride_k,       
    const scalar_t* __restrict__ gamma_q,
    const scalar_t* __restrict__ gamma_k,
    scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k,
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int threads_per_token, 
    const int tokens_per_block   
) 
{
    extern __shared__ char smem_buffer[];
    scalar_t* s_cos_sin_base = reinterpret_cast<scalar_t*>(smem_buffer);

    constexpr int HEAD_SIZE = 128;
    const int tid = threadIdx.x;
    const int local_token_idx = tid / threads_per_token;
    const int lane = tid % threads_per_token;

    const int global_token_idx = blockIdx.x * tokens_per_block + local_token_idx;
    const bool valid_token =
        local_token_idx < tokens_per_block && global_token_idx < num_tokens;

    scalar_t* my_s_cos_sin = nullptr;
    bool valid_position = false;
    if (valid_token) {
        const int64_t pos = positions[global_token_idx];
        valid_position = pos >= 0 && pos < max_position;
        if (valid_position) {
            my_s_cos_sin = s_cos_sin_base + local_token_idx * rot_dim;
            for (int i = lane; i < rot_dim; i += threads_per_token) {
                my_s_cos_sin[i] = cos_sin_cache[pos * rot_dim + i];
            }
        }
    }
    __syncthreads();
    if (!valid_token || !valid_position) return;

    const int q_boundary = num_heads * THEAD_PER_HEAD;
    const int total_threads = (num_heads + num_kv_heads) * THEAD_PER_HEAD;

    if (lane < total_threads) {
        const bool is_query = lane < q_boundary;
        const int head_idx = is_query ? (lane / THEAD_PER_HEAD) : ((lane - q_boundary) / THEAD_PER_HEAD);
        const int lane_in_head = is_query ? (lane % THEAD_PER_HEAD) : ((lane - q_boundary) % THEAD_PER_HEAD);
        
        scalar_t* head_ptr = is_query ? 
            (query + global_token_idx * query_stride + head_idx * head_stride_q) :
            (key + global_token_idx * key_stride + head_idx * head_stride_k);
        
        scalar_t* res_head_ptr = HAS_RESIDUAL ? (is_query ? 
            (residual_q + global_token_idx * query_stride + head_idx * head_stride_q) :
            (residual_k + global_token_idx * key_stride + head_idx * head_stride_k)) : nullptr;

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

  
        T_ACC sum_sq = apply_residual_and_calc_sq_4vec<T_ACC, scalar_t, VEC_SIZE, HAS_RESIDUAL>(
            v0, v1, v2, v3, res_head_ptr, o0, o1, o2, o3
        );
        sum_sq = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(sum_sq);
        T_ACC inv_rms = c10::cuda::compat::rsqrt(sum_sq / HEAD_SIZE + static_cast<T_ACC>(eps));

        if constexpr (IS_NEOX) {
            scalar_t r_cos[VEC_SIZE], r_sin[VEC_SIZE];
            *(LoadT*)r_cos = *(LoadT*)(my_s_cos_sin + o0);
            *(LoadT*)r_sin = *(LoadT*)(my_s_cos_sin + 32 + o0);

            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                T_ACC s0 = static_cast<T_ACC>(v0[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i]);
                T_ACC s1 = static_cast<T_ACC>(v1[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i]);
                T_ACC s2 = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i]);
                T_ACC s3 = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i]);

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
                

                T_ACC s0_0 = static_cast<T_ACC>(v0[i])   * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i]);
                T_ACC s0_1 = static_cast<T_ACC>(v0[i+1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o0 + i + 1]);
                
                v0[i]   = s0_0 * cos0 - s0_1 * sin0;
                v0[i+1] = s0_1 * cos0 + s0_0 * sin0;

    
                int idx_c_v1 = (o1 + i) / 2;
                scalar_t cos1 = my_s_cos_sin[idx_c_v1];
                scalar_t sin1 = my_s_cos_sin[32 + idx_c_v1];

                T_ACC s1_0 = static_cast<T_ACC>(v1[i])   * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i]);
                T_ACC s1_1 = static_cast<T_ACC>(v1[i+1]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o1 + i + 1]);

                v1[i]   = s1_0 * cos1 - s1_1 * sin1;
                v1[i+1] = s1_1 * cos1 + s1_0 * sin1;
            }
            
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                v2[i] = static_cast<T_ACC>(v2[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o2 + i]);
                v3[i] = static_cast<T_ACC>(v3[i]) * inv_rms * static_cast<T_ACC>(gamma_ptr[o3 + i]);
            }
        }
        *(LoadT*)(head_ptr + o0) = *(LoadT*)v0;
        *(LoadT*)(head_ptr + o1) = *(LoadT*)v1;
        *(LoadT*)(head_ptr + o2) = *(LoadT*)v2;
        *(LoadT*)(head_ptr + o3) = *(LoadT*)v3;
    }
}
template <typename T_ACC, typename scalar_t>
void launch_opt_rms_rope(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    const scalar_t* cos_sin_cache,
    int64_t max_position,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride_q,
    int64_t head_stride_k,
    scalar_t* gamma_q,
    scalar_t* gamma_k,
    scalar_t* residual_q_ptr, 
    scalar_t* residual_k_ptr,
    scalar_t eps,
    int num_tokens,
    bool is_neox,
    const int num_heads,
    const int num_kv_heads,
    cudaStream_t stream
) {

        bool has_residual = (residual_q_ptr != nullptr && residual_k_ptr != nullptr);

        constexpr int THREAD_PER_ROW = 8;
        constexpr int VEC = 8;
        int threads_per_token = (num_heads + num_kv_heads) * THREAD_PER_ROW;
        int target_block_size = 512; 
        int tokens_per_block = target_block_size / threads_per_token;
        if (tokens_per_block < 1) tokens_per_block = 1;
        
        int actual_block_size = tokens_per_block * threads_per_token;
        int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
        if(rot_dim == 128){
        size_t smem_size = tokens_per_block * 128 * sizeof(scalar_t);
            DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                    opt_rms_rope_qwen3<T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST,VEC,THREAD_PER_ROW>
                        <<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, cos_sin_cache, max_position,
                        rot_dim,
                        query_stride, key_stride, head_stride_q, head_stride_k, 
                        gamma_q, gamma_k, 
                        residual_q_ptr, residual_k_ptr, 
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block
                    );
                });
            });
        }else if(rot_dim == 64){
            size_t smem_size = tokens_per_block * 64 * sizeof(scalar_t);
            DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                    opt_rms_rope_qwen3_rotDim64<T_ACC, scalar_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST,4,THREAD_PER_ROW>
                        <<<grid_size, actual_block_size, smem_size, stream>>>(
                        positions, query, key, cos_sin_cache, max_position,
                        rot_dim,
                        query_stride, key_stride, head_stride_q, head_stride_k, 
                        gamma_q, gamma_k, 
                        residual_q_ptr, residual_k_ptr, 
                        eps, num_tokens, num_heads, num_kv_heads,
                        threads_per_token, tokens_per_block
                    );
                });
            });
        }else{
            return;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rms_rotary_embedding_fuse(
    Tensor& positions, 
    Tensor& query, 
    Tensor& key,
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    Tensor weight_q, 
    Tensor weight_k, 
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon) {

  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");
  
  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == positions.size(0) && (key.size(0) == positions.size(0)),
                "query, key and positions must have the same number of tokens");
  } else {
    TORCH_CHECK(query.size(0) == positions.size(0) && (key.size(0) == positions.size(0)) &&
                query.size(1) == positions.size(1) && (key.size(1) == positions.size(1)),
                "query, key and positions must have the same batch_size and seq_len");
  }
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.numel() / num_tokens;

  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key_hidden_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim <= 512);
  int query_ndim = query.dim();
  int key_ndim = key.dim();
  bool q_inner_contig = (query.stride(-1) == 1);
  if (query_ndim == positions_ndim + 2) {
      q_inner_contig = q_inner_contig && (query.stride(-2) == head_size);
  }
  bool k_inner_contig = (key.stride(-1) == 1);
  if (key_ndim == positions_ndim + 2) {
      k_inner_contig = k_inner_contig && (key.stride(-2) == head_size);
  }

  if (!q_inner_contig) {
      query = query.contiguous();
  }
  if (!k_inner_contig) {
      key = key.contiguous();
  }
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.stride(seq_dim_idx);
  int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
      
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool has_residual = residual_q.has_value() && residual_k.has_value();
  int qk_ratio = num_heads / num_kv_heads;
  bool is_allign_qk = (num_heads % num_kv_heads == 0) && (num_heads >= 1);


  auto* pos_ptr = positions.data_ptr<int64_t>(); 
  bool qwen3 =  (head_size == 128 && (num_heads + num_kv_heads) <= 128 && (rot_dim == 128 ||  rot_dim == 64));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      query.scalar_type(), "fuse_rms_rotary_embedding", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        //qwne3 opt
         if (qwen3) {
                scalar_t* res_q_ptr = residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
                scalar_t* res_k_ptr = residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;
                launch_opt_rms_rope<T_ACC, scalar_t>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    cos_sin_cache.size(0),
                    rot_dim,
                    query_stride,
                    key_stride,
                    head_stride,
                    head_stride,
                    weight_q.data_ptr<scalar_t>(),
                    weight_k.data_ptr<scalar_t>(),
                    res_q_ptr,
                    res_k_ptr,
                    static_cast<scalar_t>(epsilon),
                    (int)num_tokens,
                    is_neox,
                    num_heads,
                    num_kv_heads,
                    stream
                );
                return;
        }

        auto* wq_ptr = weight_q.data_ptr<scalar_t>();
        auto* wk_ptr = weight_k.data_ptr<scalar_t>();
        auto* res_q_ptr = residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
        auto* res_k_ptr = residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;

   
        auto launch_kernel = [&](auto kernel_tag, auto vec_size_c, auto block_size_c, auto num_warp_c, auto rot_dim_c) {
            
            constexpr int VEC_SIZE = decltype(vec_size_c)::value;
            constexpr int BLOCK_SIZE = decltype(block_size_c)::value;
            constexpr int NUM_WARP = decltype(num_warp_c)::value;
            constexpr int ROT_DIM = decltype(rot_dim_c)::value;

            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                   
                    auto run = [&](auto qk_mul_c) {
                         constexpr int QK_MUL = decltype(qk_mul_c)::value;

                         dim3 grid(num_tokens);
                         dim3 block(BLOCK_SIZE); 

                         if constexpr (std::is_same_v<decltype(kernel_tag), std::integral_constant<int, 0>>) {
                             rms_rotary_embedding_kernel<T_ACC, scalar_t, IS_NEOX_CONST, HAS_RESIDUAL_CONST, 
                                 /*BLOCK_SIZE*/ BLOCK_SIZE,
                                 /*VEC_SIZE_Q*/ VEC_SIZE, 
                                 /*VEC_SIZE_K*/ VEC_SIZE, 
                                 /*ROT_DIM*/ ROT_DIM, 
                                 /*QK_MUL*/ QK_MUL, 
                                 /*NUM_WARP*/ NUM_WARP>
                                 <<<grid, block, 0, stream>>>(
                                     positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                                     key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                     cos_sin_cache.size(0),
                                     rot_dim, query_stride, key_stride, head_stride,
                                     num_heads, num_kv_heads, head_size, num_tokens,
                                     weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
                                     res_q_ptr, res_k_ptr,
                                     epsilon
                                 );
                         } else {
                             rms_rotary_embedding_kernel_pipeline<T_ACC, scalar_t, IS_NEOX_CONST, HAS_RESIDUAL_CONST, 
                                 BLOCK_SIZE, VEC_SIZE, VEC_SIZE, ROT_DIM, QK_MUL, NUM_WARP>
                                 <<<grid, block, 0, stream>>>(
                                     positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                                     key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                     cos_sin_cache.size(0),
                                     rot_dim, query_stride, key_stride, head_stride,
                                     num_heads, num_kv_heads, head_size, num_tokens,
                                     weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
                                     res_q_ptr, res_k_ptr, 
                                     epsilon
                                 );
                         }
                    };
                    
    
                    switch (qk_ratio) {
                        case 2: run(IV(2)); break;
                        case 4: run(IV(4)); break;
                        default: run(IV(8)); break; 
                    }
                });
            });
        };

        auto USE_NORMAL_KERNEL = std::integral_constant<int, 0>{};
        auto USE_PIPELINE_KERNEL = std::integral_constant<int, 1>{};

    
        if (head_size == 128 && is_allign_qk) {
            if (num_kv_heads % 4 == 0) {
                // kernel_tag, vec_size, block_size, num_warp, rot_dim
                launch_kernel(USE_NORMAL_KERNEL, IV(2), IV(256), IV(4), IV(128));
            } 
            else if (num_kv_heads % 2 == 0) {
                launch_kernel(USE_NORMAL_KERNEL, IV(2), IV(128), IV(2), IV(128));
            }
            else if (num_heads % 3 == 0 && num_kv_heads == 1) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(256), IV(4), IV(128)); // 4*64=256
            }
            else if (num_heads % 2 == 0) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(192), IV(3), IV(128)); // 3*64=192
            }
            else if (num_heads == 1 && num_kv_heads == 1) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(128), IV(2), IV(128)); // 2*64=128
            }
        } 
        else if (head_size == 256 && is_allign_qk && num_kv_heads % 4 == 0) {
            launch_kernel(USE_NORMAL_KERNEL, IV(4), IV(256), IV(4), IV(256)); // 64*4=256
        }
        else if (head_size == 512 && is_allign_qk) {
            launch_kernel(USE_NORMAL_KERNEL, IV(8), IV(128), IV(2), IV(512)); // 64*2=128
        }
        else if (head_size == 64 && is_allign_qk) {
            launch_kernel(USE_NORMAL_KERNEL, IV(1), IV(128), IV(2), IV(64));
        }
  });
}
std::tuple<Tensor, Tensor, Tensor> qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, 
    Tensor& qkv, 
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    Tensor weight_q, 
    Tensor weight_k, 
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon) {

  TORCH_CHECK(qkv.size(-1) == q_size + 2 * kv_size, "qkv last dimension size mismatch");

  Tensor query = qkv.narrow(-1, 0, q_size);
  Tensor key = qkv.narrow(-1, q_size, kv_size);
  Tensor value = qkv.narrow(-1, q_size + kv_size, kv_size);

  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");
  
  if (positions_ndim == 1) {
    TORCH_CHECK(qkv.size(0) == positions.size(0),
                "qkv and positions must have the same number of tokens");
  } else {
    TORCH_CHECK(qkv.size(0) == positions.size(0) && qkv.size(1) == positions.size(1),
                "qkv and positions must have the same batch_size and seq_len");
  }
  
  TORCH_CHECK(q_size % head_size == 0);
  TORCH_CHECK(kv_size % head_size == 0);

  int num_heads = q_size / head_size;
  int num_kv_heads = kv_size / head_size;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim <= 512);
  
  int query_ndim = query.dim();
  int key_ndim = key.dim();
  bool q_inner_contig = (query.stride(-1) == 1);
  if (query_ndim == positions_ndim + 2) {
      q_inner_contig = q_inner_contig && (query.stride(-2) == head_size);
  }
  bool k_inner_contig = (key.stride(-1) == 1);
  if (key_ndim == positions_ndim + 2) {
      k_inner_contig = k_inner_contig && (key.stride(-2) == head_size);
  }

  if (!q_inner_contig) {
      query = query.contiguous();
  }
  if (!k_inner_contig) {
      key = key.contiguous();
  }

  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.stride(seq_dim_idx);
  int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
      
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool has_residual = residual_q.has_value() && residual_k.has_value();
  int qk_ratio = num_heads / num_kv_heads;
  bool is_allign_qk = (num_heads % num_kv_heads == 0) && (num_heads >= 1);


  auto* pos_ptr = positions.data_ptr<int64_t>(); 
  bool qwen3 =  (head_size == 128 && (num_heads + num_kv_heads) <= 128 && (rot_dim == 128 ||  rot_dim == 64));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      query.scalar_type(), "qkv_split_rms_rotary_embedding_fuse", [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
         if (qwen3) {
                scalar_t* res_q_ptr = residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
                scalar_t* res_k_ptr = residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;
                launch_opt_rms_rope<T_ACC, scalar_t>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    cos_sin_cache.size(0),
                    rot_dim,
                    query_stride,
                    key_stride,
                    head_stride,
                    head_stride,
                    weight_q.data_ptr<scalar_t>(),
                    weight_k.data_ptr<scalar_t>(),
                    res_q_ptr,
                    res_k_ptr,
                    static_cast<scalar_t>(epsilon),
                    (int)num_tokens,
                    is_neox,
                    num_heads,
                    num_kv_heads,
                    stream
                );
                return;
        }

        auto* wq_ptr = weight_q.data_ptr<scalar_t>();
        auto* wk_ptr = weight_k.data_ptr<scalar_t>();
        auto* res_q_ptr = residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr;
        auto* res_k_ptr = residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr;

   
        auto launch_kernel = [&](auto kernel_tag, auto vec_size_c, auto block_size_c, auto num_warp_c, auto rot_dim_c) {
            
            constexpr int VEC_SIZE = decltype(vec_size_c)::value;
            constexpr int BLOCK_SIZE = decltype(block_size_c)::value;
            constexpr int NUM_WARP = decltype(num_warp_c)::value;
            constexpr int ROT_DIM = decltype(rot_dim_c)::value;

            DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
                DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
                   
                    auto run = [&](auto qk_mul_c) {
                         constexpr int QK_MUL = decltype(qk_mul_c)::value;

                         dim3 grid(num_tokens);
                         dim3 block(BLOCK_SIZE); 

                         if constexpr (std::is_same_v<decltype(kernel_tag), std::integral_constant<int, 0>>) {
                             rms_rotary_embedding_kernel<T_ACC, scalar_t, IS_NEOX_CONST, HAS_RESIDUAL_CONST, 
                                 /*BLOCK_SIZE*/ BLOCK_SIZE,
                                 /*VEC_SIZE_Q*/ VEC_SIZE, 
                                 /*VEC_SIZE_K*/ VEC_SIZE, 
                                 /*ROT_DIM*/ ROT_DIM, 
                                 /*QK_MUL*/ QK_MUL, 
                                 /*NUM_WARP*/ NUM_WARP>
                                 <<<grid, block, 0, stream>>>(
                                     positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                                     key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                     cos_sin_cache.size(0),
                                     rot_dim, query_stride, key_stride, head_stride,
                                     num_heads, num_kv_heads, head_size, num_tokens,
                                     weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
                                     res_q_ptr, res_k_ptr,
                                     epsilon
                                 );
                         } else {
                             rms_rotary_embedding_kernel_pipeline<T_ACC, scalar_t, IS_NEOX_CONST, HAS_RESIDUAL_CONST, 
                                 BLOCK_SIZE, VEC_SIZE, VEC_SIZE, ROT_DIM, QK_MUL, NUM_WARP>
                                 <<<grid, block, 0, stream>>>(
                                     positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                                     key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                     cos_sin_cache.size(0),
                                     rot_dim, query_stride, key_stride, head_stride,
                                     num_heads, num_kv_heads, head_size, num_tokens,
                                     weight_q.data_ptr<scalar_t>(), weight_k.data_ptr<scalar_t>(),
                                     res_q_ptr, res_k_ptr, 
                                     epsilon
                                 );
                         }
                    };
                    
    
                    switch (qk_ratio) {
                        case 2: run(IV(2)); break;
                        case 4: run(IV(4)); break;
                        default: run(IV(8)); break; 
                    }
                });
            });
        };

        auto USE_NORMAL_KERNEL = std::integral_constant<int, 0>{};
        auto USE_PIPELINE_KERNEL = std::integral_constant<int, 1>{};

    
        if (head_size == 128 && is_allign_qk) {
            if (num_kv_heads % 4 == 0) {
                launch_kernel(USE_NORMAL_KERNEL, IV(2), IV(256), IV(4), IV(128));
            } 
            else if (num_kv_heads % 2 == 0) {
                launch_kernel(USE_NORMAL_KERNEL, IV(2), IV(128), IV(2), IV(128));
            }
            else if (num_heads % 3 == 0 && num_kv_heads == 1) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(256), IV(4), IV(128)); // 4*64=256
            }
            else if (num_heads % 2 == 0) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(192), IV(3), IV(128)); // 3*64=192
            }
            else if (num_heads == 1 && num_kv_heads == 1) {
                launch_kernel(USE_PIPELINE_KERNEL, IV(2), IV(128), IV(2), IV(128)); // 2*64=128
            }
        } 
        else if (head_size == 256 && is_allign_qk && num_kv_heads % 4 == 0) {
            launch_kernel(USE_NORMAL_KERNEL, IV(4), IV(256), IV(4), IV(256)); // 64*4=256
        }
        else if (head_size == 512 && is_allign_qk) {
            launch_kernel(USE_NORMAL_KERNEL, IV(8), IV(128), IV(2), IV(512)); // 64*2=128
        }
        else if (head_size == 64 && is_allign_qk) {
            launch_kernel(USE_NORMAL_KERNEL, IV(1), IV(128), IV(2), IV(64));
        }
  });
  return std::make_tuple(query, key, value);
}

}
}