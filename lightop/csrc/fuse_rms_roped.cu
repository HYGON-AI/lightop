
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

template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
    T val[N];
};
template <typename Element, size_t len>
struct vec {
    using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
};
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


//融合rms+rope
void rms_rotary_embedding_fuse(
    Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    Tensor& query,  // [batch_size, seq_len, num_heads_size] or
                           // [num_tokens, num_heads_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    Tensor& key,
    // null or
    // [batch_size, seq_len, num_kv_heads_size] or
    // [num_tokens, num_kv_heads_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, 
    Tensor weight_q,  // [hidden_size]
    Tensor weight_k,  // [hidden_size]
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon) {
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    (key.size(0) == positions.size(0)),
                "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (key.size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (key.size(1) == positions.size(1)),
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
  int seq_dim_idx = positions_ndim - 1;
 
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.stride(seq_dim_idx);

  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
      
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int qk_mul=num_heads/num_kv_heads;
   AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      query.scalar_type(), "rotary_embedding", [&] {
    using T_ACC = at::acc_type<scalar_t, true>;
  int rot_dim = cos_sin_cache.size(1);

  const int q_k_size_multiple = num_heads / num_kv_heads;
  bool is_allign_qk = (num_heads % num_kv_heads == 0)  && (num_heads >= 1);
    if (head_size == 128 && is_allign_qk && num_kv_heads%4 == 0) {
        constexpr int num_warp = 4;
        constexpr int block_size = 256;
        constexpr int vec_size_q = 2;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
        
      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }else{

                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }   
            
            
    }else if (head_size == 128 && is_allign_qk && num_kv_heads % 2 == 0) {
        constexpr int num_warp = 2;
        constexpr int block_size = 128;

        constexpr int vec_size_q = 2;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads / num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }else{
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 128 && is_allign_qk && num_heads % 3 == 0 && num_kv_heads == 1) {
        constexpr int num_warp = 4;
        constexpr int block_size = num_warp * 64;

        constexpr int vec_size_q = 2;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
      #define LAUNCH_ROTARY_KERNEL_128_pipeline(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel_pipeline<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads / num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }else{
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 128 && is_allign_qk && num_heads % 2 ==0) {
        constexpr int num_warp = 3;
        constexpr int block_size = num_warp * 64;

        constexpr int vec_size_q = 2;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
      #define LAUNCH_ROTARY_KERNEL_128_pipeline(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel_pipeline<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads / num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }else{
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 128 && is_allign_qk && num_heads==1 && num_kv_heads == 1) {
        constexpr int num_warp = 2;
        constexpr int block_size = num_warp * 64;

        constexpr int vec_size_q = 2;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
      #define LAUNCH_ROTARY_KERNEL_128_pipeline(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel_pipeline<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads / num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,true, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,true, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }else{
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 1, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 1, 128, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 2, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 2, 128, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 4, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 4, 128, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128_pipeline(true,false, 8, 128, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128_pipeline(false,false, 8, 128, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 256 && is_allign_qk && num_kv_heads %4 ==0) {
        constexpr int num_warp = 4;
        constexpr int block_size = 64*num_warp;
        constexpr int vec_size_q = 4;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
        
      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 256, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 256, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 256, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 256, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 256, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 256, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 256, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 256, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 256, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 256, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 256, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 256, 8);
                            }
                        } break;
                        }
                    }
            }else{
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 256, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 256, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 256, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 256, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 256, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 256, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 256, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 256, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 256, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 256, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 256, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 256, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 256, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 256, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 256, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 512 && is_allign_qk) {
        constexpr int num_warp = 2;
        constexpr int block_size = 64 * 2;
        constexpr int vec_size_q = 8;
        dim3 grid(num_tokens);
        dim3 block(64 * 2);
      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){
                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 512, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 512, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 512, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 512, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 512, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 512, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 512, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 512, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 512, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 512, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 512, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 512, 8);
                            }
                        } break;
                        }
                    }
            }else{

                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 512, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 512, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 512, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 512, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 512, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 512, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 512, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 512, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 512, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 512, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 512, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 512, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 512, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 512, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 512, 8);
                            }
                        } break;
                        }
                    }
            }   
            
    }else if (head_size == 64 && is_allign_qk) {
        constexpr int num_warp = 2;
        constexpr int block_size = 128;
        constexpr int vec_size_q = 1;
        dim3 grid(num_tokens);
        dim3 block(64 * num_warp);
        
      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,RESIDUAL, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL) \
        rms_rotary_embedding_kernel<T_ACC,scalar_t, IS_NEOX, RESIDUAL, block_size, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL, NUM_QK_MUL, num_warp><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.data_ptr<scalar_t>(), \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon);
        
            if(residual_q.has_value() && residual_k.has_value()){

                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 64, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 64, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 64, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 64, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 64, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 64, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 64, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 64, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 1, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 1, 64, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 2, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 2, 64, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 4, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 4, 64, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,true, 8, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,true, 8, 64, 8);
                            }
                        } break;
                        }
                    }
            }else{

                    if(num_heads/num_kv_heads==2){
                            switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 64, 2);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 64, 2);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 64, 2);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 64, 2);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 64, 2);
                            }
                        } break;
                        }
                    }else if(num_heads/num_kv_heads==4){
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 64, 4);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 64, 4);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 64, 4);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 64, 4);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 64, 4);
                            }
                        } break;
                        }
                    }else{
                        switch (vec_size_q) {
                        case 1:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 1, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 1, 64, 8);
                            }
                        }  break;
                        case 2:
                        {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 2, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 2, 64, 8);
                            }
                        }  break;
                        case 4: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 4, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 4, 64, 8);
                            }
                        } break;
                        default: {
                            if(is_neox){
                                LAUNCH_ROTARY_KERNEL_128(true,false, 8, 64, 8);
                            }else{
                                LAUNCH_ROTARY_KERNEL_128(false,false, 8, 64, 8);
                            }
                        } break;
                        }
                    }
            }   
    }

  }
);
}
}
}