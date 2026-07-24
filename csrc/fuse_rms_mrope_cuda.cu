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


#ifdef USE_ROCM
#include <hip/hip_fp16.h>
#endif

namespace at {
namespace native {

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


template <typename T_ACC, typename scalar_t,int Vec=4,int block_size=512, int num_warp, bool pipeline, bool is_q>
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

template <typename T_ACC, typename scalar_t,int Vec=4,int block_size=512, int num_warp, bool pipeline = false, bool is_q = false>
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
        // *(LoadT*)(input+idx)=*(LoadT*)intput_vec;
    }
}



template <typename T_ACC, typename scalar_t, bool is_interleaved, bool RESIDUAL, int vec_size, int num_warp ,int head_size, int block_size>
__global__ void fuse_rms_mrope_cuda_kernel_optimized(
    scalar_t *__restrict__ q_ptr,
    scalar_t *__restrict__ k_ptr,
    const scalar_t *__restrict__ cos,
    const scalar_t *__restrict__ sin,
    int num_tokens,
    int n_qh,
    int n_kh,
    int rd,
    int pad_n_qh,
    int pad_n_kh,
    int pad_hd,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    scalar_t* gamma_q, scalar_t* gamma_k, scalar_t* residual_q, scalar_t* residual_k, scalar_t eps
) {
    int pid = blockIdx.x; 
    if (pid >= num_tokens) return;
    const int warp_size = 64;

    __shared__ scalar_t smem_cos[head_size/2];
    __shared__ scalar_t smem_sin[head_size/2];

    int half_rd = rd / 2;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int land = tid / 64;
    int stride = land * rd;
    

    if(tid < half_rd){
            int64_t token_stride = num_tokens * half_rd;
            const scalar_t* t_cos_ptr = cos + pid * half_rd;
            const scalar_t* h_cos_ptr = t_cos_ptr + token_stride;
            const scalar_t* w_cos_ptr = h_cos_ptr + token_stride;
            const scalar_t* t_sin_ptr = sin + pid * half_rd;
            const scalar_t* h_sin_ptr = t_sin_ptr + token_stride;
            const scalar_t* w_sin_ptr = h_sin_ptr + token_stride;
            scalar_t c, s;

            bool is_h, is_w;
            if constexpr (is_interleaved) {
                is_h = ((tid % 3) == 1) && (tid <= 3 * mrope_section_h);
                is_w = ((tid % 3) == 2) && (tid <= 3 * mrope_section_w);
            } else {
                is_h = (tid >= mrope_section_t) && (tid < mrope_section_t + mrope_section_h);
                is_w = (tid >= mrope_section_t + mrope_section_h) && (tid < half_rd);
            }
            
            if (is_h) {
                c = h_cos_ptr[tid];
                s = h_sin_ptr[tid];
            } else if (is_w) {
                c = w_cos_ptr[tid];
                s = w_sin_ptr[tid];
            } else {
                // is_t
                c = t_cos_ptr[tid];
                s = t_sin_ptr[tid];
            }

            smem_cos[tid] = c;
            smem_sin[tid] = s;
        }
        
    // }
    __syncthreads();
    // -----------------------------------------------------------
    // 计算 Q / K
    // -----------------------------------------------------------

    using LoadT = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int q_stride = pid * n_qh * head_size;
    int k_stride = pid * n_kh * head_size;
    auto q_curr = q_ptr + q_stride;
    auto k_curr = k_ptr + k_stride;
    int idx = tid * vec_size;
    int head_idx_k = 0, head_idx_q = 0;
    // k&q
    for(; head_idx_k < n_kh; head_idx_k += num_warp, head_idx_q += num_warp){
        scalar_t* ptr_k = k_curr + head_idx_k * head_size;
        scalar_t* ptr_q = q_curr + head_idx_q * head_size;
        
        scalar_t input_vec_k[vec_size];
        scalar_t input_vec_q[vec_size];
        if constexpr (RESIDUAL){
            scalar_t* residual_q_ptr = residual_q + q_stride + head_idx_q * head_size;
            scalar_t* residual_k_ptr = residual_k + k_stride + head_idx_k * head_size;
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, residual_q_ptr, head_size, eps, input_vec_q);
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, residual_k_ptr, head_size, eps, input_vec_k);
        }else{
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, head_size, eps, input_vec_q);
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, head_size, eps, input_vec_k);
            }
        // *(LoadT*)input_vec_k = *(LoadT*)(ptr_k+idx);
        // *(LoadT*)input_vec_q = *(LoadT*)(ptr_q+idx);
        bool is_vec1 = tid % warp_size < 32 ? true : false;
        __shared__ scalar_t k_smem[head_size * num_warp];
        __shared__ scalar_t q_smem[head_size * num_warp];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
                k_smem[idx + i] = input_vec_k[i];
                q_smem[idx + i] = input_vec_q[i];
            }
        __syncthreads();
        if(is_vec1){
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                input_vec_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] - k_smem[idx + i + half_rd] * smem_sin[(idx + i) % half_rd];
                input_vec_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] - q_smem[idx + i + half_rd] * smem_sin[(idx + i) % half_rd];
            }
        }else{
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                input_vec_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] + k_smem[idx + i - half_rd] * smem_sin[(idx + i) % half_rd];
                input_vec_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] + q_smem[idx + i - half_rd] * smem_sin[(idx + i) % half_rd];
            }
        }
        *(LoadT*)(ptr_k + idx)=*(LoadT*)input_vec_k;
        *(LoadT*)(ptr_q + idx)=*(LoadT*)input_vec_q;
        
    }
    // q
    for(head_idx_q = n_kh; head_idx_q < n_qh; head_idx_q += num_warp){
        scalar_t* ptr_q = q_curr + head_idx_q * head_size;
        scalar_t input_vec_q[vec_size];
        if constexpr (RESIDUAL){
            scalar_t* residual_q_ptr = residual_q + q_stride + head_idx_q * head_size;
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, residual_q_ptr, head_size, eps, input_vec_q);
        }else{
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, head_size, eps, input_vec_q);
            }
        bool is_vec1 = tid % warp_size < 32 ? true : false;
        __shared__ scalar_t q_smem[head_size * num_warp];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
                q_smem[idx + i] = input_vec_q[i];
            }
        __syncthreads();
        if(is_vec1){
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                input_vec_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] - q_smem[idx + i + half_rd] * smem_sin[(idx + i) % half_rd];
            }
        }else{
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                input_vec_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] + q_smem[idx + i - half_rd] * smem_sin[(idx + i) % half_rd];
            }
        }
        *(LoadT*)(ptr_q + idx)=*(LoadT*)input_vec_q;
        
    }
}

template <typename T_ACC, typename scalar_t, bool is_interleaved, bool RESIDUAL, int vec_size, int num_warp ,int head_size, int block_size>
__global__ void fuse_rms_mrope_cuda_kernel_optimized_pipeline(
    scalar_t *__restrict__ q_ptr,
    scalar_t *__restrict__ k_ptr,
    const scalar_t *__restrict__ cos,
    const scalar_t *__restrict__ sin,
    int num_tokens,
    int n_qh,
    int n_kh,
    int rd,
    int pad_n_qh,
    int pad_n_kh,
    int pad_hd,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    scalar_t* gamma_q, scalar_t* gamma_k, scalar_t* residual_q, scalar_t* residual_k, scalar_t eps
) {
    int pid = blockIdx.x; 
    if (pid >= num_tokens) return;
    const int warp_size = 64;

    __shared__ scalar_t smem_cos[head_size/2];
    __shared__ scalar_t smem_sin[head_size/2];

    int half_rd = rd / 2;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int land = tid / 64;
    int stride = land * rd;
    if(tid < half_rd){
            int64_t token_stride = num_tokens * half_rd;
            const scalar_t* t_cos_ptr = cos + pid * half_rd;
            const scalar_t* h_cos_ptr = t_cos_ptr + token_stride;
            const scalar_t* w_cos_ptr = h_cos_ptr + token_stride;
            const scalar_t* t_sin_ptr = sin + pid * half_rd;
            const scalar_t* h_sin_ptr = t_sin_ptr + token_stride;
            const scalar_t* w_sin_ptr = h_sin_ptr + token_stride;
            scalar_t c, s;

            bool is_h, is_w;
            if constexpr (is_interleaved) {
                is_h = ((tid % 3) == 1) && (tid <= 3 * mrope_section_h);
                is_w = ((tid % 3) == 2) && (tid <= 3 * mrope_section_w);
            } else {
                is_h = (tid >= mrope_section_t) && (tid < mrope_section_t + mrope_section_h);
                is_w = (tid >= mrope_section_t + mrope_section_h) && (tid < half_rd);
            }
            
            if (is_h) {
                c = h_cos_ptr[tid];
                s = h_sin_ptr[tid];
            } else if (is_w) {
                c = w_cos_ptr[tid];
                s = w_sin_ptr[tid];
            } else {
                // is_t
                c = t_cos_ptr[tid];
                s = t_sin_ptr[tid];
            }

            smem_cos[tid] = c;
            smem_sin[tid] = s;
        }
    __syncthreads();
    // -----------------------------------------------------------
    // 计算 Q / K
    // -----------------------------------------------------------

    using LoadT = at::native::memory::aligned_vector<scalar_t, vec_size>;

    auto q_curr = q_ptr + pid * n_qh * head_size;
    auto k_curr = k_ptr + pid * n_kh * head_size;


    int head_idx_k = 0, head_idx_q = 0;
    constexpr  int num_warp_q = num_warp-1;
    if(tid >= 64){
        //q
        for(int head_idx_q = 0; head_idx_q < n_qh; head_idx_q += num_warp_q){
            scalar_t* ptr_q = q_curr + head_idx_q * head_size;
            int64_t idx = (tid-64) * vec_size;
            scalar_t input_vec_q[vec_size];
            if constexpr (RESIDUAL){
                scalar_t* residual_q_ptr = residual_q + pid * n_qh * head_size + head_idx_q * head_size;
                apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp_q, true, true>(ptr_q, gamma_q, residual_q_ptr, head_size, eps, input_vec_q);
            }else{
                apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp_q, true, true>(ptr_q, gamma_q, head_size, eps, input_vec_q);
                }
            bool is_vec1 = (tid-64) % warp_size < 32 ? true : false;
            __shared__ scalar_t q_smem[head_size * num_warp_q];
            #pragma unroll
            for (int i = 0; i < vec_size; i++) {
                    q_smem[idx + i] = input_vec_q[i];
                }
            __syncthreads();

            scalar_t output_q[vec_size];
            if(is_vec1){
                #pragma unroll
                for(int i=0;i<vec_size; i++){
                    output_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] - q_smem[idx + i + half_rd] * smem_sin[(idx + i) % half_rd];

                }
            }else{
                #pragma unroll
                for(int i=0;i<vec_size; i++){
                    output_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] + q_smem[idx + i - half_rd] * smem_sin[(idx + i) % half_rd];

                }
            }
            *(LoadT*)(ptr_q + idx)=*(LoadT*)output_q;
        
    }
      
    }else{
        // k
        scalar_t* ptr_k = k_curr ;
        int64_t idx = tid * vec_size;
        scalar_t input_vec_k[vec_size];
        if constexpr (RESIDUAL){
            scalar_t* residual_k_ptr = residual_k + pid * n_kh * head_size ;
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, residual_k_ptr, head_size, eps, input_vec_k);
        }else{
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, head_size, eps, input_vec_k);
            }
        bool is_vec1 = tid % warp_size < 32 ? true : false;
        __shared__ scalar_t k_smem[head_size * num_warp];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
                k_smem[idx + i] = input_vec_k[i];
            }
        __syncthreads();
        scalar_t output_k[vec_size];
        if(is_vec1){
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] - k_smem[(idx + i + half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }else{
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] + k_smem[(idx + i - half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }
        *(LoadT*)(ptr_k + idx)=*(LoadT*)output_k;
        
    }
}

template <typename T_ACC, typename scalar_t, bool is_interleaved, bool RESIDUAL, int vec_size, int num_warp ,int head_size, int block_size>
__global__ void fuse_rms_mrope_cuda_kernel_optimized_small(
    scalar_t *__restrict__ q_ptr,
    scalar_t *__restrict__ k_ptr,
    const scalar_t *__restrict__ cos,
    const scalar_t *__restrict__ sin,
    int num_tokens,
    int n_qh,
    int n_kh,
    int rd,
    int pad_n_qh,
    int pad_n_kh,
    int pad_hd,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    scalar_t* gamma_q, scalar_t* gamma_k, scalar_t* residual_q, scalar_t* residual_k, scalar_t eps
) {
    int pid = blockIdx.x; 
    if (pid >= num_tokens) return;
    const int warp_size = 64;
    __shared__ scalar_t smem_cos[head_size/2];
    __shared__ scalar_t smem_sin[head_size/2];

    int half_rd = rd / 2;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int land = tid / 64;
    int stride = land * rd;
    
    

    // for (int j = tid; j < half_rd; j += bsz) {
        if(tid < half_rd){
            int64_t token_stride = num_tokens * half_rd;
            const scalar_t* t_cos_ptr = cos + pid * half_rd;
            const scalar_t* h_cos_ptr = t_cos_ptr + token_stride;
            const scalar_t* w_cos_ptr = h_cos_ptr + token_stride;
            const scalar_t* t_sin_ptr = sin + pid * half_rd;
            const scalar_t* h_sin_ptr = t_sin_ptr + token_stride;
            const scalar_t* w_sin_ptr = h_sin_ptr + token_stride;
            scalar_t c, s;

            bool is_h, is_w;
            if constexpr (is_interleaved) {
                is_h = ((tid % 3) == 1) && (tid <= 3 * mrope_section_h);
                is_w = ((tid % 3) == 2) && (tid <= 3 * mrope_section_w);
            } else {
                is_h = (tid >= mrope_section_t) && (tid < mrope_section_t + mrope_section_h);
                is_w = (tid >= mrope_section_t + mrope_section_h) && (tid < half_rd);
            }
            
            if (is_h) {
                c = h_cos_ptr[tid];
                s = h_sin_ptr[tid];
            } else if (is_w) {
                c = w_cos_ptr[tid];
                s = w_sin_ptr[tid];
            } else {
                // is_t
                c = t_cos_ptr[tid];
                s = t_sin_ptr[tid];
            }

            smem_cos[tid] = c;
            smem_sin[tid] = s;
        }
        
    // }
    __syncthreads();
    // -----------------------------------------------------------
    // 计算 Q / K
    // -----------------------------------------------------------
    
    using LoadT = at::native::memory::aligned_vector<scalar_t, vec_size>;

    auto q_curr = q_ptr + pid * n_qh * head_size;
    auto k_curr = k_ptr + pid * n_kh * head_size;

    //q
    for(int head_idx_q = 0; head_idx_q < n_qh; head_idx_q += num_warp){
        scalar_t* ptr_q = q_curr + head_idx_q * head_size;
        int64_t idx = tid * vec_size;
        scalar_t input_vec_q[vec_size];
        if constexpr (RESIDUAL){
            scalar_t* residual_q_ptr = residual_q + pid * n_qh * head_size + head_idx_q * head_size;
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, residual_q_ptr, head_size, eps, input_vec_q);
        }else{
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_q, gamma_q, head_size, eps, input_vec_q);
            }
        bool is_vec1 = tid % warp_size < 32 ? true : false;
        __shared__ scalar_t q_smem[head_size * num_warp];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
                q_smem[idx + i] = input_vec_q[i];
            }
        __syncthreads();
        scalar_t output_q[vec_size];
        if(is_vec1){
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] - q_smem[(idx + i + half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }else{
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_q[i] = input_vec_q[i] * smem_cos[(idx + i)%half_rd] + q_smem[(idx + i - half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }
        *(LoadT*)(ptr_q + idx)=*(LoadT*)output_q;
        
    }
    // k
    for(int head_idx_k = 0; head_idx_k < n_kh; head_idx_k += num_warp){
        scalar_t* ptr_k = k_curr + head_idx_k * head_size;
        int64_t idx = tid * vec_size;
        scalar_t input_vec_k[vec_size];
        if constexpr (RESIDUAL){
            scalar_t* residual_k_ptr = residual_k + pid * n_kh * head_size + head_idx_k * head_size;
            apply_rmsnorm_residual<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, residual_k_ptr, head_size, eps, input_vec_k);
        }else{
            apply_rmsnorm<T_ACC, scalar_t, vec_size, block_size, num_warp, false, false>(ptr_k, gamma_k, head_size, eps, input_vec_k);
            }
        bool is_vec1 = tid % warp_size < 32 ? true : false;
        __shared__ scalar_t k_smem[head_size * num_warp];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
                k_smem[idx + i] = input_vec_k[i];
            }
        __syncthreads();
        scalar_t output_k[vec_size];
        if(is_vec1){
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] - k_smem[(idx + i + half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }else{
            #pragma unroll
            for(int i=0;i<vec_size; i++){
                output_k[i] = input_vec_k[i] * smem_cos[(idx + i)%half_rd] + k_smem[(idx + i - half_rd)%head_size + stride] * smem_sin[(idx + i) % half_rd];
            }
        }
        *(LoadT*)(ptr_k + idx)=*(LoadT*)output_k;
        
    }
}

void fuse_rms_mrope_cuda(
    Tensor &q, Tensor &k, const Tensor &cos, const Tensor &sin,
    const std::vector<int>& mrope_section,
    int head_size,
    int rotary_dim,
    bool is_interleaved,
    Tensor weight_q,  // [hidden_size]
    Tensor weight_k,  // [hidden_size]
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon
) {
    int mrope_section_t = mrope_section[0];
    int mrope_section_h = mrope_section[1];
    int mrope_section_w = mrope_section[2];

    int num_tokens = q.size(0);
    int n_qh = q.size(1) / head_size; 
    int n_kh = k.size(1) / head_size;

    int pad_hd = (head_size + 1) / 2 * 2; 
    int pad_n_qh = (n_qh + 1) / 2 * 2;
    int pad_n_kh = (n_kh + 1) / 2 * 2;

    const at::DeviceGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(head_size == 64 || head_size == 128);
    #define LAUNCH_MROTARY_KERNEL(is_interleaved, RESIDUAL, num_warp, head_size, block_size) \
        fuse_rms_mrope_cuda_kernel_optimized<T_ACC, scalar_t, is_interleaved, RESIDUAL, vec_size, num_warp, head_size, block_size><<<grid, block_size, shared_mem_size, stream>>>(\
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),\
                num_tokens, n_qh, n_kh,  rotary_dim, pad_n_qh, pad_n_kh, pad_hd,\
                mrope_section_t, mrope_section_h, mrope_section_w, weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon );

    #define LAUNCH_MROTARY_KERNEL_PIPELINE(is_interleaved, RESIDUAL, num_warp, head_size, block_size) \
        fuse_rms_mrope_cuda_kernel_optimized_pipeline<T_ACC, scalar_t, is_interleaved, RESIDUAL, vec_size, num_warp, head_size, block_size><<<grid, block_size, shared_mem_size, stream>>>(\
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),\
                num_tokens, n_qh, n_kh,  rotary_dim, pad_n_qh, pad_n_kh, pad_hd,\
                mrope_section_t, mrope_section_h, mrope_section_w, weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon );
    #define LAUNCH_MROTARY_KERNEL_SMALL(is_interleaved, RESIDUAL, num_warp, head_size, block_size) \
        fuse_rms_mrope_cuda_kernel_optimized_small<T_ACC, scalar_t, is_interleaved, RESIDUAL, vec_size, num_warp, head_size, block_size><<<grid, block_size, shared_mem_size, stream>>>(\
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),\
                num_tokens, n_qh, n_kh,  rotary_dim, pad_n_qh, pad_n_kh, pad_hd,\
                mrope_section_t, mrope_section_h, mrope_section_w, weight_q.data_ptr<scalar_t>(),weight_k.data_ptr<scalar_t>(),residual_q.has_value()?residual_q->data_ptr<scalar_t>():nullptr,residual_k.has_value()?residual_k->data_ptr<scalar_t>():nullptr,epsilon );

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "fuse_rms_mrope_cuda", [&] {

        using T_ACC = at::acc_type<scalar_t, true>;
        int shared_mem_size = (rotary_dim / 2) * 2 * sizeof(scalar_t);
        bool is_mul = n_qh%n_kh==0?true : false;
        if(num_tokens <= 256){
            if (head_size == 128 && is_mul && n_kh%4==0) {
            constexpr int num_warp = 4;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 128, block_size);
                }
            }
        }else if (head_size == 128 && is_mul && n_kh%3==0) {
            constexpr int num_warp = 3;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 128, block_size);
                }
            }

        }else if (head_size == 128 && is_mul && n_kh%2==0) {
            constexpr int num_warp = 2;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 128, block_size);
                }
            }

        }else if (head_size == 128 && is_mul && n_qh%3==0 && n_kh == 1) {
            constexpr int num_warp = 4;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_PIPELINE(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_PIPELINE(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_PIPELINE(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_PIPELINE(false, false, num_warp, 128, block_size);
                }
            }

        }else if (head_size == 128 && is_mul && n_qh%2==0 && n_kh == 1) {
            constexpr int num_warp = 3;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_PIPELINE(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_PIPELINE(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_PIPELINE(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_PIPELINE(false, false, num_warp, 128, block_size);
                }
            }

        }else if (head_size == 128) {
            constexpr int num_warp = 1;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 2;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 128, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 128, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 128, block_size);
                }
            }

        }else if(head_size == 64 && is_mul && n_kh % 4 == 0){
            constexpr int num_warp = 4;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 1;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 64, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 64, block_size);
                }
            }
            
        }else if(head_size == 64 && is_mul && n_kh % 2 == 0){
            constexpr int num_warp = 2;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 1;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 64, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 64, block_size);
                }
            }
        }else if(head_size == 64 && n_kh % 1 == 0){
            constexpr int num_warp = 1;
            constexpr int block_size = num_warp * 64;
            constexpr int vec_size = 1;
            dim3 grid(num_tokens);
            dim3 block(block_size);
            if(residual_q.has_value() && residual_k.has_value()){
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, true, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, true, num_warp, 64, block_size);
                }
            }else{
                if(is_interleaved){
                    LAUNCH_MROTARY_KERNEL_SMALL(true, false, num_warp, 64, block_size);
                }else{
                    LAUNCH_MROTARY_KERNEL_SMALL(false, false, num_warp, 64, block_size);
                }
            }
        }
        }else{
                if (head_size == 128 && is_mul && n_kh%4==0) {
                constexpr int num_warp = 4;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 128, block_size);
                    }
                }
            }else if (head_size == 128 && is_mul && n_kh%3==0) {
                constexpr int num_warp = 3;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 128, block_size);
                    }
                }

            }else if (head_size == 128 && is_mul && n_kh%2==0) {
                constexpr int num_warp = 2;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 128, block_size);
                    }
                }

            }else if (head_size == 128 && is_mul && n_qh%3==0 && n_kh == 1) {
                constexpr int num_warp = 4;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL_PIPELINE(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL_PIPELINE(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL_PIPELINE(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL_PIPELINE(false, false, num_warp, 128, block_size);
                    }
                }

            }else if (head_size == 128 && is_mul && n_qh%2==0 && n_kh == 1) {
                constexpr int num_warp = 3;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL_PIPELINE(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL_PIPELINE(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL_PIPELINE(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL_PIPELINE(false, false, num_warp, 128, block_size);
                    }
                }

            }else if (head_size == 128) {
                constexpr int num_warp = 1;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 2;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 128, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 128, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 128, block_size);
                    }
                }

            }else if(head_size == 64 && is_mul && n_kh % 4 == 0){
                constexpr int num_warp = 4;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 1;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 64, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 64, block_size);
                    }
                }
                
            }else if(head_size == 64 && is_mul && n_kh % 2 == 0){
                constexpr int num_warp = 2;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 1;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 64, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 64, block_size);
                    }
                }
            }else if(head_size == 64 && n_kh % 1 == 0){
                constexpr int num_warp = 1;
                constexpr int block_size = num_warp * 64;
                constexpr int vec_size = 1;
                dim3 grid(num_tokens);
                dim3 block(block_size);
                if(residual_q.has_value() && residual_k.has_value()){
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, true, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, true, num_warp, 64, block_size);
                    }
                }else{
                    if(is_interleaved){
                        LAUNCH_MROTARY_KERNEL(true, false, num_warp, 64, block_size);
                    }else{
                        LAUNCH_MROTARY_KERNEL(false, false, num_warp, 64, block_size);
                    }
                }
            }
        }
        

    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}} // namespace at::native