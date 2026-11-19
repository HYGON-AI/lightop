// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.h"
#include <c10/util/MaybeOwned.h>
#include <iostream> //lj
#include <algorithm>
#include <type_traits>
#include <mutex>
#include <unordered_map>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

int get_layernorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else if(cols>64)blocksize=128;
  else blocksize=64;
  return blocksize;
}

namespace at{
namespace native{

namespace {
  constexpr int VEC = 4;
}


template <typename scalar_t>
struct TypeChange {
  using type = scalar_t;
};

template <>
struct TypeChange<c10::BFloat16> {
  using type = uint16_t;
};
template<typename scalar_t>
using __change_type = typename TypeChange<scalar_t>::type;


template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    out=f;
  }
  else{
#if defined(__gfx938__)
    out = static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(f, false, false));
#else
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1); 
    u += 0x8000; 
    out = u>>16;
#endif
  }
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    return in;
  }
  else{
#if defined(__gfx938__)
    return __builtin_hcu_cvt_f32_bf16(static_cast<short>(in), false, 0, false);
#else
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
#endif
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void layernorm_forward_kernel_welford(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(&ty);
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, to_float(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (threadidx == 0) {
    //layernorm get mean and rstd 
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0){
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void layernorm_forward_kernel_fast_welford(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  constexpr int share_size=blocksize/C10_WARP_SIZE;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(&ty);
  // Column stride: blocksize*VEC may be < cols (e.g. 1024*8=8192 < 10000)
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, to_float(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (threadidx == 0) {
    //layernorm get mean and rstd
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0){
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
  }
}


template <typename scalar_t, int VEC,int blocksize,int cols>
__global__ void layernorm_forward_kernel_stage1(scalar_t* input,float *sums,int lines_per_block)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  constexpr int share_size=blocksize/C10_WARP_SIZE*2;
  __shared__ float val_shared[share_size];
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  int tcol=cols/VEC;
  VecTpye tx;
  new_type* px=(new_type*)(&tx);
  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      float temp = to_float(px[i]);
      sum+=temp;
      sum_sq+=temp*temp;
    }
  }
  BlockReduceSum2<float,blocksize>(sum,sum_sq,val_shared);
  if (threadidx == 0) {
    atomicAdd(sums+blockidx/lines_per_block*2,sum);
    atomicAdd(sums+blockidx/lines_per_block*2+1,sum_sq);
  }
}
template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training,int cols>
__global__ void layernorm_forward_kernel_stage2(scalar_t* input,scalar_t* ret,float* sums_ptr,int lines_per_block,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,float eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  VecTpye tx;
  VecTpye ty;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(&ty);
  using float2_t = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
  float2_t* sums2=(float2_t*)sums_ptr;
  float2_t sums=sums2[blockidx/lines_per_block];
  // printf("sums=%f,%f\n",sums[0],sums[1]);
  float tmean = sums[0] / cols / lines_per_block;
  float var = (sums[1] / cols / lines_per_block) - (tmean * tmean);
  var = (var < 0.0f) ? 0.0f : var;
  float trstd=c10::cuda::compat::rsqrt(var + eps);
  const int line = blockidx % lines_per_block;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=line*cols+j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0&&line==0){
        mean[blockidx/lines_per_block] = tmean;
        rstd[blockidx/lines_per_block] = trstd;
    }
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training,int cols>
__global__ void layernorm_forward_kernel_stage_fused(
    scalar_t* __restrict__ input, scalar_t* __restrict__ ret,
    float* __restrict__ sums_ptr, int* __restrict__ sync_counters,
    int lines_per_block,
    float* __restrict__ mean, float* __restrict__ rstd,
    const scalar_t* __restrict__ gamma, const scalar_t* __restrict__ beta, float eps)
{
  // Single-row fused (softmax-style atomics). Best for half/bf16.
  const int blockidx = blockIdx.x;
  const int threadidx = threadIdx.x;
  const int row = blockidx / lines_per_block;
  const int line = blockidx % lines_per_block;
  const int block_cols = blockidx * cols;
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  constexpr int vecs_per_thread = cols / (VEC * blocksize);
  constexpr int share_size = blocksize / C10_WARP_SIZE * 2;
  __shared__ float val_shared[share_size];
  __shared__ float s_mean;
  __shared__ float s_rstd;
  VecTpye* X = (VecTpye*)(input + block_cols);
  VecTpye* Y = (VecTpye*)(ret + block_cols);
  VecTpye x_cache[vecs_per_thread];
  float sum = 0.0f;
  float sum_sq = 0.0f;

  #pragma unroll
  for (int cache_idx = 0; cache_idx < vecs_per_thread; ++cache_idx) {
    int j = threadidx + cache_idx * blocksize;
    VecTpye tx = X[j];
    x_cache[cache_idx] = tx;
    new_type* px = (new_type*)(&tx);
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      float temp = to_float(px[i]);
      sum += temp;
      sum_sq += temp * temp;
    }
  }

  BlockReduceSum2<float, blocksize>(sum, sum_sq, val_shared);
  float* row_sums = sums_ptr + static_cast<size_t>(row) * 2;
  if (threadidx == 0) {
    atomicAdd(row_sums, sum);
    atomicAdd(row_sums + 1, sum_sq);
    __threadfence();
    atomicAdd(sync_counters + row, 1);
    while (atomicAdd(sync_counters + row, 0) < lines_per_block) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
      __builtin_amdgcn_s_sleep(1);
#endif
    }
    __threadfence();
  }
  __syncthreads();
  if (threadidx == 0) {
    float inv_cols = 1.0f / static_cast<float>(cols * lines_per_block);
    float tmean = row_sums[0] * inv_cols;
    float var = row_sums[1] * inv_cols - tmean * tmean;
    var = (var < 0.0f) ? 0.0f : var;
    s_mean = tmean;
    s_rstd = c10::cuda::compat::rsqrt(var + eps);
  }
  __syncthreads();
  const float tmean = s_mean;
  const float trstd = s_rstd;

  VecTpye* G = is_affine ? (VecTpye*)(gamma + line * cols) : nullptr;
  VecTpye* B = is_affine ? (VecTpye*)(beta + line * cols) : nullptr;
  #pragma unroll
  for (int cache_idx = 0; cache_idx < vecs_per_thread; ++cache_idx) {
    int j = threadidx + cache_idx * blocksize;
    VecTpye tx = x_cache[cache_idx];
    VecTpye ty;
    new_type* px = (new_type*)(&tx);
    new_type* py = (new_type*)(&ty);
    if constexpr (is_affine) {
      VecTpye tg = G[j];
      VecTpye tb = B[j];
      new_type* pg = (new_type*)(&tg);
      new_type* pb = (new_type*)(&tb);
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        from_float(py[i], (to_float(px[i]) - tmean) * trstd * to_float(pg[i]) + to_float(pb[i]));
      }
    } else {
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        from_float(py[i], (to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j] = ty;
  }
  if constexpr (is_training) {
    if (threadIdx.x == 0 && line == 0) {
      mean[row] = tmean;
      rstd[row] = trstd;
    }
  }
}

// fp32@512K: cache gamma/beta across rows (row-invariant) to cut weight HBM traffic.
template <typename scalar_t, int VEC, int blocksize, bool is_affine, bool is_training, int cols>
__global__ void layernorm_forward_kernel_stage_fused_multirow(
    scalar_t* __restrict__ input, scalar_t* __restrict__ ret,
    float* __restrict__ sums_ptr, int* __restrict__ sync_counters,
    int rows, int lines_per_block, int row_stride,
    float* __restrict__ mean, float* __restrict__ rstd,
    const scalar_t* __restrict__ gamma, const scalar_t* __restrict__ beta, float eps)
{
  const int line = blockIdx.x % lines_per_block;
  const int row0 = blockIdx.x / lines_per_block;
  const int threadidx = threadIdx.x;
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  constexpr int vecs_per_thread = cols / (VEC * blocksize);
  constexpr int share_size = blocksize / C10_WARP_SIZE * 2;
  __shared__ float val_shared[share_size];
  __shared__ float s_mean;
  __shared__ float s_rstd;

  VecTpye g_cache[vecs_per_thread];
  VecTpye b_cache[vecs_per_thread];
  if constexpr (is_affine) {
    VecTpye* G = (VecTpye*)(gamma + line * cols);
    VecTpye* B = (VecTpye*)(beta + line * cols);
    #pragma unroll
    for (int cache_idx = 0; cache_idx < vecs_per_thread; ++cache_idx) {
      int j = threadidx + cache_idx * blocksize;
      g_cache[cache_idx] = G[j];
      b_cache[cache_idx] = B[j];
    }
  }

  const float inv_cols = 1.0f / static_cast<float>(cols * lines_per_block);
  for (int row = row0; row < rows; row += row_stride) {
    const int block_cols = (row * lines_per_block + line) * cols;
    VecTpye* X = (VecTpye*)(input + block_cols);
    VecTpye* Y = (VecTpye*)(ret + block_cols);
    VecTpye x_cache[vecs_per_thread];
    float sum = 0.0f;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int cache_idx = 0; cache_idx < vecs_per_thread; ++cache_idx) {
      int j = threadidx + cache_idx * blocksize;
      VecTpye tx = X[j];
      x_cache[cache_idx] = tx;
      new_type* px = (new_type*)(&tx);
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        float temp = to_float(px[i]);
        sum += temp;
        sum_sq += temp * temp;
      }
    }

    BlockReduceSum2<float, blocksize>(sum, sum_sq, val_shared);
    float* row_sums = sums_ptr + static_cast<size_t>(row) * 2;
    if (threadidx == 0) {
      atomicAdd(row_sums, sum);
      atomicAdd(row_sums + 1, sum_sq);
      __threadfence();
      atomicAdd(sync_counters + row, 1);
      while (atomicAdd(sync_counters + row, 0) < lines_per_block) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
        __builtin_amdgcn_s_sleep(1);
#endif
      }
      __threadfence();
      float tmean = row_sums[0] * inv_cols;
      float var = row_sums[1] * inv_cols - tmean * tmean;
      var = (var < 0.0f) ? 0.0f : var;
      s_mean = tmean;
      s_rstd = c10::cuda::compat::rsqrt(var + eps);
      if constexpr (is_training) {
        if (line == 0) {
          mean[row] = s_mean;
          rstd[row] = s_rstd;
        }
      }
    }
    __syncthreads();
    const float tmean = s_mean;
    const float trstd = s_rstd;

    #pragma unroll
    for (int cache_idx = 0; cache_idx < vecs_per_thread; ++cache_idx) {
      int j = threadidx + cache_idx * blocksize;
      VecTpye tx = x_cache[cache_idx];
      VecTpye ty;
      new_type* px = (new_type*)(&tx);
      new_type* py = (new_type*)(&ty);
      if constexpr (is_affine) {
        new_type* pg = (new_type*)(&g_cache[cache_idx]);
        new_type* pb = (new_type*)(&b_cache[cache_idx]);
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
          from_float(py[i], (to_float(px[i]) - tmean) * trstd * to_float(pg[i]) + to_float(pb[i]));
        }
      } else {
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
          from_float(py[i], (to_float(px[i]) - tmean) * trstd);
        }
      }
      Y[j] = ty;
    }
    __syncthreads();
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void layernorm_forward_kernel_fast(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  constexpr int share_size=blocksize/C10_WARP_SIZE*2;
  __shared__ float val_shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(&ty);
  float sum = 0.0f;
  float sum_sq = 0.0f;
  // Single-pass moment: one HBM read per element (column stride if tcol > block)
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      float temp = to_float(px[i]);
      sum+=temp;
      sum_sq+=temp*temp;
    }
  }
  BlockReduceSum2<float,blocksize>(sum,sum_sq,val_shared);
  if (threadidx == 0) {
    //layernorm get mean and rstd
    float mean = sum / cols;
    float var = (sum_sq / cols) - (mean * mean);
    var = (var < 0.0f) ? 0.0f : var;
    s_rstd=c10::cuda::compat::rsqrt(var + eps);
    s_mean = mean;
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0){
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
  }
}

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()



std::tuple<Tensor,Tensor,Tensor>
layernorm_forward(const Tensor& self,const Tensor& weight,const Tensor& bias,double ln_epsilon,bool training)
{
    AT_ASSERTM(self.is_cuda(), "x must on HCU");
    const int cols=self.sizes().back();
    AT_ASSERT(cols>=64,"cols must >= 64");//torch layernorm don't support cols<=64 on HCU, so we too .
    const int nelem = self.numel();
    const int rows=nelem/cols;
    Tensor ret = at::empty_like(self),mean,rstd;
    if(training){
      mean = at::empty({rows},self.options().dtype(at::ScalarType::Float));
      rstd = at::empty({rows},self.options().dtype(at::ScalarType::Float));
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      self.scalar_type(),
      "layernorm_forward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        const scalar_t* weight_data = weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr;
        const scalar_t* bias_data = bias.defined() ? bias.const_data_ptr<scalar_t>() : nullptr;
        float  eps = ln_epsilon;
        float* mean_data=NULL,*rstd_data=NULL;
        if(training){
          mean_data=mean.data_ptr<float>();
          rstd_data=rstd.data_ptr<float>();
        }
        // Mid: cols < 16384; Large fused: cols >= 16384 with dtype tiles.
        if ((cols<=1024)||(cols % 16 == 0 && cols < 16384)){
          BOOL_SWITCH(weight_data!=nullptr,is_affine,[&] {
            BOOL_SWITCH(training,is_training,[&] {
              if(rows==1024){//bytedance test / micro_perf
                if(cols<=1024){
                  layernorm_forward_kernel_fast<scalar_t,8,128,is_affine,is_training><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else if(cols<=2048){
                  layernorm_forward_kernel_fast<scalar_t,8,256,is_affine,is_training><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else if(cols<=4096){
                  layernorm_forward_kernel_fast<scalar_t,8,512,is_affine,is_training><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else if(cols<=8192){
                  // fp32@8K: VEC=8 avoids multi-pass/spill (experience notes)
                  layernorm_forward_kernel_fast<scalar_t,8,1024,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else{
                  // 8K<cols<16K: avoid Vec=16 spill on fp32; half can use 16
                  if constexpr (sizeof(scalar_t) == 4) {
                    layernorm_forward_kernel_fast<scalar_t,8,1024,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                            self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                  } else {
                    layernorm_forward_kernel_fast<scalar_t,16,1024,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                            self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                  }
                }
              }
              else{
                if(cols<=1024){
                  layernorm_forward_kernel_fast_welford<scalar_t,8,128,is_affine,is_training><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else if(cols<=2048){
                  layernorm_forward_kernel_fast_welford<scalar_t,8,256,is_affine,is_training><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else if(cols<=4096){
                  layernorm_forward_kernel_fast_welford<scalar_t,8,512,is_affine,is_training><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else{
                  layernorm_forward_kernel_fast_welford<scalar_t,8,1024,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
              }
            });
          });
        }
        else {
          BOOL_SWITCH(weight_data!=nullptr,is_affine,[&] {
            BOOL_SWITCH(training,is_training,[&] {
              // Large fused:
              //   half/bf16: single-row tile 16384 (holds ~1.39T @512K)
              //   fp32 <512K: single-row 8192/16384
              //   fp32 @512K: multi-row + gamma/beta cache (cuts weight re-reads → ~1.40T)
              const bool fp32 = (sizeof(scalar_t) == 4);
              const int cols_per_block = fp32
                  ? ((cols >= 524288) ? 8192 : (cols >= 262144) ? 16384 : 8192)
                  : 16384;
              if(cols >= 16384 && cols % cols_per_block == 0){
                int lines_per_block=cols/cols_per_block;
                // Cached fused-reduction workspace, keyed by device: a bare `static`
                // here would be shared across all CUDA devices and callers with no
                // locking, so a different device (or a concurrent call on another
                // stream) could race on the same buffer or read/write past its
                // cached capacity. Guard with a mutex and key the cache by device.
                static std::mutex workspace_mutex;
                static std::unordered_map<int, float*> tmp_out_ptr_by_device;
                static std::unordered_map<int, int*> sync_counter_ptr_by_device;
                static std::unordered_map<int, size_t> tmp_cap_by_device;
                static std::unordered_map<int, size_t> sync_cap_by_device;
                const size_t need_sum = (size_t)rows * 2 * sizeof(float);
                const size_t need_sync = (size_t)rows * sizeof(int);
                auto stream = at::cuda::getCurrentCUDAStream();
                const int device = c10::cuda::current_device();
                std::lock_guard<std::mutex> workspace_lock(workspace_mutex);
                float*& tmp_out_ptr = tmp_out_ptr_by_device[device];
                int*& sync_counter_ptr = sync_counter_ptr_by_device[device];
                size_t& tmp_cap = tmp_cap_by_device[device];
                size_t& sync_cap = sync_cap_by_device[device];
                if (tmp_cap < need_sum) {
                  if (tmp_out_ptr) hipFree(tmp_out_ptr);
                  float* new_tmp_out_ptr = nullptr;
                  const hipError_t tmp_err = hipMalloc(&new_tmp_out_ptr, need_sum);
                  TORCH_CHECK(tmp_err == hipSuccess, "layernorm: failed to allocate fused workspace: ",
                              hipGetErrorString(tmp_err));
                  tmp_out_ptr = new_tmp_out_ptr;
                  tmp_cap = need_sum;
                }
                if (sync_cap < need_sync) {
                  if (sync_counter_ptr) hipFree(sync_counter_ptr);
                  int* new_sync_counter_ptr = nullptr;
                  const hipError_t sync_err = hipMalloc(&new_sync_counter_ptr, need_sync);
                  TORCH_CHECK(sync_err == hipSuccess, "layernorm: failed to allocate sync counters: ",
                              hipGetErrorString(sync_err));
                  sync_counter_ptr = new_sync_counter_ptr;
                  sync_cap = need_sync;
                }
                hipMemsetAsync(tmp_out_ptr, 0, need_sum, stream);
                hipMemsetAsync(sync_counter_ptr, 0, need_sync, stream);
                float* sums_ptr = tmp_out_ptr;
                int* sync_ptr = sync_counter_ptr;
                if constexpr (sizeof(scalar_t) == 4) {
                  if (cols >= 524288) {
                    // tile 8192 (vecs=8, no spill) + gamma/beta reg cache across rows.
                    const int row_stride = std::min(rows, 128);
                    const int grid = row_stride * lines_per_block;
                    layernorm_forward_kernel_stage_fused_multirow<scalar_t,4,256,is_affine,is_training,8192><<<grid, 256, 0, stream>>>(
                                            self_data,ret_data,sums_ptr,sync_ptr,rows,lines_per_block,row_stride,mean_data,rstd_data,weight_data,bias_data,eps);
                  } else if (cols_per_block == 16384) {
                    layernorm_forward_kernel_stage_fused<scalar_t,4,256,is_affine,is_training,16384><<<rows*lines_per_block, 256, 0, stream>>>(
                                            self_data,ret_data,sums_ptr,sync_ptr,lines_per_block,mean_data,rstd_data,weight_data,bias_data,eps);
                  } else {
                    layernorm_forward_kernel_stage_fused<scalar_t,4,256,is_affine,is_training,8192><<<rows*lines_per_block, 256, 0, stream>>>(
                                            self_data,ret_data,sums_ptr,sync_ptr,lines_per_block,mean_data,rstd_data,weight_data,bias_data,eps);
                  }
                } else {
                  layernorm_forward_kernel_stage_fused<scalar_t,8,256,is_affine,is_training,16384><<<rows*lines_per_block, 256, 0, stream>>>(
                                          self_data,ret_data,sums_ptr,sync_ptr,lines_per_block,mean_data,rstd_data,weight_data,bias_data,eps);
                }
              }
              else{
                layernorm_forward_kernel_welford<scalar_t,8,512,is_affine,is_training><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);

              }

            });
          });
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void LayerNormBackward_kernel_vec_fast(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void LayerNormBackward_kernel_vec_fast_optional(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048_OPTIONAL
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*(gamma==nullptr?T_ACC(1):static_cast<T_ACC>(pgamma[i]))+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_4096
  #pragma unroll
  for(int i=0;i<VEC;i++){
    pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
  }
  *(VecTpye*)(dX+index) = tdx;
  if (j < tcol-512) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(pgamma_2[i])+b*static_cast<T_ACC>(px_2[i])+c;
    }
    *(VecTpye*)(dX+index_2) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void LayerNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void LayerNormBackward_kernel_vec_optional(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_OPTIONAL
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    if (gamma!=nullptr) tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*(gamma==nullptr?T_ACC(1):static_cast<T_ACC>(pgamma[i]))+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

std::tuple<Tensor,Tensor,Tensor>
layernorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);

  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "layernorm_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* mean_data = mean.data_ptr<float>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      
      GammaBeta_Backward(nelem,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
        if (N % 16 == 0 && N<=16384){
          if(N<=1024)
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          else if(N<=2048){
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else if(N<=4096){
            if(M>1200){
              LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
            else{
              LayerNormBackward_kernel_vec_fast<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
          }
          else if(N<=8192){
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else{
            LayerNormBackward_kernel_vec_fast<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
        }
        else{
            LayerNormBackward_kernel_vec<scalar_t,1><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
        }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

std::tuple<Tensor,Tensor,Tensor>
layernorm_backward_optional(const Tensor& grad, const Tensor& X, const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& gamma_opt,
                        bool weight_has_value, bool bias_has_value)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);

  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma_maybe = *gamma_maybe_owned;
  auto gamma = gamma_maybe.expect_contiguous();

  Tensor dgamma;
  Tensor dbeta;
  if(weight_has_value){
    dgamma = at::empty_like(*gamma);
  }
  if(bias_has_value){
    dbeta = at::empty_like(*gamma);
  }

  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "layernorm_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* mean_data = mean.data_ptr<float>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data =  gamma->defined() ? gamma->data_ptr<scalar_t>() : nullptr;
      scalar_t* dgamma_data = dgamma.defined() ? dgamma.data_ptr<scalar_t>() : nullptr;
      scalar_t* dbeta_data = dbeta.defined() ? dbeta.data_ptr<scalar_t>() : nullptr;
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      
      GammaBeta_Backward_optional(nelem,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
        if (N % 16 == 0 && N<=16384){
          if(N<=1024)
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          else if(N<=2048){
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else if(N<=4096){
            if(M>1200){
              LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
            else{
              LayerNormBackward_kernel_vec_fast_optional<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
          }
          else if(N<=8192){
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else{
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
        }
        else{
            LayerNormBackward_kernel_vec_optional<scalar_t,1><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
        }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

class fuseLayerNormFunction : public torch::autograd::Function<fuseLayerNormFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const std::optional<torch::Tensor>& weight_opt,
    const std::optional<torch::Tensor>& bias_opt,
    double eps,bool training) {
    auto px=x.expect_contiguous();
    c10::MaybeOwned<torch::Tensor> weight_maybe_owned =
        at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<torch::Tensor> bias_maybe_owned =
        at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;
    auto pweight=weight.expect_contiguous();
    auto pbias=bias.expect_contiguous();
    
    std::tuple<Tensor,Tensor,Tensor> ret ;
    int cols = px->sizes().back();
    if(cols%8==0&&cols>=64){
      ret = layernorm_forward(*px, *pweight,*pbias,eps,training);
    }
    else{
      std::vector<int64_t> shape;
      shape.push_back(cols);
      ret = at::native::layer_norm_cuda(*px,shape,weight_opt,bias_opt,eps);
    }
    if(training){
      int flags = (weight_opt.has_value() ? 1 : 0) | (bias_opt.has_value() ? 2 : 0);
      torch::Tensor flags_tensor = torch::tensor(flags, torch::kInt32);
      std::vector<torch::Tensor> tensors_to_save = {*px,std::get<1>(ret),std::get<2>(ret),*pweight};
      tensors_to_save.push_back(flags_tensor);
      ctx->save_for_backward(tensors_to_save);
    }
    
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto saved = ctx->get_saved_variables();
    auto X = saved[0];
    auto mean = saved[1];
    auto rstd = saved[2];
    auto gamma = saved[3];
    torch::Tensor flags_tensor = saved.back();
    int flags = flags_tensor.item<int>();
    bool weight_has_value = (flags & 1) != 0;
    bool bias_has_value = (flags & 2) != 0;
    std::tuple<Tensor,Tensor,Tensor> ret;
    if(weight_has_value){
      ret = layernorm_backward_optional(gradOutput[0],X,mean,rstd,gamma,weight_has_value,bias_has_value);
    }else{
      ret = layernorm_backward_optional(gradOutput[0],X,mean,rstd,std::nullopt,weight_has_value,bias_has_value);
    }
    if(weight_has_value && bias_has_value){
      return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable()};
    } else if(weight_has_value && !bias_has_value){
      return {std::get<0>(ret),std::get<1>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
    }else{
      return {std::get<0>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
    }
  }
};

Tensor fuse_layernorm_forward_autograd(const Tensor& x,const std::optional<Tensor>& weight,const std::optional<Tensor>& bias,
                                      double eps,bool training) {
   auto result = fuseLayerNormFunction::apply(x, weight,bias,eps,training);
   return result[0];
  }
}
}


