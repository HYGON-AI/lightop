// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Mixed-precision LayerNorm: activations Half/BFloat16, weight/bias float32 (standalone .cu).
#include "layernorm.h"
#include <c10/util/MaybeOwned.h>
#include <type_traits>

namespace at {
namespace native {
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

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void mixedln_fp32w_fwd_welford(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const float* gamma,const float* beta,int cols,float eps)
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
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * gamma[jj] + (beta != nullptr ? beta[jj] : 0.f));
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
__global__ void mixedln_fp32w_fwd_fast_welford(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const float* gamma,const float* beta,int cols,float eps)
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
  int j=threadidx;
  if (j < tcol) {
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
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * gamma[jj] + (beta != nullptr ? beta[jj] : 0.f));
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
__global__ void mixedln_fp32w_fwd_stage1(scalar_t* input,float *sums,int lines_per_block)
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
__global__ void mixedln_fp32w_fwd_stage2(scalar_t* input,scalar_t* ret,float* sums_ptr,int lines_per_block,float* mean,float* rstd,
                                            const float* gamma,const float* beta,float eps)
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
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * gamma[jj] + (beta != nullptr ? beta[jj] : 0.f));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0&&blockidx%lines_per_block==0){
        mean[blockidx/lines_per_block] = tmean;
        rstd[blockidx/lines_per_block] = trstd;
    }
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void mixedln_fp32w_fwd_fast(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const float* gamma,const float* beta,int cols,float eps)
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
  int j=threadidx;
  float sum = 0.0f;
  float sum_sq = 0.0f;
  if (j < tcol) {
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
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * gamma[jj] + (beta != nullptr ? beta[jj] : 0.f));
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

template <typename scalar_t>
static void mixedln_fp32w_forward_cuda_dispatch(
    int rows, int cols, bool training,
    scalar_t* self_data, scalar_t* ret_data,
    float* mean_data, float* rstd_data,
    const float* weight_data, const float* bias_data,
    float eps, float* tmp_out_ptr, size_t tmp_out_bytes) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if ((cols <= 1024) || (cols % 16 == 0 && cols <= 16384)) {
    BOOL_SWITCH(weight_data != nullptr, is_affine, [&] {
      BOOL_SWITCH(training, is_training, [&] {
        if (rows == 1024) {
          if (cols <= 1024) {
            mixedln_fp32w_fwd_fast<scalar_t, 8, 128, is_affine, is_training>
                <<<rows, 128, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 2048) {
            mixedln_fp32w_fwd_fast<scalar_t, 16, 128, is_affine, is_training>
                <<<rows, 128, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 4096) {
            mixedln_fp32w_fwd_fast<scalar_t, 16, 256, is_affine, is_training>
                <<<rows, 256, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 8192) {
            mixedln_fp32w_fwd_fast<scalar_t, 16, 512, is_affine, is_training>
                <<<rows, 1024, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else {
            mixedln_fp32w_fwd_fast<scalar_t, 16, 1024, is_affine, is_training>
                <<<rows, 1024, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          }
        } else {
          if (cols <= 1024) {
            mixedln_fp32w_fwd_fast_welford<scalar_t, 8, 128, is_affine, is_training>
                <<<rows, 128, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 2048) {
            mixedln_fp32w_fwd_fast_welford<scalar_t, 16, 128, is_affine, is_training>
                <<<rows, 128, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 4096) {
            mixedln_fp32w_fwd_fast_welford<scalar_t, 16, 256, is_affine, is_training>
                <<<rows, 256, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else if (cols <= 8192) {
            mixedln_fp32w_fwd_fast_welford<scalar_t, 16, 512, is_affine, is_training>
                <<<rows, 1024, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
          } else {
            mixedln_fp32w_fwd_fast_welford<scalar_t, 16, 1024, is_affine, is_training>
                <<<rows, 1024, 0, stream>>>(
                    self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps
                );
          }
        }
      });
    });
  } else {
    BOOL_SWITCH(weight_data != nullptr, is_affine, [&] {
      BOOL_SWITCH(training, is_training, [&] {
        if (cols % 16384 == 0) {
          constexpr int blocksize = 256;
          constexpr int cols_per_block = 4096;
          int lines_per_block = cols / cols_per_block;
          hipMemset(tmp_out_ptr, 0, tmp_out_bytes);
          mixedln_fp32w_fwd_stage1<scalar_t, 8, blocksize, cols_per_block>
              <<<rows * lines_per_block, blocksize, 0, stream>>>(self_data, tmp_out_ptr, lines_per_block);
          mixedln_fp32w_fwd_stage2<scalar_t, 8, blocksize, is_affine, is_training, cols_per_block>
              <<<rows * lines_per_block, blocksize, 0, stream>>>(
                  self_data, ret_data, tmp_out_ptr, lines_per_block, mean_data, rstd_data, weight_data,
                  bias_data, eps);
        } else {
          mixedln_fp32w_fwd_welford<scalar_t, 8, 512, is_affine, is_training>
              <<<rows, 512, 0, stream>>>(
                  self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
        }
      });
    });
  }
}


std::tuple<Tensor, Tensor, Tensor> mixed_layernorm_fp32_params_forward(
    const Tensor& self, const Tensor& weight, const Tensor& bias, double ln_epsilon, bool training) {
  AT_ASSERTM(self.is_cuda(), "mixed_layernorm_fp32_params: input must be CUDA");
  AT_ASSERTM(
      self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::BFloat16,
      "mixed_layernorm_fp32_params: input must be Half or BFloat16");
  if (weight.defined()) {
    AT_ASSERTM(weight.scalar_type() == at::ScalarType::Float, "mixed_layernorm_fp32_params: weight must be float32");
  }
  if (bias.defined()) {
    AT_ASSERTM(weight.defined(), "mixed_layernorm_fp32_params: bias requires weight");
    AT_ASSERTM(bias.scalar_type() == at::ScalarType::Float, "mixed_layernorm_fp32_params: bias must be float32");
  }
  const int cols = self.sizes().back();
  AT_ASSERT(cols >= 64, "cols must >= 64");
  const int nelem = self.numel();
  const int rows = nelem / cols;
  Tensor ret = at::empty_like(self), mean, rstd;
  if (training) {
    mean = at::empty({rows}, self.options().dtype(at::ScalarType::Float));
    rstd = at::empty({rows}, self.options().dtype(at::ScalarType::Float));
  }
  static float* tmp_out_ptr = nullptr;
  constexpr int temp_out_size = 100000;
  if (tmp_out_ptr == nullptr) {
    hipMalloc(&tmp_out_ptr, temp_out_size);
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mixed_layernorm_fp32_params_forward",
      [&] {
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data = ret.data_ptr<scalar_t>();
        float eps = static_cast<float>(ln_epsilon);
        float *mean_data = nullptr, *rstd_data = nullptr;
        if (training) {
          mean_data = mean.data_ptr<float>();
          rstd_data = rstd.data_ptr<float>();
        }
        const float* weight_data = weight.defined() ? weight.const_data_ptr<float>() : nullptr;
        const float* bias_data = bias.defined() ? bias.const_data_ptr<float>() : nullptr;
        mixedln_fp32w_forward_cuda_dispatch<scalar_t>(
            rows, cols, training, self_data, ret_data, mean_data, rstd_data, weight_data, bias_data, eps,
            tmp_out_ptr, temp_out_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::tuple<Tensor, Tensor, Tensor>(ret, mean, rstd);
}


template <typename scalar_t, int VEC>
__device__ inline void mixedln_fp32w_bwd_load_gamma(
    const float* gamma, int j, acc_type<scalar_t, true>* pgamma_acc) {
  using T_ACC = acc_type<scalar_t, true>;
  using GVec = memory::aligned_vector<float, VEC>;
  GVec gv = *(const GVec*)(gamma + j * VEC);
  const float* pg = reinterpret_cast<const float*>(&gv);
#pragma unroll
  for (int ii = 0; ii < VEC; ++ii) {
    pgamma_acc[ii] = static_cast<T_ACC>(pg[ii]);
  }
}

#define MIXEDLN_FP32W_BWD2048                                                           \
  using T_ACC = acc_type<scalar_t, true>;                                               \
  __shared__ T_ACC ds_shared[8];                                                        \
  __shared__ T_ACC db_shared[8];                                                        \
  __shared__ T_ACC b;                                                                   \
  __shared__ T_ACC c;                                                                   \
  const int i = blockIdx.x;                                                             \
  const int j = threadIdx.x;                                                            \
  const int index = i * N + j * VEC;                                                    \
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;                                  \
  T_ACC sum1 = 0;                                                                       \
  T_ACC sum2 = 0;                                                                       \
  int tcol = N / VEC;                                                                   \
  VecTpye tx, tdy, tdx;                                                                 \
  T_ACC pgamma_acc[VEC];                                                                \
  scalar_t* px = (scalar_t*)(&tx);                                                      \
  scalar_t* pdy = (scalar_t*)(&tdy);                                                    \
  scalar_t* pdx = (scalar_t*)(&tdx);                                                    \
  if (j < tcol) {                                                                       \
    tx = *(VecTpye*)(X + index);                                                        \
    tdy = *(VecTpye*)(dY + index);                                                      \
    mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, j, pgamma_acc);                  \
    for (int ii = 0; ii < VEC; ii++) {                                                  \
      sum1 += static_cast<T_ACC>(pdy[ii]) * static_cast<T_ACC>(px[ii]) * pgamma_acc[ii]; \
      sum2 += static_cast<T_ACC>(pdy[ii]) * pgamma_acc[ii];                           \
    }                                                                                   \
  }                                                                                     \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                        \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                        \
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                     \
  T_ACC trstd = rstd[i];                                                                \
  if (threadIdx.x == 0) {                                                               \
    T_ACC tmean = mean[i];                                                              \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                 \
    b = a;                                                                              \
    c = -a * tmean - sum2 * trstd * s;                                                  \
  }                                                                                     \
  __syncthreads();

#define MIXEDLN_FP32W_BWD2048_OPT                                                      \
  using T_ACC = acc_type<scalar_t, true>;                                               \
  __shared__ T_ACC ds_shared[8];                                                        \
  __shared__ T_ACC db_shared[8];                                                        \
  __shared__ T_ACC b;                                                                   \
  __shared__ T_ACC c;                                                                   \
  const int i = blockIdx.x;                                                             \
  const int j = threadIdx.x;                                                            \
  const int index = i * N + j * VEC;                                                    \
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;                                  \
  T_ACC sum1 = 0;                                                                       \
  T_ACC sum2 = 0;                                                                       \
  int tcol = N / VEC;                                                                   \
  VecTpye tx, tdy, tdx;                                                                 \
  T_ACC pgamma_acc[VEC];                                                                \
  scalar_t* px = (scalar_t*)(&tx);                                                      \
  scalar_t* pdy = (scalar_t*)(&tdy);                                                    \
  scalar_t* pdx = (scalar_t*)(&tdx);                                                    \
  if (j < tcol) {                                                                       \
    tx = *(VecTpye*)(X + index);                                                        \
    tdy = *(VecTpye*)(dY + index);                                                      \
    if (gamma != nullptr)                                                               \
      mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, j, pgamma_acc);                \
    for (int ii = 0; ii < VEC; ii++) {                                                  \
      sum1 += static_cast<T_ACC>(pdy[ii]) * static_cast<T_ACC>(px[ii]) *                  \
              (gamma == nullptr ? T_ACC(1) : pgamma_acc[ii]);                           \
      sum2 += static_cast<T_ACC>(pdy[ii]) * (gamma == nullptr ? T_ACC(1) : pgamma_acc[ii]); \
    }                                                                                   \
  }                                                                                     \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                        \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                        \
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                     \
  T_ACC trstd = rstd[i];                                                                \
  if (threadIdx.x == 0) {                                                               \
    T_ACC tmean = mean[i];                                                              \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                 \
    b = a;                                                                              \
    c = -a * tmean - sum2 * trstd * s;                                                  \
  }                                                                                     \
  __syncthreads();

#define MIXEDLN_FP32W_BWD4096                                                          \
  using T_ACC = acc_type<scalar_t, true>;                                              \
  __shared__ T_ACC ds_shared[8];                                                       \
  __shared__ T_ACC db_shared[8];                                                       \
  __shared__ T_ACC b;                                                                  \
  __shared__ T_ACC c;                                                                  \
  const int i = blockIdx.x;                                                            \
  const int j = threadIdx.x;                                                           \
  const int index = i * N + j * VEC;                                                   \
  const int index_2 = i * N + j * VEC + 2048;                                          \
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;                               \
  T_ACC sum1 = 0;                                                                      \
  T_ACC sum2 = 0;                                                                      \
  int tcol = N / VEC;                                                                  \
  VecTpye tx, tdy, tdx, tx_2, tdy_2;                                                   \
  T_ACC pgamma_acc[VEC], pgamma_acc_2[VEC];                                            \
  scalar_t* px = (scalar_t*)(&tx);                                                     \
  scalar_t* pdy = (scalar_t*)(&tdy);                                                   \
  scalar_t* pdx = (scalar_t*)(&tdx);                                                    \
  scalar_t* px_2 = (scalar_t*)(&tx_2);                                                 \
  scalar_t* pdy_2 = (scalar_t*)(&tdy_2);                                               \
  tx = *(VecTpye*)(X + index);                                                         \
  tdy = *(VecTpye*)(dY + index);                                                       \
  mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, j, pgamma_acc);                   \
  for (int ii = 0; ii < VEC; ii++) {                                                   \
    sum1 += static_cast<T_ACC>(pdy[ii]) * static_cast<T_ACC>(px[ii]) * pgamma_acc[ii];   \
    sum2 += static_cast<T_ACC>(pdy[ii]) * pgamma_acc[ii];                              \
  }                                                                                    \
  if (j < tcol - 512) {                                                                \
    tx_2 = *(VecTpye*)(X + index_2);                                                   \
    tdy_2 = *(VecTpye*)(dY + index_2);                                                 \
    mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, j + (2048 / VEC), pgamma_acc_2);  \
    for (int ii = 0; ii < VEC; ii++) {                                                 \
      sum1 += static_cast<T_ACC>(pdy_2[ii]) * static_cast<T_ACC>(px_2[ii]) * pgamma_acc_2[ii]; \
      sum2 += static_cast<T_ACC>(pdy_2[ii]) * pgamma_acc_2[ii];                        \
    }                                                                                  \
  }                                                                                    \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                       \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                       \
  T_ACC trstd = rstd[i];                                                               \
  if (threadIdx.x == 0) {                                                             \
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                  \
    T_ACC tmean = mean[i];                                                             \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                 \
    b = a;                                                                             \
    c = -a * tmean - sum2 * trstd * s;                                                 \
  }                                                                                    \
  __syncthreads();

#define MIXEDLN_FP32W_BWDVEC                                                           \
  using T_ACC = acc_type<scalar_t, true>;                                              \
  __shared__ T_ACC ds_shared[8];                                                        \
  __shared__ T_ACC db_shared[8];                                                        \
  __shared__ T_ACC b;                                                                   \
  __shared__ T_ACC c;                                                                   \
  const int i = blockIdx.x;                                                             \
  const int j = threadIdx.x;                                                            \
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;                                \
  T_ACC sum1 = 0;                                                                       \
  T_ACC sum2 = 0;                                                                       \
  int tcol = N / VEC;                                                                   \
  VecTpye tx, tdy, tdx;                                                                 \
  T_ACC pgamma_acc[VEC];                                                                \
  scalar_t* px = (scalar_t*)(&tx);                                                      \
  scalar_t* pdy = (scalar_t*)(&tdy);                                                    \
  scalar_t* pdx = (scalar_t*)(&tdx);                                                    \
  for (int off = j; off < tcol; off += blockDim.x) {                                    \
    const int index = i * N + off * VEC;                                                \
    tx = *(VecTpye*)(X + index);                                                        \
    tdy = *(VecTpye*)(dY + index);                                                      \
    mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, off, pgamma_acc);                 \
    for (int ii = 0; ii < VEC; ii++) {                                                  \
      sum1 += static_cast<T_ACC>(pdy[ii]) * static_cast<T_ACC>(px[ii]) * pgamma_acc[ii];  \
      sum2 += static_cast<T_ACC>(pdy[ii]) * pgamma_acc[ii];                             \
    }                                                                                   \
  }                                                                                     \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                        \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                        \
  T_ACC trstd = rstd[i];                                                                \
  if (threadIdx.x == 0) {                                                               \
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                   \
    T_ACC tmean = mean[i];                                                               \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                   \
    b = a;                                                                               \
    c = -a * tmean - sum2 * trstd * s;                                                   \
  }                                                                                       \
  __syncthreads();

#define MIXEDLN_FP32W_BWDVEC_OPT                                                       \
  using T_ACC = acc_type<scalar_t, true>;                                               \
  __shared__ T_ACC ds_shared[8];                                                        \
  __shared__ T_ACC db_shared[8];                                                        \
  __shared__ T_ACC b;                                                                   \
  __shared__ T_ACC c;                                                                   \
  const int i = blockIdx.x;                                                             \
  const int j = threadIdx.x;                                                            \
  using VecTpye = memory::aligned_vector<scalar_t, VEC>;                                \
  T_ACC sum1 = 0;                                                                       \
  T_ACC sum2 = 0;                                                                       \
  int tcol = N / VEC;                                                                   \
  VecTpye tx, tdy, tdx;                                                                 \
  T_ACC pgamma_acc[VEC];                                                                \
  scalar_t* px = (scalar_t*)(&tx);                                                      \
  scalar_t* pdy = (scalar_t*)(&tdy);                                                    \
  scalar_t* pdx = (scalar_t*)(&tdx);                                                    \
  for (int off = j; off < tcol; off += blockDim.x) {                                    \
    const int index = i * N + off * VEC;                                                \
    tx = *(VecTpye*)(X + index);                                                        \
    tdy = *(VecTpye*)(dY + index);                                                      \
    if (gamma != nullptr)                                                              \
      mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, off, pgamma_acc);              \
    for (int ii = 0; ii < VEC; ii++) {                                                  \
      sum1 += static_cast<T_ACC>(pdy[ii]) * static_cast<T_ACC>(px[ii]) *                  \
              (gamma == nullptr ? T_ACC(1) : pgamma_acc[ii]);                          \
      sum2 += static_cast<T_ACC>(pdy[ii]) * (gamma == nullptr ? T_ACC(1) : pgamma_acc[ii]); \
    }                                                                                   \
  }                                                                                     \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                        \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                        \
  T_ACC trstd = rstd[i];                                                                \
  if (threadIdx.x == 0) {                                                               \
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                   \
    T_ACC tmean = mean[i];                                                               \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                   \
    b = a;                                                                               \
    c = -a * tmean - sum2 * trstd * s;                                                   \
  }                                                                                       \
  __syncthreads();

template <typename scalar_t, int VEC>
__global__ void MixedLnFp32W_Backward_vec_fast(
    int N, const scalar_t* dY, const scalar_t* X, const float* gamma, const float* mean, const float* rstd,
    scalar_t* dX) {
  MIXEDLN_FP32W_BWD2048
  if (j < tcol) {
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      pdx[i] = trstd * static_cast<T_ACC>(pdy[i]) * pgamma_acc[i] + b * static_cast<T_ACC>(px[i]) + c;
    }
    *(VecTpye*)(dX + index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void MixedLnFp32W_Backward_vec_fast_opt(
    int N, const scalar_t* dY, const scalar_t* X, const float* gamma, const float* mean, const float* rstd,
    scalar_t* dX) {
  MIXEDLN_FP32W_BWD2048_OPT
  if (j < tcol) {
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      pdx[i] = trstd * static_cast<T_ACC>(pdy[i]) * (gamma == nullptr ? T_ACC(1) : pgamma_acc[i]) +
               b * static_cast<T_ACC>(px[i]) + c;
    }
    *(VecTpye*)(dX + index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void MixedLnFp32W_Backward_vec4096(
    int N, const scalar_t* dY, const scalar_t* X, const float* gamma, const float* mean, const float* rstd,
    scalar_t* dX) {
  MIXEDLN_FP32W_BWD4096
#pragma unroll
  for (int i = 0; i < VEC; i++) {
    pdx[i] = trstd * static_cast<T_ACC>(pdy[i]) * pgamma_acc[i] + b * static_cast<T_ACC>(px[i]) + c;
  }
  *(VecTpye*)(dX + index) = tdx;
  if (j < tcol - 512) {
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      pdx[i] =
          trstd * static_cast<T_ACC>(pdy_2[i]) * pgamma_acc_2[i] + b * static_cast<T_ACC>(px_2[i]) + c;
    }
    *(VecTpye*)(dX + index_2) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void MixedLnFp32W_Backward_vec(
    int N, const scalar_t* dY, const scalar_t* X, const float* gamma, const float* mean, const float* rstd,
    scalar_t* dX) {
  MIXEDLN_FP32W_BWDVEC
  for (int off = j; off < tcol; off += blockDim.x) {
    const int index = i * N + off * VEC;
    tx = *(VecTpye*)(X + index);
    tdy = *(VecTpye*)(dY + index);
    mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, off, pgamma_acc);
#pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      pdx[ii] = trstd * static_cast<T_ACC>(pdy[ii]) * pgamma_acc[ii] + b * static_cast<T_ACC>(px[ii]) + c;
    }
    *(VecTpye*)(dX + index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void MixedLnFp32W_Backward_vec_opt(
    int N, const scalar_t* dY, const scalar_t* X, const float* gamma, const float* mean, const float* rstd,
    scalar_t* dX) {
  MIXEDLN_FP32W_BWDVEC_OPT
  for (int off = j; off < tcol; off += blockDim.x) {
    const int index = i * N + off * VEC;
    tx = *(VecTpye*)(X + index);
    tdy = *(VecTpye*)(dY + index);
    if (gamma != nullptr) mixedln_fp32w_bwd_load_gamma<scalar_t, VEC>(gamma, off, pgamma_acc);
#pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      pdx[ii] = trstd * static_cast<T_ACC>(pdy[ii]) * (gamma == nullptr ? T_ACC(1) : pgamma_acc[ii]) +
                b * static_cast<T_ACC>(px[ii]) + c;
    }
    *(VecTpye*)(dX + index) = tdx;
  }
}

template <typename scalar_t>
__global__ void mixedln_fp32w_GammaBetaBwd(
    int M, int N, const scalar_t* dY, const scalar_t* X, const float* mean, const float* rstd, float* dg,
    float* db) {
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      const int index = i * N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - mean[i]) * rstd[i];
      db_sum += cmask_dY;
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum;
  b_shared[threadIdx.y][threadIdx.x] = db_sum;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1, 16);
  sum2 = WarpReduceSum(sum2, 16);
  if (threadIdx.x == 0) {
    const int j2 = blockIdx.x * blockDim.x + threadIdx.y;
    if (j2 < N) {
      dg[j2] = static_cast<float>(sum1);
      db[j2] = static_cast<float>(sum2);
    }
  }
}

template <typename scalar_t>
__global__ void mixedln_fp32w_GammaBetaBwd_onlyW(
    int M, int N, const scalar_t* dY, const scalar_t* X, const float* mean, const float* rstd, float* dg) {
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      const int index = i * N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - mean[i]) * rstd[i];
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1, 16);
  if (threadIdx.x == 0) {
    const int j2 = blockIdx.x * blockDim.x + threadIdx.y;
    if (j2 < N) {
      dg[j2] = static_cast<float>(sum1);
    }
  }
}

template <typename scalar_t, typename T_ACC>
__global__ void mixedln_fp32w_gb_part_b(
    const T_ACC* part_grad_gamma, const T_ACC* part_grad_beta, float* dg, float* db) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC gamma = part_grad_gamma[idx];
  T_ACC beta = part_grad_beta[idx];
  gamma = WarpReduceSum(gamma);
  beta = WarpReduceSum(beta);
  if (threadIdx.x == 0) {
    dg[blockIdx.x] = static_cast<float>(gamma);
    db[blockIdx.x] = static_cast<float>(beta);
  }
}

template <typename scalar_t, typename T_ACC>
__global__ void mixedln_fp32w_gb_part_b_onlyw(const T_ACC* part_grad_gamma, float* dg) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC gamma = part_grad_gamma[idx];
  gamma = WarpReduceSum(gamma);
  if (threadIdx.x == 0) {
    dg[blockIdx.x] = static_cast<float>(gamma);
  }
}

template <typename scalar_t>
void mixedln_fp32w_gamma_beta_backward(
    int length, int N, scalar_t* dY, scalar_t* X, float* mean, float* rstd, float* dg, float* db) {
  using T_ACC = acc_type<scalar_t, true>;
  auto stream = at::cuda::getCurrentCUDAStream();
  int M = length / N;
  const int B = (N - 1) / kColwiseReduceTileSize + 1;
  if (N <= 512 && M >= 4096) {
    int part_size = 64;
    c10::TensorOptions options(DeviceType::CUDA);
    options.dtype(ScalarType::Float);
    Tensor part_grad_gamma = at::empty({N, part_size}, options);
    Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);
    T_ACC* pd_gamma = part_grad_gamma.data_ptr<T_ACC>();
    T_ACC* pd_beta = part_grad_beta.data_ptr<T_ACC>();
    gb_backward_kernel_part_a<scalar_t, T_ACC>
        <<<dim3(B, part_size), dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
            M, N, dY, X, mean, rstd, pd_gamma, pd_beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mixedln_fp32w_gb_part_b<scalar_t, T_ACC><<<N, part_size, 0, stream>>>(pd_gamma, pd_beta, dg, db);
  } else {
    mixedln_fp32w_GammaBetaBwd<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
        M, N, dY, X, mean, rstd, dg, db);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void mixedln_fp32w_gamma_beta_backward_optional(
    int length, int N, scalar_t* dY, scalar_t* X, float* mean, float* rstd, float* dg, float* db) {
  if (dg == nullptr && db == nullptr) return;
  using T_ACC = acc_type<scalar_t, true>;
  auto stream = at::cuda::getCurrentCUDAStream();
  int M = length / N;
  const int B = (N - 1) / kColwiseReduceTileSize + 1;
  if (N <= 512 && M >= 4096) {
    int part_size = 64;
    c10::TensorOptions options(DeviceType::CUDA);
    options.dtype(ScalarType::Float);
    Tensor part_grad_gamma = at::empty({N, part_size}, options);
    Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);
    T_ACC* pd_gamma = part_grad_gamma.data_ptr<T_ACC>();
    T_ACC* pd_beta = part_grad_beta.data_ptr<T_ACC>();
    if (db != nullptr) {
      gb_backward_kernel_part_a<scalar_t, T_ACC>
          <<<dim3(B, part_size), dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
              M, N, dY, X, mean, rstd, pd_gamma, pd_beta);
    } else {
      gb_backward_kernel_part_a_onlyweight<scalar_t, T_ACC>
          <<<dim3(B, part_size), dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
              M, N, dY, X, mean, rstd, pd_gamma);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (db != nullptr) {
      mixedln_fp32w_gb_part_b<scalar_t, T_ACC><<<N, part_size, 0, stream>>>(pd_gamma, pd_beta, dg, db);
    } else {
      mixedln_fp32w_gb_part_b_onlyw<scalar_t, T_ACC><<<N, part_size, 0, stream>>>(pd_gamma, dg);
    }
  } else {
    if (db != nullptr) {
      mixedln_fp32w_GammaBetaBwd<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
          M, N, dY, X, mean, rstd, dg, db);
    } else {
      mixedln_fp32w_GammaBetaBwd_onlyW<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0,
                                                   stream>>>(M, N, dY, X, mean, rstd, dg);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<Tensor, Tensor, Tensor> mixed_layernorm_fp32_params_backward(
    const Tensor& grad, const Tensor& X, const Tensor& mean, const Tensor& rstd, const Tensor& gamma) {
  int N = grad.sizes().back();
  int nelem = grad.numel();
  int M = nelem / N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  float* gamma_data = gamma.data_ptr<float>();
  float* dgamma_data = dgamma.data_ptr<float>();
  float* dbeta_data = dbeta.data_ptr<float>();
  auto cuda_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "mixed_layernorm_fp32_params_backward", [&] {
        scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
        scalar_t* X_data = X.data_ptr<scalar_t>();
        float* mean_data = mean.data_ptr<float>();
        float* rstd_data = rstd.data_ptr<float>();
        scalar_t* dX_data = dX.data_ptr<scalar_t>();
        mixedln_fp32w_gamma_beta_backward<scalar_t>(nelem, N, dY_data, X_data, mean_data, rstd_data, dgamma_data,
                                                   dbeta_data);
        if (N % 16 == 0 && N <= 16384) {
          if (N <= 1024)
            MixedLnFp32W_Backward_vec_fast<scalar_t, 8>
                <<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
          else if (N <= 2048)
            MixedLnFp32W_Backward_vec_fast<scalar_t, 8>
                <<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
          else if (N <= 4096) {
            if (M > 1200)
              MixedLnFp32W_Backward_vec_fast<scalar_t, 8>
                  <<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
            else
              MixedLnFp32W_Backward_vec_fast<scalar_t, 4>
                  <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
          } else if (N <= 8192)
            MixedLnFp32W_Backward_vec_fast<scalar_t, 8>
                <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
          else
            MixedLnFp32W_Backward_vec_fast<scalar_t, 16>
                <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
        } else {
          MixedLnFp32W_Backward_vec<scalar_t, 1>
              <<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data,
                                                                  rstd_data, dX_data);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {dX, dgamma, dbeta};
}

std::tuple<Tensor, Tensor, Tensor> mixed_layernorm_fp32_params_backward_optional(
    const Tensor& grad, const Tensor& X, const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& gamma,
    bool weight_has_value, bool bias_has_value) {
  int N = grad.sizes().back();
  int nelem = grad.numel();
  int M = nelem / N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  c10::MaybeOwned<Tensor> gamma_owned = at::borrow_from_optional_tensor(gamma);
  const Tensor& g0 = *gamma_owned;
  auto g = g0.expect_contiguous();
  Tensor dgamma, dbeta;
  if (weight_has_value) dgamma = at::empty_like(*g);
  if (bias_has_value) dbeta = at::empty_like(*g);
  auto cuda_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
      "mixed_layernorm_fp32_params_backward_opt", [&] {
        scalar_t* dY = pgrad->data_ptr<scalar_t>();
        scalar_t* Xd = X.data_ptr<scalar_t>();
        float* md = mean.data_ptr<float>();
        float* rd = rstd.data_ptr<float>();
        scalar_t* dXd = dX.data_ptr<scalar_t>();
        float* gd = g->defined() ? g->data_ptr<float>() : nullptr;
        float* dgd = dgamma.defined() ? dgamma.data_ptr<float>() : nullptr;
        float* dbd = dbeta.defined() ? dbeta.data_ptr<float>() : nullptr;
        mixedln_fp32w_gamma_beta_backward_optional<scalar_t>(nelem, N, dY, Xd, md, rd, dgd, dbd);
        if (N % 16 == 0 && N <= 16384) {
          if (N <= 1024)
            MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 8>
                <<<M, 128, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
          else if (N <= 2048)
            MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 8>
                <<<M, 256, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
          else if (N <= 4096) {
            if (M > 1200)
              MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 8>
                  <<<M, 512, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
            else
              MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 4>
                  <<<M, 1024, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
          } else if (N <= 8192)
            MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 8>
                <<<M, 1024, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
          else
            MixedLnFp32W_Backward_vec_fast_opt<scalar_t, 16>
                <<<M, 1024, 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
        } else {
          MixedLnFp32W_Backward_vec_opt<scalar_t, 1>
              <<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY, Xd, gd, md, rd, dXd);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {dX, dgamma, dbeta};
}

class fuseMixedLnFp32WFunction : public torch::autograd::Function<fuseMixedLnFp32WFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                const torch::autograd::Variable& x,
                                                const std::optional<torch::Tensor>& weight_opt,
                                                const std::optional<torch::Tensor>& bias_opt, double eps,
                                                bool training) {
    auto px = x.expect_contiguous();
    c10::MaybeOwned<Tensor> weight_owned = at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_owned;
    c10::MaybeOwned<Tensor> bias_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_owned;
    auto pweight = weight.expect_contiguous();
    auto pbias = bias.expect_contiguous();
    std::tuple<Tensor, Tensor, Tensor> ret;
    int cols = px->sizes().back();
    if (cols % 8 == 0 && cols >= 64) {
      ret = mixed_layernorm_fp32_params_forward(*px, *pweight, *pbias, eps, training);
    } else {
      std::vector<int64_t> shape = {cols};
      std::optional<Tensor> wt =
          weight_opt.has_value() ? std::optional<Tensor>(weight_opt->to(px->scalar_type())) : std::nullopt;
      std::optional<Tensor> bt =
          bias_opt.has_value() ? std::optional<Tensor>(bias_opt->to(px->scalar_type())) : std::nullopt;
      ret = at::native::layer_norm_cuda(*px, shape, wt, bt, eps);
    }
    if (training) {
      int flags = (weight_opt.has_value() ? 1 : 0) | (bias_opt.has_value() ? 2 : 0);
      Tensor flags_tensor = torch::tensor(flags, torch::kInt32);
      std::vector<Tensor> save = {*px, std::get<1>(ret), std::get<2>(ret), *pweight};
      save.push_back(flags_tensor);
      ctx->save_for_backward(save);
    }
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   const torch::autograd::variable_list& grad_out) {
    auto saved = ctx->get_saved_variables();
    auto X = saved[0];
    auto mean = saved[1];
    auto rstd = saved[2];
    auto weight = saved[3];
    int flags = saved.back().item<int>();
    bool has_w = (flags & 1) != 0;
    bool has_b = (flags & 2) != 0;
    std::tuple<Tensor, Tensor, Tensor> ret;
    if (has_w)
      ret = mixed_layernorm_fp32_params_backward_optional(grad_out[0], X, mean, rstd, weight, has_w, has_b);
    else
      ret =
          mixed_layernorm_fp32_params_backward_optional(grad_out[0], X, mean, rstd, std::nullopt, has_w, has_b);
    if (has_w && has_b)
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), torch::autograd::Variable{},
              torch::autograd::Variable{}};
    if (has_w && !has_b)
      return {std::get<0>(ret), std::get<1>(ret), torch::autograd::Variable{}, torch::autograd::Variable{},
              torch::autograd::Variable{}};
    return {std::get<0>(ret), torch::autograd::Variable{}, torch::autograd::Variable{},
            torch::autograd::Variable{}, torch::autograd::Variable{}};
  }
};

Tensor fuse_layernorm_mixed_fp32_params_autograd(const Tensor& x, const std::optional<Tensor>& weight,
                                                 const std::optional<Tensor>& bias, double eps, bool training) {
  return fuseMixedLnFp32WFunction::apply(x, weight, bias, eps, training)[0];
}

}
}
