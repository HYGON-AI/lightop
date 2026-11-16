// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.h"

namespace at{
namespace native{

namespace {
  constexpr int VEC = 4;
}

template <typename scalar_t, int VEC = 4, typename T_ACC>
// #if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
// C10_LAUNCH_BOUNDS_2(512, 4)
// #endif
__global__ void fused_rn_add_kernel_val_2048(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* gamma,int cols,scalar_t eps)
{
  __shared__ T_ACC val_shared[8];
  __shared__ T_ACC s_rstd;
  T_ACC val=0;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t residual4[VEC];
  T_ACC trstd;
  int idx = i * tcol + j;
  if(j<tcol){
    idx*=VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      input4[ii]+=residual4[ii];
      val += static_cast<T_ACC>(input4[ii])*static_cast<T_ACC>(input4[ii]);
    }
  }
  val = BlockReduceSum(val,val_shared);
  if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + static_cast<T_ACC>(eps));
  __syncthreads();
  if(j<tcol){
    trstd=s_rstd;
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii;
      output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
}

template <typename scalar_t, int VEC =4, typename T_ACC>
__global__ void fused_rn_add_kernel_val_2048_inplace(scalar_t* input, scalar_t* residual, scalar_t* gamma, int cols, scalar_t eps)
{
  __shared__ T_ACC val_shared[8];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0;
  int i = blockIdx.x;
  int j = threadIdx.x;
  int tcol = cols / VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  scalar_t input4[VEC];
  scalar_t residual4[VEC];
  T_ACC trstd;
  int idx = i * tcol + j;
  if (j < tcol) {
    idx *= VEC;
    *(LoadT*)input4 = *(LoadT*)(input + idx);
    *(LoadT*)residual4 = *(LoadT*)(residual + idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      input4[ii] += residual4[ii];
      val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);
    }
  }
  val = BlockReduceSum(val, val_shared);
  if (j == 0) s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
  __syncthreads();
  if (j < tcol) {
    trstd = s_rstd;
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj = j * VEC + ii;
      input4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(input + idx) = *(LoadT*)input4;
  }
}


template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_rn_add_kernel_val_4096(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* gamma,int cols,scalar_t eps)
{
  __shared__ T_ACC val_shared[8];
  __shared__ T_ACC s_rstd;
  T_ACC val=0;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  scalar_t input4[VEC];
  scalar_t input8[VEC];
  scalar_t output4[VEC];
  scalar_t output8[VEC];
  scalar_t residual4[VEC];
  scalar_t residual8[VEC];
  T_ACC trstd;
  int idx1 = i * tcol + j;
  int idx2 = i * tcol + j + 512;
  idx1*=VEC;
  *(LoadT*)input4 = *(LoadT*)(input+idx1);
  *(LoadT*)residual4 = *(LoadT*)(residual+idx1);
  #pragma unroll
  for (int ii = 0; ii < VEC; ii++) {
    input4[ii]+=residual4[ii];
    val += static_cast<T_ACC>(input4[ii])*static_cast<T_ACC>(input4[ii]);
  }
  if(j<tcol-512){
    idx2*=VEC;
    *(LoadT*)input8 = *(LoadT*)(input+idx2);
    *(LoadT*)residual8 = *(LoadT*)(residual+idx2);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      input8[ii]+=residual8[ii];
      val += static_cast<T_ACC>(input8[ii])*static_cast<T_ACC>(input8[ii]);
    }
  }
  val = BlockReduceSum(val,val_shared);
  if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + static_cast<T_ACC>(eps));
  __syncthreads();
  trstd=s_rstd;
  #pragma unroll
  for(int ii=0;ii<VEC;ii++){
    int jj=j*VEC+ii;
    output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
  }
  *(LoadT*)(ret+idx1)=*(LoadT*)output4;
  if(j<tcol-512){
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii+2048;
      output8[ii] = static_cast<T_ACC>(input8[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(ret+idx2)=*(LoadT*)output8;
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_rn_add_kernel_val_4096_inplace(scalar_t* input, scalar_t* residual, scalar_t* gamma, int cols, scalar_t eps)
{
    __shared__ T_ACC val_shared[8];
    __shared__ T_ACC s_rstd;
    T_ACC val = 0;
    int i = blockIdx.x;
    int j = threadIdx.x;
    int tcol = cols / VEC;
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    scalar_t input4[VEC];
    scalar_t input8[VEC];
    scalar_t residual4[VEC];
    scalar_t residual8[VEC];
    T_ACC trstd;
    int idx1 = i * tcol + j;
    int idx2 = i * tcol + j + 512;
    idx1 *= VEC;
    
    // Load data from global memory to local array
    *(LoadT*)input4 = *(LoadT*)(input + idx1);
    *(LoadT*)residual4 = *(LoadT*)(residual + idx1);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        input4[ii] += residual4[ii];
        val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);
    }
    if (j < tcol - 512) {
        idx2 *= VEC;
        *(LoadT*)input8 = *(LoadT*)(input + idx2);
        *(LoadT*)residual8 = *(LoadT*)(residual + idx2);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input8[ii] += residual8[ii];
            val += static_cast<T_ACC>(input8[ii]) * static_cast<T_ACC>(input8[ii]);
        }
    }
    val = BlockReduceSum(val, val_shared);
    if (j == 0) s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
    __syncthreads();
    trstd = s_rstd;
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        int jj = j * VEC + ii;
        input4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(input + idx1) = *(LoadT*)input4;
    if (j < tcol - 512) {
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int jj = j * VEC + ii + 2048;
            input8[ii] = static_cast<T_ACC>(input8[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        }
        *(LoadT*)(input + idx2) = *(LoadT*)input8;
    }
}


template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_rn_add_kernel_val(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* gamma,int cols,scalar_t eps)
{
  __shared__ T_ACC val_shared[8];
  __shared__ T_ACC s_rstd;
  T_ACC val=0;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t mask4[VEC];
  scalar_t residual4[VEC];
  T_ACC trstd;
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = i * tcol + off;
    idx*=VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      T_ACC temp=input4[ii]+residual4[ii];
      val += temp*temp;
    }
  }
  val = BlockReduceSum(val,val_shared);
  if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + static_cast<T_ACC>(eps));
  __syncthreads();
  trstd=s_rstd;
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = (i * tcol + off)*VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=off*VEC+ii;
      output4[ii] = static_cast<T_ACC>(input4[ii]+residual4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_rn_add_kernel_val_inplace(scalar_t* input, scalar_t* residual, scalar_t* gamma, int cols, scalar_t eps)
{
    __shared__ T_ACC val_shared[8];
    __shared__ T_ACC s_rstd;
    T_ACC val = 0;
    int i = blockIdx.x;
    int j = threadIdx.x;
    int tcol = cols / VEC;
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    float rand[VEC];
    scalar_t input4[VEC];
    scalar_t residual4[VEC];
    T_ACC trstd;

        for (int off = j; off < tcol; off += blockDim.x) {
        int idx = i * tcol + off;
        idx *= VEC;
        *(LoadT*)input4 = *(LoadT*)(input + idx);
        *(LoadT*)residual4 = *(LoadT*)(residual + idx);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            T_ACC temp = input4[ii] + residual4[ii];
            val += temp * temp;
        }
    }

    val = BlockReduceSum(val, val_shared);
    if (j == 0)
        s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
    __syncthreads();
    trstd = s_rstd;

        for (int off = j; off < tcol; off += blockDim.x) {
        int idx = i * tcol + off;
        idx *= VEC;
        *(LoadT*)input4 = *(LoadT*)(input + idx);
        *(LoadT*)residual4 = *(LoadT*)(residual + idx);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int jj = off * VEC + ii;
            input4[ii] = static_cast<T_ACC>(input4[ii] + residual4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        }
        *(LoadT*)(input + idx) = *(LoadT*)input4;     }
}


template <typename scalar_t, typename T_ACC>

#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_rn_add_kernel_train_2048(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retadd_data,float* rstd,scalar_t* gamma, int cols,scalar_t eps)
{
    __shared__ T_ACC val_shared[8];
    __shared__ T_ACC s_rstd;
  
    T_ACC val = 0;
  
    int i = blockIdx.x;
    int j = threadIdx.x;
  
    int tcol = cols / VEC;
  
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
  
  
    scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t residual4[VEC];
  
    T_ACC trstd;
  
    int idx = i * tcol + j;
  
  if (j < tcol) {
        idx *= VEC;
        *(LoadT*)input4 = *(LoadT*)(input + idx);
        *(LoadT*)residual4 = *(LoadT*)(residual + idx);
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
            input4[ii] = input4[ii]  + residual4[ii]; 
            val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);
    }
  }
  
    val = BlockReduceSum(val, val_shared);
  
  if (j == 0) {
        s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
  }
  
    __syncthreads();
  
  if (j < tcol) {
        trstd = s_rstd;
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj = j * VEC + ii;
            output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    
        *(LoadT*)(retadd_data + idx) = *(LoadT*)input4;
        *(LoadT*)(ret + idx) = *(LoadT*)output4;
    
    if (threadIdx.x == 0) {
            rstd[i] = trstd;
    }
  }
}


template <typename scalar_t, int VEC = 4, typename T_ACC>
__global__ void fused_rn_add_kernel_train_fast(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retadd_data,float* rstd,scalar_t* gamma, int cols,scalar_t eps)
{
    __shared__ T_ACC val_shared[8];
    __shared__ T_ACC s_rstd;
  
    T_ACC val = 0;
  
    int i = blockIdx.x;
    int j = threadIdx.x;
  
    int tcol = cols / VEC;
  
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
  
  
    scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t residual4[VEC];
  
    T_ACC trstd;
  
    int idx = i * tcol + j;
  
  if (j < tcol) {
        idx *= VEC;
        *(LoadT*)input4 = *(LoadT*)(input + idx);
        *(LoadT*)residual4 = *(LoadT*)(residual + idx);
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
            input4[ii] = input4[ii]  + residual4[ii]; 
            val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);
    }
  }
  
    val = BlockReduceSum(val, val_shared);
  
  if (j == 0) {
        s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
  }
  
    __syncthreads();
  
  if (j < tcol) {
        trstd = s_rstd;
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj = j * VEC + ii;
            output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }
    
        *(LoadT*)(retadd_data + idx) = *(LoadT*)input4;
        *(LoadT*)(ret + idx) = *(LoadT*)output4;
    
    if (threadIdx.x == 0) {
            rstd[i] = trstd;
    }
  }
}

template <typename scalar_t, typename T_ACC>

#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif

__global__ void fused_rn_add_kernel_train_4096(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retadd_data,float* rstd,scalar_t* gamma, int cols,scalar_t eps) {
        __shared__ T_ACC val_shared[8];      __shared__ T_ACC s_rstd;      T_ACC val = 0;      int i = blockIdx.x;      int j = threadIdx.x;      int tcol = cols / VEC;          using LoadT = memory::aligned_vector<scalar_t, VEC>;


        hiprandStatePhilox4_32_10_t state;

        scalar_t input4[VEC], input8[VEC];
    scalar_t output4[VEC], output8[VEC];
    scalar_t residual4[VEC], residual8[VEC];

    T_ACC trstd;
    int idx1 = i * tcol + j;      int idx2 = i * tcol + j + 512;      idx1 *= VEC;          *(LoadT*)input4 = *(LoadT*)(input + idx1);
    *(LoadT*)residual4 = *(LoadT*)(residual + idx1);

        #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        input4[ii] = input4[ii] + residual4[ii];          val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);      }

        if (j < tcol - 512) {
        idx2 *= VEC;                  *(LoadT*)input8 = *(LoadT*)(input + idx2);
        *(LoadT*)residual8 = *(LoadT*)(residual + idx2);

        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input8[ii] = input8[ii] + residual8[ii];
            val += static_cast<T_ACC>(input8[ii]) * static_cast<T_ACC>(input8[ii]);          }
    }

        val = BlockReduceSum(val, val_shared);

        if (j == 0) s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
    __syncthreads();

    trstd = s_rstd;

        #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        int jj = j * VEC + ii;
        output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
    }

        *(LoadT*)(retadd_data + idx1) = *(LoadT*)input4;  
    *(LoadT*)(ret + idx1) = *(LoadT*)output4;  

        if (j < tcol - 512) {
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int jj = j * VEC + ii + 2048;
            output8[ii] = static_cast<T_ACC>(input8[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        }

        *(LoadT*)(retadd_data + idx2) = *(LoadT*)input8;
        *(LoadT*)(ret + idx2) = *(LoadT*)output8;
    }

        if (j == 0) {
        rstd[i] = trstd;
    }
}


template <typename scalar_t, typename T_ACC>

#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif

__global__ void fused_rn_add_kernel_train(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retadd_data,float* rstd,scalar_t* gamma, int cols,scalar_t eps) {
        __shared__ T_ACC val_shared[8];
        __shared__ T_ACC s_rstd;
    
        T_ACC val = 0;
        int i = blockIdx.x;
        int j = threadIdx.x;
        int tcol = cols / VEC;
    
        using LoadT = memory::aligned_vector<scalar_t, VEC>;
        
        scalar_t input4[VEC];
        scalar_t residual4[VEC];
        scalar_t output4[VEC];
        T_ACC trstd;
    
        for (int off = j; off < tcol; off += blockDim.x) {
        int idx = i * tcol + off;
        idx *= VEC;
                *(LoadT*)input4 = *(LoadT*)(input + idx);
        *(LoadT*)residual4 = *(LoadT*)(residual + idx);
        
                #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
                        input4[ii] = input4[ii]  + residual4[ii];             val += static_cast<T_ACC>(input4[ii]) * static_cast<T_ACC>(input4[ii]);
        }

                *(LoadT*)(retadd_data + idx) = *(LoadT*)input4;
    }
    
        val = BlockReduceSum(val, val_shared);
    
        if (j == 0) {
                s_rstd = c10::cuda::compat::rsqrt(val / cols + static_cast<T_ACC>(eps));
    }
        __syncthreads();
    
        trstd = s_rstd;

        for (int off = j; off < tcol; off += blockDim.x) {
        int idx = (i * tcol + off) * VEC;
                *(LoadT*)input4 = *(LoadT*)(retadd_data + idx);
        
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int jj = off * VEC + ii;
                        output4[ii] = static_cast<T_ACC>(input4[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        }

                *(LoadT*)(ret + idx) = *(LoadT*)output4;
    }

        if (threadIdx.x == 0) {
        rstd[i] = trstd;
    }
}


std::tuple<Tensor, Tensor, Tensor>
fused_rn_add_forward(const Tensor &self, const Tensor &other, const Tensor &weight, double rn_epsilon, bool training, bool inplace)
{
    AT_ASSERTM(self.sizes() == other.sizes(), "self must be the same size as other");
    AT_ASSERTM(self.is_cuda() && other.is_cuda(), "self and other must be on CUDA");
    const int cols = self.sizes().back();
    AT_ASSERT(cols >= 64, "cols must be >= 64");
    AT_ASSERT(cols % 4 == 0, "cols % 4 must be 0");

    const int nelem = self.numel();
    const int rows = nelem / cols;
    Tensor rstd;
    bool eval_inplace = training == false && inplace == true;
    if(eval_inplace)
    {
      Tensor retadd;
      auto stream = at::cuda::getCurrentCUDAStream();
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "fused_rn_add_forward",
          [&]
          {
              using T_ACC = acc_type<scalar_t, true>;
              scalar_t eps = rn_epsilon;
              scalar_t *self_data = self.data_ptr<scalar_t>();
              scalar_t *other_data = other.data_ptr<scalar_t>();
              scalar_t *weight_data = weight.data_ptr<scalar_t>();
//-----------------------------------------------------------------------
              if(cols<=16384 && cols%16==0)
              {
                if (cols <= 1024)
                {
                    fused_rn_add_kernel_val_2048_inplace<scalar_t, 8, T_ACC><<<rows, 128, 0, stream>>>(self_data, other_data, weight_data, cols, eps);

                }            
                else if (cols <= 2048)
                {
                    fused_rn_add_kernel_val_2048_inplace<scalar_t, 8, T_ACC><<<rows, 256, 0, stream>>>(self_data, other_data, weight_data, cols, eps);

                }
                else if (cols <= 4096)
                {
                    fused_rn_add_kernel_val_2048_inplace<scalar_t, 8, T_ACC><<<rows, 512, 0, stream>>>(self_data, other_data, weight_data, cols, eps);
                }
                else if(cols <= 8192)
                {
                    fused_rn_add_kernel_val_2048_inplace<scalar_t, 8, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, weight_data, cols, eps);
                }
                else if(cols <= 16384)
                {
                    fused_rn_add_kernel_val_2048_inplace<scalar_t, 16, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, weight_data, cols, eps);
                }
              }
              else
              {
                fused_rn_add_kernel_val_inplace<scalar_t, T_ACC><<<rows, get_layernorm_blocksize(cols / 4), 0, stream>>>(self_data, other_data, weight_data, cols, eps);
              }
//-----------------------------------------------------------------------  
              C10_CUDA_KERNEL_LAUNCH_CHECK();
          });

      return std::tuple<Tensor, Tensor, Tensor>(self, retadd, rstd);
    }
    else
    {
      Tensor ret = at::empty_like(self);
      Tensor retadd;
      auto stream = at::cuda::getCurrentCUDAStream();
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "fused_rn_add_forward",
          [&]
          {
              using T_ACC = acc_type<scalar_t, true>;
              scalar_t eps = rn_epsilon;
              scalar_t *self_data = self.data_ptr<scalar_t>();
              scalar_t *other_data = other.data_ptr<scalar_t>();
              scalar_t *ret_data = ret.data_ptr<scalar_t>();
              scalar_t *weight_data = weight.data_ptr<scalar_t>();

              if (training)
              {
                  retadd = at::empty({nelem * 2}, self.options());
                  rstd = at::empty({rows}, self.options().dtype(at::ScalarType::Float));
                  scalar_t *retadd_data = retadd.data_ptr<scalar_t>();
                  float *rstd_data = rstd.data_ptr<float>();
              if (cols % 16 == 0 && cols<=16384)
              {
                if(cols<=1024){
                  fused_rn_add_kernel_train_fast<scalar_t, 8, T_ACC><<<rows, 128, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                }
                else if(cols<=2048){
                  fused_rn_add_kernel_train_fast<scalar_t, 8, T_ACC><<<rows, 256, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                }
                else if(cols<=4096){
                  if(rows>1200){
                  fused_rn_add_kernel_train_fast<scalar_t, 8, T_ACC><<<rows, 512, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                  }
                  else{
                  fused_rn_add_kernel_train_fast<scalar_t, 4, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                  }
                }
                else if(cols<=8192){
                  fused_rn_add_kernel_train_fast<scalar_t, 8, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                }
                else{
                  fused_rn_add_kernel_train_fast<scalar_t, 16, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
                }
              }
              else{
                      fused_rn_add_kernel_train<scalar_t, T_ACC><<<rows, 512, 0, stream>>>(self_data, other_data, ret_data,retadd_data, rstd_data, weight_data, cols, eps);
              }
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
              else
              {
                if(cols<=16384 && cols%16==0)
                {
                  if (cols <= 1024)
                  {
                      fused_rn_add_kernel_val_2048<scalar_t, 8, T_ACC><<<rows, 128, 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);

                  }            
                  else if (cols <= 2048)
                  {
                      fused_rn_add_kernel_val_2048<scalar_t, 8, T_ACC><<<rows, 256, 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);

                  }
                  else if (cols <= 4096)
                  {
                      fused_rn_add_kernel_val_2048<scalar_t, 8, T_ACC><<<rows, 512, 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);
                  }
                  else if(cols <= 8192)
                  {
                      fused_rn_add_kernel_val_2048<scalar_t, 8, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);
                  }
                  else if(cols <= 16384)
                  {
                      fused_rn_add_kernel_val_2048<scalar_t, 16, T_ACC><<<rows, 1024, 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);
                  }
                }
                else
                {
                  fused_rn_add_kernel_val<scalar_t, T_ACC><<<rows, get_layernorm_blocksize(cols / 4), 0, stream>>>(self_data, other_data, ret_data, weight_data, cols, eps);
                }
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
          });

      return std::tuple<Tensor, Tensor, Tensor>(ret, retadd, rstd);
    }
}



template <typename scalar_t, int VEC=4>
__global__ void RMSNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2)
{
  RMSNORMBACKWARD_VEC_2048
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
      pdx[i]=pdx2[i];
    }
    *(VecTpye*)(dX1+index) = tdx;
    *(VecTpye*)(dX2+index) = tdx2;
  }
}

template <typename scalar_t, int VEC=4>
__global__ void RMSNormBackward_kernel_vec_fast(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2)
{
  RMSNORMBACKWARD_VEC_2048
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
      pdx[i]=pdx2[i];
    }
    *(VecTpye*)(dX1+index) = tdx;
    *(VecTpye*)(dX2+index) = tdx2;
  }
}

template <typename scalar_t>
__global__ void RMSNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2)
{
  RMSNORMBACKWARD_VEC_4096
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  #pragma unroll
  for(int i=0;i<VEC;i++){
    pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
    pdx[i]=pdx2[i];
  }
  *(VecTpye*)(dX1+index) = tdx;
  *(VecTpye*)(dX2+index) = tdx2;
  if (j < tcol-512) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(pgamma_2[i])+b*static_cast<T_ACC>(px_2[i]);
      pdx[i]=pdx2[i];
    }
    *(VecTpye*)(dX1+index_2) = tdx;
    *(VecTpye*)(dX2+index_2) = tdx2;
  }
}

template <typename scalar_t>
__global__ void RMSNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2)
{
  RMSNORMBACKWARD_VEC
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
      pdx[i]=pdx2[i];
    }
    *(VecTpye*)(dX1+index) = tdx;
    *(VecTpye*)(dX2+index) = tdx2;
  }
}


std::tuple<Tensor,Tensor,Tensor>
fused_rn_add_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX1 = at::empty_like(*pgrad);
  Tensor dX2 = at::empty_like(dX1);
  Tensor dgamma = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "fused_rn_add_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dX1_data = dX1.data_ptr<scalar_t>();
      scalar_t* dX2_data = dX2.data_ptr<scalar_t>();
      
      Gamma_Backward(nelem,N,dY_data,X_data,rstd_data,dgamma_data);
      if (N % 16 == 0 && N<=16384){
        if(N<=1024)
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
        else if(N<=2048){
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
        }
        else if(N<=4096){
          if(M>1200){
            RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
          }
          else{
            RMSNormBackward_kernel_vec_fast<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
          }
        }
        else if(N<=8192){
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
        }
        else{
          RMSNormBackward_kernel_vec_fast<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
        }
      }
      else{
          RMSNormBackward_kernel_vec<scalar_t><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX1_data,dX2_data);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX1,dX2,dgamma);
}


class fusernAddFunction : public torch::autograd::Function<fusernAddFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& y,
    const torch::autograd::Variable& weight,
    double eps, bool training, bool inplace) {
    auto px=x.expect_contiguous();
    auto py=y.expect_contiguous();
    auto pweight=weight.expect_contiguous();
    auto ret = fused_rn_add_forward(*px,*py, *pweight,eps,training,inplace);
    ctx->save_for_backward({std::get<1>(ret),std::get<2>(ret),weight});
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto saved = ctx->get_saved_variables();
    auto X = saved[0];
    auto rstd = saved[1];
    auto gamma = saved[2];
    auto ret = fused_rn_add_backward(gradOutput[0],X,rstd,gamma);
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),
            torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};


Tensor fuse_rn_add_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,bool training, bool inplace) {
   auto result = fusernAddFunction::apply(x,y, weight,eps,training,inplace);
   return result[0];
}
}
}

