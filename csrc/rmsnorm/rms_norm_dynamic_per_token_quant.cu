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

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static __device__ inline float clamp_rms_quant_value(float value, float limit) {
  return fminf(fmaxf(value, -limit), limit);
}

template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512, bool inpit_update=false, bool APPLY_CLAMP=false>
__global__ void rms_norm_dynamic_per_token_quant_(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps,scalar_out_t* out,float* scales, int stride, float limit)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  scalar_t intput_vec[Vec];
  T_ACC trstd;
  int64_t idx = bid * tcol + tid;
  idx *= Vec;
  int64_t index = bid * stride + tid * Vec;
  if (tid < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+index);
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
          val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
      }
  }
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd=s_rstd;

  float block_absmax_val_maybe = 0.f;
  scalar_t elem[Vec];
  if (tid < tcol) {
      #pragma unroll
      for(int ii = 0;ii < Vec;ii++){
        int jj=tid * Vec + ii;
        elem[ii] = static_cast<T_ACC>(intput_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        if constexpr (APPLY_CLAMP) {
          elem[ii] = static_cast<scalar_t>(
              clamp_rms_quant_value(static_cast<float>(elem[ii]), limit));
        }
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
      }      
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
  __shared__ float s_token_scale; 
  float scale = 0.0f;     
  if(tid == 0)
  {
      scale = block_absmax_val_maybe;
    s_token_scale = scale;               // shared memory store
  }
  __syncthreads();
  float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

  scalar_out_t out_vec[Vec];
  if (tid < tcol) {
  #pragma unroll
  for(int ii = 0;ii < Vec;ii++){
      int jj = tid * Vec + ii;
      out_vec[ii] = float_to_int8_rn(elem[ii] * inv_s);
    }
    constexpr float qmax = 127.0f;
    constexpr float min_scale = 1.19209e-07f;
  *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;
  if constexpr (inpit_update){
    *(LoadT*)(input+index)=*(LoadT*)elem;
  }
  
  if(tid==0)scales[blockIdx.x] = fmaxf(scale / qmax, min_scale);
  }
}

template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec,int block_size, bool APPLY_CLAMP=false>
__global__ void rms_norm_dynamic_per_token_quant_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps, scalar_out_t* out,float* scales, int stride, float limit)
{
  constexpr int share_size=block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0.0f;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  T_ACC trstd;
  int64_t rows = bid * stride;

  for(int i = tid;i < cols;i += block_size)
  {
    val += static_cast<T_ACC>(input[rows + i])*static_cast<T_ACC>(input[rows + i]);
  }
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd = c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd = s_rstd;
  float block_absmax_val_maybe = 0.0f;
  for(int i = tid;i < cols;i += block_size){
    T_ACC normalized = static_cast<T_ACC>(input[i + rows]) * trstd *
        static_cast<T_ACC>(gamma[i]);
    if constexpr (APPLY_CLAMP) {
      normalized = clamp_rms_quant_value(normalized, limit);
    }
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, fabs(normalized));
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe,val_shared);

  __shared__ float s_token_scale; 
  float scale = 0.0f;
  if(tid == 0)
  {
      scale = block_absmax_val_maybe;
    constexpr float qmax = 127.0f;
    constexpr float min_scale = 1.19209e-07f;
    // token scale computation
    s_token_scale = scale;                 // shared memory store
    scales[blockIdx.x] = fmaxf(scale / qmax, min_scale); // global output store
  }
  __syncthreads();
  float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
  scalar_out_t out_vec[Vec];
  for(int i = tid; i < cols; i += block_size)
  {
    T_ACC normalized = static_cast<T_ACC>(input[rows + i]) * trstd *
        static_cast<T_ACC>(gamma[i]);
    if constexpr (APPLY_CLAMP) {
      normalized = clamp_rms_quant_value(normalized, limit);
    }
    out[rows + i] = float_to_int8_rn(normalized * inv_s);
    input[rows + i] = normalized;
  }
}


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512,bool inpit_update=false, bool APPLY_CLAMP=false>
__global__ void rms_norm_dynamic_per_token_quant_residual(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride, float limit)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  scalar_t intput_vec[Vec];
  scalar_t residual_vec[Vec];
  T_ACC trstd;
  int64_t idx = bid * tcol + tid;
  idx *= Vec;
  if (tid < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+idx);
      *(LoadT*)residual_vec = *(LoadT*)(residual+idx);

      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
         residual_vec[ii] += (intput_vec[ii]);  
          val += static_cast<T_ACC>(residual_vec[ii])*static_cast<T_ACC>(residual_vec[ii]);
      }
  }

  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd=s_rstd;

  float block_absmax_val_maybe = 0.f;
  scalar_t elem[Vec];
  if (tid < tcol) {
      #pragma unroll
      for(int ii = 0;ii < Vec;ii++){
        int jj=tid * Vec + ii;
        elem[ii] = static_cast<T_ACC>(residual_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
        if constexpr (APPLY_CLAMP) {
          elem[ii] = static_cast<scalar_t>(
              clamp_rms_quant_value(static_cast<float>(elem[ii]), limit));
        }
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
      }      
  }
  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
    __shared__ float s_token_scale; 
  float scale = 0.0f;     
  if(tid == 0)
  {
      scale = block_absmax_val_maybe ;
    s_token_scale = scale;                 // shared memory store
    
  }
  __syncthreads();
  float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
  scalar_out_t out_vec[Vec];
  if (tid < tcol) {
  #pragma unroll
  for(int ii = 0;ii < Vec;ii++){
      int jj = tid * Vec + ii;
      out_vec[ii] = float_to_int8_rn(elem[ii] * inv_s);
    }
  constexpr float qmax = 127.0f;
  constexpr float min_scale = 1.19209e-07f;  
  
  if(tid==0)scales[blockIdx.x] = fmaxf(scale / qmax, min_scale);
    *(LoadT*)(residual+idx) = *(LoadT*)residual_vec;
    *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;
  if constexpr (inpit_update){
    *(LoadT*)(input+idx)=*(LoadT*)elem;  
  }
    
  }  
}


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=1,int block_size=512, bool APPLY_CLAMP=false>
__global__ void rms_norm_dynamic_per_token_quant_residual_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride, float limit)
{
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val = 0.0f;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  T_ACC trstd;
  int64_t rows = bid * cols;

  for(int i = tid;i < cols;i += block_size)
  {
    float x = static_cast<T_ACC>(residual[rows + i]) + static_cast<T_ACC>(input[rows + i]);
        val += x * x;
  }
    
  val = BlockReduceSum_ROW<T_ACC,block_size>(val, val_shared);
  if (tid == 0) s_rstd=c10::cuda::compat::rsqrt(val / cols + eps);
  __syncthreads();
  trstd = s_rstd;
  float block_absmax_val_maybe = 0.0f;

  for(int i = tid;i < cols;i += block_size){
    T_ACC x = static_cast<T_ACC>(residual[i + rows]) +
        static_cast<T_ACC>(input[i + rows]);
    T_ACC normalized = static_cast<T_ACC>(x * trstd) * gamma[i];
    if constexpr (APPLY_CLAMP) {
      normalized = clamp_rms_quant_value(normalized, limit);
    }
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, fabs(normalized));
  }

  block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);

  __shared__ float s_token_scale; 
   float scale = 0.0f;
  if(tid == 0)
  {
      scale = block_absmax_val_maybe;
    constexpr float qmax = 127.0f;
    constexpr float min_scale = 1.19209e-07f;
    s_token_scale = scale;                 // shared memory store
    scales[blockIdx.x] = fmaxf(scale / qmax, min_scale);  // global output store
    
  }
   __syncthreads();
  float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
  scalar_out_t out_vec[Vec];
  for(int i = tid;i < cols;i += block_size)
  {
    residual[rows + i] = static_cast<T_ACC>(residual[i + rows]) +
        static_cast<T_ACC>(input[i + rows]);
    T_ACC normalized = static_cast<T_ACC>(residual[rows + i]) * trstd *
        static_cast<T_ACC>(gamma[i]);
    if constexpr (APPLY_CLAMP) {
      normalized = clamp_rms_quant_value(normalized, limit);
    }
    out[rows + i] = float_to_int8_rn(normalized * inv_s);
    input[rows + i] = normalized;
  }

}


template <typename scalar_in_t, bool APPLY_CLAMP>
void rms_norm_dynamic_per_token_quant_(
  torch::Tensor& out,           // [..., hidden_size]
  torch::Tensor& input,   // [..., hidden_size]
  torch::Tensor const& weight,  // [hidden_size]
  torch::Tensor& scales,        // [num_tokens]
  double const epsilon,     // Variance epsilon used in norm calculation
  std::optional<at::Tensor> residual,
  bool input_flag,
  double limit) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0 && wt_ptr % 16 == 0;
  int input_ndim = input.dim();
  int stride;
  if(input_ndim > 1){
    stride = input.stride(-2);
  }else{
    stride = input.stride(-1);
  }
  if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned && input_flag){
        using T_ACC = at::acc_type<scalar_in_t, true>;
        
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();

        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  
          "fused_add_rms_norm_kernel_out_type",
          [&] {
             using scalar_out_t = scalar_t;
        scalar_out_t* out_data = out.data_ptr<scalar_out_t>();
        float* scaless = scales.data_ptr<float>();
        if(residual_flag){
          if (hidden_size <= 512){
              if(num_tokens <= 1024){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true, APPLY_CLAMP><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else if(num_tokens <= 3072){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,true, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true, APPLY_CLAMP><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                  
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,true, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 1000){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  }
              else if(num_tokens < 8000){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,true, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }  
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                else{
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,true, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }
            else{
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,128,true, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }
          else if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,128,true, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,512,true, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,256,true, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,512,true, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
            else{
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,true, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
        }
        
      });

  }
  else  if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned && input_flag==false){
        using T_ACC = at::acc_type<scalar_in_t, true>;
        
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();

        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  
          "fused_add_rms_norm_kernel_out_type",
          [&] {
             using scalar_out_t = scalar_t;
        scalar_out_t* out_data = out.data_ptr<scalar_out_t>();
        float* scaless = scales.data_ptr<float>();
        if(residual_flag){
          if (hidden_size <= 512){
              if(num_tokens <= 1024){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false, APPLY_CLAMP><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else if(num_tokens < 4096){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,false, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false, APPLY_CLAMP><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,false, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 512){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,false, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  } 
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,false, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                else{
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
            }
            else if(hidden_size <= 8192){
              if(num_tokens <= 4){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else if(num_tokens < 64){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else if(num_tokens < 290){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else if(num_tokens < 2000){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                  
          }
            else{
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,128,false, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }
          else if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,128,false, APPLY_CLAMP><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,256,false, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,512,false, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
            else{
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,false, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
        }
        
      });

  }
  else{
        using T_ACC = at::acc_type<scalar_in_t, true>;
        scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
        scalar_in_t* other_data = nullptr;
        bool residual_flag = false;
        if(residual.has_value()){
          residual_flag = true;
          other_data = residual->data_ptr<scalar_in_t>();
        }
        scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();
        float* scaless = scales.data_ptr<float>();
        AT_DISPATCH_INTEGRAL_TYPES(
          out.scalar_type(),  // scalar_out_t 跟 output 对应
          "fused_add_rms_norm_kernel_out_type",
          [&] {
          scalar_t* out_data = out.data_ptr<scalar_t>();
        if(residual_flag){
            if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
            else if(hidden_size <= 2048){
              if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
              else{
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }  
              
            }
            else if(hidden_size <= 4096){
              if(num_tokens > 1200){
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  
              }
              else{
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
              }
            }
            else {
                rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }

        }else{
            if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                }
                else{
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256, APPLY_CLAMP><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                }                      

            }
            else if(hidden_size <= 4096){
                if(num_tokens > 1200){
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512, APPLY_CLAMP><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                  }
                else{
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
                }
            }
            else{
                  rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024, APPLY_CLAMP><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride,static_cast<float>(limit));
            }
        }

      });
  }
    }


void rms_norm_dynamic_per_token_quant(
  Tensor &out,
  Tensor& input,
  const Tensor& weight,
  Tensor scales, 
  double epsilon,
  std::optional<Tensor> residual,
  bool update_input,
  double limit
) {
  TORCH_CHECK(
      limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
      "limit must be -1 (disabled) or a finite non-negative value");
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {
      if (limit < 0.0) {
        rms_norm_dynamic_per_token_quant_<scalar_t, false>(
            out, input, weight, scales, epsilon, residual, update_input, limit);
      } else {
        rms_norm_dynamic_per_token_quant_<scalar_t, true>(
            out, input, weight, scales, epsilon, residual, update_input, limit);
      }
    });
}


// SGLang W8A8-compatible variant.  Keep this separate from the historical
// interface above: its quantization contract intentionally matches
// lmslim.layers.gemm.int8_utils.per_token_quant_int8 used by SGLang W8A8.
template <typename scalar_t, int Vec, int block_size,
          bool has_residual, bool update_input,
          bool APPLY_CLAMP, bool APPLY_SMOOTH>
__global__ __launch_bounds__(block_size)
void rms_norm_dynamic_per_token_quant_sglang_kernel(
    int8_t* out,
    scalar_t* input,
    scalar_t* norm_out,
    scalar_t* residual,
    scalar_t* residual_out,
    const scalar_t* gamma,
    int32_t cols,
    float eps,
    float* scales,
    const float* smooth_scale,
    float limit,
    bool clamp_norm_output) {
  using T_ACC = at::acc_type<scalar_t, true>;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using StoreT = at::native::memory::aligned_vector<int8_t, Vec>;
  constexpr int share_size = block_size / C10_WARP_SIZE;
  constexpr float qmax = 127.0f;
  constexpr float min_absmax = 1.0e-10f;

  __shared__ T_ACC reduce_shared[share_size];
  __shared__ T_ACC row_rstd;
  __shared__ float row_absmax;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_cols = cols / Vec;
  const int64_t offset = static_cast<int64_t>(row) * cols + tid * Vec;
  const bool active = tid < vec_cols;

  alignas(16) scalar_t input_vec[Vec];
  alignas(16) scalar_t residual_vec[Vec];
  T_ACC norm_input[Vec];
  T_ACC square_sum = 0.0f;
  if (active) {
    *reinterpret_cast<LoadT*>(input_vec) =
        *reinterpret_cast<const LoadT*>(input + offset);
    if constexpr (has_residual) {
      *reinterpret_cast<LoadT*>(residual_vec) =
          *reinterpret_cast<const LoadT*>(residual + offset);
    }
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
      T_ACC value = static_cast<T_ACC>(input_vec[i]);
      if constexpr (has_residual) {
        value += static_cast<T_ACC>(residual_vec[i]);
        residual_vec[i] = static_cast<scalar_t>(value);
      }
      norm_input[i] = value;
      square_sum += value * value;
    }
  }

  square_sum =
      BlockReduceSum_ROW<T_ACC, block_size>(square_sum, reduce_shared);
  if (tid == 0) {
    row_rstd = c10::cuda::compat::rsqrt(square_sum / cols + eps);
  }
  __syncthreads();

  alignas(16) scalar_t normalized[Vec];
  alignas(16) scalar_t clamped[Vec];
  alignas(16) scalar_t quant_input[Vec];
  float local_absmax = 0.0f;
  if (active) {
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
      const int col = tid * Vec + i;
      // Match the existing path's FP32 RMSNorm math followed by its final
      // cast to the activation dtype.  Quantization observes that rounded
      // activation, not the unrounded FP32 product.
      normalized[i] = static_cast<scalar_t>(
          norm_input[i] * row_rstd * static_cast<T_ACC>(gamma[col]));
      clamped[i] = normalized[i];
      if constexpr (APPLY_CLAMP) {
        clamped[i] = static_cast<scalar_t>(clamp_rms_quant_value(
            static_cast<float>(clamped[i]), limit));
      }
      float quant_value = static_cast<float>(clamped[i]);
      if constexpr (APPLY_SMOOTH) {
        const scalar_t scale_value =
            static_cast<scalar_t>(smooth_scale[col]);
        quant_value *= static_cast<float>(scale_value);
      }
      quant_input[i] = static_cast<scalar_t>(quant_value);
      local_absmax =
          fmaxf(local_absmax, fabsf(static_cast<float>(quant_input[i])));
    }
  }

  local_absmax =
      BlockReduceMax_ROW<T_ACC, block_size>(local_absmax, reduce_shared);
  if (tid == 0) {
    row_absmax = fmaxf(local_absmax, min_absmax);
    scales[row] = row_absmax / qmax;
  }
  __syncthreads();

  if (active) {
    alignas(8) int8_t quantized[Vec];
    alignas(16) scalar_t norm_output_vec[Vec];
    const float inv_scale = qmax / row_absmax;
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
      quantized[i] = float_to_int8_rn(
          static_cast<float>(quant_input[i]) * inv_scale);
      norm_output_vec[i] = clamp_norm_output ? clamped[i] : normalized[i];
    }
    *reinterpret_cast<StoreT*>(out + offset) =
        *reinterpret_cast<const StoreT*>(quantized);
    if (norm_out != nullptr) {
      *reinterpret_cast<LoadT*>(norm_out + offset) =
          *reinterpret_cast<const LoadT*>(norm_output_vec);
    }
    if constexpr (update_input) {
      *reinterpret_cast<LoadT*>(input + offset) =
          *reinterpret_cast<const LoadT*>(norm_output_vec);
    }
    if constexpr (has_residual) {
      *reinterpret_cast<LoadT*>(residual_out + offset) =
          *reinterpret_cast<const LoadT*>(residual_vec);
    }
  }
}


template <typename scalar_t, int Vec, int block_size>
void launch_rms_norm_dynamic_per_token_quant_sglang(
    int8_t* out,
    scalar_t* input,
    scalar_t* norm_out,
    scalar_t* residual,
    scalar_t* residual_out,
    const scalar_t* gamma,
    int num_tokens,
    int hidden_size,
    float epsilon,
    float* scales,
    bool has_residual,
    bool update_input,
    bool apply_clamp,
    bool apply_smooth,
    const float* smooth_scale,
    float limit,
    bool clamp_norm_output,
    cudaStream_t stream) {
#define LAUNCH_SGLANG_RMS_QUANT(                                            \
    HAS_RESIDUAL, UPDATE_INPUT, APPLY_CLAMP, APPLY_SMOOTH)                  \
  rms_norm_dynamic_per_token_quant_sglang_kernel<                           \
      scalar_t, Vec, block_size, HAS_RESIDUAL, UPDATE_INPUT,                \
      APPLY_CLAMP, APPLY_SMOOTH>                                            \
      <<<num_tokens, block_size, 0, stream>>>(                              \
          out, input, norm_out, residual, residual_out, gamma, hidden_size, epsilon, \
          scales, smooth_scale, limit, clamp_norm_output)

#define DISPATCH_SGLANG_RMS_QUANT(APPLY_CLAMP, APPLY_SMOOTH)                \
  do {                                                                      \
    if (has_residual) {                                                     \
      if (update_input) {                                                   \
        LAUNCH_SGLANG_RMS_QUANT(                                            \
            true, true, APPLY_CLAMP, APPLY_SMOOTH);                         \
      } else {                                                              \
        LAUNCH_SGLANG_RMS_QUANT(                                            \
            true, false, APPLY_CLAMP, APPLY_SMOOTH);                        \
      }                                                                     \
    } else if (update_input) {                                              \
      LAUNCH_SGLANG_RMS_QUANT(                                              \
          false, true, APPLY_CLAMP, APPLY_SMOOTH);                          \
    } else {                                                                \
      LAUNCH_SGLANG_RMS_QUANT(                                              \
          false, false, APPLY_CLAMP, APPLY_SMOOTH);                         \
    }                                                                       \
  } while (false)

  if (apply_clamp) {
    if (apply_smooth) {
      DISPATCH_SGLANG_RMS_QUANT(true, true);
    } else {
      DISPATCH_SGLANG_RMS_QUANT(true, false);
    }
  } else if (apply_smooth) {
    DISPATCH_SGLANG_RMS_QUANT(false, true);
  } else {
    DISPATCH_SGLANG_RMS_QUANT(false, false);
  }

#undef DISPATCH_SGLANG_RMS_QUANT
#undef LAUNCH_SGLANG_RMS_QUANT
}


static bool tensors_overlap(const Tensor& a, const Tensor& b) {
  if (a.device() != b.device() || a.numel() == 0 || b.numel() == 0) {
    return false;
  }
  const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
  const auto a_end = a_begin + a.numel() * a.element_size();
  const auto b_end = b_begin + b.numel() * b.element_size();
  return a_begin < b_end && b_begin < a_end;
}


void rms_norm_dynamic_per_token_quant_sglang(
    Tensor& out,
    Tensor& input,
    const Tensor& weight,
    Tensor scales,
    double epsilon,
    std::optional<Tensor> residual,
    bool update_input,
    double limit,
    std::optional<Tensor> smooth_scale,
    std::optional<Tensor> norm_output,
    bool clamp_norm_output,
    std::optional<Tensor> residual_output) {
  TORCH_CHECK(input.is_cuda(), "input must be on a GPU");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(weight.device() == input.device(),
              "weight must be on input.device");
  TORCH_CHECK(out.device() == input.device(),
              "out must be on input.device");
  TORCH_CHECK(scales.device() == input.device(),
              "scales must be on input.device");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "input must be FP16 or BF16");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
              "weight dtype must match input dtype");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Char,
              "out must have dtype torch.int8");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
              "scales must have dtype torch.float32");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "epsilon must be finite and positive");
  TORCH_CHECK(
      limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
      "limit must be -1 (disabled) or a finite non-negative value");

  const int hidden_size = input.size(1);
  const int num_tokens = input.size(0);
  TORCH_CHECK(hidden_size > 0 && hidden_size <= 8192 && hidden_size % 16 == 0,
              "hidden size must be divisible by 16 and in (0, 8192]");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden_size,
              "weight must have shape [hidden_size]");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == num_tokens &&
                  scales.size(1) == 1,
              "scales must have shape [num_tokens, 1]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
              "input must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "weight must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "out must be 8-byte aligned");
  TORCH_CHECK(!tensors_overlap(input, weight),
              "input and weight must not overlap");
  TORCH_CHECK(!tensors_overlap(input, out),
              "input and out must not overlap");
  TORCH_CHECK(!tensors_overlap(input, scales),
              "input and scales must not overlap");
  TORCH_CHECK(!tensors_overlap(weight, out),
              "weight and out must not overlap");
  TORCH_CHECK(!tensors_overlap(weight, scales),
              "weight and scales must not overlap");
  TORCH_CHECK(!tensors_overlap(out, scales),
              "out and scales must not overlap");

  bool has_smooth_scale = smooth_scale.has_value();
  if (has_smooth_scale) {
    TORCH_CHECK(smooth_scale->is_cuda(), "smooth_scale must be on a GPU");
    TORCH_CHECK(smooth_scale->is_contiguous(),
                "smooth_scale must be contiguous");
    TORCH_CHECK(smooth_scale->device() == input.device(),
                "smooth_scale must be on input.device");
    TORCH_CHECK(smooth_scale->scalar_type() == at::ScalarType::Float,
                "smooth_scale must have dtype torch.float32");
    TORCH_CHECK(smooth_scale->dim() == 1 &&
                    smooth_scale->size(0) == hidden_size,
                "smooth_scale must have shape [hidden_size]");
    TORCH_CHECK(!tensors_overlap(input, *smooth_scale),
                "input and smooth_scale must not overlap");
    TORCH_CHECK(!tensors_overlap(weight, *smooth_scale),
                "weight and smooth_scale must not overlap");
    TORCH_CHECK(!tensors_overlap(out, *smooth_scale),
                "out and smooth_scale must not overlap");
    TORCH_CHECK(!tensors_overlap(scales, *smooth_scale),
                "scales and smooth_scale must not overlap");
  }

  bool has_norm_output = norm_output.has_value();
  if (has_norm_output) {
    TORCH_CHECK(norm_output->is_cuda(), "norm_output must be on a GPU");
    TORCH_CHECK(norm_output->is_contiguous(),
                "norm_output must be contiguous");
    TORCH_CHECK(norm_output->device() == input.device(),
                "norm_output must be on input.device");
    TORCH_CHECK(norm_output->scalar_type() == input.scalar_type(),
                "norm_output dtype must match input dtype");
    TORCH_CHECK(norm_output->sizes() == input.sizes(),
                "norm_output shape must match input");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(norm_output->data_ptr()) % 16 == 0,
                "norm_output must be 16-byte aligned");
    TORCH_CHECK(!tensors_overlap(input, *norm_output),
                "input and norm_output must not overlap");
    TORCH_CHECK(!tensors_overlap(weight, *norm_output),
                "weight and norm_output must not overlap");
    TORCH_CHECK(!tensors_overlap(out, *norm_output),
                "out and norm_output must not overlap");
    TORCH_CHECK(!tensors_overlap(scales, *norm_output),
                "scales and norm_output must not overlap");
    if (has_smooth_scale) {
      TORCH_CHECK(!tensors_overlap(*smooth_scale, *norm_output),
                  "smooth_scale and norm_output must not overlap");
    }
  }

  bool has_residual = residual.has_value();
  if (residual_output.has_value()) {
    TORCH_CHECK(has_residual, "residual_output requires residual");
    TORCH_CHECK(residual_output->device() == input.device() &&
                    residual_output->scalar_type() == input.scalar_type() &&
                    residual_output->sizes() == input.sizes() &&
                    residual_output->is_contiguous(),
                "residual_output must match input shape/dtype/device and be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(residual_output->data_ptr()) % 16 == 0,
                "residual_output must be 16-byte aligned");
    for (const auto& tensor : {input, weight, out, scales, *residual}) {
      TORCH_CHECK(!tensors_overlap(*residual_output, tensor),
                  "residual_output must not alias another input/output");
    }
    TORCH_CHECK(!has_norm_output || !tensors_overlap(*residual_output, *norm_output),
                "residual_output must not alias norm_output");
    TORCH_CHECK(!has_smooth_scale || !tensors_overlap(*residual_output, *smooth_scale),
                "residual_output must not alias smooth_scale");
  }
  if (has_residual) {
    TORCH_CHECK(residual->sizes() == input.sizes(),
                "residual shape must match input");
    TORCH_CHECK(residual->scalar_type() == input.scalar_type(),
                "residual dtype must match input dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->device() == input.device(),
                "residual must be on input.device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(residual->data_ptr()) % 16 == 0,
                "residual must be 16-byte aligned");
    TORCH_CHECK(!tensors_overlap(input, *residual),
                "input and residual must not overlap");
    TORCH_CHECK(!tensors_overlap(weight, *residual),
                "weight and residual must not overlap");
    TORCH_CHECK(!tensors_overlap(out, *residual),
                "out and residual must not overlap");
    TORCH_CHECK(!tensors_overlap(scales, *residual),
                "scales and residual must not overlap");
    if (has_smooth_scale) {
      TORCH_CHECK(!tensors_overlap(*smooth_scale, *residual),
                  "smooth_scale and residual must not overlap");
    }
    if (has_norm_output) {
      TORCH_CHECK(!tensors_overlap(*norm_output, *residual),
                  "norm_output and residual must not overlap");
    }
  }
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_dynamic_per_token_quant_sglang",
      [&] {
        auto* input_ptr = input.data_ptr<scalar_t>();
        auto* residual_ptr =
            has_residual ? residual->data_ptr<scalar_t>() : nullptr;
        auto* residual_output_ptr = residual_output.has_value()
            ? residual_output->data_ptr<scalar_t>() : residual_ptr;
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        const auto* smooth_scale_ptr = has_smooth_scale
            ? smooth_scale->data_ptr<float>()
            : nullptr;
        auto* norm_output_ptr = has_norm_output
            ? norm_output->data_ptr<scalar_t>()
            : nullptr;
        auto* out_ptr = out.data_ptr<int8_t>();
        auto* scales_ptr = scales.data_ptr<float>();
        if (hidden_size <= 512) {
          launch_rms_norm_dynamic_per_token_quant_sglang<scalar_t, 8, 64>(
              out_ptr, input_ptr, norm_output_ptr, residual_ptr, residual_output_ptr, weight_ptr,
              num_tokens,
              hidden_size, static_cast<float>(epsilon), scales_ptr,
              has_residual, update_input, limit >= 0.0, has_smooth_scale,
              smooth_scale_ptr, static_cast<float>(limit), clamp_norm_output,
              stream);
        } else if (hidden_size <= 1024) {
          launch_rms_norm_dynamic_per_token_quant_sglang<scalar_t, 8, 128>(
              out_ptr, input_ptr, norm_output_ptr, residual_ptr, residual_output_ptr, weight_ptr,
              num_tokens,
              hidden_size, static_cast<float>(epsilon), scales_ptr,
              has_residual, update_input, limit >= 0.0, has_smooth_scale,
              smooth_scale_ptr, static_cast<float>(limit), clamp_norm_output,
              stream);
        } else if (hidden_size <= 2048) {
          launch_rms_norm_dynamic_per_token_quant_sglang<scalar_t, 8, 256>(
              out_ptr, input_ptr, norm_output_ptr, residual_ptr, residual_output_ptr, weight_ptr,
              num_tokens,
              hidden_size, static_cast<float>(epsilon), scales_ptr,
              has_residual, update_input, limit >= 0.0, has_smooth_scale,
              smooth_scale_ptr, static_cast<float>(limit), clamp_norm_output,
              stream);
        } else if (hidden_size <= 4096) {
          launch_rms_norm_dynamic_per_token_quant_sglang<scalar_t, 8, 512>(
              out_ptr, input_ptr, norm_output_ptr, residual_ptr, residual_output_ptr, weight_ptr,
              num_tokens,
              hidden_size, static_cast<float>(epsilon), scales_ptr,
              has_residual, update_input, limit >= 0.0, has_smooth_scale,
              smooth_scale_ptr, static_cast<float>(limit), clamp_norm_output,
              stream);
        } else {
          launch_rms_norm_dynamic_per_token_quant_sglang<scalar_t, 8, 1024>(
              out_ptr, input_ptr, norm_output_ptr, residual_ptr, residual_output_ptr, weight_ptr,
              num_tokens,
              hidden_size, static_cast<float>(epsilon), scales_ptr,
              has_residual, update_input, limit >= 0.0, has_smooth_scale,
              smooth_scale_ptr, static_cast<float>(limit), clamp_norm_output,
              stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// GLM/DeepSeek MLA emits [q_lora, kv_lora, q_rope] in one contiguous A
// projection. q_lora and kv_lora need independent RMSNorms, while only
// q_lora is dynamically quantized for the W8A8 q_b projection. Keeping the
// two segment blocks in one launch removes both native RMSNorm chains without
// materializing contiguous copies of the sliced views.
template <typename scalar_t, int Vec, int block_size>
__global__ void mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang_kernel(
    int8_t* q_out,
    scalar_t* packed_input,
    const scalar_t* q_gamma,
    const scalar_t* kv_gamma,
    int32_t num_tokens,
    int32_t row_stride,
    int32_t q_cols,
    int32_t kv_cols,
    float q_eps,
    float kv_eps,
    float* q_scales) {
  using T_ACC = at::acc_type<scalar_t, true>;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using StoreT = at::native::memory::aligned_vector<int8_t, Vec>;
  constexpr int share_size = block_size / C10_WARP_SIZE;
  constexpr float qmax = 127.0f;
  constexpr float min_absmax = 1.0e-10f;

  __shared__ T_ACC reduce_shared[share_size];
  __shared__ T_ACC row_rstd;
  __shared__ float row_absmax;

  // This branch is uniform for each block. The first num_tokens blocks own
  // q_lora; the second num_tokens blocks own kv_lora.
  const bool is_q = blockIdx.x < num_tokens;
  const int row = is_q ? blockIdx.x : blockIdx.x - num_tokens;
  const int cols = is_q ? q_cols : kv_cols;
  const int segment_offset = is_q ? 0 : q_cols;
  const scalar_t* gamma = is_q ? q_gamma : kv_gamma;
  const float eps = is_q ? q_eps : kv_eps;
  const int tid = threadIdx.x;
  const int vec_cols = cols / Vec;
  const bool active = tid < vec_cols;
  const int64_t packed_offset =
      static_cast<int64_t>(row) * row_stride + segment_offset + tid * Vec;

  alignas(16) scalar_t input_vec[Vec];
  T_ACC norm_input[Vec];
  T_ACC square_sum = 0.0f;
  if (active) {
    *reinterpret_cast<LoadT*>(input_vec) =
        *reinterpret_cast<const LoadT*>(packed_input + packed_offset);
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
      norm_input[i] = static_cast<T_ACC>(input_vec[i]);
      square_sum += norm_input[i] * norm_input[i];
    }
  }

  square_sum =
      BlockReduceSum_ROW<T_ACC, block_size>(square_sum, reduce_shared);
  if (tid == 0) {
    row_rstd = c10::cuda::compat::rsqrt(square_sum / cols + eps);
  }
  __syncthreads();

  alignas(16) scalar_t normalized[Vec];
  float local_absmax = 0.0f;
  if (active) {
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
      const int col = tid * Vec + i;
      // Match SGLang's native path: FP32 RMS math, then activation-dtype
      // rounding. Dynamic quantization observes this rounded value.
      normalized[i] = static_cast<scalar_t>(
          norm_input[i] * row_rstd * static_cast<T_ACC>(gamma[col]));
      if (is_q) {
        local_absmax =
            fmaxf(local_absmax, fabsf(static_cast<float>(normalized[i])));
      }
    }
  }

  if (is_q) {
    local_absmax =
        BlockReduceMax_ROW<T_ACC, block_size>(local_absmax, reduce_shared);
    if (tid == 0) {
      row_absmax = fmaxf(local_absmax, min_absmax);
      q_scales[row] = row_absmax / qmax;
    }
    __syncthreads();

    if (active) {
      alignas(8) int8_t quantized[Vec];
      const float inv_scale = qmax / row_absmax;
#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        quantized[i] = float_to_int8_rn(
            static_cast<float>(normalized[i]) * inv_scale);
      }
      const int64_t q_output_offset =
          static_cast<int64_t>(row) * q_cols + tid * Vec;
      *reinterpret_cast<StoreT*>(q_out + q_output_offset) =
          *reinterpret_cast<const StoreT*>(quantized);
    }
  }

  if (active) {
    *reinterpret_cast<LoadT*>(packed_input + packed_offset) =
        *reinterpret_cast<const LoadT*>(normalized);
  }
}


template <typename scalar_t, int Vec, int block_size>
void launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
    int8_t* q_out,
    scalar_t* packed_input,
    const scalar_t* q_gamma,
    const scalar_t* kv_gamma,
    int num_tokens,
    int row_stride,
    int q_cols,
    int kv_cols,
    float q_epsilon,
    float kv_epsilon,
    float* q_scales,
    cudaStream_t stream) {
  mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang_kernel<
      scalar_t, Vec, block_size>
      <<<num_tokens * 2, block_size, 0, stream>>>(
          q_out, packed_input, q_gamma, kv_gamma, num_tokens, row_stride,
          q_cols, kv_cols, q_epsilon, kv_epsilon, q_scales);
}


void mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
    Tensor& q_out,
    Tensor& packed_input,
    const Tensor& q_weight,
    const Tensor& kv_weight,
    Tensor q_scales,
    double q_epsilon,
    double kv_epsilon) {
  TORCH_CHECK(packed_input.is_cuda(), "packed_input must be on a GPU");
  TORCH_CHECK(packed_input.dim() == 2, "packed_input must be a 2D tensor");
  TORCH_CHECK(packed_input.is_contiguous(),
              "packed_input must be contiguous");
  TORCH_CHECK(q_weight.is_contiguous(), "q_weight must be contiguous");
  TORCH_CHECK(kv_weight.is_contiguous(), "kv_weight must be contiguous");
  TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
  TORCH_CHECK(q_scales.is_contiguous(), "q_scales must be contiguous");
  TORCH_CHECK(q_weight.device() == packed_input.device(),
              "q_weight must be on packed_input.device");
  TORCH_CHECK(kv_weight.device() == packed_input.device(),
              "kv_weight must be on packed_input.device");
  TORCH_CHECK(q_out.device() == packed_input.device(),
              "q_out must be on packed_input.device");
  TORCH_CHECK(q_scales.device() == packed_input.device(),
              "q_scales must be on packed_input.device");
  TORCH_CHECK(
      packed_input.scalar_type() == at::ScalarType::Half ||
          packed_input.scalar_type() == at::ScalarType::BFloat16,
      "packed_input must be FP16 or BF16");
  TORCH_CHECK(q_weight.scalar_type() == packed_input.scalar_type(),
              "q_weight dtype must match packed_input dtype");
  TORCH_CHECK(kv_weight.scalar_type() == packed_input.scalar_type(),
              "kv_weight dtype must match packed_input dtype");
  TORCH_CHECK(q_out.scalar_type() == at::ScalarType::Char,
              "q_out must have dtype torch.int8");
  TORCH_CHECK(q_scales.scalar_type() == at::ScalarType::Float,
              "q_scales must have dtype torch.float32");
  TORCH_CHECK(std::isfinite(q_epsilon) && q_epsilon > 0.0,
              "q_epsilon must be finite and positive");
  TORCH_CHECK(std::isfinite(kv_epsilon) && kv_epsilon > 0.0,
              "kv_epsilon must be finite and positive");

  const int num_tokens = packed_input.size(0);
  const int row_stride = packed_input.size(1);
  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be a 1D tensor");
  TORCH_CHECK(kv_weight.dim() == 1, "kv_weight must be a 1D tensor");
  const int q_cols = q_weight.size(0);
  const int kv_cols = kv_weight.size(0);
  TORCH_CHECK(q_cols > 0 && q_cols <= 8192 && q_cols % 16 == 0,
              "q hidden size must be divisible by 16 and in (0, 8192]");
  TORCH_CHECK(kv_cols > 0 && kv_cols <= 8192 && kv_cols % 16 == 0,
              "kv hidden size must be divisible by 16 and in (0, 8192]");
  TORCH_CHECK(q_cols + kv_cols <= row_stride,
              "packed_input does not contain both q and kv segments");
  TORCH_CHECK(row_stride % 8 == 0,
              "packed_input row stride must preserve 16-byte alignment");
  TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == num_tokens &&
                  q_out.size(1) == q_cols,
              "q_out must have shape [num_tokens, q_hidden_size]");
  TORCH_CHECK(q_scales.dim() == 2 && q_scales.size(0) == num_tokens &&
                  q_scales.size(1) == 1,
              "q_scales must have shape [num_tokens, 1]");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(packed_input.data_ptr()) % 16 == 0,
      "packed_input must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q_weight.data_ptr()) % 16 == 0,
              "q_weight must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(kv_weight.data_ptr()) % 16 == 0,
              "kv_weight must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 8 == 0,
              "q_out must be 8-byte aligned");
  TORCH_CHECK(!tensors_overlap(packed_input, q_weight),
              "packed_input and q_weight must not overlap");
  TORCH_CHECK(!tensors_overlap(packed_input, kv_weight),
              "packed_input and kv_weight must not overlap");
  TORCH_CHECK(!tensors_overlap(packed_input, q_out),
              "packed_input and q_out must not overlap");
  TORCH_CHECK(!tensors_overlap(packed_input, q_scales),
              "packed_input and q_scales must not overlap");
  TORCH_CHECK(!tensors_overlap(q_weight, kv_weight),
              "q_weight and kv_weight must not overlap");
  TORCH_CHECK(!tensors_overlap(q_out, q_scales),
              "q_out and q_scales must not overlap");
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(packed_input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_cols = q_cols > kv_cols ? q_cols : kv_cols;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      packed_input.scalar_type(),
      "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang",
      [&] {
        auto* packed_ptr = packed_input.data_ptr<scalar_t>();
        const auto* q_weight_ptr = q_weight.data_ptr<scalar_t>();
        const auto* kv_weight_ptr = kv_weight.data_ptr<scalar_t>();
        auto* q_out_ptr = q_out.data_ptr<int8_t>();
        auto* scales_ptr = q_scales.data_ptr<float>();
        if (max_cols <= 512) {
          launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang<
              scalar_t, 8, 64>(
              q_out_ptr, packed_ptr, q_weight_ptr, kv_weight_ptr, num_tokens,
              row_stride, q_cols, kv_cols, static_cast<float>(q_epsilon),
              static_cast<float>(kv_epsilon), scales_ptr, stream);
        } else if (max_cols <= 1024) {
          launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang<
              scalar_t, 8, 128>(
              q_out_ptr, packed_ptr, q_weight_ptr, kv_weight_ptr, num_tokens,
              row_stride, q_cols, kv_cols, static_cast<float>(q_epsilon),
              static_cast<float>(kv_epsilon), scales_ptr, stream);
        } else if (max_cols <= 2048) {
          launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang<
              scalar_t, 8, 256>(
              q_out_ptr, packed_ptr, q_weight_ptr, kv_weight_ptr, num_tokens,
              row_stride, q_cols, kv_cols, static_cast<float>(q_epsilon),
              static_cast<float>(kv_epsilon), scales_ptr, stream);
        } else if (max_cols <= 4096) {
          launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang<
              scalar_t, 8, 512>(
              q_out_ptr, packed_ptr, q_weight_ptr, kv_weight_ptr, num_tokens,
              row_stride, q_cols, kv_cols, static_cast<float>(q_epsilon),
              static_cast<float>(kv_epsilon), scales_ptr, stream);
        } else {
          launch_mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang<
              scalar_t, 8, 1024>(
              q_out_ptr, packed_ptr, q_weight_ptr, kv_weight_ptr, num_tokens,
              row_stride, q_cols, kv_cols, static_cast<float>(q_epsilon),
              static_cast<float>(kv_epsilon), scales_ptr, stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}}
