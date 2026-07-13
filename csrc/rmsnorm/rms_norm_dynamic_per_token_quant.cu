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


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512, bool inpit_update=false>
__global__ void rms_norm_dynamic_per_token_quant_(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps,scalar_out_t* out,float* scales, int stride)
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

template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec,int block_size>
__global__ void rms_norm_dynamic_per_token_quant_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int32_t const cols,float const eps, scalar_out_t* out,float* scales, int stride)
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
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(input[i + rows]) * trstd * static_cast<T_ACC>(gamma[i])));
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
    out[rows + i] = float_to_int8_rn(static_cast<T_ACC>(input[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]) * inv_s);
    input[rows + i] = static_cast<T_ACC>(input[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]);
  }
}


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=4,int block_size=512,bool inpit_update=false>
__global__ void rms_norm_dynamic_per_token_quant_residual(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride)
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


template <typename scalar_t,typename scalar_out_t,typename T_ACC,int Vec=1,int block_size=512>
__global__ void rms_norm_dynamic_per_token_quant_residual_loop(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,float const eps, scalar_out_t* out,float* scales, int stride)
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
    T_ACC x = static_cast<T_ACC>(residual[i + rows]) + static_cast<T_ACC>(input[i + rows]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(static_cast<T_ACC>(x * trstd) * gamma[i])));
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
    residual[rows + i] = static_cast<T_ACC>(residual[i + rows]) + static_cast<T_ACC>(input[i + rows]);
    out[rows + i] = float_to_int8_rn(static_cast<T_ACC>(residual[rows + i] ) * trstd * static_cast<T_ACC>(gamma[i]) * inv_s);
    input[rows + i] = static_cast<T_ACC>(residual[rows + i]) *trstd* static_cast<T_ACC>(gamma[i]);
  }

}


template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_(
  torch::Tensor& out,           // [..., hidden_size]
  torch::Tensor& input,   // [..., hidden_size]
  torch::Tensor const& weight,  // [hidden_size]
  torch::Tensor& scales,        // [num_tokens]
  double const epsilon,     // Variance epsilon used in norm calculation
  std::optional<at::Tensor> residual,
  bool input_flag) {
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
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else if(num_tokens <= 3072){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,true><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                  
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 1000){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  }
              else if(num_tokens < 8000){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }  
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                else{
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }
            else{
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }
          else if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,128,true><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,256,true><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,512,true><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }
            else{
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,true><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
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
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else if(num_tokens < 4096){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,64,false><<<num_tokens,  64, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              
                  }
            else if (hidden_size <= 1024){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  }
            else if(hidden_size <= 2048){
              if(num_tokens < 512){
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,256,false><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  } 
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              
            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,4,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                else{
                    rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
            }
            else if(hidden_size <= 8192){
              if(num_tokens <= 4){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else if(num_tokens < 64){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else if(num_tokens < 290){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else if(num_tokens < 2000){
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else{
                rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                  
          }
            else{
                  rms_norm_dynamic_per_token_quant_residual<scalar_in_t,scalar_out_t,T_ACC,16,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }

        }else{
             if (hidden_size <= 512){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }
          else if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,128,false><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
          }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,256,false><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                      }              

            }
            else if(hidden_size <= 4096){
                if(num_tokens == 512){
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,4,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                }
                else{
                    rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,512,false><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
            }
            else if(hidden_size <= 8192){
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,8,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }
            else{
                  rms_norm_dynamic_per_token_quant_<scalar_in_t,scalar_t,T_ACC,16,1024,false><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
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
                rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }
            else if(hidden_size <= 2048){
              if(num_tokens >= 1024){
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
              else{
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }  
              
            }
            else if(hidden_size <= 4096){
              if(num_tokens > 1200){
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  
              }
              else{
                  rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
              }
            }
            else {
                rms_norm_dynamic_per_token_quant_residual_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }

        }else{
            if (hidden_size <= 1024){
                rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
            }
            else if(hidden_size <= 2048){
                if(num_tokens >= 1024){
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                }
                else{
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                }                      

            }
            else if(hidden_size <= 4096){
                if(num_tokens > 1200){
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                  }
                else{
                    rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
                }
            }
            else{
                  rms_norm_dynamic_per_token_quant_loop<scalar_in_t,scalar_t,T_ACC,1,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,epsilon,out_data,scaless,stride);
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
  bool update_input
) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {
      rms_norm_dynamic_per_token_quant_<scalar_t>(out,input,weight,scales,epsilon,residual,update_input);
    });
}
}}
