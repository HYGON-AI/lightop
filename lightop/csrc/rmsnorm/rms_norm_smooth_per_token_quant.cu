#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>
#include <optional>

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


template <typename scalar_t, typename scalar_weight_t, typename scalar_out_t, typename T_ACC, bool has_weight, int Vec=4, int block_size=512, bool write_norm=false>
__global__ void rms_norm_smooth_per_token_dynamic_quant_(scalar_t* input, scalar_t* residual, scalar_weight_t* gamma, float* smooth_scale, int32_t const cols, float const eps, scalar_out_t* out, float* scales, scalar_t* after_norm, int stride, int num_tokens)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale;
  
  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
      T_ACC val = 0;
      T_ACC trstd;
      int64_t idx = bid * tcol + tid;
      idx *= Vec;
      int64_t index = bid * stride + tid * Vec;
      scalar_t intput_vec[Vec];

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
            
            T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[jj]) : static_cast<T_ACC>(1.0f);
            
            elem[ii] = static_cast<T_ACC>(intput_vec[ii]) * trstd * weight_val * static_cast<T_ACC>(smooth_scale[jj]);
            block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
          }      
      }
      block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
      
      float scale = 0.0f;    
      if(tid == 0)
      {
         scale = block_absmax_val_maybe;
         s_token_scale = scale; 
      }
      __syncthreads();
      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

      scalar_out_t out_vec[Vec];
      if (tid < tcol) {
        #pragma unroll
        for(int ii = 0;ii < Vec;ii++){
            out_vec[ii] = float_to_int8_rn(elem[ii] * inv_s);
          }
          constexpr float qmax = 127.0f;
          constexpr float min_scale = 1.19209e-07f;
          *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;
          
          if constexpr (write_norm) {
            *(LoadT*)(after_norm+index)=*(LoadT*)elem;
          }
          
          if(tid==0)scales[bid] = fmaxf(scale / qmax, min_scale);
      }
      __syncthreads();
  }
}

template <typename scalar_t, typename scalar_weight_t, typename scalar_out_t, typename T_ACC, bool has_weight, int Vec, int block_size, bool write_norm=false>
__global__ void rms_norm_smooth_per_token_dynamic_quant_loop(scalar_t* input, scalar_t* residual, scalar_weight_t* gamma, float* smooth_scale, int32_t const cols, float const eps, scalar_out_t* out, float* scales, scalar_t* after_norm, int stride, int num_tokens)
{
  constexpr int share_size=block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale; 

  int tid = threadIdx.x;
  
  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
      T_ACC val = 0.0f;
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
        T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[i]) : static_cast<T_ACC>(1.0f);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(input[i + rows]) * trstd * weight_val * static_cast<T_ACC>(smooth_scale[i])));
      }
      block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe,val_shared);

      float scale = 0.0f;
      if(tid == 0)
      {
          scale = block_absmax_val_maybe;
        constexpr float qmax = 127.0f;
        constexpr float min_scale = 1.19209e-07f;
        s_token_scale = scale;
        scales[bid] = fmaxf(scale / qmax, min_scale); 
      }
      __syncthreads();
      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
      for(int i = tid; i < cols; i += block_size)
      {
        T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[i]) : static_cast<T_ACC>(1.0f);
        T_ACC sm_val = static_cast<T_ACC>(input[rows + i]) *trstd* weight_val * static_cast<T_ACC>(smooth_scale[i]);
        out[rows + i] = float_to_int8_rn(sm_val * inv_s);
        if constexpr (write_norm) {
            after_norm[rows + i] = static_cast<scalar_t>(sm_val);
        }
      }
      __syncthreads();
  }
}


template <typename scalar_t, typename scalar_weight_t, typename scalar_out_t, typename T_ACC, bool has_weight, int Vec=4, int block_size=512, bool write_norm=false, bool write_res=true>
__global__ void rms_norm_smooth_per_token_dynamic_quant_residual(scalar_t* input, scalar_t* residual, scalar_weight_t* gamma, float* smooth_scale, int cols, float const eps, scalar_out_t* out, float* scales, scalar_t* after_norm, int stride, scalar_t* residual_out, int num_tokens)
{
  constexpr int share_size=block_size/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale; 

  int tid = threadIdx.x;
  int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;
  
  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
      T_ACC val = 0;
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
            T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[jj]) : static_cast<T_ACC>(1.0f);
            elem[ii] = static_cast<T_ACC>(residual_vec[ii]) * trstd * weight_val * static_cast<T_ACC>(smooth_scale[jj]);
            block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(elem[ii]));
          }      
      }
      block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);
      float scale = 0.0f;    
      if(tid == 0)
      {
          scale = block_absmax_val_maybe ;
        s_token_scale = scale;
      }
      __syncthreads();
      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
      scalar_out_t out_vec[Vec];
      if (tid < tcol) {
      #pragma unroll
      for(int ii = 0;ii < Vec;ii++){
          out_vec[ii] = float_to_int8_rn(elem[ii] * inv_s);
        }
      constexpr float qmax = 127.0f;
      constexpr float min_scale = 1.19209e-07f;  
      
      if(tid==0)scales[bid] = fmaxf(scale / qmax, min_scale);
        if constexpr (write_res) {
            *(LoadT*)(residual_out+idx) = *(LoadT*)residual_vec;
        }
        
        *(LoadT_out*)(out+idx) =*(LoadT_out*)out_vec;

      if constexpr (write_norm) {
        *(LoadT*)(after_norm+idx)=*(LoadT*)elem;  
      }
      }  
      __syncthreads();
  }
}

template <typename scalar_t, typename scalar_weight_t, typename scalar_out_t, typename T_ACC, bool has_weight, int Vec=1, int block_size=512, bool write_norm=false, bool write_res=true>
__global__ void rms_norm_smooth_per_token_dynamic_quant_residual_loop(scalar_t* input, scalar_t* residual, scalar_weight_t* gamma, float* smooth_scale, int cols, float const eps, scalar_out_t* out, float* scales, scalar_t* after_norm, int stride, scalar_t* residual_out, int num_tokens)
{
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale; 

  int tid = threadIdx.x;
  
  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
      T_ACC val = 0.0f;
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
        T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[i]) : static_cast<T_ACC>(1.0f);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe,fabs(static_cast<T_ACC>(static_cast<T_ACC>(x * trstd) * weight_val) * static_cast<T_ACC>(smooth_scale[i])));
      }

      block_absmax_val_maybe = BlockReduceMax_ROW<T_ACC,block_size>(block_absmax_val_maybe, val_shared);

       float scale = 0.0f;
      if(tid == 0)
      {
          scale = block_absmax_val_maybe;
        constexpr float qmax = 127.0f;
        constexpr float min_scale = 1.19209e-07f;
        s_token_scale = scale;
        scales[bid] = fmaxf(scale / qmax, min_scale);
      }
       __syncthreads();
      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
      for(int i = tid;i < cols;i += block_size)
      {
        T_ACC weight_val = has_weight ? static_cast<T_ACC>(gamma[i]) : static_cast<T_ACC>(1.0f);
        T_ACC res_val = static_cast<T_ACC>(residual[rows + i]) + static_cast<T_ACC>(input[rows + i]);

        if constexpr (write_res) {
            residual_out[rows + i] = res_val;
        }
        
        T_ACC sm_val = res_val * trstd * weight_val * static_cast<T_ACC>(smooth_scale[i]);
        out[rows + i] = float_to_int8_rn(sm_val * inv_s);
        if constexpr (write_norm) {
            after_norm[rows + i] = static_cast<scalar_t>(sm_val);
        }
      }
      __syncthreads();
  }
}

template <typename scalar_in_t, typename scalar_weight_t, bool has_weight>
void rms_norm_smooth_per_token_dynamic_quant_impl(
  torch::Tensor& out,           
  torch::Tensor& input,   
  std::optional<at::Tensor> const& weight, 
  torch::Tensor const& smooth_scale, 
  torch::Tensor& scales,        
  double const epsilon,     
  std::optional<at::Tensor> residual,
  std::optional<at::Tensor> after_norm,
  std::optional<at::Tensor> after_residual, 
  bool update_input,
  bool update_res) {  

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  
  scalar_weight_t* weight_data = nullptr;
  bool ptrs_are_aligned = (inp_ptr % 16 == 0);
  
  if constexpr (has_weight) {
      auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight->data_ptr());
      ptrs_are_aligned = ptrs_are_aligned && (wt_ptr % 16 == 0);
      weight_data = weight->data_ptr<scalar_weight_t>();
  }

  scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
  
  bool write_norm = after_norm.has_value() || update_input;
  scalar_in_t* after_norm_data = nullptr;

  if (after_norm.has_value()) {
      auto after_ptr = reinterpret_cast<std::uintptr_t>(after_norm->data_ptr());
      ptrs_are_aligned = ptrs_are_aligned && (after_ptr % 16 == 0);
      after_norm_data = after_norm->data_ptr<scalar_in_t>();
  } else if (update_input) {
      after_norm_data = self_data;
  }
  
  using T_ACC = at::acc_type<scalar_in_t, true>;
  float* smooth_data = smooth_scale.data_ptr<float>(); 
  float* scaless = scales.data_ptr<float>();

  bool residual_flag = residual.has_value();
  scalar_in_t* other_data = residual_flag ? residual->data_ptr<scalar_in_t>() : nullptr;


  bool write_res = false;
  scalar_in_t* residual_out_data = nullptr;
  if (residual_flag) {
      if (update_res) {
          residual_out_data = other_data; 
          write_res = true;
      } else if (after_residual.has_value()) {
          auto after_res_ptr = reinterpret_cast<std::uintptr_t>(after_residual->data_ptr());
          ptrs_are_aligned = ptrs_are_aligned && (after_res_ptr % 16 == 0);
          residual_out_data = after_residual->data_ptr<scalar_in_t>();
          write_res = true;
      }
  }

  int stride = (input.dim() > 1) ? input.stride(-2) : input.stride(-1);
  bool is_aligned = (hidden_size % 16 == 0) && (hidden_size <= 8192) && ptrs_are_aligned;

  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int target_blocks = num_tokens;
  if(!after_residual.has_value() && !after_norm.has_value()){
    if(num_tokens == 352) target_blocks = 320;
    if(num_tokens == 512) target_blocks = 480;
    if(num_tokens == 1024 || num_tokens == 1152) target_blocks = 1040;
  }
  if(after_residual.has_value()){
     if(num_tokens == 352) target_blocks = 320;
     if(num_tokens == 416) target_blocks = 320;
     if(num_tokens >= 480 && num_tokens <= 512)  target_blocks = 480;
     if(num_tokens >= 640 && num_tokens <= 896)  target_blocks = 640;
  }
  if(after_norm.has_value()){
    if(num_tokens == 1024 || num_tokens == 1152 || num_tokens == 1408) target_blocks = 1040;
  }
  

#define LAUNCH_ALIGNED(VEC, THREADS, WRITE_FLAG) \
  if (residual_flag) { \
      if (write_res) { \
          rms_norm_smooth_per_token_dynamic_quant_residual<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG, true><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, residual_out_data, num_tokens); \
      } else { \
          rms_norm_smooth_per_token_dynamic_quant_residual<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG, false><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, residual_out_data, num_tokens); \
      } \
  } else { \
      rms_norm_smooth_per_token_dynamic_quant_<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, num_tokens); \
  }

#define LAUNCH_LOOP(VEC, THREADS, WRITE_FLAG) \
  if (residual_flag) { \
      if (write_res) { \
          rms_norm_smooth_per_token_dynamic_quant_residual_loop<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG, true><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, residual_out_data, num_tokens); \
      } else { \
          rms_norm_smooth_per_token_dynamic_quant_residual_loop<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG, false><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, residual_out_data, num_tokens); \
      } \
  } else { \
      rms_norm_smooth_per_token_dynamic_quant_loop<scalar_in_t, scalar_weight_t, scalar_out_t, T_ACC, has_weight, VEC, THREADS, WRITE_FLAG><<<target_blocks, THREADS, 0, stream>>>(self_data, other_data, weight_data, smooth_data, hidden_size, epsilon, out_data, scaless, after_norm_data, stride, num_tokens); \
  }

  AT_DISPATCH_INTEGRAL_TYPES(out.scalar_type(), "fused_add_rms_norm_smooth_kernel_out_type", [&] {
    using scalar_out_t = scalar_t;
    scalar_out_t* out_data = out.data_ptr<scalar_out_t>();

    if (is_aligned) {
      if (write_norm) {
        if (hidden_size <= 512) { LAUNCH_ALIGNED(8, 64, true); }
        else if (hidden_size <= 1024) { LAUNCH_ALIGNED(8, 128, true); }
        else if (hidden_size <= 2048) { LAUNCH_ALIGNED(8, 256, true); }
        else if (hidden_size <= 4096) { 
          if(num_tokens == 512 || num_tokens == 1152) {
              LAUNCH_ALIGNED(4, 1024, true);
          }else{
              LAUNCH_ALIGNED(16, 256, true);
          }
         }
        else { LAUNCH_ALIGNED(8, 1024, true); } 
      } 
      else {
        if (hidden_size <= 512) { LAUNCH_ALIGNED(8, 64, false); }
        else if (hidden_size <= 1024) { LAUNCH_ALIGNED(8, 128, false); }
        else if (hidden_size <= 2048) { LAUNCH_ALIGNED(8, 256, false); }
        else if (hidden_size <= 4096) {
            if(num_tokens == 512 || num_tokens == 1152) {
                LAUNCH_ALIGNED(4, 1024, false);
            }else{
                LAUNCH_ALIGNED(16, 256, false);
            }
         }
        else { LAUNCH_ALIGNED(8, 1024, false); } 
      }
    } 
    else {
      if (write_norm) {
        if (hidden_size <= 512) { LAUNCH_LOOP(1, 64, true); }
        else if (hidden_size <= 1024) { LAUNCH_LOOP(1, 128, true); }
        else if (hidden_size <= 2048) { LAUNCH_LOOP(1, 256, true); }
        else if (hidden_size <= 4096) { 
           if(num_tokens == 512 || num_tokens == 1152) {
              LAUNCH_ALIGNED(4, 1024, true);
            }else{
              LAUNCH_ALIGNED(16, 256, true);
            }
         }
        else { LAUNCH_LOOP(1, 1024, true); }
      } else {
        if (hidden_size <= 512) { LAUNCH_LOOP(1, 64, false); }
        else if (hidden_size <= 1024) { LAUNCH_LOOP(1, 128, false); }
        else if (hidden_size <= 2048) { LAUNCH_LOOP(1, 256, false); }
        else if (hidden_size <= 4096) { LAUNCH_LOOP(1, 512, false); }
        else { LAUNCH_LOOP(1, 1024, false); }
      }
    }
  });

#undef LAUNCH_ALIGNED
#undef LAUNCH_LOOP
}

/*
字节：
    1.residual=none， after_norm=none，after_residual=none，update_input=false，update_res=false
    2.residual传进来，after_norm=none，after_residual外部申请空间传进来，update_input=false，update_res=false
    3.residual=none， after_norm外部申请空间传进来，after_residual=none，update_input=false，update_res=false
*/
void rms_norm_smooth_per_token_dynamic_quant(
  Tensor &out,                   // quant_tokens 输出 (int8)
  Tensor& input,                 // input 输入 (fp16/bf16)
  std::optional<Tensor> weight,  // RMS Norm weight / gamma (fp16/bf16/fp32)
  const Tensor& smooth_scale,    // 平滑因子 (fp32)
  Tensor scales,                 // per_token_scale 输出 (fp32)
  double epsilon,
  std::optional<Tensor> residual,
  std::optional<Tensor> after_norm,     // after_norm 接收归一化后的数据输出
  std::optional<Tensor> after_residual, //接收残差计算后的输出
  bool update_input,                // 是否原地写回 input
  bool update_res                   // 是否原地写回 res
) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_smooth_quant_kernel_input",
    [&] {
      using scalar_in_t = scalar_t;
      if (weight.has_value()) {
          AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            weight->scalar_type(),
            "fused_add_rms_norm_smooth_quant_kernel_weight",
            [&] {
              using scalar_weight_t = scalar_t;
              rms_norm_smooth_per_token_dynamic_quant_impl<scalar_in_t, scalar_weight_t, true>(
                out, input, weight, smooth_scale, scales, epsilon, residual, after_norm, after_residual, update_input, update_res 
              );
            });
      } else {
          rms_norm_smooth_per_token_dynamic_quant_impl<scalar_in_t, float, false>(
            out, input, std::nullopt, smooth_scale, scales, epsilon, residual, after_norm, after_residual, update_input, update_res 
          );
      }
    });
}

}} // namespace at::native