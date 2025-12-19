#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
#include "reduce.h"

// #include "cuda_compat.h"
// #include "../dispatch_utils.h"

namespace at {
namespace native {

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

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t,
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? silu_kernel(x) * y : x * silu_kernel(y);
}
// Activation and gating kernel template.
template <typename scalar_t, 
          bool act_first>
__global__ void act_and_mul_kernel(
    int8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, act_first>(x, y);
  }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1(
    int64_t num_tokens,
    int8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){
  // for(; token_idx < num_tokens ;token_idx += gridDim.x){
    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecInt8Type* y = (VecInt8Type*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_y[VEC];
    if (idx < d) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
      }
    }
    float row_max = 0.f;
    if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
        row_max = fmaxf(row_max,fabsf(r_y[ii]));
      }
    }

      row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);
      __shared__ float s_token_scale; 
      float scale = 0.0f;     
      if(tidx == 0){
        scale = row_max;
        // constexpr float qmax = 127.0f;
        // constexpr float min_scale = 1.19209e-07f;
        s_token_scale = scale;  
        scales[token_idx] = s_token_scale / 127.f;  
      }
      __syncthreads();

      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

      int8_t out_vec[VEC];
      if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
        }
      *y =*(VecInt8Type*)out_vec;
      
    }
  }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1_ep(
    int64_t num_tokens,
    int8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    int32_t *expert_ids
    ) {

    int64_t token_idx= blockIdx.x;
    if(expert_ids != nullptr){
      if(expert_ids[token_idx] == -1)  return;
    }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecInt8Type* y = (VecInt8Type*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_y[VEC];
    if (idx < d) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
      }
    }
    float row_max = 0.f;
    if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
        row_max = fmaxf(row_max,fabsf(r_y[ii]));
      }
    }

      row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);
      __shared__ float s_token_scale; 
      float scale = 0.0f;     
      if(tidx == 0){
        scale = row_max;
        // constexpr float qmax = 127.0f;
        // constexpr float min_scale = 1.19209e-07f;
        s_token_scale = scale;  
        scales[token_idx] = s_token_scale / 127.f;  
      }
      __syncthreads();

      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

      int8_t out_vec[VEC];
      if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
        }
      *y =*(VecInt8Type*)out_vec;
      
    }
  
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1_8192(
    int64_t num_tokens,
    int8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor
    ) {
   int64_t token_idx= blockIdx.x;
   int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(token_idx >= num_local_tokens_tensor[0] * topK)  return;
  }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecInt8Type* y = (VecInt8Type*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_y[VEC];
    if (idx < d) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
      }
    }
    float row_max = 0.f;
    if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
        row_max = fmaxf(row_max,fabsf(r_y[ii]));
      }
    }

      row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);
      __shared__ float s_token_scale; 
      float scale = 0.0f;     
      if(tidx == 0){
        scale = row_max;
        // constexpr float qmax = 127.0f;
        // constexpr float min_scale = 1.19209e-07f;
        s_token_scale = scale;  
        scales[token_idx] = s_token_scale / 127.f;  
      }
      __syncthreads();

      float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

      int8_t out_vec[VEC];
      if (idx < d) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
        }
      *y =*(VecInt8Type*)out_vec;
      
    }
  // }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt2(
    int64_t num_tokens,
    int8_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    float* __restrict__ scales,
    const int d,
    const int topK,
    int32_t *num_local_tokens_tensor) {
  int64_t token_idx = blockIdx.x;
  int num_local_tokens = num_tokens;
  if(num_local_tokens_tensor != nullptr){
    if(num_local_tokens >= num_local_tokens_tensor[0] * topK) num_local_tokens =  num_local_tokens_tensor[0] * topK;
    if(token_idx >= num_local_tokens)  return;    
  }

  for(; token_idx < num_local_tokens ;token_idx += gridDim.x){

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;

    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecInt8Type* y = (VecInt8Type*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_y[VEC];

    for (int tt = tidx; tt < d; tt += blockDim.x * VEC) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
  #pragma unroll
      for (int i = 0; i < VEC; i++) {
        r_y[i] = static_cast<float>(silu_kernel(r_x1[i]) * r_x2[i]);
      }
    }
      
    float row_max = 0.f;
    if (idx < d){
      #pragma unroll
      for(int ii = 0;ii < VEC; ii++){
        row_max = fmaxf(row_max,fabs(r_y[ii]));
      }
    }  

    row_max = BlockReduceMax_ROW<float,block_size>(row_max, val_shared);
    __shared__ float s_token_scale; 
    float scale = 0.0f;     
    if(tidx == 0){
      scale = row_max;
      constexpr float qmax = 127.0f;
      constexpr float min_scale = 1.19209e-07f;
      s_token_scale = scale;               
      scales[token_idx] = fmaxf(scale / qmax, min_scale);  
    }
    __syncthreads();

    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

    int8_t out_vec[VEC];
    for (int tt = tidx; tt < d; tt += blockDim.x * VEC) {
      #pragma unroll
      for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
        }
      *y =*(VecInt8Type*)out_vec;
    }  
}
}


// template <typename T>
// __device__ __forceinline__ T gelu_kernel(const T& x) {
//   // Equivalent to PyTorch GELU with 'none' approximation.
//   // Refer to:
//   // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
//   const float f = (float)x;
//   constexpr float ALPHA = M_SQRT1_2;
//   return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
// }

// template <typename T>
// __device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
//   // Equivalent to PyTorch GELU with 'tanh' approximation.
//   // Refer to:
//   // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
//   const float f = (float)x;
//   constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
//   constexpr float KAPPA = 0.044715;
//   float x_cube = f * f * f;
//   float inner = BETA * (f + KAPPA * x_cube);
//   return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
// }
    
// }  // namespace vllm


// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(ACT_FIRST)                                  \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  int grid_opt = num_tokens;                                             \
  if(num_local_tokens_tensor.has_value()){\
    if(expert_m != -1){\
      grid_opt = std::max(1, expert_m * topK);\
    }\
    else{\
      if(num_tokens < 8192){                                                \
        grid_opt = std::max(1, num_tokens);                                   \
      }else if(num_tokens < 16384){                                            \
        grid_opt = std::max(1, num_tokens/4);                                  \
      }else{                                                                 \
        grid_opt = std::max(1, num_tokens/8);                                      \
      }                                                                          \
    }\
  }                                                                        \
  dim3 grid(grid_opt);                                                   \
  dim3 block(std::min(d, 1024));                                               \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "act_and_mul_kernel", [&] {                         \
      if(expert_ids.has_value()){\
        if (0 == d % 8 && d <= 16384) {                                        \
          if (d <= 512) {                                                      \
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 2, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, \
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 1024) {                                              \
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, ACT_FIRST, 128> \
                <<<grid, 128, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d,\
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 2048) {                                              \
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d,\
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 4096) {                                              \
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, ACT_FIRST, 512> \
                <<<grid, 512, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(),      \
                                           scales.data_ptr<float>(),d, \
                                           expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);     \
          } else {                                                             \
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, ACT_FIRST, 1024> \
                <<<grid, 1024, 0, stream>>>(num_tokens ,\
                                            out.data_ptr<int8_t>(),          \
                                            input.data_ptr<scalar_t>(),     \
                                            scales.data_ptr<float>(),d, \
                                            expert_ids.has_value() ? expert_ids->data_ptr<int32_t>() : nullptr);    \
          }                                                                    \
        }\
        else {                                                               \
          printf("ERROR:fuse_silu_mul_quant.cu don't support this size.\n");\
        } \
      }\
      if(num_tokens < 8192 && expert_m == -1){\
        if (0 == d % 8 && d <= 16384) {                                        \
          if (d <= 512) {                                                      \
            act_and_mul_quant_kernel_opt1_8192<scalar_t, 2, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,                  \
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 1024) {                                              \
            act_and_mul_quant_kernel_opt1_8192<scalar_t, 8, ACT_FIRST, 128> \
                <<<grid, 128, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 2048) {                                              \
            act_and_mul_quant_kernel_opt1_8192<scalar_t, 8, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 4096) {                                              \
            act_and_mul_quant_kernel_opt1_8192<scalar_t, 8, ACT_FIRST, 512> \
                <<<grid, 512, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(),      \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else {                                                             \
            act_and_mul_quant_kernel_opt1_8192<scalar_t, 8, ACT_FIRST, 1024> \
                <<<grid, 1024, 0, stream>>>(num_tokens ,\
                                            out.data_ptr<int8_t>(),          \
                                            input.data_ptr<scalar_t>(),     \
                                            scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);    \
          }                                                                    \
        } else {                                                               \
          printf("ERROR:fuse_silu_mul_quant.cu don't support this size.\n");\
        } \
      }                                                                     \
        else{\
        if (0 == d % 8 && d <= 16384) {                                        \
          if (d <= 512) {                                                      \
            act_and_mul_quant_kernel_opt1<scalar_t, 2, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 1024) {                                              \
            act_and_mul_quant_kernel_opt1<scalar_t, 8, ACT_FIRST, 128> \
                <<<grid, 128, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 2048) {                                              \
            act_and_mul_quant_kernel_opt1<scalar_t, 8, ACT_FIRST, 256> \
                <<<grid, 256, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(), \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else if (d <= 4096) {                                              \
            act_and_mul_quant_kernel_opt1<scalar_t, 8, ACT_FIRST, 512> \
                <<<grid, 512, 0, stream>>>(num_tokens ,\
                                           out.data_ptr<int8_t>(),           \
                                           input.data_ptr<scalar_t>(),      \
                                           scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);     \
          } else {                                                             \
            act_and_mul_quant_kernel_opt1<scalar_t, 8, ACT_FIRST, 1024> \
                <<<grid, 1024, 0, stream>>>(num_tokens ,\
                                            out.data_ptr<int8_t>(),          \
                                            input.data_ptr<scalar_t>(),     \
                                            scales.data_ptr<float>(),d, topK, num_local_tokens_tensor.has_value() ? num_local_tokens_tensor->data_ptr<int32_t>() : nullptr);    \
          }                                                                    \
        } else {                                                               \
          printf("ERROR:fuse_silu_mul_quant.cu don't support this size.\n");\
        } \
        }\
      });

void fuse_silu_mul_quant(torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids)
{
  LAUNCH_ACTIVATION_GATE_KERNEL(true);
}

// void mul_and_silu_opt(torch::Tensor& out,    // [..., d]
//                   torch::Tensor& input)  // [..., 2 * d]
// {
//   // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
//   // applies the silu to the latter half of the input.
//   LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
// }

// void gelu_and_mul_opt(torch::Tensor& out,    // [..., d]
//                   torch::Tensor& input)  // [..., 2 * d]
// {
//   LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
// }

// void gelu_tanh_and_mul_opt(torch::Tensor& out,    // [..., d]
//                        torch::Tensor& input)  // [..., 2 * d]
// {
//   LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
// }

template <typename T>
__device__ __forceinline__ T silu_kernel_opt(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + __expf((float)-x)));
}

template <typename scalar_t>
__launch_bounds__(1024,1)
__global__ void silu_and_mul_base(scalar_t * __restrict__ output, const scalar_t * __restrict__ input, int d)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(col < d){
        scalar_t x1 = *(input + row * d * 2 + col);
        scalar_t y1 = *(input + row * d * 2 + col + d);
        *(output + row * d + col) = silu_kernel_opt(x1) * y1;
    }

    return;
}

template <typename scalar_t, int VEC>
__global__ void silu_and_mul_kernel_opt1(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  const int64_t token_idx= blockIdx.x;
  int idx = threadIdx.x * VEC;
  if (idx < d) {
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecType* y = (VecType*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    scalar_t r_y[VEC];
    *(VecType*)r_x1 = *x1;
    *(VecType*)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = silu_kernel_opt(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template <typename scalar_t, int VEC>
__global__ void silu_and_mul_kernel_opt2(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  const int64_t token_idx = blockIdx.x;
  int idx = threadIdx.x * VEC;
  for (; idx < d; idx += blockDim.x * VEC) {
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecType* y = (VecType*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    scalar_t r_y[VEC];
    *(VecType*)r_x1 = *x1;
    *(VecType*)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = silu_kernel_opt(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template <typename scalar_t, int VEC, int D,int THREAD_PER_ROW, int ROWS_PER_BLOCK>
__launch_bounds__(1024,1)
__global__ void silu_and_mul_opt3(scalar_t * __restrict__ output, const scalar_t * __restrict__ input)
{
    int t_row = threadIdx.x / THREAD_PER_ROW;
    int t_col = threadIdx.x % THREAD_PER_ROW;
    const int64_t row = blockIdx.x * ROWS_PER_BLOCK + t_row;
    const int64_t idx = t_col * VEC;
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    if(idx < D){
        const int64_t x_index = row * 2 * D + idx;
        const int64_t y_index = row * D + idx;
        VecType* x1 = (VecType*)(input + x_index);
        VecType* x2 = (VecType*)(input + x_index + D);
        VecType* y = (VecType*)(output + y_index);
        scalar_t r_x1[VEC];
        scalar_t r_x2[VEC];
        scalar_t r_y[VEC];
        *(VecType*)r_x1 = *x1;
        *(VecType*)r_x2 = *x2;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            r_y[i] = silu_kernel_opt(r_x1[i]) * r_x2[i];
        }
        *y = *(VecType*)r_y;
    }
    return;
}

void fuse_silu_and_mul(torch::Tensor& input, torch::Tensor& out)
{
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input)); 
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "fuse_silu_and_mul", [&]{
      if(384 == d){
        constexpr int VEC = 2;
        constexpr int THREAD_PER_ROW = 192;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 384, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }else if(80 == d){
        constexpr int VEC = 2;
        constexpr int THREAD_PER_ROW = 64;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 80, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }else if(160 == d){
        constexpr int VEC = 4;
        constexpr int THREAD_PER_ROW = 64;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 160, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }else if(192 == d){
        constexpr int VEC = 4;
        constexpr int THREAD_PER_ROW = 64;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 192, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }else if(96 == d){
        constexpr int VEC = 2;
        constexpr int THREAD_PER_ROW = 64;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 96, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }
      else if(48 == d){
        constexpr int VEC = 1;
        constexpr int THREAD_PER_ROW = 64;
        constexpr int ROWS_PER_BLOCK = 4;
        if(num_tokens >= 1024 & 0 == (num_tokens % ROWS_PER_BLOCK)){
          dim3 grid(num_tokens / ROWS_PER_BLOCK);
          dim3 block(THREAD_PER_ROW * ROWS_PER_BLOCK);
          silu_and_mul_opt3<scalar_t, VEC, 48, THREAD_PER_ROW, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
        }else{
          dim3 grid(num_tokens);
          dim3 block(d);
          silu_and_mul_base<<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      }
      else{
        dim3 grid(num_tokens);
        dim3 block(std::min(d, 1024));
        if (0 == d % 8 && d <= 16384) {                                        
          if (d <= 512) {                                                      
            silu_and_mul_kernel_opt1<scalar_t, 2> 
                <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),           
                                           input.data_ptr<scalar_t>(), d);     
          } else if (d <= 1024) {                                              
            silu_and_mul_kernel_opt1<scalar_t, 8>
                <<<grid, 128, 0, stream>>>(out.data_ptr<scalar_t>(),           
                                           input.data_ptr<scalar_t>(), d);     
          } else if (d <= 2048) {                                              
            silu_and_mul_kernel_opt1<scalar_t, 8> 
                <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),           
                                           input.data_ptr<scalar_t>(), d);     
          } else if (d <= 4096) {                                              
            silu_and_mul_kernel_opt1<scalar_t, 8> 
                <<<grid, 512, 0, stream>>>(out.data_ptr<scalar_t>(),           
                                           input.data_ptr<scalar_t>(), d);     
          } else {                                                             
            silu_and_mul_kernel_opt2<scalar_t, 8> 
                <<<grid, 1024, 0, stream>>>(out.data_ptr<scalar_t>(),          
                                            input.data_ptr<scalar_t>(), d);    
          }                                                                    
        } else {                                                               
            silu_and_mul_base             
                <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       
                                            input.data_ptr<scalar_t>(), d);  
        }
      }
    }
  );
}

}
}