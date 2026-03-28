#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <cmath>



namespace at {
namespace native {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = input[token_idx * 2 * d + idx];
    const scalar_t y = input[token_idx * 2 * d + d + idx];
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC,
          bool act_first>
__global__ void act_and_mul_kernel_opt1(
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
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC,
          bool act_first>
__global__ void act_and_mul_kernel_opt2(
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
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

// 一个block处理 blockDim.x * VEC 个元素
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC,
          bool act_first>
__global__ void act_and_mul_kernel_opt3(
    scalar_t* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    const int64_t total_elements,        
    const int d) {
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  
  int64_t global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  int64_t stride = gridDim.x * blockDim.x * VEC;

  for (; global_idx < total_elements; global_idx += stride) {
    const int64_t token_idx = global_idx / d;
    const int64_t idx = global_idx % d; 
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = global_idx; 

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
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC,
          bool act_first>
__global__ void act_and_mul_kernel_opt4(
    scalar_t* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    const int d) {
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  
  const int64_t token_idx = blockIdx.y;
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  int64_t stride = gridDim.x * blockDim.x * VEC;

  for (; idx < d; idx += stride) {
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
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // return (T)(((float)x) / (1.0f + __expf((float)-x)));

  constexpr float LOG2E = 1.44269504088896340736f;
  return (T)(((float)x) / (1.0f + __builtin_amdgcn_exp2f(-((float)x) * LOG2E)));
}


template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + std::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + std::tanhf(inner)));
}


#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                       \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min((int)d, 1024));                                          \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                             \
      at::ScalarType::Half,                                                    \
      at::ScalarType::BFloat16,                                                \
      input.scalar_type(), "act_and_mul_kernel", [&] {                         \
        if (0 == d % 8 && d <= 32768) {                                        \
          if (d <= 512) {                                                      \
            act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 2, ACT_FIRST>  \
                <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),           \
                                           input.data_ptr<scalar_t>(), d);     \
          } else if (d <= 1024) {                                              \
            act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8, ACT_FIRST>  \
                <<<grid, 128, 0, stream>>>(out.data_ptr<scalar_t>(),           \
                                           input.data_ptr<scalar_t>(), d);     \
          } else if (d <= 2048) {                                              \
            act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8, ACT_FIRST>  \
                <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),           \
                                           input.data_ptr<scalar_t>(), d);     \
          } else if (d <= 4096) {                                              \
            act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8, ACT_FIRST>  \
                <<<grid, 512, 0, stream>>>(out.data_ptr<scalar_t>(),           \
                                           input.data_ptr<scalar_t>(), d);     \
          } else {                                                             \
            if (num_tokens <= 1024){\
              constexpr int VEC = 8;                                           \
              constexpr int threads_per_block = 512;                          \
              int64_t elements_per_block = threads_per_block * VEC;            \
              int64_t blocks_x = (d + elements_per_block - 1) / elements_per_block; \
              dim3 grid(blocks_x, num_tokens);                            \
              dim3 block(threads_per_block);                              \
              act_and_mul_kernel_opt4<scalar_t, KERNEL<scalar_t>, VEC, ACT_FIRST> \
                  <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),\
                                                         input.data_ptr<scalar_t>(),\
                                                         d);                   \
            } else {                                                           \
              act_and_mul_kernel_opt2<scalar_t, KERNEL<scalar_t>, 8, ACT_FIRST>\
                  <<<grid, 1024, 0, stream>>>(out.data_ptr<scalar_t>(),        \
                                              input.data_ptr<scalar_t>(), d);  \
            }                                                                  \
          }                                                                    \
        } else {                                                               \
          act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>            \
              <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),           \
                                           input.data_ptr<scalar_t>(), d);     \
        }                                                                      \
      });

void silu_and_mul_opt(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel, true);
}
void gelu_and_mul_opt(torch::Tensor& out,   
                  torch::Tensor& input) 
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_kernel, true);
}
void gelu_tanh_and_mul_opt(torch::Tensor& out,  
                       torch::Tensor& input) 
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_tanh_kernel, true);
}

}  
}


