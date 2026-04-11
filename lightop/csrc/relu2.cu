#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at {
namespace native {

using bfloat16 = at::BFloat16;

template <typename T>
__device__ __forceinline__ T squared_relu_kernel(const T& x) {
  float x_float = (float)x;
  float relu_val = x_float > 0.0f ? x_float : 0.0f;
  return (T)(relu_val * relu_val);
}

template <>
__device__ __forceinline__ bfloat16 squared_relu_kernel<bfloat16>(const bfloat16& x) {
  
  #if defined(__gfx938__)
  short x_short = reinterpret_cast<const short&>(x);
  float x_f32 = __builtin_hcu_cvt_f32_bf16(x_short, false, 0, false);
  #else
  float x_f32 = (float)x;
  #endif

  float relu_val = x_f32 > 0.0f ? x_f32 : 0.0f;
  float sq_relu_f32 = relu_val * relu_val;

  #if defined(__gfx938__)
  short result_short = __builtin_hcu_cvt_bf16_f32(sq_relu_f32, false, false);
  return *(bfloat16*)&result_short;
  #else
  return (bfloat16)sq_relu_f32;
  #endif
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const int64_t offset = token_idx * d + idx;
    out[offset] = ACT_FN(input[offset]);
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC>
__global__ void act_kernel_opt1(
    scalar_t* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    const int d) {
  
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  
  const int64_t token_idx = blockIdx.x;
  int idx = threadIdx.x * VEC;
  
  if (idx < d) {
    const int64_t offset = token_idx * d + idx;
    
    VecType* x_vec = (VecType*)(input + offset);
    VecType* y_vec = (VecType*)(out + offset);
    
    scalar_t r_x[VEC];
    scalar_t r_y[VEC];
    
    *(VecType*)r_x = *x_vec;
    
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = ACT_FN(r_x[i]);
    }
    
    *y_vec = *(VecType*)r_y;
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC, int D, int THREAD_PER_ROW, int ROWS_PER_BLOCK>
__launch_bounds__(1024, 1)
__global__ void act_kernel_opt2(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input) {
  
  int t_row = threadIdx.x / THREAD_PER_ROW;
  int t_col = threadIdx.x % THREAD_PER_ROW;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + t_row;
  const int64_t idx = t_col * VEC;
  
  using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  
  if (idx < D) {
    const int64_t offset = row * D + idx;
    
    VecType* x_vec = (VecType*)(input + offset);
    VecType* y_vec = (VecType*)(out + offset);
    
    scalar_t r_x[VEC];
    scalar_t r_y[VEC];
    
    *(VecType*)r_x = *x_vec;
    
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = ACT_FN(r_x[i]);
    }
    
    *y_vec = *(VecType*)r_y;
  }
}


void relu2_opt(torch::Tensor& out, torch::Tensor& input) {
  int d = input.size(-1); 
  int64_t num_tokens = input.numel() / d;
  
  if (num_tokens == 0) {
    return;
  }
  
  dim3 grid(num_tokens);
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(), "squared_relu_kernel", [&] {
        
        if (168 == d) {
          constexpr int VEC = 8;
          constexpr int THREAD_PER_ROW = 21;
          constexpr int ROWS_PER_BLOCK = 8;
          
          if (num_tokens >= ROWS_PER_BLOCK && 0 == (num_tokens % ROWS_PER_BLOCK)) {
            dim3 grid_opt(num_tokens / ROWS_PER_BLOCK);
            dim3 block_opt(THREAD_PER_ROW * ROWS_PER_BLOCK);
            act_kernel_opt2<scalar_t, squared_relu_kernel<scalar_t>, VEC, 168, THREAD_PER_ROW, ROWS_PER_BLOCK>
                <<<grid_opt, block_opt, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
            return;
          }
        }
 
        else if (336 == d) {
          constexpr int VEC = 8;
          constexpr int THREAD_PER_ROW = 42;
          constexpr int ROWS_PER_BLOCK = 8;
          
          if (num_tokens >= ROWS_PER_BLOCK && 0 == (num_tokens % ROWS_PER_BLOCK)) {
            dim3 grid_opt(num_tokens / ROWS_PER_BLOCK);
            dim3 block_opt(THREAD_PER_ROW * ROWS_PER_BLOCK);
            act_kernel_opt2<scalar_t, squared_relu_kernel<scalar_t>, VEC, 336, THREAD_PER_ROW, ROWS_PER_BLOCK>
                <<<grid_opt, block_opt, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
            return;
          }
        } 

        else if (672 == d) {
          constexpr int VEC = 8;
          constexpr int THREAD_PER_ROW = 84;
          constexpr int ROWS_PER_BLOCK = 4;
          
          if (num_tokens >= ROWS_PER_BLOCK && 0 == (num_tokens % ROWS_PER_BLOCK)) {
            dim3 grid_opt(num_tokens / ROWS_PER_BLOCK);
            dim3 block_opt(THREAD_PER_ROW * ROWS_PER_BLOCK);
            act_kernel_opt2<scalar_t, squared_relu_kernel<scalar_t>, VEC, 672, THREAD_PER_ROW, ROWS_PER_BLOCK>
                <<<grid_opt, block_opt, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>());
            return;
          }
        }
        
        if (0 == d % 8 && d <= 4096) {
          if (d <= 1024) {
            act_kernel_opt1<scalar_t, squared_relu_kernel<scalar_t>, 8>
                <<<grid, 128, 0, stream>>>(out.data_ptr<scalar_t>(),
                                           input.data_ptr<scalar_t>(), d);
          } else if (d <= 2048) {
            act_kernel_opt1<scalar_t, squared_relu_kernel<scalar_t>, 8>
                <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),
                                           input.data_ptr<scalar_t>(), d);
          } else if (d <= 4096) {
            act_kernel_opt1<scalar_t, squared_relu_kernel<scalar_t>, 8>
                <<<grid, 512, 0, stream>>>(out.data_ptr<scalar_t>(),
                                           input.data_ptr<scalar_t>(), d);
          }
        } else {
          dim3 block(std::min((int)d, 1024));
          act_kernel<scalar_t, squared_relu_kernel<scalar_t>>
              <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                           input.data_ptr<scalar_t>(), d);
        }
      });
}

} // namespace native
} // namespace at