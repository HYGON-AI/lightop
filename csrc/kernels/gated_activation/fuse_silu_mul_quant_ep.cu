// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)  \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  // return (T)(((float)x) / (1.0f + __expf((float)-x)));
  constexpr float LOG2E = 1.44269504088896340736f;
  return (T)(((float)x) / (1.0f + __builtin_amdgcn_exp2f(-((float)x) * LOG2E)));
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
  const int32_t token_idx = blockIdx.x;
  for (int32_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, act_first>(x, y);
  }
}

template <typename scalar_t, 
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1_ep_gernel(
    int64_t num_tokens,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert
    ) {
    int64_t token_idx = blockIdx.x * T_dim + blockIdx.y;
    const int64_t num_total_tokens = num_tokens;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float shared_mem[share_size];
    __shared__ float s_token_scale; 

    for(; token_idx < num_total_tokens ; token_idx += gridDim.y){
        if (tokens_per_expert != nullptr) {
            const int32_t e = token_idx / T_dim; 
            const int32_t t = token_idx % T_dim; 
            
            if (t >= tokens_per_expert[e]) {
                break; 
            }
        }

        const int64_t input_offset = token_idx * 2 * d;
        const int64_t output_offset = token_idx * d;

        float row_max = 0.0f;
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            scalar_t val = compute<scalar_t, act_first>(x, y);
            row_max = fmaxf(row_max, fabsf(static_cast<float>(val)));
        }
        row_max = BlockReduceMax_ROW<float,block_size>(row_max, shared_mem);
        float scale = 0.0f; 
        if (threadIdx.x == 0) {
            scale = row_max;
            s_token_scale = scale;  
            scales[token_idx] = s_token_scale / 127.f; 
        }
        __syncthreads(); 

        float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
            const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
            float val = static_cast<float>(compute<scalar_t, act_first>(x, y));
            int8_t q_val = float_to_int8_rn(val * inv_s);
            out[output_offset + idx] = static_cast<int8_t>(q_val);
        }
    }
}

template <typename scalar_t, int VEC,
          bool act_first, int block_size>
__global__ void act_and_mul_quant_kernel_opt1_one_ep(
    int32_t num_tokens,                 
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int d,
    const int32_t T,
    const int32_t *tokens_per_expert
    ) {
    int32_t token_idx= blockIdx.x;

    const int32_t e = token_idx / T; // 专家索引
    const int32_t t = token_idx % T; // 专家内的 Token 索引
    
    if (tokens_per_expert != nullptr && t >= tokens_per_expert[e]) {
        return; 
    }

    constexpr int share_size=block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

    int idx = threadIdx.x * VEC;
    int tidx = threadIdx.x;
    const int32_t x_index = token_idx * 2 * d + idx;
    const int32_t y_index = token_idx * d + idx;
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
__global__ void act_and_mul_quant_kernel_opt1_ep(
    int64_t num_tokens,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int d,
    const int32_t T_dim, 
    const int32_t expert_num,
    const int32_t *tokens_per_expert
    ) {
    int64_t token_idx = blockIdx.x * T_dim + blockIdx.y;
    int64_t num_total_tokens = num_tokens;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    __shared__ float s_token_scale; 

    for(; token_idx < num_total_tokens ; token_idx += gridDim.y)
    {
        if (tokens_per_expert != nullptr) {
            const int32_t e = token_idx / T_dim; 
            const int32_t t = token_idx % T_dim; 
            
            if (t >= tokens_per_expert[e]) {
                break; 
            }
        }

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
        
        float scale = 0.0f;     
        if(tidx == 0){
            scale = row_max;
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

    } // 结束 Grid-Stride Loop
}


#ifdef USE_ROCM
// Exclusive prefix of the clamped masked counts into row_offsets[0..num_experts], by wave 0.
__device__ __forceinline__ void masked_row_offsets(
    const int32_t* __restrict__ tokens_per_expert, int num_experts, int capacity,
    int* row_offsets) {
  constexpr int Wave = C10_WARP_SIZE;
  if (threadIdx.x < Wave) {
    const int lane = threadIdx.x;
    if (lane == 0) row_offsets[0] = 0;
    int carry = 0;
    for (int base = 0; base < num_experts; base += Wave) {
      const int expert = base + lane;
      int count = expert < num_experts ? tokens_per_expert[expert] : 0;
      count = count < 0 ? 0 : (count > capacity ? capacity : count);
      #pragma unroll
      for (int delta = 1; delta < Wave; delta <<= 1) {
        const int up = __shfl_up(count, delta, Wave);
        if (lane >= delta) count += up;
      }
      if (expert < num_experts) row_offsets[expert + 1] = carry + count;
      carry += __shfl(count, Wave - 1, Wave);
    }
  }
  __syncthreads();
}

// Same masked row mapping as act_and_mul_fp8_quant_ep_masked: valid rows of all experts
// are numbered expert-major and strided over a persistent grid, so one hot expert spreads
// over every CTA instead of serialising on its own grid row. The activation and the
// per-token scale reproduce act_and_mul_quant_kernel_opt1_ep expression for expression,
// so this path is bit-identical to it and only changes how the rows are scheduled.
template <int ThreadsPerRow, int RowsPerBlock, int Segments, bool VectorAligned>
__global__ void act_and_mul_int8_quant_ep_masked(
    int8_t* __restrict__ out,
    const c10::BFloat16* __restrict__ input,
    float* __restrict__ scales,
    int d,
    int capacity,
    int num_experts,
    const int32_t* __restrict__ tokens_per_expert) {
  constexpr int Vec = 8;
  constexpr int Wave = C10_WARP_SIZE;
  static_assert(RowsPerBlock == 1 || ThreadsPerRow == Wave);
  using InputVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutputVec = at::native::memory::aligned_vector<int8_t, Vec>;
  const int row_in_block = threadIdx.x / ThreadsPerRow;
  const int lane = threadIdx.x % ThreadsPerRow;
  __shared__ float warp_max[ThreadsPerRow / Wave];
  __shared__ float inverse_scale;
  extern __shared__ int row_offsets[];
  masked_row_offsets(tokens_per_expert, num_experts, capacity, row_offsets);
  const int total_rows = row_offsets[num_experts];

  for (int row = blockIdx.x * RowsPerBlock + row_in_block;
       row < total_rows; row += gridDim.x * RowsPerBlock) {
    // Offsets are nondecreasing, so row_offsets[e] <= row holds for a prefix of experts and
    // the owner is that prefix's last one; empty experts repeat an offset.
    int expert = -1;
    for (int base = 0; base < num_experts; base += Wave) {
      const int e = base + lane % Wave;
      const uint64_t below = __ballot(e < num_experts && row_offsets[e] <= row);
      expert += __popcll(below);
      if (~below != 0) break;
    }
    const int64_t token =
        static_cast<int64_t>(expert) * capacity + (row - row_offsets[expert]);
    float values[Segments][Vec];
    float max_abs = 0.0f;
    #pragma unroll
    for (int segment = 0; segment < Segments; ++segment) {
      const int col = (lane + segment * ThreadsPerRow) * Vec;
      if (col < d) {
        InputVec gate, up;
        if constexpr (VectorAligned) {
          gate = *reinterpret_cast<const InputVec*>(input + token * 2 * d + col);
          up = *reinterpret_cast<const InputVec*>(input + token * 2 * d + d + col);
        } else {
          #pragma unroll
          for (int i = 0; i < Vec; ++i) {
            gate.val[i] = col + i < d ? input[token * 2 * d + col + i] : c10::BFloat16(0);
            up.val[i] = col + i < d ? input[token * 2 * d + d + col + i] : c10::BFloat16(0);
          }
        }
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
          // Keep bf16 rounding after SiLU and after the multiply, exactly like
          // act_and_mul_quant_kernel_opt1_ep.
          values[segment][i] =
              static_cast<float>(silu_kernel(gate.val[i]) * up.val[i]);
          max_abs = fmaxf(max_abs, fabsf(values[segment][i]));
        }
      }
    }

    float inverse;
    if constexpr (ThreadsPerRow == Wave) {
      max_abs = WarpReduceMax_ROW<float>(max_abs);
      float lane_inverse = 0.0f;
      if (lane == 0) {
        scales[token] = max_abs / 127.0f;
        lane_inverse = (max_abs == 0.0f) ? 0.0f : 127.0f / max_abs;
      }
      inverse = __shfl(lane_inverse, 0, Wave);
    } else {
      max_abs = BlockReduceMax_ROW<float, ThreadsPerRow>(max_abs, warp_max);
      if (lane == 0) {
        scales[token] = max_abs / 127.0f;
        inverse_scale = (max_abs == 0.0f) ? 0.0f : 127.0f / max_abs;
      }
      __syncthreads();
      inverse = inverse_scale;
    }

    #pragma unroll
    for (int segment = 0; segment < Segments; ++segment) {
      const int col = (lane + segment * ThreadsPerRow) * Vec;
      if (col < d) {
        OutputVec bytes;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
          bytes.val[i] = float_to_int8_rn(values[segment][i] * inverse);
        }
        if constexpr (VectorAligned) {
          *reinterpret_cast<OutputVec*>(out + token * d + col) = bytes;
        } else {
          #pragma unroll
          for (int i = 0; i < Vec; ++i) {
            if (col + i < d) out[token * d + col + i] = bytes.val[i];
          }
        }
      }
    }
    if constexpr (ThreadsPerRow != Wave) {
      __syncthreads();
    }
  }
}

template <int ThreadsPerRow, int RowsPerBlock, int Segments, bool VectorAligned>
static void launch_act_and_mul_int8_quant_ep_masked_impl(
    torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales,
    const int32_t* counts, cudaStream_t stream) {
  const int num_experts = input.size(0);
  const int capacity = input.size(1);
  const int d = input.size(2) / 2;
  // Arbitrary residency target; the grid only has to cover the device, rows are strided.
  constexpr int kBlocksPerCU = 4;
  const int64_t max_blocks =
      (static_cast<int64_t>(num_experts) * capacity + RowsPerBlock - 1) / RowsPerBlock;
  const int64_t device_blocks =
      static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
      kBlocksPerCU;
  const dim3 grid(static_cast<unsigned>(std::max<int64_t>(1, std::min(max_blocks, device_blocks))));
  const dim3 block(ThreadsPerRow * RowsPerBlock);
  const size_t smem = (static_cast<size_t>(num_experts) + 1) * sizeof(int);
  act_and_mul_int8_quant_ep_masked<ThreadsPerRow, RowsPerBlock, Segments, VectorAligned>
      <<<grid, block, smem, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<c10::BFloat16>(),
                                      scales.data_ptr<float>(), d, capacity, num_experts,
                                      counts);
}

// Select vector alignment from tensor metadata before launching the kernel, so the aligned
// kernel's instruction stream carries no scalar tail handling.
template <int ThreadsPerRow, int RowsPerBlock, int Segments>
static void launch_act_and_mul_int8_quant_ep_masked(
    torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales,
    const int32_t* counts, cudaStream_t stream) {
  if ((input.size(2) / 2) % 8 == 0) {
    launch_act_and_mul_int8_quant_ep_masked_impl<ThreadsPerRow, RowsPerBlock, Segments, true>(
        input, out, scales, counts, stream);
  } else {
    launch_act_and_mul_int8_quant_ep_masked_impl<ThreadsPerRow, RowsPerBlock, Segments, false>(
        input, out, scales, counts, stream);
  }
}
#endif

void fuse_silu_mul_quant_ep(torch::Tensor& input, 
                                 torch::Tensor& out, 
                                 torch::Tensor& scales, 
                                 torch::optional<torch::Tensor> &tokens_per_expert, 
                                 torch::optional<torch::Tensor> &num_local_tokens_tensor, 
                                 int topK,   
                                 int expert_m) 
{
  TORCH_CHECK(input.dim() == 3, "Input tensor must be 3-dimensional [E, T, H]"); 
  int32_t E = input.size(0);
  int32_t T = input.size(1); 
  int32_t H = input.size(2);
  int32_t d = H / 2;
  int32_t num_tokens = E * T; 
  
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  const int32_t* tokens_per_expert_ptr = 
        tokens_per_expert.has_value() 
            ? tokens_per_expert->data_ptr<int32_t>() 
            : nullptr;

#ifdef USE_ROCM
  // Masked EP layout: skip the padded rows entirely and stride real rows over the device.
  // Same output as the per-expert-row kernels below, so it is a scheduling-only change.
  if (d > 0 && d <= 4096 && E <= 4096 &&
      input.scalar_type() == at::ScalarType::BFloat16 &&
      tokens_per_expert_ptr != nullptr && input.is_contiguous() &&
      out.is_contiguous() && scales.is_contiguous() &&
      tokens_per_expert->is_contiguous() &&
      tokens_per_expert->numel() == E) {
    if (d <= 1024) {
      launch_act_and_mul_int8_quant_ep_masked<64, 4, 2>(
          input, out, scales, tokens_per_expert_ptr, stream);
    } else if (d <= 2048) {
      launch_act_and_mul_int8_quant_ep_masked<64, 4, 4>(
          input, out, scales, tokens_per_expert_ptr, stream);
    } else {
      launch_act_and_mul_int8_quant_ep_masked<64, 4, 8>(
          input, out, scales, tokens_per_expert_ptr, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "act_and_mul_kernel_ep", [&] {
        dim3 grid(E, 128);
        if (d <= 512) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 2, true, 256>
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                        out.data_ptr<int8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        d,
                                        T, 
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 1024) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, true, 128>
                <<<grid, 128, 0, stream>>>(num_tokens ,
                                        out.data_ptr<int8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        d,
                                        T,
                                        E, 
                                        tokens_per_expert_ptr); 
        } else if (d <= 2048) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, true, 256>
                <<<grid, 256, 0, stream>>>(num_tokens ,
                                        out.data_ptr<int8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 4096) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, true, 512>
                <<<grid, 512, 0, stream>>>(num_tokens ,
                                        out.data_ptr<int8_t>(),
                                        input.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        d,
                                        T,
                                        E,
                                        tokens_per_expert_ptr); 
        } else if (d <= 8192) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 8, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<int8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else if (d <= 16384) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 16, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<int8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else if (d <= 32768) {
            act_and_mul_quant_kernel_opt1_ep<scalar_t, 32, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<int8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        } else {
            act_and_mul_quant_kernel_opt1_ep_gernel<scalar_t, true, 1024>
                <<<grid, 1024, 0, stream>>>(num_tokens ,
                                            out.data_ptr<int8_t>(),
                                            input.data_ptr<scalar_t>(),
                                            scales.data_ptr<float>(),
                                            d,
                                            T, 
                                            E,
                                            tokens_per_expert_ptr); 
        }
      });
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

}
}

