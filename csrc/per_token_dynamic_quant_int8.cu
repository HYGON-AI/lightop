// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
// #include "reduce.h"


#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
// lj
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
// lj

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
    float dst = std::nearbyint(x);//lj
    // float dst = rintf(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic

//   dst = std::clamp(dst, i8_min, i8_max);

//   dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;//lj
    dst = fminf(fmaxf(dst, i8_min), i8_max);
    return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val =  fmaxf(val,  __shfl_down(val, offset));
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceMax_ROW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceMax_ROW<T>(val);
  if constexpr(block_size==C10_WARP_SIZE)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();//lj
    if (wid == 0&&lid<share_size) {
      val = WarpReduceMax_ROW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

template <typename scalar_t, typename scale_t,
          int VEC, int block_size, bool APPLY_CLAMP>
__global__ void per_token_dynamic_quant_int8_kernel(
    const scalar_t* __restrict__ input, 
    const float* __restrict__ smoothscale, 
    int8_t* __restrict__ output, 
    float* scale_out, 
    const int hidden_size,
    const bool round_after_smooth,
    const float limit) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    using VecFloatType = at::native::memory::aligned_vector<float, VEC>;

    int tcol = hidden_size / VEC;
    scalar_t input_vec[VEC];
    int64_t idx = bid * tcol + tid;
    idx *= VEC;

    float smoothscale_vec;
    float input_float_vec[VEC];
    float row_max = 0.f;
    if (tid < tcol) {
        *(VecType*)input_vec = *(VecType*)(input+idx);
        // *(VecFloatType)smoothscale_vec = *(VecFloatType*)(smoothscale+tid*VEC);
        // *(VecFloatType)smoothscale_vec = *reinterpret_cast<VecFloatType*>(const_cast<float*>(&smoothscale[tid * VEC]));
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            smoothscale_vec = smoothscale[tid * VEC + ii];
            if (round_after_smooth) {
                smoothscale_vec = static_cast<float>(
                    static_cast<scalar_t>(smoothscale_vec));
            }
            float input_value = static_cast<float>(input_vec[ii]);
            if constexpr (APPLY_CLAMP) {
                input_value = fminf(fmaxf(input_value, -limit), limit);
                input_value = static_cast<float>(
                    static_cast<scalar_t>(input_value));
            }
            input_float_vec[ii] = input_value * smoothscale_vec;
            if (round_after_smooth) {
                input_float_vec[ii] = static_cast<float>(
                    static_cast<scalar_t>(input_float_vec[ii]));
            }
            row_max = fmaxf(row_max,fabs(input_float_vec[ii]));
        }
    }
    row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
    __shared__ float s_token_scale;     
    if(tid == 0){
        s_token_scale = round_after_smooth ? fmaxf(row_max, 1e-10f) : row_max;
        scale_out[bid] = s_token_scale / 127.f;
    }
    __syncthreads();//lj

    // float scale_x = s_token_scale / 127.f;
    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
    int8_t out_vec[VEC];
    if (tid < tcol) {
        #pragma unroll
        for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(input_float_vec[ii] * inv_s);
        }
        *(VecInt8Type*)(output+idx) = *(VecInt8Type*)out_vec;
    }
}

template <typename scalar_t, bool APPLY_CLAMP>
void launch_per_token_dynamic_quant_int8(
    const scalar_t* input,
    const float* smoothscale,
    int8_t* output,
    float* scales,
    int hidden_size,
    int64_t num_tokens,
    bool round_after_smooth,
    float limit,
    cudaStream_t stream) {
#define LAUNCH_PER_TOKEN_SMOOTH_QUANT(VEC, THREADS)                         \
    per_token_dynamic_quant_int8_kernel<                                   \
        scalar_t, float, VEC, THREADS, APPLY_CLAMP>                        \
        <<<num_tokens, THREADS, 0, stream>>>(                              \
            input, smoothscale, output, scales, hidden_size,               \
            round_after_smooth, limit)

    if (hidden_size <= 64) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(1, 64);
    } else if (hidden_size <= 128) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(2, 64);
    } else if (hidden_size <= 256) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(4, 128);
    } else if (hidden_size <= 512) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(4, 128);
    } else if (hidden_size <= 1024) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(8, 128);
    } else if (hidden_size <= 2048) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(16, 128);
    } else if (hidden_size <= 4096) {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(16, 256);
    } else {
        LAUNCH_PER_TOKEN_SMOOTH_QUANT(16, 512);
    }

#undef LAUNCH_PER_TOKEN_SMOOTH_QUANT
}

template <int ROWS_PER_BLOCK>
__global__ void per_token_dynamic_quant_int8_bf16_k128_quarter_wave_kernel(
    const c10::BFloat16* __restrict__ input,
    const float* __restrict__ smoothscale,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    int64_t num_tokens) {
    constexpr int SUBGROUP_SIZE = C10_WARP_SIZE / 4;
    constexpr int ROWS_PER_WAVE = 4;
    static_assert(ROWS_PER_BLOCK * SUBGROUP_SIZE == 512,
                  "the K=128 fast path uses thirty-two 16-thread subgroups");
    constexpr int VEC = 8;
    const int tid = threadIdx.x;
    const int wave_lane = tid % C10_WARP_SIZE;
    const int wave = tid / C10_WARP_SIZE;
    const int subgroup = wave_lane / SUBGROUP_SIZE;
    const int lane = wave_lane % SUBGROUP_SIZE;
    const int64_t token_idx =
        static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
        wave * ROWS_PER_WAVE + subgroup;
    if (token_idx >= num_tokens) {
        return;
    }

    using InputVec =
        at::native::memory::aligned_vector<c10::BFloat16, VEC>;
    using OutputVec = at::native::memory::aligned_vector<int8_t, VEC>;
    using ScaleVec = at::native::memory::aligned_vector<float, VEC>;
    const int col = lane * VEC;
    const int64_t offset = token_idx * 128 + col;
    const InputVec input_vec =
        *reinterpret_cast<const InputVec*>(input + offset);
    const ScaleVec smooth_vec =
        *reinterpret_cast<const ScaleVec*>(smoothscale + col);
    float values[VEC];
    float row_max = 0.0f;
#pragma unroll
    for (int ii = 0; ii < VEC; ++ii) {
        values[ii] =
            static_cast<float>(input_vec.val[ii]) * smooth_vec.val[ii];
        row_max = fmaxf(row_max, fabsf(values[ii]));
    }

#pragma unroll
    for (int reduce_offset = SUBGROUP_SIZE / 2; reduce_offset > 0;
         reduce_offset >>= 1) {
        row_max = fmaxf(
            row_max, __shfl_down(row_max, reduce_offset, SUBGROUP_SIZE));
    }
    const float token_absmax = __shfl(row_max, 0, SUBGROUP_SIZE);
    if (lane == 0) {
        scale_out[token_idx] = token_absmax / 127.0f;
    }
    const float inv_scale =
        token_absmax == 0.0f ? 0.0f : 127.0f / token_absmax;
    OutputVec output_vec;
#pragma unroll
    for (int ii = 0; ii < VEC; ++ii) {
        output_vec.val[ii] = float_to_int8_rn(values[ii] * inv_scale);
    }
    *reinterpret_cast<OutputVec*>(output + offset) = output_vec;
}

void launch_per_token_dynamic_quant_int8_bf16_k128_wave(
    const c10::BFloat16* input,
    const float* smoothscale,
    int8_t* output,
    float* scales,
    int64_t num_tokens,
    cudaStream_t stream) {
    constexpr int threads = 512;
    constexpr int rows_per_block = (threads / C10_WARP_SIZE) * 4;
    const int64_t blocks =
        (num_tokens + rows_per_block - 1) / rows_per_block;
    per_token_dynamic_quant_int8_bf16_k128_quarter_wave_kernel<rows_per_block>
        <<<blocks, threads, 0, stream>>>(
            input, smoothscale, output, scales, num_tokens);
}

template <typename scalar_t, bool APPLY_CLAMP>
void launch_per_token_dynamic_quant_int8_16k(
    const scalar_t* input,
    const float* smoothscale,
    int8_t* output,
    float* scales,
    int hidden_size,
    int64_t num_tokens,
    bool round_after_smooth,
    float limit,
    cudaStream_t stream) {
    constexpr int vec = 16;
    constexpr int threads = 1024;
    per_token_dynamic_quant_int8_kernel<
        scalar_t, float, vec, threads, APPLY_CLAMP>
        <<<num_tokens, threads, 0, stream>>>(
            input, smoothscale, output, scales, hidden_size,
            round_after_smooth, limit);
}

template <typename scalar_t, bool APPLY_CLAMP>
__device__ __forceinline__ float per_token_large_hidden_transform(
    scalar_t input_value_scalar,
    float smooth_value,
    bool round_after_smooth,
    float limit) {
    if (round_after_smooth) {
        smooth_value = static_cast<float>(
            static_cast<scalar_t>(smooth_value));
    }
    float input_value = static_cast<float>(input_value_scalar);
    if constexpr (APPLY_CLAMP) {
        input_value = fminf(fmaxf(input_value, -limit), limit);
        input_value = static_cast<float>(
            static_cast<scalar_t>(input_value));
    }
    float scaled_value = input_value * smooth_value;
    if (round_after_smooth) {
        scaled_value = static_cast<float>(
            static_cast<scalar_t>(scaled_value));
    }
    return scaled_value;
}

template <typename scalar_t,
          int VEC,
          int block_size,
          bool APPLY_CLAMP>
__global__ void per_token_dynamic_quant_int8_large_hidden_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smoothscale,
    int8_t* __restrict__ output,
    float* scale_out,
    const int hidden_size,
    const bool round_after_smooth,
    const float limit) {
    const int64_t token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_vectors = hidden_size / VEC;
    const int64_t row_offset = token_idx * hidden_size;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float reduce_shared[share_size];
    __shared__ float token_absmax;

    using InputVec = at::native::memory::aligned_vector<scalar_t, VEC>;
    using OutputVec = at::native::memory::aligned_vector<int8_t, VEC>;

    float row_max = 0.0f;
    for (int vector_idx = tid; vector_idx < num_vectors;
         vector_idx += block_size) {
        const int col = vector_idx * VEC;
        const int64_t offset = row_offset + col;
        const InputVec input_vec =
            *reinterpret_cast<const InputVec*>(input + offset);

#pragma unroll
        for (int ii = 0; ii < VEC; ++ii) {
            const float scaled_value =
                per_token_large_hidden_transform<scalar_t, APPLY_CLAMP>(
                    input_vec.val[ii], smoothscale[col + ii],
                    round_after_smooth, limit);
            row_max = fmaxf(row_max, fabsf(scaled_value));
        }
    }

    row_max =
        BlockReduceMax_ROW<float, block_size>(row_max, reduce_shared);
    if (tid == 0) {
        token_absmax =
            round_after_smooth ? fmaxf(row_max, 1.0e-10f) : row_max;
        scale_out[token_idx] = token_absmax / 127.0f;
    }
    __syncthreads();

    const float inv_scale =
        token_absmax == 0.0f ? 0.0f : 127.0f / token_absmax;
    for (int vector_idx = tid; vector_idx < num_vectors;
         vector_idx += block_size) {
        const int col = vector_idx * VEC;
        const int64_t offset = row_offset + col;
        const InputVec input_vec =
            *reinterpret_cast<const InputVec*>(input + offset);
        OutputVec output_vec{};

#pragma unroll
        for (int ii = 0; ii < VEC; ++ii) {
            const float scaled_value =
                per_token_large_hidden_transform<scalar_t, APPLY_CLAMP>(
                    input_vec.val[ii], smoothscale[col + ii],
                    round_after_smooth, limit);
            output_vec.val[ii] =
                float_to_int8_rn(scaled_value * inv_scale);
        }
        *reinterpret_cast<OutputVec*>(output + offset) = output_vec;
    }
}

template <typename scalar_t, bool APPLY_CLAMP>
void launch_per_token_dynamic_quant_int8_large_hidden(
    const scalar_t* input,
    const float* smoothscale,
    int8_t* output,
    float* scales,
    int hidden_size,
    int64_t num_tokens,
    bool round_after_smooth,
    float limit,
    cudaStream_t stream) {
    constexpr int vec = 16 / sizeof(scalar_t);
    constexpr int threads = 512;
    static_assert(vec > 0 && 16 % sizeof(scalar_t) == 0,
                  "scalar type must support a 16-byte vector load");
    per_token_dynamic_quant_int8_large_hidden_kernel<
        scalar_t, vec, threads, APPLY_CLAMP>
        <<<num_tokens, threads, 0, stream>>>(
            input, smoothscale, output, scales, hidden_size,
            round_after_smooth, limit);
}

void per_token_dynamic_quant_int8(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales,
    torch::Tensor const& smoothscale,
    const bool round_after_smooth,
    const double limit) {
    
    TORCH_CHECK(input.is_cuda(), "input must be on a GPU");
    TORCH_CHECK(smoothscale.is_contiguous());
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());
    TORCH_CHECK(
        limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
        "limit must be -1 (disabled) or a finite non-negative value");

    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    int const hidden_size = input.size(-1);
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 32768 && hidden_size % 16 == 0,
                "hidden size must be divisible by 16 and in (0, 32768]");
    int64_t num_tokens = input.numel() / hidden_size;
    TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Char,
                "out must have dtype torch.int8");
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
                "scales must have dtype torch.float32");
    TORCH_CHECK(scales.numel() == num_tokens,
                "scales must contain one value per token");
    TORCH_CHECK(smoothscale.scalar_type() == at::ScalarType::Float,
                "smoothscale must have dtype torch.float32");
    TORCH_CHECK(smoothscale.dim() == 1 && smoothscale.numel() == hidden_size,
                "smoothscale must have shape [hidden_size]");
    TORCH_CHECK(input.device() == out.device() &&
                    input.device() == scales.device() &&
                    input.device() == smoothscale.device(),
                "all tensors must be on the same device");
    if (num_tokens == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
    bool ptrs_are_aligned = inp_ptr % 16 == 0;
    TORCH_CHECK(ptrs_are_aligned, "input must be 16-byte aligned");
    if (input.scalar_type() == at::ScalarType::BFloat16 &&
        num_tokens == 262144 && hidden_size == 128 &&
        !round_after_smooth && limit < 0.0) {
        launch_per_token_dynamic_quant_int8_bf16_k128_wave(
            input.data_ptr<c10::BFloat16>(), smoothscale.data_ptr<float>(),
            out.data_ptr<int8_t>(), scales.data_ptr<float>(), num_tokens,
            stream);
    } else if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned){
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "per_token_dynamic_quant_int8_kernel", [&] {
                if (limit < 0.0) {
                    launch_per_token_dynamic_quant_int8<scalar_t, false>(
                        input.data_ptr<scalar_t>(), smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                } else {
                    launch_per_token_dynamic_quant_int8<scalar_t, true>(
                        input.data_ptr<scalar_t>(), smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                }
            });
    } else if (
        hidden_size <= 16384 &&
        inp_ptr % (16 * input.element_size()) == 0 &&
        out_ptr % 16 == 0) {
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(),
            "per_token_dynamic_quant_int8_16k_kernel", [&] {
                if (limit < 0.0) {
                    launch_per_token_dynamic_quant_int8_16k<
                        scalar_t, false>(
                        input.data_ptr<scalar_t>(),
                        smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                } else {
                    launch_per_token_dynamic_quant_int8_16k<
                        scalar_t, true>(
                        input.data_ptr<scalar_t>(),
                        smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                }
            });
    } else {
        TORCH_CHECK(
            out_ptr % 16 == 0,
            "large-hidden output must be 16-byte aligned");
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(),
            "per_token_dynamic_quant_int8_large_hidden_kernel", [&] {
                if (limit < 0.0) {
                    launch_per_token_dynamic_quant_int8_large_hidden<
                        scalar_t, false>(
                        input.data_ptr<scalar_t>(),
                        smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                } else {
                    launch_per_token_dynamic_quant_int8_large_hidden<
                        scalar_t, true>(
                        input.data_ptr<scalar_t>(),
                        smoothscale.data_ptr<float>(),
                        out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        hidden_size, num_tokens, round_after_smooth,
                        static_cast<float>(limit), stream);
                }
            });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
}
