// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_gelu_hip.hip for lightop P1.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#define CUDA_CHECK(cmd)                                                                 \
    do {                                                                               \
        cudaError_t error = (cmd);                                                      \
        if (error != cudaSuccess) {                                                     \
            TORCH_CHECK(false, "HIP error: ", cudaGetErrorString(error));               \
        }                                                                              \
    } while (0)

namespace {

constexpr int kBlockThreads = 512;
constexpr float kSqrt1_2 = 0.707106781186547524400844362104849039f;
constexpr float kGeluBeta = 0.79788456080286535587989211986876f;
constexpr float kGeluKappa = 0.044715f;

struct HalfVec8 {
    __half val[8];
};

struct Bf16Vec8 {
    at::BFloat16 val[8];
};

__device__ __forceinline__ uint16_t bf16_value_bits(at::BFloat16 v) {
    return *reinterpret_cast<const uint16_t*>(&v);
}

__device__ __forceinline__ at::BFloat16 bf16_from_bits(uint16_t bits) {
    at::BFloat16 out;
    *reinterpret_cast<uint16_t*>(&out) = bits;
    return out;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    return *reinterpret_cast<const float*>(&u);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
    uint32_t u = *reinterpret_cast<uint32_t*>(&x);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

struct FloatVec4 {
    float val[4];
};

// Abramowitz & Stegun 7.1.26 (5-term): fast device erf used by custom_gelu.
// fp16 previously used a 3-term truncation (drop a4/a5) which caused ~0.08 abs
// error vs torch; keep the full 5-term poly for all dtypes (perf ~same as short).
__device__ __forceinline__ float erf_rational_exp(float t) {
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;

    float sign = t < 0.f ? -1.f : 1.f;
    float ax = fabsf(t);
    float u = 1.f / (1.f + p * ax);
    float poly = (((((a5 * u + a4) * u + a3) * u + a2) * u + a1) * u);
    return sign * (1.f - poly * __expf(-ax * ax));
}

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.f + erf_rational_exp(x * kSqrt1_2));
}

__device__ __forceinline__ float gelu_tanh(float x) {
    float x3 = x * x * x;
    float t = tanhf(kGeluBeta * (x + kGeluKappa * x3));
    return 0.5f * x * (1.f + t);
}

template <bool use_tanh>
__device__ __forceinline__ float gelu_compute(float x) {
    if constexpr (use_tanh) {
        return gelu_tanh(x);
    }
    return gelu_exact(x);
}

template <bool use_tanh>
__device__ __forceinline__ void gelu_half_vec8_lane(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t idx) {
    HalfVec8 in_vec = *reinterpret_cast<const HalfVec8*>(input + idx);
    HalfVec8 out_vec;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x = __half2float(in_vec.val[k]);
        out_vec.val[k] = __float2half(gelu_compute<use_tanh>(x));
    }
    *reinterpret_cast<HalfVec8*>(output + idx) = out_vec;
}

template <bool use_tanh>
__global__ void gelu_half_kernel(
    int64_t n,
    const __half* __restrict__ input,
    __half* __restrict__ output) {
    constexpr int vec = 8;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        gelu_half_vec8_lane<use_tanh>(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            float x = __half2float(input[i]);
            output[i] = __float2half(gelu_compute<use_tanh>(x));
        }
    }
}

template <bool use_tanh>
__device__ __forceinline__ void gelu_bf16_vec8_lane(
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output,
    int64_t idx) {
    Bf16Vec8 in_vec = *reinterpret_cast<const Bf16Vec8*>(input + idx);
    Bf16Vec8 out_vec;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x = bf16_bits_to_float(bf16_value_bits(in_vec.val[k]));
        out_vec.val[k] = bf16_from_bits(float_to_bf16_bits(gelu_compute<use_tanh>(x)));
    }
    *reinterpret_cast<Bf16Vec8*>(output + idx) = out_vec;
}

template <bool use_tanh>
__global__ void gelu_bf16_kernel(
    int64_t n,
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output) {
    constexpr int vec = 8;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        gelu_bf16_vec8_lane<use_tanh>(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            float x = bf16_bits_to_float(bf16_value_bits(input[i]));
            output[i] = bf16_from_bits(float_to_bf16_bits(gelu_compute<use_tanh>(x)));
        }
    }
}

template <bool use_tanh>
__device__ __forceinline__ void gelu_float_vec4_lane(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t idx) {
    FloatVec4 in_vec = *reinterpret_cast<const FloatVec4*>(input + idx);
    FloatVec4 out_vec;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        out_vec.val[k] = gelu_compute<use_tanh>(in_vec.val[k]);
    }
    *reinterpret_cast<FloatVec4*>(output + idx) = out_vec;
}

template <bool use_tanh>
__global__ void gelu_float_kernel(
    int64_t n,
    const float* __restrict__ input,
    float* __restrict__ output) {
    constexpr int vec = 4;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        gelu_float_vec4_lane<use_tanh>(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            output[i] = gelu_compute<use_tanh>(input[i]);
        }
    }
}

template <bool use_tanh>
void launch_gelu_half(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    int grid = static_cast<int>((n + kBlockThreads * 8 - 1) / (kBlockThreads * 8));
    gelu_half_kernel<use_tanh><<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

template <bool use_tanh>
void launch_gelu_bf16(const at::BFloat16* input, at::BFloat16* output, int64_t n, cudaStream_t stream) {
    int grid = static_cast<int>((n + kBlockThreads * 8 - 1) / (kBlockThreads * 8));
    gelu_bf16_kernel<use_tanh><<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

template <bool use_tanh>
void launch_gelu_float(const float* input, float* output, int64_t n, cudaStream_t stream) {
    // Also covers micro_perf dtype "tfloat32" (torch.float32 storage).
    int grid = static_cast<int>((n + kBlockThreads * 4 - 1) / (kBlockThreads * 4));
    gelu_float_kernel<use_tanh><<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

template <bool use_tanh>
void dispatch_gelu(at::ScalarType dtype, const void* input, void* output, int64_t n, cudaStream_t stream) {
    if (dtype == at::kBFloat16) {
        launch_gelu_bf16<use_tanh>(
            reinterpret_cast<const at::BFloat16*>(input),
            reinterpret_cast<at::BFloat16*>(output),
            n,
            stream);
    } else if (dtype == at::kHalf) {
        launch_gelu_half<use_tanh>(
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<__half*>(output),
            n,
            stream);
    } else if (dtype == at::kFloat) {
        launch_gelu_float<use_tanh>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            n,
            stream);
    } else {
        TORCH_CHECK(false, "custom_gelu: unsupported dtype");
    }
}

}  // namespace

void gelu_opt_impl(torch::Tensor input, torch::Tensor output, bool use_tanh) {
    TORCH_CHECK(input.is_cuda(), "custom_gelu: input must be CUDA/HIP tensor");
    TORCH_CHECK(output.is_cuda(), "custom_gelu: output must be CUDA/HIP tensor");
    TORCH_CHECK(input.is_contiguous(), "custom_gelu: input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "custom_gelu: output must be contiguous");
    TORCH_CHECK(input.sizes() == output.sizes(), "custom_gelu: shape mismatch");
    TORCH_CHECK(
        input.scalar_type() == output.scalar_type(),
        "custom_gelu: dtype mismatch between input and output");
    TORCH_CHECK(
        input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat,
        "custom_gelu: only float16/bfloat16/float32 are supported");

    int64_t n = input.numel();
    if (n == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (use_tanh) {
        dispatch_gelu<true>(input.scalar_type(), input.data_ptr(), output.data_ptr(), n, stream);
    } else {
        dispatch_gelu<false>(input.scalar_type(), input.data_ptr(), output.data_ptr(), n, stream);
    }
}


namespace at {
namespace native {

torch::Tensor gelu_opt(const torch::Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(input.is_cuda(), "gelu_opt: input must be CUDA/HIP");
  TORCH_CHECK(input.is_contiguous(), "gelu_opt: input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "gelu_opt: only float16/bfloat16/float32 supported");
  bool use_tanh = (approximate == "tanh");
  TORCH_CHECK(approximate == "none" || use_tanh, "gelu_opt: approximate must be none|tanh");
  auto output = torch::empty_like(input);
  gelu_opt_impl(input, output, use_tanh);
  return output;
}

torch::Tensor gelu_opt_out(
    const torch::Tensor& input, torch::Tensor& output, c10::string_view approximate) {
  bool use_tanh = (approximate == "tanh");
  TORCH_CHECK(approximate == "none" || use_tanh, "gelu_opt: approximate must be none|tanh");
  gelu_opt_impl(input, output, use_tanh);
  return output;
}

}  // namespace native
}  // namespace at
