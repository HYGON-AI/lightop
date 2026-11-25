// Ported from custom_ops/custom_silu_hip.hip for lightop P1.
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

__device__ __forceinline__ float silu_compute(float x) {
    return x / (1.f + __expf(-x));
}

__device__ __forceinline__ void silu_half_vec8_lane(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t idx) {
    HalfVec8 in_vec = *reinterpret_cast<const HalfVec8*>(input + idx);
    HalfVec8 out_vec;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x = __half2float(in_vec.val[k]);
        out_vec.val[k] = __float2half(silu_compute(x));
    }
    *reinterpret_cast<HalfVec8*>(output + idx) = out_vec;
}

__global__ void silu_half_kernel(
    int64_t n,
    const __half* __restrict__ input,
    __half* __restrict__ output) {
    constexpr int vec = 8;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        silu_half_vec8_lane(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            float x = __half2float(input[i]);
            output[i] = __float2half(silu_compute(x));
        }
    }
}

__device__ __forceinline__ void silu_bf16_vec8_lane(
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output,
    int64_t idx) {
    Bf16Vec8 in_vec = *reinterpret_cast<const Bf16Vec8*>(input + idx);
    Bf16Vec8 out_vec;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x = bf16_bits_to_float(bf16_value_bits(in_vec.val[k]));
        out_vec.val[k] = bf16_from_bits(float_to_bf16_bits(silu_compute(x)));
    }
    *reinterpret_cast<Bf16Vec8*>(output + idx) = out_vec;
}

__global__ void silu_bf16_kernel(
    int64_t n,
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output) {
    constexpr int vec = 8;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        silu_bf16_vec8_lane(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            float x = bf16_bits_to_float(bf16_value_bits(input[i]));
            output[i] = bf16_from_bits(float_to_bf16_bits(silu_compute(x)));
        }
    }
}

__device__ __forceinline__ void silu_float_vec4_lane(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t idx) {
    FloatVec4 in_vec = *reinterpret_cast<const FloatVec4*>(input + idx);
    FloatVec4 out_vec;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        out_vec.val[k] = silu_compute(in_vec.val[k]);
    }
    *reinterpret_cast<FloatVec4*>(output + idx) = out_vec;
}

__global__ void silu_float_kernel(
    int64_t n,
    const float* __restrict__ input,
    float* __restrict__ output) {
    constexpr int vec = 4;
    constexpr int bws = kBlockThreads * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        silu_float_vec4_lane(input, output, idx);
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            output[i] = silu_compute(input[i]);
        }
    }
}

void launch_silu_half(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    int grid = static_cast<int>((n + kBlockThreads * 8 - 1) / (kBlockThreads * 8));
    silu_half_kernel<<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

void launch_silu_bf16(const at::BFloat16* input, at::BFloat16* output, int64_t n, cudaStream_t stream) {
    int grid = static_cast<int>((n + kBlockThreads * 8 - 1) / (kBlockThreads * 8));
    silu_bf16_kernel<<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

void launch_silu_float(const float* input, float* output, int64_t n, cudaStream_t stream) {
    // Also covers micro_perf dtype "tfloat32" (torch.float32 storage).
    int grid = static_cast<int>((n + kBlockThreads * 4 - 1) / (kBlockThreads * 4));
    silu_float_kernel<<<dim3(grid), dim3(kBlockThreads), 0, stream>>>(
        n,
        input,
        output);;
    CUDA_CHECK(cudaGetLastError());
}

void dispatch_silu(at::ScalarType dtype, const void* input, void* output, int64_t n, cudaStream_t stream) {
    if (dtype == at::kBFloat16) {
        launch_silu_bf16(
            reinterpret_cast<const at::BFloat16*>(input),
            reinterpret_cast<at::BFloat16*>(output),
            n,
            stream);
    } else if (dtype == at::kHalf) {
        launch_silu_half(
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<__half*>(output),
            n,
            stream);
    } else if (dtype == at::kFloat) {
        launch_silu_float(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            n,
            stream);
    } else {
        TORCH_CHECK(false, "custom_silu: unsupported dtype");
    }
}

}  // namespace

void silu_opt_impl(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "custom_silu: input must be CUDA/HIP tensor");
    TORCH_CHECK(output.is_cuda(), "custom_silu: output must be CUDA/HIP tensor");
    TORCH_CHECK(input.is_contiguous(), "custom_silu: input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "custom_silu: output must be contiguous");
    TORCH_CHECK(input.sizes() == output.sizes(), "custom_silu: shape mismatch");
    TORCH_CHECK(
        input.scalar_type() == output.scalar_type(),
        "custom_silu: dtype mismatch between input and output");
    TORCH_CHECK(
        input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat,
        "custom_silu: only float16/bfloat16/float32 are supported");

    int64_t n = input.numel();
    if (n == 0) {
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_silu(input.scalar_type(), input.data_ptr(), output.data_ptr(), n, stream);
}


namespace at {
namespace native {

torch::Tensor silu_opt(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "silu_opt: input must be CUDA/HIP");
  TORCH_CHECK(input.is_contiguous(), "silu_opt: input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "silu_opt: only float16/bfloat16/float32 supported");
  auto output = torch::empty_like(input);
  silu_opt_impl(input, output);
  return output;
}

torch::Tensor silu_opt_out(const torch::Tensor& input, torch::Tensor& output) {
  silu_opt_impl(input, output);
  return output;
}

}  // namespace native
}  // namespace at
