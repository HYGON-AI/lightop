// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Elementwise natural log for LightOP. CUDA-style source; hipify at build.
// Vec8 half/bf16 + Vec4 fp32. gfx938: packed HCU cvt + amdgcn log2 * ln2.
// fp16 nvec=8 (retry after cheap log2); bf16 nvec=8; fp32 nvec=4 from 1M elems.

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

struct HalfVec8 {
    __half val[8];
};

struct Bf16Vec8 {
    at::BFloat16 val[8];
};

struct FloatVec4 {
    float val[4];
};

__device__ __forceinline__ float bf16_to_float(at::BFloat16 v) {
#if defined(__gfx938__)
    const short bits = *reinterpret_cast<const short*>(&v);
    return __builtin_hcu_cvt_f32_bf16(bits, false, 0, false);
#else
    const uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    return *reinterpret_cast<const float*>(&u);
#endif
}

__device__ __forceinline__ at::BFloat16 float_to_bf16(float x) {
#if defined(__gfx938__)
    const short bits = __builtin_hcu_cvt_bf16_f32(x, false, false);
    at::BFloat16 out;
    *reinterpret_cast<short*>(&out) = bits;
    return out;
#else
    uint32_t u = *reinterpret_cast<uint32_t*>(&x);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    at::BFloat16 out;
    *reinterpret_cast<uint16_t*>(&out) = static_cast<uint16_t>(u >> 16);
    return out;
#endif
}

__device__ __forceinline__ float log_fast(float x) {
#if defined(__gfx938__)
    constexpr float kLn2 = 0.693147180559945309417f;
    return __builtin_amdgcn_logf(x) * kLn2;
#else
    return __logf(x);
#endif
}

#if defined(__gfx938__)
__device__ __forceinline__ void log_bf16_pair(const at::BFloat16* in, at::BFloat16* out) {
    const int packed = *reinterpret_cast<const int*>(in);
    const float x0 = __builtin_hcu_cvt_f32_bf16(packed, false, 0, false);
    const float x1 = __builtin_hcu_cvt_f32_bf16(packed, false, 0, true);
    const auto y = __builtin_hcu_cvt_pk_bf16_f32(0, log_fast(x0), 0, log_fast(x1), false);
    *reinterpret_cast<int*>(out) = *reinterpret_cast<const int*>(&y);
}

__device__ __forceinline__ void log_half_pair(const __half* in, __half* out) {
    // v_log_f16 is log2; packed mul by ln2. Avoids f16<->f32 cvt + v_log_f32.
    const __half2 h = *reinterpret_cast<const __half2*>(in);
    const __half2 ln2 = __float2half2_rn(0.693147180559945309417f);
    *reinterpret_cast<__half2*>(out) = __hmul2(h2log2(h), ln2);
}
#endif

__device__ __forceinline__ float log_acc(float x) {
    return logf(x);
}

__global__ void __launch_bounds__(256, 4) log_half_kernel(
    int64_t n,
    const __half* __restrict__ input,
    __half* __restrict__ output) {
    constexpr int kBlock = 256;
    constexpr int vec = 8;
    constexpr int nvec = 1;
    constexpr int bws = kBlock * vec * nvec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        HalfVec8 in_v[1];
        HalfVec8 out_v[1];
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            in_v[v] = *reinterpret_cast<const HalfVec8*>(
                input + idx + (int64_t)v * kBlock * vec);
        }
        #pragma unroll
        for (int k = 0; k < 8; k += 2) {
            #pragma unroll
            for (int v = 0; v < nvec; ++v) {
#if defined(__gfx938__)
                log_half_pair(&in_v[v].val[k], &out_v[v].val[k]);
#else
                float x0 = __half2float(in_v[v].val[k]);
                float x1 = __half2float(in_v[v].val[k + 1]);
                out_v[v].val[k] = __float2half(log_fast(x0));
                out_v[v].val[k + 1] = __float2half(log_fast(x1));
#endif
            }
        }
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            *reinterpret_cast<HalfVec8*>(
                output + idx + (int64_t)v * kBlock * vec) = out_v[v];
        }
        return;
    }
    #pragma unroll
    for (int v = 0; v < nvec; ++v) {
        int64_t base = idx + (int64_t)v * kBlock * vec;
        #pragma unroll
        for (int k = 0; k < vec; ++k) {
            int64_t i = base + k;
            if (i < n) {
#if defined(__gfx938__)
                output[i] = __hmul(hlog2(input[i]), __float2half(0.693147180559945309417f));
#else
                float x = __half2float(input[i]);
                output[i] = __float2half(log_fast(x));
#endif
            }
        }
    }
}

__global__ void log_bf16_kernel(
    int64_t n,
    const at::BFloat16* __restrict__ input,
    at::BFloat16* __restrict__ output) {
    constexpr int kBlock = 256;
    constexpr int vec = 8;
    constexpr int nvec = 8;
    constexpr int bws = kBlock * vec * nvec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        Bf16Vec8 in_v[8];
        Bf16Vec8 out_v[8];
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            in_v[v] = *reinterpret_cast<const Bf16Vec8*>(
                input + idx + (int64_t)v * kBlock * vec);
        }
        #pragma unroll
        for (int k = 0; k < 8; k += 2) {
            #pragma unroll
            for (int v = 0; v < nvec; ++v) {
#if defined(__gfx938__)
                log_bf16_pair(&in_v[v].val[k], &out_v[v].val[k]);
#else
                float x0 = bf16_to_float(in_v[v].val[k]);
                float x1 = bf16_to_float(in_v[v].val[k + 1]);
                out_v[v].val[k] = float_to_bf16(log_fast(x0));
                out_v[v].val[k + 1] = float_to_bf16(log_fast(x1));
#endif
            }
        }
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            *reinterpret_cast<Bf16Vec8*>(
                output + idx + (int64_t)v * kBlock * vec) = out_v[v];
        }
        return;
    }
    #pragma unroll
    for (int v = 0; v < nvec; ++v) {
        int64_t base = idx + (int64_t)v * kBlock * vec;
        #pragma unroll
        for (int k = 0; k < vec; ++k) {
            int64_t i = base + k;
            if (i < n) {
                float x = bf16_to_float(input[i]);
                output[i] = float_to_bf16(log_fast(x));
            }
        }
    }
}

__global__ void log_float_simple_kernel(
    int64_t n,
    const float* __restrict__ input,
    float* __restrict__ output) {
    constexpr int kBlock = 512;
    constexpr int vec = 4;
    constexpr int bws = kBlock * vec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        FloatVec4 in_vec = *reinterpret_cast<const FloatVec4*>(input + idx);
        FloatVec4 out_vec;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            out_vec.val[k] = log_acc(in_vec.val[k]);
        }
        *reinterpret_cast<FloatVec4*>(output + idx) = out_vec;
        return;
    }
    #pragma unroll
    for (int k = 0; k < vec; ++k) {
        int64_t i = idx + k;
        if (i < n) {
            output[i] = log_acc(input[i]);
        }
    }
}

__global__ void log_float_kernel(
    int64_t n,
    const float* __restrict__ input,
    float* __restrict__ output) {
    constexpr int kBlock = 256;
    constexpr int vec = 4;
    constexpr int nvec = 4;
    constexpr int bws = kBlock * vec * nvec;
    int64_t off = (int64_t)blockIdx.x * bws;
    int64_t idx = off + (int64_t)threadIdx.x * vec;
    if (n - off >= bws) {
        FloatVec4 in_v[4];
        FloatVec4 out_v[4];
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            in_v[v] = *reinterpret_cast<const FloatVec4*>(
                input + idx + (int64_t)v * kBlock * vec);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int v = 0; v < nvec; ++v) {
                out_v[v].val[k] = log_acc(in_v[v].val[k]);
            }
        }
        #pragma unroll
        for (int v = 0; v < nvec; ++v) {
            *reinterpret_cast<FloatVec4*>(
                output + idx + (int64_t)v * kBlock * vec) = out_v[v];
        }
        return;
    }
    #pragma unroll
    for (int v = 0; v < nvec; ++v) {
        int64_t base = idx + (int64_t)v * kBlock * vec;
        #pragma unroll
        for (int k = 0; k < vec; ++k) {
            int64_t i = base + k;
            if (i < n) {
                output[i] = log_acc(input[i]);
            }
        }
    }
}

void launch_log_half(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock * 8 - 1) / (kBlock * 8));
    log_half_kernel<<<dim3(grid), dim3(kBlock), 0, stream>>>(
        n,
        input,
        output);
    CUDA_CHECK(cudaGetLastError());
}

void launch_log_bf16(const at::BFloat16* input, at::BFloat16* output, int64_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock * 64 - 1) / (kBlock * 64));
    log_bf16_kernel<<<dim3(grid), dim3(kBlock), 0, stream>>>(
        n,
        input,
        output);
    CUDA_CHECK(cudaGetLastError());
}

void launch_log_float(const float* input, float* output, int64_t n, cudaStream_t stream) {
    constexpr int64_t kNvecMinElems = 1024 * 1024;
    if (n >= kNvecMinElems) {
        constexpr int kBlock = 256;
        int grid = static_cast<int>((n + kBlock * 16 - 1) / (kBlock * 16));
        log_float_kernel<<<dim3(grid), dim3(kBlock), 0, stream>>>(
            n,
            input,
            output);
    } else {
        constexpr int kBlock = 512;
        int grid = static_cast<int>((n + kBlock * 4 - 1) / (kBlock * 4));
        log_float_simple_kernel<<<dim3(grid), dim3(kBlock), 0, stream>>>(
            n,
            input,
            output);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dispatch_log(at::ScalarType dtype, const void* input, void* output, int64_t n, cudaStream_t stream) {
    if (dtype == at::kBFloat16) {
        launch_log_bf16(
            reinterpret_cast<const at::BFloat16*>(input),
            reinterpret_cast<at::BFloat16*>(output),
            n,
            stream);
    } else if (dtype == at::kHalf) {
        launch_log_half(
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<__half*>(output),
            n,
            stream);
    } else if (dtype == at::kFloat) {
        launch_log_float(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            n,
            stream);
    } else {
        TORCH_CHECK(false, "log_opt: unsupported dtype");
    }
}

}  // namespace

void log_opt_impl(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "log_opt: input must be CUDA/HIP tensor");
    TORCH_CHECK(output.is_cuda(), "log_opt: output must be CUDA/HIP tensor");
    TORCH_CHECK(input.is_contiguous(), "log_opt: input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "log_opt: output must be contiguous");
    TORCH_CHECK(input.sizes() == output.sizes(), "log_opt: shape mismatch");
    TORCH_CHECK(
        input.scalar_type() == output.scalar_type(),
        "log_opt: dtype mismatch between input and output");
    TORCH_CHECK(
        input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat,
        "log_opt: only float16/bfloat16/float32 are supported");
    // The kernel runs on input's device/stream, so an output living elsewhere would
    // be written through a foreign pointer.
    TORCH_CHECK(
        output.device() == input.device(),
        "log_opt: input and output must live on the same device");

    int64_t n = input.numel();
    if (n == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_log(input.scalar_type(), input.data_ptr(), output.data_ptr(), n, stream);
}


namespace at {
namespace native {

torch::Tensor log_opt(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "log_opt: input must be CUDA/HIP");
  TORCH_CHECK(input.is_contiguous(), "log_opt: input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "log_opt: only float16/bfloat16/float32 supported");
  auto output = torch::empty_like(input);
  log_opt_impl(input, output);
  return output;
}

torch::Tensor log_opt_out(const torch::Tensor& input, torch::Tensor& output) {
  log_opt_impl(input, output);
  return output;
}

}  // namespace native
}  // namespace at
