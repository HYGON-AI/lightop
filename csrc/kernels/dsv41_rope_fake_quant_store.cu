// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float8_e4m3fn.h>
#include <hip/hip_fp8.h>
#include <torch/all.h>

namespace at {
namespace native {
namespace {

__device__ __forceinline__ float fp4_round_trip(float value, float scale) {
  const float x = fminf(6.f, fmaxf(-6.f, value / scale));
  const float a = fabsf(x);
  const float step = a < 2.f ? .5f : (a < 4.f ? 1.f : 2.f);
  const float magnitude = nearbyintf(a / step) * step;
  // torch.sign(+-0) is +0; nonzero negative inputs rounding to zero keep -0.
  const float rounded = x == 0.f ? 0.f : copysignf(magnitude, x);
  return static_cast<float>(at::BFloat16(rounded * scale));
}

template <typename Index>
__global__ void dsv41_rope_fake_quant_store_kernel(
    const at::BFloat16* __restrict__ input,
    const float* __restrict__ freqs,
    uint8_t* __restrict__ cache,
    const Index* __restrict__ indices,
    int64_t page_size,
    int64_t page_bytes) {
  // Two adjacent values per thread; width-8 and width-32 reductions implement
  // the per-16 fake quant and per-64 physical cache quant, respectively.
  const int64_t row = blockIdx.x;
  const int column = threadIdx.x * 2;
  const int64_t loc = indices[row];
  if (loc < 0) return;
  float x = static_cast<float>(input[row * 512 + column]);
  float y = static_cast<float>(input[row * 512 + column + 1]);
  if (column >= 448) {
#pragma clang fp contract(off)
    const float real = freqs[row * 64 + column - 448];
    const float imag = freqs[row * 64 + column - 448 + 1];
    // The reference rounds the complex product to BF16 before quantizing.
    // Disable contraction to preserve the four FP32 complex multiplications.
    const float rx = x * real - y * imag;
    const float ry = x * imag + y * real;
    x = static_cast<float>(at::BFloat16(rx));
    y = static_cast<float>(at::BFloat16(ry));
  }
  float amax = fmaxf(fabsf(x), fabsf(y));
#pragma unroll
  for (int delta = 4; delta; delta >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, delta, 8));
  }
  const float raw_scale = fminf(448.f, fmaxf(0x1p-9f, amax * (1.f / 6.f)));
  const float scale = static_cast<float>(c10::Float8_e4m3fn(raw_scale));
  x = fp4_round_trip(x, scale);
  y = fp4_round_trip(y, scale);

  const int64_t page = loc / page_size;
  const int64_t offset = loc % page_size;
  uint8_t* data = cache + page * page_bytes + offset * 576;
  if (column >= 448) {
    at::BFloat16* rope = reinterpret_cast<at::BFloat16*>(data + 448);
    rope[column - 448] = at::BFloat16(x);
    rope[column - 448 + 1] = at::BFloat16(y);
  } else {
    amax = fmaxf(fabsf(x), fabsf(y));
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, delta, 32));
    }
#if HIP_FP8_TYPE_FNUZ
    constexpr float fp8_max = 224.f;
#else
    constexpr float fp8_max = 448.f;
#endif
    const float raw = fmaxf(1.e-4f, amax) / fp8_max;
    const uint32_t bits = __float_as_uint(raw);
    const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    const float inv_scale = __uint_as_float((254 - exponent) << 23);
    const __hip_fp8_e4m3 qx(fmaxf(-fp8_max, fminf(fp8_max, x * inv_scale)));
    const __hip_fp8_e4m3 qy(fmaxf(-fp8_max, fminf(fp8_max, y * inv_scale)));
    data[column] = qx.__x;
    data[column + 1] = qy.__x;
    if (threadIdx.x % 32 == 0) {
      cache[page * page_bytes + page_size * 576 + offset * 8 + column / 64] = exponent;
    }
  }
}

}  // namespace

void dsv41_rope_fake_quant_store(
    const at::Tensor& input,
    const at::Tensor& freqs,
    at::Tensor& cache,
    const at::Tensor& indices,
    int64_t page_size) {
  TORCH_CHECK(input.is_cuda() && freqs.is_cuda() && cache.is_cuda() && indices.is_cuda(),
              "inputs and cache must be on GPU");
  TORCH_CHECK(input.device() == freqs.device() && input.device() == cache.device() &&
                  input.device() == indices.device(), "devices must match");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
                  input.size(1) == 512 && input.is_contiguous(), "input must be contiguous BF16 [N,512]");
  TORCH_CHECK(freqs.scalar_type() == at::kFloat && freqs.dim() == 2 &&
                  freqs.size(0) == input.size(0) && freqs.size(1) == 64 && freqs.is_contiguous(),
              "freqs must be contiguous FP32 [N,64], interleaved real/imag");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() == input.size(0) && indices.is_contiguous() &&
                  (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong),
              "indices must be contiguous int32/int64 [N]");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() == 2 && cache.is_contiguous(),
              "cache must be contiguous uint8 [pages,page_bytes]");
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0 &&
                  cache.size(1) == ((page_size * 584 + 575) / 576) * 576,
              "cache must use the padded V4 page layout");
  TORCH_CHECK(input.size(0) <= 2147483647, "too many token rows");
  if (input.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "dsv41_rope_fake_quant_store", [&] {
    dsv41_rope_fake_quant_store_kernel<index_t><<<input.size(0), 256, 0, stream>>>(
        input.data_ptr<at::BFloat16>(), freqs.data_ptr<float>(), cache.data_ptr<uint8_t>(),
        indices.data_ptr<index_t>(), page_size, cache.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
