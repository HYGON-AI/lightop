// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Initial candidate: preserve SGLang INT8 semantics, including zero/unit padding.
// See test/benchmarks/benchmark_quant_int8.py for gfx938 performance comparisons.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace at::native {
namespace {
template <int Bound, int Vec, bool Masked, bool FastQuant = false>
__global__ __launch_bounds__(256)
void swiglu_oai_int8_quant_rows_kernel(
    const c10::BFloat16* __restrict__ x, int8_t* __restrict__ q,
    float* __restrict__ scales, const int32_t* __restrict__ counts,
    int64_t rows, int capacity, int width,
    float alpha, float beta, float limit, bool has_limit) {
  constexpr int Wave = 64, Rows = 4;
  constexpr int Segments = (Bound + Wave * Vec - 1) / (Wave * Vec);
  const int lane = threadIdx.x & (Wave - 1);
  const int64_t row = int64_t(blockIdx.x) * Rows + threadIdx.x / Wave;
  if (row >= rows) return;
  using InVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutVec = at::native::memory::aligned_vector<int8_t, Vec>;
  if constexpr (Masked) {
    // Wave-uniform branch. Do not read uninitialized GEMM padding.
    if (row % capacity >= counts[row / capacity]) {
      if (lane == 0) scales[row] = 1.0f;
      const OutVec zero{};
      for (int col = lane * Vec; col < width; col += Wave * Vec)
        *reinterpret_cast<OutVec*>(q + row * width + col) = zero;
      return;
    }
  }
  float values[Segments * Vec];
  float maximum = 0.0f;
  const auto* gate = x + row * (2 * width);
  const auto* up = gate + width;
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Wave + lane) * Vec;
    if (col < width) {
      const InVec g = *reinterpret_cast<const InVec*>(gate + col);
      const InVec u = *reinterpret_cast<const InVec*>(up + col);
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        float gv = float(g.val[v]), uv = float(u.val[v]);
        if (has_limit) {
          gv = fminf(gv, limit);
          uv = fminf(fmaxf(uv, -limit), limit);
        }
        // Same explicit FP32 -> BF16 -> FP32 boundary as the baseline.
        const float sigmoid = 1.0f / (1.0f + __expf(-alpha * gv));
        const float value = float(c10::BFloat16((gv * sigmoid) * (uv + beta)));
        values[s * Vec + v] = value;
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
  }
#pragma unroll
  for (int offset = 32; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor(maximum, offset, Wave));
  const float scale = fmaxf(maximum, 1.0e-10f) / 127.0f;
  if (lane == 0) scales[row] = scale;
  float inv_scale = 0.0f;
  if constexpr (FastQuant) inv_scale = __fdiv_rn(1.0f, scale);
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Wave + lane) * Vec;
    if (col < width) {
      OutVec packed;
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        // No approximate reciprocal substitution at INT8 rounding boundaries.
        float value;
        if constexpr (FastQuant) {
          // With |value/scale|<=127, correctly rounded reciprocal + multiply
          // differs from correctly rounded division by < 2^-15 absolute.
          // Use a wider 1e-4 guard around half-integers and divide exactly
          // there. This preserves the integer rounding, including ties.
          const float scaled = values[s * Vec + v] * inv_scale;
          value = nearbyintf(scaled);
          if (fabsf(scaled - value) >= 0.4999f)
            value = nearbyintf(__fdiv_rn(values[s * Vec + v], scale));
        } else {
          value = nearbyintf(__fdiv_rn(values[s * Vec + v], scale));
        }
        packed.val[v] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, value)));
      }
      *reinterpret_cast<OutVec*>(q + row * width + col) = packed;
    }
  }
}

template <int Bound, int Vec, bool Masked>
__global__ __launch_bounds__(256)
void swiglu_oai_int8_quant_lut_kernel(
    const c10::BFloat16* __restrict__ x, int8_t* __restrict__ q,
    float* __restrict__ scales, const int32_t* __restrict__ counts, const float* __restrict__ gate_table,
    int64_t rows, int capacity, int width,
    float alpha, float beta, float limit, bool has_limit) {
  constexpr int Wave = 64, Rows = 4;
  constexpr int Segments = (Bound + Wave * Vec - 1) / (Wave * Vec);
  const int lane = threadIdx.x & (Wave - 1);
  const int64_t row = int64_t(blockIdx.x) * Rows + threadIdx.x / Wave;
  if (row >= rows) return;
  using InVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutVec = at::native::memory::aligned_vector<int8_t, Vec>;
  if constexpr (Masked) {
    // Wave-uniform branch. Do not read uninitialized GEMM padding.
    if (row % capacity >= counts[row / capacity]) {
      if (lane == 0) scales[row] = 1.0f;
      const OutVec zero{};
      for (int col = lane * Vec; col < width; col += Wave * Vec)
        *reinterpret_cast<OutVec*>(q + row * width + col) = zero;
      return;
    }
  }
  float values[Segments * Vec];
  float maximum = 0.0f;
  const auto* gate = x + row * (2 * width);
  const auto* up = gate + width;
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Wave + lane) * Vec;
    if (col < width) {
      const InVec g = *reinterpret_cast<const InVec*>(gate + col);
      const InVec u = *reinterpret_cast<const InVec*>(up + col);
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        float uv = float(u.val[v]);
        if (has_limit) uv = fminf(fmaxf(uv, -limit), limit);
        const float value = float(c10::BFloat16(gate_table[g.val[v].x] * (uv + beta)));
        values[s * Vec + v] = value;
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
  }
#pragma unroll
  for (int offset = 32; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor(maximum, offset, Wave));
  const float scale = fmaxf(maximum, 1.0e-10f) / 127.0f;
  if (lane == 0) scales[row] = scale;
  float inv_scale = 0.0f;
  if constexpr (true) inv_scale = __fdiv_rn(1.0f, scale);
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Wave + lane) * Vec;
    if (col < width) {
      OutVec packed;
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        // No approximate reciprocal substitution at INT8 rounding boundaries.
        float value;
        if constexpr (true) {
          // With |value/scale|<=127, correctly rounded reciprocal + multiply
          // differs from correctly rounded division by < 2^-15 absolute.
          // Use a wider 1e-4 guard around half-integers and divide exactly
          // there. This preserves the integer rounding, including ties.
          const float scaled = values[s * Vec + v] * inv_scale;
          value = nearbyintf(scaled);
          if (fabsf(scaled - value) >= 0.4999f)
            value = nearbyintf(__fdiv_rn(values[s * Vec + v], scale));
        } else {
          value = nearbyintf(__fdiv_rn(values[s * Vec + v], scale));
        }
        packed.val[v] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, value)));
      }
      *reinterpret_cast<OutVec*>(q + row * width + col) = packed;
    }
  }
}


template <int Bound, int Vec, bool Masked, int Threads>
__global__ __launch_bounds__(Threads)
void swiglu_oai_int8_quant_block_kernel(
    const c10::BFloat16* __restrict__ x, int8_t* __restrict__ q,
    float* __restrict__ scales, const int32_t* __restrict__ counts,
    int64_t rows, int capacity, int width,
    float alpha, float beta, float limit, bool has_limit) {
  constexpr int Wave = 64;
  constexpr int Segments = (Bound + Threads * Vec - 1) / (Threads * Vec);
  const int lane = threadIdx.x;
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  using InVec = at::native::memory::aligned_vector<c10::BFloat16, Vec>;
  using OutVec = at::native::memory::aligned_vector<int8_t, Vec>;
  if constexpr (Masked) {
    // Wave-uniform branch. Do not read uninitialized GEMM padding.
    if (row % capacity >= counts[row / capacity]) {
      if (lane == 0) scales[row] = 1.0f;
      const OutVec zero{};
      for (int col = lane * Vec; col < width; col += Threads * Vec)
        *reinterpret_cast<OutVec*>(q + row * width + col) = zero;
      return;
    }
  }
  float values[Segments * Vec];
  float maximum = 0.0f;
  const auto* gate = x + row * (2 * width);
  const auto* up = gate + width;
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Threads + lane) * Vec;
    if (col < width) {
      const InVec g = *reinterpret_cast<const InVec*>(gate + col);
      const InVec u = *reinterpret_cast<const InVec*>(up + col);
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        float gv = float(g.val[v]), uv = float(u.val[v]);
        if (has_limit) {
          gv = fminf(gv, limit);
          uv = fminf(fmaxf(uv, -limit), limit);
        }
        // Same explicit FP32 -> BF16 -> FP32 boundary as the baseline.
        const float sigmoid = 1.0f / (1.0f + __expf(-alpha * gv));
        const float value = float(c10::BFloat16((gv * sigmoid) * (uv + beta)));
        values[s * Vec + v] = value;
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
  }
#pragma unroll
  for (int offset = 32; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor(maximum, offset, Wave));
  __shared__ float partial[Threads / Wave];
  if (lane % Wave == 0) partial[lane / Wave] = maximum;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < Threads / Wave; ++w) maximum = fmaxf(maximum, partial[w]);
  const float scale = fmaxf(maximum, 1.0e-10f) / 127.0f;
  if (lane == 0) scales[row] = scale;
#pragma unroll
  for (int s = 0; s < Segments; ++s) {
    const int col = (s * Threads + lane) * Vec;
    if (col < width) {
      OutVec packed;
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        // No approximate reciprocal substitution at INT8 rounding boundaries.
        float value = nearbyintf(__fdiv_rn(values[s * Vec + v], scale));
        packed.val[v] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, value)));
      }
      *reinterpret_cast<OutVec*>(q + row * width + col) = packed;
    }
  }
}


template <int Bound, int Vec, bool Masked, bool FastQuant = false>
void launch(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
            const int32_t* counts, float alpha, float beta, float limit,
            bool has_limit) {
  const int64_t rows = x.numel() / x.size(-1);
  swiglu_oai_int8_quant_rows_kernel<Bound, Vec, Masked, FastQuant>
      <<<(rows + 3) / 4, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          x.data_ptr<c10::BFloat16>(), q.data_ptr<int8_t>(), s.data_ptr<float>(),
          counts, rows, int(x.size(-2)), int(x.size(-1) / 2),
          alpha, beta, limit, has_limit);
}

template <int Vec, bool Masked, bool FastQuant = false>
void dispatch(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
              const int32_t* counts, float alpha, float beta, float limit,
              bool has_limit) {
  const auto width = x.size(-1) / 2;
#define LAUNCH(B) launch<B, Vec, Masked, FastQuant>(x, q, s, counts, alpha, beta, limit, has_limit)
  if (width <= 512) LAUNCH(512);
  else if (width <= 1024) LAUNCH(1024);
  else if (width <= 2048) LAUNCH(2048);
  else if (width <= 3072) LAUNCH(3072);
  else if (width <= 4096) LAUNCH(4096);
  else LAUNCH(8192);
#undef LAUNCH
}

void validate(const torch::Tensor& x, const torch::Tensor& q,
              const torch::Tensor& s, double alpha, double beta, double limit) {
  TORCH_CHECK(x.is_cuda() && q.device() == x.device() && s.device() == x.device(),
              "All tensors must share a HIP device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                  (x.dim() == 2 || x.dim() == 3),
              "Expected contiguous BF16 [rows,2H] or [E,M,2H]");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) % 2 == 0 && x.size(-1) <= 16384,
              "Expected 1 <= H <= 8192");
  auto shape = x.sizes().vec();
  shape.back() /= 2;
  TORCH_CHECK(q.scalar_type() == at::kChar && q.is_contiguous() &&
                  q.sizes() == c10::IntArrayRef(shape), "Invalid INT8 output");
  shape.back() = 1;
  TORCH_CHECK(s.scalar_type() == at::kFloat && s.is_contiguous() &&
                  s.sizes() == c10::IntArrayRef(shape), "Invalid FP32 scales");
  TORCH_CHECK(!x.is_alias_of(q) && !x.is_alias_of(s) && !q.is_alias_of(s),
              "Buffers must not share storage");
  TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta) && std::isfinite(limit) &&
                  std::abs(alpha) <= std::numeric_limits<float>::max() &&
                  std::abs(beta) <= std::numeric_limits<float>::max() &&
                  limit >= 0 && limit <= std::numeric_limits<float>::max(),
              "Invalid OAI parameters (limit=0 means no clamp)");
  TORCH_CHECK(x.size(-2) <= std::numeric_limits<int32_t>::max() &&
                  (x.numel() / x.size(-1) + 3) / 4 <=
                      std::numeric_limits<uint32_t>::max(),
              "Input exceeds launch indexing range");
}
template <bool Masked>
void run(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
         const int32_t* counts, double alpha, double beta, double limit) {
  if (x.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const bool aligned = reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0;
  if (x.size(-1) / 2 % 8 == 0 && aligned)
    dispatch<8, Masked>(x, q, s, counts, alpha, beta, limit, limit > 0);
  else
    dispatch<1, Masked>(x, q, s, counts, alpha, beta, limit, limit > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bound, bool Masked, int Threads>
void launch_block(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
                  const int32_t* counts, double alpha, double beta, double limit) {
  const auto rows = x.numel() / x.size(-1);
  swiglu_oai_int8_quant_block_kernel<Bound, 8, Masked, Threads>
      <<<rows, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          x.data_ptr<c10::BFloat16>(), q.data_ptr<int8_t>(), s.data_ptr<float>(),
          counts, rows, int(x.size(-2)), int(x.size(-1)/2),
          alpha, beta, limit, limit > 0);
}
template <bool Masked, int Threads>
void dispatch_block(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
                    const int32_t* counts, double alpha, double beta, double limit) {
  const auto h=x.size(-1)/2;
#define BL(B) launch_block<B, Masked, Threads>(x,q,s,counts,alpha,beta,limit)
  if(h<=512) BL(512);
  else if(h<=1024) BL(1024);
  else if(h<=2048) BL(2048);
  else if(h<=3072) BL(3072);
  else if(h<=4096) BL(4096);
  else BL(8192);
#undef BL
}

__global__ void swiglu_oai_gate_table_kernel(float* table, float alpha, float limit) {
  const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=65536) return;
  c10::BFloat16 input;
  input.x=static_cast<uint16_t>(i);
  float gate=float(input);
  if(limit>0) gate=fminf(gate,limit);
  const float sigmoid=1.0f/(1.0f+__expf(-alpha*gate));
  table[i]=gate*sigmoid;
}
template <int Bound, int Vec, bool Masked>
void launch_lut(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
                const int32_t* counts, const float* table,
                double alpha, double beta, double limit) {
  const auto rows=x.numel()/x.size(-1);
  swiglu_oai_int8_quant_lut_kernel<Bound,Vec,Masked>
      <<<(rows+3)/4,256,0,at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<c10::BFloat16>(),q.data_ptr<int8_t>(),s.data_ptr<float>(),
        counts,table,rows,int(x.size(-2)),int(x.size(-1)/2),
        alpha,beta,limit,limit>0);
}
template <int Vec, bool Masked>
void dispatch_lut(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& s,
                  const int32_t* counts,const float* table,
                  double alpha,double beta,double limit) {
  const auto h=x.size(-1)/2;
#define LUT(B) launch_lut<B,Vec,Masked>(x,q,s,counts,table,alpha,beta,limit)
  if(h<=512) LUT(512);
  else if(h<=1024) LUT(1024);
  else if(h<=2048) LUT(2048);
  else if(h<=3072) LUT(3072);
  else if(h<=4096) LUT(4096);
  else LUT(8192);
#undef LUT
}
}  // namespace

void swiglu_oai_int8_quant_validate(const torch::Tensor& x,const torch::Tensor& q,const torch::Tensor& s,double alpha,double beta,double limit) { validate(x,q,s,alpha,beta,limit); }

void swiglu_oai_int8_quant_contiguous(
    const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scales,
    double alpha, double beta, double limit) {
  validate(x, q, scales, alpha, beta, limit);
  TORCH_CHECK(x.dim() == 2, "Contiguous entry expects [rows,2H]");
  run<false>(x, q, scales, nullptr, alpha, beta, limit);
}

void swiglu_oai_int8_quant_masked(
    const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scales,
    const torch::Tensor& counts, double alpha, double beta, double limit) {
  validate(x, q, scales, alpha, beta, limit);
  TORCH_CHECK(x.dim() == 3 && counts.device() == x.device() &&
                  counts.scalar_type() == at::kInt && counts.is_contiguous() &&
                  counts.dim() == 1 && counts.numel() == x.size(0),
              "Masked entry expects [E,M,2H] and device int32 counts[E]");
  TORCH_CHECK(!counts.is_alias_of(x) && !counts.is_alias_of(q) &&
                  !counts.is_alias_of(scales), "Counts must not alias buffers");
  run<true>(x, q, scales, counts.data_ptr<int32_t>(), alpha, beta, limit);
}

// Experimental launch configurations for bounded A/B, not the public default.
void swiglu_oai_int8_quant_block_config(
    const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scales,
    const torch::Tensor& counts, double alpha, double beta, double limit,
    int64_t threads) {
  validate(x,q,scales,alpha,beta,limit);
  TORCH_CHECK(threads==128 || threads==256, "Expected 128 or 256 threads");
  TORCH_CHECK(x.size(-1)/2%8==0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr())%8==0,
              "Block candidate requires vector-aligned input/output and H%8=0");
  if(x.numel()==0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  if(x.dim()==3) {
    TORCH_CHECK(counts.defined() && counts.device()==x.device() &&
                    counts.scalar_type()==at::kInt && counts.dim()==1 &&
                    counts.is_contiguous() && counts.numel()==x.size(0) &&
                    !counts.is_alias_of(x) && !counts.is_alias_of(q) &&
                    !counts.is_alias_of(scales), "Invalid counts");
    if(threads==128) dispatch_block<true,128>(x,q,scales,counts.data_ptr<int32_t>(),alpha,beta,limit);
    else dispatch_block<true,256>(x,q,scales,counts.data_ptr<int32_t>(),alpha,beta,limit);
  } else {
    if(threads==128) dispatch_block<false,128>(x,q,scales,nullptr,alpha,beta,limit);
    else dispatch_block<false,256>(x,q,scales,nullptr,alpha,beta,limit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Exact-integer guarded reciprocal candidate. Defaults remain unchanged.
void swiglu_oai_int8_quant_fast_config(
    const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scales,
    const torch::Tensor& counts, double alpha, double beta, double limit) {
  validate(x,q,scales,alpha,beta,limit);
  if(x.numel()==0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const bool vectorized = x.size(-1)/2%8==0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%8==0;
  const bool masked=x.dim()==3;
  const int32_t* ptr=nullptr;
  if(masked) {
    TORCH_CHECK(counts.defined() && counts.device()==x.device() &&
        counts.scalar_type()==at::kInt && counts.dim()==1 &&
        counts.is_contiguous() && counts.numel()==x.size(0) &&
        !counts.is_alias_of(x) && !counts.is_alias_of(q) &&
        !counts.is_alias_of(scales), "Invalid counts");
    ptr=counts.data_ptr<int32_t>();
  }
  if(masked) {
    if(vectorized) dispatch<8,true,true>(x,q,scales,ptr,alpha,beta,limit,limit>0);
    else dispatch<1,true,true>(x,q,scales,ptr,alpha,beta,limit,limit>0);
  } else {
    if(vectorized) dispatch<8,false,true>(x,q,scales,ptr,alpha,beta,limit,limit>0);
    else dispatch<1,false,true>(x,q,scales,ptr,alpha,beta,limit,limit>0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_oai_build_gate_table(torch::Tensor& table,double alpha,double limit) {
  TORCH_CHECK(table.is_cuda() && table.scalar_type()==at::kFloat &&
      table.is_contiguous() && table.numel()==65536,"Expected 65536 FP32 HIP entries");
  TORCH_CHECK(std::isfinite(alpha) && std::isfinite(limit) && limit>=0,
      "Invalid table parameters");
  const c10::cuda::CUDAGuard guard(table.device());
  swiglu_oai_gate_table_kernel<<<256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      table.data_ptr<float>(),alpha,limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void swiglu_oai_int8_quant_lut_config(
    const torch::Tensor& x,torch::Tensor& q,torch::Tensor& scales,
    const torch::Tensor& counts,const torch::Tensor& table,
    double alpha,double beta,double limit) {
  validate(x,q,scales,alpha,beta,limit);
  TORCH_CHECK(table.device()==x.device() && table.scalar_type()==at::kFloat &&
      table.is_contiguous() && table.numel()==65536 &&
      !table.is_alias_of(x) && !table.is_alias_of(q) &&
      !table.is_alias_of(scales),"Invalid gate table");
  if(x.numel()==0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const bool masked=x.dim()==3;
  const bool vec=x.size(-1)/2%8==0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%8==0;
  const int32_t* ptr=nullptr;
  if(masked) {
    TORCH_CHECK(counts.defined() && counts.device()==x.device() &&
      counts.scalar_type()==at::kInt && counts.dim()==1 && counts.is_contiguous() &&
      counts.numel()==x.size(0) && !counts.is_alias_of(x) && !counts.is_alias_of(q) &&
      !counts.is_alias_of(scales) && !counts.is_alias_of(table),"Invalid counts");
    ptr=counts.data_ptr<int32_t>();
  }
  if(masked) {
    if(vec) dispatch_lut<8,true>(x,q,scales,ptr,table.data_ptr<float>(),alpha,beta,limit);
    else dispatch_lut<1,true>(x,q,scales,ptr,table.data_ptr<float>(),alpha,beta,limit);
  } else {
    if(vec) dispatch_lut<8,false>(x,q,scales,ptr,table.data_ptr<float>(),alpha,beta,limit);
    else dispatch_lut<1,false>(x,q,scales,ptr,table.data_ptr<float>(),alpha,beta,limit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace at::native
