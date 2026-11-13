// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Native wave64 mapping follows this repository's csrc/gemm.cu.
#include "hcu_mmac.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <string_view>
#include <limits>
#include <optional>

namespace at {
namespace native {
namespace hc_mix_detail {
// Algorithm tile size, not a model shape: the last slice is zero-padded.
constexpr int kSliceSize = 256;
using B4 = short __attribute__((vector_size(8)));
using F4 = float __attribute__((vector_size(16)));

__device__ __forceinline__ void mmac(B4 a, B4 b, F4& acc) {
#if (defined(__gfx928__) || defined(__gfx92a__)) || defined(__gfx936__)
  acc = HCU_MMAC_F32_BF16(a, b, acc);
#elif defined(__gfx938__)
  acc = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, acc, false, false);
#else
  // Runtime dispatch rejects other architectures before launching.
  __builtin_trap();
#endif
}

// gfx938 RNE conversion follows softmax_autograd.cu; earlier gfx use the
// Torch software conversion. The tensor-core builtin also differs by gfx.
__device__ __forceinline__ short to_bf16(float value) {
#if defined(__gfx938__)
  return __builtin_hcu_cvt_bf16_f32(value, false, false);
#else
  return at::BFloat16(value).x;
#endif
}

// Eight waves per CTA: four contiguous K sub-slices for each N tile.
// S=ceil(K/256). Slice boundaries and arithmetic order are independent of M.
// Static dimensions are optional specializations; zero selects runtime shapes.
template <int StaticK, int StaticN>
__global__ void hc_mix_stats_partial_kernel(
    const at::BFloat16* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ ph, float* __restrict__ pl,
    float* __restrict__ ps, int64_t m, int runtime_k, int runtime_n) {
  const int K = StaticK ? StaticK : runtime_k;
  const int N = StaticN ? StaticN : runtime_n;
  const int wave = threadIdx.x / 64;
  const int lane = threadIdx.x % 64;
  const int row_lane = lane % 16;
  const int group = lane / 16;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * 16 + row_lane;
  const int n_wave = wave % 2;
  const int k_wave = wave / 2;
  const int slice = blockIdx.y;
  F4 hi = {}, lo = {};
  float sq = 0.0f;
#pragma unroll 4
  for (int tile = k_wave * 64; tile < (k_wave + 1) * 64; tile += 16) {
    // Consecutive lanes read consecutive K groups before the register transpose.
    const int load_row = lane / 4;
    const int load_group = lane % 4;
    const int64_t xrow = static_cast<int64_t>(blockIdx.x) * 16 + load_row;
    const int wrow = (StaticN ? 0 : blockIdx.z * 32) + n_wave * 16 + load_row;
    const int kk = slice * kSliceSize + tile + load_group * 4;
    B4 xv = {}, wh = {}, wl = {};
    F4 fv = {};
    if (xrow < m) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (StaticK || kk + j < K) xv[j] = x[xrow * K + kk + j].x;
    }
    if (wrow < N && (StaticK || kk < K)) {
      const float* wp = w + static_cast<int64_t>(wrow) * K + kk;
      if (StaticK || (kk + 3 < K && (reinterpret_cast<uintptr_t>(wp) & 15) == 0))
        fv = *reinterpret_cast<const F4*>(wp);
      else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (kk + j < K) fv[j] = w[static_cast<int64_t>(wrow) * K + kk + j];
      }
    }
    const int source_lane = row_lane * 4 + group;
    unsigned int x01 = static_cast<unsigned short>(xv[0]) |
                      (static_cast<unsigned int>(static_cast<unsigned short>(xv[1])) << 16);
    unsigned int x23 = static_cast<unsigned short>(xv[2]) |
                      (static_cast<unsigned int>(static_cast<unsigned short>(xv[3])) << 16);
    x01 = __shfl(x01, source_lane, 64);
    x23 = __shfl(x23, source_lane, 64);
    xv = B4{static_cast<short>(x01), static_cast<short>(x01 >> 16),
            static_cast<short>(x23), static_cast<short>(x23 >> 16)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float value = __shfl(fv[j], source_lane, 64);
      wh[j] = to_bf16(value);
      const float high = static_cast<float>(at::BFloat16(wh[j], at::BFloat16::from_bits()));
      wl[j] = to_bf16((value - high) * 256.0f);
    }
    mmac(xv, wh, hi);
    mmac(xv, wl, lo);
    if (n_wave == 0) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v = static_cast<float>(at::BFloat16(xv[j], at::BFloat16::from_bits()));
        sq += v * v;
      }
    }
  }
  __shared__ F4 hs[3][128], ls[3][128];
  __shared__ float qs[3][16];
  if (k_wave > 0) {
    hs[k_wave - 1][n_wave * 64 + lane] = hi;
    ls[k_wave - 1][n_wave * 64 + lane] = lo;
  }
  if (n_wave == 0) {
    sq += __shfl_xor(sq, 16, 64);
    sq += __shfl_xor(sq, 32, 64);
    if (k_wave > 0 && group == 0) qs[k_wave - 1][row_lane] = sq;
  }
  __syncthreads();
  if (k_wave == 0 && row < m) {
    const int idx = n_wave * 64 + lane;
    hi = (hi + hs[1][idx]) + (hs[0][idx] + hs[2][idx]);
    lo = (lo + ls[1][idx]) + (ls[0][idx] + ls[2][idx]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = (StaticN ? 0 : blockIdx.z * 32) + n_wave * 16 + group + j * 4;
      if (col < N) {
        const int64_t off = (static_cast<int64_t>(slice) * m + row) * N + col;
        ph[off] = hi[j];
        pl[off] = lo[j];
      }
    }
    if ((StaticN || blockIdx.z == 0) && n_wave == 0 && group == 0)
      ps[static_cast<int64_t>(slice) * m + row] =
          (sq + qs[1][row_lane]) + (qs[0][row_lane] + qs[2][row_lane]);
  }

}

// Larger M tile: two physical K groups execute four logical 64-element
// groups. Pair (0+2) and (1+3) before LDS; this preserves the small-M tree.
template <int K, int N, int RowTiles>
__global__ void hc_mix_stats_partial_mtile_kernel(
    const at::BFloat16* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ ph, float* __restrict__ pl, float* __restrict__ ps,
    int64_t m) {
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int n_wave = wave % 2, k_wave = wave / 2;
  const int row_lane = lane % 16, group = lane / 16;
  const int load_row = lane / 4, load_group = lane % 4;
  const int source_lane = row_lane * 4 + group;
  const int slice = blockIdx.y;
  F4 hi[RowTiles][2] = {}, lo[RowTiles][2] = {};
  float sq[RowTiles][2] = {};
#pragma unroll
  for (int q = 0; q < 2; ++q) {
#pragma unroll
    for (int tile = 0; tile < 64; tile += 16) {
      const int kk = slice * kSliceSize + (k_wave + 2*q) * 64 + tile + load_group * 4;
      const int wrow = n_wave * 16 + load_row;
      F4 fv = {};
      if (wrow < N) fv = *reinterpret_cast<const F4*>(w + wrow*K + kk);
      B4 wh = {}, wl = {};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value = __shfl(fv[j], source_lane, 64);
        wh[j] = to_bf16(value);
        const float high = static_cast<float>(at::BFloat16(wh[j], at::BFloat16::from_bits()));
        wl[j] = to_bf16((value-high)*256.0f);
      }
#pragma unroll
      for (int r = 0; r < RowTiles; ++r) {
        const int64_t xrow = static_cast<int64_t>(blockIdx.x)*(16*RowTiles) + r*16 + load_row;
        B4 xv = {};
        if (xrow < m) {
#pragma unroll
          for (int j = 0; j < 4; ++j) xv[j] = x[xrow*K+kk+j].x;
        }
        unsigned int x01 = static_cast<unsigned short>(xv[0]) |
            (static_cast<unsigned int>(static_cast<unsigned short>(xv[1])) << 16);
        unsigned int x23 = static_cast<unsigned short>(xv[2]) |
            (static_cast<unsigned int>(static_cast<unsigned short>(xv[3])) << 16);
        x01 = __shfl(x01, source_lane, 64); x23 = __shfl(x23, source_lane, 64);
        xv = B4{static_cast<short>(x01),static_cast<short>(x01>>16),
                static_cast<short>(x23),static_cast<short>(x23>>16)};
        mmac(xv, wh, hi[r][q]); mmac(xv, wl, lo[r][q]);
        if (n_wave == 0) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float v = static_cast<float>(at::BFloat16(xv[j],at::BFloat16::from_bits()));
            sq[r][q] += v*v;
          }
        }
      }
    }
  }
  __shared__ F4 hs[RowTiles][128], ls[RowTiles][128];
  __shared__ float qs[RowTiles][16];
#pragma unroll
  for (int r = 0; r < RowTiles; ++r) {
    hi[r][0] += hi[r][1]; lo[r][0] += lo[r][1];
    if (k_wave == 1) {
      hs[r][n_wave*64+lane] = hi[r][0];
      ls[r][n_wave*64+lane] = lo[r][0];
    }
    if (n_wave == 0) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        sq[r][q] += __shfl_xor(sq[r][q],16,64);
        sq[r][q] += __shfl_xor(sq[r][q],32,64);
      }
      sq[r][0] += sq[r][1];
      if (k_wave == 1 && group == 0) qs[r][row_lane] = sq[r][0];
    }
  }
  __syncthreads();
  if (k_wave == 0) {
#pragma unroll
    for (int r = 0; r < RowTiles; ++r) {
      const int64_t row = static_cast<int64_t>(blockIdx.x)*(16*RowTiles) + r*16 + row_lane;
      if (row < m) {
        const F4 hr = hi[r][0] + hs[r][n_wave*64+lane];
        const F4 lr = lo[r][0] + ls[r][n_wave*64+lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int col = n_wave*16 + group + j*4;
          if (col < N) {
            const int64_t off = (static_cast<int64_t>(slice)*m+row)*N+col;
            ph[off]=hr[j]; pl[off]=lr[j];
          }
        }
        if (n_wave == 0 && group == 0)
          ps[static_cast<int64_t>(slice)*m+row] = sq[r][0]+qs[r][row_lane];
      }
    }
  }
}


__device__ __forceinline__ F4 load4(const float* ptr) {
  if ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0)
    return *reinterpret_cast<const F4*>(ptr);
  return F4{ptr[0], ptr[1], ptr[2], ptr[3]};
}

template <int StaticK, int StaticN>
__global__ void hc_mix_stats_reduce_kernel(
    const float* __restrict__ ph, const float* __restrict__ pl,
    const float* __restrict__ ps, float* __restrict__ out,
    int64_t m, int runtime_k, int runtime_n, int splits, float eps) {
  const int K = StaticK ? StaticK : runtime_k;
  const int N = StaticN ? StaticN : runtime_n;
  const int S = StaticK ? (StaticK + kSliceSize - 1) / kSliceSize : splits;
  const int row = blockIdx.x;
  const int col = threadIdx.x % 8;
  const int group = threadIdx.x / 8;
  const int wave = threadIdx.x / 64;
  F4 hi = {}, lo = {};
  float sq = 0.0f;
#pragma unroll
  for (int s = group; s < S; s += 32) {
    if (col < 6) {
      const int ncol = (StaticN ? 0 : blockIdx.y * 24) + col * 4;
      const int64_t off = (static_cast<int64_t>(s) * m + row) * N + ncol;
      if (ncol + 3 < N) {
        hi += load4(ph + off); lo += load4(pl + off);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (ncol + j < N) { hi[j] += ph[off+j]; lo[j] += pl[off+j]; }
        }
      }
    } else if (col == 6) {
      sq += ps[static_cast<int64_t>(s) * m + row];
    }
  }
#pragma unroll
  for (int delta = 32; delta >= 8; delta /= 2) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      hi[j] += __shfl_xor(hi[j], delta, 64);
      lo[j] += __shfl_xor(lo[j], delta, 64);
    }
    sq += __shfl_xor(sq, delta, 64);
  }
  __shared__ F4 hs[4][6], ls[4][6];
  __shared__ float qs[4];
  if (group % 8 == 0) {
    if (col < 6) { hs[wave][col] = hi; ls[wave][col] = lo; }
    else if (col == 6) qs[wave] = sq;
  }
  __syncthreads();
  if (group == 0 && col < 6) {
    hi = (hs[0][col] + hs[2][col]) + (hs[1][col] + hs[3][col]);
    lo = (ls[0][col] + ls[2][col]) + (ls[1][col] + ls[3][col]);
    const float inv = rsqrtf(((qs[0] + qs[2]) + (qs[1] + qs[3])) * (1.0f / K) + eps);
    const F4 value = (hi + lo * (1.0f / 256.0f)) * inv;
    const int ncol = (StaticN ? 0 : blockIdx.y * 24) + col * 4;
    if (ncol >= N) return;
    float* dst = out + static_cast<int64_t>(row) * N + ncol;
    if (StaticN || (ncol + 3 < N && (reinterpret_cast<uintptr_t>(dst) & 15) == 0))
      *reinterpret_cast<F4*>(dst) = value;
    else {
#pragma unroll
      for (int j = 0; j < 4; ++j) if (ncol + j < N) dst[j] = value[j];
    }
  }
}

// Inference-only. Unsupported metadata returns None before allocation or launch.
bool supports_hc_mix_stats(const Tensor& x, const Tensor& w, double eps) {
  if (!x.is_cuda() || w.device() != x.device() ||
      x.scalar_type() != at::kBFloat16 || w.scalar_type() != at::kFloat)
    return false;
  // Raw-pointer kernels require ordinary contiguous matrices and resolved signs.
  if (x.layout() != at::kStrided || w.layout() != at::kStrided ||
      x.is_nested() || w.is_nested() || x.dim() != 2 || w.dim() != 2 ||
      !x.is_contiguous() || !w.is_contiguous() || x.is_neg() || w.is_neg())
    return false;
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  if (k <= 0 || w.size(1) != k || m > std::numeric_limits<int>::max() ||
      k > 65535LL * kSliceSize || n > 65535LL * 24)
    return false;
  const float eps32 = static_cast<float>(eps);
  if (!(eps32 > 0) || !std::isfinite(eps32)) return false;
  // Cached device properties; string_view avoids allocating an architecture name.
  auto arch = std::string_view(at::cuda::getDeviceProperties(x.get_device())->gcnArchName);
  arch = arch.substr(0, arch.find(':'));
  return arch == "gfx928" || arch == "gfx936" || arch == "gfx938";
}
}  // namespace hc_mix_detail

std::optional<Tensor> hc_mix_stats(const Tensor& x, const Tensor& w, double eps) {
  using namespace hc_mix_detail;
  if (!supports_hc_mix_stats(x, w, eps)) return std::nullopt;
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  const int64_t splits = (k + kSliceSize - 1) / kSliceSize;
  const c10::cuda::CUDAGuard guard(x.device());
  auto opts = x.options().dtype(at::kFloat);
  Tensor out = at::empty({m,n},opts);
  if (!m || !n) return out;

  // Per-call allocator-owned workspace; both launches use the current stream.
  Tensor ph = at::empty({splits,m,n},opts), pl = at::empty({splits,m,n},opts), ps = at::empty({splits,m},opts);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid((m+15)/16, splits, (n+31)/32);
  // Common model shape is a specialization, not the public API contract.
  if (k == 20480 && n == 24 && m >= 256 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0)
    hc_mix_stats_partial_mtile_kernel<20480,24,2><<<dim3((m+31)/32,splits),256,0,stream>>>(
        x.data_ptr<at::BFloat16>(),w.data_ptr<float>(),ph.data_ptr<float>(),
        pl.data_ptr<float>(),ps.data_ptr<float>(),m);
  else if (k == 20480 && n == 24 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0)
    hc_mix_stats_partial_kernel<20480,24><<<partial_grid,512,0,stream>>>(
        x.data_ptr<at::BFloat16>(),w.data_ptr<float>(),ph.data_ptr<float>(),
        pl.data_ptr<float>(),ps.data_ptr<float>(),m,k,n);
  else
    hc_mix_stats_partial_kernel<0,0><<<partial_grid,512,0,stream>>>(
        x.data_ptr<at::BFloat16>(),w.data_ptr<float>(),ph.data_ptr<float>(),
        pl.data_ptr<float>(),ps.data_ptr<float>(),m,k,n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 reduce_grid(m,(n+23)/24);
  if (k == 20480 && n == 24)
    hc_mix_stats_reduce_kernel<20480,24><<<reduce_grid,256,0,stream>>>(
        ph.data_ptr<float>(),pl.data_ptr<float>(),ps.data_ptr<float>(),
        out.data_ptr<float>(),m,k,n,splits,static_cast<float>(eps));
  else
    hc_mix_stats_reduce_kernel<0,0><<<reduce_grid,256,0,stream>>>(
        ph.data_ptr<float>(),pl.data_ptr<float>(),ps.data_ptr<float>(),
        out.data_ptr<float>(),m,k,n,splits,static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace native
}  // namespace at
