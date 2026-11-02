// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "hcu_mmac.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <hip/hip_runtime.h>
#include <limits>
#include <string_view>
#include <tuple>
#include <vector>

namespace at {
namespace native {
namespace hc_sink_detail {
using F2 = float __attribute__((ext_vector_type(2)));
using F4 = float __attribute__((ext_vector_type(4)));
using B4 = short __attribute__((ext_vector_type(4)));
constexpr int K = 20480, N = 24;
__device__ __forceinline__ float tf32(float v) {
  // SDK du_mma.h RNE, preserving Inf and NaN.
  unsigned u = __float_as_uint(v);
  if ((u & 0x7f800000u) != 0x7f800000u)
    u += 0xfffu + ((u >> 13) & 1u);
  else if (u & 0x1fffu)
    u |= 0x2000u;
  return __uint_as_float(u & 0xffffe000u);
}
__device__ __forceinline__ F4 mma(F2 a, F2 b, F4 c) {
#if (defined(__gfx928__) || defined(__gfx92a__)) || defined(__gfx936__)
  return HCU_MMAC_F32_TF32(a, b, c);
#elif defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(a, b, c, false, false);
#else
  __builtin_trap();
  return c;
#endif
}
__device__ __forceinline__ F4 mma(B4 a, B4 b, F4 c) {
#if (defined(__gfx928__) || defined(__gfx92a__)) || defined(__gfx936__)
  return HCU_MMAC_F32_BF16(a, b, c);
#elif defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false);
#else
  __builtin_trap();
  return c;
#endif
}
__device__ __forceinline__ float bf(short v) {
  return __uint_as_float(static_cast<unsigned>(static_cast<unsigned short>(v))
                         << 16);
}

// Two N waves, KG K groups. The group/slice reduction order is independent
// of batch padding within each specialization. X square sum is computed once.
template <bool TF, int KG, int RT, int S>
__global__ void
hc_parts_project(const BFloat16 *__restrict__ x, const void *__restrict__ w0,
                 const void *__restrict__ w1, const void *__restrict__ w2,
                 float *__restrict__ part, float *__restrict__ sqout,
                 int64_t m) {
  int lane = threadIdx.x % 64, wave = threadIdx.x / 64, nw = wave % 2,
      kw = wave / 2;
  int r = lane % 16, g = lane / 16, lr = lane / 4, lg = lane % 4,
      src = r * 4 + g;
  constexpr int E = TF ? 2 : 4, STEP = 4 * E, KS = K / S / KG;
  F4 hi[RT] = {}, lo[RT] = {}, mid[RT] = {};
  float sq[RT] = {};
  for (int kb = kw * KS; kb < (kw + 1) * KS; kb += STEP) {
    int kk = blockIdx.y * (K / S) + kb + lg * E, wr = nw * 16 + lr;
    F2 th = {}, tl = {};
    B4 bh = {}, bm = {}, bl = {};
    if constexpr (TF) {
      F2 hv = {}, lv = {};
      if (wr < N) {
        const float *hp = static_cast<const float *>(w0) + wr * K + kk;
        const float *lp = static_cast<const float *>(w1) + wr * K + kk;
        if ((reinterpret_cast<uintptr_t>(hp) & 7) == 0)
          hv = *reinterpret_cast<const F2 *>(hp);
        else
          hv = F2{hp[0], hp[1]};
        if ((reinterpret_cast<uintptr_t>(lp) & 7) == 0)
          lv = *reinterpret_cast<const F2 *>(lp);
        else
          lv = F2{lp[0], lp[1]};
      }
#pragma unroll
      for (int j = 0; j < 2; j++) {
        th[j] = __shfl(hv[j], src, 64);
        tl[j] = tf32(__shfl(lv[j], src, 64));
      }
    } else {
      const BFloat16 *weights[3] = {static_cast<const BFloat16 *>(w0),
                                    static_cast<const BFloat16 *>(w1),
                                    static_cast<const BFloat16 *>(w2)};
      B4 ww[3] = {};
#pragma unroll
      for (int c = 0; c < 3; c++) {
        unsigned a = 0, b = 0;
        if (wr < N) {
          const BFloat16 *ptr = weights[c] + wr * K + kk;
          a = unsigned(ptr[0].x) | (unsigned(ptr[1].x) << 16);
          b = unsigned(ptr[2].x) | (unsigned(ptr[3].x) << 16);
        }
        a = __shfl(a, src, 64);
        b = __shfl(b, src, 64);
        ww[c] = B4{short(a), short(a >> 16), short(b), short(b >> 16)};
      }
      bh = ww[0];
      bm = ww[1];
      bl = ww[2];
    }
#pragma unroll
    for (int rt = 0; rt < RT; rt++) {
      int64_t row = int64_t(blockIdx.x) * (16 * RT) + rt * 16 + lr;
      F2 tx = {};
      B4 bx = {};
      unsigned a = 0, b = 0;
      if (row < m) {
        const BFloat16 *ptr = x + row * K + kk;
        a = unsigned(ptr[0].x) | (unsigned(ptr[1].x) << 16);
        if constexpr (!TF)
          b = unsigned(ptr[2].x) | (unsigned(ptr[3].x) << 16);
      }
      a = __shfl(a, src, 64);
      if constexpr (TF) {
        tx = F2{bf(short(a)), bf(short(a >> 16))};
        if (nw == 0)
          sq[rt] += tx[0] * tx[0] + tx[1] * tx[1];
      } else {
        b = __shfl(b, src, 64);
        bx = B4{short(a), short(a >> 16), short(b), short(b >> 16)};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          float f = bf(bx[j]);
          if (nw == 0)
            sq[rt] += f * f;
        }
      }

      if constexpr (TF) {
        hi[rt] = mma(tx, th, hi[rt]);
        lo[rt] = mma(tx, tl, lo[rt]);
      } else {
        hi[rt] = mma(bx, bh, hi[rt]);
        mid[rt] = mma(bx, bm, mid[rt]);
        lo[rt] = mma(bx, bl, lo[rt]);
      }
    }
  }
  __shared__ F4 hs[KG > 1 ? KG - 1 : 1][RT][128],
      ls[KG > 1 ? KG - 1 : 1][RT][128],
      ms[TF ? 1 : (KG > 1 ? KG - 1 : 1)][RT][128];
  __shared__ float qs[KG > 1 ? KG - 1 : 1][RT][16];
#pragma unroll
  for (int rt = 0; rt < RT; rt++) {
    if constexpr (KG > 1) {
      if (kw) {
        hs[kw - 1][rt][nw * 64 + lane] = hi[rt];
        ls[kw - 1][rt][nw * 64 + lane] = lo[rt];
        if constexpr (!TF)
          ms[kw - 1][rt][nw * 64 + lane] = mid[rt];
      }
    }
    if (nw == 0) {
      sq[rt] += __shfl_xor(sq[rt], 16, 64);
      sq[rt] += __shfl_xor(sq[rt], 32, 64);
      if constexpr (KG > 1) {
        if (kw && g == 0)
          qs[kw - 1][rt][r] = sq[rt];
      }
    }
  }
  if constexpr (KG > 1)
    __syncthreads();
  if (kw == 0) {
#pragma unroll
    for (int rt = 0; rt < RT; rt++) {
#pragma unroll
      for (int q = 1; q < KG; q++) {
        hi[rt] += hs[q - 1][rt][nw * 64 + lane];
        lo[rt] += ls[q - 1][rt][nw * 64 + lane];
        if constexpr (!TF)
          mid[rt] += ms[q - 1][rt][nw * 64 + lane];
        if (nw == 0 && g == 0)
          sq[rt] += qs[q - 1][rt][r];
      }
      int64_t row = int64_t(blockIdx.x) * (16 * RT) + rt * 16 + r;
      if (row < m) {
        F4 result = TF ? hi[rt] + lo[rt] : (hi[rt] + mid[rt]) + lo[rt];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int col = nw * 16 + g + 4 * j;
          if (col < N)
            part[(int64_t(blockIdx.y) * m + row) * N + col] = result[j];
        }
        if (nw == 0 && g == 0)
          sqout[int64_t(blockIdx.y) * m + row] = sq[rt];
      }
    }
  }
}

// One CTA per row. Eight groups reduce disjoint slices before Sinkhorn.
template <int S>
__global__ void hc_reduce_sinkhorn(const float *part, const float *sq,
                                   const float *scale, const float *base,
                                   float *pre, float *post, float *comb,
                                   int64_t m, int iters, float rms_eps,
                                   float hc_eps) {
  int lane = threadIdx.x % 32, group = threadIdx.x / 32,
      wave = threadIdx.x / 64;
  int64_t row = blockIdx.x;
  float a = 0;
#pragma unroll
  for (int s = group; s < S; s += 8) {
    if (lane < 24)
      a += part[(int64_t(s) * m + row) * N + lane];
    else if (lane == 24)
      a += sq[int64_t(s) * m + row];
  }
  a += __shfl_xor(a, 32, 64);
  __shared__ float sums[4][32];
  if (group % 2 == 0)
    sums[wave][lane] = a;
  __syncthreads();
  if (threadIdx.x >= 32)
    return;
  a = (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane]);
  float q = __shfl(a, 24, 32), inv = rsqrtf(q * (1.0f / K) + rms_eps);
  float v = lane < 24 ? a * inv * scale[lane < 4 ? 0 : (lane < 8 ? 1 : 2)] +
                            base[lane]
                      : 0;
  if (lane < 4)
    pre[row * 4 + lane] = __fdividef(1.0f, (1.0f + expf(-v))) + hc_eps;
  if (lane >= 4 && lane < 8)
    post[row * 4 + lane - 4] = __fdividef(2.0f, (1.0f + expf(-v)));
  float c = __shfl(v, lane < 16 ? lane + 8 : 8, 32);
  float mx = fmaxf(c, __shfl_xor(c, 1, 32));
  mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
  c = expf(c - mx);
  float sum = c + __shfl_xor(c, 1, 32);
  sum += __shfl_xor(sum, 2, 32);
  c = __fdividef(c, sum) + hc_eps;
  sum = c + __shfl_xor(c, 4, 32);
  sum += __shfl_xor(sum, 8, 32);
  c = __fdividef(c, sum + hc_eps);
  for (int i = 1; i < iters; i++) {
    sum = c + __shfl_xor(c, 1, 32);
    sum += __shfl_xor(sum, 2, 32);
    c = __fdividef(c, sum + hc_eps);
    sum = c + __shfl_xor(c, 4, 32);
    sum += __shfl_xor(sum, 8, 32);
    c = __fdividef(c, sum + hc_eps);
  }
  if (lane < 16)
    comb[row * 16 + lane] = c;
}
template <bool TF, int S, int KG, int RT>
void launch(const Tensor &x, const std::vector<Tensor> &parts,
            const Tensor &scale, const Tensor &base, int64_t iters,
            double rms, double eps, const Tensor &pre,
            const Tensor &post, const Tensor &comb) {
  const int64_t m = x.size(0);
  auto opts = x.options().dtype(kFloat);
  auto p = at::empty({S, m, N}, opts), sq = at::empty({S, m}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  const void *w0 = parts[0].data_ptr(), *w1 = parts[1].data_ptr();
  const void *w2 = TF ? nullptr : parts[2].data_ptr();
  hc_parts_project<TF, KG, RT, S>
      <<<dim3((m + 16 * RT - 1) / (16 * RT), S), 128 * KG, 0, stream>>>(
          x.data_ptr<BFloat16>(), w0, w1, w2, p.data_ptr<float>(),
          sq.data_ptr<float>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  hc_reduce_sinkhorn<S><<<m, 256, 0, stream>>>(
      p.data_ptr<float>(), sq.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), pre.data_ptr<float>(), post.data_ptr<float>(),
      comb.data_ptr<float>(), m, iters, rms, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dense(const Tensor &t, const Tensor &x, ScalarType dt) {
  TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dt &&
                  t.layout() == kStrided && !t.is_nested() &&
                  t.is_contiguous() && !t.is_neg(),
              "hc mix: expected same-device contiguous tensor with matching "
              "dtype and resolved view bits");
}
std::tuple<Tensor, Tensor, Tensor> run(const Tensor &x,
                                       const std::vector<Tensor> &parts,
                                       const Tensor &scale, const Tensor &base,
                                       int64_t iters, double rms, double eps,
                                       int64_t hc, bool tf) {
  TORCH_CHECK(x.is_cuda(), "hc mix requires a gfx928, gfx936 or gfx938 GPU");
  dense(x, x, kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K &&
                  x.size(0) <= std::numeric_limits<int>::max() && hc == 4,
              "hc mix requires x[M,20480], hc_mult=4");
  int64_t m = x.size(0);

  TORCH_CHECK(parts.size() == (tf ? 2 : 3),
              "hc mix: incorrect number of weight parts");
  for (const auto &p : parts) {
    dense(p, x, tf ? kFloat : kBFloat16);
    TORCH_CHECK(p.dim() == 2 && p.size(0) == N && p.size(1) == K,
                "hc mix: weight parts must have shape [24,20480]");
  }
  dense(scale, x, kFloat);
  dense(base, x, kFloat);
  TORCH_CHECK(scale.dim() == 1 && scale.numel() == 3 && base.dim() == 1 &&
                  base.numel() == N,
              "hc mix: scale[3], base[24] required");
  TORCH_CHECK(iters >= 1 && iters <= std::numeric_limits<int>::max(),
              "hc mix: positive sinkhorn_iters required");
  TORCH_CHECK(float(rms) > 0 && std::isfinite(float(rms)) && float(eps) > 0 &&
                  std::isfinite(float(eps)),
              "hc mix: positive finite FP32 epsilons required");
  const c10::cuda::CUDAGuard guard(x.device());
  auto arch = std::string_view(
      at::cuda::getDeviceProperties(x.get_device())->gcnArchName);
  arch = arch.substr(0, arch.find(':'));
  TORCH_CHECK(arch == "gfx928" || arch == "gfx936" || arch == "gfx938",
              "hc mix: native compensated kernels require gfx928, gfx936 or gfx938");
  TORCH_CHECK(tf || (arch == "gfx938" ? m <= 131072
                                      : m == 0 || (m >= 4096 && m <= 65536)),
              "BF16x3 requires M=0..131072 on gfx938 or M=0/4096..65536 on gfx928/gfx936");
  auto opts = x.options().dtype(kFloat);
  auto pre = at::empty({m, 4}, opts), post = at::empty_like(pre),
       comb = at::empty({m, 4, 4}, opts);
  if (!m)
    return {pre, post, comb};
  // Offline-tuned gfx938 intervals; gfx928/gfx936 use the baseline configuration.
  if (tf) {
    launch<true, 80, 4, 1>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else if (arch != "gfx938") {
    launch<false, 80, 2, 2>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else if (m <= 32) {
    launch<false, 80, 4, 1>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else if (m <= 64) {
    launch<false, 40, 4, 1>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else if (m <= 128) {
    launch<false, 80, 2, 2>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else if (m <= 256) {
    launch<false, 40, 2, 2>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  } else {
    launch<false, 80, 2, 2>(x, parts, scale, base, iters, rms, eps, pre, post, comb);
  }
  return {pre, post, comb};
}
} // namespace hc_sink_detail
std::tuple<Tensor, Tensor, Tensor> hc_mix_stats_sinkhorn_deepgemm(
    const Tensor &x, const std::vector<Tensor> &parts, const Tensor &scale,
    const Tensor &base, int64_t iters, double rms, double eps, int64_t hc) {
  return hc_sink_detail::run(x, parts, scale, base, iters, rms, eps, hc, true);
}
std::tuple<Tensor, Tensor, Tensor> hc_mix_stats_sinkhorn_bf16x3(
    const Tensor &x, const std::vector<Tensor> &parts, const Tensor &scale,
    const Tensor &base, int64_t iters, double rms, double eps, int64_t hc) {
  return hc_sink_detail::run(x, parts, scale, base, iters, rms, eps, hc, false);
}
} // namespace native
} // namespace at
