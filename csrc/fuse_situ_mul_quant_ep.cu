#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "reduce.h"

namespace at {
namespace native {

namespace situ_quant_ep_detail {

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  float dst = std::nearbyint(x);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

#ifndef USE_ROCM
  #define SITU_EP_LDG(arg) __ldg(arg)
#else
  #define SITU_EP_LDG(arg) *(arg)
#endif

__device__ __forceinline__ float ep_fast_sigmoid(float x) {
  constexpr float LOG2E = 1.44269504088896340736f;
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x * LOG2E));
}

// Host precomputes `two_inv_beta`, `two_inv_linear_beta`, and
// `beta_prod = situ_beta * situ_linear_beta`, so the device path uses only
// mults + exp2 (no divisions). Matches the sglang-das Triton reference.
template <typename scalar_t>
__device__ __forceinline__ float ep_situ_activate(scalar_t gate_v,
                                                  scalar_t up_v,
                                                  float two_inv_beta,
                                                  float two_inv_linear_beta,
                                                  float beta_prod) {
  const float g = static_cast<float>(gate_v);
  const float u = static_cast<float>(up_v);
  const float gate_tanh = 2.0f * ep_fast_sigmoid(g * two_inv_beta) - 1.0f;
  const float up_tanh   = 2.0f * ep_fast_sigmoid(u * two_inv_linear_beta) - 1.0f;
  const float sig_g     = ep_fast_sigmoid(g);
  return beta_prod * gate_tanh * sig_g * up_tanh;
}

// Vectorized EP kernel with grid-stride token loop.
//
// Grid: dim3(grid_x, experts). grid_x is chosen by the host to be
// `min(tokens_per_expert, expect_m_clamped)`; the kernel then grid-strides
// over `[0, masked_m[expert])`, so we launch exactly enough blocks to
// cover the expected valid rows and never touch padding rows.
//
// Padding rows are LEFT UNTOUCHED. The downstream masked GEMM
// (`m_grouped_w4a8_gemm_nt_masked_hipc`) reads only `[0, masked_m[e])`
// per expert, so the padding output is a genuine don't-care. `scales` is
// prefilled with 1.0 on the host as a safety net for consumers that scan
// the whole tensor.
#ifdef USE_ROCM
namespace situ_wave_detail {
using at::native::situ_quant_ep_detail::ep_situ_activate;
using at::native::situ_quant_ep_detail::float_to_int8_rn;

template <int VEC, int WAVES, bool PACKED>
__device__ __forceinline__ void
wave_row_body(const at::BFloat16 *__restrict__ input, int8_t *__restrict__ out,
              float *__restrict__ scales, const int32_t *__restrict__ counts,
              int capacity, float two_inv_beta, float two_inv_linear_beta,
              float beta_prod) {
  constexpr int WIDTH = 64;
  constexpr int HIDDEN = 3072;
  constexpr int CHUNKS = HIDDEN / (WIDTH * VEC);
  const int lane = threadIdx.x % WIDTH;
  const int wave = threadIdx.x / WIDTH;
  int expert = blockIdx.y;
  int inclusive = 0, exclusive = 0, total = 0;
  if constexpr (PACKED) {
    const int subgroup_lane = lane & 31;
    const int count = subgroup_lane < 28 ? counts[subgroup_lane] : 0;
    inclusive = count;
#pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      const int other = __shfl_up(inclusive, delta, 32);
      if (subgroup_lane >= delta)
        inclusive += other;
    }
    exclusive = inclusive - count;
    total = __shfl(inclusive, 27, 32);
  } else {
    total = counts[expert];
  }
  using InputVec = at::native::memory::aligned_vector<at::BFloat16, VEC>;
  using OutputVec = at::native::memory::aligned_vector<int8_t, VEC>;
  for (int task = (blockIdx.x + blockIdx.y * gridDim.x) * WAVES + wave;
       task < total; task += gridDim.x * gridDim.y * WAVES) {
    int token = task;
    if constexpr (PACKED) {
      const unsigned long long done =
          __ballot((lane & 31) < 28 && task >= inclusive);
      expert = __popcll(done & 0xffffffffull);
      token = task - __shfl(exclusive, expert, 32);
    }
    const int64_t row = static_cast<int64_t>(expert) * capacity + token;
    float values[CHUNKS][VEC];
    float maximum = 0.f;
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
      const int col = chunk * WIDTH * VEC + lane * VEC;
      InputVec gates =
          *reinterpret_cast<const InputVec *>(input + row * 2 * HIDDEN + col);
      InputVec ups = *reinterpret_cast<const InputVec *>(
          input + row * 2 * HIDDEN + HIDDEN + col);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float value =
            ep_situ_activate(gates.val[j], ups.val[j], two_inv_beta,
                             two_inv_linear_beta, beta_prod);
        values[chunk][j] = value;
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
#pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor(maximum, offset, WIDTH));
    if (lane == 0)
      scales[row] = maximum > 0.f ? maximum / 127.f : 1.f;
    const float inverse = maximum == 0.f ? 0.f : 127.f / maximum;
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
      OutputVec quantized;
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        quantized.val[j] = float_to_int8_rn(values[chunk][j] * inverse);
      const int col = chunk * WIDTH * VEC + lane * VEC;
      *reinterpret_cast<OutputVec *>(out + row * HIDDEN + col) = quantized;
    }
  }
}

} // namespace situ_wave_detail

#endif
template <typename scalar_t, int VEC, int block_size, bool packed, bool tuned>
__device__ __forceinline__ void situ_mul_quant_ep_stride_body(
    int8_t *__restrict__ out, const scalar_t *__restrict__ input,
    float *__restrict__ scales, const int32_t *__restrict__ masked_m,
    const int hidden, const int32_t tokens_per_expert, const float two_inv_beta,
    const float two_inv_linear_beta, const float beta_prod) {
  int32_t expert = blockIdx.y;

  using VecType     = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;

  const int tidx = threadIdx.x;
  const int idx  = tidx * VEC;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float   val_shared[share_size];
  __shared__ float   s_token_scale;
  __shared__ int32_t s_valid_m;

  // Packed scheduling uses actual device counts, independent of expect_m.
  // Each 32-lane subgroup keeps the 28-expert prefix in registers. Ballot
  // finds the owner of a packed row without shared memory or a binary search.
  int32_t total_rows = 0;
  int inclusive = 0, exclusive = 0;
  if constexpr (packed) {
    const int lane = tidx & 31;
    const int count = lane < 28 ? masked_m[lane] : 0;
    inclusive = count;
#pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      int other = __shfl_up(inclusive, delta, 32);
      if (lane >= delta)
        inclusive += other;
    }
    exclusive = inclusive - count;
    total_rows = __shfl(inclusive, 27, 32);
  } else if constexpr (tuned) {
    total_rows = masked_m[expert];
  } else {
    if (tidx == 0)
      s_valid_m = masked_m[expert];
    __syncthreads();
    total_rows = s_valid_m;
  }

  // Early skew: one wave per row removes CTA barriers and exposes more rows.
  // The ordinary small-row path keeps the accepted 128-CTA schedule.
  if constexpr (tuned && packed) {
#ifdef USE_ROCM
    if (total_rows >= 512) {
      situ_wave_detail::wave_row_body<16, 4, true>(
          input, out, scales, masked_m, tokens_per_expert, two_inv_beta,
          two_inv_linear_beta, beta_prod);
      return;
    }
    if (blockIdx.x >= 128)
      return;
#endif
  }
  const bool in_range = (idx < hidden);
  const int32_t stride =
      (tuned && packed) ? 128 : static_cast<int32_t>(gridDim.x);
  for (int32_t task = static_cast<int32_t>(blockIdx.x); task < total_rows;
       task += stride) {
    int32_t token = task;
    if constexpr (packed) {
      const unsigned long long completed = static_cast<unsigned long long>(
          __ballot((tidx & 31) < 28 && task >= inclusive));
      expert = __popcll(completed & 0xffffffffull);
      token = task - __shfl(exclusive, expert, 32);
    }
    const int64_t expert_row_base =
        static_cast<int64_t>(expert) * tokens_per_expert;
    const int64_t token_row = expert_row_base + token;
    const int64_t x_index   = token_row * 2 * hidden + idx;
    const int64_t y_index   = token_row * hidden + idx;

    const VecType* x1 =
        reinterpret_cast<const VecType*>(input + x_index);
    const VecType* x2 =
        reinterpret_cast<const VecType*>(input + x_index + hidden);
    VecInt8Type* y = reinterpret_cast<VecInt8Type*>(out + y_index);

    scalar_t r_g[VEC];
    scalar_t r_u[VEC];
    float    r_y[VEC];

    if (in_range) {
      *reinterpret_cast<VecType*>(r_g) = *x1;
      *reinterpret_cast<VecType*>(r_u) = *x2;
      #pragma unroll
      for (int i = 0; i < VEC; ++i) {
        r_y[i] = ep_situ_activate(r_g[i], r_u[i],
                                  two_inv_beta, two_inv_linear_beta, beta_prod);
      }
    }

    float row_max = 0.0f;
    if (in_range) {
      #pragma unroll
      for (int i = 0; i < VEC; ++i) row_max = fmaxf(row_max, fabsf(r_y[i]));
    }
    if constexpr (packed || tuned) {
      row_max = WarpReduceMax_ROW<float>(row_max);
      if (tidx % C10_WARP_SIZE == 0)
        val_shared[tidx / C10_WARP_SIZE] = row_max;
      __syncthreads();
      if (tidx == 0) {
        row_max = 0.0f;
#pragma unroll
        for (int wave = 0; wave < share_size; ++wave)
          row_max = fmaxf(row_max, val_shared[wave]);
      }
    } else {
      row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
    }

    if (tidx == 0) {
      s_token_scale = row_max;
      scales[token_row] = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    }
    __syncthreads();

    const float inv_s =
        (s_token_scale == 0.0f) ? 0.0f : 127.0f / s_token_scale;

    if (in_range) {
      int8_t out_vec[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; ++i) {
        out_vec[i] = float_to_int8_rn(r_y[i] * inv_s);
      }
      *y = *reinterpret_cast<const VecInt8Type*>(out_vec);
    }
  }
}

template <typename scalar_t, int VEC, int block_size, bool packed = false>
__global__ void situ_mul_quant_ep_kernel_stride(
    int8_t *out, const scalar_t *input, float *scales, const int32_t *masked_m,
    int hidden, int32_t tokens_per_expert, float two_inv_beta,
    float two_inv_linear_beta, float beta_prod) {
  situ_mul_quant_ep_stride_body<scalar_t, VEC, block_size, packed, false>(
      out, input, scales, masked_m, hidden, tokens_per_expert, two_inv_beta,
      two_inv_linear_beta, beta_prod);
}

template <bool EARLY>
__global__ void situ_mul_quant_ep_kernel_tuned(
    int8_t *out, const at::BFloat16 *input, float *scales,
    const int32_t *masked_m, int hidden, int32_t tokens_per_expert,
    float two_inv_beta, float two_inv_linear_beta, float beta_prod) {
  situ_mul_quant_ep_stride_body<at::BFloat16, 16, 256, EARLY, true>(
      out, input, scales, masked_m, hidden, tokens_per_expert, two_inv_beta,
      two_inv_linear_beta, beta_prod);
}

// Non-vectorized fallback for very large hidden.
template <typename scalar_t, int block_size>
__global__ void situ_mul_quant_ep_kernel_stride_generic(
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const int32_t* __restrict__ masked_m,
    const int hidden,
    const int32_t tokens_per_expert,
    const float two_inv_beta,
    const float two_inv_linear_beta,
    const float beta_prod) {
  const int32_t expert = blockIdx.y;

  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ float   shared_mem[share_size];
  __shared__ float   s_token_scale;
  __shared__ int32_t s_valid_m;

  if (threadIdx.x == 0) s_valid_m = masked_m[expert];
  __syncthreads();

  if (s_valid_m == 0) return;

  const int32_t stride = static_cast<int32_t>(gridDim.x);
  const int64_t expert_row_base =
      static_cast<int64_t>(expert) * tokens_per_expert;

  for (int32_t token = static_cast<int32_t>(blockIdx.x);
       token < s_valid_m;
       token += stride) {
    const int64_t token_row     = expert_row_base + token;
    const int64_t input_offset  = token_row * 2 * hidden;
    const int64_t output_offset = token_row * hidden;

    float row_max = 0.0f;
    for (int idx = threadIdx.x; idx < hidden; idx += blockDim.x) {
      const scalar_t g = SITU_EP_LDG(&input[input_offset + idx]);
      const scalar_t u = SITU_EP_LDG(&input[input_offset + hidden + idx]);
      const float val =
          ep_situ_activate(g, u, two_inv_beta, two_inv_linear_beta, beta_prod);
      row_max = fmaxf(row_max, fabsf(val));
    }
    row_max = BlockReduceMax_ROW<float, block_size>(row_max, shared_mem);

    if (threadIdx.x == 0) {
      s_token_scale = row_max;
      scales[token_row] = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    }
    __syncthreads();

    const float inv_s =
        (s_token_scale == 0.0f) ? 0.0f : 127.0f / s_token_scale;

    for (int idx = threadIdx.x; idx < hidden; idx += blockDim.x) {
      const scalar_t g = SITU_EP_LDG(&input[input_offset + idx]);
      const scalar_t u = SITU_EP_LDG(&input[input_offset + hidden + idx]);
      const float val =
          ep_situ_activate(g, u, two_inv_beta, two_inv_linear_beta, beta_prod);
      out[output_offset + idx] = float_to_int8_rn(val * inv_s);
    }
  }
}

}  // namespace situ_quant_ep_detail

#ifndef SITU_EP_DISPATCH_FLOATING_TYPES
#define SITU_EP_DISPATCH_CASE_FLOATING_TYPES(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define SITU_EP_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                        \
                     SITU_EP_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#endif

// expect_m: caller hint for the expected max valid tokens per expert. When
// >0 and < tokens_per_expert, the launched grid.x is clamped to that value
// (via a small over-provision), and the kernel grid-strides over the actual
// per-expert valid range. When <=0, grid.x = tokens_per_expert for full
// coverage (backward-compat with callers that do not track expected_m).
void fuse_situ_mul_quant_ep(torch::Tensor& input,
                            torch::Tensor& out,
                            torch::Tensor& scales,
                            torch::Tensor& masked_m,
                            double situ_beta,
                            double situ_linear_beta,
                            int64_t expect_m) {
  TORCH_CHECK(input.dim() == 3,
              "input must be 3-D [experts, tokens, 2*hidden]");
  TORCH_CHECK(input.size(-1) % 2 == 0, "input last dim must be even");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
  TORCH_CHECK(masked_m.scalar_type() == at::ScalarType::Int,
              "masked_m must be int32");
  TORCH_CHECK(situ_beta > 0.0, "situ_beta must be positive");
  TORCH_CHECK(situ_linear_beta > 0.0, "situ_linear_beta must be positive");

  const int32_t experts           = static_cast<int32_t>(input.size(0));
  const int32_t tokens_per_expert = static_cast<int32_t>(input.size(1));
  const int hidden = static_cast<int>(input.size(-1) / 2);
  const int64_t rows =
      static_cast<int64_t>(experts) * tokens_per_expert;

  TORCH_CHECK(out.dim() == 3 && out.size(0) == experts &&
                  out.size(1) == tokens_per_expert && out.size(2) == hidden,
              "out shape must be [experts, tokens, hidden]");
  TORCH_CHECK(scales.numel() == rows,
              "scales must have experts * tokens elements");
  TORCH_CHECK(masked_m.numel() == experts,
              "masked_m must have `experts` elements");

  if (rows == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const float two_inv_beta        = 2.0f / static_cast<float>(situ_beta);
  const float two_inv_linear_beta = 2.0f / static_cast<float>(situ_linear_beta);
  const float beta_prod           =
      static_cast<float>(situ_beta) * static_cast<float>(situ_linear_beta);

#ifdef USE_ROCM
  // K3 EP scheduling uses actual device counts, independent of the hint.
  // Hints 1..4 pack rows: small batches keep 128 CTAs, while >=512 valid
  // rows use one wave per row. Hints 5..19 keep the per-expert grid with
  // a tuned reduction. Both switches default on and are cached below.
  static const bool packed_enabled = [] {
    const char *value = std::getenv("LIGHTOP_SITU_EP_PACKED");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();

  // Same-wheel comparison against the accepted 2026-10-10 packed kernel.
  // Cached on first invocation, just like LIGHTOP_SITU_EP_PACKED.
  static const bool tuned_enabled = [] {
    const char *value = std::getenv("LIGHTOP_SITU_EP_TUNED");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  if (tuned_enabled && packed_enabled && experts == 28 && hidden == 3072 &&
      input.scalar_type() == at::ScalarType::BFloat16 && expect_m > 0 &&
      expect_m <= 19) {
    if (expect_m <= 4) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_tuned<true>
          <<<256, 256, 0, stream>>>(
              out.data_ptr<int8_t>(), input.data_ptr<at::BFloat16>(),
              scales.data_ptr<float>(), masked_m.data_ptr<int32_t>(), hidden,
              tokens_per_expert, two_inv_beta, two_inv_linear_beta, beta_prod);
    } else {
      const int32_t grid_x =
          static_cast<int32_t>(std::min<int64_t>(expect_m, tokens_per_expert));
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_tuned<false>
          <<<dim3(grid_x, experts), 256, 0, stream>>>(
              out.data_ptr<int8_t>(), input.data_ptr<at::BFloat16>(),
              scales.data_ptr<float>(), masked_m.data_ptr<int32_t>(), hidden,
              tokens_per_expert, two_inv_beta, two_inv_linear_beta, beta_prod);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (packed_enabled && experts == 28 && hidden == 3072 &&
      input.scalar_type() == at::ScalarType::BFloat16 && expect_m > 0 &&
      expect_m <= 4) {
    situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<at::BFloat16, 16, 256,
                                                          true>
        <<<128, 256, 0, stream>>>(
            out.data_ptr<int8_t>(), input.data_ptr<at::BFloat16>(),
            scales.data_ptr<float>(), masked_m.data_ptr<int32_t>(), hidden,
            tokens_per_expert, two_inv_beta, two_inv_linear_beta, beta_prod);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  // grid.x heuristic: if the caller supplied a positive `expect_m` smaller
  // than tokens_per_expert, launch just enough blocks per expert to cover
  // the expected valid rows. Otherwise fall back to full coverage.
  int32_t grid_x;
  if (expect_m > 0 &&
      static_cast<int32_t>(expect_m) < tokens_per_expert) {
    grid_x = static_cast<int32_t>(expect_m);
    if (grid_x < 1) grid_x = 1;
  } else {
    grid_x = tokens_per_expert;
  }

  dim3 grid(static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(experts));

  SITU_EP_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_mul_quant_ep", [&] {
    if (hidden <= 512) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 2, 256>
          <<<grid, 256, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      masked_m.data_ptr<int32_t>(),
                                      hidden, tokens_per_expert,
                                      two_inv_beta, two_inv_linear_beta,
                                      beta_prod);
    } else if (hidden <= 1024) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 8, 128>
          <<<grid, 128, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      masked_m.data_ptr<int32_t>(),
                                      hidden, tokens_per_expert,
                                      two_inv_beta, two_inv_linear_beta,
                                      beta_prod);
    } else if (hidden <= 2048) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 8, 256>
          <<<grid, 256, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      masked_m.data_ptr<int32_t>(),
                                      hidden, tokens_per_expert,
                                      two_inv_beta, two_inv_linear_beta,
                                      beta_prod);
    } else if (hidden <= 4096) {
      // Split point 3072 lives in this bucket. block=256, VEC=16 covers
      // 4096 elements per block; 192 of 256 threads are active for
      // hidden=3072. Compared to the previous block=512 VEC=8 choice, the
      // smaller block halves per-block register footprint, letting more
      // blocks stay resident per CU when the total launched grid is small
      // (i.e. during decode with expect_m ~= 32).
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 16, 256>
          <<<grid, 256, 0, stream>>>(out.data_ptr<int8_t>(),
                                      input.data_ptr<scalar_t>(),
                                      scales.data_ptr<float>(),
                                      masked_m.data_ptr<int32_t>(),
                                      hidden, tokens_per_expert,
                                      two_inv_beta, two_inv_linear_beta,
                                      beta_prod);
    } else if (hidden <= 8192) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 8, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       masked_m.data_ptr<int32_t>(),
                                       hidden, tokens_per_expert,
                                       two_inv_beta, two_inv_linear_beta,
                                       beta_prod);
    } else if (hidden <= 16384) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 16, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       masked_m.data_ptr<int32_t>(),
                                       hidden, tokens_per_expert,
                                       two_inv_beta, two_inv_linear_beta,
                                       beta_prod);
    } else if (hidden <= 32768) {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride<scalar_t, 32, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       masked_m.data_ptr<int32_t>(),
                                       hidden, tokens_per_expert,
                                       two_inv_beta, two_inv_linear_beta,
                                       beta_prod);
    } else {
      situ_quant_ep_detail::situ_mul_quant_ep_kernel_stride_generic<scalar_t, 1024>
          <<<grid, 1024, 0, stream>>>(out.data_ptr<int8_t>(),
                                       input.data_ptr<scalar_t>(),
                                       scales.data_ptr<float>(),
                                       masked_m.data_ptr<int32_t>(),
                                       hidden, tokens_per_expert,
                                       two_inv_beta, two_inv_linear_beta,
                                       beta_prod);
    }
  });
}

}  // namespace native
}  // namespace at
