// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <torch/all.h>

#include "../../include/quant_fp8.cuh"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

#if defined(TORCH_VERSION_OVER_2120)
// torch>=2.12 removed C10_WARP_SIZE_STATIC, wavefront size is fixed at 64
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {
namespace {

__device__ __forceinline__ int8_t per_token_quant_int8_rn(float value) {
#ifdef USE_ROCM
    constexpr float int8_min =
        static_cast<float>(std::numeric_limits<int8_t>::min());
    constexpr float int8_max =
        static_cast<float>(std::numeric_limits<int8_t>::max());
    float rounded = std::nearbyint(value);
    rounded = fminf(fmaxf(rounded, int8_min), int8_max);
    return static_cast<int8_t>(rounded);
#else
    uint32_t result;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
    return reinterpret_cast<const int8_t&>(result);
#endif
}

template <int ROWS_PER_BLOCK, bool WITH_GATE = false>
__global__ void per_token_quant_int8_bf16_k128_kernel(
    const c10::BFloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int64_t num_tokens,
    const float* __restrict__ smooth,
    const float* __restrict__ head_weights,
    float* __restrict__ adjusted_weights) {
    constexpr int hidden_size = 128;
    constexpr int subgroup_size = 16;
    constexpr int rows_per_wave = C10_WARP_SIZE / subgroup_size;
    constexpr int values_per_lane = hidden_size / subgroup_size;
    static_assert(C10_WARP_SIZE % subgroup_size == 0,
                  "warp size must be divisible by the 16-lane subgroup");
    static_assert(values_per_lane == 8, "the fast path uses 16-lane rows");
    static_assert(ROWS_PER_BLOCK * subgroup_size == 512,
                  "the fast path uses 512 threads");

    using InputVec =
        at::native::memory::aligned_vector<c10::BFloat16, values_per_lane>;
    using OutputVec =
        at::native::memory::aligned_vector<int8_t, values_per_lane>;

    using ScaleVec =
        at::native::memory::aligned_vector<float, values_per_lane>;

    const int thread_id = threadIdx.x;
    const int wave_lane = thread_id % C10_WARP_SIZE;
    const int wave_id = thread_id / C10_WARP_SIZE;
    const int subgroup_id = wave_lane / subgroup_size;
    const int lane_id = wave_lane % subgroup_size;
    const int64_t token_id =
        static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
        wave_id * rows_per_wave + subgroup_id;
    if (token_id >= num_tokens) {
        return;
    }

    const int column = lane_id * values_per_lane;
    const int64_t offset = token_id * hidden_size + column;
    const InputVec input_vec =
        *reinterpret_cast<const InputVec*>(input + offset);
    ScaleVec smooth_vec;
    if constexpr (WITH_GATE) {
        smooth_vec = *reinterpret_cast<const ScaleVec*>(smooth + column);
    }
    float values[values_per_lane];
    float row_absmax = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_lane; ++index) {
        values[index] = static_cast<float>(input_vec.val[index]);
        if constexpr (WITH_GATE) {
            values[index] *= smooth_vec.val[index];
        }
        row_absmax = fmaxf(row_absmax, fabsf(values[index]));
    }

#pragma unroll
    for (int delta = subgroup_size / 2; delta > 0; delta >>= 1) {
        row_absmax = fmaxf(
            row_absmax, __shfl_down(row_absmax, delta, subgroup_size));
    }
    float token_absmax = __shfl(row_absmax, 0, subgroup_size);
    if constexpr (!WITH_GATE) {
        token_absmax = fmaxf(token_absmax, 1.0e-10f);
    }
    if (lane_id == 0) {
        const float token_scale = token_absmax / 127.0f;
        scale_out[token_id] = token_scale;
        if constexpr (WITH_GATE) {
            adjusted_weights[token_id] = head_weights[token_id] * token_scale;
        }
    }
    const float inverse_scale =
        WITH_GATE && token_absmax == 0.0f ? 0.0f : 127.0f / token_absmax;
    OutputVec output_vec;
#pragma unroll
    for (int index = 0; index < values_per_lane; ++index) {
        output_vec.val[index] =
            per_token_quant_int8_rn(values[index] * inverse_scale);
    }
    *reinterpret_cast<OutputVec*>(output + offset) = output_vec;
}

}  // namespace

void per_token_quant_int8_out(
    torch::Tensor& out,
    const torch::Tensor& input,
    torch::Tensor& scales,
    const c10::optional<torch::Tensor>& smooth,
    const c10::optional<torch::Tensor>& weights,
    const c10::optional<torch::Tensor>& gate_output) {
    TORCH_CHECK(input.is_cuda(), "input must be on a GPU");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) == 128,
                "input must have last dimension 128");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
                "input must have dtype torch.bfloat16");
    TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Char,
                "out must have dtype torch.int8");
    const int64_t num_tokens = input.numel() / 128;
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
                "scales must have dtype torch.float32");
    TORCH_CHECK(scales.numel() == num_tokens,
                "scales must contain one value per token");
    TORCH_CHECK(input.device() == out.device() &&
                    input.device() == scales.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0,
                "input must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "out must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(scales.data_ptr()) % 4 == 0,
                "scales must be 4-byte aligned");
    const bool with_gate = smooth.has_value() || weights.has_value() || gate_output.has_value();
    TORCH_CHECK(!with_gate || (smooth.has_value() && weights.has_value() && gate_output.has_value()),
                "smooth, weights and gate_output must be supplied together");
    if (with_gate) {
        TORCH_CHECK(smooth->scalar_type() == at::ScalarType::Float &&
                        smooth->numel() == 128 && smooth->is_contiguous());
        TORCH_CHECK(weights->scalar_type() == at::ScalarType::Float &&
                        weights->numel() == num_tokens && weights->is_contiguous());
        TORCH_CHECK(gate_output->scalar_type() == at::ScalarType::Float &&
                        gate_output->sizes() == weights->sizes() && gate_output->is_contiguous());
        TORCH_CHECK(smooth->device() == input.device() && weights->device() == input.device() &&
                        gate_output->device() == input.device());
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(smooth->data_ptr()) % 32 == 0);
        TORCH_CHECK(!out.is_alias_of(input) && !gate_output->is_alias_of(*weights) &&
                        !scales.is_alias_of(*gate_output));
    }
    if (num_tokens == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 512;
    constexpr int rows_per_block = threads / 16;
    const int64_t blocks =
        (num_tokens + rows_per_block - 1) / rows_per_block;
    if (with_gate) {
        per_token_quant_int8_bf16_k128_kernel<rows_per_block, true>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<c10::BFloat16>(), out.data_ptr<int8_t>(),
                scales.data_ptr<float>(), num_tokens, smooth->data_ptr<float>(),
                weights->data_ptr<float>(), gate_output->data_ptr<float>());
    } else {
        per_token_quant_int8_bf16_k128_kernel<rows_per_block, false>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<c10::BFloat16>(), out.data_ptr<int8_t>(),
                scales.data_ptr<float>(), num_tokens, nullptr, nullptr, nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

// Share the reduction and INT8 conversion with the existing quantization code.
template <int LANES>
__device__ __forceinline__ float quant_wave_absmax(float value) {
#if defined(USE_ROCM) && (defined(__gfx936__) || defined(__gfx938__))
  // DPP keeps reduction traffic in the register crossbar. Every stage
  // extends the valid reduction at lane LANES-1 to the next power of two.
#define QUANT_DPP(CONTROL, ROW_MASK)                                           \
  value =                                                                      \
      fmaxf(value, __uint_as_float(__builtin_amdgcn_mov_dpp(                   \
                       __float_as_uint(value), CONTROL, ROW_MASK, 0xf, true)))
  if constexpr (LANES >= 2) {
    QUANT_DPP(0xb1, 0xf);
  }
  if constexpr (LANES >= 4) {
    QUANT_DPP(0x4e, 0xf);
  }
  if constexpr (LANES >= 8) {
    QUANT_DPP(0x114, 0xf);
  }
  if constexpr (LANES >= 16) {
    QUANT_DPP(0x118, 0xf);
  }
  if constexpr (LANES >= 32) {
    QUANT_DPP(0x142, 0xa);
  }
  if constexpr (LANES >= 64) {
    QUANT_DPP(0x143, 0xc);
  }
#undef QUANT_DPP
  return __uint_as_float(
      __builtin_amdgcn_readlane(__float_as_uint(value), LANES - 1));
#else
#pragma unroll
  for (int delta = LANES / 2; delta; delta >>= 1) {
#ifdef USE_ROCM
    value = fmaxf(value, __shfl_xor(value, delta, C10_WARP_SIZE));
#else
    value =
        fmaxf(value, __shfl_xor_sync(0xffffffff, value, delta, C10_WARP_SIZE));
#endif
  }
  return value;
#endif
}

template <int THREADS>
__device__ __forceinline__ float quant_row_absmax(float value,
                                                  float floor = 1.0e-10f) {
  constexpr int wave = C10_WARP_SIZE;
  constexpr int waves = THREADS / wave;
  value = quant_wave_absmax<wave>(value);
  if constexpr (waves == 1)
    return fmaxf(value, floor);
  __shared__ float partial[waves];
  if (threadIdx.x % wave == 0)
    partial[threadIdx.x / wave] = value;
  __syncthreads();
  if (threadIdx.x < wave) {
    value = threadIdx.x < waves ? partial[threadIdx.x] : 0.0f;
    value = quant_wave_absmax<waves>(value);
    if (threadIdx.x == 0)
      partial[0] = fmaxf(value, floor);
  }
  __syncthreads();
  return partial[0];
}

template <typename Scalar>
__device__ __forceinline__ float round_activation(float value) {
  return static_cast<float>(static_cast<Scalar>(value));
}

template <>
__device__ __forceinline__ float round_activation<c10::BFloat16>(float value) {
  // Match c10's RNE and canonical NaN, with an integer select. Keeping the
  // rounding in FP32 avoids per-value divergent conversion branches.
  const uint32_t bits = __float_as_uint(value);
  const uint32_t rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
  return __uint_as_float((bits & 0x7fffffffU) > 0x7f800000U ? 0x7fc00000U
                                                            : rounded);
}

template <>
__device__ __forceinline__ float round_activation<c10::Half>(float value) {
  // Materialize the FP32 result before the half conversion. Otherwise the
  // compiler may eliminate this rounding at exact half-way boundaries.
#ifdef USE_ROCM
  asm volatile("" : "+v"(value));
#else
  asm volatile("" : "+f"(value));
#endif
  return __half2float(__float2half_rn(value));
}

// Vectorized rows share the same public operation and rounding contract.
// Aligned blocks amortize address arithmetic and pack the final stores.
template <typename Scalar, typename Position, int WIDTH, int THREADS,
          bool INVERSE, int HEAD_DIM, int ROPE_DIM, int GROUPS,
          bool FP8 = false, bool HAS_ROPE = true>
__device__ __forceinline__ void
rope_quant_vector(const Scalar *input, const float *frequencies,
                  const Position *positions, int8_t *quantized, float *scales,
                  int heads_per_group, int64_t token_stride,
                  int64_t head_stride, int64_t frequency_stride,
                  int64_t output_pitch, int output_row) {
  // Spread a head's rotary tail across every lane of its subgroup. The
  // contiguous layout made only one eighth of the lanes execute all RoPE
  // instructions while the other lanes waited at a divergent branch.
  constexpr bool striped = HAS_ROPE && HEAD_DIM == 512 && ROPE_DIM == 64 &&
                           WIDTH == 4096 && THREADS <= 256;
  constexpr int head_threads = THREADS / (WIDTH / HEAD_DIM);
  constexpr int vector = striped ? ROPE_DIM / head_threads : 8;
  constexpr int iterations = WIDTH / (vector * THREADS);
  using Input = at::native::memory::aligned_vector<Scalar, vector>;
  using Output = at::native::memory::aligned_vector<int8_t, vector>;
  const int row = blockIdx.x, token = row / GROUPS, group = row % GROUPS;
  const int64_t position = HAS_ROPE ? positions[token] : 0;
  const int64_t base =
      token * token_stride + group * heads_per_group * head_stride;
  float values[iterations][vector];
  float maximum = 0;
#pragma unroll
  for (int i = 0; i < iterations; ++i) {
    int head = striped ? threadIdx.x / head_threads
                       : vector * (i * THREADS + threadIdx.x) / HEAD_DIM;
#ifdef USE_ROCM
    if constexpr (!striped)
      head = __builtin_amdgcn_readfirstlane(head);
#endif
    const int dim = striped
                        ? i * ROPE_DIM + (threadIdx.x % head_threads) * vector
                        : vector * (i * THREADS + threadIdx.x) % HEAD_DIM;
    const Input packed = *reinterpret_cast<const Input *>(
        input + base + head * head_stride + dim);
#pragma unroll
    for (int j = 0; j < vector; ++j)
      values[i][j] = static_cast<float>(packed.val[j]);
    if constexpr (HAS_ROPE) {
      if (dim >= HEAD_DIM - ROPE_DIM) {
        using Frequency = at::native::memory::aligned_vector<float, vector>;
        const Frequency f = *reinterpret_cast<const Frequency *>(
            frequencies + position * frequency_stride + dim -
            (HEAD_DIM - ROPE_DIM));
#pragma unroll
        for (int j = 0; j < vector; j += 2) {
          const float a = values[i][j], b = values[i][j + 1];
          const float bi = __fmul_rn(b, f.val[j + 1]),
                      ai = __fmul_rn(a, f.val[j + 1]);
          values[i][j] =
              round_activation<Scalar>(fmaf(a, f.val[j], INVERSE ? bi : -bi));
          values[i][j + 1] =
              round_activation<Scalar>(fmaf(b, f.val[j], INVERSE ? -ai : ai));
        }
      }
    }
#pragma unroll
    for (int j = 0; j < vector; ++j)
      maximum = fmaxf(maximum, fabsf(values[i][j]));
  }
  const float absmax =
      quant_row_absmax<THREADS>(maximum, FP8 ? 0.0f : 1.0e-10f);
  constexpr float qmax = FP8 ? 448.0f : 127.0f;
  const float inverse = FP8 && absmax == 0 ? 0 : qmax / absmax;
  if (threadIdx.x == 0)
    scales[output_row] = absmax / qmax;
#pragma unroll
  for (int i = 0; i < iterations; ++i) {
    Output packed;
    if constexpr (FP8) {
#pragma unroll
      for (int j = 0; j < vector; j += 2) {
        const uint16_t pair = lightop::quant::float_to_fp8e4m3_x2(
            values[i][j] * inverse, values[i][j + 1] * inverse);
        packed.val[j] = static_cast<int8_t>(pair);
        packed.val[j + 1] = static_cast<int8_t>(pair >> 8);
      }
    } else {
#pragma unroll
      for (int j = 0; j < vector; ++j)
        packed.val[j] = static_cast<int8_t>(roundf(values[i][j] * inverse));
    }
    const int col = striped ? (threadIdx.x / head_threads) * HEAD_DIM +
                                  i * ROPE_DIM +
                                  (threadIdx.x % head_threads) * vector
                            : vector * (i * THREADS + threadIdx.x);
    *reinterpret_cast<Output *>(quantized + output_row * output_pitch + col) =
        packed;
  }
}

template <typename Scalar, int WIDTH, int THREADS, int GROUP_WIDTH, int SPLITS,
          bool STORE_OUTPUT>
__device__ __forceinline__ void
scale_quant_vector(const int32_t *accum, const float *input_scale,
                   const float *weight_scale, Scalar *output, int8_t *quantized,
                   float *scales) {
  constexpr int vector = 4;
  constexpr int iterations = WIDTH / (vector * THREADS);
  constexpr int groups = WIDTH / GROUP_WIDTH;
  static_assert(WIDTH % (vector * THREADS) == 0);
  using Raw = at::native::memory::aligned_vector<int32_t, vector>;
  using Weight = at::native::memory::aligned_vector<float, vector>;
  using Output = at::native::memory::aligned_vector<Scalar, vector>;
  using Quant = at::native::memory::aligned_vector<int8_t, vector>;
  static_assert(GROUP_WIDTH % (vector * C10_WARP_SIZE) == 0);
  const int token = blockIdx.x;
  const int64_t base = static_cast<int64_t>(token) * WIDTH;
  const int begin = blockIdx.y * (WIDTH / SPLITS), end = begin + WIDTH / SPLITS;
  float values[iterations][vector];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < iterations; ++i) {
    const int column = vector * (i * THREADS + threadIdx.x);
    int group = column / GROUP_WIDTH;
#ifdef USE_ROCM
    // Every wave stays inside a group, so its scale address is uniform.
    group = __builtin_amdgcn_readfirstlane(group);
#endif
    const float xs = input_scale[static_cast<int64_t>(token) * groups + group];
    const Raw raw = *reinterpret_cast<const Raw *>(accum + base + column);
    const Weight ws = *reinterpret_cast<const Weight *>(weight_scale + column);
#pragma unroll
    for (int j = 0; j < vector; ++j) {
      const float a = __fmul_rn(static_cast<float>(raw.val[j]), xs);
      const float v = round_activation<Scalar>(__fmul_rn(a, ws.val[j]));
      values[i][j] = v;
      maximum = fmaxf(maximum, fabsf(v));
    }
  }
  const float absmax = quant_row_absmax<THREADS>(maximum);
  const float inverse = 127.0f / absmax;
  if (threadIdx.x == 0 && blockIdx.y == 0)
    scales[token] = absmax / 127.0f;
#pragma unroll
  for (int i = 0; i < iterations; ++i) {
    const int column = vector * (i * THREADS + threadIdx.x);
    if (column >= begin && column < end) {
      Output floating;
      Quant quant;
#pragma unroll
      for (int j = 0; j < vector; ++j) {
        if constexpr (STORE_OUTPUT) {
          if constexpr (std::is_same_v<Scalar, c10::BFloat16>) {
            floating.val[j] =
                c10::BFloat16(static_cast<unsigned short>(
                                  __float_as_uint(values[i][j]) >> 16),
                              c10::BFloat16::from_bits());
          } else
            floating.val[j] = static_cast<Scalar>(values[i][j]);
        }
        quant.val[j] = per_token_quant_int8_rn(values[i][j] * inverse);
      }
      if constexpr (STORE_OUTPUT) {
        *reinterpret_cast<Output *>(output + base + column) = floating;
      }
      *reinterpret_cast<Quant *>(quantized + base + column) = quant;
    }
  }
}

template <typename Scalar, typename Position, int WIDTH, int THREADS,
          bool INVERSE, int HEAD_DIM = 0, int ROPE_DIM = 0, int GROUPS = 0,
          bool FP8 = false, bool HAS_ROPE = true>
__device__ __forceinline__ void rope_quant_body(
    const Scalar *__restrict__ input, const float *__restrict__ frequencies,
    const Position *__restrict__ positions, int8_t *__restrict__ quantized,
    float *__restrict__ scales, int groups, int heads_per_group, int head_dim,
    int rope_dim, int64_t token_stride, int64_t head_stride,
    int64_t frequency_stride, int64_t output_pitch, int output_row) {
  if constexpr (HEAD_DIM != 0 && WIDTH % (8 * THREADS) == 0) {
    rope_quant_vector<Scalar, Position, WIDTH, THREADS, INVERSE, HEAD_DIM,
                      ROPE_DIM, GROUPS, FP8, HAS_ROPE>(
        input, frequencies, positions, quantized, scales, heads_per_group,
        token_stride, head_stride, frequency_stride, output_pitch, output_row);
    return;
  }
  constexpr int pairs = (WIDTH + 2 * THREADS - 1) / (2 * THREADS);
  const int row = blockIdx.x;
  const int group_count = GROUPS ? GROUPS : groups;
  const int token = row / group_count;
  const int group = row % group_count;
  const int dimension = HEAD_DIM ? HEAD_DIM : head_dim;
  const int rotary = ROPE_DIM ? ROPE_DIM : rope_dim;
  const int width = HEAD_DIM ? WIDTH : heads_per_group * head_dim;
  const int64_t position = HAS_ROPE ? positions[token] : 0;
  const int64_t base =
      token * token_stride + group * heads_per_group * head_stride;
  float real[pairs], imag[pairs];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < pairs; ++i) {
    const int column = 2 * (i * THREADS + threadIdx.x);
    float a = 0.0f, b = 0.0f;
    if (column < width) {
      const int head = column / dimension;
      const int dim = column % dimension;
      const int64_t offset = base + head * head_stride + dim;
      if constexpr (HEAD_DIM != 0) {
        using Pair = at::native::memory::aligned_vector<Scalar, 2>;
        const Pair pair = *reinterpret_cast<const Pair *>(input + offset);
        a = static_cast<float>(pair.val[0]);
        b = static_cast<float>(pair.val[1]);
      } else {
        a = static_cast<float>(input[offset]);
        b = static_cast<float>(input[offset + 1]);
      }
      if (HAS_ROPE && dim >= dimension - rotary) {
        const int f = dim - (dimension - rotary);
        const float fr = frequencies[position * frequency_stride + f];
        const float fi = frequencies[position * frequency_stride + f + 1];
        const float bi = __fmul_rn(b, fi);
        const float ai = __fmul_rn(a, fi);
        const float next_a = fmaf(a, fr, INVERSE ? bi : -bi);
        const float next_b = fmaf(b, fr, INVERSE ? -ai : ai);
        // Quantization sees the materialized activation's rounding.
        a = round_activation<Scalar>(next_a);
        b = round_activation<Scalar>(next_b);
      }
    }
    real[i] = a;
    imag[i] = b;
    maximum = fmaxf(maximum, fmaxf(fabsf(a), fabsf(b)));
  }
  const float absmax =
      quant_row_absmax<THREADS>(maximum, FP8 ? 0.0f : 1.0e-10f);
  constexpr float qmax = FP8 ? 448.0f : 127.0f;
  const float inverse = FP8 && absmax == 0 ? 0 : qmax / absmax;
  if (threadIdx.x == 0)
    scales[output_row] = absmax / qmax;
#pragma unroll
  for (int i = 0; i < pairs; ++i) {
    const int column = 2 * (i * THREADS + threadIdx.x);
    if (column < width) {
      const int64_t offset = output_row * output_pitch + column;
      // This consumer uses round-half-away-from-zero, unlike the RNE
      // quantizer used by scale_per_token_quant_int8_kernel below.
      const uint16_t fp8pair = FP8 ? lightop::quant::float_to_fp8e4m3_x2(
                                         real[i] * inverse, imag[i] * inverse)
                                   : 0;
      const int8_t a = FP8 ? static_cast<int8_t>(fp8pair)
                           : static_cast<int8_t>(roundf(real[i] * inverse));
      const int8_t b = FP8 ? static_cast<int8_t>(fp8pair >> 8)
                           : static_cast<int8_t>(roundf(imag[i] * inverse));
      if constexpr (HEAD_DIM != 0) {
        using Pair = at::native::memory::aligned_vector<int8_t, 2>;
        Pair pair;
        pair.val[0] = a;
        pair.val[1] = b;
        *reinterpret_cast<Pair *>(quantized + offset) = pair;
      } else {
        quantized[offset] = a;
        quantized[offset + 1] = b;
      }
    }
  }
}

template <typename Scalar, typename Position, int WIDTH, int THREADS,
          bool INVERSE, int HEAD_DIM = 0, int ROPE_DIM = 0, int GROUPS = 0>
__global__ __launch_bounds__(THREADS) void rope_per_token_quant_int8_kernel(
    const Scalar *__restrict__ input, const float *__restrict__ frequencies,
    const Position *__restrict__ positions, int8_t *__restrict__ quantized,
    float *__restrict__ scales, int groups, int heads_per_group, int head_dim,
    int rope_dim, int64_t token_stride, int64_t head_stride,
    int64_t frequency_stride, int64_t output_pitch) {
  rope_quant_body<Scalar, Position, WIDTH, THREADS, INVERSE, HEAD_DIM, ROPE_DIM,
                  GROUPS>(input, frequencies, positions, quantized, scales,
                          groups, heads_per_group, head_dim, rope_dim,
                          token_stride, head_stride, frequency_stride,
                          output_pitch, blockIdx.x);
}

template <typename Scalar, typename Position, int WIDTH, int THREADS,
          bool INVERSE, int HEAD_DIM = 0, int ROPE_DIM = 0, int GROUPS = 0>
__global__ __launch_bounds__(THREADS) void rope_per_token_quant_fp8_kernel(
    const Scalar *__restrict__ input, const float *__restrict__ frequencies,
    const Position *__restrict__ positions, int8_t *__restrict__ quantized,
    float *__restrict__ scales, int groups, int heads_per_group, int head_dim,
    int rope_dim, int64_t token_stride, int64_t head_stride,
    int64_t frequency_stride, int64_t output_pitch, int tokens,
    bool group_major) {
  const int row = blockIdx.x;
  const int output_row =
      group_major ? (row % groups) * tokens + row / groups : row;
  rope_quant_body<Scalar, Position, WIDTH, THREADS, INVERSE, HEAD_DIM, ROPE_DIM,
                  GROUPS, true>(input, frequencies, positions, quantized,
                                scales, groups, heads_per_group, head_dim,
                                rope_dim, token_stride, head_stride,
                                frequency_stride, output_pitch, output_row);
}

template <typename Scalar, int WIDTH, int THREADS, int GROUP_WIDTH = 0,
          bool GROUP_MAJOR = false, int SPLITS = 0, bool STORE_OUTPUT = true>
__global__ __launch_bounds__(THREADS) void scale_per_token_quant_int8_kernel(
    const int32_t *__restrict__ accum, const float *__restrict__ input_scale,
    const float *__restrict__ weight_scale, Scalar *__restrict__ output,
    int8_t *__restrict__ quantized, float *__restrict__ scales, int tokens,
    int groups, int group_width, bool group_major, int splits) {
  if constexpr (GROUP_WIDTH != 0 && SPLITS != 0 && !GROUP_MAJOR &&
                WIDTH % (4 * THREADS) == 0) {
    scale_quant_vector<Scalar, WIDTH, THREADS, GROUP_WIDTH, SPLITS,
                       STORE_OUTPUT>(accum, input_scale, weight_scale, output,
                                     quantized, scales);
    return;
  }
  constexpr int values_per_thread = (WIDTH + THREADS - 1) / THREADS;
  const int token = blockIdx.x;
  const int rank = GROUP_WIDTH ? GROUP_WIDTH : group_width;
  const int width = GROUP_WIDTH ? WIDTH : groups * group_width;
  const int parts = SPLITS ? SPLITS : splits;
  const bool major = SPLITS ? GROUP_MAJOR : group_major;
  const int group_count = GROUP_WIDTH ? WIDTH / GROUP_WIDTH : groups;
  const int begin = width * blockIdx.y / parts;
  const int end = width * (blockIdx.y + 1) / parts;
  float values[values_per_thread];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    const int column = i * THREADS + threadIdx.x;
    float value = 0.0f;
    if (column < width) {
      const int group = column / rank;
      const int64_t offset =
          major ? (static_cast<int64_t>(group) * tokens + token) * rank +
                      column % rank
                : static_cast<int64_t>(token) * width + column;
      const float a = static_cast<float>(accum[offset]);
      const float scaled = __fmul_rn(
          a, input_scale[static_cast<int64_t>(token) * group_count + group]);
      // Explicitly round through the floating output dtype before absmax.
      value = round_activation<Scalar>(__fmul_rn(scaled, weight_scale[column]));
    }
    values[i] = value;
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float absmax = quant_row_absmax<THREADS>(maximum);
  const float inverse = 127.0f / absmax;
  if (threadIdx.x == 0 && blockIdx.y == 0)
    scales[token] = absmax / 127.0f;
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    const int column = i * THREADS + threadIdx.x;
    if (column >= begin && column < end) {
      const int64_t offset = static_cast<int64_t>(token) * width + column;
      if constexpr (STORE_OUTPUT) {
        if constexpr (std::is_same_v<Scalar, c10::BFloat16>) {
          // values already contain the exact BF16 rounding.
          reinterpret_cast<uint16_t *>(output)[offset] =
              __float_as_uint(values[i]) >> 16;
        } else {
          output[offset] = static_cast<Scalar>(values[i]);
        }
      }
      quantized[offset] = per_token_quant_int8_rn(values[i] * inverse);
    }
  }
}

void check_fusion_tensor(const torch::Tensor &tensor,
                         const torch::Tensor &input, at::ScalarType dtype,
                         const char *name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device(), name,
              " must be on the input GPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
}

// Conservative storage checks also cover padded views for which ATen cannot
// prove byte-level overlap. All public out buffers must own independent
// storage.
void check_fusion_outputs(std::initializer_list<torch::Tensor> inputs,
                          std::initializer_list<torch::Tensor> outputs) {
  for (auto output = outputs.begin(); output != outputs.end(); ++output) {
    if (output->numel() == 0)
      continue;
    for (const auto &input : inputs) {
      TORCH_CHECK(input.numel() == 0 || output->storage().data_ptr().get() !=
                                            input.storage().data_ptr().get(),
                  "fusion outputs must not share storage with inputs");
    }
    for (auto previous = outputs.begin(); previous != output; ++previous) {
      TORCH_CHECK(previous->numel() == 0 ||
                      output->storage().data_ptr().get() !=
                          previous->storage().data_ptr().get(),
                  "fusion outputs must have independent storage");
    }
  }
}

template <typename Scalar, typename Position, int THREADS, bool INVERSE,
          bool FP8 = false>
void launch_rope_quant(const torch::Tensor &input,
                       const torch::Tensor &frequencies,
                       const torch::Tensor &positions, torch::Tensor &quantized,
                       torch::Tensor &scales, int groups, int rope_dim,
                       cudaStream_t stream, bool group_major = false) {
  const int width = input.size(1) / groups * input.size(2);
  const int rows = input.size(0) * groups;
#define LAUNCH_ROPE_IMPL(WIDTH, HD, RD)                                        \
  if constexpr (FP8) {                                                         \
    rope_per_token_quant_fp8_kernel<Scalar, Position, WIDTH, THREADS, INVERSE, \
                                    HD, RD, (HD ? 8 : 0)>                      \
        <<<rows, THREADS, 0, stream>>>(                                        \
            input.data_ptr<Scalar>(),                                          \
            reinterpret_cast<const float *>(frequencies.data_ptr()),           \
            positions.data_ptr<Position>(),                                    \
            reinterpret_cast<int8_t *>(quantized.data_ptr()),                  \
            scales.data_ptr<float>(), groups, input.size(1) / groups,          \
            input.size(2), rope_dim, input.stride(0), input.stride(1),         \
            frequencies.stride(0) * 2,                                         \
            quantized.is_contiguous() ? width : quantized.stride(1),           \
            input.size(0), group_major);                                       \
  } else {                                                                     \
    rope_per_token_quant_int8_kernel<Scalar, Position, WIDTH, THREADS,         \
                                     INVERSE, HD, RD, (HD ? 8 : 0)>            \
        <<<rows, THREADS, 0, stream>>>(                                        \
            input.data_ptr<Scalar>(),                                          \
            reinterpret_cast<const float *>(frequencies.data_ptr()),           \
            positions.data_ptr<Position>(), quantized.data_ptr<int8_t>(),      \
            scales.data_ptr<float>(), groups, input.size(1) / groups,          \
            input.size(2), rope_dim, input.stride(0), input.stride(1),         \
            frequencies.stride(0) * 2, quantized.stride(1));                   \
  }

#define LAUNCH_ROPE(WIDTH) LAUNCH_ROPE_IMPL(WIDTH, 0, 0)
  const bool packed_pairs =
      (input.data_ptr() != nullptr &&
       reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
       reinterpret_cast<uintptr_t>(frequencies.data_ptr()) % 32 == 0 &&
       reinterpret_cast<uintptr_t>(quantized.data_ptr()) % 8 == 0 &&
       input.stride(0) % 8 == 0 && input.stride(1) % 8 == 0 &&
       quantized.stride(1) % 8 == 0);
  if (width == 4096 && input.size(2) == 512 && rope_dim == 64 && groups == 8 &&
      packed_pairs) {
    LAUNCH_ROPE_IMPL(4096, 512, 64);
  } else if (width <= 512) {
    LAUNCH_ROPE(512);
  } else if (width <= 1024) {
    LAUNCH_ROPE(1024);
  } else if (width <= 2048) {
    LAUNCH_ROPE(2048);
  } else if (width <= 4096) {
    LAUNCH_ROPE(4096);
  } else if (width <= 8192) {
    LAUNCH_ROPE(8192);
  } else {
    LAUNCH_ROPE(16384);
  }
#undef LAUNCH_ROPE
#undef LAUNCH_ROPE_IMPL
}

template <typename Scalar, typename Position, bool FP8 = false>
void dispatch_rope_quant(const torch::Tensor &input,
                         const torch::Tensor &frequencies,
                         const torch::Tensor &positions,
                         torch::Tensor &quantized, torch::Tensor &scales,
                         int groups, int rope_dim, bool inverse, int threads,
                         cudaStream_t stream, bool group_major = false) {
#define ROPE(THREADS, INVERSE)                                                 \
  launch_rope_quant<Scalar, Position, THREADS, INVERSE, FP8>(                  \
      input, frequencies, positions, quantized, scales, groups, rope_dim,      \
      stream, group_major)
  if (threads == 64) {
    if (inverse) {
      ROPE(64, true);
    } else {
      ROPE(64, false);
    }
  } else if (threads == 128) {
    if (inverse) {
      ROPE(128, true);
    } else {
      ROPE(128, false);
    }
  } else if (threads == 256) {
    if (inverse) {
      ROPE(256, true);
    } else {
      ROPE(256, false);
    }
  } else if (threads == 512) {
    if (inverse) {
      ROPE(512, true);
    } else {
      ROPE(512, false);
    }
  } else {
    if (inverse) {
      ROPE(1024, true);
    } else {
      ROPE(1024, false);
    }
  }
#undef ROPE
}

template <typename Scalar, int THREADS, bool STORE_OUTPUT>
void launch_scale_quant(const torch::Tensor &accum,
                        const torch::Tensor &input_scale,
                        const torch::Tensor &weight_scale,
                        torch::Tensor *output, torch::Tensor &quantized,
                        torch::Tensor &scales, int groups, bool group_major,
                        int splits, cudaStream_t stream) {
  const int tokens = accum.size(group_major ? 1 : 0);
  const int rank = accum.size(2);
  const int width = groups * rank;
  Scalar *output_ptr = nullptr;
  if constexpr (STORE_OUTPUT)
    output_ptr = output->data_ptr<Scalar>();
  const dim3 grid(tokens, splits);
  const bool vector_aligned =
      reinterpret_cast<uintptr_t>(accum.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(weight_scale.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(output_ptr) % (4 * sizeof(Scalar)) == 0 &&
      reinterpret_cast<uintptr_t>(quantized.data_ptr()) % 4 == 0;
#define LAUNCH_SCALE_FAST(WIDTH, RANK, MAJOR, SPLITS)                          \
  scale_per_token_quant_int8_kernel<Scalar, WIDTH, THREADS, RANK, MAJOR,       \
                                    SPLITS, STORE_OUTPUT>                      \
      <<<grid, THREADS, 0, stream>>>(                                          \
          accum.data_ptr<int32_t>(), input_scale.data_ptr<float>(),            \
          weight_scale.data_ptr<float>(), output_ptr,                          \
          quantized.data_ptr<int8_t>(), scales.data_ptr<float>(), tokens,      \
          groups, rank, group_major, splits)
#define LAUNCH_SCALE_IMPL(WIDTH, RANK) LAUNCH_SCALE_FAST(WIDTH, RANK, false, 0)
#define LAUNCH_SCALE(WIDTH) LAUNCH_SCALE_IMPL(WIDTH, 0)
  if (width == 8192 && rank == 1024 && vector_aligned && !group_major &&
      splits == 1) {
    LAUNCH_SCALE_FAST(8192, 1024, false, 1);
  } else if (width == 8192 && rank == 1024 && vector_aligned && !group_major &&
             splits == 2) {
    LAUNCH_SCALE_FAST(8192, 1024, false, 2);
  } else if (width == 8192 && rank == 1024 && vector_aligned && !group_major &&
             splits == 4) {
    LAUNCH_SCALE_FAST(8192, 1024, false, 4);
  } else if (width == 8192 && rank == 1024 && vector_aligned && !group_major &&
             splits == 8) {
    LAUNCH_SCALE_FAST(8192, 1024, false, 8);
  } else if (width == 8192 && rank == 1024) {
    LAUNCH_SCALE_IMPL(8192, 1024);
  } else if (width <= 512) {
    LAUNCH_SCALE(512);
  } else if (width <= 1024) {
    LAUNCH_SCALE(1024);
  } else if (width <= 2048) {
    LAUNCH_SCALE(2048);
  } else if (width <= 4096) {
    LAUNCH_SCALE(4096);
  } else if (width <= 8192) {
    LAUNCH_SCALE(8192);
  } else {
    LAUNCH_SCALE(16384);
  }
#undef LAUNCH_SCALE
#undef LAUNCH_SCALE_IMPL
#undef LAUNCH_SCALE_FAST
}

} // namespace

void rope_per_token_quant_out(
    const torch::Tensor &input, const torch::Tensor &frequencies,
    const torch::Tensor &positions, torch::Tensor &quantized,
    torch::Tensor &scales, int64_t groups, int64_t rope_dim, bool inverse,
    int64_t threads, bool group_major) {
  const bool fp8 = quantized.scalar_type() == at::kFloat8_e4m3fn;
  TORCH_CHECK(fp8 || quantized.scalar_type() == at::kChar,
              "quantized dtype must be INT8 or E4M3");
  TORCH_CHECK(!group_major || fp8, "group_major requires FP8 output");
  TORCH_CHECK(input.is_cuda() && input.dim() == 3,
              "input must be a GPU [tokens,heads,head_dim] tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kHalf,
              "input must be BF16 or FP16");
  TORCH_CHECK(groups > 0 && input.size(1) > 0 && input.size(1) % groups == 0,
              "groups must divide the number of heads");
  const int64_t width = input.size(1) / groups * input.size(2);
  TORCH_CHECK(input.size(2) > 0 && input.size(2) % 2 == 0 && width <= 16384,
              "head_dim must be positive/even and group width <=16384");
  TORCH_CHECK(rope_dim > 0 && rope_dim <= input.size(2) && rope_dim % 2 == 0,
              "rope_dim must be positive/even and no larger than head_dim");
  TORCH_CHECK(
      input.stride(2) == 1 && input.stride(1) >= input.size(2) &&
          input.stride(0) >= input.size(1) * input.stride(1),
      "input needs positive, non-overlapping rows and unit inner stride");
  check_fusion_tensor(frequencies, input, at::kComplexFloat, "frequencies");
  TORCH_CHECK(
      frequencies.dim() == 2 && frequencies.size(1) == rope_dim / 2 &&
          frequencies.is_contiguous() && frequencies.size(0) > 0,
      "frequencies must be contiguous complex64 [positions,rope_dim/2]");
  TORCH_CHECK(positions.scalar_type() == at::kInt ||
                  positions.scalar_type() == at::kLong,
              "positions must be int32 or int64");
  TORCH_CHECK(
      positions.device() == input.device() && positions.dim() == 1 &&
          positions.size(0) == input.size(0) && positions.is_contiguous(),
      "positions must have one contiguous entry per token on the input GPU");
  check_fusion_tensor(quantized, input,
                      fp8 ? at::kFloat8_e4m3fn : at::kChar, "quantized");
  TORCH_CHECK(
      quantized.dim() == 3 &&
          quantized.size(0) == (group_major ? groups : input.size(0)) &&
          quantized.size(1) == (group_major ? input.size(0) : groups) &&
          quantized.size(2) == width &&
          ((fp8 && quantized.is_contiguous()) ||
           (quantized.stride(2) == 1 && quantized.stride(1) >= width &&
            quantized.stride(0) == quantized.size(1) * quantized.stride(1))),
      "quantized must match the token/group layout with optional row padding");
  check_fusion_tensor(scales, input, at::kFloat, "scales");
  TORCH_CHECK(scales.is_contiguous() && scales.dim() == 3 &&
                  scales.size(0) == quantized.size(0) &&
                  scales.size(1) == quantized.size(1) && scales.size(2) == 1,
              "scales must match the token/group layout and have unit width");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 ||
                  threads == 512 || threads == 1024,
              "threads must be 64, 128, 256, 512 or 1024");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max() / groups,
              "too many quantization rows");
  check_fusion_outputs({input, frequencies, positions}, {quantized, scales});
  if (input.size(0) == 0)
    return;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define DISPATCH_ROPE(SCALAR, FP8)                                             \
  if (positions.scalar_type() == at::kLong)                                    \
    dispatch_rope_quant<SCALAR, int64_t, FP8>(                                 \
        input, frequencies, positions, quantized, scales, groups, rope_dim,    \
        inverse, threads, stream, group_major);                                \
  else                                                                         \
    dispatch_rope_quant<SCALAR, int32_t, FP8>(                                 \
        input, frequencies, positions, quantized, scales, groups, rope_dim,    \
        inverse, threads, stream, group_major)
  if (fp8) {
    if (input.scalar_type() == at::kBFloat16) {
      DISPATCH_ROPE(c10::BFloat16, true);
    } else {
      DISPATCH_ROPE(c10::Half, true);
    }
  } else if (input.scalar_type() == at::kBFloat16) {
    DISPATCH_ROPE(c10::BFloat16, false);
  } else {
    DISPATCH_ROPE(c10::Half, false);
  }
#undef DISPATCH_ROPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
void scale_quant_out_impl(const torch::Tensor &accum,
                          const torch::Tensor &input_scale,
                          const torch::Tensor &weight_scale,
                          torch::Tensor *output, torch::Tensor &quantized,
                          torch::Tensor &scales, at::ScalarType output_dtype,
                          bool group_major, int64_t splits, int64_t threads) {
  TORCH_CHECK(accum.is_cuda() && accum.dim() == 3 && accum.is_contiguous() &&
                  accum.scalar_type() == at::kInt,
              "accum must be contiguous GPU INT32 [T,G,R] or [G,T,R]");
  TORCH_CHECK(output_dtype == at::kBFloat16 || output_dtype == at::kHalf ||
                  output_dtype == at::kFloat,
              "output dtype must be BF16/FP16/FP32");
  const int64_t tokens = accum.size(group_major ? 1 : 0);
  const int64_t groups = accum.size(group_major ? 0 : 1), rank = accum.size(2);
  const int64_t width = groups * rank;
  TORCH_CHECK(groups > 0 && rank > 0 && width <= 16384,
              "full row width must be in [1,16384]");
  if (output != nullptr) {
    check_fusion_tensor(*output, accum, output_dtype, "output");
    TORCH_CHECK(output->is_contiguous() && output->dim() == 3 &&
                    output->size(0) == tokens && output->size(1) == groups &&
                    output->size(2) == rank,
                "output must be contiguous [tokens,groups,group_width]");
  }
  check_fusion_tensor(input_scale, accum, at::kFloat, "input_scale");
  TORCH_CHECK(input_scale.is_contiguous() && input_scale.dim() == 3 &&
                  input_scale.size(0) == tokens &&
                  input_scale.size(1) == groups && input_scale.size(2) == 1,
              "input_scale must be contiguous [tokens,groups,1]");
  check_fusion_tensor(weight_scale, accum, at::kFloat, "weight_scale");
  TORCH_CHECK(weight_scale.is_contiguous() && weight_scale.numel() == width,
              "weight_scale must contain groups*group_width channel scales");
  check_fusion_tensor(quantized, accum, at::kChar, "quantized");
  TORCH_CHECK(quantized.is_contiguous() && quantized.dim() == 2 &&
                  quantized.size(0) == tokens && quantized.size(1) == width,
              "quantized must be contiguous [tokens,full_row_width]");
  check_fusion_tensor(scales, accum, at::kFloat, "scales");
  TORCH_CHECK(scales.is_contiguous() && scales.dim() == 2 &&
                  scales.size(0) == tokens && scales.size(1) == 1,
              "scales must be contiguous [tokens,1]");
  TORCH_CHECK((splits == 1 || splits == 2 || splits == 4 || splits == 8) &&
                  splits <= width,
              "splits must be 1, 2, 4 or 8 and <= width");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "threads must be 128, 256, 512 or 1024");
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(), "too many tokens");
  check_fusion_outputs({accum, input_scale, weight_scale}, {quantized, scales});
  if (output != nullptr) {
    check_fusion_outputs({accum, input_scale, weight_scale, quantized, scales},
                         {*output});
  }
  if (tokens == 0)
    return;
  const at::cuda::OptionalCUDAGuard guard(device_of(accum));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define SCALE(SCALAR, STORE)                                                   \
  if (threads == 128)                                                          \
    launch_scale_quant<SCALAR, 128, STORE>(accum, input_scale, weight_scale,   \
                                           output, quantized, scales, groups,  \
                                           group_major, splits, stream);       \
  else if (threads == 256)                                                     \
    launch_scale_quant<SCALAR, 256, STORE>(accum, input_scale, weight_scale,   \
                                           output, quantized, scales, groups,  \
                                           group_major, splits, stream);       \
  else if (threads == 512)                                                     \
    launch_scale_quant<SCALAR, 512, STORE>(accum, input_scale, weight_scale,   \
                                           output, quantized, scales, groups,  \
                                           group_major, splits, stream);       \
  else                                                                         \
    launch_scale_quant<SCALAR, 1024, STORE>(accum, input_scale, weight_scale,  \
                                            output, quantized, scales, groups, \
                                            group_major, splits, stream)
#define DISPATCH_SCALE(STORE)                                                  \
  if (output_dtype == at::kBFloat16) {                                         \
    SCALE(c10::BFloat16, STORE);                                               \
  } else if (output_dtype == at::kHalf) {                                      \
    SCALE(c10::Half, STORE);                                                   \
  } else {                                                                     \
    SCALE(float, STORE);                                                       \
  }
  if (output != nullptr) {
    DISPATCH_SCALE(true);
  } else {
    DISPATCH_SCALE(false);
  }
#undef DISPATCH_SCALE
#undef SCALE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void scale_per_token_quant_int8_out(
    const torch::Tensor &accum, const torch::Tensor &input_scale,
    const torch::Tensor &weight_scale, std::optional<torch::Tensor> output,
    torch::Tensor &quantized, torch::Tensor &scales,
    at::ScalarType rounding_dtype, bool group_major, int64_t splits,
    int64_t threads) {
  scale_quant_out_impl(accum, input_scale, weight_scale,
                       output.has_value() ? &*output : nullptr, quantized,
                       scales, rounding_dtype, group_major, splits, threads);
}

namespace {
template <typename Scalar, int WIDTH, int THREADS, int HEAD_DIM = 0>
__global__ __launch_bounds__(THREADS) void per_token_quant_fp8_strided_kernel(
    const Scalar *input, int8_t *quantized, float *scales, int heads,
    int dimension, int64_t token_stride, int64_t head_stride,
    int64_t output_pitch) {
  rope_quant_body<Scalar, int64_t, WIDTH, THREADS, false, HEAD_DIM, 0, 1, true,
                  false>(input, nullptr, nullptr, quantized, scales, 1, heads,
                         dimension, 0, token_stride, head_stride, 0,
                         output_pitch, blockIdx.x);
}
template <typename Scalar, int THREADS>
void launch_strided_fp8(const torch::Tensor &input, torch::Tensor &quantized,
                        torch::Tensor &scales, cudaStream_t stream) {
  const int width = input.size(1) * input.size(2), rows = input.size(0);
#define STRIDED(W, HD)                                                         \
  per_token_quant_fp8_strided_kernel<Scalar, W, THREADS, HD>                   \
      <<<rows, THREADS, 0, stream>>>(                                          \
          input.data_ptr<Scalar>(),                                            \
          reinterpret_cast<int8_t *>(quantized.data_ptr()),                    \
          scales.data_ptr<float>(), input.size(1), input.size(2),              \
          input.stride(0), input.stride(1), width)
  const bool aligned =
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
      input.stride(0) % 8 == 0 && input.stride(1) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(quantized.data_ptr()) % 8 == 0;
  if (width == 8192 && input.size(2) == 1024 && aligned) {
    STRIDED(8192, 1024);
  } else if (width <= 512) {
    STRIDED(512, 0);
  } else if (width <= 1024) {
    STRIDED(1024, 0);
  } else if (width <= 2048) {
    STRIDED(2048, 0);
  } else if (width <= 4096) {
    STRIDED(4096, 0);
  } else if (width <= 8192) {
    STRIDED(8192, 0);
  } else {
    STRIDED(16384, 0);
  }
#undef STRIDED
}
} // namespace
void per_token_quant_fp8_strided_impl(
    const torch::Tensor &input, torch::Tensor &quantized, torch::Tensor &scales,
    int64_t threads) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 3,
              "input must be GPU [tokens,heads,dimension]");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kHalf,
              "input must be BF16 or FP16");
  const int64_t tokens = input.size(0), heads = input.size(1),
                dim = input.size(2), width = heads * dim;
  TORCH_CHECK(heads > 0 && dim > 0 && dim % 2 == 0 && width <= 16384,
              "invalid head dimensions");
  TORCH_CHECK(input.stride(2) == 1 &&
                  ((input.stride(1) >= dim &&
                    input.stride(0) >= heads * input.stride(1)) ||
                   (input.stride(0) >= dim &&
                    input.stride(1) >= tokens * input.stride(0))),
              "input needs non-overlapping token-major or head-major rows with "
              "unit inner stride");
  check_fusion_tensor(quantized, input, at::kFloat8_e4m3fn, "quantized");
  check_fusion_tensor(scales, input, at::kFloat, "scales");
  TORCH_CHECK(quantized.is_contiguous() && quantized.dim() == 2 &&
                  quantized.size(0) == tokens && quantized.size(1) == width,
              "quantized must be contiguous [tokens,heads*dimension]");
  TORCH_CHECK(scales.is_contiguous() && scales.dim() == 2 &&
                  scales.size(0) == tokens && scales.size(1) == 1,
              "scales must be contiguous [tokens,1]");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "threads must be 128,256,512,1024");
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(), "too many tokens");
  check_fusion_outputs({input}, {quantized, scales});
  if (!tokens)
    return;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define STRIDED_DISPATCH(S)                                                    \
  if (threads == 128)                                                          \
    launch_strided_fp8<S, 128>(input, quantized, scales, stream);              \
  else if (threads == 256)                                                     \
    launch_strided_fp8<S, 256>(input, quantized, scales, stream);              \
  else if (threads == 512)                                                     \
    launch_strided_fp8<S, 512>(input, quantized, scales, stream);              \
  else                                                                         \
    launch_strided_fp8<S, 1024>(input, quantized, scales, stream)
  if (input.scalar_type() == at::kBFloat16) {
    STRIDED_DISPATCH(c10::BFloat16);
  } else {
    STRIDED_DISPATCH(c10::Half);
  }
#undef STRIDED_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
