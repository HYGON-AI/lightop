/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused DeepseekV4-MLA kernel:
 *   - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last ROPE_DIM
 *   - KV side: GPT-J RoPE on last ROPE_DIM + UE8M0 FP8 quant on NoPE + paged
 *              cache insert
 *
 * Structured as one kernel with a block-specialized grid: Q blocks and KV
 * blocks are scheduled in separate grid regions.  The per-warp RMSNorm/RoPE
 * skeleton is adapted from vllm-deepseek_v4's existing
 * `fusedQKNormRopeKernel` (csrc/fused_qknorm_rope_kernel.cu).
 *
 * Assumptions (hard-coded for DeepseekV4 attention):
 *   HEAD_DIM  = 512
 *   ROPE_DIM  = 64   (RoPE applied to dims [NOPE_DIM, HEAD_DIM))
 *   NOPE_DIM  = 448
 *   QUANT_BLOCK = 64 (UE8M0 FP8 quant block)
 *   FP8_MAX   = 448.0f
 *   is_neox=false (GPT-J interleaved pairs)
 *   cos_sin_cache layout [max_pos, rope_dim] = cos || sin (cos first, sin
 *     second along last dim; each half is rope_dim/2 = 32 values)
 *
 * Cache layout per paged-cache block (block_size tokens):
 *   [0,            bs*576):          token data, 448 fp8 + 128 bf16 each
 *   [bs*576,       bs*576 + bs*8):   UE8M0 scales, 7 real + 1 pad per token
 */

#include <cmath>
#ifndef USE_ROCM
  #include <cuda_fp8.h>
#endif
#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hip/hip_runtime.h>
#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
#endif
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#ifndef VLLM_LDG
  #define VLLM_LDG(x) (*(x))
#endif

#ifndef VLLM_DISPATCH_HALF_TYPES
  #define VLLM_DISPATCH_HALF_TYPES(TYPE, NAME, ...)            \
    AT_DISPATCH_SWITCH(TYPE, NAME,                             \
                       AT_DISPATCH_CASE(at::ScalarType::Half,  \
                                        __VA_ARGS__)           \
                       AT_DISPATCH_CASE(at::ScalarType::BFloat16, \
                                        __VA_ARGS__))
#endif

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

namespace lightop {
namespace deepseek_v4 {

template <typename torch_type>
struct TypeConvert;

template <>
struct TypeConvert<c10::Half> {
  using scalar_type = __half;
  using packed_type = __half2;

  __device__ static inline float convert(scalar_type x) {
    return __half2float(x);
  }
  __device__ static inline float2 convert(packed_type x) {
    return __half22float2(x);
  }
  __device__ static inline scalar_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static inline packed_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
struct TypeConvert<c10::BFloat16> {
#ifdef USE_ROCM
  using scalar_type = __hip_bfloat16;
  using packed_type = __hip_bfloat162;
#else
  using scalar_type = __nv_bfloat16;
  using packed_type = __nv_bfloat162;
#endif

  __device__ static inline float convert(scalar_type x) {
    return __bfloat162float(x);
  }
  __device__ static inline float2 convert(packed_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static inline scalar_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static inline packed_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};

}  // namespace deepseek_v4
}  // namespace lightop

#ifdef USE_ROCM
// ROCm-compatible FP8 conversion helpers. Avoid <hip/hip_fp8.h> here: some
// DTK 26.04 header paths expose device-only helpers while compiling host code.
__device__ __forceinline__ int round_to_nearest_even_int(float val) {
  float const floored = floorf(val);
  int const base = static_cast<int>(floored);
  float const frac = val - floored;
  return base + static_cast<int>(frac > 0.5f || (frac == 0.5f && (base & 1)));
}

__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3_fnuz_fallback(
    float val) {
  if (val == 0.0f) return 0;

  bool const negative = val < 0.0f;
  float const abs_val = fabsf(val);
  constexpr float kMinSubnormal = 0x1p-10f;
  constexpr float kMinNormal = 0x1p-7f;
  constexpr float kMax = 240.0f;
  constexpr int kBias = 8;

  if (abs_val < kMinSubnormal) return 0;
  if (abs_val >= kMax) return static_cast<uint8_t>((negative ? 0x80 : 0) | 0x7f);

  int exp_field = 0;
  int mantissa = 0;
  if (abs_val < kMinNormal) {
    mantissa = round_to_nearest_even_int(abs_val / kMinSubnormal);
    if (mantissa >= 8) {
      exp_field = 1;
      mantissa = 0;
    }
  } else {
    int const exp = static_cast<int>(floorf(log2f(abs_val)));
    float const normalized = ldexpf(abs_val, -exp);
    exp_field = exp + kBias;
    mantissa = round_to_nearest_even_int((normalized - 1.0f) * 8.0f);
    if (mantissa >= 8) {
      mantissa = 0;
      ++exp_field;
    }
    if (exp_field >= 15) {
      exp_field = 15;
      mantissa = 7;
    }
  }

  uint8_t const magnitude =
      static_cast<uint8_t>((exp_field << 3) | (mantissa & 0x7));
  return static_cast<uint8_t>((negative ? 0x80 : 0) | magnitude);
}

// OCP E4M3 (a.k.a. tl.float8e4nv) software fallback.
//   Bias = 7, max finite = 448 (exp=15, mant=6), NaN = (exp=15, mant=7).
//   Subnormal step = 2^-9, smallest normal = 2^-6.
// Encodes a single fp32 into the 8-bit OCP-E4M3 byte using round-to-nearest-
// even with saturation to ±448. Operates directly on the fp32 bit pattern to
// avoid log2f / floorf / ldexpf on the hot path.
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3_ocp_fallback(
    float val) {
  uint32_t bits;
  __builtin_memcpy(&bits, &val, sizeof(bits));

  uint32_t const sign_byte = (bits >> 24) & 0x80u;
  uint32_t const abs_bits = bits & 0x7fffffffu;

  // Values strictly below half of the smallest subnormal (2^-10) round to ±0
  // (sign preserved so we match IEEE / Triton tl.float8e4nv byte output).
  // 2^-10 in fp32 ↔ exp=(127-10)=117 ↔ bits = 0x3a800000.
  constexpr uint32_t kRoundToZeroBits = 0x3a800000u;  // 2^-10
  if (abs_bits < kRoundToZeroBits) return static_cast<uint8_t>(sign_byte);

  // NaN propagation: fp32 NaN → OCP NaN (0x7f, sign ignored).
  if (abs_bits > 0x7f800000u) return 0x7f;

  int const exp_f32 = static_cast<int>(abs_bits >> 23);
  int const unbiased = exp_f32 - 127;
  uint32_t const mant23 = abs_bits & 0x7fffffu;

  // |val| >= 2^9 = 512 saturates to max finite. Caller already clamps to
  // ±448, but inf / huge inputs still need a safe answer.
  if (unbiased >= 9) {
    return static_cast<uint8_t>(sign_byte | 0x7eu);
  }

  uint32_t exp_field;
  uint32_t mantissa;

  if (unbiased >= -6) {
    // Normal range in OCP-E4M3. Round 23-bit mantissa down to 3 bits with RNE.
    constexpr uint32_t kRoundMask = 0xfffffu;   // low 20 bits dropped
    constexpr uint32_t kHalfway = 0x80000u;     // bit 19 (== 2^19)
    uint32_t const dropped = mant23 & kRoundMask;
    uint32_t mant3 = mant23 >> 20;
    bool const round_up =
        (dropped > kHalfway) || (dropped == kHalfway && (mant3 & 1u));
    mant3 += round_up ? 1u : 0u;

    exp_field = static_cast<uint32_t>(unbiased + 7);
    if (mant3 == 8u) {
      mant3 = 0u;
      ++exp_field;
    }
    // Saturate to max finite (0x7e) if we landed on/over NaN slot.
    if (exp_field >= 16u || (exp_field == 15u && mant3 == 7u)) {
      return static_cast<uint8_t>(sign_byte | 0x7eu);
    }
    mantissa = mant3;
  } else {
    // Subnormal range in OCP-E4M3 (unbiased in [-10, -7]).
    // Represent |val| as `m * 2^-9` with m ∈ {1..7} (m == 8 promotes to
    // smallest normal). We have |val| = (1.mant23) * 2^unbiased, so:
    //   m = |val| / 2^-9 = (1.mant23) * 2^(unbiased + 9)
    // Working with the 24-bit mantissa-with-leading-1, this is just a right
    // shift of `14 - unbiased` ∈ [21, 24] bits + RNE.
    uint32_t const mant_with_lead = 0x800000u | mant23;
    int const shift = 14 - unbiased;
    uint32_t const dropped_mask = (1u << shift) - 1u;
    uint32_t const dropped = mant_with_lead & dropped_mask;
    uint32_t const halfway = 1u << (shift - 1);
    uint32_t rounded = mant_with_lead >> shift;
    if (dropped > halfway || (dropped == halfway && (rounded & 1u))) {
      ++rounded;
    }
    if (rounded == 0u) return static_cast<uint8_t>(sign_byte);
    if (rounded >= 8u) {
      // Carry into smallest normal: exp_field = 1, mantissa = 0.
      return static_cast<uint8_t>(sign_byte | 0x08u);
    }
    exp_field = 0u;
    mantissa = rounded;
  }

  return static_cast<uint8_t>(sign_byte | (exp_field << 3) | mantissa);
}

__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
#if defined(__gfx938__)
  int32_t out = 0;
  out = __builtin_hcu_cvt_pk_fp8_f32(val, val, out, false);
  return static_cast<uint8_t>(out & 0xff);
#else
  return rocm_cvt_float_to_fp8_e4m3_ocp_fallback(val);
#endif
}
#endif  // USE_ROCM

namespace vllm {
namespace deepseek_v4_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  // 448
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;   // 7
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  // 8 (7 real + 1 pad)
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;  // 448 + 128 = 576
// Match the encoding chosen in rocm_cvt_float_to_fp8_e4m3: FNUZ on gfx942
// (max 240), OCP on gfx950 (max 448).
// #if defined(USE_ROCM) && !defined(__gfx950__)
// constexpr float kFp8Max = 240.0f;
// #else
// constexpr float kFp8Max = 448.0f;
// #endif
constexpr float kFp8Max = 448.0f;

// Per-warp layout:  32 lanes × 16 elems/lane = 512 elems = HEAD_DIM.
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 16

// ────────────────────────────────────────────────────────────────────────────
// Small inline helpers
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp4MaxAbs(float val) {
  // Reduce absolute max across 4 consecutive lanes (lane id & 3 group).
#ifdef USE_ROCM
  float peer = __shfl_xor(val, 1, 32);
  val = fmaxf(val, peer);
  peer = __shfl_xor(val, 2, 32);
#else
  float peer = __shfl_xor_sync(FINAL_MASK, val, 1);
  val = fmaxf(val, peer);
  peer = __shfl_xor_sync(FINAL_MASK, val, 2);
#endif
  val = fmaxf(val, peer);
  return val;
}

template <typename T>
__device__ __forceinline__ float warpSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
#ifdef USE_ROCM
    val += __shfl_xor(val, mask, 32);
#else
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
#endif
  }
  return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: 1D block-specialized: [0, q_blocks) are Q-only blocks and
// [q_blocks, gridDim.x) are KV-only blocks.  Block: 256 threads = 8 warps.
// Q blocks: each warp handles one (token, q_head) pair.
// KV blocks: each warp handles one token.
//
// With DP padding, q/kv/position_ids can have more rows than slot_mapping.
// The Q branch covers all `num_tokens_full` rows (downstream attention uses
// them).  The KV branch only inserts the first `num_tokens_insert` tokens
// (= slot_mapping length) into the paged cache.
//
template <typename weight_t>
__device__ __forceinline__ float loadWeightAsFloat(
    weight_t const* __restrict__ weight, int const idx) {
  if constexpr (std::is_same_v<weight_t, float>) {
    return weight[idx];
  } else {
    using WeightConverter = lightop::deepseek_v4::TypeConvert<weight_t>;
    typename WeightConverter::scalar_type const* typed_weight =
        reinterpret_cast<typename WeightConverter::scalar_type const*>(weight);
    return WeightConverter::convert(typed_weight[idx]);
  }
}

template <typename scalar_t_in, typename slot_t, typename weight_t>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel(
    scalar_t_in* __restrict__ q_inout,         // [N, H, 512] bf16, in place
    scalar_t_in const* __restrict__ kv_in,     // [N, 512] bf16
    weight_t const* __restrict__ kv_norm_weight,
    uint8_t* __restrict__ k_cache,             // [num_blocks, block_stride]
    slot_t const* __restrict__ slot_mapping,   // [num_tokens_insert] i32/i64
    int64_t const* __restrict__ position_ids,  // [N] i64
    float const* __restrict__ cos_sin_cache,   // [max_pos, 64] fp32
    float const eps,
    int const num_tokens_full,    // = q.size(0) = kv.size(0)
    int const num_tokens_insert,  // = slot_mapping.size(0), ≤ num_tokens_full
    int const num_heads_q,        // H
    int64_t const kv_stride_token,
    int const cache_block_size,   // tokens per paged-cache block
    int const kv_block_stride) {  // bytes per paged-cache block
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // BF16 _typeConvert specialization is unavailable on pre-Ampere.  The
  // DeepseekV4 kernel only runs with bf16 inputs in practice, so compile a
  // no-op stub for sm_70/sm_75 to keep multi-arch builds happy.
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = lightop::deepseek_v4::TypeConvert<scalar_t_in>;

    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("kFp8Max: %f", kFp8Max);
    // }


    constexpr int warpsPerBlock = 8;
    int const warpId = threadIdx.x >> 5;
    int const laneId = threadIdx.x & 31;
    int64_t const q_warps =
        static_cast<int64_t>(num_tokens_full) * num_heads_q;
    int const q_blocks =
        static_cast<int>((q_warps + warpsPerBlock - 1) / warpsPerBlock);
    bool const isKV = blockIdx.x >= q_blocks;

    int tokenIdx;
    int slotIdx;
    if (isKV) {
      int64_t const kvWarpIdx =
          (static_cast<int64_t>(blockIdx.x) - q_blocks) * warpsPerBlock +
          warpId;
      if (kvWarpIdx >= num_tokens_insert) return;
      tokenIdx = static_cast<int>(kvWarpIdx);
      slotIdx = num_heads_q;
    } else {
      int64_t const qWarpIdx =
          static_cast<int64_t>(blockIdx.x) * warpsPerBlock + warpId;
      if (qWarpIdx >= q_warps) return;
      tokenIdx = static_cast<int>(qWarpIdx / num_heads_q);
      slotIdx = static_cast<int>(qWarpIdx - static_cast<int64_t>(tokenIdx) *
                                             num_heads_q);
    }

    // PDL: wait for predecessor kernel (upstream q/kv producer) to signal
    // before touching any global memory.  No-op when PDL is not enabled on
    // the launch.  The CUDA runtime wrapper emits the griddepcontrol.wait
    // PTX with the required memory clobber internally.
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // Dim range this lane owns within the 512-wide head.
    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16

    // ── Load 16 bf16 → 16 fp32 registers (one 16-byte + one 16-byte LDG) ────
    float elements[kElemsPerLane];
    float sumOfSquares = 0.0f;

    scalar_t_in const* src_ptr;
    if (isKV) {
      src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kv_stride_token +
                dim_base;
    } else {
      int64_t const q_row_offset =
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
      src_ptr = q_inout + q_row_offset;
    }

    // Two 16-byte loads per thread (8 bf16 each).  Use uint4 as the vector
    // type and bitcast to scalar_t_in packed pairs for conversion.
    uint4 v0 = *reinterpret_cast<uint4 const*>(src_ptr);
    uint4 v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);

    {
      typename Converter::packed_type const* p0 =
          reinterpret_cast<typename Converter::packed_type const*>(&v0);
      typename Converter::packed_type const* p1 =
          reinterpret_cast<typename Converter::packed_type const*>(&v1);
// Each packed_type holds 2 bf16 → 4 packed = 8 elems per uint4.
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p0[i]);
        elements[2 * i] = f2.x;
        elements[2 * i + 1] = f2.y;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p1[i]);
        elements[8 + 2 * i] = f2.x;
        elements[8 + 2 * i + 1] = f2.y;
      }
    }

    // ── Q branch: RMSNorm with no weight (has_weight=False) ─────────────────
    // Variance + rsqrt + multiply all in fp32, no intermediate bf16 round.
    // The downstream bf16 round only happens at the final store.
    if (!isKV || kv_norm_weight != nullptr) {
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        sumOfSquares += elements[i] * elements[i];
      }
      sumOfSquares = warpSum<float>(sumOfSquares);
      float const rms_rcp =
          rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
      if (!isKV) {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          elements[i] = elements[i] * rms_rcp;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          float const weight =
              loadWeightAsFloat(kv_norm_weight, dim_base + i);
          elements[i] = elements[i] * rms_rcp * weight;
          elements[i] = Converter::convert(Converter::convert(elements[i]));
        }
      }
    }

    // ── GPT-J RoPE on dims [NOPE_DIM, HEAD_DIM) ─────────────────────────────
    // All math in fp32.  cos_sin_cache is loaded as fp32 (its native storage).
    bool const is_rope_lane = dim_base >= kNopeDim;
    if (is_rope_lane) {
      int64_t const pos = position_ids[tokenIdx];
      constexpr int kHalfRope = kRopeDim / 2;  // 32
      float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
      float const* sin_ptr = cos_ptr + kHalfRope;

      int const rope_local_base = dim_base - kNopeDim;  // in [0, 64) step 16
#pragma unroll
      for (int p = 0; p < kElemsPerLane / 2; p++) {
        int const pair_dim = rope_local_base + 2 * p;
        int const half_idx = pair_dim / 2;
        float const cos_v = VLLM_LDG(cos_ptr + half_idx);
        float const sin_v = VLLM_LDG(sin_ptr + half_idx);
        float const x_even = elements[2 * p];
        float const x_odd = elements[2 * p + 1];
        elements[2 * p] = x_even * cos_v - x_odd * sin_v;
        elements[2 * p + 1] = x_even * sin_v + x_odd * cos_v;
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Q branch: cast to bf16 and store back in place.
    // ═══════════════════════════════════════════════════════════════════════
    if (!isKV) {
      uint4 out0, out1;
      typename Converter::packed_type* po0 =
          reinterpret_cast<typename Converter::packed_type*>(&out0);
      typename Converter::packed_type* po1 =
          reinterpret_cast<typename Converter::packed_type*>(&out1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po0[i] = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po1[i] = Converter::convert(
            make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
      }
      scalar_t_in* dst =
          q_inout +
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
      *reinterpret_cast<uint4*>(dst) = out0;
      *reinterpret_cast<uint4*>(dst + 8) = out1;
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
      cudaTriggerProgrammaticLaunchCompletion();
#endif
      return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // KV branch.
    // ═══════════════════════════════════════════════════════════════════════
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id < 0) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
      cudaTriggerProgrammaticLaunchCompletion();
#endif
      return;
    }

    int64_t const block_idx = slot_id / cache_block_size;
    int64_t const pos_in_block = slot_id % cache_block_size;
    uint8_t* block_base =
        k_cache + block_idx * static_cast<int64_t>(kv_block_stride);
    uint8_t* token_fp8_ptr = block_base + pos_in_block * kTokenDataBytes;
    uint8_t* token_bf16_ptr = token_fp8_ptr + kNopeDim;
    uint8_t* token_scale_ptr =
        block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
        pos_in_block * kScaleBytesPerToken;

    // Round K to bf16 first, matching the unfused reference path where K is
    // materialized as bf16 before K quantization.  absmax, clamp, and FP8
    // quant below all run on these bf16-rounded values.
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      elements[i] = Converter::convert(Converter::convert(elements[i]));
    }

    // Per-quant-block absmax must be computed by ALL 32 lanes (warp-collective
    // shuffle requires full participation).  RoPE lanes contribute garbage,
    // but their values are gated out below via `!is_rope_lane`.
    float local_absmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      local_absmax = fmaxf(local_absmax, fabsf(elements[i]));
    }
    float const absmax = fmaxf(warp4MaxAbs(local_absmax), 1e-4f);
    float const exponent = ceilf(log2f(absmax / kFp8Max));
    float const inv_scale = exp2f(-exponent);

    if (!is_rope_lane) {
      // ── NoPE lane: UE8M0 FP8 quant ───────────────────────────────────────
      uint8_t out_bytes[kElemsPerLane];
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        float scaled = elements[i] * inv_scale;
        scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
#ifndef USE_ROCM
        __nv_fp8_storage_t s =
            __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
        out_bytes[i] = static_cast<uint8_t>(s);
#else
      out_bytes[i] = rocm_cvt_float_to_fp8_e4m3(scaled);
#endif
      }
      // One 16-byte STG per lane.
      *reinterpret_cast<uint4*>(token_fp8_ptr + dim_base) =
          *reinterpret_cast<uint4 const*>(out_bytes);

      // Lane (4k) of each 4-lane group writes the scale byte for block k<7.
      if ((laneId & 3) == 0) {
        int const q_block_idx = laneId >> 2;  // 0..6 for NoPE lanes
        float encoded = fmaxf(fminf(exponent + 127.0f, 255.0f), 0.0f);
        token_scale_ptr[q_block_idx] = static_cast<uint8_t>(encoded);
      }
      // Lane 0 also writes the padding byte at index 7.
      if (laneId == 0) {
        token_scale_ptr[kNumQuantBlocks] = 0;  // pad
      }
    } else {
      // ── RoPE lane: cast back to bf16 and store to cache bf16 tail ────────
      uint4 out0, out1;
      typename Converter::packed_type* po0 =
          reinterpret_cast<typename Converter::packed_type*>(&out0);
      typename Converter::packed_type* po1 =
          reinterpret_cast<typename Converter::packed_type*>(&out1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po0[i] = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po1[i] = Converter::convert(
            make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
      }
      int const rope_local_base = dim_base - kNopeDim;  // in [0, 64)
      scalar_t_in* bf16_dst =
          reinterpret_cast<scalar_t_in*>(token_bf16_ptr) + rope_local_base;
      *reinterpret_cast<uint4*>(bf16_dst) = out0;
      *reinterpret_cast<uint4*>(bf16_dst + 8) = out1;
    }
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t_in, typename weight_t>
__global__ void fusedDeepseekV4QNormRopeKVRopeKernel(
    scalar_t_in* __restrict__ q_inout,
    scalar_t_in* __restrict__ kv_inout,
    weight_t const* __restrict__ kv_norm_weight,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache,
    float const eps,
    int const num_tokens,
    int const num_heads_q,
    int64_t const kv_stride_token) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = lightop::deepseek_v4::TypeConvert<scalar_t_in>;

    constexpr int warpsPerBlock = 8;
    int const warpId = threadIdx.x >> 5;
    int const laneId = threadIdx.x & 31;
    int64_t const q_warps = static_cast<int64_t>(num_tokens) * num_heads_q;
    int const q_blocks =
        static_cast<int>((q_warps + warpsPerBlock - 1) / warpsPerBlock);
    bool const isKV = blockIdx.x >= q_blocks;

    int tokenIdx;
    int slotIdx;
    if (isKV) {
      int64_t const kvWarpIdx =
          (static_cast<int64_t>(blockIdx.x) - q_blocks) * warpsPerBlock +
          warpId;
      if (kvWarpIdx >= num_tokens) return;
      tokenIdx = static_cast<int>(kvWarpIdx);
      slotIdx = num_heads_q;
    } else {
      int64_t const qWarpIdx =
          static_cast<int64_t>(blockIdx.x) * warpsPerBlock + warpId;
      if (qWarpIdx >= q_warps) return;
      tokenIdx = static_cast<int>(qWarpIdx / num_heads_q);
      slotIdx = static_cast<int>(qWarpIdx - static_cast<int64_t>(tokenIdx) *
                                             num_heads_q);
    }
    int const dim_base = laneId * kElemsPerLane;

    scalar_t_in const* src_ptr;
    if (isKV) {
      src_ptr = kv_inout + static_cast<int64_t>(tokenIdx) * kv_stride_token +
                dim_base;
    } else {
      src_ptr =
          q_inout +
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
    }

    float elements[kElemsPerLane];
    float sumOfSquares = 0.0f;

    uint4 v0 = *reinterpret_cast<uint4 const*>(src_ptr);
    uint4 v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);

    {
      typename Converter::packed_type const* p0 =
          reinterpret_cast<typename Converter::packed_type const*>(&v0);
      typename Converter::packed_type const* p1 =
          reinterpret_cast<typename Converter::packed_type const*>(&v1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p0[i]);
        elements[2 * i] = f2.x;
        elements[2 * i + 1] = f2.y;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p1[i]);
        elements[8 + 2 * i] = f2.x;
        elements[8 + 2 * i + 1] = f2.y;
      }
    }

    if (!isKV || kv_norm_weight != nullptr) {
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        sumOfSquares += elements[i] * elements[i];
      }
      sumOfSquares = warpSum<float>(sumOfSquares);
      float const rms_rcp =
          rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
      if (!isKV) {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          elements[i] = elements[i] * rms_rcp;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          float const weight =
              loadWeightAsFloat(kv_norm_weight, dim_base + i);
          elements[i] = elements[i] * rms_rcp * weight;
          elements[i] = Converter::convert(Converter::convert(elements[i]));
        }
      }
    }

    bool const is_rope_lane = dim_base >= kNopeDim;
    if (is_rope_lane) {
      int64_t const pos = position_ids[tokenIdx];
      constexpr int kHalfRope = kRopeDim / 2;
      float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
      float const* sin_ptr = cos_ptr + kHalfRope;

      int const rope_local_base = dim_base - kNopeDim;
#pragma unroll
      for (int p = 0; p < kElemsPerLane / 2; p++) {
        int const pair_dim = rope_local_base + 2 * p;
        int const half_idx = pair_dim / 2;
        float const cos_v = VLLM_LDG(cos_ptr + half_idx);
        float const sin_v = VLLM_LDG(sin_ptr + half_idx);
        float const x_even = elements[2 * p];
        float const x_odd = elements[2 * p + 1];
        elements[2 * p] = x_even * cos_v - x_odd * sin_v;
        elements[2 * p + 1] = x_even * sin_v + x_odd * cos_v;
      }
    }

    uint4 out0, out1;
    typename Converter::packed_type* po0 =
        reinterpret_cast<typename Converter::packed_type*>(&out0);
    typename Converter::packed_type* po1 =
        reinterpret_cast<typename Converter::packed_type*>(&out1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po0[i] =
          Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po1[i] = Converter::convert(
          make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
    }

    scalar_t_in* dst;
    if (isKV) {
      dst = kv_inout + static_cast<int64_t>(tokenIdx) * kv_stride_token +
            dim_base;
    } else {
      dst =
          q_inout +
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
    }
    *reinterpret_cast<uint4*>(dst) = out0;
    *reinterpret_cast<uint4*>(dst + 8) = out1;
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

template <typename scalar_t_in, typename slot_t, typename weight_t>
void launchFusedDeepseekV4QNormRopeKVRopeQuantInsert(
    scalar_t_in* q_inout, scalar_t_in const* kv_in, uint8_t* k_cache,
    slot_t const* slot_mapping, weight_t const* kv_norm_weight,
    int64_t const* position_ids, float const* cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int64_t const kv_stride_token,
    int const cache_block_size,
    int const kv_block_stride, cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const q_warps =
      static_cast<int64_t>(num_tokens_full) * num_heads_q;
  int64_t const kv_warps = static_cast<int64_t>(num_tokens_insert);
  int const q_blocks =
      static_cast<int>((q_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  int const kv_blocks =
      static_cast<int>((kv_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  int const grid = q_blocks + kv_blocks;

  // PDL: enable programmatic stream serialization whenever the hardware
  // supports it (SM90+).  On pre-Hopper GPUs the attribute is unavailable,
  // so leave numAttrs = 0 and launch as a regular kernel.
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  // Host-side guard: the device kernel body is compiled as a no-op for
  // bf16 on pre-Ampere (sm_70/sm_75) because _typeConvert<BFloat16> is
  // unavailable there.  Refuse the launch loudly instead of silently
  // skipping the work.
  TORCH_CHECK(
      sm_version >= 80,
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert requires sm_80+ "
      "(Ampere or newer); got sm_",
      sm_version);
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;

  cudaLaunchKernelEx(
      &config,
      fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in, slot_t,
                                                       weight_t>,
      q_inout, kv_in, kv_norm_weight, k_cache, slot_mapping, position_ids,
      cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q,
      kv_stride_token, cache_block_size, kv_block_stride);
#else
  // ROCm: use standard kernel launch syntax (no PDL/stream serialization)
  // clang-format off
  fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in, slot_t, weight_t>
      <<<grid, kBlockSize, 0, stream>>>(
          q_inout, kv_in, kv_norm_weight, k_cache, slot_mapping, position_ids,
          cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q,
          kv_stride_token, cache_block_size, kv_block_stride);
#endif
}

template <typename scalar_t_in, typename weight_t>
void launchFusedDeepseekV4QNormRopeKVRope(
    scalar_t_in* q_inout, scalar_t_in* kv_inout,
    weight_t const* kv_norm_weight, int64_t const* position_ids,
    float const* cos_sin_cache, float const eps, int const num_tokens,
    int const num_heads_q, int64_t const kv_stride_token,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const q_warps = static_cast<int64_t>(num_tokens) * num_heads_q;
  int64_t const kv_warps = static_cast<int64_t>(num_tokens);
  int const q_blocks =
      static_cast<int>((q_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  int const kv_blocks =
      static_cast<int>((kv_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  int const grid = q_blocks + kv_blocks;

#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  TORCH_CHECK(
      sm_version >= 80,
      "fused_deepseek_v4_qnorm_rope_kv_rope requires sm_80+ "
      "(Ampere or newer); got sm_",
      sm_version);
#endif

  fusedDeepseekV4QNormRopeKVRopeKernel<scalar_t_in, weight_t>
      <<<grid, kBlockSize, 0, stream>>>(
          q_inout, kv_inout, kv_norm_weight, position_ids, cos_sin_cache, eps,
          num_tokens, num_heads_q, kv_stride_token);
}

}  // namespace deepseek_v4_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrapper
// ────────────────────────────────────────────────────────────────────────────
namespace at {
namespace native {

template <typename slot_t>
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl(
    Tensor& q,                    // [N, H, 512] bf16, in place
    Tensor const& kv,             // [N, 512] bf16 (read-only)
    Tensor const* kv_norm_weight,
    Tensor& k_cache,              // [num_blocks, block_bytes] uint8
    Tensor const& slot_mapping,   // [N] int32/int64
    Tensor const& position_ids,   // [N] int64
    Tensor const& cos_sin_cache,  // [max_pos, rope_dim] bf16
    double eps, int64_t cache_block_size, at::ScalarType slot_dtype,
    char const* slot_dtype_name) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA");
  TORCH_CHECK(kv.is_cuda(), "kv must be CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == slot_dtype,
              "slot_mapping must be ", slot_dtype_name, " CUDA");
  TORCH_CHECK(position_ids.is_cuda() && position_ids.dtype() == at::kLong,
              "position_ids must be int64 CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q shape [N, H, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  TORCH_CHECK(kv.stride(1) == 1, "kv last dim must be contiguous");
  TORCH_CHECK(kv.size(0) <= 1 || kv.stride(0) >= 512,
              "kv row stride must cover 512 elements");
  TORCH_CHECK(kv.size(0) <= 1 || kv.stride(0) % 8 == 0,
              "kv row stride must be 16-byte aligned for vectorized load");
  TORCH_CHECK(q.dtype() == kv.dtype(), "q and kv dtype must match");
  if (kv_norm_weight != nullptr) {
    TORCH_CHECK(kv_norm_weight->is_cuda() && kv_norm_weight->is_contiguous(),
                "kv_norm_weight must be contiguous CUDA");
    TORCH_CHECK(kv_norm_weight->dim() == 1 && kv_norm_weight->size(0) == 512,
                "kv_norm_weight shape [512]");
  }
  TORCH_CHECK(k_cache.dtype() == at::kByte, "k_cache must be uint8");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
              "cos_sin_cache shape [max_pos, 64]");
  TORCH_CHECK(cos_sin_cache.dtype() == at::kFloat,
              "cos_sin_cache must be float32");

  // With DP padding, slot_mapping can be shorter than q/kv/positions.
  // Q-norm+RoPE runs on all q.size(0) rows (downstream attention uses them);
  // KV quant+insert runs only on the first slot_mapping.size(0) rows.
  int const num_tokens_full = static_cast<int>(q.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens_full &&
                  static_cast<int>(position_ids.size(0)) == num_tokens_full,
              "q/kv/position_ids row counts must match");
  TORCH_CHECK(num_tokens_insert <= num_tokens_full,
              "slot_mapping must not exceed q row count");
  int const num_heads_q = static_cast<int>(q.size(1));
  int64_t const kv_stride_token = kv.stride(0);
  int const cache_block_size_i = static_cast<int>(cache_block_size);
  int const kv_block_stride = static_cast<int>(k_cache.stride(0));

  at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_HALF_TYPES(
      q.scalar_type(), "fused_deepseek_v4_qnorm_rope_kv_insert", [&] {
        using qkv_scalar_t = scalar_t;
        if (kv_norm_weight == nullptr) {
          vllm::deepseek_v4_fused_ops::
              launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<
                  qkv_scalar_t, slot_t, qkv_scalar_t>(
                  reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                  reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
                  reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
                  reinterpret_cast<slot_t const*>(slot_mapping.data_ptr()),
                  nullptr,
                  reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
                  cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                  num_tokens_full, num_tokens_insert, num_heads_q,
                  kv_stride_token, cache_block_size_i, kv_block_stride,
                  stream);
        } else {
          AT_DISPATCH_SWITCH(
              kv_norm_weight->scalar_type(), "kv_norm_weight",
              AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<
                        qkv_scalar_t, slot_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
                        reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
                        reinterpret_cast<slot_t const*>(slot_mapping.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens_full, num_tokens_insert, num_heads_q,
                        kv_stride_token, cache_block_size_i, kv_block_stride,
                        stream);
              })
              AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<
                        qkv_scalar_t, slot_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
                        reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
                        reinterpret_cast<slot_t const*>(slot_mapping.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens_full, num_tokens_insert, num_heads_q,
                        kv_stride_token, cache_block_size_i, kv_block_stride,
                        stream);
              })
              AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<
                        qkv_scalar_t, slot_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
                        reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
                        reinterpret_cast<slot_t const*>(slot_mapping.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens_full, num_tokens_insert, num_heads_q,
                        kv_stride_token, cache_block_size_i, kv_block_stride,
                        stream);
              }));
        }
      });
}
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    Tensor& q,
    Tensor const& kv,
    Tensor& k_cache,
    Tensor const& slot_mapping,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps,
    int64_t cache_block_size) {
  fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl<int64_t>(
      q, kv, nullptr, k_cache, slot_mapping, position_ids, cos_sin_cache, eps,
      cache_block_size, at::kLong, "int64");
}

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32(
    Tensor& q,
    Tensor const& kv,
    Tensor& k_cache,
    Tensor const& slot_mapping,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps,
    int64_t cache_block_size) {
  fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl<int32_t>(
      q, kv, nullptr, k_cache, slot_mapping, position_ids, cos_sin_cache, eps,
      cache_block_size, at::kInt, "int32");
}

void fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
    Tensor& q,
    Tensor const& kv,
    Tensor const& kv_norm_weight,
    Tensor& k_cache,
    Tensor const& slot_mapping,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps,
    int64_t cache_block_size) {
  fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl<int32_t>(
      q, kv, &kv_norm_weight, k_cache, slot_mapping, position_ids,
      cos_sin_cache, eps, cache_block_size, at::kInt, "int32");
}

void fused_deepseek_v4_qnorm_rope_kv_rope_impl(
    Tensor& q,
    Tensor& kv,
    Tensor const* kv_norm_weight,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA");
  TORCH_CHECK(kv.is_cuda(), "kv must be CUDA");
  TORCH_CHECK(position_ids.is_cuda() && position_ids.dtype() == at::kLong,
              "position_ids must be int64 CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q shape [N, H, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  TORCH_CHECK(kv.stride(1) == 1, "kv last dim must be contiguous");
  TORCH_CHECK(kv.size(0) <= 1 || kv.stride(0) >= 512,
              "kv row stride must cover 512 elements");
  TORCH_CHECK(kv.size(0) <= 1 || kv.stride(0) % 8 == 0,
              "kv row stride must be 16-byte aligned for vectorized load");
  TORCH_CHECK(q.dtype() == kv.dtype(), "q and kv dtype must match");
  if (kv_norm_weight != nullptr) {
    TORCH_CHECK(kv_norm_weight->is_cuda() && kv_norm_weight->is_contiguous(),
                "kv_norm_weight must be contiguous CUDA");
    TORCH_CHECK(kv_norm_weight->dim() == 1 && kv_norm_weight->size(0) == 512,
                "kv_norm_weight shape [512]");
  }
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
              "cos_sin_cache shape [max_pos, 64]");
  TORCH_CHECK(cos_sin_cache.dtype() == at::kFloat,
              "cos_sin_cache must be float32");

  int const num_tokens = static_cast<int>(q.size(0));
  TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens &&
                  static_cast<int>(position_ids.size(0)) == num_tokens,
              "q/kv/position_ids row counts must match");
  int const num_heads_q = static_cast<int>(q.size(1));
  int64_t const kv_stride_token = kv.stride(0);

  at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_HALF_TYPES(
      q.scalar_type(), "fused_deepseek_v4_qnorm_rope_kv_rope", [&] {
        using qkv_scalar_t = scalar_t;
        if (kv_norm_weight == nullptr) {
          vllm::deepseek_v4_fused_ops::launchFusedDeepseekV4QNormRopeKVRope<
              qkv_scalar_t, qkv_scalar_t>(
              reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
              reinterpret_cast<qkv_scalar_t*>(kv.data_ptr()),
              nullptr,
              reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
              cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
              num_tokens, num_heads_q, kv_stride_token, stream);
        } else {
          AT_DISPATCH_SWITCH(
              kv_norm_weight->scalar_type(), "kv_norm_weight",
              AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRope<
                        qkv_scalar_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t*>(kv.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens, num_heads_q, kv_stride_token, stream);
              })
              AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRope<
                        qkv_scalar_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t*>(kv.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens, num_heads_q, kv_stride_token, stream);
              })
              AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                using weight_scalar_t = scalar_t;
                vllm::deepseek_v4_fused_ops::
                    launchFusedDeepseekV4QNormRopeKVRope<
                        qkv_scalar_t, weight_scalar_t>(
                        reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
                        reinterpret_cast<qkv_scalar_t*>(kv.data_ptr()),
                        reinterpret_cast<weight_scalar_t const*>(
                            kv_norm_weight->data_ptr()),
                        reinterpret_cast<int64_t const*>(
                            position_ids.data_ptr()),
                        cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                        num_tokens, num_heads_q, kv_stride_token, stream);
              }));
        }
      });
}

void fused_deepseek_v4_qnorm_rope_kv_rope(
    Tensor& q,
    Tensor& kv,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps) {
  fused_deepseek_v4_qnorm_rope_kv_rope_impl(
      q, kv, nullptr, position_ids, cos_sin_cache, eps);
}

void fused_deepseek_v4_qnorm_rope_kvnorm_rope(
    Tensor& q,
    Tensor& kv,
    Tensor const& kv_norm_weight,
    Tensor const& position_ids,
    Tensor const& cos_sin_cache,
    double eps) {
  fused_deepseek_v4_qnorm_rope_kv_rope_impl(
      q, kv, &kv_norm_weight, position_ids, cos_sin_cache, eps);
}

}  // namespace native
}  // namespace at
