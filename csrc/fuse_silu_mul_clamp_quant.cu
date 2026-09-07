#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "reduce.h"

namespace at {
namespace native {

#define LIGHTOP_DISPATCH_CASE_FLOATING_TYPES(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define LIGHTOP_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                    \
      TYPE, NAME, LIGHTOP_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef USE_ROCM
#define LIGHTOP_LDG(arg) __ldg(arg)
#else
#define LIGHTOP_LDG(arg) *(arg)
#endif

static inline __device__ int8_t swiglu_clamp_float_to_int8_rn(float x) {
#ifdef USE_ROCM
  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
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

template <typename scalar_t, bool use_gelu>
__device__ __forceinline__ scalar_t swiglu_clamp_value(
    scalar_t gate, scalar_t up, float limit) {
  float gate_f = static_cast<float>(gate);
  float up_f = static_cast<float>(up);

  if constexpr (use_gelu) {
    // GELU(BF16) -> multiply(BF16) -> clamp(BF16) -> dynamic INT8 quant.
    const scalar_t gate_value = static_cast<scalar_t>(gate_f);
    const scalar_t up_value = static_cast<scalar_t>(up_f);
    const float f = static_cast<float>(gate_value);
    constexpr float ALPHA = M_SQRT1_2;
    const scalar_t activated =
        static_cast<scalar_t>(f * 0.5f * (1.0f + std::erf(f * ALPHA)));
    const scalar_t multiplied =
        static_cast<scalar_t>(activated * up_value);
    float value = static_cast<float>(multiplied);
    value = value < -limit ? -limit : (value > limit ? limit : value);
    return static_cast<scalar_t>(value);
  }
  // clamp(BF16) -> SiLU(BF16) -> multiply(BF16) -> dynamic INT8 quant.
  gate_f = gate_f > limit ? limit : gate_f;
  up_f = up_f < -limit ? -limit : (up_f > limit ? limit : up_f);
  const scalar_t gate_clamped = static_cast<scalar_t>(gate_f);
  const scalar_t up_clamped = static_cast<scalar_t>(up_f);
  constexpr float log2e = 1.44269504088896340736f;
#ifdef USE_ROCM
  const float exp_value =
      __builtin_amdgcn_exp2f(-static_cast<float>(gate_clamped) * log2e);
#else
  const float exp_value = __expf(-static_cast<float>(gate_clamped));
#endif
  const scalar_t activated = static_cast<scalar_t>(
      static_cast<float>(gate_clamped) / (1.0f + exp_value));
  return static_cast<scalar_t>(activated * up_clamped);
}

template <typename scalar_t, bool apply_smooth>
__device__ __forceinline__ scalar_t swiglu_apply_smooth(
    scalar_t value, const float* __restrict__ smooth_scale, int idx) {
  if constexpr (apply_smooth) {
    const scalar_t scale_value = static_cast<scalar_t>(smooth_scale[idx]);
    return static_cast<scalar_t>(
        static_cast<float>(value) * static_cast<float>(scale_value));
  }
  return value;
}

template <typename scalar_t,
          int vec,
          int block_size,
          bool ep,
          bool use_gelu,
          bool apply_smooth>
__global__ void swiglu_clamp_quant_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* __restrict__ mask_m) {
  int64_t token_idx = blockIdx.x;
  int64_t token_end = token_idx + 1;
  int64_t token_stride = 1;
  if constexpr (ep) {
    const int64_t expert_idx = blockIdx.x;
    int64_t valid_m = mask_m[expert_idx];
    valid_m =
        valid_m < 0 ? 0 : (valid_m > tokens_per_expert ? tokens_per_expert
                                                        : valid_m);
    token_idx = expert_idx * tokens_per_expert + blockIdx.y;
    token_end = expert_idx * tokens_per_expert + valid_m;
    token_stride = gridDim.y;
  }

  using InputVec = at::native::memory::aligned_vector<scalar_t, vec>;
  using OutputVec = at::native::memory::aligned_vector<int8_t, vec>;

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  __shared__ float row_absmax;

  for (; token_idx < token_end; token_idx += token_stride) {
    const int idx = threadIdx.x * vec;
    const int64_t input_offset = token_idx * 2 * d + idx;
    const int64_t output_offset = token_idx * d + idx;
    scalar_t gate_values[vec];
    scalar_t up_values[vec];
    scalar_t activated[vec];
    float row_max = 0.0f;

    if (idx < d) {
      *reinterpret_cast<InputVec*>(gate_values) =
          *reinterpret_cast<const InputVec*>(input + input_offset);
      *reinterpret_cast<InputVec*>(up_values) =
          *reinterpret_cast<const InputVec*>(input + input_offset + d);
#pragma unroll
      for (int i = 0; i < vec; ++i) {
        activated[i] = swiglu_apply_smooth<scalar_t, apply_smooth>(
            swiglu_clamp_value<scalar_t, use_gelu>(
                gate_values[i], up_values[i], limit),
            smooth_scale,
            idx + i);
        row_max = fmaxf(row_max, fabsf(static_cast<float>(activated[i])));
      }
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

    if (threadIdx.x == 0) {
      // Match per_token_quant_int8's non-zero minimum scale for an all-zero
      // row.
      row_absmax = fmaxf(row_max, 1.0e-10f);
      scales[token_idx] = row_absmax / 127.0f;
    }
    __syncthreads();

    if (idx < d) {
      int8_t quantized[vec];
      const float inv_scale = 127.0f / row_absmax;
#pragma unroll
      for (int i = 0; i < vec; ++i) {
        quantized[i] = swiglu_clamp_float_to_int8_rn(
            static_cast<float>(activated[i]) * inv_scale);
      }
      *reinterpret_cast<OutputVec*>(output + output_offset) =
          *reinterpret_cast<OutputVec*>(quantized);
    }
    if constexpr (ep) {
      __syncthreads();
    }
  }
}

template <typename scalar_t,
          int block_size,
          bool ep,
          bool use_gelu,
          bool apply_smooth>
__global__ void swiglu_clamp_quant_scalar_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* __restrict__ mask_m) {
  int64_t token_idx = blockIdx.x;
  int64_t token_end = token_idx + 1;
  int64_t token_stride = 1;
  if constexpr (ep) {
    const int64_t expert_idx = blockIdx.x;
    int64_t valid_m = mask_m[expert_idx];
    valid_m =
        valid_m < 0 ? 0 : (valid_m > tokens_per_expert ? tokens_per_expert
                                                        : valid_m);
    token_idx = expert_idx * tokens_per_expert + blockIdx.y;
    token_end = expert_idx * tokens_per_expert + valid_m;
    token_stride = gridDim.y;
  }

  constexpr int shared_size = block_size / C10_WARP_SIZE;
  __shared__ float reduce_scratch[shared_size];
  __shared__ float row_absmax;

  for (; token_idx < token_end; token_idx += token_stride) {
    const int64_t input_offset = token_idx * 2 * d;
    const int64_t output_offset = token_idx * d;
    float row_max = 0.0f;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t value = swiglu_apply_smooth<scalar_t, apply_smooth>(
          swiglu_clamp_value<scalar_t, use_gelu>(
              LIGHTOP_LDG(input + input_offset + idx),
              LIGHTOP_LDG(input + input_offset + d + idx),
              limit),
          smooth_scale,
          idx);
      row_max = fmaxf(row_max, fabsf(static_cast<float>(value)));
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, reduce_scratch);

    if (threadIdx.x == 0) {
      row_absmax = fmaxf(row_max, 1.0e-10f);
      scales[token_idx] = row_absmax / 127.0f;
    }
    __syncthreads();

    const float inv_scale = 127.0f / row_absmax;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t value = swiglu_apply_smooth<scalar_t, apply_smooth>(
          swiglu_clamp_value<scalar_t, use_gelu>(
              LIGHTOP_LDG(input + input_offset + idx),
              LIGHTOP_LDG(input + input_offset + d + idx),
              limit),
          smooth_scale,
          idx);
      output[output_offset + idx] = swiglu_clamp_float_to_int8_rn(
          static_cast<float>(value) * inv_scale);
    }
    if constexpr (ep) {
      __syncthreads();
    }
  }
}

// ===========================================================================
// Specialized BF16 SiLU fast path.
//
// Everything above this point is the original generic implementation and is
// left byte-identical. This block only *adds* a second implementation, and
// `swiglu_fast_launch` below returns false for every input it does not claim,
// so anything outside the narrow predicate keeps running the generic kernels
// with the generic launch configuration and the generic numerics.
//
// Claimed: BF16 + SiLU (no GELU) + no smooth scale + no EP mask + d <= 2048
// with d % 32 == 0 + limit <= 88, on ROCm. That is the shape the GLM FFN
// issues: [M, 4096] BF16 -> [M, 2048] INT8 + [M, 1] FP32, limit 10.
//
// The generic kernel spends ~64 VALU per element and reaches 564 GB/s on that
// shape. Everything below exists to cut that down until the operator is bound
// by DRAM instead: the arithmetic moves into packed 16-bit and VOP3P FP32
// forms, the row reduction moves out of LDS into DPP, and one wave takes a
// whole row so the reduction never needs a barrier.
// ===========================================================================
#ifdef USE_ROCM

// Two FP32 lanes in an adjacent VGPR pair -- the operand shape the VOP3P
// packed FP32 instructions (v_pk_mul_f32, v_pk_add_f32, v_pk_fma_f32) need.
typedef float lightop_f32x2 __attribute__((ext_vector_type(2)));

// BF16 is sign-magnitude, so both SiLU clamp bounds can be applied directly on
// the raw 16-bit patterns with the packed integer min instructions gfx9 has,
// two elements per instruction. With B = the bit pattern of
// static_cast<c10::BFloat16>(limit) and limit > 0:
//   signed   min(x, B)          == min(x, limit)    (a negative BF16 has a
//                                                    negative int16 value, so
//                                                    it never wins against a
//                                                    positive bound)
//   unsigned min(x, B | 0x8000) == max(x, -limit)   (a non-negative BF16 has
//                                                    an unsigned value below
//                                                    0x8000, so it is never
//                                                    clamped)
// This is bit-identical to clamping in FP32 and rounding back to BF16 for
// every non-NaN input, and it removes both FP32 compares and, more
// importantly, both float->BF16 round-trips the FP32 clamp needs. Verified
// exhaustively over all 65536 BF16 patterns for several limits by
// .humanize/lightop-agent/isa_semantics.cpp.
//
// Written as inline asm on purpose: spelled as a C++ vector min, clang lowers
// it to a per-lane v_cmp_lt_i32_sdwa + v_cndmask pair instead of the packed
// form, which costs 4x the instructions in a kernel that is VALU-issue bound.
__device__ __forceinline__ uint32_t swiglu_bf16x2_min_i16(uint32_t a,
                                                          uint32_t b) {
  uint32_t r;
  asm("v_pk_min_i16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

__device__ __forceinline__ uint32_t swiglu_bf16x2_min_u16(uint32_t a,
                                                          uint32_t b) {
  uint32_t r;
  asm("v_pk_min_u16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

// Same sign-magnitude property, used for the row absmax: comparing the
// magnitude bits as unsigned 16-bit integers gives the same order as comparing
// absolute values as floats. Two lanes per instruction, and no BF16 -> FP32
// unpack, which is what a float `fmaxf` would need.
__device__ __forceinline__ uint32_t swiglu_bf16x2_max_u16(uint32_t a,
                                                          uint32_t b) {
  uint32_t r;
  asm("v_pk_max_u16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

// Row absmax across one wave, without touching LDS.
//
// The generic path calls `BlockReduceMax_ROW`, which short-circuits to the
// warp form for a single-wave block -- but `WARP_SHFL_DOWN` lowers to
// `ds_bpermute_b32`, which is why the generic kernel profiles at
// SQ_INSTS_LDS = 8.00/wave with SQ_LDS_BANK_CONFLICT = 0. Those bpermutes are
// a serial dependency chain through the LDS pipe, and because shfl_down leaves
// the answer in lane 0 only, they are followed by an LDS store, a barrier and
// an LDS load just to broadcast it.
//
// DPP runs the same reduction as VALU operand modifiers, and `readlane`
// returns the total in an SGPR, so every lane sees it with no store, no
// barrier and no LDS instruction at all. Removing the barrier is what makes
// the multi-row workgroup below legal.
//
// `bound_ctrl = true` makes out-of-range lanes read 0, and the row masks on
// the two broadcast steps leave untouched rows holding `old`, also 0. Both
// rely on 0 being the identity, so this is valid only for non-negative inputs
// -- which an absmax always is. Verified against a CPU reference over 4096
// rows, including an all-zero row and a maximum that rotates through every
// lane, by .humanize/lightop-agent/dpp_reduce_check.cpp.
__device__ __forceinline__ float swiglu_wave_max(float v) {
  int x = __builtin_bit_cast(int, v);
#define LIGHTOP_SWIGLU_DPP_MAX(ctrl, row_mask)                              \
  x = __builtin_bit_cast(                                                   \
      int, fmaxf(__builtin_bit_cast(float, x),                              \
                 __builtin_bit_cast(float,                                  \
                                    __builtin_amdgcn_update_dpp(            \
                                        0, x, ctrl, row_mask, 0xf, true))))
  LIGHTOP_SWIGLU_DPP_MAX(0x111, 0xf);  // row_shr:1
  LIGHTOP_SWIGLU_DPP_MAX(0x112, 0xf);  // row_shr:2
  LIGHTOP_SWIGLU_DPP_MAX(0x114, 0xf);  // row_shr:4
  LIGHTOP_SWIGLU_DPP_MAX(0x118, 0xf);  // row_shr:8   -> row max in lane 15
  LIGHTOP_SWIGLU_DPP_MAX(0x142, 0xa);  // row_bcast:15 -> pair max in lane 31
  LIGHTOP_SWIGLU_DPP_MAX(0x143, 0xc);  // row_bcast:31 -> wave max in lane 63
#undef LIGHTOP_SWIGLU_DPP_MAX
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(x, 63));
}

// v_cvt_pk_u8_f32 rounds to nearest even, saturates to [0, 255] and inserts
// the byte at `sel` into `packed`, all in one instruction.
template <int sel>
__device__ __forceinline__ uint32_t swiglu_pack_u8(uint32_t packed, float x) {
  asm("v_cvt_pk_u8_f32 %0, %1, %2, %0" : "+v"(packed) : "v"(x), "n"(sel));
  return packed;
}

// clamp(nearbyint(x), -128, 127) for four values, packed into one output
// dword. Biasing by +128 turns the signed saturation into the unsigned
// saturation v_cvt_pk_u8_f32 already performs, and the final XOR maps the
// biased bytes back to two's complement -- so the whole INT8 epilogue costs
// one instruction per element plus one XOR per dword, replacing the generic
// path's nearbyint / two-sided select / cvt_i32 / byte-merge chain.
//
// There is deliberately no `nearbyint` here. v_cvt_pk_u8_f32 rounds to nearest
// even itself, so an explicit v_rndne_f32 ahead of it only recomputes what the
// convert is about to do. .humanize/lightop-agent/cvt_pk_u8_rounding.cpp
// checks the instruction against both candidate rules over the ties and
// near-ties that separate them: 28/28 match RNE, 15/28 match truncation.
// Adding 128 first does not disturb this, because for |x| < 128 the sum is
// exact in FP32 and the bias preserves parity, so a tie stays a tie and rounds
// to the same even neighbour.
//
// The operands arrive already scaled and biased as two packed FP32 pairs, so
// the caller does `x * inv_scale + 128` with two v_pk_fma_f32 rather than four
// scalar v_fma_f32.
__device__ __forceinline__ uint32_t swiglu_quant_dword(lightop_f32x2 lo,
                                                       lightop_f32x2 hi) {
  float a[2], b[2];
  __builtin_memcpy(a, &lo, sizeof(a));
  __builtin_memcpy(b, &hi, sizeof(b));
  uint32_t packed = 0;
  packed = swiglu_pack_u8<0>(packed, a[0]);
  packed = swiglu_pack_u8<1>(packed, a[1]);
  packed = swiglu_pack_u8<2>(packed, b[0]);
  packed = swiglu_pack_u8<3>(packed, b[1]);
  return packed ^ 0x80808080u;
}

// {hi[31:16], lo[31:16]} -- two BF16 results packed into one dword.
//
// The obvious spelling, `(hi & 0xffff0000) | (lo >> 16)`, is two instructions
// (v_lshrrev_b32 + v_and_or_b32). v_perm_b32 does it in one: the selector
// indexes bytes of the 64-bit value {src0, src1} with src0 as the upper dword,
// so 7 and 6 take src0's two high bytes and 3 and 2 take src1's. The selector
// cannot be a VOP3 literal on gfx9, hence the "s" constraint -- it is
// loop-invariant and gets hoisted into an SGPR.
__device__ __forceinline__ uint32_t swiglu_pack_bf16_hi(uint32_t hi,
                                                        uint32_t lo) {
  uint32_t r;
  asm("v_perm_b32 %0, %1, %2, %3"
      : "=v"(r)
      : "v"(hi), "v"(lo), "s"(0x07060302u));
  return r;
}

// Widen a packed BF16 pair to two FP32 lanes, low element first. This is the
// same `x << 16` / `x & 0xffff0000` the scalar path already pays; the point is
// only to land the two results in one adjacent VGPR pair so the arithmetic can
// be issued packed.
__device__ __forceinline__ lightop_f32x2 swiglu_widen_bf16x2(uint32_t pair) {
  const uint32_t words[2] = {pair << 16, pair & 0xffff0000u};
  lightop_f32x2 out;
  // Whole-vector memcpy, not __builtin_bit_cast: bit_cast of an
  // ext_vector_type silently returns element 0 on this toolchain. See
  // .humanize/lightop-agent/profile-artifacts/c8-packed-f32-rejected/.
  __builtin_memcpy(&out, words, sizeof(out));
  return out;
}

// The clamped SiLU -> multiply tail for two elements at once.
//
// The generic path evaluates `gate / (1 + exp(-gate))` with the IEEE divide
// expansion: 2x v_div_scale + v_rcp + 3x v_fma + v_div_fmas + v_div_fixup, 11
// VALU per element. This uses the algebraically equal `g*t / (1+t)` with
// `t = e^g` and a Markstein-refined reciprocal, 8 per element. The refinement
// is correctly rounded for every finite divisor, and `t` is finite only for
// gate <= 88, which is why `swiglu_fast_launch` tests `limit` before claiming
// the launch rather than branching per element.
//
// Every FP32 operation in that form has a VOP3P counterpart on gfx936, so
// handing the compiler an explicit two-lane vector packs six of them (g*log2e,
// 1+t, g*t, num*recip and the two refinement FMAs) into three instructions per
// pair. v_exp_f32 and v_rcp_f32 have no packed form, which is why this saves
// three per pair and not four.
//
// This is bit-exact against the scalar form, not an accuracy trade: a VOP3P
// FP32 op is two independent IEEE FP32 lanes, not a reduced-precision mode.
// Both BF16 roundings are the same RNE expression the generic path's
// `static_cast<scalar_t>` compiles to, minus only the NaN canonicalisation --
// bit-identical for every non-NaN float, verified exhaustively over all 2^32
// patterns by .humanize/lightop-agent/isa_semantics.cpp.
__device__ __forceinline__ uint32_t swiglu_silu_mul_pair(uint32_t gate_pair,
                                                         uint32_t up_pair) {
  constexpr float log2e = 1.44269504088896340736f;
  const lightop_f32x2 g = swiglu_widen_bf16x2(gate_pair);
  const lightop_f32x2 u = swiglu_widen_bf16x2(up_pair);

  const lightop_f32x2 arg = g * log2e;
  lightop_f32x2 t;
  t.x = __builtin_amdgcn_exp2f(arg.x);
  t.y = __builtin_amdgcn_exp2f(arg.y);

  const lightop_f32x2 den = t + 1.0f;
  // -(1 + t), built straight out of an FMA against inline constants rather
  // than written as `-den` at the use site. gfx936 VOP3P has no neg source
  // modifier, so clang materialises a negated vector as one v_xor_b32 per
  // lane: 1 per element, exactly what packing the multiplies had just saved.
  //
  // __builtin_elementwise_fma does not broadcast scalar operands the way the
  // arithmetic operators do, so the constant is splatted explicitly.
  const lightop_f32x2 minus_one = {-1.0f, -1.0f};
  const lightop_f32x2 neg_den =
      __builtin_elementwise_fma(t, minus_one, minus_one);
  const lightop_f32x2 num = g * t;
  lightop_f32x2 recip;
  recip.x = __builtin_amdgcn_rcpf(den.x);
  recip.y = __builtin_amdgcn_rcpf(den.y);
  const lightop_f32x2 quot = num * recip;
  const lightop_f32x2 resid = __builtin_elementwise_fma(neg_den, quot, num);
  const lightop_f32x2 silu = __builtin_elementwise_fma(resid, recip, quot);

  // Round to BF16 and widen straight back. `(bits >> 16) << 16` is exactly the
  // mask, so the intermediate BF16 never has to materialise as 16 bits -- the
  // generic path's codegen folds it the same way.
  uint32_t sb[2];
  __builtin_memcpy(sb, &silu, sizeof(sb));
  const uint32_t act_bits[2] = {
      (sb[0] + 0x7fffu + ((sb[0] >> 16) & 1u)) & 0xffff0000u,
      (sb[1] + 0x7fffu + ((sb[1] >> 16) & 1u)) & 0xffff0000u};
  lightop_f32x2 activated;
  __builtin_memcpy(&activated, act_bits, sizeof(activated));

  const lightop_f32x2 prod = activated * u;
  uint32_t pb[2];
  __builtin_memcpy(pb, &prod, sizeof(pb));
  const uint32_t r0 = pb[0] + 0x7fffu + ((pb[0] >> 16) & 1u);
  const uint32_t r1 = pb[1] + 0x7fffu + ((pb[1] >> 16) & 1u);
  return swiglu_pack_bf16_hi(r1, r0);
}

// `waves_per_row` waves cooperate on one row; `rows_per_block` rows share a
// workgroup. At `waves_per_row == 1` the waves are entirely independent -- no
// LDS, no barrier -- and this only changes how many workgroups the dispatcher
// has to place. Past one, a row's waves must agree on its maximum, which costs
// one LDS round-trip and one barrier.
//
// Splitting a row across waves is what removes the fill/drain ramp. The ramp is
// one wave's own end-to-end latency, so it falls roughly as the work a wave
// owns: measured on this operator's geometry with the SiLU chain replaced by an
// equal-length dependent FMA chain
// (.humanize/lightop-agent/ramp_probe.cpp, d=2048, fitting T = a + b*M):
//
//   waves/row  rows/wg   intercept   us/row     GB/s @ M=11520 / 15872 / 32000
//           1        2      2.90 us  .0088874    1120.9   1133.1   1141.0
//           1        1     23.63 us  .0089741     929.2    975.6   1054.7
//           2        1       3.42 us  .0083363    1186.6   1197.4   1213.3
//           2        4       9.86 us  .0083358    1114.5   1131.7   1185.1
//           4        1      16.35 us  .0074831    1150.7   1244.1   1281.4
//
// Two waves per row is the best of these everywhere the operator was short of
// target. Four has the lowest marginal cost of all and wins at large M, but it
// pays a 16 us intercept that costs exactly the small-M sizes this is meant to
// fix.
//
// There is deliberately no occupancy hint in `__launch_bounds__`. Occupancy is
// not this kernel's constraint, measured three ways at the same code state:
// 40 VGPRs / 6 waves (36 B scratch), 48 / 5 (16 B) and 50 / 4 (none) land
// within 0.15% of each other at M = 32000, and the two pinned forms are no
// better at M = 11520 either, which is where the kernel is furthest from the
// achievable ceiling. Pinning only buys spills.
template <int waves_per_row, int rows_per_block, bool d_is_full>
__global__ void __launch_bounds__(C10_WARP_SIZE* waves_per_row* rows_per_block)
    swiglu_clamp_quant_bf16_fast_kernel(
        const c10::BFloat16* __restrict__ input,
        int8_t* __restrict__ output,
        float* __restrict__ scales,
        int d,
        float limit,
        int64_t num_tokens) {
  // A row's waves reach a barrier below, so every wave of a workgroup must
  // agree on whether to run. With one row per workgroup the bounds check is
  // workgroup-uniform and that holds trivially; with several rows sharing a
  // workgroup a partly-empty tail block would strand the survivors.
  static_assert(waves_per_row == 1 || rows_per_block == 1,
                "a row split across waves needs one row per workgroup");

  // Elements per lane, taken as `nchunks` interleaved groups of `chunk_vec`
  // rather than one contiguous run, so every vector access stays contiguous
  // across the wave: lane L of a chunk touches bytes
  // [L*chunk_vec, (L+1)*chunk_vec). Giving a thread one long contiguous run
  // instead makes each vector instruction stride by `vec` elements per lane,
  // which touches every cache line the wave needs on every instruction and
  // multiplies L2 miss traffic.
  constexpr int chunk_vec = 8;                       // 16 B per lane per chunk
  constexpr int lanes_per_row = C10_WARP_SIZE * waves_per_row;
  constexpr int vec = 32 / waves_per_row;            // lanes_per_row x vec = 2048
  constexpr int nchunks = vec / chunk_vec;
  using InputVec = at::native::memory::aligned_vector<c10::BFloat16, chunk_vec>;

  // Lane index within the row rather than within the wave. At
  // `waves_per_row == 1` the two coincide.
  const int row_lane = static_cast<int>(threadIdx.x) % lanes_per_row;
  // `readfirstlane` is not a micro-optimization. Without it the compiler has
  // no way to know `threadIdx.x / 64` is wave-uniform, so `token_idx` -- and
  // with it every row base address derived from it -- lands in VGPRs and the
  // whole 64-bit row address arithmetic is recomputed per lane. Measured on a
  // version that omitted it: VGPRs 37 -> 54 (occupancy 6 -> 4 waves/SIMD) and
  // VALU 896 -> 1056 per wave.
  const int64_t token_idx =
      static_cast<int64_t>(blockIdx.x) * rows_per_block +
      __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x) /
                                     lanes_per_row);
  // The final workgroup is partly empty whenever `rows_per_block` does not
  // divide `num_tokens`. Its surplus waves return immediately while their
  // neighbours keep working, which is only legal because there is no barrier
  // below for them to desert -- see the static_assert above.
  if (token_idx >= num_tokens) {
    return;
  }

  const uint16_t limit_bits = static_cast<c10::BFloat16>(limit).x;
  const uint32_t limit_pair = static_cast<uint32_t>(limit_bits) * 0x00010001u;
  const uint32_t neg_limit_pair =
      static_cast<uint32_t>(limit_bits | 0x8000u) * 0x00010001u;

  const int64_t row_input_offset = token_idx * 2 * d;
  const int64_t row_output_offset = token_idx * d;

  // The activated row, held in registers between the reduction and the
  // epilogue, as packed BF16 pairs.
  uint32_t activated[vec / 2];
  // The row maximum accumulates as two packed magnitude lanes and widens to
  // FP32 once, after the chunk loop.
  uint32_t amax_pair = 0;

  // Every load for the row is issued before any of it is consumed.
  //
  // Loading and consuming one chunk at a time -- the obvious spelling -- costs
  // a startling amount here, and it does not show up in any steady-state
  // metric. `d_is_full` is what makes the split legal to express: with the
  // `idx < d` guard live, each chunk is its own basic block, the load pairs
  // cannot be hoisted past each other, and the ISA reads
  // `load, load, s_waitcnt vmcnt(0), ~185 VALU` once per chunk. Only two loads
  // are ever in flight, so a wave's own latency is
  // nchunks x (memory latency + a share of the arithmetic) rather than
  // (memory latency + all of it).
  //
  // A wave's own latency is the fill/drain ramp, which is a constant in M --
  // and that is exactly what the measurements show. Fitting the six-point sweep
  // of the one-wave-per-row form gives T(us) = 6.35 + 0.0089914*M (max residual
  // 0.33%), and hoisting the loads moved it to 8.11 + 0.0088734*M: better rate,
  // worse ramp. The same geometry with the arithmetic stripped out entirely
  // (.humanize/lightop-agent/dispatch_floor.cpp) fits 0.34 + 0.0088905*M -- same
  // grid, same accesses, no constant at all. So the ramp is the arithmetic on a
  // wave's critical path, not dispatch overhead, which is what `waves_per_row`
  // above attacks by shrinking the arithmetic a wave owns.
  //
  // Holding every chunk does not cost the VGPRs it looks like it should: the
  // input registers drain 8 per chunk while `activated` fills 4, so the peak
  // simply moves to the top of the consume loop.
  c10::BFloat16 gate_values[nchunks][chunk_vec];
  c10::BFloat16 up_values[nchunks][chunk_vec];

#pragma unroll
  for (int c = 0; c < nchunks; ++c) {
    const int idx = (c * lanes_per_row + row_lane) * chunk_vec;
    if (d_is_full || idx < d) {
      // Plain (cached) loads. Making these non-temporal costs 2.66x here
      // (317.87 us -> 846.13 us): this kernel depends on the cached load
      // returning early enough to feed a ~23 VALU/element chain, so bypassing
      // the cache only removes the overlap. See
      // .humanize/lightop-agent/profile-artifacts/c9-nt-load-rejected/.
      *reinterpret_cast<InputVec*>(gate_values[c]) =
          *reinterpret_cast<const InputVec*>(input + row_input_offset + idx);
      *reinterpret_cast<InputVec*>(up_values[c]) =
          *reinterpret_cast<const InputVec*>(input + row_input_offset + idx +
                                             d);
    }
  }

#pragma unroll
  for (int c = 0; c < nchunks; ++c) {
    const int idx = (c * lanes_per_row + row_lane) * chunk_vec;
    if (d_is_full || idx < d) {
#pragma unroll
      for (int i = 0; i < chunk_vec; i += 2) {
        uint32_t gate_pair;
        uint32_t up_pair;
        __builtin_memcpy(&gate_pair, &gate_values[c][i], sizeof(uint32_t));
        __builtin_memcpy(&up_pair, &up_values[c][i], sizeof(uint32_t));
        gate_pair = swiglu_bf16x2_min_i16(gate_pair, limit_pair);
        up_pair = swiglu_bf16x2_min_i16(
            swiglu_bf16x2_min_u16(up_pair, neg_limit_pair), limit_pair);
        const uint32_t act_pair = swiglu_silu_mul_pair(gate_pair, up_pair);
        activated[(c * chunk_vec + i) / 2] = act_pair;
        amax_pair = swiglu_bf16x2_max_u16(amax_pair, act_pair & 0x7fff7fffu);
      }
    }
  }

  const uint32_t amax_lo = amax_pair & 0xffffu;
  const uint32_t amax_hi = amax_pair >> 16;
  const float row_max =
      __builtin_bit_cast(float, (amax_lo > amax_hi ? amax_lo : amax_hi) << 16);

  // Match the generic path's non-zero minimum scale for an all-zero row.
  float absmax = fmaxf(swiglu_wave_max(row_max), 1.0e-10f);

  // When a row spans several waves, the DPP reduction above only covers one of
  // them. Cross-wave has to go through LDS on this architecture; there is no
  // shuffle that crosses a wave. It costs one dword per wave, one barrier and
  // `waves_per_row` reads, and it is the whole price of the split -- the loads
  // and the arithmetic are already done by the time anything reaches here, so
  // the barrier never stalls a load.
  if constexpr (waves_per_row > 1) {
    __shared__ float wave_absmax[waves_per_row];
    const int wave_in_row = row_lane / C10_WARP_SIZE;
    if (row_lane % C10_WARP_SIZE == 0) {
      wave_absmax[wave_in_row] = absmax;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < waves_per_row; ++w) {
      absmax = fmaxf(absmax, wave_absmax[w]);
    }
  }

  if (row_lane == 0) {
    scales[token_idx] = absmax / 127.0f;
  }
  const float inv_scale = 127.0f / absmax;

  const lightop_f32x2 scale_pair = {inv_scale, inv_scale};
  const lightop_f32x2 bias_pair = {128.0f, 128.0f};

#pragma unroll
  for (int c = 0; c < nchunks; ++c) {
    const int idx = (c * lanes_per_row + row_lane) * chunk_vec;
    if (d_is_full || idx < d) {
      // `x * inv_scale + 128` for two elements in one v_pk_fma_f32. The
      // widening is a shift and a mask either way, so a four-element group
      // costs 4 widens + 2 packed FMAs + 4 converts instead of 4 widens +
      // 4 scalar FMAs + 4 converts.
      uint32_t words[chunk_vec / 4];
#pragma unroll
      for (int j = 0; j < chunk_vec / 4; ++j) {
        const uint32_t act_lo = activated[(c * chunk_vec) / 2 + j * 2];
        const uint32_t act_hi = activated[(c * chunk_vec) / 2 + j * 2 + 1];
        words[j] = swiglu_quant_dword(
            __builtin_elementwise_fma(swiglu_widen_bf16x2(act_lo), scale_pair,
                                      bias_pair),
            __builtin_elementwise_fma(swiglu_widen_bf16x2(act_hi), scale_pair,
                                      bias_pair));
      }
      // Streaming store. This does not reduce L2 miss traffic; what it removes
      // is write back-pressure. The INT8 row is written once and never read
      // back, so holding it in L2 only makes writeback compete with the input
      // stream: `TCC_EA_WRREQ_STALL` drops 276,043 -> 22,026 and
      // `TCP_TA_DATA_STALL_CYCLES` 5.60M -> 3.06M.
      uint64_t store_word;
      __builtin_memcpy(&store_word, words, sizeof(store_word));
      __builtin_nontemporal_store(
          store_word,
          reinterpret_cast<uint64_t*>(output + row_output_offset + idx));
    }
  }
}
#endif  // USE_ROCM

// Claims the launch and returns true, or returns false to leave it to the
// generic dispatch chain below. Every condition is checked here rather than
// inside the kernel, so the specialized kernel carries no fallback scaffolding
// and no workload outside the predicate is affected in any way.
template <typename scalar_t, bool ep, bool use_gelu, bool apply_smooth>
bool swiglu_fast_launch(const torch::Tensor& input,
                        torch::Tensor& output,
                        torch::Tensor& scales,
                        int d,
                        float limit,
                        int64_t num_tokens,
                        cudaStream_t stream) {
#ifdef USE_ROCM
  if constexpr (std::is_same<scalar_t, c10::BFloat16>::value && !ep &&
                !use_gelu && !apply_smooth) {
    // One row's worth of elements, however the kernel splits it across lanes:
    // 64 lanes x 32 elements, or 128 x 16.
    constexpr int vec_per_wave = C10_WARP_SIZE_STATIC * 32;
    // d % 32 == 0 keeps every lane's 8-element chunk in bounds and 16-byte
    // aligned; d <= 2048 is one row's worth at the widths above.
    // The fast SiLU form evaluates e^gate on a gate already clamped to
    // `limit`, so it is finite exactly when limit <= 88 -- tested here, once
    // per launch, instead of per element.
    if (d <= vec_per_wave && d % 32 == 0 && limit <= 88.0f) {
      // C10_WARP_SIZE_STATIC, not C10_WARP_SIZE: torch deliberately makes the
      // latter a non-constexpr `at::cuda::warp_size()` call in host code, so it
      // cannot size a `constexpr`. The device-side uses below are in device
      // context, where the same macro is a constexpr 64.
      //
      // Both branches launch 128-thread workgroups; they differ in whether
      // those 128 threads are two independent rows or two waves cooperating on
      // one. The full-width form takes the cooperative shape, which is worth
      // 5-6% at the sizes that were short of target (see the table on the
      // kernel). The guarded form keeps one wave per row: at d < 2048 the
      // per-chunk bounds check is live anyway, which serializes the loads and
      // removes most of what the split buys.
      constexpr int block = C10_WARP_SIZE_STATIC * 2;
      // `d_is_full` is a compile-time promise that every lane's chunk is in
      // bounds, which deletes the per-chunk guard and lets all loads of a row
      // issue before the first `s_waitcnt`. d == 2048 is the target shape.
      if (d == vec_per_wave) {
        // One row per workgroup, two waves per row. Four waves per row was
        // measured and rejected: the probe predicted it would win above
        // M~16000 on its lower marginal rate, but on the real kernel it fell
        // to ~1064 GB/s against this form's ~1184 (see `attempt-ledger.md`).
        // At four waves a lane owns 8 elements, which is a single chunk, so a
        // wave issues one load pair instead of four and loses the
        // memory-level parallelism that pays for the split.
        const dim3 fast_grid(static_cast<uint32_t>(num_tokens));
        swiglu_clamp_quant_bf16_fast_kernel<2, 1, true>
            <<<fast_grid, block, 0, stream>>>(
                reinterpret_cast<const c10::BFloat16*>(
                    input.data_ptr<scalar_t>()),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                d,
                limit,
                num_tokens);
      } else {
        // Two rows per workgroup, one wave per row. Measured on this
        // operator's exact traffic with the arithmetic stripped out
        // (.humanize/lightop-agent/persistent_probe.cpp, M=32000, d=2048):
        //
        //   1 row/wg   grid 32000   315.1 us   1040 GB/s
        //   2 rows/wg  grid 16000   288.2 us   1137 GB/s
        //   4 rows/wg  grid  8000   313.9 us   1044 GB/s
        //
        // It turns back over past two, so this is not "wider is better".
        constexpr int rows_per_block = 2;
        const dim3 fast_grid(static_cast<uint32_t>(
            (num_tokens + rows_per_block - 1) / rows_per_block));
        swiglu_clamp_quant_bf16_fast_kernel<1, rows_per_block, false>
            <<<fast_grid, block, 0, stream>>>(
                reinterpret_cast<const c10::BFloat16*>(
                    input.data_ptr<scalar_t>()),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                d,
                limit,
                num_tokens);
      }
      return true;
    }
  }
#endif
  return false;
}

template <typename scalar_t, bool ep, bool use_gelu, bool apply_smooth>
void launch_swiglu_clamp_quant(
    const torch::Tensor& input,
    const float* smooth_scale,
    torch::Tensor& output,
    torch::Tensor& scales,
    int d,
    float limit,
    int64_t tokens_per_expert,
    const int32_t* mask_m,
    dim3 grid,
    cudaStream_t stream) {
  // Additive fast path. It claims only the BF16 SiLU shapes enumerated in
  // `swiglu_fast_launch` and returns false for everything else, which falls
  // straight through to the original dispatch chain below -- unchanged, and
  // still the only path for every other dtype, mode and hidden size.
  //
  // `grid.x` is the token count on the non-EP path, which is the only path the
  // predicate accepts.
  // if (swiglu_fast_launch<scalar_t, ep, use_gelu, apply_smooth>(
  //         input,
  //         output,
  //         scales,
  //         d,
  //         limit,
  //         static_cast<int64_t>(grid.x),
  //         stream)) {
  //   return;
  // }
  if (d <= 512 && d % 2 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 2, 256, ep, use_gelu, apply_smooth>
        <<<grid, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 1024 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 128, ep, use_gelu, apply_smooth>
        <<<grid, 128, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 2048 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 256, ep, use_gelu, apply_smooth>
        <<<grid, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 4096 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 512, ep, use_gelu, apply_smooth>
        <<<grid, 512, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 8192 && d % 8 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 8, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else if (d <= 16384 && d % 16 == 0) {
    swiglu_clamp_quant_kernel<scalar_t, 16, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  } else {
    swiglu_clamp_quant_scalar_kernel<
        scalar_t, 1024, ep, use_gelu, apply_smooth>
        <<<grid, 1024, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            smooth_scale,
            output.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            d,
            limit,
            tokens_per_expert,
            mask_m);
  }
}

static void fuse_silu_mul_clamp_quant_impl(
    torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    double limit,
    const torch::Tensor* mask_m,
    int expect_m,
    bool use_ep,
    bool use_gelu,
    const c10::optional<torch::Tensor>& smooth_scale) {
  TORCH_CHECK(
      input.dim() == (use_ep ? 3 : 2),
      use_ep ? "EP input must have shape [experts, tokens, 2 * hidden]"
             : "input must have shape [tokens, 2 * hidden]");
  TORCH_CHECK(input.is_cuda(), "input must be on a CUDA/HIP device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "input's last dimension must be even");
  TORCH_CHECK(limit > 0.0, "limit must be positive");

  const int64_t num_tokens =
      use_ep ? input.size(0) * input.size(1) : input.size(0);
  const int d = input.size(-1) / 2;
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Char,
              "output must be INT8");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
              "scales must be FP32");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(input.device() == output.device() &&
                  input.device() == scales.device(),
              "input, output, and scales must be on the same device");

  const float* smooth_scale_ptr = nullptr;
  if (smooth_scale.has_value()) {
    const torch::Tensor& smooth = smooth_scale.value();
    TORCH_CHECK(smooth.is_cuda(),
                "smooth_scale must be on a CUDA/HIP device");
    TORCH_CHECK(smooth.device() == input.device(),
                "input and smooth_scale must be on the same device");
    TORCH_CHECK(smooth.is_contiguous(),
                "smooth_scale must be contiguous");
    TORCH_CHECK(smooth.scalar_type() == at::ScalarType::Float,
                "smooth_scale must have torch.float32 dtype");
    TORCH_CHECK(smooth.dim() == 1 && smooth.numel() == d,
                "smooth_scale must have shape [hidden]");
    smooth_scale_ptr = smooth.data_ptr<float>();
  }

  int64_t tokens_per_expert = 0;
  const int32_t* mask_ptr = nullptr;
  dim3 grid;
  if (use_ep) {
    const int64_t num_experts = input.size(0);
    tokens_per_expert = input.size(1);
    TORCH_CHECK(
        output.dim() == 3 && output.size(0) == num_experts &&
            output.size(1) == tokens_per_expert && output.size(2) == d,
        "EP output must have shape [experts, tokens, hidden]");
    TORCH_CHECK(
        scales.dim() == 3 && scales.size(0) == num_experts &&
            scales.size(1) == tokens_per_expert && scales.size(2) == 1,
        "EP scales must have shape [experts, tokens, 1]");
    TORCH_CHECK(mask_m != nullptr, "mask_m is required for the EP path");
    const torch::Tensor& mask = *mask_m;
    TORCH_CHECK(mask.is_cuda(), "mask_m must be on a CUDA/HIP device");
    TORCH_CHECK(mask.device() == input.device(),
                "input and mask_m must be on the same device");
    TORCH_CHECK(mask.is_contiguous(), "mask_m must be contiguous");
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Int,
                "mask_m must have torch.int32 dtype");
    TORCH_CHECK(mask.dim() == 1 && mask.numel() == num_experts,
                "mask_m must have shape [experts]");
    mask_ptr = mask.data_ptr<int32_t>();

    if (num_experts != 0 && tokens_per_expert != 0) {
      const int64_t expected_blocks =
          expect_m >= 0 ? int64_t(expect_m) : tokens_per_expert;
      const int grid_y = static_cast<int>(std::max<int64_t>(
          1, std::min(tokens_per_expert, expected_blocks)));
      grid = dim3(static_cast<uint32_t>(num_experts),
                  static_cast<uint32_t>(grid_y));
    }
  } else {
    TORCH_CHECK(output.dim() == 2 && output.size(0) == num_tokens &&
                    output.size(1) == d,
                "output must have shape [tokens, hidden]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == num_tokens &&
                    scales.size(1) == 1,
                "scales must have shape [tokens, 1]");
    if (num_tokens != 0) {
      grid = dim3(static_cast<uint32_t>(num_tokens));
    }
  }

  if (num_tokens == 0 || d == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float limit_f = static_cast<float>(limit);

  LIGHTOP_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fuse_silu_mul_clamp_quant", [&] {
        if (use_ep && use_gelu && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, true, true, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep && use_gelu) {
          launch_swiglu_clamp_quant<scalar_t, true, true, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, true, false, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_ep) {
          launch_swiglu_clamp_quant<scalar_t, true, false, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              tokens_per_expert,
              mask_ptr,
              grid,
              stream);
        } else if (use_gelu && smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, false, true, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              nullptr,
              grid,
              stream);
        } else if (use_gelu) {
          launch_swiglu_clamp_quant<scalar_t, false, true, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              nullptr,
              grid,
              stream);
        } else if (smooth_scale_ptr != nullptr) {
          launch_swiglu_clamp_quant<scalar_t, false, false, true>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              nullptr,
              grid,
              stream);
        } else {
          launch_swiglu_clamp_quant<scalar_t, false, false, false>(
              input,
              smooth_scale_ptr,
              output,
              scales,
              d,
              limit_f,
              0,
              nullptr,
              grid,
              stream);
        }
      });
}

void fuse_silu_mul_clamp_quant(torch::Tensor& input,
                               torch::Tensor& output,
                               torch::Tensor& scales,
                               double limit,
                               bool use_gelu,
                               const c10::optional<torch::Tensor>& smooth_scale) {
  fuse_silu_mul_clamp_quant_impl(
      input, output, scales, limit, nullptr, -1, false, use_gelu, smooth_scale);
}

void fuse_silu_mul_clamp_quant_ep(torch::Tensor& input,
                                  torch::Tensor& output,
                                  torch::Tensor& scales,
                                  double limit,
                                  const torch::Tensor& mask_m,
                                  int expect_m,
                                  bool use_gelu,
                                  const c10::optional<torch::Tensor>& smooth_scale) {
  fuse_silu_mul_clamp_quant_impl(
      input,
      output,
      scales,
      limit,
      &mask_m,
      expect_m,
      true,
      use_gelu,
      smooth_scale);
}

}  // namespace native
}  // namespace at
