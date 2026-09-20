// E2M1 payload + per-32 E8M0 scales. Decode directly into MMAC registers;
// no intermediate dequantized cache, GEMM output or head-reduction tensor.
//
// Numerics follow DeepGEMM's MXFP4 MQA logits (sm100_mqa_logits.cuh): each
// per-32 block scale is folded into the K operand before it reaches a single
// FP32 accumulator spanning all 128 channels, and the epilogue applies relu,
// the head weight and the head reduction in FP32 with no extra scale. Both
// dispatches below implement that same contract; they differ only in how the
// block scale reaches the operand.
#include "paged_mqa_logits_fp4.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <climits>

namespace lightop::deep_gemm {
namespace {
using bf16x4 = short __attribute__((ext_vector_type(4)));
using floatx4 = float __attribute__((ext_vector_type(4)));
using bf16x8 = unsigned short __attribute__((ext_vector_type(8)));
using intx2 = int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint16_t to_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ float from_bf16(uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}
// DeepGEMM's sm100 MQA logits epilogue (the default FP32 `reduce_dtype_t`)
// keeps relu, weighting and the head reduction entirely in FP32, and adds no
// further scale for MX scale factors. It spells relu as (a + |a|) with a
// closing halving because that maps to a free absolute-value modifier on
// SM100; fmaxf is a single instruction here, so the relu is written directly.
// The two forms agree on every finite accumulator, and fmaxf additionally
// keeps a -inf dot pinned at zero instead of turning it into a NaN.
template<int HeadTiles>
__device__ __forceinline__ float reduce_heads(floatx4* values) {
    float sum_0 = 0.0f, sum_1 = 0.0f;
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
        sum_0 += values[h][0] + values[h][2];
        sum_1 += values[h][1] + values[h][3];
    }
    float sum = sum_0 + sum_1;
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    return sum;
}

// Shared epilogue for both dispatches: FP32 relu, FP32 weighting, no rounding.
template<int H, int HeadTiles>
__device__ __forceinline__ float weighted_head_sum(floatx4* acc, float (*wr)[4], int group) {
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[h][j] = h * 16 + j * 4 + group < H ? fmaxf(acc[h][j], 0.0f) * wr[h][j] : 0.0f;
        }
    }
    return reduce_heads<HeadTiles>(acc);
}

__device__ __forceinline__ uint16_t decode_fp4(unsigned code, unsigned scale) {
    const unsigned mag = code & 7u;
    const unsigned sign = (code & 8u) << 12;
    const unsigned base = mag >= 2 ? 0x3f00u + (mag << 6) : (mag == 1 ? 0x3f00u : 0u);
    const int exponent = static_cast<int>(base >> 7) + static_cast<int>(scale) - 127;
    // Common path is exact exponent adjustment in BF16, with no FP32 exp2.
    if (scale < 255 && (mag == 0 || (exponent > 0 && exponent < 255))) {
        return sign | (mag == 0 ? 0 : (exponent << 7) | (base & 0x7fu));
    }
    // Preserve exp2(scale-127), multiply, BF16 rounding for tiny/huge scales.
    // Scale 255 follows the source Triton kernel's +inf interpretation.
    const float factor = scale == 255 ? INFINITY : ldexpf(1.0f, static_cast<int>(scale) - 127);
    return to_bf16(from_bf16(sign | base) * factor);
}

__device__ __forceinline__ bf16x4 decode_fp4x4(unsigned packed, unsigned scale) {
    bf16x4 result;
    if (scale >= 2 && scale <= 252) {
        const int delta = (static_cast<int>(scale) - 127) * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned code = (packed >> (j * 4)) & 15u;
            const unsigned mag = code & 7u;
            const unsigned sign = (code & 8u) << 12;
            const int base = mag >= 2 ? 0x3f00 + (mag << 6) : 0x3f00;
            result[j] = sign | (mag == 0 ? 0 : base + delta);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) result[j] = decode_fp4((packed >> (j * 4)) & 15u, scale);
    }
    return result;
}

template<int H, int TilesPerWave>
__device__ __forceinline__ void fp4_logits_wave(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ weights,
    const int64_t* __restrict__ slots, const int64_t* __restrict__ lens,
    const uint8_t* __restrict__ table, float* __restrict__ out,
    int length, int page_size, int page_shift, int64_t table_stride,
    int64_t qb_stride, int64_t qh_stride, int64_t wb_stride) {
    constexpr int HeadTiles = (H + 15) / 16;
    constexpr int Tile = 16;
    constexpr int TokensPerCTA = 4 * Tile * TilesPerWave;
    const int b = blockIdx.y;
    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int row = lane % 16;
    const int group = lane / 16;
    const int64_t start = static_cast<int64_t>(blockIdx.x) * TokensPerCTA + wave * Tile * TilesPerWave;
    const int64_t visible = min(lens[b], static_cast<int64_t>(length));
    if (start >= visible) {
        for (int i = lane; i < Tile * TilesPerWave; i += 64) {
            if (start + i < length) out[static_cast<int64_t>(b) * length + start + i] = -INFINITY;
        }
        return;
    }

    // Each wave reuses its Q fragments across all its KV tiles. Read-only
    // cache broadcasts Q across CTAs, and no cross-wave LDS barrier is needed.
    bf16x4 qr[HeadTiles][8];
    float wr[HeadTiles][4];
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            qr[h][k] = {};
            if (h * 16 + row < H) {
                qr[h][k] = *reinterpret_cast<const bf16x4*>(
                    q + b * qb_stride + (h * 16 + row) * qh_stride + k * 16 + group * 4);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int head = h * 16 + j * 4 + group;
            wr[h][j] = head < H ? from_bf16(weights[b * wb_stride + head]) : 0.0f;
        }
    }
#pragma unroll
    for (int tile = 0; tile < TilesPerWave; ++tile) {
        const int64_t pos = start + tile * Tile + row;
        const bool valid = pos < visible;
        const int64_t slot = valid ? slots[static_cast<int64_t>(b) * length + pos] : 0;
        // All production pool pages are powers of two. Uniform runtime shifts
        // also cover non-default page sizes without 64-bit integer division.
        const int64_t page = page_shift >= 0 ? slot >> page_shift : slot / page_size;
        const int offset = page_shift >= 0 ? slot & (page_size - 1) : slot % page_size;
        const uint8_t* payload = table + page * table_stride + offset * 64;
        const uint8_t* scales = table + page * table_stride + page_size * 64 + offset * 4;
        // Gather just 64 payload bytes and 4 scales for each visible slot.
        // Invalid tail lanes use zeros and never dereference pool addresses.
        uint16_t packed[8];
        unsigned exps[4];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            packed[k] = valid ? *reinterpret_cast<const uint16_t*>(payload + k * 8 + group * 2) : 0;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) exps[k] = valid ? scales[k] : 127;
        // Every K block is decoded with its own scale already folded into the
        // BF16 operand, so all 128 scaled products enter a single FP32
        // accumulator. This is DeepGEMM's block-scaled MMA contract: scale the
        // operand, never rescale a partial accumulator.
        floatx4 acc[HeadTiles] = {};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const bf16x4 kr = decode_fp4x4(packed[k], exps[k / 2]);
#pragma unroll
            for (int h = 0; h < HeadTiles; ++h) {
#if defined(__gfx938__)
                acc[h] = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(kr, qr[h][k], acc[h], false, false);
#elif defined(__gfx936__) || defined(__gfx928__)
                acc[h] = __builtin_amdgcn_mmac_f32_16x16x16bf16(kr, qr[h][k], acc[h]);
#endif
            }
        }
        const float sum = weighted_head_sum<H, HeadTiles>(acc, wr, group);
        if (lane < 16 && pos < length) {
            out[static_cast<int64_t>(b) * length + pos] = valid ? sum : -INFINITY;
        }
    }
}

template<int H, int TilesPerWave>
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fp4_logits_kernel(
    const uint16_t* q, const uint16_t* weights, const int64_t* slots,
    const int64_t* lens, const uint8_t* table, float* out,
    int length, int page_size, int page_shift, int64_t table_stride,
    int64_t qb_stride, int64_t qh_stride, int64_t wb_stride) {
    fp4_logits_wave<H, TilesPerWave>(q, weights, slots, lens, table, out,
        length, page_size, page_shift, table_stride, qb_stride, qh_stride, wb_stride);
}

// gfx938 has FP8 K=32 MMAC, but no native block-scaled FP4 MMAC: clang rejects
// __builtin_hcu_mmac_f32_16x16x64_fp4 for gfx938 with "needs target feature
// gfx946-insts", and the block-scaled form wants gfx948-insts. Every E2M1 value
// is m * 2^e with m in {1, 1.5} and e in [-1, 2], so subtracting a per-block
// delta from the E4M3 exponent field folds that block's scale into the operand
// exactly -- the same thing DeepGEMM's block-scaled UMMA does before its single
// FP32 accumulator. What remains outside the dot is only
// 2^(max_qexp + pivot - 254), a factor common to all 128 terms, so applying it
// once to the finished dot is exact and accumulation-order independent --
// unlike rescaling separate FP32 partial sums.
//
// delta is signed: the pivot the deltas are measured from is free, not pinned
// to the token's largest block scale, which is what lets a token span more
// than the 5 exponent steps available below that maximum.
__device__ __forceinline__ intx2 fp4_to_fp8(unsigned packed, int delta) {
    // Magnitudes 0..3 come from the low constant, 4..7 from the high one.
    // Byte 0 of the low constant stays zero so code 0 keeps decoding to zero.
    // delta in [-6, 5] is the exact window, verified by enumerating all eight
    // E2M1 codes against every shift: at delta 5 the smallest byte is 0x08
    // (2^-6, the minimum E4M3 normal) and at delta -6 the largest is 0x7c
    // (384, still below the 0x7f NaN); one step further either way a byte
    // borrows to zero or carries into the sign bit.
    const unsigned hi = 0x4c484440u - static_cast<unsigned>(delta) * 0x08080808u;
    const unsigned lo = 0x3c383000u - static_cast<unsigned>(delta) * 0x08080800u;
    const unsigned even = __builtin_amdgcn_perm(hi, lo, packed & 0x07070707u)
                        | ((packed & 0x08080808u) << 4);
    const unsigned odd = __builtin_amdgcn_perm(hi, lo, (packed >> 4) & 0x07070707u)
                       | (packed & 0x80808080u);
    return {static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x05010400u)),
            static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
}

template<int H>
__global__ void pack_fp4_query(const uint16_t* q, uint8_t* packed_q, int* metadata,
                               int64_t qb_stride, int64_t qh_stride) {
    constexpr int HeadTiles = (H + 15) / 16;
    const int b = blockIdx.x, lane = threadIdx.x;
    const int row = lane % 16, group = lane / 16;
    bf16x8 raw_q[HeadTiles][4];
    unsigned max_qexp = 0;
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            bf16x8 raw = {};
            if (h * 16 + row < H)
                raw = *reinterpret_cast<const bf16x8*>(q + b * qb_stride
                        + (h * 16 + row) * qh_stride + k * 32 + group * 8);
            raw_q[h][k] = raw;
#pragma unroll
            for (int j = 0; j < 8; ++j) max_qexp = max(max_qexp, (raw[j] >> 7) & 255u);
        }
    }
#pragma unroll
    for (int delta = 1; delta < 64; delta *= 2)
        max_qexp = max(max_qexp, __shfl_xor(max_qexp, delta, 64));
    if (max_qexp == 0) max_qexp = 127;
    bool eligible = true;
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            intx2 qr = {};
            const bf16x8 raw = raw_q[h][k];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned bits = raw[j], mag = bits & 0x7fffu;
                const int exp = static_cast<int>((bits >> 7) & 255u) - max_qexp + 7;
                eligible &= mag == 0 || ((bits & 63u) == 0 && exp >= 1 && exp <= 8 && mag < 0x7f80u);
                const unsigned f8 = ((bits >> 8) & 128u)
                    | (mag == 0 ? 0u : (static_cast<unsigned>(exp) << 3) | ((bits >> 4) & 7u));
                qr[j / 4] |= f8 << ((j % 4) * 8);
            }
            if (h * 16 + row < H)
                *reinterpret_cast<intx2*>(packed_q + (b * H + h * 16 + row) * 128 + k * 32 + group * 8) = qr;
        }
    }
    const bool all_eligible = __ballot(!eligible) == 0;
    if (lane == 0) {
        metadata[b * 2] = max_qexp;
        metadata[b * 2 + 1] = all_eligible;
    }
}

template<int H, int TilesPerWave, int StaticPageShift = -1>
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fp4_fp8_logits_kernel(
    const uint16_t* q, const uint16_t* weights, const int64_t* slots,
    const int64_t* lens, const uint8_t* table, float* out,
    int length, int page_size, int page_shift, int64_t table_stride,
    int64_t qb_stride, int64_t qh_stride, int64_t wb_stride,
    const uint8_t* packed_q, const int* metadata) {
#if defined(__gfx938__)
    constexpr int HeadTiles = (H + 15) / 16;
    constexpr int Tile = 16;
    const int b = blockIdx.y, lane = threadIdx.x % 64;
    const int row = lane % 16, group = lane / 16;
    const int64_t start = static_cast<int64_t>(blockIdx.x) * (64 * TilesPerWave)
                        + (threadIdx.x / 64) * Tile * TilesPerWave;
    const int64_t visible = min(lens[b], static_cast<int64_t>(length));
    if (start >= visible) {
        for (int i = lane; i < Tile * TilesPerWave; i += 64)
            if (start + i < length) out[static_cast<int64_t>(b) * length + start + i] = -INFINITY;
        return;
    }
    intx2 qr[HeadTiles][4];
    float wr[HeadTiles][4];
    const unsigned max_qexp = metadata[b * 2];
    if (!metadata[b * 2 + 1]) {
        fp4_logits_wave<H, TilesPerWave>(q, weights, slots, lens, table, out,
            length, page_size, page_shift, table_stride, qb_stride, qh_stride, wb_stride);
        return;
    }
    // A common Q scale absorbs the original per-head/per-block Q scales
    // into the exact FP8 operands. This eliminates epilogue scale shuffles.
#pragma unroll
    for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qr[h][k] = {};
            if (h * 16 + row < H)
                qr[h][k] = *reinterpret_cast<const intx2*>(packed_q
                    + (b * H + h * 16 + row) * 128 + k * 32 + group * 8);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int head = h * 16 + j * 4 + group;
            wr[h][j] = head < H ? from_bf16(weights[b * wb_stride + head]) : 0.0f;
        }
    }
#pragma unroll
    for (int tile = 0; tile < TilesPerWave; ++tile) {
        const int64_t pos = start + tile * Tile + row;
        const bool valid = pos < visible;
        const int64_t slot = valid ? slots[static_cast<int64_t>(b) * length + pos] : 0;
        // page_size is normally 64 in the serving pool. Specializing that
        // layout removes two dynamic 64-bit page-address operations from the
        // inner loop; the generic instantiation preserves every other layout.
        const int64_t page = StaticPageShift >= 0 ? slot >> StaticPageShift
            : (page_shift >= 0 ? slot >> page_shift : slot / page_size);
        const int offset = StaticPageShift >= 0 ? slot & ((1 << StaticPageShift) - 1)
            : (page_shift >= 0 ? slot & (page_size - 1) : slot % page_size);
        const int static_page_size = StaticPageShift >= 0 ? 1 << StaticPageShift : page_size;
        const uint8_t* payload = table + page * table_stride + offset * 64;
        const uint8_t* scales = table + page * table_stride + static_page_size * 64 + offset * 4;
        // FP8 dispatch guarantees 4-byte table/row alignment. One vector
        // load supplies the four per-32 scales without changing any arithmetic.
        const unsigned packed_scales = valid ? *reinterpret_cast<const unsigned*>(scales) : 0x7f7f7f7fu;
        unsigned exps[4];
        bool scales_ok = true;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            exps[k] = (packed_scales >> (k * 8)) & 255u;
            scales_ok &= exps[k] >= 2 && exps[k] <= 252;
        }
        const unsigned min_scale = min(min(exps[0], exps[1]), min(exps[2], exps[3]));
        const unsigned max_scale = max(max(exps[0], exps[1]), max(exps[2], exps[3]));
        // Deltas are measured from a free pivot rather than from max_scale.
        // The exact fold window is delta in [-6, 5], so any pivot in
        // [max_scale - 6, min_scale + 5] keeps every block exact and a span of
        // 11 is reachable. Clamping at max_scale leaves every span <= 5 token
        // bit-identical to the pinned-pivot form; wider spans slide the pivot
        // down instead of dropping the whole wave to the BF16 path. The
        // leftover common factor must still be a normal FP32.
        const int pivot = static_cast<int>(min(max_scale, min_scale + 5u));
        scales_ok &= max_scale - min_scale <= 11
                  && static_cast<int>(max_qexp) + pivot - 127 >= 1
                  && static_cast<int>(max_qexp) + pivot - 127 <= 240;
        if (__ballot(!scales_ok)) {
            fp4_logits_wave<H, TilesPerWave>(q, weights, slots, lens, table, out,
                length, page_size, page_shift, table_stride, qb_stride, qh_stride, wb_stride);
            return;
        }
        floatx4 acc[HeadTiles] = {};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned packed = valid ? *reinterpret_cast<const unsigned*>(payload + k * 16 + group * 4) : 0;
            const intx2 kr = fp4_to_fp8(packed, pivot - static_cast<int>(exps[k]));
#pragma unroll
            for (int h = 0; h < HeadTiles; ++h) {
                acc[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(kr, qr[h][k], acc[h], false, false);
            }
        }
        // One power of two common to all 128 terms, applied to the finished
        // dot. Exact, and independent of the accumulation order.
        const float factor = __uint_as_float(
            static_cast<unsigned>(static_cast<int>(max_qexp) + pivot - 127) << 23);
#pragma unroll
        for (int h = 0; h < HeadTiles; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[h][j] *= factor;
        }
        const float sum = weighted_head_sum<H, HeadTiles>(acc, wr, group);
        if (lane < 16 && pos < length) out[static_cast<int64_t>(b) * length + pos] = valid ? sum : -INFINITY;
    }
#else
    fp4_logits_wave<H, TilesPerWave>(q, weights, slots, lens, table, out,
        length, page_size, page_shift, table_stride, qb_stride, qh_stride, wb_stride);
#endif
}
}

torch::Tensor paged_mqa_logits_fp4(
    const torch::Tensor& q, const torch::Tensor& weights,
    const torch::Tensor& slots, const torch::Tensor& lens,
    const torch::Tensor& table, int64_t page_size, std::optional<torch::Tensor> output) {
    TORCH_CHECK(q.is_cuda(), "paged_mqa_logits_fp4 unsupported: q must be a CUDA/HIP tensor");
    TORCH_CHECK(q.dim() == 3, "paged_mqa_logits_fp4 unsupported: q must have shape [B,H,128], got dim=", q.dim());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                "paged_mqa_logits_fp4 unsupported: q dtype must be BF16, got ", q.scalar_type());
    TORCH_CHECK(q.size(2) == 128,
                "paged_mqa_logits_fp4 unsupported: head_dim must be 128, got ", q.size(2));
    const int64_t batch = q.size(0), heads = q.size(1);
    TORCH_CHECK(heads == 8 || heads == 16 || heads == 32 || heads == 64,
                "paged_mqa_logits_fp4 unsupported: heads must be one of {8,16,32,64}, got ", heads);
    TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 4 == 0 && q.stride(0) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0, "q rows must have aligned contiguous D");
    TORCH_CHECK(weights.device() == q.device() && weights.scalar_type() == at::kBFloat16 &&
                weights.dim() == 2 && weights.size(0) == batch && weights.size(1) == heads && weights.stride(1) == 1,
                "weights must be BF16 [B,H] with contiguous heads on q device");
    TORCH_CHECK(slots.device() == q.device() && slots.scalar_type() == at::kLong && slots.dim() == 2 &&
                slots.size(0) == batch && slots.is_contiguous(), "slots must be contiguous int64 [B,L] on q device");
    TORCH_CHECK(lens.device() == q.device() && lens.scalar_type() == at::kLong && lens.dim() == 1 &&
                lens.size(0) == batch && lens.is_contiguous(), "lens must be contiguous int64 [B] on q device");
    TORCH_CHECK(page_size > 0 && page_size <= INT_MAX / 68, "invalid page_size");
    TORCH_CHECK(table.device() == q.device() && table.scalar_type() == at::kByte && table.dim() == 2 &&
                table.size(1) == page_size * 68 && table.stride(1) == 1 && table.stride(0) % 2 == 0 &&
                reinterpret_cast<uintptr_t>(table.data_ptr()) % 2 == 0,
                "table must be aligned uint8 [pages,page_size*68] on q device");
    TORCH_CHECK(slots.size(1) <= INT_MAX && batch <= 65535, "shape exceeds launch limits");
    // Visibility and valid slot addresses stay on device, including graph replay.
    // Caller guarantees 0 <= slots[b,j] < pages*page_size for j < min(lens[b],L).
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q.device());
    hipDeviceProp_t props;
    int device;
    TORCH_CHECK(hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&props, device) == hipSuccess,
                "cannot query GPU");
    const std::string arch(props.gcnArchName);
    TORCH_CHECK(arch.rfind("gfx938", 0) == 0 || arch.rfind("gfx936", 0) == 0 || arch.rfind("gfx928", 0) == 0,
                "FP4 MMAC requires gfx928/gfx936/gfx938");
    auto out = output.has_value() ? *output : torch::empty({batch, slots.size(1)}, q.options().dtype(at::kFloat));
    TORCH_CHECK(out.device() == q.device() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                out.size(0) == batch && out.size(1) == slots.size(1) && out.is_contiguous(),
                "out must be contiguous FP32 [B,L] on q device");
    TORCH_CHECK(!out.is_alias_of(q) && !out.is_alias_of(weights) && !out.is_alias_of(slots) &&
                !out.is_alias_of(lens) && !out.is_alias_of(table), "out must not alias inputs");
    if (batch == 0 || slots.size(1) == 0) return out;
    const int length = slots.size(1);
    auto stream = at::hip::getCurrentHIPStream();
    const int tiles_per_wave = batch * ((slots.size(1) + 255) / 256) >= 128 ? 4 : 1;
    const int tokens_per_cta = 64 * tiles_per_wave;
    const int page_shift = (page_size & (page_size - 1)) == 0 ? __builtin_ctzll(page_size) : -1;
    const dim3 grid((slots.size(1) + tokens_per_cta - 1) / tokens_per_cta, batch);
    // Measured crossover on BW1101: prepacking wins for H32 and >=48K total
    // scores. Smaller workloads avoid the extra pack launch.
    const bool fp8_workload = heads == 32 && batch * slots.size(1) >= 49152;
    const bool use_fp8 = fp8_workload && arch.rfind("gfx938", 0) == 0 &&
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
        reinterpret_cast<uintptr_t>(table.data_ptr()) % 4 == 0 && table.stride(0) % 4 == 0;
    torch::Tensor packed_q, metadata;
    if (use_fp8) {
        packed_q = torch::empty({batch, heads, 128}, q.options().dtype(at::kByte));
        metadata = torch::empty({batch, 2}, q.options().dtype(at::kInt));
        pack_fp4_query<32><<<batch, 64, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(q.data_ptr()), packed_q.data_ptr<uint8_t>(),
            metadata.data_ptr<int>(), q.stride(0), q.stride(1));
    }
#define FP4_ARGS \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint16_t*>(weights.data_ptr()), \
        slots.data_ptr<int64_t>(), lens.data_ptr<int64_t>(), table.data_ptr<uint8_t>(), out.data_ptr<float>(), \
        length, page_size, page_shift, table.stride(0), q.stride(0), q.stride(1), weights.stride(0)
    // The public FP8 fast path is H32-only. Keep the static page-64 variant
    // there so other supported heads retain their compact generic code.
    if (use_fp8 && heads == 32 && page_size == 64) {
        if (tiles_per_wave == 4) {
            fp4_fp8_logits_kernel<32, 4, 6><<<grid, 256, 0, stream>>>(
                FP4_ARGS, packed_q.data_ptr<uint8_t>(), metadata.data_ptr<int>());
        } else {
            fp4_fp8_logits_kernel<32, 1, 6><<<grid, 256, 0, stream>>>(
                FP4_ARGS, packed_q.data_ptr<uint8_t>(), metadata.data_ptr<int>());
        }
    } else {
#define FP4_KERNEL(H, T) \
    if (use_fp8) { fp4_fp8_logits_kernel<H, T><<<grid, 256, 0, stream>>>(FP4_ARGS, packed_q.data_ptr<uint8_t>(), metadata.data_ptr<int>()); } \
    else { fp4_logits_kernel<H, T><<<grid, 256, 0, stream>>>(FP4_ARGS); }
#define FP4_LAUNCH(H) \
    if (tiles_per_wave == 4) { FP4_KERNEL(H, 4); } else { FP4_KERNEL(H, 1); }
        switch (heads) {
            case 8: FP4_LAUNCH(8); break;
            case 16: FP4_LAUNCH(16); break;
            case 32: FP4_LAUNCH(32); break;
            case 64: FP4_LAUNCH(64); break;
        }
#undef FP4_LAUNCH
#undef FP4_KERNEL
    }
#undef FP4_ARGS
    TORCH_CHECK(hipGetLastError() == hipSuccess, "FP4 logits launch failed");
    return out;
}

}
