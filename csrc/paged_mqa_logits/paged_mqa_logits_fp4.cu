// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Paged MXFP4 MQA logits for the DSV4.1 low-ratio indexer, DeepGEMM's
// fp8_fp4_paged_mqa_logits contract on HCU:
//
//   logits[b, t] = sum_h w[b, h] * relu(sum_c q[b, h, c] * k[slot(b, t), c])
//   slot(b, t)   = block_table[b, t / TB] * TB + t % TB
//
// q is E2M1 [B, H, 64 bytes] with per-32 UE8M0 scales packed in an int32 per
// head, k is the fused index-K pool page [64 x 64 payload bytes][64 x 4 scale
// bytes] with an arbitrary page stride. TB = 64 walks the dense block table of
// the index pool; TB = 8 walks a candidate block table (pool slot / 8 per
// entry), which makes the same kernel DeepGEMM's sparse variant.
//
// Numerics: every per-32 block scale is folded into its operand exactly, all
// 128 products of a head enter one FP32 accumulator, and relu, the head weight
// and the head reduction stay in FP32 (DeepGEMM's block-scaled MMA contract).
//
// gfx938 has no FP4 MMAC. E2M1 codes are m * 2^e with e in [-1, 2]; shifting
// the E4M3 exponent field by a per-block delta in [-6, 5] encodes a block's
// value times 2^(scale - pivot) exactly. With one pivot per query row and one
// per KV token, the residual 2^(pq + pk - 254) is common to every term of that
// (row, token) pair and is applied once to the finished logit -- exact, and
// independent of accumulation order. Rows or tiles whose scales span more than
// 11 steps take an exact BF16 MMAC path instead (BF16 holds any E2M1 * 2^s).
//
// BF16 decode (every tile on gfx936/928, the fallback on gfx938) dominates the
// kernel there, so it is table driven: per tile, the eight E2M1 magnitudes times
// the lane's block scale become two v_perm byte tables, and each payload byte
// (two codes) costs four v_perm and one or3. Scales outside [2, 252], or a tile
// whose products would leave the FP32 normal range (BF16 MMAC flushes or coarsely
// rounds subnormal results), take an exact per-element path, scaling K up by a
// power of two that is undone on the logit; a quantizer never emits those.
//
// Work split: one CTA = 4 waves = one 256-position split (DeepGEMM's split_kv),
// one wave = 4 tiles of 16 positions. Each lane owns one position and one
// 32-channel scale group, so a tile is a single 16-byte load per lane (the
// whole 1 KB of payload, coalesced) plus one 4-byte scale word; Q, its scales
// and the head weights stay in registers across the CTA's splits. Channels are
// permuted identically in Q and K so each lane's MMAC fragments come from its
// own scale group. Persistent CTAs follow LightOp's schedule metadata; without
// one, a (splits, rows) grid is launched. Positions at or past a row's context
// length are not written (clean_logits = false).
//
// Row pairing (verify): with rows_per_request = n > 1, rows [b n, b n + n) are
// one request's draft tokens, sharing a block-table row and differing only in
// context_lens. The kernel then walks ROUTES of up to two rows of one request,
// ceil(n / 2) per request, and scores both rows of a route in one KV pass: a
// tile is loaded and decoded once and multiplied into both rows' accumulators.
// The schedule, if any, is planned on the route lengths (the longer row's);
// each row still stops writing at its own length. Pairing needs H <= 32 (two
// rows' Q in VGPRs) and is skipped below kPagedMqaLogitsFp4PairMinRequests.
#include "hcu_mmac.h"
#include "paged_mqa_logits_fp4.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <climits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace lightop::deep_gemm {
namespace {
using bf16x4 = short __attribute__((ext_vector_type(4)));
using floatx4 = float __attribute__((ext_vector_type(4)));
using intx2 = int __attribute__((ext_vector_type(2)));
using uintx4 = unsigned __attribute__((ext_vector_type(4)));

constexpr int kSplitKV = 256;
constexpr int kWaves = 4;
constexpr int kTile = 16;
constexpr int kTilesPerWave = kSplitKV / kWaves / kTile;
constexpr int kPageSlots = 64;
constexpr int kPayloadBytes = 64;
constexpr int kFp4PlannerThreads = 256;

// The legacy LightOp planner uses a logical persistent-CTA count rather than
// the physical CU count exposed by HIP.  Keep the same mapping for the new
// planner so its [ctas + 1, 2] metadata has the shape consumed by the kernel.
struct Fp4DeviceInfo {
    int physical_cus;
    int logical_sms;
};

const Fp4DeviceInfo& get_fp4_device_info() {
    static std::once_flag once;
    static Fp4DeviceInfo info{};
    std::call_once(once, [] {
        int device_id = 0;
        TORCH_CHECK(hipGetDevice(&device_id) == hipSuccess,
                    "get_paged_mqa_logits_fp4_metadata: failed to get HIP device");
        hipDeviceProp_t prop;
        TORCH_CHECK(hipGetDeviceProperties(&prop, device_id) == hipSuccess,
                    "get_paged_mqa_logits_fp4_metadata: failed to get HIP properties");
        const std::string arch(prop.gcnArchName);
        const int cus = prop.multiProcessorCount;
        info.physical_cus = cus;
        if (arch.rfind("gfx936", 0) == 0 && cus == 80) {
            info.logical_sms = 320;
        } else if (arch.rfind("gfx938", 0) == 0 && cus == 64) {
            info.logical_sms = 128;
        } else if (arch.rfind("gfx938", 0) == 0 && cus == 72) {
            info.logical_sms = 144;
        } else if ((arch.rfind("gfx928", 0) == 0 || arch.rfind("gfx92a", 0) == 0) && cus == 120) {
            info.logical_sms = 240;
        } else {
            TORCH_CHECK(false, "unsupported device config: ", arch,
                        ", CU=", cus);
        }
    });
    return info;
}

__device__ __forceinline__ int fp4_route_split_count(
    const int* context_lens, int route, int batch, int rows_per_request,
    bool pair) {
    int first = route;
    int count = 1;
    if (pair) {
        const int per_request = (rows_per_request + 1) / 2;
        const int pair_idx = route % per_request;
        first = (route / per_request) * rows_per_request + 2 * pair_idx;
        count = min(2, rows_per_request - 2 * pair_idx);
    }
    int length = context_lens[first];
    if (pair && count == 2 && first + 1 < batch)
        length = max(length, context_lens[first + 1]);
    return (length + kSplitKV - 1) / kSplitKV;
}

// One-block FP4 planner.  The legacy planner uses a serial prefix scan and a
// compile-time aligned-batch dispatch table.  This kernel keeps the prefix in
// LDS, performs a work-efficient scan, and binary-searches the scan for each
// persistent CTA start.  The route-length calculation is fused here so paired
// verification does not launch reshape/pad/amax helper kernels on the host.
__global__ void fp4_paged_logits_metadata(
    const int* context_lens, int batch, int routes, int rows_per_request,
    bool pair, int num_sms, int* schedule_metadata) {
    extern __shared__ uint32_t prefix[];
    const int tid = static_cast<int>(threadIdx.x);
    int scan_size = 1;
    while (scan_size < routes) scan_size <<= 1;

    for (int i = tid; i < scan_size; i += blockDim.x) {
        prefix[i] = i < routes
            ? static_cast<uint32_t>(fp4_route_split_count(
                  context_lens, i, batch, rows_per_request, pair))
            : 0;
    }
    __syncthreads();

    for (int offset = 1; offset < scan_size; offset <<= 1) {
        for (int i = (tid + 1) * (offset << 1) - 1;
             i < scan_size; i += blockDim.x * (offset << 1))
            prefix[i] += prefix[i - offset];
        __syncthreads();
    }

    if (tid == 0)
        prefix[scan_size - 1] = 0;
    __syncthreads();

    for (int offset = scan_size >> 1; offset > 0; offset >>= 1) {
        for (int i = (tid + 1) * (offset << 1) - 1;
             i < scan_size; i += blockDim.x * (offset << 1)) {
            const uint32_t left = prefix[i - offset];
            prefix[i - offset] = prefix[i];
            prefix[i] += left;
        }
        __syncthreads();
    }

    // Read the exclusive prefix at the last real route, then include that
    // route.  This works with or without zero padding to a power of two.
    uint32_t total_splits = prefix[routes - 1] +
        static_cast<uint32_t>(fp4_route_split_count(
            context_lens, routes - 1, batch, rows_per_request, pair));
    const uint32_t q = total_splits / static_cast<uint32_t>(num_sms);
    const uint32_t r = total_splits % static_cast<uint32_t>(num_sms);
    for (int sm_idx = tid; sm_idx <= num_sms; sm_idx += blockDim.x) {
        const uint32_t target = static_cast<uint32_t>(sm_idx) * q +
            min(static_cast<uint32_t>(sm_idx), r);
        int lo = 0, hi = routes;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            const uint32_t end = mid + 1 < routes
                ? prefix[mid + 1] : total_splits;
            if (end <= target)
                lo = mid + 1;
            else
                hi = mid;
        }
        // The final entry is the exclusive sentinel.  When target == total,
        // binary search returns lo == routes; its split offset must be zero,
        // rather than the size of the last route.
        const uint32_t previous = lo == routes ? total_splits : prefix[lo];
        schedule_metadata[sm_idx * 2] = lo;
        schedule_metadata[sm_idx * 2 + 1] = static_cast<int>(target - previous);
    }
}

__device__ __forceinline__ uint16_t to_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ float from_bf16(uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}
__device__ __forceinline__ float pow2(int e) {  // 2^(e - 127), e in [1, 254]
    return __uint_as_float(static_cast<unsigned>(e) << 23);
}

// Out of the operator's domain, scale 0xFF (NaN in MX) and products past the BF16
// range decode to zero, so no MMAC ever sees inf or NaN and every arch agrees.
__device__ __forceinline__ uint16_t decode_fp4(unsigned code, unsigned scale) {
    const unsigned mag = code & 7u;
    const unsigned sign = (code & 8u) << 12;
    if (mag == 0) return sign;
    if (scale == 255) return 0;
    const unsigned base = mag >= 2 ? 0x3f00u + (mag << 6) : 0x3f00u;
    const int exponent = static_cast<int>(base >> 7) + static_cast<int>(scale) - 127;
    if (exponent >= 255) return 0;
    if (exponent > 0) return sign | (exponent << 7) | (base & 0x7fu);
    // Build the BF16 subnormal in integers: E8M0 scales 0/1 keep the exact
    // round-to-nearest-even bits even where the FP32 path flushes subnormals.
    const unsigned mant = 0x80u | (base & 0x7fu);
    const int shift = static_cast<int>(base >> 7) + static_cast<int>(scale) - 128;
    unsigned frac;
    if (shift >= 0) {
        frac = mant << shift;
    } else {
        const int rshift = -shift;
        const unsigned q = mant >> rshift;
        const unsigned rem = mant & ((1u << rshift) - 1u);
        const unsigned halfway = 1u << (rshift - 1);
        frac = q + (rem > halfway || (rem == halfway && (q & 1u)));
    }
    return sign | (frac & 0x7fu);
}

// Out of line: scales outside [2, 252] never come from a quantizer, and inlining
// this per element would quadruple the kernel's code.
__device__ __attribute__((noinline)) bf16x4 decode_fp4x4_edge(unsigned packed, unsigned scale) {
    bf16x4 result;
#pragma unroll
    for (int j = 0; j < 4; ++j) result[j] = decode_fp4((packed >> (j * 4)) & 15u, scale);
    return result;
}

// Four E2M1 codes (low nibble first) times 2^(scale - 127) as exact BF16.
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
        result = decode_fp4x4_edge(packed, scale);
    }
    return result;
}

// BF16 of each E2M1 magnitude m times 2^(scale - 127), scale in [2, 252], as
// v_perm tables: byte m of lo/hi is the low/high byte of entry m, entry 0 zero.
struct Bf16Lut {
    unsigned lo0, lo1, hi0, hi1;
};
__device__ __forceinline__ Bf16Lut bf16_lut(int scale) {
    // Every entry stays in (0, 0x8000), so one 32-bit add updates both halves.
    const int d = (scale - 127) * 128;
    const unsigned p0 = 0x3f000000u + (static_cast<unsigned>(d) << 16);  // m = 0, 1
    const unsigned dd = static_cast<unsigned>(d) * 0x10001u;
    const unsigned p1 = 0x3fc03f80u + dd, p2 = 0x40404000u + dd, p3 = 0x40c04080u + dd;
    return {__builtin_amdgcn_perm(p1, p0, 0x06040200u), __builtin_amdgcn_perm(p3, p2, 0x06040200u),
            __builtin_amdgcn_perm(p1, p0, 0x07050301u), __builtin_amdgcn_perm(p3, p2, 0x07050301u)};
}

// The 16 bytes of `packed` (32 E2M1 codes, low nibble first) as eight bf16x4,
// bit for bit what decode_fp4x4 gives for the table's scale. Output dword i
// holds the two codes of input byte i: v_perm selects their magnitudes into
// a table index, and the signs land on bits 15 and 31.
__device__ __forceinline__ void decode_fp4x32(const uintx4& packed, const Bf16Lut& lut, bf16x4 (&out)[8]) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const unsigned x = packed[w];
        const unsigned lm = x & 0x07070707u, hm = (x >> 4) & 0x07070707u;
        const unsigned ls = (x << 4) & 0x80808080u, hs = x & 0x80808080u;
        unsigned d[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // [lm_i, 0, hm_i, 0]; index 0 reads the zero entry of either table.
            const unsigned sel = __builtin_amdgcn_perm(hm, lm, 0x0c040c00u + i * 0x00010001u);
            const unsigned lo = __builtin_amdgcn_perm(lut.lo1, lut.lo0, sel);
            const unsigned hi = __builtin_amdgcn_perm(lut.hi1, lut.hi0, sel << 8);
            const unsigned sign = __builtin_amdgcn_perm(hs, ls, 0x040c000cu + i * 0x01000100u);
            d[i] = lo | hi | sign;
        }
        out[2 * w] = __builtin_bit_cast(bf16x4, uint2{d[0], d[1]});
        out[2 * w + 1] = __builtin_bit_cast(bf16x4, uint2{d[2], d[3]});
    }
}

// Eight E2M1 codes (low nibble first) times 2^-delta as exact E4M3, delta in
// [-6, 5]: at 5 the smallest byte is 0x08 (the minimum normal), at -6 the
// largest is 0x7c (below the 0x7f NaN). Byte 0 of `lo` stays zero so code 0
// decodes to zero for any delta.
__device__ __forceinline__ intx2 fp4_to_fp8(unsigned packed, int delta) {
    const unsigned hi = 0x4c484440u - static_cast<unsigned>(delta) * 0x08080808u;
    const unsigned lo = 0x3c383000u - static_cast<unsigned>(delta) * 0x08080800u;
    const unsigned even = __builtin_amdgcn_perm(hi, lo, packed & 0x07070707u)
                        | ((packed & 0x08080808u) << 4);
    const unsigned odd = __builtin_amdgcn_perm(hi, lo, (packed >> 4) & 0x07070707u)
                       | (packed & 0x80808080u);
    return {static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x05010400u)),
            static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
}

__device__ __forceinline__ floatx4 mmac_bf16(bf16x4 a, bf16x4 b, floatx4 c) {
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false);
#elif defined(__gfx936__) || (defined(__gfx928__) || defined(__gfx92a__))
    return HCU_MMAC_F32_BF16(a, b, c);
#else
    return c;
#endif
}

__device__ __forceinline__ int wave_max(int x) {
#pragma unroll
    for (int d = 1; d < 64; d *= 2) x = max(x, __shfl_xor(x, d, 64));
    return x;
}
__device__ __forceinline__ int wave_min(int x) {
#pragma unroll
    for (int d = 1; d < 64; d *= 2) x = min(x, __shfl_xor(x, d, 64));
    return x;
}

// One query row in MMAC fragments. Lane (row = head within a tile of 16,
// group = scale group g) holds channels [32 g, 32 g + 32) of its head; FP8
// step k uses channels 32 g + 8 k .. + 8, BF16 step k channels 32 g + 4 k .. + 4.
// acc[h][j] of an MMAC holds head h * 16 + j * 4 + group of the lane's position.
template<int H>
struct RowQ {
    static constexpr int HT = H / 16;
    bf16x4 bf[HT][8];
    float w[HT][4];
    // The row's smallest and largest Q scale, which bound the BF16 subnormal shift.
    int smin, smax;
#if defined(__gfx938__)
    // FP8 fragments exist only where the FP8 MMAC does; elsewhere they would
    // cost the fragments' VGPRs for nothing.
    intx2 f8[HT][4];
    int pq;
    bool fp8_ok;
#endif

    __device__ __forceinline__ void load(const uint8_t* __restrict__ q_fp4, const int* __restrict__ q_sf,
                                         const float* __restrict__ weights, int64_t w_stride, int r,
                                         int row, int group) {
        uintx4 qb[HT];
        int sq[HT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int head = h * 16 + row;
            qb[h] = *reinterpret_cast<const uintx4*>(
                q_fp4 + (static_cast<int64_t>(r) * H + head) * kPayloadBytes + group * 16);
            sq[h] = (static_cast<unsigned>(q_sf[static_cast<int64_t>(r) * H + head]) >> (group * 8)) & 255u;
#pragma unroll
            for (int j = 0; j < 4; ++j) w[h][j] = weights[r * w_stride + h * 16 + j * 4 + group];
        }
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int k = 0; k < 8; ++k) bf[h][k] = decode_fp4x4((qb[h][k / 2] >> ((k & 1) * 16)) & 0xffffu, sq[h]);
        }
        smax = 0;
        smin = 255;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            smax = max(smax, sq[h]);
            smin = min(smin, sq[h]);
        }
        smax = wave_max(smax);
        smin = wave_min(smin);
#if defined(__gfx938__)
        pq = min(smax, smin + 5);
        fp8_ok = smax - smin <= 11;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int k = 0; k < 4; ++k) f8[h][k] = fp4_to_fp8(qb[h][k], pq - sq[h]);
        }
#endif
    }
};

template<int HT>
__device__ __forceinline__ float head_sum(const floatx4* acc, const float (*w)[4]) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int h = 0; h < HT; ++h) {
        s0 = fmaf(fmaxf(acc[h][0], 0.0f), w[h][0], s0);
        s1 = fmaf(fmaxf(acc[h][1], 0.0f), w[h][1], s1);
        s0 = fmaf(fmaxf(acc[h][2], 0.0f), w[h][2], s0);
        s1 = fmaf(fmaxf(acc[h][3], 0.0f), w[h][3], s1);
    }
    float s = s0 + s1;
    s += __shfl_xor(s, 16, 64);
    s += __shfl_xor(s, 32, 64);
    return s;
}

struct Args {
    const uint8_t* q_fp4;
    const int* q_sf;
    const uint8_t* kv;
    const float* weights;
    const int* context_lens;
    const int* block_table;
    const int* schedule;
    float* out;
    int64_t kv_page_stride;
    int64_t w_stride;
    int64_t bt_stride;
    int64_t out_stride;
    int routes;
    int rows_per_request;
    int table_blocks;
    int max_context_len;
};

// Scores split `split` of the Rows consecutive rows from `first`, which share the
// block-table row of `first`, for this wave's 64 positions. Each tile is loaded
// and decoded once for all of them; row i stops writing at visible[i].
template<int H, int TB, int Rows>
__device__ __forceinline__ void score_split(const Args& a, const RowQ<H>* q, int first, int split,
                                            const int* visible, int wave, int row, int group) {
    constexpr int HT = H / 16;
    const int base = split * kSplitKV + wave * (kTile * kTilesPerWave);
    int reach = visible[0];
    if constexpr (Rows == 2) reach = max(reach, visible[1]);
    if (base >= reach) return;
    const int* bt = a.block_table + first * a.bt_stride;

    // Issue every load of the wave's four tiles before any math, the scales
    // first: the shift test below waits on them, but not on the payloads.
    uintx4 kb[kTilesPerWave];
    unsigned sw[kTilesPerWave];
    const uint8_t* payload[kTilesPerWave];
#pragma unroll
    for (int tile = 0; tile < kTilesPerWave; ++tile) {
        const int t = base + tile * kTile + row;
        sw[tile] = 0x7f7f7f7fu;
        payload[tile] = nullptr;
        if (t < reach) {
            const int64_t slot = static_cast<int64_t>(bt[t / TB]) * TB + t % TB;
            const uint8_t* page = a.kv + (slot / kPageSlots) * a.kv_page_stride;
            const int off = static_cast<int>(slot % kPageSlots);
            payload[tile] = page + off * kPayloadBytes + group * 16;
            sw[tile] = *reinterpret_cast<const unsigned*>(page + kPageSlots * kPayloadBytes + off * 4);
        }
    }
#pragma unroll
    for (int tile = 0; tile < kTilesPerWave; ++tile) {
        kb[tile] = uintx4{0, 0, 0, 0};
        if (payload[tile]) kb[tile] = *reinterpret_cast<const uintx4*>(payload[tile]);
    }
    // Subnormal BF16 MMAC results are flushed on gfx936 and rounded to a fixed
    // 2^-149 step on gfx938. A nonzero product is a multiple of 2^(sq + sk - 256),
    // so decoding K 2^shift larger with sq_min + k_min + shift >= 130 keeps every
    // partial sum normal; the logit is scaled back after head_sum, where relu and
    // the weights commute with the power of two. shift is capped so K stays
    // decodable and the accumulator cannot overflow.
    int qmin = q[0].smin, qmax = q[0].smax;
    if constexpr (Rows == 2) {
        qmin = min(qmin, q[1].smin);
        qmax = max(qmax, q[1].smax);
    }
    // Scales a quantizer emits need no shift and sit in [2, 252], where the
    // table decode is exact. One wave-uniform test per split -- some K scale
    // byte below max(130 - qmin, 2) or above 252, by the exact SWAR byte compare
    // (bound <= 128) -- picks between two copies of the tile loop.
    const int need = 130 - qmin;
    const unsigned bound = static_cast<unsigned>(max(need, 2));
    unsigned edge = 0;
#pragma unroll
    for (int tile = 0; tile < kTilesPerWave; ++tile) {
        const unsigned x = sw[tile];
        edge |= ((x - 0x01010101u * bound) & ~x) | ((~x - 0x03030303u) & x);
    }
    const bool exact = need > 128 || __ballot((edge & 0x80808080u) != 0) != 0;

    auto score_tiles = [&](auto exact_path) {
    constexpr bool Shift = decltype(exact_path)::value;
#pragma unroll
    for (int tile = 0; tile < kTilesPerWave; ++tile) {
        const int start = base + tile * kTile;
        // Wave-uniform: the tail of a row's last split skips its empty tiles.
        if (start >= reach) break;
        const int t = start + row;
        const int sk = (sw[tile] >> (group * 8)) & 255u;
        int shift = 0;
        if constexpr (Shift) {
            const int s0 = sw[tile] & 255u, s1 = (sw[tile] >> 8) & 255u;
            const int s2 = (sw[tile] >> 16) & 255u, s3 = sw[tile] >> 24;
            const int kmax = max(max(s0, s1), max(s2, s3));
            const int kmin = min(min(s0, s1), min(s2, s3));
            shift = max(0, min(need - kmin, min(252 - kmax, 368 - qmax - kmax)));
        }
        bool need_bf16 = true;
#if defined(__gfx938__)
        const int s0 = sw[tile] & 255u, s1 = (sw[tile] >> 8) & 255u;
        const int s2 = (sw[tile] >> 16) & 255u, s3 = sw[tile] >> 24;
        const int kmax = max(max(s0, s1), max(s2, s3));
        const int kmin = min(min(s0, s1), min(s2, s3));
        const int pk = min(kmax, kmin + 5);
        int e[Rows];
        bool fp8[Rows], need_fp8 = false;
        need_bf16 = false;
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            e[i] = q[i].pq + pk - 127;
            // Only the row's own positions vote, so pairing never changes a row's path.
            const bool ok = t >= visible[i] || (kmax - kmin <= 11 && e[i] >= 1 && e[i] <= 254);
            fp8[i] = q[i].fp8_ok && __ballot(!ok) == 0;
            // Wave-uniform: decode the tile once into each format a live row needs.
            if (start < visible[i]) {
                need_fp8 |= fp8[i];
                need_bf16 |= !fp8[i];
            }
        }
        intx2 k8[4];
        if (need_fp8) {
#pragma unroll
            for (int k = 0; k < 4; ++k) k8[k] = fp4_to_fp8(kb[tile][k], pk - sk);
        }
#endif
        bf16x4 kbf[8];
        if (need_bf16) {
            if constexpr (Shift) {
#pragma unroll
                for (int k = 0; k < 8; ++k) kbf[k] = decode_fp4x4((kb[tile][k / 2] >> ((k & 1) * 16)) & 0xffffu, sk + shift);
            } else {
                decode_fp4x32(kb[tile], bf16_lut(sk), kbf);
            }
        }
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            if (start >= visible[i]) continue;
            floatx4 acc[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h) acc[h] = floatx4{0.0f, 0.0f, 0.0f, 0.0f};
            float logit;
#if defined(__gfx938__)
            if (fp8[i]) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
#pragma unroll
                    for (int h = 0; h < HT; ++h)
                        acc[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(k8[k], q[i].f8[h][k], acc[h], false, false);
                }
                // relu and the weights commute with the positive power of two.
                logit = head_sum<HT>(acc, q[i].w) * pow2(e[i]);
            } else
#endif
            {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
#pragma unroll
                    for (int h = 0; h < HT; ++h) acc[h] = mmac_bf16(kbf[k], q[i].bf[h][k], acc[h]);
                }
                logit = head_sum<HT>(acc, q[i].w);
                if constexpr (Shift) logit = ldexpf(logit, -shift);
            }
            if (group == 0 && t < visible[i]) a.out[(first + i) * a.out_stride + t] = logit;
        }
    }
    };
    if (exact) {
        score_tiles(std::true_type{});
    } else {
        score_tiles(std::false_type{});
    }
}

// First row of route `r`, and its row count: one row per route unpaired, else
// up to two rows of one request, ceil(n / 2) routes per request of n rows.
template<bool Pair>
__device__ __forceinline__ int route_rows(const Args& a, int r, int& count) {
    if constexpr (!Pair) {
        count = 1;
        return r;
    } else {
        const int n = a.rows_per_request, per = (n + 1) / 2, j = r % per;
        count = min(2, n - 2 * j);
        return (r / per) * n + 2 * j;
    }
}

template<int TB>
__device__ __forceinline__ int visible_len(const Args& a, int row) {
    return min(min(a.context_lens[row], a.max_context_len), a.table_blocks * TB);
}

// Q rows a CTA keeps in VGPRs; two H = 64 rows would halve occupancy.
template<int H, bool Pair>
constexpr int kQRows = Pair ? 2 : 1;

// Scores splits [begin, stop) of route (first, count). `loaded` is the first row
// held in q, so Q is reloaded only when the route changes.
template<int H, int TB, bool Pair>
__device__ __forceinline__ void score_route(const Args& a, RowQ<H> (&q)[kQRows<H, Pair>], int& loaded,
                                            int first, int count, int begin, int stop,
                                            const int (&visible)[2], int wave, int row, int group) {
    static_assert(!Pair || H <= 32, "paired rows need H <= 32");
    if (loaded != first) {
        q[0].load(a.q_fp4, a.q_sf, a.weights, a.w_stride, first, row, group);
        if constexpr (Pair) {
            if (count == 2) q[1].load(a.q_fp4, a.q_sf, a.weights, a.w_stride, first + 1, row, group);
        }
        loaded = first;
    }
    for (int split = begin; split < stop; ++split) {
        if constexpr (Pair) {
            if (count == 2) {
                score_split<H, TB, 2>(a, q, first, split, visible, wave, row, group);
                continue;
            }
        }
        score_split<H, TB, 1>(a, q, first, split, visible, wave, row, group);
    }
}

template<int H, int TB, bool Pair>
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fp4_paged_logits_persistent(const Args a) {
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int row = lane % 16, group = lane / 16;
    int r = a.schedule[blockIdx.x * 2], split = a.schedule[blockIdx.x * 2 + 1];
    const int end_r = a.schedule[blockIdx.x * 2 + 2], end_split = a.schedule[blockIdx.x * 2 + 3];
    RowQ<H> q[kQRows<H, Pair>];
    int loaded = -1;
    for (; r < a.routes && r <= end_r; ++r, split = 0) {
        int count;
        const int first = route_rows<Pair>(a, r, count);
        const int visible[2] = {visible_len<TB>(a, first), count == 2 ? visible_len<TB>(a, first + 1) : 0};
        // This CTA owns splits [split, stop) of the route.
        const int splits = (max(visible[0], visible[1]) + kSplitKV - 1) / kSplitKV;
        const int stop = r == end_r ? min(end_split, splits) : splits;
        if (split < stop) score_route<H, TB, Pair>(a, q, loaded, first, count, split, stop, visible, wave, row, group);
    }
}

template<int H, int TB, bool Pair>
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fp4_paged_logits_grid(const Args a) {
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int row = lane % 16, group = lane / 16;
    const int split = blockIdx.x;
    int count;
    const int first = route_rows<Pair>(a, blockIdx.y, count);
    const int visible[2] = {visible_len<TB>(a, first), count == 2 ? visible_len<TB>(a, first + 1) : 0};
    if (split * kSplitKV >= max(visible[0], visible[1])) return;
    RowQ<H> q[kQRows<H, Pair>];
    int loaded = -1;
    score_route<H, TB, Pair>(a, q, loaded, first, count, split, split + 1, visible, wave, row, group);
}
}  // namespace

torch::Tensor paged_mqa_logits_fp4(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& block_table,
    const std::optional<torch::Tensor>& schedule_meta, int64_t max_context_len,
    int64_t table_block_size, std::optional<torch::Tensor> output, int64_t rows_per_request) {
    TORCH_CHECK(q.is_cuda(), "paged_mqa_logits_fp4: q must be a CUDA/HIP tensor");
    TORCH_CHECK((q.dim() == 4 && q.size(1) == 1) || q.dim() == 3,
                "paged_mqa_logits_fp4: q must be [B,1,H,64] or [B,H,64] packed E2M1");
    TORCH_CHECK(q.scalar_type() == at::kChar || q.scalar_type() == at::kByte,
                "paged_mqa_logits_fp4: q must be int8/uint8 packed E2M1");
    TORCH_CHECK(q.is_contiguous() && q.size(-1) == 64, "paged_mqa_logits_fp4: q must be contiguous with 64 bytes per head");
    const int64_t batch = q.size(0), heads = q.size(-2);
    TORCH_CHECK(heads == 16 || heads == 32 || heads == 64, "paged_mqa_logits_fp4: heads must be 16, 32 or 64, got ", heads);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "paged_mqa_logits_fp4: q must be 16-byte aligned");
    TORCH_CHECK(q_sf.device() == q.device() && q_sf.scalar_type() == at::kInt && q_sf.is_contiguous() &&
                q_sf.numel() == batch * heads, "paged_mqa_logits_fp4: q_sf must be contiguous int32 [B,1,H]");
    TORCH_CHECK(kv_cache.device() == q.device() && kv_cache.scalar_type() == at::kByte,
                "paged_mqa_logits_fp4: kv_cache must be uint8 on q device");
    if (kv_cache.dim() == 4) {
        TORCH_CHECK(kv_cache.size(1) == kPageSlots && kv_cache.size(2) == 1 && kv_cache.size(3) == 68 &&
                    kv_cache.stride(1) == 68 && kv_cache.stride(3) == 1,
                    "paged_mqa_logits_fp4: kv_cache must be [pages,64,1,68] with packed pages");
    } else {
        TORCH_CHECK(kv_cache.dim() == 2 && kv_cache.size(1) == kPageSlots * 68 && kv_cache.stride(1) == 1,
                    "paged_mqa_logits_fp4: kv_cache must be [pages,64,1,68] or [pages,4352]");
    }
    const int64_t kv_page_stride = kv_cache.stride(0);
    TORCH_CHECK(kv_page_stride % 16 == 0 && reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % 16 == 0,
                "paged_mqa_logits_fp4: kv_cache pages must be 16-byte aligned");
    TORCH_CHECK(weights.device() == q.device() && weights.scalar_type() == at::kFloat && weights.dim() == 2 &&
                weights.size(0) == batch && weights.size(1) == heads && weights.stride(1) == 1,
                "paged_mqa_logits_fp4: weights must be FP32 [B,H] with contiguous heads");
    TORCH_CHECK(context_lens.device() == q.device() && context_lens.scalar_type() == at::kInt &&
                context_lens.is_contiguous() && context_lens.numel() == batch,
                "paged_mqa_logits_fp4: context_lens must be contiguous int32 [B]");
    TORCH_CHECK(block_table.device() == q.device() && block_table.scalar_type() == at::kInt &&
                block_table.dim() == 2 && block_table.size(0) == batch && block_table.stride(1) == 1,
                "paged_mqa_logits_fp4: block_table must be int32 [B,blocks] with contiguous blocks");
    TORCH_CHECK(table_block_size == 8 || table_block_size == 64,
                "paged_mqa_logits_fp4: table_block_size must be 8 or 64");
    TORCH_CHECK(max_context_len >= 0 && max_context_len <= INT_MAX - kSplitKV,
                "paged_mqa_logits_fp4: invalid max_context_len");
    TORCH_CHECK(rows_per_request >= 1 && batch % rows_per_request == 0,
                "paged_mqa_logits_fp4: rows_per_request must divide batch, got ", rows_per_request);
    // Sparse (block 8) tables differ per row, so only the dense path pairs rows.
    TORCH_CHECK(rows_per_request == 1 || (table_block_size == 64 && heads <= 32),
                "paged_mqa_logits_fp4: rows_per_request > 1 needs table_block_size 64 and H <= 32");
    TORCH_CHECK(batch <= 65535, "paged_mqa_logits_fp4: batch exceeds launch limits");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q.device());

    torch::Tensor out;
    if (output.has_value()) {
        out = *output;
        TORCH_CHECK(out.device() == q.device() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                    out.size(0) == batch && out.size(1) >= max_context_len && out.stride(1) == 1,
                    "paged_mqa_logits_fp4: out must be FP32 [B,>=max_context_len] with contiguous columns");
    } else {
        // DeepGEMM aligns the row stride to the split and to 1 KB.
        const int64_t aligned = (max_context_len + kSplitKV - 1) / kSplitKV * kSplitKV;
        out = torch::empty({batch, aligned}, q.options().dtype(at::kFloat)).slice(1, 0, max_context_len);
    }
    if (batch == 0 || max_context_len == 0) return out;

    Args a;
    a.q_fp4 = reinterpret_cast<const uint8_t*>(q.data_ptr());
    a.q_sf = q_sf.data_ptr<int>();
    a.kv = kv_cache.data_ptr<uint8_t>();
    a.weights = weights.data_ptr<float>();
    a.context_lens = context_lens.data_ptr<int>();
    a.block_table = block_table.data_ptr<int>();
    a.schedule = nullptr;
    a.out = out.data_ptr<float>();
    a.kv_page_stride = kv_page_stride;
    a.w_stride = weights.stride(0);
    a.bt_stride = block_table.stride(0);
    a.out_stride = out.stride(0);
    const bool pair = rows_per_request > 1 && batch / rows_per_request >= kPagedMqaLogitsFp4PairMinRequests;
    const int64_t routes = pair ? batch / rows_per_request * ((rows_per_request + 1) / 2) : batch;
    a.routes = static_cast<int>(routes);
    a.rows_per_request = static_cast<int>(rows_per_request);
    a.table_blocks = static_cast<int>(block_table.size(1));
    a.max_context_len = static_cast<int>(max_context_len);

    auto stream = at::hip::getCurrentHIPStream();
    dim3 grid;
    bool persistent = false;
    if (schedule_meta.has_value()) {
        const auto& s = *schedule_meta;
        TORCH_CHECK(s.device() == q.device() && s.scalar_type() == at::kInt && s.dim() == 2 && s.size(1) == 2 &&
                    s.size(0) >= 2 && s.is_contiguous(),
                    "paged_mqa_logits_fp4: schedule_meta must be contiguous int32 [ctas+1,2]");
        a.schedule = s.data_ptr<int>();
        grid = dim3(static_cast<unsigned>(s.size(0) - 1));
        persistent = true;
    } else {
        grid = dim3(static_cast<unsigned>((max_context_len + kSplitKV - 1) / kSplitKV), static_cast<unsigned>(routes));
    }
#define FP4_LAUNCH(H, TB, Pair) do {                                                        \
    if (persistent) { fp4_paged_logits_persistent<H, TB, Pair><<<grid, 256, 0, stream>>>(a); } \
    else { fp4_paged_logits_grid<H, TB, Pair><<<grid, 256, 0, stream>>>(a); }                \
} while (0)
#define FP4_DISPATCH_TB(H) do {                                                         \
    if (table_block_size == 64) { FP4_LAUNCH(H, 64, false); } else { FP4_LAUNCH(H, 8, false); } \
} while (0)
    switch (heads) {
        case 16: if (pair) { FP4_LAUNCH(16, 64, true); } else { FP4_DISPATCH_TB(16); } break;
        case 32: if (pair) { FP4_LAUNCH(32, 64, true); } else { FP4_DISPATCH_TB(32); } break;
        case 64: FP4_DISPATCH_TB(64); break;
    }
#undef FP4_DISPATCH_TB
#undef FP4_LAUNCH
    TORCH_CHECK(hipGetLastError() == hipSuccess, "paged_mqa_logits_fp4: launch failed");
    return out;
}

torch::Tensor get_paged_mqa_logits_fp4_metadata(
    const torch::Tensor& context_lens, int64_t block_kv, int64_t num_sms,
    int64_t rows_per_request) {
    TORCH_CHECK(context_lens.is_cuda() && context_lens.dim() == 1 &&
                    context_lens.scalar_type() == at::kInt &&
                    context_lens.is_contiguous(),
                "get_paged_mqa_logits_fp4_metadata: context_lens must be a "
                "contiguous CUDA/HIP int32 vector");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(context_lens.device());
    TORCH_CHECK(block_kv == 64,
                "get_paged_mqa_logits_fp4_metadata: block_kv must be 64");
    TORCH_CHECK(num_sms > 0 && num_sms <= INT_MAX,
                "get_paged_mqa_logits_fp4_metadata: invalid num_sms");
    TORCH_CHECK(rows_per_request >= 1 && rows_per_request <= INT_MAX,
                "get_paged_mqa_logits_fp4_metadata: invalid rows_per_request");

    const int64_t batch64 = context_lens.numel();
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX,
                "get_paged_mqa_logits_fp4_metadata: invalid batch size");
    TORCH_CHECK(batch64 % rows_per_request == 0,
                "get_paged_mqa_logits_fp4_metadata: rows_per_request must "
                "divide batch");
    const bool pair = rows_per_request > 1 &&
        batch64 / rows_per_request >= kPagedMqaLogitsFp4PairMinRequests;
    const int64_t routes64 = pair
        ? batch64 / rows_per_request * ((rows_per_request + 1) / 2)
        : batch64;
    TORCH_CHECK(routes64 <= kPagedMqaLogitsFp4MaxScheduleRows,
                "get_paged_mqa_logits_fp4_metadata: route rows exceed the "
                "LDS planner limit");
    const int batch = static_cast<int>(batch64);
    const int routes = static_cast<int>(routes64);
    int scan_size = 1;
    while (scan_size < routes) scan_size <<= 1;

    const auto& device_info = get_fp4_device_info();
    TORCH_CHECK(num_sms == device_info.physical_cus,
                "get_paged_mqa_logits_fp4_metadata: num_sms must match the "
                "physical CU count (", device_info.physical_cus, ")");
    auto schedule = torch::empty({device_info.logical_sms + 1, 2}, context_lens.options());
    auto stream = at::hip::getCurrentHIPStream();
    fp4_paged_logits_metadata<<<1, kFp4PlannerThreads,
                                static_cast<size_t>(scan_size) * sizeof(uint32_t),
                                stream>>>(
        context_lens.data_ptr<int>(), batch, routes,
        static_cast<int>(rows_per_request), pair, device_info.logical_sms,
        schedule.data_ptr<int>());
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "get_paged_mqa_logits_fp4_metadata: launch failed");
    return schedule;
}

}  // namespace lightop::deep_gemm
