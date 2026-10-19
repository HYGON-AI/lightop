// Candidate-only MXFP4/MXFP8 paged MQA on HCU. Consume the HCU-derived
// sparse layout:
//   [16B MetadataHeader][KVSplit records][wave * num_sms ScheduleEntry].
// Its fourth header word is reserved zero, unlike the official CUDA
// header's num_sms field. These metadata blobs are not binary interchangeable.
// Each schedule entry belongs to a request-aligned pair of query rows.
// Its KV splits contain the two rows' candidate union and per-row output
// slot offsets. A wave loads a union tile once, scores both query rows,
// and scatters directly to each sparse output. No dense logits or secondary
// candidate table is constructed.
//
// Prefill flattens query tokens into R rows, passes
// packed MXFP4 Q and UE8M0 scales, and plan request-aligned pairs with the
// metadata generator. Supply context_lens and logical candidate blocks
// together so each row's partial causal tail is -inf in the output. Omit
// both for whole-candidate-block scoring; the caller limits TopK to the
// valid compact prefix. Both modes read only metadata-referenced K blocks.
// The pair's rows must share the physical page mapping used by its leader.
// is_prefill selects a dedicated large-R consumer. Decode retains its
// persistent wave-column schedule; both use the same metadata and score math.
//
// Math follows paged_mqa_logits_fp4.cu: exact E2M1-to-E4M3 conversion and
// gfx938 FP8 MMAC where scales fit, BF16 MMAC on gfx936/928 and for wide
// scale spans, FP32 accumulators, ReLU, FP32 weights and FP32 head reduction.
// An exponent-normalized FP32 scalar path handles exceptional scale ranges;
// there are no FP64 operations and no BF16 weights/output branch.
#include "paged_sparse_mqa_logits.h"
#include "paged_sparse_mqa_logits_initialization.h"
#include "paged_sparse_mqa_logits_metadata.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>

namespace lightop::deep_gemm::gfx936_detail {
namespace {
using bf16x4 = short __attribute__((ext_vector_type(4)));
using floatx4 = float __attribute__((ext_vector_type(4)));
using uintx4 = unsigned __attribute__((ext_vector_type(4)));
using intx2 = int __attribute__((ext_vector_type(2)));
constexpr uint32_t kInvalidSlot = 0xffffu;
constexpr uint32_t kBlockCountMask = 0x3fffffffu;
// Preserve device bounds checks without an out-of-line __assert_fail call.
// The latter forces live query/accumulator state into a private call frame.
#define SPARSE_DEVICE_CHECK(condition) do { if (!(condition)) __builtin_trap(); } while (0)
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 64;

// HCU-derived three-field header; its aligned fourth word is reserved zero.
// The planner/consumer share num_sms as a launch argument, not a blob field.
struct alignas(16) MetadataHeader {
    uint32_t num_kv_splits, num_waves, use_unaligned_ks, reserved;
};
struct alignas(16) KVSplitHeader {
    uint32_t q_token_base, packed_num_kv_blocks, q0_slot_base, q1_slot_base;
};
struct KVBlockInfo { uint32_t physical_kv_block_idx, packed_slot_offsets; };
struct alignas(16) ScheduleEntry {
    uint32_t kv_split_begin, kv_split_end, q_token_base, num_q_tokens;
};
static_assert(sizeof(MetadataHeader) == 16 && sizeof(KVSplitHeader) == 16 &&
              sizeof(KVBlockInfo) == 8 && sizeof(ScheduleEntry) == 16);

struct Args {
    const uint8_t *q, *kv, *metadata;
    const uint8_t* kv_fp8;
    const uint8_t* q_fp8;
    const uint8_t* metadata_fp8;
    int* fp8_invalid;
    int* selected_blocks;
    int *cache_pages, *cache_count;
    int* cache_physical_pages;
    // Consumer-only slots are 0..2047. Group-mask ownership authorizes every
    // load; unwritten entries are never read. Lossless 16-bit storage changes
    // no public metadata/header byte or duplicate-physical-page fallback.
    uint16_t* union_slotmap;
    int* index_invalid;
    uint32_t* union_mask;
    const uint32_t* q_sf;
    const float* weights;
    const int *context_lens, *candidates;
    float* out;
    int* exceptional;
    int64_t q_row_stride, q_head_stride, sf_row_stride, sf_head_stride;
    int64_t page_stride, weight_stride, candidate_stride, lens_stride, out_stride;
    int64_t num_pages, metadata_bytes, max_splits;
    int rows, heads, capacity, page_kv, block_kv, page_blocks, page_block_shift, num_sms;
    bool fp8_fnuz;
    bool multiq_union_mode;
    const uint8_t* q_bf16;
    const uint8_t* k_bf16=nullptr;
    bool bf16_q_fragment=false;
    bool size_auto=false,bf16_fused_prep=false,long_packed_cache=false;int size_force=0,cache_capacity=2048,bf16_k_mode=0,bf16_k_threshold=2049;
};

template<int Lanes = 64>
__device__ __forceinline__ int wave_min(int x) {
#pragma unroll
    for (int d = 1; d < Lanes; d *= 2) x = min(x, __shfl_xor(x, d, 64));
    return x;
}
template<int Lanes = 64>
__device__ __forceinline__ int wave_max(int x) {
#pragma unroll
    for (int d = 1; d < Lanes; d *= 2) x = max(x, __shfl_xor(x, d, 64));
    return x;
}
__device__ __forceinline__ uint16_t to_bf16(float x) {
#if defined(__gfx938__) && __has_builtin(__builtin_hcu_cvt_bf16_f32)
    return static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(x, false, false));
#else
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return (u >> 16) | 0x40U;
    return (u + 0x7fffU + ((u >> 16) & 1U)) >> 16;
#endif
}
template<bool FP4>
__device__ __forceinline__ float code_value(unsigned code, bool fnuz) {
    if constexpr (FP4) {
        const float lut[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const float v = lut[code & 7];
        return code & 8 ? -v : v;
    } else {
        if (fnuz && code == 0x80u) return NAN;
        const int e = (code >> 3) & 15, m = code & 7;
        const int bias = fnuz ? 8 : 7;
        const float v = e == 0 ? ldexpf(static_cast<float>(m), 1 - bias - 3)
            : (!fnuz && e == 15 && m == 7 ? NAN
               : ldexpf(1.f + m * .125f, e - bias));
        return code & 128 ? -v : v;
    }
}
template<bool FP4>
__device__ __forceinline__ float raw_value(const uint8_t* p, int c, bool fnuz) {
    const unsigned code = FP4 ? (p[c / 2] >> ((c & 1) * 4)) & 15 : p[c];
    return code_value<FP4>(code, fnuz);
}
__device__ __forceinline__ bf16x4 fp4_to_bf16(unsigned packed, unsigned sf) {
    bf16x4 v;
    const int delta = (static_cast<int>(sf) - 127) * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned code = (packed >> (j * 4)) & 15, mag = code & 7;
        const unsigned sign = (code & 8) << 12;
        const int base = mag >= 2 ? 0x3f00 + (mag << 6) : 0x3f00;
        v[j] = sign | (mag == 0 ? 0 : base + delta);
    }
    return v;
}
__device__ __forceinline__ bf16x4 fp8_to_bf16(
    const uint8_t* p, int c, unsigned sf, bool fnuz) {
    bf16x4 v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        v[j] = to_bf16(sf == 255 ? NAN :
            ldexpf(raw_value<false>(p, c + j, fnuz), static_cast<int>(sf) - 127));
    return v;
}
// Unchanged dense-kernel E2M1 -> E4M3 mapping, valid for delta in [-6,5].
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
#elif defined(__gfx936__) || defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    return c;  // Host entry rejects unsupported architectures.
#endif
}

// BF16 permutation conversion reused from the same-repository dense FP4 kernel.
struct SparseMqaBf16Lut {
    unsigned lo0, lo1, hi0, hi1;
};
__device__ __forceinline__ SparseMqaBf16Lut sparse_mqa_bf16_lut(int scale) {
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
__device__ __forceinline__ void sparse_mqa_decode_fp4x32(const uintx4& packed, const SparseMqaBf16Lut& lut, bf16x4 (&out)[8]) {
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


template<bool FP4, int HT, bool PreparedQ = false>
struct RowQ {
    uintx4 bits[HT];
    unsigned sf[HT];
    bf16x4 bf[FP4 ? 1 : HT][FP4 ? 1 : 8];
    float weights[HT][4];
    int smin, smax;
    bool safe_weight;
    int tail_slot;
#if defined(__gfx938__)
    intx2 f8[FP4 ? HT : 1][FP4 ? 4 : 1];
    int pivot;
    bool fp8_ok;
#endif
    __device__ __forceinline__ void load(const uint8_t* qrow, const uint32_t* sfrow, const float* wrow,
              int64_t q_head_stride, int64_t sf_head_stride, int heads,
              bool fp8_fnuz, int pos, int group, const uint8_t* prepared = nullptr) {
        if constexpr (PreparedQ) {
            const auto* status = reinterpret_cast<const int*>(prepared+4096);
            safe_weight = status[1] != 0;
            tail_slot = status[2];
#if defined(__gfx938__)
            pivot = status[0];
#endif
#pragma unroll
            for (int h=0;h<HT;++h) {
                const int head=h*16+pos;
#if defined(__gfx938__)
#pragma unroll
                for (int word=0;word<4;++word)
                    f8[h][word]=reinterpret_cast<const intx2*>(prepared+head*128+group*32)[word];
#endif
#pragma unroll
                for (int j=0;j<4;++j) weights[h][j]=wrow[h*16+j*4+group];
            }
            return;
        }
        smin = 255; smax = 0;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int head = h * 16 + pos;
            const bool live = head < heads;
            const uint8_t* qp = qrow + min(head, heads - 1) * q_head_stride;
            sf[h] = live ? (sfrow[head * sf_head_stride] >> (group * 8)) & 255u : 127u;
            if (live) { smin = min(smin, static_cast<int>(sf[h])); smax = max(smax, static_cast<int>(sf[h])); }
            if constexpr (FP4) {
                bits[h] = live ? *reinterpret_cast<const uintx4*>(qp + group * 16) : uintx4{0,0,0,0};
            } else {
#pragma unroll
                for (int s = 0; s < 8; ++s)
                    bf[h][s] = live ? fp8_to_bf16(qp, group * 32 + s * 4, sf[h], fp8_fnuz) : bf16x4{0,0,0,0};
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wh = h * 16 + j * 4 + group;
                weights[h][j] = wh < heads ? wrow[wh] : 0.f;
            }
        }
        smin = wave_min(smin); smax = wave_max(smax);
        // Raw/Decode normal scales cover a wider range than preparation.
        // Restore every head before ReLU/FMA; only PreparedQ status may
        // enable late scale after proving the bounded Q/K/weight domain.
        safe_weight = false;
#if defined(__gfx938__)
        pivot = min(smax, smin + 5);
        fp8_ok = FP4 && smax - smin <= 11;
        if constexpr (FP4) {
#pragma unroll
            for (int h = 0; h < HT; ++h) {
#pragma unroll
                for (int s = 0; s < 4; ++s) f8[h][s] = fp4_to_fp8(bits[h][s], pivot - static_cast<int>(sf[h]));
            }
        }
#endif
    }
};
template<int HT>
__device__ __forceinline__ float head_sum(const floatx4 (&acc)[HT], const float (&w)[HT][4]) {
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int h = 0; h < HT; ++h) {
        s0 = fmaf(fmaxf(acc[h][0], 0.f), w[h][0], s0);
        s1 = fmaf(fmaxf(acc[h][1], 0.f), w[h][1], s1);
        s0 = fmaf(fmaxf(acc[h][2], 0.f), w[h][2], s0);
        s1 = fmaf(fmaxf(acc[h][3], 0.f), w[h][3], s1);
    }
    float sum = s0 + s1;
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    return sum;
}

// Keep Q and K unscaled during the scalar dot, combine their scale exponents,
// and normalize the four channel groups before their FP32 sum. This avoids
// overflow in separately decoded operands and uses no FP64 instructions.
template<bool FP4>
__device__ __attribute__((noinline)) float exceptional_score(
    const Args& a, int r, int group, const uint8_t* kp, unsigned ks, bool live) {
    float s0 = 0.f, s1 = 0.f;
    for (int h = 0; h < a.heads; ++h) {
        const uint8_t* qp = a.q + static_cast<int64_t>(r) * a.q_row_stride + h * a.q_head_stride;
        const unsigned qs = (a.q_sf[static_cast<int64_t>(r) * a.sf_row_stride + h * a.sf_head_stride]
                             >> (group * 8)) & 255u;
        int exponent = static_cast<int>(qs) + static_cast<int>(ks) - 254;
        float dot = 0.f, correction = 0.f;
        if (live) {
#pragma clang loop unroll(disable)
            for (int c = 0; c < 32; ++c) {
                const float qv = raw_value<FP4>(qp, group * 32 + c, a.fp8_fnuz);
                const float kv = raw_value<FP4>(kp, group * 32 + c, a.fp8_fnuz);
                if constexpr (FP4) dot = fmaf(qv,kv,dot);
                else {
                    // E4M3 products are exactly representable in FP32;
                    // TwoSum retains their low parts through cancellation.
                    const float product = qv*kv, next = dot+product, b = next-dot;
                    correction += (dot-(next-b))+(product-b);
                    dot = next;
                }
            }
        }
        if constexpr (!FP4) dot += correction;
        if (qs == 255 || ks == 255) dot = NAN;
#pragma unroll
        for (int distance = 16; distance <= 32; distance *= 2) {
            int adjustment = 0;
            dot = frexpf(dot,&adjustment);
            // A zero with an enormous scale must not discard another group's
            // smaller, nonzero value during normalization.
            exponent = dot == 0.f ? -4096 : exponent+adjustment;
            const float peer = __shfl_xor(dot,distance,64);
            const int peer_exponent = __shfl_xor(exponent,distance,64);
            const int pivot = max(exponent,peer_exponent);
            dot = ldexpf(dot,exponent-pivot)+ldexpf(peer,peer_exponent-pivot);
            exponent = pivot;
        }
        dot = ldexpf(dot,exponent);
        if (h % 4 == group) {
            const float relu = fmaxf(dot, 0.f);
            const float w = a.weights[static_cast<int64_t>(r) * a.weight_stride + h];
            if ((h / 4) & 1) s1 = fmaf(relu, w, s1); else s0 = fmaf(relu, w, s0);
        }
    }
    float sum = s0 + s1;
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    return sum;
}

// One same-stream setup launch owns output and all fresh private fill buffers.
// Marking and all scorers launch only after this kernel completes. No atomics
// or conversion work is interleaved with initialization across CTAs.

__global__ void sparse_fp8_fp4_paged_mqa_logits__initialize_output(float* out, int64_t stride, int rows, int width, int* exceptional) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *exceptional = 0;
    const int64_t n = static_cast<int64_t>(rows) * width;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        out[(i / width) * stride + i % width] = -INFINITY;
}

// Prefill's contiguous, 16-byte-aligned output needs no per-element row
// division. Fill every logical element, including absent candidate slots,
// with the same -inf bits; Decode and arbitrary strides keep the old path.
__global__ void sparse_fp8_fp4_paged_mqa_logits__initialize_output_contiguous_vec4(
    float* out, int64_t nvec, int* exceptional) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *exceptional = 0;
    auto* dst = reinterpret_cast<floatx4*>(out);
    const floatx4 value{-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for (int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
         i < nvec; i += static_cast<int64_t>(gridDim.x)*blockDim.x)
        dst[i] = value;
}

template<bool FP4, int HT, int BlockKV, bool FallbackOnly>
__attribute__((amdgpu_flat_work_group_size(kThreads,kThreads)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__decode_pair(Args a) {
    if constexpr (FallbackOnly) {
        if (*a.exceptional == 0) return;
    }
    constexpr int PB = FP4 ? 64 : 128;
    constexpr int SplitKV = FP4 ? 640 : 512;
    constexpr int BlocksPerSplit = SplitKV / BlockKV;
    constexpr int SplitBytes = sizeof(KVSplitHeader) + BlocksPerSplit * sizeof(KVBlockInfo);
    const auto* header = reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(a.num_sms == gridDim.x && header->reserved == 0 && header->use_unaligned_ks == 0 &&
           header->num_kv_splits <= a.max_splits && header->num_waves > 0);
    const uint8_t* splits = a.metadata + sizeof(MetadataHeader);
    const auto* schedule = reinterpret_cast<const ScheduleEntry*>(
        splits + static_cast<int64_t>(header->num_kv_splits) * SplitBytes);
    SPARSE_DEVICE_CHECK(sizeof(MetadataHeader) + static_cast<int64_t>(header->num_kv_splits) * SplitBytes +
           static_cast<int64_t>(header->num_waves) * a.num_sms * sizeof(ScheduleEntry) <= a.metadata_bytes);
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int pos = lane % 16, group = lane / 16;
    RowQ<FP4,HT> query0, query1;
    for (uint32_t w = 0; w < header->num_waves; ++w) {
        const ScheduleEntry entry = schedule[static_cast<int64_t>(w) * gridDim.x + blockIdx.x];
        if (entry.num_q_tokens == 0) {
            SPARSE_DEVICE_CHECK(entry.kv_split_begin == entry.kv_split_end);
            continue;
        }
        SPARSE_DEVICE_CHECK(entry.num_q_tokens <= 2 && entry.q_token_base + entry.num_q_tokens <= a.rows &&
               entry.kv_split_begin <= entry.kv_split_end && entry.kv_split_end <= header->num_kv_splits);
        const int r = entry.q_token_base;
        query0.load(a.q + static_cast<int64_t>(r) * a.q_row_stride,
                      a.q_sf + static_cast<int64_t>(r) * a.sf_row_stride,
                      a.weights + static_cast<int64_t>(r) * a.weight_stride,
                      a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group);
        if (entry.num_q_tokens == 2) query1.load(a.q + static_cast<int64_t>(r + 1) * a.q_row_stride,
                      a.q_sf + static_cast<int64_t>(r + 1) * a.sf_row_stride,
                      a.weights + static_cast<int64_t>(r + 1) * a.weight_stride,
                      a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group);
        for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++split_idx) {
            const auto* sh = reinterpret_cast<const KVSplitHeader*>(splits + static_cast<int64_t>(split_idx) * SplitBytes);
            const auto* blocks = reinterpret_cast<const KVBlockInfo*>(sh + 1);
            const int count = sh->packed_num_kv_blocks & kBlockCountMask;
            SPARSE_DEVICE_CHECK(sh->q_token_base == entry.q_token_base && count > 0 && count <= BlocksPerSplit &&
                   (sh->packed_num_kv_blocks & ~kBlockCountMask) == 0 &&
                   (entry.num_q_tokens == 2 || sh->q1_slot_base == 0xffffffffu));
            // Each wave owns disjoint union tiles; all four groups resolve the
            // same physical token, which may belong to only one query row.
            for (int tile = wave * 16; tile < count * BlockKV; tile += kWaves * 16) {
                const int t = tile + pos, block = t / BlockKV, off = t % BlockKV;
                KVBlockInfo info{0,0xffffffffu};
                if (block < count) info = blocks[block];
                int slots[2]; bool write[2], valid[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t local_slot = (info.packed_slot_offsets >> (i * 16)) & 0xffffu;
                    const uint32_t base = i == 0 ? sh->q0_slot_base : sh->q1_slot_base;
                    write[i] = i < entry.num_q_tokens && local_slot != kInvalidSlot;
                    slots[i] = write[i] ? static_cast<int>(base + local_slot) : 0;
                    SPARSE_DEVICE_CHECK(!write[i] || (slots[i] >= 0 && slots[i] < a.capacity));
                    valid[i] = write[i];
                    if (a.context_lens && write[i]) {
                        const int logical = a.candidates[static_cast<int64_t>(r+i) * a.candidate_stride + slots[i]];
                        const int len = a.context_lens[static_cast<int64_t>(r+i) * a.lens_stride];
                        SPARSE_DEVICE_CHECK(logical >= 0 && len >= 0);
                        valid[i] = static_cast<int64_t>(logical) * BlockKV + off < len;
                    }
                }
                const bool live = valid[0] || valid[1];
                const uint8_t* kp = a.kv;
                uint32_t sw = 0x7f7f7f7fu;
                if (live) {
                    // The official physical block ID is uint32. Resolve pages
                    // in that block space, avoiding a per-token 64-bit divide.
                    // Power-of-two page sizes use shifts; other legal sizes
                    // retain an exact uint32 quotient/remainder.
                    const uint32_t physical = info.physical_kv_block_idx;
                    const uint32_t page = a.page_block_shift >= 0
                        ? physical >> a.page_block_shift : physical / a.page_blocks;
                    const uint32_t page_offset = (physical - page * a.page_blocks) * BlockKV + off;
                    SPARSE_DEVICE_CHECK(page < a.num_pages);
                    const uint8_t* page_ptr = a.kv + page * a.page_stride;
                    kp = page_ptr + page_offset * PB;
                    sw = reinterpret_cast<const uint32_t*>(page_ptr + static_cast<int64_t>(a.page_kv) * PB)[page_offset];
                }
                const unsigned sf = (sw >> (group * 8)) & 255u;
                const int s0 = sw & 255, s1 = (sw >> 8) & 255, s2 = (sw >> 16) & 255, s3 = sw >> 24;
                const int lo = min(min(s0,s1),min(s2,s3)), hi = max(max(s0,s1),max(s2,s3));
                bool row_live[2] = {false,false}, normal[2] = {false,false}, fp8[2] = {false,false};
#if defined(__gfx938__)
                const int pk = min(hi,lo+5);
#endif
                // Only a row's own live candidates vote on its arithmetic
                // path. A wide-scale block exclusive to its paired row must
                // not change this row's BF16/FP8 or exceptional decision.
                auto plan_query = [&](const RowQ<FP4,HT>& q, int i)
                    __attribute__((always_inline)) {
                    row_live[i] = __ballot(valid[i]) != 0;
                    const int kmin = wave_min<16>(valid[i] ? lo : 255);
                    const int kmax = wave_max<16>(valid[i] ? hi : 0);
                    normal[i] = row_live[i] && kmin >= 16 && kmax <= 240 &&
                                q.smin >= 16 && q.smax <= 240 &&
                                q.smin + kmin >= 150 && q.smax + kmax <= 350;
#if defined(__gfx938__)
                    if constexpr (FP4) {
                        const int e = q.pivot + pk - 127;
                        const bool ok = !valid[i] || (hi-lo <= 11 && e >= 1 && e <= 254);
                        fp8[i] = normal[i] && q.fp8_ok && __ballot(!ok) == 0;
                    }
#endif
                };
                plan_query(query0, 0);
                if (entry.num_q_tokens == 2) plan_query(query1, 1);

                // Decode once into each format needed by a live normal row.
                // Other positions are zero operands, including a paired row's
                // exclusive exceptional candidates. The fallback uses raw codes.
                const bool need_bf16 = (normal[0] && !fp8[0]) || (normal[1] && !fp8[1]);
                const bool bf16_live = (normal[0] && !fp8[0] && valid[0]) ||
                                       (normal[1] && !fp8[1] && valid[1]);
                bf16x4 kbf[8];
#if defined(__gfx938__)
                const bool need_fp8 = fp8[0] || fp8[1];
                const bool fp8_live = (fp8[0] && valid[0]) || (fp8[1] && valid[1]);
                intx2 k8[4];
#endif
                if constexpr (!FallbackOnly) {
                    uintx4 kb;
                    if constexpr (FP4) {
                        bool decode_live = bf16_live;
#if defined(__gfx938__)
                        decode_live |= fp8_live;
#endif
                        kb = decode_live ? *reinterpret_cast<const uintx4*>(kp + group * 16) : uintx4{0,0,0,0};
                    }
                    if (need_bf16) {
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            if constexpr (FP4) kbf[s] = bf16_live
                                ? fp4_to_bf16((kb[s/2] >> ((s&1)*16)) & 0xffffu, sf) : bf16x4{0,0,0,0};
                            else kbf[s] = bf16_live
                                ? fp8_to_bf16(kp, group * 32 + s * 4, sf, a.fp8_fnuz) : bf16x4{0,0,0,0};
                        }
                    }
#if defined(__gfx938__)
                    if constexpr (FP4) {
                        if (need_fp8) {
#pragma unroll
                            for (int s = 0; s < 4; ++s) k8[s] = fp8_live
                                ? fp4_to_fp8(kb[s], pk - static_cast<int>(sf)) : intx2{0,0};
                        }
                    }
#endif
                }
                auto score_query = [&](const RowQ<FP4,HT>& q, int i)
                    __attribute__((always_inline)) {
                    if (!row_live[i]) return;
                    float logit;
                    if constexpr (FallbackOnly) {
                        if (normal[i]) return;
                        logit = exceptional_score<FP4>(a, r+i, group, kp, sf, valid[i]);
                    } else {
                        if (!normal[i]) {
                            if (group == 0 && valid[i]) atomicExch(a.exceptional, 1);
                            return;
                        }
                        floatx4 acc[HT];
#pragma unroll
                        for (int h = 0; h < HT; ++h) acc[h] = floatx4{0,0,0,0};
                        float residual = 1.f;
                        bool used_fp8 = false;
#if defined(__gfx938__)
                        const int e = q.pivot + pk - 127;
                        if constexpr (FP4) {
                            used_fp8 = fp8[i];
                            if (fp8[i]) {
#pragma unroll
                                for (int s = 0; s < 4; ++s) {
#pragma unroll
                                    for (int h = 0; h < HT; ++h)
                                        acc[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(k8[s], q.f8[h][s], acc[h], false, false);
                                }
                                residual = __uint_as_float(static_cast<unsigned>(e) << 23);
                            }
                        }
#endif
                        if (!used_fp8) {
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
#pragma unroll
                                for (int h = 0; h < HT; ++h) {
                                    bf16x4 qbf;
                                    if constexpr (FP4) qbf = fp4_to_bf16((q.bits[h][s/2] >> ((s&1)*16)) & 0xffffu, q.sf[h]);
                                    else qbf = q.bf[h][s];
                                    acc[h] = mmac_bf16(kbf[s], qbf, acc[h]);
                                }
                            }
                        }
                        // Restore dot exponents before extreme head weights,
                        // avoiding a large/small temporary weighted result.
                        if (!q.safe_weight) {
#pragma unroll
                            for (int h = 0; h < HT; ++h) acc[h] *= residual;
                            residual = 1.f;
                        }
                        logit = head_sum<HT>(acc, q.weights) * residual;
                    }
                    if (group == 0 && write[i])
                        a.out[static_cast<int64_t>(r+i) * a.out_stride + slots[i]*BlockKV + off] = valid[i] ? logit : -INFINITY;

                };
                score_query(query0, 0);
                if (entry.num_q_tokens == 2) score_query(query1, 1);
            }
        }
    }
}

// Prefill's block-based loads broadcast one metadata/candidate record to all
// positions of that block. Two 16-position tiles are staged per wave before
// arithmetic, while the row endpoints remain resident for the schedule entry.
struct PrefillKVTile {
    const uint8_t* payload;
    const uint8_t* fp8_payload;
    uint32_t scales;
    int slots[2];
    bool write[2], valid[2];
};

struct PrefillQueryPlan {
    bool row_live, normal, fp8;
};

template<bool FP4, int BlockKV, bool CommonSafe = false, bool WholeBlocks = false>
__device__ __forceinline__ PrefillKVTile load_prefill_tile(
    const Args& a, const KVSplitHeader& sh, const KVBlockInfo* blocks,
    int count, int num_q_tokens, int r, int t, int pos, int group,
    const int (&lengths)[2], const int (&tails)[2]) {
    constexpr int PB = FP4 ? 64 : 128;
    const int block = t / BlockKV, off = t % BlockKV;
    const int leader = pos - off;
    KVBlockInfo info{0,0xffffffffu};
    if (group == 0 && off == 0 && block < count) info = blocks[block];
    info.physical_kv_block_idx = __shfl(info.physical_kv_block_idx, leader, 64);
    info.packed_slot_offsets = __shfl(info.packed_slot_offsets, leader, 64);
    PrefillKVTile tile{};
    tile.payload = a.kv;
    tile.scales = 0x7f7f7f7fu;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t local = (info.packed_slot_offsets >> (i * 16)) & 0xffffu;
        const uint32_t base = i == 0 ? sh.q0_slot_base : sh.q1_slot_base;
        tile.write[i] = i < num_q_tokens && local != kInvalidSlot;
        tile.slots[i] = tile.write[i] ? static_cast<int>(base + local) : 0;
        SPARSE_DEVICE_CHECK(!tile.write[i] || (tile.slots[i] >= 0 && tile.slots[i] < a.capacity));
        tile.valid[i] = tile.write[i];
        if (!WholeBlocks && a.context_lens) {
            if constexpr (CommonSafe) {
                // Preparation validates all participating candidate IDs and the
                // last partial block's exact slot. Other candidates are full.
                const int tail_tokens = (lengths[i]-1) % BlockKV + 1;
                tile.valid[i] = tile.write[i] &&
                    (tile.slots[i] != tails[i] || off < tail_tokens);
            } else {
                int logical = 0;
                if (group == 0 && off == 0 && tile.write[i])
                    logical = a.candidates[static_cast<int64_t>(r+i) * a.candidate_stride + tile.slots[i]];
                logical = __shfl(logical, leader, 64);
                SPARSE_DEVICE_CHECK(!tile.write[i] || (logical >= 0 && lengths[i] >= 0));
                tile.valid[i] = tile.write[i] && static_cast<int64_t>(logical) * BlockKV + off < lengths[i];
            }
        }
    }
    if (tile.valid[0] || tile.valid[1]) {
        const uint32_t physical = info.physical_kv_block_idx;
        const uint32_t page = a.page_block_shift >= 0
            ? physical >> a.page_block_shift : physical / a.page_blocks;
        const uint32_t page_offset = (physical - page * a.page_blocks) * BlockKV + off;
        SPARSE_DEVICE_CHECK(page < a.num_pages);
        const uint8_t* page_ptr = a.kv + page * a.page_stride;
        tile.payload = page_ptr + page_offset * PB;
        // Cache page IDs are assigned once by the metadata marking pass.
        // Clamp is only relevant to the raw overflow path, which never reads
        // these fragments; cached consumers execute only without overflow.
        const int cached_page=CommonSafe ? static_cast<int>(page) :
            (a.kv_fp8 ? min(max(a.cache_pages[page],0),2047) : 0);
        tile.fp8_payload = a.kv_fp8 ? a.kv_fp8 + static_cast<int64_t>(cached_page)*8448 + page_offset*128 : nullptr;
        if constexpr (CommonSafe) {
            tile.scales = reinterpret_cast<const uint32_t*>(a.kv_fp8+static_cast<int64_t>(cached_page)*8448+8192)[page_offset];
        } else {
            tile.scales = reinterpret_cast<const uint32_t*>(page_ptr + static_cast<int64_t>(a.page_kv) * PB)[page_offset];
        }
    }
    return tile;
}

template<bool FP4, int HT, int BlockKV, bool FallbackOnly, bool FP8Only = false, bool BF16Only = false, bool FixedLayout = false, bool Predecoded = false, bool CommonSafe = false, bool OverflowOnly = false, bool Interleaved = false, bool WholeBlocks = false, bool UnionFallbackOnly = false>
__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__prefill_pair(Args a) {
    if (a.multiq_union_mode && *a.fp8_invalid==0 && *a.index_invalid==0) return;
    if constexpr (OverflowOnly) {
        if (*a.fp8_invalid == 0) return;
    }
    if constexpr (Predecoded) {
        if ((*a.fp8_invalid & 2) != 0) return;
        if constexpr (CommonSafe) {
            if (*a.fp8_invalid != 0) return;
        } else if constexpr (FP8Only) {
            if (*a.fp8_invalid == 0) return;
        }
    }
    if constexpr (CommonSafe) a.metadata=a.metadata_fp8;
    if constexpr (UnionFallbackOnly) {
        // Complete scoring owns index-collision or low-density calls.
        if (*a.index_invalid==0) return;
    }
    if constexpr (FixedLayout) {
        // Host dispatch proves these real-workload dimensions and strides.
        // Constants remove redundant dynamic 64-bit address multipliers and
        // reduce the live launch-argument state; arithmetic is unchanged.
        a.heads = 32; a.capacity = 2048; a.page_kv = 64; a.block_kv = 8;
        a.page_blocks = 8; a.page_block_shift = 3;
        a.q_row_stride = 2048; a.q_head_stride = 64;
        a.sf_row_stride = 32; a.sf_head_stride = 1;
        a.page_stride = 4352; a.out_stride = 16384;
        a.candidate_stride = 2048; a.lens_stride = 1;
    }
    if constexpr (FallbackOnly) {
        if ((*a.exceptional & 1) == 0) return;
    }
    if constexpr (BF16Only) {
        if ((*a.exceptional & 2) == 0) return;
    }
    constexpr int Waves = 2, TilesPerWave = 2;
    static_assert(!Interleaved || (FP4 && HT == 2 && BlockKV == 8 &&
                  !FallbackOnly && FP8Only && !BF16Only && FixedLayout &&
                  Predecoded && CommonSafe && !OverflowOnly),
                  "Interleaved tiles require the prepared FP4 common path");
    // appends WholeBlocks after Interleaved so existing launch arguments
    // retain their meaning. Only the prepared-null-tail dispatch enables it.
    static_assert(!WholeBlocks || (Interleaved && FP4 && HT == 2 && BlockKV == 8 &&
                  !FallbackOnly && FP8Only && !BF16Only && FixedLayout &&
                  Predecoded && CommonSafe && !OverflowOnly),
                  "WholeBlocks requires the interleaved prepared FP4 common path");
    static_assert(!UnionFallbackOnly || (WholeBlocks && CommonSafe && Predecoded &&
                  Interleaved && !FallbackOnly && !OverflowOnly),
                  "UnionFallbackOnly requires the eligible prepared whole path");
    constexpr int BlocksPerSplit = (FP4 ? 640 : 512) / BlockKV;
    constexpr int SplitBytes = sizeof(KVSplitHeader) + BlocksPerSplit * sizeof(KVBlockInfo);
    const auto* header = reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(header->reserved == 0 && header->use_unaligned_ks == 0 &&
           header->num_kv_splits <= a.max_splits && header->num_waves > 0);
    const uint8_t* splits = a.metadata + sizeof(MetadataHeader);
    const auto* schedule = reinterpret_cast<const ScheduleEntry*>(
        splits + static_cast<int64_t>(header->num_kv_splits) * SplitBytes);
    const int64_t entries = static_cast<int64_t>(header->num_waves) * a.num_sms;
    SPARSE_DEVICE_CHECK(sizeof(MetadataHeader) + static_cast<int64_t>(header->num_kv_splits) * SplitBytes +
           entries * sizeof(ScheduleEntry) <= a.metadata_bytes);
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int pos = lane % 16, group = lane / 16;
    // C003: named rows and accumulators avoid dynamic indexing of large
    // per-thread arrays. Tile staging, scheduling and score order stay fixed.
    RowQ<FP4,HT,CommonSafe> q0, q1;
    // More independent CTAs can consume the many Prefill entries. No host
    // read of num_waves is needed, and the producer's schedule width is kept.
    for (int64_t entry_idx = blockIdx.x; entry_idx < entries; entry_idx += gridDim.x) {
        const ScheduleEntry entry = schedule[entry_idx];
        if (entry.num_q_tokens == 0) {
            SPARSE_DEVICE_CHECK(entry.kv_split_begin == entry.kv_split_end);
            continue;
        }
        SPARSE_DEVICE_CHECK(entry.num_q_tokens <= 2 && entry.q_token_base + entry.num_q_tokens <= a.rows &&
               entry.kv_split_begin <= entry.kv_split_end && entry.kv_split_end <= header->num_kv_splits);
        const int r = entry.q_token_base;
        int lengths[2] = {0,0};
        if (!WholeBlocks && a.context_lens) {
            if (lane == 0) {
                lengths[0] = a.context_lens[static_cast<int64_t>(r) * a.lens_stride];
                if (entry.num_q_tokens == 2)
                    lengths[1] = a.context_lens[static_cast<int64_t>(r+1) * a.lens_stride];
            }
            lengths[0] = __shfl(lengths[0], 0, 64);
            lengths[1] = __shfl(lengths[1], 0, 64);
        }
        q0.load(a.q + static_cast<int64_t>(r) * a.q_row_stride,
                a.q_sf + static_cast<int64_t>(r) * a.sf_row_stride,
                a.weights + static_cast<int64_t>(r) * a.weight_stride,
                a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group,
                CommonSafe ? a.q_fp8+static_cast<int64_t>(r)*4112 : nullptr);
        if (entry.num_q_tokens == 2)
            q1.load(a.q + static_cast<int64_t>(r+1) * a.q_row_stride,
                    a.q_sf + static_cast<int64_t>(r+1) * a.sf_row_stride,
                    a.weights + static_cast<int64_t>(r+1) * a.weight_stride,
                    a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group,
                    CommonSafe ? a.q_fp8+static_cast<int64_t>(r+1)*4112 : nullptr);
        int tails[2] = {-1,-1};
        if constexpr (CommonSafe && !WholeBlocks) {
            tails[0]=q0.tail_slot;
            if (entry.num_q_tokens == 2) tails[1]=q1.tail_slot;
        }
        for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++split_idx) {
            const auto* sh = reinterpret_cast<const KVSplitHeader*>(splits + static_cast<int64_t>(split_idx) * SplitBytes);
            const auto* blocks = reinterpret_cast<const KVBlockInfo*>(sh + 1);
            const int count = sh->packed_num_kv_blocks & kBlockCountMask;
            SPARSE_DEVICE_CHECK(sh->q_token_base == entry.q_token_base && count > 0 && count <= BlocksPerSplit &&
                   (sh->packed_num_kv_blocks & ~kBlockCountMask) == 0 &&
                   (entry.num_q_tokens == 2 || sh->q1_slot_base == 0xffffffffu));

            for (int base = wave * TilesPerWave * 16; base < count * BlockKV;
                 base += Waves * TilesPerWave * 16) {
                PrefillKVTile tiles[TilesPerWave];
#pragma unroll
                for (int j = 0; j < TilesPerWave; ++j)
                    tiles[j] = load_prefill_tile<FP4,BlockKV,CommonSafe,WholeBlocks>(
                        a, *sh, blocks, count, entry.num_q_tokens, r, base + j*16 + pos, pos, group, lengths, tails);
#if defined(__gfx938__)
                if constexpr (Interleaved) {
                    // Only the whole-block prepared/common dispatch selects this
                    // specialization. Named tile accumulators keep all indexing
                    // static and provide eight independent MMAC chains. Every
                    // dot still consumes K32 chunks in the original s=0..3 order.
                    const bool live00 = __ballot(tiles[0].valid[0]) != 0;
                    const bool live01 = __ballot(tiles[1].valid[0]) != 0;
                    const bool live10 = entry.num_q_tokens == 2 && __ballot(tiles[0].valid[1]) != 0;
                    const bool live11 = entry.num_q_tokens == 2 && __ballot(tiles[1].valid[1]) != 0;
                    uintx4 k0lo{0,0,0,0}, k0hi{0,0,0,0};
                    uintx4 k1lo{0,0,0,0}, k1hi{0,0,0,0};
                    if (tiles[0].valid[0] || tiles[0].valid[1]) {
                        k0lo = *reinterpret_cast<const uintx4*>(tiles[0].fp8_payload + group*32);
                        k0hi = *reinterpret_cast<const uintx4*>(tiles[0].fp8_payload + group*32 + 16);
                    }
                    if (tiles[1].valid[0] || tiles[1].valid[1]) {
                        k1lo = *reinterpret_cast<const uintx4*>(tiles[1].fp8_payload + group*32);
                        k1hi = *reinterpret_cast<const uintx4*>(tiles[1].fp8_payload + group*32 + 16);
                    }
                    floatx4 acc00[HT], acc01[HT], acc10[HT], acc11[HT];
#pragma unroll
                    for (int h = 0; h < HT; ++h) {
                        acc00[h] = floatx4{0,0,0,0};
                        acc01[h] = floatx4{0,0,0,0};
                        acc10[h] = floatx4{0,0,0,0};
                        acc11[h] = floatx4{0,0,0,0};
                    }
                    auto score_chunk = [&](int s, const uintx4& lo, const uintx4& hi,
                                           floatx4 (&acc0)[HT], floatx4 (&acc1)[HT],
                                           bool live0, bool live1) __attribute__((always_inline)) {
                        const uintx4 words = s < 2 ? lo : hi;
                        const intx2 k8{static_cast<int>(words[(s & 1)*2]),
                                      static_cast<int>(words[(s & 1)*2+1])};
                        if (live0) {
#pragma unroll
                            for (int h = 0; h < HT; ++h)
                                acc0[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                    k8, q0.f8[h][s], acc0[h], false, false);
                        }
                        if (live1) {
#pragma unroll
                            for (int h = 0; h < HT; ++h)
                                acc1[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                    k8, q1.f8[h][s], acc1[h], false, false);
                        }
                    };
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        score_chunk(s, k0lo, k0hi, acc00, acc10, live00, live10);
                        score_chunk(s, k1lo, k1hi, acc01, acc11, live01, live11);
                    }
                    auto store_tile = [&](const PrefillKVTile& tile, int tile_offset,
                                          const RowQ<FP4,HT,CommonSafe>& q,
                                          floatx4 (&acc)[HT], bool live, int i)
                        __attribute__((always_inline)) {
                        if (!live) return;
                        float residual = __uint_as_float(
                            static_cast<unsigned>(q.pivot + static_cast<int>(tile.scales) - 127) << 23);
                        // Retain the original extreme-weight ordering: scale
                        // accumulators before ReLU/FMA whenever the guard says
                        // multiplying unscaled logits by weights is unsafe.
                        if (!q.safe_weight) {
#pragma unroll
                            for (int h = 0; h < HT; ++h) acc[h] *= residual;
                            residual = 1.f;
                        }
                        const float logit = head_sum<HT>(acc, q.weights) * residual;
                        if (group == 0 && tile.write[i])
                            a.out[static_cast<int64_t>(r+i) * a.out_stride + tile.slots[i]*BlockKV +
                                  (base + tile_offset + pos) % BlockKV] = tile.valid[i] ? logit : -INFINITY;
                    };
                    store_tile(tiles[0], 0, q0, acc00, live00, 0);
                    if (entry.num_q_tokens == 2) store_tile(tiles[0], 0, q1, acc10, live10, 1);
                    store_tile(tiles[1], 16, q0, acc01, live01, 0);
                    if (entry.num_q_tokens == 2) store_tile(tiles[1], 16, q1, acc11, live11, 1);
                    continue;
                }
#endif
#pragma unroll
                for (int j = 0; j < TilesPerWave; ++j) {
                    const auto& tile = tiles[j];
                    const unsigned sf = (tile.scales >> (group * 8)) & 255u;
                    const int s0 = tile.scales & 255, s1 = (tile.scales >> 8) & 255;
                    const int s2 = (tile.scales >> 16) & 255, s3 = tile.scales >> 24;
                    const int lo = min(min(s0,s1),min(s2,s3)), hi = max(max(s0,s1),max(s2,s3));
#if defined(__gfx938__)
                    const int pk = CommonSafe ? static_cast<int>(tile.scales) : min(hi,lo+5);
#endif
                    auto plan_query = [&](const RowQ<FP4,HT,CommonSafe>& q, bool valid)
                        __attribute__((always_inline)) {
                        PrefillQueryPlan plan{__ballot(valid) != 0,false,false};
                        if constexpr (CommonSafe) {
                            // Runtime preparation proves all Q/K scales in [96,160]
                            // and each Q-row/K-token span <=11. Joint scales are
                            // [192,320], residual exponents [65,193], so the exact
                            // existing FP8 and normal-dot predicates always hold.
                            plan.normal = plan.fp8 = plan.row_live;
                            return plan;
                        }
                        const bool lane_normal = lo >= 16 && hi <= 240 &&
                            q.smin >= 16 && q.smax <= 240 &&
                            q.smin + lo >= 150 && q.smax + hi <= 350;
                        plan.normal = plan.row_live && __ballot(valid && !lane_normal) == 0;
#if defined(__gfx938__)
                        if constexpr (FP4) {
                            const int e = q.pivot + pk - 127;
                            const bool ok = !valid || (hi-lo <= 11 && e >= 1 && e <= 254);
                            plan.fp8 = plan.normal && q.fp8_ok && __ballot(!ok) == 0;
                        }
#endif
                        return plan;
                    };
                    const PrefillQueryPlan p0 = plan_query(q0, tile.valid[0]);
                    const PrefillQueryPlan p1 = entry.num_q_tokens == 2
                        ? plan_query(q1, tile.valid[1]) : PrefillQueryPlan{false,false,false};
                    floatx4 acc0[HT], acc1[HT];
                    float residual0 = 1.f, residual1 = 1.f;
                    if constexpr (!FallbackOnly) {
#pragma unroll
                        for (int h = 0; h < HT; ++h) {
                            acc0[h] = floatx4{0,0,0,0};
                            acc1[h] = floatx4{0,0,0,0};
                        }
                        const bool need_bf16 = !FP8Only && ((p0.normal && !p0.fp8) || (p1.normal && !p1.fp8));
                        const bool bf16_live = !FP8Only && ((p0.normal && !p0.fp8 && tile.valid[0]) ||
                                               (p1.normal && !p1.fp8 && tile.valid[1]));
                        bool decode_live = bf16_live;
#if defined(__gfx938__)
                        const bool need_fp8 = !BF16Only && (p0.fp8 || p1.fp8);
                        const bool fp8_live = !BF16Only && ((p0.fp8 && tile.valid[0]) || (p1.fp8 && tile.valid[1]));
                        if constexpr (!Predecoded) decode_live |= fp8_live;
#endif
                        uintx4 kb;
                        if constexpr (FP4) kb = decode_live
                            ? *reinterpret_cast<const uintx4*>(tile.payload + group * 16) : uintx4{0,0,0,0};
                        // The fragment is immediately multiplied into both Q
                        // accumulators; only formats needed by a live row exist.
                        if (need_bf16) {
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
                                bf16x4 kbf;
                                if constexpr (FP4) kbf = bf16_live
                                    ? fp4_to_bf16((kb[s/2] >> ((s&1)*16)) & 0xffffu, sf) : bf16x4{0,0,0,0};
                                else kbf = bf16_live
                                    ? fp8_to_bf16(tile.payload, group * 32 + s * 4, sf, a.fp8_fnuz) : bf16x4{0,0,0,0};
                                if (p0.normal && !p0.fp8) {
#pragma unroll
                                    for (int h = 0; h < HT; ++h) {
                                        bf16x4 qbf;
                                        if constexpr (FP4) qbf = fp4_to_bf16(
                                            (q0.bits[h][s/2] >> ((s&1)*16)) & 0xffffu, q0.sf[h]);
                                        else qbf = q0.bf[h][s];
                                        acc0[h] = mmac_bf16(kbf, qbf, acc0[h]);
                                    }
                                }
                                if (p1.normal && !p1.fp8) {
#pragma unroll
                                    for (int h = 0; h < HT; ++h) {
                                        bf16x4 qbf;
                                        if constexpr (FP4) qbf = fp4_to_bf16(
                                            (q1.bits[h][s/2] >> ((s&1)*16)) & 0xffffu, q1.sf[h]);
                                        else qbf = q1.bf[h][s];
                                        acc1[h] = mmac_bf16(kbf, qbf, acc1[h]);
                                    }
                                }
                            }
                        }
#if defined(__gfx938__)
                        if constexpr (FP4) {
                            if (need_fp8) {
                                uintx4 fp8_lo{0,0,0,0}, fp8_hi{0,0,0,0};
                                if constexpr (Predecoded) {
                                    if (fp8_live) {
                                        fp8_lo = *reinterpret_cast<const uintx4*>(tile.fp8_payload + group * 32);
                                        fp8_hi = *reinterpret_cast<const uintx4*>(tile.fp8_payload + group * 32 + 16);
                                    }
                                }
#pragma unroll
                                for (int s = 0; s < 4; ++s) {
                                    intx2 k8;
                                    if constexpr (Predecoded) {
                                        const uintx4 words = s < 2 ? fp8_lo : fp8_hi;
                                        k8 = intx2{static_cast<int>(words[(s & 1)*2]),
                                                   static_cast<int>(words[(s & 1)*2+1])};
                                    } else {
                                        k8 = fp8_live ? fp4_to_fp8(kb[s], pk - static_cast<int>(sf)) : intx2{0,0};
                                    }
                                    if (p0.fp8) {
#pragma unroll
                                        for (int h = 0; h < HT; ++h)
                                            acc0[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                                k8, q0.f8[h][s], acc0[h], false, false);
                                    }
                                    if (p1.fp8) {
#pragma unroll
                                        for (int h = 0; h < HT; ++h)
                                            acc1[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                                k8, q1.f8[h][s], acc1[h], false, false);
                                    }
                                }
                                if (p0.fp8) residual0 = __uint_as_float(static_cast<unsigned>(q0.pivot + pk - 127) << 23);
                                if (p1.fp8) residual1 = __uint_as_float(static_cast<unsigned>(q1.pivot + pk - 127) << 23);
                            }
                        }
#endif
                    }
                    auto store_score = [&](const RowQ<FP4,HT,CommonSafe>& q, const PrefillQueryPlan& plan,
                                           floatx4 (&acc)[HT], float residual, int i)
                        __attribute__((always_inline)) {
                        if (!plan.row_live) return;
                        if constexpr (FP8Only) {
                            if (!plan.fp8) {
                                if (group == 0 && tile.valid[i])
                                    atomicOr(a.exceptional, plan.normal ? 2 : 1);
                                return;
                            }
                        }
                        if constexpr (BF16Only) {
                            if (plan.fp8) return;
                        }
                        float logit;
                        if constexpr (FallbackOnly) {
                            if (plan.normal) return;
                            logit = exceptional_score<FP4>(a, r+i, group, tile.payload, sf, tile.valid[i]);
                        } else {
                            if (!plan.normal) {
                                if (group == 0 && tile.valid[i]) atomicOr(a.exceptional, 1);
                                return;
                            }
                            if (!q.safe_weight) {
#pragma unroll
                                for (int h = 0; h < HT; ++h) acc[h] *= residual;
                                residual = 1.f;
                            }
                            logit = head_sum<HT>(acc, q.weights) * residual;
                        }
                        if (group == 0 && tile.write[i])
                            a.out[static_cast<int64_t>(r+i) * a.out_stride + tile.slots[i]*BlockKV +
                                  (base + j*16 + pos) % BlockKV] = tile.valid[i] ? logit : -INFINITY;
                    };
                    store_score(q0, p0, acc0, residual0, 0);
                    if (entry.num_q_tokens == 2) store_score(q1, p1, acc1, residual1, 1);
                }
            }
        }
    }
}

template<bool FP4, int HT, int BlockKV, bool FallbackOnly, bool FP8Only = false, bool BF16Only = false, bool FixedLayout = false, bool Predecoded = false, bool CommonSafe = false, bool OverflowOnly = false, bool Interleaved = false, bool WholeBlocks = false, bool UnionFallbackOnly = false, bool ImmediateExceptional = false>
__device__ __forceinline__ void prefill_paged_sparse_logits_body(Args a) {
    if (a.multiq_union_mode && *a.fp8_invalid==0 && *a.index_invalid==0) return;
    if constexpr (OverflowOnly) {
        if (*a.fp8_invalid == 0) return;
    }
    if constexpr (Predecoded) {
        if ((*a.fp8_invalid & 2) != 0) return;
        if constexpr (CommonSafe) {
            if (*a.fp8_invalid != 0) return;
        } else if constexpr (FP8Only) {
            if (*a.fp8_invalid == 0) return;
        }
    }
    if constexpr (CommonSafe) a.metadata=a.metadata_fp8;
    if constexpr (UnionFallbackOnly) {
        // Complete scoring owns index-collision or low-density calls.
        if (*a.index_invalid==0) return;
    }
    if constexpr (FixedLayout) {
        // Host dispatch proves these real-workload dimensions and strides.
        // Constants remove redundant dynamic 64-bit address multipliers and
        // reduce the live launch-argument state; arithmetic is unchanged.
        a.heads = 32; a.capacity = 2048; a.page_kv = 64; a.block_kv = 8;
        a.page_blocks = 8; a.page_block_shift = 3;
        a.q_row_stride = 2048; a.q_head_stride = 64;
        a.sf_row_stride = 32; a.sf_head_stride = 1;
        a.page_stride = 4352; a.out_stride = 16384;
        a.candidate_stride = 2048; a.lens_stride = 1;
    }
    if constexpr (FallbackOnly) {
        if ((*a.exceptional & 1) == 0) return;
    }
    if constexpr (BF16Only) {
        if ((*a.exceptional & 2) == 0) return;
    }
    constexpr int Waves = 2, TilesPerWave = 2;
    static_assert(!Interleaved || (FP4 && HT == 2 && BlockKV == 8 &&
                  !FallbackOnly && FP8Only && !BF16Only && FixedLayout &&
                  Predecoded && CommonSafe && !OverflowOnly),
                  "Interleaved tiles require the prepared FP4 common path");
    // appends WholeBlocks after Interleaved so existing launch arguments
    // retain their meaning. Only the prepared-null-tail dispatch enables it.
    static_assert(!WholeBlocks || (Interleaved && FP4 && HT == 2 && BlockKV == 8 &&
                  !FallbackOnly && FP8Only && !BF16Only && FixedLayout &&
                  Predecoded && CommonSafe && !OverflowOnly),
                  "WholeBlocks requires the interleaved prepared FP4 common path");
    static_assert(!UnionFallbackOnly || (WholeBlocks && CommonSafe && Predecoded &&
                  Interleaved && !FallbackOnly && !OverflowOnly),
                  "UnionFallbackOnly requires the eligible prepared whole path");
    constexpr int BlocksPerSplit = (FP4 ? 640 : 512) / BlockKV;
    constexpr int SplitBytes = sizeof(KVSplitHeader) + BlocksPerSplit * sizeof(KVBlockInfo);
    const auto* header = reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(header->reserved == 0 && header->use_unaligned_ks == 0 &&
           header->num_kv_splits <= a.max_splits && header->num_waves > 0);
    const uint8_t* splits = a.metadata + sizeof(MetadataHeader);
    const auto* schedule = reinterpret_cast<const ScheduleEntry*>(
        splits + static_cast<int64_t>(header->num_kv_splits) * SplitBytes);
    const int64_t entries = static_cast<int64_t>(header->num_waves) * a.num_sms;
    SPARSE_DEVICE_CHECK(sizeof(MetadataHeader) + static_cast<int64_t>(header->num_kv_splits) * SplitBytes +
           entries * sizeof(ScheduleEntry) <= a.metadata_bytes);
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int pos = lane % 16, group = lane / 16;
    // C003: named rows and accumulators avoid dynamic indexing of large
    // per-thread arrays. Tile staging, scheduling and score order stay fixed.
    RowQ<FP4,HT,CommonSafe> q0, q1;
    // More independent CTAs can consume the many Prefill entries. No host
    // read of num_waves is needed, and the producer's schedule width is kept.
    for (int64_t entry_idx = blockIdx.x; entry_idx < entries; entry_idx += gridDim.x) {
        const ScheduleEntry entry = schedule[entry_idx];
        if (entry.num_q_tokens == 0) {
            SPARSE_DEVICE_CHECK(entry.kv_split_begin == entry.kv_split_end);
            continue;
        }
        SPARSE_DEVICE_CHECK(entry.num_q_tokens <= 2 && entry.q_token_base + entry.num_q_tokens <= a.rows &&
               entry.kv_split_begin <= entry.kv_split_end && entry.kv_split_end <= header->num_kv_splits);
        const int r = entry.q_token_base;
        int lengths[2] = {0,0};
        if (!WholeBlocks && a.context_lens) {
            if (lane == 0) {
                lengths[0] = a.context_lens[static_cast<int64_t>(r) * a.lens_stride];
                if (entry.num_q_tokens == 2)
                    lengths[1] = a.context_lens[static_cast<int64_t>(r+1) * a.lens_stride];
            }
            lengths[0] = __shfl(lengths[0], 0, 64);
            lengths[1] = __shfl(lengths[1], 0, 64);
        }
        q0.load(a.q + static_cast<int64_t>(r) * a.q_row_stride,
                a.q_sf + static_cast<int64_t>(r) * a.sf_row_stride,
                a.weights + static_cast<int64_t>(r) * a.weight_stride,
                a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group,
                CommonSafe ? a.q_fp8+static_cast<int64_t>(r)*4112 : nullptr);
        if (entry.num_q_tokens == 2)
            q1.load(a.q + static_cast<int64_t>(r+1) * a.q_row_stride,
                    a.q_sf + static_cast<int64_t>(r+1) * a.sf_row_stride,
                    a.weights + static_cast<int64_t>(r+1) * a.weight_stride,
                    a.q_head_stride, a.sf_head_stride, a.heads, a.fp8_fnuz, pos, group,
                    CommonSafe ? a.q_fp8+static_cast<int64_t>(r+1)*4112 : nullptr);
        int tails[2] = {-1,-1};
        if constexpr (CommonSafe && !WholeBlocks) {
            tails[0]=q0.tail_slot;
            if (entry.num_q_tokens == 2) tails[1]=q1.tail_slot;
        }
        for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++split_idx) {
            const auto* sh = reinterpret_cast<const KVSplitHeader*>(splits + static_cast<int64_t>(split_idx) * SplitBytes);
            const auto* blocks = reinterpret_cast<const KVBlockInfo*>(sh + 1);
            const int count = sh->packed_num_kv_blocks & kBlockCountMask;
            SPARSE_DEVICE_CHECK(sh->q_token_base == entry.q_token_base && count > 0 && count <= BlocksPerSplit &&
                   (sh->packed_num_kv_blocks & ~kBlockCountMask) == 0 &&
                   (entry.num_q_tokens == 2 || sh->q1_slot_base == 0xffffffffu));

            for (int base = wave * TilesPerWave * 16; base < count * BlockKV;
                 base += Waves * TilesPerWave * 16) {
                PrefillKVTile tiles[TilesPerWave];
#pragma unroll
                for (int j = 0; j < TilesPerWave; ++j)
                    tiles[j] = load_prefill_tile<FP4,BlockKV,CommonSafe,WholeBlocks>(
                        a, *sh, blocks, count, entry.num_q_tokens, r, base + j*16 + pos, pos, group, lengths, tails);
#if defined(__gfx938__)
                if constexpr (Interleaved) {
                    // Only the whole-block prepared/common dispatch selects this
                    // specialization. Named tile accumulators keep all indexing
                    // static and provide eight independent MMAC chains. Every
                    // dot still consumes K32 chunks in the original s=0..3 order.
                    const bool live00 = __ballot(tiles[0].valid[0]) != 0;
                    const bool live01 = __ballot(tiles[1].valid[0]) != 0;
                    const bool live10 = entry.num_q_tokens == 2 && __ballot(tiles[0].valid[1]) != 0;
                    const bool live11 = entry.num_q_tokens == 2 && __ballot(tiles[1].valid[1]) != 0;
                    uintx4 k0lo{0,0,0,0}, k0hi{0,0,0,0};
                    uintx4 k1lo{0,0,0,0}, k1hi{0,0,0,0};
                    if (tiles[0].valid[0] || tiles[0].valid[1]) {
                        k0lo = *reinterpret_cast<const uintx4*>(tiles[0].fp8_payload + group*32);
                        k0hi = *reinterpret_cast<const uintx4*>(tiles[0].fp8_payload + group*32 + 16);
                    }
                    if (tiles[1].valid[0] || tiles[1].valid[1]) {
                        k1lo = *reinterpret_cast<const uintx4*>(tiles[1].fp8_payload + group*32);
                        k1hi = *reinterpret_cast<const uintx4*>(tiles[1].fp8_payload + group*32 + 16);
                    }
                    floatx4 acc00[HT], acc01[HT], acc10[HT], acc11[HT];
#pragma unroll
                    for (int h = 0; h < HT; ++h) {
                        acc00[h] = floatx4{0,0,0,0};
                        acc01[h] = floatx4{0,0,0,0};
                        acc10[h] = floatx4{0,0,0,0};
                        acc11[h] = floatx4{0,0,0,0};
                    }
                    auto score_chunk = [&](int s, const uintx4& lo, const uintx4& hi,
                                           floatx4 (&acc0)[HT], floatx4 (&acc1)[HT],
                                           bool live0, bool live1) __attribute__((always_inline)) {
                        const uintx4 words = s < 2 ? lo : hi;
                        const intx2 k8{static_cast<int>(words[(s & 1)*2]),
                                      static_cast<int>(words[(s & 1)*2+1])};
                        if (live0) {
#pragma unroll
                            for (int h = 0; h < HT; ++h)
                                acc0[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                    k8, q0.f8[h][s], acc0[h], false, false);
                        }
                        if (live1) {
#pragma unroll
                            for (int h = 0; h < HT; ++h)
                                acc1[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                    k8, q1.f8[h][s], acc1[h], false, false);
                        }
                    };
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        score_chunk(s, k0lo, k0hi, acc00, acc10, live00, live10);
                        score_chunk(s, k1lo, k1hi, acc01, acc11, live01, live11);
                    }
                    auto store_tile = [&](const PrefillKVTile& tile, int tile_offset,
                                          const RowQ<FP4,HT,CommonSafe>& q,
                                          floatx4 (&acc)[HT], bool live, int i)
                        __attribute__((always_inline)) {
                        if (!live) return;
                        float residual = __uint_as_float(
                            static_cast<unsigned>(q.pivot + static_cast<int>(tile.scales) - 127) << 23);
                        // Retain the original extreme-weight ordering: scale
                        // accumulators before ReLU/FMA whenever the guard says
                        // multiplying unscaled logits by weights is unsafe.
                        if (!q.safe_weight) {
#pragma unroll
                            for (int h = 0; h < HT; ++h) acc[h] *= residual;
                            residual = 1.f;
                        }
                        const float logit = head_sum<HT>(acc, q.weights) * residual;
                        if (group == 0 && tile.write[i])
                            a.out[static_cast<int64_t>(r+i) * a.out_stride + tile.slots[i]*BlockKV +
                                  (base + tile_offset + pos) % BlockKV] = tile.valid[i] ? logit : -INFINITY;
                    };
                    store_tile(tiles[0], 0, q0, acc00, live00, 0);
                    if (entry.num_q_tokens == 2) store_tile(tiles[0], 0, q1, acc10, live10, 1);
                    store_tile(tiles[1], 16, q0, acc01, live01, 0);
                    if (entry.num_q_tokens == 2) store_tile(tiles[1], 16, q1, acc11, live11, 1);
                    continue;
                }
#endif
#pragma unroll
                for (int j = 0; j < TilesPerWave; ++j) {
                    const auto& tile = tiles[j];
                    const unsigned sf = (tile.scales >> (group * 8)) & 255u;
                    const int s0 = tile.scales & 255, s1 = (tile.scales >> 8) & 255;
                    const int s2 = (tile.scales >> 16) & 255, s3 = tile.scales >> 24;
                    const int lo = min(min(s0,s1),min(s2,s3)), hi = max(max(s0,s1),max(s2,s3));
#if defined(__gfx938__)
                    const int pk = CommonSafe ? static_cast<int>(tile.scales) : min(hi,lo+5);
#endif
                    auto plan_query = [&](const RowQ<FP4,HT,CommonSafe>& q, bool valid)
                        __attribute__((always_inline)) {
                        PrefillQueryPlan plan{__ballot(valid) != 0,false,false};
                        if constexpr (CommonSafe) {
                            // Runtime preparation proves all Q/K scales in [96,160]
                            // and each Q-row/K-token span <=11. Joint scales are
                            // [192,320], residual exponents [65,193], so the exact
                            // existing FP8 and normal-dot predicates always hold.
                            plan.normal = plan.fp8 = plan.row_live;
                            return plan;
                        }
                        const bool lane_normal = lo >= 16 && hi <= 240 &&
                            q.smin >= 16 && q.smax <= 240 &&
                            q.smin + lo >= 150 && q.smax + hi <= 350;
                        plan.normal = plan.row_live && __ballot(valid && !lane_normal) == 0;
#if defined(__gfx938__)
                        if constexpr (FP4) {
                            const int e = q.pivot + pk - 127;
                            const bool ok = !valid || (hi-lo <= 11 && e >= 1 && e <= 254);
                            plan.fp8 = plan.normal && q.fp8_ok && __ballot(!ok) == 0;
                        }
#endif
                        return plan;
                    };
                    const PrefillQueryPlan p0 = plan_query(q0, tile.valid[0]);
                    const PrefillQueryPlan p1 = entry.num_q_tokens == 2
                        ? plan_query(q1, tile.valid[1]) : PrefillQueryPlan{false,false,false};
                    floatx4 acc0[HT], acc1[HT];
                    float residual0 = 1.f, residual1 = 1.f;
                    if constexpr (!FallbackOnly) {
#pragma unroll
                        for (int h = 0; h < HT; ++h) {
                            acc0[h] = floatx4{0,0,0,0};
                            acc1[h] = floatx4{0,0,0,0};
                        }
                        const bool need_bf16 = !FP8Only && ((p0.normal && !p0.fp8) || (p1.normal && !p1.fp8));
                        const bool bf16_live = !FP8Only && ((p0.normal && !p0.fp8 && tile.valid[0]) ||
                                               (p1.normal && !p1.fp8 && tile.valid[1]));
                        bool decode_live = bf16_live;
#if defined(__gfx938__)
                        const bool need_fp8 = !BF16Only && (p0.fp8 || p1.fp8);
                        const bool fp8_live = !BF16Only && ((p0.fp8 && tile.valid[0]) || (p1.fp8 && tile.valid[1]));
                        if constexpr (!Predecoded) decode_live |= fp8_live;
#endif
                        uintx4 kb;
                        if constexpr (FP4) kb = decode_live
                            ? *reinterpret_cast<const uintx4*>(tile.payload + group * 16) : uintx4{0,0,0,0};
                        // The fragment is immediately multiplied into both Q
                        // accumulators; only formats needed by a live row exist.
                        if (need_bf16) {
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
                                bf16x4 kbf;
                                if constexpr (FP4) kbf = bf16_live
                                    ? fp4_to_bf16((kb[s/2] >> ((s&1)*16)) & 0xffffu, sf) : bf16x4{0,0,0,0};
                                else kbf = bf16_live
                                    ? fp8_to_bf16(tile.payload, group * 32 + s * 4, sf, a.fp8_fnuz) : bf16x4{0,0,0,0};
                                if (p0.normal && !p0.fp8) {
#pragma unroll
                                    for (int h = 0; h < HT; ++h) {
                                        bf16x4 qbf;
                                        if constexpr (FP4) qbf = fp4_to_bf16(
                                            (q0.bits[h][s/2] >> ((s&1)*16)) & 0xffffu, q0.sf[h]);
                                        else qbf = q0.bf[h][s];
                                        acc0[h] = mmac_bf16(kbf, qbf, acc0[h]);
                                    }
                                }
                                if (p1.normal && !p1.fp8) {
#pragma unroll
                                    for (int h = 0; h < HT; ++h) {
                                        bf16x4 qbf;
                                        if constexpr (FP4) qbf = fp4_to_bf16(
                                            (q1.bits[h][s/2] >> ((s&1)*16)) & 0xffffu, q1.sf[h]);
                                        else qbf = q1.bf[h][s];
                                        acc1[h] = mmac_bf16(kbf, qbf, acc1[h]);
                                    }
                                }
                            }
                        }
#if defined(__gfx938__)
                        if constexpr (FP4) {
                            if (need_fp8) {
                                uintx4 fp8_lo{0,0,0,0}, fp8_hi{0,0,0,0};
                                if constexpr (Predecoded) {
                                    if (fp8_live) {
                                        fp8_lo = *reinterpret_cast<const uintx4*>(tile.fp8_payload + group * 32);
                                        fp8_hi = *reinterpret_cast<const uintx4*>(tile.fp8_payload + group * 32 + 16);
                                    }
                                }
#pragma unroll
                                for (int s = 0; s < 4; ++s) {
                                    intx2 k8;
                                    if constexpr (Predecoded) {
                                        const uintx4 words = s < 2 ? fp8_lo : fp8_hi;
                                        k8 = intx2{static_cast<int>(words[(s & 1)*2]),
                                                   static_cast<int>(words[(s & 1)*2+1])};
                                    } else {
                                        k8 = fp8_live ? fp4_to_fp8(kb[s], pk - static_cast<int>(sf)) : intx2{0,0};
                                    }
                                    if (p0.fp8) {
#pragma unroll
                                        for (int h = 0; h < HT; ++h)
                                            acc0[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                                k8, q0.f8[h][s], acc0[h], false, false);
                                    }
                                    if (p1.fp8) {
#pragma unroll
                                        for (int h = 0; h < HT; ++h)
                                            acc1[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                                                k8, q1.f8[h][s], acc1[h], false, false);
                                    }
                                }
                                if (p0.fp8) residual0 = __uint_as_float(static_cast<unsigned>(q0.pivot + pk - 127) << 23);
                                if (p1.fp8) residual1 = __uint_as_float(static_cast<unsigned>(q1.pivot + pk - 127) << 23);
                            }
                        }
#endif
                    }
                    auto store_score = [&](const RowQ<FP4,HT,CommonSafe>& q, const PrefillQueryPlan& plan,
                                           floatx4 (&acc)[HT], float residual, int i)
                        __attribute__((always_inline)) {
                        if (!plan.row_live) return;
                        if constexpr (FP8Only) {
                            if (!plan.fp8) {
                                if (group == 0 && tile.valid[i])
                                    atomicOr(a.exceptional, plan.normal ? 2 : 1);
                                return;
                            }
                        }
                        if constexpr (BF16Only) {
                            if (plan.fp8) return;
                        }
                        float logit;
                        if constexpr (FallbackOnly) {
                            if (plan.normal) return;
                            logit = exceptional_score<FP4>(a, r+i, group, tile.payload, sf, tile.valid[i]);
                        } else {
                            if constexpr (ImmediateExceptional) {
                                // plan.normal/row_live are wave-uniform. Every
                                // lane participates in exceptional_score's
                                // cross-group shuffles; live masks its dot only.
                                if (!plan.normal) {
                                    logit = exceptional_score<FP4>(a, r+i, group, tile.payload, sf, tile.valid[i]);
                                } else {
                                    if (!q.safe_weight) {
#pragma unroll
                                        for (int h = 0; h < HT; ++h) acc[h] *= residual;
                                        residual = 1.f;
                                    }
                                    logit = head_sum<HT>(acc, q.weights) * residual;
                                }
                            } else {
                            if (!plan.normal) {
                                if (group == 0 && tile.valid[i]) atomicOr(a.exceptional, 1);
                                return;
                            }
                            if (!q.safe_weight) {
#pragma unroll
                                for (int h = 0; h < HT; ++h) acc[h] *= residual;
                                residual = 1.f;
                            }
                            logit = head_sum<HT>(acc, q.weights) * residual;
                            }
                        }
                        if (group == 0 && tile.write[i])
                            a.out[static_cast<int64_t>(r+i) * a.out_stride + tile.slots[i]*BlockKV +
                                  (base + j*16 + pos) % BlockKV] = tile.valid[i] ? logit : -INFINITY;
                    };
                    store_score(q0, p0, acc0, residual0, 0);
                    if (entry.num_q_tokens == 2) store_score(q1, p1, acc1, residual1, 1);
                }
            }
        }
    }
}

// Eligible fixed-layout Prefill only. Q/K prep and remap have finished
// on this stream. Cold work reads original packed Q/K and original metadata.
__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__prefill_fallback(Args a) {
    if (*a.fp8_invalid == 0) return;
    prefill_paged_sparse_logits_body<true,2,8,false,false,false,true,false,false,true,false,false,false,true>(a);
}


// Exact normalized E2M1-to-E4M3 K fragments, prepared once per native call.
// Eligibility and raw scales remain authoritative in the original scorer.
// Allocation, conversion and launch are inside the consumer's measured scope.

// Keep the official metadata unchanged. Deduplicate its selected physical
// blocks into a private per-call bitmap; this is rebuilt on Graph replay.
__global__ void sparse_fp8_fp4_paged_mqa_logits__mark_prefill_selected_blocks(Args a) {
    constexpr int SplitBytes = 16 + 80*8;
    unsigned selected_slots=0,union_records=0;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(header->reserved==0 && header->use_unaligned_ks==0 &&
                        header->num_kv_splits<=a.max_splits);
    for (int split=blockIdx.x;split<header->num_kv_splits;split+=gridDim.x) {
        const auto* sh=reinterpret_cast<const KVSplitHeader*>(a.metadata+16+static_cast<int64_t>(split)*SplitBytes);
        const int count=sh->packed_num_kv_blocks & kBlockCountMask;
        SPARSE_DEVICE_CHECK(count>0 && count<=80);
        if (threadIdx.x<count) {
            const auto* block=reinterpret_cast<const KVBlockInfo*>(sh+1);
            const uint32_t physical=block[threadIdx.x].physical_kv_block_idx;

            if (a.index_invalid) {
                const uint32_t slots=block[threadIdx.x].packed_slot_offsets;
                selected_slots+=(slots&0xffffu)!=kInvalidSlot;
                selected_slots+=(sh->q1_slot_base!=0xffffffffu &&
                                 (slots>>16)!=kInvalidSlot);
            }
            SPARSE_DEVICE_CHECK(physical<static_cast<uint64_t>(a.num_pages)*8);
            atomicExch(a.selected_blocks+physical,1);
            const uint32_t page=physical>>3;
            const bool first_page=threadIdx.x==0 || (block[threadIdx.x-1].physical_kv_block_idx>>3)!=page;
            if (first_page && atomicCAS(a.cache_pages+page,-1,-2)==-1) {
                const int slot=atomicAdd(a.cache_count,1);
                atomicExch(a.cache_pages+page,slot);
                if (slot>=a.cache_capacity && !a.long_packed_cache) atomicOr(a.fp8_invalid,2);
                if(slot<(a.long_packed_cache?a.num_pages:a.cache_capacity)) a.cache_physical_pages[slot]=static_cast<int>(page);
            }
        }
    }
    // Original metadata is read once. Only union-eligible calls count slots;
    // two wave reductions per CTA avoid one global atomic per candidate.
    if (a.index_invalid) {
#pragma unroll
        for (int offset=32;offset>0;offset/=2)
            selected_slots+=__shfl_down(selected_slots,offset,64);
        if ((threadIdx.x&63)==0)
            atomicAdd(reinterpret_cast<unsigned*>(a.index_invalid+1),selected_slots);

    }
}


// Public producer and metadata remain untouched. A private copy has the
// identical header/split/schedule format and slots, with physical page IDs
// relocated to the bounded internal K cache. Raw fallback uses the original.
__global__ void sparse_fp8_fp4_paged_mqa_logits__remap_prefill_cache_metadata(Args a,uint32_t* output) {
    if (*a.fp8_invalid!=0) return;
    constexpr int SplitWords=164;
    const auto* words=reinterpret_cast<const uint32_t*>(a.metadata);
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    const int64_t split_end=4+static_cast<int64_t>(header->num_kv_splits)*SplitWords;
    const int64_t used=split_end+static_cast<int64_t>(header->num_waves)*a.num_sms*4;
    SPARSE_DEVICE_CHECK(used*4<=a.metadata_bytes);
    for (int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         index<used;index+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        uint32_t value=words[index];
        if (index>=4 && index<split_end) {
            const int64_t split=(index-4)/SplitWords;
            const int local=(index-4)%SplitWords;
            if (local>=4 && (local&1)==0) {
                const uint32_t count=words[4+split*SplitWords+1]&kBlockCountMask;
                if ((local-4)/2<count) {
                    const uint32_t page=value>>3;
                    SPARSE_DEVICE_CHECK(page<a.num_pages);
                    const int cached=a.cache_pages[page];
                    SPARSE_DEVICE_CHECK(cached>=0 && cached<2048);
                    value=(static_cast<uint32_t>(cached)<<3)|(value&7);
                }
            }
        }
        output[index]=value;
    }
}


__device__ __forceinline__ void prepare_prefill_k_fp8_partition(Args a, uint8_t* output, int part, int parts) {
    const int pages=*a.cache_count;
    // Mark has completed. Its overflow bit2 is equivalent to pages>2048;
    // use the immutable count to avoid racing Q's concurrent atomic bit1.
    if (pages>2048) return;
    SPARSE_DEVICE_CHECK(pages>=0);
    const int64_t groups=static_cast<int64_t>(pages)*64*4;
    for (int64_t index = static_cast<int64_t>(part)*blockDim.x+threadIdx.x;
         index < groups; index += static_cast<int64_t>(parts)*blockDim.x) {
        const int cached_page=static_cast<int>(index/256);
        const int64_t page=a.cache_physical_pages[cached_page];
        SPARSE_DEVICE_CHECK(page>=0 && page<a.num_pages);
        const int token = (index / 4) % 64, group = index % 4;
        // Only metadata-referenced physical blocks participate. Unselected
        // cache capacity is neither decoded nor allowed to reject FP8.
        if (a.selected_blocks[page*8+token/8] == 0) continue;
        SPARSE_DEVICE_CHECK(a.cache_pages[page]==cached_page);
        const uint8_t* page_ptr = a.kv + page * 4352;
        const unsigned scales = reinterpret_cast<const uint32_t*>(page_ptr+4096)[token];
        const int s0=scales&255, s1=(scales>>8)&255, s2=(scales>>16)&255, s3=scales>>24;
        const int lo=min(min(s0,s1),min(s2,s3)), hi=max(max(s0,s1),max(s2,s3));
        const int pk=min(hi,lo+5), sf=(scales>>(group*8))&255;
        const uintx4 packed=*reinterpret_cast<const uintx4*>(page_ptr+token*64+group*16);
        const bool invalid = lo < 96 || hi > 160 || hi-lo > 11;
        if (invalid && group == 0) atomicOr(a.fp8_invalid,1);
        if (group == 0)
            reinterpret_cast<uint32_t*>(output+static_cast<int64_t>(cached_page)*8448+8192)[token]=static_cast<uint32_t>(pk);
        auto* dst=reinterpret_cast<intx2*>(output+static_cast<int64_t>(cached_page)*8448+token*128+group*32);
#pragma unroll
        for (int word=0;word<4;++word) dst[word]=fp4_to_fp8(packed[word],pk-sf);
    }
}

__device__ __forceinline__ void prepare_prefill_q_fp8_partition(Args a, uint8_t* output, int part, int parts) {
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    for (int row=part*4+wave;row<a.rows;row+=parts*4) {
        const int head=lane%32;
        const unsigned scales=a.q_sf[static_cast<int64_t>(row)*32+head];
        const int s0=scales&255, s1=(scales>>8)&255, s2=(scales>>16)&255, s3=scales>>24;
        const int lo=wave_min(min(min(s0,s1),min(s2,s3)));
        const int hi=wave_max(max(max(s0,s1),max(s2,s3)));
        const int pivot=min(hi,lo+5);
        const float magnitude=fabsf(a.weights[static_cast<int64_t>(row)*a.weight_stride+head]);
        const bool edge=magnitude!=0.f && (magnitude<0x1p-60f || magnitude>0x1p30f);
        const bool safe_weight=__ballot(edge)==0;
        bool bad=lo<96 || hi>160 || hi-lo>11;
        int tail_slot=-1;
        if (a.context_lens) {
            // Strict-tail mode additionally proves that only the last
            // candidate can need per-token masking. Whole-block mode uses
            // the original metadata and never dereferences tail inputs.
            const int length=a.context_lens[row];
            const int count=length>=0 ? static_cast<int>(min(int64_t{2048},(static_cast<int64_t>(length)+7)/8)) : 0;
            const int last=length>0 ? (length-1)/8 : -1;
            bad |= length<0;
            for (int slot=lane;slot<count;slot+=64) {
                const int value=a.candidates[static_cast<int64_t>(row)*2048+slot];
                const int previous=slot>0 ? a.candidates[static_cast<int64_t>(row)*2048+slot-1] : -1;
                bad |= value<0 || value>last || value<=previous;
            }
            if (lane==0) {
                const int newest=count>0 ? a.candidates[static_cast<int64_t>(row)*2048+count-1] : -1;
                tail_slot=count>0 && newest==last ? count-1 : -1;
            }
        }
        const bool invalid=__ballot(bad)!=0;
        if (lane==0) {
            if (invalid) atomicOr(a.fp8_invalid,1);
            // The complete prepared 2Q scorer retains extreme-weight ordering.
            // This flag exists only for whole-union calls and resets each call.
            if (a.index_invalid && !safe_weight) atomicOr(a.index_invalid,4);
            auto* status=reinterpret_cast<int*>(output+static_cast<int64_t>(row)*4112+4096);
            status[0]=pivot; status[1]=safe_weight; status[2]=tail_slot;
        }
#pragma unroll
        for (int offset=0;offset<2;++offset) {
            const int group=(lane/32)*2+offset, sf=(scales>>(group*8))&255;
            const uintx4 packed=*reinterpret_cast<const uintx4*>(a.q+static_cast<int64_t>(row)*2048+head*64+group*16);
            auto* dst=reinterpret_cast<intx2*>(output+static_cast<int64_t>(row)*4112+head*128+group*32);
#pragma unroll
            for (int word=0;word<4;++word) dst[word]=fp4_to_fp8(packed[word],pivot-sf);
        }
    }
}

// Q and K write disjoint prepared buffers. Their rejection bits are atomic;
// the completed mark count is their only shared input and remains immutable.
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx938_prepare_qk_fp8(Args a, uint8_t* kout, uint8_t* qout,
                                       int kparts, int qparts) {
    // Counts are immutable after mark. Decide density in one lane while Q/K
    // partitions prepare disjoint buffers; all rejection-bit writes are atomic.
    // If conversion later rejects FP8, every index-dependent scorer returns
    // before reading this flag and the original raw fallback ignores it.
    if (blockIdx.x==0 && threadIdx.x==0 && a.index_invalid) {
        const int pages=*a.cache_count;
        SPARSE_DEVICE_CHECK(pages>=0);
        if (pages<=2048) {
            const uint64_t slots=*reinterpret_cast<const unsigned*>(a.index_invalid+1);
            const uint64_t positions=static_cast<uint64_t>(pages)*64*a.rows;
            if (slots==0 || positions>slots*8*4) atomicOr(a.index_invalid,2);
        }
    }
    if (blockIdx.x<kparts)
        prepare_prefill_k_fp8_partition(a,kout,blockIdx.x,kparts);
    else
        prepare_prefill_q_fp8_partition(a,qout,blockIdx.x-kparts,qparts);
}

// Original metadata to consumer-only lookup indices. The group mask claims
// each row/physical pair once; only its first writer stores the original slot.
// Any duplicate pair rejects the whole union call, preserving all original
// slots in the prepared 2Q fallback. Slot storage needs no initialization:
// own-row mask bits authorize reads after this kernel completes on the stream.


// Whole-union preparation fuses the unchanged private metadata copy/remap
// with mask-first slot claim. Every input is read from the ORIGINAL blob:
// no worker consumes a header or slot written by another worker in this kernel.
// Do not stop copying on index collisions; the complete private metadata is
// required by the prepared 2Q fallback after this kernel publishes all writes.
// Claim both adjacent query bits in one atomic when they share a group.
// Per-bit previous ownership remains authoritative. Any collision rejects the
// complete union call; prepared metadata is always copied for its 2Q fallback.
__device__ __forceinline__ void claim_query_pair_union(
    Args a,uint32_t qbase,uint32_t q0_base,uint32_t q1_base,
    uint32_t nrows,uint32_t packed_slots,uint32_t physical) {
    constexpr int PhysicalBlocks=16384;
    const uint32_t local0=packed_slots&0xffffu,local1=packed_slots>>16;
    const bool valid0=local0!=kInvalidSlot;
    const bool valid1=nrows==2 && local1!=kInvalidSlot;
    if (valid0) SPARSE_DEVICE_CHECK(q0_base<2048u && local0<2048u-q0_base);
    if (valid1) SPARSE_DEVICE_CHECK(q1_base<2048u && local1<2048u-q1_base);
    const uint32_t row0=qbase,row1=qbase+1u;
    const uint32_t bit0=uint32_t{1}<<(row0&31),bit1=uint32_t{1}<<(row1&31);
    if (valid0 && valid1 && row0/32==row1/32) {
        const uint32_t bits=bit0|bit1;
        const uint32_t previous=atomicOr(
            a.union_mask+static_cast<int64_t>(row0/32)*PhysicalBlocks+physical,bits);
        if ((previous&bits)!=0) atomicOr(a.index_invalid,1);
        if ((previous&bit0)==0)
            a.union_slotmap[static_cast<int64_t>(row0)*PhysicalBlocks+physical]=static_cast<int>(q0_base+local0);
        if ((previous&bit1)==0)
            a.union_slotmap[static_cast<int64_t>(row1)*PhysicalBlocks+physical]=static_cast<int>(q1_base+local1);
    } else {
        if (valid0) {
            const uint32_t previous=atomicOr(
                a.union_mask+static_cast<int64_t>(row0/32)*PhysicalBlocks+physical,bit0);
            if ((previous&bit0)!=0) atomicOr(a.index_invalid,1);
            else a.union_slotmap[static_cast<int64_t>(row0)*PhysicalBlocks+physical]=static_cast<int>(q0_base+local0);
        }
        if (valid1) {
            const uint32_t previous=atomicOr(
                a.union_mask+static_cast<int64_t>(row1/32)*PhysicalBlocks+physical,bit1);
            if ((previous&bit1)!=0) atomicOr(a.index_invalid,1);
            else a.union_slotmap[static_cast<int64_t>(row1)*PhysicalBlocks+physical]=static_cast<int>(q1_base+local1);
        }
    }
}

__global__ void sparse_fp8_fp4_paged_mqa_logits__remap_and_build_query32_union_index(Args a,uint32_t* output) {
    if (*a.fp8_invalid!=0) return;
    constexpr int SplitWords=164,PhysicalBlocks=16384;
    // Only the pre-existing density/weight bits suppress index construction.
    // Collision bit1 must never truncate the complete metadata copy.
    const bool build_union=(*a.index_invalid&(2|4))==0;
    const auto* words=reinterpret_cast<const uint32_t*>(a.metadata);
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(header->reserved==0 && header->use_unaligned_ks==0 &&
                        header->num_kv_splits<=a.max_splits && header->num_waves>0);
    const int64_t split_end=4+static_cast<int64_t>(header->num_kv_splits)*SplitWords;
    const int64_t schedule_entries=static_cast<int64_t>(header->num_waves)*a.num_sms;
    const int64_t used=split_end+schedule_entries*4;
    SPARSE_DEVICE_CHECK(used*4<=a.metadata_bytes);
    // Host validates 16-byte alignment. Every split is 656=41*16 bytes.
    if (blockIdx.x==0 && threadIdx.x==0)
        *reinterpret_cast<uintx4*>(output)=*reinterpret_cast<const uintx4*>(words);
    for (int64_t split=blockIdx.x;split<header->num_kv_splits;split+=gridDim.x) {
        const int64_t base=4+split*SplitWords;
        const uint32_t packed_count=words[base+1];
        const int count=packed_count&kBlockCountMask;
        // Validation includes zero/invalid counts even when no block is active.
        SPARSE_DEVICE_CHECK(count>0 && count<=80 && (packed_count&~kBlockCountMask)==0);
        if (threadIdx.x==0)
            *reinterpret_cast<uintx4*>(output+base)=*reinterpret_cast<const uintx4*>(words+base);
        if (threadIdx.x<80) {
            const int64_t index=base+4+static_cast<int64_t>(threadIdx.x)*2;
            // All 80 records are copied, including count..79. Only selected
            // records are relocated or allowed to contribute query mask bits.
            const uint64_t record=*reinterpret_cast<const uint64_t*>(words+index);
            uint32_t value=static_cast<uint32_t>(record);
            const uint32_t packed_slots=static_cast<uint32_t>(record>>32);
            if (threadIdx.x<count) {
                const uint32_t page=value>>3;
                SPARSE_DEVICE_CHECK(page<a.num_pages);
                const int cached=a.cache_pages[page];
                SPARSE_DEVICE_CHECK(cached>=0 && cached<2048);
                value=(static_cast<uint32_t>(cached)<<3)|(value&7u);
                SPARSE_DEVICE_CHECK(value<PhysicalBlocks);
                if (build_union) {
                    const uint32_t qbase=words[base],q0_base=words[base+2],q1_base=words[base+3];
                    const uint32_t nrows=q1_base==0xffffffffu ? 1u : 2u;
                    SPARSE_DEVICE_CHECK(qbase<=static_cast<uint32_t>(a.rows) &&
                                        nrows<=static_cast<uint32_t>(a.rows)-qbase);
                    claim_query_pair_union(a,qbase,q0_base,q1_base,nrows,packed_slots,value);
                }
            }
            *reinterpret_cast<uint64_t*>(output+index)=static_cast<uint64_t>(value)|
                (static_cast<uint64_t>(packed_slots)<<32);
        }
    }
    // Schedule records follow all splits and are copied verbatim, preserving
    // the full used metadata extent needed by every prepared 2Q fallback.
    for (int64_t entry=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         entry<schedule_entries;entry+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int64_t index=split_end+entry*4;
        *reinterpret_cast<uintx4*>(output+index)=*reinterpret_cast<const uintx4*>(words+index);
    }
}


// General 32Q candidate union; two compact pages form the 128-token tile.
// Lookup indices are derived per call, without a new producer metadata format.
__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx938_fp8_q16(Args a) {
#if defined(__gfx938__)
    if (*a.fp8_invalid!=0 || *a.index_invalid!=0) return;
    constexpr int GroupRows=16, PhysicalBlocks=16384, QStride=136, Cols=128;
    static_assert(GroupRows==16 && Cols==4*2*16);
    const int pages=*a.cache_count;
    /* Proven by private producer/host invariants; ordinary accessguards retained. */
    const int page_pairs=(pages+1)/2;
    const int query_group=blockIdx.y/2, r=blockIdx.y*GroupRows;
    const int subgroup_shift=(blockIdx.y&1)*16;
    /* Proven by private producer/host invariants; ordinary accessguards retained. */
    const int tid=threadIdx.x, wave=tid/64, lane=tid%64;
    const int pos=lane%16, column_group=lane/16;
    __shared__ __align__(16) union {
        uint8_t query[4][GroupRows][QStride];
        float logits[GroupRows][64+4];
    } shared_tile;
    static_assert(sizeof(shared_tile)==4*GroupRows*QStride,
                  "Two output phases must reuse Q LDS without growth");
    __shared__ float weights[32][GroupRows+1];
    __shared__ int q_pivot[GroupRows];
    __shared__ uint32_t block_masks[Cols/8];
    bool input_prepared=false;
    // Empty/odd second pages still take every head barrier and page-pair
    // reuse barrier. Only the mask OR branch can skip the complete pair.
    for (int pair=blockIdx.x;pair<page_pairs;pair+=gridDim.x) {
        if (tid<Cols/8) {
            const int page=pair*2+tid/8;
            block_masks[tid]=page<pages ? a.union_mask[
                static_cast<int64_t>(query_group)*PhysicalBlocks+pair*16+tid]>>subgroup_shift & 0xffffu : 0u;
        }
        __syncthreads();
        uint32_t pair_mask=0;
#pragma unroll
        for (int i=0;i<Cols/8;++i) pair_mask|=block_masks[i];
        if (pair_mask==0) { __syncthreads(); continue; }
        if (!input_prepared) {
            if (tid<GroupRows) {
                if (r+tid<a.rows) {
                    const auto* status=reinterpret_cast<const int*>(
                        a.q_fp8+static_cast<int64_t>(r+tid)*4112+4096);
                    q_pivot[tid]=status[0];
                } else { q_pivot[tid]=127; }
            }
            for (int i=tid;i<32*GroupRows;i+=256) {
                const int row=i/32, head=i%32;
                weights[head][row]=r+row<a.rows ?
                    a.weights[static_cast<int64_t>(r+row)*a.weight_stride+head] : 0.f;
            }
            __syncthreads();
            input_prepared=true;
        }
        uintx4 k0lo{0,0,0,0},k0hi{0,0,0,0},k1lo{0,0,0,0},k1hi{0,0,0,0};
        int k_pivot0=127,k_pivot1=127;
#pragma unroll
        for (int n=0;n<2;++n) {
            const int input_token=wave*32+n*16+pos;
            // Guard BOTH K payload and pivot with the selected block mask.
            // The missing odd final page has mask zero and is never read.
            if (block_masks[input_token/8]!=0) {
                const int page=pair*2+input_token/64, token=input_token%64;
                /* Proven by private producer/host invariants; ordinary accessguards retained. */
                const uint8_t* kp=a.kv_fp8+static_cast<int64_t>(page)*8448+token*128+column_group*32;
                const uintx4 lo=*reinterpret_cast<const uintx4*>(kp);
                const uintx4 hi=*reinterpret_cast<const uintx4*>(kp+16);
                const int pivot=reinterpret_cast<const uint32_t*>(
                    a.kv_fp8+static_cast<int64_t>(page)*8448+8192)[token];
                if (n==0) { k0lo=lo; k0hi=hi; k_pivot0=pivot; }
                else { k1lo=lo; k1hi=hi; k_pivot1=pivot; }
            }
        }
        intx2 key0[4],key1[4];
#pragma unroll
        for (int s=0;s<4;++s) {
            const uintx4 words0=s<2 ? k0lo : k0hi, words1=s<2 ? k1lo : k1hi;
            key0[s]={static_cast<int>(words0[(s&1)*2]),static_cast<int>(words0[(s&1)*2+1])};
            key1[s]={static_cast<int>(words1[(s&1)*2]),static_cast<int>(words1[(s&1)*2+1])};
        }
        floatx4 sum01_00{0,0,0,0},sum01_10{0,0,0,0},sum01_01{0,0,0,0},sum01_11{0,0,0,0};
        floatx4 sum23_00{0,0,0,0},sum23_10{0,0,0,0},sum23_01{0,0,0,0},sum23_11{0,0,0,0};
        // Independently complete g=0..3. The two FMA chains consume
        // heads g,g+8,g+16,g+24 and g+4,g+12,g+20,g+28 respectively.
        // No dynamic partial[head%8] array and no changed addition tree.
#pragma unroll 1
        for (int g=0;g<4;++g) {
            floatx4 even00{0,0,0,0},even10{0,0,0,0},even01{0,0,0,0},even11{0,0,0,0};
            floatx4 odd00{0,0,0,0},odd10{0,0,0,0},odd01{0,0,0,0},odd11{0,0,0,0};
            // Stage a head batch once; keep every original head/FMA order.
            // Retire prior-batch readers before overlapping Q LDS writes.
            __syncthreads();
            {
                const int base_head=0;
                for (int vec=tid;vec<4*GroupRows*8;vec+=256) {
                    const int local_head=vec/(GroupRows*8);
                    const int load_row=(vec/8)%GroupRows, load_col=(vec%8)*16;
                    uintx4 words{0,0,0,0};
                    if (r+load_row<a.rows)
                        words=*reinterpret_cast<const uintx4*>(a.q_fp8+
                            static_cast<int64_t>(r+load_row)*4112+(g+4*(base_head+local_head))*128+load_col);
                    auto* dst=reinterpret_cast<intx2*>(shared_tile.query[local_head][load_row]+load_col);
                    dst[0]=intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                    dst[1]=intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                }
            }
            __syncthreads();
#pragma unroll
            for (int k=0;k<8;++k) {
                const int head=g+4*k, buffer=k%4;
                if (k==4) {
                    __syncthreads();
                    const int base_head=4;
                for (int vec=tid;vec<4*GroupRows*8;vec+=256) {
                    const int local_head=vec/(GroupRows*8);
                    const int load_row=(vec/8)%GroupRows, load_col=(vec%8)*16;
                    uintx4 words{0,0,0,0};
                    if (r+load_row<a.rows)
                        words=*reinterpret_cast<const uintx4*>(a.q_fp8+
                            static_cast<int64_t>(r+load_row)*4112+(g+4*(base_head+local_head))*128+load_col);
                    auto* dst=reinterpret_cast<intx2*>(shared_tile.query[local_head][load_row]+load_col);
                    dst[0]=intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                    dst[1]=intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                }
                    __syncthreads();
                }
                floatx4 dot00{0,0,0,0},dot10{0,0,0,0},dot01{0,0,0,0},dot11{0,0,0,0};
#pragma unroll
                for (int s=0;s<4;++s) {
                    // Preserve D chunk order group*32+s*8. The INT8
                    // reference's group*8+s*32 loader is not interchangeable.
                    const intx2 q0=*reinterpret_cast<const intx2*>(shared_tile.query[buffer][pos]+column_group*32+s*8);
                    const intx2 q1{0,0};
                    dot00=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(q0,key0[s],dot00,false,false);
                    dot10=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(q1,key0[s],dot10,false,false);
                    dot01=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(q0,key1[s],dot01,false,false);
                    dot11=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(q1,key1[s],dot11,false,false);
                }
                const float w0=weights[head][pos], w1=0.f;
#pragma unroll
                for (int j=0;j<4;++j) {
                    if ((k&1)==0) {
                        even00[j]=fmaf(__int_as_float(max(__float_as_int(dot00[j]),0)),w0,even00[j]);
                        even10[j]=fmaf(__int_as_float(max(__float_as_int(dot10[j]),0)),w1,even10[j]);
                        even01[j]=fmaf(__int_as_float(max(__float_as_int(dot01[j]),0)),w0,even01[j]);
                        even11[j]=fmaf(__int_as_float(max(__float_as_int(dot11[j]),0)),w1,even11[j]);
                    } else {
                        odd00[j]=fmaf(__int_as_float(max(__float_as_int(dot00[j]),0)),w0,odd00[j]);
                        odd10[j]=fmaf(__int_as_float(max(__float_as_int(dot10[j]),0)),w1,odd10[j]);
                        odd01[j]=fmaf(__int_as_float(max(__float_as_int(dot01[j]),0)),w0,odd01[j]);
                        odd11[j]=fmaf(__int_as_float(max(__float_as_int(dot11[j]),0)),w1,odd11[j]);
                    }
                }

            }
            const floatx4 sum00=even00+odd00, sum10=even10+odd10;
            const floatx4 sum01=even01+odd01, sum11=even11+odd11;
            if (g==0) { sum01_00=sum00; sum01_10=sum10; sum01_01=sum01; sum01_11=sum11; }
            else if (g==1) { sum01_00+=sum00; sum01_10+=sum10; sum01_01+=sum01; sum01_11+=sum11; }
            else if (g==2) { sum23_00=sum00; sum23_10=sum10; sum23_01=sum01; sum23_11=sum11; }
            else { sum23_00+=sum00; sum23_10+=sum10; sum23_01+=sum01; sum23_11+=sum11; }
        }
        floatx4 logits00=sum01_00+sum23_00, logits10=sum01_10+sum23_10;
        floatx4 logits01=sum01_01+sum23_01, logits11=sum01_11+sum23_11;
        // qprepare excludes unsafe weights from this whole call. Construct
        // each exact power-of-two residual only after the unchanged head tree.
        // N fragments use output columns, never the input-lane pos pivot.
        const int pivot0=q_pivot[pos], pivot1=127;
#pragma unroll
        for (int j=0;j<4;++j) {
            const int output_pivot0=__shfl(k_pivot0,column_group+4*j,64);
            const int output_pivot1=__shfl(k_pivot1,column_group+4*j,64);
            logits00[j]*=__uint_as_float(static_cast<unsigned>(pivot0+output_pivot0-127)<<23);
            logits10[j]*=__uint_as_float(static_cast<unsigned>(pivot1+output_pivot0-127)<<23);
            logits01[j]*=__uint_as_float(static_cast<unsigned>(pivot0+output_pivot1-127)<<23);
            logits11[j]*=__uint_as_float(static_cast<unsigned>(pivot1+output_pivot1-127)<<23);
        }
        // Retire the last Q readers before reusing Q storage for output.
        __syncthreads();
        const bool aligned_output=(reinterpret_cast<uintptr_t>(a.out)&15u)==0;
#pragma unroll
        for (int n=0;n<2;++n) {
#pragma unroll
            for (int m=0;m<1;++m) {
                const int row=pos+m*16;
#pragma unroll
                for (int j=0;j<4;++j)
                    shared_tile.logits[row][wave*16+column_group+4*j]=
                        m==0 ? (n==0 ? logits00[j] : logits01[j]) :
                               (n==0 ? logits10[j] : logits11[j]);
            }
            __syncthreads();
            // Phase n gathers one 16-token N fragment from every wave.
            // Inverse mapping preserves each original page/block/slot.
            for (int vec=tid;vec<GroupRows*64/4;vec+=256) {
                const int row=vec/16, column=(vec%16)*4;
                if (r+row>=a.rows) continue;
                const int output_token=(column/16)*32+n*16+column%16;
                const int block=output_token/8;
                if ((block_masks[block]&(uint32_t{1}<<row))==0) continue;
                const int physical=pair*16+block;
                /* Proven by private producer/host invariants; ordinary accessguards retained. */
                const int slot=a.union_slotmap[static_cast<int64_t>(r+row)*PhysicalBlocks+physical];
                /* Proven by private producer/host invariants; ordinary accessguards retained. */
                const floatx4 value=*reinterpret_cast<const floatx4*>(shared_tile.logits[row]+column);
                float* destination=a.out+static_cast<int64_t>(r+row)*a.out_stride+slot*8+output_token%8;
                if (aligned_output) {
                    *reinterpret_cast<floatx4*>(destination)=value;
                } else {
#pragma unroll
                    for (int j=0;j<4;++j) destination[j]=value[j];
                }
            }
            // All readers finish before the next phase or next page pair.
            __syncthreads();
        }
    }
#else
    (void)a;
#endif
}

__device__ __forceinline__ int sparse_mqa_gfx936_route(const Args& a){
    const uint64_t pages=*a.cache_count,slots=*reinterpret_cast<const unsigned*>(a.index_invalid+1),records=*reinterpret_cast<const unsigned*>(a.index_invalid+2);
    int route=pages<=2048 && pages<=static_cast<uint64_t>(a.cache_capacity) && slots>0 && pages*64*a.rows*2<=slots*8*5 ? 16 : 2;
    if(a.size_force==1 || a.size_force==2)route=a.size_force;
    if(a.size_force==16 && pages<=2048 && pages<=static_cast<uint64_t>(a.cache_capacity))route=16;
    return route;
}

__device__ __forceinline__ bool sparse_mqa_gfx936_use_packed_k(const Args& a){
    if(!a.size_auto || a.rows<=2048 || a.bf16_k_mode==0)return false;
    const int route=sparse_mqa_gfx936_route(a);
    if(route!=1 && route!=2)return false;
    return (a.long_packed_cache && *a.cache_count>a.cache_capacity) || a.bf16_k_mode==1 || *a.cache_count>=a.bf16_k_threshold;
}

__global__ void sparse_fp8_fp4_paged_mqa_logits__select_prefill_bf16_union_density(Args a) {
    if(a.size_auto){a.index_invalid[3]=sparse_mqa_gfx936_route(a);return;}
    if (*a.fp8_invalid!=0) return;
    const uint64_t slots=*reinterpret_cast<const unsigned*>(a.index_invalid+1);
    const int pages=*a.cache_count;
    SPARSE_DEVICE_CHECK(pages>=0 && pages<=2048);
    // A conservative global page-union estimate, not an input-context gate.
    // Choose between two sparse consumers; every selected output survives.
    // The 1GiB union preparation budget bounds the uint32 slot counter.
    const uint64_t estimated_positions=static_cast<uint64_t>(pages)*64*a.rows;
    if (slots==0 || estimated_positions>slots*8*4)
        atomicOr(a.index_invalid,2);
}

// Private consumer-only gfx936 BF16 multi-Q path. Public metadata is unchanged.
// A call either completes here or falls back in full to the original scorer.
__global__ void sparse_fp8_fp4_paged_mqa_logits__guard_prefill_multiq_bf16(Args a) {
    const int pages=*a.cache_count;
    if ((pages>a.cache_capacity && !a.long_packed_cache) || (*a.fp8_invalid&2)!=0) return;
    SPARSE_DEVICE_CHECK(pages>=0);
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<int64_t>(a.rows)*32;
         i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const uint32_t scales=a.q_sf[i];
        bool bad=false;
#pragma unroll
        for (int g=0;g<4;++g) {
            const unsigned sf=(scales>>(g*8))&255u;
            bad|=sf<96 || sf>160;
        }
        const float magnitude=fabsf(a.weights[(i/32)*a.weight_stride+i%32]);
        bad|=!isfinite(magnitude) ||
            (magnitude!=0.f && (magnitude<0x1p-60f || magnitude>0x1p30f));
        if (bad) atomicOr(a.fp8_invalid,1);
    }
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<int64_t>(pages)*64;
         i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int cached=static_cast<int>(i/64),token=static_cast<int>(i%64);
        const int64_t physical_page=a.cache_physical_pages[cached];
        SPARSE_DEVICE_CHECK(physical_page>=0 && physical_page<a.num_pages);
        // Never inspect an unselected physical block, including padding pages.
        if (a.selected_blocks[physical_page*8+token/8]==0) continue;
        const uint32_t scales=reinterpret_cast<const uint32_t*>(
            a.kv+physical_page*4352+4096)[token];
        bool bad=false;
#pragma unroll
        for (int g=0;g<4;++g) {
            const unsigned sf=(scales>>(g*8))&255u;
            bad|=sf<96 || sf>160;
        }
        if (bad) atomicOr(a.fp8_invalid,1);
    }
}

// Build the same private row/physical lookup without copying/remapping the blob.
// Only this private index uses compact physical page IDs; raw fallback retains
// the caller's original physical IDs and all original output slots.
__global__ void sparse_fp8_fp4_paged_mqa_logits__build_prefill_multiq_raw_index(Args a) {
    if(a.size_auto && a.index_invalid[3]!=16)return;
    if (*a.fp8_invalid!=0 || (*a.index_invalid&(2|4))!=0) return;
    constexpr int SplitBytes=656,PhysicalBlocks=16384;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    SPARSE_DEVICE_CHECK(header->reserved==0 && header->use_unaligned_ks==0 &&
                        header->num_kv_splits<=a.max_splits && header->num_waves>0);
    const int64_t used=16+static_cast<int64_t>(header->num_kv_splits)*SplitBytes+
        static_cast<int64_t>(header->num_waves)*a.num_sms*16;
    SPARSE_DEVICE_CHECK(used<=a.metadata_bytes);
    for (int split=blockIdx.x;split<header->num_kv_splits;split+=gridDim.x) {
        const auto* sh=reinterpret_cast<const KVSplitHeader*>(
            a.metadata+16+static_cast<int64_t>(split)*SplitBytes);
        const int count=sh->packed_num_kv_blocks&kBlockCountMask;
        SPARSE_DEVICE_CHECK(count>0 && count<=80 &&
                            (sh->packed_num_kv_blocks&~kBlockCountMask)==0);
        const uint32_t nrows=sh->q1_slot_base==0xffffffffu ? 1u : 2u;
        SPARSE_DEVICE_CHECK(sh->q_token_base<=static_cast<uint32_t>(a.rows) &&
                            nrows<=static_cast<uint32_t>(a.rows)-sh->q_token_base);
        const auto* blocks=reinterpret_cast<const KVBlockInfo*>(sh+1);
        for (int j=threadIdx.x;j<count;j+=blockDim.x) {
            const KVBlockInfo info=blocks[j];
            const uint32_t page=info.physical_kv_block_idx>>3;
            SPARSE_DEVICE_CHECK(page<a.num_pages);
            const int cached=a.cache_pages[page];
            SPARSE_DEVICE_CHECK(cached>=0 && cached<2048);
            const uint32_t physical=(static_cast<uint32_t>(cached)<<3)|
                (info.physical_kv_block_idx&7u);
            SPARSE_DEVICE_CHECK(physical<PhysicalBlocks);
            claim_query_pair_union(a,sh->q_token_base,sh->q0_slot_base,
                sh->q1_slot_base,nrows,info.packed_slot_offsets,physical);
        }
    }
}

// Expand each original Q exactly once per consumer invocation. This private
// cache belongs to the call and is regenerated on every stream/Graph replay.
__global__ void sparse_fp8_fp4_paged_mqa_logits__prepare_prefill_q_bf16(Args a,uint8_t* output) {
    if (*a.fp8_invalid!=0 || *a.index_invalid!=0) return;
    const int64_t groups=static_cast<int64_t>(a.rows)*32*4;
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<groups;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int row=static_cast<int>(i/128),head=static_cast<int>((i/4)%32);
        const int group=static_cast<int>(i%4);
        const uintx4 packed=*reinterpret_cast<const uintx4*>(
            a.q+static_cast<int64_t>(row)*2048+head*64+group*16);
        const unsigned sf=(a.q_sf[static_cast<int64_t>(row)*32+head]>>(group*8))&255u;
        auto* dst=reinterpret_cast<bf16x4*>(output+
            static_cast<int64_t>(row)*8192+head*256+group*64);
#pragma unroll
        for (int s=0;s<8;++s)
            dst[s]=fp4_to_bf16((packed[s/2]>>((s&1)*16))&0xffffu,sf);
    }
}

// Mark completed before this launch. Q conversion, range checks and the
// density decision do not consume one another's intermediate results. Any
// rejecting atomic bit suppresses the complete union consumer after this
// kernel boundary; prepared Q is private and never read by the raw fallback.
template<bool FragmentLayout>
__device__ __forceinline__ void sparse_mqa_gfx936_prepare_q_bf16_checked_body(Args a,uint8_t* output) {
    const int pages=*a.cache_count;
    if ((pages>a.cache_capacity && !a.long_packed_cache) || (*a.fp8_invalid&2)!=0) return;
    SPARSE_DEVICE_CHECK(pages>=0);
    const bool packed_k=sparse_mqa_gfx936_use_packed_k(a);
    if (blockIdx.x==0 && threadIdx.x==0) {
        const uint64_t slots=*reinterpret_cast<const unsigned*>(a.index_invalid+1);
        const uint64_t positions=static_cast<uint64_t>(pages)*64*a.rows;
        if(a.size_auto)a.index_invalid[3]=sparse_mqa_gfx936_route(a);
        else if (slots==0 || positions>slots*8*4) atomicOr(a.index_invalid,2);
    }
    const int64_t groups=static_cast<int64_t>(a.rows)*32*4;
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<groups;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int row=static_cast<int>(i/128),head=static_cast<int>((i/4)%32);
        const int group=static_cast<int>(i%4);
        const unsigned scales=a.q_sf[i/4];
        if (group==0) {
            const float magnitude=fabsf(a.weights[static_cast<int64_t>(row)*a.weight_stride+head]);
            bool bad=!isfinite(magnitude) ||
                (magnitude!=0.f && (magnitude<0x1p-60f || magnitude>0x1p30f));
#pragma unroll
            for (int g=0;g<4;++g) {
                const unsigned sf=(scales>>(g*8))&255u;
                bad|=sf<96 || sf>160;
            }
            if (bad) atomicOr(a.fp8_invalid,1);
        }
        const unsigned sf=(scales>>(group*8))&255u;
        if (sf>=96 && sf<=160) {
            const uintx4 packed=*reinterpret_cast<const uintx4*>(
                a.q+static_cast<int64_t>(row)*2048+head*64+group*16);
            if constexpr(FragmentLayout) {
            const bool fragments=a.bf16_q_fragment && sparse_mqa_gfx936_route(a)==2;
#pragma unroll
            for (int j=0;j<8;++j) {
                const int64_t offset=fragments
                    ?static_cast<int64_t>(row/2)*16384+(head/8)*4096+(group*8+j)*128+((row%2)*8+head%8)*8
                    :static_cast<int64_t>(row)*8192+head*256+group*64+j*8;
                *reinterpret_cast<bf16x4*>(output+offset)=fp4_to_bf16((packed[j/2]>>((j&1)*16))&0xffffu,sf);
            }
            } else {
                auto* dst=reinterpret_cast<bf16x4*>(output+
                    static_cast<int64_t>(row)*8192+head*256+group*64);
#pragma unroll
                for (int j=0;j<8;++j)
                    dst[j]=fp4_to_bf16((packed[j/2]>>((j&1)*16))&0xffffu,sf);
            }
        }
    }
    if(a.bf16_fused_prep){
        for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
            i<static_cast<int64_t>(pages)*256;i+=static_cast<int64_t>(gridDim.x)*blockDim.x){
            const int cached=i/256,token=(i/4)%64,group=i%4;
            const int64_t page=a.cache_physical_pages[cached];
            SPARSE_DEVICE_CHECK(page>=0 && page<a.num_pages);
            if(a.selected_blocks[page*8+token/8]==0)continue;
            const uint8_t* input=a.kv+page*4352;
            const unsigned sf=(reinterpret_cast<const uint32_t*>(input+4096)[token]>>(group*8))&255u;
            if(sf<96 || sf>160){atomicOr(a.fp8_invalid,1);continue;}
            if(packed_k)continue;
            const uintx4 packed=*reinterpret_cast<const uintx4*>(input+token*64+group*16);
            bf16x4 decoded[8];sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
#pragma unroll
            for(int j=0;j<8;++j)*reinterpret_cast<bf16x4*>(const_cast<uint8_t*>(a.k_bf16)+static_cast<int64_t>(cached)*16384+(group*8+j)*512+token*8)=decoded[j];
        }
    }else{
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<int64_t>(pages)*64;
         i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int cached=static_cast<int>(i/64),token=static_cast<int>(i%64);
        const int64_t page=a.cache_physical_pages[cached];
        SPARSE_DEVICE_CHECK(page>=0 && page<a.num_pages);
        if (a.selected_blocks[page*8+token/8]==0) continue;
        const unsigned scales=reinterpret_cast<const uint32_t*>(a.kv+page*4352+4096)[token];
        bool bad=false;
#pragma unroll
        for (int g=0;g<4;++g) {
            const unsigned sf=(scales>>(g*8))&255u;
            bad|=sf<96 || sf>160;
        }
        if (bad) atomicOr(a.fp8_invalid,1);
    }
    }
}
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_bf16_operands_checked(Args a,uint8_t* output) {
    sparse_mqa_gfx936_prepare_q_bf16_checked_body<false>(a,output);
}
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_bf16_fragment_operands_checked(Args a,uint8_t* output) {
    sparse_mqa_gfx936_prepare_q_bf16_checked_body<true>(a,output);
}

template<int Waves=4,int QMode=0,int QStride=136,bool FragmentK=false,
         bool DirectStore=false,bool RegisterWeights=false,bool FlatSplit=false,bool Transpose=false,bool PackedK=false,int KeyGroups=2>
__device__ __forceinline__ void sparse_mqa_gfx936_bf16_direct_body(Args a){
#if defined(__gfx936__)
    if(*a.fp8_invalid || *a.index_invalid)return;
    if(a.size_auto && a.index_invalid[3]!=2)return;
    if constexpr(PackedK){if(!sparse_mqa_gfx936_use_packed_k(a))return;}
    else {if(sparse_mqa_gfx936_use_packed_k(a))return;}
    constexpr int Threads=Waves*64, BlocksPerTile=Waves*2*KeyGroups;
    const int t=threadIdx.x,wave=t/64,lane=t%64,pos=lane%16,cg=lane/16;
    const int qr=pos/8,hp=pos%8;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    const auto* splits=a.metadata+16;
    const auto* schedule=reinterpret_cast<const ScheduleEntry*>(splits+static_cast<int64_t>(header->num_kv_splits)*656);
    const int entries=FlatSplit ? header->num_kv_splits : header->num_waves*a.num_sms;
    __shared__ __align__(16) struct {
        uint8_t q[1][16][8];
        float out[2][Waves*16+4];
        float weights[32][3];
        int qp[2];uint32_t block[640];uint16_t slot[2][640];
    } sm;
    for(int ticket=blockIdx.x;ticket<entries;ticket+=gridDim.x){
        ScheduleEntry entry;
        if constexpr(FlatSplit){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(ticket)*656);
            entry={static_cast<uint32_t>(ticket),static_cast<uint32_t>(ticket+1),sh->q_token_base,sh->q1_slot_base==0xffffffffu?1u:2u};
        }else entry=schedule[ticket];
        if(entry.num_q_tokens==0 || entry.kv_split_end==entry.kv_split_begin)continue;
        const int r=entry.q_token_base;
        SPARSE_DEVICE_CHECK(r<a.rows && entry.num_q_tokens<=static_cast<uint32_t>(a.rows-r));
        if constexpr(!RegisterWeights){
            for(int i=t;i<64;i+=Threads){const int row=i/32,h=i%32;sm.weights[h][row]=row<entry.num_q_tokens?a.weights[static_cast<int64_t>(r+row)*a.weight_stride+h]:0.f;}
        }
        float weight_regs[4][Transpose?4:1]{};
        if constexpr(RegisterWeights){
#pragma unroll
            for(int step=0;step<4;++step){
                if constexpr(Transpose){
#pragma unroll
                    for(int j=0;j<4;++j)if(j/2<entry.num_q_tokens)weight_regs[step][j]=a.weights[static_cast<int64_t>(r+j/2)*a.weight_stride+step*8+cg+(j%2)*4];
                }else if(qr<entry.num_q_tokens)weight_regs[step][0]=a.weights[static_cast<int64_t>(r+qr)*a.weight_stride+step*8+hp];
            }
        }
        bf16x4 qregs[4][8]{};
        if constexpr(QMode==1){
#pragma unroll
            for(int step=0;step<4;++step){
#pragma unroll
                for(int s=0;s<8;++s)if(qr<entry.num_q_tokens)qregs[step][s]=*reinterpret_cast<const bf16x4*>(a.q_bf16+(a.bf16_q_fragment ?static_cast<int64_t>((r+qr)/2)*16384+step*4096+(cg*8+s)*128+(((r+qr)%2)*8+hp)*8:static_cast<int64_t>(r+qr)*8192+(step*8+hp)*256+cg*64+s*8));
            }
        }
        if constexpr(QMode==0){
            for(int vec=t;vec<4*16*8;vec+=Threads){
                const int step=vec/128,p=(vec/8)%16,col=(vec%8)*16,row=p/8;
                uintx4 value{};
                if(row<entry.num_q_tokens)value=*reinterpret_cast<const uintx4*>(a.q_bf16+static_cast<int64_t>(r+row)*8192+(step*8+p%8)*256+col);
                auto* dst=reinterpret_cast<intx2*>(sm.q[step][p]+col);
                dst[0]={static_cast<int>(value[0]),static_cast<int>(value[1])};dst[1]={static_cast<int>(value[2]),static_cast<int>(value[3])};
            }
        }
        __syncthreads();
        const auto* last=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_end-1)*656);
        const int count=(entry.kv_split_end-entry.kv_split_begin-1)*80+(last->packed_num_kv_blocks&kBlockCountMask);
        SPARSE_DEVICE_CHECK(count<=640);
            for(int b=t;b<count;b+=Threads){
                sm.block[b]=-1;sm.slot[0][b]=-1;sm.slot[1][b]=-1;
                if(b<count){
                    const int split=entry.kv_split_begin+b/80;
                    const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(split)*656);
                    const KVBlockInfo record=reinterpret_cast<const KVBlockInfo*>(sh+1)[b%80];
                    if constexpr(PackedK) {
                        sm.block[b]=record.physical_kv_block_idx;
                    } else {
                        const int cached=a.cache_pages[record.physical_kv_block_idx/8];
                        SPARSE_DEVICE_CHECK(cached>=0 && cached<a.cache_capacity);
                        sm.block[b]=cached*8+(record.physical_kv_block_idx%8);
                    }
                    const unsigned s0=record.packed_slot_offsets&0xffffu,s1=record.packed_slot_offsets>>16;
                    if(s0!=kInvalidSlot)sm.slot[0][b]=sh->q0_slot_base+s0;
                    if(entry.num_q_tokens==2 && s1!=kInvalidSlot)sm.slot[1][b]=sh->q1_slot_base+s1;
                }
            }
            __syncthreads();
        for(int tile=0;tile<(count+BlocksPerTile-1)/BlocksPerTile;++tile){
            bf16x4 keys[KeyGroups][8]{};int pivots[KeyGroups]{};
#pragma unroll
            for(int n=0;n<KeyGroups;++n){
                const int token=tile*(Waves*16*KeyGroups)+wave*16*KeyGroups+n*16+pos,b=token/8<count?sm.block[token/8]:-1;
                if(b>=0 && static_cast<int64_t>(b)<(PackedK?a.num_pages:a.cache_capacity)*8){
                    const int page=b/8,kt=(b%8)*8+token%8;
                    if constexpr(PackedK){
                        const int64_t physical=page;
                        SPARSE_DEVICE_CHECK(physical>=0 && physical<a.num_pages);
                        const uint8_t* input=a.kv+physical*4352;
                        const uintx4 packed=*reinterpret_cast<const uintx4*>(input+kt*64+cg*16);
                        const unsigned sf=(reinterpret_cast<const uint32_t*>(input+4096)[kt]>>(cg*8))&255u;
                        bf16x4 decoded[8];sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
#pragma unroll
                        for(int j=0;j<8;++j)keys[n][j]=decoded[j];
                    }else{
#pragma unroll
                        for(int j=0;j<8;++j)keys[n][j]=*reinterpret_cast<const bf16x4*>(a.k_bf16+static_cast<int64_t>(page)*16384+(cg*8+j)*512+kt*8);
                    }
                }
            }
            floatx4 accum[KeyGroups]{};
#pragma unroll
            for(int step=0;step<4;++step){
                if constexpr(QMode==2){
                    __syncthreads();
                    for(int vec=t;vec<16*8;vec+=Threads){
                        const int p=vec/8,col=(vec%8)*16,row=p/8;
                        uintx4 value{};
                        if(row<entry.num_q_tokens)value=*reinterpret_cast<const uintx4*>(a.q_bf16+static_cast<int64_t>(r+row)*8192+(step*8+p%8)*256+col);
                        auto* dst=reinterpret_cast<intx2*>(sm.q[0][p]+col);
                        dst[0]={static_cast<int>(value[0]),static_cast<int>(value[1])};dst[1]={static_cast<int>(value[2]),static_cast<int>(value[3])};
                    }
                    __syncthreads();
                }
                floatx4 dots[KeyGroups]{};
#pragma unroll
                for(int s=0;s<8;++s){
                    bf16x4 q{};
                    if constexpr(QMode==1)q=qregs[step][s];
                    else if constexpr(QMode==3){if(qr<entry.num_q_tokens)q=*reinterpret_cast<const bf16x4*>(a.q_bf16+(a.bf16_q_fragment ?static_cast<int64_t>((r+qr)/2)*16384+step*4096+(cg*8+s)*128+(((r+qr)%2)*8+hp)*8:static_cast<int64_t>(r+qr)*8192+(step*8+hp)*256+cg*64+s*8));}
                    else q=*reinterpret_cast<const bf16x4*>(sm.q[QMode==2?0:step][pos]+cg*64+s*8);
#pragma unroll
                    for(int n=0;n<KeyGroups;++n){
                        if constexpr(Transpose)dots[n]=mmac_bf16(keys[n][s],q,dots[n]);
                        else dots[n]=mmac_bf16(q,keys[n][s],dots[n]);
                    }
                }
#pragma unroll
                for(int n=0;n<KeyGroups;++n){
#pragma unroll
                    for(int j=0;j<4;++j){
                        float w;
                        if constexpr(Transpose)w=RegisterWeights?weight_regs[step][j]:sm.weights[step*8+cg+(j%2)*4][j/2];
                        else w=RegisterWeights?weight_regs[step][0]:sm.weights[step*8+hp][qr];
                        accum[n][j]=fmaf(fmaxf(dots[n][j],0.f),w,accum[n][j]);
                    }
                }
            }
            if constexpr(Transpose){
#pragma unroll
                for(int n=0;n<KeyGroups;++n){
#pragma unroll
                    for(int row=0;row<2;++row){
                        const float paired=accum[n][row*2]+accum[n][row*2+1];
                        const float pair01=paired+__shfl_xor(paired,16,64);
                        const float reduced=pair01+__shfl_xor(pair01,32,64);
                        const float value=reduced;
                        if(cg==0 && row<entry.num_q_tokens){
                            const int token=tile*(Waves*16*KeyGroups)+wave*16*KeyGroups+n*16+pos,slot=token/8<count?sm.slot[row][token/8]:-1;
                            if(slot>=0 && slot<2048)a.out[static_cast<int64_t>(r+row)*a.out_stride+slot*8+token%8]=value;
                        }
                    }
                }
            }else{
            floatx4 result[KeyGroups];
#pragma unroll
            for(int n=0;n<KeyGroups;++n){
#pragma unroll
                for(int j=0;j<4;++j){
                    const float paired=accum[n][j]+__shfl_xor(accum[n][j],4,64);
                    const float pair01=paired+__shfl_xor(paired,1,64);
                    const float reduced=pair01+__shfl_xor(pair01,2,64);
                    result[n][j]=reduced;
                }
            }
            __syncthreads();
#pragma unroll
            for(int n=0;n<KeyGroups;++n){
                if constexpr(DirectStore){
                    if(hp==0 && qr<entry.num_q_tokens){
#pragma unroll
                        for(int j=0;j<4;++j){
                            const int token=wave*32+n*16+cg+4*j,slot=sm.slot[qr][token/8];
                            if(slot>=0 && slot<2048)a.out[static_cast<int64_t>(r+qr)*a.out_stride+slot*8+token%8]=result[n][j];
                        }
                    }
                }else{
                    if(hp==0){
#pragma unroll
                        for(int j=0;j<4;++j)sm.out[qr][wave*16+cg+4*j]=result[n][j];
                    }
                    __syncthreads();
                    for(int vec=t;vec<2*(Waves*16)/4;vec+=Threads){
                        const int row=vec/(Waves*4),col=(vec%(Waves*4))*4;
                        const int token=(col/16)*32+n*16+col%16,slot=sm.slot[row][token/8];
                        if(row<entry.num_q_tokens && slot>=0){
                            const floatx4 value=*reinterpret_cast<const floatx4*>(sm.out[row]+col);
                            float* dst=a.out+static_cast<int64_t>(r+row)*a.out_stride+slot*8+token%8;
                            if((reinterpret_cast<uintptr_t>(dst)&15u)==0)*reinterpret_cast<floatx4*>(dst)=value;
                            else for(int j=0;j<4;++j)dst[j]=value[j];
                        }
                    }
                }
                __syncthreads();
            }
            }
        }
        __syncthreads();
    }
#endif
}

template<int Waves=2,bool FragmentK=false,bool PackedK=false>
__device__ __forceinline__ void sparse_mqa_gfx936_bf16_single_body(Args a){
#if defined(__gfx936__)
    if(*a.fp8_invalid || *a.index_invalid)return;
    if(!a.size_auto || a.index_invalid[3]!=1)return;
    if constexpr(PackedK){if(!sparse_mqa_gfx936_use_packed_k(a))return;}
    else if(sparse_mqa_gfx936_use_packed_k(a))return;
    constexpr int Threads=Waves*64,MaxBlocks=640,BlocksPerTile=Waves*4;
    const int t=threadIdx.x,lane=t%64,wave=t/64,pos=lane%16,cg=lane/16;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    const auto* splits=a.metadata+16;
    const auto* schedule=reinterpret_cast<const ScheduleEntry*>(splits+static_cast<int64_t>(header->num_kv_splits)*656);
    const int entries=header->num_waves*a.num_sms;
    __shared__ uint64_t packed_blocks[MaxBlocks];__shared__ int wave_sums[Waves],total_blocks;
    for(int ticket=blockIdx.x;ticket<entries*2;ticket+=gridDim.x){
        const int qid=ticket%2;const ScheduleEntry entry=schedule[ticket/2];
        if(qid>=entry.num_q_tokens || entry.kv_split_begin==entry.kv_split_end)continue;
        const int r=entry.q_token_base+qid;
        const auto* last=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_end-1)*656);
        const int count=(entry.kv_split_end-entry.kv_split_begin-1)*80+(last->packed_num_kv_blocks&kBlockCountMask);
        SPARSE_DEVICE_CHECK(count<=MaxBlocks && r<a.rows);
        const int span=(count+Threads-1)/Threads,begin=t*span,end=min(begin+span,count);
        int local_count=0;
        for(int i=begin;i<end;++i){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_begin+i/80)*656);
            const unsigned packed=reinterpret_cast<const KVBlockInfo*>(sh+1)[i%80].packed_slot_offsets;
            local_count+=((packed>>(qid*16))&0xffffu)!=kInvalidSlot;
        }
        int scan=local_count;
#pragma unroll
        for(int delta=1;delta<64;delta*=2){const int previous=__shfl_up(scan,delta,64);if(lane>=delta)scan+=previous;}
        if(lane==63)wave_sums[wave]=scan;
        __syncthreads();
        int offset=scan-local_count;
#pragma unroll
        for(int w=0;w<Waves;++w)if(w<wave)offset+=wave_sums[w];
        if(t==0){int total=0;for(int w=0;w<Waves;++w)total+=wave_sums[w];total_blocks=total;}
        for(int i=begin;i<end;++i){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_begin+i/80)*656);
            const KVBlockInfo info=reinterpret_cast<const KVBlockInfo*>(sh+1)[i%80];
            const unsigned slot=(info.packed_slot_offsets>>(qid*16))&0xffffu;
            if(slot!=kInvalidSlot){
                uint32_t physical_block;
                if constexpr(PackedK)physical_block=info.physical_kv_block_idx;
                else {
                    const int cached=a.cache_pages[info.physical_kv_block_idx/8];
                    SPARSE_DEVICE_CHECK(cached>=0 && cached<a.cache_capacity);
                    physical_block=static_cast<uint32_t>(cached*8+info.physical_kv_block_idx%8);
                }
                const uint32_t absolute_slot=(qid?sh->q1_slot_base:sh->q0_slot_base)+slot;
                SPARSE_DEVICE_CHECK(absolute_slot<2048);
                packed_blocks[offset]=static_cast<uint64_t>(physical_block)|(static_cast<uint64_t>(absolute_slot)<<32);
                ++offset;
            }
        }
        __syncthreads();
        const int selected=total_blocks;
        if(selected==0)continue;
        bf16x4 qregs[2][8];float weights[2][4];
#pragma unroll
        for(int step=0;step<2;++step){
#pragma unroll
            for(int s=0;s<8;++s)qregs[step][s]=*reinterpret_cast<const bf16x4*>(a.q_bf16+static_cast<int64_t>(r)*8192+(step*16+pos)*256+cg*64+s*8);
#pragma unroll
            for(int j=0;j<4;++j)weights[step][j]=a.weights[static_cast<int64_t>(r)*a.weight_stride+step*16+cg+4*j];
        }
        // blocks/slots are immutable for the entire entry, so math iterations
        // need no CTA barriers or temporary output transpose.
        for(int tile=0;tile<(selected+BlocksPerTile-1)/BlocksPerTile;++tile){
            bf16x4 keys[2][8]{};int pivots[2]={127,127};
#pragma unroll
            for(int n=0;n<2;++n){
                const int token=tile*(Waves*32)+wave*32+n*16+pos,b=token/8;
                if(b<selected){
                    const int cached=packed_blocks[b]&0xffffffffu,page=cached/8,kt=(cached%8)*8+token%8;
if constexpr(PackedK) {
                        SPARSE_DEVICE_CHECK(page>=0 && page<a.num_pages);
                        const uint8_t* input=a.kv+static_cast<int64_t>(page)*4352;
                        const uintx4 packed=*reinterpret_cast<const uintx4*>(input+kt*64+cg*16);
                        const unsigned sf=(reinterpret_cast<const uint32_t*>(input+4096)[kt]>>(cg*8))&255u;
                        bf16x4 decoded[8];sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
#pragma unroll
                        for(int j=0;j<8;++j)keys[n][j]=decoded[j];
                    } else {
#pragma unroll
                        for(int j=0;j<8;++j)keys[n][j]=*reinterpret_cast<const bf16x4*>(a.k_bf16+static_cast<int64_t>(page)*16384+(cg*8+j)*512+kt*8);
                    }
                }
            }
            float even[2]{},odd[2]{};
#pragma unroll
            for(int step=0;step<2;++step){
                floatx4 dots[2]{};
#pragma unroll
                for(int s=0;s<8;++s){
#pragma unroll
                    for(int n=0;n<2;++n)dots[n]=mmac_bf16(keys[n][s],qregs[step][s],dots[n]);
                }
#pragma unroll
                for(int n=0;n<2;++n){
                    even[n]=fmaf(fmaxf(dots[n][0],0.f),weights[step][0],even[n]);
                    even[n]=fmaf(fmaxf(dots[n][2],0.f),weights[step][2],even[n]);
                    odd[n]=fmaf(fmaxf(dots[n][1],0.f),weights[step][1],odd[n]);
                    odd[n]=fmaf(fmaxf(dots[n][3],0.f),weights[step][3],odd[n]);
                }
            }
#pragma unroll
            for(int n=0;n<2;++n){
                const float paired=even[n]+odd[n],pair01=paired+__shfl_xor(paired,16,64);
                const float reduced=pair01+__shfl_xor(pair01,32,64);
                const float value=reduced;
                const int token=tile*(Waves*32)+wave*32+n*16+pos,b=token/8;
                if(cg==0 && b<selected)a.out[static_cast<int64_t>(r)*a.out_stride+(packed_blocks[b]>>32)*8+token%8]=value;
            }
        }
        __syncthreads();
    }
#endif
}




__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w1(Args a){
    sparse_mqa_gfx936_bf16_direct_body<1,1,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w1(Args a){
    sparse_mqa_gfx936_bf16_direct_body<1,3,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w1(Args a){
    sparse_mqa_gfx936_bf16_single_body<1,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w2(Args a){
    sparse_mqa_gfx936_bf16_direct_body<2,1,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w2(Args a){
    sparse_mqa_gfx936_bf16_direct_body<2,3,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w2(Args a){
    sparse_mqa_gfx936_bf16_single_body<2,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w4(Args a){
    sparse_mqa_gfx936_bf16_direct_body<4,1,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w4(Args a){
    sparse_mqa_gfx936_bf16_direct_body<4,3,264,true,true,true,false,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w4(Args a){
    sparse_mqa_gfx936_bf16_single_body<4,true>(a);
}

template<int HeadBatch=4,bool Swizzle=false,bool CachedK=false,bool UseLut=false,int Ilp=1,int Cols=128,bool DirectQ=false>
__device__ __forceinline__ void sparse_mqa_gfx936_bf16_q16_body(Args a) {
    if(a.size_auto && a.index_invalid[3]!=16)return;
    if (*a.fp8_invalid!=0 || *a.index_invalid!=0) return;
    constexpr int GroupRows=16,PhysicalBlocks=16384,QStride=264;
    const int pages=*a.cache_count,page_pairs=(pages+Cols/64-1)/(Cols/64);
    const int query_group=blockIdx.y/2,r=blockIdx.y*GroupRows;
    const int subgroup_shift=(blockIdx.y&1)*16;
    const int tid=threadIdx.x,wave=tid/64,lane=tid%64;
    const int pos=lane%16,group=lane/16;
    __shared__ __align__(16) union {
        uint8_t query[Swizzle ? 1 : HeadBatch][GroupRows][QStride];
        bf16x4 fragments[Swizzle ? HeadBatch : 1][8][4][GroupRows];
        float logits[GroupRows][68];
    } shared_tile;
    __shared__ float weights[32][GroupRows+1];
    __shared__ uint32_t block_masks[Cols/8];
    bool input_prepared=false;
    for (int pair=blockIdx.x;pair<page_pairs;pair+=gridDim.x) {
        if (tid<Cols/8) {
            const int cached=pair*(Cols/64)+tid/8;
            block_masks[tid]=cached<pages ? (a.union_mask[
                static_cast<int64_t>(query_group)*PhysicalBlocks+pair*(Cols/8)+tid]
                >>subgroup_shift)&0xffffu : 0u;
        }
        __syncthreads();
        uint32_t pair_mask=0;
#pragma unroll
        for (int b=0;b<Cols/8;++b) pair_mask|=block_masks[b];
        if (pair_mask==0) { __syncthreads(); continue; }
        if (!input_prepared) {
            for (int i=tid;i<32*GroupRows;i+=256) {
                const int row=i/32,head=i%32;
                weights[head][row]=r+row<a.rows ?
                    a.weights[static_cast<int64_t>(r+row)*a.weight_stride+head] : 0.f;
            }
            __syncthreads();
            input_prepared=true;
        }
        bf16x4 key0[8],key1[8]{};
#pragma unroll
        for (int n=0;n<Cols/64;++n) {
            const int input_token=wave*(Cols/4)+n*16+pos;
            const bool live=block_masks[input_token/8]!=0;
            const int cached=pair*(Cols/64)+input_token/64,token=input_token%64;
            bf16x4 decoded[8];
            if constexpr (CachedK) {
#pragma unroll
                for (int s=0;s<8;++s)
                    decoded[s]=live ? *reinterpret_cast<const bf16x4*>(a.k_bf16+
                        static_cast<int64_t>(cached)*16384+(group*8+s)*512+token*8) : bf16x4{0,0,0,0};
            } else {
                uintx4 packed{0,0,0,0};unsigned sf=127;
                if (live) {
                    const int64_t physical_page=a.cache_physical_pages[cached];
                    const uint8_t* page=a.kv+physical_page*4352;
                    packed=*reinterpret_cast<const uintx4*>(page+token*64+group*16);
                    sf=(reinterpret_cast<const uint32_t*>(page+4096)[token]>>(group*8))&255u;
                }
                if constexpr (UseLut) sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
                else {
#pragma unroll
                    for (int s=0;s<8;++s)
                        decoded[s]=fp4_to_bf16((packed[s/2]>>((s&1)*16))&0xffffu,sf);
                }
            }
#pragma unroll
            for (int s=0;s<8;++s) {
                if (n==0) key0[s]=decoded[s];else key1[s]=decoded[s];
            }
        }
        floatx4 sum01_0{0,0,0,0},sum01_1{0,0,0,0};
        floatx4 sum23_0{0,0,0,0},sum23_1{0,0,0,0};
        // Match the original two FMA chains and head reduction tree exactly.
#pragma unroll 1
        for (int g=0;g<4;++g) {
            floatx4 even0{0,0,0,0},even1{0,0,0,0};
            floatx4 odd0{0,0,0,0},odd1{0,0,0,0};
#pragma unroll
            for (int k=0;k<8;k+=Ilp) {
                if (!DirectQ && k%HeadBatch==0) {
                    __syncthreads();
                    // Prepared Q has restored per-32D scales already. Keep
                    // eight-byte LDS alignment despite the padded 264B rows.
                    for (int vec=tid;vec<HeadBatch*GroupRows*16;vec+=256) {
                        const int local_head=vec/(GroupRows*16);
                        const int row=(vec/16)%GroupRows,load_col=(vec%16)*16;
                        const int head=g+4*((k/HeadBatch)*HeadBatch+local_head);
                        uintx4 words{0,0,0,0};
                        if (r+row<a.rows)
                            words=*reinterpret_cast<const uintx4*>(a.q_bf16+
                                static_cast<int64_t>(r+row)*8192+head*256+load_col);
                        if constexpr (Swizzle) {
                            const int group=load_col/64,s=(load_col%64)/8;
                            *reinterpret_cast<intx2*>(&shared_tile.fragments[local_head][s][group][row^((s/2)+group*4)])=
                                intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                            *reinterpret_cast<intx2*>(&shared_tile.fragments[local_head][s+1][group][row^((s/2)+group*4)])=
                                intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                        } else {
                            auto* dst=reinterpret_cast<intx2*>(shared_tile.query[local_head][row]+load_col);
                            dst[0]=intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                            dst[1]=intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                        }
                    }

                    __syncthreads();
                }
                floatx4 dot0[Ilp]{},dot1[Ilp]{};
#pragma unroll
                for (int s=0;s<8;++s) {
#pragma unroll
                    for (int u=0;u<Ilp;++u) {
                        bf16x4 q;
                        if constexpr (DirectQ) {
                            q=r+pos<a.rows ? *reinterpret_cast<const bf16x4*>(a.q_bf16+
                                static_cast<int64_t>(r+pos)*8192+(g+4*(k+u))*256+group*64+s*8) : bf16x4{0,0,0,0};
                        } else {
                            q=Swizzle ? shared_tile.fragments[(k+u)%HeadBatch][s][group][pos^((s/2)+group*4)] :
                                *reinterpret_cast<const bf16x4*>(shared_tile.query[(k+u)%HeadBatch][pos]+group*64+s*8);
                        }
                        dot0[u]=mmac_bf16(q,key0[s],dot0[u]);
                        if constexpr (Cols==128) dot1[u]=mmac_bf16(q,key1[s],dot1[u]);
                    }
                }
#pragma unroll
                for (int u=0;u<Ilp;++u) {
                    const float weight=weights[g+4*(k+u)][pos];
#pragma unroll
                    for (int j=0;j<4;++j) {
                        if (((k+u)&1)==0) {
                            even0[j]=fmaf(fmaxf(dot0[u][j],0.f),weight,even0[j]);
                            if constexpr (Cols==128) even1[j]=fmaf(fmaxf(dot1[u][j],0.f),weight,even1[j]);
                        } else {
                            odd0[j]=fmaf(fmaxf(dot0[u][j],0.f),weight,odd0[j]);
                            if constexpr (Cols==128) odd1[j]=fmaf(fmaxf(dot1[u][j],0.f),weight,odd1[j]);
                        }
                    }
                }
            }
            const floatx4 sum0=even0+odd0,sum1=even1+odd1;
            if (g==0) { sum01_0=sum0; sum01_1=sum1; }
            else if (g==1) { sum01_0+=sum0; sum01_1+=sum1; }
            else if (g==2) { sum23_0=sum0; sum23_1=sum1; }
            else { sum23_0+=sum0; sum23_1+=sum1; }
        }
        const floatx4 logits0=sum01_0+sum23_0,logits1=sum01_1+sum23_1;
        __syncthreads();
        const bool aligned_output=(reinterpret_cast<uintptr_t>(a.out)&15u)==0;
#pragma unroll
        for (int n=0;n<Cols/64;++n) {
#pragma unroll
            for (int j=0;j<4;++j)
                shared_tile.logits[pos][wave*16+group+4*j]=
                    n==0 ? logits0[j] : logits1[j];
            __syncthreads();
            for (int vec=tid;vec<GroupRows*64/4;vec+=256) {
                const int row=vec/16,column=(vec%16)*4;
                if (r+row>=a.rows) continue;
                const int output_token=(column/16)*(Cols/4)+n*16+column%16;
                const int block=output_token/8;
                if ((block_masks[block]&(uint32_t{1}<<row))==0) continue;
                const int physical=pair*(Cols/8)+block;
                const int slot=a.union_slotmap[static_cast<int64_t>(r+row)*PhysicalBlocks+physical];
                floatx4 value=*reinterpret_cast<const floatx4*>(shared_tile.logits[row]+column);
                if (a.context_lens) {
                    const int logical=a.candidates[static_cast<int64_t>(r+row)*a.candidate_stride+slot];
                    const int length=a.context_lens[static_cast<int64_t>(r+row)*a.lens_stride];
                    SPARSE_DEVICE_CHECK(logical>=0 && length>=0);
#pragma unroll
                    for (int j=0;j<4;++j)
                        if (static_cast<int64_t>(logical)*8+output_token%8+j>=length)
                            value[j]=-INFINITY;
                }
                float* dst=a.out+static_cast<int64_t>(r+row)*a.out_stride+slot*8+output_token%8;
                if (aligned_output) *reinterpret_cast<floatx4*>(dst)=value;
                else {
#pragma unroll
                    for (int j=0;j<4;++j) dst[j]=value[j];
                }
            }
            __syncthreads();
        }
    }
}


__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q16_legacy(Args a) {
    sparse_mqa_gfx936_bf16_q16_body<4,false,false,false,1>(a);
}
__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q16_k_cache(Args a) {
    sparse_mqa_gfx936_bf16_q16_body<2,false,true,false,1>(a);
}

__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_k_bf16(Args a) {
    if(sparse_mqa_gfx936_use_packed_k(a))return;
    if (*a.fp8_invalid!=0 || *a.index_invalid!=0) return;
    const int pages=*a.cache_count;
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<int64_t>(pages)*64*4;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int page=i/256,token=(i/4)%64,group=i%4;
        const int64_t original=a.cache_physical_pages[page];
        if (a.selected_blocks[original*8+token/8]==0) continue;
        const uint8_t* input=a.kv+original*4352;
        const uintx4 packed=*reinterpret_cast<const uintx4*>(input+token*64+group*16);
        const unsigned sf=(reinterpret_cast<const uint32_t*>(input+4096)[token]>>(group*8))&255u;
        bf16x4 decoded[8];
        sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
#pragma unroll
        for (int s=0;s<8;++s)
            *reinterpret_cast<bf16x4*>(const_cast<uint8_t*>(a.k_bf16)+
                static_cast<int64_t>(page)*16384+(group*8+s)*512+token*8)=decoded[s];
    }
}

template<int HeadBatch=4,bool Swizzle=false,bool CachedK=false,bool UseLut=false,int Ilp=1,int Cols=128,bool DirectQ=false>
__device__ __forceinline__ void sparse_mqa_gfx936_q16_shared(Args a,uint8_t* storage) {
    if(a.size_auto && a.index_invalid[3]!=16)return;
    if (*a.fp8_invalid!=0 || *a.index_invalid!=0) return;
    constexpr int GroupRows=16,PhysicalBlocks=16384,QStride=264;
    const int pages=*a.cache_count,page_pairs=(pages+Cols/64-1)/(Cols/64);
    const int query_group=blockIdx.y/2,r=blockIdx.y*GroupRows;
    const int subgroup_shift=(blockIdx.y&1)*16;
    const int tid=threadIdx.x,wave=tid/64,lane=tid%64;
    const int pos=lane%16,group=lane/16;
    struct alignas(16) SharedStorage {
    union {
        uint8_t query[Swizzle ? 1 : HeadBatch][GroupRows][QStride];
        bf16x4 fragments[Swizzle ? HeadBatch : 1][8][4][GroupRows];
        float logits[GroupRows][68];
    } shared_tile;
    float weights[32][GroupRows+1];
    uint32_t block_masks[Cols/8];
    };
    static_assert(sizeof(SharedStorage)<=11264,"BF16 Q16 shared bound");
    auto& state=*reinterpret_cast<SharedStorage*>(storage);
    auto& shared_tile=state.shared_tile;auto& weights=state.weights;auto& block_masks=state.block_masks;
    bool input_prepared=false;
    for (int pair=blockIdx.x;pair<page_pairs;pair+=gridDim.x) {
        if (tid<Cols/8) {
            const int cached=pair*(Cols/64)+tid/8;
            block_masks[tid]=cached<pages ? (a.union_mask[
                static_cast<int64_t>(query_group)*PhysicalBlocks+pair*(Cols/8)+tid]
                >>subgroup_shift)&0xffffu : 0u;
        }
        __syncthreads();
        uint32_t pair_mask=0;
#pragma unroll
        for (int b=0;b<Cols/8;++b) pair_mask|=block_masks[b];
        if (pair_mask==0) { __syncthreads(); continue; }
        if (!input_prepared) {
            for (int i=tid;i<32*GroupRows;i+=256) {
                const int row=i/32,head=i%32;
                weights[head][row]=r+row<a.rows ?
                    a.weights[static_cast<int64_t>(r+row)*a.weight_stride+head] : 0.f;
            }
            __syncthreads();
            input_prepared=true;
        }
        bf16x4 key0[8],key1[8]{};
#pragma unroll
        for (int n=0;n<Cols/64;++n) {
            const int input_token=wave*(Cols/4)+n*16+pos;
            const bool live=block_masks[input_token/8]!=0;
            const int cached=pair*(Cols/64)+input_token/64,token=input_token%64;
            bf16x4 decoded[8];
            if constexpr (CachedK) {
#pragma unroll
                for (int s=0;s<8;++s)
                    decoded[s]=live ? *reinterpret_cast<const bf16x4*>(a.k_bf16+
                        static_cast<int64_t>(cached)*16384+(group*8+s)*512+token*8) : bf16x4{0,0,0,0};
            } else {
                uintx4 packed{0,0,0,0};unsigned sf=127;
                if (live) {
                    const int64_t physical_page=a.cache_physical_pages[cached];
                    const uint8_t* page=a.kv+physical_page*4352;
                    packed=*reinterpret_cast<const uintx4*>(page+token*64+group*16);
                    sf=(reinterpret_cast<const uint32_t*>(page+4096)[token]>>(group*8))&255u;
                }
                if constexpr (UseLut) sparse_mqa_decode_fp4x32(packed,sparse_mqa_bf16_lut(sf),decoded);
                else {
#pragma unroll
                    for (int s=0;s<8;++s)
                        decoded[s]=fp4_to_bf16((packed[s/2]>>((s&1)*16))&0xffffu,sf);
                }
            }
#pragma unroll
            for (int s=0;s<8;++s) {
                if (n==0) key0[s]=decoded[s];else key1[s]=decoded[s];
            }
        }
        floatx4 sum01_0{0,0,0,0},sum01_1{0,0,0,0};
        floatx4 sum23_0{0,0,0,0},sum23_1{0,0,0,0};
        // Match the original two FMA chains and head reduction tree exactly.
#pragma unroll 1
        for (int g=0;g<4;++g) {
            floatx4 even0{0,0,0,0},even1{0,0,0,0};
            floatx4 odd0{0,0,0,0},odd1{0,0,0,0};
#pragma unroll
            for (int k=0;k<8;k+=Ilp) {
                if (!DirectQ && k%HeadBatch==0) {
                    __syncthreads();
                    // Prepared Q has restored per-32D scales already. Keep
                    // eight-byte LDS alignment despite the padded 264B rows.
                    for (int vec=tid;vec<HeadBatch*GroupRows*16;vec+=256) {
                        const int local_head=vec/(GroupRows*16);
                        const int row=(vec/16)%GroupRows,load_col=(vec%16)*16;
                        const int head=g+4*((k/HeadBatch)*HeadBatch+local_head);
                        uintx4 words{0,0,0,0};
                        if (r+row<a.rows)
                            words=*reinterpret_cast<const uintx4*>(a.q_bf16+
                                static_cast<int64_t>(r+row)*8192+head*256+load_col);
                        if constexpr (Swizzle) {
                            const int group=load_col/64,s=(load_col%64)/8;
                            *reinterpret_cast<intx2*>(&shared_tile.fragments[local_head][s][group][row^((s/2)+group*4)])=
                                intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                            *reinterpret_cast<intx2*>(&shared_tile.fragments[local_head][s+1][group][row^((s/2)+group*4)])=
                                intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                        } else {
                            auto* dst=reinterpret_cast<intx2*>(shared_tile.query[local_head][row]+load_col);
                            dst[0]=intx2{static_cast<int>(words[0]),static_cast<int>(words[1])};
                            dst[1]=intx2{static_cast<int>(words[2]),static_cast<int>(words[3])};
                        }
                    }

                    __syncthreads();
                }
                floatx4 dot0[Ilp]{},dot1[Ilp]{};
#pragma unroll
                for (int s=0;s<8;++s) {
#pragma unroll
                    for (int u=0;u<Ilp;++u) {
                        bf16x4 q;
                        if constexpr (DirectQ) {
                            q=r+pos<a.rows ? *reinterpret_cast<const bf16x4*>(a.q_bf16+
                                static_cast<int64_t>(r+pos)*8192+(g+4*(k+u))*256+group*64+s*8) : bf16x4{0,0,0,0};
                        } else {
                            q=Swizzle ? shared_tile.fragments[(k+u)%HeadBatch][s][group][pos^((s/2)+group*4)] :
                                *reinterpret_cast<const bf16x4*>(shared_tile.query[(k+u)%HeadBatch][pos]+group*64+s*8);
                        }
                        dot0[u]=mmac_bf16(q,key0[s],dot0[u]);
                        if constexpr (Cols==128) dot1[u]=mmac_bf16(q,key1[s],dot1[u]);
                    }
                }
#pragma unroll
                for (int u=0;u<Ilp;++u) {
                    const float weight=weights[g+4*(k+u)][pos];
#pragma unroll
                    for (int j=0;j<4;++j) {
                        if (((k+u)&1)==0) {
                            even0[j]=fmaf(fmaxf(dot0[u][j],0.f),weight,even0[j]);
                            if constexpr (Cols==128) even1[j]=fmaf(fmaxf(dot1[u][j],0.f),weight,even1[j]);
                        } else {
                            odd0[j]=fmaf(fmaxf(dot0[u][j],0.f),weight,odd0[j]);
                            if constexpr (Cols==128) odd1[j]=fmaf(fmaxf(dot1[u][j],0.f),weight,odd1[j]);
                        }
                    }
                }
            }
            const floatx4 sum0=even0+odd0,sum1=even1+odd1;
            if (g==0) { sum01_0=sum0; sum01_1=sum1; }
            else if (g==1) { sum01_0+=sum0; sum01_1+=sum1; }
            else if (g==2) { sum23_0=sum0; sum23_1=sum1; }
            else { sum23_0+=sum0; sum23_1+=sum1; }
        }
        const floatx4 logits0=sum01_0+sum23_0,logits1=sum01_1+sum23_1;
        __syncthreads();
        const bool aligned_output=(reinterpret_cast<uintptr_t>(a.out)&15u)==0;
#pragma unroll
        for (int n=0;n<Cols/64;++n) {
#pragma unroll
            for (int j=0;j<4;++j)
                shared_tile.logits[pos][wave*16+group+4*j]=
                    n==0 ? logits0[j] : logits1[j];
            __syncthreads();
            for (int vec=tid;vec<GroupRows*64/4;vec+=256) {
                const int row=vec/16,column=(vec%16)*4;
                if (r+row>=a.rows) continue;
                const int output_token=(column/16)*(Cols/4)+n*16+column%16;
                const int block=output_token/8;
                if ((block_masks[block]&(uint32_t{1}<<row))==0) continue;
                const int physical=pair*(Cols/8)+block;
                const int slot=a.union_slotmap[static_cast<int64_t>(r+row)*PhysicalBlocks+physical];
                floatx4 value=*reinterpret_cast<const floatx4*>(shared_tile.logits[row]+column);
                if (a.context_lens) {
                    const int logical=a.candidates[static_cast<int64_t>(r+row)*a.candidate_stride+slot];
                    const int length=a.context_lens[static_cast<int64_t>(r+row)*a.lens_stride];
                    SPARSE_DEVICE_CHECK(logical>=0 && length>=0);
#pragma unroll
                    for (int j=0;j<4;++j)
                        if (static_cast<int64_t>(logical)*8+output_token%8+j>=length)
                            value[j]=-INFINITY;
                }
                float* dst=a.out+static_cast<int64_t>(r+row)*a.out_stride+slot*8+output_token%8;
                if (aligned_output) *reinterpret_cast<floatx4*>(dst)=value;
                else {
#pragma unroll
                    for (int j=0;j<4;++j) dst[j]=value[j];
                }
            }
            __syncthreads();
        }
    }
}

template<int Waves=4,int QMode=0,int QStride=136,bool FragmentK=false,
         bool DirectStore=false,bool RegisterWeights=false,bool FlatSplit=false,bool Transpose=false>
__device__ __forceinline__ void sparse_mqa_gfx936_q2_shared(Args a,uint8_t* storage){
#if defined(__gfx936__)
    if(*a.fp8_invalid || *a.index_invalid)return;
    if(a.size_auto && a.index_invalid[3]!=2)return;
    constexpr int Threads=Waves*64, BlocksPerTile=Waves*4;
    const int t=threadIdx.x,wave=t/64,lane=t%64,pos=lane%16,cg=lane/16;
    const int qr=pos/8,hp=pos%8;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    const auto* splits=a.metadata+16;
    const auto* schedule=reinterpret_cast<const ScheduleEntry*>(splits+static_cast<int64_t>(header->num_kv_splits)*656);
    const int entries=FlatSplit ? header->num_kv_splits : header->num_waves*a.num_sms;
    struct alignas(16) SharedStorage {
        uint8_t q[1][16][8];
        float out[2][Waves*16+4];
        float weights[32][3];
        int qp[2];uint32_t block[640];uint16_t slot[2][640];
    };
    static_assert(sizeof(SharedStorage)<=11264,"BF16 Q2 shared bound");
    auto& sm=*reinterpret_cast<SharedStorage*>(storage);
    for(int ticket=blockIdx.y*gridDim.x+blockIdx.x;ticket<entries;ticket+=gridDim.x*gridDim.y){
        ScheduleEntry entry;
        if constexpr(FlatSplit){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(ticket)*656);
            entry={static_cast<uint32_t>(ticket),static_cast<uint32_t>(ticket+1),sh->q_token_base,sh->q1_slot_base==0xffffffffu?1u:2u};
        }else entry=schedule[ticket];
        if(entry.num_q_tokens==0 || entry.kv_split_end==entry.kv_split_begin)continue;
        const int r=entry.q_token_base;
        SPARSE_DEVICE_CHECK(r<a.rows && entry.num_q_tokens<=static_cast<uint32_t>(a.rows-r));
        if constexpr(!RegisterWeights){
            for(int i=t;i<64;i+=Threads){const int row=i/32,h=i%32;sm.weights[h][row]=row<entry.num_q_tokens?a.weights[static_cast<int64_t>(r+row)*a.weight_stride+h]:0.f;}
        }
        float weight_regs[4][Transpose?4:1]{};
        if constexpr(RegisterWeights){
#pragma unroll
            for(int step=0;step<4;++step){
                if constexpr(Transpose){
#pragma unroll
                    for(int j=0;j<4;++j)if(j/2<entry.num_q_tokens)weight_regs[step][j]=a.weights[static_cast<int64_t>(r+j/2)*a.weight_stride+step*8+cg+(j%2)*4];
                }else if(qr<entry.num_q_tokens)weight_regs[step][0]=a.weights[static_cast<int64_t>(r+qr)*a.weight_stride+step*8+hp];
            }
        }
        bf16x4 qregs[4][8]{};
        if constexpr(QMode==1){
#pragma unroll
            for(int step=0;step<4;++step){
#pragma unroll
                for(int s=0;s<8;++s)if(qr<entry.num_q_tokens)qregs[step][s]=*reinterpret_cast<const bf16x4*>(a.q_bf16+static_cast<int64_t>(r+qr)*8192+(step*8+hp)*256+cg*64+s*8);
            }
        }
        if constexpr(QMode==0){
            for(int vec=t;vec<4*16*8;vec+=Threads){
                const int step=vec/128,p=(vec/8)%16,col=(vec%8)*16,row=p/8;
                uintx4 value{};
                if(row<entry.num_q_tokens)value=*reinterpret_cast<const uintx4*>(a.q_bf16+static_cast<int64_t>(r+row)*8192+(step*8+p%8)*256+col);
                auto* dst=reinterpret_cast<intx2*>(sm.q[step][p]+col);
                dst[0]={static_cast<int>(value[0]),static_cast<int>(value[1])};dst[1]={static_cast<int>(value[2]),static_cast<int>(value[3])};
            }
        }
        __syncthreads();
        const auto* last=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_end-1)*656);
        const int count=(entry.kv_split_end-entry.kv_split_begin-1)*80+(last->packed_num_kv_blocks&kBlockCountMask);
        SPARSE_DEVICE_CHECK(count<=640);
            for(int b=t;b<count;b+=Threads){
                sm.block[b]=-1;sm.slot[0][b]=-1;sm.slot[1][b]=-1;
                if(b<count){
                    const int split=entry.kv_split_begin+b/80;
                    const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(split)*656);
                    const KVBlockInfo record=reinterpret_cast<const KVBlockInfo*>(sh+1)[b%80];
                    const int cached=a.cache_pages[record.physical_kv_block_idx/8];
                    SPARSE_DEVICE_CHECK(cached>=0 && cached<a.cache_capacity);
                    sm.block[b]=cached*8+(record.physical_kv_block_idx%8);
                    const unsigned s0=record.packed_slot_offsets&0xffffu,s1=record.packed_slot_offsets>>16;
                    if(s0!=kInvalidSlot)sm.slot[0][b]=sh->q0_slot_base+s0;
                    if(entry.num_q_tokens==2 && s1!=kInvalidSlot)sm.slot[1][b]=sh->q1_slot_base+s1;
                }
            }
            __syncthreads();
        for(int tile=0;tile<(count+BlocksPerTile-1)/BlocksPerTile;++tile){
            bf16x4 keys[2][8]{};int pivots[2]={127,127};
#pragma unroll
            for(int n=0;n<2;++n){
                const int token=tile*(Waves*32)+wave*32+n*16+pos,b=token/8<count?sm.block[token/8]:-1;
                if(b>=0 && b<a.cache_capacity*8){
                    const int page=b/8,kt=(b%8)*8+token%8;
#pragma unroll
                    for(int s=0;s<8;++s)keys[n][s]=*reinterpret_cast<const bf16x4*>(a.k_bf16+static_cast<int64_t>(page)*16384+(cg*8+s)*512+kt*8);
                }
            }
            floatx4 accum[2]{};
#pragma unroll
            for(int step=0;step<4;++step){
                if constexpr(QMode==2){
                    __syncthreads();
                    for(int vec=t;vec<16*8;vec+=Threads){
                        const int p=vec/8,col=(vec%8)*16,row=p/8;
                        uintx4 value{};
                        if(row<entry.num_q_tokens)value=*reinterpret_cast<const uintx4*>(a.q_bf16+static_cast<int64_t>(r+row)*8192+(step*8+p%8)*256+col);
                        auto* dst=reinterpret_cast<intx2*>(sm.q[0][p]+col);
                        dst[0]={static_cast<int>(value[0]),static_cast<int>(value[1])};dst[1]={static_cast<int>(value[2]),static_cast<int>(value[3])};
                    }
                    __syncthreads();
                }
                floatx4 dots[2]{};
#pragma unroll
                for(int s=0;s<8;++s){
                    bf16x4 q{};
                    if constexpr(QMode==1)q=qregs[step][s];
                    else if constexpr(QMode==3){if(qr<entry.num_q_tokens)q=*reinterpret_cast<const bf16x4*>(a.q_bf16+static_cast<int64_t>(r+qr)*8192+(step*8+hp)*256+cg*64+s*8);}
                    else q=*reinterpret_cast<const bf16x4*>(sm.q[QMode==2?0:step][pos]+cg*64+s*8);
#pragma unroll
                    for(int n=0;n<2;++n){
                        if constexpr(Transpose)dots[n]=mmac_bf16(keys[n][s],q,dots[n]);
                        else dots[n]=mmac_bf16(q,keys[n][s],dots[n]);
                    }
                }
#pragma unroll
                for(int n=0;n<2;++n){
#pragma unroll
                    for(int j=0;j<4;++j){
                        float w;
                        if constexpr(Transpose)w=RegisterWeights?weight_regs[step][j]:sm.weights[step*8+cg+(j%2)*4][j/2];
                        else w=RegisterWeights?weight_regs[step][0]:sm.weights[step*8+hp][qr];
                        accum[n][j]=fmaf(fmaxf(dots[n][j],0.f),w,accum[n][j]);
                    }
                }
            }
            if constexpr(Transpose){
#pragma unroll
                for(int n=0;n<2;++n){
#pragma unroll
                    for(int row=0;row<2;++row){
                        const float paired=accum[n][row*2]+accum[n][row*2+1];
                        const float pair01=paired+__shfl_xor(paired,16,64);
                        const float reduced=pair01+__shfl_xor(pair01,32,64);
                        const float value=reduced;
                        if(cg==0 && row<entry.num_q_tokens){
                            const int token=tile*(Waves*32)+wave*32+n*16+pos,slot=token/8<count?sm.slot[row][token/8]:-1;
                            if(slot>=0 && slot<2048)a.out[static_cast<int64_t>(r+row)*a.out_stride+slot*8+token%8]=value;
                        }
                    }
                }
            }else{
            floatx4 result[2];
#pragma unroll
            for(int n=0;n<2;++n){
#pragma unroll
                for(int j=0;j<4;++j){
                    const float paired=accum[n][j]+__shfl_xor(accum[n][j],4,64);
                    const float pair01=paired+__shfl_xor(paired,1,64);
                    const float reduced=pair01+__shfl_xor(pair01,2,64);
                    result[n][j]=reduced;
                }
            }
            __syncthreads();
#pragma unroll
            for(int n=0;n<2;++n){
                if constexpr(DirectStore){
                    if(hp==0 && qr<entry.num_q_tokens){
#pragma unroll
                        for(int j=0;j<4;++j){
                            const int token=wave*32+n*16+cg+4*j,slot=sm.slot[qr][token/8];
                            if(slot>=0 && slot<2048)a.out[static_cast<int64_t>(r+qr)*a.out_stride+slot*8+token%8]=result[n][j];
                        }
                    }
                }else{
                    if(hp==0){
#pragma unroll
                        for(int j=0;j<4;++j)sm.out[qr][wave*16+cg+4*j]=result[n][j];
                    }
                    __syncthreads();
                    for(int vec=t;vec<2*(Waves*16)/4;vec+=Threads){
                        const int row=vec/(Waves*4),col=(vec%(Waves*4))*4;
                        const int token=(col/16)*32+n*16+col%16,slot=sm.slot[row][token/8];
                        if(row<entry.num_q_tokens && slot>=0){
                            const floatx4 value=*reinterpret_cast<const floatx4*>(sm.out[row]+col);
                            float* dst=a.out+static_cast<int64_t>(r+row)*a.out_stride+slot*8+token%8;
                            if((reinterpret_cast<uintptr_t>(dst)&15u)==0)*reinterpret_cast<floatx4*>(dst)=value;
                            else for(int j=0;j<4;++j)dst[j]=value[j];
                        }
                    }
                }
                __syncthreads();
            }
            }
        }
        __syncthreads();
    }
#endif
}

template<int Waves=2,bool FragmentK=false>
__device__ __forceinline__ void sparse_mqa_gfx936_q1_shared(Args a,uint8_t* storage){
#if defined(__gfx936__)
    if(*a.fp8_invalid || *a.index_invalid)return;
    if(!a.size_auto || a.index_invalid[3]!=1)return;
    constexpr int Threads=Waves*64,MaxBlocks=640,BlocksPerTile=Waves*4;
    const int t=threadIdx.x,lane=t%64,wave=t/64,pos=lane%16,cg=lane/16;
    const auto* header=reinterpret_cast<const MetadataHeader*>(a.metadata);
    const auto* splits=a.metadata+16;
    const auto* schedule=reinterpret_cast<const ScheduleEntry*>(splits+static_cast<int64_t>(header->num_kv_splits)*656);
    const int entries=header->num_waves*a.num_sms;
    struct alignas(16) SharedStorage {uint64_t packed_blocks[MaxBlocks];int wave_sums[Waves],total_blocks;};
    static_assert(sizeof(SharedStorage)<=11264,"BF16 Q1 shared bound");
    auto& state=*reinterpret_cast<SharedStorage*>(storage);auto& packed_blocks=state.packed_blocks;auto& wave_sums=state.wave_sums;auto& total_blocks=state.total_blocks;
    for(int ticket=blockIdx.y*gridDim.x+blockIdx.x;ticket<entries*2;ticket+=gridDim.x*gridDim.y){
        const int qid=ticket%2;const ScheduleEntry entry=schedule[ticket/2];
        if(qid>=entry.num_q_tokens || entry.kv_split_begin==entry.kv_split_end)continue;
        const int r=entry.q_token_base+qid;
        const auto* last=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_end-1)*656);
        const int count=(entry.kv_split_end-entry.kv_split_begin-1)*80+(last->packed_num_kv_blocks&kBlockCountMask);
        SPARSE_DEVICE_CHECK(count<=MaxBlocks && r<a.rows);
        const int span=(count+Threads-1)/Threads,begin=t*span,end=min(begin+span,count);
        int local_count=0;
        for(int i=begin;i<end;++i){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_begin+i/80)*656);
            const unsigned packed=reinterpret_cast<const KVBlockInfo*>(sh+1)[i%80].packed_slot_offsets;
            local_count+=((packed>>(qid*16))&0xffffu)!=kInvalidSlot;
        }
        int scan=local_count;
#pragma unroll
        for(int delta=1;delta<64;delta*=2){const int previous=__shfl_up(scan,delta,64);if(lane>=delta)scan+=previous;}
        if(lane==63)wave_sums[wave]=scan;
        __syncthreads();
        int offset=scan-local_count;
#pragma unroll
        for(int w=0;w<Waves;++w)if(w<wave)offset+=wave_sums[w];
        if(t==0){int total=0;for(int w=0;w<Waves;++w)total+=wave_sums[w];total_blocks=total;}
        for(int i=begin;i<end;++i){
            const auto* sh=reinterpret_cast<const KVSplitHeader*>(splits+static_cast<int64_t>(entry.kv_split_begin+i/80)*656);
            const KVBlockInfo info=reinterpret_cast<const KVBlockInfo*>(sh+1)[i%80];
            const unsigned slot=(info.packed_slot_offsets>>(qid*16))&0xffffu;
            if(slot!=kInvalidSlot){
                const int cached=a.cache_pages[info.physical_kv_block_idx/8];
                SPARSE_DEVICE_CHECK(cached>=0 && cached<a.cache_capacity);
                const uint32_t absolute_slot=(qid?sh->q1_slot_base:sh->q0_slot_base)+slot;
                SPARSE_DEVICE_CHECK(absolute_slot<2048);
                packed_blocks[offset]=static_cast<uint64_t>(cached*8+info.physical_kv_block_idx%8)|(static_cast<uint64_t>(absolute_slot)<<32);
                ++offset;
            }
        }
        __syncthreads();
        const int selected=total_blocks;
        if(selected==0)continue;
        bf16x4 qregs[2][8];float weights[2][4];
#pragma unroll
        for(int step=0;step<2;++step){
#pragma unroll
            for(int s=0;s<8;++s)qregs[step][s]=*reinterpret_cast<const bf16x4*>(a.q_bf16+static_cast<int64_t>(r)*8192+(step*16+pos)*256+cg*64+s*8);
#pragma unroll
            for(int j=0;j<4;++j)weights[step][j]=a.weights[static_cast<int64_t>(r)*a.weight_stride+step*16+cg+4*j];
        }
        // blocks/slots are immutable for the entire entry, so math iterations
        // need no CTA barriers or temporary output transpose.
        for(int tile=0;tile<(selected+BlocksPerTile-1)/BlocksPerTile;++tile){
            bf16x4 keys[2][8]{};int pivots[2]={127,127};
#pragma unroll
            for(int n=0;n<2;++n){
                const int token=tile*(Waves*32)+wave*32+n*16+pos,b=token/8;
                if(b<selected){
                    const int cached=packed_blocks[b]&0xffffffffu,page=cached/8,kt=(cached%8)*8+token%8;
#pragma unroll
                    for(int s=0;s<8;++s)keys[n][s]=*reinterpret_cast<const bf16x4*>(a.k_bf16+static_cast<int64_t>(page)*16384+(cg*8+s)*512+kt*8);
                }
            }
            float even[2]{},odd[2]{};
#pragma unroll
            for(int step=0;step<2;++step){
                floatx4 dots[2]{};
#pragma unroll
                for(int s=0;s<8;++s){
#pragma unroll
                    for(int n=0;n<2;++n)dots[n]=mmac_bf16(keys[n][s],qregs[step][s],dots[n]);
                }
#pragma unroll
                for(int n=0;n<2;++n){
                    even[n]=fmaf(fmaxf(dots[n][0],0.f),weights[step][0],even[n]);
                    even[n]=fmaf(fmaxf(dots[n][2],0.f),weights[step][2],even[n]);
                    odd[n]=fmaf(fmaxf(dots[n][1],0.f),weights[step][1],odd[n]);
                    odd[n]=fmaf(fmaxf(dots[n][3],0.f),weights[step][3],odd[n]);
                }
            }
#pragma unroll
            for(int n=0;n<2;++n){
                const float paired=even[n]+odd[n],pair01=paired+__shfl_xor(paired,16,64);
                const float reduced=pair01+__shfl_xor(pair01,32,64);
                const float value=reduced;
                const int token=tile*(Waves*32)+wave*32+n*16+pos,b=token/8;
                if(cg==0 && b<selected)a.out[static_cast<int64_t>(r)*a.out_stride+(packed_blocks[b]>>32)*8+token%8]=value;
            }
        }
        __syncthreads();
    }
#endif
}


__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_query_adaptive(Args a){
#if defined(__gfx936__)
    if(*a.fp8_invalid || *a.index_invalid)return;
    extern __shared__ __align__(16) uint8_t storage[];
    const int route=a.index_invalid[3];
    if(route==16)sparse_mqa_gfx936_q16_shared<2,false,true,false,1>(a,storage);
    else if(route==2)sparse_mqa_gfx936_q2_shared<4,1,264,true,true,true,false,true>(a,storage);
    else if(route==1)sparse_mqa_gfx936_q1_shared<4,true>(a,storage);
#endif
}


__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w1(Args a){
    sparse_mqa_gfx936_bf16_direct_body<1,1,264,true,true,true,false,true,true,1>(a);
}

__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w1(Args a){
    sparse_mqa_gfx936_bf16_direct_body<1,3,264,true,true,true,false,true,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w2(Args a){
    sparse_mqa_gfx936_bf16_direct_body<2,1,264,true,true,true,false,true,true,1>(a);
}

__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w2(Args a){
    sparse_mqa_gfx936_bf16_direct_body<2,3,264,true,true,true,false,true,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w4(Args a){
    sparse_mqa_gfx936_bf16_direct_body<4,1,264,true,true,true,false,true,true,1>(a);
}

__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w4(Args a){
    sparse_mqa_gfx936_bf16_direct_body<4,3,264,true,true,true,false,true,true>(a);
}

__attribute__((amdgpu_flat_work_group_size(64,64)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w1(Args a){
    sparse_mqa_gfx936_bf16_single_body<1,true,true>(a);
}
__attribute__((amdgpu_flat_work_group_size(128,128)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w2(Args a){
    sparse_mqa_gfx936_bf16_single_body<2,true,true>(a);
}
__attribute__((amdgpu_flat_work_group_size(256,256)))
__global__ void sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w4(Args a){
    sparse_mqa_gfx936_bf16_single_body<4,true,true>(a);
}
int64_t ceil_div(int64_t a, int64_t b) { return (a+b-1)/b; }
}  // namespace

torch::Tensor fp8_fp4_paged_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& qsf, const torch::Tensor& kv,
    const torch::Tensor& weights, const torch::Tensor& metadata,
    int64_t capacity, int64_t block_kv,
    const std::optional<torch::Tensor>& context_lens,
    const std::optional<torch::Tensor>& candidates,
    const std::optional<torch::Tensor>& output, std::optional<int64_t> num_sms_opt,
    bool is_prefill, const PrefillSparseMQAMetadataPlan* metadata_plan = nullptr) {
    TORCH_CHECK(q.is_cuda(), "fp8_fp4_paged_sparse_mqa_logits: Q must be a HIP tensor");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q.device());
    const bool fp4 = q.scalar_type() == at::kChar || q.scalar_type() == at::kByte;
    TORCH_CHECK(fp4 || q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                q.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                "fp8_fp4_paged_sparse_mqa_logits: Q must be packed FP4 int8/uint8 or E4M3 FP8");
    const int pb = fp4 ? 64 : 128;
    TORCH_CHECK((q.dim() == 3 || (q.dim() == 4 && q.size(1) == 1)) &&
                q.size(-1) == pb && q.stride(-1) == 1 && q.stride(-2) >= pb &&
                q.stride(-2)%16 == 0 && q.stride(0) >= 0 && q.stride(0)%16 == 0 &&
                reinterpret_cast<uintptr_t>(q.data_ptr())%16 == 0,
                "fp8_fp4_paged_sparse_mqa_logits: Q must be [R,1,H,64/128] or [R,H,64/128], vector aligned with unit payload stride");
    const int64_t rows = q.size(0), heads = q.size(-2);
    TORCH_CHECK(rows > 0 && rows <= (1<<20) && heads > 0 && heads <= 32 && heads%4 == 0,
                "fp8_fp4_paged_sparse_mqa_logits: R must be in [1,2^20] and H a multiple of four in [4,32]");
    TORCH_CHECK(capacity > 0 && capacity <= 4096 && capacity%4 == 0 && (block_kv == 8 || block_kv == 16),
                "fp8_fp4_paged_sparse_mqa_logits: K must be a positive multiple of four <=4096, sparse_block_kv 8 or 16");
    TORCH_CHECK(qsf.device() == q.device() && qsf.scalar_type() == at::kInt &&
                qsf.dim() == q.dim()-1 && qsf.size(0) == rows && qsf.size(-1) == heads &&
                (qsf.dim() == 2 || qsf.size(1) == 1) && qsf.stride(0) >= 0 && qsf.stride(-1) >= 0,
                "fp8_fp4_paged_sparse_mqa_logits: packed Q scales must be matching int32 [R,1,H] or [R,H]");
    TORCH_CHECK(kv.device() == q.device() && kv.scalar_type() == at::kByte && kv.dim() == 4 &&
                kv.size(0) > 0 && kv.size(1) > 0 && kv.size(1) <= 65536 && kv.size(1)%block_kv == 0 &&
                kv.size(2) == 1 && kv.size(3) == pb+4 && kv.stride(1) == pb+4 && kv.stride(3) == 1 &&
                kv.stride(0) >= kv.size(1)*(pb+4) && kv.stride(0)%16 == 0 &&
                reinterpret_cast<uintptr_t>(kv.data_ptr())%16 == 0,
                "fp8_fp4_paged_sparse_mqa_logits: KV must be aligned uint8 [P,page_kv,1,68/132] with page-local payload/scale SOA");
    TORCH_CHECK(weights.device() == q.device() && weights.scalar_type() == at::kFloat &&
                weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == heads &&
                weights.stride(1) == 1 && weights.stride(0) >= 0,
                "fp8_fp4_paged_sparse_mqa_logits: weights must be FP32 [R,H], unit head stride");
    TORCH_CHECK(context_lens.has_value() == candidates.has_value(),
                "fp8_fp4_paged_sparse_mqa_logits: context_lens and sparse_kv_block_indices must be provided together");
    if (context_lens) {
        TORCH_CHECK(context_lens->device() == q.device() && context_lens->scalar_type() == at::kInt &&
                    context_lens->dim() == 1 && context_lens->size(0) == rows && context_lens->stride(0) >= 0,
                    "fp8_fp4_paged_sparse_mqa_logits: context_lens must be int32 [R]");
        TORCH_CHECK(candidates->device() == q.device() && candidates->scalar_type() == at::kInt &&
                    candidates->dim() == 2 && candidates->size(0) == rows && candidates->size(1) == capacity &&
                    candidates->stride(1) == 1 && candidates->stride(0) >= 0,
                    "fp8_fp4_paged_sparse_mqa_logits: sparse_kv_block_indices must be int32 [R,K], unit slot stride");
    }
    hipDeviceProp_t prop;
    TORCH_CHECK(hipGetDeviceProperties(&prop,q.get_device()) == hipSuccess,
                "fp8_fp4_paged_sparse_mqa_logits: cannot query HIP device");
    const std::string arch(prop.gcnArchName);
    TORCH_CHECK(arch.rfind("gfx938",0) == 0 || arch.rfind("gfx936",0) == 0 || arch.rfind("gfx928",0) == 0,
                "fp8_fp4_paged_sparse_mqa_logits: unsupported architecture ",arch);
    const int64_t num_sms = num_sms_opt.value_or(prop.multiProcessorCount);
    TORCH_CHECK(num_sms > 0 && num_sms <= 256, "fp8_fp4_paged_sparse_mqa_logits: num_sms must be in [1,256]");
    const int64_t blocks_per_split = (fp4 ? 640 : 512)/block_kv;
    const int64_t max_splits = rows*ceil_div(capacity,blocks_per_split);
    const int64_t bytes = 16 + max_splits*(16 + blocks_per_split*8) + ceil_div(max_splits,num_sms)*num_sms*16;
    TORCH_CHECK(metadata.device() == q.device() && metadata.scalar_type() == at::kByte &&
                metadata.dim() == 1 && metadata.is_contiguous() && metadata.numel() >= bytes &&
                reinterpret_cast<uintptr_t>(metadata.data_ptr())%16 == 0,
                "fp8_fp4_paged_sparse_mqa_logits: metadata must be aligned official uint8 blob, need at least ",bytes," bytes");
    const int64_t width = capacity*block_kv;
    torch::Tensor out;
    if (output) {
        out = *output;
        TORCH_CHECK(out.device() == q.device() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                    out.size(0) == rows && out.size(1) == width && out.stride(1) == 1 && out.stride(0) >= width,
                    "fp8_fp4_paged_sparse_mqa_logits: output must be FP32 [R,K*sparse_block_kv] with unit column stride");
        for (const torch::Tensor* input : {&q,&qsf,&kv,&weights,&metadata})
            TORCH_CHECK(!out.is_alias_of(*input), "fp8_fp4_paged_sparse_mqa_logits: output cannot alias inputs");
        if (context_lens)
            TORCH_CHECK(!out.is_alias_of(*context_lens) && !out.is_alias_of(*candidates),
                        "fp8_fp4_paged_sparse_mqa_logits: output cannot alias tail inputs");
    }
    Args a{};
    // A single sparse-call flag keeps exceptional arithmetic out of the
    // common MMAC kernel. The fallback reads the same official metadata and
    // returns immediately when all selected tiles use normal scales.
    // Private storage is assigned below after dispatch eligibility is known.
    a.q = reinterpret_cast<const uint8_t*>(q.data_ptr()); a.q_sf = reinterpret_cast<const uint32_t*>(qsf.data_ptr<int>());
    a.kv = kv.data_ptr<uint8_t>(); a.metadata = metadata.data_ptr<uint8_t>();
    a.weights = weights.data_ptr<float>(); a.out = output ? out.data_ptr<float>() : nullptr;
    a.q_row_stride = q.stride(0); a.q_head_stride = q.stride(-2);
    a.sf_row_stride = qsf.stride(0); a.sf_head_stride = qsf.stride(-1);
    a.page_stride = kv.stride(0); a.weight_stride = weights.stride(0); a.out_stride = output ? out.stride(0) : ceil_div(width,512)*512;
    a.num_pages = kv.size(0); a.metadata_bytes = metadata.numel(); a.max_splits = max_splits;
    a.rows = rows; a.heads = heads; a.capacity = capacity; a.page_kv = kv.size(1); a.block_kv = block_kv;
    a.num_sms = num_sms;
    a.page_blocks = kv.size(1) / block_kv;
    a.page_block_shift = (a.page_blocks & (a.page_blocks-1)) == 0
        ? __builtin_ctz(static_cast<unsigned>(a.page_blocks)) : -1;
    a.fp8_fnuz = q.scalar_type() == at::ScalarType::Float8_e4m3fnuz;
    // Optimized Prefill dispatch is intrinsic to this API. Geometry, GPU
    // metadata and numeric guards still select the supported kernel/fallback.
    a.size_auto=is_prefill && !context_lens && arch.rfind("gfx936",0)==0;
    if(a.size_auto){
        const char* k_mode=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_BF16_K_MODE");a.bf16_k_mode=k_mode?std::atoi(k_mode):2;
        TORCH_CHECK(a.bf16_k_mode>=0 && a.bf16_k_mode<=2,"BF16 K mode must be 0(cached),1(packed),2(footprint-auto)");
        const char* k_cut=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_BF16_K_THRESHOLD");if(k_cut)a.bf16_k_threshold=std::max(1,std::min(65535,std::atoi(k_cut)));
        const char* prep=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_BF16_FUSED_PREP");a.bf16_fused_prep=!prep || std::strcmp(prep,"0")!=0;
        const char* force=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_SIZE_FORCE");a.size_force=force?std::atoi(force):0;
        const char* cap=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_SIZE_CACHE_PAGES");
        a.cache_capacity=static_cast<int>(std::min<int64_t>(cap?std::max(1,std::atoi(cap)):65535,std::max<int64_t>(2048,kv.size(0))));
        a.cache_capacity=std::min(a.cache_capacity,65535);
        const char* long_env=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_LONG_PACKED");
        a.long_packed_cache=a.rows>2048 && a.bf16_k_mode!=0 && (!long_env || std::strcmp(long_env,"0")!=0);
        // Dense 16Q requires at most 2048 selected pages. Above that range,
        // the 2Q consumer reads packed physical K directly. No expanded K
        // allocation or cached-ID representability is needed for those pages.
        if(a.long_packed_cache)a.cache_capacity=std::min(a.cache_capacity,2048);
    }

    if (context_lens) {
        a.context_lens = context_lens->data_ptr<int>(); a.lens_stride = context_lens->stride(0);
        a.candidates = candidates->data_ptr<int>(); a.candidate_stride = candidates->stride(0);
    }
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    bool metadata_ready=metadata_plan==nullptr;
    const auto make_metadata=[&](bool mark) {
        if(metadata_ready) return;
        PrefillSparseMQASelection selection;
        if(mark) {
            selection.selected_blocks=a.selected_blocks;selection.cache_pages=a.cache_pages;
            selection.cache_count=a.cache_count;selection.cache_physical_pages=a.cache_physical_pages;
            selection.index_invalid=a.index_invalid;
            selection.invalid=a.fp8_invalid;selection.selected_slots=a.index_invalid ? a.index_invalid+1 : nullptr;
            selection.num_pages=a.num_pages;selection.cache_capacity=a.cache_capacity;
            selection.long_packed_cache=a.long_packed_cache;
        }
        prepare_prefill_sparse_mqa_logits_metadata(metadata,*metadata_plan,a.page_kv,q.scalar_type(),block_kv,num_sms,selection);
        metadata_ready=true;
    };

    // capacity is a multiple of four and block_kv is 8/16, so width is
    // divisible by four. Capture replays fill the complete fixed output,
    // independent of changing metadata, eligibility or candidate prefix.
    const bool vector_fill = is_prefill && a.out_stride == width &&
        reinterpret_cast<uintptr_t>(a.out)%16 == 0 && width%4 == 0;
    const int head_tiles = ceil_div(heads,16);
    // The fixed-layout consumer already guards every tail read on the
    // optional pointer. Prepared Q/K and cache relocation apply equally to
    // whole-block mode; strict mode retains its original stride proof.
    const bool fixed_prefill_tail_layout = !context_lens ||
        (context_lens->stride(0) == 1 && candidates->stride(0) == 2048);
    const bool fixed_prefill_layout = fp4 && heads == 32 && capacity == 2048 &&
        block_kv == 8 && kv.size(1) == 64 && q.stride(0) == 2048 &&
        q.stride(-2) == 64 && qsf.stride(0) == 32 && qsf.stride(-1) == 1 &&
        kv.stride(0) == 4352 && a.out_stride == 16384 && fixed_prefill_tail_layout;
    torch::Tensor prepared_k, prepared_q, eligibility, selected, cache_pages, prepared_metadata, cache_physical_pages;
    torch::Tensor union_slots, union_masks, index_invalid;
    // Bound only the additional private preparation buffers. rows is already
    // validated <= 2^20, so its Q-buffer term is safe in int64. Compare the
    // metadata before subtraction, and divide the remaining budget before
    // comparing P, avoiding overflow from huge metadata/page-pool inputs.
    constexpr int64_t kPreparationBudgetBytes = int64_t{1} << 30;
    constexpr int64_t kBookkeepingBytesPerPage = (8 + 1) * sizeof(int);
    const int64_t preparation_base_bytes = int64_t{2048} * 64 * 132 +
        rows * 4112 + int64_t{2} * sizeof(int) + int64_t{2048} * sizeof(int);
    bool preparation_budget_ok = preparation_base_bytes <= kPreparationBudgetBytes;
    if (preparation_budget_ok) {
        int64_t remaining = kPreparationBudgetBytes - preparation_base_bytes;
        if (metadata.numel() > remaining) {
            preparation_budget_ok = false;
        } else {
            remaining -= metadata.numel();
            preparation_budget_ok = kv.size(0) <= remaining / kBookkeepingBytesPerPage;
        }
    }
    const bool prepare_k_fp8 = is_prefill && fixed_prefill_layout && rows >= 64 &&
        arch.rfind("gfx938",0) == 0 && preparation_budget_ok;
    // API rows<=2^20 bounds all products in int64. Union budget rejection
    // preserves the independent original prepared-versus-raw decision.
    const int64_t union_groups=ceil_div(rows,32);
    const int64_t union_workspace_bytes=rows*int64_t{16384}*sizeof(uint16_t)+
        union_groups*int64_t{16384}*sizeof(uint32_t)+int64_t{2}*sizeof(int);
    const int64_t union_base_bytes=preparation_base_bytes+union_workspace_bytes;
    bool union_budget_ok=union_base_bytes<=kPreparationBudgetBytes;
    if (union_budget_ok) {
        int64_t remaining=kPreparationBudgetBytes-union_base_bytes;
        if (metadata.numel()>remaining) union_budget_ok=false;
        else {
            remaining-=metadata.numel();
            union_budget_ok=kv.size(0)<=remaining/kBookkeepingBytesPerPage;
        }
    }
    const bool prepare_union=prepare_k_fp8 && !context_lens && union_budget_ok;
    // Keep the independent GPU stages, but own private buffers with one
    // allocator allocation. All initialization still runs on the same stream.
    torch::Tensor workspace,exceptional;
    bool single_workspace=false;
    if (prepare_k_fp8 && vector_fill) {
        const auto align_piece=[](int64_t n) { return (n+255)/256*256; };
        const int64_t private_bytes=align_piece(4)+align_piece(8)+
            align_piece(a.num_pages*4)+align_piece(2048*4)+
            align_piece(rows*4112)+align_piece(a.num_pages*8*4)+
            align_piece(metadata.numel())+align_piece(int64_t{2048}*64*132)+
            (prepare_union ? align_piece(8)+align_piece(rows*int64_t{16384}*2)+
                align_piece(union_groups*int64_t{16384}*4) : 0);
        if (private_bytes<=kPreparationBudgetBytes) {
            const int64_t output_bytes=output ? 0 : align_piece(rows*width*4);
            workspace=torch::empty({private_bytes+output_bytes},kv.options());
            uint8_t* base=workspace.data_ptr<uint8_t>();
            int64_t offset=0;
            const auto take=[&](int64_t bytes) {
                uint8_t* ptr=base+offset;offset+=align_piece(bytes);return ptr;
            };
            a.exceptional=reinterpret_cast<int*>(take(4));
            a.fp8_invalid=reinterpret_cast<int*>(take(8));a.cache_count=a.fp8_invalid+1;
            a.cache_pages=reinterpret_cast<int*>(take(a.num_pages*4));
            a.cache_physical_pages=reinterpret_cast<int*>(take(2048*4));
            a.q_fp8=take(rows*4112);
            a.selected_blocks=reinterpret_cast<int*>(take(a.num_pages*8*4));
            a.metadata_fp8=take(metadata.numel());
            a.kv_fp8=take(int64_t{2048}*64*132);
            if (prepare_union) {
                a.index_invalid=reinterpret_cast<int*>(take(8));
                a.union_slotmap=reinterpret_cast<uint16_t*>(take(rows*int64_t{16384}*2));
                a.union_mask=reinterpret_cast<uint32_t*>(take(union_groups*int64_t{16384}*4));
            }
            if (!output) {
                auto* output_ptr=take(rows*width*4);
                auto output_storage=workspace.storage();
                auto output_impl=c10::make_intrusive<c10::TensorImpl>(
                    std::move(output_storage),workspace.key_set(),
                    caffe2::TypeMeta::Make<float>());
                output_impl->set_storage_offset((output_ptr-base)/sizeof(float));
                output_impl->set_sizes_contiguous({rows,width});
                out=at::Tensor(std::move(output_impl));
                a.out=out.data_ptr<float>();a.out_stride=width;
            }
            TORCH_CHECK(offset==private_bytes+output_bytes,"Sparse workspace layout mismatch");
            single_workspace=true;
        }
    }
    if (!out.defined()) {
        out=torch::empty({ceil_div(rows,2)*2,ceil_div(width,512)*512},q.options().dtype(at::kFloat))
            .slice(0,0,rows).slice(1,0,width);
        a.out=out.data_ptr<float>();a.out_stride=out.stride(0);
    }
    if (!single_workspace) {
        exceptional=torch::empty({1},q.options().dtype(at::kInt));
        a.exceptional=exceptional.data_ptr<int>();
    }
    const char* fusion_env=std::getenv("LIGHTOP_PREFILL_SPARSE_PREP_FUSION");
    const std::string fusion_mode=fusion_env ? fusion_env : "0";
    TORCH_CHECK(fusion_mode=="0" || fusion_mode=="1" || fusion_mode=="2" || fusion_mode=="3",
                "LIGHTOP_PREFILL_SPARSE_PREP_FUSION must be 0..3");
    const bool fused_bf16_setup=is_prefill && fixed_prefill_layout && rows>=64 &&
        arch.rfind("gfx936",0)==0 && vector_fill &&
        (fusion_mode=="1" || fusion_mode=="3");
    const bool fused_setup=prepare_k_fp8 && vector_fill;
    if (!fused_setup && !fused_bf16_setup) {
    if (vector_fill) {
        const int64_t nvec = rows*width/4;
        const int fill_grid = static_cast<int>(std::min<int64_t>(ceil_div(nvec,256),4096));
        sparse_fp8_fp4_paged_mqa_logits__initialize_output_contiguous_vec4<<<fill_grid,256,0,stream>>>(a.out,nvec,a.exceptional);
    } else {
        const int fill_grid = static_cast<int>(std::min<int64_t>(ceil_div(rows*width,256),4096));
        sparse_fp8_fp4_paged_mqa_logits__initialize_output<<<fill_grid,256,0,stream>>>(a.out,a.out_stride,rows,width,a.exceptional);
    }
    TORCH_CHECK(hipGetLastError() == hipSuccess, "fp8_fp4_paged_sparse_mqa_logits: output initialization launch failed");
    }
    // BF16 has no FP8-span restriction. The conservative [96,160] guard
    // proves normal BF16 operands/dots; any rejected call uses raw fallback.
    // BF16-specific actual additional buffers: prepared Q, row/physical slots,
    // group masks, compact physical page IDs, two 2-int flags, and per-pool-page
    // selected-block/map bookkeeping. No private metadata or K cache is made.
    // The measured consumer is the default; unsupported layouts and scale
    // ranges retain their existing guarded fallbacks.
    constexpr bool sparse_opt=true;
    const bool want_bf16_k_cache=sparse_opt && true;
    const int64_t bf16_union_base_bytes=(want_bf16_k_cache ? static_cast<int64_t>(a.cache_capacity)*16384 : 0)+rows*(int64_t{8192}+int64_t{16384}*sizeof(uint16_t))+
        union_groups*int64_t{16384}*sizeof(uint32_t)+static_cast<int64_t>(a.long_packed_cache?a.num_pages:a.cache_capacity)*sizeof(int)+
        int64_t{6}*sizeof(int);
    bool bf16_union_budget_ok=bf16_union_base_bytes<=kPreparationBudgetBytes;
    if (bf16_union_budget_ok) {
        const int64_t remaining=kPreparationBudgetBytes-bf16_union_base_bytes;
        bf16_union_budget_ok=kv.size(0)<=remaining/kBookkeepingBytesPerPage;
    }
    const bool prepare_bf16_union=is_prefill && fixed_prefill_layout && rows>=64 &&
        arch.rfind("gfx936",0)==0 && bf16_union_budget_ok;
    a.bf16_q_fragment=fused_bf16_setup && a.size_auto && a.rows>2048 && (a.rows%2)==0;
    if (fused_bf16_setup && !prepare_bf16_union) {
        const int64_t nvec=rows*width/4;
        const int fill_grid=static_cast<int>(std::min<int64_t>(ceil_div(nvec,256),4096));
        sparse_fp8_fp4_paged_mqa_logits__initialize_output_contiguous_vec4<<<fill_grid,256,0,stream>>>(a.out,nvec,a.exceptional);
        TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 rejected setup initialization failed");
    }
    if (prepare_bf16_union) {
        a.multiq_union_mode=true;
        prepared_q=torch::empty({rows,8192},kv.options());
        a.q_bf16=prepared_q.data_ptr<uint8_t>();
        if (want_bf16_k_cache) {
            prepared_k=torch::empty({a.cache_capacity,16384},kv.options());
            a.k_bf16=prepared_k.data_ptr<uint8_t>();
        }
        eligibility=fused_bf16_setup ? torch::empty({2},q.options().dtype(at::kInt)) : torch::zeros({2},q.options().dtype(at::kInt));
        a.fp8_invalid=eligibility.data_ptr<int>();
        a.cache_count=eligibility.data_ptr<int>()+1;
        cache_pages=fused_bf16_setup ? torch::empty({kv.size(0)},q.options().dtype(at::kInt)) : torch::full({kv.size(0)},-1,q.options().dtype(at::kInt));
        a.cache_pages=cache_pages.data_ptr<int>();
        cache_physical_pages=torch::empty({a.long_packed_cache?a.num_pages:a.cache_capacity},q.options().dtype(at::kInt));
        a.cache_physical_pages=cache_physical_pages.data_ptr<int>();
        selected=fused_bf16_setup ? torch::empty({kv.size(0)*8},q.options().dtype(at::kInt)) : torch::zeros({kv.size(0)*8},q.options().dtype(at::kInt));
        a.selected_blocks=selected.data_ptr<int>();
        index_invalid=fused_bf16_setup ? torch::empty({a.size_auto?4:2},q.options().dtype(at::kInt)) : torch::zeros({a.size_auto?4:2},q.options().dtype(at::kInt));
        a.index_invalid=index_invalid.data_ptr<int>();
        union_slots=torch::empty({rows,16384},q.options().dtype(at::kShort));
        union_masks=fused_bf16_setup ? torch::empty({union_groups,16384},q.options().dtype(at::kInt)) : torch::zeros({union_groups,16384},q.options().dtype(at::kInt));
        a.union_slotmap=reinterpret_cast<uint16_t*>(union_slots.data_ptr<int16_t>());
        a.union_mask=reinterpret_cast<uint32_t*>(union_masks.data_ptr<int>());
        if (fused_bf16_setup) {
            const int64_t nvec=rows*width/4;
            const int fill_grid=static_cast<int>(std::min<int64_t>(ceil_div(nvec,256),4096));
            ::lightop::deep_gemm::sparse_mqa_detail::sparse_fp8_fp4_paged_mqa_logits__initialize_prefill_output_and_scratch<true><<<fill_grid,256,0,stream>>>(a,nvec,union_groups*int64_t{16384});
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 fused setup failed");
        }
        if(metadata_plan) make_metadata(true);
        else sparse_fp8_fp4_paged_mqa_logits__mark_prefill_selected_blocks<<<num_sms*8,128,0,stream>>>(a);
        TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q selected block marking failed");
        if (fused_bf16_setup) {
            if(a.bf16_q_fragment) {
                sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_bf16_fragment_operands_checked<<<num_sms*8,256,0,stream>>>(a,prepared_q.data_ptr<uint8_t>());
            } else {
                sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_bf16_operands_checked<<<num_sms*8,256,0,stream>>>(a,prepared_q.data_ptr<uint8_t>());
            }
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 fused checked preparation failed");
            sparse_fp8_fp4_paged_mqa_logits__build_prefill_multiq_raw_index<<<num_sms*8,128,0,stream>>>(a);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q private indexing failed");
        } else {
            sparse_fp8_fp4_paged_mqa_logits__guard_prefill_multiq_bf16<<<num_sms*8,256,0,stream>>>(a);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q eligibility guard failed");
            sparse_fp8_fp4_paged_mqa_logits__select_prefill_bf16_union_density<<<1,1,0,stream>>>(a);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q density selection failed");
            sparse_fp8_fp4_paged_mqa_logits__build_prefill_multiq_raw_index<<<num_sms*8,128,0,stream>>>(a);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q private indexing failed");
            sparse_fp8_fp4_paged_mqa_logits__prepare_prefill_q_bf16<<<num_sms*8,256,0,stream>>>(a,prepared_q.data_ptr<uint8_t>());
            TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q Q preparation failed");
        }
        if (want_bf16_k_cache && !(fused_bf16_setup && a.bf16_fused_prep)) {
            sparse_fp8_fp4_paged_mqa_logits__gfx936_prepare_k_bf16<<<num_sms*8,256,0,stream>>>(a);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"Sparse MQA BF16 key preparation failed");
        }
        const dim3 sparse_grid(static_cast<unsigned>(num_sms),static_cast<unsigned>(ceil_div(rows,16)));
        if(a.size_auto){
            if(a.rows<=2048){
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_query_adaptive<<<dim3(static_cast<unsigned>(num_sms),static_cast<unsigned>((a.rows+15)/16)),256,11264,stream>>>(a);
            }else{
            const char* mode_env=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_BF16_MODE");const int mode=mode_env?std::atoi(mode_env):1;
            const char* wave_env=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_BF16_WAVES");const int waves2=wave_env?std::atoi(wave_env):4,waves1=wave_env?std::atoi(wave_env):4;
            TORCH_CHECK((waves2==1 || waves2==2 || waves2==4) && (waves1==1 || waves1==2 || waves1==4),"BF16 waves must be 1,2,4");
            const char* grid_env=std::getenv("LIGHTOP_SPARSE_MQA_LOGITS_SIZE_GRID");const int grid2=static_cast<int>(num_sms)*(grid_env?std::max(1,std::min(256,std::atoi(grid_env))):32),grid1=static_cast<int>(num_sms)*(grid_env?std::max(1,std::min(256,std::atoi(grid_env))):24);
            if(a.size_force!=1 && a.size_force!=2){
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q16_k_cache<<<sparse_grid,256,0,stream>>>(a);
            }
            if(a.size_force!=1){
            if(waves2==1){
                if(mode==3){
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w1<<<grid2,64,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w1<<<grid2,64,0,stream>>>(a);
                    }
                }else{
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w1<<<grid2,64,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w1<<<grid2,64,0,stream>>>(a);
                    }
                }
            }
            else if(waves2==2){
                if(mode==3){
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w2<<<grid2,128,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w2<<<grid2,128,0,stream>>>(a);
                    }
                }else{
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w2<<<grid2,128,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w2<<<grid2,128,0,stream>>>(a);
                    }
                }
            }
            else if(waves2==4){
                if(mode==3){
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_direct_w4<<<grid2,256,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_direct_w4<<<grid2,256,0,stream>>>(a);
                    }
                }else{
                    if(a.bf16_k_mode!=1 || a.rows<=2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_resident_w4<<<grid2,256,0,stream>>>(a);
                    }
                    if(a.bf16_k_mode!=0 && a.rows>2048){
                        sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q2_packed_resident_w4<<<grid2,256,0,stream>>>(a);
                    }
                }
            }
            }
            if(a.size_force==1){
            if(waves1==1){
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w1<<<grid1,64,0,stream>>>(a);
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w1<<<grid1,64,0,stream>>>(a);
            }
            else if(waves1==2){
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w2<<<grid1,128,0,stream>>>(a);
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w2<<<grid1,128,0,stream>>>(a);
            }
            else if(waves1==4){
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_selected_w4<<<grid1,256,0,stream>>>(a);
                sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q1_packed_w4<<<grid1,256,0,stream>>>(a);
            }
            }
            }
        }else if (sparse_opt)
            sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q16_k_cache<<<sparse_grid,256,0,stream>>>(a);
        else
            sparse_fp8_fp4_paged_mqa_logits__gfx936_bf16_q16_legacy<<<sparse_grid,256,0,stream>>>(a);
        TORCH_CHECK(hipGetLastError()==hipSuccess,"BF16 multi-Q consumer failed");
    }

    if (prepare_k_fp8) {
        if (!single_workspace) {
        prepared_k = torch::empty({2048,64,132}, kv.options());
        eligibility = fused_setup ? torch::empty({2}, q.options().dtype(at::kInt)) : torch::zeros({2}, q.options().dtype(at::kInt));
        a.kv_fp8 = prepared_k.data_ptr<uint8_t>();
        a.fp8_invalid = eligibility.data_ptr<int>();
        a.cache_count = eligibility.data_ptr<int>()+1;
        cache_pages = fused_setup ? torch::empty({kv.size(0)},q.options().dtype(at::kInt)) : torch::full({kv.size(0)},-1,q.options().dtype(at::kInt));
        a.cache_pages = cache_pages.data_ptr<int>();
        cache_physical_pages=torch::empty({2048},q.options().dtype(at::kInt));
        a.cache_physical_pages=cache_physical_pages.data_ptr<int>();
        prepared_q = torch::empty({rows,4112}, kv.options());
        a.q_fp8 = prepared_q.data_ptr<uint8_t>();
        selected = fused_setup ? torch::empty({kv.size(0)*8}, q.options().dtype(at::kInt)) : torch::zeros({kv.size(0)*8}, q.options().dtype(at::kInt));
        a.selected_blocks = selected.data_ptr<int>();
        if (prepare_union) {
            index_invalid=fused_setup ? torch::empty({2},q.options().dtype(at::kInt)) : torch::zeros({2},q.options().dtype(at::kInt));
            a.index_invalid=index_invalid.data_ptr<int>();
            union_slots=torch::empty({rows,16384},q.options().dtype(at::kShort));
            union_masks=fused_setup ? torch::empty({union_groups,16384},q.options().dtype(at::kInt)) : torch::zeros({union_groups,16384},q.options().dtype(at::kInt));
            a.union_slotmap=reinterpret_cast<uint16_t*>(union_slots.data_ptr<int16_t>());
            a.union_mask=reinterpret_cast<uint32_t*>(union_masks.data_ptr<int>());
        }
        }
        if (fused_setup) {
            const int64_t nvec=rows*width/4;
            const int fill_grid=static_cast<int>(std::min<int64_t>(ceil_div(nvec,256),4096));
        ::lightop::deep_gemm::sparse_mqa_detail::sparse_fp8_fp4_paged_mqa_logits__initialize_prefill_output_and_scratch<true><<<fill_grid,256,0,stream>>>(a,nvec,prepare_union ? union_groups*int64_t{16384} : 0);
            TORCH_CHECK(hipGetLastError()==hipSuccess,"Prefill fused output and scratch initialization failed");
        }
        if(metadata_plan) make_metadata(true);
        else sparse_fp8_fp4_paged_mqa_logits__mark_prefill_selected_blocks<<<num_sms*8,128,0,stream>>>(a);
        TORCH_CHECK(hipGetLastError()==hipSuccess,"Prefill selected block marking failed");
        const int prepare_grid=static_cast<int>(num_sms*8);
        const int guard_grid = static_cast<int>(std::min<int64_t>(ceil_div(rows,4),4096));
        sparse_fp8_fp4_paged_mqa_logits__gfx938_prepare_qk_fp8<<<prepare_grid+guard_grid,256,0,stream>>>(a,const_cast<uint8_t*>(a.kv_fp8),const_cast<uint8_t*>(a.q_fp8),prepare_grid,guard_grid);
        TORCH_CHECK(hipGetLastError()==hipSuccess,"Prefill combined Q/K FP8 preparation launch failed");
        if (!single_workspace) {
            prepared_metadata=torch::empty_like(metadata);
            a.metadata_fp8=prepared_metadata.data_ptr<uint8_t>();
        }
        if (prepare_union) {
            sparse_fp8_fp4_paged_mqa_logits__remap_and_build_query32_union_index<<<num_sms*8,128,0,stream>>>(
                a,reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(a.metadata_fp8)));
            TORCH_CHECK(hipGetLastError()==hipSuccess,"32Q fused metadata remap and union indexing failed");
        } else {
            sparse_fp8_fp4_paged_mqa_logits__remap_prefill_cache_metadata<<<num_sms*8,256,0,stream>>>(a,reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(a.metadata_fp8)));
            TORCH_CHECK(hipGetLastError()==hipSuccess,"Prefill cache metadata relocation failed");
        }
    }
    make_metadata(false);
#define PREFILL_CALL(FP4,HT,BLOCK,FALLBACK,FP8,BF16) do { \
    if constexpr (FP4 && HT == 2 && BLOCK == 8) { \
        if (fixed_prefill_layout) { \
            if constexpr (!FALLBACK && FP8 && !BF16) { \
                if (prepare_k_fp8) { \
                    if (!context_lens) { \
                        if (prepare_union) { \
                            sparse_fp8_fp4_paged_mqa_logits__gfx938_fp8_q16<<<dim3(static_cast<unsigned>(num_sms),static_cast<unsigned>(ceil_div(rows,16))),256,0,stream>>>(a); \
                            sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,true,true,true,false,true,true,true><<<num_sms*8,128,0,stream>>>(a); \
                        } else { \
                            sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,true,true,true,false,true,true><<<num_sms*8,128,0,stream>>>(a); \
                        } \
                    } else { \
                        sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,true,true,true><<<num_sms*8,128,0,stream>>>(a); \
                    } \
                    sparse_fp8_fp4_paged_mqa_logits__prefill_fallback<<<num_sms*8,128,0,stream>>>(a); \
                } else { \
                    sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,true,false><<<num_sms*8,128,0,stream>>>(a); \
                } \
            } else { \
                sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,true,false><<<num_sms*8,128,0,stream>>>(a); \
            } \
        } else { \
            sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,false><<<num_sms*8,128,0,stream>>>(a); \
        } \
    } else { \
        sparse_fp8_fp4_paged_mqa_logits__prefill_pair<FP4,HT,BLOCK,FALLBACK,FP8,BF16,false><<<num_sms*8,128,0,stream>>>(a); \
    } \
} while (0)
#define LAUNCH(FP4,HT,BLOCK) do { \
    if (is_prefill) { \
        if constexpr (FP4) { \
            if (arch.rfind("gfx938",0) == 0) { \
                PREFILL_CALL(FP4,HT,BLOCK,false,true,false); \
                if (!prepare_k_fp8) { PREFILL_CALL(FP4,HT,BLOCK,false,false,true); } \
            } else { \
                PREFILL_CALL(FP4,HT,BLOCK,false,false,false); \
            } \
        } else { \
            PREFILL_CALL(FP4,HT,BLOCK,false,false,false); \
        } \
        if (!prepare_k_fp8) { PREFILL_CALL(FP4,HT,BLOCK,true,false,false); } \
    } else { \
        sparse_fp8_fp4_paged_mqa_logits__decode_pair<FP4,HT,BLOCK,false><<<num_sms,kThreads,0,stream>>>(a); \
        sparse_fp8_fp4_paged_mqa_logits__decode_pair<FP4,HT,BLOCK,true><<<num_sms,kThreads,0,stream>>>(a); \
    } \
} while (0)
    if (fp4) {
        if (head_tiles == 1) { if (block_kv == 8) { LAUNCH(true,1,8); } else { LAUNCH(true,1,16); } }
        else { if (block_kv == 8) { LAUNCH(true,2,8); } else { LAUNCH(true,2,16); } }
    } else {
        if (head_tiles == 1) { if (block_kv == 8) { LAUNCH(false,1,8); } else { LAUNCH(false,1,16); } }
        else { if (block_kv == 8) { LAUNCH(false,2,8); } else { LAUNCH(false,2,16); } }
    }
#undef LAUNCH
#undef PREFILL_CALL
    TORCH_CHECK(hipGetLastError() == hipSuccess, "fp8_fp4_paged_sparse_mqa_logits: sparse consumer launch failed");
    return out;
}
}  // namespace lightop::deep_gemm::gfx936_detail

#undef SPARSE_DEVICE_CHECK
