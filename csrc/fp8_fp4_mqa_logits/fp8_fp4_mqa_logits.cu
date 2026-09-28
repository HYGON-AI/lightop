// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "fp8_fp4_mqa_logits.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <string>

#include "utils.h"

namespace lightop::fp4_mqa {
namespace {

using detail::bf16_t;
using detail::floatx4;
using detail::uint32x4_t;
namespace wrap = detail::even_wrap;

struct Tile32x256 {
    static constexpr int M = 32;
    static constexpr int N = 256;
    static constexpr int KHalfN = 128;
    static constexpr int K = 128;
    static constexpr int PanelK = 64;
    static constexpr int Waves = 4;
    static constexpr int NTilesPerWave = 4;
    static constexpr int MTilesPerWave = 2;
};

static_assert(wrap::is_supported_v<bf16_t, Tile32x256::M, Tile32x256::K,
                                   Tile32x256::PanelK, Tile32x256::Waves>);

// K-only LDS layout. A 16B slot holds 8 BF16 values. Rotating the three
// K-slot bits puts the two 16B chunks loaded by adjacent lanes in distinct
// bank quartets; the row-slot parity supplies the other bit needed by the
// MMAC B-read lane groups. All operations are integer coordinate transforms:
// no lane shuffle or data permutation is required.
__device__ __forceinline__ int k_lds_elem_index(int row, int col, int panel) {
    const int row_block = row >> 5;
    const int within = row & 31;
    const int wrap_id = within & 3;
    const int row_slot = within >> 2;
    const int k_slot = col >> 3;
    const int rotated_k = ((k_slot & 3) << 1) | (k_slot >> 2);
    const int bank_slot = (wrap_id << 1) ^ (row_slot & 1) ^ rotated_k ^
                          (panel << 2);
    return (row_block * 4 + wrap_id) * 512 +
           (row_slot * 8 + bank_slot) * 8 + (col & 7);
}

__device__ __forceinline__ uint16_t round_bf16(float x) {
    const uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

// Same E2M1/UE8M0 bit semantics as paged_mqa_logits_fp4::decode_fp4.
__device__ __forceinline__ uint16_t decode_fp4_bf16(unsigned code, unsigned scale) {
    const unsigned mag = code & 7u;
    const unsigned sign = (code & 8u) << 12;
    const unsigned base = mag >= 2 ? 0x3f00u + (mag << 6)
                                   : (mag == 1 ? 0x3f00u : 0u);
    const int exponent = static_cast<int>(base >> 7) + static_cast<int>(scale) - 127;
    if (scale < 255 && (mag == 0 || (exponent > 0 && exponent < 255)))
        return sign | (mag == 0 ? 0u : (static_cast<unsigned>(exponent) << 7)
                                      | (base & 0x7fu));
    const float factor = scale == 255 ? INFINITY
                                      : ldexpf(1.0f, static_cast<int>(scale) - 127);
    return round_bf16(__uint_as_float((sign | base) << 16) * factor);
}

// One 16-byte packed load covers exactly one 32-value MXFP4 scale group.
// Each output is 8 adjacent BF16 values, so it can be written as one LDS b128.
__device__ __forceinline__ void decode_group32(
    const uint32x4_t& packed, unsigned scale, uint32x4_t (&out)[4]) {
#pragma unroll
    for (int group8 = 0; group8 < 4; ++group8) {
        uint32x4_t decoded = {0, 0, 0, 0};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int byte_index = group8 * 4 + pair;
            const unsigned byte = (packed[byte_index / 4] >> ((byte_index % 4) * 8)) & 255u;
            const unsigned lo = decode_fp4_bf16(byte & 15u, scale);
            const unsigned hi = decode_fp4_bf16(byte >> 4, scale);
            decoded[pair] = lo | (hi << 16);
        }
        out[group8] = decoded;
    }
}

// Four neighboring lanes load the four 16B chunks of one K row. Every group
// of eight lanes therefore loads two adjacent 64B rows, i.e. a full 128B
// aligned interval across one buffer_load_dwordx4 instruction's lanes.
// This is an address-layout property; cache/DRAM transactions are hardware
// dependent. Each 128-row half of the N tile
// stores both decoded 64-D panels in 32 KiB LDS. B fragments are copied to
// VGPRs before the next half overwrites LDS.
template <bool NTail>
__device__ __forceinline__ void stage_k_half_tile(
    bf16_t (&k_lds)[2][Tile32x256::KHalfN * Tile32x256::PanelK],
    const int32_t* k_scale,
    int tile_n, int num_k_tokens, const uint32x4_t& k_rsrc) {
    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int chunk = lane & 3;
    uint32x4_t raw[2];
    uint32_t sf[2];
#pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int local_n = wave * 32 + iter * 16 + (lane >> 2);
        const int abs_n = tile_n + local_n;
        const bool valid = !NTail || abs_n < num_k_tokens;
        const int packed_offset = valid ? abs_n * 64 + chunk * 16 : -1;
        detail::buffer_load_dwordx4_vgpr(raw[iter], k_rsrc,
                                                          packed_offset);
        sf[iter] = valid ? static_cast<uint32_t>(k_scale[abs_n]) : 0x7f7f7f7fu;
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    const int panel = chunk >> 1;
#pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int local_n = wave * 32 + iter * 16 + (lane >> 2);
        const unsigned scale = (sf[iter] >> (chunk * 8)) & 255u;
        uint32x4_t decoded[4];
        decode_group32(raw[iter], scale, decoded);
#pragma unroll
        for (int group8 = 0; group8 < 4; ++group8) {
            const int local_k = (chunk & 1) * 32 + group8 * 8;
            const int lds_index = k_lds_elem_index(local_n, local_k, panel);
            *reinterpret_cast<uint32x4_t*>(k_lds[panel] + lds_index) = decoded[group8];
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));

// C and D are distinct operands. Passing a loop-invariant zero C to each
// head's first MMAC lets the zero VGPR quartet be initialized before the H
// loop, while every later MMAC consumes the previous D. The BF16 even-wrap
// operand layout uses the unmodified instruction on both gfx936 and gfx938.
__device__ __forceinline__ floatx4 mmac_bf16_inline(
    const uint32_t a01[2], const uint32_t b01[2], floatx4 c) {
#if defined(__gfx936__) || defined(__gfx938__)
    const uint32x2_t a = {a01[0], a01[1]};
    const uint32x2_t b = {b01[0], b01[1]};
    floatx4 d;
    asm volatile("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %3"
                 : "=&v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
#else
    // Lightop also builds code objects for other architectures. The host
    // dispatch rejects those devices before launching this kernel.
    return c;
#endif
}

// Keep eight independent M16xN16 accumulators in flight. For each K32
// fragment, issue the first BF16 MMAC for every dot before its dependent
// second MMAC. This follows m64.cu's independent-dot latency coverage while
// retaining the even-wrap Q stage.
__device__ __forceinline__ void bf16_mmac_head_interleaved(
    bf16_t q_lds[][Tile32x256::M * Tile32x256::PanelK],
    const uint32x4_t* b_reg, floatx4 (&dot)[2][4], int wave,
    floatx4 zero_c) {
    uint32x4_t a_reg[2][2];
#pragma unroll
    for (int frag = 0; frag < 4; ++frag) {
        const int stage = frag & 1;
        const int panel = frag >> 1;
        const int k_base = (frag & 1) * 32;
#pragma unroll
        for (int mu = 0; mu < 2; ++mu) {
            a_reg[stage][mu] =
                detail::even_wrap::ds_read_mmac_tile<
                    bf16_t, Tile32x256::M, Tile32x256::K,
                    Tile32x256::PanelK, Tile32x256::Waves>(
                    q_lds[panel], wave, mu * 16, k_base);
        }
        if (frag == 0) continue;
        const int previous = stage ^ 1;
        const int previous_frag = frag - 1;
        asm volatile("s_setprio 1" ::: "memory");
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int nu = 0; nu < 4; ++nu) {
                const uint32x4_t bv = b_reg[previous_frag * 4 + nu];
                const uint32_t b01[2] = {bv[2 * half], bv[2 * half + 1]};
#pragma unroll
                for (int mu = 0; mu < 2; ++mu) {
                    const uint32x4_t av = a_reg[previous][mu];
                    const uint32_t a01[2] = {av[2 * half], av[2 * half + 1]};
                    if (previous_frag == 0 && half == 0)
                        dot[mu][nu] = mmac_bf16_inline(a01, b01, zero_c);
                    else
                        dot[mu][nu] = mmac_bf16_inline(a01, b01, dot[mu][nu]);
                }
            }
        }
        asm volatile("s_setprio 0" ::: "memory");
    }

    asm volatile("s_setprio 1" ::: "memory");
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
            const uint32x4_t bv = b_reg[3 * 4 + nu];
            const uint32_t b01[2] = {bv[2 * half], bv[2 * half + 1]};
#pragma unroll
            for (int mu = 0; mu < 2; ++mu) {
                const uint32x4_t av = a_reg[1][mu];
                const uint32_t a01[2] = {av[2 * half], av[2 * half + 1]};
                dot[mu][nu] = mmac_bf16_inline(a01, b01, dot[mu][nu]);
            }
        }
    }
    asm volatile("s_setprio 0" ::: "memory");
}

// m64.cu's paired epilogue: expose eight independent max instructions before
// consuming either packet in the weighted sum. The two FP32 lanes share W.
__device__ __forceinline__ void accum_relu_paired(
    floatx4 (&sum)[2][4], const floatx4 (&dot)[2][4], float w0, float w1) {
    const float weights[2] = {w0, w1};
#pragma unroll
    for (int mu = 0; mu < 2; ++mu) {
        const detail::floatx2 weight = {weights[mu], weights[mu]};
#pragma unroll
        for (int nu = 0; nu < 4; nu += 2) {
            floatx4 activated0, activated1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                activated0[j] = detail::relu0_f32(dot[mu][nu][j]);
                activated1[j] = detail::relu0_f32(dot[mu][nu + 1][j]);
            }
            floatx4 s0 = sum[mu][nu];
            floatx4 s1 = sum[mu][nu + 1];
            s0.xy = activated0.xy * weight + s0.xy;
            s0.zw = activated0.zw * weight + s0.zw;
            s1.xy = activated1.xy * weight + s1.xy;
            s1.zw = activated1.zw * weight + s1.zw;
            sum[mu][nu] = s0;
            sum[mu][nu + 1] = s1;
        }
    }
}

template <bool MTail, bool NTail>
__global__ __launch_bounds__(256, 1) void bf16_fp4_32x256_kernel(
    const bf16_t* __restrict__ q, const uint8_t* __restrict__ k_payload,
    const int32_t* __restrict__ k_scale, const float* __restrict__ weights,
    const int32_t* __restrict__ ks, const int32_t* __restrict__ ke,
    float* __restrict__ out, int num_q_tokens, int num_k_tokens,
    int num_heads, int out_width) {
#if defined(__gfx936__) || defined(__gfx938__)
    constexpr int BM = Tile32x256::M;
    constexpr int BN = Tile32x256::N;
    constexpr int BK = Tile32x256::K;
    constexpr int WK = Tile32x256::PanelK;
    constexpr int Waves = Tile32x256::Waves;
    constexpr int MTiles = Tile32x256::MTilesPerWave;
    constexpr int NTiles = Tile32x256::NTilesPerWave;

    // Two 16 KiB K panels + two 8 KiB Q stages. The next head's direct-to-LDS
    // loads can run while MMAC reads the current head from the other stage.
    __shared__ bf16_t k_lds[2][Tile32x256::KHalfN * WK];
    __shared__ bf16_t q_lds[2][2][BM * WK];
    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;  // absolute K coordinate
    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int n_block = wave * 32;

    // Every wave votes on the same 32 Q rows before staging this K tile.
    const int probe_row = tile_m + (lane & 31);
    const bool overlap = probe_row < num_q_tokens &&
        tile_n < ke[probe_row] && tile_n + BN > ks[probe_row];
    if (__ballot(overlap) == 0) return;

    uint32x4_t b_reg[(BK / 32) * NTiles];
    const uint32x4_t k_rsrc = detail::make_buffer_rsrc(k_payload);

#pragma unroll
    for (int half = 0; half < BN / Tile32x256::KHalfN; ++half) {
        stage_k_half_tile<NTail>(k_lds, k_scale,
                                 tile_n + half * Tile32x256::KHalfN,
                                 num_k_tokens, k_rsrc);
#pragma unroll
        for (int panel = 0; panel < BK / WK; ++panel) {
#pragma unroll
            for (int ku = 0; ku < WK / 32; ++ku) {
                const int frag = panel * (WK / 32) + ku;
#pragma unroll
                for (int nu = 0; nu < NTiles / 2; ++nu) {
                    const int row = n_block + nu * 16 + (lane & 15);
                    const int col = ku * 32 + (lane >> 4) * 8;
                    const int lds_index = k_lds_elem_index(row, col, panel);
                    b_reg[frag * NTiles + half * (NTiles / 2) + nu] =
                        *reinterpret_cast<const uint32x4_t*>(k_lds[panel] + lds_index);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    }

    const auto w_plan = detail::precompute_w_shfl_xor16(lane);
    floatx4 zero_c = {0.f, 0.f, 0.f, 0.f};
    // Keep one physical zero quartet live across all heads.
    asm volatile("" : "+v"(zero_c));

    const int valid_m = MTail ? min(BM, num_q_tokens - tile_m) : BM;

    floatx4 c_out[MTiles][NTiles];
#pragma unroll
    for (int mu = 0; mu < MTiles; ++mu)
#pragma unroll
        for (int nu = 0; nu < NTiles; ++nu)
            c_out[mu][nu] = {0.f, 0.f, 0.f, 0.f};

    const uint32x4_t w_rsrc = detail::make_buffer_rsrc(
        weights + static_cast<size_t>(tile_m) * num_heads);
    // Prime head 0 before the loop. After each head's weight is unpacked,
    // issue the next head's Q and W before the current head's MMAC. The LDS
    // stages are disjoint; wait for current Q reads and pending VMEM writes
    // before the barrier that makes the next stage visible to every wave.
    bf16_t* q_tile = const_cast<bf16_t*>(
        q + static_cast<size_t>(tile_m) * num_heads * BK);
    detail::issue_q_head<bf16_t, BM, BK, WK, Waves, MTail>(
        q_lds[0], q_tile, num_heads * BK, 0, valid_m);
    float w_raw;
    detail::issue_w_head_vgpr<BM, MTail>(
        w_raw, w_rsrc, num_heads, 0, lane, valid_m);
    detail::wait_vmcnt0_dep(w_raw);
    asm volatile("s_barrier" ::: "memory");

    // Q remains BF16 throughout the head loop and MMAC schedule.
    for (int head = 0; head < num_heads; ++head) {
        float w0, w1;
        detail::unpack_w_shfl_xor16_pre(w_raw, w_plan, w0, w1);
        float next_w_raw;
        if (head + 1 < num_heads) {
            detail::issue_q_head<bf16_t, BM, BK, WK, Waves, MTail>(
                q_lds[(head + 1) & 1], q_tile, num_heads * BK, head + 1,
                valid_m);
            detail::issue_w_head_vgpr<BM, MTail>(
                next_w_raw, w_rsrc, num_heads, head + 1, lane, valid_m);
        }
        floatx4 c_head[MTiles][NTiles];
        bf16_mmac_head_interleaved(q_lds[head & 1], b_reg, c_head, wave, zero_c);
        accum_relu_paired(c_out, c_head, w0, w1);
        if (head + 1 < num_heads) {
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            detail::wait_vmcnt0_dep(next_w_raw);
            asm volatile("s_barrier" ::: "memory");
            w_raw = next_w_raw;
        }
    }

    // The output has local columns, but K staging used absolute K tiles so
    // all 32 Q rows may safely share one K panel even when their ks differ.
    // The host initializes invalid columns to -inf; only valid values are stored.
#pragma unroll
    for (int mu = 0; mu < MTiles; ++mu) {
        const int row = tile_m + mu * 16 + (lane % 16);
        if (row >= num_q_tokens) continue;
        const int start = ks[row];
        const int end = ke[row];
#pragma unroll
        for (int nu = 0; nu < NTiles; ++nu) {
            const floatx4 scores = detail::pack_c_contig4(c_out[mu][nu], lane);
            const int abs_col = tile_n + (nu / (NTiles / 2)) * Tile32x256::KHalfN +
                                n_block + (nu % (NTiles / 2)) * 16 + (lane / 16) * 4;
            const int local_col = abs_col - start;
            if (abs_col >= start && abs_col + 3 < end && local_col + 3 < out_width &&
                (local_col & 3) == 0) {
                *reinterpret_cast<floatx4*>(out + static_cast<size_t>(row) * out_width +
                                             local_col) = scores;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = abs_col + j;
                    const int local = col - start;
                    if (col >= start && col < end && local >= 0 && local < out_width)
                        out[static_cast<size_t>(row) * out_width + local] = scores[j];
                }
            }
        }
    }
#else
    // Even-wrap dwordx4 direct-to-LDS is also architecture-specific.
    // The host rejects unsupported devices; do not emit those instructions
    // into Lightop's gfx928 (or older) code objects.
    __builtin_trap();
#endif
}

template <bool MTail, bool NTail>
void launch(const bf16_t* q, const uint8_t* k_payload, const int32_t* k_scale,
            const float* weights, const int32_t* ks, const int32_t* ke,
            float* out,
            int t, int l, int h, int width, hipStream_t stream) {
    const dim3 grid((l + Tile32x256::N - 1) / Tile32x256::N,
                    (t + Tile32x256::M - 1) / Tile32x256::M);
    bf16_fp4_32x256_kernel<MTail, NTail><<<grid, 256, 0, stream>>>(
        q, k_payload, k_scale, weights, ks, ke, out, t, l, h, width);
}

}  // namespace

torch::Tensor fp8_fp4_mqa_logits(
    const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::tuple<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, bool clean_logits,
    int64_t max_seqlen_k) {
    const auto& q_data = std::get<0>(q);
    const auto& q_scale = std::get<1>(q);
    const auto& k_data = std::get<0>(kv);
    const auto& k_scale = std::get<1>(kv);
    const auto& ks = cu_seq_len_k_start;
    const auto& ke = cu_seq_len_k_end;

    TORCH_CHECK(q_data.scalar_type() == at::kBFloat16 && !q_scale.has_value(),
                "fp8_fp4_mqa_logits: only (BF16 Q, None) is implemented; FP4 Q and FP8 Q are unsupported");
    TORCH_CHECK(q_data.is_cuda() && q_data.dim() == 3 && q_data.size(2) == 128 &&
                q_data.is_contiguous(), "q must be contiguous BF16 [T,H,128] on HIP");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q_data.data_ptr()) % 4 == 0,
                "q data pointer must be 4-byte aligned for buffer_load_dwordx4");
    const int64_t t = q_data.size(0), h = q_data.size(1);
    TORCH_CHECK(h > 0 && h <= INT_MAX / 256,
                "head count must be positive and fit BF16 Q buffer offset limits");
    TORCH_CHECK(k_data.is_cuda() && k_data.device() == q_data.device() &&
                k_data.scalar_type() == at::kChar && k_data.dim() == 2 &&
                k_data.size(1) == 64 && k_data.is_contiguous(),
                "K must be contiguous packed E2M1 FP4 int8 [L,64]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(k_data.data_ptr()) % 4 == 0,
                "K data pointer must be 4-byte aligned for buffer_load_dwordx4");
    const int64_t l = k_data.size(0);
    TORCH_CHECK(k_scale.is_cuda() && k_scale.device() == q_data.device() &&
                k_scale.scalar_type() == at::kInt && k_scale.dim() == 1 &&
                k_scale.size(0) == l && k_scale.is_contiguous(),
                "K scale must be packed UE8M0 int32 [L]");
    TORCH_CHECK(weights.is_cuda() && weights.device() == q_data.device() &&
                weights.scalar_type() == at::kFloat && weights.dim() == 2 &&
                weights.size(0) == t && weights.size(1) == h &&
                weights.is_contiguous(), "weights must be contiguous FP32 [T,H]");
    TORCH_CHECK(ks.is_cuda() && ke.is_cuda() && ks.device() == q_data.device() &&
                ke.device() == q_data.device() && ks.scalar_type() == at::kInt &&
                ke.scalar_type() == at::kInt && ks.dim() == 1 && ke.dim() == 1 &&
                ks.size(0) == t && ke.size(0) == t && ks.is_contiguous() &&
                ke.is_contiguous(), "ks/ke must be contiguous int32 [T]");
    TORCH_CHECK(!clean_logits, "clean_logits=True is unsupported for compressed logits");
    TORCH_CHECK(max_seqlen_k > 0 && max_seqlen_k % 4 == 0 &&
                max_seqlen_k <= INT_MAX, "max_seqlen_k must be positive, 4-aligned and int32-sized");
    TORCH_CHECK(l <= INT_MAX / 64 && t <= INT_MAX / (h * 256) &&
                t <= 65535LL * Tile32x256::M,
                "shape exceeds 32x256 launch or buffer offset limits");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q_data.device());
    hipDeviceProp_t properties{};
    int device = 0;
    TORCH_CHECK(hipGetDevice(&device) == hipSuccess &&
                hipGetDeviceProperties(&properties, device) == hipSuccess,
                "cannot query HIP device");
    const std::string arch(properties.gcnArchName);
    TORCH_CHECK(arch.rfind("gfx936", 0) == 0 || arch.rfind("gfx938", 0) == 0,
                "fp8_fp4_mqa_logits requires gfx936 or gfx938, got ", arch);

    auto out = torch::full({t, max_seqlen_k}, -INFINITY,
                           q_data.options().dtype(at::kFloat));
    if (t == 0 || l == 0) return out;
    const auto stream = at::hip::getCurrentHIPStream();
    const auto* q_ptr = reinterpret_cast<const bf16_t*>(q_data.data_ptr());
    const auto* k_ptr = reinterpret_cast<const uint8_t*>(k_data.data_ptr());
    const auto* sf_ptr = k_scale.data_ptr<int32_t>();
    const auto* w_ptr = weights.data_ptr<float>();
    const auto* ks_ptr = ks.data_ptr<int32_t>();
    const auto* ke_ptr = ke.data_ptr<int32_t>();
    auto* out_ptr = out.data_ptr<float>();

    const bool mtail = t % Tile32x256::M != 0;
    const bool ntail = l % Tile32x256::N != 0;
    if (mtail && ntail)
        launch<true, true>(q_ptr, k_ptr, sf_ptr, w_ptr, ks_ptr, ke_ptr,
                           out_ptr, t, l, h, max_seqlen_k, stream);
    else if (mtail)
        launch<true, false>(q_ptr, k_ptr, sf_ptr, w_ptr, ks_ptr, ke_ptr,
                            out_ptr, t, l, h, max_seqlen_k, stream);
    else if (ntail)
        launch<false, true>(q_ptr, k_ptr, sf_ptr, w_ptr, ks_ptr, ke_ptr,
                            out_ptr, t, l, h, max_seqlen_k, stream);
    else
        launch<false, false>(q_ptr, k_ptr, sf_ptr, w_ptr, ks_ptr, ke_ptr,
                             out_ptr, t, l, h, max_seqlen_k, stream);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "BF16/FP4 MQA kernel launch failed");
    return out;
}

}  // namespace lightop::fp4_mqa
