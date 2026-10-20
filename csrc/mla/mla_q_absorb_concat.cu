// MLA Q-absorb BMM + concat on gfx936, using LightOp's BF16 MMAC pattern.
// Original weight layout, FP32 accumulation, one final BF16 rounding.
// Tile choices are fixed after graph tests on BS2/3/4 and real checkpoint layers.
// No runtime tuning, weight repack, BLAS call or private scratch buffer.
// BM: token rows per block; NT: N tiles per wave; NW: waves per block.
// BK: reduction stage width; PIPE: prefetch the next stage before MMAC.
// Each wave uses BF16 MMACs; the epilogue writes [M,128,512+64] directly.
// N-tile zero rotates the Q rope columns directly into the concatenated output.
// The BF16 products are rounded separately, matching native BF16 RoPE.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {
using BFloat16 = at::BFloat16;
using BFloat16x4 = short __attribute__((vector_size(8)));
using Float4 = float __attribute__((vector_size(16)));
using UInt4 = uint32_t __attribute__((vector_size(16)));

struct alignas(16) BFloat16x8 {
    BFloat16x4 lo, hi;
};

__device__ __forceinline__ Float4 mma_bf16(BFloat16x4 a, BFloat16x4 b, Float4 c) {
#if defined(__gfx936__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    return c;
#endif
}

__device__ __forceinline__ UInt4 make_buffer_resource(const BFloat16 *p) {
    UInt4 r = {};
    *reinterpret_cast<uint64_t *>(&r) = reinterpret_cast<uint64_t>(p);
    r[2] = 0x80000000;
    r[3] = 0x00020000;
    return r;
}

__device__ __forceinline__ BFloat16x8 load_bf16x8(UInt4 r, int off) {
    auto v = __builtin_amdgcn_buffer_load_dwordx4(r, 0, off, false, false);
    return __builtin_bit_cast(BFloat16x8, v);
}

// Do not contract these two BF16 products into one FP32 FMA.
__device__ __forceinline__ BFloat16 rotate_bf16(const BFloat16 *x, const BFloat16 *cos_sin, int d,
                                                bool neox) {
    const int partner = neox ? (d + 32) % 64 : (d ^ 1);
    const int frequency = neox ? d % 32 : d / 2;
    const bool negate = neox ? d < 32 : (d % 2 == 0);
    const float a = float(BFloat16(float(x[d]) * float(cos_sin[frequency])));
    const float rotated = negate ? -float(x[partner]) : float(x[partner]);
    const float b = float(BFloat16(rotated * float(cos_sin[frequency + 32])));
    return BFloat16(a + b);
}

// Zero padded token rows before staging; weights retain their original layout.
template <int BM, int BN, int BK, int NW, int K>
__device__ __forceinline__ void
preload_tiles(UInt4 ra, UInt4 rw,
              BFloat16x8 (&a_prefetch)[(BM * BK + NW * 64 * 8 - 1) / (NW * 64 * 8)],
              BFloat16x8 (&b_prefetch)[BN * BK / (NW * 64 * 8)], int m_offset, int n_offset,
              int num_tokens, int base, int head_stride, int thread_id) {
    constexpr int T = NW * 64, AV = (BM * BK + T * 8 - 1) / (T * 8), BV = BN * BK / (T * 8);
#pragma unroll
    for (int v = 0; v < AV; ++v) {
        int i = thread_id * 8 + v * T * 8, r = i / BK, c = i % BK;
        a_prefetch[v] = {};
        if (r < BM && m_offset + r < num_tokens) {
            a_prefetch[v] = load_bf16x8(ra, ((m_offset + r) * 128 * head_stride + base + c) * 2);
        }
    }
#pragma unroll
    for (int v = 0; v < BV; ++v) {
        int i = thread_id * 8 + v * T * 8, r = i / BK, c = i % BK;
        b_prefetch[v] = load_bf16x8(rw, ((n_offset + r) * K + base + c) * 2);
    }
}

// XOR swizzle the vector columns to spread the compute reads across LDS banks.
template <int BM, int BN, int BK, int NW>
__device__ __forceinline__ void
store_tiles_to_lds(BFloat16 *a_lds, BFloat16 *b_lds,
                   const BFloat16x8 (&a_prefetch)[(BM * BK + NW * 64 * 8 - 1) / (NW * 64 * 8)],
                   const BFloat16x8 (&b_prefetch)[BN * BK / (NW * 64 * 8)], int thread_id) {
    constexpr int T = NW * 64, AV = (BM * BK + T * 8 - 1) / (T * 8), BV = BN * BK / (T * 8);
#pragma unroll
    for (int v = 0; v < AV; ++v) {
        int i = thread_id * 8 + v * T * 8, r = i / BK, c = i % BK;
        if (r < BM) {
            *reinterpret_cast<BFloat16x8 *>(a_lds + r * BK + (c ^ ((r & (BK / 8 - 1)) * 8))) =
                a_prefetch[v];
        }
    }
#pragma unroll
    for (int v = 0; v < BV; ++v) {
        int i = thread_id * 8 + v * T * 8, r = i / BK, c = i % BK;
        *reinterpret_cast<BFloat16x8 *>(b_lds + r * BK + (c ^ ((r & (BK / 8 - 1)) * 8))) =
            b_prefetch[v];
    }
}

template <int BM, int NT, int NW, int BK, int PIPE, int K, int N>
__global__ __launch_bounds__(NW * 64) void mla_q_absorb_rope_concat_kernel(
    const BFloat16 *__restrict__ a, const BFloat16 *__restrict__ w,
    const BFloat16 *__restrict__ q_pe, const int64_t *__restrict__ positions,
    const BFloat16 *__restrict__ cos_sin, BFloat16 *__restrict__ y, int num_tokens, int head_stride,
    int rope_token_stride, int rope_head_stride, int cache_token_stride, bool neox) {
    constexpr int MT = BM / 16, BN = NT * NW * 16, T = NW * 64;
    constexpr int STAGE_BYTES = (BM + BN) * BK * 2, OUT_BYTES = BM * BN * 2;
    constexpr int SM_BYTES = STAGE_BYTES > OUT_BYTES ? STAGE_BYTES : OUT_BYTES;
    __shared__ Float4 shared_storage[(SM_BYTES + 15) / 16];
    int thread_id = threadIdx.x, lane_id = thread_id % 64, n_wave_id = thread_id / 64,
        head_id = blockIdx.z;
    int m_offset = blockIdx.y * BM, n_offset = blockIdx.x * BN, k_lane_offset = (lane_id >> 4) * 8;
    BFloat16 *a_lds = reinterpret_cast<BFloat16 *>(shared_storage);
    BFloat16 *b_lds = a_lds + BM * BK;
    a += head_id * head_stride;
    w += head_id * K * N;
    UInt4 ra = make_buffer_resource(a), rw = make_buffer_resource(w);
    BFloat16x8 a_prefetch[(BM * BK + T * 8 - 1) / (T * 8)], b_prefetch[BN * BK / (T * 8)];
    Float4 accumulators[MT][NT] = {};
    preload_tiles<BM, BN, BK, NW, K>(ra, rw, a_prefetch, b_prefetch, m_offset, n_offset, num_tokens,
                                     0, head_stride, thread_id);
    store_tiles_to_lds<BM, BN, BK, NW>(a_lds, b_lds, a_prefetch, b_prefetch, thread_id);
    __syncthreads();
#pragma unroll
    for (int base = 0; base < K; base += BK) {
        BFloat16x8 a_fragments[BK / 32][MT], b_fragments[BK / 32][NT];
#pragma unroll
        for (int kk = 0; kk < BK / 32; ++kk) {
#pragma unroll
            for (int r = 0; r < MT; ++r) {
                int row = (lane_id & 15) + r * 16, k = kk * 32 + k_lane_offset;
                a_fragments[kk][r] = *reinterpret_cast<const BFloat16x8 *>(
                    a_lds + row * BK + (k ^ ((row & (BK / 8 - 1)) * 8)));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int row = n_wave_id * NT * 16 + n * 16 + (lane_id & 15),
                    k = kk * 32 + k_lane_offset;
                b_fragments[kk][n] = *reinterpret_cast<const BFloat16x8 *>(
                    b_lds + row * BK + (k ^ ((row & (BK / 8 - 1)) * 8)));
            }
        }
        if constexpr (PIPE) {
            __syncthreads();
            if (base + BK < K) {
                preload_tiles<BM, BN, BK, NW, K>(ra, rw, a_prefetch, b_prefetch, m_offset, n_offset,
                                                 num_tokens, base + BK, head_stride, thread_id);
            }
        }
#pragma unroll
        for (int kk = 0; kk < BK / 32; ++kk) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
#pragma unroll
                    for (int r = 0; r < MT; ++r) {
                        accumulators[r][n] =
                            mma_bf16(half ? a_fragments[kk][r].hi : a_fragments[kk][r].lo,
                                     half ? b_fragments[kk][n].hi : b_fragments[kk][n].lo,
                                     accumulators[r][n]);
                    }
                }
            }
        }
        if constexpr (!PIPE) {
            __syncthreads();
        }
        if (base + BK < K) {
            if constexpr (!PIPE) {
                preload_tiles<BM, BN, BK, NW, K>(ra, rw, a_prefetch, b_prefetch, m_offset, n_offset,
                                                 num_tokens, base + BK, head_stride, thread_id);
            }
            store_tiles_to_lds<BM, BN, BK, NW>(a_lds, b_lds, a_prefetch, b_prefetch, thread_id);
        }
        __syncthreads();
    }
    // Reuse LDS after the final read barrier, then emit aligned 16-byte stores.
    BFloat16 *output_lds = reinterpret_cast<BFloat16 *>(shared_storage);
    {
#pragma unroll
        for (int r = 0; r < MT; ++r) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int row = (lane_id & 15) + r * 16,
                        col = n_wave_id * NT * 16 + n * 16 + (lane_id >> 4) + j * 4;
                    int dest = col ^ ((row & (BN / 8 - 1)) * 8);
                    output_lds[row * BN + dest] = BFloat16(accumulators[r][n][j]);
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x * 8; i < BM * BN; i += T * 8) {
        int row = i / BN, col = i % BN;
        UInt4 packed = *reinterpret_cast<const UInt4 *>(output_lds + row * BN +
                                                        (col ^ ((row & (BN / 8 - 1)) * 8)));
        if (m_offset + row < num_tokens) {
            *reinterpret_cast<UInt4 *>(y + (m_offset + row) * 128 * (N + 64) + head_id * (N + 64) +
                                       n_offset + col) = packed;
        }
    }
    // Disjoint epilogue stores need no cross-block synchronization.
    // Only N-tile zero owns the Q RoPE columns; K is not read or modified.
    if (blockIdx.x == 0) {
        for (int i = thread_id; i < BM * 64; i += T) {
            const int row = m_offset + i / 64;
            const int d = i % 64;
            if (row < num_tokens) {
                const BFloat16 *trig = cos_sin + positions[row] * cache_token_stride;
                y[(row * 128 + head_id) * (N + 64) + N + d] = rotate_bf16(
                    q_pe + row * rope_token_stride + head_id * rope_head_stride, trig, d, neox);
            }
        }
    }
}

template <int BM, int NT, int NW, int BK, int PIPE, int K, int N>
void launch_mla_bmm(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &q_pe,
                    const torch::Tensor &positions, const torch::Tensor &cos_sin, torch::Tensor &y,
                    bool neox) {
    dim3 grid(N / (NT * NW * 16), (x.size(0) + BM - 1) / BM, 128);
    mla_q_absorb_rope_concat_kernel<BM, NT, NW, BK, PIPE, K, N>
        <<<grid, NW * 64, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<BFloat16>(), w.data_ptr<BFloat16>(), q_pe.data_ptr<BFloat16>(),
            positions.data_ptr<int64_t>(), cos_sin.data_ptr<BFloat16>(), y.data_ptr<BFloat16>(),
            x.size(0), x.stride(1), q_pe.stride(0), q_pe.stride(1), cos_sin.stride(0), neox);
}

// Fixed tiles validated on BS2/3/4. V-up is not part of this implementation.
void dispatch_mla_q_absorb_concat(const torch::Tensor &x, const torch::Tensor &w,
                                  const torch::Tensor &q_pe, const torch::Tensor &positions,
                                  const torch::Tensor &cos_sin, torch::Tensor &y, bool neox) {
    if (x.size(0) == 20) {
        launch_mla_bmm<32, 1, 4, 64, 0, 128, 512>(x, w, q_pe, positions, cos_sin, y, neox);
    } else {
        launch_mla_bmm<16, 1, 2, 64, 1, 128, 512>(x, w, q_pe, positions, cos_sin, y, neox);
    }
}

torch::Tensor mla_q_absorb_rope_concat_impl(torch::Tensor x, torch::Tensor w, torch::Tensor q_pe,
                                            torch::Tensor positions, torch::Tensor cos_sin,
                                            bool neox) {
    constexpr int K = 128, N = 512;
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && x.scalar_type() == at::kBFloat16 &&
                    w.scalar_type() == at::kBFloat16,
                "MLA BMM requires BF16 tensors on the same HCU");
    TORCH_CHECK(x.dim() == 3 && (x.size(0) == 10 || x.size(0) == 15 || x.size(0) == 20) &&
                    x.size(1) == 128 && x.size(2) == K,
                "MLA BMM supports M10/15/20, H128 and its fixed K");
    TORCH_CHECK(x.stride(2) == 1 && x.stride(0) == 128 * x.stride(1) &&
                    (x.stride(1) == 128 || x.stride(1) == 192),
                "unsupported MLA input layout");
    TORCH_CHECK(w.sizes() == at::IntArrayRef({128, K, N}) &&
                    w.strides() == at::IntArrayRef({65536, 1, K}),
                "MLA BMM requires the original head-major weight storage");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "MLA BMM requires aligned tensors");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(
        std::strncmp(at::cuda::getDeviceProperties(x.get_device())->gcnArchName, "gfx936", 6) == 0,
        "MLA BMM is validated on gfx936");
    TORCH_CHECK(q_pe.device() == x.device() && q_pe.scalar_type() == at::kBFloat16 &&
                    q_pe.sizes() == at::IntArrayRef({x.size(0), 128, 64}),
                "Q-absorb concat requires BF16 rope columns [M,128,64] on the same HCU");
    TORCH_CHECK(q_pe.stride(2) == 1 && q_pe.stride(0) == 128 * q_pe.stride(1) &&
                    (q_pe.stride(1) == 64 || q_pe.stride(1) == 192) &&
                    reinterpret_cast<uintptr_t>(q_pe.data_ptr()) % 16 == 0,
                "unsupported Q-absorb rope layout or alignment");
    TORCH_CHECK(positions.device() == x.device() && positions.scalar_type() == at::kLong &&
                    positions.dim() == 1 && positions.numel() == x.size(0) &&
                    positions.is_contiguous(),
                "RoPE positions must be contiguous int64 [M] on the same HCU");
    TORCH_CHECK(cos_sin.device() == x.device() && cos_sin.scalar_type() == at::kBFloat16 &&
                    cos_sin.dim() == 2 && cos_sin.size(0) > 0 && cos_sin.size(1) == 64 &&
                    cos_sin.stride(1) == 1,
                "RoPE requires the original BF16 [max_position,64] cos/sin cache");
    // Positions are trusted model indices, as in the existing native RoPE API.
    auto y = torch::empty({x.size(0), 128, N + 64}, x.options());
    dispatch_mla_q_absorb_concat(x, w, q_pe, positions, cos_sin, y, neox);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
} // namespace

namespace at {
namespace native {
torch::Tensor mla_q_absorb_rope_concat(torch::Tensor x, torch::Tensor w, torch::Tensor q_pe,
                                       torch::Tensor positions, torch::Tensor cos_sin, bool neox) {
    return mla_q_absorb_rope_concat_impl(x, w, q_pe, positions, cos_sin, neox);
}
} // namespace native
} // namespace at
