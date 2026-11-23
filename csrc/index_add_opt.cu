// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_index_add_hip.hip for lightop P1.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <cstdlib>

static const char* lightop_getenv_dual(const char* lightop_key, const char* legacy_key) {
    const char* env = std::getenv(lightop_key);
    if (env == nullptr) {
        env = std::getenv(legacy_key);
    }
    return env;
}

#define CUDA_CHECK(cmd)                                                                          \
    do {                                                                                        \
        cudaError_t error = (cmd);                                                               \
        if (error != cudaSuccess) {                                                              \
            TORCH_CHECK(false, "HIP error: ", cudaGetErrorString(error), " at ", __FILE__, ":", \
                        __LINE__);                                                              \
        }                                                                                       \
    } while (0)

static constexpr int kWavefrontSize = 64;
static constexpr int kMaxBlockThreads = 512;
static constexpr int kMultiRowThreads = 256;
static constexpr int64_t kMultiRowDimThreshold = 256;  // probe: pipe wins for D>=256
#define GRID_REMAP_MULT 4097

// GRID_REMAP_MULT * block_idx (mod block_count) is only a bijection on
// [0, block_count) when gcd(GRID_REMAP_MULT, block_count) == 1; otherwise the
// mapping collapses and some destination rows are visited twice (double-add)
// while others are skipped entirely.
static constexpr unsigned long long grid_remap_gcd(unsigned long long a, unsigned long long b) {
    return b == 0 ? a : grid_remap_gcd(b, a % b);
}

static bool grid_remap_is_bijective(int64_t block_count) {
    return grid_remap_gcd(GRID_REMAP_MULT, static_cast<unsigned long long>(block_count)) == 1ULL;
}

__device__ __forceinline__ int map_block_idx(int block_idx, int block_count, bool use_remap) {
    if (!use_remap || block_count <= 1 || block_count == 65536 || block_count <= 4096) {
        return block_idx;
    }
    const uint32_t bc = static_cast<uint32_t>(block_count);
    return static_cast<int>((static_cast<uint64_t>(block_idx) * GRID_REMAP_MULT) % bc);
}

static int round_up_wave(int threads) {
    threads = (threads + kWavefrontSize - 1) / kWavefrontSize * kWavefrontSize;
    return threads < kWavefrontSize ? kWavefrontSize : threads;
}

// 1 block per row, software-pipelined float4 RMW (prefetch next before store current).
__global__ __launch_bounds__(512, 2) void index_add_unique_fp32_pipe_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const float* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    float* dst_row_ptr = dst + dst_row * dim;

    const int64_t nvec = dim >> 2;
    int64_t vi = static_cast<int64_t>(threadIdx.x);
    if (vi < nvec) {
        float4 s = *reinterpret_cast<const float4*>(src_row_ptr + (vi << 2));
        float4 d = *reinterpret_cast<const float4*>(dst_row_ptr + (vi << 2));
        for (vi += blockDim.x; vi < nvec; vi += blockDim.x) {
            float4 s_next = *reinterpret_cast<const float4*>(src_row_ptr + (vi << 2));
            float4 d_next = *reinterpret_cast<const float4*>(dst_row_ptr + (vi << 2));
            d.x += s.x;
            d.y += s.y;
            d.z += s.z;
            d.w += s.w;
            *reinterpret_cast<float4*>(dst_row_ptr + ((vi - blockDim.x) << 2)) = d;
            s = s_next;
            d = d_next;
        }
        {
            const int64_t last = vi - blockDim.x;
            d.x += s.x;
            d.y += s.y;
            d.z += s.z;
            d.w += s.w;
            *reinterpret_cast<float4*>(dst_row_ptr + (last << 2)) = d;
        }
    }
    for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
        dst_row_ptr[i] += src_row_ptr[i];
    }
}

// Multi-row per block: warp-per-row, 256 threads, float4x2 vectorized RMW.
__global__ __launch_bounds__(256, 4) void index_add_unique_fp32_multirow_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim, int rows_per_block) {
    const int base = static_cast<int>(blockIdx.x) * rows_per_block;
    const int warp = threadIdx.x / kWavefrontSize;
    const int lane = threadIdx.x % kWavefrontSize;
    const int warps = blockDim.x / kWavefrontSize;

    for (int r = warp; r < rows_per_block; r += warps) {
        const int src_row = base + r;
        if (src_row >= src_batch) {
            return;
        }
        const int64_t dst_row = index[src_row];
        const float* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
        float* dst_row_ptr = dst + dst_row * dim;

        const int64_t nvec = dim >> 2;
        const int64_t stride = static_cast<int64_t>(kWavefrontSize) * 2;
        for (int64_t vi = static_cast<int64_t>(lane) * 2; vi + 1 < nvec; vi += stride) {
            const int64_t i0 = vi << 2;
            const int64_t i1 = (vi + 1) << 2;
            float4 s0 = *reinterpret_cast<const float4*>(src_row_ptr + i0);
            float4 s1 = *reinterpret_cast<const float4*>(src_row_ptr + i1);
            float4 d0 = *reinterpret_cast<const float4*>(dst_row_ptr + i0);
            float4 d1 = *reinterpret_cast<const float4*>(dst_row_ptr + i1);
            d0.x += s0.x;
            d0.y += s0.y;
            d0.z += s0.z;
            d0.w += s0.w;
            d1.x += s1.x;
            d1.y += s1.y;
            d1.z += s1.z;
            d1.w += s1.w;
            *reinterpret_cast<float4*>(dst_row_ptr + i0) = d0;
            *reinterpret_cast<float4*>(dst_row_ptr + i1) = d1;
        }
        for (int64_t vi = (nvec & ~int64_t{1}) + lane; vi < nvec; vi += kWavefrontSize) {
            const int64_t i = vi << 2;
            float4 s = *reinterpret_cast<const float4*>(src_row_ptr + i);
            float4 d = *reinterpret_cast<const float4*>(dst_row_ptr + i);
            d.x += s.x;
            d.y += s.y;
            d.z += s.z;
            d.w += s.w;
            *reinterpret_cast<float4*>(dst_row_ptr + i) = d;
        }
        for (int64_t i = (nvec << 2) + lane; i < dim; i += kWavefrontSize) {
            dst_row_ptr[i] += src_row_ptr[i];
        }
    }
}

// Bitset mark: detect duplicate dst indices (atomicOr). Sets *conflict=1 on collision.
__global__ void index_add_mark_unique_kernel(const int64_t* __restrict__ index, int64_t src_batch,
                                             unsigned int* __restrict__ bits, int* __restrict__ conflict) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= src_batch) {
        return;
    }
    const int64_t dst = index[i];
    const unsigned int mask = 1u << (static_cast<unsigned int>(dst) & 31u);
    unsigned int* word = bits + (dst >> 5);
    const unsigned int old = atomicOr(word, mask);
    if (old & mask) {
        atomicExch(conflict, 1);
    }
}

// Row-tiled FP32 atomics: 1 block/src-row, float4 load + 4×atomicAdd.
__global__ __launch_bounds__(512, 2) void index_add_atomic_fp32_row_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const float* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    float* dst_row_ptr = dst + dst_row * dim;

    const int64_t nvec = dim >> 2;
    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
        const float4 s = *reinterpret_cast<const float4*>(src_row_ptr + (vi << 2));
        float* dp = dst_row_ptr + (vi << 2);
        atomicAdd(dp + 0, s.x);
        atomicAdd(dp + 1, s.y);
        atomicAdd(dp + 2, s.z);
        atomicAdd(dp + 3, s.w);
    }
    for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(dst_row_ptr + i, src_row_ptr[i]);
    }
}

// Dual-row FP32: interleaved atomics when the pair targets distinct dst rows;
// sequential+syncthreads when the pair collides on the same dst (correctness).
// dst is intentionally non-__restrict__ so the compiler may not assume dst0!=dst1.
__global__ __launch_bounds__(512, 2) void index_add_atomic_fp32_dualrow_kernel(
    const float* __restrict__ src, float* dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, bool use_remap) {
    const int pair = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    const int src_row0 = pair * 2;
    if (src_row0 >= src_batch) {
        return;
    }
    const int64_t nvec = dim >> 2;
    const float* src0 = src + static_cast<int64_t>(src_row0) * dim;
    const int64_t dst_row0 = index[src_row0];
    float* dst0 = dst + dst_row0 * dim;

    const int src_row1 = src_row0 + 1;
    if (src_row1 >= src_batch) {
        for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
            const float4 s = *reinterpret_cast<const float4*>(src0 + (vi << 2));
            float* dp = dst0 + (vi << 2);
            atomicAdd(dp + 0, s.x);
            atomicAdd(dp + 1, s.y);
            atomicAdd(dp + 2, s.z);
            atomicAdd(dp + 3, s.w);
        }
        for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
            atomicAdd(dst0 + i, src0[i]);
        }
        return;
    }

    const int64_t dst_row1 = index[src_row1];
    const float* src1 = src + static_cast<int64_t>(src_row1) * dim;
    float* dst1 = dst + dst_row1 * dim;

    if (dst_row0 != dst_row1) {
        for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
            const int64_t off = vi << 2;
            float4 s0 = *reinterpret_cast<const float4*>(src0 + off);
            float4 s1 = *reinterpret_cast<const float4*>(src1 + off);
            float* d0 = dst0 + off;
            float* d1 = dst1 + off;
            atomicAdd(d0 + 0, s0.x);
            atomicAdd(d1 + 0, s1.x);
            atomicAdd(d0 + 1, s0.y);
            atomicAdd(d1 + 1, s1.y);
            atomicAdd(d0 + 2, s0.z);
            atomicAdd(d1 + 2, s1.z);
            atomicAdd(d0 + 3, s0.w);
            atomicAdd(d1 + 3, s1.w);
        }
        for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
            atomicAdd(dst0 + i, src0[i]);
            atomicAdd(dst1 + i, src1[i]);
        }
        return;
    }

    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
        const float4 s = *reinterpret_cast<const float4*>(src0 + (vi << 2));
        float* dp = dst0 + (vi << 2);
        atomicAdd(dp + 0, s.x);
        atomicAdd(dp + 1, s.y);
        atomicAdd(dp + 2, s.z);
        atomicAdd(dp + 3, s.w);
    }
    for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(dst0 + i, src0[i]);
    }
    __syncthreads();
    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
        const float4 s = *reinterpret_cast<const float4*>(src1 + (vi << 2));
        float* dp = dst1 + (vi << 2);
        atomicAdd(dp + 0, s.x);
        atomicAdd(dp + 1, s.y);
        atomicAdd(dp + 2, s.z);
        atomicAdd(dp + 3, s.w);
    }
    for (int64_t i = (nvec << 2) + threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(dst1 + i, src1[i]);
    }
}

// Multi-row atomic FP32 for tiny D (warp-per-row).
__global__ __launch_bounds__(256, 4) void index_add_atomic_fp32_multirow_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim, int rows_per_block) {
    const int base = static_cast<int>(blockIdx.x) * rows_per_block;
    const int warp = threadIdx.x / kWavefrontSize;
    const int lane = threadIdx.x % kWavefrontSize;
    const int warps = blockDim.x / kWavefrontSize;

    for (int r = warp; r < rows_per_block; r += warps) {
        const int src_row = base + r;
        if (src_row >= src_batch) {
            return;
        }
        const int64_t dst_row = index[src_row];
        const float* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
        float* dst_row_ptr = dst + dst_row * dim;

        const int64_t nvec = dim >> 2;
        for (int64_t vi = lane; vi < nvec; vi += kWavefrontSize) {
            const int64_t i = vi << 2;
            float4 s = *reinterpret_cast<const float4*>(src_row_ptr + i);
            float* dp = dst_row_ptr + i;
            atomicAdd(dp + 0, s.x);
            atomicAdd(dp + 1, s.y);
            atomicAdd(dp + 2, s.z);
            atomicAdd(dp + 3, s.w);
        }
        for (int64_t i = (nvec << 2) + lane; i < dim; i += kWavefrontSize) {
            atomicAdd(dst_row_ptr + i, src_row_ptr[i]);
        }
    }
}

// Torch-style flat grid-stride (IndexIsMajor): consecutive threads hit consecutive cols
// of the same src row → same atomic destination pattern as indexFuncLargeIndex.
__global__ __launch_bounds__(256, 8) void index_add_atomic_fp32_flat_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim) {
    const int64_t n = src_batch * dim;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; tid < n;
         tid += stride) {
        const int64_t src_row = tid / dim;
        const int64_t col = tid - src_row * dim;
        const int64_t dst_row = index[src_row];
        atomicAdd(dst + dst_row * dim + col, src[src_row * dim + col]);
    }
}

// Flat float4: amortize index lookup; 4 atomics per thread (torch VecSize=4 style).
__global__ __launch_bounds__(256, 8) void index_add_atomic_fp32_flat_vec_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim) {
    const int64_t nvec_row = dim >> 2;
    const int64_t nvec = src_batch * nvec_row;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; tid < nvec;
         tid += stride) {
        const int64_t src_row = tid / nvec_row;
        const int64_t vcol = tid - src_row * nvec_row;
        const int64_t dst_row = index[src_row];
        const int64_t off = (vcol << 2);
        const float4 s = *reinterpret_cast<const float4*>(src + src_row * dim + off);
        float* dp = dst + dst_row * dim + off;
        atomicAdd(dp + 0, s.x);
        atomicAdd(dp + 1, s.y);
        atomicAdd(dp + 2, s.z);
        atomicAdd(dp + 3, s.w);
    }
}

// 1-block-per-row scalar atomics (no float4) — coalesced atomics beat float4-atomic on gfx938.
__global__ __launch_bounds__(512, 2) void index_add_atomic_fp32_row_scalar_kernel(
    const float* __restrict__ src, float* __restrict__ dst, const int64_t* __restrict__ index,
    int64_t src_batch, int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const float* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    float* dst_row_ptr = dst + dst_row * dim;
    // Dual-stride ILP: two independent atomic streams per thread.
    const int64_t stride = static_cast<int64_t>(blockDim.x) * 2;
    for (int64_t i = threadIdx.x; i < dim; i += stride) {
        atomicAdd(dst_row_ptr + i, src_row_ptr[i]);
        const int64_t i2 = i + blockDim.x;
        if (i2 < dim) {
            atomicAdd(dst_row_ptr + i2, src_row_ptr[i2]);
        }
    }
}

template <typename T>
struct HalfTypeOps;

template <>
struct HalfTypeOps<at::Half> {
    static __device__ __forceinline__ float to_float(at::Half v) {
        return __half2float(static_cast<__half>(v));
    }
    static __device__ __forceinline__ at::Half from_float(float v) { return at::Half(__float2half_rn(v)); }
};

template <>
struct HalfTypeOps<at::BFloat16> {
    static __device__ __forceinline__ float to_float(at::BFloat16 v) {
        const uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
        const uint32_t u = static_cast<uint32_t>(bits) << 16;
        return *reinterpret_cast<const float*>(&u);
    }
    static __device__ __forceinline__ at::BFloat16 from_float(float v) {
        uint32_t u = *reinterpret_cast<uint32_t*>(&v);
        const uint32_t lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb;
        at::BFloat16 out;
        *reinterpret_cast<uint16_t*>(&out) = static_cast<uint16_t>(u >> 16);
        return out;
    }
};

// CAS-based half/bf16 atomic add (ROCm has no native half atomicAdd).
template <typename T>
__device__ __forceinline__ void atomic_add_half(T* addr, T val) {
    unsigned int* address_as_ui =
        reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(addr) - (reinterpret_cast<size_t>(addr) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        T cur;
        *reinterpret_cast<uint16_t*>(&cur) =
            static_cast<uint16_t>((reinterpret_cast<size_t>(addr) & 2) ? (old >> 16) : (old & 0xffffu));
        const T out = HalfTypeOps<T>::from_float(HalfTypeOps<T>::to_float(cur) + HalfTypeOps<T>::to_float(val));
        const uint16_t bits = *reinterpret_cast<const uint16_t*>(&out);
        const unsigned int neu =
            (reinterpret_cast<size_t>(addr) & 2) ? ((old & 0xffffu) | (static_cast<unsigned int>(bits) << 16))
                                                : ((old & 0xffff0000u) | bits);
        old = atomicCAS(address_as_ui, assumed, neu);
    } while (assumed != old);
}

// One CAS updates two packed halves (dst row contiguous → naturally 4B-aligned pairs).
template <typename T>
__device__ __forceinline__ void atomic_add_half2_u32(unsigned int* addr_u32, unsigned int src_u32) {
    unsigned int old = *addr_u32;
    unsigned int assumed;
    do {
        assumed = old;
        const T* ov = reinterpret_cast<const T*>(&assumed);
        const T* sv = reinterpret_cast<const T*>(&src_u32);
        unsigned int neu;
        T* nv = reinterpret_cast<T*>(&neu);
        nv[0] = HalfTypeOps<T>::from_float(HalfTypeOps<T>::to_float(ov[0]) + HalfTypeOps<T>::to_float(sv[0]));
        nv[1] = HalfTypeOps<T>::from_float(HalfTypeOps<T>::to_float(ov[1]) + HalfTypeOps<T>::to_float(sv[1]));
        old = atomicCAS(addr_u32, assumed, neu);
    } while (assumed != old);
}

// FP16: packed __hadd2 inside CAS (avoids scalar float round-trips).
template <>
__device__ __forceinline__ void atomic_add_half2_u32<at::Half>(unsigned int* addr_u32,
                                                                unsigned int src_u32) {
    unsigned int old = *addr_u32;
    unsigned int assumed;
    do {
        assumed = old;
        const __half2 sum =
            __hadd2(*reinterpret_cast<const __half2*>(&assumed), *reinterpret_cast<const __half2*>(&src_u32));
        const unsigned int neu = *reinterpret_cast<const unsigned int*>(&sum);
        old = atomicCAS(addr_u32, assumed, neu);
    } while (assumed != old);
}

// BF16: packed bf16x2 add inside CAS.
template <>
__device__ __forceinline__ void atomic_add_half2_u32<at::BFloat16>(unsigned int* addr_u32,
                                                                   unsigned int src_u32) {
    unsigned int old = *addr_u32;
    unsigned int assumed;
    do {
        assumed = old;
        // Forward-declare uses bf16x2_add_u32 defined below; keep local convert here to
        // avoid ordering issues — inlined float path matching HalfTypeOps.
        const at::BFloat16* ov = reinterpret_cast<const at::BFloat16*>(&assumed);
        const at::BFloat16* sv = reinterpret_cast<const at::BFloat16*>(&src_u32);
        unsigned int neu;
        at::BFloat16* nv = reinterpret_cast<at::BFloat16*>(&neu);
        nv[0] = HalfTypeOps<at::BFloat16>::from_float(HalfTypeOps<at::BFloat16>::to_float(ov[0]) +
                                                       HalfTypeOps<at::BFloat16>::to_float(sv[0]));
        nv[1] = HalfTypeOps<at::BFloat16>::from_float(HalfTypeOps<at::BFloat16>::to_float(ov[1]) +
                                                       HalfTypeOps<at::BFloat16>::to_float(sv[1]));
        old = atomicCAS(addr_u32, assumed, neu);
    } while (assumed != old);
}

// atomic_add_half2_u32 reinterprets the destination as a 4B-aligned unsigned int*
// and issues a single atomicCAS on it, which is only well-defined when that address
// is actually 4B-aligned. Row base addresses are offset by dst_row * dim * sizeof(T);
// when dim is odd this offset's alignment depends on dst_row's parity (data-dependent,
// not known at launch time), so some rows land on a misaligned address at runtime.
// atomic_add_half itself stays correct for any 2B-aligned address (it locates the
// containing 4B word via `addr & 2`), so misaligned rows fall back to it scalarly.
__device__ __forceinline__ bool is_u32_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0x3u) == 0u;
}

template <typename T>
__device__ __forceinline__ void atomic_add_half_row(
    T* __restrict__ dst_row_ptr, const T* __restrict__ src_row_ptr, int64_t dim, int lane, int lane_stride) {
    // The packed path does a 4B load from src and a 4B atomicCAS on dst, so both
    // row starts must be naturally aligned; an odd dim makes their alignment
    // depend on the row index, and src/dst row parity can differ (index remaps
    // rows independently).
    if (!is_u32_aligned(dst_row_ptr) || !is_u32_aligned(src_row_ptr)) {
        for (int64_t i = lane; i < dim; i += lane_stride) {
            atomic_add_half(dst_row_ptr + i, src_row_ptr[i]);
        }
        return;
    }
    const int64_t npair = dim >> 1;
    for (int64_t pi = lane; pi < npair; pi += lane_stride) {
        const unsigned int s = *reinterpret_cast<const unsigned int*>(src_row_ptr + (pi << 1));
        atomic_add_half2_u32<T>(reinterpret_cast<unsigned int*>(dst_row_ptr + (pi << 1)), s);
    }
    if ((dim & 1) && lane == 0) {
        atomic_add_half(dst_row_ptr + (npair << 1), src_row_ptr[npair << 1]);
    }
}

// 1 block/src-row: half2-packed CAS.
template <typename T>
__global__ __launch_bounds__(512, 2) void index_add_atomic_half_row_kernel(
    const T* __restrict__ src, T* __restrict__ dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const T* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    T* dst_row_ptr = dst + dst_row * dim;

    atomic_add_half_row<T>(dst_row_ptr, src_row_ptr, dim, threadIdx.x, blockDim.x);
}

// Dual-row half CAS: interleaved when dst rows differ; sequential on collision.
template <typename T>
__global__ __launch_bounds__(512, 2) void index_add_atomic_half_dualrow_kernel(
    const T* __restrict__ src, T* dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, bool use_remap) {
    const int pair = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    const int src_row0 = pair * 2;
    if (src_row0 >= src_batch) {
        return;
    }
    const T* src0 = src + static_cast<int64_t>(src_row0) * dim;
    const int64_t dst_row0 = index[src_row0];
    T* dst0 = dst + dst_row0 * dim;

    const int src_row1 = src_row0 + 1;
    if (src_row1 >= src_batch) {
        atomic_add_half_row<T>(dst0, src0, dim, threadIdx.x, blockDim.x);
        return;
    }

    const int64_t dst_row1 = index[src_row1];
    const T* src1 = src + static_cast<int64_t>(src_row1) * dim;
    T* dst1 = dst + dst_row1 * dim;

    if (dst_row0 != dst_row1) {
        atomic_add_half_row<T>(dst0, src0, dim, threadIdx.x, blockDim.x);
        atomic_add_half_row<T>(dst1, src1, dim, threadIdx.x, blockDim.x);
        return;
    }

    atomic_add_half_row<T>(dst0, src0, dim, threadIdx.x, blockDim.x);
    __syncthreads();
    atomic_add_half_row<T>(dst1, src1, dim, threadIdx.x, blockDim.x);
}

template <typename T>
__global__ __launch_bounds__(256, 4) void index_add_atomic_half_multirow_kernel(
    const T* __restrict__ src, T* __restrict__ dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, int rows_per_block) {
    const int base = static_cast<int>(blockIdx.x) * rows_per_block;
    const int warp = threadIdx.x / kWavefrontSize;
    const int lane = threadIdx.x % kWavefrontSize;
    const int warps = blockDim.x / kWavefrontSize;

    for (int r = warp; r < rows_per_block; r += warps) {
        const int src_row = base + r;
        if (src_row >= src_batch) {
            return;
        }
        const int64_t dst_row = index[src_row];
        const T* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
        T* dst_row_ptr = dst + dst_row * dim;

        atomic_add_half_row<T>(dst_row_ptr, src_row_ptr, dim, lane, kWavefrontSize);
    }
}

// Generic float-accum path (bf16 / fallback).
template <typename T>
__device__ __forceinline__ uint4 half_vec8_add(uint4 d_bits, uint4 s_bits) {
    const T* s_data = reinterpret_cast<const T*>(&s_bits);
    const T* d_data = reinterpret_cast<const T*>(&d_bits);
    uint4 out_bits;
    T* out_data = reinterpret_cast<T*>(&out_bits);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        out_data[k] = HalfTypeOps<T>::from_float(HalfTypeOps<T>::to_float(d_data[k]) +
                                                  HalfTypeOps<T>::to_float(s_data[k]));
    }
    return out_bits;
}

template <>
__device__ __forceinline__ uint4 half_vec8_add<at::Half>(uint4 d_bits, uint4 s_bits) {
    uint4 out;
    const __half2* d2 = reinterpret_cast<const __half2*>(&d_bits);
    const __half2* s2 = reinterpret_cast<const __half2*>(&s_bits);
    __half2* o2 = reinterpret_cast<__half2*>(&out);
#pragma unroll
    for (int k = 0; k < 4; ++k) o2[k] = __hadd2(d2[k], s2[k]);
    return out;
}

// Packed BF16x2 add via float (RNE-ish truncate), 4× fewer scalar convert loops than generic.
__device__ __forceinline__ uint32_t bf16x2_add_u32(uint32_t d, uint32_t s) {
    float d0 = __uint_as_float(d << 16);
    float d1 = __uint_as_float(d & 0xffff0000u);
    float s0 = __uint_as_float(s << 16);
    float s1 = __uint_as_float(s & 0xffff0000u);
    uint32_t u0 = __float_as_uint(d0 + s0);
    uint32_t u1 = __float_as_uint(d1 + s1);
    u0 += 0x7fffu + ((u0 >> 16) & 1u);
    u1 += 0x7fffu + ((u1 >> 16) & 1u);
    return (u0 >> 16) | (u1 & 0xffff0000u);
}

__device__ __forceinline__ uint4 bf16_vec8_add(uint4 d, uint4 s) {
    uint4 o;
    const uint32_t* dp = reinterpret_cast<const uint32_t*>(&d);
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(&s);
    uint32_t* op = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) op[i] = bf16x2_add_u32(dp[i], sp[i]);
    return o;
}

template <>
__device__ __forceinline__ uint4 half_vec8_add<at::BFloat16>(uint4 d_bits, uint4 s_bits) {
    return bf16_vec8_add(d_bits, s_bits);
}

// 1 block per row, 2-wide software-pipelined uint4 (vec8 half) RMW for ILP.
template <typename T>
__global__ __launch_bounds__(512, 2) void index_add_unique_half_pipe_kernel(
    const T* __restrict__ src, T* __restrict__ dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const T* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    T* dst_row_ptr = dst + dst_row * dim;

    const int64_t nvec = dim >> 3;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * 2;
    int64_t vi = static_cast<int64_t>(threadIdx.x) * 2;
    if (vi + 1 < nvec) {
        uint4 s0 = *reinterpret_cast<const uint4*>(src_row_ptr + (vi << 3));
        uint4 s1 = *reinterpret_cast<const uint4*>(src_row_ptr + ((vi + 1) << 3));
        uint4 d0 = *reinterpret_cast<const uint4*>(dst_row_ptr + (vi << 3));
        uint4 d1 = *reinterpret_cast<const uint4*>(dst_row_ptr + ((vi + 1) << 3));
        for (vi += stride; vi + 1 < nvec; vi += stride) {
            uint4 s0n = *reinterpret_cast<const uint4*>(src_row_ptr + (vi << 3));
            uint4 s1n = *reinterpret_cast<const uint4*>(src_row_ptr + ((vi + 1) << 3));
            uint4 d0n = *reinterpret_cast<const uint4*>(dst_row_ptr + (vi << 3));
            uint4 d1n = *reinterpret_cast<const uint4*>(dst_row_ptr + ((vi + 1) << 3));
            *reinterpret_cast<uint4*>(dst_row_ptr + ((vi - stride) << 3)) = half_vec8_add<T>(d0, s0);
            *reinterpret_cast<uint4*>(dst_row_ptr + ((vi - stride + 1) << 3)) =
                half_vec8_add<T>(d1, s1);
            s0 = s0n;
            s1 = s1n;
            d0 = d0n;
            d1 = d1n;
        }
        {
            const int64_t last = vi - stride;
            *reinterpret_cast<uint4*>(dst_row_ptr + (last << 3)) = half_vec8_add<T>(d0, s0);
            *reinterpret_cast<uint4*>(dst_row_ptr + ((last + 1) << 3)) = half_vec8_add<T>(d1, s1);
        }
    }
    for (int64_t v = (nvec & ~int64_t{1}) + threadIdx.x; v < nvec; v += blockDim.x) {
        uint4 s = *reinterpret_cast<const uint4*>(src_row_ptr + (v << 3));
        uint4 d = *reinterpret_cast<const uint4*>(dst_row_ptr + (v << 3));
        *reinterpret_cast<uint4*>(dst_row_ptr + (v << 3)) = half_vec8_add<T>(d, s);
    }
    for (int64_t i = (nvec << 3) + threadIdx.x; i < dim; i += blockDim.x) {
        float acc = HalfTypeOps<T>::to_float(dst_row_ptr[i]) + HalfTypeOps<T>::to_float(src_row_ptr[i]);
        dst_row_ptr[i] = HalfTypeOps<T>::from_float(acc);
    }
}

// Dedicated BF16 pipe kernel (__launch_bounds__ 512,2): packed bf16_vec8_add + software pipe.
// Uses 2-wide depth (matches FP16 ILP schedule). A 4-live-wide variant was slower on gfx938.
__global__ __launch_bounds__(512, 2) void index_add_unique_bf16_pipe_kernel(
    const at::BFloat16* __restrict__ src, at::BFloat16* __restrict__ dst,
    const int64_t* __restrict__ index, int64_t src_batch, int64_t dim, bool use_remap) {
    const int src_row = map_block_idx(blockIdx.x, gridDim.x, use_remap);
    if (src_row >= src_batch) {
        return;
    }
    const int64_t dst_row = index[src_row];
    const at::BFloat16* __restrict__ src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
    at::BFloat16* __restrict__ dst_row_ptr = dst + dst_row * dim;

    const int64_t nvec = dim >> 3;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * 2;
    int64_t vi = static_cast<int64_t>(threadIdx.x) * 2;
    if (vi + 1 < nvec) {
        uint4 s0 = *reinterpret_cast<const uint4*>(src_row_ptr + (vi << 3));
        uint4 s1 = *reinterpret_cast<const uint4*>(src_row_ptr + ((vi + 1) << 3));
        uint4 d0 = *reinterpret_cast<const uint4*>(dst_row_ptr + (vi << 3));
        uint4 d1 = *reinterpret_cast<const uint4*>(dst_row_ptr + ((vi + 1) << 3));
        for (vi += stride; vi + 1 < nvec; vi += stride) {
            uint4 s0n = *reinterpret_cast<const uint4*>(src_row_ptr + (vi << 3));
            uint4 s1n = *reinterpret_cast<const uint4*>(src_row_ptr + ((vi + 1) << 3));
            uint4 d0n = *reinterpret_cast<const uint4*>(dst_row_ptr + (vi << 3));
            uint4 d1n = *reinterpret_cast<const uint4*>(dst_row_ptr + ((vi + 1) << 3));
            *reinterpret_cast<uint4*>(dst_row_ptr + ((vi - stride) << 3)) = bf16_vec8_add(d0, s0);
            *reinterpret_cast<uint4*>(dst_row_ptr + ((vi - stride + 1) << 3)) = bf16_vec8_add(d1, s1);
            s0 = s0n;
            s1 = s1n;
            d0 = d0n;
            d1 = d1n;
        }
        {
            const int64_t last = vi - stride;
            *reinterpret_cast<uint4*>(dst_row_ptr + (last << 3)) = bf16_vec8_add(d0, s0);
            *reinterpret_cast<uint4*>(dst_row_ptr + ((last + 1) << 3)) = bf16_vec8_add(d1, s1);
        }
    }
    for (int64_t v = (nvec & ~int64_t{1}) + threadIdx.x; v < nvec; v += blockDim.x) {
        uint4 s = *reinterpret_cast<const uint4*>(src_row_ptr + (v << 3));
        uint4 d = *reinterpret_cast<const uint4*>(dst_row_ptr + (v << 3));
        *reinterpret_cast<uint4*>(dst_row_ptr + (v << 3)) = bf16_vec8_add(d, s);
    }
    for (int64_t i = (nvec << 3) + threadIdx.x; i < dim; i += blockDim.x) {
        float acc = HalfTypeOps<at::BFloat16>::to_float(dst_row_ptr[i]) +
                    HalfTypeOps<at::BFloat16>::to_float(src_row_ptr[i]);
        dst_row_ptr[i] = HalfTypeOps<at::BFloat16>::from_float(acc);
    }
}



// Multi-row per block: warp-per-row, 256 threads, uint4 vec8 RMW (tiny D).
template <typename T>
__global__ __launch_bounds__(256, 4) void index_add_unique_half_multirow_kernel(
    const T* __restrict__ src, T* __restrict__ dst, const int64_t* __restrict__ index, int64_t src_batch,
    int64_t dim, int rows_per_block) {
    const int base = static_cast<int>(blockIdx.x) * rows_per_block;
    const int warp = threadIdx.x / kWavefrontSize;
    const int lane = threadIdx.x % kWavefrontSize;
    const int warps = blockDim.x / kWavefrontSize;

    for (int r = warp; r < rows_per_block; r += warps) {
        const int src_row = base + r;
        if (src_row >= src_batch) {
            return;
        }
        const int64_t dst_row = index[src_row];
        const T* src_row_ptr = src + static_cast<int64_t>(src_row) * dim;
        T* dst_row_ptr = dst + dst_row * dim;

        const int64_t nvec = dim >> 3;
        const int64_t stride = static_cast<int64_t>(kWavefrontSize) * 2;
        for (int64_t vi = static_cast<int64_t>(lane) * 2; vi + 1 < nvec; vi += stride) {
            const int64_t i0 = vi << 3;
            const int64_t i1 = (vi + 1) << 3;
            uint4 s0 = *reinterpret_cast<const uint4*>(src_row_ptr + i0);
            uint4 s1 = *reinterpret_cast<const uint4*>(src_row_ptr + i1);
            uint4 d0 = *reinterpret_cast<const uint4*>(dst_row_ptr + i0);
            uint4 d1 = *reinterpret_cast<const uint4*>(dst_row_ptr + i1);
            *reinterpret_cast<uint4*>(dst_row_ptr + i0) = half_vec8_add<T>(d0, s0);
            *reinterpret_cast<uint4*>(dst_row_ptr + i1) = half_vec8_add<T>(d1, s1);
        }
        for (int64_t vi = (nvec & ~int64_t{1}) + lane; vi < nvec; vi += kWavefrontSize) {
            const int64_t i = vi << 3;
            uint4 s = *reinterpret_cast<const uint4*>(src_row_ptr + i);
            uint4 d = *reinterpret_cast<const uint4*>(dst_row_ptr + i);
            *reinterpret_cast<uint4*>(dst_row_ptr + i) = half_vec8_add<T>(d, s);
        }
        for (int64_t i = (nvec << 3) + lane; i < dim; i += kWavefrontSize) {
            float acc =
                HalfTypeOps<T>::to_float(dst_row_ptr[i]) + HalfTypeOps<T>::to_float(src_row_ptr[i]);
            dst_row_ptr[i] = HalfTypeOps<T>::from_float(acc);
        }
    }
}

static void launch_index_add_unique_fp32(const float* src, float* dst, const int64_t* index,
                                         int64_t src_batch, int64_t dim, cudaStream_t stream) {
    if (dim < kMultiRowDimThreshold) {
        // Warp-per-row for tiny D only; pipe is faster from D>=256 on gfx938.
        const int rows_per_block = (dim <= 256) ? 8 : (dim <= 1024 ? 4 : 2);
        const int blocks = static_cast<int>((src_batch + rows_per_block - 1) / rows_per_block);
        index_add_unique_fp32_multirow_kernel<<<blocks, kMultiRowThreads, 0, stream>>>(
            src, dst, index, src_batch, dim, rows_per_block);
    } else {
        int threads = 256;
        if (dim >= 16384) {
            threads = 512;
        }
        const int blocks = static_cast<int>(src_batch);
        const bool use_remap = grid_remap_is_bijective(blocks);
        index_add_unique_fp32_pipe_kernel<<<blocks, threads, 0, stream>>>(src, dst, index, src_batch,
                                                                         dim, use_remap);
    }
}

static void launch_index_add_atomic_fp32(const float* src, float* dst, const int64_t* index,
                                         int64_t src_batch, int64_t dim, cudaStream_t stream) {
    // Mode override for A/B: 0=auto, 1=row-vec4, 2=row-scalar, 3=flat, 4=dual, 5=flat-vec4
    const char* mode_env = lightop_getenv_dual("LIGHTOP_INDEX_ADD_ATOMIC_FP32", "CUSTOM_OPS_INDEX_ADD_ATOMIC_FP32");
    const int mode = (mode_env && mode_env[0] >= '0' && mode_env[0] <= '5' && mode_env[1] == '\0')
                         ? (mode_env[0] - '0')
                         : 0;

    auto launch_flat = [&](bool vec4) {
        const int threads = 256;
        const int64_t work = vec4 ? (src_batch * (dim >> 2)) : (src_batch * dim);
        int blocks = static_cast<int>((work + threads - 1) / threads);
        constexpr int kMaxBlocks = 65535;
        if (blocks > kMaxBlocks) {
            blocks = kMaxBlocks;
        }
        if (vec4 && (dim & 3) == 0) {
            index_add_atomic_fp32_flat_vec_kernel<<<blocks, threads, 0, stream>>>(src, dst, index,
                                                                                 src_batch, dim);
        } else {
            index_add_atomic_fp32_flat_kernel<<<blocks, threads, 0, stream>>>(src, dst, index,
                                                                            src_batch, dim);
        }
    };

    auto launch_row_scalar = [&]() {
        int threads = 256;
        if (dim < 256) {
            threads = round_up_wave(static_cast<int>(dim));
            if (threads > 256) {
                threads = 256;
            }
        } else if (dim >= 4096) {
            threads = 512;
        }
        const bool use_remap = grid_remap_is_bijective(src_batch);
        index_add_atomic_fp32_row_scalar_kernel<<<static_cast<int>(src_batch), threads, 0, stream>>>(
            src, dst, index, src_batch, dim, use_remap);
    };

    if (mode == 1) {
        int threads = (dim >= 4096) ? 512 : 256;
        const bool use_remap = grid_remap_is_bijective(src_batch);
        index_add_atomic_fp32_row_kernel<<<static_cast<int>(src_batch), threads, 0, stream>>>(
            src, dst, index, src_batch, dim, use_remap);
        return;
    }
    if (mode == 2) {
        launch_row_scalar();
        return;
    }
    if (mode == 3) {
        launch_flat(/*vec4=*/false);
        return;
    }
    if (mode == 4 && src_batch >= 2) {
        int threads = (dim >= 4096) ? 512 : 256;
        const int pairs = static_cast<int>((src_batch + 1) / 2);
        const bool use_remap = grid_remap_is_bijective(pairs);
        index_add_atomic_fp32_dualrow_kernel<<<pairs, threads, 0, stream>>>(src, dst, index, src_batch,
                                                                           dim, use_remap);
        return;
    }
    if (mode == 5) {
        launch_flat(/*vec4=*/true);
        return;
    }

    // Default (mode 0): coalesced row-scalar atomics beat torch flat (~730→1300+ GB/s).
    // Small grids lack occupancy with 1-block/row → fall back to flat (ATen-like).
    if (dim < 64) {
        const int rows_per_block = (dim <= 16) ? 8 : 4;
        const int blocks = static_cast<int>((src_batch + rows_per_block - 1) / rows_per_block);
        index_add_atomic_fp32_multirow_kernel<<<blocks, kMultiRowThreads, 0, stream>>>(
            src, dst, index, src_batch, dim, rows_per_block);
        return;
    }
    if (src_batch < 1024) {
        launch_flat(/*vec4=*/false);
        return;
    }
    launch_row_scalar();
}

template <typename T>
static void launch_index_add_atomic_half(const T* src, T* dst, const int64_t* index, int64_t src_batch,
                                         int64_t dim, cudaStream_t stream) {
    if (dim < 256) {
        const int rows_per_block = (dim <= 64) ? 8 : 4;
        const int blocks = static_cast<int>((src_batch + rows_per_block - 1) / rows_per_block);
        index_add_atomic_half_multirow_kernel<T><<<blocks, kMultiRowThreads, 0, stream>>>(
            src, dst, index, src_batch, dim, rows_per_block);
        return;
    }
    int threads = 256;
    if (dim >= 4096) {
        threads = 512;
    } else if (dim < 512) {
        threads = round_up_wave(static_cast<int>(dim >> 1));
        if (threads > 256) {
            threads = 256;
        }
    }
    // Default dual-row for half/bf16 (~+5–10% BW). CUSTOM_OPS_INDEX_ADD_ATOMIC_ROW=1 → 1-row.
    const char* row_env = lightop_getenv_dual("LIGHTOP_INDEX_ADD_ATOMIC_ROW", "CUSTOM_OPS_INDEX_ADD_ATOMIC_ROW");
    const bool force_row = row_env && row_env[0] == '1' && row_env[1] == '\0';
    if (!force_row && src_batch >= 2) {
        const int pairs = static_cast<int>((src_batch + 1) / 2);
        const bool use_remap = grid_remap_is_bijective(pairs);
        index_add_atomic_half_dualrow_kernel<T><<<pairs, threads, 0, stream>>>(src, dst, index,
                                                                              src_batch, dim, use_remap);
    } else {
        const bool use_remap = grid_remap_is_bijective(src_batch);
        index_add_atomic_half_row_kernel<T><<<static_cast<int>(src_batch), threads, 0, stream>>>(
            src, dst, index, src_batch, dim, use_remap);
    }
}

// Device-side uniqueness probe with reused workspace + pinned D2H (avoids torch::zeros/.item).
// src_batch > dst_batch ⇒ pigeonhole collision (skip probe).
static bool probe_indices_unique(torch::Tensor index, int64_t src_batch, int64_t dst_batch,
                                 cudaStream_t stream) {
    if (src_batch > dst_batch) {
        return false;
    }
    if (src_batch <= 1) {
        return true;
    }

    const int64_t nwords = (dst_batch + 31) / 32;
    // Persist across calls: micro_perf hammers the same op with similar shapes.
    static thread_local torch::Tensor bits;
    static thread_local torch::Tensor conflict;
    static thread_local int32_t* pinned_conflict = nullptr;
    if (pinned_conflict == nullptr) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_conflict), sizeof(int32_t),
                                cudaHostAllocPortable));
    }
    if (!bits.defined() || bits.numel() < nwords || bits.device() != index.device()) {
        bits = torch::empty({nwords}, index.options().dtype(torch::kInt32));
    }
    if (!conflict.defined() || conflict.device() != index.device()) {
        conflict = torch::empty({1}, index.options().dtype(torch::kInt32));
    }

    CUDA_CHECK(cudaMemsetAsync(bits.data_ptr<int32_t>(), 0,
                             static_cast<size_t>(nwords) * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(conflict.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    const int threads = 256;
    const int blocks = static_cast<int>((src_batch + threads - 1) / threads);
    index_add_mark_unique_kernel<<<blocks, threads, 0, stream>>>(
        index.data_ptr<int64_t>(), src_batch, reinterpret_cast<unsigned int*>(bits.data_ptr<int32_t>()),
        conflict.data_ptr<int32_t>());
    CUDA_CHECK(cudaMemcpyAsync(pinned_conflict, conflict.data_ptr<int32_t>(), sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return *pinned_conflict == 0;
}

static bool index_add_probe_enabled() {
    // Default ON: micro_perf permutations recover unique-path BW when assume_unique=False.
    // Set CUSTOM_OPS_INDEX_ADD_PROBE=0 to always take the atomic path.
    const char* env = lightop_getenv_dual("LIGHTOP_INDEX_ADD_PROBE", "CUSTOM_OPS_INDEX_ADD_PROBE");
    if (env == nullptr) {
        return true;
    }
    return !(env[0] == '0' && env[1] == '\0');
}

// Probe is ~10-20us (memset+mark+pinned D2H). Skip only for tiny ops where atomic is fine.
static bool index_add_probe_worth_it(int64_t src_batch, int64_t dim, int64_t elem_size) {
    // Traffic ≈ 3 * src_batch * dim * elem at ~1500 GB/s.
    const double bytes = 3.0 * static_cast<double>(src_batch) * static_cast<double>(dim) *
                         static_cast<double>(elem_size);
    const double est_us = bytes / 1.5e6;  // bytes / (GB/s) → µs
    return est_us >= 15.0;
}

static void launch_index_add_unique_fp16(const at::Half* src, at::Half* dst, const int64_t* index,
                                         int64_t src_batch, int64_t dim, cudaStream_t stream) {
    if (dim < kMultiRowDimThreshold) {
        const int rows_per_block = (dim <= 256) ? 8 : (dim <= 1024 ? 4 : 2);
        const int blocks = static_cast<int>((src_batch + rows_per_block - 1) / rows_per_block);
        index_add_unique_half_multirow_kernel<at::Half><<<blocks, kMultiRowThreads, 0, stream>>>(
            src, dst, index, src_batch, dim, rows_per_block);
    } else {
        // 2-wide pipe + __hadd2
        const int64_t pairs = (dim >> 3) >> 1;  // nvec/2
        int threads = 256;
        if (dim >= 4096) {
            threads = 512;
        } else if (dim >= 1024) {
            threads = 256;
        } else {
            threads = 128;
        }
        if (pairs > 0 && pairs < threads) {
            threads = round_up_wave(static_cast<int>(pairs));
            if (threads > 512) {
                threads = 512;
            }
        }
        const int blocks = static_cast<int>(src_batch);
        const bool use_remap = grid_remap_is_bijective(blocks);
        index_add_unique_half_pipe_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            src, dst, index, src_batch, dim, use_remap);
    }
}

static void launch_index_add_unique_bf16(const at::BFloat16* src, at::BFloat16* dst,
                                         const int64_t* index, int64_t src_batch, int64_t dim,
                                         cudaStream_t stream) {
    if (dim < kMultiRowDimThreshold) {
        const int rows_per_block = (dim <= 256) ? 8 : (dim <= 1024 ? 4 : 2);
        const int blocks = static_cast<int>((src_batch + rows_per_block - 1) / rows_per_block);
        index_add_unique_half_multirow_kernel<at::BFloat16><<<blocks, kMultiRowThreads, 0, stream>>>(
            src, dst, index, src_batch, dim, rows_per_block);
    } else {
        // Match FP16 thread schedule; 512 threads for D>=4096 (and large D).
        const int64_t pairs = (dim >> 3) >> 1;
        int threads = 256;
        if (dim >= 4096) {
            threads = 512;
        } else if (dim >= 1024) {
            threads = 256;
        } else {
            threads = 128;
        }
        if (pairs > 0 && pairs < threads) {
            threads = round_up_wave(static_cast<int>(pairs));
            if (threads > 512) {
                threads = 512;
            }
        }
        const int blocks = static_cast<int>(src_batch);
        const bool use_remap = grid_remap_is_bijective(blocks);
        index_add_unique_bf16_pipe_kernel<<<blocks, threads, 0, stream>>>(src, dst, index, src_batch,
                                                                         dim, use_remap);
    }
}





void index_add_opt_impl(torch::Tensor& dst, torch::Tensor index, torch::Tensor src, bool assume_unique) {
    TORCH_CHECK(dst.is_cuda(), "custom_ops index_add: dst must be CUDA/HIP");
    TORCH_CHECK(src.is_cuda(), "custom_ops index_add: src must be CUDA/HIP");
    TORCH_CHECK(index.is_cuda(), "custom_ops index_add: index must be CUDA/HIP");
    TORCH_CHECK(dst.dim() == 2 && src.dim() == 2, "custom_ops index_add: expected 2D tensors");
    TORCH_CHECK(dst.size(1) == src.size(1), "custom_ops index_add: dim size mismatch");
    TORCH_CHECK(index.dim() == 1 && index.size(0) == src.size(0),
                "custom_ops index_add: index must be [src_batch]");
    TORCH_CHECK(index.scalar_type() == at::kLong, "custom_ops index_add: index must be int64");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && index.is_contiguous(),
                "custom_ops index_add: tensors must be contiguous");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
                "custom_ops index_add: dtype mismatch between dst and src");

    const int64_t src_batch = src.size(0);
    const int64_t dst_batch = dst.size(0);
    const int64_t dim = src.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(dst.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t elem_size = dst.element_size();

    // assume_unique=True → fast unique path (caller guarantees injective index).
    // assume_unique=False → only take the unique RMW path when the device-side
    //   probe actually confirms no duplicate indices; equal src/dst batch size
    //   does not by itself imply the index is a permutation (duplicates with
    //   src_batch == dst_batch are a legal input whose repeats must still
    //   accumulate), so it must never be treated as proof of uniqueness.
    //   Otherwise fall back to the vectorized atomic path (fp32 atomicAdd /
    //   half CAS), which is correct regardless of duplicates.
    bool unique = assume_unique;
    if (!unique && index_add_probe_enabled() &&
        index_add_probe_worth_it(src_batch, dim, elem_size)) {
        unique = probe_indices_unique(index, src_batch, dst_batch, stream);
    }

    if (dst.scalar_type() == at::kFloat) {
        // Also covers micro_perf dtype "tfloat32" (torch.float32 storage).
        const float* src_ptr = src.data_ptr<float>();
        float* dst_ptr = dst.data_ptr<float>();
        const int64_t* idx_ptr = index.data_ptr<int64_t>();
        if (unique) {
            launch_index_add_unique_fp32(src_ptr, dst_ptr, idx_ptr, src_batch, dim, stream);
        } else {
            launch_index_add_atomic_fp32(src_ptr, dst_ptr, idx_ptr, src_batch, dim, stream);
        }
    } else if (dst.scalar_type() == at::kHalf) {
        if (unique) {
            launch_index_add_unique_fp16(src.data_ptr<at::Half>(), dst.data_ptr<at::Half>(),
                                         index.data_ptr<int64_t>(), src_batch, dim, stream);
        } else {
            launch_index_add_atomic_half<at::Half>(src.data_ptr<at::Half>(), dst.data_ptr<at::Half>(),
                                                   index.data_ptr<int64_t>(), src_batch, dim, stream);
        }
    } else if (dst.scalar_type() == at::kBFloat16) {
        if (unique) {
            launch_index_add_unique_bf16(src.data_ptr<at::BFloat16>(), dst.data_ptr<at::BFloat16>(),
                                         index.data_ptr<int64_t>(), src_batch, dim, stream);
        } else {
            launch_index_add_atomic_half<at::BFloat16>(
                src.data_ptr<at::BFloat16>(), dst.data_ptr<at::BFloat16>(), index.data_ptr<int64_t>(),
                src_batch, dim, stream);
        }
    } else {
        TORCH_CHECK(false, "custom_ops index_add: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}


namespace at {
namespace native {

torch::Tensor index_add_opt(
    const torch::Tensor& dst,
    int64_t dim,
    const torch::Tensor& index,
    const torch::Tensor& src,
    bool assume_unique) {
  TORCH_CHECK(dim == 0, "index_add_opt: only dim=0 supported");
  auto out = dst.clone();
  index_add_opt_impl(out, index, src, assume_unique);
  return out;
}

torch::Tensor index_add_opt_out(
    torch::Tensor& dst,
    int64_t dim,
    const torch::Tensor& index,
    const torch::Tensor& src,
    bool assume_unique) {
  TORCH_CHECK(dim == 0, "index_add_opt: only dim=0 supported");
  index_add_opt_impl(dst, index, src, assume_unique);
  return dst;
}

}  // namespace native
}  // namespace at
