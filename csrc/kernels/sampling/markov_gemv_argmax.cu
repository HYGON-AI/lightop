// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "markov_gemv_argmax.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

namespace lightop::sampling {
namespace {

using BFloat16 = c10::BFloat16;

constexpr int kRank = 256;
constexpr int kThreads = 256;
constexpr int kLanes = 16;
constexpr int kRows = kThreads / kLanes;
constexpr int kVector = 8;
constexpr int kItems = kRank / kLanes;
constexpr int kSharedPitch = kLanes + 1;

struct ArgMaxPair {
    float value;
    int index;
};

// Match torch.argmax: the first NaN wins; equal finite values and infinities
// choose the lowest vocabulary index. INT_MAX denotes an invalid tail lane.
__device__ __forceinline__ ArgMaxPair better(ArgMaxPair a, ArgMaxPair b) {
    const bool a_nan = isnan(a.value);
    const bool b_nan = isnan(b.value);
    if (a_nan || b_nan) {
        if (a_nan != b_nan) {
            return a_nan ? a : b;
        }
        return a.index < b.index ? a : b;
    }
    if (a.value == b.value) {
        return a.index < b.index ? a : b;
    }
    return a.value > b.value ? a : b;
}

#ifdef USE_ROCM
// Order by value first, then by the smallest vocabulary index. All NaN
// payloads share the highest value key; both signed zeros share one key.
// Key zero is reserved for invalid lanes, below every valid float value.
__device__ __forceinline__ uint64_t ordered_argmax_key(float value, int index) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    if (magnitude == 0) {
        ordered = 0x80000000u;
    }
    if (magnitude > 0x7f800000u) {
        ordered = 0xffffffffu;
    }
    return (static_cast<uint64_t>(ordered) << 32) |
        static_cast<uint32_t>(INT_MAX - index);
}

__device__ __forceinline__ uint64_t shuffle_key_xor(uint64_t key, int delta, int width) {
    const uint32_t lo = __shfl_xor(static_cast<uint32_t>(key), delta, width);
    const uint32_t hi = __shfl_xor(static_cast<uint32_t>(key >> 32), delta, width);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

__device__ __forceinline__ uint64_t key_max(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}
using BFloat16x4 = short __attribute__((vector_size(8)));
using Float4 = float __attribute__((vector_size(16)));
using UInt4 = uint32_t __attribute__((vector_size(16)));
constexpr int kMatrixRows = 128;
constexpr int kMatrixWaves = kMatrixRows / 16;
constexpr int kMatrixThreads = kMatrixWaves * 64;

__device__ __forceinline__ UInt4 make_weight_resource(const BFloat16* p) {
    UInt4 resource = {};
    *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(p);
    // Tile-relative offsets stay inside this 2 GiB buffer descriptor.
    resource[2] = 0x80000000;
    resource[3] = 0x00020000;
    return resource;
}

template <int Rows, int Threads>
__device__ __forceinline__ void preload_weight_stage(
    UInt4 resource, uint4 (&packets)[Rows * 64 / (Threads * 8)],
    int vocab_begin, int vocab_size, int stage, int thread) {
#pragma unroll
    for (int i = 0; i < Rows * 64 / (Threads * 8); ++i) {
        const int item = (thread + i * Threads) * 8;
        const int row = item / 64;
        const int k = item % 64;
        packets[i] = {};
        if (vocab_begin + row < vocab_size) {
            auto value = __builtin_amdgcn_buffer_load_dwordx4(
                resource, 0, (row * kRank + stage + k) * 2, false, true);
            packets[i] = __builtin_bit_cast(uint4, value);
        }
    }
    return;
}

template <int Rows, int Threads>
__device__ __forceinline__ void store_weight_stage(
    BFloat16* shared_weight, const uint4 (&packets)[Rows * 64 / (Threads * 8)], int thread) {
#pragma unroll
    for (int i = 0; i < Rows * 64 / (Threads * 8); ++i) {
        const int item = (thread + i * Threads) * 8;
        const int row = item / 64;
        const int k = item % 64;
        const int swizzle = (row & 15) * 4;
        *reinterpret_cast<uint2*>(shared_weight + row * 64 + (k ^ swizzle)) =
            make_uint2(packets[i].x, packets[i].y);
        *reinterpret_cast<uint2*>(shared_weight + row * 64 + ((k + 4) ^ swizzle)) =
            make_uint2(packets[i].z, packets[i].w);
    }
    return;
}

// Only one to four latent rows are live. Spread their aligned BF16x4
// fragments across LDS banks and retain the complete rank for all K stages.
__device__ __forceinline__ int latent_swizzle(int row) {
    return (row & 1) * 4 + ((row >> 1) & 1) * 32;
}

template <int Batch, int Threads>
__device__ __forceinline__ void store_latent(
    const BFloat16* latent, BFloat16* shared_latent, int thread) {
    for (int item = thread; item < Batch * kRank; item += Threads) {
        const int row = item / kRank;
        const int k = item % kRank;
        shared_latent[row * kRank + (k ^ latent_swizzle(row))] = latent[item];
    }
    if (thread == 0) {
        // Inactive MMAC columns all read this one aligned zero fragment.
        *reinterpret_cast<uint2*>(shared_latent + Batch * kRank) = make_uint2(0, 0);
    }
    return;
}

// gfx936 matrix-core path. Stage vector weight loads in registers and issue
// the next stage's reads before MMAC. The 4-BF16 XOR layout retains aligned
// 8-byte LDS reads while distributing the matrix fragments across the banks.
template <int Batch, typename LogitType>
__global__ __launch_bounds__(kMatrixThreads) void markov_mmac_add_partial_argmax_kernel(
    const BFloat16* __restrict__ latent,
    const BFloat16* __restrict__ weight,
    const LogitType* __restrict__ base_logits,
    LogitType* __restrict__ corrected_logits,
    uint64_t* __restrict__ partials,
    int vocab_size,
    int64_t base_row_stride,
    int num_tiles) {
#if defined(__gfx936__)
    constexpr int Rows = kMatrixRows;
    constexpr int Stage = 64;
    constexpr int Swizzle = 4;
    constexpr int Waves = kMatrixWaves;
    constexpr int Threads = Waves * 64;
    constexpr int SwizzleMask = Stage / Swizzle - 1;
    __shared__ BFloat16 shared_weight[Rows * Stage];
    __shared__ uint2 latent_storage[(Batch * kRank + 4) / 4];
    auto* shared_latent = reinterpret_cast<BFloat16*>(latent_storage);
    __shared__ uint64_t wave_winners[Batch * Waves];
    const int thread = threadIdx.x;
    const int lane = thread % 64;
    const int wave = thread / 64;
    const int vocab_begin = blockIdx.x * Rows;
    Float4 accumulator = {};

    uint4 packets[Rows * Stage / (Threads * 8)];
    const UInt4 resource = make_weight_resource(weight + static_cast<int64_t>(vocab_begin) * kRank);
    preload_weight_stage<Rows, Threads>(resource, packets, vocab_begin, vocab_size, 0, thread);
    store_weight_stage<Rows, Threads>(shared_weight, packets, thread);
    store_latent<Batch, Threads>(latent, shared_latent, thread);
    __syncthreads();
#pragma unroll
    for (int stage = 0; stage < kRank; stage += Stage) {
        BFloat16x4 weight_fragments[Stage / 32][2];
        BFloat16x4 latent_fragments[Stage / 32][2];
#pragma unroll
        for (int kk = 0; kk < Stage / 32; ++kk) {
            const int column = lane & 15;
            const int k = kk * 32 + (lane >> 4) * 8;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int offset = k + half * 4;
                const int latent_index = column < Batch ?
                    column * kRank + ((stage + offset) ^ latent_swizzle(column)) :
                    Batch * kRank;
                latent_fragments[kk][half] = *reinterpret_cast<const BFloat16x4*>(
                    shared_latent + latent_index);
                const int row = wave * 16 + (lane & 15);
                weight_fragments[kk][half] = *reinterpret_cast<const BFloat16x4*>(
                    shared_weight + row * Stage +
                    (offset ^ ((row & SwizzleMask) * Swizzle)));
            }
        }
        __syncthreads();
        if (stage + Stage < kRank) {
            preload_weight_stage<Rows, Threads>(
                resource, packets, vocab_begin, vocab_size, stage + Stage, thread);
        }
#pragma unroll
        for (int kk = 0; kk < Stage / 32; ++kk) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                accumulator = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                    weight_fragments[kk][half], latent_fragments[kk][half], accumulator);
            }
        }
        if (stage + Stage < kRank) {
            store_weight_stage<Rows, Threads>(shared_weight, packets, thread);
        }
        __syncthreads();
    }
    const int batch = lane >> 4;
    if (batch < Batch) {
        const int vocab = vocab_begin + wave * 16 + (lane & 15);
        uint64_t winner = 0;
        if (vocab < vocab_size) {
            // Preserve the separate BF16 projection rounding before logits add.
            const BFloat16 bias = BFloat16(accumulator[0]);
            const LogitType corrected = LogitType(
                float(bias) + float(base_logits[batch * base_row_stride + vocab]));
            corrected_logits[static_cast<int64_t>(batch) * vocab_size + vocab] = corrected;
            winner = ordered_argmax_key(float(corrected), vocab);
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            winner = key_max(winner, shuffle_key_xor(winner, offset, 16));
        }
        if ((lane & 15) == 0) {
            wave_winners[batch * Waves + wave] = winner;
        }
    }
    __syncthreads();
    if (thread < Batch) {
        const int batch_index = thread;
        uint64_t winner = wave_winners[batch_index * Waves];
#pragma unroll
        for (int wave_index = 1; wave_index < Waves; ++wave_index) {
            winner = key_max(winner, wave_winners[batch_index * Waves + wave_index]);
        }
        partials[batch_index * num_tiles + blockIdx.x] = winner;
    }
#endif
    return;
}

// Keep the tile scan distributed over eight waves, then exchange only one
// winner per wave. This needs one block barrier instead of a full shared tree.
template <int Threads>
__global__ __launch_bounds__(Threads) void markov_argmax_wave_finish_kernel(
    const uint64_t* __restrict__ partials, int64_t* __restrict__ tokens, int num_tiles) {
    constexpr int Waves = Threads / 64;
    __shared__ uint64_t wave_winners[Waves];
    const int thread = threadIdx.x;
    const int lane = thread % 64;
    uint64_t winner = 0;
    for (int tile = thread; tile < num_tiles; tile += Threads) {
        winner = key_max(winner, partials[blockIdx.x * num_tiles + tile]);
    }
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        winner = key_max(winner, shuffle_key_xor(winner, offset, 64));
    }
    if (lane == 0) {
        wave_winners[thread / 64] = winner;
    }
    __syncthreads();
    if (thread == 0) {
        winner = wave_winners[0];
#pragma unroll
        for (int wave = 1; wave < Waves; ++wave) {
            winner = key_max(winner, wave_winners[wave]);
        }
        tokens[blockIdx.x] = INT_MAX - static_cast<uint32_t>(winner);
    }
    return;
}

#endif

__device__ __forceinline__ float shuffle_xor(float value, int offset, int width) {
#ifdef USE_ROCM
    return __shfl_xor(value, offset, width);
#else
    return __shfl_xor_sync(0xffffffff, value, offset, width);
#endif
}

__device__ __forceinline__ int shuffle_xor(int value, int offset, int width) {
#ifdef USE_ROCM
    return __shfl_xor(value, offset, width);
#else
    return __shfl_xor_sync(0xffffffff, value, offset, width);
#endif
}


template <int Batch, typename LogitType>
__global__ void markov_gemv_add_partial_argmax_kernel(
    const BFloat16* __restrict__ latent,
    const BFloat16* __restrict__ weight,
    const LogitType* __restrict__ base_logits,
    LogitType* __restrict__ corrected_logits,
    ArgMaxPair* __restrict__ partials,
    int vocab_size,
    int64_t base_row_stride,
    int num_tiles) {
    __shared__ float shared_latent[Batch * kItems * kSharedPitch];
    __shared__ float shared_logits[Batch * kRows];

    const int thread = threadIdx.x;
    const int lane = thread % kLanes;
    const int row = thread / kLanes;
    const int vocab = blockIdx.x * kRows + row;

    // Transpose the small latent vectors once. Dot-product reads then use
    // consecutive shared-memory banks rather than strided BF16 accesses.
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
        const int item = (thread / (kLanes * kVector)) * kVector + thread % kVector;
        const int column = (thread % (kLanes * kVector)) / kVector;
        shared_latent[batch * kItems * kSharedPitch + item * kSharedPitch + column] =
            static_cast<float>(latent[batch * kRank + thread]);
    }
    __syncthreads();

    float sums[Batch] = {};
    if (vocab < vocab_size) {
#pragma unroll
        for (int block = 0; block < kRank / (kLanes * kVector); ++block) {
            const int column = block * kLanes * kVector + lane * kVector;
            const uint4 packed = *reinterpret_cast<const uint4*>(
                weight + static_cast<int64_t>(vocab) * kRank + column);
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int item = 0; item < kVector; ++item) {
                const uint32_t bits = (words[item / 2] >> (16 * (item % 2))) & 0xffffu;
                const float w = __uint_as_float(bits << 16);
#pragma unroll
                for (int batch = 0; batch < Batch; ++batch) {
                    const float x = shared_latent[
                        batch * kItems * kSharedPitch +
                        (block * kVector + item) * kSharedPitch + lane];
                    sums[batch] = fmaf(x, w, sums[batch]);
                }
            }
        }
    }

#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
#pragma unroll
        for (int offset = kLanes / 2; offset > 0; offset /= 2) {
            sums[batch] += shuffle_xor(sums[batch], offset, kLanes);
        }
        if (lane == 0) {
            float value = -INFINITY;
            if (vocab < vocab_size) {
                // Round the Linear output to BF16; preserve the logits output dtype.
                const BFloat16 bias = static_cast<BFloat16>(sums[batch]);
                const LogitType corrected = static_cast<LogitType>(
                    static_cast<float>(bias) +
                    static_cast<float>(base_logits[batch * base_row_stride + vocab]));
                corrected_logits[static_cast<int64_t>(batch) * vocab_size + vocab] = corrected;
                value = static_cast<float>(corrected);
            }
            shared_logits[batch * kRows + row] = value;
        }
    }
    __syncthreads();

    // Every subgroup participates in the shuffle, including on wave64.
    // Only thread zero writes the one winner for this vocabulary tile.
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
        const int local_row = thread % kRows;
        const int index = blockIdx.x * kRows + local_row;
        ArgMaxPair winner = {
            shared_logits[batch * kRows + local_row],
            index < vocab_size ? index : INT_MAX};
#pragma unroll
        for (int offset = kRows / 2; offset > 0; offset /= 2) {
            winner = better(winner, {
                shuffle_xor(winner.value, offset, kRows),
                shuffle_xor(winner.index, offset, kRows)});
        }
        if (thread == 0) {
            partials[batch * num_tiles + blockIdx.x] = winner;
        }
    }
    return;
}

__global__ void markov_argmax_finish_kernel(
    const ArgMaxPair* __restrict__ partials,
    int64_t* __restrict__ tokens,
    int num_tiles) {
    __shared__ ArgMaxPair winners[kThreads];
    const int thread = threadIdx.x;
    ArgMaxPair winner = {-INFINITY, INT_MAX};
    for (int tile = thread; tile < num_tiles; tile += kThreads) {
        winner = better(winner, partials[blockIdx.x * num_tiles + tile]);
    }
    winners[thread] = winner;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            winners[thread] = better(winners[thread], winners[thread + offset]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        tokens[blockIdx.x] = winners[0].index;
    }
    return;
}

template <int Batch, typename LogitType>
void launch(
    const torch::Tensor& latent,
    const torch::Tensor& weight,
    const torch::Tensor& base_logits,
    torch::Tensor& corrected,
    torch::Tensor& tokens,
    torch::Tensor& workspace,
    int vocab_size,
    int num_tiles,
    bool matrix_core) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* partials = reinterpret_cast<ArgMaxPair*>(workspace.data_ptr<int32_t>());
#ifdef USE_ROCM
    if (matrix_core) {
        markov_mmac_add_partial_argmax_kernel<Batch, LogitType>
            <<<num_tiles, kMatrixThreads, 0, stream>>>(
                latent.data_ptr<BFloat16>(), weight.data_ptr<BFloat16>(),
                base_logits.data_ptr<LogitType>(), corrected.data_ptr<LogitType>(),
                reinterpret_cast<uint64_t*>(partials),
                vocab_size, base_logits.stride(0), num_tiles);
    } else
#endif
    {
        markov_gemv_add_partial_argmax_kernel<Batch, LogitType>
            <<<num_tiles, kThreads, 0, stream>>>(
                latent.data_ptr<BFloat16>(), weight.data_ptr<BFloat16>(),
                base_logits.data_ptr<LogitType>(), corrected.data_ptr<LogitType>(), partials,
                vocab_size, base_logits.stride(0), num_tiles);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef USE_ROCM
    if (matrix_core) {
        markov_argmax_wave_finish_kernel<kMatrixThreads>
            <<<Batch, kMatrixThreads, 0, stream>>>(
                reinterpret_cast<const uint64_t*>(partials),
                tokens.data_ptr<int64_t>(), num_tiles);
    } else
#endif
    {
        markov_argmax_finish_kernel<<<Batch, kThreads, 0, stream>>>(
            partials, tokens.data_ptr<int64_t>(), num_tiles);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
}

template <typename LogitType>
void dispatch(
    const torch::Tensor& latent,
    const torch::Tensor& weight,
    const torch::Tensor& base_logits,
    torch::Tensor& corrected,
    torch::Tensor& tokens,
    torch::Tensor& workspace,
    int vocab_size,
    int num_tiles,
    bool matrix_core) {
    switch (latent.size(0)) {
        case 1:
            launch<1, LogitType>(latent, weight, base_logits, corrected, tokens,
                                workspace, vocab_size, num_tiles, matrix_core);
            break;
        case 2:
            launch<2, LogitType>(latent, weight, base_logits, corrected, tokens,
                                workspace, vocab_size, num_tiles, matrix_core);
            break;
        case 3:
            launch<3, LogitType>(latent, weight, base_logits, corrected, tokens,
                                workspace, vocab_size, num_tiles, matrix_core);
            break;
        case 4:
            launch<4, LogitType>(latent, weight, base_logits, corrected, tokens,
                                workspace, vocab_size, num_tiles, matrix_core);
            break;
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> markov_gemv_add_argmax(
    const torch::Tensor& latent,
    const torch::Tensor& weight,
    const torch::Tensor& base_logits) {
    TORCH_CHECK(latent.is_cuda() && weight.is_cuda() && base_logits.is_cuda(),
                "markov_gemv_add_argmax requires GPU tensors");
    TORCH_CHECK(latent.device() == weight.device() && latent.device() == base_logits.device(),
                "all tensors must be on the same GPU");
    TORCH_CHECK(latent.scalar_type() == torch::kBFloat16 &&
                weight.scalar_type() == torch::kBFloat16,
                "latent and weight must be BF16");
    TORCH_CHECK(base_logits.scalar_type() == torch::kBFloat16 ||
                base_logits.scalar_type() == torch::kFloat32,
                "base_logits must be BF16 or FP32");
    TORCH_CHECK(latent.dim() == 2 && weight.dim() == 2 && base_logits.dim() == 2,
                "all inputs must have two dimensions");
    const auto batch = latent.size(0);
    const auto vocab = weight.size(0);
    TORCH_CHECK(batch >= 1 && batch <= 4, "supported batch sizes are 1, 2, 3 and 4");
    TORCH_CHECK(latent.size(1) == kRank && weight.size(1) == kRank,
                "Markov rank must be 256");
    TORCH_CHECK(vocab > 0 && vocab <= INT_MAX, "invalid vocabulary size");
    TORCH_CHECK(base_logits.size(0) == batch && base_logits.size(1) == vocab,
                "base_logits must have shape [batch, vocabulary]");
    TORCH_CHECK(latent.is_contiguous() && weight.is_contiguous(),
                "latent and weight must be contiguous");
    TORCH_CHECK(base_logits.stride(1) == 1 && base_logits.stride(0) >= vocab,
                "base_logits must have contiguous, non-overlapping rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
                "weight must be aligned to 16 bytes");

    const c10::cuda::CUDAGuard guard(latent.device());
    bool matrix_core = false;
    int tile_rows = kRows;
#ifdef USE_ROCM
    const std::string architecture = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    matrix_core = architecture.rfind("gfx936", 0) == 0;
    if (matrix_core) {
        tile_rows = kMatrixRows;
    }
#endif
    const int num_tiles = (vocab + tile_rows - 1) / tile_rows;
    auto corrected = torch::empty({batch, vocab}, base_logits.options());
    auto tokens = torch::empty({batch}, latent.options().dtype(torch::kInt64));
    auto workspace = torch::empty({batch, num_tiles, 2}, latent.options().dtype(torch::kInt32));
    if (base_logits.scalar_type() == torch::kFloat32) {
        dispatch<float>(latent, weight, base_logits, corrected, tokens, workspace,
                        vocab, num_tiles, matrix_core);
    } else {
        dispatch<BFloat16>(latent, weight, base_logits, corrected, tokens, workspace,
                           vocab, num_tiles, matrix_core);
    }
    return {corrected, tokens};
}


namespace {

#ifdef USE_ROCM
using Int2 = int __attribute__((vector_size(8)));

// Decode the existing monotonic score/low-ID key back to the public pair
// format. NaN payloads and signed zeros may be canonicalized; their ordering
// and winning token IDs are unchanged. Zero key denotes an invalid lane.
__device__ __forceinline__ ArgMaxPair argmax_pair_from_key(uint64_t key) {
    if (key == 0) return {-INFINITY, INT_MAX};
    const uint32_t ordered = static_cast<uint32_t>(key >> 32);
    const uint32_t bits = (ordered & 0x80000000u) ?
        (ordered ^ 0x80000000u) : ~ordered;
    return {__uint_as_float(bits), INT_MAX - static_cast<int>(static_cast<uint32_t>(key))};
}

// One aligned packet per thread covers a TileRows x 64 stage for the
// existing four-threads-per-row block shape. Read the next stage into VGPRs
// before computing on the current fragments; final scheduling is compiler-led.
__device__ __forceinline__ uint4 load_fp8_weight_stage64(
    const uint8_t* weight, int begin, int valid_width, int tid, int stage) {
    const int row = tid / 4, k = (tid % 4) * 16;
    uint4 data = {};
    if (begin + row < valid_width) {
        data = *reinterpret_cast<const uint4*>(
            weight + static_cast<int64_t>(begin + row) * 256 + stage + k);
    }
    return data;
}

__device__ __forceinline__ void store_fp8_weight_stage64(
    uint8_t* shared_weight, uint4 data, int tid) {
    const int row = tid / 4, k = (tid % 4) * 16;
    const int swizzle = (row & 7) * 8;
    *reinterpret_cast<uint2*>(shared_weight + row * 64 + (k ^ swizzle)) =
        make_uint2(data.x, data.y);
    *reinterpret_cast<uint2*>(shared_weight + row * 64 + ((k + 8) ^ swizzle)) =
        make_uint2(data.z, data.w);
}

// One block covers TileRows vocabulary entries and 16 requests. Stage coalesced
// The default stages K256; opt-in K64 reuses one smaller weight buffer.
// Both consume rank256 with the same eight MMACs in the same K order.
// The operand/output lane mapping follows bmm_scale/fused_bmm{,_utils}.
template <typename LogitType, typename IndexType, int TileRows, bool PackedArgmax, int KStage>
__global__ __launch_bounds__((TileRows / 16) * 64) void dsv41_markov_fp8_partial_kernel(
    const LogitType* __restrict__ base, const IndexType* __restrict__ prev,
    const uint8_t* __restrict__ embedding, const float* __restrict__ xs,
    const uint8_t* __restrict__ weight, const float* __restrict__ ws,
    ArgMaxPair* __restrict__ partials, int batch, int valid_width,
    int vocab, int start, int parts, int64_t base_stride, int64_t prev_stride,
    int64_t xs_stride, int64_t ws_stride) {
#if defined(__gfx938__)
    static_assert(TileRows == 64 || TileRows == 128);
    static_assert(KStage == 64 || KStage == 256);
    constexpr int kWaves = TileRows / 16;
    constexpr int kBlockThreads = kWaves * 64;
    __shared__ uint4 weight_storage[TileRows * KStage / 16];
    __shared__ uint4 latent_storage[16 * 256 / 16];
    __shared__ float latent_scales[16];
    __shared__ std::conditional_t<PackedArgmax, uint64_t, ArgMaxPair> winners[16 * kWaves];
    auto* shared_weight = reinterpret_cast<uint8_t*>(weight_storage);
    auto* shared_latent = reinterpret_cast<uint8_t*>(latent_storage);
    const int tid = threadIdx.x, lane = tid & 63, wave = tid / 64;
    const int begin = blockIdx.x * TileRows, batch_begin = blockIdx.y * 16;
    // Global 16-byte packets are consecutive within each row. The XOR layout
    // preserves aligned 8-byte matrix fragments and spreads LDS banks.
    if constexpr (KStage == 256) {
#pragma unroll
        for (int packet = tid; packet < TileRows * 256 / 16; packet += kBlockThreads) {
            const int row = packet / 16, k = (packet % 16) * 16;
            uint4 data = {};
            if (begin + row < valid_width) {
                data = *reinterpret_cast<const uint4*>(
                    weight + static_cast<int64_t>(begin + row) * 256 + k);
            }
            const int swizzle = (row & 15) * 8;
            *reinterpret_cast<uint2*>(shared_weight + row * 256 + (k ^ swizzle)) =
                make_uint2(data.x, data.y);
            *reinterpret_cast<uint2*>(shared_weight + row * 256 + ((k + 8) ^ swizzle)) =
                make_uint2(data.z, data.w);
        }
    } else {
        const uint4 data = load_fp8_weight_stage64(weight, begin, valid_width, tid, 0);
        store_fp8_weight_stage64(shared_weight, data, tid);
    }
    if (tid < 16 * 256 / 16) {
        const int row = tid / 16, k = (tid % 16) * 16;
        const int b = batch_begin + row;
        int64_t token = b < batch ? static_cast<int64_t>(prev[b * prev_stride]) : -1;
        const bool valid = token >= 0 && token < vocab;
        uint4 data = {};
        if (valid) {
            data = *reinterpret_cast<const uint4*>(embedding + token * 256 + k);
        }
        const int swizzle = row * 8;
        *reinterpret_cast<uint2*>(shared_latent + row * 256 + (k ^ swizzle)) =
            make_uint2(data.x, data.y);
        *reinterpret_cast<uint2*>(shared_latent + row * 256 + ((k + 8) ^ swizzle)) =
            make_uint2(data.z, data.w);
        if (k == 0) latent_scales[row] = valid ? xs[token * xs_stride] : 0.0f;
    }
    __syncthreads();
    Float4 acc = {};
    if constexpr (KStage == 256) {
#pragma unroll
        for (int k = 0; k < 256; k += 32) {
            const int row = wave * 16 + (lane & 15);
            const int offset = k + (lane >> 4) * 8;
            const Int2 a = *reinterpret_cast<const Int2*>(
                shared_weight + row * 256 + (offset ^ ((row & 15) * 8)));
            const int column = lane & 15;
            const Int2 b = *reinterpret_cast<const Int2*>(
                shared_latent + column * 256 + (offset ^ (column * 8)));
            acc = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                a, b, acc, false, false);
        }
    } else {
#pragma unroll
        for (int stage = 0; stage < 256; stage += 64) {
            Int2 a[2], b[2];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int row = wave * 16 + (lane & 15);
                const int offset = kk * 32 + (lane >> 4) * 8;
                a[kk] = *reinterpret_cast<const Int2*>(
                    shared_weight + row * 64 + (offset ^ ((row & 7) * 8)));
                const int column = lane & 15;
                b[kk] = *reinterpret_cast<const Int2*>(
                    shared_latent + column * 256 + ((stage + offset) ^ (column * 8)));
            }
            uint4 next = {};
            if (stage + 64 < 256) {
                // All waves have consumed the current weight buffer before
                // it can be reused. Stage fragments remain live in registers.
                __syncthreads();
                next = load_fp8_weight_stage64(weight, begin, valid_width, tid, stage + 64);
            }
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                acc = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                    a[kk], b[kk], acc, false, false);
            }
            if (stage + 64 < 256) {
                store_fp8_weight_stage64(shared_weight, next, tid);
                __syncthreads();
            }
        }
    }
    const int local_vocab = begin + wave * 16 + (lane & 15);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int local_batch = (lane >> 4) + reg * 4;
        const int b = batch_begin + local_batch;
        ArgMaxPair best = {-INFINITY, INT_MAX};
        if (b < batch && local_vocab < valid_width) {
            // Preserve the original FP8 GEMM epilogue: FP32 token scale,
            // FP32 channel scale, BF16 projection rounding, FP32 base add.
            float projection = __fmul_rn(acc[reg], latent_scales[local_batch]);
            projection = __fmul_rn(projection, ws[local_vocab * ws_stride]);
            const BFloat16 bias = BFloat16(projection);
            best = {__fadd_rn(float(bias), float(base[b * base_stride + local_vocab])),
                    start + local_vocab};
        }
        if constexpr (PackedArgmax) {
            uint64_t key = best.index == INT_MAX ? 0 : ordered_argmax_key(best.value, best.index);
#pragma unroll
            for (int delta = 8; delta > 0; delta /= 2) {
                key = key_max(key, shuffle_key_xor(key, delta, 16));
            }
            if ((lane & 15) == 0) winners[local_batch * kWaves + wave] = key;
        } else {
#pragma unroll
            for (int delta = 8; delta > 0; delta /= 2) {
                best = better(best, {shuffle_xor(best.value, delta, 16),
                                     shuffle_xor(best.index, delta, 16)});
            }
            if ((lane & 15) == 0) winners[local_batch * kWaves + wave] = best;
        }
    }
    __syncthreads();
    if (tid < 16 && batch_begin + tid < batch) {
        if constexpr (PackedArgmax) {
            uint64_t key = 0;
#pragma unroll
            for (int w = 0; w < kWaves; ++w) key = key_max(key, winners[tid * kWaves + w]);
            partials[(batch_begin + tid) * parts + blockIdx.x] = argmax_pair_from_key(key);
        } else {
            ArgMaxPair best = {-INFINITY, INT_MAX};
#pragma unroll
            for (int w = 0; w < kWaves; ++w) best = better(best, winners[tid * kWaves + w]);
            partials[(batch_begin + tid) * parts + blockIdx.x] = best;
        }
    }
#endif
}

__global__ void dsv41_markov_compact_kernel(
    const ArgMaxPair* input, ArgMaxPair* output, int parts, int out_parts) {
    const int lane = threadIdx.x;
    const int p = blockIdx.x * 32 + lane;
    ArgMaxPair best = {-INFINITY, INT_MAX};
    if (lane < 32 && p < parts) best = input[blockIdx.y * parts + p];
#pragma unroll
    for (int delta = 32; delta > 0; delta /= 2) {
        best = better(best, {shuffle_xor(best.value, delta, 64),
                             shuffle_xor(best.index, delta, 64)});
    }
    if (lane == 0) output[blockIdx.y * out_parts + blockIdx.x] = best;
}

// TP1 avoids rank/row division in the packed scan. Both paths retain the
// public score/ID-bit input format and write the next token on this stream.
template <bool PackedArgmax, bool RankLocal = false>
__global__ void dsv41_markov_merge_kernel(
    const ArgMaxPair* input, int64_t* output,
    int rows, int parts, int world, int64_t output_stride) {
    __shared__ std::conditional_t<PackedArgmax, uint64_t, ArgMaxPair> winners[4];
    const int tid = threadIdx.x, lane = tid & 63;
    if constexpr (PackedArgmax) {
        uint64_t best = 0;
        for (int p = tid; p < parts * world; p += 256) {
            int offset;
            if constexpr (RankLocal) {
                offset = blockIdx.x * parts + p;
            } else {
                offset = ((p / parts) * rows + blockIdx.x) * parts + p % parts;
            }
            ArgMaxPair other = input[offset];
            other.index &= INT_MAX;
            if (other.index != INT_MAX) {
                best = key_max(best, ordered_argmax_key(other.value, other.index));
            }
        }
#pragma unroll
        for (int delta = 32; delta > 0; delta /= 2) {
            best = key_max(best, shuffle_key_xor(best, delta, 64));
        }
        if (lane == 0) winners[tid / 64] = best;
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int w = 0; w < 4; ++w) best = key_max(best, winners[w]);
            output[blockIdx.x * output_stride] =
                best == 0 ? INT_MAX : INT_MAX - static_cast<int>(static_cast<uint32_t>(best));
        }
    } else {
        ArgMaxPair best = {-INFINITY, INT_MAX};
        for (int p = tid; p < parts * world; p += 256) {
            ArgMaxPair other = input[((p / parts) * rows + blockIdx.x) * parts + p % parts];
            // Some transports use the float sign bit as a Lamport sentinel for 0.
            other.index &= INT_MAX;
            if (other.index != INT_MAX) best = better(best, other);
        }
#pragma unroll
        for (int delta = 32; delta > 0; delta /= 2) {
            best = better(best, {shuffle_xor(best.value, delta, 64),
                                 shuffle_xor(best.index, delta, 64)});
        }
        if (lane == 0) winners[tid / 64] = best;
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int w = 0; w < 4; ++w) best = better(best, winners[w]);
            output[blockIdx.x * output_stride] = best.index;
        }
    }
}
#endif

void check_markov_tensor(const torch::Tensor& t, const torch::Tensor& reference) {
    TORCH_CHECK(t.is_cuda() && t.device() == reference.device(),
                "all Markov tensors must be on the same GPU");
    TORCH_CHECK(!t.requires_grad(), "Markov sampling is inference-only");
}

void check_markov_pairs(const torch::Tensor& pairs) {
    TORCH_CHECK(pairs.is_cuda() && pairs.scalar_type() == torch::kFloat32 &&
                pairs.is_contiguous() && pairs.numel() > 0 && pairs.numel() <= INT_MAX,
                "partials must be nonempty contiguous GPU FP32 score/ID-bit pairs");
}

#ifdef USE_ROCM
template <typename LogitType, typename IndexType, int TileRows, bool PackedArgmax, int KStage>
void launch_dsv41_partial_tile(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int start, int vocab, torch::Tensor& out) {
    dsv41_markov_fp8_partial_kernel<LogitType, IndexType, TileRows, PackedArgmax, KStage>
        <<<dim3(out.size(1), (base.size(0) + 15) / 16), (TileRows / 16) * 64, 0,
           at::cuda::getCurrentCUDAStream()>>>(
        base.data_ptr<LogitType>(), prev.data_ptr<IndexType>(),
        static_cast<const uint8_t*>(embedding.data_ptr()), xs.data_ptr<float>(),
        static_cast<const uint8_t*>(weight.data_ptr()), ws.data_ptr<float>(),
        reinterpret_cast<ArgMaxPair*>(out.data_ptr<float>()),
        base.size(0), weight.size(0), vocab, start, out.size(1),
        base.stride(0), prev.stride(0), xs.stride(0), ws.stride(0));
}
template <typename LogitType, typename IndexType, bool PackedArgmax, int KStage>
void launch_dsv41_partial(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int start, int vocab, torch::Tensor& out, int tile_rows) {
    if (tile_rows == 64) {
        launch_dsv41_partial_tile<LogitType, IndexType, 64, PackedArgmax, KStage>(
            base, prev, embedding, xs, weight, ws, start, vocab, out);
    } else {
        launch_dsv41_partial_tile<LogitType, IndexType, 128, PackedArgmax, KStage>(
            base, prev, embedding, xs, weight, ws, start, vocab, out);
    }
}
template <typename LogitType, typename IndexType, int KStage>
void dispatch_dsv41_partial(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int start, int vocab, torch::Tensor& out, int tile_rows, bool packed_argmax) {
    if (packed_argmax) {
        launch_dsv41_partial<LogitType, IndexType, true, KStage>(
            base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows);
    } else {
        launch_dsv41_partial<LogitType, IndexType, false, KStage>(
            base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows);
    }
}
template <typename LogitType, typename IndexType>
void dispatch_dsv41_stage(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int start, int vocab, torch::Tensor& out, int tile_rows, bool packed_argmax, int k_stage) {
    if (k_stage == 64) {
        dispatch_dsv41_partial<LogitType, IndexType, 64>(
            base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax);
    } else {
        dispatch_dsv41_partial<LogitType, IndexType, 256>(
            base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax);
    }
}
#endif
}  // namespace

// Preserve the previously exported ten-argument C++ symbol for older callers.
void dsv41_markov_fp8_partials(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int64_t start, int64_t vocab, torch::Tensor out, int64_t tile_rows) {
    dsv41_markov_fp8_partials(
        base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, false);
}

void dsv41_markov_fp8_partials(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int64_t start, int64_t vocab, torch::Tensor out, int64_t tile_rows,
    bool packed_argmax) {
    dsv41_markov_fp8_partials(
        base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax, 256);
}

void dsv41_markov_fp8_partials(const torch::Tensor& base, const torch::Tensor& prev,
    const torch::Tensor& embedding, const torch::Tensor& xs,
    const torch::Tensor& weight, const torch::Tensor& ws,
    int64_t start, int64_t vocab, torch::Tensor out, int64_t tile_rows,
    bool packed_argmax, int64_t k_stage) {
    for (const auto& t : {base, prev, embedding, xs, weight, ws, out})
        check_markov_tensor(t, base);
    TORCH_CHECK(base.dim() == 2 && base.size(0) >= 1 && base.size(0) <= 32 &&
                base.size(1) > 0 && base.size(1) <= INT_MAX - 127 &&
                base.stride(1) == 1 && base.stride(0) >= base.size(1) &&
                (base.scalar_type() == torch::kBFloat16 || base.scalar_type() == torch::kFloat32),
                "base must be BF16/FP32 [1..32, padded vocab] with contiguous rows");
    TORCH_CHECK(prev.dim() == 1 && prev.size(0) == base.size(0) && prev.stride(0) > 0 &&
                (prev.scalar_type() == torch::kInt32 || prev.scalar_type() == torch::kInt64),
                "prev must be an int32/int64 batch vector");
    TORCH_CHECK(vocab > 0 && vocab < INT_MAX && start >= 0 && start <= vocab,
                "invalid global vocabulary bounds");
    TORCH_CHECK(embedding.dim() == 2 && weight.dim() == 2 &&
                embedding.size(1) == 256 && weight.size(1) == 256 &&
                embedding.size(0) >= vocab && weight.size(0) <= base.size(1) &&
                weight.size(0) <= vocab - start && embedding.is_contiguous() &&
                weight.is_contiguous() && embedding.scalar_type() == torch::kFloat8_e4m3fn &&
                weight.scalar_type() == torch::kFloat8_e4m3fn,
                "embedding and weight must be contiguous E4M3FN rank-256 matrices");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(embedding.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
                "FP8 matrices must be 16-byte aligned");
    TORCH_CHECK(xs.dim() == 2 && ws.dim() == 2 && xs.size(0) == embedding.size(0) &&
                ws.size(0) == weight.size(0) && xs.size(1) == 1 && ws.size(1) == 1 &&
                xs.stride(0) > 0 && ws.stride(0) > 0 &&
                xs.scalar_type() == torch::kFloat32 && ws.scalar_type() == torch::kFloat32,
                "scales must be FP32 [operand rows, 1] with positive row strides");
    check_markov_pairs(out);
    TORCH_CHECK(tile_rows == 64 || tile_rows == 128, "tile_rows must be 64 or 128");
    TORCH_CHECK(k_stage == 64 || k_stage == 256, "k_stage must be 64 or 256");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == base.size(0) &&
                out.size(1) == (base.size(1) + tile_rows - 1) / tile_rows && out.size(2) == 2,
                "out must be [batch, ceil(padded vocab / tile_rows), 2]");
#ifdef USE_ROCM
    const c10::cuda::CUDAGuard guard(base.device());
    const std::string architecture = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    TORCH_CHECK(architecture.rfind("gfx938", 0) == 0, "native FP8 Markov requires gfx938");
    if (base.scalar_type() == torch::kFloat32) {
        if (prev.scalar_type() == torch::kInt32)
            dispatch_dsv41_stage<float, int32_t>(base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax, k_stage);
        else
            dispatch_dsv41_stage<float, int64_t>(base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax, k_stage);
    } else {
        if (prev.scalar_type() == torch::kInt32)
            dispatch_dsv41_stage<BFloat16, int32_t>(base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax, k_stage);
        else
            dispatch_dsv41_stage<BFloat16, int64_t>(base, prev, embedding, xs, weight, ws, start, vocab, out, tile_rows, packed_argmax, k_stage);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "native FP8 Markov requires an HCU build");
#endif
}

void dsv41_markov_compact_partials(const torch::Tensor& input, torch::Tensor out) {
    check_markov_tensor(input, input);
    check_markov_tensor(out, input);
    check_markov_pairs(input);
    check_markov_pairs(out);
    TORCH_CHECK(input.dim() == 3 && input.size(0) >= 1 && input.size(0) <= 32 && input.size(2) == 2 &&
                out.dim() == 3 && out.size(0) == input.size(0) &&
                out.size(1) == (input.size(1) + 31) / 32 && out.size(2) == 2,
                "compact out must contain one pair per 32 input pairs");
#ifdef USE_ROCM
    const c10::cuda::CUDAGuard guard(input.device());
    dsv41_markov_compact_kernel<<<dim3(out.size(1), out.size(0)), 64, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const ArgMaxPair*>(input.data_ptr<float>()),
        reinterpret_cast<ArgMaxPair*>(out.data_ptr<float>()), input.size(1), out.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "native Markov merge requires an HCU build");
#endif
}

// Keep the original C++ symbol callable by previously compiled clients.
void dsv41_markov_merge_partials(const torch::Tensor& input,
    int64_t rows, int64_t world, torch::Tensor out) {
    dsv41_markov_merge_partials(input, rows, world, out, false);
}

void dsv41_markov_merge_partials(const torch::Tensor& input,
    int64_t rows, int64_t world, torch::Tensor out, bool packed_argmax) {
    check_markov_tensor(input, input);
    check_markov_tensor(out, input);
    check_markov_pairs(input);
    TORCH_CHECK(rows > 0 && rows <= 32 && world > 0 && world <= INT_MAX / (rows * 2) &&
                input.numel() % (world * rows * 2) == 0,
                "invalid rank-major candidate geometry");
    TORCH_CHECK(out.dim() == 1 && out.size(0) == rows && out.stride(0) > 0 &&
                out.scalar_type() == torch::kInt64, "out must be a strided int64 batch vector");
#ifdef USE_ROCM
    const c10::cuda::CUDAGuard guard(input.device());
    const auto* pairs = reinterpret_cast<const ArgMaxPair*>(input.data_ptr<float>());
    auto* tokens = out.data_ptr<int64_t>();
    const int parts = input.numel() / (world * rows * 2);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (packed_argmax && world == 1) {
        dsv41_markov_merge_kernel<true, true><<<rows, 256, 0, stream>>>(
            pairs, tokens, rows, parts, world, out.stride(0));
    } else if (packed_argmax) {
        dsv41_markov_merge_kernel<true, false><<<rows, 256, 0, stream>>>(
            pairs, tokens, rows, parts, world, out.stride(0));
    } else {
        dsv41_markov_merge_kernel<false, false><<<rows, 256, 0, stream>>>(
            pairs, tokens, rows, parts, world, out.stride(0));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "native Markov merge requires an HCU build");
#endif
}

}  // namespace lightop::sampling
