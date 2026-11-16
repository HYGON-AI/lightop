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

}  // namespace lightop::sampling
