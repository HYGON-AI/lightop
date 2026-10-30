// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// DSA Indexer Q-only preparation: BF16 RoPE, 128-point Hadamard,
// BF16 normalization, per-head INT8 quantization and FP32 gate scaling.
// One wave owns one Q head. K and its cache are never read or written.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <tuple>

namespace {
using BFloat16 = at::BFloat16;
constexpr int kHeads = 32;
constexpr int kWidth = 128;
constexpr int kWave = 64;
constexpr int kThreads = 256;

__device__ __forceinline__ float round_bf16(float value) {
    return static_cast<float>(static_cast<BFloat16>(value));
}

__device__ __forceinline__ int8_t quantize_int8(float value) {
    float rounded = nearbyintf(value);
    rounded = fminf(127.0f, fmaxf(-128.0f, rounded));
    return static_cast<int8_t>(rounded);
}

template <typename Position>
__global__ __launch_bounds__(kThreads) void indexer_q_rope_hadamard_quant_kernel(
    const BFloat16 *__restrict__ query, const Position *__restrict__ positions,
    const BFloat16 *__restrict__ cos_sin, const float *__restrict__ raw_gate,
    int8_t *__restrict__ output, float *__restrict__ adjusted_gate, int num_tokens,
    int query_stride0, int query_stride1, int cache_stride, bool neox, float hadamard_scale,
    float head_scale, float softmax_scale) {
    const int lane = threadIdx.x % kWave;
    const int head_row = blockIdx.x * (kThreads / kWave) + threadIdx.x / kWave;
    if (head_row >= num_tokens * kHeads) {
        return;
    }
    const int token = head_row / kHeads;
    const int head = head_row % kHeads;
    const BFloat16 *input = query + token * query_stride0 + head * query_stride1;
    const BFloat16 *trig = cos_sin + static_cast<int64_t>(positions[token]) * cache_stride;
    const int partner = neox ? lane ^ 32 : lane ^ 1;
    const int frequency = neox ? lane % 32 : lane / 2;
    const bool negate = neox ? lane < 32 : (lane % 2 == 0);
    const float a = static_cast<float>(input[lane]);
    const float b = static_cast<float>(input[partner]);
    const float c = static_cast<float>(trig[frequency]);
    const float s = static_cast<float>(trig[frequency + 32]);
    // Native eager RoPE rounds both products and then the addition to BF16.
    float low = round_bf16(round_bf16(a * c) + round_bf16((negate ? -b : b) * s));
    float high = static_cast<float>(input[lane + 64]);
#pragma unroll
    for (int distance = 1; distance < 64; distance *= 2) {
        const float other_low = __shfl_xor(low, distance, kWave);
        const float other_high = __shfl_xor(high, distance, kWave);
        low = (lane & distance) ? other_low - low : low + other_low;
        high = (lane & distance) ? other_high - high : high + other_high;
    }
    const float sum = low + high;
    const float difference = low - high;
    // Match the legacy Hadamard GEMM's BF16 output, then the separate scale.
    low = round_bf16(round_bf16(sum) * hadamard_scale);
    high = round_bf16(round_bf16(difference) * hadamard_scale);
    float maximum = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
    for (int distance = 32; distance > 0; distance /= 2) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, distance, kWave));
    }
    // This is the original no-smooth, no-clamp INT8 contract, including zero.
    const float inverse_scale = maximum == 0.0f ? 0.0f : 127.0f / maximum;
    output[head_row * kWidth + lane] = quantize_int8(low * inverse_scale);
    output[head_row * kWidth + lane + 64] = quantize_int8(high * inverse_scale);
    if (lane == 0) {
        const float q_scale = maximum / 127.0f;
        // Preserve the three independent FP32 scalar multiplication stages.
        volatile float weight = raw_gate[head_row] * head_scale;
        volatile float scaled_weight = weight * softmax_scale;
        adjusted_gate[head_row] = scaled_weight * q_scale;
    }
}
} // namespace

namespace at::native {
std::tuple<torch::Tensor, torch::Tensor>
indexer_q_rope_hadamard_quant(const torch::Tensor &query, const torch::Tensor &positions,
                              const torch::Tensor &cos_sin, const torch::Tensor &raw_gate,
                              bool neox, double hadamard_scale, double head_scale,
                              double softmax_scale) {
    TORCH_CHECK(query.is_cuda() && positions.is_cuda() && cos_sin.is_cuda() && raw_gate.is_cuda(),
                "Indexer Q inputs must be on the GPU");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && cos_sin.scalar_type() == at::kBFloat16,
                "Indexer Q and cos/sin must be BF16");
    TORCH_CHECK(raw_gate.scalar_type() == at::kFloat, "raw_gate must be FP32");
    TORCH_CHECK(query.dim() == 3 && query.size(1) == kHeads && query.size(2) == kWidth &&
                    query.stride(2) == 1 && query.size(0) <= 36,
                "Indexer Q must have shape [M<=36,32,128] and contiguous head columns");
    TORCH_CHECK(positions.dim() == 1 && positions.is_contiguous() &&
                    positions.size(0) == query.size(0) &&
                    (positions.scalar_type() == at::kLong || positions.scalar_type() == at::kInt),
                "positions must be contiguous INT32/INT64 [M]");
    TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.stride(1) == 1,
                "cos_sin must have shape [positions,64]");
    TORCH_CHECK(raw_gate.dim() == 2 && raw_gate.size(0) == query.size(0) &&
                    raw_gate.size(1) == kHeads && raw_gate.is_contiguous(),
                "raw_gate must be contiguous FP32 [M,32]");
    TORCH_CHECK(query.device() == positions.device() && query.device() == cos_sin.device() &&
                    query.device() == raw_gate.device(),
                "Indexer Q inputs must share a device");
    TORCH_CHECK(std::isfinite(hadamard_scale) && std::isfinite(head_scale) &&
                    std::isfinite(softmax_scale),
                "Indexer Q scales must be finite");
    const c10::cuda::CUDAGuard guard(query.device());
    auto output = torch::empty(query.sizes(), query.options().dtype(at::kChar));
    auto weights = torch::empty(raw_gate.sizes(), raw_gate.options());
    const int tokens = query.size(0);
    if (tokens == 0) {
        return {output, weights};
    }
    const int blocks = (tokens * kHeads + kThreads / kWave - 1) / (kThreads / kWave);
#define LAUNCH(POSITION_TYPE)                                                                      \
    indexer_q_rope_hadamard_quant_kernel<POSITION_TYPE>                                            \
        <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(                               \
            query.data_ptr<BFloat16>(), positions.data_ptr<POSITION_TYPE>(),                       \
            cos_sin.data_ptr<BFloat16>(), raw_gate.data_ptr<float>(), output.data_ptr<int8_t>(),   \
            weights.data_ptr<float>(), tokens, query.stride(0), query.stride(1),                   \
            cos_sin.stride(0), neox, static_cast<float>(hadamard_scale),                           \
            static_cast<float>(head_scale), static_cast<float>(softmax_scale))
    if (positions.scalar_type() == at::kLong) {
        LAUNCH(int64_t);
    } else {
        LAUNCH(int32_t);
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, weights};
}
} // namespace at::native
