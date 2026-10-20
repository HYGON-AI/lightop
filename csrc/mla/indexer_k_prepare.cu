// DSA Indexer K preparation and page-planar INT8 cache writes.
// One wave handles one 128-element K vector; Q is never read or written.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {
using BFloat16 = at::BFloat16;
constexpr int kWidth = 128;
constexpr int kWave = 64;
constexpr int kThreads = 64;
constexpr int kPageSize = 64;
constexpr int kPageBytes = kPageSize * (kWidth + sizeof(float));

__device__ __forceinline__ float bf16(float value) {
    return static_cast<float>(static_cast<BFloat16>(value));
}

__device__ __forceinline__ double wave_sum(double value) {
#pragma unroll
    for (int distance = 32; distance > 0; distance /= 2) {
        value += __shfl_xor(value, distance, kWave);
    }
    return value;
}

__device__ __forceinline__ int8_t quantize(float value) {
    // Index-K cache uses round-away-from-zero, unlike Indexer Q's nearbyint.
    const float rounded = value >= 0.0f ? floorf(value + 0.5f) : ceilf(value - 0.5f);
    return static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, rounded)));
}

template <typename Position, typename Location>
__global__ __launch_bounds__(kThreads) void indexer_k_norm_rope_hadamard_store_kernel(
    const BFloat16 *__restrict__ key, const float *__restrict__ weight,
    const float *__restrict__ bias, const Position *__restrict__ positions,
    const BFloat16 *__restrict__ cos_sin, const Location *__restrict__ locations,
    uint8_t *remote_cache, uint8_t *owner_cache, int num_tokens, int key_stride, int trig_stride,
    int64_t num_pages, bool neox, double norm_epsilon, float hadamard_scale, float quant_epsilon) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= num_tokens) {
        return;
    }
    const int64_t location = static_cast<int64_t>(locations[token]);
    // Padding must not write either cache or dereference a padded position.
    if (location < 0 || location >= num_pages * kPageSize) {
        return;
    }
    const BFloat16 *input = key + token * key_stride;
    // BF16 input sums are exact in FP64. Higher precision statistics avoid
    // FP32 reduction-order ties crossing the following BF16 rounding boundary.
    const double input_low = static_cast<float>(input[lane]);
    const double input_high = static_cast<float>(input[lane + kWave]);
    const double mean = wave_sum(input_low + input_high) / kWidth;
    const double centered_low = input_low - mean;
    const double centered_high = input_high - mean;
    const double variance =
        wave_sum(centered_low * centered_low + centered_high * centered_high) / kWidth;
    const double inverse_std = 1.0 / sqrt(variance + norm_epsilon);
    float low = bf16(static_cast<float>((centered_low * inverse_std) * weight[lane] +
                                        (bias ? static_cast<double>(bias[lane]) : 0.0)));
    float high = bf16(static_cast<float>((centered_high * inverse_std) * weight[lane + kWave] +
                                         (bias ? static_cast<double>(bias[lane + kWave]) : 0.0)));
    const BFloat16 *trig = cos_sin + static_cast<int64_t>(positions[token]) * trig_stride;
    const int partner = neox ? lane ^ 32 : lane ^ 1;
    const int frequency = neox ? lane % 32 : lane / 2;
    const bool negate = neox ? lane < 32 : lane % 2 == 0;
    const float other = __shfl(low, partner, kWave);
    const float cosine = static_cast<float>(trig[frequency]);
    const float sine = static_cast<float>(trig[frequency + 32]);
    low = bf16(bf16(low * cosine) + bf16((negate ? -other : other) * sine));
#pragma unroll
    for (int distance = 1; distance < 64; distance *= 2) {
        const float other_low = __shfl_xor(low, distance, kWave);
        const float other_high = __shfl_xor(high, distance, kWave);
        low = (lane & distance) ? other_low - low : low + other_low;
        high = (lane & distance) ? other_high - high : high + other_high;
    }
    const float sum = low + high;
    const float difference = low - high;
    low = bf16(bf16(sum) * hadamard_scale);
    high = bf16(bf16(difference) * hadamard_scale);
    float maximum = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
    for (int distance = 32; distance > 0; distance /= 2) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, distance, kWave));
    }
    const float scale = fmaxf(maximum, quant_epsilon) / 127.0f;
    const int8_t low_q = quantize(low / scale);
    const int8_t high_q = quantize(high / scale);
    const int64_t page = location / kPageSize;
    const int offset = location % kPageSize;
    uint8_t *remote = remote_cache + page * kPageBytes;
    reinterpret_cast<int8_t *>(remote)[offset * kWidth + lane] = low_q;
    reinterpret_cast<int8_t *>(remote)[offset * kWidth + lane + kWave] = high_q;
    if (lane == 0) {
        reinterpret_cast<float *>(remote + kPageSize * kWidth)[offset] = scale;
    }
    if (owner_cache != remote_cache) {
        uint8_t *owner = owner_cache + page * kPageBytes;
        reinterpret_cast<int8_t *>(owner)[offset * kWidth + lane] = low_q;
        reinterpret_cast<int8_t *>(owner)[offset * kWidth + lane + kWave] = high_q;
        if (lane == 0) {
            reinterpret_cast<float *>(owner + kPageSize * kWidth)[offset] = scale;
        }
    }
}
} // namespace

namespace at::native {
void indexer_k_norm_rope_hadamard_store(const torch::Tensor &key, const torch::Tensor &weight,
                                        const c10::optional<torch::Tensor> &bias,
                                        const torch::Tensor &positions,
                                        const torch::Tensor &cos_sin,
                                        const torch::Tensor &locations, torch::Tensor remote_cache,
                                        torch::Tensor owner_cache, bool neox, double norm_epsilon,
                                        double hadamard_scale, double quant_epsilon) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == at::kBFloat16 && key.dim() == 3 &&
                    key.size(0) <= 20 && key.size(1) == 1 && key.size(2) == kWidth &&
                    key.stride(2) == 1,
                "Indexer K must be BF16 [M<=20,1,128] on GPU");
    const auto device = key.device();
    const int tokens = key.size(0);
    const auto parameter_ok = [&](const torch::Tensor &tensor) {
        return tensor.device() == device && tensor.scalar_type() == at::kFloat &&
               tensor.dim() == 1 && tensor.numel() == kWidth && tensor.is_contiguous();
    };
    TORCH_CHECK(parameter_ok(weight) && (!bias.has_value() || parameter_ok(*bias)),
                "LayerNorm weight and optional bias must be contiguous FP32 [128]");
    const auto index_ok = [&](const torch::Tensor &tensor) {
        return tensor.device() == device && tensor.dim() == 1 && tensor.size(0) == tokens &&
               tensor.is_contiguous() &&
               (tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong);
    };
    TORCH_CHECK(index_ok(positions) && index_ok(locations),
                "positions and locations must be contiguous INT32/INT64 [M]");
    TORCH_CHECK(cos_sin.device() == device && cos_sin.scalar_type() == at::kBFloat16 &&
                    cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.stride(1) == 1,
                "cos_sin must be BF16 [positions,64]");
    const auto cache_ok = [&](const torch::Tensor &tensor) {
        return tensor.device() == device && tensor.scalar_type() == at::kByte &&
               tensor.dim() == 2 && tensor.size(1) == kPageBytes && tensor.is_contiguous();
    };
    TORCH_CHECK(cache_ok(remote_cache) && cache_ok(owner_cache) &&
                    remote_cache.sizes() == owner_cache.sizes(),
                "cache buffers must have matching contiguous UINT8 [pages,8448] layouts");
    TORCH_CHECK(std::isfinite(norm_epsilon) && norm_epsilon > 0.0 &&
                    std::isfinite(hadamard_scale) && std::isfinite(quant_epsilon) &&
                    quant_epsilon > 0.0,
                "normalization and quantization epsilons must be positive and finite");
    const c10::cuda::CUDAGuard guard(device);
    if (tokens == 0) {
        return;
    }
#define LAUNCH(POSITION, LOCATION)                                                                 \
    indexer_k_norm_rope_hadamard_store_kernel<POSITION, LOCATION>                                  \
        <<<tokens, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(                               \
            key.data_ptr<BFloat16>(), weight.data_ptr<float>(),                                    \
            bias.has_value() ? bias->data_ptr<float>() : nullptr, positions.data_ptr<POSITION>(),  \
            cos_sin.data_ptr<BFloat16>(), locations.data_ptr<LOCATION>(),                          \
            remote_cache.data_ptr<uint8_t>(), owner_cache.data_ptr<uint8_t>(), tokens,             \
            key.stride(0), cos_sin.stride(0), remote_cache.size(0), neox,                          \
            static_cast<float>(norm_epsilon), static_cast<float>(hadamard_scale),                  \
            static_cast<float>(quant_epsilon))
    if (positions.scalar_type() == at::kLong) {
        if (locations.scalar_type() == at::kLong) {
            LAUNCH(int64_t, int64_t);
        } else {
            LAUNCH(int64_t, int32_t);
        }
    } else if (locations.scalar_type() == at::kLong) {
        LAUNCH(int32_t, int64_t);
    } else {
        LAUNCH(int32_t, int32_t);
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace at::native
