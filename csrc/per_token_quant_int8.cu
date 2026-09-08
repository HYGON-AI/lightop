#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <limits>

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {
namespace {

__device__ __forceinline__ int8_t per_token_quant_int8_rn(float value) {
#ifdef USE_ROCM
    constexpr float int8_min =
        static_cast<float>(std::numeric_limits<int8_t>::min());
    constexpr float int8_max =
        static_cast<float>(std::numeric_limits<int8_t>::max());
    float rounded = std::nearbyint(value);
    rounded = fminf(fmaxf(rounded, int8_min), int8_max);
    return static_cast<int8_t>(rounded);
#else
    uint32_t result;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
    return reinterpret_cast<const int8_t&>(result);
#endif
}

template <int ROWS_PER_BLOCK>
__global__ void per_token_quant_int8_bf16_k128_kernel(
    const c10::BFloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int64_t num_tokens) {
    constexpr int hidden_size = 128;
    constexpr int subgroup_size = 16;
    constexpr int rows_per_wave = C10_WARP_SIZE / subgroup_size;
    constexpr int values_per_lane = hidden_size / subgroup_size;
    static_assert(C10_WARP_SIZE % subgroup_size == 0,
                  "warp size must be divisible by the 16-lane subgroup");
    static_assert(values_per_lane == 8, "the fast path uses 16-lane rows");
    static_assert(ROWS_PER_BLOCK * subgroup_size == 512,
                  "the fast path uses 512 threads");

    using InputVec =
        at::native::memory::aligned_vector<c10::BFloat16, values_per_lane>;
    using OutputVec =
        at::native::memory::aligned_vector<int8_t, values_per_lane>;

    const int thread_id = threadIdx.x;
    const int wave_lane = thread_id % C10_WARP_SIZE;
    const int wave_id = thread_id / C10_WARP_SIZE;
    const int subgroup_id = wave_lane / subgroup_size;
    const int lane_id = wave_lane % subgroup_size;
    const int64_t token_id =
        static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
        wave_id * rows_per_wave + subgroup_id;
    if (token_id >= num_tokens) {
        return;
    }

    const int column = lane_id * values_per_lane;
    const int64_t offset = token_id * hidden_size + column;
    const InputVec input_vec =
        *reinterpret_cast<const InputVec*>(input + offset);
    float values[values_per_lane];
    float row_absmax = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_lane; ++index) {
        values[index] = static_cast<float>(input_vec.val[index]);
        row_absmax = fmaxf(row_absmax, fabsf(values[index]));
    }

#pragma unroll
    for (int delta = subgroup_size / 2; delta > 0; delta >>= 1) {
        row_absmax = fmaxf(
            row_absmax, __shfl_down(row_absmax, delta, subgroup_size));
    }
    const float token_absmax =
        fmaxf(__shfl(row_absmax, 0, subgroup_size), 1.0e-10f);
    if (lane_id == 0) {
        scale_out[token_id] = token_absmax / 127.0f;
    }
    const float inverse_scale = 127.0f / token_absmax;
    OutputVec output_vec;
#pragma unroll
    for (int index = 0; index < values_per_lane; ++index) {
        output_vec.val[index] =
            per_token_quant_int8_rn(values[index] * inverse_scale);
    }
    *reinterpret_cast<OutputVec*>(output + offset) = output_vec;
}

}  // namespace

void per_token_quant_int8_out(
    torch::Tensor& out,
    const torch::Tensor& input,
    torch::Tensor& scales) {
    TORCH_CHECK(input.is_cuda(), "input must be on a GPU");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) == 128,
                "input must have last dimension 128");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
                "input must have dtype torch.bfloat16");
    TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Char,
                "out must have dtype torch.int8");
    const int64_t num_tokens = input.numel() / 128;
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float,
                "scales must have dtype torch.float32");
    TORCH_CHECK(scales.numel() == num_tokens,
                "scales must contain one value per token");
    TORCH_CHECK(input.device() == out.device() &&
                    input.device() == scales.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0,
                "input must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0,
                "out must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(scales.data_ptr()) % 4 == 0,
                "scales must be 4-byte aligned");
    if (num_tokens == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 512;
    constexpr int rows_per_block = threads / 16;
    const int64_t blocks =
        (num_tokens + rows_per_block - 1) / rows_per_block;
    per_token_quant_int8_bf16_k128_kernel<rows_per_block>
        <<<blocks, threads, 0, stream>>>(
            input.data_ptr<c10::BFloat16>(), out.data_ptr<int8_t>(),
            scales.data_ptr<float>(), num_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
