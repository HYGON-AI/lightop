// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <algorithm>

namespace at {
namespace native {
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...) \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

// =====================================
// fake_quant_per_tensor_affine
// =====================================

// 1、向量化
template <typename scalar_t, int VEC>
__global__ void fake_quantize_per_tensor_vec_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t total_vec,
    float scale,
    float inv_scale,
    float zero_point,
    float quant_min,
    float quant_max){

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    const VecType* in_vec_ptr = reinterpret_cast<const VecType*>(input);
    VecType* out_vec_ptr = reinterpret_cast<VecType*>(output);

    for (int64_t i = idx; i < total_vec; i += stride) {
        VecType in_vec = in_vec_ptr[i];
        VecType out_vec;

        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            float val = static_cast<float>(in_vec.val[j]);
            float q = std::nearbyint(val * inv_scale) + zero_point;
            q = fminf(fmaxf(q, quant_min), quant_max);
            out_vec.val[j] = static_cast<scalar_t>((q - zero_point) * scale);
        }

        out_vec_ptr[i] = out_vec;
    }
}

// 2、标量化
template <typename scalar_t>
__global__ void fake_quantize_per_tensor_scalar_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t total,
    int64_t scale,
    float inv_scale,
    float zero_point,
    float quant_min,
    float quant_max) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        float val = static_cast<float>(input[i]);
        float q = std::nearbyint(val * inv_scale) + zero_point;
        q = fminf(fmaxf(q, quant_min), quant_max);
        output[i] = static_cast<scalar_t>((q - zero_point) * scale);
    }
}

// =====================================
// fake_quant_per_channl_affine
// =====================================

// 1、向量化
template <typename scalar_t, int VEC>
__global__ void fake_quantize_per_channel_vec_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale,
    const int32_t* __restrict__ zero_point,
    scalar_t* __restrict__ output,
    int64_t Outer, int64_t Channel, int64_t Inner,
    float quant_min, float quant_max) {
    
    // 一个thread处理VEC个元素
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total_vec = (Outer * Channel * Inner) / VEC;
    for (int64_t i = idx; i < total_vec; i += stride) {
        // 计算当前向量块所属的channel
        // Inner是VEC的倍数，所以一个向量块必定属于同一个channel

        int64_t inner_vec_count = Inner / VEC;
        int64_t c = (i / inner_vec_count) % Channel;

        float s = scale[c];
        float inv_s = 1.0f / s;
        float zp = static_cast<float>(zero_point[c]);

        using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
        VecType in_vec = *(reinterpret_cast<const VecType*>(input) + i);
        VecType out_vec;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float val = static_cast<float>(in_vec.val[j]);
            float q = std::nearbyint(val * inv_s) + zp;
            q = fminf(fmaxf(q, quant_min), quant_max);
            out_vec.val[j] = static_cast<scalar_t>((q - zp) * s);
        }

        *(reinterpret_cast<VecType*>(output) + i) = out_vec;
    }
}

// 2、标量化
template <typename scalar_t>
__global__ void fake_quantize_per_channel_scalar_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale,
    const int32_t* __restrict__ zero_point,
    scalar_t* __restrict__ output,
    int64_t Outer, int64_t Channel, int64_t Inner,
    float quant_min, float quant_max) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = Outer * Channel * Inner;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t c = (i / Inner) % Channel;
        float s = scale[c];
        float inv_s = 1.0f / s;
        float zp = static_cast<float>(zero_point[c]);

        float val = static_cast<float>(input[i]);
        float q = std::nearbyint(val * inv_s) + zp;
        q = fminf(fmaxf(q, quant_min), quant_max);
        output[i] = static_cast<scalar_t>((q - zp) * s);
    }
}

// tensor api
torch::Tensor fake_quantize_per_tensor_affine_custom(
    const torch::Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int64_t total_elements = input_contig.numel();
    if (total_elements == 0) return output;

    float s = static_cast<float>(scale);
    float inv_s  = 1.0f / s;
    float zp = static_cast<float>(zero_point);
    float qmin = static_cast<float>(quant_min);
    float qmax = static_cast<float>(quant_max);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input_contig.data_ptr());
    auto out_ptr = reinterpret_cast<std::uintptr_t>(output.data_ptr());

    // 检查内存是否16字节对齐
    bool ptrs_aligned = (inp_ptr % 16 == 0) && (out_ptr % 16 == 0);

    VLLM_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "fake_quantize_per_tensor_affine",
        [&] {
            constexpr int VEC = 16 / sizeof(scalar_t);
            int threads = 256;
            
            if (ptrs_aligned) {
                int64_t vec_total_elements = (total_elements / VEC) * VEC; // 能被VEC整除的长度
                int64_t tail_elements = total_elements % VEC;
                if (vec_total_elements > 0) {
                    int64_t total_vec = vec_total_elements / VEC;
                    int blocks = std::min(static_cast<int64_t>(65535), (total_vec + threads - 1) / threads);
                    fake_quantize_per_tensor_vec_kernel<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                        input_contig.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        total_vec,
                        s, inv_s, zp, qmin, qmax
                    );
                }
                if (tail_elements > 0) {
                    const scalar_t* input_tail = input_contig.data_ptr<scalar_t>() + vec_total_elements;
                    scalar_t* output_tail = output.data_ptr<scalar_t>() + vec_total_elements;

                    fake_quantize_per_tensor_scalar_kernel<scalar_t><<<1, 32, 0, stream>>>(
                        input_tail,
                        output_tail,
                        tail_elements,
                        s, inv_s, zp, qmin, qmax
                    );
                }
            } else {
                int blocks = std::min(static_cast<int64_t>(65535), (total_elements + threads - 1) / threads);
                fake_quantize_per_tensor_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    input_contig.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    total_elements,
                    s, inv_s, zp, qmin, qmax
                );
            }
        }
    );
    
    return output;
}

// channel api
torch::Tensor fake_quantize_per_channel_affine_custom(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
    
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous for optimal performance");

    if (axis < 0) axis += input.dim();
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Invalid axis");

    int64_t Channel = input.size(axis);
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == Channel, "scale size must match channel size");
    TORCH_CHECK(zero_point.dim() == 1 && zero_point.size(0) == Channel, "zero_point size must mtch channel size");

    auto scale_contig = scale.contiguous().to(torch::kFloat32).to(input.device());
    auto zp_contig = zero_point.contiguous().to(torch::kInt32).to(input.device());

    // [Outer, Channel, Inner]
    int64_t Outer = 1;
    int64_t Inner = 1;
    for (int i = 0; i < axis; i++) {
        Outer *= input.size(i);
    }
    for (int i = axis + 1; i < input.dim(); i++){
        Inner *= input.size(i);
    }
    auto output = torch::empty_like(input);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    auto out_ptr = reinterpret_cast<std::uintptr_t>(output.data_ptr());
    
    bool ptrs_aligned = (inp_ptr % 16 == 0) && (out_ptr % 16 == 0);
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "fake_quantize_per_channel_affine",
        [&] {
            constexpr int VEC = 16 / sizeof(scalar_t);
            int threads = 256;
            if (ptrs_aligned && (Inner % VEC == 0)) {
                int64_t total_vec = (Outer * Channel * Inner) / VEC;
                int blocks = std::min(static_cast<int64_t>(65535), (total_vec + threads - 1) / threads);

                fake_quantize_per_channel_vec_kernel<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    scale_contig.data_ptr<float>(),
                    zp_contig.data_ptr<int32_t>(),
                    output.data_ptr<scalar_t>(),
                    Outer, Channel, Inner,
                    static_cast<float>(quant_min), static_cast<float>(quant_max)
                );
            } else {
                int64_t total = Outer * Channel * Inner;
                int blocks = std::min(static_cast<int64_t>(65535), (total + threads - 1) / threads);

                fake_quantize_per_channel_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    scale_contig.data_ptr<float>(),
                    zp_contig.data_ptr<int32_t>(),
                    output.data_ptr<scalar_t>(),
                    Outer, Channel, Inner,
                    static_cast<float>(quant_min), static_cast<float>(quant_max)
                );
            }
        }
    );
    return output;
}

} // namespace native
} // namespace at