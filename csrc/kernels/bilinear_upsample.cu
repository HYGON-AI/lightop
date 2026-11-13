// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Bilinear upsample forward & backward CUDA kernels, aligned with:
//   nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
//
// Supported dtypes: float32, float16, bfloat16
//
// Coordinate mapping (align_corners=False):
//   src = (dst + 0.5) * (src_size / dst_size) - 0.5

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"

namespace at {
namespace native {

// Each thread writes one output pixel.
// Grid layout: (ceil(W_out/BLOCK_X), ceil(H_out/BLOCK_Y), N*C)
template <typename T>
__global__ void bilinear_upsample_kernel(
    const T* __restrict__ input,   // [N, C, H_in,  W_in ]
    T*       __restrict__ output,  // [N, C, H_out, W_out]
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;   // flattened N*C index

    if (w_out >= W_out || h_out >= H_out) return;

    
    // 计算得到原来的中心坐标  -0.5？
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    float src_x = (w_out + 0.5f) * scale_x - 0.5f;

    // 得到原来的起始点坐标 左上角
    int y0 = static_cast<int>(floorf(src_y));
    int x0 = static_cast<int>(floorf(src_x));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));
    x0 = std::max(0, std::min(x0, W_in - 1));
    x1 = std::max(0, std::min(x1, W_in - 1));

    // Fractional part used as interpolation weight toward the larger index
    float alpha = src_y - floorf(src_y);  // weight for y1
    float beta  = src_x - floorf(src_x);  // weight for x1

    const T* in_nc = input + (long)nc * H_in * W_in;
    float I00 = static_cast<float>(in_nc[y0 * W_in + x0]);
    float I01 = static_cast<float>(in_nc[y0 * W_in + x1]);
    float I10 = static_cast<float>(in_nc[y1 * W_in + x0]);
    float I11 = static_cast<float>(in_nc[y1 * W_in + x1]);

    float val = (1.0f - alpha) * (1.0f - beta) * I00
              + (1.0f - alpha) * beta  * I01
              + alpha  * (1.0f - beta) * I10
              + alpha  * beta  * I11;

    output[(long)nc * H_out * W_out + h_out * W_out + w_out] = static_cast<T>(val);
}

// ============================================================
//  Backward kernel
//  Grid: 与前向相同 (ceil(W_out/32), ceil(H_out/8), N*C)
//  Each thread reads one grad_output pixel and atomically
//  accumulates weighted gradients to the 4 corresponding
//  input pixel positions.
//
//  反向推导（链式法则）：
//    forward:  output = (1-α)(1-β)·I00 + (1-α)β·I01 + α(1-β)·I10 + αβ·I11
//    backward:
//      grad_input[y0,x0] += (1-α)(1-β) · grad_output
//      grad_input[y0,x1] += (1-α)  β   · grad_output
//      grad_input[y1,x0] +=   α (1-β)  · grad_output
//      grad_input[y1,x1] +=   α    β   · grad_output
//
//  grad_input_f32 为 float32 累加缓冲区：
//    - fp16/bf16 输入：中间缓冲区避免 ROCm 上 fp16/bf16 原子加的兼容问题
//    - fp32 输入：直接传入 grad_input 本身的指针，无需额外缓冲区
// ============================================================
// template <typename T, int vec>
// __global__ void bilinear_upsample_backward_kernel(
//     const T*   __restrict__ grad_output,    // [N, C, H_out, W_out]
//     float*     __restrict__ grad_input_f32, // [N, C, H_in,  W_in ] 已预置零
//     const int H_in,
//     const int W_in,
//     const int H_out,
//     const int W_out,
//     const float scale_y,   // H_in / H_out
//     const float scale_x    // W_in / W_out
// ) {
//     int w_out = blockIdx.x * blockDim.x + threadIdx.x;
//     int h_out = blockIdx.y * blockDim.y + threadIdx.y;
//     int nc    = blockIdx.z;

//     if (w_out >= W_out || h_out >= H_out) return;

//     // 与前向完全相同的坐标映射
//     float src_y = (h_out + 0.5f) * scale_y - 0.5f;
//     float src_x = (w_out + 0.5f) * scale_x - 0.5f;

//     int y0 = static_cast<int>(floorf(src_y));
//     int x0 = static_cast<int>(floorf(src_x));
//     int y1 = y0 + 1;
//     int x1 = x0 + 1;

//     y0 = std::max(0, std::min(y0, H_in - 1));
//     y1 = std::max(0, std::min(y1, H_in - 1));
//     x0 = std::max(0, std::min(x0, W_in - 1));
//     x1 = std::max(0, std::min(x1, W_in - 1));

//     float alpha = src_y - floorf(src_y);
//     float beta  = src_x - floorf(src_x);

//     // 读取 grad_output（转为 float 参与后续计算）
//     float g = static_cast<float>(
//         grad_output[(long)nc * H_out * W_out + h_out * W_out + w_out]);

//     // 向 4 个输入像素散射（scatter-add），使用 float32 atomicAdd 保证精度
//     float* gin = grad_input_f32 + nc * H_in * W_in;
//     // atomicAdd(&gin[y0 * W_in + x0], (1.0f - alpha) * (1.0f - beta) * g);
//     // atomicAdd(&gin[y0 * W_in + x1], (1.0f - alpha) *          beta  * g);
//     // atomicAdd(&gin[y1 * W_in + x0],          alpha  * (1.0f - beta) * g);
//     // atomicAdd(&gin[y1 * W_in + x1],          alpha  *          beta  * g);
//     #ifdef __gfx936__
//     __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x0), (1.0f - alpha) * (1.0f - beta) * g);
//     __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x1), (1.0f - alpha) *          beta  * g);
//     __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x0),          alpha  * (1.0f - beta) * g);
//     __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x1),          alpha  *          beta  * g);
//     #else
//     atomicAdd(&gin[y0 * W_in + x0], (1.0f - alpha) * (1.0f - beta) * g);
//     atomicAdd(&gin[y0 * W_in + x1], (1.0f - alpha) *          beta  * g);
//     atomicAdd(&gin[y1 * W_in + x0],          alpha  * (1.0f - beta) * g);
//     atomicAdd(&gin[y1 * W_in + x1],          alpha  *          beta  * g);
//     #endif

//     // gin[y0 * W_in + x0] += (1.0f - alpha) * (1.0f - beta) * g;
//     // gin[y0 * W_in + x1] += (1.0f - alpha) *          beta  * g;
//     // gin[y1 * W_in + x0] +=          alpha  * (1.0f - beta) * g;
//     // gin[y1 * W_in + x1] +=          alpha  *          beta  * g;
// }

template <typename T, int vec>
__global__ void bilinear_upsample_backward_kernel_fp32(
    const T*   __restrict__ grad_output,    // [N, C, H_out, W_out]
    float*     __restrict__ grad_input_f32, // [N, C, H_in,  W_in ] 已预置零
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = (blockIdx.x * blockDim.x + threadIdx.x)*vec;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    // 与前向完全相同的坐标映射
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    //float src_x = (w_out + 0.5f) * scale_x - 0.5f;
    float src_x[vec];
    for(int i = 0; i < vec; i++) {
        src_x[i] = (w_out + 0.5f + i) * scale_x - 0.5f;
    }

    int y0 = static_cast<int>(floorf(src_y));
    //int x0 = static_cast<int>(floorf(src_x));
    int x0[vec];
    int x1[vec];
    float beta[vec];
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        x0[i] = static_cast<int>(floorf(src_x[i]));
        x1[i] = x0[i] + 1;
        x0[i] = std::max(0, std::min(x0[i], W_in - 1));
        x1[i] = std::max(0, std::min(x1[i], W_in - 1));
        beta[i] = src_x[i] - x0[i]; 
    }
    int y1 = y0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));

    float alpha = src_y - floorf(src_y);
    //float beta  = src_x - floorf(src_x);
    

    // 读取 grad_output（转为 float 参与后续计算）
    //float g = static_cast<float>(grad_output[nc * H_out * W_out + h_out * W_out + w_out]);
    using vec_grad = __attribute__((__vector_size__(vec * sizeof(float)))) float;
    vec_grad g_vec = *(vec_grad*)(grad_output + nc * H_out * W_out + h_out * W_out + w_out);

    // 向 4 个输入像素散射（scatter-add），使用 float32 atomicAdd 保证精度
    float* gin = grad_input_f32 + nc * H_in * W_in;
    #ifdef __gfx936__

        if constexpr (1) {
            #pragma unroll
            for(int i = 0; i < vec; i++) {
                float g = static_cast<float>(g_vec[i]);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x0[i]), (1.0f - alpha) * (1.0f - beta[i]) * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x1[i]), (1.0f - alpha) *          beta[i]  * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x0[i]),          alpha  * (1.0f - beta[i]) * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x1[i]),          alpha  *          beta[i]  * g);
            }
        } else {
            // x4的原子加有问题 x1[0] x1[1] x1[2] x1[3] 不一定是连续的值排列
            vec_grad xy00, xy01, xy10, xy11;
            #pragma unroll
            for(int i = 0; i < vec; i++) {
                float g = static_cast<float>(g_vec[i]);
                xy00[i] = (1.0f - alpha) * (1.0f - beta[i]) * g;
                xy01[i] = (1.0f - alpha) *          beta[i]  * g;
                xy10[i] =          alpha  * (1.0f - beta[i]) * g;
                xy11[i] =          alpha  *          beta[i]  * g;
            }
        
        
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y0 * W_in + x0[0]), xy00);
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y1 * W_in + x0[0]), xy10);

            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y1 * W_in + x1[0]), xy11);
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y0 * W_in + x1[0]), xy01);
        }
    #else
        #pragma unroll
        for(int i = 0; i < vec; i++) {
            float g = static_cast<float>(g_vec[i]);
            atomicAdd(&gin[y0 * W_in + x0[i]], (1.0f - alpha) * (1.0f - beta[i]) * g);
            atomicAdd(&gin[y0 * W_in + x1[i]], (1.0f - alpha) *          beta[i]  * g);
            atomicAdd(&gin[y1 * W_in + x0[i]],          alpha  * (1.0f - beta[i]) * g);
            atomicAdd(&gin[y1 * W_in + x1[i]],          alpha  *          beta[i]  * g);
        }
    #endif
}

template <typename T, int vec>
__global__ void bilinear_upsample_backward_kernel_fp16(
    const T*   __restrict__ grad_output,    // [N, C, H_out, W_out]
    float*     __restrict__ grad_input_f32, // [N, C, H_in,  W_in ] 已预置零
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = (blockIdx.x * blockDim.x + threadIdx.x) * vec;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    // 与前向完全相同的坐标映射
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    //float src_x = (w_out + 0.5f) * scale_x - 0.5f;
    float src_x[vec];
    for(int i = 0; i < vec; i++) {
        src_x[i] = (w_out + 0.5f + i) * scale_x - 0.5f;
    }

    int y0 = static_cast<int>(floorf(src_y));
    //int x0 = static_cast<int>(floorf(src_x));
    int x0[vec];
    int x1[vec];
    float beta[vec];
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        x0[i] = static_cast<int>(floorf(src_x[i]));
        x1[i] = x0[i] + 1;
        x0[i] = std::max(0, std::min(x0[i], W_in - 1));
        x1[i] = std::max(0, std::min(x1[i], W_in - 1));
        beta[i] = src_x[i] - x0[i]; 
    }
    int y1 = y0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));

    float alpha = src_y - floorf(src_y);
    //float beta  = src_x - floorf(src_x);
    

    // 读取 grad_output（转为 float 参与后续计算）
    //float g = static_cast<float>(grad_output[nc * H_out * W_out + h_out * W_out + w_out]);
    using vec_grad = __attribute__((__vector_size__(vec * sizeof(_Float16)))) _Float16;
    vec_grad g_vec = *(vec_grad*)(grad_output + nc * H_out * W_out + h_out * W_out + w_out);

    // 向 4 个输入像素散射（scatter-add），使用 float32 atomicAdd 保证精度
    float* gin = grad_input_f32 + nc * H_in * W_in;
    #ifdef __gfx936__
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        float g = static_cast<float>(g_vec[i]);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x0[i]), (1.0f - alpha) * (1.0f - beta[i]) * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x1[i]), (1.0f - alpha) *          beta[i]  * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x0[i]),          alpha  * (1.0f - beta[i]) * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x1[i]),          alpha  *          beta[i]  * g);
    }
    #else
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        float g = static_cast<float>(g_vec[i]);
        atomicAdd(&gin[y0 * W_in + x0[i]], (1.0f - alpha) * (1.0f - beta[i]) * g);
        atomicAdd(&gin[y0 * W_in + x1[i]], (1.0f - alpha) *          beta[i]  * g);
        atomicAdd(&gin[y1 * W_in + x0[i]],          alpha  * (1.0f - beta[i]) * g);
        atomicAdd(&gin[y1 * W_in + x1[i]],          alpha  *          beta[i]  * g);
    }
    #endif
}

// ============================================================
//  Forward host wrapper
//  支持 dtype: float32 / float16 / bfloat16
// ============================================================
torch::Tensor bilinear_upsample(
    const torch::Tensor&       x,
    const std::vector<int64_t>& output_size
) {
    TORCH_CHECK(x.dim() == 4,
        "bilinear_upsample: input must be 4-D [N, C, H, W], got shape ", x.sizes());
    TORCH_CHECK(output_size.size() == 2,
        "bilinear_upsample: output_size must have exactly 2 elements [H_out, W_out]");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float   ||
                x.scalar_type() == at::ScalarType::Half    ||
                x.scalar_type() == at::ScalarType::BFloat16,
        "bilinear_upsample: only float32 / float16 / bfloat16 are supported, got ",
        x.scalar_type());

    auto x_contig = x.contiguous();
    auto stream   = at::cuda::getCurrentCUDAStream();

    const int N    = static_cast<int>(x_contig.size(0));
    const int C    = static_cast<int>(x_contig.size(1));
    const int H_in = static_cast<int>(x_contig.size(2));
    const int W_in = static_cast<int>(x_contig.size(3));
    const int H_out = static_cast<int>(output_size[0]);
    const int W_out = static_cast<int>(output_size[1]);

    TORCH_CHECK(H_out > 0 && W_out > 0,
        "bilinear_upsample: output_size must be positive, got [", H_out, ", ", W_out, "]");

    const float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    const float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (W_out + BLOCK_X - 1) / BLOCK_X,
        (H_out + BLOCK_Y - 1) / BLOCK_Y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_contig.scalar_type(), "bilinear_upsample",
        [&]() {
            bilinear_upsample_kernel<scalar_t><<<grid, block, 0, stream>>>(
                x_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                H_in, W_in, H_out, W_out,
                scale_y, scale_x
            );
        }
    );

    return output;
}

// ============================================================
//  Backward host wrapper
//  支持 dtype: float32 / float16 / bfloat16
//
//  Parameters:
//    grad_output - 4-D tensor [N, C, H_out, W_out]
//    input_size  - 原始输入形状 {N, C, H_in, W_in}
//
//  Returns:
//    grad_input  - 4-D tensor [N, C, H_in, W_in]，dtype 与 grad_output 相同
// ============================================================
torch::Tensor bilinear_upsample_backward(
    const torch::Tensor&        grad_output,
    const std::vector<int64_t>& input_size
) {
    TORCH_CHECK(grad_output.dim() == 4,
        "bilinear_upsample_backward: grad_output must be 4-D, got shape ", grad_output.sizes());
    TORCH_CHECK(input_size.size() == 4,
        "bilinear_upsample_backward: input_size must have 4 elements [N, C, H_in, W_in]");
    TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Float   ||
                grad_output.scalar_type() == at::ScalarType::Half    ||
                grad_output.scalar_type() == at::ScalarType::BFloat16,
        "bilinear_upsample_backward: only float32 / float16 / bfloat16 are supported, got ",
        grad_output.scalar_type());

    auto go_contig = grad_output.contiguous();
    auto stream    = at::cuda::getCurrentCUDAStream();

    const int N     = static_cast<int>(input_size[0]);
    const int C     = static_cast<int>(input_size[1]);
    const int H_in  = static_cast<int>(input_size[2]);
    const int W_in  = static_cast<int>(input_size[3]);
    const int H_out = static_cast<int>(go_contig.size(2));
    const int W_out = static_cast<int>(go_contig.size(3));

    const float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    const float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_X = 64;
    constexpr int vec = 2;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (W_out + BLOCK_X*vec - 1) / (BLOCK_X * vec),
        (H_out + BLOCK_Y - 1) / BLOCK_Y,
        N * C
    );
    

    if (go_contig.scalar_type() == at::ScalarType::Float) {
        // fp32 路径：grad_input 本身是 float32，直接传指针，省去中间缓冲区
        auto grad_input = torch::zeros({N, C, H_in, W_in}, go_contig.options());
        bilinear_upsample_backward_kernel_fp32<float, vec><<<grid, block, 0, stream>>>(
            go_contig.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            H_in, W_in, H_out, W_out,
            scale_y, scale_x
        );
        return grad_input;
    }

    // fp16 / bf16 路径：先向 float32 缓冲区累加，再 cast 回原始 dtype
    auto grad_input_fp32 = torch::zeros(
        {N, C, H_in, W_in},
        go_contig.options().dtype(torch::kFloat32));
    
    if(H_in == 16 && W_in == 31) {
        constexpr int BLOCK_Y1 = 32;
        constexpr int BLOCK_X1 = 32;
        constexpr int vec1 = 4;
        dim3 block1(BLOCK_X1, BLOCK_Y1);
        dim3 grid1(
            (W_out + BLOCK_X1*vec1 - 1) / (BLOCK_X1 * vec1),
            (H_out + BLOCK_Y1 - 1) / BLOCK_Y1,
            N * C
        );
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            go_contig.scalar_type(), "bilinear_upsample_backward",
            [&]() {
                bilinear_upsample_backward_kernel_fp16<scalar_t, vec1><<<grid1, block1, 0, stream>>>(
                    go_contig.data_ptr<scalar_t>(),
                    grad_input_fp32.data_ptr<float>(),
                    H_in, W_in, H_out, W_out,
                    scale_y, scale_x
                );
            }
        );
    }
    else{
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            go_contig.scalar_type(), "bilinear_upsample_backward",
            [&]() {
                bilinear_upsample_backward_kernel_fp16<scalar_t, vec><<<grid, block, 0, stream>>>(
                    go_contig.data_ptr<scalar_t>(),
                    grad_input_fp32.data_ptr<float>(),
                    H_in, W_in, H_out, W_out,
                    scale_y, scale_x
                );
            }
        );
    }

    return grad_input_fp32.to(grad_output.dtype());
}

} // namespace native
} // namespace at
