// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cstdint>

// 新增：CUDA Stream兼容头文件
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include <torch/extension.h>
#include <cuda_runtime_api.h>

// 自定义CUDA错误检查函数
// #define checkCUDA(call) do { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     } \
// } while (0)

// 获取当前CUDA流（PyTorch兼容）
// cudaStream_t getCurrentCUDAStream() {
//     return at::cuda::getCurrentCUDAStream();
// }

#define PACK_FACTOR 8
#define WARP_SIZE 64
#define MEM_ACCESS_SIZE 128


namespace at {
namespace native {

__forceinline__ __device__ void dequantize_s4_to_fp16x2(half2 const &source, uint4 *result) {

    uint32_t *h        = reinterpret_cast<uint32_t *>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    //              : "=r"(h[0])
    //              : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    h[0] = (i4s & BOTTOM_MASK) | I4s_TO_F16s_MAGIC_NUM;

    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    //              : "=r"(h[1])
    //              : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    h[1] = (i4s & TOP_MASK) | I4s_TO_F16s_MAGIC_NUM;

    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    //              : "=r"(h[2])
    //              : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    h[2] = (top_i4s & BOTTOM_MASK) | I4s_TO_F16s_MAGIC_NUM;

    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    //              : "=r"(h[3])
    //              : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    h[3] = (top_i4s & TOP_MASK) | I4s_TO_F16s_MAGIC_NUM;

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    half2 *_h = reinterpret_cast<half2*>(h);

    _h[0] = __hsub2(_h[0], __float2half2_rn(1024.0f));
    _h[1] = __hfma2(_h[1], __float2half2_rn(0.0625f), __float2half2_rn(-64.0f));
    _h[2] = __hsub2(_h[2], __float2half2_rn(1024.0f));
    _h[3] = __hfma2(_h[3], __float2half2_rn(0.0625f), __float2half2_rn(-64.0f));
   
}

template<typename half_t, int NPerBlock, int BlockSize, int GroupSize>
__global__ void awq_w4a16_dequant_trans_kernel(const uint32_t *weight,
                            const half_t *scales,
                            const half_t *zeros,
                            half_t *outputs,
                            const int IC,
                            const int OC) {

    // using half2_t  = typename packed_as<half_t, 2>::type; // half2
    using half2_t  = half2;


    const int kStride            = 64; // 64个元素是一组，处理scale和zeros用
    const int kElemsPerThread    = MEM_ACCESS_SIZE / 4; // 一个线程处理32个元素，一个block处理8192个元素
    const int kThreadsNumPerTile = kStride / kElemsPerThread; // 64 / 32 = 2
    constexpr int Num         = NPerBlock; // 一个block处理2行，2 * batchsize
    constexpr int kInterleave = 4;


    alignas(16) uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    alignas(16) half_t half_weight_buffer[kElemsPerThread]; // dequant后的结果
    alignas(16) half_t local_scale[NPerBlock]; // dequant的scale
    alignas(16) half_t local_scaled_zeros[NPerBlock]; // dequant的零点

    
    // 一个block处理两行数据，一行数据代表原有数据的4行，所以一个block处理8行数据
    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave; // 8 * bid
    
    // 两个线程一组，先算出第几组，然后再算每个线程在这一组中的位置。
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave; // (tid / 2) % 4 
    const int act_k_offset   = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride + // (tx / 8) * 64 + (tx % 2) * 32, 例0, 2, 4, 6取相同input
                             (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;
    
    //  IC是unpack数据一行有多少数据，所以要除PACK_FACTOR，找到weight的对应位置
    const uint32_t *blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;  // IC是unpack数据一行有多少数据，所以要除PACK_FACTOR，找到weight的对应位置
    
    // weight的行数对应scale和zeros的列。OC是输出的unpack size。
    // blk_row_offset + thd_row_offset是scale和zeros的列数，group_offset是行数。
    const half_t *scale_ptr        = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const half_t *zeros_ptr        = zeros + blk_row_offset + thd_row_offset + group_offset * OC;

    // step
    const int act_forward_step   = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;
 
    int gy = blk_row_offset;
    // Main loop iteration, each block completes the outputs for several OCs
    // 0 ~ 8192; < 12288; += 8192 
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread) {
        // if (blockIdx.x == 0 && threadIdx.x == 1) {
        //     printf("【第一个Block-第一个线程】kk = %d\n", kk);
        // }
        #pragma unroll 
        for (int idx = 0; idx < NPerBlock; ++idx) { // 0, 1
            int y_offset = idx * kInterleave + kk / IC;

            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            // idx * kInterleave * IC + kk = idx * 12288 + kk, 128 int4 per thread
            *((float4 *)(local_qweights)) = *((float4 *)(blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));
            local_scale[idx]              = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx]       = *(zeros_ptr + idx * kInterleave);

            // Map int4 qweight to fp format
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i) { // 0 ~ 4
                // Converts 32 bits (8 x int4) to 8 fp16
                // caution shuffle packed fp16
                // half_weight_buffer存了32个s4转换后的fp16 weight, 且位置已经重排。
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2_t *>(local_qweights + i),
                                        reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

            // half_weight_buffer is 32 fp16
            // Dequantize (apply s/z) and shuffle elements to match the weight packing format
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i) {    // 4
                #pragma unroll
                for (int j = 0; j < PACK_FACTOR / 2; ++j) {  // 4
                    half2_t w = *reinterpret_cast<half2_t *>(half_weight_buffer + (i * PACK_FACTOR + j * 2));
                    w         = __hfma2(w, __halves2half2(local_scale[idx], local_scale[idx]), __halves2half2(local_scaled_zeros[idx], local_scaled_zeros[idx]));

                    int col = kk % IC + i * PACK_FACTOR + j * 2;
                    int row = gy + y_offset;

                    /*
                    if (blockIdx.x == 0 && threadIdx.x == 1) {
                        printf("【第一个Block-第一个线程】col = %d\n", col);
                        printf("【第一个Block-第一个线程】row = %d\n", row);
                        printf("【第一个Block-第一个线程】idx = %d\n", row * IC + col);
                    }
                    outputs[row * IC + col] = w.x;
                    outputs[row * IC + col + 1] = w.y;
                    */

                    // trans
                    outputs[row + col * OC] = w.x;
                    outputs[row + (col + 1) * OC] = w.y;
                }
            }
            
        }
        
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }
}


torch::Tensor awq_w4a16_dequant_trans(torch::Tensor _weight, torch::Tensor _scaling_factors, torch::Tensor _zeros, int IC, int OC, int group_size) {
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    

        std::vector<int64_t> weight_shape = _weight.sizes().vec();
        assert(weight_shape.size() == 2 && "_weight must be 2D Tensor!");
        std::vector<int64_t> output_shape = {IC, OC};


        using half_t = half;
        assert(_zeros.dtype() == at::kHalf && "_zeros must be half (float16) dtype!");
        assert(_scaling_factors.dtype() == at::kHalf && "_scaling_factors must be half (float16) dtype!");
        auto weight          = reinterpret_cast<uint32_t*>(_weight.data_ptr()); 
        auto zeros           = reinterpret_cast<half_t*>(_zeros.data_ptr()); 
        auto scaling_factors = reinterpret_cast<half_t*>(_scaling_factors.data_ptr());  

        torch::Tensor _output = torch::empty(output_shape, torch::TensorOptions().dtype(at::kHalf).device(_weight.device()));
        auto out_put         = reinterpret_cast<half_t*>(_output.data_ptr());  

        
        static constexpr int N_PER_BLOCK  = 2;
        static constexpr int K_INTERLEAVE = 4;
        static constexpr int BLOCK_SIZE   = 256;


        dim3 num_blocks(OC / N_PER_BLOCK / K_INTERLEAVE);
        dim3 num_threads(BLOCK_SIZE); // 一个block处理 2 行，每行 4 unpack行，共 8 行


        constexpr int GROUP_SIZE = 64;
        assert(group_size == GROUP_SIZE);

        awq_w4a16_dequant_trans_kernel<half_t, N_PER_BLOCK, BLOCK_SIZE, GROUP_SIZE>
                    <<<num_blocks, num_threads, 0, stream>>>
                    (weight, scaling_factors, zeros, out_put, IC, OC);
        // checkCUDA(cudaGetLastError());

        return _output;
}

} // namespace native
} // namespace at



// namespace py = pybind11;

// 绑定函数到Python模块，import的名字叫awq_dequant
// PYBIND11_MODULE(awq_dequant, m) {
//     m.doc() = "AWQ 4bit to 16bit dequantization and trans CUDA kernel"; // 模块文档
    
//     // 调用的函数在python中叫awq_w4a16_dequant_trans，注册的是cu中的awq_w4a16_dequant_trans函数
//     m.def("awq_w4a16_dequant_trans",
//           &awq_w4a16_dequant_trans, // py::arg可省略
//           py::arg("_weight"), // 4bit量化权重，uint32类型，2D张量 [OC, IC//8]
//           py::arg("_scaling_factors"), // scale因子，float16类型，1D张量
//           py::arg("_zeros"), // zero点，float16类型，1D张量
//           py::arg("IC"), // 输出通道数IC
//           py::arg("OC"), // 输入通道数OC
//           py::arg("group_size")); // AWQ分组大小（仅支持64）
// }