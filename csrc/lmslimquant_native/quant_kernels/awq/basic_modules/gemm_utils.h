#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"

namespace instance {
using profiler_InstanceType = void (*)(const void* A,
                                       const void* B0,
                                       const void* B1,
                                       void* C,
                                       int M,
                                       int N,
                                       int K,
                                       int StrideA,
                                       int StrideB,
                                       int StrideC,
                                       int SplitK_Batch,
                                       int Group,
                                       void* splitK_workspace, // 用于SplitK添加pad的显存空间
                                       hipStream_t stream_id);

} // namespace instance

namespace weight_only_tools {

__host__ __device__ ck::half8_t dequantize_s4_to_fp16x2(uint32_t const& source) {
    ck::half8_t result;
    uint32_t* h                           = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& i4s                   = reinterpret_cast<uint32_t const&>(source);
    static constexpr uint32_t BOT_MASK    = 0x000f000f;
    static constexpr uint32_t TOP_MASK    = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;       // `64`
    static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4; // `64` >> 4
    const uint32_t top_i4s                = i4s >> 8;

    h[0] = (i4s & BOT_MASK) | MAGIC_NUM_2;
    h[1] = (i4s & TOP_MASK) | MAGIC_NUM_1;
    h[2] = (top_i4s & BOT_MASK) | MAGIC_NUM_2;
    h[3] = (top_i4s & TOP_MASK) | MAGIC_NUM_1;
    h[0] <<= 4;
    h[2] <<= 4;

    return result;
}

struct Dequation {
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};
    static constexpr auto I4 = ck::Number<4>{};
    static constexpr auto I5 = ck::Number<5>{};
    static constexpr auto I6 = ck::Number<6>{};
    static constexpr auto I7 = ck::Number<7>{};

    // __host__ __device__ constexpr void
    __host__ __device__ void operator()(ck::half8_t& b, const uint32_t& b0, const uint32_t& b1) const // 65.7TF 3840 69.49TF 7680 72.45TF 11520
    {
        // 1uint32 --> 8个half
        b = dequantize_s4_to_fp16x2(b0);

        // 把量化因子转成half
        const ck::half_t* zero_scale = reinterpret_cast<const ck::half_t*>(&b1);
        ck::static_for<0, 8, 1>{}([&](auto i) { b[i.value] = (b[i.value] - zero_scale[0]) * zero_scale[1]; });
    }
    // pack8_data
    constexpr const static bool is_pack8_invocable = true;
};

__inline__ __device__ ck::half8_t dequant_convert(uint32_t const& source, const uint32_t& zero_scale) {
    auto out = dequantize_s4_to_fp16x2(source);
    auto tmp = reinterpret_cast<const ck::half_t*>(&zero_scale);

    ck::half8_t halves;
    ck::static_for<0, 8, 1>{}([&](auto i) { halves[i.value] = (out[i.value] - tmp[0]) * tmp[1]; });

    return halves;
}

__device__ __forceinline__ float warpReduceSum(float sum, int threadNum) {
    if(threadNum >= 64)
        sum += __shfl_down(sum, 32);
    if(threadNum >= 32)
        sum += __shfl_down(sum, 16);
    if(threadNum >= 16)
        sum += __shfl_down(sum, 8);
    if(threadNum >= 8)
        sum += __shfl_down(sum, 4);
    if(threadNum >= 4)
        sum += __shfl_down(sum, 2);
    if(threadNum >= 2)
        sum += __shfl_down(sum, 1);
    return sum;
}

__device__ void amd_assembly_product_1x2(const ck::half2_t* a, const ck::half2_t* b, float* c0) {
    asm volatile("\n \
                v_dot2_f32_f16 %0, %1, %2, %0\n \
                v_dot2_f32_f16 %0, %3, %4, %0\n \
                "
                 : "=v"(*c0)
                 : "v"(*a), "v"(*(b)), "v"(*(a + 1)), "v"(*(b + 1)), "0"(*c0));
}

__device__ void amd_assembly_outer_product_1x4(const ck::half2_t* a, const ck::half2_t* b, float* c0) {
    amd_assembly_product_1x2(a, b, c0);
    amd_assembly_product_1x2(a + 2, b + 2, c0);
}

__device__ void amd_assembly_product_1x4(const ck::half8_t& a, const ck::half8_t& b, float& c) {
    const ck::half2_t* p_a_half2 = reinterpret_cast<const ck::half2_t*>(&a);
    const ck::half2_t* p_b_half2 = reinterpret_cast<const ck::half2_t*>(&b);

    asm volatile("\n \
            v_dot2_f32_f16 %0, %1, %2, %0\n \
            v_dot2_f32_f16 %0, %3, %4, %0\n \
            v_dot2_f32_f16 %0, %5, %6, %0\n \
            v_dot2_f32_f16 %0, %7, %8, %0\n \
            "
                 : "=v"(c)
                 : "v"(p_a_half2[0]),
                   "v"(p_b_half2[0]),
                   "v"(p_a_half2[1]),
                   "v"(p_b_half2[1]),
                   "v"(p_a_half2[2]),
                   "v"(p_b_half2[2]),
                   "v"(p_a_half2[3]),
                   "v"(p_b_half2[3]),
                   "0"(c));
}

} // namespace weight_only_tools
