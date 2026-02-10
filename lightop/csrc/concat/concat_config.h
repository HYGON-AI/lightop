#pragma once
#include <unordered_map>
#include <functional>
#include "hip/hip_bf16.h"  
#include "hip/hip_fp16.h"
namespace at {
namespace native {
///////////////////////////////////////////////////////////////////////Gemm params///////////////////////////////////////////////////////////////////////////////////////

// GemmParams结构体定义
template <typename T>
struct GemmParams {
    GemmParams(const T* input_A,
               const T* input_B,
               const T* ptr_C,
               uint32_t M,
               uint32_t N, 
               uint32_t a_section_num,
               uint32_t b_section_num,
               uint32_t c_section_num, 
               uint32_t stride_am,
               uint32_t stride_an,
               uint32_t stride_ak,
               uint32_t stride_bm, 
               uint32_t stride_bn,
               uint32_t stride_bk)
        : input_A(input_A),
          input_B(input_B),
          ptr_C(ptr_C),
          M(M),
          N(N),
          a_section_num(a_section_num),
          b_section_num(b_section_num),
          c_section_num(c_section_num),
          stride_am(stride_am),
          stride_an(stride_an),
          stride_ak(stride_ak),
          stride_bm(stride_bm),
          stride_bn(stride_bn),
          stride_bk(stride_bk)
         {}

    const T* input_A;                   
    const T* input_B;                  
    const T* ptr_C;                   
    uint32_t M;                  
    uint32_t N;                  
    uint32_t a_section_num;                   
    uint32_t b_section_num;
    uint32_t c_section_num;                      
    uint32_t stride_am;               
    uint32_t stride_an;               
    uint32_t stride_ak;              
    uint32_t stride_bm;              
    uint32_t stride_bn;               
    uint32_t stride_bk;                           
};


///////////////////////////////////////////////////////////////////////launch_function///////////////////////////////////////////////////////////////////////////////////////

using bhalf_t = __hip_bfloat16;
using half  = __half;
template<int per_block_A, int per_block_B, int Blcoksize,int WARP_NUM, typename T>
void launch_concat_prefill(const GemmParams<T>& params);

template<int per_block_A, int per_block_B, int Blcoksize,int WARP_NUM, typename T>
void launch_concat_decode(const GemmParams<T>& params);

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
using KernelFunc = std::function<void(const GemmParams<bhalf_t>&)>;
static std::unordered_map<int, KernelFunc> kernel_maps_prefill = {
    {0, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 8, 1024, 4, bhalf_t>(p); }},
    {1, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<8, 1, 1024, 1, bhalf_t>(p); }},
    {2, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<8, 1, 1024, 2, bhalf_t>(p); }},
    {3, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<8, 1, 1024, 4, bhalf_t>(p); }},    
    {4, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 1, 1024, 1, bhalf_t>(p); }},
    {5, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 1, 1024, 2, bhalf_t>(p); }},
    {6, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 8, 1024, 4, bhalf_t>(p); }},
    {7, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<32, 1, 1024, 1, bhalf_t>(p); }},
    {8, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<32, 1, 1024, 1, bhalf_t>(p); }},
    {9, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<32, 1, 1024, 1, bhalf_t>(p); }},
    {10, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 2, 1024, 1, bhalf_t>(p); }},
    {11, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 2, 1024, 1, bhalf_t>(p); }},
    {12, [](const GemmParams<bhalf_t>& p) { launch_concat_prefill<16, 2, 1024, 1, bhalf_t>(p); }},

};

static std::unordered_map<int, KernelFunc> kernel_maps_decode = {
    {0, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<4, 8, 1024, 4, bhalf_t>(p); }},
    {1, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<8, 1, 1024, 1, bhalf_t>(p); }},
    {2, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<8, 1, 1024, 2, bhalf_t>(p); }},
    {3, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<8, 1, 1024, 4, bhalf_t>(p); }},    
    {4, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<16, 1, 1024, 1, bhalf_t>(p); }},
    {5, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<16, 1, 1024, 2, bhalf_t>(p); }},
    {6, [](const GemmParams<bhalf_t>& p) { launch_concat_decode<16, 8, 1024, 4, bhalf_t>(p); }},
};



using KernelFunc3 = std::function<void(const GemmParams<half>&)>;
static std::unordered_map<int, KernelFunc3> kernel_maps_prefill_half = {
    {0, [](const GemmParams<half>& p) { launch_concat_prefill<16, 8, 1024, 4, half>(p); }},
    {1, [](const GemmParams<half>& p) { launch_concat_prefill<8, 1, 1024, 1, half>(p); }},
    {2, [](const GemmParams<half>& p) { launch_concat_prefill<8, 1, 1024, 2, half>(p); }},
    {3, [](const GemmParams<half>& p) { launch_concat_prefill<8, 1, 1024, 4, half>(p); }},    
    {4, [](const GemmParams<half>& p) { launch_concat_prefill<16, 1, 1024, 1, half>(p); }},
    {5, [](const GemmParams<half>& p) { launch_concat_prefill<16, 1, 1024, 2, half>(p); }},
    {6, [](const GemmParams<half>& p) { launch_concat_prefill<16, 8, 1024, 4, half>(p); }},
    {7, [](const GemmParams<half>& p) { launch_concat_prefill<32, 1, 1024, 1, half>(p); }},
    {8, [](const GemmParams<half>& p) { launch_concat_prefill<32, 1, 1024, 1, half>(p); }},
    {9, [](const GemmParams<half>& p) { launch_concat_prefill<32, 1, 1024, 1, half>(p); }},
    {10, [](const GemmParams<half>& p) { launch_concat_prefill<16, 2, 1024, 1, half>(p); }},
    {11, [](const GemmParams<half>& p) { launch_concat_prefill<16, 2, 1024, 1, half>(p); }},
    {12, [](const GemmParams<half>& p) { launch_concat_prefill<16, 2, 1024, 1, half>(p); }},

};

static std::unordered_map<int, KernelFunc3> kernel_maps_decode_half = {
    {0, [](const GemmParams<half>& p) { launch_concat_decode<4, 8, 1024, 4, half>(p); }},
    {1, [](const GemmParams<half>& p) { launch_concat_decode<8, 1, 1024, 1, half>(p); }},
    {2, [](const GemmParams<half>& p) { launch_concat_decode<8, 1, 1024, 2, half>(p); }},
    {3, [](const GemmParams<half>& p) { launch_concat_decode<8, 1, 1024, 4, half>(p); }},    
    {4, [](const GemmParams<half>& p) { launch_concat_decode<16, 1, 1024, 1, half>(p); }},
    {5, [](const GemmParams<half>& p) { launch_concat_decode<16, 1, 1024, 2, half>(p); }},
    {6, [](const GemmParams<half>& p) { launch_concat_decode<16, 8, 1024, 4, half>(p); }},
};

using KernelFunc2 = std::function<void(const GemmParams<uint8_t>&)>;
static std::unordered_map<int, KernelFunc2> kernel_maps_prefill_fp8 = {
    {0, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 8, 1024, 4, uint8_t>(p); }},
    {1, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<8, 1, 1024, 1, uint8_t>(p); }},
    {2, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<8, 1, 1024, 2, uint8_t>(p); }},
    {3, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<8, 1, 1024, 4, uint8_t>(p); }},    
    {4, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 1, 1024, 1, uint8_t>(p); }},
    {5, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 1, 1024, 2, uint8_t>(p); }},
    {6, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 8, 1024, 4, uint8_t>(p); }},
    {7, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<32, 1, 1024, 1, uint8_t>(p); }},
    {8, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<32, 1, 1024, 1, uint8_t>(p); }},
    {9, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<32, 1, 1024, 1, uint8_t>(p); }},
    {10, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 2, 1024, 1, uint8_t>(p); }},
    {11, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 2, 1024, 1, uint8_t>(p); }},
    {12, [](const GemmParams<uint8_t>& p) { launch_concat_prefill<16, 2, 1024, 1, uint8_t>(p); }},

};

static std::unordered_map<int, KernelFunc2> kernel_maps_decode_fp8 = {
    {0, [](const GemmParams<uint8_t>& p) { launch_concat_decode<4, 8, 1024, 4, uint8_t>(p); }},
    {1, [](const GemmParams<uint8_t>& p) { launch_concat_decode<8, 1, 1024, 1, uint8_t>(p); }},
    {2, [](const GemmParams<uint8_t>& p) { launch_concat_decode<8, 1, 1024, 2, uint8_t>(p); }},
    {3, [](const GemmParams<uint8_t>& p) { launch_concat_decode<8, 1, 1024, 4, uint8_t>(p); }},    
    {4, [](const GemmParams<uint8_t>& p) { launch_concat_decode<16, 1, 1024, 1, uint8_t>(p); }},
    {5, [](const GemmParams<uint8_t>& p) { launch_concat_decode<16, 1, 1024, 2, uint8_t>(p); }},
    {6, [](const GemmParams<uint8_t>& p) { launch_concat_decode<16, 8, 1024, 4, uint8_t>(p); }},
};

}
}