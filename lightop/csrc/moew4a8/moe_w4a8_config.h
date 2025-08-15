#pragma once
#include <unordered_map>
#include <functional>
// MOE prefill和decode的分界线
#define MOE_THRESHOLD 128  

namespace at {
namespace native {
///////////////////////////////////////////////////////////////////////Gemm params///////////////////////////////////////////////////////////////////////////////////////

// GemmParams结构体定义
template <typename T>
struct GemmParams {
    GemmParams(const T* ptr_A,
               const T* ptr_B0,
               bhalf_t* ptr_C,
               float* ptr_A_scale,
               float* ptr_B_scale,
               const float* topk_weights,
               const int32_t* sorted_token_ids,
               const int32_t* expert_ids,
               const int32_t num_tokens_post_pad,
               const int32_t* num_tokens_post_pad_ptr,
               uint32_t size_m,
               uint32_t size_n, 
               uint32_t size_k,
               uint32_t size_kofb, 
               uint32_t stride_asm,
               uint32_t stride_ask,
               uint32_t stride_bse,
               uint32_t stride_bsn, 
               uint32_t stride_bsk,
               uint32_t sorted_token_lens,
               uint32_t top_k,
               uint32_t delta)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          size_kofb(size_kofb),
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta)
         {}

    const T* ptr_A;                    // input
    const T* ptr_B0;                   // weight
    bhalf_t* ptr_C;                   // output
    float* ptr_A_scale;                // input scale
    float* ptr_B_scale;                // weight scale
    const float* topk_weights;         // topk weights
    const int32_t* sorted_token_ids;   // sorted token ids
    const int32_t* expert_ids;         // expert ids
    const int32_t num_tokens_post_pad;// num tokens after padding
    const int32_t* num_tokens_post_pad_ptr;// num tokens after padding
    uint32_t size_m;                   // input size m
    uint32_t size_n;                   // output size n
    uint32_t size_k;                    // input size k of a
    uint32_t size_kofb;                     // input size k of b
    uint32_t stride_asm;               // input scale stride m
    uint32_t stride_ask;               // input scale stride k
    uint32_t stride_bse;               // weight scale stride expert
    uint32_t stride_bsn;               // weight scale stride n
    uint32_t stride_bsk;               // weight scale stride k
    uint32_t sorted_token_lens;        // sorted token length
    uint32_t top_k;                    // top k
    uint32_t delta;                    // delta
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w4a8_gemm_first_stage(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w4a8_gemm_second_stage(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w4a8_gemm_first_stage_decode(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w4a8_gemm_second_stage_decode(const GemmParams<T>& params);

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
using KernelFunc = std::function<void(const GemmParams<char>&)>;

static std::unordered_map<int, KernelFunc> kernel_maps_gemm1_prefill = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {0, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 16, 64, char>(p); }},

    {1, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 32, 16, 64, char>(p); }},
    {2, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {3, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 128, 128, 16, 32, 64, char>(p); }},
    {4, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 32, 32, 64, char>(p); }},
    //{5, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {6, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 256, 128, 16, 64, 64, char>(p); }},
    //{7, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{8, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {9, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 32, 16, 64, char>(p); }},
    {10, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {11, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 256, 128, 32, 32, 64, char>(p); }},
    //{12, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {13, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 16, 64, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {14, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 32, 128, 16, 16, 64, char>(p); }},
    {15, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 32, 64, char>(p); }},
    {16, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 32, 16, 64, char>(p); }},
    {17, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {18, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 16, 64, char>(p); }},
    {19, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 32, 16, 64, char>(p); }},
    {20, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 16, 32, 64, char>(p); }},
    {21, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {22, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 16, 16, 64, char>(p); }},
    {23, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 16, 32, 64, char>(p); }},
    {24, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<128, 32, 128, 32, 16, 64, char>(p); }},
    {25, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<128, 64, 128, 32, 32, 64, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {26, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 16, 128, 16, 16, 64, char>(p); }},
    {27, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
    {28, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 32, 128, 16, 32, 64, char>(p); }},
    {29, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
    {30, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
    // 2 warp
    {31, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
    {32, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 32, 128, 16, 32, 64, char>(p); }},
    {33, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 64, 64, char>(p); }},
    //{34, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 16, 128, 64, char>(p); }}, //寄存器溢出 156
    // 4 warp
    {35, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 16, 128, 16, 16, 64, char>(p); }},
    {36, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 16, 32, 64, char>(p); }},
    // {37, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 16, 64, 64, char>(p); }}, // 寄存器溢出 120
    // {38, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 16, 128, 64, char>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {39, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 32, 16, 32, char>(p); }},
    {40, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 64, 16, 32, char>(p); }},

    {41, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 32, 32, 32, char>(p); }},
    {42, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 32, 32, char>(p); }},

    {43, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 256, 128, 32, 64, 32, char>(p); }},
    {44, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 64, 32, char>(p); }},
    
    // 8warp-4stages-warpk
    {45, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 16, 32, char>(p); }},
    // 8warp
    {46, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 32, 32, char>(p); }},

    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {47, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {48, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {49, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<128, 64, 128, 32, 32, 32, char>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {50, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 256, 16, 16, 64, char>(p); }},
    {51, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 256, 32, 16, 64, char>(p); }},
    {52, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 256, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {53, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 128, 256, 16, 32, 64, char>(p); }},
    {54, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 256, 32, 32, 64, char>(p); }},
    //{55, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {56, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 128, 128, 16, 64, 64, char>(p); }},
    //{57, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{58, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {59, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 256, 32, 16, 64, char>(p); }},
    {60, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 256, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {61, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 256, 256, 32, 32, 64, char>(p); }},
    //{62, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {63, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 256, 256, 64, 16, 64, char>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {64, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 16, 64, char>(p); }},
    {65, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 32, 16, 64, char>(p); }},
    {66, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {67, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 128, 128, 16, 32, 64, char>(p); }},
    {68, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 32, 32, 64, char>(p); }},
    // 4warp BLOCKN 256 
    {69, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 256, 128, 16, 64, 64, char>(p); }},
    // 8warp BLOCKN 128 
    {70, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 32, 16, 64, char>(p); }},
    {71, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 64, 16, 64, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {74, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 32, 128, 16, 16, 64, char>(p); }},
    {75, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 32, 64, char>(p); }},
    {76, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 32, 16, 64, char>(p); }},
    {77, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {78, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 16, 64, char>(p); }},
    {79, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 32, 16, 64, char>(p); }},
    {80, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 128, 128, 16, 32, 64, char>(p); }},
    {81, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 128, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {82, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 16, 16, 64, char>(p); }},
    {83, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 64, 128, 16, 32, 64, char>(p); }},
    {84, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<128, 32, 128, 32, 16, 64, char>(p); }},
    {85, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<128, 64, 128, 32, 32, 64, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {86, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 16, 128, 16, 16, 64, char>(p); }},
    {87, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
    {88, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 32, 128, 16, 32, 64, char>(p); }},
    {89, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
    {90, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
    // 2 warp
    {91, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
    {92, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 32, 128, 16, 32, 64, char>(p); }},
    {93, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<32, 64, 128, 16, 64, 64, char>(p); }},
    // 4 warp
    {94, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 16, 128, 16, 16, 64, char>(p); }},
    {95, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage<64, 32, 128, 16, 32, 64, char>(p); }},

};


// gemm2  目前仅支持部分 config   
static std::unordered_map<int, KernelFunc> kernel_maps_gemm2_prefill = {
    // ======================================== 当前已支持的config 4warp + 二级流水 =============================
    // 4warp BLOCKN 64 
    {0, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<16, 64, 128, 16, 16, 64, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<32, 64, 128, 32, 16, 64, char>(p); }},
    {2, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<64, 64, 128, 64, 16, 64, char>(p); }},
    // // 4warp BLOCKN 128 
    {3, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<16, 128, 128, 16, 32, 64, char>(p); }},
    {4, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<32, 128, 128, 32, 32, 64, char>(p); }},
    {5, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<16, 256, 128, 16, 64, 64, char>(p); }},

    {6, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<32, 32, 128, 16, 16, 64, char>(p); }},
    {7, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<32, 64, 128, 16, 32, 64, char>(p); }},
    {8, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<64, 32, 128, 32, 16, 64, char>(p); }},
    {9, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<64, 64, 128, 32, 32, 64, char>(p); }},

    {10, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<64, 32, 128, 16, 32, 64, char>(p); }},


    {11, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<64, 64, 64, 64, 16, 32, char>(p); }},
    // // 4warp BLOCKN 128 选中的
    // // 4warp BLOCKN 256 
    // // 4 warp:  2,2,1
    // // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {12, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<32, 64, 64, 16, 32, 32, char>(p); }},
    {13, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage<128, 128, 64, 128, 32, 32, char>(p); }},
};

///////////////////////////////////////////////////////////////////////decode_config///////////////////////////////////////////////////////////////////////////////////////


static std::unordered_map<int, KernelFunc> kernel_maps_gemm1_decode = {
    // MN 16 16 
    {0, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 64, 16, 16, 64, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 128, 16, 16, 64, char>(p); }},
    {2, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 256, 16, 16, 64, char>(p); }},
    {3, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 512, 16, 16, 64, char>(p); }},
    // MN 16 32  N方向两个warp
    {4, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 64, 16, 16, 64, char>(p); }},
    {5, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 128, 16, 16, 64, char>(p); }},
    {6, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 256, 16, 16, 64, char>(p); }},
    {7, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
    // MN 16 32  N方向重复算
    {8, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 64, 16, 32, 64, char>(p); }},
    {9, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 128, 16, 32, 64, char>(p); }},
    {10, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 256, 16, 32, 64, char>(p); }},
    {11, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 32, 64, char>(p); }},
    // MN 16 64 N方向两个warp+重复算两次
    {12, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 64, 16, 32, 64, char>(p); }},
    {13, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 128, 16, 32, 64, char>(p); }},
    {14, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 256, 16, 32, 64, char>(p); }},
    {15, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 512, 16, 32, 64, char>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {16, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 128, 16, 16, 128, char>(p); }},
    {17, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 256, 16, 16, 128, char>(p); }},
    {18, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 512, 16, 16, 128, char>(p); }},
    {19, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 1024, 16, 16, 128, char>(p); }},
    // MN 16 32  N方向两个warp
    {20, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 128, 16, 16, 128, char>(p); }},
    {21, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 256, 16, 16, 128, char>(p); }},
    {22, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 16, 128, char>(p); }},
    {23, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 16, 128, char>(p); }},
    // MN 16 32  N方向重复算
    {24, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 128, 16, 32, 128, char>(p); }},
    {25, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 256, 16, 32, 128, char>(p); }},
    {26, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 32, 128, char>(p); }},
    {27, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 32, 128, char>(p); }},
    // MN 16 64 N方向两个warp+重复算两次
    // {28, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 128, 16, 32, 128, char>(p); }},
    // {29, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 256, 16, 32, 128, char>(p); }},
    // {30, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 512, 16, 32, 128, char>(p); }},
    // {31, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 1024, 16, 32, 128, char>(p); }},
    // 16 warp
    {28, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 1024, 16, 16, 64, char>(p); }},
    {29, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 32, 64, char>(p); }},
    {30, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 16, 128, char>(p); }},
    {31, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 32, 128, char>(p); }},

    {32, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 384, 16, 32, 64, char>(p); }},
    {33, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 384, 16, 16, 64, char>(p); }},
    {34, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 384, 16, 16, 64, char>(p); }},
    {35, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 384, 16, 32, 64, char>(p); }},
    {36, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 16, 1024, 16, 16, 64, char>(p); }},
    {37, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 32, 64, char>(p); }},
    {38, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
    {39, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},

    {40, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
    {41, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 512, 16, 32, 64, char>(p); }},
    {42, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 256, 16, 16, 64, char>(p); }},
    {43, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 128, 256, 32, 16, 64, char>(p); }},
    // {40, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 1024, 16, 32, 128, char>(p); }},
    // {41, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 64, 1024, 16, 64, 128, char>(p); }},
    // {42, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 16, 128, char>(p); }},
    // {43, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_first_stage_decode<16, 32, 1024, 16, 32, 256, char>(p); }}
    
    
};

static std::unordered_map<int, KernelFunc> kernel_maps_gemm2_decode = {
    // MN 16 16 
    {0, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 64, 16, 16, 64, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 128, 16, 16, 64, char>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {2, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 64, 16, 16, 64, char>(p); }},
    {3, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 16, 64, char>(p); }},
    {4, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 64, 16, 16, 64, char>(p); }},
    {5, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 16, 64, char>(p); }},
    {6, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 64, 16, 16, 64, char>(p); }},
    {7, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 16, 64, char>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {8, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 64, 16, 32, 64, char>(p); }},
    {9, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 32, 64, char>(p); }},
    {10, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 64, 16, 32, 64, char>(p); }},
    {11, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 32, 64, char>(p); }},
    {12, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 64, 16, 32, 64, char>(p); }},
    {13, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 32, 64, char>(p); }},
    {14, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 256, 64, 16, 32, 64, char>(p); }},
    {15, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 256, 128, 16, 32, 64, char>(p); }},
    // warpk 128
    {16, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 128, 16, 16, 128, char>(p); }},
    {17, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 128, 16, 16, 128, char>(p); }},
    {18, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 16, 128, char>(p); }},
    {19, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 16, 128, char>(p); }},
    {20, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 16, 128, char>(p); }},
    {21, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 16, 128, char>(p); }},
    {22, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 16, 128, char>(p); }},
    {23, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 16, 128, char>(p); }},
    {24, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 16, 128, char>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {25, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 32, 128, char>(p); }},
    // {26, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 128, 16, 32, 128, char>(p); }},
    // {27, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 32, 128, char>(p); }},
    // {28, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 128, 16, 32, 128, char>(p); }},
    // {29, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 32, 128, char>(p); }},
    {26, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 256, 64, 16, 32, 64, char>(p); }},
    {27, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 512, 64, 16, 32, 64, char>(p); }},
    {28, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 256, 128, 16, 32, 128, char>(p); }},
    {29, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 512, 128, 16, 32, 128, char>(p); }},
    {30, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 128, 128, 16, 32, 128, char>(p); }},
    {31, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 256, 128, 16, 32, 128, char>(p); }},

    {32, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 64, 384, 16, 32, 64, char>(p); }},
    {33, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 384, 16, 32, 64, char>(p); }},
    {34, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 384, 16, 16, 64, char>(p); }},
    {35, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 384, 16, 32, 64, char>(p); }},

    {36, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 1024, 16, 16, 64, char>(p); }},
    {37, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 1024, 16, 32, 64, char>(p); }},
    {38, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
    {39, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},

    {40, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 16, 1024, 16, 16, 64, char>(p); }},
    {41, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 1024, 16, 32, 64, char>(p); }},
    {42, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
    {43, [](const GemmParams<char>& p) { launch_moe_w4a8_gemm_second_stage_decode<16, 32, 512, 16, 16, 64, char>(p); }},
};

}
}