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
               uint32_t stride_asm,
               uint32_t stride_ask,
               uint32_t stride_bse,
               uint32_t stride_bsn, 
               uint32_t stride_bsk,
               uint32_t sorted_token_lens,
               uint32_t top_k,
               uint32_t delta,
               bool is_marlin
            )
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
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta),
          is_marlin(is_marlin)
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
    uint32_t size_k;                   // input size k
    uint32_t stride_asm;               // input scale stride m
    uint32_t stride_ask;               // input scale stride k
    uint32_t stride_bse;               // weight scale stride expert
    uint32_t stride_bsn;               // weight scale stride n
    uint32_t stride_bsk;               // weight scale stride k
    uint32_t sorted_token_lens;        // sorted token length
    uint32_t top_k;                    // top k
    uint32_t delta;                    // delta
    bool is_marlin;                    // 是否使用weight重排
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_first_stage_prefill(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_second_stage_prefill(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_first_stage_decode(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_second_stage_decode(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_fp8_first_stage_prefill(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_fp8_second_stage_prefill(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_fp8_first_stage_decode(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_fp8_second_stage_decode(const GemmParams<T>& params);

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
using KernelFunc = std::function<void(const GemmParams<char>&)>;

using KernelFuncFp8 = std::function<void(const GemmParams<uint8_t>&)>;

static std::unordered_map<int, KernelFunc> kernel_maps_gemm1_prefill = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 16, 64, char>(p); }},
    {501, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 32, 16, 64, char>(p); }},
    {502, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 128, 128, 16, 32, 64, char>(p); }},
    {504, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 32, 32, 64, char>(p); }},
    //{5, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 256, 128, 16, 64, 64, char>(p); }},
    //{507, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 32, 16, 64, char>(p); }},
    {510, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 256, 128, 32, 32, 64, char>(p); }},
    //{512, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 16, 64, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 32, 128, 16, 16, 64, char>(p); }},
    {515, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 32, 64, char>(p); }},
    {516, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 32, 16, 64, char>(p); }},
    {517, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 16, 64, char>(p); }},
    {519, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 32, 16, 64, char>(p); }},
    {520, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 16, 32, 64, char>(p); }},
    {521, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 16, 16, 64, char>(p); }},
    {523, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 16, 32, 64, char>(p); }},
    {524, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<128, 32, 128, 32, 16, 64, char>(p); }},
    {525, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<128, 64, 128, 32, 32, 64, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 16, 128, 16, 16, 64, char>(p); }},
    {527, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {528, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 32, 128, 16, 32, 64, char>(p); }},
    {529, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    {530, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    // 2 warp
    {531, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {532, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 32, 128, 16, 32, 64, char>(p); }},
    {533, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 64, 64, char>(p); }},
    //{534, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 16, 128, 64, char>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 16, 128, 16, 16, 64, char>(p); }},
    {536, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 16, 32, 64, char>(p); }},
    // {37, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 16, 64, 64, char>(p); }}, // 寄存器溢出 120
    // {38, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 16, 128, 64, char>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 32, 16, 32, char>(p); }},
    {540, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 64, 16, 32, char>(p); }},

    {541, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 32, 32, 32, char>(p); }},
    {542, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 32, 32, char>(p); }},

    {543, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 256, 128, 32, 64, 32, char>(p); }},
    {544, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 64, 32, char>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 16, 32, char>(p); }},
    // 8warp
    {546, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 32, 32, char>(p); }},

    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<128, 64, 128, 32, 32, 32, char>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 256, 16, 16, 64, char>(p); }},
    {551, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 256, 32, 16, 64, char>(p); }},
    {552, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 256, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 128, 256, 16, 32, 64, char>(p); }},
    {554, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 256, 32, 32, 64, char>(p); }},
    //{555, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 128, 128, 16, 64, 64, char>(p); }},
    //{557, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 256, 32, 16, 64, char>(p); }},
    {560, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 256, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 256, 256, 32, 32, 64, char>(p); }},
    //{562, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 256, 256, 64, 16, 64, char>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 16, 64, char>(p); }},
    {565, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 32, 16, 64, char>(p); }},
    {566, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 128, 128, 16, 32, 64, char>(p); }},
    {568, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 32, 32, 64, char>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 256, 128, 16, 64, 64, char>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 32, 16, 64, char>(p); }},
    {571, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 64, 16, 64, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 32, 128, 16, 16, 64, char>(p); }},
    {575, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 32, 64, char>(p); }},
    {576, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 32, 16, 64, char>(p); }},
    {577, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 16, 64, char>(p); }},
    {579, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 32, 16, 64, char>(p); }},
    {580, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 128, 128, 16, 32, 64, char>(p); }},
    {581, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 128, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 16, 16, 64, char>(p); }},
    {583, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 64, 128, 16, 32, 64, char>(p); }},
    {584, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<128, 32, 128, 32, 16, 64, char>(p); }},
    {585, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<128, 64, 128, 32, 32, 64, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 16, 128, 16, 16, 64, char>(p); }},
    {587, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {588, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 32, 128, 16, 32, 64, char>(p); }},
    {589, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    {590, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    // 2 warp
    {591, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {592, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 32, 128, 16, 32, 64, char>(p); }},
    {593, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<32, 64, 128, 16, 64, 64, char>(p); }},
    // 4 warp
    {594, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 16, 128, 16, 16, 64, char>(p); }},
    {595, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_prefill<64, 32, 128, 16, 32, 64, char>(p); }},
};


// gemm2        
static std::unordered_map<int, KernelFunc> kernel_maps_gemm2_prefill = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 128, 16, 16, 64, char>(p); }},
    {501, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 32, 16, 64, char>(p); }},
    {502, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 128, 128, 16, 32, 64, char>(p); }},
    {504, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 32, 64, char>(p); }},
    //{505, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 256, 128, 16, 64, 64, char>(p); }},
    //{507, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 16, 64, char>(p); }},
    {510, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 256, 128, 32, 32, 64, char>(p); }},
    //{12, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 256, 128, 64, 16, 64, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 32, 128, 16, 16, 64, char>(p); }},
    {515, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 16, 32, 64, char>(p); }},
    {516, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 128, 32, 16, 64, char>(p); }},
    {517, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 16, 16, 64, char>(p); }},
    {519, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 32, 16, 64, char>(p); }},
    {520, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 16, 32, 64, char>(p); }},
    {521, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 32, 32, 64, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 128, 16, 16, 64, char>(p); }},
    {523, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 16, 32, 64, char>(p); }},
    {524, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<128, 32, 128, 32, 16, 64, char>(p); }},
    {525, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<128, 64, 128, 32, 32, 64, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 16, 128, 16, 16, 64, char>(p); }},
    {527, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {528, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 32, 128, 16, 32, 64, char>(p); }},
    {529, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    {530, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 128, 16, 64, 64, char>(p); }},
    // 2 warp
    {531, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 16, 128, 16, 16, 64, char>(p); }},
    {532, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 32, 128, 16, 32, 64, char>(p); }},
    {533, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 16, 64, 64, char>(p); }},
    //{534, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 16, 128, 64, char>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 16, 128, 16, 16, 64, char>(p); }},
    {536, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 128, 16, 32, 64, char>(p); }},
    // {537, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 16, 64, 64, char>(p); }}, // 寄存器溢出 120
    // {538, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 16, 128, 64, char>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 32, 16, 32, char>(p); }},
    {540, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 64, 16, 32, char>(p); }},

    {541, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 32, 32, char>(p); }},
    {542, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 32, 32, char>(p); }},

    {543, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 256, 128, 32, 64, 32, char>(p); }},
    {544, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 256, 128, 64, 64, 32, char>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 16, 32, char>(p); }},
    // 8warp
    {546, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 256, 128, 64, 32, 32, char>(p); }},
    
    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 32, 32, 32, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<128, 64, 128, 32, 32, 32, char>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 128, 16, 16, 64, char>(p); }},
    {551, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 128, 32, 16, 64, char>(p); }},
    {552, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 128, 64, 16, 64, char>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 128, 128, 16, 32, 64, char>(p); }},
    {554, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 32, 64, char>(p); }},
    //{555, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 128, 128, 16, 64, 64, char>(p); }},
    //{557, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 16, 64, char>(p); }},
    {560, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 16, 64, char>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 128, 32, 32, 64, char>(p); }},
    //{562, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 128, 64, 16, 64, char>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 64, 16, 16, 32, char>(p); }},
    {565, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 64, 32, 16, 32, char>(p); }},
    {566, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 64, 64, 16, 32, char>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 128, 64, 16, 32, 32, char>(p); }},
    {568, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 64, 32, 32, 32, char>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 256, 64, 16, 64, 32, char>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 64, 32, 16, 32, char>(p); }},
    {571, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 64, 64, 16, 32, char>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 32, 64, 16, 16, 32, char>(p); }},
    {575, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 64, 16, 32, 32, char>(p); }},
    {576, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 64, 32, 16, 32, char>(p); }},
    {577, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 64, 32, 32, 32, char>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 64, 16, 16, 32, char>(p); }},
    {579, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 64, 32, 16, 32, char>(p); }},
    {580, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 128, 64, 16, 32, 32, char>(p); }},
    {581, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 128, 64, 32, 32, 32, char>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 64, 16, 16, 32, char>(p); }},
    {583, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 64, 64, 16, 32, 32, char>(p); }},
    {584, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<128, 32, 64, 32, 16, 32, char>(p); }},
    {585, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<128, 64, 64, 32, 32, 32, char>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 16, 64, 16, 16, 32, char>(p); }},
    {587, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 16, 64, 16, 16, 32, char>(p); }},
    {588, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 32, 64, 16, 32, 32, char>(p); }},
    {589, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 64, 16, 64, 32, char>(p); }},
    {590, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<16, 64, 64, 16, 64, 32, char>(p); }},
    // 2 warp
    {591, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 16, 64, 16, 16, 32, char>(p); }},
    {592, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 32, 64, 16, 32, 32, char>(p); }},
    {593, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<32, 64, 64, 16, 64, 32, char>(p); }},
    // 4 warp
    {594, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 16, 64, 16, 16, 32, char>(p); }},
    {595, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_prefill<64, 32, 64, 16, 32, 32, char>(p); }},
};

///////////////////////////////////////////////////////////////////////decode_config///////////////////////////////////////////////////////////////////////////////////////


static std::unordered_map<int, KernelFunc> kernel_maps_gemm1_decode = {
    // MN 16 16 
    {0, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 64, 16, 16,  64, 2, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 128, 16, 16, 64, 2, char>(p); }},
    {2, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 256, 16, 16, 64, 2, char>(p); }},
    {3, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 64, 2, char>(p); }},
    // MN 16 32  N方向两个warp
    {4, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 64, 16, 16,  64, 2, char>(p); }},
    {5, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 16, 64, 2, char>(p); }},
    {6, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 16, 64, 2, char>(p); }},
    {7, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},
    // MN 16 32  N方向重复算
    {8, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 64, 16, 32, 64, 2, char>(p); }},
    {9, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 32, 64, 2, char>(p); }},
    {10, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 32, 64, 2, char>(p); }},
    {11, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 32, 64, 2, char>(p); }},
    // MN 16 64 N方向两个warp+重复算两次
    {12, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 64, 16, 32, 64, 2, char>(p); }},
    {13, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 128, 16, 32, 64, 2, char>(p); }},
    {14, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16, 32, 64, 2, char>(p); }},
    {15, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16, 32, 64, 2, char>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {16, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 128, 16, 16, 128, 2,char>(p); }},
    {17, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 256, 16, 16, 128, 2,char>(p); }},
    {18, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 128, 2,char>(p); }}, //8提升明显
    {19, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 1024, 16, 16, 128,2, char>(p); }},
    // MN 16 32  N方向两个warp
    {20, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 16, 128, 2,char>(p); }},
    {21, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 16, 128, 2,char>(p); }},
    {22, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 128, 2,char>(p); }},
    {23, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 16, 128,2, char>(p); }},
    // MN 16 32  N方向重复算
    {24, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 32, 128, 2, char>(p); }},
    {25, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 32, 128, 2, char>(p); }},
    {26, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 32, 128, 2, char>(p); }},
    {27, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 32, 128,2,  char>(p); }},
    // 16 warp
    {28, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 1024, 16, 16, 64, 2, char>(p); }},
    {29, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 32, 64, 2, char>(p); }}, //8提升明显
    {30, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 16, 128,2,  char>(p); }},
    {31, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 32, 128,2,  char>(p); }},

    {32, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 384, 16, 32, 64, 2, char>(p); }},
    {33, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 384, 16, 16, 64, 2, char>(p); }},
    {34, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 384, 16, 16, 64, 2, char>(p); }},
    {35, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 384, 16, 32, 64, 2, char>(p); }},
    {36, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 1024, 16, 16, 64, 2, char>(p); }},
    {37, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 32, 64, 2, char>(p); }},
    {38, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},
    {39, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},

    {40, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},
    {41, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16, 32, 64, 2, char>(p); }},
    {42, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16, 16, 64, 2, char>(p); }},
    {43, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 128, 256, 32, 16, 64, 2, char>(p); }},

    // MN 16 64 N方向两个warp+重复算两次
    {44, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 128, 16, 32, 128, 2, char>(p); }},
    {45, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16, 32, 128, 2, char>(p); }},
    {46, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16, 32, 128, 2, char>(p); }},
    {47, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 1024, 16, 32, 128, 2, char>(p); }},

    {48, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 128, 16,  64, 128, 2, char>(p); }},
    {49, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16,  64, 128, 2, char>(p); }},
    {50, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16,  64, 128, 2, char>(p); }},
    {51, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 1024, 16, 64, 128, 2, char>(p); }},

    {52, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16, 32, 256, 2, char>(p); }},
    {53, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16, 32, 256, 2, char>(p); }},
    {54, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 1024, 16, 32, 256, 2, char>(p); }},
    {55, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 2048, 16, 32, 256, 2, char>(p); }},

    {56, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 32, 256, 2, char>(p); }}, 
    {57, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 32, 256, 2, char>(p); }},
    {58, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 1024, 16, 32, 256, 2, char>(p); }},
    {59, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 2048, 16, 32, 256, 2, char>(p); }},
    
// stages 4
    {60, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 64, 16, 16, 64, 4, char>(p); }},
    {61, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 128, 16, 16, 64, 4, char>(p); }},
    {62, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 256, 16, 16, 64, 4, char>(p); }},
    {63, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 64, 4, char>(p); }},
    // MN 16 32  N方向两个warp
    {64, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 64, 16, 16, 64, 4, char>(p); }},
    {65, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 16, 64, 4, char>(p); }},
    {66, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 16, 64, 4, char>(p); }},
    {67, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 64, 4, char>(p); }},
    // MN 16 32  N方向重复算
    {68, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 64, 16, 32, 64, 4, char>(p); }},
    {69, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 32, 64, 4, char>(p); }},
    {70, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 32, 64, 4, char>(p); }},
    {71, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 32, 64, 4, char>(p); }},
    // MN 4, 16 64 N方向两个warp+重复算两次
    {72, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 64, 16, 32, 64, 4, char>(p); }},
    {73, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 128, 16, 32, 64, 4, char>(p); }},
    {74, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 256, 16, 32, 64, 4, char>(p); }},
    {75, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 64, 512, 16, 32, 64, 4, char>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {76, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 128, 16, 16, 128, 4, char>(p); }},
    {77, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 256, 16, 16, 128, 4, char>(p); }},
    {78, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 16, 512, 16, 16, 128, 4, char>(p); }},
    // MN 16 32  N方向两个warp
    {79, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 16, 128, 4, char>(p); }},
    {80, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 16, 128, 4, char>(p); }},
    {81, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 16, 128, 4, char>(p); }},
     // MN 16 32  N方向重复算
    {82, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 128, 16, 32, 128, 4, char>(p); }},
    {83, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 256, 16, 32, 128, 4, char>(p); }},
    {84, [](const GemmParams<char>& p) { launch_moe_w8a8_first_stage_decode<16, 32, 512, 16, 32, 128, 4, char>(p); }},
    
};

static std::unordered_map<int, KernelFunc> kernel_maps_gemm2_decode = {
    // MN 16 16 
    // MN 16 16 
    {0, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 64, 16, 16, 64, 2, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 128, 16, 16, 64, 2, char>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {2, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 16, 64, 2, char>(p); }},
    {3, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 16, 64, 2, char>(p); }},
    {4, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 16, 64, 2, char>(p); }},
    {5, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 128, 16, 16, 64, 2, char>(p); }},
    {6, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 16, 64, 2, char>(p); }},
    {7, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 16, 64, 2, char>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {8, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 32, 64, 2, char>(p); }},
    {9, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 32, 64, 2, char>(p); }},
    {10, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 32, 64, 2, char>(p); }},
    {11, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 128, 16, 32, 64, 2, char>(p); }},
    {12, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2, char>(p); }},
    {13, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 32, 64, 2, char>(p); }},
    {14, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 32, 64, 2, char>(p); }},
    {15, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 32, 64, 2, char>(p); }},
    // warpk 128
    {16, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 128, 16, 16, 128, 2, char>(p); }},
    {17, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 128, 16, 16, 128, 2, char>(p); }},
    {18, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 16, 128, 2, char>(p); }},
    {19, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 16, 128, 2, char>(p); }},
    {20, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 128, 16, 16, 128, 2, char>(p); }},
    {21, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 128, 16, 16, 128, 2, char>(p); }},
    {22, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, char>(p); }},
    {23, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, char>(p); }},
    {24, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, char>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {25, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 32, 128, 2, char>(p); }},
    {26, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 32, 64, 2, char>(p); }},
    {27, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 32, 64, 2, char>(p); }},
    {28, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 32, 128, 2, char>(p); }},

    {29, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 128, 16, 32, 128, 2, char>(p); }},
    {30, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 32, 128, 2, char>(p); }},
    {31, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 32, 128, 2, char>(p); }},

    {32, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 384, 16, 32, 64, 2, char>(p); }},
    {33, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 384, 16, 32, 64, 2, char>(p); }},
    {34, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 384, 16, 16, 64, 2, char>(p); }},
    {35, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 384, 16, 32, 64, 2, char>(p); }},

    {36, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 1024, 16, 16, 64, 2, char>(p); }},
    {37, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 1024, 16, 32, 64, 2, char>(p); }},
    {38, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},
    {39, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},

    {40, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 1024, 16, 16, 64, 2, char>(p); }},
    {41, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 1024, 16, 32, 64, 2, char>(p); }},
    {42, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},
    {43, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, char>(p); }},

    {44, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 128, 16, 64, 128, 2, char>(p); }},
    {45, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 1024, 128, 16, 64, 128, 2, char>(p); }},
    {46, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 128, 16, 32, 128, 2, char>(p); }},
    {47, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 16, 128, 2, char>(p); }},

    {48, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 2, char>(p); }},
    {49, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 2, char>(p); }},
    {50, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 32, 64, 2, char>(p); }},
    {51, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 16, 64, 2, char>(p); }},

    {52, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 128, 64, 2, char>(p); }},
    {53, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 128, 64, 2, char>(p); }},
    {54, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 2, char>(p); }},
    {55, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 2, char>(p); }},

    {56, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 128, 16, 64, 128, 2, char>(p); }},
    {57, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 64, 128, 2, char>(p); }},
    {58, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 4, char>(p); }},
    {59, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 128, 64, 4, char>(p); }},

// stage4
    {60, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 16, 64, 16, 16, 64, 4, char>(p); }},
    {61, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 128, 64, 4, char>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {62, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 16, 64, 4, char>(p); }},
    {63, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 16, 64, 4, char>(p); }},
    {64, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 16, 64, 4, char>(p); }},
    {65, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 16, 64, 4, char>(p); }},
    {66, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 16, 64, 4, char>(p); }},
    {67, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 16, 64, 4, char>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {68, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, char>(p); }},
    {69, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, char>(p); }},
    {70, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 32, 64, 4, char>(p); }},
    {71, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 64, 64, 16, 32, 64, 4, char>(p); }},
    {72, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 32, 64, 4, char>(p); }},
    {73, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 32, 64, 4, char>(p); }},
    {74, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, char>(p); }},
    {75, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 128, 16, 32, 64, 4, char>(p); }},
    // warpk 128
    {76, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, char>(p); }},
    {77, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, char>(p); }},
    {78, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 32, 64, 4, char>(p); }},
    {79, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, char>(p); }},

    {80, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 4, char>(p); }},
    {81, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 64, 64, 4, char>(p); }},
    {82, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 512, 64, 16, 32, 64, 4, char>(p); }},
    {83, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 256, 64, 16, 16, 64, 4, char>(p); }},

    {84, [](const GemmParams<char>& p) { launch_moe_w8a8_second_stage_decode<16, 128, 64, 16, 128, 64, 4, char>(p); }}
};

////////////////////////////////////////////////////////////////prefill_config//////////////////////////////////////////////////////////////
static std::unordered_map<int, KernelFuncFp8> kernel_maps_gemm1_fp8_prefill = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {501, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {502, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 64, 16, 64, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {504, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 32, 32, 64, uint8_t>(p); }},
    //{5, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 32, 64, uint8_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 256, 128, 16, 64, 64, uint8_t>(p); }},
    //{507, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 256, 128, 32, 64, 64, uint8_t>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 64, 64, uint8_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 32, 16, 64, uint8_t>(p); }},
    {510, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 16, 64, uint8_t>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 256, 128, 32, 32, 64, uint8_t>(p); }},
    //{512, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 32, 64, uint8_t>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 16, 64, uint8_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {515, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {516, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {517, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {519, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {520, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {521, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {523, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {524, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<128, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {525, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<128, 64, 128, 32, 32, 64, uint8_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {527, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {528, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {529, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    {530, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    // 2 warp
    {531, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {532, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {533, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 64, 64, uint8_t>(p); }},
    //{534, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 16, 128, 64, uint8_t>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {536, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 16, 32, 64, uint8_t>(p); }},
    // {37, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 16, 64, 64, uint8_t>(p); }}, // 寄存器溢出 120
    // {38, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 16, 128, 64, uint8_t>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 32, 16, 32, uint8_t>(p); }},
    {540, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 64, 16, 32, uint8_t>(p); }},

    {541, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 32, 32, 32, uint8_t>(p); }},
    {542, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 32, 32, uint8_t>(p); }},

    {543, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 256, 128, 32, 64, 32, uint8_t>(p); }},
    {544, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 64, 32, uint8_t>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 16, 32, uint8_t>(p); }},
    // 8warp
    {546, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 32, 32, uint8_t>(p); }},

    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<128, 64, 128, 32, 32, 32, uint8_t>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 256, 16, 16, 64, uint8_t>(p); }},
    {551, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 256, 32, 16, 64, uint8_t>(p); }},
    {552, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 256, 64, 16, 64, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 128, 256, 16, 32, 64, uint8_t>(p); }},
    {554, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 256, 32, 32, 64, uint8_t>(p); }},
    //{555, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 32, 64, uint8_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 128, 128, 16, 64, 64, uint8_t>(p); }},
    //{557, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 256, 128, 32, 64, 64, uint8_t>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 64, 64, uint8_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 256, 32, 16, 64, uint8_t>(p); }},
    {560, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 256, 64, 16, 64, uint8_t>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 256, 256, 32, 32, 64, uint8_t>(p); }},
    //{562, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 128, 64, 32, 64, uint8_t>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 256, 256, 64, 16, 64, uint8_t>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {565, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {566, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 64, 16, 64, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {568, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 32, 32, 64, uint8_t>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 256, 128, 16, 64, 64, uint8_t>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 32, 16, 64, uint8_t>(p); }},
    {571, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 64, 16, 64, uint8_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {575, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {576, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {577, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {579, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {580, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {581, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 128, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {583, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {584, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<128, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {585, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<128, 64, 128, 32, 32, 64, uint8_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {587, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {588, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {589, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    {590, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    // 2 warp
    {591, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {592, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {593, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<32, 64, 128, 16, 64, 64, uint8_t>(p); }},
    // 4 warp
    {594, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {595, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_prefill<64, 32, 128, 16, 32, 64, uint8_t>(p); }},
};


// gemm2        
static std::unordered_map<int, KernelFuncFp8> kernel_maps_gemm2_fp8_prefill = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {501, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {502, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 64, 16, 64, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {504, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 32, 64, uint8_t>(p); }},
    //{505, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 32, 64, uint8_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 256, 128, 16, 64, 64, uint8_t>(p); }},
    //{507, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 256, 128, 32, 64, 64, uint8_t>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 256, 128, 64, 64, 64, uint8_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 16, 64, uint8_t>(p); }},
    {510, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 16, 64, uint8_t>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 256, 128, 32, 32, 64, uint8_t>(p); }},
    //{12, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 256, 128, 64, 32, 64, uint8_t>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 256, 128, 64, 16, 64, uint8_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {515, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {516, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {517, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {519, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {520, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {521, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 32, 32, 64, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 128, 16, 16, 64, uint8_t>(p); }},
    {523, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 16, 32, 64, uint8_t>(p); }},
    {524, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<128, 32, 128, 32, 16, 64, uint8_t>(p); }},
    {525, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<128, 64, 128, 32, 32, 64, uint8_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {527, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {528, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {529, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    {530, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 128, 16, 64, 64, uint8_t>(p); }},
    // 2 warp
    {531, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {532, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 32, 128, 16, 32, 64, uint8_t>(p); }},
    {533, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 16, 64, 64, uint8_t>(p); }},
    //{534, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 16, 128, 64, uint8_t>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 16, 128, 16, 16, 64, uint8_t>(p); }},
    {536, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 128, 16, 32, 64, uint8_t>(p); }},
    // {537, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 16, 64, 64, uint8_t>(p); }}, // 寄存器溢出 120
    // {538, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 16, 128, 64, uint8_t>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 32, 16, 32, uint8_t>(p); }},
    {540, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 64, 16, 32, uint8_t>(p); }},

    {541, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 32, 32, uint8_t>(p); }},
    {542, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 32, 32, uint8_t>(p); }},

    {543, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 256, 128, 32, 64, 32, uint8_t>(p); }},
    {544, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 256, 128, 64, 64, 32, uint8_t>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 16, 32, uint8_t>(p); }},
    // 8warp
    {546, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 256, 128, 64, 32, 32, uint8_t>(p); }},
    
    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<128, 64, 128, 32, 32, 32, uint8_t>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t>(p); }},
    {551, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 128, 32, 16, 64, uint8_t>(p); }},
    {552, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 128, 64, 16, 64, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 128, 128, 16, 32, 64, uint8_t>(p); }},
    {554, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 32, 64, uint8_t>(p); }},
    //{555, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 32, 64, uint8_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 128, 128, 16, 64, 64, uint8_t>(p); }},
    //{557, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 64, 64, uint8_t>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 64, 64, uint8_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 16, 64, uint8_t>(p); }},
    {560, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 16, 64, uint8_t>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 128, 32, 32, 64, uint8_t>(p); }},
    //{562, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 32, 64, uint8_t>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 128, 64, 16, 64, uint8_t>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 64, 16, 16, 32, uint8_t>(p); }},
    {565, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 64, 32, 16, 32, uint8_t>(p); }},
    {566, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 64, 64, 16, 32, uint8_t>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 128, 64, 16, 32, 32, uint8_t>(p); }},
    {568, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 64, 32, 32, 32, uint8_t>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 256, 64, 16, 64, 32, uint8_t>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 64, 32, 16, 32, uint8_t>(p); }},
    {571, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 64, 64, 16, 32, uint8_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 32, 64, 16, 16, 32, uint8_t>(p); }},
    {575, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 64, 16, 32, 32, uint8_t>(p); }},
    {576, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 64, 32, 16, 32, uint8_t>(p); }},
    {577, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 64, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 64, 16, 16, 32, uint8_t>(p); }},
    {579, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 64, 32, 16, 32, uint8_t>(p); }},
    {580, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 128, 64, 16, 32, 32, uint8_t>(p); }},
    {581, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 128, 64, 32, 32, 32, uint8_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 64, 16, 16, 32, uint8_t>(p); }},
    {583, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 64, 64, 16, 32, 32, uint8_t>(p); }},
    {584, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<128, 32, 64, 32, 16, 32, uint8_t>(p); }},
    {585, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<128, 64, 64, 32, 32, 32, uint8_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 16, 64, 16, 16, 32, uint8_t>(p); }},
    {587, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 16, 64, 16, 16, 32, uint8_t>(p); }},
    {588, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 32, 64, 16, 32, 32, uint8_t>(p); }},
    {589, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 64, 16, 64, 32, uint8_t>(p); }},
    {590, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<16, 64, 64, 16, 64, 32, uint8_t>(p); }},
    // 2 warp
    {591, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 16, 64, 16, 16, 32, uint8_t>(p); }},
    {592, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 32, 64, 16, 32, 32, uint8_t>(p); }},
    {593, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<32, 64, 64, 16, 64, 32, uint8_t>(p); }},
    // 4 warp
    {594, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 16, 64, 16, 16, 32, uint8_t>(p); }},
    {595, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_prefill<64, 32, 64, 16, 32, 32, uint8_t>(p); }},
};



///////////////////////////////////////////////////////////////////////decode_config_fp8///////////////////////////////////////////////////////////////////////////////////////

static std::unordered_map<int, KernelFuncFp8> kernel_maps_gemm1_fp8_decode = {
    // MN 16 16 
    {0, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 64, 16, 16,  64, 2, uint8_t>(p); }},
    {1, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 128, 16, 16, 64, 2, uint8_t>(p); }},
    {2, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 256, 16, 16, 64, 2, uint8_t>(p); }},
    {3, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 512, 16, 16, 64, 2, uint8_t>(p); }},
    // MN 16 32  N方向两个warp
    {4, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 64, 16, 16,  64, 2, uint8_t>(p); }},
    {5, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t>(p); }},
    {6, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 16, 64, 2, uint8_t>(p); }},
    {7, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},
    // MN 16 32  N方向重复算
    {8, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {9, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 32, 64, 2, uint8_t>(p); }},
    {10, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 32, 64, 2, uint8_t>(p); }},
    {11, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 32, 64, 2, uint8_t>(p); }},
    // MN 16 64 N方向两个warp+重复算两次
    {12, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {13, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 128, 16, 32, 64, 2, uint8_t>(p); }},
    {14, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16, 32, 64, 2, uint8_t>(p); }},
    {15, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16, 32, 64, 2, uint8_t>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {16, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 128, 16, 16, 128, 2,uint8_t>(p); }},
    {17, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 256, 16, 16, 128, 2,uint8_t>(p); }},
    {18, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 512, 16, 16, 128, 2,uint8_t>(p); }}, //8提升明显
    {19, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 1024, 16, 16, 128,2, uint8_t>(p); }},
    // MN 16 32  N方向两个warp
    {20, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 16, 128, 2,uint8_t>(p); }},
    {21, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 16, 128, 2,uint8_t>(p); }},
    {22, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 128, 2,uint8_t>(p); }},
    {23, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 16, 128,2, uint8_t>(p); }},
    // MN 16 32  N方向重复算
    {24, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {25, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 32, 128, 2, uint8_t>(p); }},
    {26, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 32, 128, 2, uint8_t>(p); }},
    {27, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 32, 128,2,  uint8_t>(p); }},
    // 16 warp
    {28, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 1024, 16, 16, 64, 2, uint8_t>(p); }},
    {29, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 32, 64, 2, uint8_t>(p); }}, //8提升明显
    {30, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 16, 128,2,  uint8_t>(p); }},
    {31, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 32, 128,2,  uint8_t>(p); }},

    {32, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 384, 16, 32, 64, 2, uint8_t>(p); }},
    {33, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 384, 16, 16, 64, 2, uint8_t>(p); }},
    {34, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 384, 16, 16, 64, 2, uint8_t>(p); }},
    {35, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 384, 16, 32, 64, 2, uint8_t>(p); }},
    {36, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 1024, 16, 16, 64, 2, uint8_t>(p); }},
    {37, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 32, 64, 2, uint8_t>(p); }},
    {38, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},
    {39, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},

    {40, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},
    {41, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16, 32, 64, 2, uint8_t>(p); }},
    {42, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16, 16, 64, 2, uint8_t>(p); }},
    {43, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 128, 256, 32, 16, 64, 2, uint8_t>(p); }},

    // MN 16 64 N方向两个warp+重复算两次
    {44, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {45, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16, 32, 128, 2, uint8_t>(p); }},
    {46, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16, 32, 128, 2, uint8_t>(p); }},
    {47, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 1024, 16, 32, 128, 2, uint8_t>(p); }},

    {48, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 128, 16,  64, 128, 2, uint8_t>(p); }},
    {49, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16,  64, 128, 2, uint8_t>(p); }},
    {50, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16,  64, 128, 2, uint8_t>(p); }},
    {51, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 1024, 16, 64, 128, 2, uint8_t>(p); }},

    {52, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16, 32, 256, 2, uint8_t>(p); }},
    {53, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16, 32, 256, 2, uint8_t>(p); }},
    {54, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 1024, 16, 32, 256, 2, uint8_t>(p); }},
    {55, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 2048, 16, 32, 256, 2, uint8_t>(p); }},

    {56, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 32, 256, 2, uint8_t>(p); }}, 
    {57, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 32, 256, 2, uint8_t>(p); }},
    {58, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 1024, 16, 32, 256, 2, uint8_t>(p); }},
    {59, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 2048, 16, 32, 256, 2, uint8_t>(p); }},
    
// stages 4
    {60, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {61, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 128, 16, 16, 64, 4, uint8_t>(p); }},
    {62, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 256, 16, 16, 64, 4, uint8_t>(p); }},
    {63, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 512, 16, 16, 64, 4, uint8_t>(p); }},
    // MN 16 32  N方向两个warp
    {64, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {65, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 16, 64, 4, uint8_t>(p); }},
    {66, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 16, 64, 4, uint8_t>(p); }},
    {67, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 64, 4, uint8_t>(p); }},
    // MN 16 32  N方向重复算
    {68, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {69, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 32, 64, 4, uint8_t>(p); }},
    {70, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 32, 64, 4, uint8_t>(p); }},
    {71, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 32, 64, 4, uint8_t>(p); }},
    // MN 4, 16 64 N方向两个warp+重复算两次
    {72, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {73, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 128, 16, 32, 64, 4, uint8_t>(p); }},
    {74, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 256, 16, 32, 64, 4, uint8_t>(p); }},
    {75, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 64, 512, 16, 32, 64, 4, uint8_t>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {76, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 128, 16, 16, 128, 4, uint8_t>(p); }},
    {77, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 256, 16, 16, 128, 4, uint8_t>(p); }},
    {78, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 16, 512, 16, 16, 128, 4, uint8_t>(p); }},
    // MN 16 32  N方向两个warp
    {79, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 16, 128, 4, uint8_t>(p); }},
    {80, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 16, 128, 4, uint8_t>(p); }},
    {81, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 16, 128, 4, uint8_t>(p); }},
     // MN 16 32  N方向重复算
    {82, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 128, 16, 32, 128, 4, uint8_t>(p); }},
    {83, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 256, 16, 32, 128, 4, uint8_t>(p); }},
    {84, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_first_stage_decode<16, 32, 512, 16, 32, 128, 4, uint8_t>(p); }},
    
};

static std::unordered_map<int, KernelFuncFp8> kernel_maps_gemm2_fp8_decode = {
    // MN 16 16 
    // MN 16 16 
    {0, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 64, 16, 16, 64, 2, uint8_t>(p); }},
    {1, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 128, 16, 16, 64, 2, uint8_t>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {2, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 16, 64, 2, uint8_t>(p); }},
    {3, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t>(p); }},
    {4, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 16, 64, 2, uint8_t>(p); }},
    {5, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 128, 16, 16, 64, 2, uint8_t>(p); }},
    {6, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 16, 64, 2, uint8_t>(p); }},
    {7, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 16, 64, 2, uint8_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {8, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {9, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 128, 16, 32, 64, 2, uint8_t>(p); }},
    {10, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {11, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 128, 16, 32, 64, 2, uint8_t>(p); }},
    {12, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {13, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 32, 64, 2, uint8_t>(p); }},
    {14, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {15, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 32, 64, 2, uint8_t>(p); }},
    // warpk 128
    {16, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {17, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {18, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {19, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {20, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {21, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {22, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {23, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, uint8_t>(p); }},
    {24, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 16, 128, 2, uint8_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {25, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {26, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {27, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {28, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 32, 128, 2, uint8_t>(p); }},

    {29, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {30, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {31, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 32, 128, 2, uint8_t>(p); }},

    {32, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 384, 16, 32, 64, 2, uint8_t>(p); }},
    {33, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 384, 16, 32, 64, 2, uint8_t>(p); }},
    {34, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 384, 16, 16, 64, 2, uint8_t>(p); }},
    {35, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 384, 16, 32, 64, 2, uint8_t>(p); }},

    {36, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 1024, 16, 16, 64, 2, uint8_t>(p); }},
    {37, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 1024, 16, 32, 64, 2, uint8_t>(p); }},
    {38, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},
    {39, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},

    {40, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 1024, 16, 16, 64, 2, uint8_t>(p); }},
    {41, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 1024, 16, 32, 64, 2, uint8_t>(p); }},
    {42, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},
    {43, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 512, 16, 16, 64, 2, uint8_t>(p); }},

    {44, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 128, 16, 64, 128, 2, uint8_t>(p); }},
    {45, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 1024, 128, 16, 64, 128, 2, uint8_t>(p); }},
    {46, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 128, 16, 32, 128, 2, uint8_t>(p); }},
    {47, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 16, 128, 2, uint8_t>(p); }},

    {48, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 64, 64, 2, uint8_t>(p); }},
    {49, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 64, 64, 2, uint8_t>(p); }},
    {50, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 32, 64, 2, uint8_t>(p); }},
    {51, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 16, 64, 2, uint8_t>(p); }},

    {52, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 128, 64, 2, uint8_t>(p); }},
    {53, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 128, 64, 2, uint8_t>(p); }},
    {54, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 128, 64, 2, uint8_t>(p); }},
    {55, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 128, 64, 2, uint8_t>(p); }},

    {56, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 128, 16, 64, 128, 2, uint8_t>(p); }},
    {57, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 64, 128, 2, uint8_t>(p); }},
    {58, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 128, 64, 4, uint8_t>(p); }},
    {59, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 128, 64, 4, uint8_t>(p); }},

// stage4
    {60, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 16, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {61, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 128, 64, 4, uint8_t>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {62, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {63, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {64, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {65, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {66, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 16, 64, 4, uint8_t>(p); }},
    {67, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 16, 64, 4, uint8_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {68, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {69, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {70, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {71, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 64, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {72, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {73, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {74, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {75, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 128, 16, 32, 64, 4, uint8_t>(p); }},
    // warpk 128
    {76, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 32, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {77, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {78, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {79, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 32, 64, 4, uint8_t>(p); }},

    {80, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 64, 64, 4, uint8_t>(p); }},
    {81, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 64, 64, 4, uint8_t>(p); }},
    {82, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 512, 64, 16, 32, 64, 4, uint8_t>(p); }},
    {83, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 256, 64, 16, 16, 64, 4, uint8_t>(p); }},

    {84, [](const GemmParams<uint8_t>& p) { launch_moe_w8a8_fp8_second_stage_decode<16, 128, 64, 16, 128, 64, 4, uint8_t>(p); }}
};

}
}