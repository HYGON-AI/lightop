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
struct GemmParams2 {
    GemmParams2(const T* ptr_A,
               const int32_t* ptr_B0,
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
               uint32_t delta,
               uint32_t expert_num)
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
          delta(delta),
          expert_num(expert_num)
         {}

    const T* ptr_A;                    // input
    const int32_t* ptr_B0;                   // weight
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
    uint32_t expert_num;
};

template <typename T>
struct GemmParams3 {
    GemmParams3(const T* ptr_A,
               const T* ptr_B0,
               bhalf_t* ptr_C,
               const float* topk_weights,
               const int32_t* sorted_token_ids,
               const int32_t* expert_ids,
               const int32_t num_tokens_post_pad,
               const int32_t* num_tokens_post_pad_ptr,
               uint32_t size_m,
               uint32_t size_n, 
               uint32_t size_k,
               uint32_t size_kofb,             // input size k of b
               uint32_t sorted_token_lens,
               uint32_t top_k,
               uint32_t delta,
               uint32_t expert_num)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          size_kofb(size_kofb),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta),
          expert_num(expert_num)
         {}

    const T* ptr_A;                    // input
    const T* ptr_B0;                   // weight
    bhalf_t* ptr_C;                   // output
    const float* topk_weights;         // topk weights
    const int32_t* sorted_token_ids;   // sorted token ids
    const int32_t* expert_ids;         // expert ids
    const int32_t num_tokens_post_pad;// num tokens after padding
    const int32_t* num_tokens_post_pad_ptr;// num tokens after padding
    uint32_t size_m;                   // input size m
    uint32_t size_n;                   // output size n
    uint32_t size_k;                    // input size k of a
    uint32_t size_kofb;                     // input size k of b
    uint32_t sorted_token_lens;        // sorted token length
    uint32_t top_k;                    // top k
    uint32_t delta;                    // delta
    uint32_t expert_num;
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

// template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
// void launch_MOE_WI4AI8_PERCHANNEL_HIP_NN_UP_prefill(const GemmParams<T>& params);

// template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
// void launch_MOE_WI4AI8_PERCHANNEL_HIP_NN_DOWN_prefill(const GemmParams<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void launch_MOE_W16A16_HIP_NT_UP_decode(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_DOWN_decode(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_UP_prefill(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_DOWN_prefill(const GemmParams3<T>& params);

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
// using KernelFunc = std::function<void(const GemmParams<char>&)>;
using KernelFunc4 = std::function<void(const GemmParams3<bhalf_t>&)>;



static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_decode_marlin_w16a16 = {
    {300, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    {301, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 64, 16, 16, 16, 1, bhalf_t>(p); }},
    {302, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 128, 16, 16, 16, 1, bhalf_t>(p); }},
    //kpack2
    {303, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    {304, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 128, 16, 16, 32, 1, bhalf_t>(p); }},
    {305, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 256, 16, 16, 32, 1, bhalf_t>(p); }},
    //
    {306, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    {307, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    {308, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 256, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    //KPACK2
    {309, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    {310, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    {311, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 256, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    //MIX
    {312, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 64, 16, 16, 32, 1, bhalf_t>(p); }},
    {313, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 128, 16, 16, 32, 1, bhalf_t>(p); }}, 
    {314, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 32, 16, 16, 16, 1, bhalf_t>(p); }},
    {315, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 64, 16, 16, 16, 1, bhalf_t>(p); }},   
};

static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_decode_marlin_w16a16 = {
    {300, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 16, 16, 16, 16, bhalf_t>(p); }},
    {301, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 64, 16, 16, 16, bhalf_t>(p); }},
    // {302, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 128, 16, 16, 16, bhalf_t>(p); }},
    //kpack2
    {303, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 32, 16, 16, 32, bhalf_t>(p); }},
    // {304, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 128, 16, 16, 32, bhalf_t>(p); }},
    // {305, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 256, 16, 16, 32, bhalf_t>(p); }},
    
    {306, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 16, 16, 16, 16, bhalf_t>(p); }},
    {307, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 16, 16, 16, 16, bhalf_t>(p); }},
    {308, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 16, 16, 16, 16, bhalf_t>(p); }},
   
    {309, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 32, 16, 16, 32, bhalf_t>(p); }},
    {310, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 32, 16, 16, 32, bhalf_t>(p); }},
    {311, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 32, 16, 16, 32, bhalf_t>(p); }},
    //MIX
    {312, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 64, 16, 16, 32, bhalf_t>(p); }},
    // {313, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 128, 16, 16, 32, bhalf_t>(p); }}, 
    {314, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 32, 16, 16, 16, bhalf_t>(p); }},
    {315, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 64, 16, 16, 16, bhalf_t>(p); }}, 

};


static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_decode_w16a16 = {
    //初始基准，验证精度
    {401, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    //默认仅在n方向分布warp
    {402, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    {403, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    {404, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 128, 16, 16, 16, 16, 1, bhalf_t>(p); }},
    //默认仅在k方向分布warp
    {405, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 32, 16, 16, 16, 1, bhalf_t>(p); }},
    {406, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 64, 16, 16, 16, 1, bhalf_t>(p); }},
    {407, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 128, 16, 16, 16, 1, bhalf_t>(p); }},
    //同时在n和k方向分布
    {408, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 64, 16, 16, 16, 1, bhalf_t>(p); }},
    {409, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 32, 16, 16, 16, 1, bhalf_t>(p); }},
    //kpack2默认
    {410, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    //多warp在k方向的kpack2
    {411, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 64, 16, 16, 32, 1, bhalf_t>(p); }},
    {412, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 128, 16, 16, 32, 1, bhalf_t>(p); }},
    {413, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 256, 16, 16, 32, 1, bhalf_t>(p); }},
    //多warp在n方向的kpack2
    {414, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 32, 16, 16, 32, 1, bhalf_t>(p); }},
    {415, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 128, 32, 16, 16, 32, 1, bhalf_t>(p); }},    
    //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
    {416, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 128, 16, 16, 32, 1, bhalf_t>(p); }},
    {417, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 64, 16, 16, 32, 1, bhalf_t>(p); }},
};


static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_decode_w16a16 = {
    //初始基准，验证精度
    {401, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 16, 16, 16, 16, bhalf_t>(p); }},
    //默认仅在n方向分布warp
    {402, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 16, 16, 16, 16, bhalf_t>(p); }},
    {403, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 16, 16, 16, 16, bhalf_t>(p); }},
    {404, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 128, 16, 16, 16, 16, bhalf_t>(p); }},
    //默认仅在k方向分布warp
    {405, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 32, 16, 16, 16, bhalf_t>(p); }},
    {406, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 64, 16, 16, 16, bhalf_t>(p); }},
    // {407, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 16, bhalf_t>(p); }}, 
    //同时在n和k方向分布
    {408, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 64, 16, 16, 16, bhalf_t>(p); }},
    {409, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 32, 16, 16, 16, bhalf_t>(p); }},
    //kpack2默认
    {410, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 32, 16, 16, 32, bhalf_t>(p); }},
    //多warp在k方向的kpack2
    {411, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    // {412, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 32, bhalf_t>(p); }},
    // {413, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 256, 16, 16, 32, bhalf_t>(p); }},
    //多warp在n方向的kpack2
    {414, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 32, 16, 16, 32, bhalf_t>(p); }},
    {415, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 128, 32, 16, 16, 32, bhalf_t>(p); }},    
    //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
    // {416, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 128, 16, 16, 32, bhalf_t>(p); }},
    // {417, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 64, 16, 16, 32, bhalf_t>(p); }},
};



static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_prefill_w16a16 = {
   ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {501, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {502, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {504, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    //{5, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
    //{507, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    {510, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 32, 64, bhalf_t>(p); }},
    //{512, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 16, 64, bhalf_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {515, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {516, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {517, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {519, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {520, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {521, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {523, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {524, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {525, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {527, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {528, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {529, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    {530, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    // 2 warp
    {531, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {532, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {533, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    //{534, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 128, 64, bhalf_t>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {536, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    // {37, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 120
    // {38, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 16, 128, 64, bhalf_t>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 32, bhalf_t>(p); }},
    {540, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 32, bhalf_t>(p); }},

    {541, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 32, bhalf_t>(p); }},
    {542, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 32, bhalf_t>(p); }},

    {543, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 32, bhalf_t>(p); }},
    {544, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 32, bhalf_t>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 32, bhalf_t>(p); }},
    // 8warp
    {546, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 32, bhalf_t>(p); }},

    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 32, bhalf_t>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 64, bhalf_t>(p); }},
    {551, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 256, 32, 16, 64, bhalf_t>(p); }},
    {552, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 256, 64, 16, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 256, 16, 32, 64, bhalf_t>(p); }},
    {554, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 256, 32, 32, 64, bhalf_t>(p); }},
    //{555, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 64, 64, bhalf_t>(p); }},
    //{557, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 256, 32, 16, 64, bhalf_t>(p); }},
    {560, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 256, 64, 16, 64, bhalf_t>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 256, 32, 32, 64, bhalf_t>(p); }},
    //{562, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 256, 64, 16, 64, bhalf_t>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {565, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {566, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {568, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    {571, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {575, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {576, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {577, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {579, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {580, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {581, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {583, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {584, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {585, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {587, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {588, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {589, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    {590, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    // 2 warp
    {591, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {592, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {593, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    // 4 warp
    {594, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {595, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
};

// gemm2        
static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_prefill_w16a16 = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {501, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {502, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {503, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {504, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    //{505, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
    //{507, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
    //{508, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    {510, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
    // 8warp BLOCKN 256 
    {511, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 32, 64, bhalf_t>(p); }},
    //{12, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
    // 16warp
    {513, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 16, 64, bhalf_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {515, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {516, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {517, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {519, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {520, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {521, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
    {523, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    {524, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
    {525, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {527, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {528, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {529, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    {530, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    // 2 warp
    {531, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {532, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    {533, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
    //{534, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 16, 128, 64, bhalf_t>(p); }}, //寄存器溢出 156
    // 4 warp
    {535, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    {536, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
    // {537, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 16, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 120
    // {538, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 16, 128, 64, bhalf_t>(p); }}, // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 32, bhalf_t>(p); }},
    {540, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 32, bhalf_t>(p); }},

    {541, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 32, bhalf_t>(p); }},
    {542, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 32, bhalf_t>(p); }},

    {543, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 32, bhalf_t>(p); }},
    {544, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 32, bhalf_t>(p); }},
    
    // 8warp-4stages-warpk
    {545, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 32, bhalf_t>(p); }},
    // 8warp
    {546, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 32, bhalf_t>(p); }},
    
    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 32, bhalf_t>(p); }},

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    {551, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    {552, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {553, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
    {554, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    //{555, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 64, 64, bhalf_t>(p); }},
    //{557, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
    //{558, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    {560, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
    // 8warp BLOCKN 256 
    {561, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
    //{562, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
    // 16warp
    {563, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 16, 32, bhalf_t>(p); }},
    {565, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 32, 16, 32, bhalf_t>(p); }},
    {566, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 64, 16, 32, bhalf_t>(p); }},
    // 4warp BLOCKN 128 
    {567, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 64, 16, 32, 32, bhalf_t>(p); }},
    {568, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 32, 32, 32, bhalf_t>(p); }},
    // 4warp BLOCKN 256 
    {569, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 256, 64, 16, 64, 32, bhalf_t>(p); }},
    // 8warp BLOCKN 128 
    {570, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 32, 16, 32, bhalf_t>(p); }},
    {571, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 64, 64, 16, 32, bhalf_t>(p); }},
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 64, 16, 16, 32, bhalf_t>(p); }},
    {575, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 32, 32, bhalf_t>(p); }},
    {576, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 32, 16, 32, bhalf_t>(p); }},
    {577, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 16, 32, bhalf_t>(p); }},
    {579, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 32, 16, 32, bhalf_t>(p); }},
    {580, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 16, 32, 32, bhalf_t>(p); }},
    {581, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 64, 32, 32, 32, bhalf_t>(p); }},
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 16, 16, 32, bhalf_t>(p); }},
    {583, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 16, 32, 32, bhalf_t>(p); }},
    {584, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 32, 64, 32, 16, 32, bhalf_t>(p); }},
    {585, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 64, 32, 32, 32, bhalf_t>(p); }},
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    {587, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    {588, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 64, 16, 32, 32, bhalf_t>(p); }},
    {589, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, bhalf_t>(p); }},
    {590, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, bhalf_t>(p); }},
    // 2 warp
    {591, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    {592, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 64, 16, 32, 32, bhalf_t>(p); }},
    {593, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 64, 32, bhalf_t>(p); }},
    // 4 warp
    {594, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    {595, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 16, 32, 32, bhalf_t>(p); }},
};

}
}