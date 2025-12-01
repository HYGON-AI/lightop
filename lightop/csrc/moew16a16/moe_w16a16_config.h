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

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill(const GemmParams3<T>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill(const GemmParams3<T>& params);

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


// static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_prefill_w16a16 = {
//     //初始基准，验证精度
//     {201, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 16, 16, 16, 16, 1, bhalf_t>(p); }},
//     //默认仅在n方向分布warp
//     {202, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 16, 16, 16, 16, 1, bhalf_t>(p); }},
//     {203, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 16, 16, 16, 16, 1, bhalf_t>(p); }},
//     {204, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 16, 16, 16, 16, 1, bhalf_t>(p); }},
//     //默认仅在k方向分布warp
//     {205, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 32, 16, 16, 16, 1, bhalf_t>(p); }},
//     {206, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 64, 16, 16, 16, 1, bhalf_t>(p); }},
//     {207, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 16, 1, bhalf_t>(p); }},
//     //同时在n和k方向分布
//     {208, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 64, 16, 16, 16, 1, bhalf_t>(p); }},
//     {209, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 32, 16, 16, 16, 1, bhalf_t>(p); }},
//     //kpack2默认
//     {210, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 32, 16, 16, 32, 1, bhalf_t>(p); }},
//     //多warp在k方向的kpack2
//     {211, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 64, 16, 16, 32, 1, bhalf_t>(p); }},
//     {212, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 32, 1, bhalf_t>(p); }},
//     {213, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 256, 16, 16, 32, 1, bhalf_t>(p); }},
//     //多warp在n方向的kpack2
//     {214, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 32, 16, 16, 32, 1, bhalf_t>(p); }},
//     {215, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 32, 16, 16, 32, 1, bhalf_t>(p); }},    
//     //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
//     {216, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 128, 16, 16, 32, 1, bhalf_t>(p); }},
//     {217, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 64, 16, 16, 32, 1, bhalf_t>(p); }},
// };


// static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_prefill_w16a16 = {
//     //初始基准，验证精度
//     {201, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 16, 16, 16, 16, bhalf_t>(p); }},
//     //默认仅在n方向分布warp
//     {202, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 16, 16, 16, 16, bhalf_t>(p); }},
//     {203, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 16, 16, 16, 16, bhalf_t>(p); }},
//     {204, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 16, 16, 16, 16, bhalf_t>(p); }},
//     //默认仅在k方向分布warp
//     {205, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 32, 16, 16, 16, bhalf_t>(p); }},
//     {206, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 16, bhalf_t>(p); }},
//     // {407, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 16, bhalf_t>(p); }}, 
//     //同时在n和k方向分布
//     {208, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 64, 16, 16, 16, bhalf_t>(p); }},
//     {209, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 32, 16, 16, 16, bhalf_t>(p); }},
//     //kpack2默认
//     {210, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 32, 16, 16, 32, bhalf_t>(p); }},
//     //多warp在k方向的kpack2
//     {211, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 32, bhalf_t>(p); }},
//     // {412, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 32, bhalf_t>(p); }},
//     // {413, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 256, 16, 16, 32, bhalf_t>(p); }},
//     //多warp在n方向的kpack2
//     {214, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 32, 16, 16, 32, bhalf_t>(p); }},
//     {215, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 32, 16, 16, 32, bhalf_t>(p); }},    
//     //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
//     // {416, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 128, 16, 16, 32, bhalf_t>(p); }},
//     // {417, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 64, 16, 16, 32, bhalf_t>(p); }},
// };


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

// marlin
static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_prefill_marlin_w16a16 = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {100, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {101, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {102, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {103, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {104, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     //{5, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {106, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
     //{107, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
     //{108, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {109, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {110, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
     // 8warp BLOCKN 256 
     {111, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 32, 64, bhalf_t>(p); }},
     //{512, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
     // 16warp
    //  {113, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 16, 64, bhalf_t>(p); }},
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {114, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {115, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {116, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {117, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  2,4,1
    //  // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {118, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {119, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
     {120, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {121, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {122, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {123, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {124, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {125, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {126, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {127, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {128, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {129, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     {130, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     // 2 warp
    //  {131, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {132, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {133, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     //{134, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 16, 128, 64, bhalf_t>(p); }}, //寄存器溢出 156
     // 4 warp
    //  {135, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {136, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     // {37, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 16, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 120
     // {38, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 16, 128, 64, bhalf_t>(p); }}, // 寄存器溢出 120
     
     ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
     // 4warp-4stages-warpk
    //  {139, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 32, bhalf_t>(p); }},
    //  {140, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 32, bhalf_t>(p); }},
 
     {141, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 32, bhalf_t>(p); }},
     {142, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 32, bhalf_t>(p); }},
 
     {143, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 32, bhalf_t>(p); }},
     {144, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 32, bhalf_t>(p); }},
     
     // 8warp-4stages-warpk
    //  {145, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 32, bhalf_t>(p); }},
     // 8warp
     {146, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 32, bhalf_t>(p); }},
 
     ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {147, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {148, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {149, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 32, bhalf_t>(p); }},
 
     ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {150, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 64, bhalf_t>(p); }},
    //  {151, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 256, 32, 16, 64, bhalf_t>(p); }},
    //  {152, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 256, 64, 16, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {153, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 256, 16, 32, 64, bhalf_t>(p); }},
     {154, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 256, 32, 32, 64, bhalf_t>(p); }},
     //{155, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {156, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 64, 64, bhalf_t>(p); }},
     //{157, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
     //{158, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {159, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 256, 32, 16, 64, bhalf_t>(p); }},
    //  {160, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 256, 64, 16, 64, bhalf_t>(p); }},
     // 8warp BLOCKN 256 
     {161, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 256, 32, 32, 64, bhalf_t>(p); }},
     //{162, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
     // 16warp
    //  {163, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 256, 64, 16, 64, bhalf_t>(p); }},
     
     ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    //  // 4warp BLOCKN 64 
    //  {164, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {165, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {166, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {167, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {168, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 256 
     {169, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
     // 8warp BLOCKN 128 
    //  {170, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {171, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {174, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {175, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {176, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {177, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {178, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {179, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
     {180, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {181, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {182, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {183, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {184, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {185, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {186, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {187, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {188, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {189, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     {190, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     // 2 warp
    //  {191, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {192, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {193, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     // 4 warp
    //  {194, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {195, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
 };
 
 // gemm2        
 static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_prefill_marlin_w16a16 = {
     ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {100, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {101, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {102, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {103, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {104, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     //{105, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {106, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 128, 16, 64, 64, bhalf_t>(p); }},
     //{107, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
     //{108, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {109, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {110, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
     // 8warp BLOCKN 256 
     {111, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 128, 32, 32, 64, bhalf_t>(p); }},
     //{12, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
     // 16warp
    //  {113, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 16, 64, bhalf_t>(p); }},
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {114, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {115, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {116, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {117, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {118, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {119, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 16, 64, bhalf_t>(p); }},
     {120, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {121, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {122, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 16, 16, 64, bhalf_t>(p); }},
     {123, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 16, 32, 64, bhalf_t>(p); }},
    //  {124, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 32, 128, 32, 16, 64, bhalf_t>(p); }},
     {125, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 64, bhalf_t>(p); }},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {126, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 16, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {127, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {128, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {129, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     {130, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     // 2 warp
    //  {131, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {132, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     {133, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 64, 64, bhalf_t>(p); }},
     //{134, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 16, 128, 64, bhalf_t>(p); }}, //寄存器溢出 156
     // 4 warp
    //  {135, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 16, 128, 16, 16, 64, bhalf_t>(p); }},
     {136, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 16, 32, 64, bhalf_t>(p); }},
     // {137, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 16, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 120
     // {138, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 16, 128, 64, bhalf_t>(p); }}, // 寄存器溢出 120
     
     ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
     // 4warp-4stages-warpk
    //  {139, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 32, bhalf_t>(p); }},
    //  {140, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 32, bhalf_t>(p); }},
 
     {141, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 32, bhalf_t>(p); }},
     {142, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 32, bhalf_t>(p); }},
 
     {143, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 32, bhalf_t>(p); }},
     {144, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 32, bhalf_t>(p); }},
     
     // 8warp-4stages-warpk
    //  {145, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 32, bhalf_t>(p); }},
     // 8warp
     {146, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 32, bhalf_t>(p); }},
     
     ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {147, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {148, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {149, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 32, bhalf_t>(p); }},
 
     ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {150, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(p); }},
    //  {151, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {152, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {153, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, bhalf_t>(p); }},
     {154, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     //{155, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {156, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 64, 64, bhalf_t>(p); }},
     //{157, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 64, 64, bhalf_t>(p); }}, // 寄存器溢出 60
     //{158, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 64, 64, bhalf_t>(p); }}, //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {159, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, bhalf_t>(p); }},
    //  {160, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
     // 8warp BLOCKN 256 
     {161, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, bhalf_t>(p); }},
     //{162, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, bhalf_t>(p); }}, //寄存器溢出28
     // 16warp
    //  {163, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, bhalf_t>(p); }},
     
     ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {164, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 16, 32, bhalf_t>(p); }},
    //  {165, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 32, 16, 32, bhalf_t>(p); }},
    //  {166, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 64, 16, 32, bhalf_t>(p); }},
     // 4warp BLOCKN 128 
     {167, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 64, 16, 32, 32, bhalf_t>(p); }},
     {168, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 32, 32, 32, bhalf_t>(p); }},
     // 4warp BLOCKN 256 
     {169, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 64, 16, 64, 32, bhalf_t>(p); }},
     // 8warp BLOCKN 128 
    //  {170, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 32, 16, 32, bhalf_t>(p); }},
    //  {171, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 64, 64, 16, 32, bhalf_t>(p); }},
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {174, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 64, 16, 16, 32, bhalf_t>(p); }},
     {175, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 32, 32, bhalf_t>(p); }},
    //  {176, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 32, 16, 32, bhalf_t>(p); }},
     {177, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {178, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 16, 32, bhalf_t>(p); }},
    //  {179, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 32, 16, 32, bhalf_t>(p); }},
     {180, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 16, 32, 32, bhalf_t>(p); }},
     {181, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 64, 32, 32, 32, bhalf_t>(p); }},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {182, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 16, 16, 32, bhalf_t>(p); }},
     {183, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 16, 32, 32, bhalf_t>(p); }},
    //  {184, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 32, 64, 32, 16, 32, bhalf_t>(p); }},
     {185, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 64, 32, 32, 32, bhalf_t>(p); }},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {186, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 32, bhalf_t>(p); }},
    //  {187, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, bhalf_t>(p); }},
     {188, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 32, 64, 16, 32, 32, bhalf_t>(p); }},
     {189, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, bhalf_t>(p); }},
     {190, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, bhalf_t>(p); }},
     // 2 warp
    //  {191, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, bhalf_t>(p); }},
     {192, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 64, 16, 32, 32, bhalf_t>(p); }},
     {193, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 64, 32, bhalf_t>(p); }},
     // 4 warp
    //  {194, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 16, 64, 16, 16, 32, bhalf_t>(p); }},
     {195, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 16, 32, 32, bhalf_t>(p); }},
 };

}
}