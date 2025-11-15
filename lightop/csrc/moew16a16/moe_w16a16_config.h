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

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
// using KernelFunc = std::function<void(const GemmParams<char>&)>;
using KernelFunc4 = std::function<void(const GemmParams3<bhalf_t>&)>;



// static std::unordered_map<int, KernelFunc4> kernel_maps_gemm1_decode_marlin_w16a16 = {
//     {401, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 16, 128, 16, 16, 16, 1, bhalf_t>(p); }},
//     {402, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 256, 16, 16, 64, 1, bhalf_t>(p); }},
//     {403, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 512, 16, 16, 64, 1, bhalf_t>(p); }},
//     {404, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 256, 16, 16, 32, 1, bhalf_t>(p); }},
//     //支持EP8 TP1时N比较大时，增大N分块，之前为64，现增加到128\256
//     {413, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 256, 16, 16, 64, 1, bhalf_t>(p); }},
//     {414, [](const GemmParams3<bhalf_t>& p) { launch_MO_W16A16_MARLIN_HIP_NN_UP_decode<16, 256, 128, 16, 16, 64, 1, bhalf_t>(p); }},
//     //支持EP8 TP1时blocksizeM=32的情况：
//     {415, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<32, 128, 256, 32, 16, 64, 1, bhalf_t>(p); }},
//     {416, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<32, 256, 128, 32, 16, 64, 1, bhalf_t>(p); }},
//     {417, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<32, 64, 256, 32, 16, 64, 1, bhalf_t>(p); }},
//     {418, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<32, 64, 512, 32, 16, 64, 1, bhalf_t>(p); }},

//     {419, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<64, 64, 256, 64, 16, 64, 1, bhalf_t>(p); }},
//     {420, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<64, 128, 256, 64, 16, 64, 1, bhalf_t>(p); }},
//     {421, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<128, 64, 256, 128, 16, 64, 1, bhalf_t>(p); }},
//     {422, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<128, 128, 256, 128, 16, 64, 1, bhalf_t>(p); }},   
// };

// static std::unordered_map<int, KernelFunc4> kernel_maps_gemm2_decode_marlin_w16a16 = {
//     {401, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 64, 16, 16, 64, bhalf_t>(p); }},
//     {402, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 512, 64, 16, 16, 64, bhalf_t>(p); }},
//     {415, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<32, 256, 64, 32, 16, 64, bhalf_t>(p); }},
//     {416, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<32, 512, 64, 32, 16, 64, bhalf_t>(p); }},
//     {417, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<64, 256, 64, 64, 16, 64, bhalf_t>(p); }},
//     {418, [](const GemmParams3<bhalf_t>& p) { launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<64, 512, 64, 64, 16, 64, bhalf_t>(p); }},

// };


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


}
}