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
struct GemmParams3 {
    GemmParams3(const T* ptr_A,
               const T* ptr_B0,
               T* ptr_C,
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
    T* ptr_C;                   // output
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

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode(const GemmParams3<T>& params){
    launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,splitk, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode(const GemmParams3<T>& params){
    launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void kernel_launch_MOE_W16A16_HIP_NT_UP_decode(const GemmParams3<T>& params){
    launch_MOE_W16A16_HIP_NT_UP_decode<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,splitk, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode(const GemmParams3<T>& params){
    launch_MOE_W16A16_HIP_NT_DOWN_decode<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_HIP_NT_UP_prefill(const GemmParams3<T>& params){
    launch_MOE_W16A16_HIP_NT_UP_prefill<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill(const GemmParams3<T>& params){
    launch_MOE_W16A16_HIP_NT_DOWN_prefill<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill(const GemmParams3<T>& params){
    launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill(const GemmParams3<T>& params){
    launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
}

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
// using KernelFunc = std::function<void(const GemmParams<char>&)>;
template<typename T>
using KernelFunc4 = std::function<void(const GemmParams3<T>&)>;


template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_decode_marlin_w16a16(){
    return{
        {300, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 16, 16, 16, 16, 1, T> },
        {301, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 64, 16, 16, 16, 1, T>},
        {302, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 128, 16, 16, 16, 1, T>},
        //kpack2
        {303, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 32, 16, 16, 32, 1, T>},
        {304, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 128, 16, 16, 32, 1, T>},
        {305, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 32, 256, 16, 16, 32, 1, T>},
        //
        {306, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 16, 16, 16, 16, 1, T>},
        {307, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 16, 16, 16, 16, 1, T>},
        {308, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 256, 16, 16, 16, 16, 1, T>},
        //KPACK2
        {309, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 32, 16, 16, 32, 1, T>},
        {310, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 32, 16, 16, 32, 1, T>},
        {311, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 256, 32, 16, 16, 32, 1, T>},
        //MIX
        {312, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 64, 16, 16, 32, 1, T>},
        {313, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 128, 16, 16, 32, 1, T>},
        {314, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 128, 32, 16, 16, 16, 1, T>},
        {315, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 64, 16, 16, 16, 1, T>}, 
    };
}

static auto kernel_maps_gemm1_decode_marlin_w16a16_half = make_kernel_maps_gemm1_decode_marlin_w16a16<half>();
static auto kernel_maps_gemm1_decode_marlin_w16a16_bhalf_t = make_kernel_maps_gemm1_decode_marlin_w16a16<bhalf_t>();

template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_decode_marlin_w16a16(){
    return{
    {300, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 16, 16, 16, 16, T>}, 
    {301, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 64, 16, 16, 16, T>}, 
    {302, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 128, 16, 16, 16, T>}, 
    //kpack2
    {303, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 32, 16, 16, 32, T>}, 
    {304, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 128, 16, 16, 32, T>}, 
    {305, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 32, 256, 16, 16, 32, T>}, 
    {306, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 16, 16, 16, 16, T>}, 
    {307, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 16, 16, 16, 16, T>}, 
    {308, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 16, 16, 16, 16, T>}, 
   
    {309, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 32, 16, 16, 32, T>}, 
    {310, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 32, 16, 16, 32, T>}, 
    {311, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 32, 16, 16, 32, T>}, 
    //MIX
    {312, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 64, 16, 16, 32, T>}, 
    {313, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 128, 16, 16, 32, T>}, 
    {314, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 128, 32, 16, 16, 16, T>}, 
    {315, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 64, 64, 16, 16, 16, T>},  

};
}

static auto kernel_maps_gemm2_decode_marlin_w16a16_half = make_kernel_maps_gemm2_decode_marlin_w16a16<half>();
static auto kernel_maps_gemm2_decode_marlin_w16a16_bhalf_t = make_kernel_maps_gemm2_decode_marlin_w16a16<bhalf_t>();
template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_decode_w16a16(){
    return{
    //初始基准，验证精度
    {401, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 16, 16, 16, 16, 1, T>},  
    //默认仅在n方向分布warp
    {402, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 16, 16, 16, 16, 1, T>},  
    {403, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 16, 16, 16, 16, 1, T>},  
    {404, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 128, 16, 16, 16, 16, 1, T>},  
    //默认仅在k方向分布warp
    {405, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 32, 16, 16, 16, 1, T>},  
    {406, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 64, 16, 16, 16, 1, T>},  
    {407, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 128, 16, 16, 16, 1, T>},  
    //同时在n和k方向分布
    {408, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 64, 16, 16, 16, 1, T>},  
    {409, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 32, 16, 16, 16, 1, T>},  
    //kpack2默认
    {410, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 32, 16, 16, 32, 1, T>},  
    //多warp在k方向的kpack2
    {411, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 64, 16, 16, 32, 1, T>},  
    {412, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 128, 16, 16, 32, 1, T>},  
    {413, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 256, 16, 16, 32, 1, T>},  
    //多warp在n方向的kpack2
    {414, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 32, 16, 16, 32, 1, T>},  
    {415, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 128, 32, 16, 16, 32, 1, T>},  
    //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
    {416, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 32, 128, 16, 16, 32, 1, T>},  
    {417, &kernel_launch_MOE_W16A16_HIP_NT_UP_decode<16, 64, 64, 16, 16, 32, 1, T>},  
};
}

static auto kernel_maps_gemm1_decode_w16a16_half = make_kernel_maps_gemm1_decode_w16a16<half>();
static auto kernel_maps_gemm1_decode_w16a16_bhalf_t = make_kernel_maps_gemm1_decode_w16a16<bhalf_t>();
template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_decode_w16a16(){
    return{
    //初始基准，验证精度
    {401, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 16, 16, 16, 16, T>},  
    //默认仅在n方向分布warp
    {402, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 16, 16, 16, 16, T>},  
    {403, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 16, 16, 16, 16, T>},  
    {404, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 128, 16, 16, 16, 16, T>},  
    //默认仅在k方向分布warp
    {405, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 32, 16, 16, 16, T>},  
    {406, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 64, 16, 16, 16, T>},  
    // {407, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 16, T>},   
    //同时在n和k方向分布
    {408, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 64, 16, 16, 16, T>},  
    {409, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 32, 16, 16, 16, T>},  
    //kpack2默认
    {410, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 32, 16, 16, 32, T>},  
    //多warp在k方向的kpack2
    {411, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 64, 16, 16, 32, T>},  
    // {412, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 128, 16, 16, 32, T>},  
    // {413, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 256, 16, 16, 32, T>},  
    //多warp在n方向的kpack2
    {414, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 32, 16, 16, 32, T>},  
    {415, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 128, 32, 16, 16, 32, T>},      
    //FINAL 多warp在k方向的kpack2 叠加n方向分布warp
    // {416, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 32, 128, 16, 16, 32, T>},  
    // {417, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 64, 64, 16, 16, 32, T>},  
};
}

static auto kernel_maps_gemm2_decode_w16a16_half = make_kernel_maps_gemm2_decode_w16a16<half>();
static auto kernel_maps_gemm2_decode_w16a16_bhalf_t = make_kernel_maps_gemm2_decode_w16a16<bhalf_t>();
template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_prefill_w16a16(){
   ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
   return{ 
   // 4warp BLOCKN 64 
    {500, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, T>},  
    {501, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, T>},  
    {502, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, T>},  
    // 4warp BLOCKN 128 
    {503, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},  
    {504, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, T>},  
    //{5, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, T>},  
    //{507, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, T>},   // 寄存器溢出 60
    //{508, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, T>},   //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, T>},  
    {510, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, T>},  
    // 8warp BLOCKN 256 
    {511, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 32, 64, T>},  
    //{512, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
    // 16warp
    {513, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 16, 64, T>},  
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, T>},  
    {515, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, T>},  
    {516, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, T>},  
    {517, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, T>},  
    {519, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, T>},  
    {520, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, T>},  
    {521, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, T>},  
    {523, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, T>},  
    {524, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, T>},  
    {525, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, T>},  
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, T>},  
    {527, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
    {528, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, T>},  
    {529, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},  
    {530, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},  
    // 2 warp
    {531, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
    {532, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, T>},  
    {533, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, T>},  
    //{534, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 128, 64, T>},   //寄存器溢出 156
    // 4 warp
    {535, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, T>},  
    {536, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, T>},  
    // {37, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 64, 64, T>},   // 寄存器溢出 120
    // {38, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 16, 128, 64, T>},   // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 32, T>},  
    {540, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 32, T>},  

    {541, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 32, T>},  
    {542, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 32, T>},  

    {543, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 32, T>},  
    {544, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 32, T>},  
    
    // 8warp-4stages-warpk
    {545, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 32, T>},  
    // 8warp
    {546, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 32, T>},  

    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 32, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 32, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 32, T>},  

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 64, T>},  
    {551, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 256, 32, 16, 64, T>},  
    {552, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 256, 64, 16, 64, T>},  
    // 4warp BLOCKN 128 
    {553, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 256, 16, 32, 64, T>},  
    {554, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 256, 32, 32, 64, T>},  
    //{555, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 64, 64, T>},  
    //{557, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, T>},   // 寄存器溢出 60
    //{558, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, T>},   //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 256, 32, 16, 64, T>},  
    {560, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 256, 64, 16, 64, T>},  
    // 8warp BLOCKN 256 
    {561, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 256, 256, 32, 32, 64, T>},  
    //{562, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
    // 16warp
    {563, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 256, 256, 64, 16, 64, T>},  
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, T>},  
    {565, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, T>},  
    {566, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, T>},  
    // 4warp BLOCKN 128 
    {567, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},  
    {568, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, T>},  
    // 4warp BLOCKN 256 
    {569, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, T>},  
    // 8warp BLOCKN 128 
    {570, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, T>},  
    {571, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, T>},  
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, T>},  
    {575, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, T>},  
    {576, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, T>},  
    {577, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, T>},  
    {579, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, T>},  
    {580, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, T>},  
    {581, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, T>},  
    {583, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, T>},  
    {584, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, T>},  
    {585, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, T>},  
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, T>},  
    {587, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
    {588, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, T>},  
    {589, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},  
    {590, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},  
    // 2 warp
    {591, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
    {592, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, T>},  
    {593, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, T>},  
    // 4 warp
    {594, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, T>},  
    {595, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, T>},  
    //kpack2 stage1
    // {501, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 128, T>},  
    // {502, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 128, T>},  
    // {503, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 128, 16, 16, 128, T>},  
    //kpack2 STAGE2
    {596, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 128, T>},  
    // {505, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 256, 16, 16, 128, T>},  //out of memory
    {597, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 128, T>},  
    // {507, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 256, 16, 16, 128, T>},  //out of memory
    {598, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 64, 256, 32, 16, 128, T>},  
    // {509, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<32, 128, 256, 32, 16, 128, T>},  //out of memory
    {599, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 64, 256, 64, 16, 128, T>},  
    // {511, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<64, 128, 256, 64, 16, 128, T>},  //out of memory
    // {512, &kernel_launch_MOE_W16A16_HIP_NT_UP_prefill<16, 128, 256, 16, 32, 128, T>},  //out of memory 
    };
}

static auto kernel_maps_gemm1_prefill_w16a16_half = make_kernel_maps_gemm1_prefill_w16a16<half>();
static auto kernel_maps_gemm1_prefill_w16a16_bhalf_t = make_kernel_maps_gemm1_prefill_w16a16<bhalf_t>();

// gemm2 
template <typename T>       
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_prefill_w16a16(){
    return{
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {500, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, T>},  
    {501, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, T>},  
    {502, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, T>},  
    // 4warp BLOCKN 128 
    {503, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, T>},  
    {504, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},  
    //{505, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {506, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 256, 128, 16, 64, 64, T>},  
    //{507, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 64, T>},   // 寄存器溢出 60
    //{508, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 64, T>},   //寄存器溢出 392
    // 8warp BLOCKN 128 
    {509, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, T>},  
    {510, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
    // 8warp BLOCKN 256 
    {511, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 32, 64, T>},  
    //{12, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
    // 16warp
    {513, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 16, 64, T>},  
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {514, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 128, 16, 16, 64, T>},  
    {515, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 32, 64, T>},  
    {516, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 32, 16, 64, T>},  
    {517, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 64, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {518, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 16, 64, T>},  
    {519, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 16, 64, T>},  
    {520, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 16, 32, 64, T>},  
    {521, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 64, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {522, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 16, 16, 64, T>},  
    {523, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 16, 32, 64, T>},  
    {524, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 32, 128, 32, 16, 64, T>},  
    {525, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 64, T>},  
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {526, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 128, 16, 16, 64, T>},  
    {527, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, T>},  
    {528, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 128, 16, 32, 64, T>},  
    {529, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, T>},  
    {530, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, T>},  
    // 2 warp
    {531, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, T>},  
    {532, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 128, 16, 32, 64, T>},  
    {533, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 16, 64, 64, T>},  
    //{534, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 16, 128, 64, T>},   //寄存器溢出 156
    // 4 warp
    {535, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 16, 128, 16, 16, 64, T>},  
    {536, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 128, 16, 32, 64, T>},  
    // {537, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 16, 64, 64, T>},   // 寄存器溢出 120
    // {538, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 16, 128, 64, T>},   // 寄存器溢出 120
    
    ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
    // 4warp-4stages-warpk
    {539, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 32, T>},  
    {540, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 32, T>},  

    {541, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 32, T>},  
    {542, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 32, T>},  

    {543, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 32, T>},  
    {544, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 32, T>},  
    
    // 8warp-4stages-warpk
    {545, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 32, T>},  
    // 8warp
    {546, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 32, T>},  
    
    ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {547, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 32, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {548, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 32, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {549, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 32, T>},  

    ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {550, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, T>},  
    {551, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, T>},  
    {552, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, T>},  
    // 4warp BLOCKN 128 
    {553, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, T>},  
    {554, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},  
    //{555, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
    // 4warp BLOCKN 256 
    {556, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 64, 64, T>},  
    //{557, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 64, 64, T>},   // 寄存器溢出 60
    //{558, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 64, 64, T>},   //寄存器溢出 392
    // 8warp BLOCKN 128 
    {559, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, T>},  
    {560, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
    // 8warp BLOCKN 256 
    {561, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},  
    //{562, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   //寄存器溢出28
    // 16warp
    {563, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
    
    ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    // 4warp BLOCKN 64 
    {564, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 16, 32, T>},  
    {565, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 32, 16, 32, T>},  
    {566, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 64, 16, 32, T>},  
    // 4warp BLOCKN 128 
    {567, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 64, 16, 32, 32, T>},  
    {568, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 32, 32, 32, T>},  
    // 4warp BLOCKN 256 
    {569, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 256, 64, 16, 64, 32, T>},  
    // 8warp BLOCKN 128 
    {570, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 32, 16, 32, T>},  
    {571, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 64, 64, 16, 32, T>},  
    ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
    // 4 warp:  2,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {574, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 64, 16, 16, 32, T>},  
    {575, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 32, 32, T>},  
    {576, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 32, 16, 32, T>},  
    {577, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 32, 32, 32, T>},  
    // 8 warp:  2,4,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {578, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 16, 32, T>},  
    {579, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 32, 16, 32, T>},  
    {580, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 64, 16, 32, 32, T>},  
    {581, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 64, 32, 32, 32, T>},  
    // 8 warp:  4,2,1
    // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    {582, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 16, 16, 32, T>},  
    {583, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 64, 16, 32, 32, T>},  
    {584, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 32, 64, 32, 16, 32, T>},  
    {585, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<128, 64, 64, 32, 32, 32, T>},  
    /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
    // 1 warp
    {586, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 32, T>},  
    {587, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, T>},  
    {588, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 32, 64, 16, 32, 32, T>},  
    {589, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, T>},  
    {590, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, T>},  
    // 2 warp
    {591, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, T>},  
    {592, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 32, 64, 16, 32, 32, T>},  
    {593, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 64, 16, 64, 32, T>},  
    // 4 warp
    {594, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 16, 64, 16, 16, 32, T>},  
    {595, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 32, 64, 16, 32, 32, T>},  
    //kpack2
    {596, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 16, 128, 16, 16, 128, T>},  
    {597, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 128, T>},  
    {698, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 16, 128, 32, 16, 128, T>},  
    {599, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 16, 16, 128, T>},  
    {600, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 16, 128, T>},  
    {601, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 128, T>},  
    {602, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 128, T>},  
    {603, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 128, T>},  
    {604, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 16, 128, 16, 16, 128, T>},  
    {605, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 128, T>},  
    {606, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 16, 32, 128, T>},  
    {607, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 128, T>},      
    {608, &kernel_launch_MOE_W16A16_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 128, T>},  
    };
}

static auto kernel_maps_gemm2_prefill_w16a16_half = make_kernel_maps_gemm2_prefill_w16a16<half>();
static auto kernel_maps_gemm2_prefill_w16a16_bhalf_t = make_kernel_maps_gemm2_prefill_w16a16<bhalf_t>();

// marlin
template <typename T>
static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_prefill_marlin_w16a16(){
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    return {
    // 4warp BLOCKN 64 
     {100, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     {101, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     {102, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     {107, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     {108, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 64, 32, 32, 32, T>},
     {109, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 64, 32, 32, 32, T>}, 
     {110, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 64, 32, 32, 32, T>},
     {112, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 64, 32, 32, 32, T>},
     // 4warp BLOCKN 128 
     {103, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     {104, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 64, T>},
     {105, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 32, 64, T>},
     //{5, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, T>}, // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {106, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 256, 128, 16, 64, 64, T>},
     {113, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 64, 64, 64, 32, T>},   
     {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 64, 64, 64, 32, T>},   
     // 8warp BLOCKN 128 
    //  {109, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, T>},  
    //  {110, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, T>},  
     // 8warp BLOCKN 256 
     {111, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 64, 32, 32, 32, T>},
     //{512, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
     // 16warp
    //  {113, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 16, 64, T>},  
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, T>},  
     {115, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 32, 64, T>},
    //  {116, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, T>},  
     {117, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 64, T>},
     // 8 warp:  2,4,1
    //  // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {118, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, T>},  
    //  {119, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, T>},  
     {120, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 16, 32, 64, T>},
     {121, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 64, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {122, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, T>},  
     {123, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 16, 32, 64, T>},
    //  {124, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, T>},  
     {125, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 64, T>},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {126, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, T>},  
    //  {127, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
     {128, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 32, 128, 16, 32, 64, T>},
     {129, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},
     {130, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 64, 64, T>},
     // 2 warp
    //  {131, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
     {132, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 32, 64, T>},
     {133, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 64, 64, T>},
     //{134, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 16, 128, 64, T>},   //寄存器溢出 156
     // 4 warp
    //  {135, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, T>},  
     {136, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 32, 64, T>},
     // {37, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 16, 64, 64, T>},   // 寄存器溢出 120
     // {38, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 16, 128, 64, T>},   // 寄存器溢出 120
     
     ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
     // 4warp-4stages-warpk
    //  {139, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 32, T>},  
    //  {140, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 32, T>},  
 
     {141, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 32, 32, T>},
     {142, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 32, T>},
 
     {143, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 32, T>},
     {144, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 32, T>},
     
     // 8warp-4stages-warpk
    //  {145, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 32, T>},  
     // 8warp
     {146, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 32, T>},
 
     ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {147, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 32, 32, T>},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {148, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 32, 32, 32, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {149, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 128, 32, 32, 32, T>},
 
     ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {150, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 256, 16, 16, 64, T>},  
    //  {151, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 256, 32, 16, 64, T>},  
    //  {152, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 256, 64, 16, 64, T>},  
     // 4warp BLOCKN 128 
     {153, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 256, 16, 32, 64, T>},
     {154, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 256, 32, 32, 64, T>},
     //{155, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {156, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 128, 16, 64, 64, T>},
     //{157, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 128, 32, 64, 64, T>},   // 寄存器溢出 60
     //{158, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 64, 64, T>},   //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {159, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 256, 32, 16, 64, T>},  
    //  {160, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 256, 64, 16, 64, T>},  
     // 8warp BLOCKN 256 
     {161, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 256, 256, 32, 32, 64, T>},
     //{162, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
     // 16warp
    //  {163, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 256, 256, 64, 16, 64, T>},  
     
     ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
    //  // 4warp BLOCKN 64 
    //  {164, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, T>},  
    //  {165, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 32, 16, 64, T>},  
    //  {166, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 64, 16, 64, T>},  
     // 4warp BLOCKN 128 
     {167, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 128, 64, 16, 32, 32, T>},
     {168, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 64, 32, 32, 32, T>},
     // 4warp BLOCKN 256 
     {169, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 256, 64, 16, 64, 32, T>},
     // 8warp BLOCKN 128 
    //  {170, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 128, 32, 16, 64, T>},  
    //  {171, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 128, 64, 16, 64, T>},  
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {174, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 128, 16, 16, 64, T>},  
     {175, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 64, 16, 32, 32, T>},
    //  {176, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 32, 16, 64, T>},  
     {177, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 64, 32, 32, 32, T>},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {178, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 128, 16, 16, 64, T>},  
    //  {179, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 128, 32, 16, 64, T>},  
     {180, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 128, 64, 16, 32, 32, T>},
     {181, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 128, 64, 32, 32, 32, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {182, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 128, 16, 16, 64, T>},  
     {183, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 64, 64, 16, 32, 32, T>},
    //  {184, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 32, 128, 32, 16, 64, T>},  
     {185, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<128, 64, 64, 32, 32, 32, T>},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {186, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 16, 128, 16, 16, 64, T>},  
    //  {187, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
     {188, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 32, 64, 16, 32, 32, T>},
     {189, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 64, 16, 64, 32, T>},
     {190, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 64, 16, 64, 32, T>},
     // 2 warp
    //  {191, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 16, 128, 16, 16, 64, T>},  
     {192, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 32, 64, 16, 32, 32, T>},
     {193, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<32, 64, 64, 16, 64, 32, T>},
     // 4 warp
    //  {194, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 16, 128, 16, 16, 64, T>},  
     {195, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<64, 32, 64, 16, 32, 32, T>},

    };
}

static auto kernel_maps_gemm1_prefill_marlin_w16a16_half = make_kernel_maps_gemm1_prefill_marlin_w16a16<half>();
static auto kernel_maps_gemm1_prefill_marlin_w16a16_bhalf_t = make_kernel_maps_gemm1_prefill_marlin_w16a16<bhalf_t>();

template <typename T>
 static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_prefill_marlin_w16a16(){
    return{
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
     // 4warp BLOCKN 64 
     {100, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 16, 16, 32, 16, T>},
     {101, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 32, 16, 32, 32, T>},
     {102, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 16, 16, 32, 16, T>},
     {107, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 32, 16, 32, 32, T>},
     {108, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 32, 32, 32, 32, T>},
     {109, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 32, 64, 32, 32, T>}, 
     {110, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 32, 32, 32, 32, T>},
     {112, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 32, 64, 32, 32, T>},
     // 4warp BLOCKN 128 
     {103, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, T>},
     {104, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},
     {105, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 16, 16, 32, 16, T>},
     //{105, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {106, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 128, 16, 64, 64, T>},
     {113, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 64, 64, 64, 32, T>},   
     {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 32, 64, 64, 32, T>},   
     //{108, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 64, T>},   //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {109, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, T>},  
    //  {110, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
     // 8warp BLOCKN 256 
     {111, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 128, 32, 32, 64, T>},
     //{12, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 64, T>},   //寄存器溢出28
     // 16warp
    //  {113, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 16, 64, T>},  
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 128, 16, 16, 64, T>},  
     {115, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 32, 64, T>},
    //  {116, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 32, 16, 64, T>},  
     {117, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 64, T>},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {118, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 16, 64, T>},  
    //  {119, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 16, 64, T>},  
     {120, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 16, 32, 64, T>},
     {121, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 64, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {122, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 16, 16, 64, T>},  
     {123, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 16, 32, 64, T>},
    //  {124, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 32, 128, 32, 16, 64, T>},  
     {125, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 64, T>},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {126, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 16, 128, 16, 16, 64, T>},  
    //  {127, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, T>},  
     {128, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 32, 128, 16, 32, 64, T>},
     {129, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, T>},
     {130, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 64, 64, T>},
     // 2 warp
    //  {131, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 128, 16, 16, 64, T>},  
     {132, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 128, 16, 32, 64, T>},
     {133, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 16, 64, 64, T>},
     //{134, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 16, 128, 64, T>},   //寄存器溢出 156
     // 4 warp
    //  {135, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 16, 128, 16, 16, 64, T>},  
     {136, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 128, 16, 32, 64, T>},
     // {137, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 16, 64, 64, T>},   // 寄存器溢出 120
     // {138, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 16, 128, 64, T>},   // 寄存器溢出 120
     
     ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
     // 4warp-4stages-warpk
    //  {139, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 32, T>},  
    //  {140, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 32, T>},  
 
     {141, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 32, T>},
     {142, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 32, T>},
 
     {143, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 256, 128, 32, 64, 32, T>},
     {144, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 64, 32, T>},
     
     // 8warp-4stages-warpk
    //  {145, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 32, T>},  
     // 8warp
     {146, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 256, 128, 64, 32, 32, T>},
     
     ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {147, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 32, 32, 32, T>},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {148, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 32, 32, 32, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
     {149, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 128, 32, 32, 32, T>},
 
     ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {150, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, T>},  
    //  {151, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 128, 32, 16, 64, T>},  
    //  {152, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 128, 64, 16, 64, T>},  
     // 4warp BLOCKN 128 
     {153, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 32, 64, T>},
     {154, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},
     //{155, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   // 寄存器溢出 32
     // 4warp BLOCKN 256 
     {156, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 128, 16, 64, 64, T>},
     //{157, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 64, 64, T>},   // 寄存器溢出 60
     //{158, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 64, 64, T>},   //寄存器溢出 392
     // 8warp BLOCKN 128 
    //  {159, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 16, 64, T>},  
    //  {160, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
     // 8warp BLOCKN 256 
     {161, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 128, 32, 32, 64, T>},
     //{162, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 32, 64, T>},   //寄存器溢出28
     // 16warp
    //  {163, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 128, 64, 16, 64, T>},  
     
     ///////////////////////////////////////////////////////N_warp-235b////////////////////////////////////////////////
     // 4warp BLOCKN 64 
    //  {164, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 16, 32, T>},  
    //  {165, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 32, 16, 32, T>},  
    //  {166, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 64, 16, 32, T>},  
     // 4warp BLOCKN 128 
     {167, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 128, 64, 16, 32, 32, T>},
     {168, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 32, 32, 32, T>},
     // 4warp BLOCKN 256 
     {169, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 256, 64, 16, 64, 32, T>},
     // 8warp BLOCKN 128 
    //  {170, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 32, 16, 32, T>},  
    //  {171, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 64, 64, 16, 32, T>},  
     ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
     // 4 warp:  2,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {174, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 64, 16, 16, 32, T>},  
     {175, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 32, 32, T>},
    //  {176, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 32, 16, 32, T>},  
     {177, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 32, 32, 32, T>},
     // 8 warp:  2,4,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {178, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 16, 32, T>},  
    //  {179, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 32, 16, 32, T>},  
     {180, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 128, 64, 16, 32, 32, T>},
     {181, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 128, 64, 32, 32, 32, T>},
     // 8 warp:  4,2,1
     // thread: 2,2,1    1,2,1    2,1,1  1,1,1
    //  {182, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 16, 16, 32, T>},  
     {183, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 64, 64, 16, 32, 32, T>},
    //  {184, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 32, 64, 32, 16, 32, T>},  
     {185, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<128, 64, 64, 32, 32, 32, T>},
     /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
     // 1 warp
    //  {186, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 16, 64, 16, 16, 32, T>},  
    //  {187, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, T>},  
     {188, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 32, 64, 16, 32, 32, T>},
     {189, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, T>},
     {190, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 64, 16, 64, 32, T>},
     // 2 warp
    //  {191, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 16, 64, 16, 16, 32, T>},  
     {192, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 32, 64, 16, 32, 32, T>},
     {193, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<32, 64, 64, 16, 64, 32, T>},
     // 4 warp
    //  {194, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 16, 64, 16, 16, 32, T>},  
     {195, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<64, 32, 64, 16, 32, 32, T>},
    };
}

static auto kernel_maps_gemm2_prefill_marlin_w16a16_half = make_kernel_maps_gemm2_prefill_marlin_w16a16<half>();
static auto kernel_maps_gemm2_prefill_marlin_w16a16_bhalf_t = make_kernel_maps_gemm2_prefill_marlin_w16a16<bhalf_t>();


}
}