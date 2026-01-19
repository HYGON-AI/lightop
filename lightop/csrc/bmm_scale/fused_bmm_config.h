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
    GemmParams(const T* ptr_q_nope,
               const T* ptr_w_uv,
               unsigned char* ptr_output,
               float* ptr_a_scale,
               float* ptr_b_scale,
               bhalf_t* ptr_bias,
               uint32_t batch_size,
               uint32_t n_size, 
               uint32_t p_size,
               uint32_t l_size,
               uint32_t stride_ab,
               uint32_t stride_an,
               uint32_t stride_ap,
               uint32_t stride_bn,
               uint32_t stride_bl,
               uint32_t stride_bp,
               uint32_t stride_asn,
               uint32_t stride_asb,
               uint32_t stride_ask,
               uint32_t stride_bsn,
               uint32_t stride_bsl,
               uint32_t stride_bsk,
               bool has_bias
            )
        : ptr_q_nope(ptr_q_nope),
          ptr_w_uv(ptr_w_uv),
          ptr_output(ptr_output),
          ptr_a_scale(ptr_a_scale),
          ptr_b_scale(ptr_b_scale),
          ptr_bias(ptr_bias),
          batch_size(batch_size),
          n_size(n_size),
          p_size(p_size),
          l_size(l_size),
          stride_ab(stride_ab),
          stride_an(stride_an),
          stride_ap(stride_ap),
          stride_bn(stride_bn),
          stride_bl(stride_bl),
          stride_bp(stride_bp),
          stride_asn(stride_asn),
          stride_asb(stride_asb),
          stride_ask(stride_ask),
          stride_bsn(stride_bsn),
          stride_bsl(stride_bsl),
          stride_bsk(stride_bsk),
          has_bias(has_bias)
         {}

    const T* ptr_q_nope;               // input q_nope
    const T* ptr_w_uv;                 // weight w_uv
    unsigned char* ptr_output;         // output
    float* ptr_a_scale;                // input scale a_scale
    float* ptr_b_scale;                // weight scale b_scale
    bhalf_t* ptr_bias;                 // bias
    uint32_t batch_size;               // batch size
    uint32_t n_size;                   // output size n
    uint32_t p_size;                   // input size p
    uint32_t l_size;                   // input size l
    uint32_t stride_ab;                // input stride ab   
    uint32_t stride_an;                // input stride an
    uint32_t stride_ap;                // input stride ap
    uint32_t stride_bn;                // weight stride bn
    uint32_t stride_bl;                // weight stride bl
    uint32_t stride_bp;                // weight stride bp
    uint32_t stride_asn;               // input scale stride asm
    uint32_t stride_asb;               // input scale stride ask
    uint32_t stride_ask;               // weight scale stride bse
    uint32_t stride_bsn;               // weight scale stride bsn
    uint32_t stride_bsl;               // weight scale stride bsk
    uint32_t stride_bsk;               // weight scale stride bsk
    bool has_bias;                     // has bias
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_fused_bmm_decode(const GemmParams<T>& params);

///////////////////////////////////////////////////////////////////////kernel_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
using KernelFunc = std::function<void(const GemmParams<char>&)>;

static std::unordered_map<int, KernelFunc> kernel_maps_fused_bmm = {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp warp_k=64
    {0, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 64,  128, 16, 16, 64, char>(p); }},
    {1, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 128, 16, 32, 64, char>(p); }},
    {2, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 128, 16, 64, 64, char>(p); }},
    // 8warp warp_k=64
    {3, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 128, 16, 16, 64, char>(p); }},
    {4, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 128, 16, 32, 64, char>(p); }},
    // 4warp warp_k=64
    // {5, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 64,  64, 16, 16, 32, char>(p); }},
    // {6, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 64, 16, 32, 32, char>(p); }},
    // {7, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 64, 16, 64, 32, char>(p); }},
    // // 8warp warp_k=64
    // {8, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 64, 16, 16, 32, char>(p); }},
    // {9, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 64, 16, 32, 32, char>(p); }},
    // 4warp warp_k=64, warp_k=block_k
    {10, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 64,  64, 16, 16, 64, char>(p); }},
    {11, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 64, 16, 32, 64, char>(p); }},
    {12, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 64, 16, 64, 64, char>(p); }},
    // 8warp warp_k=64, warp_k=block_k
    {13, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 64, 16, 16, 64, char>(p); }},
    {14, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 64, 16, 32, 64, char>(p); }},
    // 4warp warp_k=64, warp_k=block_k
    // {15, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 64,  32, 16, 16, 32, char>(p); }},
    // {16, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 32, 16, 32, 32, char>(p); }},
    // {17, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 32, 16, 64, 32, char>(p); }},
    // // 8warp warp_k=64, warp_k=block_k
    // {18, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 128, 32, 16, 16, 32, char>(p); }},
    // {19, [](const GemmParams<char>& p) { launch_fused_bmm_decode<16, 256, 32, 16, 32, 32, char>(p); }},
};

}
}