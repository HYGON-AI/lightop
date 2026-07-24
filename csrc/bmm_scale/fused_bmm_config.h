#pragma once
#include <unordered_map>
#include <functional>
// MOE prefill和decode的分界线
#define MOE_THRESHOLD 128

namespace at {
namespace native {
///////////////////////////////////////////////////////////////////////Gemm params///////////////////////////////////////////////////////////////////////////////////////

// GemmParams结构体定义
template <typename T, typename OutputType=unsigned char>
struct GemmParams {
    GemmParams(const T* ptr_q_nope,
               const T* ptr_w_uv,
               OutputType* ptr_output,
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
               uint32_t stride_ob,
               uint32_t stride_on,
               uint32_t stride_ol,
               uint32_t stride_asb,
               uint32_t stride_asn,
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
          stride_ob(stride_ob),
          stride_on(stride_on),
          stride_ol(stride_ol),
          stride_asb(stride_asb),
          stride_asn(stride_asn),
          stride_ask(stride_ask),
          stride_bsn(stride_bsn),
          stride_bsl(stride_bsl),
          stride_bsk(stride_bsk),
          has_bias(has_bias)
         {}

    const T* ptr_q_nope;               // input q_nope
    const T* ptr_w_uv;                 // weight w_uv
    OutputType* ptr_output;            // output
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
    uint32_t stride_ob;                // output stride ob
    uint32_t stride_on;                // output stride on
    uint32_t stride_ol;                // output stride ol
    uint32_t stride_asb;               // input scale stride asb
    uint32_t stride_asn;               // input scale stride asn
    uint32_t stride_ask;               // weight scale stride bse
    uint32_t stride_bsn;               // weight scale stride bsn
    uint32_t stride_bsl;               // weight scale stride bsk
    uint32_t stride_bsk;               // weight scale stride bsk
    bool has_bias;                     // has bias
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename U, typename OutputType, bool LDS_BYPASS>
void launch_fused_bmm_decode(const GemmParams<T, OutputType>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename U, typename OutputType, bool LDS_BYPASS>
void kernel_launch_fused_bmm_decode(const GemmParams<T, OutputType>& params){
    launch_fused_bmm_decode<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T, U, OutputType, LDS_BYPASS>(params);
}

///////////////////////////////////////////////////////////////////////kernel_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
template<typename T, typename OutputType>
using KernelFunc = std::function<void(const GemmParams<T, OutputType>&)>;

template <typename T, typename U, typename OutputType>
static std::unordered_map<int, KernelFunc<T, OutputType>> make_kernel_maps_batched_gemm_w8a8(){
    return {
    ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
    // 4warp warp_k=64
    {0, &kernel_launch_fused_bmm_decode<16, 64,  128, 16, 16, 64, T, U, OutputType, false>},
    {1, &kernel_launch_fused_bmm_decode<16, 128, 128, 16, 32, 64, T, U, OutputType, false>},
    {2, &kernel_launch_fused_bmm_decode<16, 256, 128, 16, 64, 64, T, U, OutputType, false>},
    // 8warp warp_k=64
    {3, &kernel_launch_fused_bmm_decode<16, 128, 128, 16, 16, 64, T, U, OutputType, false>},
    {4, &kernel_launch_fused_bmm_decode<16, 256, 128, 16, 32, 64, T, U, OutputType, false>},
    // 4warp warp_k=64
    // {5, &kernel_launch_fused_bmm_decode<16, 64,  64, 16, 16, 32, T, U, OutputType, false>},
    // {6, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 32, 32, T, U, OutputType, false>},
    // {7, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 64, 32, T, U, OutputType, false>},
    // // 8warp warp_k=64
    // {8, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 16, 32, T, U, OutputType, false>},
    // {9, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 32, 32, T, U, OutputType, false>},
    // 4warp warp_k=64, warp_k=block_k
    {10, &kernel_launch_fused_bmm_decode<16, 64,  64, 16, 16, 64, T, U, OutputType, false>},
    {11, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 32, 64, T, U, OutputType, false>},
    {12, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 64, 64, T, U, OutputType, false>},
    // 8warp warp_k=64, warp_k=block_k
    {13, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 16, 64, T, U, OutputType, false>},
    {14, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 32, 64, T, U, OutputType, false>},
    // 4warp warp_k=64, warp_k=block_k
    // {15, &kernel_launch_fused_bmm_decode<16, 64,  32, 16, 16, 32, T, U, OutputType, false>},
    // {16, &kernel_launch_fused_bmm_decode<16, 128, 32, 16, 32, 32, T, U, OutputType, false>},
    // {17, &kernel_launch_fused_bmm_decode<16, 256, 32, 16, 64, 32, T, U, OutputType, false>},
    // // 8warp warp_k=64, warp_k=block_k
    // {18, &kernel_launch_fused_bmm_decode<16, 128, 32, 16, 16, 32, T, U, OutputType, false>},
    // {19, &kernel_launch_fused_bmm_decode<16, 256, 32, 16, 32, 32, T, U, OutputType, false>},
    ///////////////////////////////////////////////////////LDS_BYPASS/////////////////////////////////////////
    // 4warp warp_k=64
    {50, &kernel_launch_fused_bmm_decode<16, 64,  128, 16, 16, 64, T, U, OutputType, true>},
    {51, &kernel_launch_fused_bmm_decode<16, 128, 128, 16, 32, 64, T, U, OutputType, true>},
    {52, &kernel_launch_fused_bmm_decode<16, 256, 128, 16, 64, 64, T, U, OutputType, true>},
    // 8warp warp_k=64
    {53, &kernel_launch_fused_bmm_decode<16, 128, 128, 16, 16, 64, T, U, OutputType, true>},
    {54, &kernel_launch_fused_bmm_decode<16, 256, 128, 16, 32, 64, T, U, OutputType, true>},
    // 4warp warp_k=64
    // {5, &kernel_launch_fused_bmm_decode<16, 64,  64, 16, 16, 32, T, U, OutputType, false>},
    // {6, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 32, 32, T, U, OutputType, false>},
    // {7, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 64, 32, T, U, OutputType, false>},
    // // 8warp warp_k=64
    // {8, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 16, 32, T, U, OutputType, false>},
    // {9, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 32, 32, T, U, OutputType, false>},
    // 4warp warp_k=64, warp_k=block_k
    {60, &kernel_launch_fused_bmm_decode<16, 64,  64, 16, 16, 64, T, U, OutputType, true>},
    {61, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 32, 64, T, U, OutputType, true>},
    {62, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 64, 64, T, U, OutputType, true>},
    // 8warp warp_k=64, warp_k=block_k
    {63, &kernel_launch_fused_bmm_decode<16, 128, 64, 16, 16, 64, T, U, OutputType, true>},
    {64, &kernel_launch_fused_bmm_decode<16, 256, 64, 16, 32, 64, T, U, OutputType, true>},
    // 4warp warp_k=64, warp_k=block_k
    // {15, &kernel_launch_fused_bmm_decode<16, 64,  32, 16, 16, 32, T, U, OutputType, true>},
    // {16, &kernel_launch_fused_bmm_decode<16, 128, 32, 16, 32, 32, T, U, OutputType, true>},
    // {17, &kernel_launch_fused_bmm_decode<16, 256, 32, 16, 64, 32, T, U, OutputType, true>},
    // // 8warp warp_k=64, warp_k=block_k
    // {18, &kernel_launch_fused_bmm_decode<16, 128, 32, 16, 16, 32, T, U, OutputType, true>},
    // {19, &kernel_launch_fused_bmm_decode<16, 256, 32, 16, 32, 32, T, U, OutputType, true>},
    };
}

static auto kernel_maps_batched_gemm_int8_fp8 = make_kernel_maps_batched_gemm_w8a8<char, int, unsigned char>();
static auto kernel_maps_batched_gemm_fp8_fp8 = make_kernel_maps_batched_gemm_w8a8<fp8_t, float, unsigned char>();
static auto kernel_maps_batched_gemm_int8_bf16 = make_kernel_maps_batched_gemm_w8a8<char, int, bhalf_t>();
static auto kernel_maps_batched_gemm_fp8_bf16 = make_kernel_maps_batched_gemm_w8a8<fp8_t, float, bhalf_t>();

}
}