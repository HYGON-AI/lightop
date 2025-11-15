#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>

#include "aiter_hip_common.h"

using half_t  = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

template <typename T>
struct __attribute__((packed)) GemmInfo {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    T* d;
    T* c;
    int8_t* a;
    int8_t* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t  alpha[16];
    int8_t   beta[16];
    float* scaleA=nullptr;
    float* scaleB=nullptr;
};

namespace at {
namespace native {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_w8a8_gemm_nt_contig_asm_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,                   // (M)
    torch::Tensor b_scale,                   // (E, N)
    torch::Tensor m_indices,
    uint32_t experts_num) {
    GemmInfo<OutputType> d_prob;
    const int size_m = input.size(0);        //(M, K)
    const int size_n = b_qweight.size(1) * 16;    //(E, N // 16, K * 16)
    const int size_k = input.size(1);

    uint32_t lda = size_k;
    uint32_t ldb = size_k;
    uint32_t ldc = size_n;
    uint32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    size_t stride_a = size_m * size_k;
    size_t stride_b = size_n * size_k;
    size_t stride_c = size_m * size_n;
    size_t stride_d = size_m * size_n;
    {
    d_prob.m = size_n;
    d_prob.n = size_m;
    d_prob.batch = 1;
    d_prob.k = size_k;
    d_prob.strideD1 = ldd;
    d_prob.strideD2 = stride_d;
    d_prob.strideC1 = ldc;
    d_prob.strideC2 = stride_c;
    d_prob.strideA1 = ldb;
    d_prob.strideA2 = stride_b;
    d_prob.strideB1 = lda;
    d_prob.strideB2 = stride_a;
    memcpy(d_prob.alpha, &d_alpha, sizeof(float));
    memcpy(d_prob.beta,  &d_beta, sizeof(float));
    d_prob.a = b_qweight.data_ptr<int8_t>();
    d_prob.b = input.data_ptr<int8_t>();
    d_prob.d = static_cast<OutputType*>(output.data_ptr());
    d_prob.scaleA = b_scale.data_ptr<float>();
    d_prob.scaleB = a_scale.data_ptr<float>();
    }
    
    size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    int wg_M = DIVIDE(size_m, BLOCKM);
    int wg_N = DIVIDE(size_n, BLOCKN);
    wgTile += wg_M * wg_N;

	struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;                                
    }hipFunctionArgs;

    uint8_t* d_argsPtr;
    size_t arg_size = sizeof(d_prob);
    hipMalloc((void**)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, &d_prob, arg_size, hipMemcpyHostToDevice);

    hipFunctionArgs.gemm_count = 1;
    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    string fp16_co_file_tn = "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_FP16.co";
    string bf16_co_file_tn = "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.co";
	static AiterAsmKernel deepgemm_w8a8_fp16_tn("DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT256X256X128_FP16", fp16_co_file_tn.c_str());
	static AiterAsmKernel deepgemm_w8a8_bf16_tn("DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT256X256X128_BF16", bf16_co_file_tn.c_str());

    AiterAsmKernel *impl_ptr = nullptr;
    if (output.scalar_type() == at::ScalarType::BFloat16)
        impl_ptr = &deepgemm_w8a8_bf16_tn;
    else 
        impl_ptr = &deepgemm_w8a8_fp16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}


torch::Tensor m_grouped_w8a8_gemm_nt_contig_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  int mode) {
  
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    uint32_t experts_num = b_qweight.sizes()[0];
    if (mode == 1000) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else {
        TORCH_CHECK(false, mode, " mode error.");
    }
  return output;
}

}
}