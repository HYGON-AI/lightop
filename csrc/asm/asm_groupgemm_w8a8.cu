// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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
    T*  bias=nullptr;
    int biasType;
};


namespace at{
namespace native {

void _group_gemm_w8a8_256x256x128_TN_impl_fp16(
    const uint32_t& group_num,
    std::vector<torch::Tensor>& A_list,
    std::vector<torch::Tensor>& B_list,
    std::vector<torch::Tensor>& D_list,
    std::vector<torch::Tensor>& As_list,
    std::vector<torch::Tensor>& Bs_list,
    std::vector<float>& alphas,
    c10::optional<std::vector<torch::Tensor>>& C_list, 
    c10::optional<std::vector<float>>& betas
) {
    std::vector<GemmInfo<__half>> d_prob(group_num);
    for (int i = 0; i < group_num; ++i) {
        size_t batch =  A_list[i].dim() == 3 ? A_list[i].size(0) : 1;
        size_t size_m = A_list[i].dim() == 3 ? A_list[i].size(1) : A_list[i].size(0);
        size_t size_k = A_list[i].dim() == 3 ? A_list[i].size(2) : A_list[i].size(1);
        size_t size_n = B_list[i].dim() == 3 ? B_list[i].size(1) : B_list[i].size(0);
        size_t lda = size_k;
        size_t ldb = size_k;
        float d_alpha = alphas[i];
        float d_beta = 0.0;
        if (betas.has_value())
            d_beta = betas.value()[i];
        size_t ldc = size_n;
        size_t ldd = size_n;
        size_t stride_a = size_m * size_k;
        size_t stride_b = size_n * size_k;
        size_t stride_c = size_m * size_n;
        size_t stride_d = size_m * size_n;
        
        size_t 	size_mn =  ldc * size_m +(stride_c*(batch-1));
        size_t 	size_mk =  (stride_a*(batch-1)+(lda*size_m));
        size_t 	size_nk =  (stride_b*(batch-1)+(ldb*size_n));
        d_prob[i].m = size_n;
        d_prob[i].n = size_m;
        d_prob[i].batch = batch;
        d_prob[i].k = size_k;
        d_prob[i].strideD1 = ldd;
        d_prob[i].strideD2 = stride_d;
        d_prob[i].strideC1 = ldc;
        d_prob[i].strideC2 = stride_c;
        d_prob[i].strideA1 = ldb;
        d_prob[i].strideA2 = stride_b;
        d_prob[i].strideB1 = lda;
        d_prob[i].strideB2 = stride_a;
        memcpy(d_prob[i].alpha, &d_alpha, sizeof(float));
        memcpy(d_prob[i].beta, &d_beta, sizeof(float));

        d_prob[i].a = B_list[i].data_ptr<int8_t>();
        d_prob[i].b = A_list[i].data_ptr<int8_t>();
        d_prob[i].d = reinterpret_cast<half_t *>(D_list[i].data_ptr());
        if (C_list.has_value())
            d_prob[i].c = reinterpret_cast<half_t *>(C_list.value()[i].data_ptr());
        d_prob[i].scaleA = Bs_list[i].data_ptr<float>();
        d_prob[i].scaleB = As_list[i].data_ptr<float>();
    }

    const unsigned int workDim = 3;
    const unsigned int threadTile[2] = { 16, 8 };
    const unsigned int groupSize[2] = { 16, 32 };
    size_t localWorkSize[3] = { 768, 1, 1 };
    unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
    unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);

    size_t wgTile = 0;
    for (int i = 0; i < group_num; ++i) {
        int wg_M=(d_prob[i].m+macroTile0-1)/macroTile0;
        int wg_N=(d_prob[i].n+macroTile1-1)/macroTile1;
        wgTile += wg_M * wg_N * d_prob[i].batch;
    }
    uint8_t* d_argsPtr;
    size_t arg_size = group_num * sizeof(GemmInfo<__half>);
    hipMalloc((void **)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, d_prob.data(), arg_size, hipMemcpyHostToDevice);

    struct __attribute__((packed)) {
        uint32_t gemm_count;
        void const* DeviceUserArguments;
        void const* argsPtr;
        uint32_t wgm;
        unsigned int gsu;
    }hipFunctionArgs;
    hipFunctionArgs.gemm_count=group_num;
    hipFunctionArgs.DeviceUserArguments=d_argsPtr;
    hipFunctionArgs.argsPtr=nullptr;
    hipFunctionArgs.wgm=1;
    hipFunctionArgs.gsu=1;

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    string fp16_co_file_tn = "grougemm_w8a8_tensor/GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_FP16.co";
    // string bf16_co_file_tn = "grougemm_w8a8_tensor/GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_BF16.co";
    static AiterAsmKernel groupgemm_w8a8_fp16_tn("GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_FP16", fp16_co_file_tn.c_str());
    // static AiterAsmKernel groupgemm_w8a8_bf16_tn("GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_BF16", bf16_co_file_tn.c_str());
    AiterAsmKernel *impl_ptr = nullptr;
    impl_ptr = &groupgemm_w8a8_fp16_tn;
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

void _group_gemm_w8a8_256x256x128_TN_impl_bf16(
    const uint32_t& group_num,
    std::vector<torch::Tensor>& A_list,
    std::vector<torch::Tensor>& B_list,
    std::vector<torch::Tensor>& D_list,
    std::vector<torch::Tensor>& As_list,
    std::vector<torch::Tensor>& Bs_list,
    std::vector<float>& alphas,
    c10::optional<std::vector<torch::Tensor>>& C_list, 
    c10::optional<std::vector<float>>& betas
) {
    std::vector<torch::Tensor> D_ref;
    std::vector<GemmInfo<bhalf_t>> d_prob(group_num);
    for (int i = 0; i < group_num; ++i) {
        size_t batch =  A_list[i].dim() == 3 ? A_list[i].size(0) : 1;
        size_t size_m = A_list[i].dim() == 3 ? A_list[i].size(1) : A_list[i].size(0);
        size_t size_k = A_list[i].dim() == 3 ? A_list[i].size(2) : A_list[i].size(1);
        size_t size_n = B_list[i].dim() == 3 ? B_list[i].size(1) : B_list[i].size(0);
        size_t lda = size_k;
        size_t ldb = size_k;
        float d_alpha = alphas[i];
        float d_beta = 0.0;
        if (betas.has_value())
            d_beta = betas.value()[i];
        size_t ldc = size_n;
        size_t ldd = size_n;
        size_t stride_a = size_m * size_k;
        size_t stride_b = size_n * size_k;
        size_t stride_c = size_m * size_n;
        size_t stride_d = size_m * size_n;
        
        size_t 	size_mn =  ldc * size_m +(stride_c*(batch-1));
        size_t 	size_mk =  (stride_a*(batch-1)+(lda*size_m));
        size_t 	size_nk =  (stride_b*(batch-1)+(ldb*size_n));
        d_prob[i].m = size_n;
        d_prob[i].n = size_m;
        d_prob[i].batch = batch;
        d_prob[i].k = size_k;
        d_prob[i].strideD1 = ldd;
        d_prob[i].strideD2 = stride_d;
        d_prob[i].strideC1 = ldc;
        d_prob[i].strideC2 = stride_c;
        d_prob[i].strideA1 = ldb;
        d_prob[i].strideA2 = stride_b;
        d_prob[i].strideB1 = lda;
        d_prob[i].strideB2 = stride_a;
        
        memcpy(d_prob[i].alpha, &d_alpha, sizeof(float));
        memcpy(d_prob[i].beta, &d_beta, sizeof(float));

        d_prob[i].a = B_list[i].data_ptr<int8_t>();
        d_prob[i].b = A_list[i].data_ptr<int8_t>();
        d_prob[i].d = reinterpret_cast<bhalf_t *>(D_list[i].data_ptr());
        if (C_list.has_value())
            d_prob[i].c = reinterpret_cast<bhalf_t *>(C_list.value()[i].data_ptr());
        d_prob[i].scaleA = Bs_list[i].data_ptr<float>();
        d_prob[i].scaleB = As_list[i].data_ptr<float>();

    }

    const unsigned int workDim = 3;
    const unsigned int threadTile[2] = { 16, 8 };
    const unsigned int groupSize[2] = { 16, 32 };
    size_t localWorkSize[3] = { 768, 1, 1 };
    unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
    unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);

    size_t wgTile = 0;
    for (int i = 0; i < group_num; ++i) {
        int wg_M=(d_prob[i].m+macroTile0-1)/macroTile0;
        int wg_N=(d_prob[i].n+macroTile1-1)/macroTile1;
        wgTile += wg_M * wg_N * d_prob[i].batch;
    }
    uint8_t* d_argsPtr;
    size_t arg_size = group_num * sizeof(GemmInfo<bhalf_t>);
    hipMalloc((void **)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, d_prob.data(), arg_size, hipMemcpyHostToDevice);

    struct __attribute__((packed)) {
        uint32_t gemm_count;
        void const* DeviceUserArguments;
        void const* argsPtr;
        uint32_t wgm;
        unsigned int gsu;
    }hipFunctionArgs;
    hipFunctionArgs.gemm_count=group_num;
    hipFunctionArgs.DeviceUserArguments=d_argsPtr;
    hipFunctionArgs.argsPtr=nullptr;
    hipFunctionArgs.wgm=0;
    hipFunctionArgs.gsu=1;

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    string bf16_co_file_tn = "grougemm_w8a8_tensor/GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_BF16.co";
    static AiterAsmKernel groupgemm_w8a8_bf16_tn("GroupGemm_W8A8_I8_PERTENSOR_ASM_TN_MT256X256X128_BF16", bf16_co_file_tn.c_str());
    AiterAsmKernel *impl_ptr = nullptr;
    impl_ptr = &groupgemm_w8a8_bf16_tn;
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

std::vector<torch::Tensor> grougemm_w8a8_pertensor_asm(
    std::vector<torch::Tensor>& A_list,
    std::vector<torch::Tensor>& B_list,
    std::vector<torch::Tensor>& D_list,
    std::vector<torch::Tensor>& As_list,
    std::vector<torch::Tensor>& Bs_list,
    std::vector<float>& alphas,
    c10::optional<std::vector<torch::Tensor>>& C_list, 
    c10::optional<std::vector<float>>& betas,
    torch::Dtype output_dtype
) {
    uint32_t group_num = A_list.size();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A_list[0]));
    if (output_dtype == torch::kFloat16)
        _group_gemm_w8a8_256x256x128_TN_impl_fp16(
            group_num,
            A_list,
            B_list,
            D_list,
            As_list,
            Bs_list,
            alphas,
            C_list,
            betas
        );
    else if (output_dtype == torch::kBFloat16)
        _group_gemm_w8a8_256x256x128_TN_impl_bf16(
            group_num,
            A_list,
            B_list,
            D_list,
            As_list,
            Bs_list,
            alphas,
            C_list,
            betas
        );
    else
        TORCH_CHECK(false, "group gemm w8a8 only supports outputType in [Float16, BFloat16]");
    return D_list;
}

}
}