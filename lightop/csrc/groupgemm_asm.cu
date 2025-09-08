#include <iostream>
#include <stdio.h>
#include <string.h>
#include <random>
#include "hip/hip_runtime.h"
#include <hip/hip_hcc.h>
#include "rocblas.h"
#include "internal/rocblas-auxiliary.h"

#include <torch/extension.h>
#include <string>
#include <sstream>
#include "groupgemm_asm.h"
#include <ATen/dtk_macros.h>
#include "hip/hip_runtime.h"
#include <hipblaslt/hipblaslt.h>
// #include "hip/hip_bf16.h"
// #include "hip/hip_fp16.h"
#include <ATen/hip/HIPContext.h>
#include <hip/hip_hcc.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>
#include <filesystem>
#include <torch/all.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"

using bhalf_t = hip_bfloat16;
enum class BiasType : int
{
    Float,
    Double,
    ComplexFloat,
    ComplexDouble,
    Half,
    Int8x4,
    Int32, //6,
    BFloat16,
    Int8,
    Float8,
    BFloat8,
    XFloat32,
    Float8BFloat8,
    BFloat8Float8,
    Count,
    None = Count
};

enum class ActivationType : uint32_t
{
    None = 0,
    Abs,
    Clippedrelu,
    Gelu,
    Leakyrelu,
    Relu, //5,
    Sigmoid,
    Tanh,
    DGelu,
    Geluscaling,
    All,
    Exp, // Verification use only.
    Count
};


namespace at {
namespace native {

template <typename Ti, typename To, typename Tc, typename Talpha = Tc, typename Tbias = Tc>
int makeArg(
    torch::Tensor& a, torch::Tensor& b, torch::Tensor& d,
    GPUProb<Ti, To, Tc, Talpha, Tbias>& d_prob,
    uint32_t M, uint32_t N, uint32_t K,
    // size_t batch_count,
    BiasType use_bias_type,
    ActivationType use_activation_type){

    uint32_t lda = N;
    uint32_t ldb = K;
    uint32_t ldc = N;
    uint32_t ldd = N;

    uint32_t stride_a = N * K;
    uint32_t stride_b = M * K;
    uint32_t stride_c = M * N;
    uint32_t stride_d = M * N;

    Tc h_alpha = 1;
    Tc h_beta  = 0;
    int batchcount = 1;

    d_prob.a = reinterpret_cast<Ti*>(b.data_ptr());
    d_prob.b = reinterpret_cast<Ti*>(a.data_ptr());
    // d_prob.c = reinterpret_cast<To*>(c.data_ptr());
    d_prob.c = nullptr;
    d_prob.d = reinterpret_cast<To*>(d.data_ptr());
    
    d_prob.size_0 = static_cast<uint32_t>(N);
    d_prob.size_1 = static_cast<uint32_t>(M);
    d_prob.size_2 = static_cast<uint32_t>(batchcount);
    d_prob.size_3 = static_cast<uint32_t>(K);

    d_prob.strideD1 = static_cast<unsigned int>(ldd);
    d_prob.strideD2 = static_cast<unsigned int>(stride_d);
    d_prob.strideC1 = static_cast<unsigned int>(ldc);
    d_prob.strideC2 = static_cast<unsigned int>(stride_c);
    d_prob.strideA1 = static_cast<unsigned int>(lda);
    d_prob.strideA2 = static_cast<unsigned int>(stride_a);
    d_prob.strideB1 = static_cast<unsigned int>(ldb);
    d_prob.strideB2 = static_cast<unsigned int>(stride_b);

    d_prob.alpha = h_alpha;
    d_prob.beta = h_beta; 
    d_prob.scaleAlphaVec == nullptr;       
    d_prob.bias =  nullptr;       
    d_prob.bias_type = static_cast<unsigned int>(use_bias_type);
    d_prob.strideBias = 0;
    d_prob.activation_0 = 2;
    d_prob.activation_1 = 2;
    d_prob.activationType = static_cast<unsigned int>(use_activation_type);

    return 0;
}

int _groupgemm_nn_impl(std::vector<torch::Tensor>& a_list,
                       std::vector<torch::Tensor>& b_list,
                       std::vector<torch::Tensor>& d_list,
                       const std::vector<uint32_t>& M_list,
                       const std::vector<uint32_t>& N_list,
                       const std::vector<uint32_t>& K_list,
                       int device_id)
{
    using Ti = bhalf_t;
    using To = bhalf_t;
    using Tc = float;
    using Talpha = Tc;
    using Tbias = Tc;
    /* grid sizes */
    const unsigned int workDim = 3;
    const unsigned int threadTile[2] = { 16, 8 };
    const unsigned int groupSize[2] = { 16, 32 };
    size_t localWorkSize[3] = { 512, 1, 1 };
    unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
    unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);

    int gemm_count=static_cast<int>(a_list.size());;
    int wgRight=0;//gemm对于每个size的WG的偏移量
    
    std::vector<GPUProb<Ti,To,Tc,Talpha,Tbias>> d_prob(gemm_count);  
    std::vector<uint32_t> arg_wgRight(gemm_count);
        
    for (int i = 0; i < gemm_count; ++i) {
        makeArg<Ti,To,Tc,Talpha,Tbias>(a_list[i], b_list[i], d_list[i],
                                    d_prob[i],
                                    M_list[i], N_list[i], K_list[i],
                                    BiasType::Float, ActivationType::None);
    }


    for (int i=0;i<gemm_count;i++)
    {
        arg_wgRight[i]=wgRight;
        int wg_M=(d_prob[i].size_0+macroTile0-1)/macroTile0;
        int wg_N=(d_prob[i].size_1+macroTile1-1)/macroTile1;
        wgRight += wg_M * wg_N * d_prob[i].size_2;  
    }

    size_t arg_wgRight_size= gemm_count*sizeof(uint32_t);
    size_t arg_size = gemm_count*sizeof(GPUProb<Ti,To,Tc,Talpha,Tbias>);
    
    HIP_CALL(hipSetDevice(device_id));
    uint8_t* d_argsPtr;
    HIP_CALL(hipMalloc((void**)&d_argsPtr, arg_wgRight_size + arg_size));
    HIP_CALL(hipMemcpy(d_argsPtr,
                    arg_wgRight.data(),
                    arg_wgRight_size,
                    hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_argsPtr + arg_wgRight_size,
                    d_prob.data(),
                    arg_size,
                    hipMemcpyHostToDevice));

    //计算launch等参数
    size_t numWorkItems[3]={static_cast<size_t>(wgRight),1,1};
    
    //GPU_launch
	struct __attribute__((packed)){
        uint32_t gemm_count;
        void const* DeviceUserArguments;
        void const* argsPtr;
        uint32_t kipWgTableGen;
        unsigned int gsu;
    }hipFunctionArgs;
    
    hipFunctionArgs.gemm_count=gemm_count;
    hipFunctionArgs.DeviceUserArguments=nullptr;
    hipFunctionArgs.argsPtr=d_argsPtr;
    hipFunctionArgs.kipWgTableGen=0;
    hipFunctionArgs.gsu=1;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	static AiterAsmKernel groupgemm_NN_BF16("Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT256x256x16_MI16x16x1_SN_GSU1_K1_MIWT8_4", "groupgemm.co");
	AiterAsmKernel *impl_ptr = &groupgemm_NN_BF16;
    size_t size = sizeof(hipFunctionArgs);
	impl_ptr->launch_kernel_ext({&hipFunctionArgs,
                             &size,
                             numWorkItems[0], 
                             numWorkItems[1], 
                             numWorkItems[2],
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    hipFree(d_argsPtr);
    return 0;
}

Tensor groupgemm_dispatch(Tensor &a, Tensor &b, Tensor &batch_sizes, bool trans_b) {
    TORCH_CHECK(b.dim() == 3, "b 必须是三维 [?, K, N]");
    int device_id = a.device().index(); 
    auto options = torch::TensorOptions()
                    .dtype(a.scalar_type())
                    .device(a.device());

    const uint32_t M = a.size(0);
    const uint32_t K = a.size(1);
    const uint32_t N = b.size(2);
    TORCH_CHECK(b.size(1) == K, "b.size(1) 必须等于 a.size(1)");

    Tensor d = torch::empty({M, N}, options);

    const int parts_num = batch_sizes.size(0);
    int64_t row_offset = 0;

    std::vector<Tensor> a_parts, b_parts, d_parts;
    std::vector<uint32_t> M_list, N_list, K_list;

    a_parts.reserve(parts_num);
    b_parts.reserve(parts_num);
    d_parts.reserve(parts_num);
    M_list.reserve(parts_num);
    N_list.reserve(parts_num);
    K_list.reserve(parts_num);
    Tensor batch_sizes_cpu = batch_sizes.to(torch::kCPU);
    int64_t* batch_ptr = batch_sizes_cpu.data_ptr<int64_t>();

    for (int i = 0; i < parts_num; ++i) {
        // const int64_t rows = batch_sizes[i].item<int64_t>();
        const int64_t rows = batch_ptr[i];
        a_parts.emplace_back(a.narrow(0, row_offset, rows));
        b_parts.emplace_back(b[i]);
        d_parts.emplace_back(d.narrow(0, row_offset, rows));

        M_list.emplace_back(static_cast<uint32_t>(rows));
        N_list.emplace_back(N);
        K_list.emplace_back(K);

        row_offset += rows;
    }

    _groupgemm_nn_impl(a_parts, b_parts, d_parts, M_list, N_list, K_list, device_id);
    return d;
}


Tensor groupgemm(Tensor &a, Tensor &b, Tensor &batch_sizes, bool trans_b){
    return groupgemm_dispatch(a, b, batch_sizes, trans_b);
}

}
} // namespace at

