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
struct GemmArg {
    torch::Tensor a, b, c, d;     
    uint32_t M, N, K;              
    uint32_t batchcount;       
    float alpha, beta;         
    torch::Tensor bias;        
    int biastype;         
    uint32_t activationType;     
    c10::optional<torch::Tensor> scaleAlphaVec; 
};


template <typename Ti, typename To, typename Tc, typename Talpha = Tc, typename Tbias = Tc>
int makeArg(
    const GemmArg& args,
    GPUProb<Ti, To, Tc, Talpha, Tbias>& d_prob)
{

    uint32_t lda = args.N;
    uint32_t ldb = args.K;
    uint32_t ldc = args.N;
    uint32_t ldd = args.N;

    uint32_t stride_a = args.N * args.K;
    uint32_t stride_b = args.M * args.K;
    uint32_t stride_c = args.M * args.N;
    uint32_t stride_d = args.M * args.N;

    Tc h_alpha = 1;
    Tc h_beta  = 0;

    d_prob.a = reinterpret_cast<Ti*>(args.b.data_ptr());
    d_prob.b = reinterpret_cast<Ti*>(args.a.data_ptr());
    d_prob.c = reinterpret_cast<To*>(args.c.data_ptr());
    d_prob.d = reinterpret_cast<To*>(args.d.data_ptr());
    
    d_prob.size_0 = static_cast<uint32_t>(args.N);
    d_prob.size_1 = static_cast<uint32_t>(args.M);
    d_prob.size_2 = static_cast<uint32_t>(args.batchcount);
    d_prob.size_3 = static_cast<uint32_t>(args.K);

    d_prob.strideD1 = static_cast<uint32_t>(ldd);
    d_prob.strideD2 = static_cast<uint32_t>(stride_d);
    d_prob.strideC1 = static_cast<uint32_t>(ldc);
    d_prob.strideC2 = static_cast<uint32_t>(stride_c);
    d_prob.strideA1 = static_cast<uint32_t>(lda);
    d_prob.strideA2 = static_cast<uint32_t>(stride_a);
    d_prob.strideB1 = static_cast<uint32_t>(ldb);
    d_prob.strideB2 = static_cast<uint32_t>(stride_b);

    if(args.scaleAlphaVec.has_value()) {
        // d_prob.scaleAlphaVec = reinterpret_cast<Talpha*>(args.scaleAlphaVec.value().data_ptr());
        d_prob.scaleAlphaVec = nullptr;
    } else {
        d_prob.scaleAlphaVec = nullptr;
    }
    d_prob.alpha = d_prob.scaleAlphaVec == nullptr ? h_alpha : 1;
    d_prob.beta = h_beta;
    // d_prob.bias =  reinterpret_cast<Tbias*>(args.bias.data_ptr());         
    d_prob.bias =  nullptr;       
    d_prob.bias_type = static_cast<uint32_t>(args.biastype);
    d_prob.strideBias = 0;
    d_prob.activation_0 = 2;
    d_prob.activation_1 = 2;
    d_prob.activationType = static_cast<uint32_t>(args.activationType);

    return 0;
}

std::vector<torch::Tensor> groupgemm(const std::vector<GemmArg>& args)
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

    int gemm_count=static_cast<int>(args.size());
    int wgRight=0;//gemm对于每个size的WG的偏移量
    
    std::vector<GPUProb<Ti,To,Tc,Talpha,Tbias>> d_prob(gemm_count);
    
    std::vector<uint32_t> arg_wgRight(gemm_count);
    
    for (int i=0;i<gemm_count;i++)
    {
        makeArg<Ti, To, Tc, Talpha, Tbias>(args[i], d_prob[i]);
        arg_wgRight[i]=wgRight;
        int wg_M=(d_prob[i].size_0+macroTile0-1)/macroTile0;
        int wg_N=(d_prob[i].size_1+macroTile1-1)/macroTile1;
        wgRight += wg_M * wg_N * d_prob[i].size_2;  
    }

    size_t arg_wgRight_size= gemm_count*sizeof(uint32_t);
    size_t arg_size = gemm_count*sizeof(GPUProb<Ti,To,Tc,Talpha,Tbias>);
    
    uint8_t* d_argsPtr;
    hipMalloc((void **)&d_argsPtr,arg_wgRight_size+arg_size);
    hipMemcpy(d_argsPtr,arg_wgRight.data(),arg_wgRight_size,hipMemcpyHostToDevice);
    hipMemcpy(d_argsPtr+arg_wgRight_size,d_prob.data(),arg_size,hipMemcpyHostToDevice);

    //计算launch等参数
    size_t numWorkItems[3]={localWorkSize[0]*wgRight,1,1};
    
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
	static AiterAsmKernel groupgemm("Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT256x256x16_MI16x16x1_SN_GSU1_K1_MIWT8_4", "groupgemm.co");
	AiterAsmKernel *impl_ptr = &groupgemm;
    size_t size = sizeof(hipFunctionArgs);
	impl_ptr->launch_kernel({&hipFunctionArgs,
                             &size,
                             numWorkItems[0], 
                             numWorkItems[1], 
                             numWorkItems[2],
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    hipDeviceSynchronize();
    hipFree(d_argsPtr);
    std::vector<torch::Tensor> outputs;
    for(const auto& arg : args) outputs.push_back(arg.d);
    return outputs;
}

}
} // namespace at