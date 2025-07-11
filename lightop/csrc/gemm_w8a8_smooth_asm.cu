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
#include "gemm_w8a8_smooth_asm.h"
#include <ATen/dtk_macros.h>
#include "hip/hip_runtime.h"
#include <c10/util/BFloat16.h>
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
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

using half_t = __half;
using bhalf_t = __hip_bfloat16;
template<typename T>
struct TypeNameResolver {
    static constexpr const char* name = "unknown";
};

template<>
struct TypeNameResolver<half_t> {
    static constexpr const char* name = "FP16";
};

template<>
struct TypeNameResolver<bhalf_t> { // 如果你也在用 bfloat16
    static constexpr const char* name = "BF16";
};


namespace at {
namespace native {


template <typename OutputType>
void _gemm_w8a8_smooth_256x256x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 1;	//
	int depth= 128;//atoi(argv[8]);
	int SU = 0;					//kernel中没有StaggerU相关代码
	size_t strideC1J = ldc;
	size_t strideC2K = stridec;
	size_t strideA1L = lda;
	size_t strideA2K = stridea;
	size_t strideB1L = ldb;
	size_t strideB2K = strideb;



    unsigned int staggerUIter = SU; // how many stride-sized clicks to stagger start offset
	int unrollLoopIters = size_k/depth/1; // /DepthU/GSU
	int tmp;
	if(depth==8)
		tmp=16;
	else if(depth==16)
		tmp=8;
	while (staggerUIter>1) {
		if (unrollLoopIters >= (staggerUIter*tmp)) {
		break;}
		staggerUIter /= 2; // step down to smaller stagger
	}
	if (staggerUIter>=1) staggerUIter -= 1;


	/* grid sizes */
	const unsigned int workDim = 3;
	const unsigned int threadTile[2] = { 16, 8 };
	const unsigned int groupSize[2] = { 16, 32 };
	size_t localWorkSize[3] = { 768, 1, 1 };//需要额外线程
	size_t globalWorkSize[1][3];
	globalWorkSize[0][2] = 1;
	globalWorkSize[0][2] *= 1;  // batch = 1
	unsigned int sizeOfC0 = size_m;
	unsigned int sizeOfC1 = size_n;
	unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
	unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);
	unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
	unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
	// b/c single kernel, add extra work-group here if edge needed
	if (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }
	if (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }

	unsigned int problemNumGroupTiles0 = totalWorkGroups0;
	unsigned int problemNumGroupTiles1 = totalWorkGroups1;
	const unsigned smallNumMagicShift = 31; // bozo, review
	unsigned magicNumberProblemNumGroupTiles0 = (1L<<smallNumMagicShift) / problemNumGroupTiles0 + 1; // bozo, review
	unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
	unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
	if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
	unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);
	globalWorkSize[0][0] = totalWorkGroups0;
	globalWorkSize[0][1] = totalWorkGroups1;

	uint64_t tensor2dSizeC = 1 * std::max(size_m, strideC1J) * std::max(size_n, strideC2K);
	uint64_t tensor2dSizeA = 1;
	uint64_t tensor2dSizeAStride = 0;
	uint64_t tensor2dSizeAOffset = 0;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeA -= tensor2dSizeAOffset;

	uint64_t tensor2dSizeB = 1;
	uint64_t tensor2dSizeBStride = 0;
	uint64_t tensor2dSizeBOffset = 0;
	tensor2dSizeBStride = std::max(tensor2dSizeB*size_k, (uint64_t)strideB1L);
	tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB*size_k;
	tensor2dSizeB = tensor2dSizeBStride;
	tensor2dSizeBStride = std::max(tensor2dSizeB*size_n, (uint64_t)strideB2K);
	tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB*size_n;
	tensor2dSizeB = tensor2dSizeBStride;
	tensor2dSizeB -= tensor2dSizeBOffset;

    hipFunctionArgs_wa8a_smooth<OutputType> params;



	params.tensor2dSizeC = tensor2dSizeC;
	params.tensor2dSizeB = tensor2dSizeB;
	params.tensor2dSizeA = tensor2dSizeA;
	params.dataD = D_data;
	params.dataC = nullptr; // C_data is not used in this kernel
	params.dataA = A_data;
	params.dataB = B_data;
	params.offseta = 0;
	params.offsetb =0;
	params.offsetc=0;
	params.offsetd=0;
	params.alpha = 1;
	params.beta = 0;
	params.strideD1J = strideC1J;
	params.strideD2K = strideC2K;
	params.strideC1J = strideC1J;
	params.strideC2K = strideC2K;
	params.strideA1L = strideA1L;
	params.strideA2K = strideA2K;
	params.strideB1L = strideB1L;
	params.strideB2K = strideB2K;
	params.sizeI = size_m;
	params.sizeJ = size_n;
	params.sizeK = 1;
	params.sizeL = size_k;
	params.dataAs = As_data;
	params.dataBs = Bs_data;
	params.staggerUIter = staggerUIter;
	params.problemNumGroupTiles0 = problemNumGroupTiles0;
	params.problemNumGroupTiles1 = problemNumGroupTiles1;
	params.numFullBlocks = numFullBlocks;
	params.wgmRemainder1=wgmRemainder1;
	params.magicNumberWgmRemainder1 =magicNumberWgmRemainder1;
	size_t paramsSize = sizeof(params);

	
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static AiterAsmKernel gemm_w8a8_smooth("Cijk_Alik_Bljk_I8II_BH_MT256x256x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn2_K1_KLA_LDL1_LRVW8_MAC_MMAC_BW1_MMAC_KM0_MMAC_KME1_NLCA1_NLCB1_OLDSR0_PGR1_PLR1_RK1_SU0_SUM0_SUS256_SVW4_TT16_8_TLM0_ULDSWM0_USFGRO0_VAW1_VSn1_VW8_WSS0_WG16_32_1_WGM1", "gemm_w8a8_smooth_256x256x128_TN_FP16.co");
	gemm_w8a8_smooth.launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});

    // hipDeviceSynchronize();
    return;
}



torch::Tensor gemm_w8a8_smooth_256x256x128_TN_dispatch(torch::Tensor &A,
                                              torch::Tensor &B,
                                              torch::Tensor &As,
                                              torch::Tensor &Bs,
                                              const int &M, const int &N,
                                              const int &K,
                                              c10::optional<torch::Tensor> bias,
                                              torch::Dtype output_dtype)
{

    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(output_dtype)
                                       .device(A.device());
	torch::Tensor D = torch::empty({M, N}, options);
	const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    if (output_dtype == torch::kFloat16)
        _gemm_w8a8_smooth_256x256x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_256x256x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


torch::Tensor gemm_w8a8_smooth_256x256x128_TN_bridg(torch::Tensor &A,
                                           torch::Tensor &B,
                                           torch::Tensor &As,
                                           torch::Tensor &Bs,
                                           c10::optional<torch::Tensor> bias,
                                           torch::Dtype output_dtype, int M, int N, int K)
{

    return gemm_w8a8_smooth_256x256x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
}



torch::Tensor gemm_w8a8_smooth_asm(torch::Tensor &A,
									torch::Tensor &B,
									torch::Tensor &As,
									torch::Tensor &Bs,
									c10::optional<torch::Tensor> bias,
									torch::Dtype output_dtype,
									int M, int N, int K)
{
    return gemm_w8a8_smooth_256x256x128_TN_bridg(A, B, As, Bs, bias, output_dtype,
												  M, N, K);
}
}
} // namespace at