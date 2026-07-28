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
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16 = false)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
    // static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT256x256x128_TT16x8_WG16x32x2_Squant", "gemm_w8a8_smooth_256x256x128_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT256x256x128_TT16x8_WG16x32x2_Squant", "gemm_w8a8_smooth_256x256x128_TN_BF16.co");
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT256x256x128_TT16x8_WG16x32x2_Squant", "gemm_w8a8_smooth_256x256x128_WGM8_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT256x256x128_TT16x8_WG16x32x2_Squant", "gemm_w8a8_smooth_256x256x128_WGM8_TN_BF16.co");
	
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;

	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
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
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_256x256x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}



template <typename OutputType>
void _gemm_w8a8_smooth_256x128x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 16, 4 };
	const unsigned int groupSize[2] = { 16, 32 };
	size_t localWorkSize[3] = { 512, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT256x128x128_TT16x8_WG16x16_Squant", "gemm_w8a8_smooth_256x128x128_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT256x128x128_TT16x8_WG16x16_Squant", "gemm_w8a8_smooth_256x128x128_TN_BF16.co");

	// static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT256x128x128_TT16x8_WG16x16_Squant", "gemm_w8a8_smooth_256x128x128_WGM8_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT256x128x128_TT16x8_WG16x16_Squant", "gemm_w8a8_smooth_256x128x128_WGM8_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_256x128x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_256x128x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_256x128x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_128x128x64_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 8, 8 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT128x128x64_TT8x8_WG16x16_Squant", "gemm_w8a8_smooth_128x128x64_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT128x128x64_TT8x8_WG16x16_Squant", "gemm_w8a8_smooth_128x128x64_TN_BF16.co");

	// static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT128x128x64_TT8x8_WG16x16_Squant", "gemm_w8a8_smooth_128x128x64_WGM8_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT128x128x64_TT8x8_WG16x16_Squant", "gemm_w8a8_smooth_128x128x64_WGM8_TN_BF16.co");

	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_128x128x64_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_128x128x64_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_128x128x64_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_64x64x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 4, 4 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_64x64x128_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_64x64x128_TN_BF16.co");

	// static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_WGM8_64x64x128_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_WGM8_64x64x128_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_64x64x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_64x64x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_64x64x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}

template <typename OutputType>
void _gemm_w8a8_smooth_64x64x128_TN_WARPN4_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 4, 4 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_64x64x128_WGM1_TN_FP16_WARPM1_WARPK1_WARPN4.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_64x64x128_WGM1_TN_BF16_WARPM1_WARPK1_WARPN4.co");

	// static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_WGM8_64x64x128_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x64x128_TT4x4_WG16x16_Squant", "gemm_w8a8_smooth_WGM8_64x64x128_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}


torch::Tensor gemm_w8a8_smooth_64x64x128_TN_WARPN4_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_64x64x128_TN_WARPN4_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_64x64x128_TN_WARPN4_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_64x96x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 4, 6 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x96x128_TT4x6_WG16x16_Squant", "gemm_w8a8_smooth_64x96x128_WGM8_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x96x128_TT4x6_WG16x16_Squant", "gemm_w8a8_smooth_64x96x128_WGM8_TN_BF16.co");

	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}


torch::Tensor gemm_w8a8_smooth_64x96x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_64x96x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_64x96x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_64x32x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 4, 2 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_64x32x128_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_64x32x128_TN_BF16.co");

	// static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT64x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_64x32x128_WGM8_TN_FP16.co");
	// static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT64x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_64x32x128_WGM8_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_64x32x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_64x32x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_64x32x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_32x32x256_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 2, 2 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT32x32x256_TT2x2_WG16x16_Squant", "gemm_w8a8_smooth_32x32x256_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT32x32x256_TT2x2_WG16x16_Squant", "gemm_w8a8_smooth_32x32x256_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_32x32x256_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_32x32x256_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_32x32x256_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_128x32x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 8, 2 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT128x32x128_TT8x2_WG16x16_Squant", "gemm_w8a8_smooth_128x32x128_WGM8_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT128x32x128_TT8x2_WG16x16_Squant", "gemm_w8a8_smooth_128x32x128_WGM8_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_128x32x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_128x32x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_128x32x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}


template <typename OutputType>
void _gemm_w8a8_smooth_128x96x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{


	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 8, 6 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT128x96x128_TT8x6_WG16x16_Squant", "gemm_w8a8_smooth_128x96x128_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT128x96x128_TT8x6_WG16x16_Squant", "gemm_w8a8_smooth_128x96x128_TN_BF16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;
}



torch::Tensor gemm_w8a8_smooth_128x96x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_128x96x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_128x96x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}

template <typename OutputType>
void _gemm_w8a8_smooth_32x32x128_TN_impl(
    const int8_t *A_data, int8_t *B_data,
    float *As_data, float *Bs_data, // scales
    OutputType *bias_data,
    OutputType *D_data,
    const int &size_m, const int &size_n, const int &size_k,
    bool isBfloat16)
{
	size_t lda=size_k;
	size_t ldb=size_k;
	size_t ldc=size_m;
	size_t stridea=lda;
	size_t strideb=ldb;
	size_t stridec=ldc;
	int workgroupmapping = 8;	//
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
	const unsigned int threadTile[2] = { 2, 2 };
	const unsigned int groupSize[2] = { 16, 16 };
	size_t localWorkSize[3] = { 256, 1, 1 };
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
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
	tensor2dSizeA = tensor2dSizeAStride;
	tensor2dSizeAStride = std::max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
	tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
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
	static AiterAsmKernel gemm_w8a8_smooth_fp16("Cijk_Alik_Bljk_I8II_BH_MT32x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_32x32x128_WGM8_TN_FP16.co");
	static AiterAsmKernel gemm_w8a8_smooth_bf16("Cijk_Alik_Bljk_I8II_BH_MT32x32x128_TT4x2_WG16x16_Squant", "gemm_w8a8_smooth_32x32x128_WGM8_TN_BP16.co");
	AiterAsmKernel *impl_ptr = &gemm_w8a8_smooth_fp16;
    if (isBfloat16) 
        impl_ptr = &gemm_w8a8_smooth_bf16;
	impl_ptr->launch_kernel({&params,
                             &paramsSize,
                             globalWorkSize[0][0],           // gdx
                             globalWorkSize[0][1],           // gdy
                             globalWorkSize[0][2],           // gdz
                             localWorkSize[0],               // bdx: 12 wv64
                             localWorkSize[1],               // bdy
                             localWorkSize[2],               // bdz
                             stream});
    return;

}



torch::Tensor gemm_w8a8_smooth_32x32x128_TN_dispatch(torch::Tensor &A,
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
        _gemm_w8a8_smooth_32x32x128_TN_impl<half_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
                                               nullptr,
                                               reinterpret_cast<half_t *>(D.data_ptr()),
                                               N, M, K, 0);
    else if (output_dtype == torch::kBFloat16)
        _gemm_w8a8_smooth_32x32x128_TN_impl<bhalf_t>(B.data_ptr<int8_t>(),
                                               A.data_ptr<int8_t>(),
                                               Bs.data_ptr<float>(),
                                               As.data_ptr<float>(),
											nullptr,
											reinterpret_cast<bhalf_t *>(D.data_ptr()),
											N, M, K, 1);

    else
        TORCH_CHECK(false, "gemm_w8a8_smooth only supports outputType in [Float16, BFloat16]");

    return D;
}

enum class Tunning_slide_block : uint8_t { // 使用 8 位无符号整数
    B_32x32x128 = 0,
    B_32x32x256 = 1,
    B_64x32x128 = 2,
	B_64x64x128 = 3,
	B_128x32x128 = 4,
	B_128x96x128 = 5,
	B_128x128x64 = 6,
	B_256x128x128 = 7,
	B_256x256x128 = 8,
	B_64x64x128_WARPN4 = 9,
	B_64x96x128 = 10
};

torch::Tensor gemm_w8a8_smooth_asm_bridg(torch::Tensor &A,
                                           torch::Tensor &B,
                                           torch::Tensor &As,
                                           torch::Tensor &Bs,
                                           c10::optional<torch::Tensor> bias,
                                           torch::Dtype output_dtype, int M, int N, int K,
										   bool isTunning_slide_block, int tunning_slide_block)
{

	if(isTunning_slide_block){
		switch (static_cast<Tunning_slide_block>(tunning_slide_block)) {
			case Tunning_slide_block::B_32x32x128:
				return gemm_w8a8_smooth_32x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_32x32x256:
				return gemm_w8a8_smooth_32x32x256_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_64x32x128:
				return gemm_w8a8_smooth_64x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_64x64x128:
				return gemm_w8a8_smooth_64x64x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_128x32x128:
				return gemm_w8a8_smooth_128x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_128x96x128:
				return gemm_w8a8_smooth_128x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_128x128x64:
				return gemm_w8a8_smooth_128x128x64_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_256x128x128:
				return gemm_w8a8_smooth_256x128x128_TN_dispatch(A,B ,As,Bs,M,N,K,bias ,output_dtype);
			case Tunning_slide_block::B_256x256x128:
				return gemm_w8a8_smooth_256x256x128_TN_dispatch(A,B ,As,Bs,M,N,K,bias ,output_dtype);
			case Tunning_slide_block::B_64x64x128_WARPN4:
				return gemm_w8a8_smooth_64x64x128_TN_WARPN4_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			case Tunning_slide_block::B_64x96x128:
				return gemm_w8a8_smooth_64x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			default:
				TORCH_CHECK(false,"Unsupported tunning slide block type: ", tunning_slide_block);
		}
	}else{
		static int BW_CUS = 80;
		int numBlocks_32x32asm = (N + 32 -1)/32 * ((M + 32 -1)/32);
		int numBlocks_64x32asm = (N + 64 -1)/64 * ((M + 32 -1)/32);
		int numBlocks_64x64asm = (N + 64 -1)/64 * ((M + 64 -1)/64);
		int numBlocks_64x96asm = (N + 64 -1)/64 * ((M + 96 -1)/96);
		int numBlocks_128x32asm = (N + 128 -1)/128 * ((M + 32 -1)/32);
		int numBlocks_128x96asm = (N + 128 -1)/128 * ((M + 96 -1)/96);
		int numBlocks_128x128asm = (N + 128 -1)/128 *((M + 128 -1)/128);
		int numBlocks_256x128asm = (N + 256 -1)/256 * ((M + 128 -1)/128);
		int numBlocks_256x256asm = (N + 256 -1)/256 * ((M + 256 -1)/256);
		
		if(K == 128){
			if(M<=64){
				return gemm_w8a8_smooth_128x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			}else if(M<=256){
				return gemm_w8a8_smooth_128x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			}else{
				if(numBlocks_32x32asm < BW_CUS)
					return gemm_w8a8_smooth_32x32x256_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				if(numBlocks_64x32asm < BW_CUS)
					return gemm_w8a8_smooth_64x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				if(numBlocks_64x64asm < BW_CUS)
					return gemm_w8a8_smooth_64x64x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				if(numBlocks_128x96asm < BW_CUS)
					return gemm_w8a8_smooth_128x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				if(numBlocks_128x128asm < BW_CUS)
					return gemm_w8a8_smooth_128x128x64_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				if(numBlocks_256x128asm < BW_CUS*1.5)
					return gemm_w8a8_smooth_256x128x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				return gemm_w8a8_smooth_256x256x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			}
		}else{
			if(numBlocks_32x32asm < BW_CUS)
				return gemm_w8a8_smooth_32x32x256_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			if(numBlocks_64x32asm < BW_CUS)
				return gemm_w8a8_smooth_64x32x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			if(numBlocks_64x64asm < BW_CUS) {
				// if ((M >= 16 && M <=64) || (M > 128 && M < 256)  && N >= 4096)
				// 	return gemm_w8a8_smooth_64x64x128_TN_WARPN4_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
				// else 
					return gemm_w8a8_smooth_64x64x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			}
			if(numBlocks_64x96asm < BW_CUS)
				return gemm_w8a8_smooth_64x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			if(numBlocks_128x96asm < BW_CUS)
				return gemm_w8a8_smooth_128x96x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			if(numBlocks_128x128asm < BW_CUS)
				return gemm_w8a8_smooth_128x128x64_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			if(numBlocks_256x128asm < BW_CUS*1.5)
				return gemm_w8a8_smooth_256x128x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
			return gemm_w8a8_smooth_256x256x128_TN_dispatch(A, B, As, Bs, M, N, K, bias, output_dtype);
		}
	}
	
}



torch::Tensor gemm_w8a8_smooth_asm(torch::Tensor &A,
									torch::Tensor &B,
									torch::Tensor &As,
									torch::Tensor &Bs,
									c10::optional<torch::Tensor> bias,
									torch::Dtype output_dtype,
									int M, int N, int K,
									bool isTunning_slide_block, int tunning_slide_block)
{
    return gemm_w8a8_smooth_asm_bridg(A, B, As, Bs, bias, output_dtype,
												  M, N, K, isTunning_slide_block, tunning_slide_block);
}
}
} // namespace at