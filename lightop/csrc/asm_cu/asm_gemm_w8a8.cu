#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>


#include "aiter_hip_common.h"

#define DEBUG 0

using half_t = __half;
using bhalf_t = __hip_bfloat16;

enum GRAD_TYPE {
  nograd,
  wgrad,
  xgrad
};

/* module function args */
template <typename T>
struct HipFunctionArgs {
  // Size of Tensor's packed dims, in elements
  uint32_t gemm_count;
  uint32_t internalArgs;
  uint32_t internalArgs1;
  unsigned int numWorkGroups;
  unsigned int size0;
  unsigned int size1;
  unsigned int size2;
  unsigned int size3;
  T *dataD;
  T *dataC;
  const int8_t *dataA;
  const int8_t *dataB;
  unsigned int strideD1;
  unsigned int strideD2;
  unsigned int strideC1;
  unsigned int strideC2;
  unsigned int strideA1;
  unsigned int strideA2;
  unsigned int strideB1;
  unsigned int strideB2;
  float alpha;
  float beta;
  const float *scaleA;
  const float *scaleB;
  void *debugBuffer;
};

namespace at {
namespace native {

template <typename OutputType>
void _gemm_w8a8_128x256x128_TN_impl(
  const int8_t *A_data, int8_t *B_data,
  float *As_data, float *Bs_data, // scales
  OutputType *bias_data,
  OutputType *C_data,
  OutputType *D_data,
  const int &size_m, const int &size_n, const int &size_k,
  bool isBfloat16, GRAD_TYPE gradType = nograd) {
  
  size_t strideC1 = size_n;
  size_t strideC2 = 0;
  size_t strideA1 = gradType == xgrad ? size_n : size_k;
  size_t strideA2 = 0;
  size_t strideB1 = size_k;
  size_t strideB2 = 0;

  /* grid sizes */
  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  unsigned int totalWorkGroups0 = size_n / 128;
  unsigned int totalWorkGroups1 = size_m / 256;
  if (totalWorkGroups0*128 < size_n) { totalWorkGroups0++; }
  if (totalWorkGroups1*256 < size_m) { totalWorkGroups1++; }
  uint32_t numWorkGroups = totalWorkGroups0 * totalWorkGroups1;
  globalWorkSize[0] = numWorkGroups * globalWorkSize[0];

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.gemm_count = 1;
  hipFunctionArgs.internalArgs = 1;
  hipFunctionArgs.internalArgs1 = 16;
  hipFunctionArgs.numWorkGroups = numWorkGroups;
  hipFunctionArgs.size0 = size_n;
  hipFunctionArgs.size1 = size_m;
  hipFunctionArgs.size2 = 1;
  hipFunctionArgs.size3 = size_k;
  hipFunctionArgs.dataD = D_data;
  hipFunctionArgs.dataC = C_data;
  hipFunctionArgs.dataA = B_data;
  hipFunctionArgs.dataB = A_data;
  hipFunctionArgs.strideD1 = strideC1;
  hipFunctionArgs.strideD2 = strideC2;
  hipFunctionArgs.strideC1 = strideC1;
  hipFunctionArgs.strideC2 = strideC2;
  hipFunctionArgs.strideA1 = strideA1;
  hipFunctionArgs.strideA2 = strideA2;
  hipFunctionArgs.strideB1 = strideB1;
  hipFunctionArgs.strideB2 = strideB2;
  hipFunctionArgs.alpha = 1.0;
  hipFunctionArgs.beta = 0.0;
  hipFunctionArgs.scaleA = Bs_data;
  hipFunctionArgs.scaleB = As_data;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  int debug_size = 768 * 4;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  // printf("%s:%d top_k:%d topk:%f\n", __FILE__, __LINE__, top_k, hipFunctionArgs.topk);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  string fp16_co_file = "gemm_w8a8_block/gemm_w8a8_128x256x128_TN_FP16.co";
  string bf16_co_file = "gemm_w8a8_block/gemm_w8a8_128x256x128_TN_BF16.co";
  if (gradType == wgrad) {
    fp16_co_file = "gemm_w8a8_block/blockwise_w8a8_gemm_BF16_wgrad_scale_T.co";
    bf16_co_file = "gemm_w8a8_block/blockwise_w8a8_gemm_BF16_wgrad_scale_T.co";
  } else if (gradType == xgrad) {
    fp16_co_file = "gemm_w8a8_block/gemm_w8a8_128x256x128_NN_FP16.co";
    bf16_co_file = "gemm_w8a8_block/gemm_w8a8_128x256x128_NN_BF16.co";
  }
	static AiterAsmKernel gemm_w8a8_fp16("Cijk_Alik_Bljk_I8HS_BH_SABB_UserArgs_MT128x256x128_SN_K1_SB2_SSA0_SSB1024_SWMNK1_8_1_TT8_16", fp16_co_file.c_str());
	static AiterAsmKernel gemm_w8a8_bf16("Cijk_Alik_Bljk_I8HS_BH_SABB_UserArgs_MT128x256x128_SN_K1_SB2_SSA0_SSB1024_SWMNK1_8_1_TT8_16", bf16_co_file.c_str());
	AiterAsmKernel *impl_ptr = &gemm_w8a8_fp16;
  if (isBfloat16) 
    impl_ptr = &gemm_w8a8_bf16;
	impl_ptr->launch_kernel_ext({
    &hipFunctionArgs,
    &argsSize,
    globalWorkSize[0],           // gdx
    globalWorkSize[1],           // gdy
    globalWorkSize[2],           // gdz
    localWorkSize[0],            // bdx: 12 wv64
    localWorkSize[1],            // bdy
    localWorkSize[2],            // bdz
    stream
  });
    #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2]);
  printf("\ndebug:\n");
  hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / 2; ++i) {
    if (i % 16 == 0) printf("\n");
    // if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // printf("%3d:%10.4f\t", i, p_debug[i]);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    printf("%3d:%6d\t", i, p_debug[i]);
    // printf("%3d\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = 2 * i;
    // printf("i:%d %10.4f %6d\t", i, p_debug[index], *reinterpret_cast<uint32_t*>(p_debug + index + 1));
    // printf("i:%d bidz:%d sa:%10.4f sb:%10.4f v:%10.4f\t", i, *reinterpret_cast<int32_t*>(p_debug+index), p_debug[index+1], p_debug[index+2], p_debug[index+3]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + index + 4);
    // printf("i:%d scale:%10.4f %10.4f value:%10.4f %10.4f out:%10.4f %10.4f\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], static_cast<float>(tmp[0]), static_cast<float>(tmp[2])
    //   );
  }
  printf("\n");
  #endif

  return;
}

torch::Tensor gemm_w8a8_asm(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &As,
  torch::Tensor &Bs,
  c10::optional<torch::Tensor> &C,
  c10::optional<torch::Tensor> &D_,
  c10::optional<torch::Tensor> bias,
  torch::Dtype output_dtype, int M, int N, int K, int gradType = 0) {
  
  torch::Tensor D;
  if (D_.has_value()) {
    D = D_.value();
  } else {
    torch::TensorOptions options = torch::TensorOptions().dtype(output_dtype).device(A.device());
    D = torch::empty({M, N}, options);
  }
	const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  if (output_dtype == torch::kFloat16)
    _gemm_w8a8_128x256x128_TN_impl<half_t>(
      A.data_ptr<int8_t>(),
      B.data_ptr<int8_t>(),
      As.data_ptr<float>(),
      Bs.data_ptr<float>(),
      nullptr,
      C.has_value() ? reinterpret_cast<half_t *>(C.value().data_ptr()) : nullptr,
      reinterpret_cast<half_t *>(D.data_ptr()),
      M, N, K, 0, static_cast<GRAD_TYPE>(gradType));
  else if (output_dtype == torch::kBFloat16)
    _gemm_w8a8_128x256x128_TN_impl<bhalf_t>(
      A.data_ptr<int8_t>(),
      B.data_ptr<int8_t>(),
      As.data_ptr<float>(),
      Bs.data_ptr<float>(),
      nullptr,
      C.has_value() ? reinterpret_cast<bhalf_t *>(C.value().data_ptr()) : nullptr,
      reinterpret_cast<bhalf_t *>(D.data_ptr()),
      M, N, K, 1, static_cast<GRAD_TYPE>(gradType));
  else
    TORCH_CHECK(false, "gemm_w8a8 only supports outputType in [Float16, BFloat16]");
  return D;
}

} // namespace native
} // namespace at