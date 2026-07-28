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

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

/* module function args */
template <typename T>
struct HipFunctionArgs {
  uint32_t numWorkGroups0;
  uint32_t numWorkGroups1;
  T* ptr_C;                          // output
  const char* ptr_A;              // input
  const char* ptr_B;               // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  // const float* topk_weights;         // topk weights
  // const int32_t* expert_offsets;     // null
  const int32_t* masked_m;       // tokens per expert
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_ase;               // weight scale stride expert
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  uint32_t numFullBlocks;
  uint32_t wgmRemainder1;
  uint32_t magicNumberWgmRemainder1;
  void *debugBuffer;
};

namespace at {
namespace native {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  uint32_t expected_m_per_group, 
  uint32_t experts_num) {

  const int size_m = input.size(1); 
  const int size_n = b_qweight.size(1) * 16;
  const int size_k = input.size(2);
  const int workgroupmapping = 8;	//



  /* grid sizes */
  size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  auto need_size_m = std::min(size_m, expected_m_per_group*1.3);
  globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
  globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
  globalWorkSize[2] = experts_num;

  unsigned int problemNumGroupTiles0 = globalWorkSize[0];
  unsigned int problemNumGroupTiles1 = globalWorkSize[1];
  const unsigned smallNumMagicShift = 31; // bozo, review
  unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
  unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
  if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
  unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.numWorkGroups0 = globalWorkSize[0];
  hipFunctionArgs.numWorkGroups1 = globalWorkSize[1];
  
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = (const char*)b_qweight.data_ptr();
  hipFunctionArgs.ptr_B = (const char*)input.data_ptr();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  // hipFunctionArgs.topk_weights = nullptr;
  // hipFunctionArgs.expert_offsets = nullptr;
  hipFunctionArgs.masked_m = masked_m.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_ase = b_scale.stride(0);
  hipFunctionArgs.stride_asm = b_scale.stride(1);
  hipFunctionArgs.stride_ask = b_scale.stride(2);
  hipFunctionArgs.stride_bse = a_scale.stride(0);
  hipFunctionArgs.stride_bsn = a_scale.stride(1);
  hipFunctionArgs.stride_bsk = a_scale.stride(2);
  hipFunctionArgs.numFullBlocks  = numFullBlocks;
  hipFunctionArgs.wgmRemainder1  = wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  std::cout << "_m_grouped_marlin_fp8_gemm_nt_masked_asm_impl "
          << " size_m: " << size_m
          << " need_size_m: " << need_size_m
          << " size_n: " << size_n
          << " size_k: " << size_k
          << " expected_m_per_group: " << expected_m_per_group
          << " experts_num: " << experts_num
          << std::endl;
  int bytes = 64;
  int debug_size = BLOCKDIM * bytes;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  char funcName[1024], fp8_groupgemm_co_file[1024];
  memset(funcName, 0x0, sizeof(funcName));
  memset(fp8_groupgemm_co_file, 0x0, sizeof(fp8_groupgemm_co_file));
  
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    sprintf(funcName, "DEEPGEMM_FP8_I8_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_MASKED", BLOCKM, BLOCKN, BLOCKK);
    sprintf(fp8_groupgemm_co_file, "deepgemm_channel/fp8/deepgemm_groupgemm_masked_fp8_marlin_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
  } else {
    TORCH_CHECK(false, "moe_groupgemm_fp8 only supports outputType in [BFloat16]");
  }

  static AiterAsmKernel moe_groupgemm_marlin_fp8(funcName, fp8_groupgemm_co_file);
  moe_groupgemm_marlin_fp8.launch_kernel_ext({
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
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d argsSize:%ld\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2], argsSize);
  printf("numWorkGroups0:%p\nnumWorkGroups1:%p\nptr_C:%p\nptr_A:%p\nptr_B:%p\nptr_A_scale:%p\nptr_B_scale:%p\nproblem_size:%p\nexperts_num:%p\n"
    "size_m:%p\nsize_n:%p\nsize_k:%p\nstride_ase:%p\nstride_asm:%p\nstride_ask:%p\nstride_bse:%p\nstride_bsn:%p\nstride_bsk:%p\nnumFullBlocks:%p\nwgmRemainder1:%p\nmagicNumberWgmRemainder1:%p\ndebugBuffer%p\n",
    &hipFunctionArgs.numWorkGroups0, &hipFunctionArgs.numWorkGroups1, &hipFunctionArgs.ptr_C,
    &hipFunctionArgs.ptr_A, &hipFunctionArgs.ptr_B, &hipFunctionArgs.ptr_A_scale,
    &hipFunctionArgs.ptr_B_scale,// &hipFunctionArgs.topk_weights, &hipFunctionArgs.expert_offsets, 
    &hipFunctionArgs.masked_m, &hipFunctionArgs.experts_num, &hipFunctionArgs.size_m,
    &hipFunctionArgs.size_n, &hipFunctionArgs.size_k, &hipFunctionArgs.stride_ase, &hipFunctionArgs.stride_asm, 
    &hipFunctionArgs.stride_ask, &hipFunctionArgs.stride_bse, &hipFunctionArgs.stride_bsn,
    &hipFunctionArgs.stride_bsk, &hipFunctionArgs.numFullBlocks, &hipFunctionArgs.wgmRemainder1, &hipFunctionArgs.magicNumberWgmRemainder1, &hipFunctionArgs.debugBuffer
  );
  printf("\ndebug:\n");
  hipMemcpy(debug_host, hipFunctionArgs.debugBuffer, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 8 == 0) printf("\n");
    if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    printf("%6d\t", p_debug[i]);
    // printf("%10d\t", p_debug[i]);
    // printf("%10.4f\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = split * i;
    // printf("tid:%d %3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   "%3d %3d %3d %3d %3d %3d %3d %3d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ],
    //   p_debug[index+4 ], p_debug[index+5 ], p_debug[index+6 ], p_debug[index+7 ],
    //   p_debug[index+8 ], p_debug[index+9 ], p_debug[index+10], p_debug[index+11],
    //   p_debug[index+12], p_debug[index+13], p_debug[index+14], p_debug[index+15]
    //   // ,
    //   // p_debug[index+16], p_debug[index+17], p_debug[index+18], p_debug[index+19],
    //   // p_debug[index+20], p_debug[index+21], p_debug[index+22], p_debug[index+23],
    //   // p_debug[index+24], p_debug[index+25], p_debug[index+26], p_debug[index+27],
    //   // p_debug[index+28], p_debug[index+29], p_debug[index+30], p_debug[index+31]
    // );
    // printf("tid:%d %6d %6d %6d %6d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ]
    // );
    // int8_t * tmp = reinterpret_cast<int8_t*>(p_debug + index);
    // printf("tid:%d %3d %3d %3d %3d 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X %3d %3d %3d %3d\n", i, 
    //   tmp[0], tmp[1], tmp[2], tmp[3],
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], p_debug[index+4 ], p_debug[index+5 ], 
    //   tmp[20], tmp[21], tmp[22], tmp[23] 
    // );
  }
  printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}

torch::Tensor m_grouped_marlin_fp8_gemm_nt_masked_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m, 
  int expected_m_per_group,
  int mode) {
  
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  uint32_t experts_num = b_qweight.sizes()[0];
  if (mode == 1000) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
        _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 256, 128, scalar_t>(
          input, b_qweight, output, a_scale, b_scale,
          masked_m, expected_m_per_group, experts_num);
      }
    );
  } else if(mode == 1001) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
        _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 128, 128, scalar_t, 768>(
          input, b_qweight, output, a_scale, b_scale,
          masked_m, expected_m_per_group, experts_num);
      }
    );
  }else if(mode == 1002) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
        _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 64, 128, scalar_t, 512>(
          input, b_qweight, output, a_scale, b_scale,
          masked_m, expected_m_per_group, experts_num);
      }
    );
  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
  return output;
}

} // namespace native
} // namespace at