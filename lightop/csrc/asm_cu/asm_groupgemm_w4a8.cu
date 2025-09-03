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
  const int8_t* ptr_A;               // input
  const int8_t* ptr_B;               // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  const float* topk_weights;         // topk weights
  const int32_t* expert_offsets;     // expert ids
  const int32_t* problem_size;       // expert ids
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  void *debugBuffer;
};

namespace at {
namespace native {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType>
void _moe_groupgemm_w4a8_TN_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor problem_size,
  torch::Tensor expert_offsets, uint32_t experts_num) {

  const int size_m = input.size(0); 
  const int size_n = b_qweight.size(1); 
  const int size_k = input.size(1);

  bool first_stage = true;

  const float* topk_weights_ptr; // 第一阶段这里为null
  if (topk_weights.has_value()){
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
    first_stage = false;
  }

  // int num_pad = num_tokens_post_pad[0].item<int>();
  // int sorted_token_lens = sorted_token_ids.size(0);

  /* grid sizes */
  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  globalWorkSize[0] = DIVIDE(size_n, BLOCKN);
  // globalWorkSize[2] = std::min(DIVIDE(sorted_token_lens, BLOCKM), DIVIDE(num_pad, BLOCKM));
  globalWorkSize[1] = experts_num;
  globalWorkSize[2] = DIVIDE(problem_size.max().item<int>(), BLOCKM);

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.numWorkGroups0 = globalWorkSize[0];
  hipFunctionArgs.numWorkGroups1 = globalWorkSize[2];
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = b_qweight.data_ptr<int8_t>();
  hipFunctionArgs.ptr_B = input.data_ptr<int8_t>();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  hipFunctionArgs.topk_weights = topk_weights_ptr;
  hipFunctionArgs.expert_offsets = expert_offsets.data_ptr<int32_t>();
  hipFunctionArgs.problem_size = problem_size.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_asm = a_scale.stride(0);
  hipFunctionArgs.stride_ask = a_scale.stride(1);
  hipFunctionArgs.stride_bse = b_scale.stride(0);
  hipFunctionArgs.stride_bsn = b_scale.stride(1);
  hipFunctionArgs.stride_bsk = b_scale.stride(2);
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  int bytes = 20;
  int debug_size = 768 * bytes;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  // printf("%s:%d isBfloat16:%d \n", __FILE__, __LINE__, isBfloat16);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  char funcName[1024], first_stage_co_file[1024], second_stage_co_file[1024];
  memset(funcName, 0x0, sizeof(funcName));
  memset(first_stage_co_file, 0x0, sizeof(first_stage_co_file));
  memset(second_stage_co_file, 0x0, sizeof(second_stage_co_file));
  sprintf(funcName, "Cijk_Alik_Bljk_I8HS_BH_SABB_UserArgs_MT%dx%dx%d_SN_K1_SB2_SSA0_SSB1024_SWMNK1_8_1_TT8_16", BLOCKM, BLOCKN, BLOCKK);
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    sprintf(first_stage_co_file, "moe_w4a8_channel/moe_w4a8_%dx%dx%d_TN_BF16_FirstStage.co", BLOCKM, BLOCKN, BLOCKK);
    sprintf(second_stage_co_file, "moe_w4a8_channel/moe_w4a8_%dx%dx%d_TN_BF16_SecondStage.co", BLOCKM, BLOCKN, BLOCKK);
  } else {
    TORCH_CHECK(false, "moe_w4a8 only supports outputType in [BFloat16]");
  }

  static AiterAsmKernel moe_w4a8_first_stage(funcName, first_stage_co_file);
  static AiterAsmKernel moe_w4a8_second_stage(funcName, second_stage_co_file);
  if(first_stage){
    // moe_w4a8_first_stage.launch_kernel_ext({
    //   &hipFunctionArgs,
    //   &argsSize,
    //   globalWorkSize[0],           // gdx
    //   globalWorkSize[1],           // gdy
    //   globalWorkSize[2],           // gdz
    //   localWorkSize[0],            // bdx: 12 wv64
    //   localWorkSize[1],            // bdy
    //   localWorkSize[2],            // bdz
    //   stream
    // });
  } else {
    // moe_w4a8_second_stage.launch_kernel_ext({
    //   &hipFunctionArgs,
    //   &argsSize,
    //   globalWorkSize[0],           // gdx
    //   globalWorkSize[1],           // gdy
    //   globalWorkSize[2],           // gdz
    //   localWorkSize[0],            // bdx: 12 wv64
    //   localWorkSize[1],            // bdy
    //   localWorkSize[2],            // bdz
    //   stream
    // });
  }


  #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2]);
  // printf("\ndebug:\n");
  // hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  // using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 8 == 0) printf("\n");
    // if (i % 8 == 0) printf("\ni:%d ", i / 8);
    // printf("%3d:%10.4f\t", i, p_debug[i]);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    // printf("%3d\t", p_debug[i]);
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
  // }
  // printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}

torch::Tensor moe_groupgemm_w4a8_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor problem_size, 
  torch::Tensor expert_offsets,
  int mode) {
  
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  uint32_t experts_num = b_qweight.sizes()[0];
  if (mode == 1000) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "moe_groupgemm_w4a8_TN_impl",[&] {
        _moe_groupgemm_w4a8_TN_impl<128, 256, 128, scalar_t>(
          input, b_qweight, output, a_scale, b_scale,
          topk_weights, problem_size, expert_offsets, experts_num);
      }
    );
  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
  return output;
}

} // namespace native
} // namespace at