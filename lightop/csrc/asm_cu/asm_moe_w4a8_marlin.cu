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
  T* ptr_C;                   // output
  const int32_t* ptr_A;                    // input
  const int8_t* ptr_B0;                   // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  const float* topk_weights;         // topk weights
  const int32_t* sorted_token_ids;   // sorted token ids
  const int32_t* expert_ids;         // expert ids
  const int32_t* num_tokens_post_pad_ptr;// num tokens after padding
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bn;               // weight scale stride n
  uint32_t stride_bk;               // weight scale stride k
  uint32_t topk;                     // top k
  float topk_rcip;                   // top k
  uint32_t numFullBlocks;
  uint32_t wgmRemainder1;
  uint32_t magicNumberWgmRemainder1;
  void *debugBuffer;
};

namespace at {
namespace native {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int IS_WGM8 = false>
void _moe_w4a8_marlin_TN_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  uint32_t top_k, uint32_t experts_num) {

  const int size_m = input.size(0); 
  const int size_n = b_qweight.size(2) / 4; 
  const int size_k = input.size(1);

  bool first_stage = true;

  const float* topk_weights_ptr; // 第一阶段这里为null
  if (topk_weights.has_value()){
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
    first_stage = false;
  }

  // int num_pad = num_tokens_post_pad[0].item<int>();
  int sorted_token_lens = sorted_token_ids.size(0);

  /* grid sizes */
  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  globalWorkSize[0] = DIVIDE(size_n, BLOCKN);
  // globalWorkSize[2] = std::min(DIVIDE(sorted_token_lens, BLOCKM), DIVIDE(num_pad, BLOCKM));
  globalWorkSize[2] = DIVIDE(sorted_token_lens, BLOCKM);

  uint32_t workgroupmapping = 8;
  uint32_t numFullBlocks = globalWorkSize[2] / workgroupmapping;
  uint32_t wgmRemainder1 = globalWorkSize[2] % workgroupmapping;
  if (wgmRemainder1 = 0) wgmRemainder1 = workgroupmapping;

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.numWorkGroups0 = globalWorkSize[0];
  hipFunctionArgs.numWorkGroups1 = globalWorkSize[2];
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = b_qweight.data_ptr<int32_t>();
  hipFunctionArgs.ptr_B0 = input.data_ptr<int8_t>();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  hipFunctionArgs.topk_weights = topk_weights_ptr;
  hipFunctionArgs.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
  hipFunctionArgs.expert_ids = expert_ids.data_ptr<int32_t>();
  hipFunctionArgs.num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_asm = a_scale.stride(0);
  hipFunctionArgs.stride_ask = a_scale.stride(1);
  hipFunctionArgs.stride_bse = b_scale.stride(0);
  hipFunctionArgs.stride_bn = b_qweight.size(2) * 4;
  hipFunctionArgs.stride_bk = b_qweight.size(1);
  hipFunctionArgs.topk = top_k;
  hipFunctionArgs.topk_rcip = 1.0 / top_k;
  hipFunctionArgs.numFullBlocks = numFullBlocks;
  hipFunctionArgs.wgmRemainder1 = wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 = ((1L<<31) / wgmRemainder1 + 1);;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  int bytes = 128;
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
    sprintf(first_stage_co_file, "moe_w4a8_channel/moe_w4a8_marlin_%dx%dx%d_TN_BF16_%sFirstStage.co", BLOCKM, BLOCKN, BLOCKK, IS_WGM8?"WGM8_":"");
    sprintf(second_stage_co_file, "moe_w4a8_channel/moe_w4a8_marlin_%dx%dx%d_TN_BF16_%sSecondStage.co", BLOCKM, BLOCKN, BLOCKK, IS_WGM8?"WGM8_":"");
  } else if (output.scalar_type() == at::ScalarType::Float) { // debug
    sprintf(first_stage_co_file, "moe_w4a8_channel/moe_w4a8_marlin_%dx%dx%d_TN_BF16_FirstStage.co", BLOCKM, BLOCKN, BLOCKK);
    sprintf(second_stage_co_file, "moe_w4a8_channel/moe_w4a8_marlin_%dx%dx%d_TN_BF16_SecondStage.co", BLOCKM, BLOCKN, BLOCKK);
  } else {
    TORCH_CHECK(false, "moe_w4a8_marlin only supports outputType in [BFloat16]");
  }

  static AiterAsmKernel moe_w4a8_marlin_first_stage(funcName, first_stage_co_file);
  static AiterAsmKernel moe_w4a8_marlin_second_stage(funcName, second_stage_co_file);
  AiterAsmKernel *impl_ptr = &moe_w4a8_marlin_first_stage;
  if(!first_stage){
    impl_ptr = &moe_w4a8_marlin_second_stage;
  }
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
  printf("argsSize:%ld\nnumWorkGroups0:%p\nnumWorkGroups1:%p\nptr_C:%p\nptr_A:%p\ptr_B0:%p\nptr_A_scale:%p\nptr_B_scale:%p\ntopk_weights:%p\sorted_token_ids:%p\n"
    "expert_ids:%pnum_tokens_post_pad_ptr:%p\nexperts_num:%p\nsize_m:%p\nsize_n:%p\nsize_k:%p\nstride_asm:%p\nstride_ask:%p\nstride_bse:%p\stride_bn:%p\n"
    "stride_bk:%p\ntopk:%p\ntopk_rcip:%p\nnumFullBlocks:%p\nwgmRemainder1:%p\nmagicNumberWgmRemainder1:%p\ndebugBuffer%p\n", argsSize,
    &hipFunctionArgs.numWorkGroups0, &hipFunctionArgs.numWorkGroups1, &hipFunctionArgs.ptr_C,
    &hipFunctionArgs.ptr_A, &hipFunctionArgs.ptr_B0, &hipFunctionArgs.ptr_A_scale,
    &hipFunctionArgs.ptr_B_scale, &hipFunctionArgs.topk_weights, &hipFunctionArgs.sorted_token_ids, 
    &hipFunctionArgs.expert_ids, &hipFunctionArgs.num_tokens_post_pad_ptr, &hipFunctionArgs.experts_num, &hipFunctionArgs.size_m,
    &hipFunctionArgs.size_n, &hipFunctionArgs.size_k, &hipFunctionArgs.stride_asm, 
    &hipFunctionArgs.stride_ask, &hipFunctionArgs.stride_bse, &hipFunctionArgs.stride_bn,
    &hipFunctionArgs.stride_bk, &hipFunctionArgs.topk, &hipFunctionArgs.topk_rcip, &hipFunctionArgs.numFullBlocks, 
    &hipFunctionArgs.wgmRemainder1, &hipFunctionArgs.magicNumberWgmRemainder1, &hipFunctionArgs.debugBuffer
  );
  printf("\ndebug:\n");
  hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 16 == 0) printf("\n");
    // if (i % 32 == 0) printf("\ni:%d ", i / 32);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f\t", p_debug[i]);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    // printf("%6d\t", p_debug[i]);
    // printf("%10d\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = split * i;
    // // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + index + 8);
    // printf("tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], p_debug[index+4 ], p_debug[index+5 ], p_debug[index+6 ]
    // );
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + index);
    // printf("tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", i, 
    //   static_cast<float>(tmp[0]), static_cast<float>(tmp[2]), static_cast<float>(tmp[4]), static_cast<float>(tmp[6]), 
    //   static_cast<float>(tmp[8]), static_cast<float>(tmp[10]), static_cast<float>(tmp[12]), static_cast<float>(tmp[14])
    // );
    // printf("tid:%d %6d %6d %10.4f\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], *reinterpret_cast<float*>(p_debug + index + 2)
    // );
    // int8_t * tmp = reinterpret_cast<int8_t*>(p_debug + index);
    // printf("tid:%d %6d 0X%08X 0X%08X 0X%08X 0X%08X "
    //   "%3d %3d %3d %3d %3d %3d %3d %3d\n", i, 
    //   p_debug[index+0 ], 
    //   p_debug[index+0 ], p_debug[index+1 ], 
    //   p_debug[index+2 ], p_debug[index+3 ], 
    //   tmp[8], tmp[9], tmp[10], tmp[11],
    //   tmp[12], tmp[13], tmp[14], tmp[15]
    // );
  }
  printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}

torch::Tensor moe_w4a8_marlin_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  uint32_t top_k,
  int mode) {
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  uint32_t experts_num = b_qweight.sizes()[0];
  if (mode == 1000) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "moe_w4a8_marlin_TN_impl",[&] {
        _moe_w4a8_marlin_TN_impl<128, 512, 128, scalar_t>(
          input, b_qweight, output, a_scale, b_scale,
          topk_weights, sorted_token_ids, expert_ids,
          num_tokens_post_pad, top_k, experts_num);
      }
    );
  } else if (mode == 1001) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "moe_w4a8_marlin_TN_impl",[&] {
        _moe_w4a8_marlin_TN_impl<128, 512, 128, scalar_t, true>(
          input, b_qweight, output, a_scale, b_scale,
          topk_weights, sorted_token_ids, expert_ids,
          num_tokens_post_pad, top_k, experts_num);
      }
    );
  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
  return output;
}

} // namespace native
} // namespace at