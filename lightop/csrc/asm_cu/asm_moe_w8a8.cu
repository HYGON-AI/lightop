#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>


#include "aiter_hip_common.h"


using half_t = __half;
using bhalf_t = __hip_bfloat16;

#define BLOCKM 128
#define BLOCKN 256
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

/* module function args */
template <typename T>
struct HipFunctionArgs {
  uint32_t numWorkGroups0;
  uint32_t numWorkGroups1;
  T* ptr_C;                   // output
  const int8_t* ptr_A;                    // input
  const int8_t* ptr_B0;                   // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  const float* topk_weights;         // topk weights
  const int32_t* sorted_token_ids;   // sorted token ids
  const int32_t* expert_ids;         // expert ids
  const int32_t* num_tokens_post_pad_ptr;// num tokens after padding
  int32_t num_tokens_post_pad;// num tokens after padding
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  uint32_t sorted_token_lens;        // sorted token length
  void *debugBuffer;
  // uint32_t topk;                    // top k
  float topk;                    // top k
};

namespace at {
namespace native {

template <typename OutputType>
void _moe_w8a8_256x128x128_TN_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  uint32_t top_k, bool isBfloat16) {

  const int size_m = input.size(0); 
  const int size_n = b_qweight.size(1); 
  const int size_k = input.size(1);

  bool first_stage = true;

  const float* topk_weights_ptr; // 第一阶段这里为null
  if (topk_weights.has_value()){
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
    first_stage = false;
  }

  int num_pad = num_tokens_post_pad[0].item<int>();
  int sorted_token_lens = sorted_token_ids.size(0);

  /* grid sizes */
  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  unsigned int totalWorkGroups0 = DIVIDE(size_n, BLOCKN);
  unsigned int totalWorkGroups1 = 1;
  globalWorkSize[0] = totalWorkGroups0;
  globalWorkSize[1] = totalWorkGroups1;
  globalWorkSize[2] = std::min(DIVIDE(sorted_token_lens, BLOCKM), DIVIDE(num_pad, BLOCKM));

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.numWorkGroups0 = globalWorkSize[0];
  hipFunctionArgs.numWorkGroups1 = globalWorkSize[2];
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = b_qweight.data_ptr<int8_t>();
  hipFunctionArgs.ptr_B0 = input.data_ptr<int8_t>();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  hipFunctionArgs.topk_weights = topk_weights_ptr;
  hipFunctionArgs.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
  hipFunctionArgs.expert_ids = expert_ids.data_ptr<int32_t>();
  hipFunctionArgs.num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
  hipFunctionArgs.num_tokens_post_pad = num_pad;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_asm = a_scale.stride(0);
  hipFunctionArgs.stride_ask = a_scale.stride(1);
  hipFunctionArgs.stride_bse = b_scale.stride(0);
  hipFunctionArgs.stride_bsn = b_scale.stride(1);
  hipFunctionArgs.stride_bsk = b_scale.stride(2);
  hipFunctionArgs.sorted_token_lens = sorted_token_ids.size(0);
  hipFunctionArgs.topk = 1.0 / top_k;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if 0
  int debug_size = 768 * 128;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  // printf("%s:%d top_k:%d topk:%f\n", __FILE__, __LINE__, top_k, hipFunctionArgs.topk);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  string fp16_co_file = "moe_w8a8_channel/moe_w8a8_128x256x128_TN_FP16.co";
  string bf16_co_file = "moe_w8a8_channel/moe_w8a8_128x256x128_TN_BF16.co";
	static AiterAsmKernel moe_w8a8_fp16("Cijk_Alik_Bljk_I8HS_BH_SABB_UserArgs_MT128x256x128_SN_K1_SB2_SSA0_SSB1024_SWMNK1_8_1_TT8_16", fp16_co_file.c_str());
	static AiterAsmKernel moe_w8a8_bf16("Cijk_Alik_Bljk_I8HS_BH_SABB_UserArgs_MT128x256x128_SN_K1_SB2_SSA0_SSB1024_SWMNK1_8_1_TT8_16", bf16_co_file.c_str());
	AiterAsmKernel *impl_ptr = &moe_w8a8_fp16;
  if (isBfloat16) 
    impl_ptr = &moe_w8a8_bf16;
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

  #if 0
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2]);
  printf("\ndebug:\n");
  hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  // using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / 32; ++i) {
    // if (i % 8 == 0) printf("\n");
    // if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // printf("%3d:%10.4f\t", i, p_debug[i]);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    // printf("%3d\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    int index = 32 * i;
    printf("i:%d 0:%6d 1:%6d 2:%6d 3:%6d 4:%6d 5:%6d 6:%6d 7:%6d "
      "8:%6d 9:%6d 10:%6d 11:%6d 12:%6d 13:%6d 14:%6d 15:%6d "
      "16:%6d 17:%6d 18:%6d 19:%6d 20:%6d 21:%6d 22:%6d 23:%6d "
      "24:%6d 25:%6d 26:%6d 27:%6d 28:%6d 29:%6d 30:%6d 31:%6d\n", i,
      p_debug[index + 0], p_debug[index + 1], p_debug[index + 2], p_debug[index + 3], 
      p_debug[index + 4], p_debug[index + 5], p_debug[index + 6], p_debug[index + 7], 
      p_debug[index + 8], p_debug[index + 9], p_debug[index +10], p_debug[index +11], 
      p_debug[index +12], p_debug[index +13], p_debug[index +14], p_debug[index +15], 
      p_debug[index +16], p_debug[index +17], p_debug[index +18], p_debug[index +19], 
      p_debug[index +20], p_debug[index +21], p_debug[index +22], p_debug[index +23], 
      p_debug[index +24], p_debug[index +25], p_debug[index +26], p_debug[index +27], 
      p_debug[index +28], p_debug[index +29], p_debug[index +30], p_debug[index +31]
      );
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

torch::Tensor moe_w8a8_asm(
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

  if (output.dtype() == torch::kFloat16)
    _moe_w8a8_256x128x128_TN_impl<half_t>(
      input, b_qweight, output, a_scale, b_scale,
      topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_pad, top_k, 0);
  else if (output.dtype() == torch::kBFloat16)
    _moe_w8a8_256x128x128_TN_impl<bhalf_t>(
      input, b_qweight, output, a_scale, b_scale,
      topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_pad, top_k, 1);
  else if (output.dtype() == torch::kFloat32)
    _moe_w8a8_256x128x128_TN_impl<float>(
      input, b_qweight, output, a_scale, b_scale,
      topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_pad, top_k, 1);
  else
    TORCH_CHECK(false, "moe_w8a8 only supports outputType in [Float16, BFloat16, Float32]");
  return output;
}

} // namespace native
} // namespace at