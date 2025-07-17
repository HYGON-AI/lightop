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
  uint32_t topk;                    // top k
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
  int64_t top_k, bool isBfloat16) {

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
  hipFunctionArgs.topk = top_k;
  // torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt).device(input.device());
  // torch::Tensor debug = torch::empty({768}, options);
  void * debug_host, * debug_device;
  int debug_size = 768 * 4;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  // printf("%s:%d isbf16:%d\n", __FILE__, __LINE__, isBfloat16);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  string fp16_co_file = "moe_w8a8_block/moe_w8a8_128x256x128_TN_FP16.co";
  string bf16_co_file = "moe_w8a8_block/moe_w8a8_128x256x128_TN_BF16.co";
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

  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2]);
  printf("\ndebug:\n");
  hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  // using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
    if (i % 8 == 0) printf("\n");
    printf("%3d:%6d\t", i, p_debug[i]);
  }
  printf("\n");
  
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
  int64_t top_k,
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
  else
    TORCH_CHECK(false, "moe_w8a8 only supports outputType in [Float16, BFloat16]");
  return output;
}

} // namespace native
} // namespace at