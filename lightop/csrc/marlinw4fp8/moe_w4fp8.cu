#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// 8warp 取消归约 复用 sorted_token_id
#include "kernel_512_64.cu"
#include "kernel_512_64_reduce.cu"
#include "awq_marlin_repack_w4a8.cu"
#include <torch/extension.h>
#include <iostream>

namespace at {
namespace native {  
namespace marlin_moe_w4fp8{
void marlin_mm_bf16_per_channel(
                              const void* A, 
                              const void* A_Scales, 
                              const void* B, 
                              const void* B_Scales, 
                              void* C, 
                              void* sorted_token_ids, 
                              void* expert_ids,
                              void* num_tokens_past_padded, 
                              std::optional<torch::Tensor> topk_weights,
                              int moe_block_size, 
                              int top_k, 
                              bool mul_topk_weights,
                              bool is_ep,
                              int prob_m, 
                              int prob_n, 
                              int prob_k,
                              void* workspace,
                              void* global_reduce_buffer,
                              int dev, 
                              cudaStream_t stream, 
                              int thread_k, 
                              int thread_n,
                              int sms,
                              int blocks_per_cu, // 一个CU上并行执行多少block
                              bool is_reduce
                            ) {

  const int4* A_ptr = (const int4*)A;
  // const int4* A_scales_ptr = (const int4*)A_Scales;
  const float* A_scales_ptr = (const float*)A_Scales;
  const int4* B_ptr = (const int4*)B;
  // const int4* B_scales_ptr = (const int4*)B_Scales;
  const float* B_scales_ptr = (const float*)B_Scales;
  int4* C_ptr = (int4*)C;

  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;


  const float* topk_weights_ptr;
  if(mul_topk_weights){
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
  }

  // 全局规约的信号量                        
  int* locks = (int*)workspace;
  // 全局归约的缓冲区
  int4* global_reduce_buffer_int4 = (int4*)global_reduce_buffer;

  int max_shared_mem = 0;

  max_shared_mem = 20000;
  MOE_W4A8_F8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 1, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
                                                      A_ptr, 
                                                      A_scales_ptr, 
                                                      B_ptr, 
                                                      B_scales_ptr,
                                                      C_ptr, 
                                                      sorted_token_ids_ptr, 
                                                      expert_ids_ptr, 
                                                      num_tokens_past_padded_ptr,
                                                      topk_weights_ptr, 
                                                      top_k, 
                                                      mul_topk_weights, 
                                                      is_ep,
                                                      prob_m,
                                                      prob_n, 
                                                      prob_k,
                                                      locks,
                                                      global_reduce_buffer_int4
                                                    );   
}  // namespace 

}

torch::Tensor moe_marlin_w4fp8(
                                  torch::Tensor& a_q, 
                                  torch::Tensor& a_scales, 
                                  torch::Tensor& c,
                                  torch::Tensor& b_q_weight, 
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace, 
                                  torch::Tensor& global_reduce_buffer,  // 用于全局规约的缓冲区
                                  torch::Tensor& sorted_token_ids, 
                                  torch::Tensor& expert_ids,
                                  torch::Tensor& num_tokens_past_padded, 
                                  std::optional<torch::Tensor> topk_weights,
                                  int64_t moe_block_size, 
                                  int64_t top_k, 
                                  bool is_ep,
                                  int64_t size_m, 
                                  int64_t size_n,
                                  int64_t size_k,
                                  int64_t thread_n,
                                  int64_t thread_k,
                                  int64_t blocks_per_cu,
                                  bool is_reduce
                                ) {
  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a_q));
  auto options = torch::TensorOptions().dtype(a_q.dtype()).device(a_q.device());
  int dev = a_q.get_device();

  bool mul_topk_weights = topk_weights.has_value();
  at::native::marlin_moe_w4fp8::marlin_mm_bf16_per_channel(
                                          a_q.data_ptr(), 
                                          a_scales.data_ptr(), 
                                          b_q_weight.data_ptr(), 
                                          b_scales.data_ptr(),
                                          c.data_ptr(),
                                          sorted_token_ids.data_ptr(),
                                          expert_ids.data_ptr(), 
                                          num_tokens_past_padded.data_ptr(),
                                          topk_weights, 
                                          moe_block_size, 
                                          top_k, 
                                          mul_topk_weights,
                                          is_ep,
                                          size_m, 
                                          size_n, 
                                          size_k,  
                                          workspace.data_ptr(),
                                          global_reduce_buffer.data_ptr(),
                                          dev,
                                          at::cuda::getCurrentCUDAStream(dev), 
                                          thread_k, 
                                          thread_n, 
                                          80,
                                          blocks_per_cu,
                                          is_reduce
                                        );

  return c;

} // end moe_marlin_w4a8

} // end native 
} // end at
