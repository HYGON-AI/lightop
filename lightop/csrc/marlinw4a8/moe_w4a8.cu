#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// 算子实现文件
#include "kernel.cu"

#include <torch/extension.h>
#include <iostream>

namespace at {
namespace native {  
namespace marlin_moe_w4a8{

void marlin_mm_bf16_per_channel(
                              const void* A, 
                              const void* A_Scales, 
                              const void* B, 
                              const void* B_Scales, 
                              void* C, 
                              void* sorted_token_ids, 
                              void* expert_ids,
                              void* num_tokens_past_padded, 
                              void* topk_weights,
                              int moe_block_size, 
                              int top_k, 
                              bool mul_topk_weights,
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
                              int blocks_per_cu // 一个CU上并行执行多少block
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
  const float* topk_weights_ptr = (const float*)topk_weights;

  // 全局规约的信号量                        
  int* locks = (int*)workspace;
  // 全局归约的缓冲区
  int4* global_reduce_buffer_int4 = (int4*)global_reduce_buffer;

  int max_shared_mem = 0;

  //  X, 128, 128
  if(thread_n==128 && thread_k == 128){
    switch(moe_block_size / 16) {
      case 1: // 16-128-128
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 1, 8, 4, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );   
        break;  
      case 2: // 32-128-128
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 2, 8, 4, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );    
        break;      
      case 3: // 48-128-128
        max_shared_mem = 13000; // 3  4  5
        Marlin<256, 3, 8, 4, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );       
        break;   
      case 4: // 64-128-128
        max_shared_mem = 20000; // 3
        Marlin<256, 4, 8, 4, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );  
        break;        
      case 5: // 80-128-128
        max_shared_mem = 21200;
        // 80, 128, 128
        Marlin<256, 5, 8, 4, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );     
        break;   
        
      default:
        printf("仅支持 moe_block_size = 16、32、48、64、80 四个值, 当前值: %d \n", moe_block_size);
        break;
    }  
  }else if(thread_n==256 && thread_k == 64){
    switch(moe_block_size / 16) {
      case 1: // 16-256-64
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 1, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            );   
        break;  
      case 2: // 32-256-64
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 2, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
        break;      
      case 3: // 48-256-64
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 3, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
    
        break;   
      case 4: // 64-256-64
        max_shared_mem = 10000; // 3 4 5 6 
        Marlin<256, 4, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 

        break;        
      case 5: // 80-256-64
        max_shared_mem = 10880; // 3 4 5 6 
        Marlin<256, 5, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
    
        break;   
      case 6: // 96-256-64
        max_shared_mem = 13056; // 3 4 5 
        Marlin<256, 6, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
    
        break;   
      case 7: // 112-256-64
        max_shared_mem = 15232; // 3 4
        Marlin<256, 7, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
    
        break;   
      case 8: // 128-256-64
        max_shared_mem = 17408; // 3 4
        Marlin<256, 8, 16, 2, 2><<<sms * blocks_per_cu, 256, max_shared_mem, stream>>>(
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
                                                              prob_m,
                                                              prob_n, 
                                                              prob_k,
                                                              locks,
                                                              global_reduce_buffer_int4
                                                            ); 
    
        break;   
        
      default:
        printf("仅支持 moe_block_size = 16、32、48、64、80 112、128, 当前值: %d \n", moe_block_size);
        break;
    }  
  }else{
    printf(" config not support!!!!");
  }
}

}  // namespace 

torch::Tensor moe_marlin_w4a8(
                                  torch::Tensor& a_q, 
                                  torch::Tensor& a_scales, 
                                  std::optional<torch::Tensor> const& c_or_none,
                                  torch::Tensor& b_q_weight, 
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace, 
                                  torch::Tensor& global_reduce_buffer,  // 用于全局规约的缓冲区
                                  torch::Tensor& sorted_token_ids, 
                                  torch::Tensor& expert_ids,
                                  torch::Tensor& num_tokens_past_padded, 
                                  torch::Tensor& topk_weights,
                                  int64_t moe_block_size, 
                                  int64_t top_k, 
                                  bool mul_topk_weights, 
                                  bool is_ep,
                                  int64_t size_m, 
                                  int64_t size_n,
                                  int64_t size_k,
                                  int64_t thread_n,
                                  int64_t thread_k,
                                  int64_t blocks_per_cu
                                ) {
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a_q.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a_q));
  auto options = torch::TensorOptions().dtype(a_q.dtype()).device(a_q.device());
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m * top_k,
                "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m * topk = ", size_m * top_k);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m * top_k, size_n}, options);
  }

  int dev = a_q.get_device();

  at::native::marlin_moe_w4a8::marlin_mm_bf16_per_channel(
                                          a_q.data_ptr(), 
                                          a_scales.data_ptr(), 
                                          b_q_weight.data_ptr(), 
                                          b_scales.data_ptr(),
                                          c.data_ptr(),
                                          sorted_token_ids.data_ptr(),
                                          expert_ids.data_ptr(), 
                                          num_tokens_past_padded.data_ptr(),
                                          topk_weights.data_ptr(), 
                                          moe_block_size, 
                                          top_k, 
                                          mul_topk_weights,
                                          size_m, 
                                          size_n, 
                                          size_k,  
                                          workspace.data_ptr(),
                                          global_reduce_buffer.data_ptr(),
                                          dev,
                                          at::cuda::getCurrentCUDAStream(dev), 
                                          thread_k, 
                                          thread_n, 
                                          sms,
                                          blocks_per_cu
                                        );

  return c;

} // end moe_marlin_w4a8

} // end native 
} // end at
