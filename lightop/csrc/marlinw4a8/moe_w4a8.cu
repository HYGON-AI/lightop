#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// 算子实现文件
#include "kernel.cu"
// 8warp 取消归约 复用 sorted_token_id
#include "kernel_512_64.cu"
#include "kernel_512_64_reduce.cu"
#include "awq_marlin_repack_w4a8.cu"
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
  // X-256-128
  if(thread_n==256 && thread_k == 128){
    switch(moe_block_size / 16) {
      case 1: // 16-256-128
        max_shared_mem = 20000;
        if (!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 1, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 1, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 2: // 32-256-128
        max_shared_mem = 20000;
        if (!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 2, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 2, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 3: // 48-256-128
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 3, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 3, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 5: // 80-256-128
        max_shared_mem = 21120;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 5, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 5, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        } 
        break;
      // case 8: // 128-256-128 
      //   max_shared_mem = 33792;
      //   Marlin_256_128<512, 8, 16, 4, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
      //                                                         A_ptr, 
      //                                                         A_scales_ptr, 
      //                                                         B_ptr, 
      //                                                         B_scales_ptr,
      //                                                         C_ptr, 
      //                                                         sorted_token_ids_ptr, 
      //                                                         expert_ids_ptr, 
      //                                                         num_tokens_past_padded_ptr,
      //                                                         topk_weights_ptr, 
      //                                                         top_k, 
      //                                                         mul_topk_weights, 
      //                                                         prob_m,
      //                                                         prob_n, 
      //                                                         prob_k,
      //                                                         locks,
      //                                                         global_reduce_buffer_int4
      //                                                       );   
      //   break;
      default:
        printf("X-256-128 config 仅支持 moe_block_size = 16、32、48、64、80 四个值, 当前值: %d \n", moe_block_size);
        break;
    }
  }else if(thread_n==128 && thread_k == 256){
    switch(moe_block_size / 16) {
      case 1: // 16-128-256
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 1, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 1, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 2: // 32-128-256
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 2, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 2, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 3: // 48-128-256
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 3, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 3, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 5: // 80-128-256
        max_shared_mem = 21120;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 5, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 5, 8, 8, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 8: // 128-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 8, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 8, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      default:
        printf(" X-512-65 config 仅支持 moe_block_size = 16、32、48、64、80、128四个值, 当前值: %d \n", moe_block_size);
        break;
    }
  }else if(thread_n==512 && thread_k == 64){
    switch(moe_block_size / 16) {
      case 1: // 16-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 1, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 1, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 2: // 32-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 2, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 2, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 3: // 48-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 3, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 3, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 5: // 80-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 5, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 5, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      case 8: // 128-512-64
        max_shared_mem = 20000;
        if(!is_reduce){
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64<512, 8, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }else{
          MOE_W4A8_I8_PERCHANNEL_HIP_NT_marlin_512_64_reduce<512, 8, 32, 2, 2><<<sms * blocks_per_cu, 512, max_shared_mem, stream>>>(
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
        }
        break;
      default:
        printf(" X-512-65 config 仅支持 moe_block_size = 16、32、48、64、80、128四个值, 当前值: %d \n", moe_block_size);
        break;
    }
  }else{
    printf("config not support!!!!");
  }

}  // namespace 

}

torch::Tensor moe_marlin_w4a8(
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
  at::native::marlin_moe_w4a8::marlin_mm_bf16_per_channel(
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


// uint4 [32, 64] 大小的重排
torch::Tensor awq_marlin_repack_w4a8(torch::Tensor& b_q_weight,  
                                    torch::Tensor& marlin_weight, 
                                    int64_t size_batch, 
                                    int64_t size_k, 
                                    int64_t size_n
                                  ){
  // Verify compatibility with marlin tile of 16x128
  TORCH_CHECK(size_k % 32 == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", 32); // K维度 必须是 32 的倍数
  TORCH_CHECK(size_n % 64 == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", 64); // N维度 必须是 64 的倍数

  // Verify device and strides
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  uint8_t const* b_q_weight_ptr = reinterpret_cast<uint8_t const*>(b_q_weight.data_ptr());
  uint32_t* marlin_weight_ptr = reinterpret_cast<uint32_t*>(marlin_weight.data_ptr());

  // Get dev info
  int dev = b_q_weight.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  dim3 gridDim;
  gridDim.x = size_batch; // 有几个专家矩阵需要处理
  gridDim.y = size_k / 32; // 专家矩阵的K维度 包含 [32, 64]数据块的数目
  gridDim.z = size_n / 64; // 专家矩阵的N维度 包含 [32, 64]数据块的数目

  // // 不用 LDS
  at::native::marlin_moe_w4a8::awq_marlin_w4a8_repack_kernel<256, 4><<<gridDim, 256, 0, stream>>>(
                                                                  b_q_weight_ptr,
                                                                  marlin_weight_ptr, 
                                                                  size_batch,
                                                                  size_k, 
                                                                  size_n
                                                                  );
  return marlin_weight;
}

} // end native 
} // end at
