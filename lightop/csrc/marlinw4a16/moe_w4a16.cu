#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.cu"
#include "awq_marlin_repack.cu"
#include "kernel_gemm2.cu"

#include <torch/extension.h>
#include <iostream>

namespace at {
namespace native {
namespace marlin_moe_w4a16{

template <typename scalar_t>
void marlin_mm_impl(const void* A, const void* B, void* C, void* s_zp,
              void* sorted_token_ids, void* expert_ids,
              void* num_tokens_past_padded, void* topk_weights,
              int moe_block_size, int top_k, bool mul_topk_weights,
              int prob_m, int prob_n, int prob_k, void* workspace,
              int num_groups, int group_size,
              int dev, cudaStream_t stream, int thread_k, int thread_n,
              int sms);

void marlin_mm(const void* A, const void* B, void* C, void* s_zp,
              void* sorted_token_ids, void* expert_ids,
              void* num_tokens_past_padded, void* topk_weights,
              int moe_block_size, int top_k, bool mul_topk_weights,
              int prob_m, int prob_n, int prob_k, void* workspace,
              int num_groups, int group_size,
              int dev, cudaStream_t stream, int thread_k, int thread_n,
              int sms) {
  // 默认使用fp16，保持向后兼容
  marlin_mm_impl<__half>(A, B, C, s_zp, sorted_token_ids, expert_ids,
                         num_tokens_past_padded, topk_weights, moe_block_size, top_k, mul_topk_weights,
                         prob_m, prob_n, prob_k, workspace, num_groups, group_size,
                         dev, stream, thread_k, thread_n, sms);
}

template <typename scalar_t>
void marlin_mm_impl(const void* A, const void* B, void* C, void* s_zp,
              void* sorted_token_ids, void* expert_ids,
              void* num_tokens_past_padded, void* topk_weights,
              int moe_block_size, int top_k, bool mul_topk_weights,
              int prob_m, int prob_n, int prob_k, void* workspace,
              int num_groups, int group_size,
              int dev, cudaStream_t stream, int thread_k, int thread_n,
              int sms) {
  int group_blocks = 0;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                " is not divisible by group_blocks = ", group_blocks);
  }

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  const int4* s_zp_ptr = (const int4*)s_zp;

  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Marlin_test<__half, 256, 1, 16, 4, 2, 4><<<sms*3, 256, 20000, stream>>>(
  //     A_ptr, B_ptr, C_ptr, s_zp_ptr,
  //     sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
  //     topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
  //     prob_n, prob_k, locks); 

  // deepseek gemm1
  if(prob_k==7168 && prob_n==512){
    switch(moe_block_size / 16) {
      case 1: // 16-256-64
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_256_128_reduce<scalar_t, 256, 1, 16, 4, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks); 
        break;

      case 2: // 32-256-64
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_256_128_reduce<scalar_t, 256, 2, 16, 4, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks); 
        break;

      case 3: // 48-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 3, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);         
        break;
      case 4: // 64-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 4, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);          
        break;
      case 5: // 80-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 5, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);          
        break;

      default:
        printf("仅支持 moe_block_size = 16、32、48、64、80 四个值, 当前值: %d \n", moe_block_size);
        break;
    };
  }else if(prob_k==256 && prob_n==7168){ //  deepseek gemm2
    switch(moe_block_size / 16) {
      case 1: // 16-256-64
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 1, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);    
        break;

      case 2: // 32-256-64
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 2, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);    
        break;

      case 3: // 48-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 3, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);         
        break;
      case 4: // 64-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 4, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);          
        break;
      case 5: // 80-512-32
        MOE_W4A16_f16_PERCHANNEL_HIP_NT_marlin_512_32_reduce<scalar_t, 256, 5, 32, 2, 2, 4><<<sms*3, 256, 20000, stream>>>(
            A_ptr, B_ptr, C_ptr, s_zp_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
            topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
            prob_n, prob_k, locks);          
        break;

      default:
        printf("仅支持 moe_block_size = 16、32、48、64、80 四个值, 当前值: %d \n", moe_block_size);
        break;
    };      
  }else{
    TORCH_CHECK(0, "Not adapted to the current size!!!"); // N维度 必须是 128 的倍数
  }

}

}

torch::Tensor moe_marlin_w4a16(torch::Tensor& a, 
                              std::optional<torch::Tensor> const& c_or_none,
                              torch::Tensor& b_q_weight, 
                              torch::Tensor& b_scale_zeros,
                              std::optional<torch::Tensor> const& g_idx_or_none,
                              std::optional<torch::Tensor> const& perm_or_none, 
                              torch::Tensor& workspace, 
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
                              bool is_k_full, 
                              bool use_atomic_add, 
                              bool use_fp32_reduce,
                              bool is_zp_float
                            ) {
  int pack_factor = 8;
  int temp27;
  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 80,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % at::native::marlin_moe_w4a16::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", at::native::marlin_moe_w4a16::tile_size);
  TORCH_CHECK((size_k / at::native::marlin_moe_w4a16::tile_size) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_k = ", size_k,
              ", tile_size = ", at::native::marlin_moe_w4a16::tile_size);
  TORCH_CHECK(
      b_q_weight.size(2) % at::native::marlin_moe_w4a16::tile_size == 0,
      "b_q_weight.size(2) = ", b_q_weight.size(2),
      " is not divisible by tile_size = ", at::native::marlin_moe_w4a16::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / at::native::marlin_moe_w4a16::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
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

  
  int num_groups = -1;
  int group_size = -1;

  num_groups = b_scale_zeros.size(1);

  if (num_groups > 1) {
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  // Verify workspace size
  TORCH_CHECK(size_n % at::native::marlin_moe_w4a16::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              at::native::marlin_moe_w4a16::min_thread_n);

  TORCH_CHECK(workspace.numel() == sms * 3,
              "workspace.numel = ", workspace.numel(),
              " is == ", sms * 3);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    at::native::marlin_moe_w4a16::marlin_mm_impl<__half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        b_scale_zeros.data_ptr<int>(),
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(), num_tokens_past_padded.data_ptr(),
        topk_weights.data_ptr(), moe_block_size, top_k, mul_topk_weights,
        size_m, size_n, size_k, workspace.data_ptr(),
        num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    at::native::marlin_moe_w4a16::marlin_mm_impl<uint16_t>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(), c.data_ptr<at::BFloat16>(),
        b_scale_zeros.data_ptr<int>(),
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(), num_tokens_past_padded.data_ptr(),
        topk_weights.data_ptr(), moe_block_size, top_k, mul_topk_weights,
        size_m, size_n, size_k, workspace.data_ptr(),
        num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms);
  } else {
    TORCH_CHECK(false,
                "moe_wna16_marlin_gemm only supports Half and BFloat16");
  }

  return c;
} // end moe_wna16_marlin_gemm

// 目前只支持了 uint4 数据的量化
torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k, int64_t size_n, int64_t num_bits){
  // Verify compatibility with marlin tile of 16x128
  TORCH_CHECK(size_k % at::native::marlin_moe_w4a16::tile_k_size == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", at::native::marlin_moe_w4a16::tile_k_size); // K维度 必须是 16 的倍数
  TORCH_CHECK(size_n % at::native::marlin_moe_w4a16::tile_n_size == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", at::native::marlin_moe_w4a16::tile_n_size); // N维度 必须是 128 的倍数

  TORCH_CHECK(num_bits == 4,
              "num_bits must be 4. Got = ", num_bits); // 目前只支持 uint4 量化
  int const pack_factor = 32 / num_bits; // 一个int32 需要包含几个 量化后的权重数据  8

  // Verify B   量化后B矩阵size  size_k, size_n / pack_factor
  TORCH_CHECK(b_q_weight.size(0) == size_k,
              "b_q_weight.size(0) = ", b_q_weight.size(0),
              " is not size_k = ", size_k);
  TORCH_CHECK((size_n / pack_factor) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_n = ", size_n, ", pack_factor = ", pack_factor);
  
  // Verify device and strides
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto options = torch::TensorOptions()
                    .dtype(b_q_weight.dtype())
                    .device(b_q_weight.device());
  // 申请一个 [size_k /16, size_n * 16 / 8] 的Tensor 即 Marlin 处理后的结果Tensor
  torch::Tensor out = torch::empty(
      {size_k / at::native::marlin_moe_w4a16::tile_size, size_n * at::native::marlin_moe_w4a16::tile_size / pack_factor},
      options);

  // Get ptrs 转成 uint32 的指针
  uint32_t const* b_q_weight_ptr =
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  // Get dev info
  int dev = b_q_weight.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  
  // 目前只支持 uint4
  if (false) {
  }else if(num_bits==4){
    at::native::marlin_moe_w4a16::awq_marlin_repack_kernel<256, 4><<<blocks, 256, max_shared_mem, stream>>>(b_q_weight_ptr, out_ptr, size_k, size_n);
  }
  else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits);
  }

  return out;
}

} 
}