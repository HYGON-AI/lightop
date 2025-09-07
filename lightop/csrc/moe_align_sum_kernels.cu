#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#define WARP_SIZE 1
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace at {
namespace native {

  template <typename scalar_t,int smem_size>
__global__ void moe_align_block_size_kernel_opt_fused(
    const scalar_t* __restrict__ topk_ids,      // 输入：每个令牌对应的专家ID
    int32_t* __restrict__ sorted_token_ids,     // 输出：排序后的令牌原始索引（此内核不填充此项）
    int32_t* __restrict__ expert_ids,           // 输出：每个块对应的专家ID
    int32_t* __restrict__ total_tokens_post_pad,// 输出：填充后的总令牌数
    int32_t num_experts,                        // 输入：专家总数
    int32_t padded_num_experts,                 // 输入：填充到 warp 大小倍数的专家数
    int32_t experts_per_warp,                   // 输入：每个 warp 处理的专家数
    int32_t block_size,                         // 输入：对齐用的块大小
    size_t numel) {             // 中间缓存：用于存储每个专家的令牌数和累积和

  // 声明共享内存，用于块内线程间高效通信，存储每个专家的令牌计数
   __shared__ int32_t shared_counts[smem_size+1];
  int32_t *cumsum = shared_counts;
  // 计算当前线程所在的 warp ID 和 warp 内的线程 ID (lane_id)
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  // 计算当前 warp 负责处理的起始专家 ID
  const int my_expert_start = warp_id * experts_per_warp;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  // [优化点 1: 并行初始化]
  // 使用 warp-stride 循环代替原有的循环。
  // 原有写法中，一个 warp 内的所有线程会重复执行相同的工作，效率低下。
  // 优化后，warp 内的每个线程 (lane_id) 分工合作，各自负责初始化不同的计数器，
  // 从而完全并行化，避免了重复劳动和写入冲突。
  for (int i = tid; i <= num_experts; i += stride) {
      shared_counts[i] = 0;
  }

  // 同步块内所有线程，确保初始化完成
  __syncthreads();

  // 获取线程的唯一 ID 和步长，用于并行遍历所有令牌


  // [步骤1: 计数]
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    atomicAdd(&shared_counts[expert_id], 1);
  }

  // 同步块内所有线程，确保计数完成
  __syncthreads();

  // 计算前缀和与填充
  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    shared_counts[threadIdx.x] = CEILDIV(expert_count, block_size) * block_size;
  }
  __syncthreads();

  int offset = 1;
   for (int d = num_experts / 2; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            shared_counts[bi] += shared_counts[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    if(tid == 0) *total_tokens_post_pad = shared_counts[num_experts-1];
     //最后一位清0
    if (tid == 0)
    {
        shared_counts[num_experts - 1] = 0;
    }
    __syncthreads();

   //下扫
    for (int d = 1; d < num_experts; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
			//下扫两步，看图
            //bi的值要传给ai  
            //ai的值要和bi相加
            int v = shared_counts[ai];
            shared_counts[ai] = shared_counts[bi];
            shared_counts[bi] += v;
        }
        __syncthreads();
    }
   
 if(tid == 0) shared_counts[num_experts] = *total_tokens_post_pad;
    __syncthreads();
  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  __syncthreads();
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }

}

template <typename scalar_t,int seme_size>
__global__ void moe_align_block_size_kernel_opt(
    const scalar_t* __restrict__ topk_ids,      // 输入：每个令牌对应的专家ID
    int32_t* __restrict__ sorted_token_ids,     // 输出：排序后的令牌原始索引（此内核不填充此项）
    int32_t* __restrict__ expert_ids,           // 输出：每个块对应的专家ID
    int32_t* __restrict__ total_tokens_post_pad,// 输出：填充后的总令牌数
    int32_t num_experts,                        // 输入：专家总数
    int32_t padded_num_experts,                 // 输入：填充到 warp 大小倍数的专家数
    int32_t experts_per_warp,                   // 输入：每个 warp 处理的专家数
    int32_t block_size,                         // 输入：对齐用的块大小
    size_t numel,                               // 输入：令牌总数
    int32_t* __restrict__ cumsum) {             // 中间缓存：用于存储每个专家的令牌数和累积和


  // 声明共享内存，用于块内线程间高效通信，存储每个专家的令牌计数
   __shared__ int32_t shared_counts[seme_size+1];
  
  // 计算当前线程所在的 warp ID 和 warp 内的线程 ID (lane_id)
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  // 计算当前 warp 负责处理的起始专家 ID
  const int my_expert_start = warp_id * experts_per_warp;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  // [优化点 1: 并行初始化]
  // 使用 warp-stride 循环代替原有的循环。
  // 原有写法中，一个 warp 内的所有线程会重复执行相同的工作，效率低下。
  // 优化后，warp 内的每个线程 (lane_id) 分工合作，各自负责初始化不同的计数器，
  // 从而完全并行化，避免了重复劳动和写入冲突。
  for (int i = tid; i <= num_experts; i += stride) {
      shared_counts[i] = 0;
  }

  // 同步块内所有线程，确保初始化完成
  __syncthreads();

  // 获取线程的唯一 ID 和步长，用于并行遍历所有令牌


  // [步骤1: 计数]
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    atomicAdd(&shared_counts[expert_id], 1);
  }

  // 同步块内所有线程，确保计数完成
  __syncthreads();

  // 计算前缀和与填充
  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    shared_counts[threadIdx.x] = CEILDIV(expert_count, block_size) * block_size;
  }
  __syncthreads();

  int offset = 1;
   for (int d = num_experts / 2; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            shared_counts[bi] += shared_counts[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    if(tid == 0) *total_tokens_post_pad = shared_counts[num_experts-1];
     //最后一位清0
    if (tid == 0)
    {
        shared_counts[num_experts - 1] = 0;
    }
    __syncthreads();

   //下扫
    for (int d = 1; d < num_experts; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
			//下扫两步，看图
            //bi的值要传给ai  
            //ai的值要和bi相加
            int v = shared_counts[ai];
            shared_counts[ai] = shared_counts[bi];
            shared_counts[bi] += v;
        }
        __syncthreads();
    }
   
 if(tid == 0) shared_counts[num_experts] = *total_tokens_post_pad;
    __syncthreads();
  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
//  if(tid == 0) cumsum[257] = shared_counts[257];

  if(tid < 64){
     *reinterpret_cast<int4 *>(&cumsum[tid*4]) =  *reinterpret_cast<int4 *>(&shared_counts[tid*4]);
    // cumsum[tid] = shared_counts[tid];
  }

}


template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum) {
  extern __shared__ int32_t shared_counts[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];

      cumsum[i] =
          cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
  }

}

// taken from
// https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
template <typename scalar_t>
__global__ void sgl_ep_moe_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* cumsum,
    int32_t start_expert, int32_t end_expert) {
  __shared__ int32_t shared_counts[32][8];
  __shared__ int32_t local_offsets[256];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    if (expert_id >= start_expert && expert_id < end_expert) {
      expert_id -= start_expert;
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] =
          cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
    local_offsets[threadIdx.x] = cumsum[threadIdx.x];
  }

  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= start_expert && expert_id < end_expert) {
      expert_id -= start_expert;
      int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }
}

// template <typename scalar_t, int TOPK>
// __global__ void moe_sum_kernel(
//     scalar_t* __restrict__ out,          // [..., d]
//     const scalar_t* __restrict__ input,  // [..., topk, d]
//     const int d) {
//   const int64_t token_idx = blockIdx.x;
//   for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
//     scalar_t x = 0.0;
// #pragma unroll
//     for (int k = 0; k < TOPK; ++k) {
//       x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
//     }
//     out[token_idx * d + idx] = x;
//   }
// }

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] +=
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

        AT_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
     
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = std::max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          auto small_batch_expert_kernel =
              moe_align_block_size_small_batch_expert_kernel<
                  scalar_t>;
          small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        } else {
      
          
          if(num_experts == 256){
            {
                auto align_kernel = moe_align_block_size_kernel_opt<scalar_t,256>;
            auto options_int = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
                torch::Tensor cumsum_buffer = torch::zeros({num_experts + 1}, options_int);
              size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
              size_t shared_mem_size =
                 ( num_experts + 1) * sizeof(int32_t);
               if(topk_ids.numel() <= 256){
                  threads = 256;
                }else if(topk_ids.numel() <= 512){
                    threads = 512;
                }else{
                    threads = 1024;
                }
              align_kernel<<<1, threads, shared_mem_size, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  experts_ids.data_ptr<int32_t>(),
                  num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                  padded_num_experts, experts_per_warp, block_size,
                  topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());

              int block_threads = std::min(64, (int)threads);
              const int num_blocks =
                  (topk_ids.numel() + block_threads - 1) / block_threads;
              const int max_blocks = 65535;
               int actual_blocks = std::min(num_blocks, max_blocks);
            
              auto sort_kernel =
                  count_and_sort_expert_tokens_kernel<scalar_t>;
              sort_kernel<<<actual_blocks , block_threads, 0, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
            }
              
          }else{
            auto align_kernel = moe_align_block_size_kernel<scalar_t>;
            auto options_int = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
            torch::Tensor cumsum_buffer = torch::zeros({num_experts + 1}, options_int);
            size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
            size_t shared_mem_size =  num_warps * experts_per_warp * sizeof(int32_t);

            align_kernel<<<1, threads, shared_mem_size, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                experts_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                padded_num_experts, experts_per_warp, block_size,
                topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());

            const int block_threads = std::min(256, (int)threads);
            const int num_blocks =
                (topk_ids.numel() + block_threads - 1) / block_threads;
            const int max_blocks = 65535;
            const int actual_blocks = std::min(num_blocks, max_blocks);
              
            auto sort_kernel =
                count_and_sort_expert_tokens_kernel<scalar_t>;
            sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
          }
        }
      });
}

// void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
//              torch::Tensor& output)  // [num_tokens, hidden_size]
// {
//   const int hidden_size = input.size(-1);
//   const auto num_tokens = output.numel() / hidden_size;
//   const int topk = input.size(1);

//   dim3 grid(num_tokens);
//   dim3 block(std::min(hidden_size, 1024));
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   switch (topk) {
//     case 2:
//       VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
//         moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
//             output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
//             hidden_size);
//       });
//       break;

//     case 3:
//       VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
//         moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
//             output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
//             hidden_size);
//       });
//       break;

//     case 4:
//       VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
//         moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
//             output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
//             hidden_size);
//       });
//       break;

//     case 8:
//       VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
//         moe_sum_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
//             output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
//             hidden_size);
//       });
//       break;
      
//     default:
//       at::sum_out(output, input, 1);
//       break;
//   }
// }

}  // namespace at
}  // namespace native


