#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cub/cub.cuh>
#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <cub/block/block_scan.cuh>
#define WARP_SIZE 1

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

namespace at {
namespace native {

/* numel <= 256 */
template <typename scalar_t, int smem_size, int TOPK, bool Is_EP,int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_fused_small_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ expert_map,
    bool* __restrict__ expert_mask, int32_t* __restrict__ num_local_tokens, int32_t sorted_token_ids_len,bool Is_fuse_fill) {
  const int topk_numel = numel;
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  if(Is_fuse_fill){
      for (int i = tid; i < sorted_token_ids_len; i += stride) {
         sorted_token_ids[i] = topk_numel;
    }
  }

  __shared__ int32_t shared_counts[smem_size + 1];
  
  // CUB Scan 需要额外的共享内存
  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  // 保存总和
  // int32_t block_total_sum;

  __shared__ int32_t local_expert_map[smem_size];

  //初始化
  for (int i = tid; i < num_experts; i += stride) {
    shared_counts[i] = 0;
  }
   if (expert_map != nullptr) {
      if(tid < num_experts) local_expert_map[tid] = expert_map[tid];
    }
  // // 单独处理最后一个元素
  // if (tid == 0) {
  //   shared_counts[num_experts] = 0;
  // }
  __syncthreads();

  //直方图统计
  bool is_valid_expert = true;
  int expert_id = 0;

  if(tid < numel){
      expert_id = topk_ids[tid];
      if (expert_map) {
          if (local_expert_map[expert_id] == -1) {
              is_valid_expert = false;
          }
      }

      if constexpr(Is_EP) {
          if (is_valid_expert) {
              atomicAdd(&shared_counts[expert_id], 1);
          }
      } else {
          atomicAdd(&shared_counts[expert_id], 1);
      }
  }
  __syncthreads();

  // Padding 
  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    // (a + b - 1) & ~(b - 1)
    shared_counts[threadIdx.x] = (expert_count + block_size - 1) & ~(block_size - 1);
  }
  
  __syncthreads(); 

  //执行 CUB 扫描
  int32_t my_padded_count = (threadIdx.x < num_experts) ? shared_counts[threadIdx.x] : 0;
  int32_t exclusive_prefix_sum = 0;
  cub::Sum scan_op;

  BlockScan(temp_storage).ExclusiveScan(
      my_padded_count,      
      exclusive_prefix_sum,  
      0,                     
      scan_op                
  );
  // 总和等于最后一个线程的排他和+自己padd的值
  if (tid == CUB_BLOCK_THREADS - 1) {
    *total_tokens_post_pad = exclusive_prefix_sum + my_padded_count;
     shared_counts[num_experts] = exclusive_prefix_sum + my_padded_count;
  }

  //__syncthreads();
  if (threadIdx.x < num_experts) {
    shared_counts[threadIdx.x] = exclusive_prefix_sum;
  }
  __syncthreads(); 
  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      if (expert_map != nullptr) {
        expert_ids[i / block_size] = local_expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }
  __syncthreads(); 
  if(tid < numel){
      if constexpr(Is_EP){
          if(is_valid_expert){
              int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1); 
              sorted_token_ids[rank_post_pad] = tid;
          }
      }else{
              int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
              sorted_token_ids[rank_post_pad] = tid;
      }
       if (expert_mask) expert_mask[tid] = is_valid_expert;
  }
}
template <typename scalar_t, int smem_size, int TOPK, int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_fused_cub(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ expert_map,
    bool *__restrict__ expert_mask, int32_t *__restrict__ num_local_tokens,
    bool Is_EP, int32_t sorted_token_ids_len, bool Is_fuse_fill) {

  const int topk_numel = numel;
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (Is_fuse_fill) {
    for (int i = tid; i < sorted_token_ids_len; i += stride) {
      sorted_token_ids[i] = topk_numel;
    }
  }

  __shared__ int32_t shared_counts[smem_size + 1];
  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = tid; i <= num_experts; i += stride) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    // EP
    if (Is_EP) {
      if (expert_map[expert_id] != -1) {
        atomicAdd(&shared_counts[expert_id], 1);
        if (expert_mask) expert_mask[i] = true;
      } else {
        if (expert_mask) expert_mask[i] = false;
      }
    } else {
      atomicAdd(&shared_counts[expert_id], 1);
      if (expert_mask) {
        if (expert_map && expert_map[expert_id] == -1) {
          expert_mask[i] = false;
        } else {
          expert_mask[i] = true;
        }
      }
    }
  }

  __syncthreads();


  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    shared_counts[threadIdx.x] = ((expert_count + block_size - 1) / block_size) * block_size;
  }
  __syncthreads();


  int32_t my_padded_count = (threadIdx.x < num_experts) ? shared_counts[threadIdx.x] : 0;
  int32_t exclusive_prefix_sum = 0;
  cub::Sum scan_op;

  BlockScan(temp_storage).ExclusiveScan(
      my_padded_count,      
      exclusive_prefix_sum,  
      0,                     
      scan_op                
  );
  // 总和等于最后一个线程的排他和+自己padd的值
  if (tid == num_experts - 1) {
    *total_tokens_post_pad = exclusive_prefix_sum + my_padded_count;
     shared_counts[num_experts] = exclusive_prefix_sum + my_padded_count;
  }


  if (threadIdx.x < num_experts) {
    shared_counts[threadIdx.x] = exclusive_prefix_sum;
  }


  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      if (expert_map != nullptr) {
        expert_ids[i / block_size] = expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }
  __syncthreads();
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (Is_EP) {
      if (expert_map[expert_id] != -1) {
        int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
      }
    } else {
      int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }
}
template <typename scalar_t, int smem_size, int TOPK>
__global__ void moe_align_block_size_kernel_opt_fused(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ expert_map,
    bool *__restrict__ expert_mask, int32_t *__restrict__ num_local_tokens,bool Is_EP,int32_t sorted_token_ids_len,bool Is_fuse_fill) {
   const int topk_numel = numel;
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

   if(Is_fuse_fill){
      for (int i = tid; i < sorted_token_ids_len; i += stride) {
         sorted_token_ids[i] = topk_numel;
    }
  }
  __shared__ int32_t shared_counts[smem_size + 1];
  int32_t *cumsum = shared_counts;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
 

  for (int i = tid; i <= num_experts; i += stride) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    //EP
    if(Is_EP){
        if(expert_map[expert_id] != -1){
           atomicAdd(&shared_counts[expert_id], 1);
           if(expert_mask)  expert_mask[i] = true;
        }else{
           if(expert_mask)  expert_mask[i] = false;
        }
    }else{
        atomicAdd(&shared_counts[expert_id], 1);
        if (expert_mask) {
          if (expert_map && expert_map[expert_id] == -1) {
            expert_mask[i] = false;
          } else {
            expert_mask[i] = true;
          }
        }
    }
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    shared_counts[threadIdx.x] = CEILDIV(expert_count, block_size) * block_size;
  }
  __syncthreads();

  int offset = 1;
  for (int d = num_experts / 2; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      shared_counts[bi] += shared_counts[ai];
    }
    offset *= 2;
    __syncthreads();
  }
  if (tid == 0)
    *total_tokens_post_pad = shared_counts[num_experts - 1];
  if (tid == 0) {
    shared_counts[num_experts - 1] = 0;
  }
  __syncthreads();

  // 下扫
  for (int d = 1; d < num_experts; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      // bi的值要传给ai
      // ai的值要和bi相加
      int v = shared_counts[ai];
      shared_counts[ai] = shared_counts[bi];
      shared_counts[bi] += v;
    }
    __syncthreads();
  }

  if (tid == 0)
    shared_counts[num_experts] = *total_tokens_post_pad;
  __syncthreads();
  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      if (expert_map != nullptr) {
        expert_ids[i / block_size] = expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }
  __syncthreads();
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if(Is_EP){
        if(expert_map[expert_id] != -1){
            int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
            sorted_token_ids[rank_post_pad] = i;
        }
    }else{
        int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
  }
}
template <typename scalar_t, int seme_size, int TOPK, int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_cub(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum,
    int32_t *__restrict__ expert_map, bool *__restrict__ expert_mask,
    int32_t *__restrict__ num_local_tokens, bool Is_EP, int32_t sorted_token_ids_len, bool Is_fuse_fill,
    int32_t *__restrict__ global_count 
) {
    const int topk_numel = numel;

    size_t effective_numel = numel;
    if (num_local_tokens != nullptr) {
        if (num_local_tokens[0] == 0) {
            return;
        } else {
            effective_numel = num_local_tokens[0] * TOPK;
        }
    }

    const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_stride = blockDim.x * gridDim.x;
    const int tid = threadIdx.x;

    __shared__ int32_t shared_counts[seme_size + 1];

    for (int i = tid; i <= num_experts; i += blockDim.x) {
        shared_counts[i] = 0;
    }

    if (Is_fuse_fill) {
        for (size_t i = global_tid; i < sorted_token_ids_len; i += global_stride) {
            sorted_token_ids[i] = topk_numel;
        }
    }

    typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    __syncthreads();

    for (size_t i = global_tid; i < effective_numel; i += global_stride) {
        int expert_id = topk_ids[i];
        
        // EP 处理
        if (Is_EP) {
            if (expert_map[expert_id] != -1) {
                atomicAdd(&shared_counts[expert_id], 1);
                if (expert_mask) expert_mask[i] = true;
            } else {
                if (expert_mask) expert_mask[i] = false;
            }
        } else {
            atomicAdd(&shared_counts[expert_id], 1);
            if (expert_mask) {
                if (expert_map && expert_map[expert_id] == -1) {
                    expert_mask[i] = false;
                } else {
                    expert_mask[i] = true;
                }
            }
        }
    }

    __syncthreads();
    if (tid < num_experts) {
        int count = shared_counts[tid];
        if (count > 0) {
            atomicAdd(&cumsum[tid], count);
        }
    }

    // // 全局同步
    // __threadfence();

    __shared__ bool is_last_block;
    if (tid == 0) {
        int ticket = atomicAdd(global_count, 1);
        is_last_block = (ticket == (gridDim.x - 1));
    }
    __syncthreads();

    if (!is_last_block) {
        return;
    }

    int32_t my_padded_count = (tid < num_experts) ? CEILDIV(cumsum[tid] , block_size) * block_size : 0;
    int32_t exclusive_prefix_sum = 0;
    cub::Sum scan_op;

    BlockScan(temp_storage).ExclusiveScan(
        my_padded_count,
        exclusive_prefix_sum,
        0, 
        scan_op
    );


    if (tid == CUB_BLOCK_THREADS - 1) {
        int32_t total = exclusive_prefix_sum + my_padded_count;
        *total_tokens_post_pad = total;
        shared_counts[num_experts] = total;
    }

    if (tid < num_experts) {
        shared_counts[tid] = exclusive_prefix_sum;
    }
    __syncthreads();

    
    if (tid < num_experts) {
        for (int i = shared_counts[tid]; i < shared_counts[tid + 1]; i += block_size) {
            if (expert_map != nullptr) {
                expert_ids[i / block_size] = expert_map[tid];
            } else {
                expert_ids[i / block_size] = tid;
            }
        }
    }

    if (tid < num_experts) {
        cumsum[tid] = shared_counts[tid];
    }
}

template <typename scalar_t, int seme_size, int TOPK>
__global__ void moe_align_block_size_kernel_opt(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum,
    int32_t *__restrict__ expert_map, bool *__restrict__ expert_mask,
    int32_t *__restrict__ num_local_tokens,bool Is_EP,int32_t sorted_token_ids_len,bool Is_fuse_fill) {
   const int topk_numel = numel;
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }

  __shared__ int32_t shared_counts[seme_size + 1];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
     if(Is_fuse_fill){
      for (int i = tid; i < sorted_token_ids_len; i += stride) {
         sorted_token_ids[i] = topk_numel;
    }
  }
  for (int i = tid; i <= num_experts; i += stride) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    //EP
    if(Is_EP){
        if(expert_map[expert_id] != -1){
           atomicAdd(&shared_counts[expert_id], 1);
           if(expert_mask)  expert_mask[i] = true;
        }else{
           if(expert_mask)  expert_mask[i] = false;
        }
    }else{
        atomicAdd(&shared_counts[expert_id], 1);
        if (expert_mask) {
          if (expert_map && expert_map[expert_id] == -1) {
            expert_mask[i] = false;
          } else {
            expert_mask[i] = true;
          }
        }
    }
  }
  __syncthreads();

  if (threadIdx.x < num_experts) {
    int expert_count = shared_counts[threadIdx.x];
    shared_counts[threadIdx.x] = CEILDIV(expert_count, block_size) * block_size;
  }
  __syncthreads();

  int offset = 1;
  for (int d = num_experts / 2; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      shared_counts[bi] += shared_counts[ai];
    }
    offset *= 2;
    __syncthreads();
  }
  if (tid == 0)
    *total_tokens_post_pad = shared_counts[num_experts - 1];

  if (tid == 0) {
    shared_counts[num_experts - 1] = 0;
  }
  __syncthreads();

  for (int d = 1; d < num_experts; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      int v = shared_counts[ai];
      shared_counts[ai] = shared_counts[bi];
      shared_counts[bi] += v;
    }
    __syncthreads();
  }

  if (tid == 0)
    shared_counts[num_experts] = *total_tokens_post_pad;
  __syncthreads();
  if (threadIdx.x < num_experts) {
    for (int i = shared_counts[threadIdx.x]; i < shared_counts[threadIdx.x + 1];
         i += block_size) {
      if (expert_map != nullptr) {
        expert_ids[i / block_size] = expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }

  if (tid < 64) {
    *reinterpret_cast<int4 *>(&cumsum[tid * 4]) =
        *reinterpret_cast<int4 *>(&shared_counts[tid * 4]);
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum,
    int32_t *__restrict__ expert_map) {
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
    int expert_id = topk_ids[i]; // Original code for no map
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
      if (expert_map) {
        expert_ids[i / block_size] = expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }
}

template <typename scalar_t, int TOPK>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ cumsum_buffer,
    size_t numel, int32_t *__restrict__ num_local_tokens) {
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  int expert_id;
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  //const size_t stride = blockDim.x * gridDim.x;
  __shared__ int32_t lds_count[256];
   if(threadIdx.x < 256){
     lds_count[threadIdx.x ] = 0;
  }
  __syncthreads();
  if(tid < numel){
    expert_id = topk_ids[tid];
    atomicAdd(&lds_count[expert_id], 1);
  }
  __syncthreads();
  if(threadIdx.x < 256){
     lds_count[threadIdx.x ] = atomicAdd(&cumsum_buffer[threadIdx.x ], lds_count[threadIdx.x ]);
  }
  // for (size_t i = threadIdx.x; i < 256; i += blockDim.x) {
    
  // }
  __syncthreads();
  if(tid < numel){
    int32_t rank_post_pad = atomicAdd(&lds_count[expert_id], 1);
     sorted_token_ids[rank_post_pad] = tid;
  }
}

template <typename scalar_t, int TOPK,int seme_size,int num_experts>
__global__ void count_and_sort_expert_tokens_kernel_EP(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ cumsum_buffer,
    size_t numel, int32_t *__restrict__ num_local_tokens,int32_t* expert_map,bool Is_EP) {
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  int expert_id;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
   const int stride = blockDim.x * gridDim.x;
  //const size_t stride = blockDim.x * gridDim.x;
  __shared__ int32_t lds_count[seme_size];
   if(threadIdx.x < num_experts){
     lds_count[threadIdx.x ] = 0;
  }
  __syncthreads();
   for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (Is_EP) {
      if (expert_map != nullptr) {
          int32_t local_idx = expert_map[expert_id];
          if (local_idx != -1) {
              atomicAdd(&lds_count[local_idx], 1);
          }
      }
    } else {
        atomicAdd(&lds_count[expert_id], 1);
    }
  }
  __syncthreads();

  if(threadIdx.x < num_experts){
     lds_count[threadIdx.x ] = atomicAdd(&cumsum_buffer[threadIdx.x ], lds_count[threadIdx.x ]);
  }

  __syncthreads();
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (Is_EP) {
      if (expert_map != nullptr) {
           int32_t local_idx = expert_map[expert_id];
           if (local_idx != -1) {
               int32_t rank_post_pad = atomicAdd(&lds_count[local_idx], 1);
               sorted_token_ids[rank_post_pad] = i; 
           }
      }
    } else {
          int32_t rank_post_pad = atomicAdd(&lds_count[expert_id], 1);
          sorted_token_ids[rank_post_pad] = i; 
    }
  }
}
// taken from
// https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
template <typename scalar_t>
__global__ void sgl_ep_moe_align_block_size_kernel(
    scalar_t *__restrict__ topk_ids, int32_t *sorted_token_ids,
    int32_t *expert_ids, int32_t *total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t *cumsum, int32_t start_expert,
    int32_t end_expert) {
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

template <typename scalar_t, int TOPK, bool Is_bf16>
__global__ void
moe_sum_kernel(scalar_t *__restrict__ out,         // [..., d]
               const scalar_t *__restrict__ input, // [..., topk, d]
               const int d, scalar_t *__restrict__ bias, float factor) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    if (!Is_bf16)
      factor = 1.0f / factor;
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += input[token_idx * TOPK * d + k * d + idx];
    }
    if (bias == nullptr) {
      out[token_idx * d + idx] = x * factor;
    } else {
      if (Is_bf16) {
        out[token_idx * d + idx] = x * factor + bias[token_idx * d + idx];
      } else {
        out[token_idx * d + idx] = x + bias[token_idx * d + idx] * factor;
      }
    }
  }
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_reg(scalar_t *__restrict__ out,
                                   const scalar_t *__restrict__ input,
                                   scalar_t *__restrict__ bias, float factor,
                                   bool *expert_mask,int32_t *num_local_tokens) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  const int token_idx = blockIdx.y;
   if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
  }
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        VecType sum_vec = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] += current_vec.val[i];
            }
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        VecType sum_vec = input_base_ptr[0 * d_vec + current_idx];
#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec.val[i] += current_vec.val[i];
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
    }
  }
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_reg_bf16_add_bias(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    scalar_t *__restrict__ bias, float factor, bool *expert_mask,int32_t *num_local_tokens,int num_tokens) {
  int token_idx = blockIdx.y;
  int token_num = num_tokens;    
  if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
    token_num = std::min(token_num,num_local_tokens[0]);
  }
  using VecType = Vector<scalar_t, VEC_SIZE>;
  for(token_idx = blockIdx.y;token_idx < token_num; token_idx += gridDim.y){
      const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        Vector<float, VEC_SIZE> sum_vec_f32 = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec_f32.val[i] += static_cast<float>(current_vec.val[i]);
            }
          }
        }
        VecType sum_vec_rounded;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_rounded.val[i] = static_cast<scalar_t>(sum_vec_f32.val[i]);
        }
        if (bias != nullptr) {
          VecType temp_mul_rounded;
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(temp_mul_rounded.val[i]) +
                static_cast<float>(bias_val.val[i]));
          }
          out_base_ptr[current_idx] = sum_vec_rounded;
        } else {
          VecType temp_mul_rounded;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
          out_base_ptr[current_idx] = temp_mul_rounded;
        }
      }
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        Vector<float, VEC_SIZE> sum_vec_f32;
        VecType initial_vec = input_base_ptr[0 * d_vec + current_idx];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_f32.val[i] = static_cast<float>(initial_vec.val[i]);
        }

#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_f32.val[i] += static_cast<float>(current_vec.val[i]);
          }
        }
        VecType sum_vec_rounded;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_rounded.val[i] = static_cast<scalar_t>(sum_vec_f32.val[i]);
        }
        if (bias != nullptr) {
          VecType temp_mul_rounded;
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(temp_mul_rounded.val[i]) +
                static_cast<float>(bias_val.val[i]));
          }
          out_base_ptr[current_idx] = sum_vec_rounded;
        } else {
          VecType temp_mul_rounded;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
          out_base_ptr[current_idx] = temp_mul_rounded;
        }
      }
    }
  }
  }
}


template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec, 
          int total_threads_per_token, int TILE_WIDTH, 
          bool HasBias>
__global__ void moe_sum_kernel_reg_bf16_add_bias_opt(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ bias,
    float factor,
    int num_tokens) {

    int64_t token_idx = blockIdx.y;
    using VecType = Vector<scalar_t, VEC_SIZE>;
    for(;token_idx < num_tokens; token_idx += gridDim.y){
          const int64_t token_offset_input = token_idx * TOPK * d_vec;
          const int64_t token_offset_out   = token_idx * d_vec;
         
          const VecType* input_base = reinterpret_cast<const VecType*>(input + token_offset_input);
          VecType* out_base = reinterpret_cast<VecType*>(out + token_offset_out);
          
          const VecType* bias_base = nullptr;
          if constexpr (HasBias) {
              bias_base = reinterpret_cast<const VecType*>(bias + token_offset_out);
          }

          const int tid = threadIdx.x;
          const int num_vectors = d_vec / VEC_SIZE;
          for (int tile_idx = blockIdx.x * TILE_WIDTH; tile_idx < num_vectors; tile_idx += total_threads_per_token) {
              const int vec_idx = tile_idx + tid;
              
              if (vec_idx < num_vectors) { 
                  float acc[VEC_SIZE] = {0.0f};

                  #pragma unroll VEC_SIZE
                  for (int k = 0; k < TOPK; ++k) {
                    int input_vec_idx = k * num_vectors + vec_idx;
                    VecType loaded_vec = input_base[input_vec_idx];
                    #pragma unroll VEC_SIZE
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        acc[i] += static_cast<float>(loaded_vec.val[i]);
                    }
                  }

                  VecType result_vec;
                  #pragma unroll VEC_SIZE
                  for(int i = 0; i < VEC_SIZE; i++){
                      result_vec.val[i] = static_cast<scalar_t>(acc[i]);
                  }
                  if constexpr (HasBias) {
                      VecType b_val = bias_base[vec_idx];
                      #pragma unroll VEC_SIZE
                      for (int i = 0; i < VEC_SIZE; ++i) {
                          result_vec.val[i]  = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) * factor);
                          result_vec.val[i] = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) + static_cast<float>(b_val.val[i]));
                      }
                  } else {
                      #pragma unroll VEC_SIZE
                      for (int i = 0; i < VEC_SIZE; ++i) {
                          result_vec.val[i]  = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) * factor);
                      }
                  }
                  out_base[vec_idx] = result_vec;
              }
          }
    }
    
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_shared_mem(scalar_t *__restrict__ out,
                                          const scalar_t *__restrict__ input,
                                          scalar_t *__restrict__ bias,
                                          float factor, bool *expert_mask,int32_t* num_local_tokens) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  __shared__ VecType tile[TOPK][TILE_WIDTH];
  const int token_idx = blockIdx.y;
    if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
  }
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {

      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          tile[k][tid_in_block] = input_base_ptr[k * d_vec + current_idx];
        }
      }
      __syncthreads();
      if (current_idx < d_vec) {
        VecType sum_vec = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = tile[k][tid_in_block];
#pragma unroll
            for (int i = 0; i < 8; i++) {
              sum_vec.val[i] += current_vec.val[i];
            }
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
      __syncthreads();
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {

      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          tile[k][tid_in_block] = input_base_ptr[k * d_vec + current_idx];
        }
      }
      __syncthreads();
      if (current_idx < d_vec) {
        VecType sum_vec = tile[0][tid_in_block];
#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = tile[k][tid_in_block];
#pragma unroll
          for (int i = 0; i < 8; i++) {
            sum_vec.val[i] += current_vec.val[i];
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, const int32_t *__restrict__ expert_map) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t *cumsum = shared_mem;
  int32_t *tokens_cnts = (int32_t *)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id =
          expert_map[logical_expert_id]; // Get physical ID
      ++tokens_cnts[(threadIdx.x + 1) * num_experts + mapped_expert_id];
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
    }
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
  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      int32_t rank_post_pad =
          tokens_cnts[threadIdx.x * num_experts + mapped_expert_id] +
          cumsum[mapped_expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[threadIdx.x * num_experts + mapped_expert_id];
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad =
          tokens_cnts[threadIdx.x * num_experts + expert_id] +
          cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[threadIdx.x * num_experts + expert_id];
    }
  }
}
#define DISPATCH_MOE_ALIGN_KERNEL(TEMPLATE_CAPACITY) \
    if (topk_ids.numel() <= TEMPLATE_CAPACITY) { \
         /* 小size*/ \
        threads = TEMPLATE_CAPACITY; \
        if (Is_EP) { \
            moe_align_block_size_kernel_opt_fused_small_size<scalar_t, TEMPLATE_CAPACITY, 8, true, TEMPLATE_CAPACITY><<< \
                1, threads, 0, stream>>>( \
                topk_ids.data_ptr<scalar_t>(), \
                sorted_token_ids.data_ptr<int32_t>(), \
                experts_ids.data_ptr<int32_t>(), \
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts, \
                padded_num_experts, experts_per_warp, block_size, topk_ids.numel(), \
                expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr, \
                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, \
                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr, \
                sorted_token_ids.numel(), Is_fuse_fill); \
        } else { \
            moe_align_block_size_kernel_opt_fused_small_size<scalar_t, TEMPLATE_CAPACITY, 8, false, TEMPLATE_CAPACITY><<< \
                1, threads, 0, stream>>>( \
                topk_ids.data_ptr<scalar_t>(), \
                sorted_token_ids.data_ptr<int32_t>(), \
                experts_ids.data_ptr<int32_t>(), \
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts, \
                padded_num_experts, experts_per_warp, block_size, topk_ids.numel(), \
                expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr, \
                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, \
                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr, \
                sorted_token_ids.numel(), Is_fuse_fill); \
        } \
    } else if (topk_ids.numel() < 16384) { \
        /* 中等size*/ \
        auto align_kernel = moe_align_block_size_kernel_opt_fused_cub<scalar_t, TEMPLATE_CAPACITY, 8, TEMPLATE_CAPACITY>; \
        size_t shared_mem_size = (num_experts + 1) * sizeof(int32_t); \
            if (topk_ids.numel() <= 256) {   \
                 threads = 128;                                       \
              } else if (topk_ids.numel() <= 512) {                   \
                 threads = 256;                                       \
              } else if (topk_ids.numel() <= 1024){                   \
                 threads = 512;                                        \
              }else{                                                   \
                 threads = 1024;                                        \
              }                                                         \
        if(threads <  TEMPLATE_CAPACITY) threads = TEMPLATE_CAPACITY; \
        \
        align_kernel<<<1, threads, shared_mem_size, stream>>>( \
            topk_ids.data_ptr<scalar_t>(), \
            sorted_token_ids.data_ptr<int32_t>(), \
            experts_ids.data_ptr<int32_t>(), \
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts, \
            padded_num_experts, experts_per_warp, block_size, \
            topk_ids.numel(), \
            expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr, \
            expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, \
            num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr, \
            Is_EP, sorted_token_ids.numel(), Is_fuse_fill); \
    } else { \
        /* 大size */ \
        auto align_kernel = moe_align_block_size_kernel_opt_cub<scalar_t, TEMPLATE_CAPACITY, 8, 512>; \
        auto options_int = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device()); \
        torch::Tensor cumsum_buffer = torch::zeros({TEMPLATE_CAPACITY + 2}, options_int); \
        size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp); \
        size_t shared_mem_size = num_experts * sizeof(int32_t); \
        /* num_expert > 512 ?*/                                             \
        threads = 512; \
        int num_blocks = std::min(80*2,(topk_ids.numel() + threads - 1) / threads); \
        /*torch::Tensor global_count = torch::zeros({1}, options_int); */ \
        \
        align_kernel<<<num_blocks, threads, shared_mem_size, stream>>>( \
            topk_ids.data_ptr<scalar_t>(), \
            sorted_token_ids.data_ptr<int32_t>(), \
            experts_ids.data_ptr<int32_t>(), \
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts, \
            padded_num_experts, experts_per_warp, block_size, \
            topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(), \
            expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr, \
            expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, \
            num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr, \
            Is_EP, sorted_token_ids.numel(), Is_fuse_fill,(cumsum_buffer.data_ptr<int32_t>() + TEMPLATE_CAPACITY + 1)); \
        \
        const int block_threads = 1024; \
        num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads; \
        const int max_blocks = 65535; \
        const int actual_blocks = std::min(num_blocks, max_blocks); \
        \
        auto sort_kernel = count_and_sort_expert_tokens_kernel_EP<scalar_t, 8, TEMPLATE_CAPACITY, TEMPLATE_CAPACITY>; \
        sort_kernel<<<std::max(actual_blocks/2,1), block_threads, 0, stream>>>( \
            topk_ids.data_ptr<scalar_t>(), \
            sorted_token_ids.data_ptr<int32_t>(), \
            cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(), \
            num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr, \
            expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr, \
            Is_EP); \
    }

void moe_align_block_size(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    const torch::optional<torch::Tensor> &expert_map =  torch::nullopt,
    const torch::optional<torch::Tensor> &expert_mask =  torch::nullopt, // bool
    const torch::optional<torch::Tensor> &num_local_tokens =  torch::nullopt,bool Is_EP = false,bool Is_fuse_fill = true) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  AT_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = std::max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          auto small_batch_expert_kernel =
              moe_align_block_size_small_batch_expert_kernel<scalar_t>;
          small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(),
              expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                     : nullptr);
        } else {
            int template_capacity = 0;
            if (num_experts > 64 && num_experts <= 128) {
                template_capacity = 128;
            } else if (num_experts > 128 && num_experts <= 256) {
                template_capacity = 256;
            } else if (num_experts > 256 && num_experts <= 512) {
                template_capacity = 512;
            } else if (num_experts > 512 && num_experts <= 1024) {
                template_capacity = 1024;
            }

            bool dispatched = false;

            if (template_capacity > 0) {
                if (template_capacity == 128) {
                    DISPATCH_MOE_ALIGN_KERNEL(128);
                    dispatched = true;
                } else if (template_capacity == 256) {
                    DISPATCH_MOE_ALIGN_KERNEL(256);
                    dispatched = true;
                } else if (template_capacity == 512) {
                    DISPATCH_MOE_ALIGN_KERNEL(512);
                    dispatched = true;
                } else if (template_capacity == 1024) {
                    DISPATCH_MOE_ALIGN_KERNEL(1024);
                    dispatched = true;
                }
            }
            //baseline
            if (!dispatched) {
                auto align_kernel = moe_align_block_size_kernel<scalar_t>;
                auto options_int = torch::TensorOptions()
                                       .dtype(torch::kInt)
                                       .device(topk_ids.device());
                torch::Tensor cumsum_buffer =
                    torch::zeros({num_experts + 1}, options_int);
                size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
                size_t shared_mem_size =
                    num_warps * experts_per_warp * sizeof(int32_t);

                align_kernel<<<1, threads, shared_mem_size, stream>>>(
                    topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                    padded_num_experts, experts_per_warp, block_size,
                    topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
                    expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                           : nullptr);

                const int block_threads = std::min(256, (int)threads);
                const int num_blocks =
                    (topk_ids.numel() + block_threads - 1) / block_threads;
                const int max_blocks = 65535;
                const int actual_blocks = std::min(num_blocks, max_blocks);

                auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t, 8>;
                sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
                    topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(),
                    num_local_tokens.has_value()
                        ? num_local_tokens->data_ptr<int32_t>()
                        : nullptr);
            }
        }
      });
}

void moe_sum(torch::Tensor &input,  // [num_tokens, topk, hidden_size]
             torch::Tensor &output, // [num_tokens, hidden_size]
             const torch::optional<torch::Tensor> &bias = torch::nullopt ,
             const torch::optional<torch::Tensor> &expert_mask = torch::nullopt ,
             const torch::optional<torch::Tensor> &num_local_tokens = torch::nullopt ,
             const float factor = 1.f,const int expect_m = -1) {
  const int hidden_size = input.size(-1);
  int  num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  input = input.is_contiguous() ? input : input.contiguous();
  switch (topk) {
  case 2:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 2, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 2, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;

  case 3:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 3, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 3, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;

  case 4:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 4, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 4, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;
  case 8: {
    /* deepseek */
    if (hidden_size == 7168) {
      int gridy = num_tokens;
      if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
          if(expect_m > 0){
            gridy = expect_m;
          }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
              gridy /= 2; 
            } else if(num_tokens >= 4096)
            { 
               gridy /= 4;
            }
          }
      }
      const int x_dim = 7;
      dim3 Grid(x_dim, gridy);
      const int block_size = 128;
      const int VEC_SIZE = 8;
      if (num_tokens <= 512) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
              if (input.scalar_type() == at::ScalarType::BFloat16) {
                if(expert_mask.has_value() || num_local_tokens.has_value()){
                  moe_sum_kernel_reg_bf16_add_bias<
                    scalar_t, 8, VEC_SIZE, 7168 / VEC_SIZE, 7 * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,
                    expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                            : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                }else{
                  if(bias.has_value()){
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 8, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }else{
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 8, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, false><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }
                }
              } else {
                moe_sum_kernel_reg<scalar_t, 8, VEC_SIZE, 7168 / VEC_SIZE,
                                   7 * block_size, block_size, false>
                    <<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        1.0 / factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
              }
            });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_shared_mem", [&] {
              if (input.scalar_type() == at::ScalarType::BFloat16) {
                  if(expert_mask.has_value() || num_local_tokens.has_value()){
                  moe_sum_kernel_reg_bf16_add_bias<
                    scalar_t, 8, VEC_SIZE, 7168 / VEC_SIZE, 7 * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,
                    expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                            : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                }else{
                  /* deepseek */
                  if(bias.has_value()){
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 8, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }else{
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 8, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, false><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }
                }
              } else {
                moe_sum_kernel_shared_mem<scalar_t, 8, VEC_SIZE,
                                          7168 / VEC_SIZE, 7 * block_size,
                                          block_size, false>
                    <<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        1.0 / factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
              }
            });
      }
    } else if (hidden_size == 2048 || hidden_size == 4096 || hidden_size == 8192){   
      #define DISPATCH_HIDDEN_SIZE(HIDDEN_SIZE_VAL, ...)       \
            switch (HIDDEN_SIZE_VAL) {                         \
              case 2048: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 2048;        \
                constexpr int CONST_X_DIM = 2;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 4096: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 4096;        \
                constexpr int CONST_X_DIM = 4;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 8192: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 8192;        \
                constexpr int CONST_X_DIM = 8;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              default:                                         \
                TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
            }
    int gridy = num_tokens;
    if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
        if(expect_m > 0){
            gridy = expect_m;
        }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
                gridy /= 2; 
            } else if(num_tokens >= 4096) { 
               gridy /= 4;
            }
        }
    }

    if(num_tokens == 288) gridy = 256;
    const int block_size = 128;
    const int VEC_SIZE = 8;
    if (num_tokens <= 512) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
                
                DISPATCH_HIDDEN_SIZE(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                        if(expert_mask.has_value() || num_local_tokens.has_value()){
                             moe_sum_kernel_reg_bf16_add_bias<
                                scalar_t, 8, VEC_SIZE, 
                                CONST_HIDDEN_SIZE / VEC_SIZE, 
                                CONST_X_DIM * block_size,    
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,
                                num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_reg<scalar_t, 8, VEC_SIZE, 
                                           CONST_HIDDEN_SIZE / VEC_SIZE, 
                                           CONST_X_DIM * block_size,    
                                           block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                }); 
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_shared_mem", [&] {
                 DISPATCH_HIDDEN_SIZE(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                         if(expert_mask.has_value() || num_local_tokens.has_value()){
                            moe_sum_kernel_reg_bf16_add_bias<
                            scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                            block_size, true><<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr,
                            num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_shared_mem<scalar_t, 8, VEC_SIZE,
                                                  CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                                                  block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr,
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                 });
            });
    }        
    }else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              moe_sum_kernel<scalar_t, 8, true><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            } else {
              moe_sum_kernel<scalar_t, 8, false><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            }
          });
    }
  } break;
  case 9: {
    if (hidden_size == 7168) {
      int gridy = num_tokens;
      if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
          if(expect_m > 0){
            gridy = expect_m;
          }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
              gridy /= 2; 
            } else if(num_tokens >= 4096)
            { 
               gridy /= 4;
            }
          }
      }
      const int x_dim = 7;
      dim3 Grid(x_dim, gridy);
      const int block_size = 128;
      const int VEC_SIZE = 8;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
              if (input.scalar_type() == at::ScalarType::BFloat16) {
                if(expert_mask.has_value() || num_local_tokens.has_value()){
                  moe_sum_kernel_reg_bf16_add_bias<
                    scalar_t, 9, VEC_SIZE, 7168 / VEC_SIZE, 7 * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,
                    expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                            : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                }else{
                  if(bias.has_value()){
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 9, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, true><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }else{
                    moe_sum_kernel_reg_bf16_add_bias_opt<
                    scalar_t, 9, VEC_SIZE, 7168, x_dim * block_size,
                    block_size, false><<<Grid, block_size, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                    factor,num_tokens);
                  }
                }
              } else {
                moe_sum_kernel_reg<scalar_t, 9, VEC_SIZE, 7168 / VEC_SIZE,
                                   7 * block_size, block_size, false>
                    <<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        1.0 / factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
              }
            });
    } else if (hidden_size == 2048 || hidden_size == 4096 || hidden_size == 8192){   
      #define DISPATCH_HIDDEN_SIZE_9(HIDDEN_SIZE_VAL, ...)       \
            switch (HIDDEN_SIZE_VAL) {                          \
              case 2048: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 2048;        \
                constexpr int CONST_X_DIM = 2;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 4096: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 4096;        \
                constexpr int CONST_X_DIM = 4;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 8192: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 8192;        \
                constexpr int CONST_X_DIM = 8;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              default:                                         \
                TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
            }
    int gridy = num_tokens;
    if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
        if(expect_m > 0){
            gridy = expect_m;
        }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
                gridy /= 2; 
            } else if(num_tokens >= 4096) { 
               gridy /= 4;
            }
        }
    }
    const int block_size = 128;
    const int VEC_SIZE = 8;
    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
                DISPATCH_HIDDEN_SIZE_9(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                        if(expert_mask.has_value() || num_local_tokens.has_value()){
                             moe_sum_kernel_reg_bf16_add_bias<
                                scalar_t, 9, VEC_SIZE, 
                                CONST_HIDDEN_SIZE / VEC_SIZE, 
                                CONST_X_DIM * block_size,    
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,
                                num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_reg<scalar_t, 9, VEC_SIZE, 
                                           CONST_HIDDEN_SIZE / VEC_SIZE, 
                                           CONST_X_DIM * block_size,    
                                           block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                }); 
            });        
    }else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              moe_sum_kernel<scalar_t, 9, true><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            } else {
              moe_sum_kernel<scalar_t, 9, false><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            }
          });
    }
  } break;

  default:
    at::sum_out(output, input, 1);
    break;
  }
}
} // namespace native
} // namespace at