#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#define WARP_SIZE 1

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

namespace at {
namespace native {

template <typename scalar_t, int smem_size>
__global__ void moe_align_block_size_kernel_opt_fused(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ expert_map) {

  __shared__ int32_t shared_counts[smem_size + 1];
  int32_t *cumsum = shared_counts;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (int i = tid; i <= num_experts; i += stride) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      atomicAdd(&shared_counts[mapped_expert_id], 1);
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = topk_ids[i];
      atomicAdd(&shared_counts[expert_id], 1);
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
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  __syncthreads();
  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      int32_t rank_post_pad = atomicAdd(&shared_counts[mapped_expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }
}

template <typename scalar_t, int seme_size>
__global__ void moe_align_block_size_kernel_opt(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum,
    int32_t *__restrict__ expert_map) {

  __shared__ int32_t shared_counts[seme_size + 1];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (int i = tid; i <= num_experts; i += stride) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      atomicAdd(&shared_counts[mapped_expert_id], 1);
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = topk_ids[i];
      atomicAdd(&shared_counts[expert_id], 1);
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
      expert_ids[i / block_size] = threadIdx.x;
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

  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id =
          expert_map[logical_expert_id]; // Get physical ID
      int warp_idx = mapped_expert_id / experts_per_warp;
      int expert_offset = mapped_expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = topk_ids[i]; // Original code for no map
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
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
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ cumsum_buffer,
    size_t numel, const int32_t *__restrict__ expert_map) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      int32_t rank_post_pad = atomicAdd(&cumsum_buffer[mapped_expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
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

template <typename scalar_t, int TOPK>
__global__ void
moe_sum_kernel(scalar_t *__restrict__ out,         // [..., d]
               const scalar_t *__restrict__ input, // [..., topk, d]
               const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += input[token_idx * TOPK * d + k * d + idx];
    }
    out[token_idx * d + idx] = x;
  }
}
template <typename scalar_t, int TOPK, int group_size, int VEC_SIZE, int d_vec,
          int d, int total_threads_per_token>
__global__ void moe_sum_kernel_opt(scalar_t *__restrict__ out, // [..., d]
                                   const scalar_t *__restrict__ input) {

  using VecType = Vector<scalar_t, VEC_SIZE>;

  const int token_idx = blockIdx.y;
  const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  //  const int total_threads_per_token = group_size * blockDim.x;

  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  for (int idx = start_idx; idx < d_vec; idx += total_threads_per_token) {
    VecType sum_vec = input_base_ptr[0 * d_vec + idx];
    for (int k = 1; k < TOPK; k += 2) {
      VecType vec1 = input_base_ptr[k * d_vec + idx];
      VecType vec2;
      if (k + 1 < TOPK) {
        vec2 = input_base_ptr[(k + 1) * d_vec + idx];
      }
      __half2 *sum_h2 = reinterpret_cast<__half2 *>(sum_vec.val);
      const __half2 *current_h2_1 = reinterpret_cast<const __half2 *>(vec1.val);
      sum_h2[0] = __hadd2(sum_h2[0], current_h2_1[0]);
      sum_h2[1] = __hadd2(sum_h2[1], current_h2_1[1]);
      sum_h2[2] = __hadd2(sum_h2[2], current_h2_1[2]);
      sum_h2[3] = __hadd2(sum_h2[3], current_h2_1[3]);
      if (k + 1 < TOPK) {
        const __half2 *current_h2_2 =
            reinterpret_cast<const __half2 *>(vec2.val);
        sum_h2[0] = __hadd2(sum_h2[0], current_h2_2[0]);
        sum_h2[1] = __hadd2(sum_h2[1], current_h2_2[1]);
        sum_h2[2] = __hadd2(sum_h2[2], current_h2_2[2]);
        sum_h2[3] = __hadd2(sum_h2[3], current_h2_2[3]);
      }
    }
    out_base_ptr[idx] = sum_vec;
  }
}
template <typename scalar_t, int TOPK, int group_size, int VEC_SIZE, int d_vec,
          int d, int total_threads_per_token>
__global__ void
moe_sum_kernel_fully_unrolled(scalar_t *__restrict__ out, // [..., d]
                              const scalar_t *__restrict__ input) {

  using VecType = Vector<scalar_t, VEC_SIZE>;

  const int token_idx = blockIdx.y;
  const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;

  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  for (int idx = start_idx; idx < d_vec; idx += total_threads_per_token) {

    VecType sum_vec = input_base_ptr[0 * d_vec + idx];

    VecType vec1 = input_base_ptr[1 * d_vec + idx];
    VecType vec2 = input_base_ptr[2 * d_vec + idx];
    VecType vec3 = input_base_ptr[3 * d_vec + idx];
    VecType vec4 = input_base_ptr[4 * d_vec + idx];
    VecType vec5 = input_base_ptr[5 * d_vec + idx];
    VecType vec6 = input_base_ptr[6 * d_vec + idx];
    VecType vec7 = input_base_ptr[7 * d_vec + idx];
    __half2 *sum_h2 = reinterpret_cast<__half2 *>(sum_vec.val);

#define ACCUMULATE_VEC(vec)                                                    \
  do {                                                                         \
    const __half2 *h2_ptr = reinterpret_cast<const __half2 *>(&(vec.val));     \
    sum_h2[0] = __hadd2(sum_h2[0], h2_ptr[0]);                                 \
    sum_h2[1] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[2] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[3] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[4] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[5] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[6] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
    sum_h2[7] = __hadd2(sum_h2[1], h2_ptr[1]);                                 \
  } while (0)

    ACCUMULATE_VEC(vec1);
    ACCUMULATE_VEC(vec2);
    ACCUMULATE_VEC(vec3);
    ACCUMULATE_VEC(vec4);
    ACCUMULATE_VEC(vec5);
    ACCUMULATE_VEC(vec6);
    ACCUMULATE_VEC(vec7);

#undef ACCUMULATE_VEC

    out_base_ptr[idx] = sum_vec;
  }
}

template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token,int TILE_WIDTH>
__global__ void moe_sum_kernel_reg(scalar_t *__restrict__ out,
                                   const scalar_t *__restrict__ input) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  const int token_idx = blockIdx.y;
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;

  for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
       tile_start_idx += total_threads_per_token) {
    const int current_idx = tile_start_idx + tid_in_block;
    if (current_idx < d_vec) {
      VecType sum_vec = input_base_ptr[0 * d_vec + current_idx];
#pragma unroll
      for (int k = 1; k < TOPK; ++k) {
        VecType current_vec = input_base_ptr[k * d_vec + current_idx];
        #pragma unroll
            for(int i = 0; i < 8; i ++){
                sum_vec.val[i] += current_vec.val[i];
            }
      }
      out_base_ptr[current_idx] = sum_vec;
    }
  }
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,int total_threads_per_token,int TILE_WIDTH>
__global__ void moe_sum_kernel_shared_mem(scalar_t *__restrict__ out,
                                          const scalar_t *__restrict__ input) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  __shared__ VecType tile[TOPK][TILE_WIDTH];
  const int token_idx = blockIdx.y;
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;

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
            for(int i = 0; i < 8; i ++){
                sum_vec.val[i] += current_vec.val[i];
            }
      }
      out_base_ptr[current_idx] = sum_vec;
    }
    __syncthreads();
  }
}
// 内外预取
template <typename scalar_t, int TOPK, int group_size, int VEC_SIZE>
__global__ void
moe_sum_kernel_2level_db(scalar_t *__restrict__ out,         // [..., d]
                         const scalar_t *__restrict__ input, // [..., topk, d]
                         const int d) {

  using VecType = Vector<scalar_t, VEC_SIZE>;

  const int d_vec = d / VEC_SIZE;
  const int token_idx = blockIdx.x / group_size;
  const int start_idx = (blockIdx.x % group_size) * blockDim.x + threadIdx.x;
  const int total_threads_per_token = group_size * blockDim.x;

  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  int current_idx = start_idx;

  VecType current_k0_vec, current_k1_vec;
  if (current_idx < d_vec) {
    current_k0_vec = input_base_ptr[0 * d_vec + current_idx];
    if (TOPK > 1) {
      current_k1_vec = input_base_ptr[1 * d_vec + current_idx];
    }
  }
  for (; current_idx < d_vec; current_idx += total_threads_per_token) {
    const int next_idx = current_idx + total_threads_per_token;

    VecType prefetch_k0_vec, prefetch_k1_vec;
    if (next_idx < d_vec) {
      prefetch_k0_vec = input_base_ptr[0 * d_vec + next_idx];
      if (TOPK > 1) {
        prefetch_k1_vec = input_base_ptr[1 * d_vec + next_idx];
      }
    }

    VecType sum_vec = current_k0_vec;

    if (TOPK > 1) {

      VecType compute_k_vec = current_k1_vec;
      VecType prefetch_k_vec;

      if (2 < TOPK) {
        prefetch_k_vec = input_base_ptr[2 * d_vec + current_idx];
      }

#pragma unroll 4
      for (int k = 1; k < TOPK - 1; ++k) {

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          sum_vec.val[i] += compute_k_vec.val[i];
        }

        compute_k_vec = prefetch_k_vec;

        if (k + 2 < TOPK) {
          prefetch_k_vec = input_base_ptr[(k + 2) * d_vec + current_idx];
        }
      }

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        sum_vec.val[i] += compute_k_vec.val[i];
      }
    }

    out_base_ptr[current_idx] = sum_vec;

    current_k0_vec = prefetch_k0_vec;
    current_k1_vec = prefetch_k1_vec;
  }
}
// 内层topk预取
template <typename scalar_t, int TOPK, int group_size, int VEC_SIZE>
__global__ void moe_sum_kernel_prefetch_topk(
    scalar_t *__restrict__ out,         // [..., d]
    const scalar_t *__restrict__ input, // [..., topk, d]
    const int d) {

  using VecType = Vector<scalar_t, VEC_SIZE>;

  const int d_vec = d / VEC_SIZE;
  const int token_idx = blockIdx.x / group_size;
  const int start_idx = (blockIdx.x % group_size) * blockDim.x + threadIdx.x;
  const int total_threads_per_token = group_size * blockDim.x;

  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  for (int idx = start_idx; idx < d_vec; idx += total_threads_per_token) {
    VecType sum_vec = input_base_ptr[0 * d_vec + idx];

    if (TOPK > 1) {
      VecType current_vec = input_base_ptr[1 * d_vec + idx];

#pragma unroll 4
      for (int k = 1; k < TOPK - 1; ++k) {
        VecType prefetch_vec = input_base_ptr[(k + 1) * d_vec + idx];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          sum_vec.val[i] += current_vec.val[i];
        }
        current_vec = prefetch_vec;
      }

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        sum_vec.val[i] += current_vec.val[i];
      }
    }
    out_base_ptr[idx] = sum_vec;
  }
}
// 外层预取
template <typename scalar_t, int TOPK, int TILE_M, int TILE_D_VEC, int VEC_SIZE>
__global__ void moe_sum_kernel_prefetch(scalar_t *__restrict__ out,
                                        const scalar_t *__restrict__ input,
                                        const int num_tokens, const int d) {

  using VecType = Vector<scalar_t, VEC_SIZE>;
  const int d_vec = d / VEC_SIZE;
  const int token_start_block = blockIdx.x * TILE_M;
  const int dim_start_block_vec = blockIdx.y * blockDim.x * TILE_D_VEC;
  const int local_tid = threadIdx.x;
  const VecType *input_base_ptr = reinterpret_cast<const VecType *>(input);
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out);
  for (int m = 0; m < TILE_M; ++m) {
    const int token_idx = token_start_block + m;
    if (token_idx >= num_tokens) {
      break;
    }
    for (int d_i = 0; d_i < TILE_D_VEC; ++d_i) {
      const int dim_offset_vec =
          dim_start_block_vec + d_i * blockDim.x + local_tid;

      if (dim_offset_vec >= d_vec) {
        break;
      }
      const VecType *current_input_ptr =
          input_base_ptr + token_idx * TOPK * d_vec + dim_offset_vec;

      VecType sum_vec = current_input_ptr[0 * d_vec];
      VecType next_vec = current_input_ptr[1 * d_vec];
#pragma unroll
      for (int k = 1; k < TOPK - 1; ++k) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          sum_vec.val[i] += next_vec.val[i];
        }
        next_vec = current_input_ptr[(k + 1) * d_vec];
      }

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        sum_vec.val[i] += next_vec.val[i];
      }
      out_base_ptr[token_idx * d_vec + dim_offset_vec] = sum_vec;
    }
  }
}
// 三级流水
template <typename scalar_t, int TOPK, int group_size, int VEC_SIZE>
__global__ void
moe_sum_kernel_3stage(scalar_t *__restrict__ out,         // [..., d]
                      const scalar_t *__restrict__ input, // [..., topk, d]
                      const int d) {

  using VecType = Vector<scalar_t, VEC_SIZE>;

  const int d_vec = d / VEC_SIZE;
  const int token_idx = blockIdx.x / group_size;
  const int start_idx = (blockIdx.x % group_size) * blockDim.x + threadIdx.x;
  const int total_threads_per_token = group_size * blockDim.x;

  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  for (int idx = start_idx; idx < d_vec; idx += total_threads_per_token) {

    VecType sum_vec = input_base_ptr[0 * d_vec + idx];

    VecType vec_k1 = input_base_ptr[1 * d_vec + idx];
    VecType vec_k2 = input_base_ptr[2 * d_vec + idx];
#pragma unroll 4
    for (int k = 1; k < TOPK - 2; ++k) {

      VecType vec_k3 = input_base_ptr[(k + 2) * d_vec + idx];
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        sum_vec.val[i] += vec_k1.val[i];
      }
      vec_k1 = vec_k2;
      vec_k2 = vec_k3;
    }
#pragma unroll 8
    for (int i = 0; i < VEC_SIZE; ++i) {
      sum_vec.val[i] += vec_k1.val[i];
    }
#pragma unroll 8
    for (int i = 0; i < VEC_SIZE; ++i) {
      sum_vec.val[i] += vec_k2.val[i];
    }

    out_base_ptr[idx] = sum_vec;
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
// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          const torch::optional<torch::Tensor> &expert_map) {
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
          if (num_experts == 256) {
            if (topk_ids.numel() <= 1024) {
              auto align_kernel =
                  moe_align_block_size_kernel_opt_fused<scalar_t, 256>;
              size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
              size_t shared_mem_size = (num_experts + 1) * sizeof(int32_t);
              if (topk_ids.numel() <= 256) {
                threads = 256;
              } else if (topk_ids.numel() <= 512) {
                threads = 512;
              } else {
                threads = 1024;
              }
              align_kernel<<<1, threads, shared_mem_size, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  experts_ids.data_ptr<int32_t>(),
                  num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                  padded_num_experts, experts_per_warp, block_size,
                  topk_ids.numel(),
                  expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                         : nullptr);
            } else {
              auto align_kernel =
                  moe_align_block_size_kernel_opt<scalar_t, 256>;
              auto options_int = torch::TensorOptions()
                                     .dtype(torch::kInt)
                                     .device(topk_ids.device());
              torch::Tensor cumsum_buffer =
                  torch::zeros({num_experts + 1}, options_int);
              size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
              size_t shared_mem_size = num_experts * sizeof(int32_t);
              threads = 1024;
              align_kernel<<<1, threads, shared_mem_size, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  experts_ids.data_ptr<int32_t>(),
                  num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                  padded_num_experts, experts_per_warp, block_size,
                  topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
                  expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                         : nullptr);

              const int block_threads = std::min(64, (int)threads);
              const int num_blocks =
                  (topk_ids.numel() + block_threads - 1) / block_threads;
              const int max_blocks = 65535;
              const int actual_blocks = std::min(num_blocks, max_blocks);

              auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
              sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
                  topk_ids.data_ptr<scalar_t>(),
                  sorted_token_ids.data_ptr<int32_t>(),
                  cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(),
                  expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                         : nullptr);
            }

          } else {
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

            auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
            sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(),
                expert_map.has_value() ? expert_map->data_ptr<int32_t>()
                                       : nullptr);
          }
        }
      });
}

void moe_sum(torch::Tensor &input,  // [num_tokens, topk, hidden_size]
             torch::Tensor &output) // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
  case 2:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
              output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
              hidden_size);
        });
    break;

  case 3:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
              output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
              hidden_size);
        });
    break;

  case 4:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
              output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
              hidden_size);
        });
    break;
  case 8: {
    if (hidden_size == 7168) {
      dim3 Grid(7, num_tokens);
      const int block_size = 128;
      const int VEC_SIZE = 8;
      if (num_tokens <= 512) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel", [&] {
              moe_sum_kernel_reg<scalar_t, 8, VEC_SIZE, 7168 / VEC_SIZE, 7 * block_size,block_size>
                  <<<Grid, block_size, 0, stream>>>(output.data_ptr<scalar_t>(),
                                             input.data_ptr<scalar_t>());
            });
      } else {
       AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel", [&] {
              moe_sum_kernel_shared_mem<scalar_t, 8, VEC_SIZE, 7168 / VEC_SIZE, 7 * block_size,block_size>
                  <<<Grid, block_size, 0, stream>>>(output.data_ptr<scalar_t>(),
                                             input.data_ptr<scalar_t>());
            });
      }
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            moe_sum_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size);
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
