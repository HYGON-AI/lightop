#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE 64
namespace at {
namespace native {

constexpr int EP_SCAN_MAX_EXPERTS = 1024;

template <typename T, int N> struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];
  __device__ inline T &operator[](int i) { return val[i]; }
  __device__ inline const T &operator[](int i) const { return val[i]; }
};

static inline __device__ int8_t float_to_int8_rn_fused(float x) {
  int32_t val = __float2int_rn(x);
  val = std::max(-128, std::min(127, val));
  return static_cast<int8_t>(val);
}

static inline __device__ uint32_t fp32_to_bits_lightop(float f) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

static inline __device__ float fp32_from_bits_lightop(uint32_t u) {
  union {
    float f;
    uint32_t u;
  } v;
  v.u = u;
  return v.f;
}

inline __device__ uint8_t float_to_fp8e4m3_lightop(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  uint32_t f_bits = fp32_to_bits_lightop(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = 0x7f;
  } else if (f_bits < (UINT32_C(121) << 23)) {
    f_bits = fp32_to_bits_lightop(fp32_from_bits_lightop(f_bits) +
                                  fp32_from_bits_lightop(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;
    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

inline __device__ uint8_t float_to_fp8e5m2_lightop(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
  uint32_t f_bits = fp32_to_bits_lightop(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else if (f_bits < (UINT32_C(113) << 23)) {
    f_bits = fp32_to_bits_lightop(fp32_from_bits_lightop(f_bits) +
                                  fp32_from_bits_lightop(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint32_t mant_odd = (f_bits >> 21) & 1;
    f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

inline __device__ uint32_t float_to_fp8e4m3_x4_lightop(float a, float b,
                                                       float c, float d) {
#if defined(__gfx938__)
  int32_t res;
  res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
  res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
  return res;
#else
  uint32_t out;
  reinterpret_cast<uint8_t *>(&out)[0] = float_to_fp8e4m3_lightop(a);
  reinterpret_cast<uint8_t *>(&out)[1] = float_to_fp8e4m3_lightop(b);
  reinterpret_cast<uint8_t *>(&out)[2] = float_to_fp8e4m3_lightop(c);
  reinterpret_cast<uint8_t *>(&out)[3] = float_to_fp8e4m3_lightop(d);
  return out;
#endif
}

inline __device__ uint32_t float_to_fp8e5m2_x4_lightop(float a, float b,
                                                       float c, float d) {
#if defined(__gfx938__)
  int32_t res;
  res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
  res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
  return res;
#else
  uint32_t out;
  reinterpret_cast<uint8_t *>(&out)[0] = float_to_fp8e5m2_lightop(a);
  reinterpret_cast<uint8_t *>(&out)[1] = float_to_fp8e5m2_lightop(b);
  reinterpret_cast<uint8_t *>(&out)[2] = float_to_fp8e5m2_lightop(c);
  reinterpret_cast<uint8_t *>(&out)[3] = float_to_fp8e5m2_lightop(d);
  return out;
#endif
}

template <typename T, int Num_threads>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int offset = Num_threads / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down(val, offset));
  }
  return val;
}

template <int CUB_BLOCK_THREADS>
__global__ void
ep_scan_kernel(const int32_t *__restrict__ num_tokens_per_expert,
               int32_t *__restrict__ expert_counters, int num_experts,
               int alignment) {
  int tid = threadIdx.x;
  int32_t count = 0;
  if (tid < num_experts)
    count = num_tokens_per_expert[tid];

  int32_t padded_count = 0;
  if (tid < num_experts)
    padded_count = (count + alignment - 1) & ~(alignment - 1);

  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int32_t start_offset = 0;
  cub::Sum scan_op;
  BlockScan(temp_storage).ExclusiveScan(padded_count, start_offset, 0, scan_op);

  if (tid < num_experts) {
    expert_counters[tid] = start_offset;
  }
}

template <int CUB_BLOCK_THREADS, bool FILL_PADDED_WITH_EXPERT>
__global__ void ep_scan_fill_m_indices_kernel(
    const int32_t *__restrict__ num_tokens_per_expert,
    int32_t *__restrict__ expert_counters, int32_t *__restrict__ m_indices,
    int num_experts, int alignment) {
  int tid = threadIdx.x;
  int32_t count = 0;
  if (tid < num_experts) {
    count = num_tokens_per_expert[tid];
  }

  int32_t padded_count = 0;
  if (tid < num_experts) {
    padded_count = (count + alignment - 1) & ~(alignment - 1);
  }

  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int32_t start_offset = 0;
  cub::Sum scan_op;
  BlockScan(temp_storage).ExclusiveScan(padded_count, start_offset, 0, scan_op);

  if (tid < num_experts) {
    expert_counters[tid] = start_offset;
    int32_t fill_value = FILL_PADDED_WITH_EXPERT ? tid : -1;
    for (int32_t i = 0; i < padded_count; ++i) {
      m_indices[start_offset + i] = fill_value;
    }
  }
}

template <typename scalar_t, int BLOCK_SIZE, int GROUP_SIZE, int num_warps>
__launch_bounds__(BLOCK_SIZE) __global__ void quant_scatter_opt_kernel(
    const scalar_t *__restrict__ input, 
    const int64_t *__restrict__ topk_ids, // int64_t
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ expert_counters, int8_t *__restrict__ aq_out,
    float *__restrict__ aq_scale_out, int32_t *__restrict__ inv_perm,
    int32_t *__restrict__ m_indices, int num_tokens, int K, int num_experts,
    int hidden_size, bool has_expert_map) {
  constexpr int VEC_SIZE = 8;
  extern __shared__ __align__(16) int8_t shmem_int8[];

  __shared__ float s_max_abs;
  __shared__ float warp_maxs[num_warps];
  __shared__ bool s_any_valid;
  const int tid = threadIdx.x;
  const int lane_id = tid & 63;
  const int warp_id = tid / 64;
  __shared__ bool all_ineffective;

  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens;
       token_idx += gridDim.x) {

    int group_id = tid / GROUP_SIZE;
    int group_lane = tid % GROUP_SIZE;
    constexpr int NUM_GROUPS = BLOCK_SIZE / GROUP_SIZE;

    __shared__ int64_t s_dest_rows[NUM_GROUPS];
    __shared__ bool s_is_valid[NUM_GROUPS];

    if (tid == 0)
      s_any_valid = false;
    __syncthreads();

    if (tid < K) {
      int expert_id = (int)topk_ids[token_idx * K + tid]; 
      bool is_valid = expert_id >= 0 && expert_id < num_experts;

      if (is_valid && has_expert_map) {
        int mapped_id = expert_map[expert_id];
        if (mapped_id == -1)
          is_valid = false;
        else
          expert_id = mapped_id;
      }

      s_is_valid[tid] = is_valid;

      if (is_valid) {
        s_any_valid = true;
        int64_t dest_row = atomicAdd(&expert_counters[expert_id], 1);
        s_dest_rows[tid] = dest_row;

        int64_t row_idx = token_idx * K + tid;
        inv_perm[row_idx] = (int32_t)dest_row;
        if (m_indices)
          m_indices[dest_row] = expert_id;
      } else {
        int64_t row_idx = token_idx * K + tid;
        inv_perm[row_idx] = -1;
      }
    }

    __syncthreads();

    if (!s_any_valid) {
      continue;
    }

    float local_max_abs = 0.0f;
    int64_t input_offset = token_idx * hidden_size;

    using VecType = AlignedVector<scalar_t, VEC_SIZE>;
    int num_vecs = hidden_size / VEC_SIZE;

    for (int i = tid; i < num_vecs; i += blockDim.x) {
      VecType vec_input = *reinterpret_cast<const VecType *>(
          input + input_offset + i * VEC_SIZE);

#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        float val = static_cast<float>(vec_input[v]);
        local_max_abs = fmaxf(local_max_abs, fabsf(val));
      }
    }

    local_max_abs = warpReduceMax<float, 64>(local_max_abs);

    if (lane_id == 0)
      warp_maxs[warp_id] = local_max_abs;

    __syncthreads();

    if (warp_id == 0) {
      float block_val = (tid < num_warps) ? warp_maxs[tid] : 0.0f;
      block_val = warpReduceMax<float, num_warps>(block_val);

      if (tid == 0) {
        s_max_abs = block_val;
      }
    }
    __syncthreads();

    float inv_scale = 127.0f / s_max_abs;

    float scale_for_output = 1.0 / inv_scale;

    for (int i = tid; i < num_vecs; i += blockDim.x) {
      int offset = i * VEC_SIZE;
      VecType vec_input =
          *reinterpret_cast<const VecType *>(input + input_offset + offset);

      int64_t packed = 0;
      int8_t *packed_ptr = reinterpret_cast<int8_t *>(&packed);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        packed_ptr[v] = float_to_int8_rn_fused(
            static_cast<float>(vec_input[v]) * inv_scale);
      }
      *reinterpret_cast<int64_t *>(&shmem_int8[offset]) = packed;
    }

    __syncthreads();

    for (int k_base = 0; k_base < K; k_base += NUM_GROUPS) {

      int k = k_base + group_id;

      if (s_is_valid[group_id]) {
        int64_t dest_row = s_dest_rows[group_id];
        int64_t dst_offset = dest_row * hidden_size;
        int8_t *dst_ptr = aq_out + dst_offset;
        if (group_lane == 0) {
          aq_scale_out[dest_row] = scale_for_output;
        }
        using CopyType = int4;
        int num_copy_vecs = hidden_size / sizeof(CopyType);

        CopyType *src_shmem_vec = reinterpret_cast<CopyType *>(shmem_int8);
        CopyType *dst_global_vec = reinterpret_cast<CopyType *>(dst_ptr);

        for (int i = group_lane; i < num_copy_vecs; i += GROUP_SIZE) {
          dst_global_vec[i] = src_shmem_vec[i];
        }
      }
    }
  }
}

template <typename T, typename ScaleT, int SCATTER_THREADS_PER_GROUP,
          int SCATTER_MAX_K>
__launch_bounds__(1024) __global__
    void ep_scatter_kernel(const T *__restrict__ aq,            // int8
                           const ScaleT *__restrict__ aq_scale, // float
                           const int64_t *__restrict__ topk_ids, // int64_t
                           const int32_t *__restrict__ expert_map,
                           int32_t *__restrict__ expert_counters,
                           T *__restrict__ aq_out,
                           ScaleT *__restrict__ aq_scale_out,
                           int32_t *__restrict__ inv_perm,
                           int32_t *__restrict__ m_indices, int num_tokens,
                           int K, int num_experts, int hidden_size,
                           int scale_hidden_size, bool has_expert_map) {

  extern __shared__ int4 shmem_data_int4[];

  int64_t token_idx = blockIdx.x;
  int tid = threadIdx.x;
  for(;token_idx <  num_tokens; token_idx += gridDim.x){
         int64_t global_src_offset = token_idx * hidden_size;
        const int2 *src_vec_global =
            reinterpret_cast<const int2 *>(aq + global_src_offset);
        int2 *dst_vec_shmem = reinterpret_cast<int2 *>(shmem_data_int4);
        int H_vec = hidden_size / 8;

        for (int i = tid; i < H_vec; i += blockDim.x) {
            dst_vec_shmem[i] = src_vec_global[i];
        }

        __shared__ int dest_row_shared[SCATTER_MAX_K];
        __shared__ bool is_valid_shared[SCATTER_MAX_K];

        int k_idx = tid / SCATTER_THREADS_PER_GROUP;
        int lane_id = tid % SCATTER_THREADS_PER_GROUP;

        if (lane_id == 0) {
            bool is_valid = k_idx < K;
            int64_t row_idx = token_idx * K + k_idx;
            int expert_id = -1;

           if (is_valid) {
              int global_expert_id = static_cast<int>(topk_ids[row_idx]);

              if (has_expert_map) {
                if (global_expert_id < 0) {
                  is_valid = false;
                } else {
                  int mapped_id = expert_map[global_expert_id];
                  if (mapped_id < 0 || mapped_id >= num_experts) {
                    is_valid = false;
                  } else {
                    expert_id = mapped_id;
                  }
                }
              } else {
                if (global_expert_id < 0 || global_expert_id >= num_experts) {
                  is_valid = false;
                } else {
                  expert_id = global_expert_id;
                }
              }
          }

            is_valid_shared[k_idx] = is_valid;

            if (k_idx < K) {
                if (is_valid) {
                    int dest_row = atomicAdd(&expert_counters[expert_id], 1);
                    dest_row_shared[k_idx] = dest_row;
                    inv_perm[row_idx] = dest_row;
                    if (m_indices != nullptr) {
                        m_indices[dest_row] = expert_id;
                    }
                    aq_scale_out[dest_row] = aq_scale[token_idx];
                } else {
                    inv_perm[row_idx] = -1;
                }
            }
        }

        __syncthreads();

        H_vec = hidden_size / 16;
        if (is_valid_shared[k_idx]) {
            int64_t dest_row = dest_row_shared[k_idx];
            int64_t dst_offset = dest_row * hidden_size;
            int4 *dst_vec_global = reinterpret_cast<int4 *>(aq_out + dst_offset);
            const int4 *src_vec_shmem = shmem_data_int4;
        #pragma unroll
            for (int v = lane_id; v < H_vec; v += SCATTER_THREADS_PER_GROUP) {
            dst_vec_global[v] = src_vec_shmem[v];
            }
        }
         __syncthreads();
  }
 
}

template <typename ScaleT, int SCATTER_MAX_K>
__launch_bounds__(1024) __global__ void ep_scatter_kernel_1024_opt(
    const int8_t *__restrict__ aq,            
    const ScaleT *__restrict__ aq_scale, 
    const int64_t *__restrict__ topk_ids, //int64_t
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ expert_counters,
    int8_t *__restrict__ aq_out,
    ScaleT *__restrict__ aq_scale_out,
    int32_t *__restrict__ inv_perm,
    int32_t *__restrict__ m_indices,
    int num_tokens,
    int K,
    int num_experts,
    int hidden_size,
    bool has_expert_map) {

  //存储 2 个 Token 的数据
  extern __shared__ float4 shmem_data_f4[]; 

  int tid = threadIdx.x;
  
  int sub_block_idx = tid / 512;     
  int lane_in_sub = tid % 512;      
  
  int64_t global_token_idx = blockIdx.x * 2 + sub_block_idx;

  // 如果 num_tokens 是奇数会越界
  if (global_token_idx >= num_tokens) return;


  int H_vec = hidden_size / 16; 
  float4* my_shmem = shmem_data_f4 + (sub_block_idx * H_vec);

  int64_t global_src_offset = global_token_idx * hidden_size;

  if (lane_in_sub < H_vec) {
      my_shmem[lane_in_sub] = *reinterpret_cast<const float4*>(aq + global_src_offset + lane_in_sub * 16);
  }

  
  int group_id = lane_in_sub / 64; 
  int lane_id = lane_in_sub % 64;
  int32_t my_dest_row = -1;
 // int my_valid_int = 0;    
//   int shmem_meta_offset = sub_block_idx * SCATTER_MAX_K;

  if (group_id < K) {
      if (lane_id == 0) {
          int64_t row_idx = global_token_idx * K + group_id;
          int global_expert_id = static_cast<int>(topk_ids[row_idx]);
          int expert_id = -1;
          bool is_valid = true;

          if (has_expert_map) {
              if (global_expert_id < 0) {
                  is_valid = false;
              } else {
                  int mapped_id = expert_map[global_expert_id];

                  if (mapped_id < 0 || mapped_id >= num_experts) {
                      is_valid = false;
                  } else {
                      expert_id = mapped_id;
                  }
              }
          } else {
              if (global_expert_id < 0 || global_expert_id >= num_experts) {
                  is_valid = false;
              } else {
                  expert_id = global_expert_id;
              }
          }

          if (is_valid) {
        
              my_dest_row = atomicAdd(&expert_counters[expert_id], 1);
             // my_valid_int = 1;
              inv_perm[row_idx] = my_dest_row;
              if (m_indices) m_indices[my_dest_row] = expert_id;
              aq_scale_out[my_dest_row] = aq_scale[global_token_idx];
          } else {
              inv_perm[row_idx] = -1;
          }
      }
    }
      __syncthreads(); 
      
    my_dest_row = __shfl(my_dest_row, 0);

     if (my_dest_row != -1) {
          int64_t dest_row = static_cast<int64_t>(my_dest_row);
          int64_t dst_offset = dest_row * hidden_size;
          int8_t *dst_ptr_base = aq_out + dst_offset;

          int v = lane_id;
          #pragma unroll 4
          for (; v < H_vec; v += 64) {
             float4 val = my_shmem[v];
             *reinterpret_cast<float4*>(dst_ptr_base + v * 16) = val;
          }
    }
}

template <typename topk_t, int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS) __global__ void ep_build_m_indices_kernel(
    const topk_t *__restrict__ topk_ids, int32_t *__restrict__ m_indices,
    int64_t topk_numel, int64_t total_elements, int num_experts,
    int alignment) {
  __shared__ int32_t counts[EP_SCAN_MAX_EXPERTS];
  __shared__ int32_t starts[EP_SCAN_MAX_EXPERTS];

  const int tid = threadIdx.x;
  for (int expert = tid; expert < num_experts; expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();

  for (int64_t idx = tid; idx < topk_numel; idx += blockDim.x) {
    int expert = static_cast<int>(topk_ids[idx]);
    if (expert >= 0 && expert < num_experts) {
      atomicAdd(&counts[expert], 1);
    }
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    const int32_t count = counts[tid];
    padded_count = ((count + alignment - 1) / alignment) * alignment;
  }

  typedef cub::BlockScan<int32_t, BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int32_t start = 0;
  cub::Sum scan_op;
  BlockScan(temp_storage).ExclusiveScan(padded_count, start, 0, scan_op);

  if (tid < num_experts) {
    starts[tid] = start;
  }
  __syncthreads();

  for (int64_t idx = tid; idx < total_elements; idx += blockDim.x) {
    m_indices[idx] = -1;
  }
  __syncthreads();

  for (int expert = 0; expert < num_experts; ++expert) {
    const int32_t count = counts[expert];
    const int32_t start_offset = starts[expert];
    for (int32_t offset = tid; offset < count; offset += blockDim.x) {
      const int64_t dst = static_cast<int64_t>(start_offset) + offset;
      if (dst < total_elements) {
        m_indices[dst] = expert;
      }
    }
  }
}

template <typename scalar_t, int BLOCK_SIZE, int SCATTER_THREADS_PER_GROUP,
          int SCATTER_MAX_K>
__launch_bounds__(BLOCK_SIZE) __global__ void ep_fp8_quant_scatter_kernel(
    const scalar_t *__restrict__ input, const int64_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ expert_counters, uint8_t *__restrict__ aq_out,
    float *__restrict__ aq_scale_out, int32_t *__restrict__ inv_perm,
    int32_t *__restrict__ m_indices, int num_tokens, int K, int num_experts,
    int hidden_size, int fp8type, bool has_expert_map) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  extern __shared__ __align__(16) uint8_t shmem_fp8[];

  __shared__ float warp_maxs[NUM_WARPS];
  __shared__ float s_scale;
  __shared__ bool s_any_valid;
  __shared__ int dest_row_shared[SCATTER_MAX_K];
  __shared__ bool is_valid_shared[SCATTER_MAX_K];

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens;
       token_idx += gridDim.x) {
    if (tid == 0) {
      s_any_valid = false;
    }
    __syncthreads();

    int group_id = tid / SCATTER_THREADS_PER_GROUP;
    int group_lane = tid % SCATTER_THREADS_PER_GROUP;

    if (group_lane == 0) {
      bool is_valid = group_id < K;
      int64_t row_idx = token_idx * K + group_id;
      int expert_id = -1;

      if (is_valid) {
        expert_id = static_cast<int>(topk_ids[row_idx]);
        is_valid = expert_id >= 0 && expert_id < num_experts;
      }

      if (is_valid && has_expert_map) {
        int mapped_id = expert_map[expert_id];
        if (mapped_id < 0 || mapped_id >= num_experts) {
          is_valid = false;
        } else {
          expert_id = mapped_id;
        }
      }

      is_valid_shared[group_id] = is_valid;
      if (group_id < K) {
        if (is_valid) {
          s_any_valid = true;
          int dest_row = atomicAdd(&expert_counters[expert_id], 1);
          dest_row_shared[group_id] = dest_row;
          inv_perm[row_idx] = dest_row;
          m_indices[dest_row] = expert_id;
        } else {
          inv_perm[row_idx] = -1;
        }
      }
    }
    __syncthreads();

    if (!s_any_valid) {
      continue;
    }

    const scalar_t *input_row = input + token_idx * hidden_size;
    float local_max_abs = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
      local_max_abs =
          fmaxf(local_max_abs, fabsf(static_cast<float>(input_row[i])));
    }

    local_max_abs = warpReduceMax<float, WARP_SIZE>(local_max_abs);
    if (lane_id == 0) {
      warp_maxs[warp_id] = local_max_abs;
    }
    __syncthreads();

    if (warp_id == 0) {
      float block_max = (tid < NUM_WARPS) ? warp_maxs[tid] : 0.0f;
      block_max = warpReduceMax<float, NUM_WARPS>(block_max);
      if (tid == 0) {
        const float fp8_max = (fp8type == 0) ? 448.0f : 57344.0f;
        const float min_scale = 1.0f / (fp8_max * 512.0f);
        s_scale = fmaxf(block_max / fp8_max, min_scale);
      }
    }
    __syncthreads();

    float inv_scale = 1.0f / s_scale;
    for (int i = tid * 4; i < hidden_size; i += BLOCK_SIZE * 4) {
      float v0 = static_cast<float>(input_row[i]) * inv_scale;
      float v1 = static_cast<float>(input_row[i + 1]) * inv_scale;
      float v2 = static_cast<float>(input_row[i + 2]) * inv_scale;
      float v3 = static_cast<float>(input_row[i + 3]) * inv_scale;
      uint32_t packed =
          (fp8type == 0)
              ? float_to_fp8e4m3_x4_lightop(v0, v1, v2, v3)
              : float_to_fp8e5m2_x4_lightop(v0, v1, v2, v3);
      *reinterpret_cast<uint32_t *>(shmem_fp8 + i) = packed;
    }
    __syncthreads();

    if (is_valid_shared[group_id]) {
      int dest_row = dest_row_shared[group_id];
      uint8_t *dst_ptr =
          aq_out + static_cast<int64_t>(dest_row) * hidden_size;
      if (group_lane == 0) {
        aq_scale_out[dest_row] = s_scale;
      }
      int H_vec = hidden_size / 16;
      int4 *dst_vec_global = reinterpret_cast<int4 *>(dst_ptr);
      const int4 *src_vec_shmem = reinterpret_cast<const int4 *>(shmem_fp8);
      for (int v = group_lane; v < H_vec; v += SCATTER_THREADS_PER_GROUP) {
        dst_vec_global[v] = src_vec_shmem[v];
      }
    }
    __syncthreads();
  }
}

void ep_fused_quant_scatter(torch::Tensor input, // fp16 or bf16
                            torch::Tensor topk_ids,
                            torch::optional<torch::Tensor> expert_map_opt,
                            torch::Tensor expert_num_tokens,
                            torch::Tensor aq_out, torch::Tensor aq_scale_out,
                            torch::Tensor m_indices, torch::Tensor inv_perm,
                            int local_num_experts, int alignment) {

  int num_tokens = input.size(0);
  int hidden_size = input.size(1);
  int K = topk_ids.size(1);

  TORCH_CHECK(hidden_size % 16 == 0,
              "Hidden size must be divisible by 16 for int4 copy");
  TORCH_CHECK(local_num_experts > 0 && local_num_experts <= EP_SCAN_MAX_EXPERTS,
              "ep_fused_quant_scatter supports 1 to 1024 experts");

  auto input_tensor = input.expect_contiguous();
  const int32_t *expert_map_ptr = nullptr;
  bool has_expert_map = false;
  if (expert_map_opt.has_value()) {
    expert_map_ptr = expert_map_opt.value().data_ptr<int32_t>();
    has_expert_map = true;
  }

  int32_t *expert_counters_ptr = expert_num_tokens.data_ptr<int32_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // scan
  {
    ep_scan_kernel<EP_SCAN_MAX_EXPERTS>
        <<<1, EP_SCAN_MAX_EXPERTS, 0, stream>>>(
        expert_counters_ptr, expert_counters_ptr, local_num_experts, alignment);
  }
  // scatter
  {

    constexpr int BLOCK_SIZE = 1024;
    constexpr int GROUP_SIZE = 128;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);
    size_t shmem_size = hidden_size * sizeof(int8_t);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "ep_fused_quant_scatter", [&] {
          quant_scatter_opt_kernel<scalar_t, BLOCK_SIZE, GROUP_SIZE, NUM_WARPS>
              <<<grid, block, shmem_size, stream>>>(
                  input_tensor->data_ptr<scalar_t>(),
                  topk_ids.data_ptr<int64_t>(), //data_ptr<int64_t>()
                  expert_map_ptr,
                  expert_counters_ptr, aq_out.data_ptr<int8_t>(),
                  aq_scale_out.data_ptr<float>(), inv_perm.data_ptr<int32_t>(),
                  m_indices.data_ptr<int32_t>(), num_tokens, K,
                  local_num_experts, hidden_size, has_expert_map);
        });
  }
}

void ep_build_m_indices(torch::Tensor topk_ids, torch::Tensor m_indices,
                        int local_num_experts, int alignment) {
  if (topk_ids.numel() == 0 || m_indices.numel() == 0) {
    return;
  }

  TORCH_CHECK(topk_ids.is_contiguous(),
              "ep_build_m_indices expects contiguous topk_ids");
  TORCH_CHECK(m_indices.is_contiguous(),
              "ep_build_m_indices expects contiguous m_indices");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Long ||
                  topk_ids.scalar_type() == at::ScalarType::Int,
              "ep_build_m_indices expects int32 or int64 topk_ids");
  TORCH_CHECK(m_indices.scalar_type() == at::ScalarType::Int,
              "ep_build_m_indices expects int32 m_indices");
  TORCH_CHECK(local_num_experts > 0 && local_num_experts <= EP_SCAN_MAX_EXPERTS,
              "ep_build_m_indices supports 1 to 1024 experts");
  TORCH_CHECK(alignment > 0, "ep_build_m_indices expects positive alignment");

  constexpr int BLOCK_THREADS = EP_SCAN_MAX_EXPERTS;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "ep_build_m_indices", [&] {
    ep_build_m_indices_kernel<scalar_t, BLOCK_THREADS>
        <<<1, BLOCK_THREADS, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), m_indices.data_ptr<int32_t>(),
            topk_ids.numel(), m_indices.numel(), local_num_experts,
            alignment);
  });
}

void ep_fused_fp8_quant_scatter(torch::Tensor input, torch::Tensor topk_ids,
                                torch::optional<torch::Tensor> expert_map_opt,
                                torch::Tensor expert_num_tokens,
                                torch::Tensor aq_out,
                                torch::Tensor aq_scale_out,
                                torch::Tensor m_indices,
                                torch::Tensor inv_perm,
                                int local_num_experts, int alignment,
                                int fp8type, bool fill_padded_m_indices) {
  int num_tokens = input.size(0);
  int hidden_size = input.size(1);
  int K = topk_ids.size(1);

  if (num_tokens == 0) {
    return;
  }

  TORCH_CHECK(input.dim() == 2 && aq_out.dim() == 2,
              "ep_fused_fp8_quant_scatter expects 2D tensors");
  TORCH_CHECK(input.is_contiguous() && aq_out.is_contiguous(),
              "ep_fused_fp8_quant_scatter expects contiguous tensors");
  TORCH_CHECK(aq_out.element_size() == 1,
              "ep_fused_fp8_quant_scatter expects one-byte fp8 output");
  TORCH_CHECK(aq_scale_out.scalar_type() == at::ScalarType::Float,
              "ep_fused_fp8_quant_scatter expects fp32 output scales");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Long,
              "ep_fused_fp8_quant_scatter expects int64 topk_ids");
  TORCH_CHECK(expert_num_tokens.scalar_type() == at::ScalarType::Int &&
                  inv_perm.scalar_type() == at::ScalarType::Int &&
                  m_indices.scalar_type() == at::ScalarType::Int,
              "ep_fused_fp8_quant_scatter expects int32 index tensors");
  TORCH_CHECK(hidden_size % 16 == 0,
              "ep_fused_fp8_quant_scatter hidden size must be divisible by 16");
  TORCH_CHECK(K <= 8, "ep_fused_fp8_quant_scatter supports topk <= 8");
  TORCH_CHECK(fp8type == 0 || fp8type == 1,
              "ep_fused_fp8_quant_scatter fp8type must be 0(e4m3) or 1(e5m2)");
  TORCH_CHECK(local_num_experts > 0 && local_num_experts <= EP_SCAN_MAX_EXPERTS,
              "ep_fused_fp8_quant_scatter supports 1 to 1024 experts");

  const int32_t *expert_map_ptr = nullptr;
  bool has_expert_map = false;
  if (expert_map_opt.has_value()) {
    expert_map_ptr = expert_map_opt.value().data_ptr<int32_t>();
    has_expert_map = true;
  }

  int32_t *expert_counters_ptr = expert_num_tokens.data_ptr<int32_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (fill_padded_m_indices) {
    ep_scan_fill_m_indices_kernel<EP_SCAN_MAX_EXPERTS, true>
        <<<1, EP_SCAN_MAX_EXPERTS, 0, stream>>>(
        expert_counters_ptr, expert_counters_ptr, m_indices.data_ptr<int32_t>(),
        local_num_experts, alignment);
  } else {
    ep_scan_fill_m_indices_kernel<EP_SCAN_MAX_EXPERTS, false>
        <<<1, EP_SCAN_MAX_EXPERTS, 0, stream>>>(
        expert_counters_ptr, expert_counters_ptr, m_indices.data_ptr<int32_t>(),
        local_num_experts, alignment);
  }

  constexpr int BLOCK_SIZE = 1024;
  constexpr int SCATTER_THREADS_PER_GROUP = 128;
  constexpr int SCATTER_MAX_K = 8;
  dim3 grid(num_tokens);
  dim3 block(BLOCK_SIZE);
  size_t shmem_size = hidden_size * sizeof(uint8_t);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "ep_fused_fp8_quant_scatter", [&] {
        ep_fp8_quant_scatter_kernel<scalar_t, BLOCK_SIZE,
                                    SCATTER_THREADS_PER_GROUP, SCATTER_MAX_K>
            <<<grid, block, shmem_size, stream>>>(
                input.data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                expert_map_ptr, expert_counters_ptr,
                reinterpret_cast<uint8_t *>(aq_out.data_ptr()),
                aq_scale_out.data_ptr<float>(), inv_perm.data_ptr<int32_t>(),
                m_indices.data_ptr<int32_t>(), num_tokens, K,
                local_num_experts, hidden_size, fp8type, has_expert_map);
      });
}

template <typename scalar_t, int K_MAX>
__global__ void ep_gather_kernel(
    const scalar_t* __restrict__ input_tensor,    
    const int64_t* __restrict__ topk_ids,               
    const float* __restrict__ topk_weights,     
    const int* __restrict__ src_indices,      //inverm        
    const int* __restrict__ expert_map,            
    scalar_t* __restrict__ output_tensor,        
    const int hidden_size,
    const int topk_num,
    const bool has_expert_map,
    const int num_tokens                         
) {
  
    constexpr int VEC_SIZE = 8;
    using VecType = AlignedVector<scalar_t, VEC_SIZE>;

    const int col_offset_base = blockIdx.x * blockDim.x * VEC_SIZE;
    const int tid = threadIdx.x;
    const int feat_idx = col_offset_base + tid * VEC_SIZE;
    if (feat_idx >= hidden_size) return;

    for (int token_idx = blockIdx.y; token_idx < num_tokens; token_idx += gridDim.y) {
        float acc[VEC_SIZE] = {0.0f};
        #pragma unroll
        for (int k = 0; k < K_MAX; ++k) {
             if (k >= topk_num) {
                 continue;
             }
             size_t flat_idx = static_cast<size_t>(token_idx) * topk_num + k;
             int expert_id = topk_ids[flat_idx];
             if (has_expert_map && expert_id >= 0) {
                 expert_id = expert_map[expert_id];
             }
             int src_idx = src_indices[flat_idx];
             if (expert_id >= 0 && src_idx >= 0) {
                 float w = topk_weights[flat_idx];
                 
                 size_t src_offset = static_cast<size_t>(src_idx) * hidden_size + feat_idx;
                 const scalar_t* src_ptr = input_tensor + src_offset;
                 
                 VecType loaded_vec = *reinterpret_cast<const VecType*>(src_ptr);
                 
                 #pragma unroll
                 for (int i = 0; i < VEC_SIZE; ++i) {
                     acc[i] += static_cast<float>(loaded_vec.val[i]) * w;
                 }
             }
        }

        VecType out_vec;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec.val[i] = static_cast<scalar_t>(acc[i]);
        }

        size_t out_offset = static_cast<size_t>(token_idx) * hidden_size + feat_idx;
        VecType* out_vec_ptr = reinterpret_cast<VecType*>(output_tensor + out_offset);
        *out_vec_ptr = out_vec;
    }
}
void ep_scatter(torch::Tensor aq, torch::Tensor aq_scale,
                torch::Tensor topk_ids,
                torch::optional<torch::Tensor> expert_map_opt,
                torch::Tensor expert_num_tokens, torch::Tensor aq_out,
                torch::Tensor aq_scale_out, torch::Tensor m_indices,
                torch::Tensor inv_perm, int local_num_experts, int alignment) {

  int num_tokens = aq.size(0);
  int H = aq.size(1);
  int HS = (aq_scale.dim() > 1) ? aq_scale.size(1) : 1;
  int K = topk_ids.size(1);

  if (num_tokens == 0) {
    return;
  }

  TORCH_CHECK(aq.dim() == 2, "ep_scatter expects aq to be 2D");
  TORCH_CHECK(aq_out.dim() == 2, "ep_scatter expects aq_out to be 2D");
  TORCH_CHECK(aq.element_size() == 1 && aq_out.element_size() == 1,
              "ep_scatter expects one-byte activation tensors");
  TORCH_CHECK(aq.is_contiguous() && aq_out.is_contiguous(),
              "ep_scatter expects contiguous activation tensors");
  TORCH_CHECK(aq_scale.scalar_type() == at::ScalarType::Float &&
                  aq_scale_out.scalar_type() == at::ScalarType::Float,
              "ep_scatter expects fp32 scale tensors");
  TORCH_CHECK(aq_scale.is_contiguous() && aq_scale_out.is_contiguous(),
              "ep_scatter expects contiguous scale tensors");
  TORCH_CHECK(HS == 1,
              "ep_scatter expects one fp32 scale per token");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Long,
              "ep_scatter expects int64 topk_ids");
  TORCH_CHECK(expert_num_tokens.scalar_type() == at::ScalarType::Int,
              "ep_scatter expects int32 expert_num_tokens");
  TORCH_CHECK(inv_perm.scalar_type() == at::ScalarType::Int &&
                  m_indices.scalar_type() == at::ScalarType::Int,
              "ep_scatter expects int32 index tensors");
  TORCH_CHECK(H % 16 == 0,
              "ep_scatter hidden size must be divisible by 16");
  TORCH_CHECK(K <= 8, "ep_scatter supports topk <= 8");
  TORCH_CHECK(local_num_experts > 0 && local_num_experts <= EP_SCAN_MAX_EXPERTS,
              "ep_scatter supports 1 to 1024 experts");

  const int32_t *expert_map_ptr = nullptr;
  bool has_expert_map = false;

  if (expert_map_opt.has_value()) {
    expert_map_ptr = expert_map_opt.value().data_ptr<int32_t>();
    has_expert_map = true;
  }

  int32_t *expert_counters_ptr = expert_num_tokens.data_ptr<int32_t>();
  const int8_t *aq_ptr = reinterpret_cast<const int8_t *>(aq.data_ptr());
  int8_t *aq_out_ptr = reinterpret_cast<int8_t *>(aq_out.data_ptr());
  float *aq_scale_ptr = aq_scale.data_ptr<float>();
  float *aq_scale_out_ptr = aq_scale_out.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Scan
  {
    dim3 grid(1);
    dim3 block(EP_SCAN_MAX_EXPERTS);
    constexpr int CUB_BLOCK_THREADS = EP_SCAN_MAX_EXPERTS;
    ep_scan_fill_m_indices_kernel<CUB_BLOCK_THREADS, false>
        <<<grid, block, 0, stream>>>(
            expert_counters_ptr, expert_counters_ptr,
            m_indices.data_ptr<int32_t>(), local_num_experts, alignment);
  }

  // Scatter
  {
    if(num_tokens <= 4096){
            constexpr int SCATTER_MAX_K = 8;
            using scale_t = float;
            int threads = 1024;
            int blocks = (num_tokens + 1) / 2;
            dim3 grid(blocks);
            dim3 block(threads);
            size_t shmem_size = 2 * (H / 16) * sizeof(float4);
            ep_scatter_kernel_1024_opt<scale_t, SCATTER_MAX_K>
                <<<grid, block, shmem_size, stream>>>(
                aq_ptr,
                aq_scale_ptr,
                topk_ids.data_ptr<int64_t>(), //data_ptr<int64_t>()
                expert_map_ptr, 
                expert_counters_ptr,
                aq_out_ptr,
                aq_scale_out_ptr,
                inv_perm.data_ptr<int32_t>(), 
                m_indices.data_ptr<int32_t>(),
                num_tokens, K, local_num_experts, H, has_expert_map);
    }else{
        constexpr int SCATTER_THREADS_PER_GROUP = 128;
        constexpr int SCATTER_MAX_K = 8;
        using data_t = int8_t;
        using scale_t = float;
        int threads = SCATTER_THREADS_PER_GROUP * SCATTER_MAX_K;
        int blocks = num_tokens;

        dim3 grid(blocks);
        dim3 block(threads);

        size_t shmem_size = H * sizeof(data_t);
        ep_scatter_kernel<data_t, scale_t, SCATTER_THREADS_PER_GROUP,
                            SCATTER_MAX_K><<<grid, block, shmem_size, stream>>>(
                aq_ptr, aq_scale_ptr,
                topk_ids.data_ptr<int64_t>(), //data_ptr<int64_t>()
                expert_map_ptr, expert_counters_ptr,
                aq_out_ptr, aq_scale_out_ptr,
                inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                num_tokens, K, local_num_experts, H, HS, has_expert_map);
    }
  }
}

//unpermute
void ep_gather(
    torch::Tensor& a,           
    torch::Tensor& topk_ids,     
    torch::Tensor& topk_weights, 
    torch::Tensor& inv_perm,     
    torch::optional<torch::Tensor> expert_map,
    torch::Tensor& output        
) {
 
    const int num_tokens = output.size(0);
    const int hidden_size = output.size(1);
    const int topk_num = topk_ids.size(1);

    const int* expert_map_ptr = nullptr;
    bool has_expert_map = false;
    if (expert_map.has_value() && expert_map->defined()) {
        expert_map_ptr = expert_map->data_ptr<int>();
        has_expert_map = true;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ep_gather",
        [&] {
            constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
            TORCH_CHECK(hidden_size % VEC_SIZE == 0, "Hidden size must be divisible by vector size");

            int threads_per_block = 256;
            int elements_per_thread = VEC_SIZE;
            int elements_per_block = threads_per_block * elements_per_thread;
            int grid_x = (hidden_size + elements_per_block - 1) / elements_per_block;
            int grid_y = num_tokens;

            bool is_deepseek_config = (hidden_size == 7168) && (VEC_SIZE == 8);
            // 7168 * 8 128线程*8 = 7168 loop 8次
            if (is_deepseek_config) {
                threads_per_block = 128;
                grid_x = 7;
            }

            if(num_tokens >= 256 && num_tokens <= 512 && hidden_size == 4096) {
                threads_per_block = 128;
                grid_x = 4;
            };
            // if (grid_y > 65535) grid_y = 65535;
            dim3 block(threads_per_block);
            dim3 grid(grid_x, grid_y);

            auto a_tensor = a.expect_contiguous();
            if (topk_num <= 2) {
                ep_gather_kernel<scalar_t, 2><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    a_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                    inv_perm.data_ptr<int>(), expert_map_ptr, output.data_ptr<scalar_t>(),
                    hidden_size, topk_num, has_expert_map, num_tokens
                );
            } else if (topk_num <= 4) {
                ep_gather_kernel<scalar_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    a_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                    inv_perm.data_ptr<int>(), expert_map_ptr, output.data_ptr<scalar_t>(),
                    hidden_size, topk_num, has_expert_map, num_tokens
                );
            } else {
                ep_gather_kernel<scalar_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    a_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                    inv_perm.data_ptr<int>(), expert_map_ptr, output.data_ptr<scalar_t>(),
                    hidden_size, topk_num, has_expert_map, num_tokens
                );
            }
        }
    );
}

template <typename scalar_t, int BLOCK_SIZE, int MAX_K>
__global__ void quant_scatter_smooth_opt_kernel(
    const scalar_t *__restrict__ input,
    const int64_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ expert_offsets,
    int8_t *__restrict__ aq_out,
    float *__restrict__ aq_scale_out,
    int32_t *__restrict__ inv_perm,
    int32_t *__restrict__ m_indices,
    const float *__restrict__ smooth_scale, 
    int num_tokens, int K, int num_experts,
    int hidden_size, bool has_expert_map) {

    int64_t token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ int s_expert_ids[MAX_K];
    __shared__ int64_t s_dest_rows[MAX_K];
    __shared__ bool s_valid[MAX_K];
    __shared__ float s_warp_maxs[32]; 
    __shared__ float s_current_inv_scale; 
    __shared__ bool s_any_valid;
    
    if (tid == 0) s_any_valid = false;
    __syncthreads();
    
    if (tid < K) {
        int expert_id = (int)topk_ids[token_idx * K + tid];
        bool is_valid = expert_id >= 0 && expert_id < num_experts;
        if (is_valid && has_expert_map) {
            int mapped_id = expert_map[expert_id];
            if (mapped_id == -1) is_valid = false;
            else expert_id = mapped_id;
        }
        s_valid[tid] = is_valid;
        s_expert_ids[tid] = expert_id;
        if (is_valid) {
            s_any_valid = true;
            int64_t dest_row = atomicAdd(&expert_offsets[expert_id], 1);
            s_dest_rows[tid] = dest_row;
            int64_t row_idx = token_idx * K + tid;
            inv_perm[row_idx] = (int32_t)dest_row;
            if (m_indices) m_indices[dest_row] = expert_id;
        } else {
            inv_perm[token_idx * K + tid] = -1;
        }
    }
    __syncthreads(); 

    if(!s_any_valid) return;

    constexpr int ELEMENTS_PER_VEC = 16;
    using VecType = AlignedVector<scalar_t, ELEMENTS_PER_VEC>;
    using OutType = AlignedVector<int8_t, ELEMENTS_PER_VEC>; 
    using SmoothVecType = AlignedVector<float, ELEMENTS_PER_VEC>;

    int64_t input_base_offset = token_idx * hidden_size;

    #pragma unroll 1 
    for (int k = 0; k < MAX_K; ++k) {
        if (k >= K) break; 
        bool is_active = s_valid[k];
        if(!is_active) continue;
        
        int expert = s_expert_ids[k];
        float local_max_abs = 0.0f; 
        for (int col_idx = tid * ELEMENTS_PER_VEC; col_idx < hidden_size; col_idx += BLOCK_SIZE * ELEMENTS_PER_VEC) {
            VecType curr_input_vec = *reinterpret_cast<const VecType*>(input + input_base_offset + col_idx);
            const float* smooth_ptr = smooth_scale + (int64_t)expert * hidden_size + col_idx;
            SmoothVecType curr_smooth = *reinterpret_cast<const SmoothVecType*>(smooth_ptr);

            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_VEC; ++e) {
                float val = static_cast<float>(curr_input_vec[e]);
                float smoothed = val * curr_smooth[e];
                local_max_abs = fmaxf(local_max_abs, fabsf(smoothed));
            }
        }

        local_max_abs = warpReduceMax<float, WARP_SIZE>(local_max_abs);
        if (lane_id == 0) s_warp_maxs[warp_id] = local_max_abs;
        __syncthreads(); 

        if (warp_id == 0) {
            float block_val = (tid < num_warps) ? s_warp_maxs[tid] : 0.0f;
            block_val = warpReduceMax<float, WARP_SIZE>(block_val); 
            if (tid == 0) {
                float max_val = block_val;
                max_val = fmaxf(max_val, 1e-6f); // 防止除0
                float inv_scale = 127.0f / max_val;
                float final_scale = max_val / 127.0f; 
                aq_scale_out[s_dest_rows[k]] = final_scale; 
                s_current_inv_scale = inv_scale;
            }
        }
        __syncthreads(); 
     
        float inv_scale = s_current_inv_scale; 
        
        for (int col_idx = tid * ELEMENTS_PER_VEC; col_idx < hidden_size; col_idx += BLOCK_SIZE * ELEMENTS_PER_VEC) {
            VecType curr_input_vec = *reinterpret_cast<const VecType*>(input + input_base_offset + col_idx);
            const float* smooth_ptr = smooth_scale + (int64_t)expert * hidden_size + col_idx;
            SmoothVecType curr_smooth = *reinterpret_cast<const SmoothVecType*>(smooth_ptr);
            
            int8_t packed_vals[ELEMENTS_PER_VEC];
            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_VEC; ++e) {
                float val = static_cast<float>(curr_input_vec[e]);
                float smoothed = val * curr_smooth[e]; 
                packed_vals[e] = float_to_int8_rn_fused(smoothed * inv_scale);
            }
            
            int64_t dst_offset = s_dest_rows[k] * hidden_size + col_idx;
            *reinterpret_cast<OutType*>(aq_out + dst_offset) = *reinterpret_cast<OutType*>(packed_vals);
        }
    }
}


template <typename scalar_t, int BLOCK_SIZE, int VEC_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) quant_scatter_smooth_low_reg_kernel(
    const scalar_t *__restrict__ input,
    const int64_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ expert_offsets,
    int8_t *__restrict__ aq_out,
    float *__restrict__ aq_scale_out,
    int32_t *__restrict__ inv_perm,
    int32_t *__restrict__ m_indices,
    const float *__restrict__ smooth_scale, 
    int num_tokens, int K, int hidden_size, bool has_expert_map) {


    int64_t token_idx = blockIdx.x*2;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;


    __shared__ float s_warp_maxs[32]; 
    __shared__ float s_block_scale; 
    __shared__ int64_t s_dest_row;


    float r_input[VEC_SIZE];
    int idx = tid * VEC_SIZE;
    for(int ii = 0; ii < 2; ii++){
        token_idx += ii;
        if (token_idx >= num_tokens) break;
        int64_t input_base_offset = token_idx * hidden_size;
        using VecType = AlignedVector<scalar_t, VEC_SIZE>;
        const VecType* in_ptr = reinterpret_cast<const VecType*>(input + input_base_offset + idx);
        VecType in_val;
    
        bool is_load = true;

        for(int k=0; k<K; ++k) {
            int expert_id = (int)topk_ids[token_idx * K + k];
            if (expert_id < 0) {
                continue;
            }
            
            if (has_expert_map) {
                int mapped = expert_map[expert_id];
                if (mapped == -1) {
                    continue; 
                }
                expert_id = mapped;
            }

            if(is_load){
                in_val = *in_ptr;
                #pragma unroll
                for(int i=0; i<VEC_SIZE; ++i) {
                    r_input[i] = static_cast<float>(in_val.val[i]);
                }
                is_load = false;
            }

            if (tid == 0) {
                int32_t dr = atomicAdd(&expert_offsets[expert_id], 1);
                s_dest_row = dr;
                inv_perm[token_idx * K + k] = dr;
                if (m_indices) m_indices[dr] = expert_id;
            }
            __syncthreads(); 
            
            int64_t dest_row = s_dest_row;
            
            float local_max = 0.0f;
            float r_smooth[VEC_SIZE]; 

            const float* smooth_ptr = smooth_scale + (int64_t)expert_id * hidden_size + idx;
            using SmoothVec = AlignedVector<float, VEC_SIZE>;
            const SmoothVec* s_vec_ptr = reinterpret_cast<const SmoothVec*>(smooth_ptr);
            *reinterpret_cast<SmoothVec *>(r_smooth) = *s_vec_ptr; 

            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                float val = r_input[i] * r_smooth[i];
                local_max = fmaxf(local_max, fabsf(val));
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down( local_max, offset));
            }
            
            if (lane_id == 0) {
                s_warp_maxs[warp_id] = local_max;
            }
            __syncthreads(); 

            if (warp_id == 0) {
                float block_max = (tid < (BLOCK_SIZE / WARP_SIZE)) ? s_warp_maxs[tid] : 0.0f;
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    block_max = fmaxf(block_max, __shfl_down( block_max, offset));
                }
                
                if (tid == 0) {
                    block_max = fmaxf(block_max, 1e-6f);
                    float scale = block_max / 127.0f;
                    aq_scale_out[dest_row] = scale;
                    s_block_scale = 127.0f / block_max; 
                }
            }
            __syncthreads(); 
            
            float inv_scale = s_block_scale;

            int8_t out_buf[VEC_SIZE];
            #pragma unroll
            for(int i=0; i<VEC_SIZE; ++i) {
                float val = r_input[i] * r_smooth[i];
                out_buf[i] = float_to_int8_rn_fused(val * inv_scale);
            }
            int64_t dst_offset = dest_row * hidden_size + idx;
            using OutVec = AlignedVector<int8_t, VEC_SIZE>;
            *reinterpret_cast<OutVec*>(aq_out + dst_offset) = *reinterpret_cast<OutVec*>(out_buf);
        }
    }
}

void ep_fused_smooth_quant_scatter(
    torch::Tensor input, 
    torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt,
    torch::Tensor expert_offsets,
    torch::Tensor smooth_scale,
    torch::Tensor aq_out, 
    torch::Tensor aq_scale_out,
    torch::Tensor m_indices, 
    torch::Tensor inv_perm,
    int local_num_experts, int alignment) {

  int num_tokens = input.size(0);
  int hidden_size = input.size(1);
  int K = topk_ids.size(1);

  TORCH_CHECK(smooth_scale.scalar_type() == at::ScalarType::Float, "Smooth scale must be float32");
  TORCH_CHECK(K == 4 || K == 8, "TopK must be 4 or 8");

  auto input_tensor = input.expect_contiguous();
  const int32_t *expert_map_ptr = nullptr;
  bool has_expert_map = false;
  if (expert_map_opt.has_value()) {
    expert_map_ptr = expert_map_opt.value().data_ptr<int32_t>();
    has_expert_map = true;
  }

  int32_t *expert_offsets_ptr = expert_offsets.data_ptr<int32_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  constexpr int VEC_SIZE = 16; 
  size_t shmem_size = 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "ep_fused_smooth_quant_scatter", [&] {

        if (hidden_size == 1024) {
            constexpr int CURR_BLOCK_SIZE = 64;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else if (hidden_size == 2048) {
            constexpr int CURR_BLOCK_SIZE = 128;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else if (hidden_size == 4096) {
            constexpr int CURR_BLOCK_SIZE = 256;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else if (hidden_size == 7168) {
            constexpr int CURR_BLOCK_SIZE = 448;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else if (hidden_size == 8192) {
            constexpr int CURR_BLOCK_SIZE = 512;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else if (hidden_size == 16384) {
            constexpr int CURR_BLOCK_SIZE = 1024;
            dim3 grid((num_tokens + 1) / 2);
            dim3 block(CURR_BLOCK_SIZE);
            quant_scatter_smooth_low_reg_kernel<scalar_t, CURR_BLOCK_SIZE, VEC_SIZE>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, hidden_size, has_expert_map);
        } else {
            constexpr int FALLBACK_BLOCK_SIZE = 256;
            dim3 grid(num_tokens);
            dim3 block(FALLBACK_BLOCK_SIZE);
            if (K <= 8) {
                 quant_scatter_smooth_opt_kernel<scalar_t, FALLBACK_BLOCK_SIZE, 8>
                <<<grid, block, shmem_size, stream>>>(
                    input_tensor->data_ptr<scalar_t>(), topk_ids.data_ptr<int64_t>(),
                    expert_map_ptr, expert_offsets_ptr, 
                    aq_out.data_ptr<int8_t>(), aq_scale_out.data_ptr<float>(),
                    inv_perm.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),
                    smooth_scale.data_ptr<float>(), num_tokens, K, local_num_experts, hidden_size, has_expert_map);
            } else {
                
            }
        }
      });
}

} // namespace native
} // namespace at
