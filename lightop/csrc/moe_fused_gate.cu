#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/all.h>
#include <cfloat>
#include <type_traits>
#include <hip/hip_bf16.h>
#if defined(__gfx938__)
  constexpr float LOG2_E_VAL = 1.44269504089f; 
  __device__ inline float exp_func(float x) {
      return __builtin_amdgcn_exp2f(x * LOG2_E_VAL);
  }
#else
  __device__ inline float exp_func(float x) {
      return __expf(x);
  }
#endif

namespace at {
namespace native {
typedef __hip_bfloat16 __nv_bfloat16;

#ifndef USE_ROCM
 #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
   __shfl_xor_sync(uint32_t(-1), var, lane_mask)
 #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
   __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
 #define VLLM_SHFL_BCAST_SYNC_WIDTH(var, src_lane, width) \
   __shfl_sync(uint32_t(-1), var, src_lane, width)
#else
 #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
 #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
   __shfl_xor(var, lane_mask, width)
 #define VLLM_SHFL_BCAST_SYNC_WIDTH(var, src_lane, width) \
   __shfl(var, src_lane, width)
#endif

#define WARP_SIZE 64

/// Aligned array type
template <
 typename T,
   /// Number of elements in the array
   int N,
   /// Alignment requirement in bytes
   int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
   T data[N];

public:
   __device__ T& operator[](int index) {
     return data[index];
   }

   __device__ const T& operator[](int index) const {
     return data[index];
   }
};

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARPS_PER_CTA = 1;
static constexpr int MAX_VPT = 32; 

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;


/* deepseek 使用,写死了，四个线程一个gourp, vpt_per_thread=8*/
template <typename T, typename T_bias, typename Params>
__device__ void moe_fused_gate_impl(
  void* input,
  void* bias,
  float* output_ptr,
  int32_t* indices_ptr,
  int num_rows,
  int topk_group,
  int topk,
  int n_share_experts_fusion,
  double routed_scaling_factor,
  Params params) 
{
 
  int tidx = threadIdx.x;
  int group_row = tidx / params.THREADS_PER_ROW; 
  int thread_row = blockIdx.x * params.ROWS_PER_CTA + group_row;

   int topk_routed = topk - n_share_experts_fusion;

  if (thread_row >= num_rows) {
    return;
  }
  constexpr float Negative_inf = -FLT_MAX;
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T_bias*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;
  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  T_bias* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;

  Array<float, 8> row_chunk; 
  Array<float, 8> scores_chunk;
  using vectype = Array<T, 8>;
  using vectype_bias = Array<T_bias, 8>;
  vectype in = *((vectype*)(thread_read_ptr));
  vectype_bias b = *((vectype_bias*)(bias_thread_read_ptr));
  for(int kk =0; kk < 8; kk++){
   row_chunk[kk]=1.0f/(1.0f + exp_func(-static_cast<float>(in[kk])));
   scores_chunk[kk]=static_cast<float>(b[kk])+row_chunk[kk];
  }


  /////////////////////// 线程组计算top2 ///////////////////////////// 
  // int my_local_start_id = first_elt_read_by_thread;
 float max_val_group = scores_chunk[0];
 float max_val_second_group = Negative_inf;
  int max_val_group_idx = first_elt_read_by_thread;  //缓存top1
  int max_val_second_group_idx = first_elt_read_by_thread;
 for (int ii = 1; ii < params.VPT; ++ii) {
   float val = scores_chunk[ii];
   int curr_idx = first_elt_read_by_thread + ii;
   if (val > max_val_group) {
    max_val_second_group = max_val_group;
    max_val_second_group_idx = max_val_group_idx;
    max_val_group = val;
    max_val_group_idx = curr_idx;
   } else if (val > max_val_second_group) {
     max_val_second_group = val;
     max_val_second_group_idx = curr_idx;
   }
 }
  float tmp_top1 = max_val_group;
  float tmp_top2 = max_val_second_group;
  for (int mask = 1; mask < 4; mask *= 2) {
    float other_top1 = VLLM_SHFL_XOR_SYNC_WIDTH(tmp_top1, mask, params.THREADS_PER_ROW);
    float other_top2 = VLLM_SHFL_XOR_SYNC_WIDTH(tmp_top2, mask, params.THREADS_PER_ROW);
    float old_top1 = tmp_top1;
    //top1 = max(My1,Other1)
    tmp_top1 = fmaxf(old_top1, other_top1);
    // top2 = max(min(My1, Other1), max(My2, Other2))
    tmp_top2 = fmaxf(fminf(old_top1, other_top1), fmaxf(tmp_top2, other_top2));
  }
  float max_sum_group = tmp_top1 + tmp_top2; // Group Score 
  int my_group_logical_id = thread_group_idx / 4;
  int my_lane_in_group = thread_group_idx % 4;
 
  int my_group_start_expert_id = my_group_logical_id * (params.VPT * 4);

  float temp_group_score = max_sum_group;
  // 只让每个 Group 的第 0 号线程参与竞争，其设为负无穷
  if (my_lane_in_group != 0) {
    temp_group_score = Negative_inf;
  }
  bool is_my_group_in_top_k = false;
  /////////////////////// 选组 ///////////////////////////// 
  for (int k_idx = 0; k_idx < topk_group; ++k_idx) {
    float max_val_reduce = temp_group_score;
    int winning_group_id = my_group_start_expert_id;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_reduce, mask, params.THREADS_PER_ROW);
      int other_group = VLLM_SHFL_XOR_SYNC_WIDTH(winning_group_id, mask, params.THREADS_PER_ROW);

      // if (other_max > max_val_reduce || (other_max == max_val_reduce && other_group < winning_group_id)) {
      if (other_max > max_val_reduce ) {
        max_val_reduce = other_max;
        winning_group_id = other_group;
      }
    }
    if (my_group_start_expert_id == winning_group_id) {
      is_my_group_in_top_k = true;
      temp_group_score = Negative_inf; // 赢过的组退出下一轮竞争
    }
  }
  //提前屏蔽被淘汰的组
  if (!is_my_group_in_top_k) {
    max_val_group = Negative_inf;
    max_val_second_group = Negative_inf;
    for (int ii = 0; ii < params.VPT; ++ii) {
      scores_chunk[ii] = Negative_inf;
    }
  }
  /////////////////////// topk ///////////////////////////// 
  // 第 k 个结果由 thread_group_idx == k 的线程写入
  float my_stored_prob = 0.0f;
  int my_stored_idx = 0;
  float output_sum = 0.0f;
  if(topk_routed > 0){
    constexpr int k_idx=0;
    float local_max = max_val_group;
    int local_idx = max_val_group_idx;
    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    int winner_thread_id = global_best_expert_id / params.VPT;
    int winner_offset = global_best_expert_id % params.VPT;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
     max_val_group = Negative_inf; 
    }
  }
  if(topk_routed > 1){
    constexpr int k_idx=1;
    float local_max;
    int local_idx; 
    if (max_val_group == Negative_inf) { 
        local_max = max_val_second_group;
        local_idx = max_val_second_group_idx;
      } else {
        local_max = max_val_group;
        local_idx = max_val_group_idx;
    }
    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    int winner_thread_id = global_best_expert_id / params.VPT;
    int winner_offset = global_best_expert_id % params.VPT;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
    }
  }
  for (int k_idx = 2; k_idx < topk_routed; ++k_idx) {
    float local_max = Negative_inf;
    int local_idx = first_elt_read_by_thread; 
    for (int ii = 0; ii < params.VPT; ++ii) {
      float val = scores_chunk[ii];
      if (val > local_max) {
        local_max = val;
        local_idx = first_elt_read_by_thread + ii;
      }
    }

    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    int winner_thread_id = global_best_expert_id / params.VPT;
    int winner_offset = global_best_expert_id % params.VPT;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
   if(k_idx + 1 < topk){
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
    }
   }
  }
  //共享专家逻辑
   if (n_share_experts_fusion > 0 && thread_group_idx >= topk_routed && thread_group_idx < topk) {
      // 共享专家ID从 NUM_EXPERTS 开始递增
      int share_offset = thread_group_idx - topk_routed;
      my_stored_idx = params.NUM_EXPERTS + share_offset;
      // 共享专家权重 = 常规路由之和 / Scaling routed_scaling_factor
      my_stored_prob = output_sum / static_cast<float>(routed_scaling_factor);
  }
  // 路由专家: stored_prob / output_sum -> 归一化权重
  // 共享专家: (output_sum / scale) / output_sum -> 1 / scale (固定权重)
  if (thread_group_idx < topk) {
    int const idx = topk * thread_row + thread_group_idx;
    output_ptr[idx] = my_stored_prob /output_sum;
    indices_ptr[idx] = my_stored_idx;
  }
}
template <typename T, typename T_bias, typename Params, int VPT_Capacity>
__device__ void moe_fused_gate_impl_expert_384(
  void* input,
  void* bias,
  float* output_ptr,
  int32_t* indices_ptr,
  int num_rows,
  int topk_group,
  int topk,
  int n_share_experts_fusion,
  double routed_scaling_factor,
  Params params) 
{
  int tidx = threadIdx.x;
  int group_row = tidx / params.THREADS_PER_ROW; 
  int thread_row = blockIdx.x * params.ROWS_PER_CTA + group_row;

  if (thread_row >= num_rows) {
    return;
  }
  constexpr float Negative_inf = -FLT_MAX;
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T_bias*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;
  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  
  int first_elt_read_by_thread = thread_group_idx * VPT_Capacity;
  
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  T_bias* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;

  float row_chunk[VPT_Capacity]; 
  float scores_chunk[VPT_Capacity];
  
  T in[VPT_Capacity];
  T_bias b[VPT_Capacity];

  #pragma unroll
  for(int i=0; i < VPT_Capacity; ++i) {
      in[i] = thread_read_ptr[i];
      b[i] = bias_thread_read_ptr[i];
  }

  #pragma unroll
  for(int kk = 0; kk < VPT_Capacity; kk++){
      row_chunk[kk] = 1.0f/(1.0f + exp_func(-static_cast<float>(in[kk])));
      scores_chunk[kk] = static_cast<float>(b[kk]) + row_chunk[kk];
  }

  /////////////////////// 线程组计算top2 ///////////////////////////// 
 float max_val_group = scores_chunk[0];
 float max_val_second_group = Negative_inf;
  int max_val_group_idx = first_elt_read_by_thread;  //缓存top1
  int max_val_second_group_idx = first_elt_read_by_thread;
 
 #pragma unroll
 for (int ii = 1; ii < VPT_Capacity; ++ii) {
   float val = scores_chunk[ii];
   int curr_idx = first_elt_read_by_thread + ii;
   if (val > max_val_group) {
    max_val_second_group = max_val_group;
    max_val_second_group_idx = max_val_group_idx;
    max_val_group = val;
    max_val_group_idx = curr_idx;
   } else if (val > max_val_second_group) {
     max_val_second_group = val;
     max_val_second_group_idx = curr_idx;
   }
 }

  float tmp_top1 = max_val_group;
  float tmp_top2 = max_val_second_group;
  for (int mask = 1; mask < 4; mask *= 2) {
    float other_top1 = VLLM_SHFL_XOR_SYNC_WIDTH(tmp_top1, mask, params.THREADS_PER_ROW);
    float other_top2 = VLLM_SHFL_XOR_SYNC_WIDTH(tmp_top2, mask, params.THREADS_PER_ROW);
    float old_top1 = tmp_top1;
    //top1 = max(My1,Other1)
    tmp_top1 = fmaxf(old_top1, other_top1);
    // top2 = max(min(My1, Other1), max(My2, Other2))
    tmp_top2 = fmaxf(fminf(old_top1, other_top1), fmaxf(tmp_top2, other_top2));
  }
  float max_sum_group = tmp_top1 + tmp_top2; // Group Score 
  int my_group_logical_id = thread_group_idx / 4;
  int my_lane_in_group = thread_group_idx % 4;
  
  int my_group_start_expert_id = my_group_logical_id * (VPT_Capacity * 4);

  float temp_group_score = max_sum_group;
  // 只让每个 Group 的第 0 号线程参与竞争，其设为负无穷
  if (my_lane_in_group != 0) {
    temp_group_score = Negative_inf;
  }
  bool is_my_group_in_top_k = false;
  /////////////////////// 选组 ///////////////////////////// 
  for (int k_idx = 0; k_idx < topk_group; ++k_idx) {
    float max_val_reduce = temp_group_score;
    int winning_group_id = my_group_start_expert_id;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_reduce, mask, params.THREADS_PER_ROW);
      int other_group = VLLM_SHFL_XOR_SYNC_WIDTH(winning_group_id, mask, params.THREADS_PER_ROW);

      if (other_max > max_val_reduce ) {
        max_val_reduce = other_max;
        winning_group_id = other_group;
      }
    }
    if (my_group_start_expert_id == winning_group_id) {
      is_my_group_in_top_k = true;
      temp_group_score = Negative_inf; // 赢过的组退出下一轮竞争
    }
  }
  //提前屏蔽被淘汰的组
  if (!is_my_group_in_top_k) {
    max_val_group = Negative_inf;
    max_val_second_group = Negative_inf;
    // Loop bound uses VPT_Capacity
    #pragma unroll
    for (int ii = 0; ii < VPT_Capacity; ++ii) {
      scores_chunk[ii] = Negative_inf;
    }
  }
  /////////////////////// topk ///////////////////////////// 
  // 第 k 个结果由 thread_group_idx == k 的线程写入
  float my_stored_prob = 0.0f;
  int my_stored_idx = 0;
  float output_sum = 0.0f;
  {
    constexpr int k_idx=0;
    float local_max = max_val_group;
    int local_idx = max_val_group_idx;
    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    // Replaced params.VPT with VPT_Capacity
    int winner_thread_id = global_best_expert_id / VPT_Capacity;
    int winner_offset = global_best_expert_id % VPT_Capacity;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
     max_val_group = Negative_inf; 
    }
  }
  {
    constexpr int k_idx=1;
    float local_max;
    int local_idx; 
    if (max_val_group == Negative_inf) { 
        local_max = max_val_second_group;
        local_idx = max_val_second_group_idx;
      } else {
        local_max = max_val_group;
        local_idx = max_val_group_idx;
    }
    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    // Replaced params.VPT with VPT_Capacity
    int winner_thread_id = global_best_expert_id / VPT_Capacity;
    int winner_offset = global_best_expert_id % VPT_Capacity;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
    }
  }
  for (int k_idx = 2; k_idx < topk; ++k_idx) {
    float local_max = Negative_inf;
    int local_idx = first_elt_read_by_thread; 
    // Loop bound uses VPT_Capacity
    #pragma unroll
    for (int ii = 0; ii < VPT_Capacity; ++ii) {
      float val = scores_chunk[ii];
      if (val > local_max) {
        local_max = val;
        local_idx = first_elt_read_by_thread + ii;
      }
    }

    float global_max_val = local_max;
    int global_best_expert_id = local_idx;
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val ) {
        global_max_val = other_max;
        global_best_expert_id = other_expert;
      }
    }
    // Replaced params.VPT with VPT_Capacity
    int winner_thread_id = global_best_expert_id / VPT_Capacity;
    int winner_offset = global_best_expert_id % VPT_Capacity;
    float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
    output_sum += best_prob;
    if (thread_group_idx == k_idx) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
   if(k_idx + 1 < topk){
    if (thread_group_idx == winner_thread_id) {
     scores_chunk[winner_offset] = Negative_inf;
    }
   }
  }
  if (thread_group_idx < topk) {
    int const idx = topk * thread_row + thread_group_idx;
    output_ptr[idx] = my_stored_prob /output_sum;
    indices_ptr[idx] = my_stored_idx;
  }
}

template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
 static constexpr int VPT = VPT_;
 static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
 static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
 static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
 static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
 static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

// 智普优化
template <typename T, typename T_bias, typename Params, int VPT_LOCAL>
__global__ void moe_fused_gate_kernel_expert160_topk8(
  void* input,
  void* bias,
  float* output_ptr,
  int32_t* indices_ptr,
  int64_t num_rows,
  int64_t n_share_experts_fusion,
  double routed_scaling_factor,
  Params params) 
{
  constexpr float Negative_inf = -FLT_MAX;
  constexpr int NUM_EXPERTS = 160;

  int tidx = threadIdx.x;
  int group_row = tidx / params.THREADS_PER_ROW; 
  int thread_row = blockIdx.x * params.ROWS_PER_CTA + group_row;
  
  if (thread_row >= num_rows) {
      return;
  }

  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T_bias*>(bias);
  
  auto* thread_row_input = input_ptr + thread_row * NUM_EXPERTS;
  auto* thread_row_bias = bias_ptr; 

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int start_col = thread_group_idx * VPT_LOCAL;

  float scores_chunk[VPT_LOCAL];
  float row_chunk[VPT_LOCAL];

 
  constexpr int EFFECTIVE_THREADS = NUM_EXPERTS / VPT_LOCAL;
  
  if (thread_group_idx < EFFECTIVE_THREADS) {
      using VecT = AlignedArray<T, VPT_LOCAL>;
      using VecBias = AlignedArray<T_bias, VPT_LOCAL>;

      VecT input_vec = *reinterpret_cast<VecT*>(thread_row_input + start_col);
      VecBias bias_vec = *reinterpret_cast<VecBias*>(thread_row_bias + start_col);

      #pragma unroll
      for(int i=0; i < VPT_LOCAL; ++i) {
          float in = static_cast<float>(input_vec[i]);
          float b = static_cast<float>(bias_vec[i]);
          float prob = 1.0f / (1.0f + exp_func(-in)); 
          row_chunk[i] = prob; 
          scores_chunk[i] = b + prob; 
      }
  } else {
      #pragma unroll
      for(int i=0; i < VPT_LOCAL; ++i) {
          scores_chunk[i] = Negative_inf;
          row_chunk[i] = 0.0f; 
      }
  }

  float my_stored_prob = 0.0f;
  int my_stored_idx = 0;
  float output_sum = 0.0f;
  
  int topk_routed = 8 - n_share_experts_fusion;


  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float local_max = Negative_inf;
    int local_idx = (thread_group_idx < EFFECTIVE_THREADS) ? start_col : 0;
  
    #pragma unroll
    for (int i = 0; i < VPT_LOCAL; ++i) {
      if (scores_chunk[i] > local_max) {
        local_max = scores_chunk[i];
        local_idx = start_col + i;
      }
    }

    float global_max_val = local_max;
    int global_best_expert_id = local_idx;

    #pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(global_max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(global_best_expert_id, mask, params.THREADS_PER_ROW);
      if (other_max > global_max_val || (other_max == global_max_val && other_expert < global_best_expert_id)) {
            global_max_val = other_max;
            global_best_expert_id = other_expert;
      }
    }
    int winner_thread = global_best_expert_id / VPT_LOCAL;
    int winner_offset = global_best_expert_id % VPT_LOCAL;
    
    float best_prob_val = 0.0f;
  
    if (thread_group_idx == winner_thread) {
       scores_chunk[winner_offset] = Negative_inf;
       best_prob_val = row_chunk[winner_offset];
    }
    
    float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(best_prob_val, winner_thread, params.THREADS_PER_ROW);

    if (k < topk_routed) {
        output_sum += best_prob;
    }

    if (thread_group_idx == k) {
      my_stored_prob = best_prob;
      my_stored_idx = global_best_expert_id;
    }
  }

  // 共享专家逻辑
  if (n_share_experts_fusion > 0 && thread_group_idx >= topk_routed && thread_group_idx < 8) {
      int share_offset = thread_group_idx - topk_routed;
      my_stored_idx = NUM_EXPERTS + share_offset; 
      my_stored_prob = output_sum / static_cast<float>(routed_scaling_factor);
  }

  if (thread_group_idx < 8) { // 只前8个线程写回结果
    int const out_idx = 8 * thread_row + thread_group_idx;
    output_ptr[out_idx] = my_stored_prob / output_sum;
    indices_ptr[out_idx] = my_stored_idx;
  }
}

template <
   typename T,
   typename T_bias, 
   int VPT, 
   int NUM_EXPERTS,
   int THREADS_PER_ROW, 
   int ROWS_PER_WARP,
   int ROWS_PER_CTA,
   int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(
   void* input,
   void* bias,
   float* output_ptr,
   int32_t* indices_ptr,
   int64_t num_rows,
   int64_t topk_group,
   int64_t topk,
   int64_t n_share_experts_fusion,
   double routed_scaling_factor) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
    if (NUM_EXPERTS == 384) {
      moe_fused_gate_impl_expert_384<T, T_bias, decltype(params), VPT>(
          input,
          bias,
          output_ptr,
          indices_ptr,
          num_rows,
          topk_group,
          topk,
          n_share_experts_fusion,
          routed_scaling_factor,
          params);
  } else {
    moe_fused_gate_impl<T, T_bias, decltype(params)>(
          input,
          bias,
          output_ptr,
          indices_ptr,
          num_rows,
          topk_group,
          topk,
          n_share_experts_fusion,
          routed_scaling_factor,
          params);
  }
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, T_bias, EXPERTS, EXPERT_GROUP) \
 do {               \
   constexpr int N_WARP = 2;\
   constexpr int VPT_PER_GROUP = (EXPERTS) / (EXPERT_GROUP);    \
   constexpr int VPT_PER_THREAD = VPT_PER_GROUP / 4;      \
   /* threads per row 变成了之前的四倍*/ \
   constexpr int THREADS_PER_ROW_NEW = (EXPERT_GROUP) * 4;    \
   constexpr int ROWS_PER_WARP = ((THREADS_PER_ROW_NEW) <= WARP_SIZE) ? (WARP_SIZE / (THREADS_PER_ROW_NEW)) : 1; \
   constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP; \
   moe_fused_gate_kernel<T, T_bias, VPT_PER_THREAD, EXPERTS, THREADS_PER_ROW_NEW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
     <<<num_blocks, block_dim, 0, stream>>>( \
       input.data_ptr(),          \
       bias.data_ptr(),          \
       output.data_ptr<float>(),        \
       indices.data_ptr<int32_t>(),      \
       num_rows,              \
       topk_group,            \
       topk,                \
       n_share_experts_fusion,        \
       routed_scaling_factor);        \
   dispatched = true;              \
 } while (0)

 #define LAUNCH_MOE_GATE_E160_NE1(T, T_bias, THREADS_PER_ROW_VAL, VPT_VAL, N_WARP_VAL) \
 do { \
   constexpr int N_WARP = N_WARP_VAL; \
   constexpr int THREADS_PER_ROW = THREADS_PER_ROW_VAL; \
   constexpr int VPT = VPT_VAL; \
   constexpr int EXPERTS = 160; \
   constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW; \
   constexpr int ROWS_PER_CTA = N_WARP * ROWS_PER_WARP; \
   dim3 block_dim_spec(WARP_SIZE*N_WARP); \
   int64_t num_blocks_spec = (num_rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA; \
   KernelParams<VPT, EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, N_WARP> params_spec; \
   moe_fused_gate_kernel_expert160_topk8<T, T_bias, decltype(params_spec), VPT> \
     <<<num_blocks_spec, block_dim_spec, 0, stream>>>( \
       input.data_ptr(),          \
       bias.data_ptr(),          \
       output.data_ptr<float>(),        \
       indices.data_ptr<int32_t>(),       \
       num_rows,              \
       n_share_experts_fusion,        \
       routed_scaling_factor, \
       params_spec);        \
   dispatched = true; \
 } while (0)

struct KernelParamsDynamic {
 int VPT;
 int NUM_EXPERTS;
 int THREADS_PER_ROW;
 int ROWS_PER_WARP;
 int ROWS_PER_CTA;
 int WARPS_PER_CTA;
};

template <typename T, typename T_bias>
__global__ void moe_fused_gate_kernel_dynamic(
   void* input,
   void* bias,
   float* output_ptr,
   int32_t* indices_ptr,
   int64_t num_rows,
   int64_t num_experts,
   int64_t num_expert_group,
   int64_t topk_group,
   int64_t topk,
   int64_t n_share_experts_fusion,
   double routed_scaling_factor) {
 KernelParamsDynamic params;
 params.NUM_EXPERTS = num_experts;    // e.g, for deepseek v3, this is 256
 params.VPT = (num_experts / num_expert_group) / 4;
 params.THREADS_PER_ROW = num_expert_group * 4;
 params.WARPS_PER_CTA = WARPS_PER_CTA;   
 params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / params.THREADS_PER_ROW);
 params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;
   moe_fused_gate_impl<T, T_bias>(
     input,
     bias,
     output_ptr,
     indices_ptr,
     num_rows,
     topk_group,
     topk,
     n_share_experts_fusion,
     routed_scaling_factor,
     params);
}
std::vector<torch::Tensor> moe_fused_gate(
   torch::Tensor& input, //[num_tokens,num_expert]
   torch::Tensor& bias,
   int64_t num_expert_group,
   int64_t topk_group,
   int64_t topk,
   int64_t n_share_experts_fusion,
   double routed_scaling_factor) {
 int64_t num_rows = input.size(0);
 int32_t num_experts = input.size(1);
 auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
 auto output = torch::empty({num_rows, topk}, options);
 auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

 int64_t threads_per_row = num_expert_group * 4;

 int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / threads_per_row);
 int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
 int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
 const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
 dim3 block_dim(WARP_SIZE, WARPS_PER_CTA); // This is correct for the new thread_row calculation

 int computed_vpt = num_experts / num_expert_group;
 int computed_vpt_per_thread = computed_vpt / 4;

 //智普优化
 bool dispatched = false;
 if (num_experts == 160 && topk == 8 && num_expert_group == 1 && topk_group == 1) {
    if (input.scalar_type() == at::kBFloat16) {
        if (bias.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_E160_NE1(__nv_bfloat16, float, 32, 8, 2);
        } else if (bias.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_E160_NE1(__nv_bfloat16, __nv_bfloat16, 32, 8, 2);
        }
    } else if (input.scalar_type() == at::kHalf) {
         if (bias.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_E160_NE1(half, half, 32, 8, 2);
         }
    } else if (input.scalar_type() == at::kFloat) {
         if (bias.scalar_type() == at::kFloat) {
            if (num_rows < 512) {
                LAUNCH_MOE_GATE_E160_NE1(float, float, 64, 4, 1);
            } else {
                LAUNCH_MOE_GATE_E160_NE1(float, float, 32, 8, 2);
            }
         }
    }
    return {output, indices};
 }

 if (input.scalar_type() == at::kBFloat16) {
   if (bias.scalar_type() == at::kFloat) {
    switch (num_experts) {
     case 384: // 支持 384 专家
        if (num_expert_group == 16) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 384, 16);
        } else if (num_expert_group == 8) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 384, 8);
        } else if (num_expert_group == 4) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 384, 4);
        }
        break;
     case 256:
      if (num_expert_group == 16) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 256, 16);
      }
      else if (num_expert_group == 8) { 
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 256, 8);
      }
      break;
     case 128:
      if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 128, 8);
      }
      else if (num_expert_group == 4) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, float, 128, 4);
      }
      break;
     default:
      break;
    }
   } else if (bias.scalar_type() == at::kBFloat16) {
    switch (num_experts) {
     case 384: // 支持 384 专家
        if (num_expert_group == 16) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 384, 16);
        } else if (num_expert_group == 8) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 384, 8);
        } else if (num_expert_group == 4) {
          LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 384, 4);
        }
        break;
     case 256:
      if (num_expert_group == 16) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 256, 16);
      } else if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 256, 8);
      }
      break;
     case 128:
      if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 128, 8);
      } else if (num_expert_group == 4) {
       LAUNCH_MOE_GATE_CONFIG(__nv_bfloat16, __nv_bfloat16, 128, 4);
      }
      break;
     default:
      break;
    }
   }
 } else if (input.scalar_type() == at::kHalf) {
   if (bias.scalar_type() == at::kHalf) {
    switch (num_experts) {
     case 384: // 支持 384 专家
      if (num_expert_group == 16) {
        LAUNCH_MOE_GATE_CONFIG(half, half, 384, 16);
      } else if (num_expert_group == 8) {
        LAUNCH_MOE_GATE_CONFIG(half, half, 384, 8);
      } else if (num_expert_group == 4) {
        LAUNCH_MOE_GATE_CONFIG(half, half, 384, 4);
      }
      break;
     case 256:
      if (num_expert_group == 16) {
       LAUNCH_MOE_GATE_CONFIG(half, half, 256, 16);
      } else if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(half, half, 256, 8);
      }
      break;
     case 128:
      if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(half, half, 128, 8);
      } else if (num_expert_group == 4) {
       LAUNCH_MOE_GATE_CONFIG(half, half, 128, 4);
      }
      break;
     default:
      break;
    }
   }
 } else if (input.scalar_type() == at::kFloat) {
   if (bias.scalar_type() == at::kFloat) {
    switch (num_experts) {
     case 384: // 支持 384 专家
      if (num_expert_group == 16) {
        LAUNCH_MOE_GATE_CONFIG(float, float, 384, 16);
      } else if (num_expert_group == 8) {
        LAUNCH_MOE_GATE_CONFIG(float, float, 384, 8);
      } else if (num_expert_group == 4) {
        LAUNCH_MOE_GATE_CONFIG(float, float, 384, 4);
      }
      break;
     case 256:
      if (num_expert_group == 16) {
       LAUNCH_MOE_GATE_CONFIG(float, float, 256, 16);
      } else if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(float, float, 256, 8);
      }
      break;
     case 128:
      if (num_expert_group == 8) {
       LAUNCH_MOE_GATE_CONFIG(float, float, 128, 8);
      } else if (num_expert_group == 4) {
       LAUNCH_MOE_GATE_CONFIG(float, float, 128, 4);
      }
      break;
     default:
      break;
    }
   }
 }
 
 if (!dispatched) {
   // Fallback to the dynamic kernel if none of the supported combinations match.
   
   if (input.scalar_type() == at::kBFloat16) {
    if (bias.scalar_type() == at::kFloat) {
     moe_fused_gate_kernel_dynamic<__nv_bfloat16, float><<<num_blocks, block_dim, 0, stream>>>(
       input.data_ptr(),
       bias.data_ptr(),
       output.data_ptr<float>(),
       indices.data_ptr<int32_t>(),
       num_rows,
       num_experts,
       num_expert_group,
       topk_group,
       topk,
       n_share_experts_fusion,
       routed_scaling_factor);
    } else if (bias.scalar_type() == at::kBFloat16) {
     moe_fused_gate_kernel_dynamic<__nv_bfloat16, __nv_bfloat16><<<num_blocks, block_dim, 0, stream>>>(
       input.data_ptr(),
       bias.data_ptr(),
       output.data_ptr<float>(),
       indices.data_ptr<int32_t>(),
       num_rows,
       num_experts,
       num_expert_group,
       topk_group,
       topk,
       n_share_experts_fusion,
       routed_scaling_factor);
    } else {
     TORCH_CHECK(false, "Unsupported bias data type for bf16 input in moe_fused_gate");
    }
   } else if (input.scalar_type() == at::kHalf) {
    if (bias.scalar_type() == at::kHalf) {
     moe_fused_gate_kernel_dynamic<half, half><<<num_blocks, block_dim, 0, stream>>>(
       input.data_ptr(),
       bias.data_ptr(),
       output.data_ptr<float>(),
       indices.data_ptr<int32_t>(),
       num_rows,
       num_experts,
       num_expert_group,
       topk_group,
       topk,
       n_share_experts_fusion,
       routed_scaling_factor);
    } else {
      TORCH_CHECK(false, "Unsupported bias data type for fp16 input in moe_fused_gate");
    }
   } else if (input.scalar_type() == at::kFloat) {
    if (bias.scalar_type() == at::kFloat) {
     moe_fused_gate_kernel_dynamic<float, float><<<num_blocks, block_dim, 0, stream>>>(
       input.data_ptr(),
       bias.data_ptr(),
       output.data_ptr<float>(),
       indices.data_ptr<int32_t>(),
       num_rows,
       num_experts,
       num_expert_group,
       topk_group,
       topk,
       n_share_experts_fusion,
       routed_scaling_factor);
    } else {
     TORCH_CHECK(false, "Unsupported bias data type for fp32 input in moe_fused_gate");
    }
   } else {
    TORCH_CHECK(false, "Unsupported input data type for moe_fused_gate");
   }
 }
 return {output, indices};
}
}
}