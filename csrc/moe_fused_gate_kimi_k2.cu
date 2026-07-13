#include <ATen/hip/HIPContext.h>
#include <cfloat>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <torch/all.h>
#include <type_traits>

namespace at {
namespace native {

static constexpr int WARP_SIZE = 64; 
static constexpr int NUM_EXPERTS = 384;

template <typename T>
__device__ __forceinline__ T warp_shfl_xor(T var, int lane_mask) {
    return __shfl_xor(var, lane_mask);
}

template <typename T>
__device__ __forceinline__ T warp_shfl_idx(T var, int src_lane) {
    return __shfl(var, src_lane);
}


template <int LOGICAL_WARP_SIZE>
__global__ void kimi_k2_moe_fused_gate_kernel_opt(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output_ptr, 
    int32_t* __restrict__ indices_ptr,
    int64_t num_rows, 
    int64_t topk, 
    bool renormalize,
    float routed_scaling_factor, 
    bool apply_routed_scaling_factor_on_output) {


    constexpr int VPT = NUM_EXPERTS / LOGICAL_WARP_SIZE;
    constexpr int TOKENS_PER_PHYSICAL_WARP = WARP_SIZE / LOGICAL_WARP_SIZE;

  
    int64_t global_physical_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int physical_lane_id = threadIdx.x % WARP_SIZE;
    
    int sub_warp_id = physical_lane_id / LOGICAL_WARP_SIZE; 
    int logical_lane_id = physical_lane_id % LOGICAL_WARP_SIZE;

    int64_t row_idx = global_physical_warp_id * TOKENS_PER_PHYSICAL_WARP + sub_warp_id;

    if (row_idx >= num_rows) return;


    float local_scores[VPT];
    float local_output_scores[VPT];

    const float* input_row = input + row_idx * NUM_EXPERTS;
    const float* bias_ptr = bias;

    #pragma unroll
    for (int k = 0; k < VPT; ++k) {
        int expert_idx = k * LOGICAL_WARP_SIZE + logical_lane_id; 
        
        float val = input_row[expert_idx];
        float b = bias_ptr[expert_idx];
        
        float sigmoid = 1.0f / (1.0f + __expf(-val));
        
        local_output_scores[k] = sigmoid;
        local_scores[k] = sigmoid + b;
    }


    float my_stored_prob = 0.0f;
    int my_stored_idx = 0;
    float output_sum = 0.0f;

    for (int k = 0; k < topk; ++k) {
        float my_max_val = -FLT_MAX;
        int my_max_idx = -1;

        #pragma unroll
        for (int i = 0; i < VPT; ++i) {
            if (local_scores[i] > my_max_val) {
                my_max_val = local_scores[i];
                my_max_idx = i * LOGICAL_WARP_SIZE + logical_lane_id;
            }
        }

     
        float global_max_val = my_max_val;
        int global_best_expert_id = my_max_idx;

        #pragma unroll
        for (int mask = LOGICAL_WARP_SIZE / 2; mask > 0; mask /= 2) {
            float other_val = warp_shfl_xor(global_max_val, mask);
            int other_idx = warp_shfl_xor(global_best_expert_id, mask);
            if (other_val > global_max_val) {
                global_max_val = other_val;
                global_best_expert_id = other_idx;
            }
        }

       
        int winner_logical_lane = global_best_expert_id % LOGICAL_WARP_SIZE;
        int winner_offset = global_best_expert_id / LOGICAL_WARP_SIZE; 
        
        float val_to_broadcast = 0.0f;

      
        if (logical_lane_id == winner_logical_lane) {
             val_to_broadcast = local_output_scores[winner_offset];
             
             
             local_scores[winner_offset] = -FLT_MAX;
        }
        
  
        int winner_physical_lane = winner_logical_lane + (sub_warp_id * LOGICAL_WARP_SIZE);
        float global_output_val = warp_shfl_idx(val_to_broadcast, winner_physical_lane);

      
        if (renormalize) {
            output_sum += global_output_val;
        }

        
        if (logical_lane_id == k) {
            my_stored_prob = global_output_val;
            my_stored_idx = global_best_expert_id;
        }
    }

    if (logical_lane_id < topk) {
        float val = my_stored_prob;
        if (renormalize) {
            val /= output_sum;
            if (apply_routed_scaling_factor_on_output) {
                val *= static_cast<float>(routed_scaling_factor);
            }
        }
        int write_idx = row_idx * topk + logical_lane_id;
        output_ptr[write_idx] = val;
        indices_ptr[write_idx] = my_stored_idx;
    }
}

std::vector<at::Tensor> kimi_k2_moe_fused_gate(
    at::Tensor &input, 
    at::Tensor &bias, 
    int64_t topk,
    bool renormalize, 
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {

    int64_t num_rows = input.size(0);
    int32_t num_experts = input.size(1);

    TORCH_CHECK(num_experts == NUM_EXPERTS, "Supports only 384 experts");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 input");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 bias");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty({num_rows, topk}, options);
    auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(num_rows <= 2048){
          constexpr int SUB_WARP_SIZE = 64; 
          constexpr int TOKENS_PER_PHYSICAL_WARP = WARP_SIZE / SUB_WARP_SIZE; // 1
          
          int64_t TOTAL_WARPS = (num_rows + TOKENS_PER_PHYSICAL_WARP - 1) / TOKENS_PER_PHYSICAL_WARP;
          
          int WARPS_PER_CTA = 4;
          dim3 block_dim(WARP_SIZE * WARPS_PER_CTA);
          
          int64_t grid_size = (TOTAL_WARPS + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

          kimi_k2_moe_fused_gate_kernel_opt<SUB_WARP_SIZE><<<grid_size, block_dim, 0, stream>>>(
              input.data_ptr<float>(),
              bias.data_ptr<float>(),
              output.data_ptr<float>(),
              indices.data_ptr<int32_t>(),
              num_rows,
              topk,
              renormalize,
              static_cast<float>(routed_scaling_factor),
              apply_routed_scaling_factor_on_output
          );
    } else {
          constexpr int SUB_WARP_SIZE = 32; 
          constexpr int TOKENS_PER_PHYSICAL_WARP = WARP_SIZE / SUB_WARP_SIZE;
          
          int64_t TOTAL_WARPS = (num_rows + TOKENS_PER_PHYSICAL_WARP - 1) / TOKENS_PER_PHYSICAL_WARP;
          
          int WARPS_PER_CTA = 2; 
          dim3 block_dim(WARP_SIZE * WARPS_PER_CTA);
          
          int64_t grid_size = (TOTAL_WARPS + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
          
          kimi_k2_moe_fused_gate_kernel_opt<SUB_WARP_SIZE><<<grid_size, block_dim, 0, stream>>>(
              input.data_ptr<float>(),
              bias.data_ptr<float>(),
              output.data_ptr<float>(),
              indices.data_ptr<int32_t>(),
              num_rows,
              topk,
              renormalize,
              static_cast<float>(routed_scaling_factor),
              apply_routed_scaling_factor_on_output
          );
    }
  
    return {output, indices};
}

} // namespace native
} // namespace at