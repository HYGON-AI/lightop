#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/all.h>
#include <cfloat>
#include <type_traits>
#include <hip/hip_bf16.h>


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

template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
 if constexpr (std::is_same<T, at::Half>::value) {
 	return static_cast<float>(a) > static_cast<float>(b);
 } else {
 	return a > b;
 }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
 if constexpr (std::is_same<T, at::Half>::value) {
 	return static_cast<float>(a) == static_cast<float>(b);
 } else {
 	return a == b;
 }
}


// Fixed constants common to both dynamic and static template versions:
static constexpr int WARPS_PER_CTA = 1;
// 现在约束的是 per-thread VPT
static constexpr int MAX_VPT = 32; // maximum VPT we support, > params.VPT = (num_expert / num_expert_group) / 4

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

    if (thread_row >= num_rows) {
        return;
    }

    constexpr float Negative_inf = -FLT_MAX;
    int topk_excluding_share_expert_fusion = topk - (n_share_experts_fusion > 0 ? 1 : 0);

    auto* input_ptr = reinterpret_cast<T*>(input);
    auto* bias_ptr = reinterpret_cast<T_bias*>(bias);
    
    auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;
    
    int thread_group_idx = tidx % params.THREADS_PER_ROW;

    int first_elt_read_by_thread = thread_group_idx * params.VPT;

    T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
    T_bias* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;

    Array<float, 8> row_chunk; 
    Array<float, 8> bias_chunk;
    

    using AccessType = AlignedArray<T, 8>; 
    using BiasAccessType = AlignedArray<T_bias, 8>;

    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
    const BiasAccessType* vec_bias_thread_read_ptr = reinterpret_cast<const BiasAccessType*>(bias_thread_read_ptr);

    #pragma unroll 8
    for (int ii = 0; ii < params.VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(vec_thread_read_ptr[0][ii]);
        bias_chunk[ii] = static_cast<float>(vec_bias_thread_read_ptr[0][ii]);
    }

 
    #pragma unroll 8
    for (int ii = 0; ii < params.VPT; ++ii) {
        float x = -row_chunk[ii]; 
        float sigmoid_val = __fdividef(1.0f, 1.0f + __expf(x));
        row_chunk[ii] = sigmoid_val; 
    }

    Array<float, 8> scores_chunk;
    #pragma unroll 8
    for (int ii = 0; ii < params.VPT; ++ii) {
        scores_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
    }


    /////////////////////// 线程组计算top2 ///////////////////////////// 
   // int my_local_start_id = first_elt_read_by_thread;
    float max_val_group = Negative_inf;
    float max_val_second_group = Negative_inf;
	int max_val_group_idx = first_elt_read_by_thread;   //缓存top1
	int max_val_second_group_idx = first_elt_read_by_thread;
    #pragma unroll 8
    for (int ii = 0; ii < params.VPT; ++ii) {
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

    #pragma unroll
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
    #pragma unroll
    for (int k_idx = 0; k_idx < topk_group; ++k_idx) {
        float max_val_reduce = temp_group_score;
        int winning_group_id = my_group_start_expert_id;
        #pragma unroll
        for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_reduce, mask, params.THREADS_PER_ROW);
            int other_group = VLLM_SHFL_XOR_SYNC_WIDTH(winning_group_id, mask, params.THREADS_PER_ROW);

            if (other_max > max_val_reduce || (other_max == max_val_reduce && other_group < winning_group_id)) {
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
        #pragma unroll
        for (int ii = 0; ii < params.VPT; ++ii) {
            scores_chunk[ii] = Negative_inf;
        }
    }
	/////////////////////// topk ///////////////////////////// 
    // 第 k 个结果由 thread_group_idx == k 的线程写入
    float my_stored_prob = 0.0f;
    int my_stored_idx = 0;
    float output_sum = 0.0f; 
    #pragma unroll
    for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
        
            float local_max = Negative_inf;
        int local_idx = first_elt_read_by_thread; 
		//top1和top2都是从缓存里取
        if (k_idx == 0) {
            local_max = max_val_group;
            local_idx = max_val_group_idx;
        } else if (k_idx == 1) {
            if (max_val_group == Negative_inf) { 
                local_max = max_val_second_group;
                local_idx = max_val_second_group_idx;
            } else {
                local_max = max_val_group;
                local_idx = max_val_group_idx;
            }
        } else {
            #pragma unroll
            for (int ii = 0; ii < params.VPT; ++ii) {
                float val = scores_chunk[ii];
                if (val > local_max) {
                    local_max = val;
                    local_idx = first_elt_read_by_thread + ii;
                }
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

        //  广播给其他人，计算赢家属于哪个线程、哪个位置
        int winner_thread_id = global_best_expert_id / params.VPT;
        int winner_offset = global_best_expert_id % params.VPT;
        float val_to_broadcast = (thread_group_idx == winner_thread_id) ? row_chunk[winner_offset] : 0.0f;
        // 广播给所有人：现在大家都知道第 k 个专家是谁，概率是多少
        float best_prob = VLLM_SHFL_BCAST_SYNC_WIDTH(val_to_broadcast, winner_thread_id, params.THREADS_PER_ROW);
        // 所有人都要算，为了最后的归一化
        output_sum += best_prob;

        // 寄存器暂存，不用共享内存了
        if (thread_group_idx == k_idx) {
            my_stored_prob = best_prob;
            my_stored_idx = global_best_expert_id;
        }
		//mask这轮的赢家
		if(k_idx + 1 < topk_excluding_share_expert_fusion){
			if (thread_group_idx == winner_thread_id) {
				scores_chunk[winner_offset] = Negative_inf;
				if (k_idx == 0) {
                		max_val_group = Negative_inf; 
           		 }
			}
		}
    }

   //归一化因子
    float inv_sum = __fdividef(1.0f, output_sum);

    //共享专家，todo: routed_scaling_factor是double，传进来的是倒数更好
    if (n_share_experts_fusion > 0) {
        if (thread_group_idx == 0) {
            int last_idx = topk * thread_row + topk_excluding_share_expert_fusion;
            int expert_offset = thread_row % n_share_experts_fusion;
            indices_ptr[last_idx] = params.NUM_EXPERTS + expert_offset;
            output_ptr[last_idx] = output_sum / routed_scaling_factor; 
        }
    }

    if (thread_group_idx < topk_excluding_share_expert_fusion) {
        int64_t const idx = topk * thread_row + thread_group_idx;
        output_ptr[idx] = my_stored_prob * inv_sum;
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

template <
 	typename T,
 	typename T_bias, 
 	int VPT,  //VPT_PER_THREAD  32/4 = 8
 	int NUM_EXPERTS,
 	int THREADS_PER_ROW,  //THREADS_PER_ROW_NEW
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

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, T_bias, EXPERTS, EXPERT_GROUP) \
 do {                              \
 	/* 优化之前是一个线程一个gourp,opt后四个线程一个gourp*/ \
 	constexpr int VPT_PER_GROUP = (EXPERTS) / (EXPERT_GROUP);  	  \
 	constexpr int VPT_PER_THREAD = VPT_PER_GROUP / 4;  		  \
 	/* threads per row 变成了之前的四倍*/ \
 	constexpr int THREADS_PER_ROW_NEW = (EXPERT_GROUP) * 4;  	  \
 	constexpr int ROWS_PER_WARP = ((THREADS_PER_ROW_NEW) <= WARP_SIZE) ? (WARP_SIZE / (THREADS_PER_ROW_NEW)) : 1; \
 	constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP; \
 	moe_fused_gate_kernel<T, T_bias, VPT_PER_THREAD, EXPERTS, THREADS_PER_ROW_NEW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
 		  <<<num_blocks, block_dim, 0, stream>>>( \
 			  input.data_ptr(),  				  \
 			  bias.data_ptr(),  				  \
 			  output.data_ptr<float>(), 			  \
 			  indices.data_ptr<int32_t>(),  		  \
 			  num_rows,  						  \
 			  topk_group,  					  \
 			  topk,  							  \
 			  n_share_experts_fusion,  			  \
 			  routed_scaling_factor);  			  \
 	dispatched = true;  						  \
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
 params.NUM_EXPERTS = num_experts;  	  // e.g, for deepseek v3, this is 256
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

//之前是一个线程一个group，现在改为4个线程一个gourp
 int64_t threads_per_row = num_expert_group * 4;

 int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / threads_per_row);
 int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
 int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
 const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
 dim3 block_dim(WARP_SIZE, WARPS_PER_CTA); // This is correct for the new thread_row calculation

 // Check 1: Ensure that num_experts is a power of 2.
 TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

 // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
 TORCH_CHECK(
 		num_experts % num_expert_group == 0,
 		"num_experts must be divisible by num_expert_group, but got ",
 		num_experts,
 		" / ",
 		num_expert_group);

 int computed_vpt = num_experts / num_expert_group;
 
 //四个线程一个gourep
 int computed_vpt_per_thread = computed_vpt / 4;
 TORCH_CHECK(
 		computed_vpt_per_thread <= MAX_VPT,
 		"Per thread experts: (num_experts / num_expert_group / 4) = (",
 		computed_vpt_per_thread,
 		") exceeds the maximum supported (",
 		MAX_VPT,
 		")");


 // Dispatch to templated kernel for known compile-time configurations.
 bool dispatched = false;

 if (input.scalar_type() == at::kBFloat16) {
 	if (bias.scalar_type() == at::kFloat) {
 	  switch (num_experts) {
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