#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include <torch/extension.h>

#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"

#include "reduce.h"


class uint16x2 {
    public:
        uint16_t x;
        uint16_t y;
        __host__ __device__ uint16x2() = default;
        __host__ __device__ uint16x2(uint16_t x_, uint16_t y_) : x(x_), y(y_) {}
};

union BF16Union {
    uint32_t as_uint32;
    struct {
        uint16_t x;  // 第一个bf16元素
        uint16_t y;  // 第二个bf16元素
    };
};

static __device__ float inline num2float(const uint16_t x) {
    // Convert bf16 to float
    uint32_t u = static_cast<uint32_t>(x) << 16;
    return *(float*)(&u);
  }

static __host__ __device__ uint16_t inline float2num(const float x) {
    // Convert float to bf16
    uint32_t& u = *(uint32_t*)(&x);

    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}

template<typename scalar_t2>
static __device__ inline scalar_t2 atomic_add_bf16(scalar_t2* p_dst, const scalar_t2& x) {

    // bf16版本，使用atomicCAS
    // 对于bf16，scalar_t2实际上是__half2，但存储的是bf16数据
    uint32_t* dword_addr = reinterpret_cast<uint32_t*>(p_dst);

    BF16Union cur_val, old_val, new_val, x_val;
    cur_val.as_uint32 = *dword_addr;
    x_val.as_uint32 = reinterpret_cast<const uint32_t&>(x);
    do {
        old_val = cur_val;
        
        // 转换为float进行计算
        float f1 = num2float(cur_val.x);
        float f2 = num2float(cur_val.y);

        float fx1 = num2float(x_val.x);
        float fx2 = num2float(x_val.y);
        
        // 执行加法并转换回bf16
        new_val.x = float2num(f1 + fx1); 
        new_val.y = float2num(f2 + fx2);
        // 原子比较并交换
        cur_val.as_uint32 = atomicCAS(dword_addr, old_val.as_uint32, new_val.as_uint32); //builting
    } while (cur_val.as_uint32 != old_val.as_uint32);

    return x;

}


namespace at{
namespace native{
    using bf16_t = __hip_bfloat16;

    template <typename T>
    __inline__ __device__ T WarpReduceSum1(T val,int max=32) {
    for (int offset = max; offset > 0; offset >>= 1) {
        val += WARP_SHFL_DOWN(val, offset);
    }
    return val;
    }

    template <typename T>
    __inline__ __device__ T BlockReduceSum1(T val, T* shared) {

        const int lid = threadIdx.x % C10_WARP_SIZE;
        const int wid = threadIdx.x / C10_WARP_SIZE;
        const int block_size=blockDim.x;
        const int share_size=block_size/C10_WARP_SIZE;
        val = WarpReduceSum(val);
        if(block_size==C10_WARP_SIZE)return val;
        if (lid == 0&&wid<share_size) {
            shared[wid] = val;
        }
        __syncthreads();
        if (wid == 0&&lid<share_size) {
            val= shared[lid];
            val = WarpReduceSum1(val,share_size/2);
        }
        return val;
    }

    // template <int VEC>
    // __global__ void __launch_bounds__(1024) count_exper_m_kernel(
    //     bool* routing_map,        // [num_experts, num_tokens]
    //     int* expert_m,
    //     const int num_experts,
    //     const int num_tokens
    // ) {
    //     using boolVEC = __attribute__((__vector_size__(VEC * sizeof(uint8_t)))) uint8_t;
    //     int tid = threadIdx.x;
    //     bool* map_ptr = routing_map + blockIdx.x * num_tokens;
    //     int sum = 0;
    //     for(int i = tid * VEC; i < num_tokens; i += VEC * blockDim.x) {
    //         boolVEC tmp_map = *(boolVEC*)(&map_ptr[i]);
    //         #pragma unroll
    //         for(int j = 0; j < VEC && i+j < num_tokens; j++) { //这里
    //             if(tmp_map[j]) {
    //                 sum++;
    //             }
    //         }
    //     }

    //     __shared__ int shared[32];
    //     sum = BlockReduceSum1<int>(sum, shared);
    //     __syncthreads();
    //     if(tid == 0) {
    //         expert_m[blockIdx.x] = sum;
    //     }
    // }

    template <int VEC>
    __global__ void __launch_bounds__(1024) count_exper_m_kernel(
        bool* routing_map,        // [num_experts, num_tokens]
        int* expert_m,
        const int num_experts,
        const int num_tokens
    ) {
        using boolVEC = __attribute__((__vector_size__(VEC * sizeof(uint8_t)))) uint8_t;
        int tid = threadIdx.x;
        bool* map_ptr = routing_map + blockIdx.x * num_tokens;
        int sum = 0;
        int loop = num_tokens / VEC;
        int remain = num_tokens % VEC;

        for( ; tid < loop; tid += blockDim.x) {
            boolVEC tmp_map = *(boolVEC*)(&map_ptr[tid * VEC]);

            #pragma unroll
            for(int j = 0; j < VEC; j++) { //这里
                if(tmp_map[j]) {
                    sum++;
                }
            }
        }

        tid = num_tokens - remain + threadIdx.x;
        for( ; tid < num_tokens; tid += blockDim.x) {
            if(map_ptr[tid]) {
                sum++;
            }
        }

        __shared__ int shared[32];
        sum = BlockReduceSum1<int>(sum, shared);
        __syncthreads();
        if(threadIdx.x == 0) {
            expert_m[blockIdx.x] = sum;
        }
    }

    template <int VEC, int EXPERT> // VEC = EXPERT
    __global__ void count_exper_summ_kernel(
        int* expert_m,
        int* expert_m_count,
        int* count_total,
        int* expert_m_max,
        const int num_experts,
        const int num_tokens
    ) {
        using value_VEC = __attribute__((__vector_size__(VEC * sizeof(int)))) int;
        int expert_m_reg[EXPERT];
        int m_sum_reg[EXPERT] = {0};
        int sum = 0;
        int max = -1;
        if(threadIdx.x == 0){
            for(int i = 0; i < EXPERT; i+=VEC) {
                *(value_VEC*)(&expert_m_reg[i]) = *(value_VEC*)(&expert_m[i]);
            }

            max = expert_m_reg[0];
            for(int i = 1; i < EXPERT; i++) {
                sum += expert_m_reg[i-1];
                m_sum_reg[i] = sum;

                if(expert_m_reg[i] > max) {
                    max = expert_m_reg[i];
                }
            }

            #pragma unroll
            for(int i = 0; i < EXPERT; i+=VEC) {
                *(value_VEC*)(&expert_m_count[i]) = *(value_VEC*)(&m_sum_reg[i]);
            }


            sum += expert_m_reg[EXPERT-1];
            count_total[0] = sum;
            expert_m_max[0] = max;
        }
    }

    // template<int VEC = 16>
    // __global__ void __launch_bounds__(1024) sorted_indices_optimized_kernel(
    //     bool* routing_map,          // [num_experts, num_tokens]
    //     int* sorted_indices,              // output: [num_selected_tokens]
    //     int* expert_m,
    //     int* expert_m_count,
    //     const int num_tokens,
    //     const int num_experts
    // ) {
    //     int tid = threadIdx.x;
    //     int offset_start = expert_m_count[blockIdx.x];
    //     int offset_end = offset_start + expert_m[blockIdx.x];
    //     routing_map = routing_map + blockIdx.x*num_tokens;
    //     //using bool_vec = __attribute__((__vector_size__(VEC * sizeof(uint8_t)))) uint8_t;
    //     using bool_vec = bool;
    //     for(int idx = tid*VEC; idx < num_tokens; idx += blockDim.x*VEC) {
    //         bool_vec routing_map_vec = *(bool_vec*)(&routing_map[idx]);

    //         int prefix_count = offset_start;
    //         for(int t = 0; t < idx; ++t){ 
    //             if(routing_map[t])
    //                 prefix_count++;
    //         }

    //         // if(prefix_count >= offset_end) //todo
    //         //     return;

    //         #pragma unroll
    //         for(int i = 0; i < VEC && i+idx < num_tokens; i++) { //这里
    //             //if(routing_map_vec[i]) {
    //             if(routing_map_vec) {
    //                 sorted_indices[prefix_count] = idx + i;
    //                 prefix_count++;
    //             }
    //         }
    //     }
    // }


    template<int VEC = 16>
    __global__ void __launch_bounds__(1024) sorted_indices_optimized_kernel(
        bool* routing_map,          // [num_experts, num_tokens]
        int* sorted_indices,              // output: [num_selected_tokens]
        int* expert_m,
        int* expert_m_count,
        const int num_tokens,
        const int num_experts
    ) {
        int tid = threadIdx.x;
        int offset_start = expert_m_count[blockIdx.x];
        //int offset_end = offset_start + expert_m[blockIdx.x];
        
        routing_map = routing_map + blockIdx.x*num_tokens;

        int loop = num_tokens / VEC;
        int remain = num_tokens % VEC;

        using bool_vec = __attribute__((__vector_size__(VEC * sizeof(uint8_t)))) uint8_t;
        for( ; tid < loop; tid += blockDim.x) {
            bool_vec routing_map_vec = *(bool_vec*)(&routing_map[tid * VEC]);

            int prefix_count = offset_start;
            for(int t = 0; t < tid * VEC; ++t){ 
                if(routing_map[t])
                    prefix_count++;
            }

            #pragma unroll
            for(int i = 0; i < VEC; i++) { //这里
                if(routing_map_vec[i]) {
                    sorted_indices[prefix_count] = tid * VEC + i;
                    prefix_count++;
                }
            }
        }

        tid = num_tokens - remain + threadIdx.x;
        int prefix_count = offset_start;
        for( ; tid < num_tokens; tid += blockDim.x) {
            if(routing_map[tid]) {
                for(int t = 0; t < tid; ++t){ 
                    if(routing_map[t])
                        prefix_count++;
                }
                sorted_indices[prefix_count] = tid;
                prefix_count++;
            }
        }
    }
    
    template<typename T, int VEC = 8>
    __global__ void __launch_bounds__(1024) permute_tokens_kernel_vec(
        const T* input_tokens,           // [num_tokens, hidden_size]
        T* output_tokens,               // [num_selected_tokens, hidden_size]
        const int* sorted_indices,      // [num_selected_tokens]
        const int hidden_size,
        const int num_selected_tokens
    ) {
        int token_idx = blockIdx.x;
        int tid = threadIdx.x;
        using vec_T = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
        for(int i = tid * VEC; i < hidden_size; i += blockDim.x * VEC) {
            vec_T input_tokens_vec = *(vec_T*)(&input_tokens[sorted_indices[token_idx] * hidden_size + i]);
            *(vec_T*)(&output_tokens[token_idx * hidden_size + i]) = input_tokens_vec;
        }

    }

    template<typename T, int VEC = 16>
    __global__ void __launch_bounds__(1024) permute_probs_kernel_vec(
        const T* probs,                 // [num_tokens, num_experts]
        T* permuted_probs,             // [num_selected_tokens]
        int* expert_m_count,
        bool* routing_map,       // [num_experts, num_tokens]
        const int num_tokens,
        const int num_experts
    ) {
        int tid = threadIdx.x;
        int offset_start = expert_m_count[blockIdx.x];
        // if(tid == 0)
        //     printf("blockIdx.x: %d, offset_start: %d\n", blockIdx.x, offset_start);
        routing_map = routing_map + blockIdx.x*num_tokens;
        using bool_vec = __attribute__((__vector_size__(VEC * sizeof(uint8_t)))) uint8_t;
        
        for(int idx = tid*VEC; idx < num_tokens; idx += blockDim.x*VEC) {
            bool_vec routing_map_vec = *(bool_vec*)(&routing_map[idx]);

            int prefix_count = offset_start;
            for(int t = 0; t < idx; ++t){
                if(routing_map[t])
                    prefix_count++;
            }

            #pragma unroll
            for(int i = 0; i < VEC && i+idx < num_tokens; i++) { //这里
                if(routing_map_vec[i]) {
                    int token_idx = (blockIdx.x*num_tokens + idx + i) % num_tokens;
                    permuted_probs[prefix_count] = probs[token_idx * num_experts + blockIdx.x];
                    prefix_count++;
                }
            }
        }
    }

    template<typename T>
    __global__ void unpermute_with_probs_kernel(
        const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
        T* output_tokens,              // [num_tokens, hidden_size]
        const int* sorted_indices,     // [num_selected_tokens]
        const T* permuted_probs,       // [num_selected_tokens]
        const int hidden_size,
        const int num_tokens,
        const int num_selected_tokens
    ) {
        int permuted_token_idx = blockIdx.x;
        int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
        
        if (permuted_token_idx < num_selected_tokens && hidden_idx < hidden_size) {
            int dst_token_idx = sorted_indices[permuted_token_idx];
            T prob_weight = permuted_probs[permuted_token_idx];
            T weighted_value = permuted_tokens[permuted_token_idx * hidden_size + hidden_idx] * prob_weight;
            
            atomicAdd(&output_tokens[dst_token_idx * hidden_size + hidden_idx], weighted_value);
        }
    }

    // template<typename T, int VEC = 8>
    // __global__ void __launch_bounds__(1024) unpermute_with_probs_kernel_fast(
    //     const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
    //     T* output_tokens,              // [num_tokens, hidden_size]
    //     const int* sorted_indices,     // [num_selected_tokens]
    //     const T* permuted_probs,       // [num_selected_tokens]
    //     const int hidden_size,
    //     const int num_tokens,
    //     const int num_selected_tokens
    // ) {
    //     int permuted_token_idx = blockIdx.x;
    //     int tid = threadIdx.x;

    //     using vec_T = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
     
    //     T prob_weight = permuted_probs[permuted_token_idx];
    //     int dst_token_idx = sorted_indices[permuted_token_idx];
    //     for(int i = tid * VEC; i < hidden_size; i += blockDim.x * VEC) {
    //         vec_T permuted_tokens_vec = *(vec_T*)(&permuted_tokens[permuted_token_idx * hidden_size + i]);
    //         T* permute_tokens_T = (T*)(&permuted_tokens_vec);

            
    //         #pragma unroll
    //         for(int j = 0; j < VEC; j++) {
    //             permute_tokens_T[j] = permute_tokens_T[j] * prob_weight;
    //         }

    //         #pragma unroll
    //         for(int j = 0; j < VEC; j++){
    //             atomicAdd(&output_tokens[dst_token_idx * hidden_size + i + j], permute_tokens_T[j]); //todo bf16的原子加优化
    //         }
    //     }
    // }

    template<typename T, int VEC = 8>
    __global__ void __launch_bounds__(1024) unpermute_with_probs_kernel_fast(
        const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
        T* output_tokens,              // [num_tokens, hidden_size]
        const int* sorted_indices,     // [num_selected_tokens]
        const T* permuted_probs,       // [num_selected_tokens]
        const int hidden_size,
        const int num_tokens,
        const int num_selected_tokens
    ) {
        int permuted_token_idx = blockIdx.x;
        int tid = threadIdx.x;

        using vec_T = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
        using bf16_2 = uint16x2;
        T prob_weight = permuted_probs[permuted_token_idx];
        int dst_token_idx = sorted_indices[permuted_token_idx];
        for(int i = tid * VEC; i < hidden_size; i += blockDim.x * VEC) {
        
            #pragma unroll
            for(int j = 0; j < VEC; j+=2){
                
                uint16x2 permute_tokens_T = *(uint16x2*)(&permuted_tokens[permuted_token_idx * hidden_size + i+j]);
                T* tmp_t = (T*)(&permute_tokens_T);
                tmp_t[0] = (tmp_t[0] * prob_weight);
                tmp_t[1] = (tmp_t[1] * prob_weight); 
                bf16_2* output_c = reinterpret_cast<bf16_2*>(&output_tokens[dst_token_idx * hidden_size + i + j]);
                atomic_add_bf16<bf16_2>(output_c, permute_tokens_T);
            }
        }
    }

    template<typename T, int VEC = 8>
    __global__ void __launch_bounds__(1024) unpermute_tokens_kernel_fast(
        const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
        T* output_tokens,              // [num_tokens, hidden_size]
        const int* sorted_indices,     // [num_selected_tokens]
        const int hidden_size,
        const int num_tokens,
        const int num_selected_tokens
    ) {
        int permuted_token_idx = blockIdx.x;
        int tid = threadIdx.x;

        using vec_T = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
        using bf16_2 = uint16x2;
     
        int dst_token_idx = sorted_indices[permuted_token_idx];
        for(int i = tid * VEC; i < hidden_size; i += blockDim.x * VEC) {
            #pragma unroll
            for(int j = 0; j < VEC; j+=2){
                uint16x2 permute_tokens_T = *(uint16x2*)(&permuted_tokens[permuted_token_idx * hidden_size + i+j]);
                bf16_2* output_c = reinterpret_cast<bf16_2*>(&output_tokens[dst_token_idx * hidden_size + i + j]);
                atomic_add_bf16<bf16_2>(output_c, permute_tokens_T);
            }
        }
    }

    template<typename T, int VEC = 8, int TC = 2>
    __global__ void __launch_bounds__(1024) unpermute_with_probs_kernel_fast2(
        const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
        T* output_tokens,              // [num_tokens, hidden_size]
        const int* sorted_indices,     // [num_selected_tokens]
        const T* permuted_probs,       // [num_selected_tokens]
        const int hidden_size,
        const int num_tokens,
        const int num_selected_tokens
    ) {
        int permuted_token_idx = blockIdx.x / TC;
        const int tid_offset = hidden_size / TC;

        int tid_start = (blockIdx.x % TC) * tid_offset;
        int tid_end = tid_start + tid_offset;
        int tid = threadIdx.x;

        using vec_T = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
     
        T prob_weight = permuted_probs[permuted_token_idx];
        int dst_token_idx = sorted_indices[permuted_token_idx];

        //for(int i = tid * VEC; i < hidden_size; i += blockDim.x * VEC) {
        for(int i = tid * VEC + tid_start; i < tid_end; i += blockDim.x * VEC) {
            vec_T permuted_tokens_vec = *(vec_T*)(&permuted_tokens[permuted_token_idx * hidden_size + i]);
            T* permute_tokens_T = (T*)(&permuted_tokens_vec);

            
            #pragma unroll
            for(int j = 0; j < VEC; j++) {
                permute_tokens_T[j] = permute_tokens_T[j] * prob_weight;
                //atomicAdd(&output_tokens[dst_token_idx * hidden_size + i + j], permute_tokens_T[j]); //todo bf16的原子加优化
                // #ifdef __gfx936__
                // __builtin_amdgcn_global_atomic_fadd_f32((float*)(output_tokens + dst_token_idx * hidden_size + i + j), (float)permute_tokens_T[j]);
                // #else
                // atomicAdd(&output_tokens[dst_token_idx * hidden_size + i + j], permute_tokens_T[j]);
                // #endif
            }

            #pragma unroll
            for(int j = 0; j < VEC; j++){
                atomicAdd(&output_tokens[dst_token_idx * hidden_size + i + j], permute_tokens_T[j]); //todo bf16的原子加优化
            }
        }
    }

    // CUDA kernel for unpermuting tokens
    template<typename T>
    __global__ void unpermute_tokens_kernel(
        const T* permuted_tokens,       // [num_selected_tokens, hidden_size]
        T* output_tokens,              // [num_tokens, hidden_size]
        const int* sorted_indices,     // [num_selected_tokens]
        const int hidden_size,
        const int num_tokens,
        const int num_selected_tokens
    ) {
        int permuted_token_idx = blockIdx.x;
        int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
        
        if (permuted_token_idx < num_selected_tokens && hidden_idx < hidden_size) {
            int dst_token_idx = sorted_indices[permuted_token_idx];
            atomicAdd(&output_tokens[dst_token_idx * hidden_size + hidden_idx],
                    permuted_tokens[permuted_token_idx * hidden_size + hidden_idx]);
        }
    }

//-----------------------------------------------------------------------------------------------------下发后的permute python 接口--------------------------------------------------------------------------------------------------//
#if 0
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> groupgemm_permute(
        const torch::Tensor& tokens,
        const torch::Tensor& routing_map
    ) {
        auto device = tokens.device();
        auto stream = at::cuda::getCurrentCUDAStream();
        
        int num_tokens = tokens.size(0);
        int hidden_size = tokens.size(1);
        int num_experts = routing_map.size(1);

        auto routing_map_bool = routing_map.to(torch::kBool).transpose(0, 1).contiguous();
        
        // Count selected tokens
        auto count_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto expert_m_max = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    
        auto expert_m = torch::zeros({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto expert_m_count = torch::zeros({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        
        
        dim3 block_size(512);
        dim3 grid_size(1);
        count_exper_m_kernel<8><<<num_experts, block_size, 256, stream>>>(
            routing_map_bool.data_ptr<bool>(),
            expert_m.data_ptr<int>(),
            num_experts,
            num_tokens
        );

        // 获取count expert_m expert_m_count
    #if 1 // 2 ms
        if (num_experts == 16) {
            count_exper_summ_kernel<4, 16><<<1, 1, 0, stream>>>(
                expert_m.data_ptr<int>(),
                expert_m_count.data_ptr<int>(),
                count_tensor.data_ptr<int>(),
                expert_m_max.data_ptr<int>(),
                num_experts,
                num_tokens
            );
        } else if (num_experts == 8) {
            count_exper_summ_kernel<4, 8><<<1, 1, 0, stream>>>(
                expert_m.data_ptr<int>(),
                expert_m_count.data_ptr<int>(),
                count_tensor.data_ptr<int>(),
                expert_m_max.data_ptr<int>(),
                num_experts,
                num_tokens
            );
        } else if(num_experts == 4) {
            count_exper_summ_kernel<4, 4><<<1, 1, 0, stream>>>(
                expert_m.data_ptr<int>(),
                expert_m_count.data_ptr<int>(),
                count_tensor.data_ptr<int>(),
                expert_m_max.data_ptr<int>(),
                num_experts,
                num_tokens
            );
        } else if(num_experts == 2) {
            count_exper_summ_kernel<2, 2><<<1, 1, 0, stream>>>(
                expert_m.data_ptr<int>(),
                expert_m_count.data_ptr<int>(),
                count_tensor.data_ptr<int>(),
                expert_m_max.data_ptr<int>(),
                num_experts,
                num_tokens
            );
        } else{
            printf("ERROR: permute_with_indices_cuda only support 2, 4, 8, 16 experts\n");
        }
    #endif
        
    
        
        int num_selected_tokens = count_tensor.item<int>();
        auto sorted_indices = torch::zeros({num_selected_tokens}, 
                                         torch::TensorOptions().dtype(torch::kInt32).device(device));
        sorted_indices_optimized_kernel<4><<<num_experts, 1024, 0, stream>>>(
            routing_map_bool.data_ptr<bool>(),
            sorted_indices.data_ptr<int>(),
            expert_m.data_ptr<int>(),
            expert_m_count.data_ptr<int>(),
            num_tokens,
            num_experts
        );
    
        
        // Permute tokens
        auto permuted_tokens = torch::empty({num_selected_tokens, hidden_size}, tokens.options());
        int permute_grid = num_selected_tokens;
        int permute_block = std::min(hidden_size / 8, 1024);
        permute_tokens_kernel_vec<bf16_t, 8><<<permute_grid, permute_block, 0, stream>>>(
            (bf16_t*)tokens.data_ptr(),
            (bf16_t*)permuted_tokens.data_ptr(),
            sorted_indices.data_ptr<int>(),
            hidden_size,
            num_selected_tokens
        );
    

        return std::make_tuple(permuted_tokens, sorted_indices, expert_m, expert_m_count, expert_m_max);
    }


    torch::Tensor groupgemm_unpermute(
        const torch::Tensor& permuted_tokens,
        const torch::Tensor& sorted_indices,
        const std::vector<int64_t>& restore_shape,
        //const torch::Tensor& probs,
        //const torch::Tensor& routing_map,
        std::optional<torch::Tensor> probs,
        std::optional<torch::Tensor> routing_map,
        const torch::Tensor& expert_m_count
    ) {
        auto device = permuted_tokens.device();
        auto stream = at::cuda::getCurrentCUDAStream();
        
        int num_selected_tokens = permuted_tokens.size(0);
        int hidden_size = permuted_tokens.size(1);
        int num_tokens = restore_shape[0];
        
        // Create output tensor filled with zeros
        auto output_tokens = torch::zeros(restore_shape, permuted_tokens.options());
        
        // Convert sorted_indices to int32 for CUDA kernel
        auto sorted_indices_int = sorted_indices.to(torch::kInt32);
        
        dim3 block(256);
        dim3 grid(num_selected_tokens, (hidden_size + block.x - 1) / block.x);
        
        //bool use_probs = probs.defined() && probs.numel() > 0;
        bool use_probs = probs.has_value();
        
        if (use_probs) {
            // Need to permute probs first
            
            int num_experts = routing_map.value().size(1);
            auto routing_map_bool = routing_map.value().to(torch::kBool).transpose(0, 1).contiguous();
            
            auto permuted_probs = torch::empty({num_selected_tokens}, probs.value().options());
            
            dim3 prob_grid((num_experts * num_tokens + 255) / 256);
            dim3 prob_block(256);
            
    
            permute_probs_kernel_vec<bf16_t, 8><<<num_experts, 512, 0, stream>>>(
                //(bf16_t*)probs.data_ptr(),
                (bf16_t*)probs.value().data_ptr(),
                (bf16_t*)permuted_probs.data_ptr(),
                expert_m_count.data_ptr<int>(),
                routing_map_bool.data_ptr<bool>(),
                num_tokens,
                num_experts
            );
          

            unpermute_with_probs_kernel_fast<bf16_t, 4><<<num_selected_tokens, 512, 0, stream>>>(
                (bf16_t*)permuted_tokens.data_ptr(),
                (bf16_t*)output_tokens.data_ptr(),
                sorted_indices_int.data_ptr<int>(),
                (bf16_t*)permuted_probs.data_ptr(),
                hidden_size,
                num_tokens,
                num_selected_tokens
            );
         
        } else {
            // printf("unpermute_with_probs_kernel_fast null\n");
         
            unpermute_tokens_kernel_fast<bf16_t, 4><<<num_selected_tokens, 512, 0, stream>>>(
                (bf16_t*)permuted_tokens.data_ptr(),
                (bf16_t*)output_tokens.data_ptr(),
                sorted_indices_int.data_ptr<int>(),
                hidden_size,
                num_tokens,
                num_selected_tokens
            );
        }
        
        return output_tokens;
    }
#endif
    ////////////////////////////////////////////////////////////////// torch_permute 接口////////////////////////////////////////////////////////////////////
    template <int VEC> // VEC
    __global__ void  count_expert_cm(
        int* expert_m,
        int* expert_m_count,
        int* count_total,
        const int num_experts,
        const int num_tokens
    ) {
        using value_VEC = __attribute__((__vector_size__(VEC * sizeof(int)))) int;
        int sum = 0;
        int count_vec[VEC];
        if(threadIdx.x == 0){
            for(int i = 0; i < num_experts; i+=VEC) {
                value_VEC expert_m_reg = *(value_VEC*)(&expert_m[i]);

                #pragma unroll
                for(int j = 0; j < VEC; ++j){
                    count_vec[j] = sum;
                    sum += expert_m_reg[j];
                }

                *(value_VEC*)(expert_m_count + i) = *(value_VEC*)(&count_vec);
            }
            count_total[0] = sum;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> groupgemm_permute(
        const torch::Tensor& tokens,
        const torch::Tensor& routing_map
    ) {
        auto device = tokens.device();
        auto stream = at::cuda::getCurrentCUDAStream();
        
        int num_tokens = tokens.size(0);
        int hidden_size = tokens.size(1);
        int num_experts = routing_map.size(1);

        auto routing_map_bool = routing_map.to(torch::kBool).transpose(0, 1).contiguous();
        
        // Count selected tokens
        auto count_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        auto expert_m = torch::zeros({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto expert_m_count = torch::zeros({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        
        
        int block_size = std::min(std::max(num_tokens / 8, 256), 512); // =512
        count_exper_m_kernel<8><<<num_experts, block_size, 256, stream>>>(
            routing_map_bool.data_ptr<bool>(),
            expert_m.data_ptr<int>(),
            num_experts,
            num_tokens
        );

        // 获取count expert_m expert_m_count
        count_expert_cm<4><<<1, 1, 0, stream>>>(
            expert_m.data_ptr<int>(),
            expert_m_count.data_ptr<int>(),
            count_tensor.data_ptr<int>(),
            num_experts,
            num_tokens
        );

        
        //int num_selected_tokens = count_tensor.item<int>();
        int num_selected_tokens = num_tokens * 8;  // topk固定为8
        auto sorted_indices = torch::zeros({num_selected_tokens}, 
                                        torch::TensorOptions().dtype(torch::kInt32).device(device));
        //sorted_indices_optimized_kernel<4><<<num_experts, 1024, 0, stream>>>(
        block_size = std::min(std::max(num_tokens / 4, 256), 512);
        sorted_indices_optimized_kernel<4><<<num_experts, block_size, 0, stream>>>(
            routing_map_bool.data_ptr<bool>(),
            sorted_indices.data_ptr<int>(),
            expert_m.data_ptr<int>(),
            expert_m_count.data_ptr<int>(),
            num_tokens,
            num_experts
        );

        
        // Permute tokens
        auto permuted_tokens = torch::empty({num_selected_tokens, hidden_size}, tokens.options());
        int permute_grid = num_selected_tokens;
        int permute_block = std::min(hidden_size / 8, 512); // 1024 512?
       permute_tokens_kernel_vec<bf16_t, 8><<<permute_grid, permute_block, 0, stream>>>(
            (bf16_t*)tokens.data_ptr(),
            (bf16_t*)permuted_tokens.data_ptr(),
            sorted_indices.data_ptr<int>(),
            hidden_size,
            num_selected_tokens
        );


        return std::make_tuple(permuted_tokens, sorted_indices, expert_m_count);
    }

    torch::Tensor groupgemm_unpermute(
        const torch::Tensor& permuted_tokens,
        const torch::Tensor& sorted_indices,
        const std::vector<int64_t>& restore_shape,
        std::optional<torch::Tensor> probs,
        std::optional<torch::Tensor> routing_map,
        const torch::Tensor& expert_m_count
    ) {
        auto device = permuted_tokens.device();
        auto stream = at::cuda::getCurrentCUDAStream();
        
        int num_selected_tokens = permuted_tokens.size(0);
        int hidden_size = permuted_tokens.size(1);
        int num_tokens = restore_shape[0];
        
        // Create output tensor filled with zeros
        auto output_tokens = torch::zeros(restore_shape, permuted_tokens.options());
        
        // Convert sorted_indices to int32 for CUDA kernel
        auto sorted_indices_int = sorted_indices.to(torch::kInt32);
        
        dim3 block(256);
        dim3 grid(num_selected_tokens, (hidden_size + block.x - 1) / block.x);
        
        //bool use_probs = probs.defined() && probs.numel() > 0;
        bool use_probs = probs.has_value();
        
        if (use_probs) {
            // Need to permute probs first
            
            int num_experts = routing_map.value().size(1);
            auto routing_map_bool = routing_map.value().to(torch::kBool).transpose(0, 1).contiguous();
            
            auto permuted_probs = torch::empty({num_selected_tokens}, probs.value().options());
            
            dim3 prob_grid((num_experts * num_tokens + 255) / 256);
            dim3 prob_block(256);
            
    
            permute_probs_kernel_vec<bf16_t, 8><<<num_experts, 512, 0, stream>>>(
                //(bf16_t*)probs.data_ptr(),
                (bf16_t*)probs.value().data_ptr(),
                (bf16_t*)permuted_probs.data_ptr(),
                expert_m_count.data_ptr<int>(),
                routing_map_bool.data_ptr<bool>(),
                num_tokens,
                num_experts
            );
          

            unpermute_with_probs_kernel_fast<bf16_t, 4><<<num_selected_tokens, 512, 0, stream>>>(
                (bf16_t*)permuted_tokens.data_ptr(),
                (bf16_t*)output_tokens.data_ptr(),
                sorted_indices_int.data_ptr<int>(),
                (bf16_t*)permuted_probs.data_ptr(),
                hidden_size,
                num_tokens,
                num_selected_tokens
            );
         
        } else {
            // printf("unpermute_with_probs_kernel_fast null\n");
         
            unpermute_tokens_kernel_fast<bf16_t, 4><<<num_selected_tokens, 512, 0, stream>>>(
                (bf16_t*)permuted_tokens.data_ptr(),
                (bf16_t*)output_tokens.data_ptr(),
                sorted_indices_int.data_ptr<int>(),
                hidden_size,
                num_tokens,
                num_selected_tokens
            );
        }
        
        return output_tokens;
    }

}//namespace native
}//namespace at

