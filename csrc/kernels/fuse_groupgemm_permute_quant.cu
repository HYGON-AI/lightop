// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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
    // if(threadIdx.x == 0) {
    //     printf("cur_val: %f, x_val: %f\n", cur_val.x, cur_val.y);
    // }
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

    static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

    using scalar_t = __hip_bfloat16;

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

template<typename T, int VEC, int block_size, int num>
__global__ void __launch_bounds__(1024) fuse_permute_tokens_quant_kernel_vec(
    const T* input_tokens,
    int8_t* output_tokens,
    float* scales,
    const int* sorted_indices,
    const int hidden_size,
    const int num_selected_tokens
) {
    constexpr int THREAD_WORK_SIZE = block_size * VEC;
    constexpr int NUM_LOOPS = (num + THREAD_WORK_SIZE - 1) / THREAD_WORK_SIZE;

    __shared__ float val_shared[block_size / C10_WARP_SIZE];
    __shared__ float s_block_wide_max;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= num_selected_tokens) {
        return;
    }
    const int source_row_offset = sorted_indices[token_idx] * hidden_size;
    const int dest_row_offset = token_idx * hidden_size;

    using VecType = at::native::memory::aligned_vector<T, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    
    T loaded_data[NUM_LOOPS][VEC];
    float thread_local_max = 0.0f;
    int loop_iter = 0;

    for (int i = tid * VEC; i < hidden_size; i += THREAD_WORK_SIZE) {
        const VecType loaded_vec = *reinterpret_cast<const VecType*>(input_tokens + source_row_offset + i);
        *reinterpret_cast<VecType*>(loaded_data[loop_iter]) = loaded_vec;

        #pragma unroll
        for (int ii = 0; ii < VEC; ++ii) {
            thread_local_max = fmaxf(thread_local_max, fabsf(static_cast<float>(loaded_data[loop_iter][ii])));
        }
        loop_iter++;
    }

    float row_max = BlockReduceMax_ROW<float, block_size>(thread_local_max, val_shared);
    
    if (tid == 0) {
        scales[token_idx] = row_max / 127.0f;
        s_block_wide_max = row_max;
    }

    __syncthreads(); 

    const float inv_s = (s_block_wide_max == 0.0f) ? 0.0f : 127.0f / s_block_wide_max;
    
    
    for (int iter = 0; iter < NUM_LOOPS; ++iter) {
        const int i = (iter * block_size + tid) * VEC;
        if (i >= hidden_size) {
            continue;
        }
        int8_t out_vec[VEC];
        #pragma unroll
        for (int ii = 0; ii < VEC; ++ii) {
            out_vec[ii] = float_to_int8_rn(static_cast<float>(loaded_data[iter][ii]) * inv_s);
        }

        *reinterpret_cast<VecInt8Type*>(output_tokens + dest_row_offset + i) = *reinterpret_cast<VecInt8Type*>(out_vec);
    }
}

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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fuse_groupgemm_permute_quant(
        const torch::Tensor& tokens,
        const torch::Tensor& routing_map,
        torch::Tensor& scales
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
        // auto permuted_tokens = torch::empty({num_selected_tokens, hidden_size}, tokens.options());
        auto permuted_tokens = torch::empty({num_selected_tokens, hidden_size}, 
                                            tokens.options().dtype(torch::kInt8));
        int permute_grid = num_selected_tokens;
        int permute_block = std::min(hidden_size / 8, 512); // 1024 512?
        if(hidden_size<=4096){
            fuse_permute_tokens_quant_kernel_vec<scalar_t, 8, 512, 4096><<<permute_grid, permute_block, 0, stream>>>(
                (scalar_t*)tokens.data_ptr(),
                (int8_t*)permuted_tokens.data_ptr(),
                (float*)scales.data_ptr(),
                sorted_indices.data_ptr<int>(),
                hidden_size,
                num_selected_tokens
            );
        }else if(hidden_size<=8192){
            fuse_permute_tokens_quant_kernel_vec<scalar_t, 8, 512, 8192><<<permute_grid, permute_block, 0, stream>>>(
                (scalar_t*)tokens.data_ptr(),
                (int8_t*)permuted_tokens.data_ptr(),
                (float*)scales.data_ptr(),
                sorted_indices.data_ptr<int>(),
                hidden_size,
                num_selected_tokens
            );
        }else if(hidden_size<=16384){
            fuse_permute_tokens_quant_kernel_vec<scalar_t, 8, 512, 16384><<<permute_grid, permute_block, 0, stream>>>(
                (scalar_t*)tokens.data_ptr(),
                (int8_t*)permuted_tokens.data_ptr(),
                (float*)scales.data_ptr(),
                sorted_indices.data_ptr<int>(),
                hidden_size,
                num_selected_tokens
            );
        }else{
            fuse_permute_tokens_quant_kernel_vec<scalar_t, 8, 512, 32768><<<permute_grid, permute_block, 0, stream>>>(
                (scalar_t*)tokens.data_ptr(),
                (int8_t*)permuted_tokens.data_ptr(),
                (float*)scales.data_ptr(),
                sorted_indices.data_ptr<int>(),
                hidden_size,
                num_selected_tokens
            );
        }
        return std::make_tuple(permuted_tokens, scales, sorted_indices, expert_m_count);
    }


}//namespace native
}//namespace at

