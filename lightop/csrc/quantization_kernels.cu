#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>

#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"

#include "reduce.h"

namespace at {
namespace native {
    using bf16_t = __hip_bfloat16;

    static __device__ inline uint16_t f32_to_bf16(float f){
  
        uint32_t u = *(uint32_t*)(&f);
        u += 0x7fff + ((u >> 16) & 1); 
        return u>>16;
    
    }

    // Warp级别的reduce操作
    template <typename T>
    __inline__ __device__ T warp_reduce_max(T val) {
        for (int offset = C10_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            T other = WARP_SHFL_DOWN(val, offset);
            val = (val > other) ? val : other;
        }
        return val;
    }

    template <typename T>
    __inline__ __device__ T warp_reduce_min(T val) {
        for (int offset = C10_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            T other = WARP_SHFL_DOWN(val, offset);
            val = (val < other) ? val : other;
        }
        return val;
    }

    // 前向量化kernel：bf16 -> int8 (非对称量化) 适用于bs<=1024
    template<int VEC = 8, int HEADN = 10>
    __global__ void __launch_bounds__(1024) quantize_int8_small_kernel(
        const bf16_t* __restrict__ input,          // [bs_seq, head_num, head_dim]
        int8_t* __restrict__ output,               // [bs_seq, head_num, head_dim]
        bf16_t* __restrict__ scale_zero,           // [bs_seq, head_num, 2]
        const int bs_seq,
        const int head_num,
        const int head_dim
    ) {
        const int batch_idx = blockIdx.x;
        int tid = threadIdx.x;
        int warp_id = tid / C10_WARP_SIZE;
        int lane_id = tid % C10_WARP_SIZE;
    
        const bf16_t* input_ptr = input + batch_idx * head_num * head_dim + warp_id * head_dim;
        int8_t* output_ptr = output + batch_idx * head_num * head_dim + warp_id * head_dim;
        bf16_t* scale_zero_ptr = scale_zero + batch_idx * head_num * 2 + warp_id * 2;
        
        float input_val = float(input_ptr[lane_id]);
        float local_max = input_val;
        float local_min = input_val;
  
        // Warp内reduce 找最大和最小值
        local_max = warp_reduce_max(local_max);
        local_min = warp_reduce_min(local_min);

        // 只有线程0拿到最大和最小值
        local_max = __shfl_sync(0xffffffffffffffff, local_max, 0);
        local_min = __shfl_sync(0xffffffffffffffff, local_min, 0);


        float scale = (local_max - local_min) / 255.0f;
        scale = fmaxf(scale, 1e-8f);  // 防止除0
        int zero = int(-local_min / scale);
        float q_tmp_data = int(input_val / scale) + zero; //待验证 是否需要中转int32舍去尾数
        q_tmp_data = fminf(fmaxf(q_tmp_data, 0.0f), 255.0f);
        int8_t q_data = static_cast<int8_t>(static_cast<int>(q_tmp_data) - 128);


        //写回数据
        output_ptr[lane_id] = q_data;
        if(lane_id == 0){
            scale_zero_ptr[0] = f32_to_bf16(scale);
            scale_zero_ptr[1] = f32_to_bf16(zero);
        }
        
    }

    // 针对大shape m>1024
    template<int WARP_SIZE = 4, int VEC = 8>
    __global__ void __launch_bounds__(1024) quantize_int8_large_kernel(
        const bf16_t* __restrict__ input,          // [bs_seq, head_num, head_dim]
        int8_t* __restrict__ output,               // [bs_seq, head_num, head_dim]
        bf16_t* __restrict__ scale_zero,           // [bs_seq, head_num, 2]
        const int bs_seq,
        const int head_num,
        const int bs_head,                         // bs_head = bs_seq*head_num
        const int head_dim
    ) {
        //using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
        using bf16_8 = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
        using int8_8 = __attribute__((__vector_size__(VEC * sizeof(int8_t)))) int8_t;
        const int bidx = blockIdx.x;
        int tid = threadIdx.x;
        int warp_id = tid / C10_WARP_SIZE; 
        int lane_id = tid % C10_WARP_SIZE;
        constexpr int thread_dim = 64/VEC;  //64/vec=8  64个元素在8个线程里面
        constexpr int WARP_DIM = WARP_SIZE * 64;
        int dim_id = tid / thread_dim;
        int dim_lane_id = tid % thread_dim;
        
        const bf16_t* input_ptr = input + bidx * WARP_DIM * VEC + tid * VEC;
        int8_t* output_ptr = output + bidx * WARP_DIM * VEC + tid * VEC;
        bf16_t* scale_zero_ptr = scale_zero +  bidx * WARP_DIM * VEC / 64 * 2 + (tid * VEC / 64) * 2;
        __shared__ float s_max[WARP_SIZE*64/thread_dim][thread_dim + 2];
        __shared__ float s_min[WARP_SIZE*64/thread_dim][thread_dim + 2];
        const bf16_8 input_vec = *reinterpret_cast<const bf16_8*>(input_ptr);
        const bf16_t* input_vec8 = reinterpret_cast<const bf16_t*>(&input_vec);
        int8_8 output_vec8;

        bf16_t local_max = input_vec8[0];
        bf16_t local_min = input_vec8[0];
        for(int i = 1; i < 8; i++){
            local_max = local_max > input_vec8[i] ? local_max : input_vec8[i];
            local_min = local_min < input_vec8[i] ? local_min : input_vec8[i];
        }
        
        // 拷贝到lds
        s_max[dim_id][dim_lane_id+2] = float(local_max);
        s_min[dim_id][dim_lane_id+2] = float(local_min);

        __syncthreads();

        // 计算head_dim维度最大和最小值
        if(dim_lane_id == 0){
            s_max[dim_id][0] = s_max[dim_id][2];
            s_min[dim_id][1] = s_min[dim_id][2];
            for(int i = 1; i < thread_dim; i++){
                s_max[dim_id][0] = s_max[dim_id][i+2] > s_max[dim_id][0] ? s_max[dim_id][i+2] : s_max[dim_id][0];
                s_min[dim_id][1] = s_min[dim_id][i+2] < s_min[dim_id][1] ? s_min[dim_id][i+2] : s_min[dim_id][1];
            }
        }
        __syncthreads();

        float val_max = s_max[dim_id][0];
        float val_min = s_min[dim_id][1];
        float scale = (val_max - val_min) / 255.0f;
        scale = fmaxf(scale, 1e-8f);  // 防止除0
        int zero = int(-val_min / scale);

        #pragma unroll
        for(int i = 0; i < VEC; i++){
            float q_tmp_data = float(input_vec8[i]);
            q_tmp_data = int(q_tmp_data / scale) + zero; //待验证 是否需要中转int32舍去尾数
            q_tmp_data = fminf(fmaxf(q_tmp_data, 0.0f), 255.0f);
            output_vec8[i] = static_cast<int8_t>(static_cast<int>(q_tmp_data) - 128);
        }
        *reinterpret_cast<int8_8*>(output_ptr) = output_vec8;

        if(dim_lane_id == 0){
            scale_zero_ptr[0] = f32_to_bf16(scale);
            scale_zero_ptr[1] = f32_to_bf16(zero);
        }
    }

    // 反向反量化kernel：int8 -> bf16 small
    template<int VEC = 8>
    __global__ void __launch_bounds__(1024) dequantize_int8_kernel(
        const int8_t* __restrict__ input,          // [bs_seq, head_num, head_dim]
        const bf16_t* __restrict__ scale_zero,     // [bs_seq, head_num, 2]
        bf16_t* __restrict__ output,               // [bs_seq, head_num, head_dim]
        const int bs_seq,
        const int head_num,
        const int head_dim
    ) {
        const int batch_idx = blockIdx.x;
        int tid = threadIdx.x;
        int warp_id = tid / C10_WARP_SIZE;
        int lane_id = tid % C10_WARP_SIZE;

        const int8_t* input_ptr = input + batch_idx * head_num * head_dim + warp_id * head_dim;
        bf16_t* output_ptr = output + batch_idx * head_num * head_dim + warp_id * head_dim;
        const bf16_t* scale_zero_ptr = scale_zero + batch_idx * head_num * 2 + warp_id * 2;

        int8_t input_data = input_ptr[lane_id];
        bf16_t scale = scale_zero_ptr[0];
        bf16_t zero = scale_zero_ptr[1];
        float tmp_out = (int(input_data) + 128 - int(zero)) * float(scale);
        bf16_t outptut_data = f32_to_bf16(tmp_out);

        output_ptr[lane_id] = outptut_data;
    }

    template<int WARP_SIZE = 4, int VEC = 8>
    __global__ void __launch_bounds__(1024) dequantize_int8_large_kernel(
        const int8_t* __restrict__ input,          // [bs_seq, head_num, head_dim]
        const bf16_t* __restrict__ scale_zero,     // [bs_seq, head_num, 2]
        bf16_t* __restrict__ output,               // [bs_seq, head_num, head_dim]
        const int bs_seq,
        const int head_num,
        const int bs_head,
        const int head_dim
    ) {
        using bf16_8 = __attribute__((__vector_size__(VEC * sizeof(uint16_t)))) unsigned short;
        using int8_8 = __attribute__((__vector_size__(VEC * sizeof(int8_t)))) int8_t;
        const int bidx = blockIdx.x;
        int tid = threadIdx.x;
        int warp_id = tid / C10_WARP_SIZE; 
        int lane_id = tid % C10_WARP_SIZE;
        constexpr int thread_dim = 64/VEC;  //64/vec=8  64个元素在4个线程里面
        constexpr int WARP_DIM = WARP_SIZE * 64;
        int dim_id = tid / thread_dim;
        int dim_lane_id = tid % thread_dim;
        
        const int8_t* input_ptr = input + bidx * WARP_DIM * VEC + tid * VEC;
        bf16_t* output_ptr = output + bidx * WARP_DIM * VEC + tid * VEC;
        const bf16_t* scale_zero_ptr = scale_zero +  bidx * WARP_DIM * VEC / 64 * 2 + (tid * VEC / 64) * 2;

        const int8_8 input_vec8 = *reinterpret_cast<const int8_8*>(input_ptr);
        bf16_8 output_vec8;
        const bf16_t scale = scale_zero_ptr[0];
        const bf16_t zero = scale_zero_ptr[1];
        #pragma unroll
        for(int i = 0; i < VEC; i++){
            float tmp_out = (int(input_vec8[i]) + 128 - int(zero)) * float(scale);
            output_vec8[i] = f32_to_bf16(tmp_out);
        }
        *reinterpret_cast<bf16_8*>(output_ptr) = output_vec8;
////////////////////////////////////////////////////////////////////////////////////////
        // const int batch_idx = blockIdx.x;
        // int tid = threadIdx.x;
        // int warp_id = tid / C10_WARP_SIZE;
        // int lane_id = tid % C10_WARP_SIZE;

        // const int8_t* input_ptr = input + batch_idx * head_num * head_dim + warp_id * head_dim;
        // bf16_t* output_ptr = output + batch_idx * head_num * head_dim + warp_id * head_dim;
        // const bf16_t* scale_zero_ptr = scale_zero + batch_idx * head_num * 2 + warp_id * 2;

        // int8_t input_data = input_ptr[lane_id];
        // bf16_t scale = scale_zero_ptr[0];
        // bf16_t zero = scale_zero_ptr[1];
        // float tmp_out = (int(input_data) + 128 - int(zero)) * float(scale);
        // bf16_t outptut_data = f32_to_bf16(tmp_out);

        // output_ptr[lane_id] = outptut_data;
    }

    // Python绑定：前向量化
    void destindex_copy_quantize_int8_cuda(
        const torch::Tensor& input,           // [bs_seq, head_num, head_dim], bf16
        torch::Tensor& scale_zero,             // [bs_seq, head_num, 2], bf16
        torch::Tensor& output                // [bs_seq, head_num, head_dim], int8
    ) {
        auto device = input.device();
        auto stream = at::cuda::getCurrentCUDAStream();

        int bs_seq = input.size(0);
        int head_num = input.size(1);
        int head_dim = input.size(2);
        int bs_head = bs_seq * head_num;

        // dim3 grid(bs_seq);
        // dim3 block(head_num*64);
        // quantize_int8_asym_kernel<8><<<grid, block, 0, stream>>>(
        //     reinterpret_cast<const bf16_t*>(input.data_ptr()),
        //     output.data_ptr<int8_t>(),
        //     reinterpret_cast<bf16_t*>(scale_zero.data_ptr()),
        //     bs_seq,
        //     head_num,
        //     head_dim
        // );

        constexpr int WARP_SIZE = 4;
        constexpr int VEC = 8;
        int grid_num = (bs_head*head_dim + WARP_SIZE * 64 * VEC - 1) / (WARP_SIZE * 64*VEC);
        dim3 grid(grid_num);
        dim3 block(WARP_SIZE * 64);
        quantize_int8_large_kernel<WARP_SIZE, VEC><<<grid, block, 0, stream>>>(
            reinterpret_cast<const bf16_t*>(input.data_ptr()),
            output.data_ptr<int8_t>(),
            reinterpret_cast<bf16_t*>(scale_zero.data_ptr()),
            bs_seq,
            head_num,
            bs_head,
            head_dim
        );

    }

    // Python绑定：反向反量化
    void destindex_dequantize_int8_cuda(
        const torch::Tensor& input,           // [bs_seq, head_num, head_dim], int8
        const torch::Tensor& scale_zero,      // [bs_seq, head_num, 2], bf16
        torch::Tensor& output                 // [bs_seq, head_num, head_dim], bf16
    ) {
        auto device = input.device();
        auto stream = at::cuda::getCurrentCUDAStream();

        int bs_seq = input.size(0);
        int head_num = input.size(1);
        int head_dim = input.size(2);
        int bs_head = bs_seq * head_num;

        // dim3 grid(bs_seq);
        // dim3 block(head_num*64);
        // dequantize_int8_kernel<8><<<grid, block, 0, stream>>>(
        //     input.data_ptr<int8_t>(),
        //     reinterpret_cast<const bf16_t*>(scale_zero.data_ptr()),
        //     reinterpret_cast<bf16_t*>(output.data_ptr()),
        //     bs_seq,
        //     head_num,
        //     head_dim
        // );

        constexpr int WARP_SIZE = 4;
        constexpr int VEC = 8;
        int grid_num = (bs_head*head_dim + WARP_SIZE * 64 * VEC - 1) / (WARP_SIZE * 64 * VEC);
        dim3 grid(grid_num);
        dim3 block(WARP_SIZE * 64);
        dequantize_int8_large_kernel<WARP_SIZE, VEC><<<grid, block, 0, stream>>>(
            input.data_ptr<int8_t>(),
            reinterpret_cast<const bf16_t*>(scale_zero.data_ptr()),
            reinterpret_cast<bf16_t*>(output.data_ptr()),
            bs_seq,
            head_num,
            bs_head,
            head_dim
        );
    }

} // namespace native
} // namespace at

