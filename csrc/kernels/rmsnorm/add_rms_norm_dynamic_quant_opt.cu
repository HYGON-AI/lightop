// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_addrmsnormdynamicquant_hip.hip for lightop.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>

// 1. 基础定义
#define BLOCK_SIZE 128 
#define WARP_SIZE 64

// 针对 4096 优化的特定配置
#define BLOCK_SIZE_4096 512  // 512 线程，最大化占用率
#define MAX_WARPS_4096 (BLOCK_SIZE_4096 / WARP_SIZE) // 8 Warps

#define BLOCK_SIZE_5120 512  // 512 线程，最大化占用率
#define MAX_WARPS_5120 (BLOCK_SIZE_5120 / WARP_SIZE) // 10 Warps

// 定义向量类型
struct Bf16Vec8 {
    __hip_bfloat16 val[8];
};

struct Char4Vec {
    char val[4];
};

struct Bf16Vec10 {
    __hip_bfloat16 val[10];
};

// ============================================================================
// 1. Device Helper Functions
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// ============================================================================
// 2. HIP Kernel: HS5120 (Original Logic Renamed)
// ============================================================================

__global__ void addrmsnormdynamicquant_kernel_hs5120(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    int warps_per_block = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    float rms_val = 0.0f;
    float local_scale_recip = 1.0f;

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        if (actual_max > 0.0f) local_scale_recip = actual_max / 127.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        if (i < hidden_size) {
            float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
            float val_norm = val_add * rms * norm_weight[i];
            float val_smooth = val_norm * smooth_scale[i];
            float quant_val = __builtin_roundf(val_smooth * scale);
            quant_val = fminf(quant_val, 127.0f);
            quant_val = fmaxf(quant_val, -127.0f);
            q_row[i] = static_cast<int8_t>(quant_val);
        }
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs5120_res(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_res,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    __hip_bfloat16* ar_row = after_res + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    int warps_per_block = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
            
        ar_row[i] = static_cast<at::BFloat16>(val_add);
        
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    float rms_val = 0.0f;
    float local_scale_recip = 1.0f;

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        if (actual_max > 0.0f) local_scale_recip = actual_max / 127.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        if (i < hidden_size) {
            float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
            float val_norm = val_add * rms * norm_weight[i];
            float val_smooth = val_norm * smooth_scale[i];
            float quant_val = __builtin_roundf(val_smooth * scale);
            quant_val = fminf(quant_val, 127.0f);
            quant_val = fmaxf(quant_val, -127.0f);
            q_row[i] = static_cast<int8_t>(quant_val);
        }
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs5120_norm(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_norm,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    __hip_bfloat16* an_row = after_norm + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    int warps_per_block = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    float rms_val = 0.0f;
    float local_scale_recip = 1.0f;

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        if (actual_max > 0.0f) local_scale_recip = actual_max / 127.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        if (i < hidden_size) {
            float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
            float val_norm = val_add * rms * norm_weight[i];
            an_row[i] = static_cast<at::BFloat16>(val_norm);
            float val_smooth = val_norm * smooth_scale[i];
            float quant_val = __builtin_roundf(val_smooth * scale);
            quant_val = fminf(quant_val, 127.0f);
            quant_val = fmaxf(quant_val, -127.0f);
            q_row[i] = static_cast<int8_t>(quant_val);
        }
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs5120_pro(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 5120;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    int8_t* q_row = quant_tokens + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_5120];
    __shared__ float s_warp_max[MAX_WARPS_5120];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 10 个元素
    int base_idx = tid * 10;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[10];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_5120) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec10 d_vec = *reinterpret_cast<const Bf16Vec10*>(x_row + base_idx);
    Bf16Vec10 r_vec = *reinterpret_cast<const Bf16Vec10*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 10; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_5120; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }
        
        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;
        
        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }
        
        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }
    
    __syncthreads();
    
    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用 r_val_smooth

    #pragma unroll
    for (int k = 0; k < 10; k++) {
        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        
        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs5120_pro_res(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_res,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 5120;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    int8_t* q_row = quant_tokens + row_offset;
    __hip_bfloat16* ar_row = after_res + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_5120];
    __shared__ float s_warp_max[MAX_WARPS_5120];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 10 个元素
    int base_idx = tid * 10;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[10];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_5120) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec10 d_vec = *reinterpret_cast<const Bf16Vec10*>(x_row + base_idx);
    Bf16Vec10 r_vec = *reinterpret_cast<const Bf16Vec10*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 10; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);

        ar_row[base_idx + k] = static_cast<at::BFloat16>(val_add);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_5120; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }
        
        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;
        
        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }
        
        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }
    
    __syncthreads();
    
    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用 r_val_smooth

    #pragma unroll
    for (int k = 0; k < 10; k++) {
        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        
        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs5120_pro_norm(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_norm,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 5120;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    int8_t* q_row = quant_tokens + row_offset;
    __hip_bfloat16* an_row = after_norm + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_5120];
    __shared__ float s_warp_max[MAX_WARPS_5120];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 10 个元素
    int base_idx = tid * 10;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[10];
    float r_val_norm[10];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_5120) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec10 d_vec = *reinterpret_cast<const Bf16Vec10*>(x_row + base_idx);
    Bf16Vec10 r_vec = *reinterpret_cast<const Bf16Vec10*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 10; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        r_val_norm[k] = val_norm; // 存入寄存器，待 RMS 算出后再乘以 rms 写回 after_norm

        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_5120; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }
        
        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;
        
        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }
        
        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }
    
    __syncthreads();
    
    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用 r_val_smooth

    #pragma unroll
    for (int k = 0; k < 10; k++) {
        an_row[base_idx + k] = static_cast<at::BFloat16>(r_val_norm[k] * rms);

        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);

        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

// ============================================================================
// 3. HIP Kernel: HS4096 (Single-Pass Register Caching)
// ============================================================================
// 核心优化：
// 1. 单遍扫描：利用寄存器数组缓存加载的数据，算完 RMS 后直接使用，不再读内存。
// 2. 向量化写入：使用 char4 一次性写入 4 个 int8 结果。

__global__ void addrmsnormdynamicquant_kernel_hs4096(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 4096;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    int8_t* q_row = quant_tokens + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_4096];
    __shared__ float s_warp_max[MAX_WARPS_4096];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 8 个元素
    int base_idx = tid * 8;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[8];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_4096) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec8 d_vec = *reinterpret_cast<const Bf16Vec8*>(x_row + base_idx);
    Bf16Vec8 r_vec = *reinterpret_cast<const Bf16Vec8*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_4096; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }
        
        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;
        
        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }
        
        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }
    
    __syncthreads();
    
    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用 r_val_smooth

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        
        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs4096_res(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_res,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 4096;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    __hip_bfloat16* ar_row = after_res + row_offset;
    int8_t* q_row = quant_tokens + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_4096];
    __shared__ float s_warp_max[MAX_WARPS_4096];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 8 个元素
    int base_idx = tid * 8;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[8];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_4096) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec8 d_vec = *reinterpret_cast<const Bf16Vec8*>(x_row + base_idx);
    Bf16Vec8 r_vec = *reinterpret_cast<const Bf16Vec8*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);
        
        ar_row[base_idx + k] = static_cast<at::BFloat16>(val_add);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_4096; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }
        
        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;
        
        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }
        
        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }
    
    __syncthreads();
    
    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用r_val_smooth

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        
        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_hs4096_norm(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_norm,
    int64_t num_tokens,
    float eps
) {
    const int64_t hidden_size = 4096;
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;

    int64_t row_offset = row_idx * hidden_size;
    const __hip_bfloat16* x_row = hidden_states + row_offset;
    const __hip_bfloat16* res_row = residual + row_offset;
    __hip_bfloat16* an_row = after_norm + row_offset;
    int8_t* q_row = quant_tokens + row_offset;

    // 静态共享内存
    __shared__ float s_warp_sum[MAX_WARPS_4096];
    __shared__ float s_warp_max[MAX_WARPS_4096];
    __shared__ float s_final_rms;
    __shared__ float s_final_scale;

    // 每个线程处理 8 个元素
    int base_idx = tid * 8;

    // --- 寄存器缓存区 (关键优化) ---
    // 存储转换后的 float 数据，避免二次转换和二次读取
    float r_val_smooth[8];
    float r_val_norm[8];

    // --- 阶段 1: 加载与统计 ---
    /*
    if (tid < MAX_WARPS_4096) {
        s_warp_sum[tid] = 0.0f;
        s_warp_max[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    // 1. 向量化加载 (128-bit)
    Bf16Vec8 d_vec = *reinterpret_cast<const Bf16Vec8*>(x_row + base_idx);
    Bf16Vec8 r_vec = *reinterpret_cast<const Bf16Vec8*>(res_row + base_idx);

    // 2. 数据处理与统计累加
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float val_add = static_cast<float>(d_vec.val[k]) + static_cast<float>(r_vec.val[k]);

        local_sum_sq += val_add * val_add;

        float val_norm = val_add * norm_weight[base_idx + k];
        r_val_norm[k] = val_norm; // 存入寄存器，待 RMS 算出后再乘以 rms 写回 after_norm

        float val_smooth = val_norm * smooth_scale[base_idx + k];
        r_val_smooth[k] = val_smooth; // 存入寄存器

        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    // 3. Warp 归约
    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // --- 阶段 2: 计算 Scale ---
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float total_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_WARPS_4096; ++i) {
            total_sum_sq += s_warp_sum[i];
            total_max = fmaxf(total_max, s_warp_max[i]);
        }

        float rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = total_max * rms;

        float scale_val = 1.0f;
        if (actual_max > 0.0f) {
            scale_val = actual_max / 127.0f;
        }

        per_token_scale[row_idx] = scale_val;
        s_final_rms = rms;
        s_final_scale = scale_val;
    }

    __syncthreads();

    float rms = s_final_rms;
    float scale = 1.0f / s_final_scale;

    // --- 阶段 3: 量化写入 (使用寄存器缓存数据) ---
    // 注意：这里不再读取 global memory，直接使用r_val_smooth

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        an_row[base_idx + k] = static_cast<at::BFloat16>(r_val_norm[k] * rms);

        float quant_val = __builtin_roundf(r_val_smooth[k] * rms * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);

        q_row[base_idx+k] = static_cast<int8_t>(quant_val);
    }
}

// ============================================================================
// 4. HIP Kernel: Generic (Fallback)
// ============================================================================

__global__ void addrmsnormdynamicquant_kernel_generic(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;
    
    int warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        float rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        float local_scale_recip = (actual_max > 0.0f) ? (actual_max / 127.0f) : 1.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        float val_norm = val_add * rms * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        float quant_val = __builtin_roundf(val_smooth * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        q_row[i] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_generic_res(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_res,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;
    
    int warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    __hip_bfloat16* ar_row = after_res + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        ar_row[i] = static_cast<at::BFloat16>(val_add);
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        float rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        float local_scale_recip = (actual_max > 0.0f) ? (actual_max / 127.0f) : 1.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        float val_norm = val_add * rms * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        float quant_val = __builtin_roundf(val_smooth * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        q_row[i] = static_cast<int8_t>(quant_val);
    }
}

__global__ void addrmsnormdynamicquant_kernel_generic_norm(
    const __hip_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ norm_weight,
    const float* __restrict__ smooth_scale,
    const __hip_bfloat16* __restrict__ residual,
    int8_t* __restrict__ quant_tokens,
    float* __restrict__ per_token_scale,
    __hip_bfloat16* __restrict__ after_norm,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    if (row_idx >= num_tokens) return;

    int64_t tid = threadIdx.x;
    int64_t warp_id = tid / WARP_SIZE;
    int64_t lane_id = tid % WARP_SIZE;
    
    int warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const __hip_bfloat16* x_row = hidden_states + row_idx * hidden_size;
    const __hip_bfloat16* res_row = residual + row_idx * hidden_size;
    __hip_bfloat16* an_row = after_norm + row_idx * hidden_size;
    int8_t* q_row = quant_tokens + row_idx * hidden_size;

    extern __shared__ float s_mem[];
    float* s_block_sum_sq = s_mem;
    float* s_block_max_abs = s_mem + warps_per_block;
    float* s_rms = s_mem + 2 * warps_per_block;
    float* s_scale_recip = s_mem + 2 * warps_per_block + 1;

    /*
    if (tid < warps_per_block) {
        s_block_sum_sq[tid] = 0.0f;
        s_block_max_abs[tid] = 0.0f;
    }
    __syncthreads();
    */

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        local_sum_sq += val_add * val_add;
        float val_norm = val_add * norm_weight[i];
        float val_smooth = val_norm * smooth_scale[i];
        local_max_abs = fmaxf(local_max_abs, fabsf(val_smooth));
    }

    float warp_sum = warp_reduce_sum(local_sum_sq);
    float warp_max = warp_reduce_max(local_max_abs);

    if (lane_id == 0) {
        s_block_sum_sq[warp_id] = warp_sum;
        s_block_max_abs[warp_id] = warp_max;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum_sq = 0.0f;
        float block_max = 0.0f;
        for(int i = 0; i < warps_per_block; ++i) {
            total_sum_sq += s_block_sum_sq[i];
            block_max = fmaxf(block_max, s_block_max_abs[i]);
        }
        float rms_val = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + eps);
        float actual_max = block_max * rms_val;
        float local_scale_recip = (actual_max > 0.0f) ? (actual_max / 127.0f) : 1.0f;
        
        per_token_scale[row_idx] = local_scale_recip;
        *s_rms = rms_val;
        *s_scale_recip = local_scale_recip;
    }
    
    __syncthreads();
    
    float rms = *s_rms;
    float scale = 1.0f / (*s_scale_recip);

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        float val_add = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        float val_norm = val_add * rms * norm_weight[i];
        an_row[i] = static_cast<at::BFloat16>(val_norm);
        float val_smooth = val_norm * smooth_scale[i];
        float quant_val = __builtin_roundf(val_smooth * scale);
        quant_val = fminf(quant_val, 127.0f);
        quant_val = fmaxf(quant_val, -127.0f);
        q_row[i] = static_cast<int8_t>(quant_val);
    }
}

// ============================================================================
// 5. Host Dispatcher
// ============================================================================
std::vector<torch::Tensor> addrmsnormdynamicquant_forward_hip(
    torch::Tensor& hidden_states,
    torch::Tensor& norm_weight,
    torch::Tensor& smooth_scale,
    torch::Tensor& residual,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    torch::Tensor& after_res,
    torch::Tensor& after_norm,
    int64_t output_mode,
    double eps
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t num_tokens = hidden_states.size(0);
    int64_t hidden_size = hidden_states.size(1);

    const __hip_bfloat16* data_ptr = reinterpret_cast<const __hip_bfloat16*>(hidden_states.data_ptr<at::BFloat16>());
    const float* weight_ptr = reinterpret_cast<const float*>(norm_weight.data_ptr<float>());
    const float* scale_ptr = reinterpret_cast<const float*>(smooth_scale.data_ptr<float>());
    const __hip_bfloat16* residual_ptr = reinterpret_cast<const __hip_bfloat16*>(residual.data_ptr<at::BFloat16>());

    int8_t * qt_ptr = reinterpret_cast<int8_t*>(quant_tokens.data_ptr<int8_t>());
    float * pts_ptr = reinterpret_cast<float*>(per_token_scale.data_ptr<float>());
    
    dim3 grid(static_cast<unsigned int>(num_tokens));
    
    if (hidden_size == 5120) {
        dim3 block(BLOCK_SIZE);
        int warps_per_block = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        int shared_mem_size = (2 * warps_per_block + 2) * sizeof(float);

        if(output_mode == 1)
        {
            /*
            __hip_bfloat16* after_res_ptr = reinterpret_cast<__hip_bfloat16*>(after_res.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs5120_res<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_res_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
            */
            dim3 block(BLOCK_SIZE_5120);
            int warps_per_block = (BLOCK_SIZE_5120 + WARP_SIZE - 1) / WARP_SIZE;
            int shared_mem_size = (2 * warps_per_block + 2) * sizeof(float);
            
            __hip_bfloat16* after_res_ptr = reinterpret_cast<__hip_bfloat16*>(after_res.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs5120_pro_res<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_res_ptr, num_tokens, static_cast<float>(eps)
            );
 
        }
        else if(output_mode ==2)
        {
            /*
            __hip_bfloat16* after_norm_ptr = reinterpret_cast<__hip_bfloat16*>(after_norm.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs5120_norm<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_norm_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
            */
            dim3 block(BLOCK_SIZE_5120);
            int warps_per_block = (BLOCK_SIZE_5120 + WARP_SIZE - 1) / WARP_SIZE;
            int shared_mem_size = (2 * warps_per_block + 2) * sizeof(float);
            
            __hip_bfloat16* after_norm_ptr = reinterpret_cast<__hip_bfloat16*>(after_norm.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs5120_pro_norm<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_norm_ptr, num_tokens, static_cast<float>(eps)
            );
 
        }
        else
        {
            addrmsnormdynamicquant_kernel_hs5120<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
            /*
            dim3 block(BLOCK_SIZE_5120);
            int warps_per_block = (BLOCK_SIZE_5120 + WARP_SIZE - 1) / WARP_SIZE;
            int shared_mem_size = (2 * warps_per_block + 2) * sizeof(float);
 
            addrmsnormdynamicquant_kernel_hs5120_pro<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, num_tokens, static_cast<float>(eps)
            );
            */
        }
    } 
    else if (hidden_size == 4096) {
        dim3 block(BLOCK_SIZE_4096);
        
        if(output_mode == 1)
        {
            __hip_bfloat16* after_res_ptr = reinterpret_cast<__hip_bfloat16*>(after_res.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs4096_res<<<grid, block, 0, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_res_ptr, num_tokens, static_cast<float>(eps)
            );
        }
        else if(output_mode == 2)
        {
            __hip_bfloat16* after_norm_ptr = reinterpret_cast<__hip_bfloat16*>(after_norm.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_hs4096_norm<<<grid, block, 0, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_norm_ptr, num_tokens, static_cast<float>(eps)
            );
        }
        else
        {
            addrmsnormdynamicquant_kernel_hs4096<<<grid, block, 0, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, num_tokens, static_cast<float>(eps)
            );
        }
    } 
    else {
        dim3 block(BLOCK_SIZE);
        int warps_per_block = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        int shared_mem_size = (2 * warps_per_block + 2) * sizeof(float);

        if(output_mode == 1)
        {
            __hip_bfloat16* after_res_ptr = reinterpret_cast<__hip_bfloat16*>(after_res.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_generic_res<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_res_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
        }
        else if(output_mode == 2)
        {
            __hip_bfloat16* after_norm_ptr = reinterpret_cast<__hip_bfloat16*>(after_norm.data_ptr<at::BFloat16>());
            addrmsnormdynamicquant_kernel_generic_norm<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, after_norm_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
        }
        else
        {
            addrmsnormdynamicquant_kernel_generic<<<grid, block, shared_mem_size, stream>>>(
                data_ptr, weight_ptr, scale_ptr, residual_ptr,
                qt_ptr, pts_ptr, num_tokens, hidden_size, static_cast<float>(eps)
            );
        }
    }

    if(output_mode == 1) return {quant_tokens, per_token_scale, after_res};
    else if(output_mode == 2) return {quant_tokens, per_token_scale, after_norm};
    else return {quant_tokens, per_token_scale};
}


namespace at {
namespace native {

std::vector<torch::Tensor> add_rms_norm_dynamic_quant(
    torch::Tensor& hidden_states,
    torch::Tensor& norm_weight,
    torch::Tensor& smooth_scale,
    torch::Tensor& residual,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    torch::Tensor& after_res,
    torch::Tensor& after_norm,
    int64_t output_mode,
    double eps) {
  TORCH_CHECK(hidden_states.is_cuda(), "add_rms_norm_dynamic_quant: CUDA/HIP required");
  TORCH_CHECK(hidden_states.dim() == 2, "add_rms_norm_dynamic_quant: hidden_states must be 2D");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kBFloat16,
      "add_rms_norm_dynamic_quant: only bfloat16 activations supported");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "add_rms_norm_dynamic_quant: residual must be bf16");
  TORCH_CHECK(norm_weight.scalar_type() == at::kFloat, "add_rms_norm_dynamic_quant: weight must be float32");
  TORCH_CHECK(smooth_scale.scalar_type() == at::kFloat, "add_rms_norm_dynamic_quant: smooth_scale must be float32");
  TORCH_CHECK(quant_tokens.scalar_type() == at::kChar, "add_rms_norm_dynamic_quant: quant_tokens must be int8");
  TORCH_CHECK(per_token_scale.scalar_type() == at::kFloat, "add_rms_norm_dynamic_quant: per_token_scale must be float32");
  TORCH_CHECK(output_mode >= 0 && output_mode <= 2, "add_rms_norm_dynamic_quant: output_mode must be 0|1|2");
  return addrmsnormdynamicquant_forward_hip(
      hidden_states,
      norm_weight,
      smooth_scale,
      residual,
      quant_tokens,
      per_token_scale,
      after_res,
      after_norm,
      output_mode,
      eps);
}

}  // namespace native
}  // namespace at
