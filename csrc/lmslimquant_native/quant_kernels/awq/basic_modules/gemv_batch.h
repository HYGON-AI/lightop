#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdint.h>

#include <hip/hip_fp16.h>
#include "gemm_utils.h"

#pragma once

namespace gemv_batch {

#define MAX_THREADS_PER_BLOCK 1024
#define GemvPerThreadCnt 16 // 每个线程处理的元素（int8/int4）个数，为16的倍数(至少是VEC_L的2倍，隐藏访存和计算形成流水)，需要被k整除，m为1时无效
#define VEC_L 8             // 1个向量处理8个int8/int4
#define READ_CNT (GemvPerThreadCnt / VEC_L)

// int4
template <int Batch, int NPerBlock, int SplitKBatch>
__global__ void weight_only_int4_gemv_groupwise(const ck::half8_t* input,
                                                const uint32_t* weight_int4,
                                                const uint32_t* zero_scales,
                                                ck::half_t* output,
                                                int m,
                                                int n,
                                                int k,
                                                int StrideA,
                                                int StrideB,
                                                int StrideC,
                                                int GroupSize,
                                                int k_iteration) {
    float sum;
    float block_sum       = 0;
    const int K_Cnt       = k / VEC_L;
    const int BStride_Cnt = StrideB / VEC_L;
    const int BGroup_Cnt  = StrideB / GroupSize;

    const int laneId = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;

    int tid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    static __shared__ float warpLevelSums[NPerBlock][SplitKBatch];

    for(int iteration = 0; iteration < k_iteration; iteration++) {
        int gidx   = tid + iteration * blockDim.x;
        int gdatax = READ_CNT * gidx;

        uint32_t zero_scale = zero_scales[row * BGroup_Cnt + gidx * GemvPerThreadCnt / GroupSize];
        // auto p_zero_scale = reinterpret_cast<ck::half_t*>(&zero_scale);

        sum = 0;
        for(int it = 0; it < READ_CNT / 2; it++) {
            if(gdatax + 2 * it + 1 < K_Cnt) { // 8
                ck::half8_t vec_val_1 = input[gdatax + 2 * it];
                ck::half8_t mat_h_1   = weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it], zero_scale);
                ck::half8_t vec_val_2 = input[gdatax + 2 * it + 1];
                ck::half8_t mat_h_2   = weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it + 1], zero_scale);
                weight_only_tools::amd_assembly_product_1x4(vec_val_1, mat_h_1, sum);
                weight_only_tools::amd_assembly_product_1x4(vec_val_2, mat_h_2, sum);
            }
        }

        // sum *= __half2float(p_zero_scale[1]);
        sum = weight_only_tools::warpReduceSum(sum, blockDim.x);

        if(blockDim.x <= warpSize) {
            if(tid == 0) {
                block_sum += sum;
            }
            continue;
        }

        if(laneId == 0)
            warpLevelSums[threadIdx.y][warpId] = sum;
        __syncthreads();

        sum = warpLevelSums[threadIdx.y][laneId];

        if(warpId == 0)
            sum = weight_only_tools::warpReduceSum(sum, SplitKBatch);
        if(tid == 0) {
            block_sum += sum;
        }
    }

    if(tid == 0) {
        output[row] = __float2half(block_sum);
    }
}

template <int Batch, int NPerBlock, int SplitKBatch>
__global__ void weight_only_int4_batch2_gemv_groupwise(const ck::half8_t* input,
                                                       const uint32_t* weight_int4,
                                                       const uint32_t* zero_scales,
                                                       ck::half_t* output,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       int StrideA,
                                                       int StrideB,
                                                       int StrideC,
                                                       int GroupSize,
                                                       int k_iteration) {
    float sum[Batch]       = {0};
    float block_sum[Batch] = {0};
    const int K_Cnt        = k / VEC_L;
    const int AStride_Cnt  = StrideA / VEC_L;
    const int BStride_Cnt  = StrideB / VEC_L;
    const int BGroup_Cnt   = StrideB / GroupSize;

    int tid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    static __shared__ ck::half8_t shared_mat[NPerBlock][SplitKBatch * warpSize * READ_CNT + 1];
    static __shared__ float warpLevelSums[NPerBlock][SplitKBatch];
    static __shared__ float warpLevelSums_b[NPerBlock][SplitKBatch];

    for(int iteration = 0; iteration < k_iteration; iteration++) {
        int gidx   = tid + iteration * blockDim.x;
        int gdatax = READ_CNT * gidx;
        int ldatax = READ_CNT * tid;

        uint32_t zero_scale = zero_scales[row * BGroup_Cnt + gidx * GemvPerThreadCnt / GroupSize];
        for(int it_s = 0; it_s < READ_CNT / 2; it_s++) {
            if(gdatax + 2 * it_s + 1 < K_Cnt) {
                shared_mat[threadIdx.y][ldatax + 2 * it_s] = weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s], zero_scale);
                shared_mat[threadIdx.y][ldatax + 2 * it_s + 1] =
                    weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s + 1], zero_scale);
            }
        }
        __syncthreads();

        for(int b = 0; b < Batch; b = b + 2) {
            sum[b]     = 0;
            sum[b + 1] = 0;
            for(int it = 0; it < READ_CNT / 2; it++) {
                if(gdatax + 2 * it + 1 < K_Cnt) {
                    ck::half8_t vec_val_1    = input[gdatax + b * AStride_Cnt + 2 * it];
                    ck::half8_t vec_val_1_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it];
                    ck::half8_t mat_h_1      = shared_mat[threadIdx.y][ldatax + 2 * it];

                    ck::half8_t vec_val_2    = input[gdatax + b * AStride_Cnt + 2 * it + 1];
                    ck::half8_t vec_val_2_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it + 1];
                    ck::half8_t mat_h_2      = shared_mat[threadIdx.y][ldatax + 2 * it + 1];

                    weight_only_tools::amd_assembly_product_1x4(vec_val_1, mat_h_1, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2, mat_h_2, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_1_b2, mat_h_1, sum[b + 1]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2_b2, mat_h_2, sum[b + 1]);
                }
            }

            sum[b]     = weight_only_tools::warpReduceSum(sum[b], blockDim.x);
            sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], blockDim.x);

            if(blockDim.x <= warpSize) {
                if(tid == 0) {
                    block_sum[b] += sum[b];
                    block_sum[b + 1] += sum[b + 1];
                }
                continue;
            }

            const int laneId = threadIdx.x % warpSize;
            const int warpId = threadIdx.x / warpSize;
            if(laneId == 0) {
                warpLevelSums[threadIdx.y][warpId]   = sum[b];
                warpLevelSums_b[threadIdx.y][warpId] = sum[b + 1];
            }
            __syncthreads();

            sum[b]     = 0;
            sum[b]     = warpLevelSums[threadIdx.y][laneId];
            sum[b + 1] = 0;
            sum[b + 1] = warpLevelSums_b[threadIdx.y][laneId];

            if(warpId == 0) {
                sum[b]     = weight_only_tools::warpReduceSum(sum[b], SplitKBatch);
                sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], SplitKBatch);
            }
            if(tid == 0) {
                block_sum[b] += sum[b];
                block_sum[b + 1] += sum[b + 1];
            }
        }
    }
    if(tid == 0) {
        for(int b = 0; b < Batch; b++) {
            output[row + b * StrideC] = __float2half(block_sum[b]);
        }
    }
}

template <int Batch, int NPerBlock, int SplitKBatch>
__global__ void weight_only_int4_batch3_gemv_groupwise(const ck::half8_t* input,
                                                       const uint32_t* weight_int4,
                                                       const uint32_t* zero_scales,
                                                       ck::half_t* output,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       int StrideA,
                                                       int StrideB,
                                                       int StrideC,
                                                       int GroupSize,
                                                       int k_iteration) {

    float sum[Batch]       = {0};
    float block_sum[Batch] = {0};
    const int K_Cnt        = k / VEC_L;
    const int AStride_Cnt  = StrideA / VEC_L;
    const int BStride_Cnt  = StrideB / VEC_L;
    const int BGroup_Cnt   = StrideB / GroupSize;

    int tid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    static __shared__ ck::half8_t shared_mat[NPerBlock][SplitKBatch * warpSize * READ_CNT + 1];
    static __shared__ float warpLevelSums[NPerBlock][SplitKBatch];
    static __shared__ float warpLevelSums_b2[NPerBlock][SplitKBatch];
    static __shared__ float warpLevelSums_b3[NPerBlock][SplitKBatch];

    for(int iteration = 0; iteration < k_iteration; iteration++) {
        int gidx   = tid + iteration * blockDim.x;
        int gdatax = READ_CNT * gidx;
        int ldatax = READ_CNT * tid;

        uint32_t zero_scale = zero_scales[row * BGroup_Cnt + gidx * GemvPerThreadCnt / GroupSize];

        for(int it_s = 0; it_s < READ_CNT / 2; it_s++) {
            if(gdatax + 2 * it_s + 1 < K_Cnt) {
                shared_mat[threadIdx.y][ldatax + 2 * it_s] = weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s], zero_scale);
                shared_mat[threadIdx.y][ldatax + 2 * it_s + 1] =
                    weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s + 1], zero_scale);
            }
        }
        __syncthreads();

        for(int b = 0; b < Batch; b = b + 3) {
            sum[b]     = 0;
            sum[b + 1] = 0;
            sum[b + 2] = 0;
            for(int it = 0; it < READ_CNT / 2; it++) {
                if(gdatax + 2 * it + 1 < K_Cnt) {
                    ck::half8_t vec_val_1    = input[gdatax + b * AStride_Cnt + 2 * it];
                    ck::half8_t vec_val_1_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it];
                    ck::half8_t vec_val_1_b3 = input[gdatax + (b + 2) * AStride_Cnt + 2 * it];
                    ck::half8_t mat_h_1      = shared_mat[threadIdx.y][ldatax + 2 * it];

                    ck::half8_t vec_val_2    = input[gdatax + b * AStride_Cnt + 2 * it + 1];
                    ck::half8_t vec_val_2_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it + 1];
                    ck::half8_t vec_val_2_b3 = input[gdatax + (b + 2) * AStride_Cnt + 2 * it + 1];
                    ck::half8_t mat_h_2      = shared_mat[threadIdx.y][ldatax + 2 * it + 1];

                    weight_only_tools::amd_assembly_product_1x4(vec_val_1, mat_h_1, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2, mat_h_2, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_1_b2, mat_h_1, sum[b + 1]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2_b2, mat_h_2, sum[b + 1]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_1_b3, mat_h_1, sum[b + 2]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2_b3, mat_h_2, sum[b + 2]);
                }
            }

            sum[b]     = weight_only_tools::warpReduceSum(sum[b], blockDim.x);
            sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], blockDim.x);
            sum[b + 2] = weight_only_tools::warpReduceSum(sum[b + 2], blockDim.x);

            if(blockDim.x <= warpSize) {
                if(tid == 0) {
                    block_sum[b] += sum[b];
                    block_sum[b + 1] += sum[b + 1];
                    block_sum[b + 2] += sum[b + 2];
                }
                continue;
            }

            const int laneId = threadIdx.x % warpSize;
            const int warpId = threadIdx.x / warpSize;
            if(laneId == 0) {
                warpLevelSums[threadIdx.y][warpId]    = sum[b];
                warpLevelSums_b2[threadIdx.y][warpId] = sum[b + 1];
                warpLevelSums_b3[threadIdx.y][warpId] = sum[b + 2];
            }
            __syncthreads();

            sum[b]     = 0;
            sum[b]     = warpLevelSums[threadIdx.y][laneId];
            sum[b + 1] = 0;
            sum[b + 1] = warpLevelSums_b2[threadIdx.y][laneId];
            sum[b + 2] = 0;
            sum[b + 2] = warpLevelSums_b3[threadIdx.y][laneId];

            if(warpId == 0) {
                sum[b]     = weight_only_tools::warpReduceSum(sum[b], SplitKBatch);
                sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], SplitKBatch);
                sum[b + 2] = weight_only_tools::warpReduceSum(sum[b + 2], SplitKBatch);
            }
            if(tid == 0) {
                block_sum[b] += sum[b];
                block_sum[b + 1] += sum[b + 1];
                block_sum[b + 2] += sum[b + 2];
            }
        }
    }
    if(tid == 0) {
        for(int b = 0; b < Batch; b++) {
            output[row + b * StrideC] = __float2half(block_sum[b]);
        }
    }
}

template <int Batch, int NPerBlock, int SplitKBatch>
__global__ void weight_only_int4_batch4_gemv_groupwise(const ck::half8_t* input,
                                                       const uint32_t* weight_int4,
                                                       const uint32_t* zero_scales,
                                                       ck::half_t* output,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       int StrideA,
                                                       int StrideB,
                                                       int StrideC,
                                                       int GroupSize,
                                                       int k_iteration) {

    float sum[Batch]       = {0};
    float block_sum[Batch] = {0};
    const int K_Cnt        = k / VEC_L;
    const int AStride_Cnt  = StrideA / VEC_L;
    const int BStride_Cnt  = StrideB / VEC_L;
    const int BGroup_Cnt   = StrideB / GroupSize;

    int tid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    static __shared__ ck::half8_t shared_mat[NPerBlock][SplitKBatch * warpSize * READ_CNT + 1];
    static __shared__ float warpLevelSums[NPerBlock][SplitKBatch];
    static __shared__ float warpLevelSums_b[NPerBlock][SplitKBatch];

    for(int iteration = 0; iteration < k_iteration; iteration++) {
        int gidx   = tid + iteration * blockDim.x;
        int gdatax = READ_CNT * gidx;
        int ldatax = READ_CNT * tid;

        uint32_t zero_scale = zero_scales[row * BGroup_Cnt + gidx * GemvPerThreadCnt / GroupSize];
        for(int it_s = 0; it_s < READ_CNT / 2; it_s++) {
            if(gdatax + 2 * it_s + 1 < K_Cnt) {
                shared_mat[threadIdx.y][ldatax + 2 * it_s] = weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s], zero_scale);
                shared_mat[threadIdx.y][ldatax + 2 * it_s + 1] =
                    weight_only_tools::dequant_convert(weight_int4[row * BStride_Cnt + gdatax + 2 * it_s + 1], zero_scale);
            }
        }
        __syncthreads();

        for(int b = 0; b < Batch; b = b + 2) {
            sum[b]     = 0;
            sum[b + 1] = 0;
            for(int it = 0; it < READ_CNT / 2; it++) {
                if(gdatax + 2 * it + 1 < K_Cnt) {
                    ck::half8_t vec_val_1    = input[gdatax + b * AStride_Cnt + 2 * it];
                    ck::half8_t vec_val_1_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it];
                    ck::half8_t mat_h_1      = shared_mat[threadIdx.y][ldatax + 2 * it];

                    ck::half8_t vec_val_2    = input[gdatax + b * AStride_Cnt + 2 * it + 1];
                    ck::half8_t vec_val_2_b2 = input[gdatax + (b + 1) * AStride_Cnt + 2 * it + 1];
                    ck::half8_t mat_h_2      = shared_mat[threadIdx.y][ldatax + 2 * it + 1];

                    weight_only_tools::amd_assembly_product_1x4(vec_val_1, mat_h_1, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2, mat_h_2, sum[b]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_1_b2, mat_h_1, sum[b + 1]);
                    weight_only_tools::amd_assembly_product_1x4(vec_val_2_b2, mat_h_2, sum[b + 1]);
                }
            }

            sum[b]     = weight_only_tools::warpReduceSum(sum[b], blockDim.x);
            sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], blockDim.x);

            if(blockDim.x <= warpSize) {
                if(tid == 0) {
                    block_sum[b] += sum[b];
                    block_sum[b + 1] += sum[b + 1];
                }
                continue;
            }

            const int laneId = threadIdx.x % warpSize;
            const int warpId = threadIdx.x / warpSize;
            if(laneId == 0) {
                warpLevelSums[threadIdx.y][warpId]   = sum[b];
                warpLevelSums_b[threadIdx.y][warpId] = sum[b + 1];
            }
            __syncthreads();

            sum[b]     = 0;
            sum[b]     = warpLevelSums[threadIdx.y][laneId];
            sum[b + 1] = 0;
            sum[b + 1] = warpLevelSums_b[threadIdx.y][laneId];

            if(warpId == 0) {
                sum[b]     = weight_only_tools::warpReduceSum(sum[b], SplitKBatch);
                sum[b + 1] = weight_only_tools::warpReduceSum(sum[b + 1], SplitKBatch);
            }
            if(tid == 0) {
                block_sum[b] += sum[b];
                block_sum[b + 1] += sum[b + 1];
            }
        }
    }
    if(tid == 0) {
        for(int b = 0; b < Batch; b++) {
            output[row + b * StrideC] = __float2half(block_sum[b]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int SplitKBatch, int NPerBlock>
void launch_gemv_batch_wrapper_kernel(const ck::half8_t* input,
                                      const uint32_t* weight,
                                      const uint32_t* zero_scales,
                                      ck::half_t* output,
                                      int m,
                                      int n,
                                      int k,
                                      int StrideA,
                                      int StrideB,
                                      int StrideC,
                                      int GroupSize,
                                      hipStream_t stream_id) {
    // 设置参数
    const int KerBlock = SplitKBatch * warpSize * GemvPerThreadCnt;
    int block_dim_x    = SplitKBatch * warpSize;
    int block_dim_y    = NPerBlock;
    int k_iteration    = (k + KerBlock - 1) / KerBlock; // k整除KPERBLOCK时性能最优

    if(k % 16 != 0) {
        printf("wrong! mod(k=%d, 16) != 0\n", k);
        return;
    }
    if(n % NPerBlock != 0) {
        printf("wrong! mod(n=%d, NPerBlock=%d) != 0\n", n, NPerBlock);
        return;
    }
    if(block_dim_x * block_dim_y > MAX_THREADS_PER_BLOCK) {
        printf("wrong! block_dim_x=%d * block_dim_y=%d need <= MAX_THREADS_PER_BLOCK=%d\n", block_dim_x, block_dim_y, MAX_THREADS_PER_BLOCK);
        return;
    }
    if(SplitKBatch > warpSize) {
        printf("wrong! SplitKBatch=%d > warpSize=%d\n", SplitKBatch, warpSize);
        return;
    }
    if(SplitKBatch * NPerBlock > 16) {
        printf("wrong! NPerBlock=%d * SplitKBatch=%d > 16\n", NPerBlock, SplitKBatch);
        return;
    }

    // 设置kernel参数
    dim3 grid_dim(1, n / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);

    switch(m) {
        case 1:
            weight_only_int4_gemv_groupwise<1, NPerBlock, SplitKBatch>
                <<<grid_dim, block_dim, 0, stream_id>>>(input, weight, zero_scales, output, m, n, k, StrideA, StrideB, StrideC, GroupSize, k_iteration);
            break;
        case 2:
            weight_only_int4_batch2_gemv_groupwise<2, NPerBlock, SplitKBatch>
                <<<grid_dim, block_dim, 0, stream_id>>>(input, weight, zero_scales, output, m, n, k, StrideA, StrideB, StrideC, GroupSize, k_iteration);
            break;
        case 3:
            weight_only_int4_batch3_gemv_groupwise<3, NPerBlock, SplitKBatch>
                <<<grid_dim, block_dim, 0, stream_id>>>(input, weight, zero_scales, output, m, n, k, StrideA, StrideB, StrideC, GroupSize, k_iteration);
            break;
        case 4:
            weight_only_int4_batch4_gemv_groupwise<4, NPerBlock, SplitKBatch>
                <<<grid_dim, block_dim, 0, stream_id>>>(input, weight, zero_scales, output, m, n, k, StrideA, StrideB, StrideC, GroupSize, k_iteration);
            break;
        default: printf("WeightOnlyGemvKernel type only support m <= 4\n"); break;
    }
}

} // namespace gemv_batch

void run_gemv_batch_wrapper(
    const void* A, const void* B0, const void* B1, void* C, int m, int n, int k, int StrideA, int StrideB, int StrideC, int GroupSize, hipStream_t stream_id) {
    // 可调, 具体场景需要寻最优参数
    const auto input       = reinterpret_cast<const ck::half8_t*>(A);
    const auto weight      = reinterpret_cast<const uint32_t*>(B0);
    const auto zero_scales = reinterpret_cast<const uint32_t*>(B1);
    auto output            = reinterpret_cast<ck::half_t*>(C);

    auto launch_kernel = [&](auto SplitKBatch_, auto NPerBlock_) {
        return gemv_batch::launch_gemv_batch_wrapper_kernel<SplitKBatch_, NPerBlock_>(
            input, weight, zero_scales, output, m, n, k, StrideA, StrideB, StrideC, GroupSize, stream_id);
    };

    switch(m) {
        case 1: launch_kernel(ck::integral_constant<int, 4>{}, ck::integral_constant<int, 1>{}); break;
        case 2: launch_kernel(ck::integral_constant<int, 2>{}, ck::integral_constant<int, 2>{}); break;
        case 3: launch_kernel(ck::integral_constant<int, 1>{}, ck::integral_constant<int, 4>{}); break;
        case 4: launch_kernel(ck::integral_constant<int, 1>{}, ck::integral_constant<int, 2>{}); break;
        default: printf("WeightOnlyGemvKernel type only support m <= 4\n"); break;
    }
}
