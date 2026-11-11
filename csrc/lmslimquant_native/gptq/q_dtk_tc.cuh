// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include "hcu_mmac.h"
#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef USE_ROCM
#include "compat.cuh"
#endif
#include "matrix_view.cuh"
#include "qdq_2.cuh"
#include "qdq_3.cuh"
#include "qdq_4.cuh"
#include "qdq_8.cuh"

namespace vllm {
namespace gptq {

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using floatx4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
template<bool is_half=true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, floatx4& reg_c)
{
    #if defined(__gfx936__) || (defined(__gfx928__) || defined(__gfx92a__))
        // if constexpr (is_half)
            reg_c=HCU_MMAC_F32_F16(reg_a,reg_b,reg_c);
        //  else
        //     reg_c=__builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*)&reg_a,*(v4bh*)&reg_b,reg_c);
    #endif
    #if defined(__gfx938__)
        __builtin_trap();  // HCU_MMAC_F32_F16 gfx938编译不通过强制终止
    #endif

}

struct half4x2{
  half4_t data[2];
};
struct half4x8{
  half4_t data[8];
};
struct half4x4{
  half4_t data[4];
};
#define WARP_SIZE 64
#define MMA_SIZE 16

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))
template <int WARP_NUM=4, int BLOCK_N_SIZE=16, int BLOCK_M_SIZE=16, int BLOCK_K_SIZE=512>
__global__ void tc_gptq_int8(const __half* ha, const uint32_t* hw, const uint32_t* zero, const __half* scale, __half* ho, int size_m, int size_n,int size_k)
{
    constexpr int n_num=BLOCK_N_SIZE/MMA_SIZE;
    constexpr int m_num=BLOCK_M_SIZE/MMA_SIZE;
    int size_mma = 8;
    int offset_a = size_mma * MMA_SIZE; // 128
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x;
    const int block_m = blockIdx.y;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);

    int threadinwarp = tid % WARP_SIZE;

    int row_a = threadinwarp % MMA_SIZE + block_m * BLOCK_M_SIZE;
    int col_a = threadinwarp / MMA_SIZE; //注意有向右4个的偏移

    int row_b = threadinwarp / MMA_SIZE; //注意有向下的4个偏移
    int col_b = threadinwarp % MMA_SIZE;

    half4x8 tmp_a[m_num];
    for(int j = 0; j < m_num; j++)
    {
        for(int i = 0; i < size_mma; i++)
        {
            tmp_a[j].data[i]={0,0,0,0};
        }
    }
    floatx4 d_vec[m_num][n_num];
    for(int i = 0; i < m_num; i++)
    {
        for(int j = 0; j < n_num; j++)
        {
            d_vec[i][j]={0,0,0,0};
        }
    }

    int temp_row_a;

    for (int i = warp_idx * offset_a; i < size_k; i += offset_a * WARP_NUM) // 128 = size_mma * MMA_SIZE groupsize 是128 (4个half压缩一个uint32，一次取4个uint32), 4*size_mma=32，awq是4*8=32,
    {
        for (int m = 0; m < m_num; m++)
        {
            temp_row_a = row_a + m * MMA_SIZE;
            if ( temp_row_a < size_m)
                tmp_a[m] = *(half4x8*)(ha + temp_row_a * size_k + i + col_a * 4 * size_mma); //单次偏移是4，一次取16个数据，所以是4 * 16个数据, 需要4个tensor core一次计算。
        }
        for (int n = 0; n < n_num; n++)
        {
            half4x8 tmp_b;
            int tmp_cow_b = block_n * BLOCK_N_SIZE + n * MMA_SIZE + col_b;
            if (tmp_cow_b < size_n)
            {
                int tmp_row_b = row_b * size_mma + i / 4;
                int tmp_z = zero[(tmp_row_b / 32) * size_n/4+ tmp_cow_b / 4];
                int z_col_offset = tmp_cow_b % 4;
                __half tmp_s = scale[(tmp_row_b / 32) * size_n + tmp_cow_b];
                for (int ii = 0; ii < size_mma; ii++)
                {

                    int int_tmp_b = hw[(tmp_row_b + ii) * size_n + tmp_cow_b];
                    for (int j = 0; j < 4; j++)
                    {
                        tmp_b.data[ii][j] = (__int2half_rn(exb(int_tmp_b, j * 8, 0xff) - (((tmp_z >> (8 * z_col_offset)) & 0xff) + 1))) * tmp_s;
                    }
                }
            }
            for (int m = 0; m < m_num; m++)
            {
                for(int n_mma= 0; n_mma < size_mma; n_mma++)
                {
                    builtin_amdgcn_mmac(tmp_a[m].data[n_mma], tmp_b.data[n_mma], d_vec[m][n]);
                }
            }

        }
    }

    int row_c = threadinwarp % MMA_SIZE + block_m * BLOCK_M_SIZE;
    int col_c = threadinwarp / MMA_SIZE; //开始位置，一次向后偏移4
    if constexpr(WARP_NUM > 1){
        extern __shared__ floatx4 out_smem[];
        int lds_size_n_offset = WARP_NUM / 2 * 16 * size_m / 4;
            #pragma unroll
            for (int i = WARP_NUM; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    for (int m = 0; m < m_num; m++)
                    {
                        temp_row_a = row_a + m * MMA_SIZE;
                        if (temp_row_a < size_m)
                        {
                            for(int n = 0; n < n_num; n++){
                                out_smem[(warp_idx - mid) * 4 * size_m + temp_row_a * 4 + col_c + lds_size_n_offset * n] = d_vec[m][n];
                            }
                        }
                    }

                }
                __syncthreads();
                if (warp_idx < mid) {
                    for (int m = 0; m < m_num; m++)
                    {
                        temp_row_a = row_a + m * MMA_SIZE;
                        if (temp_row_a < size_m)
                        {
                            for(int n = 0; n< n_num; n++)
                            {
                                floatx4 tmp = out_smem[warp_idx * 4 * size_m + temp_row_a * 4 + col_c + lds_size_n_offset * n];
                                #pragma unroll
                                for (int index = 0; index < 4; index++) {
                                    d_vec[m][n][index] += tmp[index];
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
    }

    if (warp_idx == 0)
    {
        for(int m=0; m < m_num; m++)
        {
            temp_row_a = row_a + m * MMA_SIZE;
            if (temp_row_a < size_m) {
                for(int n=0; n < n_num; n++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        if (block_n * BLOCK_N_SIZE + n * MMA_SIZE + col_c + i * 4 < size_n)
                            ho[temp_row_a * size_n + block_n * BLOCK_N_SIZE + n * MMA_SIZE + col_c + i * 4] = static_cast<_Float16>(d_vec[m][n][i]);
                    }
                }
            }
        }
    }

}

#define NPerBlock_SWITCH(n_per_block, ...)    \
  [&] {                                       \
    if (n_per_block == 16) {                  \
      constexpr static int NPerBlock = 16;    \
      return __VA_ARGS__();                   \
    } else if (n_per_block == 32){            \
      constexpr static int NPerBlock = 32;    \
      return __VA_ARGS__();                   \
    } else if (n_per_block == 64){            \
      constexpr static int NPerBlock = 64;    \
      return __VA_ARGS__();                   \
    } else {                                  \
      constexpr static int NPerBlock = 128;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define MPerBlock_SWITCH(m_per_block, ...)    \
  [&] {                                       \
    if (m_per_block == 16) {                  \
      constexpr static int MPerBlock = 16;    \
      return __VA_ARGS__();                   \
    } else if (m_per_block == 32){            \
      constexpr static int MPerBlock = 32;    \
      return __VA_ARGS__();                   \
    } else if (m_per_block == 64){            \
      constexpr static int MPerBlock = 64;    \
      return __VA_ARGS__();                   \
    } else if (m_per_block == 128){            \
      constexpr static int MPerBlock = 128;   \
      return __VA_ARGS__();                   \
    } else if (m_per_block == 256){            \
      constexpr static int MPerBlock = 256;   \
      return __VA_ARGS__();                   \
    } else {                                  \
      constexpr static int MPerBlock = 512;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()

static void select_param(int size_m,int size_n, int size_k, int &m_per_block,int &n_per_block)
{
    n_per_block=32;
    m_per_block=16;

    m_per_block =  ((size_m + MMA_SIZE - 1) / MMA_SIZE) * MMA_SIZE;

    // if (m_per_block > 64) m_per_block=64;
    //qwen2 7b
    if (size_n / size_k > 2) n_per_block = 128;
}

void gemm_tc_gptq_int8(const half* a, const uint32_t* b_q_weight, const uint32_t* b_gptq_qzeros,
                        const half* b_gptq_scales, half* c, int size_m, int size_n, int size_k)
{

    const int warp_num = 4;
    // const int block_n = 16; // 32 64
    // const int block_m = 16;
    int n_per_block, m_per_block;
    select_param(size_m, size_n, size_k, m_per_block, n_per_block);

    int block_size = WARP_SIZE * warp_num;
    // dim3 gridDim;
    // gridDim.x = DIVIDE(size_n, block_n);
    // // gridDim.y = DIVIDE(size_m, m_count);
    // // gridDim.z = DIVIDE(size_k, BLOCK_K_SIZE);
    // gridDim.y = DIVIDE(size_m, block_m);
    // gridDim.z = 1;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    NPerBlock_SWITCH(n_per_block,[&] {
        MPerBlock_SWITCH(m_per_block,[&] {
        int lds_size=warp_num/2*NPerBlock*size_m*sizeof(float);
        dim3 gridDim;
        gridDim.x = DIVIDE(size_n, NPerBlock);
        // gridDim.y = DIVIDE(size_m, m_count);
        // gridDim.z = DIVIDE(size_k, BLOCK_K_SIZE);
        gridDim.y = DIVIDE(size_m, MPerBlock);
        gridDim.z = 1;


        tc_gptq_int8<warp_num, NPerBlock, MPerBlock><<<gridDim, block_size, lds_size, stream>>>(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c, size_m, size_n, size_k);
        });
    });
  }
}}
