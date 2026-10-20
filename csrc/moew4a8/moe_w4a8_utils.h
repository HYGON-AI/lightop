// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
#include "intrinsic.h"

namespace at{
namespace native{

#define buffer_load_lds_tile_pad(WARP_NUM, N_row_len, M, N, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id)\
{\
    int bytes_per_Element = 2;\
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {\
        bytes_per_Element = 1;\
    }\
    int Element_per_dword = 4/bytes_per_Element;\
    int lane_M_idx = lane_id >> 4;\
    int lane_N_idx = lane_id & 15;\
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int padding = (warp_loop & 7);\
        int gsOffset   = global_offset/Element_per_dword;\
        int gvOffset   = (std::min(warp_loop * 4 + lane_M_idx, max_M_len - 1) * N_row_len)/Element_per_dword  +  lane_N_idx;\
        int lds_offset = lds_stage_offset/Element_per_dword + padding + (warp_loop & 7) * 64 + (warp_loop / 8) * 32 * 34/2;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

// B矩阵buffer_load读
#define buffer_load_lds_tile_pad_split(WARP_NUM, N_row_len, BLOCK_N, WARP_K, WARP_N, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                         vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 1;\
    int Element_per_dword = 4/bytes_per_Element;\
    int thread_num_n = WARP_K*bytes_per_Element/4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\
    int lane_N_idx = lane_id % thread_num_n;\
    for(int m_idx = 0 ; m_idx < BLOCK_N /( WARP_N * WARP_NUM); m_idx++) {\
        for(int load = 0; load < (WARP_N * WARP_K * bytes_per_Element) / (4/*1 dword*/ * 64); load++) {\
            int block_row = load * thread_num_m + lane_M_idx;\
            int g_row = (std::min(sorted_token_ids_offset[block_row] / top_k,  max_n_len_offset - 1)); \
            int phase = (block_row/per_phase) % max_phase; \
            int g_col = lane_N_idx; \
            int vec_size_dword = vec_size / Element_per_dword; \
            int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
            int gsOffset = m_idx * WARP_N * WARP_NUM * N_row_len / Element_per_dword + warp_id * WARP_N * N_row_len / Element_per_dword+ global_offset / Element_per_dword;\
            int gvOffset = block_row * N_row_len / Element_per_dword + g_col_swizzle;\
            int lds_offset = warp_id * WARP_K * WARP_N / Element_per_dword + lds_stage_offset/Element_per_dword + m_idx * WARP_N * WARP_NUM * WARP_K / Element_per_dword + load * 64;\
            builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
        }\
    }\
}

// A矩阵buffer_load读                                                                                          
#define buffer_load_lds_tile_pad_sorted_token(WARP_NUM, N_row_len, WARP_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 1;\
    int Element_per_dword = 4/bytes_per_Element;\
    int thread_num_n = WARP_K*bytes_per_Element/4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (WARP_M*WARP_K*bytes_per_Element) / (4*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size / Element_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/Element_per_dword;\
        int gvOffset   = g_row * N_row_len/Element_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/Element_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

// B矩阵buffer_load读                                                                                          
#define buffer_load_lds_tile_pad_weight(WARP_NUM, N_row_len, WARP_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 1;\
    int Element_per_dword = 4/bytes_per_Element;\
    int thread_num_n = WARP_K*bytes_per_Element/4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (WARP_M*WARP_K*bytes_per_Element) / (4*64);\  
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int g_row = block_row; \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size / Element_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/Element_per_dword;\
        int gvOffset   = g_row * N_row_len/Element_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/Element_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

#define ds_read2_tile_pad_no_wait(M,n_idx,WARP_NUM,Element,lds_v8i8,precompute_offset,lds_stage_offset_loop,reg,loop)\
{\
for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
    inline_ds_read2_b32_no_wait(lds_v8i8, precompute_offset[m_idx * (WARP_K/MFMA_K) + n_idx] + lds_stage_offset_loop, reg[loop * (M / 16) + m_idx][n_idx].int2_array[0], 1);\
}\
}

template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  gemm_nt_first_stage(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,8> A_reg[][WARP_K/32], 
    union_vec<Element,8> B_reg[][WARP_K/32], 
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    const int warp_n_num = BLOCK_N / WARP_N;
    int warp_n_id = warp_id % warp_n_num;
    int warp_m_id = warp_id / warp_n_num;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)]; 
    typename vec<Element,8>::type *B_lds_v8i8 = (typename vec<Element,8>::type *)(B_lds);
    typename vec<Element,8>::type *A_lds_v8i8 = (typename vec<Element,8>::type *)(A_lds);

    // Reference: TritonGPU swizzled shared-memory layout.
    const int type_width_in_bit = 8;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int vec_size = 8;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length);
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size);

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K; // A tile拿 1 * 16x64的大小
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K; // B tile拿 4 * 32x64的大小,
#if 1
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); // 配置全局显存信息

        #pragma unroll // 预加载STAGES-1的数据
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);                
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * WARP_K; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);
            }
        }
    }

    // 读取A矩阵
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            // int lds_offset = min_tile_k * 4 + min_tile_n*32 + (lane_id & 15)*16 + ((lane_id & 15)>>2) + ((lane_id/16) & 1)*8 + (lane_id/32); // warp_layout：16x4
            // 线程布局16x4,取16x64的数据,k方向的stride为WARP_K,也正好是WARP_K个byte,换算成dword:WARP_K/4,k方向每个线程取MFMA_K/4=8个元素,即2个dword
            int lds_col = (min_tile_k * MFMA_K) / 4 + (lane_id/16)*2;
            int lds_row = (lane_id & 15);
            int vec_size_dword = vec_size/4/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/4 + (min_tile_m * MFMA_M + lds_row)*WARP_K/4/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 读取B矩阵
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            { // 走这里
                int lds_col = (min_tile_k * MFMA_K) / 4 + (lane_id/16)*2;
                int lds_row = (lane_id & 15);
                int vec_size_dword = vec_size/4/*dword*/;
                int phase = (lds_row/per_phase) % max_phase;
                int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
                int lds_offset = warp_n_id * (WARP_N*WARP_K)/4 + min_tile_n * (MFMA_N*WARP_K/4) + lds_row * (WARP_K/4) + col_swizzle;/*dword*/
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K) + min_tile_k] = lds_offset;
            } 
        }
    }

    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * WARP_K; 
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 

            buffer_load_lds_tile_pad_weight(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                    vec_size, per_phase, max_phase);  
        }

        vmcnt_wait(((BLOCK_N * WARP_K) / (4*64)/WARP_NUM + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) {
                    C_reg[0][min_tile_m*(WARP_N / MFMA_N) + min_tile_n] = mmac<Element>(
                        *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                        C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]);        
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait(((BLOCK_N * WARP_K) / (4*64)/WARP_NUM + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                        *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                        C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
                }
            }
        }
    }

    // 为了最后一次等到vmcnt_wait=0
    vmcnt_wait(0);
    {
        stage_id_reg_A = STAGES-1;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                    *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                    *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
            }
        }
    }
    lgkmcnt_wait_barrier(0);

#endif
}

using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
// MOE-DECODE-W4A8 针对小shape优化
template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AI8_PERCHANNEL_HIP_NT_decode_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[][STAGES], // 2 = warpk/mmak
    union_vec<Element,WARP_K/8> B_reg[][STAGES], // 2 = warpk/mmak
    // union_vec<Element,8> A_reg[][2], // 2 = warpk/mmak
    // union_vec<Element,8> B_reg[][2], // 2 = warpk/mmak
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int8_t * tb;  
    int8x8_t b_vec;
    for(int i = warp_k_id * WARP_K; i < size_k; i += warp_k_num * WARP_K){

        // load A
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / 64; k_tile++){ // todo 一个线程连续拷贝16还是隔开拷贝  col_id * 16 * WARP_K/64 + k_tile * 16
                A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + i + col_id * 16 + k_tile * 64);
                //A_reg[m_tile].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + i + col_id * 16 * WARP_K/64 + k_tile * 16);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / 64; k_tile++){
                B_reg[n_tile][0].int2_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx2(weight_ptr, 0, (g_row * size_k/2) + (i/2) + (col_id * 8) + (k_tile * 32));
                //B_reg[n_tile].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + i + col_id * 16 * WARP_K/64 + k_tile * 16);
            }
        }

        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                tb = (&B_reg[n_tile][0].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F) - 8 ;
                    b_vec[1] = ((tb[0] >> 4) & 0x0F) - 8;
                    b_vec[2] = (tb[1]& 0x0F)- 8;
                    b_vec[3] = ((tb[1]>> 4) & 0x0F)- 8;
                    b_vec[4] = (tb[2]& 0x0F)- 8;
                    b_vec[5] = ((tb[2]>> 4) & 0x0F)- 8;
                    b_vec[6] = (tb[3]& 0x0F)- 8 ;
                    b_vec[7] = ((tb[3]>> 4) & 0x0F) - 8 ;
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&b_vec), 
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }
    } // k_loop end
    

    // warp间的规约 
    extern __shared__ int out_smem[]; // 声明lds信息
    if constexpr (warp_k_num > 1){
        constexpr int lds_n_offset = warp_k_num * BLOCK_N;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ // todo 连续拷贝还是隔开拷贝 -> 差别不大 n_tile*4 + col_id * 4 *  WARP_N / MFMA_N
                *(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
                //*(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }

        __syncthreads();


        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0, 0, 0, 0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            //out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N  + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                        }
                    }
                }
            }
        }
    }
}
#define buffer_load_lds_tile_pad_sorted_token_marlin(WARP_NUM, N_row_len, WARP_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id_M,warp_id_K, lane_id,vec_size,per_phase,max_phase)\
{\
    int bytes_per_Element = 1;\
    int Element_per_dword = 4/bytes_per_Element;\
    int thread_K_perWarp = WARP_K * 2 / vec_size;\
    int thread_M_perWarp = 64 / thread_K_perWarp;\ 
    int lane_M_idx = lane_id / thread_K_perWarp;\  
    int lane_N_idx = lane_id % thread_K_perWarp;\  
    const int lds_load_num_per_warp = (WARP_M * WARP_K * 2 * bytes_per_Element) / (vec_size * 64);\
    int WARP_NUM_M=2;\
    int WARP_NUM_K=1;\
    for(int load = 0,warp_loop = warp_id_M; load < (lds_load_num_per_warp / WARP_NUM_M); warp_loop += WARP_NUM_M, ++load) {\
        for(int iter_K = 0; iter_K < std::min(1,(BLOCK_K / WARP_NUM_K / WARP_K)); iter_K++){\
            int block_row = warp_loop * thread_M_perWarp + lane_M_idx;\ 
            int g_col = lane_N_idx ;\ 
            int phase = (block_row/per_phase) % max_phase; \
            int vec_size_dword = vec_size / Element_per_dword; \ 
            int g_col_swizzle = (g_col ^ phase) * vec_size_dword;\ 
            int gsOffset   = global_offset/Element_per_dword;\
            int sorted_token_ids_tt = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
            int g_row = (std::min(sorted_token_ids_tt / top_k,  max_n_len_offset - 1)); \
            int gvOffset   = (g_row * seqlen_A_stride + iter_K * WARP_NUM_K *  thread_K_perWarp * vec_size  + warp_id_K * WARP_K) / Element_per_dword +  g_col_swizzle;\
            int lds_offset = lds_stage_offset + (warp_loop * thread_M_perWarp * WARP_K * 2 + iter_K * WARP_NUM * 16 * WARP_K) + warp_id_K * WARP_K;\
            amdgcn_buffer_load_dwordx4_lds((lds_ptr+lds_offset), global_ptr, gvOffset*4, gsOffset);\
        }\
    }\
}

#define ds_readb128_tile_pad_no_wait(M,n_idx,WARP_NUM,Element,lds_v16i8,precompute_offset,lds_stage_offset_loop,reg,loop)\
{\
    for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
        inline_ds_read_b128_no_wait(lds_v16i8 + precompute_offset[m_idx+ n_idx]+ lds_stage_offset_loop, reg[m_idx][n_idx].int4_array[0]);\
    }\
}

template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    int splitk,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AI8_PERCHANNEL_MARLIN_HIP_NN_UP_decode_kernel(
    const Element* input_ptr, 
    const int* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    int8_vec<WARP_K/8> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    intx4 C_reg[][4 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    const int size_k = seqlen_A_stride;//7168
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_k_num = BLOCK_K/WARP_K;
    int warp_k_id = warp_id % warp_k_num;//0:0 1:1 2:0 3:1
    int warp_n_id = warp_id / warp_k_num;//0:0 1:0 2:1 3:1
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    int i = 0;
    int B_stride_k = WARP_K / 32;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                A_reg[m_tile][0].int4_array[k_tile] = *(intx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
        }
    }

    i=1; 
    union_vec<Element,8> midval;
    int itk = size_k/warp_k_num/WARP_K/splitk; //112
    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                A_reg[m_tile][1].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
            }
        }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                        int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                        midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                        midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&midval.int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                   A_reg[m_tile][0].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16);
                    // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){  
                    B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
                    // B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k + B_stride_k * warp_k_id + k_tile) * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + lane_id * 4);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                    int first_val = B_reg[n_tile][1].data[4*k_tile + N_PACKID];
                    midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                    midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&midval.int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i + 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                    // int first_val = B_reg[n_tile][0].int_array[k_tile];
                    int first_val = B_reg[n_tile][0].data[4*k_tile + N_PACKID];
                    union_vec<Element,8> midval;
                    midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                    midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;

                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&midval.int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                    }
                }
            }
        }        
    }
    extern __shared__ int out_smem[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=65;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *(intx4*)(out_smem+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 16  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + k];
                    }
                }
            }
        
        __syncthreads();

        if(warp_k_id == 0){ //存在多个warp的warpkid=0，这些warp都需要和附近的warpknum块数据进行reduce
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num-1; k_tile++){//warp迭代
                        #pragma unroll
                        for(int k = 0; k < 4; k++){//4组数据按组迭代
                            intx4 temp = *(intx4*)(out_smem + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 16  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + k][i] += temp[i];
                            }
                        }
                    }
                }
            }
        __syncthreads();
        }
    }
    
}
template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    int splitk,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AI8_PERCHANNEL_MARLIN_HIP_NN_UP_decode_warp_specialization_kernel(
    const Element* input_ptr, 
    const int* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    int8_vec<WARP_K/8> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 16384
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    bhalf_t* g_output,
    int stride_asm
    ) 
{
    const int size_k = seqlen_A_stride;  //7168
    int lane_id = threadIdx.x & 63; // thread_id
    //符合mmac 列主序
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_m_num = BLOCK_M/WARP_M; //1
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N / (WARP_N * 4); // 8
    constexpr int type_width_in_bit = 8;
    constexpr int num_banks = 32;
    constexpr int bank_bit_width = 32;
    constexpr int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit; // 128
    constexpr int vec_sizeA = 16; 
    const int per_phaseA = std::max(1, elems_per_one_banks_row / WARP_K / 2);
    const int max_phaseA = 32 / (vec_sizeA * type_width_in_bit / bank_bit_width); // 8
    int warp_m_id = 0; //默认不分布 
    int warp_k_id = warp_id % warp_k_num; 
    int warp_n_id = warp_id / warp_k_num; 
    // int g_row_A[WARP_M / MFMA_M]; //1
    int g_row_B[WARP_N / MFMA_N]; //1
    int i = 0;
    int B_stride_k = WARP_K / 32; //2
    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; //1 * 2 =2
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * BLOCK_K * 2;
    int A_lds_stage_offset_loop = A_lds_stage_offset * warp_k_num;
    union_vec<Element,8> midval[8];
    int itk = size_k/warp_k_num/WARP_K/splitk; //112 
    if(warp_id>7 ){
        if(warp_id > 9) return;

        //----------------------------------------------------Init----------------------------------------------------------//

        //warp8 9 负责将A预取到lds，A的大小为16x128 需要迭代56次
        //----------------------------------------------------Prefetch----------------------------------------------------------//
        int warp_id_M = warp_id % 4; 
        auto A_ptr = input_ptr; 
        int A_block_buffer_load_global_offset = 0; 
        int A_lds_stage_offset_loop = 0; 
        buffer_load_lds_tile_pad_sorted_token_marlin(WARP_NUM, seqlen_A_stride, WARP_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id_M,0, lane_id,vec_sizeA,per_phaseA,max_phaseA);                
        vmcnt_wait(0);
        A_lds_stage_offset_loop = A_lds_stage_offset * warp_k_num ;
        A_ptr = (input_ptr + WARP_K * 2);
        buffer_load_lds_tile_pad_sorted_token_marlin(WARP_NUM, seqlen_A_stride, WARP_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id_M,0, lane_id,vec_sizeA,per_phaseA,max_phaseA);                
        //----------------------------------------------------Main loop----------------------------------------------------------//
        for(i=2;i<(size_k/warp_k_num/WARP_K/2/splitk);i+=2){
            
            S_BARRIER;
            vmcnt_wait(0);
            
            A_ptr = (input_ptr + i * warp_k_num * WARP_K * 2); 
            A_lds_stage_offset_loop = 0;
            buffer_load_lds_tile_pad_sorted_token_marlin(WARP_NUM, seqlen_A_stride, WARP_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id_M,0, lane_id,vec_sizeA,per_phaseA,max_phaseA);                
            S_BARRIER;
            vmcnt_wait(0); 

            A_ptr = (input_ptr + (i+1) * warp_k_num * WARP_K * 2); 
            A_lds_stage_offset_loop = A_lds_stage_offset * warp_k_num;
            buffer_load_lds_tile_pad_sorted_token_marlin(WARP_NUM, seqlen_A_stride, WARP_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id_M,0, lane_id,vec_sizeA,per_phaseA,max_phaseA);                
 
        }
        S_BARRIER;
        vmcnt_wait(0);
    }
    else if(warp_id<4){
        //----------------------------------------------------Init----------------------------------------------------------//
        intx4 C_reg[1][4 * (WARP_M/16)*(WARP_N/16)];
        for(int init=0; init< (4 * (WARP_M/16)*(WARP_N/16)); init++){
            C_reg[0][init] = {0,0,0,0};
        }
        int warp_n_id  = warp_id;
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
        }
        #pragma unroll 
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_k=0; min_tile_k<(WARP_K/(MFMA_K )); min_tile_k++) {
                int lds_col = col_id + 4 * min_tile_k;
                int phase = (row_id / per_phaseA) % max_phaseA;
                int col_swizzle = (lds_col ^ phase) * vec_sizeA;       
                int lds_offset =(min_tile_m * MFMA_M + row_id) * WARP_K * 2 + col_swizzle + warp_k_id * WARP_M * WARP_K + (min_tile_k * 0 * WARP_NUM * WARP_M * WARP_K ); 
                precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + min_tile_k] = lds_offset; 
            }
        }
        //----------------------------------------------------Prefetch----------------------------------------------------------//
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,((i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
            }
        }
        S_BARRIER;
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
            for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){
                inline_ds_read_b128_no_wait(A_lds + precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx], A_reg[min_tile_m][0].int4_array[k_idx]);         
            }
        }
        //----------------------------------------------------Main loop----------------------------------------------------------//
        for(i=1 ; i < itk ; i += 4){
            //b-->REG1
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    // A_reg[min_tile_m][1].int4_array[k_idx] = *(intx4*)(A_lds +(precompute_A_lds_offset[1]));
                    inline_ds_read_b128_no_wait(A_lds + precompute_A_lds_offset[1], A_reg[min_tile_m][1].int4_array[k_idx]);
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    // B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][1].vec[k_tile],0,(((i) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            }
            lgkmcnt_wait(1);
            vmcnt(2);
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            S_BARRIER;
 
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,(((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            }
            lgkmcnt_wait(0);//等待完成即可mmac与下次的执行进行遮掩
            vmcnt(2);
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][1].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);            
            S_BARRIER;
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    inline_ds_read_b128_no_wait(A_lds + A_lds_stage_offset_loop +precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx], A_reg[min_tile_m][0].int4_array[k_idx]);
                }
            }
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    
                    inline_ds_read_b128_no_wait(A_lds + A_lds_stage_offset_loop +precompute_A_lds_offset[1], A_reg[min_tile_m][1].int4_array[k_idx]);
                }
            } 
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    // B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][1].vec[k_tile],0,(((i+2) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            }
            lgkmcnt_wait(1);
            vmcnt(2);
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);           
            S_BARRIER;
            if((i +3)< itk){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                        buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,(((i+3) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                    }
                }
                
            
                vmcnt(2);               
            }else{
                vmcnt(0);  
            } 
            lgkmcnt_wait(0);//等待完成即可mmac与下次的执行进行遮掩
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][1].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            S_BARRIER;
            if((i +3)< itk){
                for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                    for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                        inline_ds_read_b128_no_wait(A_lds + precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx], A_reg[min_tile_m][0].int4_array[k_idx]);
                    }
                }
            }
        }
        if((i +3)< itk){
            lgkmcnt_wait(0);
            vmcnt(0);
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            S_BARRIER;            
        }
        //----------------------------------------------------Write back----------------------------------------------------------//
        if(warp_k_id == 0){
            float *b_scale_ptr = weight_scale_ptr + warp_n_id * WARP_N * 4 + col_id * 4;
            using vec_bf16_4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) unsigned short;
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
                float a_scale = input_scale_ptr[std::min(token_index / top_k,  max_n_len_offset - 1) * stride_asm]; //计算M方向的偏移
                for (int i = 0; i < 4; i++) {
                    floatx4 bscale4;
                    if(row_id==0) bscale4=*(floatx4*)(b_scale_ptr+i*16);
                    if(token_index < max_n_len_offset * top_k){
                        vec_bf16_4 D;
                        for (int j = 0; j < 4; j++) {
                            float bscale=bscale4[j];
                            bscale=__shfl(bscale,col_id*16);
                            ((bhalf_t*)&D)[j] = f32_to_bf16(static_cast<float> (((int*)&C_reg[0][m_tile*4+j])[i]  ) * a_scale * bscale);
                        }
                        *reinterpret_cast<vec_bf16_4*>(&g_output[token_index * (seqlen_B_stride>>2) + warp_n_id * WARP_N * 4 + col_id * 4 + i * 16]) = D;
                    }
                }
            }
        }
    }
else{//线程5 6 7 8
        //----------------------------------------------------Init----------------------------------------------------------//
        int warp_n_id  = warp_id;
        intx4 C_reg1[1][4 * (WARP_M/16)*(WARP_N/16)];
        for(int init=0; init< (4 * (WARP_M/16)*(WARP_N/16)); init++){
            C_reg1[0][init] = {0,0,0,0};
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
        }
        #pragma unroll 
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_k=0; min_tile_k<(WARP_K/(MFMA_K )); min_tile_k++) {
                int lds_col = col_id + 4 * min_tile_k;
                int phase = (row_id / per_phaseA) % max_phaseA;
                int col_swizzle = (lds_col ^ phase) * vec_sizeA;       
                int lds_offset =(min_tile_m * MFMA_M + row_id) * WARP_K * 2 + col_swizzle + warp_k_id * WARP_M * WARP_K + (min_tile_k * 0 * WARP_NUM * WARP_M * WARP_K ); 
                precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + min_tile_k] = lds_offset; 
            }
        }
        //----------------------------------------------------Prefetch----------------------------------------------------------//
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,((i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
            }
        }
        S_BARRIER;  
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
            for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                inline_ds_read_b128_no_wait(A_lds + precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx], A_reg[min_tile_m][0].int4_array[k_idx]); 
            }
        }
               
        //----------------------------------------------------main loop ----------------------------------------------------------//
        for(i = 1; i < itk ; i += 4){
            lgkmcnt_wait(0);
            vmcnt_wait(0);            
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    inline_ds_read_b128_no_wait(A_lds + (precompute_A_lds_offset[1]), A_reg[min_tile_m][1].int4_array[k_idx]);
                    }
            }
            //b-->REG1 取1
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][1].vec[k_tile],0,(((i) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            }
          
            
            
            lgkmcnt_wait(0);//等待完成即可mmac与下次的执行进行遮掩
            vmcnt_wait(0);
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][1].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    inline_ds_read_b128_no_wait(A_lds + A_lds_stage_offset_loop +(precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx]), A_reg[min_tile_m][0].int4_array[k_idx]);
                    }
            }
            //b-->REG1
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,(((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            } 
            lgkmcnt_wait(0);
            vmcnt_wait(0);
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                inline_ds_read_b128_no_wait(A_lds + A_lds_stage_offset_loop +(precompute_A_lds_offset[1]), A_reg[min_tile_m][1].int4_array[k_idx]);
                        }
            }
            //b-->REG1
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][1].vec[k_tile],0,(((i+2) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                }
            }              
            
            lgkmcnt_wait(0);
            vmcnt_wait(0);
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][1].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);
            if((i +3)< itk){
                //b-->REG1
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                        buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,(((i+3) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                    }
                }
                
                
            
                //A-->lds 0
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++){
                for(int k_idx = 0; k_idx < (WARP_K / MFMA_K / 2); k_idx++){         
                    inline_ds_read_b128_no_wait(A_lds + (precompute_A_lds_offset[min_tile_m * (WARP_K/(MFMA_K)) + k_idx]), A_reg[min_tile_m][0].int4_array[k_idx]);
                    }
            }
            }                
        }
        if((i +3)< itk){
            lgkmcnt_wait(0);
            vmcnt_wait(0);
            //mmac
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                            midval[ 4 * k_tile + N_PACKID].int_array[0] = first_val & 0xf0f0f0f0 ;
                            midval[ 4 * k_tile + N_PACKID].int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        }
                    }
                }
            s_setprio(1);
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                            C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2_asm<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&(midval[ 4 * k_tile + N_PACKID]).int8x8_array[0]), 
                                C_reg1[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                            }
                    }
                }
            }
            s_setprio(0);            
        }
        //----------------------------------------------------Write back----------------------------------------------------------// 
        if(warp_k_id == 0){
        float *b_scale_ptr = weight_scale_ptr + warp_n_id * WARP_N * 4 + col_id * 4;
        using vec_bf16_4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) unsigned short;
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            float a_scale = input_scale_ptr[std::min(token_index / top_k,  max_n_len_offset - 1) * stride_asm]; //计算M方向的偏移
            for (int i = 0; i < 4; i++) {
                floatx4 bscale4;
                if(row_id==0) bscale4=*(floatx4*)(b_scale_ptr+i*16);
                if(token_index < max_n_len_offset * top_k){
                    vec_bf16_4 D;
                    for (int j = 0; j < 4; j++) {
                        float bscale=bscale4[j];
                        bscale=__shfl(bscale,col_id*16);
                        ((bhalf_t*)&D)[j] = f32_to_bf16(static_cast<float> (((int*)&C_reg1[0][m_tile*4+j])[i]  ) * a_scale * bscale);
                    }
                    *reinterpret_cast<vec_bf16_4*>(&g_output[token_index * (seqlen_B_stride>>2) + warp_n_id * WARP_N * 4 + col_id * 4 + i * 16]) = D;
                }
            }
        }
        }    
    }
}


// MOE-DECODE-W4A8 针对小shape优化
template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AI8_PERCHANNEL_MARLIN_HIP_NN_DOWN_decode_kernel(
    const Element* input_ptr, 
    const int* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[WARP_M/16][1], // 2 = warpk/mmak
    int8_vec<WARP_K/8> B_reg[WARP_N/16][1], // 2 = warpk/mmak
    intx4 C_reg[][4 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    const int size_k = seqlen_A_stride;//7168
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_k_num = BLOCK_K/WARP_K;
    int warp_k_id = warp_id % warp_k_num;//0:0 1:1 2:0 3:1
    int warp_n_id = warp_id / warp_k_num;//0:0 1:0 2:1 3:1
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    int i = 0;
    int B_stride_k = WARP_K / 32;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
    }

    union_vec<Element,8> midval;
    int itk = size_k/warp_k_num/WARP_K; //112
    for(; i < itk ; i += 1){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                // A_reg[m_tile][1].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
                buffer_load_reg_dwordx4(input_ptr,A_reg[m_tile][0].int4_array[k_tile],0, g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,((i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                // B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
            }
        }
        vmcnt_wait(0);
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                        int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                        midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                        midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&midval.int8x8_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                        }
                }
            }
        }
    } 


    extern __shared__ int out_smem[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=65;
        constexpr int lds_offset = warp_k_num * 16 * lineoffset;
        if(warp_k_id>0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *(intx4*)(out_smem+(m_tile * MFMA_M) * lds_offset + n_tile * 16+ (warp_k_id-1 +  warp_n_id * warp_k_num)  * lineoffset * 16 +  row_id * lineoffset + col_id * 16  + k * 4) 
                          = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile + k];
                    }
                }
            }
        }
        __syncthreads();

        if(warp_k_id == 0){ //存在多个warp的warpkid=0，这些warp都需要和附近的warpknum块数据进行reduce
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num-1; k_tile++){//warp迭代
                        #pragma unroll
                        for(int k = 0; k < 4; k++){//4组数据按组迭代
                            intx4 temp = *(intx4*)(out_smem+ (m_tile * MFMA_M) * lds_offset + n_tile * 16 + ( k_tile +  warp_n_id * warp_k_num ) * lineoffset *16 + row_id * lineoffset + col_id * 16  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile + k ][i] += temp[i];
                            }
                        }
                    }
                }
            }
        }
    }
}

template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    int splitk,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AF8_PERCHANNEL_MARLIN_HIP_NN_UP_decode_kernel(
    const Element* input_ptr, 
    const int* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec_fp8<Element,WARP_K/4> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    int8_vec<WARP_K/8> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    floatx4 C_reg[][4 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    const int size_k = seqlen_A_stride;//7168
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_k_num = BLOCK_K/WARP_K;
    int warp_k_id = warp_id % warp_k_num;//0:0 1:1 2:0 3:1
    int warp_n_id = warp_id / warp_k_num;//0:0 1:0 2:1 3:1
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    int i = 0;
    int B_stride_k = WARP_K / 32;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                A_reg[m_tile][0].int4_array[k_tile] = *(intx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
        }
    }
    i=1; 
    union_vec_fp8<Element,8> midval;
    int itk = size_k/warp_k_num/WARP_K/splitk; //112
    //新增一个int8到fp8的映射表，因为是int4转的int8，实际上int8中只有16个数据。
    //该方案因为数组过大，需要额外的load加载和等待
    // static const uint8_t lookup_table[16] = {
    //     0xd0, 0xce, 0xcc, 0xca, 0xc8, 0xc4, 0xc0, 0xb8,  // key: -8 ~ -1
    //     0x00, 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e   // key:  0 ~  7
    // };
    // 定义 FP8 E4M3 查找表
    // 负数查找表（int4: -8, -7, -6, -5, -4, -3, -2, -1）
    // constexpr uint32_t neg0 = 0xb8c0c4c8; // -8, -7, -6, -5
    // constexpr uint32_t neg1 = 0xcaccced0; // -4, -3, -2, -1
    // // 正数查找表（int4: 0,1,2,3,4,5,6,7）
    // constexpr uint32_t pos0 = 0x4e4c4a48; // 7,6,5,4
    // constexpr uint32_t pos1 = 0x44403800; // 3,2,1,0

    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                A_reg[m_tile][1].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
            }
        }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                        int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                        // uint32_t sign = ((first_val & 0x88888888) | 0x66442200) >> 1;
                        // uint32_t lut_idx = first_val & 0x77777777;
                        // // =============================== 处理低4个 int4 (bit 0~3) ===============================
                        // uint32_t sign_low = sign & 0x0f0f0f0f;
                        // uint32_t lut_idx_low4 = lut_idx & 0x0f0f0f0f;
                        // uint32_t output_low_neg = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_low_neg)
                        //               : "v"(neg0), "v"(neg1), "v"(lut_idx_low4));

                        // uint32_t output_low_pos = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_low_pos)
                        //               : "v"(pos0), "v"(pos1), "v"(lut_idx_low4));

                        // uint32_t output_low = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_low)
                        //               : "v"(output_low_neg), "v"(output_low_pos), "v"(sign_low));

                        // // =============================== 处理高4个 int4 (bit 4~7) ===============================
                        // uint32_t sign_high = (sign >> 4) & 0x0f0f0f0f;
                        // uint32_t lut_idx_high4 = (lut_idx >> 4) & 0x0f0f0f0f;

                        // uint32_t output_high_neg = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_high_neg)
                        //               : "v"(neg0), "v"(neg1), "v"(lut_idx_high4));

                        // uint32_t output_high_pos = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_high_pos)
                        //               : "v"(pos0), "v"(pos1), "v"(lut_idx_high4));

                        // uint32_t output_high = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //               : "=v"(output_high)
                        //               : "v"(output_high_neg), "v"(output_high_pos), "v"(sign_high));
                        // midval.int_array[0] = output_low;
                        // midval.int_array[1] = output_high;
                        midval.int2_array[0] = amd_assembly_i4_to_inx2(first_val);
                        // midval.uint8_array[0] = lookup_table(midval.int8_array[0]);
                        // midval.uint8_array[1] = lookup_table(midval.int8_array[1]);
                        // midval.uint8_array[2] = lookup_table(midval.int8_array[2]);
                        // midval.uint8_array[3] = lookup_table(midval.int8_array[3]);
                        // midval.uint8_array[4] = lookup_table(midval.int8_array[4]);
                        // midval.uint8_array[5] = lookup_table(midval.int8_array[5]);
                        // midval.uint8_array[6] = lookup_table(midval.int8_array[6]);
                        // midval.uint8_array[7] = lookup_table(midval.int8_array[7]);
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac_fp8_fp8<Element>(
                                (&A_reg[m_tile][0].int2_array[k_tile]), 
                                (&midval.int2_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                   A_reg[m_tile][0].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16);
                    // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){  
                    B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
                    // B_reg[n_tile][0].vec[k_tile]  = *(intx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k + B_stride_k * warp_k_id + k_tile) * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + lane_id * 4);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                    int first_val = B_reg[n_tile][1].data[4*k_tile + N_PACKID];
                    // midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                    // midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                    // midval.uint8_array[0] = lookup_table(midval.int8_array[0]);
                    // midval.uint8_array[1] = lookup_table(midval.int8_array[1]);
                    // midval.uint8_array[2] = lookup_table(midval.int8_array[2]);
                    // midval.uint8_array[3] = lookup_table(midval.int8_array[3]);
                    // midval.uint8_array[4] = lookup_table(midval.int8_array[4]);
                    // midval.uint8_array[5] = lookup_table(midval.int8_array[5]);
                    // midval.uint8_array[6] = lookup_table(midval.int8_array[6]);
                    // midval.uint8_array[7] = lookup_table(midval.int8_array[7]);
                    // uint32_t sign = ((first_val & 0x88888888) | 0x66442200) >> 1;
                    // uint32_t lut_idx = first_val & 0x77777777;
                    // // =============================== 处理低4个 int4 (bit 0~3) ===============================
                    // uint32_t sign_low = sign & 0x0f0f0f0f;
                    // uint32_t lut_idx_low4 = lut_idx & 0x0f0f0f0f;
                    // uint32_t output_low_neg = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low_neg)
                    //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_low4));

                    // uint32_t output_low_pos = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low_pos)
                    //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_low4));

                    // uint32_t output_low = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low)
                    //                 : "v"(output_low_neg), "v"(output_low_pos), "v"(sign_low));

                    // // =============================== 处理高4个 int4 (bit 4~7) ===============================
                    // uint32_t sign_high = (sign >> 4) & 0x0f0f0f0f;
                    // uint32_t lut_idx_high4 = (lut_idx >> 4) & 0x0f0f0f0f;

                    // uint32_t output_high_neg = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high_neg)
                    //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_high4));

                    // uint32_t output_high_pos = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high_pos)
                    //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_high4));

                    // uint32_t output_high = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high)
                    //                 : "v"(output_high_neg), "v"(output_high_pos), "v"(sign_high));
                    // midval.int_array[0] = output_low;
                    // midval.int_array[1] = output_high;   
                    midval.int2_array[0] = amd_assembly_i4_to_inx2(first_val);                 
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac_fp8_fp8<Element>(
                                (&A_reg[m_tile][1].int2_array[k_tile]), 
                                (&midval.int2_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i + 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                    // int first_val = B_reg[n_tile][0].int_array[k_tile];
                    int first_val = B_reg[n_tile][0].data[4*k_tile + N_PACKID];
                    union_vec_fp8<Element,8> midval;
                    // midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                    // midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                    // midval.uint8_array[0] = lookup_table(midval.int8_array[0]);
                    // midval.uint8_array[1] = lookup_table(midval.int8_array[1]);
                    // midval.uint8_array[2] = lookup_table(midval.int8_array[2]);
                    // midval.uint8_array[3] = lookup_table(midval.int8_array[3]);
                    // midval.uint8_array[4] = lookup_table(midval.int8_array[4]);
                    // midval.uint8_array[5] = lookup_table(midval.int8_array[5]);
                    // midval.uint8_array[6] = lookup_table(midval.int8_array[6]);
                    // midval.uint8_array[7] = lookup_table(midval.int8_array[7]);
                    // uint32_t sign = ((first_val & 0x88888888) | 0x66442200) >> 1;
                    // uint32_t lut_idx = first_val & 0x77777777;
                    // // =============================== 处理低4个 int4 (bit 0~3) ===============================
                    // uint32_t sign_low = sign & 0x0f0f0f0f;
                    // uint32_t lut_idx_low4 = lut_idx & 0x0f0f0f0f;
                    // uint32_t output_low_neg = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low_neg)
                    //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_low4));

                    // uint32_t output_low_pos = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low_pos)
                    //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_low4));

                    // uint32_t output_low = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_low)
                    //                 : "v"(output_low_neg), "v"(output_low_pos), "v"(sign_low));

                    // // =============================== 处理高4个 int4 (bit 4~7) ===============================
                    // uint32_t sign_high = (sign >> 4) & 0x0f0f0f0f;
                    // uint32_t lut_idx_high4 = (lut_idx >> 4) & 0x0f0f0f0f;

                    // uint32_t output_high_neg = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high_neg)
                    //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_high4));

                    // uint32_t output_high_pos = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high_pos)
                    //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_high4));

                    // uint32_t output_high = 0;
                    // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                    //                 : "=v"(output_high)
                    //                 : "v"(output_high_neg), "v"(output_high_pos), "v"(sign_high));
                    // midval.int_array[0] = output_low;
                    // midval.int_array[1] = output_high;
                    midval.int2_array[0] = amd_assembly_i4_to_inx2(first_val);
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac_fp8_fp8<Element>(
                                (&A_reg[m_tile][0].int2_array[k_tile]), 
                                (&midval.int2_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                    }
                }
            }
        }        
    }
    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=65;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *(floatx4*)(out_smemf+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 16  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + k];
                    }
                }
            }
        
        __syncthreads();

        if(warp_k_id == 0){ //存在多个warp的warpkid=0，这些warp都需要和附近的warpknum块数据进行reduce
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num-1; k_tile++){//warp迭代
                        #pragma unroll
                        for(int k = 0; k < 4; k++){//4组数据按组迭代
                            floatx4 temp = *(floatx4*)(out_smemf + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 16  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + k][i] += temp[i];
                            }
                        }
                    }
                }
            }
        __syncthreads();
        }
    }
    
}

// MOE-DECODE-W4A8 针对小shape优化
template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_WI4AF8_PERCHANNEL_MARLIN_HIP_NN_DOWN_decode_kernel(
    const Element* input_ptr, 
    const int* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec_fp8<Element,WARP_K/4> A_reg[WARP_M/16][1], // 2 = warpk/mmak
    int8_vec<WARP_K/8> B_reg[WARP_N/16][1], // 2 = warpk/mmak
    floatx4 C_reg[][4 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    const int size_k = seqlen_A_stride;//7168
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int warp_k_num = BLOCK_K/WARP_K;
    int warp_k_id = warp_id % warp_k_num;//0:0 1:1 2:0 3:1
    int warp_n_id = warp_id / warp_k_num;//0:0 1:0 2:1 3:1
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    int i = 0;
    int B_stride_k = WARP_K / 32;
    // 定义 FP8 E4M3 查找表
    // 负数查找表（int4: -8, -7, -6, -5, -4, -3, -2, -1）
    constexpr uint32_t neg0 = 0xb8c0c4c8; // -8, -7, -6, -5
    constexpr uint32_t neg1 = 0xcaccced0; // -4, -3, -2, -1
    // 正数查找表（int4: 0,1,2,3,4,5,6,7）
    constexpr uint32_t pos0 = 0x4e4c4a48; // 7,6,5,4
    constexpr uint32_t pos1 = 0x44403800; // 3,2,1,0
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * WARP_N * 4 * 4 + row_id * 4 + (col_id % 2) * 128 +(col_id/2)*seqlen_B_stride;
    }

    union_vec_fp8<Element,8> midval;
    int itk = size_k/warp_k_num/WARP_K; //112
    for(; i < itk ; i += 1){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
                // A_reg[m_tile][1].int4_array[k_tile] = *(intx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
                buffer_load_reg_dwordx4(input_ptr,A_reg[m_tile][0].int4_array[k_tile],0, g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                buffer_load_reg_dwordx4(weight_ptr,B_reg[n_tile][0].vec[k_tile],0,((i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64) * sizeof(int));
                // B_reg[n_tile][1].vec[k_tile]  = *(intx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
            }
        }
        vmcnt_wait(0);
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 4; N_PACKID++){
                        int first_val = B_reg[n_tile][0].data[ 4 * k_tile + N_PACKID];
                        // midval.int_array[0] = first_val & 0xf0f0f0f0 ;
                        // midval.int_array[1] = (first_val << 4) & 0xf0f0f0f0;
                        // midval.uint8_array[0] = lookup_table(midval.int8_array[0]);
                        // midval.uint8_array[1] = lookup_table(midval.int8_array[1]);
                        // midval.uint8_array[2] = lookup_table(midval.int8_array[2]);
                        // midval.uint8_array[3] = lookup_table(midval.int8_array[3]);
                        // midval.uint8_array[4] = lookup_table(midval.int8_array[4]);
                        // midval.uint8_array[5] = lookup_table(midval.int8_array[5]);
                        // midval.uint8_array[6] = lookup_table(midval.int8_array[6]);
                        // midval.uint8_array[7] = lookup_table(midval.int8_array[7]);
                        // uint32_t sign = ((first_val & 0x88888888) | 0x66442200) >> 1;
                        // uint32_t lut_idx = first_val & 0x77777777;
                        // // =============================== 处理低4个 int4 (bit 0~3) ===============================
                        // uint32_t sign_low = sign & 0x0f0f0f0f;
                        // uint32_t lut_idx_low4 = lut_idx & 0x0f0f0f0f;
                        // uint32_t output_low_neg = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_low_neg)
                        //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_low4));

                        // uint32_t output_low_pos = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_low_pos)
                        //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_low4));

                        // uint32_t output_low = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_low)
                        //                 : "v"(output_low_neg), "v"(output_low_pos), "v"(sign_low));

                        // // =============================== 处理高4个 int4 (bit 4~7) ===============================
                        // uint32_t sign_high = (sign >> 4) & 0x0f0f0f0f;
                        // uint32_t lut_idx_high4 = (lut_idx >> 4) & 0x0f0f0f0f;

                        // uint32_t output_high_neg = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_high_neg)
                        //                 : "v"(neg0), "v"(neg1), "v"(lut_idx_high4));

                        // uint32_t output_high_pos = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_high_pos)
                        //                 : "v"(pos0), "v"(pos1), "v"(lut_idx_high4));

                        // uint32_t output_high = 0;
                        // asm volatile ("v_perm_b32 %0, %1, %2, %3"
                        //                 : "=v"(output_high)
                        //                 : "v"(output_high_neg), "v"(output_high_pos), "v"(sign_high));
                        // midval.int_array[0] = output_low;
                        // midval.int_array[1] = output_high;
                        midval.int2_array[0] = amd_assembly_i4_to_inx2(first_val);
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID] = mmac_fp8_fp8<Element>(
                                (&A_reg[m_tile][0].int2_array[k_tile]), 
                                (&midval.int2_array[0]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 4 + N_PACKID]);  
                        }
                }
            }
        }
    } 


    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=65;
        constexpr int lds_offset = warp_k_num * 16 * lineoffset;
        if(warp_k_id>0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *(floatx4*)(out_smemf+(m_tile * MFMA_M) * lds_offset + n_tile * 16+ (warp_k_id-1 +  warp_n_id * warp_k_num)  * lineoffset * 16 +  row_id * lineoffset + col_id * 16  + k * 4) 
                          = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile + k];
                    }
                }
            }
        }
        __syncthreads();

        if(warp_k_id == 0){ //存在多个warp的warpkid=0，这些warp都需要和附近的warpknum块数据进行reduce
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num-1; k_tile++){//warp迭代
                        #pragma unroll
                        for(int k = 0; k < 4; k++){//4组数据按组迭代
                            floatx4 temp = *(intx4*)(out_smemf+ (m_tile * MFMA_M) * lds_offset + n_tile * 16 + ( k_tile +  warp_n_id * warp_k_num ) * lineoffset *16 + row_id * lineoffset + col_id * 16  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile + k ][i] += temp[i];
                            }
                        }
                    }
                }
            }
        }
    }
}

template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  gemm_nt_two_stage_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[][2], // 2 = stage
    union_vec<Element,WARP_K/8> B_reg[][2], // 2 = stage
    // union_vec<Element,8> A_reg[][2], // 2 = warpk/mmak
    // union_vec<Element,8> B_reg[][2], // 2 = warpk/mmak
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int STAGE = 2;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    // block_k = warp_k*warp_k_num
    // k_stage =  warp_k * stage
    // 一次性读取warp_k
    int k_start = warp_k_id * WARP_K;
    
    // 提前读取token_idx到reg
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N + row_id) * size_k/2 + col_id * 8;
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    // read 0
    // load A
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        // int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        // int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
            //A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + k_start + col_id * 16 + k_tile * 64);
            //A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][0].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
        }
    }

    // load B
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        // int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            //B_reg[n_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + k_start + col_id * 16 + k_tile * 64);
            //B_reg[n_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row_B[n_tile] + k_tile * 64 + k_start);
            // buffer_load_reg_dwordx2(weight_ptr, B_reg[n_tile][0].int2_array[k_tile], (g_row_B[n_tile] + k_tile * 32 + k_start/2), 0);
            // buffer_load_reg_dwordx2(weight_ptr, B_reg[n_tile][0].int2_array[k_tile], (g_row_B[n_tile] + k_tile * 32 + k_start/2), 0);
            // B_reg[n_tile][0].int2_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx2(weight_ptr, 0, (g_row_B[n_tile] + k_tile * 32 + k_start/2));
        }

    }

    int8_t * tb;  
    int8x8_t b_vec;    
    const int stage_offset = warp_k_num * WARP_K;
    for( ; k_start < size_k - (STAGE - 1) * stage_offset * STAGE; k_start += stage_offset * STAGE){

        ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
        // load A
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            // int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            // int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ // todo 一个线程连续拷贝16还是隔开拷贝  col_id * 16 * WARP_K/64 + k_tile * 16
                //A_reg[m_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset + col_id * 16 + k_tile * 64);
                //A_reg[m_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][1].int4_array[k_tile], 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
               
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            // int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //B_reg[n_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset + col_id * 16 + k_tile * 64);
                //B_reg[n_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
                // buffer_load_reg_dwordx2(weight_ptr, B_reg[n_tile][1].int2_array[k_tile], g_row_B[n_tile] + (STAGE - 1) * stage_offset/2 + k_tile * 32 + k_start/2, 0);
            }
        }
        vmcnt_wait(WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K);

        ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    tb = (&B_reg[n_tile][0].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F);
                    b_vec[1] = ((tb[0] >> 4) & 0x0F);
                    b_vec[2] = (tb[1]& 0x0F);
                    b_vec[3] = ((tb[1]>> 4) & 0x0F);
                    b_vec[4] = (tb[2]& 0x0F);
                    b_vec[5] = ((tb[2]>> 4) & 0x0F);
                    b_vec[6] = (tb[3]& 0x0F);
                    b_vec[7] = ((tb[3]>> 4) & 0x0F);
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&b_vec), 
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }

        ///////////////////////////////////////////////////////////// read 0 ////////////////////////////////////////////////////////////
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            // int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            // int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ // todo 一个线程连续拷贝16还是隔开拷贝  col_id * 16 * WARP_K/64 + k_tile * 16
                //A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset * 2 + col_id * 16 + k_tile * 64);
                //A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset * 2 + k_tile * 64 + k_start);
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][0].int4_array[k_tile], 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset * 2 + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            //int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //B_reg[n_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset * 2 + col_id * 16 + k_tile * 64);
                //B_reg[n_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset * 2 + k_tile * 64 + k_start);
                // buffer_load_reg_dwordx2(weight_ptr, B_reg[n_tile][0].int2_array[k_tile], g_row_B[n_tile] + (STAGE - 1) * stage_offset + k_tile * 32 + k_start/2, 0);
            }
        }
        vmcnt_wait(WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K);

        ///////////////////////////////////////////////////////////// mmac 1 ////////////////////////////////////////////////////////////
        
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    tb = (&B_reg[n_tile][1].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F);
                    b_vec[1] = ((tb[0] >> 4) & 0x0F);
                    b_vec[2] = (tb[1]& 0x0F);
                    b_vec[3] = ((tb[1]>> 4) & 0x0F);
                    b_vec[4] = (tb[2]& 0x0F);
                    b_vec[5] = ((tb[2]>> 4) & 0x0F);
                    b_vec[6] = (tb[3]& 0x0F);
                    b_vec[7] = ((tb[3]>> 4) & 0x0F); 

                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&b_vec), 
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//

    if(k_start + stage_offset < size_k){ // size_k / block 为偶数
        ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
        // load A
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            // int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            // int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ // todo 一个线程连续拷贝16还是隔开拷贝  col_id * 16 * WARP_K/64 + k_tile * 16
                //A_reg[m_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset + col_id * 16 + k_tile * 64);
                //A_reg[m_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][1].int4_array[k_tile], 0, g_row_A[m_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            //int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //B_reg[n_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + k_start + (STAGE - 1) * stage_offset + col_id * 16 + k_tile * 64);
                //B_reg[n_tile][1].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
                // buffer_load_reg_dwordx2(weight_ptr, B_reg[n_tile][1].int2_array[k_tile], g_row_B[n_tile] + (STAGE - 1) * stage_offset/2 + k_tile * 32 + k_start/2, 0);
            }
        }
        vmcnt_wait(WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K);

        ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    tb = (&B_reg[n_tile][0].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F);
                    b_vec[1] = ((tb[0] >> 4) & 0x0F);
                    b_vec[2] = (tb[1]& 0x0F);
                    b_vec[3] = ((tb[1]>> 4) & 0x0F);
                    b_vec[4] = (tb[2]& 0x0F);
                    b_vec[5] = ((tb[2]>> 4) & 0x0F);
                    b_vec[6] = (tb[3]& 0x0F);
                    b_vec[7] = ((tb[3]>> 4) & 0x0F);
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&b_vec),
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }

        ///////////////////////////////////////////////////////////// mmac 1 ////////////////////////////////////////////////////////////
        vmcnt_wait(0);
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    tb = (&B_reg[n_tile][1].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F);
                    b_vec[1] = ((tb[0] >> 4) & 0x0F);
                    b_vec[2] = (tb[1]& 0x0F);
                    b_vec[3] = ((tb[1]>> 4) & 0x0F);
                    b_vec[4] = (tb[2]& 0x0F);
                    b_vec[5] = ((tb[2]>> 4) & 0x0F);
                    b_vec[6] = (tb[3]& 0x0F);
                    b_vec[7] = ((tb[3]>> 4) & 0x0F);
            
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                (&b_vec), 
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }
    }
    else{ // size_k / block 为奇数
    
        vmcnt_wait(0);
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    tb = (&B_reg[n_tile][0].int8_array[k_tile*4]);
                    b_vec[0] = (tb[0]& 0x0F);
                    b_vec[1] = ((tb[0] >> 4) & 0x0F);
                    b_vec[2] = (tb[1]& 0x0F);
                    b_vec[3] = ((tb[1]>> 4) & 0x0F);
                    b_vec[4] = (tb[2]& 0x0F);
                    b_vec[5] = ((tb[2]>> 4) & 0x0F);
                    b_vec[6] = (tb[3]& 0x0F);
                    b_vec[7] = ((tb[3]>> 4) & 0x0F);
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                (&b_vec),
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }
    }

    // warp间的规约 
    extern __shared__ int out_smem[]; // 声明lds信息
    if constexpr (warp_k_num > 1){
        constexpr int lds_n_offset = warp_k_num * BLOCK_N;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ // todo 连续拷贝还是隔开拷贝 -> 差别不大 n_tile*4 + col_id * 4 *  WARP_N / MFMA_N
                *(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
                //*(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }

        __syncthreads();


        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0, 0, 0, 0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            //out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N  + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ gemm_decode_four_stage -------------------------------------------------------- //
template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGE,  
    int GROUP_N,
    int GROUP_K,
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  gemm_nt_four_stage_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[][STAGE], // 2 = stage
    union_vec<Element,WARP_K/8> B_reg[][STAGE], // 2 = stage
    // union_vec<Element,8> A_reg[][2], // 2 = warpk/mmak
    // union_vec<Element,8> B_reg[][2], // 2 = warpk/mmak
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_stride; 
    int lane_id = threadIdx.x & 63; 
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K; 
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num; 


    int k_start = warp_k_id * WARP_K; 
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    const int stage_offset = warp_k_num * WARP_K;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N + row_id) * size_k/2 + col_id * 8;
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    #pragma unroll
    for(int c_idx = 0; c_idx < (WARP_M/MFMA_M)*(WARP_N/MFMA_N); c_idx++){
                C_reg[0][c_idx] = {0,0,0,0};
    }
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //global load version
                A_reg[m_tile][i].int4_array[k_tile] = *(intx4*)(input_ptr + g_row_A[m_tile] + k_tile * 64 + k_start);
                // bufferload ASM version 
                // buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][0].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start); 
                // vmcnt_wait(0);
                // bufferload  builtin version 
                // A_reg[m_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0,g_row_A[m_tile] + k_tile * 64 + k_start);
         }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //global load version
                B_reg[n_tile][i].int2_array[k_tile] = *(intx2*)(weight_ptr + g_row_B[n_tile] + k_tile * 32 + k_start/2);
                // bufferload ASM version 
                // buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][0].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 + k_start);
                // vmcnt_wait(0);
                // bufferload  builtin version        
                // B_reg[n_tile][0].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row_B[n_tile] + k_tile * 64 + k_start);
            }
            
        }
        k_start += stage_offset;
    }
    k_start -= stage_offset;

    for( ; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE){

        #pragma unroll
        for(int i = 0; i < STAGE; ++i){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                    A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile] = *(intx4*)(input_ptr + g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

            // load B
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                    B_reg[n_tile][(i+STAGE-1)%STAGE].int2_array[k_tile] = *(intx2*)(weight_ptr + g_row_B[n_tile] + (i + 1) * stage_offset / 2 + k_tile * 32 + k_start/2);
                }
            }


            ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    int8_t * tb = (&(B_reg[n_tile][i%STAGE].int8_array[k_tile*4]));
                    int8x8_t b_vec;
                    b_vec[0] = (tb[0] & 0x0F) - 8;
                    b_vec[1] = ((tb[0] >> 4) & 0x0F)  - 8;
                    b_vec[2] = (tb[1]& 0x0F)- 8;
                    b_vec[3] = ((tb[1]>> 4) & 0x0F) - 8;
                    b_vec[4] = (tb[2]& 0x0F)- 8;
                    b_vec[5] = ((tb[2]>> 4) & 0x0F) - 8;
                    b_vec[6] = (tb[3]& 0x0F)- 8;
                    b_vec[7] = ((tb[3]>> 4) & 0x0F) - 8;
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i%STAGE].int8x8_array[k_tile]), 
                                    (&b_vec),
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        } // stage end

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//
    if(k_start + stage_offset < size_k){ // size_k / block 为偶数

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                A_reg[m_tile][(STAGE-1)].int4_array[k_tile] = *(intx4*)(input_ptr + g_row_A[m_tile] + stage_offset + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                B_reg[n_tile][(STAGE-1)].int2_array[k_tile] = *(intx2*)(weight_ptr + g_row_B[n_tile] + stage_offset/2 + k_tile * 32 + k_start/2);
            }
        }

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    int8_t * tb = (&B_reg[n_tile][0].int8_array[k_tile*4]); 
                    int8x8_t b_vec;
                    b_vec[0] = (tb[0]& 0x0F) - 8 ;
                    b_vec[1] = ((tb[0] >> 4) & 0x0F) - 8 ;
                    b_vec[2] = (tb[1]& 0x0F) - 8 ;
                    b_vec[3] = ((tb[1]>> 4) & 0x0F) - 8 ;
                    b_vec[4] = (tb[2]& 0x0F) - 8 ;
                    b_vec[5] = ((tb[2]>> 4) & 0x0F) - 8 ;
                    b_vec[6] = (tb[3]& 0x0F) - 8 ;
                    b_vec[7] = ((tb[3]>> 4) & 0x0F) - 8 ;
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8x8_array[k_tile]), 
                                (&b_vec),
                                C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                }
            }
        }
        //  --------------------------------------------- load3 mmac ---------------------------------- //

        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int i = 1; i < STAGE; ++i){
            

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
            
                    int8_t * tb = (&B_reg[n_tile][i].int8_array[k_tile*4]);
                    int8x8_t b_vec;
                    b_vec[0] = (tb[0]& 0x0F) - 8;
                    b_vec[1] = ((tb[0] >> 4) & 0x0F) - 8;
                    b_vec[2] = (tb[1]& 0x0F) - 8;
                    b_vec[3] = ((tb[1]>> 4) & 0x0F) - 8;
                    b_vec[4] = (tb[2]& 0x0F) - 8;
                    b_vec[5] = ((tb[2]>> 4) & 0x0F) - 8;
                    b_vec[6] = (tb[3]& 0x0F) - 8;
                    b_vec[7] = ((tb[3]>> 4) & 0x0F) - 8;
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8x8_array[k_tile]), 
                                    (&b_vec),
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }

        }

    }
    else{ // todo 不为block_k的倍数
        // if(threadIdx.x == 0 && bidx == 0)
        //     printf("run 1041\n");
        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int i = 1; i < STAGE; ++i){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    int8_t * tb = (&B_reg[n_tile][i-1].int8_array[k_tile*4]);
                    int8x8_t b_vec;
                    b_vec[0] = (tb[0]& 0x0F) - 8;
                    b_vec[1] = ((tb[0] >> 4) & 0x0F) - 8;
                    b_vec[2] = (tb[1]& 0x0F) - 8;
                    b_vec[3] = ((tb[1]>> 4) & 0x0F) - 8;
                    b_vec[4] = (tb[2]& 0x0F) - 8;
                    b_vec[5] = ((tb[2]>> 4) & 0x0F) - 8;
                    b_vec[6] = (tb[3]& 0x0F) - 8;
                    b_vec[7] = ((tb[3]>> 4) & 0x0F) - 8;
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac2<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i-1].int8t_array[k_tile]), 
                                    (&b_vec),
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        }
    }

    // warp间的规约 
    extern __shared__ int out_smem[]; // 声明lds信息
    if constexpr (warp_k_num > 1){
        constexpr int lds_n_offset = warp_k_num * BLOCK_N;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ // todo 连续拷贝还是隔开拷贝 -> 差别不大 n_tile*4 + col_id * 4 *  WARP_N / MFMA_N
                *(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
                //*(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }

        __syncthreads();


        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0, 0, 0, 0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            //out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N  + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                        }
                    }
                }
            }
        }
    }
    }
} //namespace end
} //namespace end
