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

                                                                                   
#define buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, MARLIN_n_stride, BLOCK_N, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, marlin_warp_stride, warp_id, lane_id,\
    marlin_warp_n, marlin_warp_k)\
{\
    const int element_num_per_thread = 8;\
    int bytes_per_Element = 2;\
    const int lds_load_num = (BLOCK_N*WARP_K*bytes_per_Element) / (element_num_per_thread*64*bytes_per_Element);\
    int warp_num_n = BLOCK_N/marlin_warp_n;\
    int warp_num_k = lds_load_num/warp_num_n;\
    if(WARP_NUM > lds_load_num){ \
        if(warp_id < lds_load_num){ \ 
            int warp_n_id = warp_id % warp_num_n;\
            int warp_k_id = warp_id / warp_num_n;\
            int gsOffset   = global_offset;\
            int gvOffset   = (warp_k_id * MARLIN_n_stride + warp_n_id * marlin_warp_stride + lane_id * element_num_per_thread * bytes_per_Element);\
            int lds_offset = lds_stage_offset + (warp_n_id * warp_num_k + warp_k_id) * 64 * element_num_per_thread * bytes_per_Element;\
            buffer_load_lds_dwordx4(global_ptr,lds_ptr, (lds_offset/bytes_per_Element), gvOffset, gsOffset);\
        } \
    }else{ \
        for(int load = 0; load < lds_load_num/WARP_NUM; ++load) {\
            int warp_linear  = (load * WARP_NUM + warp_id); \
            int warp_n_id = warp_linear % warp_num_n; \
            int warp_k_id = warp_linear / warp_num_n; \
            int gsOffset   = global_offset;\
            int gvOffset   = warp_k_id * MARLIN_n_stride + warp_n_id * marlin_warp_stride + lane_id * element_num_per_thread * bytes_per_Element;\
            int lds_offset = lds_stage_offset + (warp_n_id * warp_num_k + warp_k_id) * 64 * element_num_per_thread * bytes_per_Element;\
            buffer_load_lds_dwordx4(global_ptr, lds_ptr, (lds_offset/bytes_per_Element), gvOffset, gsOffset);\
        } \
    }\
}


#define buffer_load_lds_tile_pad_weight_marlin_remiander(WARP_NUM, MARLIN_n_stride, BLOCK_N, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, marlin_warp_stride, warp_id, lane_id,\
    marlin_warp_n, marlin_warp_k,warp_n_id_real)\
{\
    const int element_num_per_thread = 8;\
    int bytes_per_Element = 2;\
    const int lds_load_num = (BLOCK_N*WARP_K*bytes_per_Element) / (element_num_per_thread*64*bytes_per_Element);\
    int warp_num_n = BLOCK_N/marlin_warp_n;\
    int warp_num_k = lds_load_num/warp_num_n;\
    if(WARP_NUM > lds_load_num){ \
        if(warp_id < lds_load_num){ \ 
            int warp_n_id = warp_id % warp_num_n;\
            int warp_k_id = warp_id / warp_num_n;\
            int gsOffset   = global_offset;\
            int gvOffset   = (warp_k_id * MARLIN_n_stride + std::min(warp_n_id,warp_n_id_real) * marlin_warp_stride + lane_id * element_num_per_thread * bytes_per_Element);\
            int lds_offset = lds_stage_offset + (warp_n_id * warp_num_k + warp_k_id) * 64 * element_num_per_thread * bytes_per_Element;\
            buffer_load_lds_dwordx4(global_ptr,lds_ptr, (lds_offset/bytes_per_Element), gvOffset, gsOffset);\
        } \
    }else{ \
        for(int load = 0; load < lds_load_num/WARP_NUM; ++load) {\
            int warp_linear  = (load * WARP_NUM + warp_id); \
            int warp_n_id = warp_linear % warp_num_n; \
            int warp_k_id = warp_linear / warp_num_n; \
            int gsOffset   = global_offset;\
            int gvOffset   = warp_k_id * MARLIN_n_stride + std::min(warp_n_id,warp_n_id_real) * marlin_warp_stride + lane_id * element_num_per_thread * bytes_per_Element;\
            int lds_offset = lds_stage_offset + (warp_n_id * warp_num_k + warp_k_id) * 64 * element_num_per_thread * bytes_per_Element;\
            buffer_load_lds_dwordx4(global_ptr, lds_ptr, (lds_offset/bytes_per_Element), gvOffset, gsOffset);\
        } \
    }\
}

#define ds_read_b128_tile_pad_no_wait(M,k_idx,WARP_NUM,Element,lds_v4bf16,precompute_offset,lds_stage_offset_loop,reg)\
{\
for(int m_idx = 0; m_idx < M / marlin_warp_n; m_idx ++){\
    inline_ds_read_b128_no_wait_marlin(lds_v4bf16, precompute_offset[m_idx * (WARP_K/marlin_warp_k) + k_idx] + lds_stage_offset_loop, reg[m_idx][k_idx].int4_array[0]);\
}\
}

#define ds_read2_tile_pad_no_wait(M,n_idx,WARP_NUM,Element,lds_v8i8,precompute_offset,lds_stage_offset_loop,reg,loop)\
{\
for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
    inline_ds_read2_b32_no_wait(lds_v8i8, precompute_offset[m_idx * (WARP_K/MFMA_K) + n_idx] + lds_stage_offset_loop, reg[loop * (M / 16) + m_idx][n_idx].int2_array[0], 1);\
}\
}


using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;

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
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NN_UP_decode_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    f16_vec<WARP_K/2,Element> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    floatx4 C_reg[][2 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_k: 12288
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    const int size_k = seqlen_A_stride;//2048  sizeofkb是128
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1

    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    int i = 0;
    int B_stride_k = WARP_K / 16;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * WARP_K / 4; 
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * 64 * 8 + lane_id * 8;
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }   
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
        }
    }
    i=1; 
    int itk = size_k/warp_k_num/WARP_K; 

    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                A_reg[m_tile][1].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }

        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                   A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                   // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
                // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
                // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i+1,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
                // }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64); 
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){                 
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }        
    }
    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=33;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 2; k++){
                        *(floatx4*)(out_smemf+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 8  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k];
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
                        for(int k = 0; k < 2; k++){//4组数据按组迭代
                            floatx4 temp = *(floatx4*)(out_smemf + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 8  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k][i] += temp[i];
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
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NN_DOWN_decode_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    f16_vec<WARP_K/2,Element> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    floatx4 C_reg[][2 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_k: 12288
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const float* topk_weights) 
{
    const int size_k = seqlen_A_stride;//6144 sizeofkb是5
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N/WARP_N/2; //1
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    int i = 0;
    int B_stride_k = WARP_K / 16;
    int k_block = warp_k_num * WARP_K; //64
    int itk = size_k/ k_block ; //2
    int remainder = size_k % k_block; 
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * WARP_K / 4; 
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * 64 * 8 + lane_id * 8;
    }
    if(itk!=0){
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
        }
    }
    }
    i=1; 


    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                A_reg[m_tile][1].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
        // printf(" 迭代%d次 first read A_reg[m_tile][1],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][1].data[0]), static_cast<float>(A_reg[0][1].data[1]), static_cast<float>(A_reg[0][1].data[2]), static_cast<float>(A_reg[0][1].data[3]));
        // }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64);
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                   A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                   // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
                // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
                // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i+1,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
                // }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64); 
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){                 
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }        
    }

    if(remainder > 0){
        //仅迭代一次，不需要进行预取，直接读取然后计算写回即可
        int k0 = (size_k/ k_block) * k_block; //128
        if((k0 + warp_k_id * WARP_K) < size_k) {    
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                   A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + k0  + k_tile * 16); 
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (itk * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 64); 
                   }
            }
        
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        #pragma unroll
                        for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){                 
                            C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                        }
                    }
                }
            }
        }                    
    }
    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=33;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 2; k++){
                        *(floatx4*)(out_smemf+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 8  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k];
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
                        for(int k = 0; k < 2; k++){//4组数据按组迭代
                            floatx4 temp = *(floatx4*)(out_smemf + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 8  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k][i] += temp[i];
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
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NN_UP_decode_kpack2_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    f16_vec<WARP_K/2,Element> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    floatx4 C_reg[][2 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_k: 12288
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) 
{
    //此kernel适配kpack2即warpk=32
    const int size_k = seqlen_A_stride;//2048  sizeofkb是128
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1

    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    int i = 0;
    int B_stride_k = WARP_K / 16;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * WARP_K / 4; 
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * 64 * 8 + row_id * 8 + (col_id % 2)* 256 + (col_id/2)*seqlen_B_stride;
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }
   
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128);
        }
    }
    i=1; 
    int itk = size_k/warp_k_num/WARP_K; 

    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
        // printf(" 迭代%d次 first read A_reg[m_tile][1],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][1].data[0]), static_cast<float>(A_reg[0][1].data[1]), static_cast<float>(A_reg[0][1].data[2]), static_cast<float>(A_reg[0][1].data[3]));
        // }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128);
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                   // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
                // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
                // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i+1,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
                // }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128); 
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){                 
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }        
    }
    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=33;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 2; k++){
                        *(floatx4*)(out_smemf+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 8  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k];
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
                        for(int k = 0; k < 2; k++){//4组数据按组迭代
                            floatx4 temp = *(floatx4*)(out_smemf + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 8  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k][i] += temp[i];
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
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NN_DOWN_decode_kpack2_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 2 = warpk/mmak
    f16_vec<WARP_K/2,Element> B_reg[WARP_N/16][2], // 2 = warpk/mmak
    floatx4 C_reg[][2 * (WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_k: 12288
    int top_k, // 8/1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const float* topk_weights) 
{
    //此kernel适配kpack2即warpk=32
    const int size_k = seqlen_A_stride;//2048  sizeofkb是128
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1

    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    int i = 0;
    int B_stride_k = WARP_K / 16;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * WARP_K / 4; 
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  warp_k_id * B_stride_k * seqlen_B_stride + warp_n_id * 64 * 8 + row_id * 8 + (col_id % 2)* 256 + (col_id/2)*seqlen_B_stride;
    }
    int itk = size_k/BLOCK_K; 
    if(itk!=0){
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }
   
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128);
        }
    }
    }
    i=1;
    // int k_block =  warp_k_num * WARP_K;
    
    int remainder = size_k % BLOCK_K; 
    for(; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }
        // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
        // printf(" 迭代%d次 first read A_reg[m_tile][1],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][1].data[0]), static_cast<float>(A_reg[0][1].data[1]), static_cast<float>(A_reg[0][1].data[2]), static_cast<float>(A_reg[0][1].data[3]));
        // }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (i * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128);
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                        }
                }
            }
        }
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                   // A_reg[m_tile][0].int2_array[k_tile] = *(intx2*)(input_ptr+ g_row_A[m_tile] + (i + 1)* warp_k_num * WARP_K + warp_k_id  * WARP_K + k_tile * MFMA_K );
                }
            }
                // if(bidx==0 && bidy==0 && warp_id==0 && threadIdx.x==0){
                // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i+1,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
                // }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + ((i+1) * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128); 
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
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){                 
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }        
    }

    if(remainder > 0){
        //仅迭代一次，不需要进行预取，直接读取然后计算写回即可
        int k0 = (size_k/ BLOCK_K) * BLOCK_K; //128
        if((k0 + warp_k_id * WARP_K) < size_k) {

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + k0  + k_tile * 16); 
                   }
            }

            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + (itk * warp_k_num * B_stride_k) * seqlen_B_stride + g_row_B[n_tile] + k_tile * 128); 
                    }
            }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    #pragma unroll
                    for(int N_PACKID = 0; N_PACKID < 2; N_PACKID++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[2 * k_tile + N_PACKID]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + N_PACKID]);  
                    }
                }
            }
        }
        }                    
    }
    extern __shared__ float out_smemf[]; // 声明lds信息

    if constexpr (warp_k_num > 1){
        //计算一个warp需要的空间即：64x16
        constexpr int lineoffset=33;
        // constexpr int lds_offset = (warp_k_num-1) * warp_n_num * lineoffset;
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            if(warp_k_id>0){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k = 0; k < 2; k++){
                        *(floatx4*)(out_smemf+ n_tile * 16+ (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * lineoffset * 16 +  row_id * lineoffset + col_id * 8  + k * 4) 
                          = C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k];
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
                        for(int k = 0; k < 2; k++){//4组数据按组迭代
                            floatx4 temp = *(floatx4*)(out_smemf + n_tile * 16 + ( k_tile +  warp_n_id * (warp_k_num-1) ) * lineoffset *16 + row_id * lineoffset + col_id * 8  + k * 4);
                            #pragma unroll
                            for(int i = 0; i < 4; i++){ //组内迭代
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile) * 2 + k][i] += temp[i];
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
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_UP_decode_kpack2_kernel2(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 
    int warp_n_id = warp_id / warp_k_num; // 0 

    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[WARP_N / MFMA_N]; //1
    
    int i = 0;
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  (n_tile * MFMA_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
            A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
        }
    }
    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
            }
        }
        // if(bidx ==0 && bidy==0 && warp_id ==0 && lane_id ==0){
        // S_BARRIER;S_BARRIER;S_BARRIER;
        // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
        // printf("  迭代%d次 first read B_reg[n_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(B_reg[0][0].data[0]),static_cast<float>( B_reg[0][0].data[1]), static_cast<float>(B_reg[0][0].data[2]), static_cast<float>(B_reg[0][0].data[3]));
        // }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                }
            }
        }

        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 32); 
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 32);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                }
            }        
        }
    }
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
        

        //假设这里是4个warp，blockm n 都是16 而blockk是16*2 =32 而对应的warp都是16 写入到kwarp个独立的空间中
        constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 * lds_n_offset + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();
        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 * lds_n_offset + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id];
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        // *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8]);
      }
    } 
  } 

}

template<bool Is_store_A, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_UP_decode_kpack2_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[BLOCK_N/WARP_N][WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 
    int warp_n_id = warp_id / warp_k_num; // 0 

    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[BLOCK_N / WARP_N][WARP_N / MFMA_N]; //1 
    int i = 0;
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[warp_n_id][n_tile] =  (warp_n_id * MFMA_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
            A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
           B_reg[warp_n_id][n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
        }
    }
    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                B_reg[warp_n_id][n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
            }
        }
        // if(bidx ==0 && bidy==0 && warp_id ==0 && lane_id ==0){
        // S_BARRIER;S_BARRIER;S_BARRIER;
        // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
        // printf("  迭代%d次 first read B_reg[n_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(B_reg[0][0].data[0]),static_cast<float>( B_reg[0][0].data[1]), static_cast<float>(B_reg[0][0].data[2]), static_cast<float>(B_reg[0][0].data[3]));
        // }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                }
            }
        }

        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 32); 
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                     B_reg[warp_n_id][n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 32);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                    C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                }
            }        
        }
    }
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
        
        //假设这里是4个warp，blockm n 都是16 而blockk是16*2 =32 而对应的warp都是16 写入到kwarp个独立的空间中
        constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();

        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[warp_n_id][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id];
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        // *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8]);
      }
    } 
  } 

}



template<bool Is_store_A, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_UP_decode_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[BLOCK_N/WARP_N][WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //2
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 1 
    int warp_n_id = warp_id / warp_k_num; // 0

    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[BLOCK_N / WARP_N][WARP_N / MFMA_N]; //1 

    int i = 0; 
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            g_row_B[warp_n_id][n_tile] =  (warp_n_id * WARP_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
        }
    
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }

        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            B_reg[warp_n_id][n_tile][0].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 16);
            }
        }
    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                A_reg[m_tile][1].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
            }
        }  
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                    B_reg[warp_n_id][n_tile][1].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 16);
                }
            }
  
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                    }
                }
            }
        
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                   A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                }
            }            
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                        B_reg[warp_n_id][n_tile][0].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 16);
                    }
                }
            }             
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                    }
                }
            }
        
        
    } 
    //尾处理
    if((i - 1) < itk){        
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                    }
                }        
            }
    }
    
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
    constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32  
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  //一行存64个数 相同warpn的数存储在一起
                                        + warp_n_id * WARP_N + n_tile * 16  
                                        + col_id * 4  
                                        + warp_k_id * BLOCK_N] ) = C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();
        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[warp_n_id][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id];
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      

//     if(warp_k_id == 0 || warp_k_num == 1){
//     for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
//         for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
//         int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
//           if(token_index < max_n_len_offset * top_k){
//             for (int j = 0; j < 4; j++) {
//                 Element D = f32_to_f16<Element>((C_reg[0][(m_tile*(WARP_N/MFMA_N)) + n_tile][j]));
//                 *reinterpret_cast<Element*>(&g_output[token_index * size_n + n_tile * 16 + col_id + j * 4 ]) = D;
//             }
//           }
//         }
//    }     
// }
}





// #define buffer_load_lds_tile_pad_split(WARP_NUM, WARP_NUM_N, N_row_len, BLOCK_N, WARP_K, WARP_N, Element, global_ptr, lds_ptr, lds_stage_offset, max_M_len, warp_id, lane_id,vec_size,per_phase,max_phase)\
// {\
//     int bytes_per_Element = 1;\
//     int Element_per_dword = 4/bytes_per_Element;\
//     int lane_M_idx = lane_id / 8;\
//     int lane_N_idx = lane_id % 8;\
//     for(int load = 0,warp_loop = warp_id; load < (WARP_N * WARP_NUM_N * WARP_K * bytes_per_Element) / (/*2 dword*/ 8 * 64)/WARP_NUM; warp_loop += WARP_NUM,load++) {\
//             int block_row =  warp_loop * 8 + lane_M_idx;\
//             int g_col =lane_N_idx;\
//             int phase = (block_row/per_phase) % max_phase; \
//             int vec_size_dword = vec_size / Element_per_dword; \ 
//             int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
//             int gsOffset = 0;\
//             int gvOffset = block_row * N_row_len/2 / Element_per_dword + g_col_swizzle ;\
//             int lds_offset = warp_loop* 8 * WARP_K + lds_stage_offset;\ 
//             amdgcn_buffer_load_dwordx2_lds((lds_ptr+lds_offset),(global_ptr),(gvOffset*4),(gsOffset));\
//     }\
// }


// #define buffer_load_lds_tile_pad_sorted_token(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
//                                                 vec_size, per_phase, max_phase)\
// {\
//     int bytes_per_Element = 2;\
//     const int bytes_num_per_dword = 4;\
//     const int bytes_num_per_dwordx2 = 8;\
//     int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dwordx2;\
//     int thread_num_m = 64/thread_num_n;\
//     int lane_M_idx = lane_id / thread_num_n;\ 
//     int lane_N_idx = lane_id % thread_num_n;\
//     const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dwordx2*64);\
//     for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
//         int block_row = warp_loop * thread_num_m + lane_M_idx;\
//         int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
//         int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
//         int phase = (block_row/per_phase) % max_phase; \
//         int g_col = lane_N_idx; \
//         int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
//         int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
//         int gsOffset   = global_offset/bytes_num_per_dword;\
//         int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
//         int lds_offset = lds_stage_offset/bytes_num_per_dword + warp_loop * 64 * 2;\
//         builtin_buffer_load_dword_lds2(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
//     }\
// }

//直接变成4x16的了 每个线程都是dwordx4 col为16
//读取的是16x128的数据
//需要4个线程
//perphas=1 一行数据就是128字了
//maxphase 8
#define buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    const int bytes_num_per_dwordx4 = 16;\
    int thread_num_n = WARP_K * bytes_per_Element / bytes_num_per_dwordx4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M * WARP_K * bytes_per_Element) / (bytes_num_per_dwordx4 * 64);\
    if(warp_id < lds_load_num){\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row / per_phase ) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
        int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
        int gsOffset   = global_offset/bytes_num_per_dword;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_num_per_dword + warp_loop * 64 * vec_size_dword;\
        builtin_buffer_load_dword_lds4(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
    }\
}

#define buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    const int bytes_num_per_dwordx2 = 8;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dwordx2;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dwordx2*64);\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
        int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
        int gsOffset   = global_offset;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_per_Element + warp_loop * 64 * 4;\
        buffer_load_lds_dwordx2(global_ptr,lds_ptr,  lds_offset, (gvOffset*4), gsOffset);\
    }\
}



#define buffer_load_lds_tile_pad_sorted_token_dwordx2_blocksizek32(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    const int bytes_num_per_dwordx2 = 8;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dwordx2;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dwordx2*64);\
    if(warp_id < lds_load_num){\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
        int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
        int gsOffset   = global_offset;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_per_Element + warp_loop * 64 * 4;\
        buffer_load_lds_dwordx2(global_ptr,lds_ptr,  lds_offset, (gvOffset*4), gsOffset);\
    }\
    }\
}
//这里是2x32的线程layout布局
//4个warp，迭代2次读取完成 一共是16x64的数据
#define buffer_load_lds_tile_pad_sorted_token(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dword;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dword*64);\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/bytes_num_per_dword;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_num_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}


#define buffer_load_lds_tile_pad_sorted_token_asm(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dword;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dword*64);\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
        int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)); \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_per_Element + warp_loop * 64 * 2;\
        buffer_load_lds_dwordx1(global_ptr, lds_ptr,  lds_offset,(gvOffset*4), gsOffset);\
    }\
}

// B矩阵buffer_load读                                                                                          
#define buffer_load_lds_tile_pad_weight(WARP_NUM, N_row_len, BLOCK_N, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dword;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_N*WARP_K*bytes_per_Element) / (bytes_num_per_dword*64);\  
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int g_row = block_row; \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size * bytes_per_Element / bytes_num_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/bytes_num_per_dword;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_num_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

#define buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, N_row_len, BLOCK_N, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    const int bytes_num_per_dwordx4 = 16;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dwordx4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_N*WARP_K*bytes_per_Element) / (bytes_num_per_dwordx4*64);\
    if(warp_id<lds_load_num){\  
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int g_row = block_row; \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size * bytes_per_Element / bytes_num_per_dword; \
        int g_col_swizzle = (g_col ^ phase) * vec_size_dword; \
        int gsOffset   = global_offset/bytes_num_per_dword;\
        int gvOffset   = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_num_per_dword + warp_loop * 64 * vec_size_dword;\
        builtin_buffer_load_dword_lds4(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
    }\
}

#define ds_read2_tile_pad_no_wait(M,n_idx,WARP_NUM,Element,lds_v4bf16,precompute_offset,lds_stage_offset_loop,reg,loop)\
{\
for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
    inline_ds_read2_b32_no_wait(lds_v4bf16, precompute_offset[m_idx * (WARP_K/MFMA_K) + n_idx] + lds_stage_offset_loop, reg[loop * (M / 16) + m_idx][n_idx].int2_array[0], 1);\
}\
}

#define ds_read_b128_tile_pad_no_wait_dword_lds4(M,n_idx,WARP_NUM,Element,lds_v8bf16,precompute_offset,lds_stage_offset_loop,reg,loop)\
{\
for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
    inline_ds_read_b128_no_wait(lds_v8bf16 + (precompute_offset[m_idx * (WARP_K/MFMA_K/2) + n_idx] / 4) + lds_stage_offset_loop, reg[loop * (M / 16) + m_idx][n_idx].int4_array[0]);\
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
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_prefill_kpack2_stage2_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,8> A_reg[][4], 
    union_vec<Element,8> B_reg[][4], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    constexpr int bytes_per_dwordx4 = 16;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;


    const int warp_n_num = BLOCK_N / WARP_N; //4

    int warp_n_id = warp_id % warp_n_num;//0 1 2 3 
    int warp_m_id = warp_id / warp_n_num;//0

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K/2)]; //4 一次线程读取四次，算8次
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K/2)]; //4 一次线程读取四次，算8次

    typename vec<Element,8>::type *B_lds_v8bf16 = (typename vec<Element,8>::type *)(B_lds); //stride为16字节的指针
    typename vec<Element,8>::type *A_lds_v8bf16 = (typename vec<Element,8>::type *)(A_lds); //stride为16字节的指针

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit; //64xf16
    const int simd_width = 16; //MMAC LAYOUT :ROW LEN = 16
    const int inner_dim_length = WARP_K;
    const int vec_size = 8; //element per thread
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //1
    const int max_phase0 = std::min(simd_width / per_phase, inner_dim_length / vec_size); //16
    const int max_phase = std::min(max_phase0,32 / (vec_size * type_width_in_bit / bank_bit_width)); // 8

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element;  //128x128
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); 
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);

            }
        }
    }
    
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K/2); min_tile_k++) {

            int lds_col = (min_tile_k * MFMA_K * 2 ) / 8 + (lane_id / 16) ;
            int lds_row = (lane_id & 15);
            int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;

            int phase = (lds_row/per_phase) % max_phase;
            
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;

            int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K/2) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
{
    //16X64
    // #pragma unroll 
    // for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
    //     #pragma unroll
    //     for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
    //         int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword/*dword*/ + (lane_id/16)*2/*每个线程持有4个bf16,共2个dword*/;
    //         int lds_row = (lane_id & 15);
    //         int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
    //         int phase = (lds_row/per_phase) % max_phase;
    //         int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
    //         int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
    //         precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
    //     }
    // }
    // 构建B矩阵读取索引 64X64
    // #pragma unroll 
    // for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
    //     #pragma unroll
    //     for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
    //         {
    //             int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword/*dword*/ + (lane_id/16)*2/**每个线程持有4个bf16,共2个dword*/;
    //             int lds_row = (lane_id & 15);
    //             int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
    //             int phase = (lds_row/per_phase) % max_phase;
    //             int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
    //             int lds_offset = warp_n_id * (WARP_N*WARP_K*bytes_per_element)/bytes_per_dword + min_tile_n * (MFMA_N*WARP_K*bytes_per_element/bytes_per_dword) + lds_row * (WARP_K*bytes_per_element/bytes_per_dword) + col_swizzle;/*dword*/
    //             precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K) + min_tile_k] = lds_offset;
    //         } 
    //     }
    // }
}
    //每个线程也要读取四次
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K/2); min_tile_k++) { 
            {
                int lds_col = (min_tile_k * MFMA_K * 2) / 8 + (lane_id/16);
                int lds_row = (lane_id & 15);
                int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
                int phase = (lds_row/per_phase) % max_phase;
                int col_swizzle = ((lds_col) ^ phase) * vec_size_dword;
                int lds_offset = warp_n_id * (WARP_N*WARP_K*bytes_per_element)/bytes_per_dword + min_tile_n * (MFMA_N*WARP_K*bytes_per_element/bytes_per_dword) + lds_row * (WARP_K*bytes_per_element/bytes_per_dword) + col_swizzle;/*dword*/
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K/2) + min_tile_k] = lds_offset;
            } 
        }
    }

    int itk = size_k/WARP_K;
    int k=1;
    for(; k < itk; k+=2){
        int k_offset = k * WARP_K;
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = 0; 
            int A_lds_stage_offset_loop = A_lds_stage_offset;

            buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = 0; 
            int B_lds_stage_offset_loop = B_lds_stage_offset; 

            buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                    vec_size, per_phase, max_phase);  
        }

        vmcnt_wait(((BLOCK_N * WARP_K) / (vec_size*64) / WARP_NUM + ((BLOCK_M * WARP_K) / (vec_size * 64)/WARP_NUM)) * (STAGES - 1));

        stage_id_reg_A = 0;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K / 2; k_idx++){ 
            { // 加载A矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dwordx4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dwordx4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K/2; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) {
                    for(int kpack2=0; kpack2<2; kpack2++) {
                        C_reg[0][min_tile_m*(WARP_N / MFMA_N) + min_tile_n] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[kpack2*4]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[kpack2*4]), 
                            C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]);        
                    }
                }
            }
        }

        if((k+1) < itk)
        {
            k_offset = (k+1) * WARP_K;
            auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
            auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset); // 配置全局显存信息
            { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = 0; 
            int A_lds_stage_offset_loop = 0;

            buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = 0; 
            int B_lds_stage_offset_loop = 0; 

            buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                    vec_size, per_phase, max_phase);  
            }
            vmcnt_wait(((BLOCK_N * WARP_K) / (vec_size*64) / WARP_NUM + ((BLOCK_M * WARP_K) / (vec_size * 64)/WARP_NUM)) * (STAGES - 1));            
        }
        else{
            vmcnt_wait(0);
        }

        {
        stage_id_reg_A = 1;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K / 2; k_idx++){ 
            // 加载A矩阵
            ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dwordx4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dwordx4, B_reg, stage_id_reg_B);
        }
        }

        lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K/2; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                for(int kpack2=0; kpack2<2; kpack2++) {
                C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                    *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[kpack2*4]), 
                    *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[kpack2*4]), 
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
                }
            }
        }
    }        
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait(((BLOCK_N * WARP_K) / (16*64)/WARP_NUM + (BLOCK_M*WARP_K) / (16*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K / 2; k_idx++){ 
                // 加载A矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dwordx4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dwordx4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K/2; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                    for(int kpack2=0; kpack2 < 2; kpack2++) {
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[kpack2*4]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[kpack2*4]), 
                        C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
                    }
                }
            }
        }
    
    }
    // 为了最后一次等到vmcnt_wait=0
    if((k+1) < itk){
        vmcnt_wait(0);
    {
        stage_id_reg_A = 0;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K / 2; k_idx++){ 
            // 加载A矩阵
            ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dwordx4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dwordx4, B_reg, stage_id_reg_B);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K/2; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                for(int kpack2=0; kpack2<2; kpack2++) {
                C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                    *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[kpack2*4]), 
                    *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[kpack2*4]), 
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
                }
            }
        }
    }
    }
    __syncthreads();
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
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_prefill_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,4> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;


    const int warp_n_num = BLOCK_N / WARP_N; //4

    int warp_n_id = warp_id % warp_n_num;//0 1 2 3 
    int warp_m_id = warp_id / warp_n_num;//0

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; //4
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)]; //4

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit; //64xf16
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //2
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //8
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); 
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);

            }
        }
    }
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword/*dword*/ + (lane_id/16)*2/*每个线程持有4个bf16,共2个dword*/;
            int lds_row = (lane_id & 15);
            int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引 64X64
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            {
                int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword/*dword*/ + (lane_id/16)*2/**每个线程持有4个bf16,共2个dword*/;
                int lds_row = (lane_id & 15);
                int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
                int phase = (lds_row/per_phase) % max_phase;
                int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
                int lds_offset = warp_n_id * (WARP_N*WARP_K*bytes_per_element)/bytes_per_dword + min_tile_n * (MFMA_N*WARP_K*bytes_per_element/bytes_per_dword) + lds_row * (WARP_K*bytes_per_element/bytes_per_dword) + col_swizzle;/*dword*/
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K) + min_tile_k] = lds_offset;
            } 
        }
    }

    //main loop 2048/64 = 32次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
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
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dword, B_reg, stage_id_reg_B);
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
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
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
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dword, B_reg, stage_id_reg_B);
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
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
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
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dword, B_reg, stage_id_reg_B);
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
                    *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                    *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
            }
        }
    }
    lgkmcnt_wait_barrier(0);


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
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,8> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = (input_ptr + 0); 
        auto B_ptr = (weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * (WARP_K/marlin_warp_k) * seqlen_B_stride; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop,
                                                            marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;
            
            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) { 
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride + 
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 2048/16 = 128次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, 
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
        }

        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            } 
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {  
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                    }
                }
            }
        }
    }

    // 为了最后一次等到vmcnt_wait=0
    vmcnt_wait(0);
    {
        stage_id_reg_A = ((size_k/WARP_K-1))%STAGES;	
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                #pragma unroll
                for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                    C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                }
            }
        }
    }
    lgkmcnt_wait_barrier(0);
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
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_remainder(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,8> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;
    int warp_n_id_1 = warp_n_id;
    if((bidy * (BLOCK_N* marlin_warp_k)) + warp_n_id *marlin_warp_k * marlin_warp_n >= seqlen_B_k){
        warp_n_id_1 =0;
    }
    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = (input_ptr + 0); 
        auto B_ptr = (weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * (WARP_K/marlin_warp_k) * seqlen_B_stride; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight_marlin_remiander(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop,
                                                            marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k,warp_n_id_1);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;
            
            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) { 
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride + 
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 2048/16 = 128次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_pad_weight_marlin_remiander(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, 
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k,warp_n_id_1);
                                    
        }

        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            } 
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {  
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                    }
                }
            }
        }
    }
    vmcnt_wait(0);
    {
        stage_id_reg_A = ((size_k/WARP_K-1))%STAGES;	
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                #pragma unroll
                for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                    C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                }
            }
        }
    }

    __syncthreads();
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
    bool tail_k_process,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_DOWN(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,8> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = (input_ptr + 0); 
        auto B_ptr = (weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * (WARP_K/marlin_warp_k) * seqlen_B_stride; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop,
                                                            marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;
            
            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    //dowrdx1 swizzle offset 
    // #pragma unroll 
    // for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
    //     #pragma unroll
    //     for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
    //         int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword/*dword*/ + (lane_id/16)*2/*每个线程持有4个bf16,共2个dword*/;
    //         int lds_row = (lane_id & 15);
    //         int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
    //         int phase = (lds_row/per_phase) % max_phase;
    //         int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
    //         int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
    //         precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
    //     }
    // }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) { 
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride + 
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 2048/16 = 128次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, 
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
        }

        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            } 
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {  
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                    }
                }
            }
        }
    }

    // 为了最后一次等到vmcnt_wait=0
    vmcnt_wait(0);
    {
        stage_id_reg_A = ((size_k/WARP_K-1))%STAGES;	
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                #pragma unroll
                for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                    C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                        *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]); 
                }
            }
        }
    }

    if constexpr (tail_k_process){
        int k_offset = (size_k/ WARP_K) * WARP_K; //128  160-128=32 warpK=64 or warpk=32,剩下16需要处理
        int kb_offset = (size_k/ WARP_K) * (WARP_K/marlin_warp_k) * seqlen_B_k; //8 max：10

        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息

        int remain_WARP_K = size_k - k_offset; //32
        //重新匹配swizzle参数
        int inner_dim_length_remain = remain_WARP_K;//32
        //elems_per_one_banks_row = 64
        int per_phase_A = std::max(1, elems_per_one_banks_row / inner_dim_length_remain); //2
        int max_phase_A = std::min(simd_width / per_phase_A, inner_dim_length_remain / vec_size); //8
        
        #pragma unroll 
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_k=0; min_tile_k<(remain_WARP_K/MFMA_K); min_tile_k++) {
                int lds_row = (lane_id & 15);
                int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16; //0 1 2 3 4 5 6 7
            
                int vec_size_dword = vec_size/2/*dword*/;
                int phase = (lds_row/per_phase_A) % max_phase_A;
                int col_swizzle = (lds_col ^ phase) * vec_size_dword;
                // int col_swizzle = lds_col * vec_size_dword ;
                int lds_offset = warp_m_id * (WARP_M*remain_WARP_K)/2 + (min_tile_m * MFMA_M + lds_row) * remain_WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
                precompute_A_lds_offset[min_tile_m * (remain_WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
            }
        }
        //这里要读取2次 //lds的offset必须是字节单位
        #pragma unroll 
        for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) { 
            #pragma unroll
            for(int min_tile_k=0; min_tile_k<(remain_WARP_K/marlin_warp_k); min_tile_k++) {

                int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
                
                int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (remain_WARP_K/marlin_warp_k) * marlin_warp_stride + 
                                min_tile_n * (remain_WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
                precompute_B_lds_offset[min_tile_n*(remain_WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
            }
        }
        {     
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, remain_WARP_K, Element, A_ptr, A_lds, 0, 0, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase_A, max_phase_A);                 
            buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, remain_WARP_K, Element, B_ptr, B_lds, 0, 
                                                        0, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
        }

        vmcnt_wait(0);
        //之前是读取2次，现在是读取4次
        stage_id_reg_A = 0;
        // int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        // int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        for(int k_idx = 0; k_idx < remain_WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
            //     ds_read2_tile_pad_no_wait2(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A,remain_WARP_K); // stage_id_reg_A: 0, 1
            for(int m_idx = 0; m_idx < WARP_M / 16; m_idx ++){\
                A_reg[m_idx][k_idx].int2_array[0] = *(intx2*)((uint8_t*)A_lds+ precompute_A_lds_offset[m_idx * (remain_WARP_K/MFMA_K) + k_idx] * 4);
            }
                // ds_read2_tile_pad_no_wait2(16, k_idx, 4, Element, A_lds, precompute_A_lds_offset, 0, A_reg, 0,32);
            }
            { // 加载B矩阵
                for(int m_idx = 0; m_idx < WARP_N / marlin_warp_n; m_idx ++){
                    B_reg[m_idx][k_idx].int4_array[0] = *(intx4*)((uint8_t*)B_lds + precompute_B_lds_offset[m_idx * (remain_WARP_K/marlin_warp_k) + k_idx]);
                }
                // B_reg[m_idx][k_idx].int4_array[0] = *(intx4*)B_lds_v4bf16;
                // ds_read_b128_tile_pad_no_wait2(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg,remain_WARP_K);
            } 
        }

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<remain_WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
                            
    }      
    __syncthreads();
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
    bool tail_k_process,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_DOWN_Blocksizek32_weight_bypass_lds(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,8> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k; //96
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;
            
            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    //main loop 
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){
        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_dwordx2_blocksizek32(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                
        }
        vmcnt_wait(0);
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N/marlin_warp_n; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                B_reg[n_tile][k_tile].int4_array[0]  = *(intx4*)(B_ptr + warp_n_id * (WARP_N/marlin_warp_n) * 64 * 8 + lane_id * 8 + k_tile * seqlen_B_k + n_tile * 64 * 8);
            }
        }
        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { 
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            } 
        }
        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
    }
    __syncthreads();
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
    bool tail_k_process,  
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_DOWN_Blocksizek32(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,4> A_reg[][WARP_K/16], 
    union_vec<Element,8> B_reg[][WARP_K/16], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k; //96
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename vec<Element,4>::type *B_lds_v4bf16 = (typename vec<Element,4>::type *)(B_lds);
    typename vec<Element,4>::type *A_lds_v4bf16 = (typename vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; 
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;
            
            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) { 
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride + 
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){
        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_dwordx2_blocksizek32(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, 
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
        }

        vmcnt_wait(0);
        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            } 
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]), 
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);    
                    }
                }
            }
        }
    }
    __syncthreads();
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
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_prefill_kpack2_stage1_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset, 
    union_vec<Element,8> A_reg[][4], 
    union_vec<Element,8> B_reg[][4], 
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k, 
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    constexpr int bytes_per_dwordx4 = 16;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;


    const int warp_n_num = BLOCK_N / WARP_N; //4

    int warp_n_id = warp_id % warp_n_num;//0 1 2 3 
    int warp_m_id = warp_id / warp_n_num;//0

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K/2)]; //4 一次线程读取四次，算8次
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K/2)]; //4 一次线程读取四次，算8次

    typename vec<Element,8>::type *B_lds_v8bf16 = (typename vec<Element,8>::type *)(B_lds); //stride为16字节的指针
    typename vec<Element,8>::type *A_lds_v8bf16 = (typename vec<Element,8>::type *)(A_lds); //stride为16字节的指针

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit; //64xf16
    const int simd_width = 16; //MMAC LAYOUT :ROW LEN = 16
    const int inner_dim_length = WARP_K;
    const int vec_size = 8; //element per thread
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //1
    const int max_phase0 = std::min(simd_width / per_phase, inner_dim_length / vec_size); //16
    const int max_phase = std::min(max_phase0,32 / (vec_size * type_width_in_bit / bank_bit_width)); // 8

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; 
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element;  //128x128
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); 
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); 

        #pragma unroll 
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase); 
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);

            }
        }
    }

    //单线程读取四次，warp间重复读取
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K/2); min_tile_k++) {

            int lds_col = (min_tile_k * MFMA_K * 2 ) / 8 + (lane_id / 16) ;
            int lds_row = (lane_id & 15);
            int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;

            int phase = (lds_row/per_phase) % max_phase;
            
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;

            int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K/2) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }

    //每个线程也要读取四次
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K/2); min_tile_k++) { 
            {
                int lds_col = (min_tile_k * MFMA_K * 2) / 8 + (lane_id/16);
                int lds_row = (lane_id & 15);
                int vec_size_dword = (vec_size*bytes_per_element)/bytes_per_dword/*dword*/;
                int phase = (lds_row/per_phase) % max_phase;
                int col_swizzle = ((lds_col) ^ phase) * vec_size_dword;
                int lds_offset = warp_n_id * (WARP_N*WARP_K*bytes_per_element)/bytes_per_dword + min_tile_n * (MFMA_N*WARP_K*bytes_per_element/bytes_per_dword) + lds_row * (WARP_K*bytes_per_element/bytes_per_dword) + col_swizzle;/*dword*/
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K/2) + min_tile_k] = lds_offset;
            } 
        }
    }
    
    for(int k=0; k < size_k/WARP_K; k++){
        int k_offset = k * WARP_K;
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = 0; 
            int A_lds_stage_offset_loop = 0;

            buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = 0; 
            int B_lds_stage_offset_loop = 0; 

            buffer_load_lds_tile_pad_weight_dword_lds4(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                    vec_size, per_phase, max_phase);  
        }

        vmcnt_wait(0);

        stage_id_reg_A = 0;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K / 2; k_idx++){ 
            { // 加载A矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dwordx4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait_dword_lds4(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8bf16, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dwordx4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K/2; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) {
                    for(int kpack2=0; kpack2<2; kpack2++) {
                        C_reg[0][min_tile_m*(WARP_N / MFMA_N) + min_tile_n] = mmac<Element>(
                            *(typename vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[kpack2*4]), 
                            *(typename vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[kpack2*4]), 
                            C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]);        
                    }
                }
            }
        }
    }
    __syncthreads();
    


}


template<bool Is_store_A, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_DOWN_decode_kpack2_kernel2(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output,
    const float* topk_weights) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 
    int warp_n_id = warp_id / warp_k_num; // 0 

    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[WARP_N / MFMA_N]; //1
    float topk_weight_value;
    int i = 0;
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        topk_weight_value = topk_weights[sorted_token_idx];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] =  (n_tile * MFMA_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
            A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
           B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
        }
    }
    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                B_reg[n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
            }
        }
        // if(bidx ==0 && bidy==0 && warp_id ==0 && lane_id ==0){
        // S_BARRIER;S_BARRIER;S_BARRIER;
        // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
        // printf("  迭代%d次 first read B_reg[n_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(B_reg[0][0].data[0]),static_cast<float>( B_reg[0][0].data[1]), static_cast<float>(B_reg[0][0].data[2]), static_cast<float>(B_reg[0][0].data[3]));
        // }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                }
            }
        }

        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 32); 
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                     B_reg[n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 32);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[0][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                }
            }        
        }
    }
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
        

        //假设这里是4个warp，blockm n 都是16 而blockk是16*2 =32 而对应的warp都是16 写入到kwarp个独立的空间中
        constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 * lds_n_offset + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();
        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 * lds_n_offset + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id] * topk_weight_value;
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        // *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8]);
      }
    } 
  } 



//     if(warp_k_id == 0 || warp_k_num == 1){
//     for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
//         for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
//         int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
//           if(token_index < max_n_len_offset * top_k){
//             for (int j = 0; j < 4; j++) {
//                 Element D = f32_to_f16<Element>((C_reg[0][(m_tile*(WARP_N/MFMA_N))][j]) * topk_weight_value );
//                 *reinterpret_cast<Element*>(&g_output[token_index * size_n + 0 * 16 + col_id + j * 4 ]) = D;
//             }
//           }
//         }
//    }     
// }


}

template<bool Is_store_A, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_DOWN_decode_kpack2_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[BLOCK_N/WARP_N][WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output,
    const float* topk_weights) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //1
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 
    int warp_n_id = warp_id / warp_k_num; // 0 
    float topk_weight_value;
    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[BLOCK_N / WARP_N][WARP_N / MFMA_N]; //1 
    int i = 0;
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        topk_weight_value = topk_weights[sorted_token_idx];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[warp_n_id][n_tile] =  (warp_n_id * MFMA_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
    }

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
            A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32 );
        }
    }
        
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K / 2; k_tile++){ 
           B_reg[warp_n_id][n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
        }
    }
    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K /2; k_tile++){ 
                A_reg[m_tile][1].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 32);
            }
        }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                B_reg[warp_n_id][n_tile][1].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 32);
            }
        }
        // if(bidx ==0 && bidy==0 && warp_id ==0 && lane_id ==0){
        // S_BARRIER;S_BARRIER;S_BARRIER;
        // printf(" 迭代%d次 first read A_reg[m_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(A_reg[0][0].data[0]), static_cast<float>(A_reg[0][0].data[1]), static_cast<float>(A_reg[0][0].data[2]), static_cast<float>(A_reg[0][0].data[3]));
        // printf("  迭代%d次 first read B_reg[n_tile][0],%f,%f,%f,%f\n", i,static_cast<float>(B_reg[0][0].data[0]),static_cast<float>( B_reg[0][0].data[1]), static_cast<float>(B_reg[0][0].data[2]), static_cast<float>(B_reg[0][0].data[3]));
        // }
        // mmac
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                }
            }
        }

        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                   A_reg[m_tile][0].floatx4_array[k_tile] = *(floatx4*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 32); 
                }
            }
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K/2; k_tile++){ 
                     B_reg[warp_n_id][n_tile][0].floatx4_array[k_tile]  = *(floatx4*)(weight_ptr + g_row_B[warp_n_id][n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 32);
                }
            }
        }
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                    C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                }
            }
        }
    } 
    //尾处理
    if((i - 1) < itk){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                    C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                }
            }        
        }
    }
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
        
        //假设这里是4个warp，blockm n 都是16 而blockk是16*2 =32 而对应的warp都是16 写入到kwarp个独立的空间中
        constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();

        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }


            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                            
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[warp_n_id][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id] * topk_weight_value;
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        // *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8]);
      }
    } 
  } 



//     if(warp_k_id == 0 || warp_k_num == 1){
//     for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
//         for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
//         int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
//           if(token_index < max_n_len_offset * top_k){
//             for (int j = 0; j < 4; j++) {
//                 Element D = f32_to_f16<Element>((C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N))][j]) * topk_weight_value );
//                 *reinterpret_cast<Element*>(&g_output[token_index * size_n + warp_n_id * 16 + col_id + j * 4 ]) = D;
//             }
//           }
//         }
//    }     
// }

}



template<bool Is_store_A, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGES,  
    typename Element>                                                                                                            
__forceinline__ __device__ void  MOE_W16A16_HIP_NT_DOWN_decode_kernel(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    int max_n_len_offset, 
    f16_vec<WARP_K/4,Element> A_reg[WARP_M/16][2], // 4
    f16_vec<WARP_K/4,Element> B_reg[BLOCK_N/WARP_N][WARP_N/16][2], // 4
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)], //1
    int warp_id, //0~3
    int seqlen_A_stride, // size_k: 2048
    int seqlen_B_stride, // size_kb: 2048
    int top_k, // 8
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy,
    const int size_n,
    Element* g_output,
    const float* topk_weights) 
{
    const int size_k = seqlen_A_stride;//2048=sizeofkb
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;
    constexpr int warp_k_num = BLOCK_K/WARP_K; //2
    constexpr int warp_n_num = BLOCK_N/WARP_N; //1
    int warp_k_id = warp_id % warp_k_num; // 0 1 
    int warp_n_id = warp_id / warp_k_num; // 0
    float topk_weight_value;
    int g_row_A[WARP_M / MFMA_M]; //1 
    int g_row_B[BLOCK_N / WARP_N][WARP_N / MFMA_N]; //1 

    int i = 0; 
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        topk_weight_value = topk_weights[sorted_token_idx];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + warp_k_id * WARP_K + col_id * (WARP_K / 4); //这是kpack才是16 默认是8个int8，这里就是4个bf16
    }
    // 2048 * 768大小
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            g_row_B[warp_n_id][n_tile] =  (warp_n_id * WARP_N + row_id) * seqlen_B_stride + warp_k_id * WARP_K  + col_id * (WARP_K / 4);
        }
    
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr + g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16 );
        }
    }

        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            B_reg[warp_n_id][n_tile][0].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 16);
            }
        }

    int itk = size_k/warp_k_num/WARP_K; // 2048/4/16 =  32
    for(i=1; i < itk ; i += 2){
    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            A_reg[m_tile][1].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + i * warp_k_num * WARP_K  + k_tile * 16);
        }
    }
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
            B_reg[warp_n_id][n_tile][1].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + (i * warp_k_num * WARP_K)  + k_tile * 16);
            }
        }
  
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                        
                    }
                }
            }
        
        if((i + 1) < itk){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                   A_reg[m_tile][0].floatx2_array[k_tile] = *(floatx2*)(input_ptr+ g_row_A[m_tile] + (i + 1) * warp_k_num * WARP_K  + k_tile * 16); 
                }
            }            
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){ 
                        B_reg[warp_n_id][n_tile][0].floatx2_array[k_tile]  = *(floatx2*)(weight_ptr + g_row_B[warp_n_id][n_tile] + ((i + 1) * warp_k_num * WARP_K)  + k_tile * 16);
                    }
                }
            }             
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){               
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][1].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][1].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  

                    }
                }
            }
        
        
    } 
    //尾处理
    if((i - 1) < itk){        
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)] = mmac<Element>(
                                (*(typename vec<Element, 4>::type*)&A_reg[m_tile][0].floatx2_f16_array[k_tile]), 
                                (*(typename vec<Element, 4>::type*)&B_reg[warp_n_id][n_tile][0].floatx2_f16_array[k_tile]), 
                                C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N) + n_tile)]);  
                    }
                }        
            }      
    } 
    
    extern __shared__ float out_smemf[]; // 声明lds信息
    uint16_t* lds_bf16 = (uint16_t*)out_smemf;
    if constexpr (warp_k_num > 1){ //现在就2个warp来测试
    constexpr int lds_n_offset = warp_k_num * BLOCK_N; // 2 * 16 = 32  
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                *(floatx4*)( &out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  //一行存64个数 相同warpn的数存储在一起
                                        + warp_n_id * WARP_N + n_tile * 16  
                                        + col_id * 4  
                                        + warp_k_id * BLOCK_N] ) = C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile];
            }
        }
        __syncthreads();
        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile] = {0.0, 0.0, 0.0, 0.0};
                }
            }

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num; k_tile++){ //0 1 
                        for(int i = 0; i < 4; ++i){
                            C_reg[warp_n_id][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                            out_smemf[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16  + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / MFMA_M; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / MFMA_N; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[warp_n_id][min_tile_m * WARP_N/MFMA_N + min_tile_n][reg_id] * topk_weight_value;
          int index = min_tile_m * MFMA_M * BLOCK_N 
                    + min_tile_n * MFMA_N 
                    + warp_n_id * WARP_N 
                    + (lane_id & 15 ) * BLOCK_N 
                    + lane_id / 16 
                    + reg_id * 4 
                    + (min_tile_m * MFMA_M + (lane_id % 16) )/2 * 2/*padding*/;
          lds_bf16[index] = f32_to_f16<Element>(value);
        }
      }
    }
  }

  __syncthreads();

  
  { 
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_N / 8;  //2
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
    int m_idx = tid / N_thread;  //00 11 22 33 44 
    int n_idx = tid % N_thread;  //01 01 01 01 01
    for(; m_idx < BLOCK_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_idx, int(sorted_token_lens - 1))];
      if(token_index < max_n_len_offset * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&lds_bf16[m_idx * BLOCK_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      


//     if(warp_k_id == 0 || warp_k_num == 1){
//     for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
//         for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
//         int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
//           if(token_index < max_n_len_offset * top_k){
//             for (int j = 0; j < 4; j++) {
//                 Element D = f32_to_f16<Element>((C_reg[warp_n_id][(m_tile*(WARP_N/MFMA_N))][j]) * topk_weight_value );
//                 *reinterpret_cast<Element*>(&g_output[token_index * size_n + warp_n_id * 16 + col_id + j * 4 ]) = D;
//             }
//           }
//         }
//    }     
// }

}
} //namespace end
} //namespace end
