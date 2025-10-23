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


template <uint16_t BLOCK_M,
          uint16_t BLOCK_N
          >
struct Scheduler {
public:
    int bidx;
    int bidy;
    int size_m_topk;
    int n_blocks;       // n方向的block数目 N/block_n
    int current_idx;    // 当前处理的block索引 blockIdx.x +
    int num_tokens_pad;
    const int32_t* __restrict__ sorted_token_ids;

public:
    // 构造函数，根据定义的变量进行初始化
    __device__ Scheduler(int _bidx, int _bidy, int _size_m_topk, int _n_blocks, int _current_idx, int _num_tokens_pad, const int32_t* __restrict__ _sorted_token_ids)
        : bidx(_bidx),
          bidy(_bidy),
          size_m_topk(_size_m_topk),
          n_blocks(_n_blocks),
          current_idx(_current_idx),
          num_tokens_pad(_num_tokens_pad),
          sorted_token_ids(_sorted_token_ids)
    {}

 
    __device__ bool get_next_block() {
        get_swizzled_block_idx();

        if(sorted_token_ids[bidx * BLOCK_M] >= size_m_topk || bidx * BLOCK_M >= num_tokens_pad) 
            return false;
        
        return true;  // 找到有效tile
    }

  
    __device__ int get_bidx() const {
        return bidx;
    }

  
    __device__ int get_bidy() const {
        return bidy;
    }
  
    __device__ void get_swizzled_block_idx() {
        bidx = current_idx / n_blocks;
        bidy = current_idx % n_blocks;
        // todo block swizlle
        // bidx = current_idx % n_blocks;
        // bidy = current_idx / n_blocks;
    }

 
    __device__ void advance() {
        current_idx += gridDim.x;  // 跳过gridDim.x个索引，实现不同block的任务交错
    }
};


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
__forceinline__ __device__ void  gemm_nt_first_stage_prefill(
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

    // Ref triton swizzle: http://112.11.119.99:10068/dcutoolkit/deeplearing/triton/-/blob/triton-lang/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
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

    // 构建A矩阵读取索引
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
    // 构建B矩阵读取索引
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            {
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
__forceinline__ __device__ void  gemm_nt_first_stage_marlin_prefill(
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
    int seqlen_B_stride, // size_k: 16384
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

    // Ref triton swizzle: http://112.11.119.99:10068/dcutoolkit/deeplearing/triton/-/blob/triton-lang/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
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
                int B_block_buffer_load_global_offset = stage_id_num * seqlen_B_stride; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight(WARP_NUM, WARP_K, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);
            }
        }
    }

    // 构建A矩阵读取索引
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
    // 构建B矩阵读取索引
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            {
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
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k * seqlen_B_stride); // 配置全局显存信息
        { 
            // 预加载A矩阵->lds
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 预加载B矩阵->lds
            int B_block_buffer_load_global_offset = (STAGES - 1) * seqlen_B_stride; 
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 

            buffer_load_lds_tile_pad_weight(WARP_NUM, WARP_K, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
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

// MOE-DECODE-W8A8 针对小shape优化
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
__forceinline__ __device__ void  gemm_nt_first_stage_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[], // 2 = warpk/mmak
    union_vec<Element,WARP_K/4> B_reg[], // 2 = warpk/mmak
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


    for(int i = warp_k_id * WARP_K; i < size_k; i += warp_k_num * WARP_K){

        // load A
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
            int g_row = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1));
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / 64; k_tile++){ // todo 一个线程连续拷贝16还是隔开拷贝  col_id * 16 * WARP_K/64 + k_tile * 16
                A_reg[m_tile].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + i + col_id * 16 + k_tile * 64);
                //A_reg[m_tile].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(input_ptr, 0, g_row * size_k + i + col_id * 16 * WARP_K/64 + k_tile * 16);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            int g_row = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / 64; k_tile++){
                B_reg[n_tile].int4_array[k_tile] = builtin_amdgcn_buffer_load_reg_dwordx4(weight_ptr, 0, g_row * size_k + i + col_id * 16 + k_tile * 64);
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile].int8t_array[k_tile]), 
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
    union_vec<Element,WARP_K/4> B_reg[][2], // 2 = stage
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
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N + row_id) * size_k + col_id * 16;
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
            buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][0].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 + k_start);
        }

    }

    
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
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][1].int4_array[k_tile], 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile][0].int8t_array[k_tile]), 
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
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][0].int4_array[k_tile], 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset * 2 + k_tile * 64 + k_start);
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile][1].int8t_array[k_tile]), 
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
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][1].int4_array[k_tile], 0, g_row_B[n_tile] + (STAGE - 1) * stage_offset + k_tile * 64 + k_start);
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile][0].int8t_array[k_tile]), 
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][1].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile][1].int8t_array[k_tile]), 
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
                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                                *(typename vec<Element, 8>::type*)(&B_reg[n_tile][0].int8t_array[k_tile]), 
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
    union_vec<Element,WARP_K/4> B_reg[][STAGE], // 2 = stage
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
    const int stage_offset = warp_k_num * WARP_K;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_idx / top_k,  max_n_len_offset - 1)) * size_k + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N + row_id) * size_k + col_id * 16;
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    // read 0
    // load A
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 + k_start);
            }
        }
        k_start += stage_offset;
    }
    k_start -= stage_offset;
    
    for( ; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE){

        #pragma unroll
        for(int i = 0; i < STAGE; ++i){
            ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
            // load A
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

            // load B
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

            ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        } // stage end

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//
    if(k_start + stage_offset < size_k){ // size_k / block 为偶数
        //  --------------------------------------------- load mmac ---------------------------------- //
        int i = 0;
        if constexpr (STAGE == 4){ //TODO: STAGE为4 尾处理有四种可能方式 需要size_k来确定loop次数，size_k如果为动态传入会有寄存器溢出 目前写死解决 
            constexpr int epilogue_tile =  ( 7168/BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;        
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        } else{ // two stage
            constexpr int epilogue_tile =  1;      
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        }
        //  --------------------------------------------- load mmac ---------------------------------- //

        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int ii = 1; ii < STAGE; ++ii){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
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
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i-1].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i-1].int8t_array[k_tile]), 
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

// ------------------------------------------------------------------------------------------------ gemm_decode_marlin -------------------------------------------------------- //
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
__forceinline__ __device__ void  gemm_nt_marlin_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    union_vec<Element,WARP_K/4> A_reg[][STAGE], // 2 = stage
    union_vec<Element,WARP_K/4> B_reg[][STAGE], // 2 = stage
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
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;


    //warp内部k连续[warpn, 64] warp外部在N方向连续 
    const int size_n = scale_B_stride_e;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    //warp内部k连续[warpn, 64] warp外部在K方向连续 
    // const int size_n = 256;
    // int k_start = warp_k_id * WARP_K;
    // int k_start_b = warp_k_id * WARP_K * WARP_N;
    // const int stage_offset = warp_k_num * WARP_K;
    // const int stage_offset_b = warp_k_num * WARP_K * WARP_N;
    
    
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
        g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N)* 64 + col_id * 16; //warp内部k连续[warpn, 64] warp外部在N方向连续 
        //g_row_B[n_tile] = (warp_n_id * WARP_N) * size_k + (row_id + n_tile * MFMA_N)* 64 + col_id * 16; //warp内部k连续[warpn, 64] warp外部在K方向连续 
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    // read 0
    // load A
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * WARP_N + k_start_b);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;
    
    for( ; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE, k_start_b += stage_offset_b * STAGE){

        #pragma unroll
        for(int i = 0; i < STAGE; ++i){
            ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
            // load A
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

            // load B
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * WARP_N + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

            ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        } // stage end

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//
    if(k_start + stage_offset < size_k){ // size_k / block 为偶数
        //  --------------------------------------------- load mmac ---------------------------------- //
        int i = 0;
        if constexpr (STAGE == 4){ //TODO: STAGE为4 尾处理有四种可能方式 需要size_k来确定loop次数，size_k如果为动态传入会有寄存器溢出 目前写死解决 
            constexpr int epilogue_tile =  ( 7168/BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;        
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * WARP_N + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        } else{ // two stage
            constexpr int epilogue_tile =  1;      
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * WARP_N + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        }
        //  --------------------------------------------- load mmac ---------------------------------- //

        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int ii = 1; ii < STAGE; ++ii){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }

        }

    }
    else{ // todo 不为block_k的倍数
        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int i = 1; i < STAGE; ++i){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i-1].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i-1].int8t_array[k_tile]), 
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