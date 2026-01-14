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

// A矩阵buffer_load读                                                                                          
#define buffer_load_lds_tile_q_nope(WARP_NUM, n_size, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
    vec_size, per_phase, max_phase)\
{\
    int bytes_per_element = 1;\
    const int element_num_per_thread = 4/*buffer_load_dword => 1 dword: 4 bytes*/; \
    int element_per_dword = element_num_per_thread/bytes_per_element;\
    int thread_num_n = WARP_K*bytes_per_element/element_num_per_thread;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    int np_size = n_size * N_row_len * bytes_per_element;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_element) / (bytes_per_element*element_per_dword*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int block_offset = block_row + bidx * BLOCK_M;\
        int g_row = std::min(block_offset, max_M_len - 1);\
        int g_offset = g_row * np_size + bidz * N_row_len * bytes_per_element;\
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size / element_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/element_per_dword;\
        int gvOffset   = g_offset/element_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/element_per_dword + warp_loop * 64;\
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

// B矩阵buffer_load读                                                                                          
#define buffer_load_lds_tile_w_uv(WARP_NUM, N_row_len, BLOCK_N, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, warp_id, lane_id,\
    vec_size, per_phase, max_phase)\
{\
    int bytes_per_element = 1;\
    const int element_num_per_thread = 4/*buffer_load_dword => 1 dword: 4 bytes*/; \
    int element_per_dword = element_num_per_thread/bytes_per_element;\
    int thread_num_n = WARP_K*bytes_per_element/element_num_per_thread;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_N*WARP_K*bytes_per_element) / (element_num_per_thread*64*bytes_per_element);\  
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int g_row = block_row; \
        int phase = (block_row/per_phase) % max_phase; \
        int g_col = lane_N_idx; \
        int vec_size_dword = vec_size / element_per_dword; \
        int g_col_swizzle = ((g_col/vec_size_dword) ^ phase) * vec_size_dword + (g_col%vec_size_dword); \
        int gsOffset   = global_offset/element_per_dword;\
        int gvOffset   = g_row * N_row_len * bytes_per_element/element_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/element_per_dword + warp_loop * 64;\
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


template<
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
__forceinline__ __device__ void  batch_gemm_impl(
    const Element* q_nope, 
    const Element* w_uv, 
    Element* A_lds,
    Element* B_lds,
    union_vec<Element,8> A_reg[][WARP_K/32], 
    union_vec<Element,8> B_reg[][WARP_K/32], 
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int batch_size,
    int n_size,
    int seqlen_A_stride, // size_k: 128
    int seqlen_B_stride, // size_k: 128
    const int bidx,
    const int bidz) {

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

    const int bytes_per_dword = 4;
    const int bytes_per_element = 1;

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element; // A tile拿 1 * 16x64的大小
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element; // B tile拿 4 * 32x64的大小,
#if 1
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto q_nope_ptr = tcp_cache_swizzle_func<64, Element>(q_nope + 0); // 配置全局显存信息
        auto w_uv_ptr = tcp_cache_swizzle_func<64, Element>(w_uv + 0); // 配置全局显存信息

        #pragma unroll // 预加载STAGES-1的数据
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_q_nope(WARP_NUM, n_size, seqlen_A_stride, BLOCK_M, WARP_K, Element, q_nope_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, batch_size, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);                
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element; 
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_w_uv(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, w_uv_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, warp_id, lane_id,
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
            int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword + (lane_id/16)*2;
            int lds_row = (lane_id & 15);
            int vec_size_dword = vec_size*bytes_per_element/bytes_per_dword/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K*bytes_per_element)/bytes_per_dword 
                                + (min_tile_m * MFMA_M + lds_row)*WARP_K*bytes_per_element/bytes_per_dword/*dword*/ + col_swizzle/*dword*/; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            {
                int lds_col = (min_tile_k * MFMA_K * bytes_per_element) / bytes_per_dword + (lane_id/16)*2;
                int lds_row = (lane_id & 15);
                int vec_size_dword = vec_size*bytes_per_element/bytes_per_dword/*dword*/;
                int phase = (lds_row/per_phase) % max_phase;
                int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
                int lds_offset = warp_n_id * (WARP_N*WARP_K*bytes_per_element)/bytes_per_dword 
                                + (min_tile_n * MFMA_N + lds_row)*WARP_K*bytes_per_element/bytes_per_dword + col_swizzle/*dword*/;
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K) + min_tile_k] = lds_offset;
            } 
        }
    }

    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K * bytes_per_element;
        auto q_nope_ptr = tcp_cache_swizzle_func<64, Element>(q_nope + k_offset); // 配置全局显存信息
        auto w_uv_ptr = tcp_cache_swizzle_func<64, Element>(w_uv + k_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_q_nope(WARP_NUM, n_size, seqlen_A_stride, BLOCK_M, WARP_K, Element, q_nope_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, batch_size, warp_id, lane_id,
                                                  vec_size, per_phase, max_phase);                  

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element; 
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_w_uv(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, w_uv_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, warp_id, lane_id,
                                                    vec_size, per_phase, max_phase);  
        }

        vmcnt_wait(((BLOCK_N * WARP_K) / (bytes_per_dword*64)/WARP_NUM + (BLOCK_M*WARP_K) / (bytes_per_dword*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dword, B_reg, stage_id_reg_B);
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
        vmcnt_wait(((BLOCK_N * WARP_K) / (bytes_per_dword*64)/WARP_NUM + (BLOCK_M*WARP_K) / (bytes_per_dword*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/bytes_per_dword, B_reg, stage_id_reg_B);
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

} //namespace end
} //namespace end
