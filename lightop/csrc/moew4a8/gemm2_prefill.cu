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

// A矩阵buffer_load读 未加入 swizzle，先验证准确                                                                                       
#define buffer_load_lds_tile_pad_sorted_token_gemm2_prefill(WARP_NUM, \
                                              N_row_len, \
                                              WARP_M, \
                                              WARP_K, \
                                              Element, \
                                              global_ptr, \
                                              lds_ptr, \
                                              global_offset, \
                                              lds_stage_offset, \
                                              max_M_len, \
                                              warp_id, \
                                              lane_id,\
                                              vec_size, \
                                              per_phase, \
                                              max_phase)\
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


// B矩阵buffer_load读 改为 w4a8  暂时未引入 swizzle                                                                                    
#define buffer_load_lds_tile_pad_weight_w4a8(\
                                        WARP_NUM, \
                                        N_row_len, \
                                        WARP_M, \
                                        WARP_K, \
                                        Element, \
                                        global_ptr, \
                                        lds_ptr, \
                                        global_offset, \
                                        lds_stage_offset, \
                                        warp_id, \
                                        lane_id, \
                                        vec_size, \
                                        per_phase, \
                                        max_phase) \
{\
    int bytes_per_Element = 1;\
    int Element_per_dword = 4/bytes_per_Element;\
    int thread_num_n = WARP_K*bytes_per_Element/4;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\ 
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (WARP_M*WARP_K*bytes_per_Element) / (4*64);\  
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int g_row = warp_loop * thread_num_m + lane_M_idx;\
        int g_col = lane_N_idx; \
        int gsOffset   = global_offset/Element_per_dword;\
        int gvOffset   = g_row * N_row_len/Element_per_dword + g_col;\
        int lds_offset = lds_stage_offset/Element_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}


/*
         从lds中读取数据到 vgpr 中
    M： 每个warp需要 load 的行索引大小
    n_idx： 列方向当前 load 的是 warp 需要计算的第几个 mmac 块
    WARP_NUM： 总的 warp 数
    Element： char
    lds_v8i8： int8x8 的向量化指针
    precompute_offset： 预先计算出来的 需要 load 的偏移 
    lds_stage_offset_loop： 当前执行计算的数据块 在lds中的偏移
    reg: 接收数据的 寄存器
*/ 
#define ds_read2_tile_pad_no_wait_w4a8_token(\
                                        M,\
                                        n_idx,\
                                        lds_v8i8,\
                                        precompute_offset,\
                                        lds_stage_offset_loop,\
                                        reg\
                                    )\
{\
for(int m_idx = 0; m_idx < M / 16; m_idx ++){\
    inline_ds_read2_b32_no_wait(lds_v8i8, lds_stage_offset_loop + precompute_offset[m_idx][n_idx], reg[m_idx][n_idx].int2_array[0], 1);\
}\
}

// 流水版本 
template<bool Is_store_A, // false
    int A_prefetch_level, // 0
    int B_prefetch_level, // 0
    int WARP_NUM,  // 4
    int BLOCK_M,   // 32
    int BLOCK_N,   // 128
    int BLOCK_K,   // 64
    int WARP_M,    // 32
    int WARP_N,    // 32
    int WARP_K,    // 32
    int STAGES,    // 2
    typename Element, 
    typename ElementAccum = int32_t>                                                                                                            
__forceinline__ __device__ void  gemm_nt_first_stage_w4a8_pipe(
    const Element* input_ptr,  // token 矩阵的起始位置
    const Element* weight_ptr, // 权重矩阵 当前处理的起始位置
    Element* A_lds,  // lds中A缓存区的起始地址
    Element* B_lds,  // lds中B缓存区的起始地址
    float* input_scale_ptr, // 全局 inout_scale 的起始位置
    float* weight_scale_ptr, // 当前处理的起始位置
    int max_n_len_offset, // 标记 A矩阵 的最大长度
    union_vec<Element,8> A_reg[][WARP_K/32], // union_vec<Element, 8>  A_reg: [2][1] 使用哪个缓存区；M维度的索引；K维度的索引
    union_vec<Element,8> B_reg[][WARP_K/32], // union_vec<Element, 8> B_reg：[2][1] N维度的索引；K维度的索引
    intx4 C_reg[WARP_M/16][WARP_N/16],  // intx4 C_reg[2][2] :M维度的索引；N维度的索引
    int warp_id,  //当前线程的warp索引
    int seqlen_A_stride, // size_k: 256
    int seqlen_B_stride, // size_k: 128 这里由于 uint4 pack成 int8 B矩阵的行数据减半
    int scale_A_stride_m, // 1
    int scale_A_stride_k, // 1
    int scale_B_stride_e, // 7168
    int scale_B_stride_n, // 1
    int scale_B_stride_k, // 1
    int top_k, // 1
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx 
) {
    const int size_k = seqlen_A_stride; // 256
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    const int warp_n_num = BLOCK_N / WARP_N;
    int warp_n_id = warp_id % warp_n_num;
    int warp_m_id = warp_id / warp_n_num;

    int precompute_A_lds_offset[WARP_M/MFMA_M][WARP_K/MFMA_K]; 
    int precompute_B_lds_offset[WARP_N/MFMA_N][WARP_K/MFMA_K];

    // A矩阵的指针仍然保持 强转成 8个int8的向量数据类型
    typename vec<Element,8>::type *A_lds_v8i8 = (typename vec<Element,8>::type *)(A_lds);

    // B矩阵 改为int类型
    int32_t* B_lds_int32 = (int32_t*)B_lds;

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

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K; // A矩阵 LDS中一个数据块的大小偏移是 [32, 32]
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * (WARP_K / 2); // B矩阵 LDS中一个数据块的大小偏移是 [128, 16] k维度是折半后的

    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;
    
    // 预取下一阶段的 数据到LDS中
    {
        // 转为 stride 为 128字节 的全局指针
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); // 配置全局显存信息
        
        #pragma unroll // 预加载STAGES-1的数据
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){
            // 加载下一数据块的A矩阵数据到 LDS中
            int A_block_buffer_load_global_offset = stage_id_num * WARP_K; 
            int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_gemm2_prefill(WARP_NUM, 
                                                seqlen_A_stride, 
                                                BLOCK_M, 
                                                WARP_K, 
                                                Element, 
                                                A_ptr, 
                                                A_lds, 
                                                A_block_buffer_load_global_offset, 
                                                A_lds_stage_offset_loop, 
                                                BLOCK_K, 
                                                warp_id, 
                                                lane_id,
                                                vec_size, 
                                                per_phase, 
                                                max_phase);  

            int B_block_buffer_load_global_offset = stage_id_num * WARP_K / 2;  // 从全局拿B矩阵时， 每次偏移 WARP_K / 2
            int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; // B 一次计算拿 128*32 大小的B矩阵

            buffer_load_lds_tile_pad_weight_w4a8(WARP_NUM,  // 4
                                                seqlen_B_stride,  // 这里传进来的已经是 折半后的 一行的数据个数了
                                                BLOCK_N,  // 128
                                                WARP_K / 2,  // K维度折半 
                                                Element,  // 
                                                B_ptr,  // 
                                                B_lds,  // 整个 LDS 缓冲区的起始地址
                                                B_block_buffer_load_global_offset, // 全局内存的偏移量  k维度折半
                                                B_lds_stage_offset_loop,  // 当前写入 LDS 的起始位置 k维度折半
                                                warp_id,  // warp索引
                                                lane_id,  // 线程在当前 warp 中的索引
                                                vec_size, // 当前调整正确性，未加入swizzle 
                                                per_phase, 
                                                max_phase); 
        }
    }

    // 预先计算 lds中 load A矩阵的偏移
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
            precompute_A_lds_offset[min_tile_m][min_tile_k] = lds_offset; // 装载了两次,
        }
    }

    // [128, 16] 的B矩阵中 每个warp load 的偏移量
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            { // 走这里
                // 整个warp 负责的 [WARP_N, WARP_K/2] 数据块中 线程的索引
                //  行索引： 当前mmac之前的mmac偏移 + 当前的偏移
                int row = min_tile_n * MFMA_N + (lane_id & 15);
                //  列索引： 当前mmac之前的mmac偏移 dword  + 当前mmac的列索引
                // int lds_col = min_tile_k * MFMA_K / 2 / 4 + lane_id / 16;
                int col = min_tile_k * MFMA_K / 8 + lane_id / 16;
                // 计算 lds当前计算起始地址的偏移 dword
                int lds_offset = warp_n_id * (WARP_N*WARP_K / 2) / 4 +
                                 row * (WARP_K / 2 / 4) +
                                 col;
                precompute_B_lds_offset[min_tile_n][min_tile_k] = lds_offset;
            } 
        }
    }

    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){
    // for(int k=0; k < 2; k++){
        // 等待预取完成
        vmcnt_wait(0);

        // 当前迭代中 k方向的偏移
        int k_offset = k * WARP_K;
        int k_offset_b = k * (WARP_K / 2);
        // A_ptr 偏移到当前迭代的起始位置
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        // B_ptr 偏移到当前迭代的起始位置 
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + k_offset_b); // 配置全局显存信息
        { 
            // 加载下一次计算需要的 A矩阵
            // 全局地址每次都需要 偏移 WARP_K大小
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K; 
            // 写入位置 1 0 1 0 1 0 1 0迭代使用 双lds双缓冲
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token_gemm2_prefill(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);                

            // 加载下一次计算需要的 B矩阵
            // 全局地址每次都需要 偏移 WARP_K大小
            int B_block_buffer_load_global_offset = (STAGES - 1) * WARP_K / 2; 
            // 写入位置 1 0 1 0 1 0 1 0迭代使用 双lds双缓冲
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 
            buffer_load_lds_tile_pad_weight_w4a8(WARP_NUM,  // 4
                                                seqlen_B_stride,  // 这里传进来的已经是 折半后的 一行的数据个数了
                                                BLOCK_N,  // 128
                                                WARP_K / 2,  // K维度折半 
                                                Element,  // 
                                                B_ptr,  // 
                                                B_lds,  // 整个 LDS 缓冲区的起始地址
                                                B_block_buffer_load_global_offset, // 全局内存的偏移量  k维度折半
                                                B_lds_stage_offset_loop,  // 当前写入 LDS 的起始位置 k维度折半
                                                warp_id,  // warp索引
                                                lane_id,  // 线程在当前 warp 中的索引
                                                vec_size, // 当前调整正确性，未加入swizzle 
                                                per_phase, 
                                                max_phase); 
        }
        // =================================== 本次迭代的运算部分 ================================================
        // 当前使用的 lds双缓冲区的那一部分 0 1 0 1 0 1 0 1
        stage_id_reg_A = k & (STAGES-1);
        // AB指针均偏移到LDS当前运算位置
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;

        // 从LDS中拆解出 需要的B矩阵数据到vgpr中
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵数据到 VGPR中
            ds_read2_tile_pad_no_wait_w4a8_token(WARP_M, k_idx, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop / 4, A_reg); 
            
            // 加载B矩阵数据到 VGPR 中
            for(int n_idx = 0; n_idx < WARP_N / 16; n_idx ++){
                // 计算偏移          
                int b_lds_dword_idx = B_lds_stage_offset_loop / 4 + precompute_B_lds_offset[n_idx][k_idx];
                // 偏移
                int first_int = B_lds_int32[b_lds_dword_idx];
                int8_t* first_byte_ptr = reinterpret_cast<int8_t*>(&first_int);
                int secone_int = first_int >> 4;
                int8_t* second_byte_ptr = reinterpret_cast<int8_t*>(&secone_int);
                // 赋值
                B_reg[n_idx][k_idx].int8_array[0] = first_byte_ptr[0] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[1] = second_byte_ptr[0] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[2] = first_byte_ptr[1] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[3] = second_byte_ptr[1] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[4] = first_byte_ptr[2] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[5] = second_byte_ptr[2] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[6] = first_byte_ptr[3] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[7] = second_byte_ptr[3] & 0x0f;
            } 
        }

        // =========================== B矩阵中各寄存器中 的值减去 8  ========================

        for(int n_idx = 0; n_idx < WARP_N / 16; n_idx ++){
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
                B_reg[n_idx][k_idx].int8_array[0] -= 8;
                B_reg[n_idx][k_idx].int8_array[1] -= 8;
                B_reg[n_idx][k_idx].int8_array[2] -= 8;
                B_reg[n_idx][k_idx].int8_array[3] -= 8;
                B_reg[n_idx][k_idx].int8_array[4] -= 8;
                B_reg[n_idx][k_idx].int8_array[5] -= 8;
                B_reg[n_idx][k_idx].int8_array[6] -= 8;
                B_reg[n_idx][k_idx].int8_array[7] -= 8;
            }
        }
        // ============================ 执行 mmac 运算 ================================
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                    C_reg[min_tile_m][min_tile_n] = mmac<Element>(
                                                        *(typename vec<Element, 8>::type*)(&A_reg[min_tile_m][min_tile_k].scalar_array[0]), 
                                                        *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                                                        C_reg[min_tile_m][min_tile_n]
                                                    ); 
                }
            }
        }
    }

    // 
    vmcnt_wait(0);

    {   
        // 使用最后一次for循环的另一块LDS数据
        stage_id_reg_A = (stage_id_reg_A + 1) & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 从LDS中拆解出 需要的B矩阵数据到vgpr中
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵数据到 VGPR中
            ds_read2_tile_pad_no_wait_w4a8_token(WARP_M, k_idx, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop / 4, A_reg); 
            
            // 加载B矩阵数据到 VGPR 中
            for(int n_idx = 0; n_idx < WARP_N / 16; n_idx ++){
                // 计算偏移          
                int b_lds_dword_idx = B_lds_stage_offset_loop / 4 + precompute_B_lds_offset[n_idx][k_idx];
                // 偏移
                int first_int = B_lds_int32[b_lds_dword_idx];
                int8_t* first_byte_ptr = reinterpret_cast<int8_t*>(&first_int);
                int secone_int = first_int >> 4;
                int8_t* second_byte_ptr = reinterpret_cast<int8_t*>(&secone_int);
                // 赋值
                B_reg[n_idx][k_idx].int8_array[0] = first_byte_ptr[0] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[1] = second_byte_ptr[0] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[2] = first_byte_ptr[1] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[3] = second_byte_ptr[1] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[4] = first_byte_ptr[2] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[5] = second_byte_ptr[2] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[6] = first_byte_ptr[3] & 0x0f;
                B_reg[n_idx][k_idx].int8_array[7] = second_byte_ptr[3] & 0x0f;
            } 
        }
        // =========================== B矩阵中各寄存器中 的值减去 8 ==========
        for(int n_idx = 0; n_idx < WARP_N / 16; n_idx ++){
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
                B_reg[n_idx][k_idx].int8_array[0] -= 8;
                B_reg[n_idx][k_idx].int8_array[1] -= 8;
                B_reg[n_idx][k_idx].int8_array[2] -= 8;
                B_reg[n_idx][k_idx].int8_array[3] -= 8;
                B_reg[n_idx][k_idx].int8_array[4] -= 8;
                B_reg[n_idx][k_idx].int8_array[5] -= 8;
                B_reg[n_idx][k_idx].int8_array[6] -= 8;
                B_reg[n_idx][k_idx].int8_array[7] -= 8;
            }
        }
    }

    // ============================ 执行 mmac 运算 ================================
    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                C_reg[min_tile_m][min_tile_n] = mmac<Element>(
                                                    *(typename vec<Element, 8>::type*)(&A_reg[min_tile_m][min_tile_k].scalar_array[0]), 
                                                    *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                                                    C_reg[min_tile_m][min_tile_n]
                                                ); 
            }
        }
    }
}


}
}