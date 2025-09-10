#include "marlin.h"

namespace at {
namespace native {
namespace marlin_moe_w4a8{
    template <int const num_threads, int const num_bits>
    __global__ void awq_marlin_w4a8_repack_kernel(
                        uint8_t const* b_q_weight_ptr,
                        uint32_t* out_ptr,
                        int64_t size_batch,
                        int64_t size_k, 
                        int64_t size_n
                    ) {
        int batch_idx =  blockIdx.x;  // 当前 block需要重排的 权重矩阵 [32, 64]数据块是第几个专家
        int k_tile_idx =  blockIdx.y; //当前 block需要重排的 权重矩阵 [32, 64]数据块是第几个专家K维度的第几个任务块
        int n_tile_idx =  blockIdx.z; //当前 block需要重排的 权重矩阵 [32, 64]数据块是第几个专家N维度的第几个任务块
        
        // 重排 [32, 64]的数据 
        constexpr int tile_k_size = 32;
        constexpr int tile_n_size = 64;

        // // =============== debug ==============================
        
        // 当前block需要加载的 [32, 64]数据块的 N K 起始索引 (uint4角度)
        int first_n = n_tile_idx * tile_n_size; // n维度的 起始索引(uint4)
        int first_k = k_tile_idx * tile_k_size; // k维度的 起始索引(uint4)
        // 从 uint8 全局内存中取数据的 uint8 索引
        int gl_rd_n_idx = first_n + threadIdx.x % 64;
        int gl_rd_k_idx = first_k / 2 + (threadIdx.x / 64) * 4;

        int gl_rd_idx =  batch_idx * size_n * size_k / 2 + gl_rd_n_idx * (size_k / 2) + gl_rd_k_idx;

        uint32_t vals[8]; // 使用uint32 的 低4bit 记录
        for(int i=0;i<4;i++){
            uint8_t pack_int8_val = b_q_weight_ptr[gl_rd_idx + i];
            // 提取高 4 bit：右移 4 位，然后保留低 4 bit
            uint32_t high_nibble = (pack_int8_val >> 4) & 0xF;
            // 提取低 4 bit：直接与 0xF 与运算
            uint32_t low_nibble = pack_int8_val & 0xF;
            // 其中一种放置方式
            // vals[i * 2 + 0] = low_nibble;
            // vals[i * 2 + 1] = high_nibble;
            vals[i * 2 + 0] = high_nibble;
            vals[i * 2 + 1] = low_nibble;
        } 
        
        // =========== 重组后以int32的数据写回 =======================
        constexpr int pack_idx[8] = {4, 0, 5, 1, 6, 2, 7, 3};
        uint32_t res = 0;
        // 8个 uint4 数据按 指定顺序 打包成 int32
        for (int i = 0; i < 8; i++) {
            res |= vals[pack_idx[i]] << (i * 4);
        }
        // 结果矩阵的 int32 索引
        constexpr int tile_size = tile_k_size * tile_n_size / 8; // 一个 [16, 128]的数据块包含多少 个int32数据
        int n_tiles = size_n / tile_n_size; //  size_n
        int out_offset = batch_idx * size_n * size_k / 8 + (k_tile_idx * n_tiles + n_tile_idx) * tile_size;
        // 处理完每个线程拿到自己负责线程 需要的 1个int32 写回全局内存
        out_ptr[out_offset + threadIdx.x] = res;        
    }

}
}
}