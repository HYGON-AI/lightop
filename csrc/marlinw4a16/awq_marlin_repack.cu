// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "marlin.h"
#include "marlin_dtypes.h"

namespace at {
namespace native {
namespace marlin_moe_w4a16{
    
    template <int const num_threads, int const num_bits>
    __global__ void awq_marlin_repack_kernel(
        uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
        int size_k, int size_n) {

        constexpr int pack_factor = 32 / num_bits; // 8
        
        int k_tiles = size_k / tile_k_size; //  size_k / 16  448
        int n_tiles = size_n / tile_n_size; //  size_n / 128   4
        int block_k_tiles = div_ceil(k_tiles, gridDim.x);  // K 维度 每个 block 需要处理的数据块数

        auto start_k_tile = blockIdx.x * block_k_tiles; // 开始处理的 K 维度 数据块 索引
        if (start_k_tile >= k_tiles) {
            return;
        }

        int finish_k_tile = device_min(start_k_tile + block_k_tiles, k_tiles); // 结束处理的 K维度 数据块边界

        extern __shared__ int4 sh[];

        constexpr int tile_n_ints = tile_n_size / pack_factor; // 数据块N方向含有的 int32 个数 16

        constexpr int stage_n_threads = tile_n_ints / 4; // n方向需要用几个线程进行处理  4
        constexpr int stage_k_threads = tile_k_size;  // k方向需要几个线程进行处理  16 
        constexpr int stage_size = stage_k_threads * stage_n_threads; // 一个数据块需要几个线程进行处理（int4） 

        // load 一个 [16, 128](uint4)的块数据到LDS中
        auto fetch_to_shared = [&](int pipe, // 标志写入的是第几个数据块
                                    int k_tile_id, // k方向的行索引 
                                    int n_tile_id) { // n方向的数据块索引
            // 越过N维度边界之后，直接退出
            if (n_tile_id >= n_tiles) {
                return;
            }

            int first_n = n_tile_id * tile_n_size; // n维度的 起始索引(uint4)
            int first_n_packed = first_n / pack_factor; // n维度的 起始索引(int32)

            int4* sh_ptr = sh + stage_size * pipe; // 指针偏移到 当前的写入缓冲区

            if (threadIdx.x < stage_size) { // 只有需要load的线程执行 load 操作，一个线程 load 128bit
                auto k_id = threadIdx.x / stage_n_threads; // load时，K维度索引
                auto n_id = threadIdx.x % stage_n_threads; // load时，N维度索引

                int first_k = k_tile_id * tile_k_size; // awq量化后的 起始k维度索引

                // 直接读取数据到 LDS中
                    // 以int32 为数据宽度，计算偏移
                int read_int32_offset = (first_k + k_id) * (size_n / pack_factor) + first_n_packed + (n_id * 4);
                    // load 128bit数据到 LDS中
                sh_ptr[k_id * stage_n_threads + n_id] = *reinterpret_cast<const int4*>(&(b_q_weight_ptr[read_int32_offset]));
            }
        };

        auto repack_tile = [&](int pipe,           // 标志处理哪个LDS缓冲区中的数据
                                int k_tile_id,     // K维度的数据块索引
                                int n_tile_id) {   // N维度的数据块索引
            // 如果越过N维度的边界，退出函数执行
            if (n_tile_id >= n_tiles) {
                return;
            }

            auto warp_id = threadIdx.x / 64;  // 当前线程属于的warp id
            auto th_id = threadIdx.x % 64;    // 当前线程在 warp 内部的索引

            int tc_col = th_id % 16;  // 每个线程在列方向的索引（uint4）
            int tc_row = (th_id / 16) * 4; // 每个线程在行方向的索引(uint4)

            constexpr int tc_offsets[4] = {0, 1, 2, 3}; // 一个线程[16, 16]中 load 4个数据列方向上的偏移

            int cur_n = warp_id * 32 + tc_col; // 每个 warp 起始的列索引 （uint4）
            int cur_n_packed = cur_n / pack_factor; // 每个 warp 起始的列索引 （int32）
            int cur_n_pos = cur_n % pack_factor; // 当前线程负责的数据在 int32打包中所在的打包位置

            constexpr int sh_stride = tile_n_ints;  // 16 N维度int32数据元素的个数
            constexpr uint32_t mask = (1 << num_bits) - 1; // 保留 32位中的低 4 位bit位

            int4* sh_stage_ptr = sh + stage_size * pipe; // LDS中的数据位置
            uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr); // 转成uint32指针

            // Undo interleaving
            int cur_n_pos_unpacked;
            if constexpr (num_bits == 4) {
                constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // 按照原先的索引，去int32的哪个 位置去拿实际的数据
                cur_n_pos_unpacked = undo_pack[cur_n_pos];
            } else {
                constexpr int undo_pack[4] = {0, 2, 1, 3};
                cur_n_pos_unpacked = undo_pack[cur_n_pos];
            }

            // 拿到自己线程需要处理的 8个 uint4 数据
            uint32_t vals[8]; // 保存 每个线程 执行计算的 8个uint4数据
            for (int i = 0; i < 4; i++) {
                int cur_elem = tc_row + tc_offsets[i]; // 获取当前 线程处理的行索引

                int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
                int packed_src_1 = sh_stage_int_ptr[cur_n_packed + 2 + sh_stride * cur_elem];

                vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
                vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
            }

            constexpr int tile_size = tile_k_size * tile_n_size / pack_factor; // 一个 [16, 128]的数据块包含多少 个int32数据
            int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size; // 计算当前 [16, 128]处理好的数据应该存放的起始偏移（int32）

            // 将每个线程需要处理的8个数据先按照奇偶混排 打包成 int32 ,再写到对应的位置
            // Result of:
            // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
            if constexpr (num_bits == 4) {
                constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

                uint32_t res = 0;
                // 8个 uint4 数据按 奇偶混排 顺序打包成 int32
                for (int i = 0; i < 8; i++) {
                    res |= vals[pack_idx[i]] << (i * 4);
                }
                // 处理完每个线程拿到自己负责线程 需要的 1个int32
                out_ptr[out_offset + th_id * 4 + warp_id] = res;

            } else {
                constexpr int pack_idx[4] = {0, 2, 1, 3};

                uint32_t res1 = 0;
                uint32_t res2 = 0;
                for (int i = 0; i < 4; i++) {
                    res1 |= vals[pack_idx[i]] << (i * 8);
                    res2 |= vals[4 + pack_idx[i]] << (i * 8);
                }

                out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
                out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
            }
        };

        // 每个迭代 需要处理K维度的一个数据块 所在的一行
        for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
            int n_tile_id = 0;
            while (n_tile_id < n_tiles) {
                for(int pipe = 0; pipe < repack_stages; pipe++){
                    fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
                }
                // 等待所有 warp 将数据load到 LDS中
                __syncthreads();

                // 等待 global 到 LDS 的 load 完成，这里使用的是直接赋值的方式，由编译器进行 swait 判断
                
                for (int pipe = 0; pipe < repack_stages; pipe++) { 
                    // 迭代处理一个 [16, 128] 的数据块 
                    repack_tile(pipe, k_tile_id, n_tile_id + pipe);
                }
                n_tile_id += repack_stages;
            }
        }
    }

}
}
}