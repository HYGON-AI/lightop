// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "awq_w4a16_common.h"

namespace at {
namespace native {

template <typename scalar_t>
__global__ void awq_w4a16_dequant_marlin_kernel(scalar_t* ret, int32_t* weight, int32_t* scales_zeros, int N, int K, int MarlinTileK, int G)
{

    constexpr int AWQQuantSize = 8;
    constexpr int VecSize = 8;
    const int WarpSize = C10_WARP_SIZE;
    const int NumWarps = __builtin_amdgcn_readfirstlane(blockDim.x / WarpSize);
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / WarpSize);
    const int lane = tid % WarpSize;
    const int n = blockIdx.x * NumWarps + warp_idx;
    if(n >= N) return;
    const int n_int_weight_offset = (n*MarlinTileK / AWQQuantSize);
    weight += n_int_weight_offset;
    const int n_int_scale_offset = n*K/G;
    scales_zeros += n_int_scale_offset;
    const int n_ret_offset = n*K;
    ret += n_ret_offset;
    const int marlin_tile_k_int_offset = (N*MarlinTileK / AWQQuantSize);
    const int num_marlin_tile_per_warp = WarpSize*VecSize / MarlinTileK;  // 通常可以一次性跳跃8个MarlinTileK，当MarlinTileK=256
    half8_t b_vec;
    for(int k = lane*VecSize; k < K; k += WarpSize*VecSize) {
        const int marlin_tile_k_outer_id =  k / MarlinTileK;
        const int marlin_tile_k_inner_id =  (k % MarlinTileK) / AWQQuantSize;
        const int tb = *(weight + marlin_tile_k_outer_id * marlin_tile_k_int_offset + marlin_tile_k_inner_id);
        half2_t t_scale = *(half2_t*)(scales_zeros + k/G);
        dequantize_s4_to_fp16x2(tb, (uint32_t*)&b_vec);
        auto b_vec_half = reinterpret_cast<half_t*>(&b_vec);

        #pragma unroll
        for(int kk = 0; kk < 8; kk++) {
            b_vec_half[kk]  = (b_vec_half[kk] - t_scale[0]) * t_scale[1];
        }
        *(half8_t*)(ret + k) = b_vec;

    }


}

  
void gemm_awq_w4a16_dequant(Tensor& ret, const Tensor& weight,const Tensor& scales_zeros, bool use_marlin)
{
    // int temp1;
    // constexpr int BlockN = 16*16;
    // constexpr int BlockK = 128;
    // constexpr int AWQQuantSize = 8;
    // const int N = use_marlin ? weight.sizes()[1]/16 : weight.sizes()[0];
    // const int K = weight.sizes()[0]*BlockK;
    // const int GroupSize = K / scales_zeros.sizes()[1];
    // auto stream = at::cuda::getCurrentCUDAStream();
    // auto out_dtype = ret.scalar_type();
    // // printf("gemm_awq_w4a16_dequant: N=%d, K=%d, GroupSize=%d\n", N, K, GroupSize);
    // Output_Type_SWITCH(out_dtype,[&] {
    //     const int num_block_N = (N-1)/(BlockN)+1;
    //     const int num_block_K = (K-1)/(BlockK)+1;
    //     constexpr int NumWarps = BlockN/16;
    //     // printf("num_block_N=%d, num_block_K=%d\n", num_block_N, num_block_K);
    //     awq_w4a16_dequant_marlin_kernel<scalar_t,NumWarps,BlockN,BlockK>
    //     <<<dim3(num_block_N,num_block_K),dim3(NumWarps*C10_WARP_SIZE)>>>((scalar_t*)ret.data_ptr(),
    //     (int32_t*)weight.data_ptr(),(int32_t*)scales_zeros.data_ptr(),N,K,GroupSize);
    // });
    
    const int N = ret.sizes()[0];
    const int K = ret.sizes()[1];
    const int MarlinTileK = K%256 == 0 ? 256 : 128;
    const int G =K / scales_zeros.sizes()[1];
    auto stream = at::cuda::getCurrentCUDAStream();
    const int NumWarps = 4;
    const int NumBlocks = (N+NumWarps-1)/NumWarps;
    auto out_dtype = ret.scalar_type();
    Output_Type_SWITCH(out_dtype,[&] {
        awq_w4a16_dequant_marlin_kernel<scalar_t><<<NumBlocks, NumWarps*C10_WARP_SIZE, 0, stream>>>((scalar_t*)ret.data_ptr(),
        (int32_t*)weight.data_ptr(),(int32_t*)scales_zeros.data_ptr(),N,K,MarlinTileK,G);
    });
}
 

}
}
