// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "awq_w4a16_common.h"
#include "awq_w4a16.h"
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace native {




  template <typename scalar_t, int NUM_K_WARPS,int NPerWarp,int KPerWarp, int NUM_N_WARPS, bool UseMarlin>
  __global__ void __launch_bounds__(NUM_K_WARPS*NUM_N_WARPS*C10_WARP_SIZE, 2)  gemm_awq_w4a16_kernel(scalar_t* d, scalar_t* a, int32_t* b, int32_t* scale_zeros, int M, int N, int K, int GroupSize)
  {
  
    auto scale_n = reinterpret_cast<const half2_t*>(scale_zeros);
    constexpr int KSizePerWarp = KPerWarp;            // 每个warp计算的k的大小
    constexpr int NSizePerWarp = 16;                // 每个warp计算的n的大小
    constexpr int tc_number    = KSizePerWarp / 16; // 每个warp调用tensor core的个数  
    constexpr int AWQQuantSize = 8;
    constexpr int NLoop = NPerWarp / NSizePerWarp;
    constexpr int k_per_thread = tc_number * 4;
    constexpr int warpSize = 64;
    constexpr int kNWarps = NUM_K_WARPS;
    constexpr int KSizePerWarpQuant = KPerWarp / AWQQuantSize; // 我们的warpk=128， 量化后压缩成16
    constexpr int BlockM = 16;
    
    const int bidm = blockIdx.x;
    a += bidm * BlockM * K;
    d += bidm * BlockM * N;
    M = MIN(BlockM, M - bidm * BlockM);

    const int num_element_per_n_warp = NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*M : NPerWarp*M;
    const int StrideA  = K;
    const int StrideB  = K;
    const int StrideC  = N;
    
    
    

    const int warp_n_idx = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int bwid       = __builtin_amdgcn_readfirstlane(blockIdx.y*NUM_N_WARPS+warp_n_idx);
    const int tid       = threadIdx.x;
    const int warp_idx  = __builtin_amdgcn_readfirstlane(tid / warpSize);
    const int lane      = tid % warpSize;
    const int row_id    = lane % NSizePerWarp;
    const int col_id    = lane / NSizePerWarp;  
    const bool row_id_valid = row_id < M;
    const int N_lenth = __builtin_amdgcn_readfirstlane(N - NPerWarp*bwid);
    if(N_lenth <=0) return;
  
    const int block_n = NPerWarp * bwid;
    d += block_n;
    // b += block_n * StrideB / AWQQuantSize;
    if constexpr(UseMarlin){
      // b += block_n * StrideB / AWQQuantSize;
      b +=  block_n * KSizePerWarpQuant;
    } else {
      b += block_n * StrideB / AWQQuantSize;
    }
    scale_n += block_n * StrideB / GroupSize;
    
    using A_VEC_TYPE = __attribute__( (__vector_size__(k_per_thread * sizeof(_Float16)) )) _Float16;
    using B_VEC_TYPE = __attribute__( (__vector_size__((k_per_thread / AWQQuantSize) * sizeof(int32_t)) )) int32_t;
  
    A_VEC_TYPE a_vec{};
    float4_t d_vec[NLoop]; 
    #pragma unroll
    for(int i = 0; i < NLoop; i++) {
        d_vec[i] = {0, 0, 0, 0};
    }
    for(int i = warp_idx * KSizePerWarp; i < K; i += KSizePerWarp * kNWarps) {
      const int k_idx = i + col_id * k_per_thread;
      if(row_id < M) {
        a_vec = *(A_VEC_TYPE*)(a + row_id * StrideA + k_idx);
      }
      for(int nn = 0; nn < NLoop; nn++) {
        const int n_idx = NSizePerWarp * nn + row_id;
        // int weight_offset = (n_idx * StrideB + k_idx) / AWQQuantSize;
        int weight_offset;
        const int line_offset = n_idx * StrideB + k_idx;
        if constexpr(UseMarlin){
          weight_offset = n_idx * KSizePerWarpQuant + (col_id*k_per_thread)/AWQQuantSize + (k_idx / KPerWarp)*N*KSizePerWarpQuant;
        } else {
          weight_offset = line_offset / AWQQuantSize;
        }
        auto tb         = *(B_VEC_TYPE*)(b + weight_offset);
        half2_t t_scale = scale_n[line_offset / GroupSize];
        half8_t b_vec;
        #pragma unroll
        for(int ii = 0; ii < tc_number/2; ii++) {
            dequantize_s4_to_fp16x2(tb[ii], (uint32_t*)&b_vec);
            auto b_vec_half = reinterpret_cast<half_t*>(&b_vec);
            #pragma unroll
            for(int kk = 0; kk < 8; kk++) {
                b_vec_half[kk]  = (b_vec_half[kk] - t_scale[0]) * t_scale[1];
            }
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2), *((half4_t*)&b_vec), d_vec[nn]);
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2 + 1), *((half4_t*)&b_vec + 1), d_vec[nn]);
        } 
      }    
    }
    extern __shared__ float start_smem[];
    float4_t* out_smem = reinterpret_cast<float4_t*>(start_smem + num_element_per_n_warp*warp_n_idx);
    half_t* out_smem_output_ptr = reinterpret_cast<half_t*>(start_smem) + num_element_per_n_warp*warp_n_idx;
  
    if constexpr(NUM_K_WARPS>1){
        int lds_size_n_offset=NUM_K_WARPS/2*16*M/4;
          #pragma unroll
          for (int i = NUM_K_WARPS; i > 1; i /= 2) {
              int mid = i / 2;
              if(row_id_valid)
              {
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * M+row_id*4+col_id+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
              }
              __syncthreads();
              if (warp_idx < mid) {
                  for(int nn=0;nn<NLoop;nn++){
                    if(row_id_valid)
                    {
                      float4_t tmp=out_smem[warp_idx* 4 * M+row_id*4+col_id+lds_size_n_offset*nn];
                      #pragma unroll
                      for (int i = 0; i < 4; i++) {
                          d_vec[nn][i] += tmp[i];
                      }
                    }
                  }
              }
              __syncthreads();
          }
    }
    using out_vec_dtype = half4_t;
    out_vec_dtype out_vec_16bit;
    if(warp_idx == 0) {
      if (row_id_valid) 
      { 
        #pragma unroll
        for(int nn = 0; nn < NLoop; nn++) {
            int out_smem_offset = nn*NSizePerWarp*M;
            #pragma unroll
            for(int i = 0; i < 4; i++) {
              out_smem_output_ptr[out_smem_offset + row_id * NSizePerWarp + i*4 + col_id] = static_cast<half_t>(d_vec[nn][i]);
            }
        }
      }
    }
    __syncthreads();   
    {
      if(row_id_valid)
      {
        for(int nn = warp_idx; nn < NLoop; nn+=NUM_K_WARPS) {
          int out_smem_offset = nn*NSizePerWarp*M;
          const bool n_idx_valid = nn*NSizePerWarp+col_id * 4 < N_lenth;
          if(n_idx_valid)
          {
            out_vec_16bit = *reinterpret_cast<out_vec_dtype*>(out_smem_output_ptr + out_smem_offset + row_id * NSizePerWarp + col_id * 4);
            *reinterpret_cast<out_vec_dtype*>(d + row_id * N + nn * NSizePerWarp + col_id * 4) = out_vec_16bit;
          }
        }
      }
  
    }
  
  }


  template <typename scalar_t, int NUM_K_WARPS,int NPerWarp,int KPerWarp, int NUM_N_WARPS, bool UseMarlin>
  __global__ void gemm_awq_w4a16_kernel_with_splitk(scalar_t* d, scalar_t* a, int32_t* b, int32_t* scale_zeros, float* splitk_work_space, int M, int N, int K, int GroupSize)
  {
  
    auto scale_n      = reinterpret_cast<const half2_t*>(scale_zeros);
  
    constexpr int KSizePerWarp = KPerWarp;            // 每个warp计算的k的大小
    constexpr int NSizePerWarp = 16;                // 每个warp计算的n的大小
    constexpr int tc_number    = KSizePerWarp / 16; // 每个warp调用tensor core的个数
    constexpr int AWQQuantSize = 8;
    constexpr int NLoop = NPerWarp / NSizePerWarp;
    constexpr int k_per_thread = tc_number * 4;
    constexpr int kNWarps = NUM_K_WARPS;
    constexpr int KSizePerWarpQuant = KPerWarp / AWQQuantSize; // 我们的warpk=128， 量化后压缩成16
    constexpr int BlockM = 16;


    const int bidm = __builtin_amdgcn_readfirstlane(blockIdx.x);
    a += bidm * BlockM * K;
    d += bidm * BlockM * N;
    splitk_work_space += int64_t(blockIdx.z) * M * N + int64_t(bidm) * BlockM * N;
    M = MIN(BlockM, M - bidm * BlockM);

    const int bidy      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int bidx      = __builtin_amdgcn_readfirstlane(blockIdx.z);  // splitk

    const int tid       = threadIdx.x;
    const int bwid = __builtin_amdgcn_readfirstlane(bidy*NUM_N_WARPS + threadIdx.y);
    const int warp_idx  = __builtin_amdgcn_readfirstlane(tid / warpSize);
    const int lane      = tid % warpSize;
    const int row_id    = lane % NSizePerWarp;
    const int col_id    = lane / NSizePerWarp;
    const int SplitKBatch = __builtin_amdgcn_readfirstlane(gridDim.z);
    const int part_k    = ((K / KSizePerWarp - 1) / SplitKBatch + 1) * KSizePerWarp;
    const int N_lenth = N - NPerWarp*bwid;
    if(N_lenth<=0) return;
    const int StrideA              = K;
    const int StrideB              = K;
    const int StrideC              = N;
    const int warp_n_idx = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int  num_element_per_n_warp= NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*M : 0;    
  
    float4_t d_vec[NLoop];
    #pragma unroll
    for(int i = 0; i < NLoop; i++) {
        d_vec[i] = {0, 0, 0, 0};
    }
    const int block_n = NPerWarp * bwid;



    a += bidx * part_k;
    d += block_n;
    // b += (block_n * StrideB + part_k * bidx) / AWQQuantSize;
    if constexpr(UseMarlin){
      b +=  block_n * KSizePerWarpQuant  + (part_k * bidx) / KSizePerWarp * (N * KSizePerWarpQuant);
    } else {
      b += (block_n * StrideB + part_k * bidx) / AWQQuantSize;
    }


    scale_n += block_n * StrideB / GroupSize;
    splitk_work_space += block_n;
    const int length_per_block = (bidx == SplitKBatch - 1 ? K - part_k * bidx : part_k);
    using A_VEC_TYPE = __attribute__( (__vector_size__(k_per_thread * sizeof(_Float16)) )) _Float16;
    using B_VEC_TYPE = __attribute__( (__vector_size__((k_per_thread / AWQQuantSize) * sizeof(int32_t)) )) int32_t;
    A_VEC_TYPE a_vec{};
    #pragma unroll
    // for(int i = 0; i < k_per_thread; i++) {
        // *((half_t*)&a_vec + i) = 0;
    // } 
      
    for(int i = warp_idx * KSizePerWarp; i < length_per_block; i += KSizePerWarp * kNWarps) {
      if(row_id < M) {
          a_vec = *(A_VEC_TYPE*)(a + row_id * StrideA + i + col_id * k_per_thread);
      }
      const int k_idx = i + col_id * k_per_thread;
      #pragma unroll
      for(int nn = 0; nn < NLoop; nn++) {
        const int n_idx = NSizePerWarp * nn + row_id;
        // int line_offset = n_idx * StrideB + k_idx;
        int weight_offset;
        const int line_offset = n_idx * StrideB + k_idx;
        if constexpr(UseMarlin){
          weight_offset = n_idx * KSizePerWarpQuant + (col_id*k_per_thread)/AWQQuantSize + (k_idx / KPerWarp)*N*KSizePerWarpQuant;
        } else {
          weight_offset = line_offset / AWQQuantSize;
        }
        auto tb         = *(B_VEC_TYPE*)(b + weight_offset);  
        // A split can start inside a quantization group. Divide the absolute
        // K index once, rather than separately rounding its split/local parts.
        half2_t t_scale = scale_n[(line_offset + part_k * bidx) / GroupSize];
        half8_t b_vec;
        #pragma unroll
        for(int ii = 0; ii < tc_number/2; ii++) {
            dequantize_s4_to_fp16x2(tb[ii], (uint32_t*)&b_vec);
            auto b_vec_half = reinterpret_cast<half_t*>(&b_vec);
            #pragma unroll
            for(int kk = 0; kk < 8; kk++) {
                b_vec_half[kk]  = (b_vec_half[kk] - t_scale[0]) * t_scale[1];
            }
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2), *((half4_t*)&b_vec), d_vec[nn]);
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2 + 1), *((half4_t*)&b_vec + 1), d_vec[nn]);
        } 

      }
    }
  
    extern __shared__ float start_smem[];
    float4_t* out_smem = reinterpret_cast<float4_t*>(start_smem + num_element_per_n_warp*warp_n_idx);
    if constexpr (NUM_K_WARPS > 1) {
      const int lds_size_n_offset = NUM_K_WARPS / 2 * 16 * M / 4;
      #pragma unroll
      for (int width = NUM_K_WARPS; width > 1; width /= 2) {
        const int mid = width / 2;
        if (row_id < M && warp_idx >= mid && warp_idx < width) {
          for (int nn = 0; nn < NLoop; ++nn)
            out_smem[(warp_idx-mid)*4*M + row_id*4 + col_id + lds_size_n_offset*nn] = d_vec[nn];
        }
        __syncthreads();
        if (row_id < M && warp_idx < mid) {
          for (int nn = 0; nn < NLoop; ++nn) {
            const auto tmp = out_smem[warp_idx*4*M + row_id*4 + col_id + lds_size_n_offset*nn];
            for (int i = 0; i < 4; ++i) d_vec[nn][i] += tmp[i];
          }
        }
        __syncthreads();
      }
    }
    // Each split writes a disjoint plane. A second kernel reduces these planes
    // on the same stream, without global counters or cross-CTA synchronization.
    if (warp_idx == 0 && row_id < M) {
      for (int nn = 0; nn < NLoop; ++nn)
        for (int i = 0; i < 4; ++i)
          splitk_work_space[row_id*N + nn*NSizePerWarp + i*4 + col_id] = d_vec[nn][i];
    }
  }

__global__ void reduce_awq_splits(half_t* output, const float* partials,
                                 int64_t count, int splits) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(blockDim.x) * gridDim.x) {
    float sum = 0;
    for (int split = 0; split < splits; ++split) sum += partials[int64_t(split)*count + i];
    output[i] = half_t(sum);
  }
}

void gemm_awq_w4a16(Tensor& ret, const Tensor& x, const Tensor& weight,
    const Tensor& scales_zeros, int num_n_warps, int num_k_warps,
    int n_per_warp, int k_per_warp, int splitk, bool use_marlin) {
  TORCH_CHECK(x.dim() == 2 && ret.dim() == 2 && scales_zeros.dim() == 2 && weight.dim() == 2,
              "AWQ GEMM tensors must be matrices");
  const int64_t m64 = x.size(0), k64 = x.size(1), n64 = scales_zeros.size(0);
  TORCH_CHECK(m64 <= INT32_MAX && n64 <= INT32_MAX && k64 <= INT32_MAX &&
              m64*n64 <= INT32_MAX && m64*k64 <= INT32_MAX && n64*k64 <= INT32_MAX,
              "AWQ GEMM dimensions exceed kernel indexing range");
  const int M = m64, N = n64, K = k64;
  TORCH_CHECK(ret.size(0) == M && ret.size(1) == N, "AWQ output shape mismatch");
  lightop::awq::check_tensor(x, at::kHalf, x.device(), m64*k64, "input");
  lightop::awq::check_tensor(ret, at::kHalf, x.device(), m64*n64, "output");
  lightop::awq::check_tensor(weight, at::kInt, x.device(), n64*(k64/8), "weight");
  lightop::awq::check_tensor(scales_zeros, at::kInt, x.device(), scales_zeros.numel(), "scale_zero");
  at::assert_no_overlap(ret, x);
  at::assert_no_overlap(ret, weight);
  at::assert_no_overlap(ret, scales_zeros);
  TORCH_CHECK((num_n_warps == 1 || num_n_warps == 2) &&
      (num_k_warps == 1 || num_k_warps == 2 || num_k_warps == 4 || num_k_warps == 8 || num_k_warps == 16) &&
      num_n_warps*num_k_warps <= 16 &&
      (n_per_warp == 16 || n_per_warp == 32 || n_per_warp == 64 || n_per_warp == 128) &&
      (k_per_warp == 128 || k_per_warp == 256) && splitk > 0,
      "invalid AWQ launch configuration");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  if (M == 0 || N == 0) return;
  if (K == 0) { ret.zero_(); return; }
  const int64_t groups = scales_zeros.size(1);
  TORCH_CHECK(groups > 0 && K % groups == 0, "invalid AWQ group count");
  const int GroupSize = K / groups;
  TORCH_CHECK(GroupSize % 8 == 0, "AWQ group_size must be divisible by 8");
  if (use_marlin) {
    TORCH_CHECK(K % k_per_warp == 0 && weight.size(0) == K / k_per_warp &&
                weight.size(1) == n64*(k_per_warp/8), "invalid Marlin weight shape");
  } else {
    TORCH_CHECK(weight.size(0) == N && weight.size(1) == K/8, "invalid column-major weight shape");
  }
  // Vector loads and block-wide barriers require complete tiles. Tail shapes
  // and smaller groups use the source dequantizer with the current-stream BLAS.
  if (N % (num_n_warps*n_per_warp) || K % k_per_warp || GroupSize % (k_per_warp/4)) {
    auto dequant = at::empty({N, K}, x.options());
    if (use_marlin && k_per_warp != (K % 256 == 0 ? 256 : 128)) {
      auto ordinary = weight.view({K/k_per_warp,N,k_per_warp/8})
                          .permute({1,0,2}).contiguous().view({N,K/8});
      lightop::awq::dequant_colmajor(dequant, ordinary, scales_zeros, N, K, K, GroupSize);
    } else {
      gemm_awq_w4a16_dequant(dequant, weight, scales_zeros, use_marlin);
    }
    at::mm_out(ret, x, dequant.transpose(0,1));
    return;
  }
  const int tiles = K / k_per_warp;
  splitk = std::min(splitk, tiles);
  const int tiles_per_split = (tiles + splitk - 1) / splitk;
  splitk = (tiles + tiles_per_split - 1) / tiles_per_split;
  Tensor partials;
  if (splitk > 1) partials = at::empty({splitk,M,N}, x.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream();
  UseMarlin_SWITCH(use_marlin,[&] {
    NPerWarp_SWITCH(n_per_warp,[&] {
      NUM_N_WARPS_SWITCH(num_n_warps,[&] {
        NUM_K_WARPS_SWITCH(num_k_warps,[&] {
          KPerWarp_SWITCH(k_per_warp,[&] {
            constexpr int BlockM = 16;
            const int blocks = (N-1)/(NPerWarp*NUM_N_WARPS)+1;
            const int m_blocks = (M-1)/BlockM+1;
            const int64_t lds = NUM_K_WARPS > 1
                ? NUM_K_WARPS/2*NPerWarp*MIN(BlockM,M)*sizeof(float)
                : (splitk > 1 ? 0 : NPerWarp*MIN(BlockM,M)*sizeof(half_t));
            if (splitk > 1) {
              gemm_awq_w4a16_kernel_with_splitk<half_t,NUM_K_WARPS,NPerWarp,KPerWarp,NUM_N_WARPS,UseMarlin>
                  <<<dim3(m_blocks,blocks,splitk),dim3(NUM_K_WARPS*C10_WARP_SIZE,NUM_N_WARPS),lds*NUM_N_WARPS,stream>>>(
                  (half_t*)ret.data_ptr(),(half_t*)x.data_ptr(),(int32_t*)weight.data_ptr(),
                  (int32_t*)scales_zeros.data_ptr(),partials.data_ptr<float>(),M,N,K,GroupSize);
            } else {
              gemm_awq_w4a16_kernel<half_t,NUM_K_WARPS,NPerWarp,KPerWarp,NUM_N_WARPS,UseMarlin>
                  <<<dim3(m_blocks,blocks),dim3(NUM_K_WARPS*C10_WARP_SIZE,NUM_N_WARPS),lds*NUM_N_WARPS,stream>>>(
                  (half_t*)ret.data_ptr(),(half_t*)x.data_ptr(),(int32_t*)weight.data_ptr(),
                  (int32_t*)scales_zeros.data_ptr(),M,N,K,GroupSize);
            }
          });
        });
      });
    });
  });
  lightop::awq::check_launch();
  if (splitk > 1) {
    const int blocks = int(std::min<int64_t>((m64*n64+255)/256,65535));
    reduce_awq_splits<<<blocks,256,0,stream>>>((half_t*)ret.data_ptr(),partials.data_ptr<float>(),m64*n64,splitk);
    lightop::awq::check_launch();
  }
}
}  // namespace native
}  // namespace at
