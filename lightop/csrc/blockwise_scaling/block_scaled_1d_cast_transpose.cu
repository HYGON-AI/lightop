/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <condition_variable>
#include <type_traits>
#include <utility>
#include "common.h"
#include "block_scaled_1d_cast_transpose.h"
#include <ATen/ATen.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

namespace at { 
namespace native {


constexpr size_t kThreadsPerWarp = 64;

// Hyperparameters for performance tuning
constexpr int kTileDim64 = 64;
constexpr int kTileDim = 128;  // Fixed to 128 beacause we are using 1x128 and 128x1 quantization
constexpr int kNVecIn = 8;     // The number of elements each LDG touches
constexpr int kNVecOut = 16;   // The number of elements each STG touches
constexpr int kNVecSMem = 4;   // The number of elements each LDS/STS touches
constexpr int kThreadsPerBlock = 512;  // Thread block size, 8 warps in total
 
// Auto-calculated constants, do not modify directly)
static_assert(kNVecIn % kNVecSMem == 0, "kNVecIn must be divisible by kNVecSMem");
static_assert(kNVecOut % kNVecSMem == 0, "kNVecOut must be divisible by kNVecSMem");
constexpr int kSMemRow = kTileDim64;
constexpr int kSMemCol = (kTileDim / kNVecSMem) ;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;
constexpr int kNumThreadsStore = kTileDim / kNVecOut;
static_assert(kNumThreadsLoad <= kThreadsPerWarp, "kNumThreadsLoad must be <= kThreadsPerWarp");
static_assert(kNumThreadsStore <= kThreadsPerWarp, "kNumThreadsStore must be <= kThreadsPerWarp");
constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;




constexpr int kFP32SMemCol = kTileDim / kNVecSMem;
constexpr int kFP32SMemSize = kSMemRow * kFP32SMemCol * kNVecSMem;

template <bool kAligned, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) block_scaled_1d_cast_transpose_kernel_rolwise(
    const IType* const input, OType* const output_c, 
    CType* const tile_scales_inv_c, const size_t row_length,
    const size_t num_rows, const size_t scale_stride_x, const size_t scale_stride_y,
    const float epsilon,FP8BlockwiseRowwiseOption rowwise_option,
    const bool pow_2_scaling) {
  bool return_rowwise = rowwise_option != FP8BlockwiseRowwiseOption::NONE;

  using SMemVec = Vec<IType, kNVecSMem>;
  using OVec = Vec<OType, kNVecOut>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };
  
  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(smem_base);
  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsLoad;  // stride in rows of shared memory 
    constexpr int num_iterations = kTileDim64 / r_stride;//64/16=4
    const int c_s =
        (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);  // Column in shared memory 
    int r_s = threadIdx.x / kNumThreadsLoad;                      // Row in shared memory 
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem;    // Column in global memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim64 + r_s;       // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = c_g < row_length ? std::min(static_cast<size_t>(kNVecIn), row_length - c_g)
                                            : 0;            // For not aligned case
    const IType* input_g = &input[r_g * row_length + c_g];  // Input address in global memory
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      // Step 1.1: Load from global memory (input) to registers
      if constexpr (kAligned) {
        input_vec.input_type.load_from(input_g);
      } else {
        if (r_g < num_rows) {
          input_vec.input_type.load_from_elts(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
      // Step 1.2: Write to shared memory - row Major
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        // row Major Store
       smem[r * kSMemCol + c] = input_vec.smem_type.data.elt[i];
      }
      // Step 1.3: Update input address, row index of shared memory, (and row index of global memory for not aligned case)
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();
 
  // Step 2: Cast and store to output_c
  if (return_rowwise) {
    constexpr int r_stride =
        kThreadsPerBlock / kNumThreadsStore;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim64 / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);  // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsStore;                       // Row in shared memory 
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem;    // Column in global memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim64 + r_s;       // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = c_g < row_length ? std::min(static_cast<size_t>(kNVecOut), row_length - c_g)
                                            : 0;          // For not aligned case
    OType* output_g = &output_c[r_g * row_length + c_g];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsStore * kNumThreadsStore;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsStore) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;

#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut / kNVecSMem];
      // Step 2.1: Load from shared memory to registers - Column Major
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        // Column Major Read
        smem_vec[i] = smem[r * kSMemCol + c];
      }
    
      // Step 2.2: Compute local amax
      CType amax = 0;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[j]));
        }
      }
   
      // Step 2.3: Reduce amax
#pragma unroll
      for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
       
        //const float other_amax =__shfl_xor_sync((unsigned long long)(mask), amax, delta, kThreadsPerWarp);
        const float other_amax =__shfl_xor( amax, delta, kThreadsPerWarp);
        __builtin_assume(amax >= 0);
        __builtin_assume(other_amax >= 0);
       
        amax = fmaxf(amax, other_amax);
      }
     
      CType scale;
      // Step 2.4: Compute scale
      scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
      // Step 2.5: Write scale_inv
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
      }
      if (write_scale_inv) {
        CType scale_inv = 1.0 / scale;
        size_t row_idx = static_cast<size_t>(blockIdx.y) * kTileDim64 + r_s;//
        size_t col_idx = static_cast<size_t>(blockIdx.x);
        tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
      }
      // Step 2.6: Quantize
      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          if constexpr (std::is_same_v<OType, int8_t>) {
            output_vec.data.elt[i * kNVecSMem + j] = static_cast<OType>(lroundf(fmaxf(
                -127.0f, fminf(127.0f, static_cast<CType>(smem_vec[i].data.elt[j]) * scale))));
          } else {
            output_vec.data.elt[i * kNVecSMem + j] =
                static_cast<OType>(static_cast<CType>(smem_vec[i].data.elt[j]) * scale);
          }
        }
      }
      // Step 2.7: Store output_c
      if constexpr (kAligned) {
        output_vec.store_to(output_g);
      } else {
        if (r_g < num_rows) {
          output_vec.store_to_elts(output_g, 0, num_ele);
        }
      }
      // Step 2.8: Update output address, row index of shared memory (and row index of global memory for not aligned case)
      output_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }


}

constexpr int kTileDim64_Colwise=64;
constexpr int kNVecSMem_Colwise = 2;
constexpr int kSMemRow_Colwise = kTileDim;
constexpr int kSMemCol_Colwise = (kTileDim64_Colwise / kNVecSMem_Colwise) ;
constexpr int kSMemSize_Colwise = kSMemRow_Colwise * kSMemCol_Colwise * kNVecSMem_Colwise;
constexpr int kNumThreadsLoad_Colwise = kTileDim64_Colwise / kNVecIn;
constexpr int kNumThreadsStore_Colwise = kTileDim / kNVecOut;
constexpr int kThreadsPerBlock_Colwise =256;
constexpr int kNumWarps_Colwise=kThreadsPerBlock_Colwise/kThreadsPerWarp;

template <bool kAligned, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock_Colwise) block_scaled_1d_cast_transpose_kernel_colwise(
    const IType* const input,  OType* const output_t, CType* const tile_scales_inv_t, const size_t row_length,
    const size_t num_rows,  const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
   FP8BlockwiseColumnwiseOption columnwise_option,const bool pow_2_scaling) {


  bool return_columnwise_gemm_ready =
      columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
  bool return_columnwise_compact =
      columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT;

  using SMemVec = Vec<IType, kNVecSMem_Colwise>;
  using OVec = Vec<OType, kNVecOut>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem_Colwise> smem_type;
  };

  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(smem_base);
  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock_Colwise / kNumThreadsLoad_Colwise;  // stride in rows of shared memory 
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsLoad_Colwise) * (kNVecIn / kNVecSMem_Colwise);  // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsLoad_Colwise;                      // Row in shared memory
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim64_Colwise + c_s * kNVecSMem_Colwise;    // Column in global memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;       // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = c_g < row_length ? std::min(static_cast<size_t>(kNVecIn), row_length - c_g)
                                            : 0;            // For not aligned case
    const IType* input_g = &input[r_g * row_length + c_g];  // Input address in global memory
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      // Step 1.1: Load from global memory (input) to registers
      if constexpr (kAligned) {
        input_vec.input_type.load_from(input_g);
      } else {
        if (r_g < num_rows) {
          input_vec.input_type.load_from_elts(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
      // Step 1.2: Write to shared memory - Row Major
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem_Colwise; ++i) {
        int c = c_s + i;
        int r = r_s;
        // Row Major Store
        smem[r * kSMemCol_Colwise + c] = input_vec.smem_type.data.elt[i];
      }
      // Step 1.3: Update input address, row index of shared memory, (and row index of global memory for not aligned case)
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();

  // Step 3: Transpose, cast and store to output_t
  if (return_columnwise_gemm_ready) {
    constexpr int c_stride =
        kThreadsPerBlock_Colwise / kNumThreadsStore_Colwise;  // Stride in columns of shared memory 
    constexpr int num_iterations = kTileDim64_Colwise / (c_stride * kNVecSMem_Colwise);
    const int r_s = (threadIdx.x % kNumThreadsStore_Colwise) * kNVecOut;  // Row in shared memory 
    int c_s = threadIdx.x / kNumThreadsStore_Colwise;                     // Column in shared memory
    size_t r_g =
        static_cast<size_t>(blockIdx.x) * kTileDim64_Colwise + c_s * kNVecSMem_Colwise;     // Row in global memory
    const size_t c_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;  // Column in global memory
    const size_t stride_g =
        static_cast<size_t>(c_stride) * kNVecSMem_Colwise * num_rows;  // Stride in global memory
    const size_t num_ele = c_g < num_rows ? std::min(static_cast<size_t>(kNVecOut), num_rows - c_g)
                                          : 0;          // For not aligned case
    OType* output_g = &output_t[r_g * num_rows + c_g];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsStore_Colwise * kNumThreadsStore_Colwise;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsStore_Colwise) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore_Colwise) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut];
      // Step 3.1: Load from shared memory to registers - Column Major
#pragma unroll
      for (int i = 0; i < kNVecOut; ++i) {
        int r = r_s + i;
        int c = c_s;
        // Row Major Read
        smem_vec[i] = smem[r * kSMemCol_Colwise + c];
      }
#pragma unroll
      for (int smem_idx = 0; smem_idx < kNVecSMem_Colwise; ++smem_idx) {
        // Step 3.2: Compute local amax
        CType amax = 0;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[smem_idx]));
        }
        // Step 3.3: Reduce amax
#pragma unroll
        for (int delta = kNumThreadsStore_Colwise / 2; delta > 0; delta /= 2) {

          // const float other_amax =
          //     __shfl_xor_sync((unsigned long long)(mask), amax, delta, kThreadsPerWarp);
          const float other_amax =__shfl_xor( amax, delta, kThreadsPerWarp);

          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }

        // Step 3.4: Compute scale
        CType scale;
        scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
        }
        if (write_scale_inv) {
          CType scale_inv = 1.0 / scale;
          size_t row_idx = static_cast<size_t>(blockIdx.x) * kTileDim64_Colwise + c_s * kNVecSMem_Colwise + smem_idx;
          size_t col_idx = static_cast<size_t>(blockIdx.y);
          tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
        }
        // Step 3.6: Quantize
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          if constexpr (std::is_same_v<OType, int8_t>) {
            output_vec.data.elt[i] = static_cast<OType>(lroundf(
                fmaxf(-127.0f,
                      fminf(127.0f, static_cast<CType>(smem_vec[i].data.elt[smem_idx]) * scale))));
          } else {
            output_vec.data.elt[i] =
                static_cast<OType>(static_cast<CType>(smem_vec[i].data.elt[smem_idx]) * scale);
          }
        }
        // Step 3.7: Store output_t
        if constexpr (kAligned) {
          output_vec.store_to(output_g + smem_idx * num_rows);
        } else {
          if (r_g + smem_idx < row_length) {
            output_vec.store_to_elts(output_g + smem_idx * num_rows, 0, num_ele);
          }
        }
      }
      // Step 3.8: Update output address, column index of shared memory (and row index of global memory for not aligned case)
      output_g += stride_g;
      c_s += c_stride;
      if constexpr (!kAligned) {
        r_g += c_stride * kNVecSMem_Colwise;
      }
    }
  }


  if (return_columnwise_compact) {
    // thread tile should be 4x16, 16 means 8 smem reads
    constexpr int kThreadTileRow = kTileDim / kThreadsPerWarp;
    constexpr int kThreadTileCol = kNVecOut;
    using RegVec = Vec<IType, kThreadTileCol>;
    using RegScaleVec = Vec<CType, kThreadTileCol>;//float,16 
    constexpr int num_smem_reads = kNVecOut / kNVecSMem_Colwise;
    // c_stride will not be used here because we only have one iteration
    // constexpr int c_stride = kThreadTileCol * kNumWarps / kNVecSMem;
    constexpr int num_iterations =
        kTileDim64_Colwise / (kNumWarps_Colwise * kThreadTileCol);  // should be only one iteration 
    static_assert(num_iterations == 1,
                  "num_iterations should be 1 for columnwise non-transpose case");
    const int thr_idx_in_warp = threadIdx.x % kThreadsPerWarp;
    const int warp_idx = threadIdx.x / kThreadsPerWarp;
    const int r_s = thr_idx_in_warp * kThreadTileRow;               // Row in shared memory
    int c_s = warp_idx * num_smem_reads;                            // Column in shared memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;  // Row in global memory
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim64_Colwise + c_s * kNVecSMem_Colwise;  // Column in global memory
    const size_t num_ele = c_g < row_length
                               ? std::min(static_cast<size_t>(kThreadTileCol), row_length - c_g)
                               : 0;  // For not aligned case
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      RegVec reg_vec[kThreadTileRow];
      RegScaleVec thr_scale;

      // Step 3.1: Load from shared memory to registers
#pragma unroll
      for (int i = 0; i < kThreadTileRow; ++i) {
        int r = r_s + i;
#pragma unroll
        for (int j = 0; j < num_smem_reads; ++j) {
          int c = c_s + j;
          SMemVec smem_vec = smem[r * kSMemCol_Colwise + c];
          // copy smem_vec to reg vec with its elements
#pragma unroll
          for (int k = 0; k < kNVecSMem_Colwise; ++k) {
            reg_vec[i].data.elt[j * kNVecSMem_Colwise + k] = smem_vec.data.elt[k];
          }
        }
      }
#pragma unroll
      for (int reg_idx = 0; reg_idx < kThreadTileCol; ++reg_idx) {
        // Step 3.2: Compute local amax
        CType amax = 0;
#pragma unroll
        for (int i = 0; i < kThreadTileRow; ++i) {
          amax = fmaxf(amax, fabsf(reg_vec[i].data.elt[reg_idx]));
        }
        // Step 3.3: Reduce amax
        const bool is_src_lane = thr_idx_in_warp == 0;
#pragma unroll
        for (int delta = kThreadsPerWarp / 2; delta > 0; delta /= 2) {

          // const float other_amax =
          //     __shfl_xor_sync((unsigned long long)(0xFFFFFFFF), amax, delta, kThreadsPerWarp);
          const float other_amax =__shfl_xor( amax, delta, kThreadsPerWarp);

          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        // Step 3.4: Compute scale
        CType scale;
        scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
        thr_scale.data.elt[reg_idx] = scale;
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (c_g + reg_idx < row_length);
        }
        if (write_scale_inv) {
          CType scale_inv = 1.0 / scale;
          size_t row_idx = static_cast<size_t>(blockIdx.y);
          size_t col_idx = static_cast<size_t>(blockIdx.x) * kTileDim64_Colwise + c_s * kNVecSMem_Colwise + reg_idx;
          tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
        }
      }
      // Step 3.6: Quantize
      for (int row_idx = 0; row_idx < kThreadTileRow; ++row_idx) {
        OType* output_g =
            &output_t[(r_g + row_idx) * row_length + c_g];  // Output address in global memory
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kThreadTileCol; ++i) {
          if constexpr (std::is_same_v<OType, int8_t>) {
            output_vec.data.elt[i] = static_cast<OType>(lroundf(
                fmaxf(-127.0f,
                      fminf(127.0f, static_cast<CType>(reg_vec[row_idx].data.elt[i]) * thr_scale.data.elt[i]))));
          } else {
            output_vec.data.elt[i] = static_cast<OType>(
                static_cast<CType>(reg_vec[row_idx].data.elt[i]) * thr_scale.data.elt[i]);
          }
        }
        // Step 3.7: Store output_t
        if constexpr (kAligned) {
          output_vec.store_to(output_g);
        } else {
          if (r_g + row_idx < num_rows) {
            output_vec.store_to_elts(output_g, 0, num_ele);
          }
        }
      }
      // Step 3.8: Update output address, column index of shared memory
      // this section shouldn't matter since we only have one iteration
    }
  }

}


void blockwise_scaled_1d_cast_transpose(const Tensor& input,std::optional<Tensor> output,std::optional<Tensor> scale_inv,
                                        std::optional<Tensor> output_t,std::optional<Tensor> scale_inv_t,std::optional<Tensor> noop,
                                        const float epsilon,const bool pow2_scale,const bool all_gather_usage, at::ScalarType out_dtype) {

  const size_t row_length = input.sizes().size() > 0 ? input.sizes().at(input.sizes().size() - 1) : 1u;
  size_t num_elements = row_length;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.sizes().size() - 1) && (input.sizes().size() > 0); ++i) {
    num_rows *= input.sizes().at(i);
    num_elements *= input.sizes().at(i);
  }
 
  // Early return if the input tensor is empty
  if (num_elements == 0) {
    return;
  }

  // Options for scale layout of cuBLAS GEMM kernel.
  size_t scale_stride_x = 0;
  size_t scale_stride_y = 0;
  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;

  FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
  if (output.has_value()) {
        
        rowwise_option = all_gather_usage ? FP8BlockwiseRowwiseOption::ROWWISE_COMPACT
                                         : FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
  }
  if (output_t.has_value()) {
  
        columnwise_option = all_gather_usage
                            ? FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT
                            : FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
  }

  if (rowwise_option != FP8BlockwiseRowwiseOption::NONE) {
    TORCH_CHECK(rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY ||
                   rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_COMPACT,
               "Unexpected rowwise enum value");
    TORCH_CHECK(input.sizes() == output.value().sizes(), "Input and output must have the same shape.");
    TORCH_CHECK(scale_inv.value().sizes().size() == 2, "Scale dimension must be 2.");
    size_t scale_k = scale_inv.value().sizes()[1];
    bool rowwise_compact = rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_COMPACT;
    scale_stride_x = rowwise_compact ? 1 : scale_k;
    scale_stride_y = rowwise_compact ? scale_k : 1;
  }

  if (columnwise_option != FP8BlockwiseColumnwiseOption::NONE) {
    TORCH_CHECK(output_t.value().sizes().size() == input.sizes().size(),
               "output_t must have same number of dimensions as input.");

    if (output_t.value().sizes().size() > 0) {
      if (columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY) {
        TORCH_CHECK(output_t.value().sizes()[0] == row_length, "Wrong dimension 0 of output_t.");
        for (size_t i = 1; i < output_t.value().sizes().size(); ++i) {
          TORCH_CHECK(output_t.value().sizes().at(i) == input.sizes().at(i - 1), "Wrong dimension in output_t");
        }
      } else {
        TORCH_CHECK(columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT,
                   "Unexpected columnwise option enum value");
        TORCH_CHECK(output_t.value().sizes()[0] == input.sizes()[0], "Wrong dimension 0 of output_t.");
        TORCH_CHECK(
            input.sizes() == output_t.value().sizes(),
            "Input and output_t must have the same shape for columnwise non-transpose case.");
      }
    }
   
    TORCH_CHECK(output.value().scalar_type() == output_t.value().scalar_type(), "output and output_t need to have the same dtype.");
    TORCH_CHECK(scale_inv_t.value().sizes().size() == 2, "Scale_t dimension must be 2.");
    bool columnwise_compact = columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT;
    size_t scale_t_k = scale_inv_t.value().sizes()[1];
    scale_t_stride_x = columnwise_compact ? 1 : scale_t_k;
    scale_t_stride_y = columnwise_compact ? scale_t_k : 1;
  }
 
  const size_t block_len = 128;

  const size_t num_blocks_x = DIVUP(row_length, (size_t)block_len);
  const size_t num_blocks_y = DIVUP(num_rows, (size_t)(block_len/2));
  
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.scalar_type(), InputType,
      
      TRANSFORMER_ENGINE_TYPE_SWITCH_8BIT(
          out_dtype, OutputType,
          
          dim3 grid(num_blocks_x, num_blocks_y, 1);

          const bool full_tile = row_length % block_len == 0 && num_rows % block_len == 0;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              full_tile, kAligned,
              while (true) {
                if (128 == block_len) {
                   
                    if (rowwise_option != FP8BlockwiseRowwiseOption::NONE)
                    {
                      size_t smem_bytes = kFP32SMemSize * sizeof(InputType);
                      if (smem_bytes >= 48 * 1024) {
                        cudaError_t err = cudaFuncSetAttribute(
                            (const void*)&block_scaled_1d_cast_transpose_kernel_rolwise<
                                kAligned, float, InputType, OutputType>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                        
                      }
                  
                      block_scaled_1d_cast_transpose_kernel_rolwise<kAligned, float, InputType,
                                                                OutputType>
                          <<<grid, kThreadsPerBlock, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                              reinterpret_cast<const InputType*>(input.data_ptr()),
                              reinterpret_cast<OutputType*>(output.value().data_ptr()),
                              reinterpret_cast<float*>(scale_inv.value().data_ptr()),
                              row_length, num_rows,scale_stride_x, scale_stride_y, 
                              epsilon, rowwise_option, pow2_scale);
                    }
                    if (columnwise_option != FP8BlockwiseColumnwiseOption::NONE)
                    {
                  
                        size_t smem_bytes = kSMemSize_Colwise * sizeof(InputType);
                        const size_t num_blocks_x = DIVUP(row_length, (size_t)(block_len/2));//1024/64=16
                        const size_t num_blocks_y = DIVUP(num_rows, (size_t)(block_len));//2
                        dim3 grid(num_blocks_x, num_blocks_y, 1);
                        cudaError_t err = cudaFuncSetAttribute(
                            (const void*)&block_scaled_1d_cast_transpose_kernel_colwise<
                                kAligned, float, InputType, OutputType>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                        block_scaled_1d_cast_transpose_kernel_colwise<kAligned, float, InputType,
                                                                OutputType>
                          <<<grid, kThreadsPerBlock_Colwise, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                             reinterpret_cast<const InputType*>(input.data_ptr()),
                              reinterpret_cast<OutputType*>(output_t.value().data_ptr()),
                              reinterpret_cast<float*>(scale_inv_t.value().data_ptr()), 
                              row_length, num_rows,
                              scale_t_stride_x, scale_t_stride_y,
                              epsilon, columnwise_option, pow2_scale);
                    }
                   
                  break;
                }
              }

              )  // kAligned
          )      // OutputType
      )          // InputType
  TORCH_CHECK_CUDA(cudaGetLastError());
}


}  // 
}  // 