// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Even-wrap DirectToLDS helpers: exactly 4 wrap amounts {0,2,4,6}.
//
// Geometry (16-bit element):
//   - One issue: 64 lanes × 8 elems = 512 elems = 8 rows × WK(K-bytes=128B)
//   - 4 wraps × 8 rows = M-block of 32
//   - Row map: m = m_block*32 + wrap_id + (lane/8)*4 , wrap_id ∈ [0,4)
//   - wrap_amount = wrap_id * 2
//
// Warp scheduling (unified):
//   issue     = access_id * kNWarps + warp_id
//   wrap_id   = issue % 4
//   row_block = issue / 4
//   Covers all (M-block, wrap) when access_cnt * kNWarps == BM/8.
//
// IMPORTANT — swizzle ≠ op_test/demos/gemm_mmac_wrap_128_demo.cu:
//   that demo still uses 8 wrap groups + stride-8 (m%8, row_block*64).
//
// Read path (MMAC 16×32 tile):
//   Call ds_read_mmac_tile(lds, warp_id, m_base, k_base) with m_base%16==0,
//   k_base%32==0 (k_base panel-local in [0, WK)). Component maps lane → (m,k),
//   even-wrap index, and issues ds_read_b128. First/last dwordx2 → mmac K-lo/hi.
//
// Shape gate (compile-time):
//   - BK * sizeof(E) >= 128 && (BK * sizeof(E)) % 128 == 0
//   - WK * sizeof(E) == 128
//   - BM >= 32 && BM % 32 == 0
//   - kNWarps >= 1 && access_cnt * kNWarps == (BM/32)*4
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

#ifndef WARP_SIZE_GPU
#define WARP_SIZE_GPU 64
#endif

namespace lightop::fp4_mqa::detail::even_wrap {

using uint32x4_t = uint32_t __attribute__((ext_vector_type(4)));

template <typename Element, int BM, int BK, int WK, int kNWarps>
struct Traits {
  static constexpr int kElemBytes = static_cast<int>(sizeof(Element));
  static constexpr int kBkBytes = BK * kElemBytes;
  static constexpr int kWkBytes = WK * kElemBytes;

  static constexpr int kElemsPerAccess = 16 / kElemBytes;
  static constexpr int kNumWraps = 4;
  static constexpr int kRowsPerWrapIssue = 8;
  static constexpr int kMBlock = kNumWraps * kRowsPerWrapIssue;  // 32
  static constexpr int kElemsPerWarpAccess =
      WARP_SIZE_GPU * kElemsPerAccess;  // 512 for 2B

  static constexpr bool kWarpsOk = (kNWarps >= 1);

  // Distinct M-blocks touched by one access when issues are consecutive.
  static constexpr int kMBlocksPerAccess =
      (kNWarps + kNumWraps - 1) / kNumWraps;

  static constexpr bool kBkOk =
      (kBkBytes >= 128) && (kBkBytes % 128 == 0);
  static constexpr bool kWkOk = (kWkBytes == 128);
  static constexpr bool kBmOk =
      (BM >= kMBlock) && (BM % kMBlock == 0) &&
      (BM / kMBlock >= kMBlocksPerAccess);

  static constexpr int kAccessCnt =
      (kNWarps > 0)
          ? ((BM * WK * kElemBytes) / (16 * kNWarps * WARP_SIZE_GPU))
          : 0;

  static constexpr int kIndependentWrapIssues =
      (BM / kMBlock) * kNumWraps;  // BM/8

  static constexpr bool kAccessOk =
      kWarpsOk && (kAccessCnt >= 1) &&
      ((BM * WK * kElemBytes) % (16 * kNWarps * WARP_SIZE_GPU) == 0) &&
      (kAccessCnt * kNWarps == kIndependentWrapIssues);

  static constexpr bool kWrapOk =
      kBmOk && (kIndependentWrapIssues >= kNumWraps) &&
      (kIndependentWrapIssues % kNumWraps == 0);

  static constexpr bool kOk =
      kBkOk && kWkOk && kBmOk && kWarpsOk && kAccessOk && kWrapOk;

};

template <typename Element, int BM, int BK, int WK, int kNWarps>
inline constexpr bool is_supported_v =
    Traits<Element, BM, BK, WK, kNWarps>::kOk;

template <typename Element, int BM, int BK, int WK, int kNWarps>
using EnableIfSupported = std::enable_if_t<
    is_supported_v<Element, BM, BK, WK, kNWarps>, int>;

// Resolve (wrap_id, row_block) for one access_id / warp.
template <typename Element, int BM, int BK, int WK, int kNWarps>
__device__ __forceinline__ void wrap_coords(int access_id, int warp_id,
                                             int& wrap_id, int& row_block) {
  using T = Traits<Element, BM, BK, WK, kNWarps>;
  const int issue = access_id * kNWarps + warp_id;
  wrap_id = issue % T::kNumWraps;
  row_block = issue / T::kNumWraps;
}

// kRowGuard: when true, rows >= valid_rows load with voffset=-1 so the HW drops
// the (out-of-bounds) gmem access — used for the M-tail (Q) / N-tail (KV) block.
// Contiguous dir (K = head_dim) is always full, so dwordx4 stays valid; only the
// row selection is masked. Default false → bit-identical to the aligned path.
template <typename Element, int BM, int BK, int WK, int kNWarps,
          bool kRowGuard = false,
          EnableIfSupported<Element, BM, BK, WK, kNWarps> = 0>
inline __device__ void prefetch_panel_to_lds(
    Element* lds_base,
    Element* gmem_src,
    int stride_element,
    int k_panel,
    int k_col_offset = 0,
    int valid_rows = 0) {
  using T = Traits<Element, BM, BK, WK, kNWarps>;

  const int warp_id = threadIdx.x / WARP_SIZE_GPU;
  const int lane_id = threadIdx.x % WARP_SIZE_GPU;

  struct PtrWrapper {
    uint32_t former;
    uint32_t latter;
  };
  PtrWrapper glob_ptr;
  *reinterpret_cast<uint64_t*>(&glob_ptr) =
      reinterpret_cast<uint64_t>(gmem_src);

  uint32x4_t global_addr = {0};
  global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
  global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
  global_addr[2] = 0x80000000u;
  global_addr[3] = 0x00020000u;
  const int offset_s = 0;
  const int k_col_base = k_col_offset + k_panel * WK;

  #pragma unroll
  for (int access_id = 0; access_id < T::kAccessCnt; ++access_id) {
    int wrap_id = 0, row_block = 0;
    wrap_coords<Element, BM, BK, WK, kNWarps>(access_id, warp_id, wrap_id,
                                               row_block);
    const int wrap_amount = wrap_id * 2;
    const int issue = access_id * kNWarps + warp_id;

    const int row =
        row_block * T::kMBlock + wrap_id + (lane_id / 8) * 4;
    const int col = (lane_id % 8) * T::kElemsPerAccess + k_col_base;
    int offset_v =
        (row * stride_element + col) * T::kElemBytes;
    if constexpr (kRowGuard) {
      // OOB row → -1: buffer resource bounds-check drops the load (returns 0),
      // same mechanism as weight_voff_lane. Output for these rows is dropped.
      if (row >= valid_rows) offset_v = -1;
    }

    const int lds_elem = issue * T::kElemsPerWarpAccess;
    const int lds_addr =
        static_cast<int>(reinterpret_cast<uintptr_t>(lds_base)) +
        lds_elem * T::kElemBytes;
    const int target_addr =
        __builtin_amdgcn_readfirstlane(lds_addr + (wrap_amount << 16));

    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, %3, offen offset:0, lds \n" ::
            "v"(offset_v),
        "s"(target_addr), "s"(global_addr), "s"(offset_s)
        :);
  }
}

template <typename Element, int BM, int BK, int WK, int kNWarps,
          EnableIfSupported<Element, BM, BK, WK, kNWarps> = 0>
__device__ __forceinline__ int lds_elem_index_from_mk(int m, int k) {
  using T = Traits<Element, BM, BK, WK, kNWarps>;
  const int m_block = m / T::kMBlock;
  const int wrap = m % T::kNumWraps;
  const int slot = (m % T::kMBlock) / 4;
  const int wrap_amount = wrap * 2;
  const int L = slot * 8 + (k / 8);
  const int phys = (L + wrap_amount) % WARP_SIZE_GPU;
  const int aid = m_block * T::kNumWraps + wrap;
  return aid * T::kElemsPerWarpAccess + phys * 8 + (k % 8);
}

// MMAC lane map for one 16×32 tile origin (m_base, k_base).
__device__ __forceinline__ void mmac_lane_mk(int m_base, int k_base, int lane_id,
                                             int& m, int& k) {
  m = m_base + (lane_id % 16);
  k = k_base + (lane_id / 16) * 8;
}

template <typename Element, int BM, int BK, int WK, int kNWarps,
          EnableIfSupported<Element, BM, BK, WK, kNWarps> = 0>
__device__ __forceinline__ int lds_elem_index_mmac_tile(int m_base, int k_base,
                                                        int lane_id) {
  int m = 0, k = 0;
  mmac_lane_mk(m_base, k_base, lane_id, m, k);
  return lds_elem_index_from_mk<Element, BM, BK, WK, kNWarps>(m, k);
}

template <typename Element>
__device__ __forceinline__ uint32x4_t ds_read_b128_at(Element* lds_base,
                                                      int elem_index) {
  return *reinterpret_cast<const uint32x4_t*>(lds_base + elem_index);
}

// Load one MMAC 16×32 tile from even-wrap LDS for warp `warp_id`.
// m_base % 16 == 0, k_base % 32 == 0, k_base in [0, WK). All lanes of that
// warp must call with the same (warp_id, m_base, k_base). Returns b128:
// [0:1]=K-lo mmac, [2:3]=K-hi mmac.
template <typename Element, int BM, int BK, int WK, int kNWarps,
          EnableIfSupported<Element, BM, BK, WK, kNWarps> = 0>
__device__ __forceinline__ uint32x4_t ds_read_mmac_tile(Element* lds_base,
                                                         int warp_id,
                                                         int m_base,
                                                         int k_base) {
  const int lane_id = threadIdx.x - warp_id * WARP_SIZE_GPU;
  const int idx =
      lds_elem_index_mmac_tile<Element, BM, BK, WK, kNWarps>(m_base, k_base,
                                                              lane_id);
  return ds_read_b128_at(lds_base, idx);
}

}  // namespace lightop::fp4_mqa::detail::even_wrap
