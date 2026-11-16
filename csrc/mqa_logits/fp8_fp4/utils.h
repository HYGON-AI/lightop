// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Device helpers retained from the validated even-wrap MQA logits prototype.
#pragma once

#include "lds_even_wrap.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace lightop::fp4_mqa::detail {

using bf16_t = __hip_bfloat16;
using floatx4 = float __attribute__((ext_vector_type(4)));
using floatx2 = float __attribute__((ext_vector_type(2)));
using uint32x4_t = even_wrap::uint32x4_t;

__device__ __forceinline__ uint32x4_t make_buffer_rsrc(const void* gmem_base) {
  struct PtrWrapper {
    uint32_t former;
    uint32_t latter;
  };
  PtrWrapper glob_ptr;
  *reinterpret_cast<uint64_t*>(&glob_ptr) =
      reinterpret_cast<uint64_t>(gmem_base);

  uint32x4_t rsrc = {0};
  rsrc[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
  rsrc[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
  rsrc[2] = 0x80000000u;
  rsrc[3] = 0x00020000u;
  return rsrc;
}

// Async MUBUF dwordx4 -> VGPR. Ready only after a later s_waitcnt vmcnt.
__device__ __forceinline__ void buffer_load_dwordx4_vgpr(uint32x4_t& dst,
                                                         uint32x4_t rsrc,
                                                         int v_offset_bytes) {
  const int offset_s = 0;
  asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen\n"
               : "=v"(dst)
               : "v"(v_offset_bytes), "s"(rsrc), "s"(offset_s)
               : "memory");
}

__device__ __forceinline__ void wait_vmcnt0_dep(float& dep) {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n\t" : "+v"(dep) : : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

// Async MUBUF dword -> VGPR (Weights). Ready after a later s_waitcnt.
__device__ __forceinline__ void buffer_load_dword_vgpr(float& dst,
                                                      uint32x4_t rsrc,
                                                      int v_offset_bytes) {
  const int offset_s = 0;
  asm volatile("buffer_load_dword %0, %1, %2, %3 offen\n"
               : "=v"(dst)
               : "v"(v_offset_bytes), "s"(rsrc), "s"(offset_s)
               : "memory");
}

template <int BM, bool kMGuard = false>
__device__ __forceinline__ int weight_voff_lane(int lane_id, int num_heads,
                                                 int h, int valid_m = BM) {
  // Only lanes [0, BM) touch gmem; upper lanes use -1 so HW drops the load.
  // kMGuard (M-tail): rows [valid_m, BM) are also OOB → -1 (avoids reading past
  // the Weights allocation).
  const int limit = kMGuard ? (valid_m < BM ? valid_m : BM) : BM;
  // [M,H]: base is (row0, head 0); neighboring lanes advance by one row.
  const int idx = lane_id * num_heads + h;
  return (lane_id < limit) ? idx * static_cast<int>(sizeof(float)) : -1;
}

// One dword / wave: lane L < BM loads W[L]; lane >= BM issues with voffset=-1.
template <int BM, bool kMGuard = false>
__device__ __forceinline__ void issue_w_head_vgpr(float& w,
                                                  uint32x4_t w_rsrc,
                                                  int num_heads,
                                                  int head,
                                                  int lane_id,
                                                  int valid_m = BM) {
  buffer_load_dword_vgpr(
      w, w_rsrc,
      weight_voff_lane<BM, kMGuard>(lane_id, num_heads, head, valid_m));
}

// Loop-invariant part of unpack_w_shfl_xor16: the two ds_bpermute source-lane
// byte-addresses depend only on lane_id, so precompute once and hoist out of
// the per-head loop. `__shfl`/`__shfl_xor` recompute the lane address (mbcnt +
// and/or/xor/cmp/cndmask, ~13 VALU) on every call; the explicit-bpermute path
// below keeps only the two gathers + the data select in the hot loop.
struct WShflXor16Plan {
  int addr_base;  // ds_bpermute byte addr: pull w from lane (lane%32)
  int addr_peer;  // ds_bpermute byte addr: xor-16 peer of the shuffled value
  bool lo;        // (lane%32) < 16 → row lives in the low half
};

__device__ __forceinline__ WShflXor16Plan precompute_w_shfl_xor16(int lane_id) {
  constexpr int BM = 32;
  const int src = lane_id % BM;           // 0..31
  WShflXor16Plan p;
  p.addr_base = src << 2;                 // ds_bpermute addr = laneIndex * 4
  p.addr_peer = (lane_id ^ 16) << 2;      // xor-16 on the wave lane
  p.lo = src < 16;
  return p;
}

__device__ __forceinline__ void unpack_w_shfl_xor16_pre(float w_loaded,
                                                        const WShflXor16Plan& p,
                                                        float& w0, float& w1) {
  const float w_base = __int_as_float(
      __builtin_amdgcn_ds_bpermute(p.addr_base, __float_as_int(w_loaded)));
  const float w_peer = __int_as_float(
      __builtin_amdgcn_ds_bpermute(p.addr_peer, __float_as_int(w_base)));
  w0 = p.lo ? w_base : w_peer;
  w1 = p.lo ? w_peer : w_base;
}

// Issue one K-panel (WK) of head into q_stage[panel]. Per-wave access count =
// Traits::kAccessCnt (BM128/4w → 4). Callers that split waitcnt by panel MUST
// issue panels in order p=0 then p=1 so FIFO drain matches panel readiness.
template <typename scalar_t, int BM, int BK, int WK, int kNWarps,
          bool kRowGuard = false>
__device__ __forceinline__ void issue_q_panel(scalar_t q_stage[][BM * WK],
                                              scalar_t* q_tile,
                                              int stride_q,
                                              int head,
                                              int panel,
                                              int valid_rows = 0) {
  even_wrap::prefetch_panel_to_lds<scalar_t, BM, BK, WK,
                                                   kNWarps, kRowGuard>(
      q_stage[panel], q_tile, stride_q, panel, /*k_col_offset=*/head * BK,
      valid_rows);
}

template <typename scalar_t, int BM, int BK, int WK, int kNWarps,
          bool kRowGuard = false>
__device__ __forceinline__ void issue_q_head(scalar_t q_stage[][BM * WK],
                                             scalar_t* q_tile,
                                             int stride_q,
                                             int head,
                                             int valid_rows = 0) {
  constexpr int kPanels = BK / WK;
  #pragma unroll
  for (int p = 0; p < kPanels; ++p) {
    issue_q_panel<scalar_t, BM, BK, WK, kNWarps, kRowGuard>(
        q_stage, q_tile, stride_q, head, p, valid_rows);
  }
}

// MFMA C layout -> 4 contiguous cols (same as mika pack_c_contig4).
__device__ __forceinline__ floatx4 pack_c_contig4(floatx4 c, int lane_id) {
  const int a = (lane_id >> 4) & 3;
  const float c0 = c[0];
  const float c1 = c[1];
  const float c2 = c[2];
  const float c3 = c[3];

  const float send16 = (a == 0) ? c1 : (a == 1) ? c0 : (a == 2) ? c3 : c2;
  const float send32 = (a == 0) ? c2 : (a == 1) ? c3 : (a == 2) ? c0 : c1;
  const float send48 = (a == 0) ? c3 : (a == 1) ? c2 : (a == 2) ? c1 : c0;

  const float recv16 = __shfl_xor(send16, 16);
  const float recv32 = __shfl_xor(send32, 32);
  const float recv48 = __shfl_xor(send48, 48);

  float tmp0, tmp1, tmp2, tmp3;
  if (a == 0) {
    tmp0 = c0;
    tmp1 = recv16;
    tmp2 = recv32;
    tmp3 = recv48;
  } else if (a == 1) {
    tmp0 = recv16;
    tmp1 = c1;
    tmp2 = recv48;
    tmp3 = recv32;
  } else if (a == 2) {
    tmp0 = recv32;
    tmp1 = recv48;
    tmp2 = c2;
    tmp3 = recv16;
  } else {
    tmp0 = recv48;
    tmp1 = recv32;
    tmp2 = recv16;
    tmp3 = c3;
  }
  return floatx4{tmp0, tmp1, tmp2, tmp3};
}

// Force single-instruction ReLU: v_max with 0.
__device__ __forceinline__ float relu0_f32(float x) {
  float y;
  asm volatile("v_max_f32_e32 %0, 0, %1" : "=v"(y) : "v"(x));
  return y;
}
}  // namespace lightop::fp4_mqa::detail
