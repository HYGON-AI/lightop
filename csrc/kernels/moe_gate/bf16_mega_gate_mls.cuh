// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(__gfx938__)
namespace mega_gate_mls {
using u4 = unsigned __attribute__((ext_vector_type(4)));
using i4 = int __attribute__((ext_vector_type(4)));

// Layout and B16 encodings follow deepgemm/csrc/include/intrinsic.h.
// Keep this small wrapper local rather than introducing a DeepGEMM dependency.
// The 32x16 B16 instruction with T=1 loads 16 rows of 32 elements.
// Stride is in elements. Bit 31 selects the matching transposed DS layout.
__device__ __forceinline__ void load(const BFloat16* src, short* dst, int stride) {
  const auto address = reinterpret_cast<uintptr_t>(src);
  const u4 desc = {
      unsigned(__builtin_amdgcn_readfirstlane(unsigned(address))),
      unsigned(__builtin_amdgcn_readfirstlane(unsigned(address >> 32) & 0xffffu)),
      unsigned(stride), 0};
  const unsigned offset = __builtin_amdgcn_readfirstlane(
      unsigned(reinterpret_cast<uintptr_t>(dst))) | 0x80000000u;
  // DTK 26.04 assigns a VGPR to the builtin's wave-private LDS address.
  // Explicit SGPR constraints also make the five-cycle descriptor hazard clear.
  asm volatile("s_nop 4\n\tmatrix_load_32x16_b16 %0, %1, moffset:0 t lds"
               :: "s"(desc), "s"(offset) : "memory");
}
__device__ __forceinline__ void load_w(const BFloat16* src, short* dst, int stride) {
  const auto address = reinterpret_cast<uintptr_t>(src);
  const u4 desc = {
      unsigned(__builtin_amdgcn_readfirstlane(unsigned(address))),
      unsigned(__builtin_amdgcn_readfirstlane(unsigned(address >> 32) & 0xffffu)),
      unsigned(stride), 0};
  const unsigned offset = __builtin_amdgcn_readfirstlane(
      unsigned(reinterpret_cast<uintptr_t>(dst))) | 0x80000000u;
  // DTK 26.04 assigns a VGPR to the builtin's wave-private LDS address.
  // Explicit SGPR constraints also make the five-cycle descriptor hazard clear.
  asm volatile("s_nop 4\n\tmatrix_load_32x32_b16 %0, %1, moffset:0 t lds"
               :: "s"(desc), "s"(offset) : "memory");
}
__device__ __forceinline__ i4 read(short* src) {
  const unsigned offset = __builtin_amdgcn_readfirstlane(
      unsigned(reinterpret_cast<uintptr_t>(src)));
  i4 result;
  asm volatile("s_add_u32 m0, %1, 0x80000000\n\ts_nop 0\n\t"
               "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0"
               : "=v"(result) : "s"(offset) : "m0", "memory");
  return result;
}
// DS produces K=(lane/16)*4+[0:4,16:20]. Preserve the original
// K=(lane/16)*8+[0:8] fragments, including their FP32 accumulation order.
__device__ __forceinline__ half4x2 fragment(i4 v, int lane) {
  const int group = lane / 16;
  const int src = lane % 16 + (group % 2) * 32;
  i4 lo, hi;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo[i] = __shfl(v[i], src, 64);
    hi[i] = __shfl(v[i], src + 16, 64);
  }
  const int h = group < 2 ? 0 : 2;
  i4 result = {lo[h], lo[h + 1], hi[h], hi[h + 1]};
  return __builtin_bit_cast(half4x2, result);
}
// MLS transpose layout rotates each row's four 16-byte groups by row/2.
// This address mapping preserves the original contiguous-eight BF16 fragment
// without cross-lane shuffles for W. Validated for all 16 rows and 64 lanes.
__device__ __forceinline__ i4 read_direct(short* src) {
  const int lane = threadIdx.x % 64, row = lane % 16;
  const int group = (lane / 16 + row / 2) % 4;
  const unsigned offset = unsigned(reinterpret_cast<uintptr_t>(
      src + row * 32 + group * 8));
  i4 result;
  // DTK's ds_read_v4i builtin adds global cache fences around each read.
  // This is wave-private LDS: wait_load()/wait_read() delimit its lifetime.
  asm volatile("ds_read_b128 %0, %1"
               : "=v"(result) : "v"(offset) : "memory");
  return result;
}
__device__ __forceinline__ half4x2 fragment_direct(i4 v, int) {
  return __builtin_bit_cast(half4x2, v);
}
__device__ __forceinline__ void wait_load() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ void wait_read() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
} // namespace mega_gate_mls
#endif
