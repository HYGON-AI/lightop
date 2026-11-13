// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <iostream>

namespace at {
namespace native {
namespace marlin_moe_w4fp8{
// Marlin params
#define S_BARRIER asm volatile("s_barrier")
// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 1;  // 4 pipeline stages fit into shared memory

// static constexpr int min_thread_n = 64;
static constexpr int min_thread_n = 128;
static constexpr int min_thread_k = 64; 
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 8;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

// 存储加载到 VGPR 中 打包好的权重矩阵
using UINT32X4 = Vec<uint32_t, 4>;
using intx4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

using V2i = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using V4f   = __attribute__((__vector_size__(4 * sizeof(float)))) float;

using FragA = Vec<uint32_t, 2>; // 计算时 参与一次mmac运算的 A矩阵 向量数据类型  fp8 是 int 类型
using FragB = Vec<uint32_t, 2>; // 计算时 参与一次mmac运算的 B矩阵 向量数据类型  fp8 是 int 类型
using FragC = Vec<float, 4>;   // fp8 mmac的结果 4 个 float32
using bhalf_t = __hip_bfloat16;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

static __device__ inline uint16_t f32_to_bf16(float f){
    uint32_t u = *(uint32_t*)(&f);
    u += 0x7fff + ((u >> 16) & 1); 
    return u>>16;
}

#ifdef __gfx938__

__device__ inline void mmac_16_16_32_fp8_fp8(
                                    const FragA& frag_a,
                                    const FragB& frag_b,
                                    FragC& frag_c
                                  ) {
  *(reinterpret_cast<V4f*>(&frag_c)) = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(*(reinterpret_cast<const V2i*>(&frag_a)), *(reinterpret_cast<const V2i*>(&frag_b)), *(reinterpret_cast<V4f*>(&frag_c)), false, false);
}

#endif

// 仅在 BW gfx 936 上才编译实际算子
#if defined (__gfx936__) || (__gfx938__)

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      state = __atomic_load_n(lock, __ATOMIC_ACQUIRE);
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    atomicAdd(lock, val);
  }
}

__device__ __forceinline__ int cvta_to_shared(const void* ptr) {
    return (reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}

// 按照pred 的值进行 128bit的 load
__device__ __forceinline__ void cp_async4_pred(void* smem_ptr, const int4* glob_ptr, int offset_v, bool pred = true) {
  if(pred){
    int smem_offset = __builtin_amdgcn_readfirstlane(cvta_to_shared(smem_ptr));
    typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
    uint32x4_t global_addr = {0};
    *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
    global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset), "s"(global_addr)
    :);
  }
}

// 进行 128bit的 load
__device__ __forceinline__ void cp_async4(void* smem_ptr, const int4* glob_ptr, int offset_v) {
  int smem_offset = __builtin_amdgcn_readfirstlane(cvta_to_shared(smem_ptr));
  typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
  uint32x4_t global_addr = {0};
  *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
  global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
  global_addr[2] = 0x80000000;
  global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
  //  000
  asm volatile(
      "s_mov_b32 m0, %1 \n\t"
      "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
      "s"(smem_offset), "s"(global_addr)
  :);
}

// 步长设置为 128bit从而避免 * 16 从global load 128bit 到 VGPR 中
__device__ __forceinline__ void cp_async4_vgpr_s128(UINT32X4* dst_vgpr, const int4* glob_ptr, int offset_v) {

  intx4_t res;
  typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
  uint32x4_t global_addr = {0};
  *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
  global_addr[1] += 0x00100000; // 128bit的步长，并且启用idxen的地址计算模式
  global_addr[2] = 0x80000000;
  global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
  //  000
  asm volatile(
    "buffer_load_dwordx4 %0, %1, %2 ,0 idxen\n" 
    " \n\t" :"=v"(res),
    "+v"(offset_v), "+s"(global_addr)
  );

  // res = __builtin_amdgcn_buffer_load_dwordx4(global_addr, 0, offset_v, false, false);

  *reinterpret_cast<intx4_t*>(dst_vgpr) = res;
}


__device__ inline void cp_async_wait_0() {
  // asm volatile("s_waitcnt vmcnt(0);");
  asm volatile(
    "s_waitcnt vmcnt(0);\n"
    "s_barrier;\n"
  );
}

#endif

}  // namespace marlin
}
}
