#pragma once

#include <torch/all.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
namespace at {
namespace native {
namespace marlin_moe_w4a16{

// Marlin params

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

using I4 = Vec<int, 4>;
using intx4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

__device__ inline int device_min(int a, int b) {
    return a < b ? a : b;
}

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

/*
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(const T* ptr) {
    // First get the address as a size_t to handle all pointer sizes
    size_t addr = reinterpret_cast<size_t>(ptr);

    // Extract the lower 32 bits which represent the shared memory offset
    // This is safe because shared memory addresses are always within 32-bit range
    return static_cast<uint32_t>(addr & 0xFFFFFFFF);
}
*/
__device__ __forceinline__ int cvta_to_shared(const void* ptr) {
    return (reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}

// 利用 buffer_load_dwordx4 的边界判断去掉分支
__device__ __forceinline__ void cp_async4_pred_a(void* smem_ptr, int m, int k, const int4* glob_ptr, int offset_v) {
  int smem_offset = __builtin_amdgcn_readfirstlane(cvta_to_shared(smem_ptr));
  typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
  uint32x4_t global_addr = {0};
  *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
  global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
  
  const uint32_t mk_value = m * k * 2;
  
  global_addr[2] = mk_value;
  global_addr[3] = 0x00020000;
#ifdef __gfx936__
  asm volatile(
      "s_mov_b32 m0, %1 \n\t"
      "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
      "s"(smem_offset), "s"(global_addr)
  :);
#endif
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
    // #if defined(__gfx936__)
    //   asm volatile(
    //       "s_mov_b32 m0, %1 \n\t"
    //       "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
    //       "s"(smem_offset), "s"(global_addr)
    //   :);
    // #endif
#ifdef __gfx936__
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset), "s"(global_addr)
    :);
#endif
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
  // #if defined(__gfx936__) || defined(__gfx928__)
  //   asm volatile(
  //       "s_mov_b32 m0, %1 \n\t"
  //       "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
  //       "s"(smem_offset), "s"(global_addr)
  //   :);
  // #endif
#ifdef __gfx936__
  asm volatile(
      "s_mov_b32 m0, %1 \n\t"
      "buffer_load_dwordx4 %0, %2, 0 ,offen  offset:0, lds \n" ::"v"(offset_v),
      "s"(smem_offset), "s"(global_addr)
  :);
#endif
}

// 从global load 128bit 到 VGPR 中
__device__ __forceinline__ void cp_async4_vgpr(I4* dst_vgpr, const int4* glob_ptr, int offset_v) {

  intx4_t res;

  typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
  uint32x4_t global_addr = {0};
  *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
  global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
  global_addr[2] = 0x80000000;
  global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
  //  000
  // #if defined(__gfx936__) || defined(__gfx928__)
  //   asm volatile(
  //     "buffer_load_dwordx4 %0, %1, %2 ,0 offen\n" 
  //     " \n\t" :"=v"(res),
  //     "+v"(offset_v), "+s"(global_addr)
  //   );
  // #endif

  asm volatile(
    "buffer_load_dwordx4 %0, %1, %2 ,0 offen\n" 
    " \n\t" :"=v"(res),
    "+v"(offset_v), "+s"(global_addr)
  );
  
  // res = __builtin_amdgcn_buffer_load_dwordx4(global_addr, 0, offset_v, false, false);

  *reinterpret_cast<intx4_t*>(dst_vgpr) = res;
}

// 步长设置为 128bit从而避免 * 16 从global load 128bit 到 VGPR 中
__device__ __forceinline__ void cp_async4_vgpr_s128(I4* dst_vgpr, const int4* glob_ptr, int offset_v) {

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

  *reinterpret_cast<intx4_t*>(dst_vgpr) = res;
}

// // 设备函数：将 int32 转换为 half2
// __device__ void int32_to_half2_print(int packed_value) {
//   // Step 1: Extract the two 16-bit halves from the int32 value
//   unsigned short low_half_bits = static_cast<unsigned short>(packed_value & 0xFFFF); // Low 16 bits
//   unsigned short high_half_bits = static_cast<unsigned short>((packed_value >> 16) & 0xFFFF); // High 16 bits

//   // Step 2: Convert the 16-bit parts to __half using bit-casting
//   __half low_half = __ushort_as_half(low_half_bits);
//   __half high_half = __ushort_as_half(high_half_bits);
//   // 
//   printf(" %f %f ",__half2float(low_half), __half2float(high_half));
// }

// __device__ inline void cp_async_wait(int n) {
//   // __builtin_amdgcn_s_waitcnt(n);
//   // 直接嵌入汇编
//   asm volatile("s_waitcnt vmcnt(%0);" :: "n"(n));
// }

// template <int n>
// __device__ inline void cp_async_wait() {
//     // 使用"n"约束确保参数是编译时常量
//     asm volatile("s_waitcnt vmcnt(%0);" :: "n"(n));
// }

// // 预先计算所有可能的指令数, 以将n变为 编译时常量
// __device__ inline void cp_async_wait(int n) {
//     switch(n) {
//         case 0: asm volatile("s_waitcnt vmcnt(0);"); break;
//         case 1: asm volatile("s_waitcnt vmcnt(1);"); break;
//         case 2: asm volatile("s_waitcnt vmcnt(2);"); break;
//         case 3: asm volatile("s_waitcnt vmcnt(3);"); break;
//         case 4: asm volatile("s_waitcnt vmcnt(4);"); break;
//         case 5: asm volatile("s_waitcnt vmcnt(5);"); break;
//         case 6: asm volatile("s_waitcnt vmcnt(6);"); break;
//         case 7: asm volatile("s_waitcnt vmcnt(7);"); break;
//         case 8: asm volatile("s_waitcnt vmcnt(8);"); break;
//         case 9: asm volatile("s_waitcnt vmcnt(9);"); break;
//         case 10: asm volatile("s_waitcnt vmcnt(10);"); break;

//         default: 
//           asm volatile("s_waitcnt vmcnt(10);"); break;
//           break;
//     }
// }

// // 预先计算所有可能的指令数, 以将n变为 编译时常量
// __device__ inline void cp_async_wait(int n) {
//     switch(n) {
//         case 0: __builtin_amdgcn_s_waitcnt(0); break;
//         case 1: __builtin_amdgcn_s_waitcnt(1); break;
//         case 2: __builtin_amdgcn_s_waitcnt(2); break;
//         case 3: __builtin_amdgcn_s_waitcnt(3); break;
//         case 4: __builtin_amdgcn_s_waitcnt(4); break;
//         case 5: __builtin_amdgcn_s_waitcnt(5); break;
//         case 6: __builtin_amdgcn_s_waitcnt(6); break;
//         case 7: __builtin_amdgcn_s_waitcnt(7); break;
//         case 8: __builtin_amdgcn_s_waitcnt(8); break;
//         case 9: __builtin_amdgcn_s_waitcnt(9); break;
//         case 10: __builtin_amdgcn_s_waitcnt(10); break;

//         default: 
//           __builtin_amdgcn_s_waitcnt(20); break;
//           break;
//     }
// }

__device__ inline void cp_async_wait_0() {
  asm volatile("s_waitcnt vmcnt(0);");
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      // asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
      //             : "=r"(state)
      //             : "l"(lock));
      state = __atomic_load_n(lock, __ATOMIC_ACQUIRE);
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

}
}
}  // namespace marlin

