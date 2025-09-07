#pragma once
#include "marlin.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_vector_types.h>

namespace at {
namespace native {
namespace marlin_moe_w4a16{

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<__half> {
 public:
  using scalar_t = __half;
  using scalar_t2 = __half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<__half2, 2>;
  //
  using FragB = Vec<__half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<__half2, 1>;
  using FragZP = Vec<__half2, 4>;

  using I4 = Vec<int, 4>;

  static __device__ float inline num2float(const __half x) {
    return __half2float(x);
  }

  static __device__ __half2 inline num2num2(const __half x) {
    return __half2half2(x);
  }

  static __device__ __half2 inline num2num2g(const __half x) {
    return __half2half2(x);
  }



  static __host__ __device__ __half inline float2num(const float x) {
    return __float2half(x);
  }

};

class uint16x2 {
  public:
  uint16_t x;
  uint16_t y;
  __host__ __device__ uint16x2() = default;
  __host__ __device__ uint16x2(uint16_t x_, uint16_t y_) : x(x_), y(y_) {}
};


// Add bf16 support using uint16_t
template <>
class ScalarType<uint16_t> {
 public:
  using scalar_t = uint16_t;
  using scalar_t2 = uint16x2;  // Use __half2 for bf16 as well, but will be reinterpreted

  // Matrix fragments for tensor core instructions
  using FragA = Vec<__half2, 2>;
  using FragB = Vec<uint16_t, 4>;  // Store actual bf16 values as uint16_t
  using FragC = Vec<float, 4>;
  // Note: FragS and FragZP are kept as fp16 to reuse existing dequantization logic
  using FragS = Vec<__half2, 1>;
  using FragZP = Vec<__half2, 4>;

  using I4 = Vec<int, 4>;

  static __device__ float inline num2float(const uint16_t x) {
    // Convert bf16 to float
    uint32_t u = static_cast<uint32_t>(x) << 16;
    return *(float*)(&u);
  }

  static __device__ __half2 inline num2num2(const uint16_t x) {
    // Convert bf16 to __half2 (reuse the same value)
    __half h = __ushort_as_half(x);
    return __half2half2(h);
  }
  static __device__ uint16x2 inline num2num2g(const uint16_t x) {
    uint16x2 res = {x, x};
    return res;
  }  

  static __device__ __half2 inline num2num2(const __half x) {
    // Convert fp16 to __half2 (for scale and zero point)
    return __half2half2(x);
  }

  static __host__ __device__ uint16_t inline float2num_fast(const float x) {
    return reinterpret_cast<const uint16_t*>(&x)[1];
  }


  static __host__ __device__ uint16_t inline float2num(const float x) {
    // Convert float to bf16
    uint32_t& u = *(uint32_t*)(&x);
    
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
  }


};

// Union 用于在 uint32_t 和两个 uint16_t bf16 值之间进行类型转换
union BF16Union {
  uint32_t as_uint32;
  struct {
    uint16_t x;  // 第一个bf16元素
    uint16_t y;  // 第二个bf16元素
  };
};

union FragUnion {
    typename ScalarType<__half>::FragB frag_fp16;
    typename ScalarType<uint16_t>::FragB frag_bf16;
};


// 通用atomicAdd函数，支持fp16和bf16
template<typename scalar_t2>
__device__ inline scalar_t2 atomic_add_generic(scalar_t2* p_dst, const scalar_t2& x) {

  if constexpr (std::is_same<scalar_t2, __half2>::value) {

    return atomicAdd(p_dst, x);
    // if(threadIdx.x & 63 == 0) 
  } else {
    // bf16版本，使用atomicCAS
    // 对于bf16，scalar_t2实际上是__half2，但存储的是bf16数据
    uint32_t* dword_addr = reinterpret_cast<uint32_t*>(p_dst);
    
    BF16Union cur_val, old_val, new_val, x_val;
    cur_val.as_uint32 = *dword_addr;
    x_val.as_uint32 = reinterpret_cast<const uint32_t&>(x);
    do {
      old_val = cur_val;
      
      // 转换为float进行计算
      float f1 = ScalarType<uint16_t>::num2float(cur_val.x);
      float f2 = ScalarType<uint16_t>::num2float(cur_val.y);
      float fx1 = ScalarType<uint16_t>::num2float(x_val.x);
      float fx2 = ScalarType<uint16_t>::num2float(x_val.y);
      
      // 执行加法并转换回bf16
      new_val.x = ScalarType<uint16_t>::float2num(f1 + fx1);
      new_val.y = ScalarType<uint16_t>::float2num(f2 + fx2);
      
      // 原子比较并交换
      cur_val.as_uint32 = atomicCAS(dword_addr, old_val.as_uint32, new_val.as_uint32);
    } while (cur_val.as_uint32 != old_val.as_uint32);
    
    return x;
  }
}

// 通用mul2函数，支持fp16和bf16
template<typename scalar_t2>
__device__ inline scalar_t2 mul2(const scalar_t2& a, const scalar_t2& b) {
  if constexpr (std::is_same<scalar_t2, __half2>::value) {
    // fp16版本，直接使用__hmul2
    return __hmul2(a, b);
  } else {
    // bf16版本，减少寄存器用量，避免中间变量


    uint16x2 res;
    res.x = ScalarType<uint16_t>::float2num(
      ScalarType<uint16_t>::num2float(a.x) * ScalarType<uint16_t>::num2float(b.x)
    );
    res.y = ScalarType<uint16_t>::float2num(
      ScalarType<uint16_t>::num2float(a.y) * ScalarType<uint16_t>::num2float(b.y)
    );
    return res;
  }
}


} // namespace marlin_moe_w4a16
}  
}