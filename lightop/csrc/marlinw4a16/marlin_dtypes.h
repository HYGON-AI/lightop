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

  static __device__ __half2 inline nums2num2(const __half x1, const __half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ __half inline float2num(const float x) {
    return __float2half(x);
  }

};

}  // namespace marlin
}
}