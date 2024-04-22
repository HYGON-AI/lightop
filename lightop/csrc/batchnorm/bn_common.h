#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {
    #if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

constexpr unsigned MAX_GRID_SIZE = 65535u;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

constexpr int MAX_BLOCK_SIZE_1024 = 1024;
static int getNumThreads_1024(int nElem) {

  int threadSizes[6] = { 16, 32, 64, 128, 256, MAX_BLOCK_SIZE_1024 };

  for (int i = 0; i != 6; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE_1024;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, C10_WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffling reduction.
// First each warp (of C10_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most C10_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than C10_WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[C10_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % C10_WARP_SIZE == 0) {
    shared[tid / C10_WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / C10_WARP_SIZE && tid < C10_WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / C10_WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

struct InvStd {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> get_packed_accessor(
    const Tensor& t, c10::string_view var_name) {
  constexpr auto expect_type = c10::CppTypeToScalarType<scalar_t>::value;
  const auto actual_type = t.scalar_type();
  TORCH_CHECK(actual_type == expect_type, "Expected ", var_name,
              " to have type ", expect_type, " but got ", actual_type);
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(
    const Tensor& t, c10::string_view var_name) {
  if (!t.defined()) {
    const std::array<index_t, dim> zeros{{0}};
    return GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return get_packed_accessor<scalar_t, dim, PtrTraits, index_t>(t, var_name);
}

} } // namespace at::native