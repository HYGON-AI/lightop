#pragma once

#include "hip/hip_runtime.h"
#include "hipcub/hipcub.hpp"
#include "hiprand.h"
#include "hiprand_kernel.h"

namespace lightop {
namespace sampling {

using namespace hipcub;

using MaxReduceOp = hipcub::Max;
using MinReduceOp = hipcub::Min;

#define DCU_WARP_SIZE 64

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

#define __INLINE__ inline __attribute__((always_inline)) __device__

template <typename float_t, size_t vec_size>
struct vec_t {
  __INLINE__ float_t& operator[](size_t i);
  __INLINE__ const float_t& operator[](size_t i) const;
  __INLINE__ void fill(float_t val);
  __INLINE__ void load(const float_t* ptr);
  __INLINE__ void store(float_t* ptr) const;
  __INLINE__ void load_global_acquire(float* addr);
  __INLINE__ void store_global_release(float* addr) const;
  __INLINE__ void load_global_volatile(float* addr);
  __INLINE__ void store_global_volatile(float* addr) const;
  template <typename T>
  __INLINE__ void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  __INLINE__ void cast_load(const T* ptr);
  template <typename T>
  __INLINE__ void cast_store(T* ptr) const;
};

template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  __INLINE__ static void cast(dst_t* dst, const src_t* src) {
    #pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
__INLINE__ void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                               const vec_t<src_float_t, vec_size>& src) {
  vec_cast<tgt_float_t, src_float_t>::cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
__INLINE__ void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                               const src_float_t* src_ptr) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
__INLINE__ void cast_store_impl(tgt_float_t* dst_ptr,
                                const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr) {
  *addr = val;
}

__device__ __forceinline__ int4 ld_global_acquire(int4* addr) {
  int4 val = *addr;
  return val;
}

__device__ __forceinline__ void st_global_volatile(int4 const& val, int4* addr) {
  *addr = val;
}

__device__ __forceinline__ int4 ld_global_volatile(int4* addr) {
  int4 val = *addr;
  return val;
}

template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];

  __INLINE__ float& operator[](size_t i) { return ((float*)(data))[i]; }
  __INLINE__ const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  __INLINE__ float* ptr() { return reinterpret_cast<float*>(&data); }
  __INLINE__ void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  __INLINE__ void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  __INLINE__ void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  __INLINE__ void store_global_release(float* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      st_global_release(*(int4*)(data + i), (int4*)(addr + i * 4));
    }
  }
  __INLINE__ void load_global_acquire(float* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      *((int4*)(data + i)) = ld_global_acquire((int4*)(addr + i * 4));
    }
  }
  __INLINE__ void store_global_volatile(float* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      st_global_volatile(*(int4*)(data + i), (int4*)(addr + i * 4));
    }
  }
  __INLINE__ void load_global_volatile(float* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      *((int4*)(data + i)) = ld_global_volatile((int4*)(addr + i * 4));
    }
  }
  template <typename T>
  __INLINE__ void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  __INLINE__ void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <typename T>
struct ValueCount {
  T value;
  int count;

  __device__ ValueCount operator+(const ValueCount& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ ValueCount& operator+=(const ValueCount& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    float deterministic_scan[BLOCK_THREADS / DCU_WARP_SIZE];
    typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    int32_t last_valid_id;
    float max_val;
    union {
      float value;
      ValueCount<float> pair;
    } block_aggregate;
  };
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan algorithm.
 * \note This implementation is slower than the cub::BlockScan, but it is deterministic.
 */
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float* in_data, float* out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  float* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
  float thread_data[VEC_SIZE];
  float thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < DCU_WARP_SIZE; offset *= 2) {
    float tmp = __shfl_up_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  float warp_sum = __shfl_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffffffffffff);
  if (threadIdx.x % DCU_WARP_SIZE == DCU_WARP_SIZE - 1) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = DCU_WARP_SIZE / 2; offset >= 1; offset /= 2) {
    float tmp = __shfl_xor_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / DCU_WARP_SIZE] = warp_sum;
  __syncthreads();

  if (threadIdx.x < DCU_WARP_SIZE) {
    float warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / DCU_WARP_SIZE) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < DCU_WARP_SIZE; offset *= 2) {
      float tmp = __shfl_up_sync(0xffffffffffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % DCU_WARP_SIZE == DCU_WARP_SIZE - 1) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = DCU_WARP_SIZE / 2; offset >= 1; offset /= 2) {
      float tmp = __shfl_xor_sync(0xffffffffffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / DCU_WARP_SIZE) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / DCU_WARP_SIZE] + thread_exclusive_prefix_sum + thread_data[i];
  }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  __syncthreads(); // added
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }

  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, MaxReduceOp{});
  __syncthreads(); // added
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType* probs, IdType* output, IdType* indices,
                                           IdType* top_k_arr, uint32_t top_k_val, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  int round = 0;
  do {
    round += 1;
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = hiprand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_gt_pivot_0{0, 0}, threadlocal_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        threadlocal_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                temp_storage.block_prim.reduce_value_count)
                                .Sum(threadlocal_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                temp_storage.block_prim.reduce_value_count)
                                .Sum(threadlocal_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    if (aggregate_gt_pivot_0.count < k) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, IdType* output, IdType* indices,
                                           float* top_p_arr, float top_p_val, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  float top_p = (top_p_arr == nullptr) ? top_p_val : top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = hiprand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }

    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                .Sum(threadlocal_aggregate_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

    aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                .Sum(threadlocal_aggregate_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;

    if (aggregate_gt_pivot_0 < top_p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1 < top_p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = hiprand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_0{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

}
}
