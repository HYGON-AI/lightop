// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>
#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif


#if defined(TORCH_VERSION_OVER_2120)
// torch>=2.12 removed C10_WARP_SIZE_STATIC, DCU wavefront size is fixed at 64
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {

namespace softmax_math {


template <typename T>
__host__ __device__ constexpr T max(T x)
{
    return x;
}

template <typename T>
__host__ __device__ constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T, typename T0>
__host__ __device__ constexpr T max(T x, T0 y)
{
    auto y0 = static_cast<T>(y);
    return x > y0 ? x : y0;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");

    return max(x, max(ys...));
}

template <typename T>
__host__ __device__ constexpr T min(T x)
{
    return x;
}

template <typename T>
__host__ __device__ constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");

    return min(x, min(ys...));
}

template <typename T>
__host__ __device__ constexpr T clamp(const T& x, const T& lowerbound, const T& upperbound)
{
    return min(max(x, lowerbound), upperbound);
}

// disallow implicit type casting
template <typename T>
__device__ T exp(T x);

// TODO: add f16 support using v_exp_f16

template <>
__device__ float exp<float>(float x)
{
    return __expf(x);
}

template <>
__device__ double exp<double>(double x)
{
    return exp(x);
}

static inline __host__ float exp(float x) { return std::expf(x); }

static inline __host__ double exp(double x) { return std::exp(x); }

// disallow implicit type casting
template <typename T>
__device__ T log(T x);

template <>
__device__ float log<float>(float x)
{
    return __logf(x);
}

template <>
__device__ double log<double>(double x)
{
    return log(x);
}

template<typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

template<typename T, typename acc_t>
struct AddFloat
{
  __device__ __forceinline__ acc_t operator()(acc_t sum, T v) const {
    return sum + v;
  }
};

template<typename T>
struct Max {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? b : a;
    }
};


}   // namespace softmax_math

namespace softmax_op {

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_batch_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename acc_t, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t& sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc_t b = WARP_SHFL_XOR(sum, offset, WARP_SIZE);
        sum = r(sum, b);
    }
}


template<typename input_t, typename acc_t>
struct OnlineSoftmaxUpdate
{
    __device__ __forceinline__ void operator()(const input_t v, acc_t &max_v, acc_t &sum_v) const {
        acc_t bigger = softmax_math::max(max_v, v);
        sum_v = sum_v * softmax_math::exp(max_v - bigger) + softmax_math::exp(v - bigger);
        max_v = bigger;
    }
};

template<typename input_t, typename acc_t>
struct OnlineSoftmaxCombineUpdate
{
    __device__ __forceinline__ void operator()(const acc_t &max_old, const acc_t &sum_old, acc_t &max_new, acc_t &sum_new) const {
        acc_t bigger = softmax_math::max(max_old, max_new);
        sum_new = sum_new * softmax_math::exp(max_new - bigger) + 
                  sum_old * softmax_math::exp(max_old - bigger);
        max_new = bigger;
    }
};

template<typename input_t, typename acc_t, typename output_t>
struct SoftMaxForwardEpilogue {
    __device__ __forceinline__ SoftMaxForwardEpilogue(acc_t max_v, acc_t sum_v)
        : max_v_(max_v), sum_v_(sum_v), logsum_(std::log(sum_v)) {}

    __device__ __forceinline__ output_t operator()(input_t input) const {
        return static_cast<output_t>(std::exp(input - max_v_ - logsum_));
    }

    const acc_t max_v_;
    const acc_t sum_v_;
    const acc_t logsum_;
};

template<typename input_t, typename acc_t, typename output_t>
struct LogSoftMaxForwardEpilogue {
    __device__ __forceinline__ LogSoftMaxForwardEpilogue(acc_t max_v, acc_t sum_v)
        : max_v_(max_v), logsum_(std::log(sum_v)) {}

    __device__ __forceinline__ output_t operator()(input_t input) const {
        return static_cast<output_t>(input - max_v_ - logsum_);
    }

    const acc_t max_v_;
    const acc_t logsum_;
};

template<typename input_t, typename acc_t, typename output_t>
struct SoftMaxBackwardEpilogue {
    __device__ __forceinline__ SoftMaxBackwardEpilogue(acc_t sum_v) : sum_v_(sum_v) {}

    // gradOutput that we get here is really gradOutput * output
    __device__ __forceinline__ input_t operator()(output_t gradOutput, output_t output) const {
        return static_cast<input_t>(gradOutput - output * sum_v_);
    }

    const acc_t sum_v_;
};

template<typename input_t, typename acc_t, typename output_t>
struct LogSoftMaxBackwardEpilogue {
    __device__ __forceinline__ LogSoftMaxBackwardEpilogue(acc_t sum_v) : sum_v_(sum_v) {}

    __device__ __forceinline__ input_t operator()(output_t gradOutput, output_t output) const {
        return static_cast<input_t>(gradOutput - std::exp(static_cast<acc_t>(output)) * sum_v_);
    }

    const acc_t sum_v_;
};

bool softmax_spatial_fwd_use_transpose(int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    const int outer_size_log2 = ::log2(outer_size);
    const int dim_size_log2 = ::log2(dim_size);
    const int inner_size_log2 = ::log2(inner_size);
    const int size_log2_sum = dim_size_log2 + inner_size_log2;
    const bool condition1 = inner_size_log2 <= 10 && outer_size_log2 <= 5 && dim_size_log2 <= 15 && size_log2_sum <= 22;
    const bool condition2 = inner_size_log2 > 6 && outer_size_log2 > 1 && dim_size_log2 > 6 && size_log2_sum > 12;
    bool condition3 = false;
    if (inner_size_log2 <= 7) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 6 && dim_size_log2 <= 15) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 7 && dim_size_log2 <= 14) condition3 = true;
        } else if (outer_size_log2 <= 4) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 13) condition3 = true;
        } else if (outer_size_log2 <= 5) {
            if (dim_size_log2 >= 9 && dim_size_log2 <= 12) condition3 = true;
        }
    } else if (inner_size_log2 <= 8) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 6 && dim_size_log2 <= 14) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 7 && dim_size_log2 <= 13) condition3 = true;
        } else if (outer_size_log2 <= 4) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 12) condition3 = true;
        }
    } else if (inner_size_log2 <= 9) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 6 && dim_size_log2 <= 13) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 7 && dim_size_log2 <= 12) condition3 = true;
        } else if (outer_size_log2 <= 4) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 11) condition3 = true;
        }
    } else if (inner_size_log2 <= 10) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 6 && dim_size_log2 <= 11) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 7 && dim_size_log2 <= 10) condition3 = true;
        }
    }
    // printf("fwd outer_size_log2=%d, dim_size_log2=%d, inner_size_log2=%d, \n", outer_size_log2, dim_size_log2, inner_size_log2);
    // printf("fwd condition1=%d, condition2=%d, condition3=%d, all=%d, \n", condition1, condition2, condition3, condition1 && condition2 && condition3);
    return (condition1 && condition2 && condition3);
}

bool softmax_spatial_bwd_use_transpose(int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    const int outer_size_log2 = ::log2(outer_size);
    const int dim_size_log2 = ::log2(dim_size);
    const int inner_size_log2 = ::log2(inner_size);
    const int size_log2_sum = dim_size_log2 + inner_size_log2;
    const bool condition1 = inner_size_log2 <= 9 && outer_size_log2 <= 4 && dim_size_log2 <= 14 && size_log2_sum <= 21;
    const bool condition2 = inner_size_log2 > 7 && outer_size_log2 > 1 && dim_size_log2 > 6 && size_log2_sum > 13;

    bool condition3 = false;
    if (inner_size_log2 <= 7) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 15) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 14) condition3 = true;
        } else if (outer_size_log2 <= 4) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 13) condition3 = true;
        }
    } else if (inner_size_log2 <= 8) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 14) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 13) condition3 = true;
        } else if (outer_size_log2 <= 4) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 11) condition3 = true;
        }
    } else if (inner_size_log2 <= 9) {
        if(outer_size_log2 <= 2) {
            if (dim_size_log2 >= 8 && dim_size_log2 <= 13) condition3 = true;
        } else if (outer_size_log2 <= 3) {
            if (dim_size_log2 >= 9 && dim_size_log2 <= 10) condition3 = true;
        }
    }
    // printf("bwd outer_size_log2=%d, dim_size_log2=%d, inner_size_log2=%d, \n", outer_size_log2, dim_size_log2, inner_size_log2);
    // printf("bwd condition1=%d, condition2=%d, condition3=%d, all=%d, \n", condition1, condition2, condition3, condition1 && condition2 && condition3);
    return (condition1 && condition2 && condition3);
}

}   // namespace softmax_op

} // namespace native
} // namespace at
