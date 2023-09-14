#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <numeric>
#include <torch/autograd.h>
#include <torch/extension.h>
#include <type_traits>
#include <ATen/ATen.h>

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
    #include <hip/hip_runtime.h>
    #include <hipcub/hipcub.hpp>
    #include <hiprand_kernel.h>
#else

    #include <cuda.h>
    #include <cuda_fp16.h>
    #include <curand_kernel.h>

#endif

namespace at {
namespace native {

namespace {

void compute_n1_n2(const Tensor &input,
                   int normalized_shape,
                   int &n1, int &n2) {

    const int input_ndim = input.ndimension();
    n2 = 1;
    int l = input.ndimension() - 1;
    for (; l >= 0; l--) {
        n2 *= input.sizes()[l];
        if (n2 >= normalized_shape) {
            break;
        }
    }
    TORCH_CHECK(n2 == normalized_shape, "normalized_shape must be same with the last dims of input");

    n1 = 1;
    for (int j = 0; j < l; ++j) {
        n1 *= input.sizes()[j];
    }
}

}

namespace norms_device_kernel_tools {

#if defined(__HIP_PLATFORM_HCC__)
    #define BLOCK_WARP_SIZE (64)
#else
    #define BLOCK_WARP_SIZE (32)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// welford modules ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename U>
struct WelfordData_LN {
    U mean;
    U sigma2;
    U count;

    C10_HOST_DEVICE WelfordData_LN()  : mean(0.f), sigma2(0.f), count(0.f) {}

    C10_HOST_DEVICE WelfordData_LN(U mean, U sigma2, U count) : mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U>
struct WelfordData_RMS {
    U sigma2;

    C10_HOST_DEVICE WelfordData_RMS() :  sigma2(0.f) {}

    C10_HOST_DEVICE WelfordData_RMS(U sigma2) : sigma2(sigma2) {}
};

/** 传入值，更新均值和方差 */
template<typename U>
__device__ void Welford_LN_OnlineSum(const U &val, WelfordData_LN<U> &curr_sum) {
    curr_sum.count += U(1);
    U delta = val - curr_sum.mean;
    curr_sum.mean = curr_sum.mean + delta * (1.f / curr_sum.count); //proper division is slow, this is less accurate but noticeably faster
    curr_sum.sigma2 += delta * (val - curr_sum.mean);
}

template<typename U>
__device__ void Welford_RMS_OnlineSum(const U &val, WelfordData_RMS<U> &curr_sum) {
    curr_sum.sigma2 += val * val;
}

/** 合并两个welford的均值和方差 */
template<typename U>
__device__ void Welford_LN_Combine(WelfordData_LN<U> &dataA, const WelfordData_LN<U> &dataB) {
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    if (count > U(0)) {
        auto coef = 1.f / count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
        auto nA = dataA.count * coef;
        auto nB = dataB.count * coef;
        dataA.mean = nA * dataA.mean + nB * dataB.mean;
        dataA.sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
        dataA.mean = U(0);
        dataA.sigma2 = U(0);
    }
    dataA.count = count;
}

template<typename U>
__device__ void Welford_RMS_Combine(WelfordData_RMS<U> &dataA, const WelfordData_RMS<U> &dataB) {
    dataA.sigma2 += dataB.sigma2;
}

/** 计算welford的结果 */
template<typename T, typename U, int VEC = 4>
__device__ WelfordData_LN<U> Welford_LN_MuSigma2(
    const T *__restrict__ vals_row,
    const int n2, U *buf,
    const int &num_x,
    const int &thread_x) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
    using vec_t = memory::aligned_vector<T, VEC>;
    const vec_t *vals_row_vec = reinterpret_cast<const vec_t *>(vals_row);
    const int n_vec_to_read = n2 / VEC;
    WelfordData_LN<U> wd(0.f, 0.f, 0.f);

    // synchronization is implicit initialize with standard Welford algorithm
    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data = vals_row_vec[l];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            Welford_LN_OnlineSum<U>(static_cast<U>(data.val[k]), wd);
        }
    }
    // deal tail data
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        Welford_LN_OnlineSum<U>(static_cast<U>(vals_row[l_last]), wd);
    }

    // intra-warp reductions
    #pragma unroll
    for (int offset = (BLOCK_WARP_SIZE >> 1); offset > 0; (offset >>= 1)) {
        WelfordData_LN<U> wdB{WARP_SHFL_DOWN(wd.mean, offset, BLOCK_WARP_SIZE),
                              WARP_SHFL_DOWN(wd.sigma2, offset, BLOCK_WARP_SIZE),
                              WARP_SHFL_DOWN(wd.count, offset, BLOCK_WARP_SIZE)};
        Welford_LN_Combine<U>(wd, wdB);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    U *mean_sigma_buf = buf;
    U *count_buf = mean_sigma_buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) { // offset: 2,1,0
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            mean_sigma_buf[2 * wrt_y] = wd.mean;
            mean_sigma_buf[2 * wrt_y + 1] = wd.sigma2;
            count_buf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordData_LN<U> wdB{mean_sigma_buf[2 * threadIdx.y], mean_sigma_buf[2 * threadIdx.y + 1], count_buf[threadIdx.y]};
            Welford_LN_Combine<U>(wd, wdB);
        }
        __syncthreads();
    }
    // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mean_sigma_buf[0] = wd.mean;
        mean_sigma_buf[1] = wd.sigma2 / U(n2);
    }
    __syncthreads();

    return WelfordData_LN<U>{mean_sigma_buf[0], mean_sigma_buf[1], 0.f};
}

template<typename T, typename U, int VEC = 4>
__device__ WelfordData_RMS<U> Welford_RMS_Sigma2(
    const T *__restrict__ vals_row,
    const int n2, U *buf,
    const int &num_x,
    const int &thread_x) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
    using vec_t = memory::aligned_vector<T, VEC>;
    const vec_t *vals_row_vec = reinterpret_cast<const vec_t *>(vals_row);
    const int n_vec_to_read = n2 / VEC;
    WelfordData_RMS<U> wd(0.f);

    // synchronization is implicit initialize with standard Welford algorithm
    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data = vals_row_vec[l];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            Welford_RMS_OnlineSum<U>(static_cast<U>(data.val[k]), wd);
        }
    }
    // deal tail data
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        Welford_RMS_OnlineSum<U>(static_cast<U>(vals_row[l_last]), wd);
    }

    // intra-warp reductions
    #pragma unroll
    for (int offset = (BLOCK_WARP_SIZE >> 1); offset > 0; (offset >>= 1)) {
        WelfordData_RMS<U> wdB{WARP_SHFL_DOWN(wd.sigma2, offset, BLOCK_WARP_SIZE)};
        Welford_RMS_Combine<U>(wd, wdB);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    //    if (blockDim.y > 1) {
    U *sigma_buf = buf;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            sigma_buf[2 * wrt_y] = wd.sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordData_RMS<U> wdB{sigma_buf[2 * threadIdx.y]};
            Welford_RMS_Combine<U>(wd, wdB);
        }
        __syncthreads();
    }
    // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sigma_buf[0] = wd.sigma2 / U(n2);
    }
    __syncthreads();

    return WelfordData_RMS<U>{sigma_buf[0]};
}

// https://github.com/NVIDIA/apex/issues/246
template<typename T>
struct SharedMemory;

template<>
struct SharedMemory<float> {
    __device__ float *getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template<>
struct SharedMemory<double> {
    __device__ double *getPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};

} // norms_device_kernel_tools


namespace norms_device_kernel {
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// norms forward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename U=float, int VEC = 4>
__device__ void cuApplyLnNorm_(
    T *__restrict__ output_vals,
    U *__restrict__ mean,
    U *__restrict__ rstd,
    const T *__restrict__ in_vals,
    const int &n1,
    const int &n2,
    const U epsilon,
    const T *__restrict__ gamma,
    const T *__restrict__ beta) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensors are contiguous
    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = n2 / VEC;

    int i1 = blockIdx.x;
    const T *input_row = in_vals + i1 * n2;
    T *output_row = output_vals + i1 * n2;

    norms_device_kernel_tools::SharedMemory<U> shared;
    U *buf = shared.getPointer();
    using vec_t = memory::aligned_vector<T, VEC>;
    const vec_t *input_row_vec = reinterpret_cast<const vec_t *>(input_row);
    vec_t *output_row_vec = reinterpret_cast<vec_t *>(output_row);

    norms_device_kernel_tools::WelfordData_LN<U> wd = norms_device_kernel_tools::Welford_LN_MuSigma2<T, U, VEC>(input_row, n2, buf, num_x, thread_x);
    U rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + epsilon);
    if (thread_x == 0) {
        mean[i1] = wd.mean;
        rstd[i1] = rstd_val;
    }

    // vectorized operator
    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data = input_row_vec[l];
        vec_t out;

        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr && beta != nullptr) {
            #pragma unroll
            for (int ii = 0; ii < VEC; ii++) {  // ln norm, affine=True
                out.val[ii] = gamma[l * VEC + ii] * static_cast<T>(rstd_val * (static_cast<U>(data.val[ii]) - wd.mean)) + beta[l * VEC + ii];
            }
        } else {
            #pragma unroll
            for (int ii = 0; ii < VEC; ii++) {
                out.val[ii] = static_cast<T>(rstd_val * (static_cast<U>(data.val[ii]) - wd.mean));
            }
        }
        output_row_vec[l] = out;
    }

    // deal tail data
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr && beta != nullptr) {
            output_row[l_last] = gamma[l_last] * static_cast<T>(rstd_val * (input_row[l_last] - wd.mean)) + beta[l_last];
        } else {
            output_row[l_last] = static_cast<T>(rstd_val * (input_row[l_last] - wd.mean));
        }
    }
}

template<typename T, typename U=float, int VEC = 4>
__device__ void cuApplyRmsNorm_(
    T *__restrict__ output_vals,
    U *__restrict__ rstd,
    const T *__restrict__ in_vals,
    const int &n1,
    const int &n2,
    const U epsilon,
    const T *__restrict__ gamma) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensors are contiguous
    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = n2 / VEC;

    int i1 = blockIdx.x;
    const T *input_row = in_vals + i1 * n2;
    T *output_row = output_vals + i1 * n2;

    norms_device_kernel_tools::SharedMemory<U> shared;
    U *buf = shared.getPointer();
    using vec_t = memory::aligned_vector<T, VEC>;
    const vec_t *input_row_vec = reinterpret_cast<const vec_t *>(input_row);
    vec_t *output_row_vec = reinterpret_cast<vec_t *>(output_row);

    norms_device_kernel_tools::WelfordData_RMS<U> wd = norms_device_kernel_tools::Welford_RMS_Sigma2(input_row, n2, buf, num_x, thread_x);
    U rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + epsilon);

    if (thread_x == 0) {
        rstd[i1] = rstd_val;
    }
    // vectorized operator
    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data = input_row_vec[l];
        vec_t out;

        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr) {
            #pragma unroll
            for (int ii = 0; ii < VEC; ii++) {  // ln norm, affine=True
                out.val[ii] = gamma[l * VEC + ii] * static_cast<T>(rstd_val * (static_cast<U>(data.val[ii])));
            }
        } else {
            #pragma unroll
            for (int ii = 0; ii < VEC; ii++) {
                out.val[ii] = static_cast<T>(rstd_val * (static_cast<U>(data.val[ii])));
            }
        }
        output_row_vec[l] = out;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr) {
            output_row[l_last] = gamma[l_last] * static_cast<T>(rstd_val * (input_row[l_last]));
        } else {
            output_row[l_last] = static_cast<T>(rstd_val * (input_row[l_last]));
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// norms backward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__inline__ __device__ T WarpReduceSum(T val) {
    for (int offset = (BLOCK_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val += WARP_SHFL_DOWN(val, offset, BLOCK_WARP_SIZE);
    }
    return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / BLOCK_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x * blockDim.y / BLOCK_WARP_SIZE;
    }
};

// Sums `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `BLOCK_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template<typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
    const int tid = B::Tid();
    const int lid = tid % BLOCK_WARP_SIZE;
    const int wid = tid / BLOCK_WARP_SIZE;
    val = WarpReduceSum(val);
    __syncthreads(); // prevent races when BlockReduces are called in a row.
    if (lid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (tid < B::Warps()) ? shared[lid] : T(0);
    if (wid == 0) {
        val = WarpReduceSum(val);
    }
    return val;
}

//// compute grad gamma and beta
template<typename T, typename U>
__global__ void cuApply_Ln_GammaBetaBackward_Simple(
    int64_t n1, int64_t n2,
    const T *dY, const T *X,
    const U *mean, const U *rstd,
    T *grad_gamma, T *grad_beta) {  // assert grad_gamma!=nullptr and grad_beta!=nullptr

    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n2) { return; }

    U sum1 = 0;
    U sum2 = 0;
    for (int64_t i = 0; i < n1; ++i) {
        const int64_t index = i * n2 + j;
        sum1 += static_cast<U>(dY[index]) * (static_cast<U>(X[index]) - mean[i]) * rstd[i];
        sum2 += static_cast<U>(dY[index]);
    }
    grad_gamma[j] = sum1;
    grad_beta[j] = sum2;
}

template<typename T, typename U>
__global__ void cuApply_Rms_GammaBackward_Simple(
    int64_t n1, int64_t n2,
    const T *dY, const T *X,
    const U *rstd, T *grad_gamma) {  // assert grad_gamma!=nullptr

    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n2) { return; }

    U sum1 = 0;
    for (int64_t i = 0; i < n1; ++i) {
        const int64_t index = i * n2 + j;
        sum1 += static_cast<U>(dY[index]) * static_cast<U>(X[index]) * rstd[i];
    }
    grad_gamma[j] = sum1;
}

template<typename T, typename U, int VEC = 8>
__global__ void cuApply_Ln_GammaBetaBackward_Vec(
    int64_t n1, int64_t n2,
    const T *dY, const T *X,
    const U *mean, const U *rstd,
    T *grad_gamma, T *grad_beta) {

    alignas(sizeof(double))
    extern __shared__ char s_data1[];
    U *s_data_typed = reinterpret_cast<U *>(&s_data1);
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t1d_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int64_t t2d_idx = blockDim.x * blockDim.y + t1d_idx;

    T dYs[VEC];
    T Xs[VEC];
    U *means = s_data_typed;
    U *rstds = s_data_typed + blockDim.x * blockDim.y;
    U dg_sum = 0;
    U db_sum = 0;

    if (j >= n2) { return; }

    int bcounter;
    for (bcounter = 0; bcounter < n1 / (blockDim.y * VEC); bcounter++) {
        int offset = (bcounter * blockDim.y + threadIdx.y) * VEC;
#pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int temp_offset = offset + ii;

            if (threadIdx.x == 0) {
                means[ii * blockDim.y + threadIdx.y] = mean[temp_offset];
                rstds[ii * blockDim.y + threadIdx.y] = rstd[temp_offset];
            }
            dYs[ii] = dY[temp_offset * n2 + j];
            Xs[ii] = X[temp_offset * n2 + j];
        }
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            dg_sum += static_cast<U>(dYs[ii]) * (static_cast<U>(Xs[ii]) - means[ii * blockDim.y + threadIdx.y]) * rstds[ii * blockDim.y + threadIdx.y];
            db_sum += static_cast<U>(dYs[ii]);
        }
        __syncthreads();
    }
    int offset = (bcounter * blockDim.y + threadIdx.y) * VEC;
    for (int ii = 0; ii < VEC; ii++) {
        int temp_offset = offset + ii;
        U mean_val, rstd_val; // we don't use smem in the tail to avoid awkward synchronizations, perf penalty is negligible
        if (temp_offset < n1) {
            mean_val = mean[temp_offset];
            rstd_val = rstd[temp_offset];
            dYs[0] = dY[temp_offset * n2 + j];
            Xs[0] = X[temp_offset * n2 + j];
            dg_sum += static_cast<U>(dYs[0]) * (static_cast<U>(Xs[0]) - mean_val) * rstd_val;
            db_sum += static_cast<U>(dYs[0]);
        }
    }
    s_data_typed[t1d_idx] = dg_sum;
    s_data_typed[t2d_idx] = db_sum;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int temp_offset = offset * blockDim.x;
            s_data_typed[t1d_idx] += s_data_typed[t1d_idx + temp_offset];
            s_data_typed[t2d_idx] += s_data_typed[t2d_idx + temp_offset];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        grad_gamma[j] = static_cast<T>(s_data_typed[threadIdx.x]);
        grad_beta[j] = static_cast<T>(s_data_typed[threadIdx.x + blockDim.x * blockDim.y]);
    }
}

template<typename T, typename U, int VEC = 8>
__global__ void cuApply_Rms_GammaBackward_Vec(
    int64_t n1, int64_t n2,
    const T *dY, const T *X,
    const U *rstd,
    T *grad_gamma) {

    alignas(sizeof(double))
    extern __shared__ char s_data1[];
    U *s_data_typed = reinterpret_cast<U *>(&s_data1);
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t1d_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (j >= n2) { return; }

    T dYs[VEC];
    T Xs[VEC];
    U *rstds = s_data_typed;
    U dg_sum = 0;

    int bcounter;
    for (bcounter = 0; bcounter < n1 / (blockDim.y * VEC); bcounter++) {
        int offset = (bcounter * blockDim.y + threadIdx.y) * VEC;
#pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            int temp_offset = offset + ii;
            if (threadIdx.x == 0) {
                rstds[ii * blockDim.y + threadIdx.y] = rstd[temp_offset];
            }
            dYs[ii] = dY[temp_offset * n2 + j];
            Xs[ii] = X[temp_offset * n2 + j];
        }
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            dg_sum += static_cast<U>(dYs[ii]) * static_cast<U>(Xs[ii]) * rstds[ii * blockDim.y + threadIdx.y];
        }
        __syncthreads();
    }
    int offset = (bcounter * blockDim.y + threadIdx.y) * VEC;
    for (int ii = 0; ii < VEC; ii++) {
        int temp_offset = offset + ii;
        U rstd_val; // we don't use smem in the tail to avoid awkward synchronizations, perf penalty is negligible
        if (temp_offset < n1) {
            rstd_val = rstd[temp_offset];
            dYs[0] = dY[temp_offset * n2 + j];
            Xs[0] = X[temp_offset * n2 + j];
            dg_sum += static_cast<U>(dYs[0]) * static_cast<U>(Xs[0]) * rstd_val;
        }
    }
    s_data_typed[t1d_idx] = dg_sum;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int temp_offset = offset * blockDim.x;
            s_data_typed[t1d_idx] += s_data_typed[t1d_idx + temp_offset];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        grad_gamma[j] = static_cast<T>(s_data_typed[threadIdx.x]);
    }
}

//// compute grad input
template<typename T, typename U, int VEC = 4>
__device__ __inline__ void cuApply_Ln_GradInput_(
    const T *__restrict__ dY,
    const T *__restrict__ X,
    const U *__restrict__ mean,
    const U *__restrict__ rstd,
    const T *__restrict__ gamma,
    T *dX,
    const int n1, const int n2,
    U *buf) {

    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;

    const auto i1 = blockIdx.x;
    const U mean_val = mean[i1];
    const U rstd_val = rstd[i1];
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);

    const int row_size = i1 * n2;
    const T *X_row = X + row_size;
    const T *dY_row = dY + row_size;
    T *dX_row = dX + row_size;

    using vec_t = memory::aligned_vector<T, VEC>;
    const int n_vec_to_read = n2 / VEC;
    const vec_t *X_row_vec = reinterpret_cast<const vec_t *>(X_row);
    const vec_t *dY_row_vec = reinterpret_cast<const vec_t *>(dY_row);
    vec_t *gamma_vec;
    if(gamma != nullptr) {
        gamma_vec= reinterpret_cast<vec_t *>(const_cast<T *>(gamma));    // const_cast<T *>
    }

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_X = X_row_vec[l];
        vec_t data_dY = dY_row_vec[l];
        vec_t data_gamma;
        if(gamma != nullptr){
            data_gamma = gamma_vec[l];
        }

        for (int k = 0; k < VEC; k++) {
            U gamma_val = (gamma != nullptr) ? static_cast<U>(data_gamma.val[k]) : U(1);
            const U c_h = static_cast<U>(data_X.val[ k]);
            const U c_loss = static_cast<U>(data_dY.val[ k]);
            sum_loss1 += c_loss * gamma_val;
            sum_loss2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
        }
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        U gamma_val = (gamma != nullptr) ? static_cast<U>(gamma[l_last]) : U(1);
        const U c_h = static_cast<U>(X_row[l_last]);
        const U c_loss = static_cast<U>(dY_row[l_last]);
        sum_loss1 += c_loss * gamma_val;
        sum_loss2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }

    sum_loss1 = BlockReduceSum<U>(sum_loss1, buf);
    sum_loss2 = BlockReduceSum<U>(sum_loss2, buf);
    if (thread_x == 0) {
        buf[0] = sum_loss1;
        buf[1] = sum_loss2;
    }
    __syncthreads();

    sum_loss1 = buf[0];
    sum_loss2 = buf[1];
    U fH = U(n2);
    U term1 = (U(1) / fH) * rstd_val;

    for (int l = thread_x; l < n2; l += blockDim.x) {
        const U x = X_row[l];
        const U dy = dY_row[l];
        U gamma_val = (gamma != nullptr) ? static_cast<U>(gamma[l]) : U(1);
        U f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * sum_loss2;
        f_grad_input -= sum_loss1;
        f_grad_input *= term1;
        dX_row[l] = f_grad_input;
    }
}

template<typename T, typename U, int VEC = 4>
__device__ __inline__ void cuApply_Rms_GradInput_(
    const T *__restrict__ dY,
    const T *__restrict__ X,
    const U *__restrict__ rstd,
    const T *__restrict__ gamma,
    T *dX,
    const int n1, const int n2,
    U *buf) {

    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;

    const auto i1 = blockIdx.x;
    const U rstd_val = rstd[i1];
    U sum_loss2 = U(0);

    const int row_size = i1 * n2;
    const T *X_row = X + row_size;
    const T *dY_row = dY + row_size;
    T *dX_row = dX + row_size;

    using vec_t = memory::aligned_vector<T, VEC>;
    const int n_vec_to_read = n2 / VEC;
    const vec_t *X_row_vec = reinterpret_cast<const vec_t *>(X_row);
    const vec_t *dY_row_vec = reinterpret_cast<const vec_t *>(dY_row);
    vec_t *gamma_vec;
    if(gamma != nullptr) {
        gamma_vec= reinterpret_cast<vec_t *>(const_cast<T *>(gamma));    // const_cast<T *>
    }

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_X = X_row_vec[l];
        vec_t data_dY = dY_row_vec[l];
        vec_t data_gamma;
        if(gamma != nullptr){
            data_gamma = gamma_vec[l];
        }

        for (int k = 0; k < VEC; k++) {
            U gamma_val = (gamma != nullptr) ? static_cast<U>(data_gamma.val[k]) : U(1);
            sum_loss2 += static_cast<U>(data_dY.val[ k]) * gamma_val * static_cast<U>(data_X.val[k]) * rstd_val;
        }
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        U gamma_val = (gamma != nullptr) ? static_cast<U>(gamma[l_last]) : U(1);
        sum_loss2 += static_cast<U>(dY_row[l_last]) * gamma_val * static_cast<U>(X_row[l_last]) * rstd_val;
    }

    sum_loss2 = BlockReduceSum(sum_loss2, buf);
    if (thread_x == 0) {
        buf[0] = sum_loss2;
    }
    __syncthreads();
    sum_loss2 = buf[0];
    U fH = n2;
    U term1 = (U(1) / fH) * rstd_val;

    for (int l = thread_x; l < n2; l += blockDim.x) {
        const U x = X_row[l];
        const U dy = dY_row[l];
        U gamma_val = (gamma != nullptr) ? static_cast<U>(gamma[l]) : U(1);
        U f_grad_input = fH * gamma_val * dy;
        f_grad_input -= x * rstd_val * sum_loss2;
        f_grad_input *= term1;
        dX_row[l] = f_grad_input;
    }
}

} // norms_device_kernel

namespace add_drops_device_kernel {
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// add drop forward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename U, int VEC = 4>
__device__ void cuApplyAdd_(    // only add
    T *__restrict__ output_vals,
    const T *__restrict__ input,
    const T *__restrict__ residual,
    const int n1, const int n2) {

    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;

    using vec_t = memory::aligned_vector<T, VEC>;
    const int n_vec_to_read = n2 / VEC;
    int i1 = blockIdx.x;

    const int row_size = i1 * n2;
    const T *input_row = input + row_size;
    const T *residual_row = residual + row_size;
    T *output_row = output_vals + row_size;

    const vec_t *input_row_vec = reinterpret_cast<const vec_t *>(input_row);
    const vec_t *residual_row_vec = reinterpret_cast<const vec_t *>(residual_row);
    vec_t *output_row_vec = reinterpret_cast<vec_t *>(output_row);

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_in = input_row_vec[l];
        vec_t data_residual = residual_row_vec[l];
        vec_t data_out;

        for (int k = 0; k < VEC; ++k) {
            data_out.val[k] = data_in.val[k] + data_residual.val[k];
        }
        output_row_vec[l] = data_out;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        output_row[l_last] = input_row[l_last] + residual_row[l_last];
    }
}

template<typename T, typename U, int VEC = 4>
__device__ void cuApplyAddDrops_(   // dropmask != nullptr and residual != nullptr
    T *__restrict__ output_vals,
    const T *__restrict__ input,
    const T *__restrict__ residual,
    const int n1, const int n2,
    uint8_t *__restrict__ drop_mask,
    U drop_keep_rate,
    const PhiloxCudaState &philox_args) {

    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;
    int i1 = blockIdx.x;

    // prepare rand seed and state
    auto seeds = at::cuda::philox::unpack(philox_args);
    curandStatePhilox4_32_10_t state;
    int rand_idx = i1 * num_x + thread_x;
    curand_init(std::get<0>(seeds), rand_idx, std::get<1>(seeds), &state);
    U pinv = U(1) / drop_keep_rate;

    // prepare vectorized operator
    using vec_t = memory::aligned_vector<T, VEC>;
    using vec_t_uint8 = memory::aligned_vector<uint8_t, VEC>;
    const int n_vec_to_read = n2 / VEC;

    const int row_size = i1 * n2;
    const T *input_row = input + row_size;
    uint8_t *drop_mask_row = drop_mask + row_size;
    T *output_row = output_vals + row_size;
    const vec_t *input_row_vec = reinterpret_cast<const vec_t *>(input_row);
    vec_t_uint8 *drop_mask_row_vec = reinterpret_cast<vec_t_uint8 *>(drop_mask_row);
    vec_t *output_row_vec = reinterpret_cast<vec_t *>(output_row);

    // residual if not nullptr
    T *residual_row;
    vec_t *residual_row_vec;
    if (residual != nullptr) {
        residual_row = const_cast<T *>(residual + row_size);
        residual_row_vec = reinterpret_cast<vec_t *>(residual_row);
    }

    int l = thread_x;
    float4 rand;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_in = input_row_vec[l];
        vec_t_uint8 data_drop_mask;
        vec_t data_residual;
        if (residual != nullptr) {
            data_residual = residual_row_vec[l];
        }
        vec_t data_out;

        rand = curand_uniform4(&state);
        for (int k = 0; k < VEC; ++k) {
            data_drop_mask.val[k] = (uint8_t)((&rand.x)[k] <= drop_keep_rate);
            U curr_val = static_cast<T>(static_cast<U>(data_in.val[k]) * U(data_drop_mask.val[k]) * pinv);
            if (residual != nullptr) {
                curr_val += static_cast<U>(data_residual.val[k]);
            }
            data_out.val[k] = curr_val;
        }
        drop_mask_row_vec[l] = data_drop_mask;
        output_row_vec[l] = data_out;
    }

    if (l * VEC < n2) {
        rand = curand_uniform4(&state);
        rand.x = rand.x <= drop_keep_rate;
        rand.y = rand.y <= drop_keep_rate;
        rand.z = rand.z <= drop_keep_rate;
        rand.w = rand.w <= drop_keep_rate;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        int simple_idx = l_last - l * VEC;  // assert simple_idx dur 0~3
        drop_mask_row[l_last] = (uint8_t)((&rand.x)[simple_idx] <= drop_keep_rate);
        U curr_val = static_cast<T>(static_cast<U>(input_row[l_last]) * U((&rand.x)[simple_idx]) * pinv);
        if (residual != nullptr) {
            curr_val += static_cast<U>(residual_row[l_last]);
        }
        output_row[l_last] = curr_val;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// add drop backward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Dropout Add Backward ,  assert drop_mask != nullptr*/
template<typename T, typename U, int VEC = 4>
__device__ void cuApplyAddDropsGradient_(T *grad_in, T *residual_grads,
                                         const T *grad_output, const uint8_t *drop_mask,
                                         const int n1, const int n2, const U pinv) {
    // dim3 block_dims(warp_size, 4, 1);  // MI100 wavefront/warp = 64
    // const dim3 grid_dims(n1);
    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;

    int i1 = blockIdx.x;
    const int row_size = i1 * n2;

    const T *grad_output_row = grad_output + row_size;
    const uint8_t *drop_mask_row = drop_mask + row_size;
    T *grad_in_row = grad_in + row_size;
    // residual grad
    T *residual_grads_row;
    if (residual_grads != nullptr) {
        residual_grads_row = residual_grads + row_size;
    }

    // prepare vectorized operator
    using vec_t = memory::aligned_vector<T, VEC>;
    using vec_t_uint8 = memory::aligned_vector<uint8_t, VEC>;
    const int n_vec_to_read = n2 / VEC;

    const vec_t *grad_output_row_vec = reinterpret_cast<const vec_t *>(grad_output_row);
    const vec_t_uint8 *drop_mask_row_vec = reinterpret_cast<const vec_t_uint8 *>(drop_mask_row);
    vec_t *grad_in_row_vec = reinterpret_cast<vec_t *>(grad_in_row);
    vec_t *residual_grads_row_vec;
    if (residual_grads != nullptr) {
        residual_grads_row_vec = reinterpret_cast<vec_t *>(residual_grads_row);
    }

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_grad_output = grad_output_row_vec[l];
        vec_t_uint8 data_drop_mask = drop_mask_row_vec[l];
        vec_t data_residual_grads, data_grad_in;

        for (int k = 0; k < VEC; ++k) {
            T grad_out_val = data_grad_output.val[k];
            if (residual_grads != nullptr) {
                data_residual_grads.val[k] = grad_out_val;
            }
            grad_out_val *= static_cast<T>(static_cast<U>(data_drop_mask.val[k]) * pinv);
            data_grad_in.val[k] = grad_out_val;
        }

        if (residual_grads != nullptr) {
            residual_grads_row_vec[l] = data_residual_grads;
        }
        grad_in_row_vec[l] = data_grad_in;
    }

    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        T grad_out_val = grad_output_row[l_last];
        if (residual_grads != nullptr) {
            residual_grads_row[l_last] = grad_out_val;
        }
        grad_out_val *= static_cast<T>(static_cast<U>(drop_mask_row[l_last]) * pinv);
        grad_in_row[l_last] = grad_out_val;
    }
}

} // add_drops_device_kernel

} // native
} // at
