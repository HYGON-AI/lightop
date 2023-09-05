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
    curr_sum.count = curr_sum.count + U(1);
    U delta = val - curr_sum.mean;
    curr_sum.mean = curr_sum.mean + delta * (1.f / curr_sum.count); //proper division is slow, this is less accurate but noticeably faster
//    curr_sum.mean = curr_sum.mean + delta / curr_sum.count; //proper division is slow, this is less accurate but noticeably faster
    curr_sum.sigma2 = curr_sum.sigma2 + delta * (val - curr_sum.mean);
}

template<typename U>
__device__ void Welford_RMS_OnlineSum(const U &val, WelfordData_RMS<U> &curr_sum) {
    curr_sum.sigma2 = curr_sum.sigma2 + val * val;
}

/** 合并两个welford的均值和方差 */
template<typename U>
__device__ void Welford_LN_Combine(WelfordData_LN<U> &dataA, const WelfordData_LN<U> dataB) {
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
__device__ void Welford_RMS_Combine(WelfordData_RMS<U> &dataA, const WelfordData_RMS<U> dataB) {
    dataA.sigma2 = dataA.sigma2 + dataB.sigma2;
}

/** 计算welford的结果 */
template<typename T, typename U, int VEC = 4>
__device__ void Welford_LN_MuSigma2(
    const T *__restrict__ vals_row,
    const int n2,
    U &mu, U &sigma2, U *buf,
    const int &GPU_WARP_SIZE,
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
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        Welford_LN_OnlineSum<U>(static_cast<U>(vals_row[l_last]), wd);
    }

    // intra-warp reductions
    #pragma unroll
    for (int offset = (GPU_WARP_SIZE >> 1); offset > 0; (offset >>= 1)) {
        WelfordData_LN<U> wdB{WARP_SHFL_DOWN(wd.mean, offset), WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        Welford_LN_Combine<U>(wd, wdB);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    // assert blockDim.y > 1, otherwise result is error
    U *mean_sigma_buf = buf;
    U *count_buf = (U *) (mean_sigma_buf + blockDim.y);
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
        mean_sigma_buf[1] = wd.sigma2;
    }
    __syncthreads();

    mu = mean_sigma_buf[0];
    sigma2 = mean_sigma_buf[1] / U(n2);
}

template<typename T, typename U, int VEC = 4>
__device__ void Welford_RMS_Sigma2(
    const T *__restrict__ vals_row,
    const int n2,
    U &sigma2, U *buf,
    const int &GPU_WARP_SIZE,
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
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        Welford_RMS_OnlineSum<U>(static_cast<U>(vals_row[l_last]), wd);
    }

    // intra-warp reductions
    #pragma unroll
    for (int offset = (GPU_WARP_SIZE >> 1); offset > 0; (offset >>= 1)) {
        WelfordData_RMS<U> wdB{WARP_SHFL_DOWN(wd.sigma2, offset)};
        Welford_RMS_Combine<U>(wd, wdB);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
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
        sigma_buf[0] = wd.sigma2;
    }
    __syncthreads();

    sigma2 = sigma_buf[0] / U(n2);
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
    U *__restrict__ invvar,
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

    U mu, sigma2;
    norms_device_kernel_tools::Welford_LN_MuSigma2<T, U, VEC>(input_row, n2, mu, sigma2, buf, blockDim.x, num_x, thread_x);

    U rstd_val = c10::cuda::compat::rsqrt(sigma2 + epsilon);
    if (thread_x == 0) {
        mean[i1] = mu;
        invvar[i1] = rstd_val;
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
                out.val[ii] = gamma[l * VEC + ii] * static_cast<T>(rstd_val * (static_cast<U>(data.val[ii]) - mu)) + beta[l * VEC + ii];
            }
        } else {
            #pragma unroll
            for (int ii = 0; ii < VEC; ii++) {
                out.val[ii] = static_cast<T>(rstd_val * (static_cast<U>(data.val[ii]) - mu));
            }
        }
        output_row_vec[l] = out;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr && beta != nullptr) {
            output_row[l_last] = gamma[l_last] * static_cast<T>(rstd_val * (input_row[l_last] - mu)) + beta[l_last];
        } else {
            output_row[l_last] = static_cast<T>(rstd_val * (input_row[l_last] - mu));
        }
    }
}

template<typename T, typename U=float, int VEC = 4>
__device__ void cuApplyRmsNorm_(
    T *__restrict__ output_vals,
    U *__restrict__ invvar,
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

    U sigma2;
    norms_device_kernel_tools::Welford_RMS_Sigma2(input_row, n2, sigma2, buf, blockDim.x, num_x, thread_x);
    U rstd_val = c10::cuda::compat::rsqrt(sigma2 + epsilon);
    if (thread_x == 0) {
        invvar[i1] = rstd_val;
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

template<typename T, typename U>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U *warp_buf1,
    U *warp_buf2,
    const T *input,
    const T *dout,
    const int i1_end,
    const int n2,
    const U *__restrict__ mean,
    const U *__restrict__ invvar,
    bool rms_only
) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_mean;
        if (!rms_only) {
            curr_mean = mean[i1];
        }
        U curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                if (!rms_only) {
                    warp_buf1[write_idx] = curr_dout;
                    warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
                } else {
                    warp_buf2[write_idx] = curr_dout * (curr_input) * curr_invvar;
                }
            } else {
                if (!rms_only) {
                    warp_buf1[write_idx] = U(0);
                }
                warp_buf2[write_idx] = U(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (!rms_only) {
                warp_buf1[write_idx] = U(0);
            }
            warp_buf2[write_idx] = U(0);
        }
    }
}

template<typename T, typename U>
__device__ void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U *warp_buf1,
    U *warp_buf2,
    const T *input,
    const T *dout,
    const int n1,
    const int n2,
    const U *__restrict__ mean,
    const U *__restrict__ invvar,
    bool rms_only
) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < n1) {
        U curr_mean;
        if (!rms_only) { curr_mean = mean[i1]; }
        U curr_invvar = invvar[i1];

        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                if (!rms_only) {
                    warp_buf1[write_idx] += curr_dout;
                    warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
                } else {
                    warp_buf2[write_idx] += curr_dout * (curr_input) * curr_invvar;
                }
            }
        }
    }
}

template<typename T, typename U>
__device__ void cuApplyComputePartGradGammaBeta_(
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const int n1,
    const int n2,
    const U *__restrict__ mean,
    const U *__restrict__ invvar,
    U epsilon,
    U *part_grad_gamma,
    U *part_grad_beta,
    bool rms_only) {
    const int block_square = blockDim.y * blockDim.y;
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;

    norms_device_kernel_tools::SharedMemory<U> shared;
    U *buf = shared.getPointer(); // buf has at least blockDim.x * block_square + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements

    U *warp_buf1 = (U *) buf;
    U *warp_buf2 = warp_buf1 + block_square * row_stride;
    // compute partial sums from strided input
    // do this to increase number of loads in flight
    int i1_block = blockIdx.x;
    cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
                           warp_buf1, warp_buf2, input, dout, n1, n2, mean, invvar, rms_only);
    __syncthreads();

    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k * blockDim.y;
        int idx1 = row1 * row_stride + threadIdx.x;
        if (!rms_only) {
            acc1 += warp_buf1[idx1];
        }
        acc2 += warp_buf2[idx1];
    }
    if (!rms_only) {
        warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
    }
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();

    // sum all warps
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1 * row_stride + threadIdx.x;
            int idx2 = row2 * row_stride + threadIdx.x;
            if (!rms_only) {
                warp_buf1[idx1] += warp_buf1[idx2];
            }
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        if (!rms_only) {
            part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
        }
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}


template<typename T, typename U>
__device__ void cuApplyComputeGradGammaBeta_(
    const U *part_grad_gamma,
    const U *part_grad_beta,
    const int n1,
    const int n2,
    T *grad_gamma,
    T *grad_beta,
    bool rms_only) {
    // sum partial gradients for gamma and beta
    norms_device_kernel_tools::SharedMemory<U> shared;
    U *buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = blockDim.x;

    if (i2 >= n2) { return; }
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = warp_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr = nullptr;

    if (!rms_only) {
        part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    }
    for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
        if (!rms_only) {
            sum_beta += part_grad_beta_ptr[warp_offset * n2];
        }
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
            buf[write_idx] = sum_gamma;
            if (!rms_only) {
                buf[write_idx + nbsize3] = sum_beta;
            }
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
            const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
            sum_gamma += buf[read_idx];
            if (!rms_only) {
                sum_beta += buf[read_idx + nbsize3];
            }
        }
        __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        if (!rms_only) {
            grad_beta[i2] = sum_beta;
        }
    }
}


template<typename T, typename U, int VEC = 4>
__device__ void cuApplyComputeGradInput_(
    T *grad_input,
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const int n1,
    const int n2,
    const U *__restrict__ mean,
    const U *__restrict__ invvar,
    U epsilon,
    const T *gamma,
    bool rms_only) {
    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;
    int i1 = blockIdx.x;
    // prepare vectorized operator
    using vec_t = memory::aligned_vector<T, VEC>;
    const int n_vec_to_read = n2 / VEC;
    const int row_size = i1 * n2;
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    U c_mean;
    if (!rms_only) {
        c_mean = mean[i1];
    }
    const U c_invvar = invvar[i1];
    const T *k_input = input + row_size;
    const T *k_dout = dout + row_size;
    const vec_t *input_row_vec = reinterpret_cast<const vec_t *>(k_input);
    const vec_t *dout_row_vec = reinterpret_cast<const vec_t *>(k_dout);

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_in = input_row_vec[l];
        vec_t data_dout = dout_row_vec[l];
        if (gamma != nullptr) {
            const vec_t *gamma_vec = reinterpret_cast<const vec_t *>(gamma);
            vec_t data_gamma = gamma_vec[l];

            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                const U c_h = static_cast<U>(data_in.val[k]);
                const U c_loss = static_cast<U>(data_dout.val[k]);
                if (!rms_only) {
                    sum_loss1 += c_loss * data_gamma.val[k];
                    sum_loss2 += c_loss * data_gamma.val[k] * (c_h - c_mean) * c_invvar;
                } else {
                    sum_loss2 += c_loss * data_gamma.val[k] * (c_h) * c_invvar;
                }
            }
        } else {
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                const U c_h = static_cast<U>(data_in.val[k]);
                const U c_loss = static_cast<U>(data_dout.val[k]);
                if (!rms_only) {
                    sum_loss1 += c_loss;
                    sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
                } else {
                    sum_loss2 += c_loss * (c_h) * c_invvar;
                }
            }
        }
    }
    // deal last data
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        // computation is performed in T, X is cast to T and result is implicitly cast to T
        if (gamma != nullptr) {
            const U c_h = static_cast<U>(k_input[l_last]);
            const U c_loss = static_cast<U>(k_dout[l_last]);
            if (!rms_only) {
                sum_loss1 += c_loss * gamma[l];
                sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
            } else {
                sum_loss2 += c_loss * gamma[l] * (c_h) * c_invvar;
            }
        } else {
            const U c_h = static_cast<U>(k_input[l_last]);
            const U c_loss = static_cast<U>(k_dout[l_last]);
            if (!rms_only) {
                sum_loss1 += c_loss;
                sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            } else {
                sum_loss2 += c_loss * (c_h) * c_invvar;
            }
        }
    }

    // intra-warp reductions
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (!rms_only) {
            sum_loss1 += WARP_SHFL_XOR(sum_loss1, offset);
        }
        sum_loss2 += WARP_SHFL_XOR(sum_loss2, offset);
    }
    // inter-warp reductions
    // if (blockDim.y > 1) {
    norms_device_kernel_tools::SharedMemory<U> shared;
    U *buf = shared.getPointer();
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
            if (!rms_only) {
                buf[2 * wrt_i] = sum_loss1;
            }
            buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
            const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
            if (!rms_only) {
                sum_loss1 += buf[2 * read_i];
            }
            sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        if (!rms_only) {
            buf[2 * threadIdx.x] = sum_loss1;
        }
        buf[2 * threadIdx.x + 1] = sum_loss2;
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        if (!rms_only) {
            sum_loss1 = buf[2 * threadIdx.x];
        }
        sum_loss2 = buf[2 * threadIdx.x + 1];
    }
    // }

    // all threads now have the two sums over l
    U fH = (U) n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + row_size;
    for (int l = thread_x; l < n2; l += num_x) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * ((gamma != nullptr) ? gamma[l] : T(1));
        if (!rms_only) {
            f_grad_input -= sum_loss1;
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
            f_grad_input -= (c_h) * c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
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

    const int total_elements = n1 * n2;
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

        for (int k = 0; k < 4; ++k) {
            data_out.val[k] = data_in.val[k] + data_residual.val[k];
        }
        output_row_vec[l] = data_out;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        output_row[l_last] = input_row[l_last] + residual_row[l_last];
    }
}

template<typename T, typename U, int VEC = 4>
__device__ void cuApplyAddDrops_(   // dropmask !=nullptr
    T *__restrict__ output_vals,
    const T *__restrict__ input,
    const T *__restrict__ residual,
    const int n1, const int n2,
    uint8_t *__restrict__ drop_mask,
    U drop_keep_rate,
    const PhiloxCudaState philox_args) {

    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;
    int i1 = blockIdx.x;

    // prepare rand seed and state
    auto seeds = at::cuda::philox::unpack(philox_args);
    curandStatePhilox4_32_10_t state;
    int rand_idx = i1 * num_x + thread_x;
    curand_init(std::get<0>(seeds), rand_idx, std::get<1>(seeds), &state);
    U pinv = U(1) / drop_keep_rate;

    const int total_elements = n1 * n2;

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
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_in = input_row_vec[l];
        vec_t_uint8 data_drop_mask;
        vec_t data_residual;
        if (residual != nullptr) { data_residual = residual_row_vec[l]; }
        vec_t data_out;

        float4 rand;
        rand = curand_uniform4(&state);
        rand.x = rand.x <= drop_keep_rate;
        rand.y = rand.y <= drop_keep_rate;
        rand.z = rand.z <= drop_keep_rate;
        rand.w = rand.w <= drop_keep_rate;

        for (int k = 0; k < 4; ++k) {
            data_drop_mask.val[k] = (uint8_t)(&rand.x)[k];
            U curr_val = static_cast<T>(static_cast<U>(data_in.val[k]) * U((&rand.x)[k]) * pinv);
            if (residual != nullptr) {
                curr_val += static_cast<U>(data_residual.val[k]);
            }
            data_out.val[k] = curr_val;
        }
        drop_mask_row_vec[l] = data_drop_mask;
        output_row_vec[l] = data_out;
    }

    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        float4 rand;
        rand = curand_uniform4(&state);
        rand.x = rand.x <= drop_keep_rate;
        rand.y = rand.y <= drop_keep_rate;
        rand.z = rand.z <= drop_keep_rate;
        rand.w = rand.w <= drop_keep_rate;

        int simple_idx = l_last - l * VEC;  // assert simple_idx dur 0~3
        drop_mask_row[l_last] = (uint8_t)(&rand.x)[simple_idx];
        U curr_val = static_cast<T>(static_cast<U>(input_row[l_last]) * U((&rand.x)[simple_idx]) * pinv);
        if (residual != nullptr) { curr_val += static_cast<U>(residual_row[l_last]); }
        output_row[l_last] = curr_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// add drop backward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename U, int VEC = 4>
__device__ void cuApplyAddGradient_(T *grad_in, T *residual_grads,
                                    const T *grad_output, const int n1, const int n2) {
    // dim3 block_dims(warp_size, 4, 1);  // MI100 wavefront/warp = 64
    // const dim3 grid_dims(n1);
    const int num_x = blockDim.x * blockDim.y;
    const int thread_x = threadIdx.x + threadIdx.y * blockDim.x;

    int i1 = blockIdx.x;
    const int row_size = i1 * n2;
    const T *grad_output_row = grad_output + row_size;
    T *grad_in_row = grad_in + row_size;
    T *residual_grads_row = residual_grads + row_size;

    // prepare vectorized operator
    using vec_t = memory::aligned_vector<T, VEC>;
    const int n_vec_to_read = n2 / VEC;
    const vec_t *grad_output_row_vec = reinterpret_cast<const vec_t *>(grad_output_row);
    vec_t *grad_in_row_vec = reinterpret_cast<vec_t *>(grad_in_row);
    vec_t *residual_grads_row_vec = reinterpret_cast<vec_t *>(residual_grads_row);

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_grad_output = grad_output_row_vec[l];
        vec_t data_residual_grads, data_grad_in;

        for (int k = 0; k < 4; ++k) {
            T grad_out_val = data_grad_output.val[k];
            data_residual_grads.val[k] = grad_out_val;
            data_grad_in.val[k] = grad_out_val;
        }
        residual_grads_row_vec[l] = data_residual_grads;
        grad_in_row_vec[l] = data_grad_in;
    }
    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        T grad_out_val = grad_output_row[l_last];
        residual_grads_row[l_last] = grad_out_val;
        grad_in_row[l_last] = grad_out_val;
    }
}

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
    if (residual_grads != nullptr) { residual_grads_row = residual_grads + row_size; }

    // prepare vectorized operator
    using vec_t = memory::aligned_vector<T, VEC>;
    using vec_t_uint8 = memory::aligned_vector<uint8_t, VEC>;
    const int n_vec_to_read = n2 / VEC;

    const vec_t *grad_output_row_vec = reinterpret_cast<const vec_t *>(grad_output_row);
    const vec_t_uint8 *drop_mask_row_vec = reinterpret_cast<const vec_t_uint8 *>(drop_mask_row);
    vec_t *grad_in_row_vec = reinterpret_cast<vec_t *>(grad_in_row);
    vec_t *residual_grads_row_vec;
    if (residual_grads != nullptr) { residual_grads_row_vec = reinterpret_cast<vec_t *>(residual_grads_row); }

    int l = thread_x;
    for (; l < n_vec_to_read; l += num_x) {
        vec_t data_grad_output = grad_output_row_vec[l];
        vec_t_uint8 data_drop_mask = drop_mask_row_vec[l];
        vec_t data_residual_grads, data_grad_in;

        for (int k = 0; k < 4; ++k) {
            T grad_out_val = data_grad_output.val[k];
            if (residual_grads != nullptr) { data_residual_grads.val[k] = grad_out_val; }
            grad_out_val *= static_cast<T>(static_cast<U>(data_drop_mask.val[k]) * pinv);
            data_grad_in.val[k] = grad_out_val;
        }
        if (residual_grads != nullptr) { residual_grads_row_vec[l] = data_residual_grads; }
        grad_in_row_vec[l] = data_grad_in;
    }

    for (int l_last = l * VEC; l_last < n2; ++l_last) {
        T grad_out_val = grad_output_row[l_last];
        if (residual_grads != nullptr) { residual_grads_row[l_last] = grad_out_val; }
        grad_out_val *= static_cast<T>(static_cast<U>(drop_mask_row[l_last]) * pinv);
        grad_in_row[l_last] = grad_out_val;
    }
}

} // add_drops_device_kernel

} // native
} // at