#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <THC/THCGeneral.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/autograd.h>

using namespace std;

namespace at{
namespace native{

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      combine_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct WelfordOps {
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    return res_t(acc.m2 / acc.nf, static_cast<scalar_t>(acc.mean));
  }

  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      __shfl_down(acc.mean, offset)
      , __shfl_down(acc.m2, offset)
      , __shfl_down(acc.n, offset)
      , __shfl_down(acc.nf, offset)
    };
  }
};

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val,const ReduceOp& op,int max=32) {
#pragma unroll
  for (int offset = max; offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  if (wid == 0) {
    val= shared[lid];
    val = WarpReduce(val,op,blockDim.x/C10_WARP_SIZE/2);
  }
  return val;
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val,int max=32) {
#pragma unroll
  for (int offset = max; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}


template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
if (wid == 0) {
    val= shared[lid];
    val = WarpReduceSum(val,blockDim.x/C10_WARP_SIZE/2);
  }
  return val;
}

template <typename scalar_t>
__global__ void GammaBetaBackwardSimple(int M,int N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                                   const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int i = 0; i < M; ++i) {
      #pragma clang fp contract(off)
      int index = i * N + j;
      sum1 += static_cast<T_ACC>(dY[index]) *(static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
      sum2 += dY[index];
    }
    dg[j] = sum1;
    db[j] = sum2;
  }
}

#define kColwiseReduceTileSize 32

template <typename scalar_t>
__global__ void GammaBetaBackward(int M,int N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      #pragma clang fp contract(off)
      const int index = i * N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
      db_sum += cmask_dY;
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum;
  b_shared[threadIdx.y][threadIdx.x] = db_sum;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1,16);
  sum2 = WarpReduceSum(sum2,16);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
        dg[j] = sum1;
        db[j] = sum2;
    }
  }
}
}//namespace native
}//namesapace at
#define LAYERNORMBACKWARD                                                                \
  using T_ACC = acc_type<scalar_t, true>;                                                \
  __shared__ T_ACC ds_shared[8];                                                         \
  __shared__ T_ACC db_shared[8];                                                         \
 const int i = blockIdx.x;                                                               \
  T_ACC sum1 = 0;                                                                        \
  T_ACC sum2 = 0;                                                                        \
  for (int j = threadIdx.x; j < N; j += blockDim.x) {                                    \
    const int index = i * N + j;                                                         \
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);                                  \
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;      \
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;                                     \
  }                                                                                      \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                         \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                         \
  __shared__ T_ACC b;                                                                    \
  __shared__ T_ACC c;                                                                    \
  if (threadIdx.x == 0) {                                                                \
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                    \
    T_ACC tmean = static_cast<T_ACC>(mean[i]);                                           \
    T_ACC trstd = static_cast<T_ACC>(rstd[i]);                                           \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                   \
    b = a;                                                                               \
    c = -a * tmean - sum2 * trstd * s;                                                   \
  }                                                                                      \
  __syncthreads();


int get_layernorm_blocksize(int cols);