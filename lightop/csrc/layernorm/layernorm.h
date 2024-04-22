#include "reduce.h"

using namespace std;

namespace at{
namespace native{
constexpr int vec_size = 4;
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
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
};

template <typename scalar_t>
__global__ void GammaBetaBackward(int M,int N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
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


template <typename scalar_t,typename T_ACC>
__global__ void gb_backward_kernel_part_a(int M,int N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,T_ACC* dg,T_ACC* db)
{
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  const int bidy=blockIdx.y;
  const int gdimy=gridDim.y;
  int col_block=(M-1)/gdimy+1;
  int off=col_block*blockIdx.y;
  if(bidy==gdimy-1){
    col_block=M-off;
  }
  if (j < N) {
    for (int i = threadIdx.y ; i < col_block; i += blockDim.y) {
      const int i2=i+off;
      const int index = i2* N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i2])) * static_cast<T_ACC>(rstd[i2]);
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
      if (dg != nullptr) {
        dg[j*gdimy+bidy] = sum1;
      }
      if (db != nullptr) {
        db[j*gdimy+bidy] = sum2;
      }
    }
  }
}

template <typename scalar_t,typename T_ACC>
__global__ void gb_backward_kernel_part_b(const T_ACC* part_grad_gamma,const T_ACC* part_grad_beta,scalar_t* dg,scalar_t* db)
{
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  T_ACC gamma=part_grad_gamma[idx];
  T_ACC beta=part_grad_beta[idx];
  gamma=WarpReduceSum(gamma);
  beta=WarpReduceSum(beta);
  if(threadIdx.x==0){
    dg[blockIdx.x]=gamma;
    db[blockIdx.x]=beta;
  }
}

template <typename scalar_t>
void GammaBeta_Backward(int length,int N,scalar_t* dY_data,scalar_t* X_data,scalar_t* mean_data,scalar_t* rstd_data,scalar_t* dgamma_data,scalar_t* dbeta_data)
{
    using T_ACC = acc_type<scalar_t, true>;
    auto stream = at::cuda::getCurrentCUDAStream();
    int M=length/N;
    const int B =(N - 1) / kColwiseReduceTileSize + 1;
    if(N<=512&&M>=4096){
      int part_size=64;
      c10::TensorOptions options(DeviceType::CUDA);
      options.dtype(ScalarType::Float);
      Tensor part_grad_gamma = at::empty({N,part_size}, options);
      Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);
      T_ACC *pd_gamma = part_grad_gamma.template data_ptr<T_ACC>();
      T_ACC *pd_beta = part_grad_beta.template data_ptr<T_ACC>();
      gb_backward_kernel_part_a<<<dim3(B,part_size), dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
          M,N,dY_data,X_data,mean_data,rstd_data,pd_gamma,pd_beta);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      gb_backward_kernel_part_b<<<N,part_size>>>(pd_gamma,pd_beta,dgamma_data,dbeta_data);
    }
    else{
        GammaBetaBackward<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
                        M,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}//namespace native
}//namesapace at


#define LAYERNORMBACKWARD_VEC_2048  \
  using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC db_shared[8];\
  __shared__ T_ACC b;\
  __shared__ T_ACC c;\
  const int i = blockIdx.x;\
  const int j = threadIdx.x;\
  const int index = i * N + j*VEC;\
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;\
  T_ACC sum1 = 0;\
  T_ACC sum2 = 0;\
  int tcol=N/VEC;\
  VecTpye tx;\
  VecTpye tdy;\
  VecTpye tdx;\
  VecTpye tgamma;\
  scalar_t* px=(scalar_t*)(&tx);\
  scalar_t* pdy=(scalar_t*)(&tdy);\
  scalar_t* pdx=(scalar_t*)(&tdx);\
  scalar_t* pgamma=(scalar_t*)(&tgamma);\
  if (j < tcol) {\
    tx = *(VecTpye*)(X+index);\
    tdy = *(VecTpye*)(dY+index);\
    tgamma = *(VecTpye*)(gamma+j*VEC);\
    for(int i=0;i<VEC;i++){\
      sum1+=static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(px[i])*pgamma[i];\
      sum2+=static_cast<T_ACC>(pdy[i])*pgamma[i];\
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);\
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
  T_ACC trstd = static_cast<T_ACC>(rstd[i]);\
  if (threadIdx.x == 0) {\
    T_ACC tmean = static_cast<T_ACC>(mean[i]);\
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;\
    b = a;\
    c = -a * tmean - sum2 * trstd * s;\
  }\
  __syncthreads();

#define LAYERNORMBACKWARD_VEC_4096  \
   using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC db_shared[8];\
  __shared__ T_ACC b;\
  __shared__ T_ACC c;\
  const int i = blockIdx.x;\
  const int j = threadIdx.x;\
  const int index = i * N + j*VEC;\
  const int index_2 = i * N + j*VEC+2048;\
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;\
  T_ACC sum1 = 0;\
  T_ACC sum2 = 0;\
  int tcol=N/VEC;\
  VecTpye tx;\
  VecTpye tdy;\
  VecTpye tdx;\
  VecTpye tgamma;\
  VecTpye tx_2;\
  VecTpye tdy_2;\
  VecTpye tgamma_2;\
  scalar_t* px=(scalar_t*)(&tx);\
  scalar_t* pdy=(scalar_t*)(&tdy);\
  scalar_t* pdx=(scalar_t*)(&tdx);\
  scalar_t* pgamma=(scalar_t*)(&tgamma);\
  scalar_t* px_2=(scalar_t*)(&tx_2);\
  scalar_t* pdy_2=(scalar_t*)(&tdy_2);\
  scalar_t* pgamma_2=(scalar_t*)(&tgamma_2);\
  tx = *(VecTpye*)(X+index);\
  tdy = *(VecTpye*)(dY+index);\
  tgamma = *(VecTpye*)(gamma+j*VEC);\
  for(int i=0;i<VEC;i++){\
    sum1+=static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(px[i])*pgamma[i];\
    sum2+=static_cast<T_ACC>(pdy[i])*pgamma[i];\
  }\
  if (j < tcol-512) {\
    tx_2 = *(VecTpye*)(X+index_2);\
    tdy_2 = *(VecTpye*)(dY+index_2);\
    tgamma_2 = *(VecTpye*)(gamma+j*VEC+2048);\
    for(int i=0;i<VEC;i++){\
      sum1+=static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(px_2[i])*pgamma_2[i];\
      sum2+=static_cast<T_ACC>(pdy_2[i])*pgamma_2[i];\
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);\
  T_ACC trstd=static_cast<T_ACC>(rstd[i]);\
  if (threadIdx.x == 0) {\
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
    T_ACC tmean = static_cast<T_ACC>(mean[i]);\
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;\
    b = a;\
    c = -a * tmean - sum2 * trstd * s;\
  }\
  __syncthreads();


#define LAYERNORMBACKWARD_VEC  \
  using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC db_shared[8];\
  __shared__ T_ACC b;\
  __shared__ T_ACC c;\
  const int i = blockIdx.x;\
  const int j = threadIdx.x;\
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;\
  T_ACC sum1 = 0;\
  T_ACC sum2 = 0;\
  int tcol=N/VEC;\
  VecTpye tx;\
  VecTpye tdy;\
  VecTpye tdx;\
  VecTpye tgamma;\
  scalar_t* px=(scalar_t*)(&tx);\
  scalar_t* pdy=(scalar_t*)(&tdy);\
  scalar_t* pdx=(scalar_t*)(&tdx);\
  scalar_t* pgamma=(scalar_t*)(&tgamma);\
  for(int off=j;off<tcol;off+=blockDim.x){\
    const int index = i * N + off*VEC;\
    tx = *(VecTpye*)(X+index);\
    tdy = *(VecTpye*)(dY+index);\
    tgamma = *(VecTpye*)(gamma+off*VEC); \
    for(int i=0;i<VEC;i++){\
      sum1+=static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(px[i])*pgamma[i];\
      sum2+=static_cast<T_ACC>(pdy[i])*pgamma[i];\
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);\
  T_ACC trstd=static_cast<T_ACC>(rstd[i]);\
  if (threadIdx.x == 0) {\
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
    T_ACC tmean = static_cast<T_ACC>(mean[i]);\
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;\
    b = a;\
    c = -a * tmean - sum2 * trstd * s;\
  }\
  __syncthreads();


int get_layernorm_blocksize(int cols);
