#include "reduce.h"

using namespace std;

namespace at{
namespace native{
constexpr int vec_size = 4;

template <typename scalar_t>
__global__ void GammaBackward(int M,int N,const scalar_t* dY,const scalar_t* X,const float* rstd,scalar_t* dg)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      const int index = i * N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * rstd[i];
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1,16);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
        dg[j] = sum1;
    }
  }
}


template <typename scalar_t,typename T_ACC>
__global__ void g_backward_kernel_part_a(int M,int N,const scalar_t* dY,const scalar_t* X,const float* rstd,T_ACC* dg)
{
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
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
      dg_sum += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * rstd[i2];
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1,16);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
        dg[j*gdimy+bidy] = sum1;
    }
  }
}

template <typename scalar_t,typename T_ACC>
__global__ void g_backward_kernel_part_b(const T_ACC* part_grad_gamma,scalar_t* dg)
{
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  T_ACC gamma=part_grad_gamma[idx];
  gamma=WarpReduceSum(gamma);
  if(threadIdx.x==0){
    dg[blockIdx.x]=gamma;
  }
}

template <typename scalar_t>
void Gamma_Backward(int length,int N,scalar_t* dY_data,scalar_t* X_data,float* rstd_data,scalar_t* dgamma_data)
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
      g_backward_kernel_part_a<<<dim3(B,part_size), dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(M,N,dY_data,X_data,rstd_data,pd_gamma);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      g_backward_kernel_part_b<<<N,part_size>>>(pd_gamma,dgamma_data);
    }
    else{
        GammaBackward<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(M,N,dY_data,X_data,rstd_data,dgamma_data);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}//namespace native
}//namesapace at

#define RMSNORMBACKWARD_VEC_2048  \
  using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC b;\
  __shared__ T_ACC c;\
  const int i = blockIdx.x;\
  const int j = threadIdx.x;\
  const int index = i * N + j*VEC;\
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;\
  T_ACC sum1 = 0;\
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
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
  float trstd = rstd[i];\
  if (threadIdx.x == 0) {\
    b = - sum1 * trstd * trstd * trstd * s;\
  }\
  __syncthreads();

#define RMSNORMBACKWARD_VEC_4096  \
   using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC b;\
  const int i = blockIdx.x;\
  const int j = threadIdx.x;\
  const int index = i * N + j*VEC;\
  const int index_2 = i * N + j*VEC+2048;\
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;\
  T_ACC sum1 = 0;\
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
  }\
  if (j < tcol-512) {\
    tx_2 = *(VecTpye*)(X+index_2);\
    tdy_2 = *(VecTpye*)(dY+index_2);\
    tgamma_2 = *(VecTpye*)(gamma+j*VEC+2048);\
    for(int i=0;i<VEC;i++){\
      sum1+=static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(px_2[i])*pgamma_2[i];\
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  float trstd=rstd[i];\
  if (threadIdx.x == 0) {\
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
    b = - sum1 * trstd * trstd * trstd * s;\
  }\
  __syncthreads();


#define RMSNORMBACKWARD_VEC  \
  using T_ACC = acc_type<scalar_t, true>;\
  __shared__ T_ACC ds_shared[8];\
  __shared__ T_ACC b;\
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
    }\
  }\
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);\
  float trstd=rstd[i];\
  if (threadIdx.x == 0) {\
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);\
    b = - sum1 * trstd * trstd * trstd * s;\
  }\
  __syncthreads();


int get_layernorm_blocksize(int cols);
