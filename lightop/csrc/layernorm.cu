#include "layernorm.h"

namespace at{
namespace native{

#define VEC 4
typedef int IndexType ;
constexpr int kColwiseReduceTileSize = 32;
constexpr int kCUDANumThreads = 256;
constexpr int kCUDABlockReduceNumThreads=512;

template <typename scalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,IndexType cols,scalar_t eps)
{
  //dropout do nothing in val mode
  IndexType i=blockIdx.x;
  // add + layernorm get mean and rstd
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, IndexType, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, IndexType, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[kCUDABlockReduceNumThreads/C10_WARP_SIZE];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;

  for (IndexType j = threadIdx.x; j < cols; j += blockDim.x) {
    IndexType index = i * cols + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(input[index]));
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);

  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  if (threadIdx.x == 0) {
    thrust::tie(s_rstd, s_mean) = welford_op.project(val);
    mean[i] = s_mean;
    s_rstd=c10::cuda::compat::rsqrt(s_rstd + static_cast<T_ACC>(eps));
    rstd[i] = s_rstd;
  }
  __syncthreads();
  //layernorm  (x-mean)*rstd*gamma+beta
  scalar_t tmean=s_mean;
  scalar_t trstd=s_rstd;
  for (IndexType j = threadIdx.x; j < cols; j += blockDim.x) {
    IndexType index = i * cols + j;
    ret[index] = (static_cast<T_ACC>(input[index]) - static_cast<T_ACC>(tmean))*static_cast<T_ACC>(trstd) 
                 * static_cast<T_ACC>(gamma[j]) + static_cast<T_ACC>(beta[j]);
  }
}


std::tuple<Tensor,Tensor,Tensor>
layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon)
{
    AT_ASSERTM(self.is_cuda(), "x must on dcu");
    const IndexType cols=self.sizes().back();
    AT_ASSERT(cols>=512,"cols must >= 512");
    const IndexType nelem = self.numel();
    const IndexType rows=nelem/cols;
    Tensor ret = at::empty_like(self);
    Tensor mean = at::empty({rows},self.options());
    Tensor rstd = at::empty({rows},self.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "layernorm_forward",
      [&] {
        scalar_t eps = ln_epsilon;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* mean_data=mean.data_ptr<scalar_t>();
        scalar_t* rstd_data=rstd.data_ptr<scalar_t>();
        scalar_t* weight_data=weight.data_ptr<scalar_t>();
        scalar_t* bias_data=bias.data_ptr<scalar_t>();
        layernorm_forward_kernel<scalar_t><<<rows, kCUDABlockReduceNumThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                   self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}

template <typename scalar_t>
__global__ void GammaBetaBackwardSimple(IndexType M,IndexType N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                                   const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  const IndexType j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (IndexType i = 0; i < M; ++i) {
      IndexType index = i * N + j;
      sum1 += dg == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]) *
                            (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
      sum2 += dY[index];
    }
    if (dg != nullptr) {
      dg[j] = sum1;
    }
    if (db != nullptr) {
      db[j] = sum2;
    }
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void GammaBetaBackward(IndexType M,IndexType N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const IndexType j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (j < N) {
    for (IndexType i = threadIdx.y; i < M; i += blockDim.y * 2) {
      const IndexType i1 = i;
      const IndexType i2 = i + blockDim.y;
      const IndexType index1 = i1 * N + j;
      const IndexType index2 = i2 * N + j;
      dg_sum1 += dg == nullptr ? T_ACC(0)
                               : static_cast<T_ACC>(dY[index1]) *
              (static_cast<T_ACC>(X[index1]) - static_cast<T_ACC>(mean[i1])) *
              static_cast<T_ACC>(rstd[i1]);
      db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]);
      if (i2 < M) {
        dg_sum2 += dg == nullptr ? T_ACC(0)
                                 : static_cast<T_ACC>(dY[index2]) *
                (static_cast<T_ACC>(X[index2]) - static_cast<T_ACC>(mean[i2])) *
                static_cast<T_ACC>(rstd[i2]);
        db_sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index2]);
      }
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1,16);
  sum2 = WarpReduceSum(sum2,16);
  if (threadIdx.x == 0) {
    const IndexType j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = WarpReduceSum(sum1,16);
  sum2 = WarpReduceSum(sum2,16);
  if (threadIdx.x == 0) {
    const IndexType j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel(IndexType N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,
                                         const scalar_t* rstd, scalar_t* dX)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const IndexType i = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (IndexType j = threadIdx.x; j < N; j += blockDim.x) {
    const IndexType index = i * N + j;
    const T_ACC gamma_v = gamma[j];
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);
  __shared__ T_ACC b;
  __shared__ T_ACC c;
  if (threadIdx.x == 0) {
    b = (sum2 * static_cast<T_ACC>(mean[i]) - sum1) * static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(rstd[i]) *static_cast<T_ACC>(rstd[i]) * s;
    c = -(b * static_cast<T_ACC>(mean[i]) + sum2 * static_cast<T_ACC>(rstd[i]) * s);
  }
  __syncthreads();
  for (IndexType j = threadIdx.x; j < N; j += blockDim.x) {
    const IndexType index = i * N + j;
    const T_ACC gamma_v = gamma[j];
    dX[index] = static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(dY[index]) * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
  }
}

std::tuple<Tensor,Tensor,Tensor>
layernorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  IndexType N=grad.sizes().back();
  IndexType M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  constexpr int kThreadX = kColwiseReduceTileSize;
  constexpr int kThreadY = kColwiseReduceTileSize / 2;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "layernorm_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      scalar_t* mean_data = mean.data_ptr<scalar_t>();
      scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      if (M < 512) {
        // For small batch size, do colwise reduce directly.
        const IndexType B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
        GammaBetaBackwardSimple<scalar_t><<<B, kCUDANumThreads, 0, cuda_stream>>>(
               M,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      else{
        const IndexType B =(N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
        GammaBetaBackward<scalar_t><<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                      M,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      LayerNormBackward_kernel<scalar_t><<<M, kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

class fuseLayerNormFunction : public torch::autograd::Function<fuseLayerNormFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& weight,
    const torch::autograd::Variable& bias,
    double eps) {
    auto ret = layernorm_forward(x, weight,bias,eps);
    ctx->save_for_backward({x,std::get<1>(ret),std::get<2>(ret),weight});
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto saved = ctx->get_saved_variables();
    auto X = saved[0];
    auto mean = saved[1];
    auto rstd = saved[2];
    auto gamma = saved[3];
    auto ret = layernorm_backward(gradOutput[0],X,mean,rstd,gamma);
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable()};
  }
};
Tensor fuse_layernorm_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps) {
   auto result = fuseLayerNormFunction::apply(x, weight,bias,eps);
   return result[0];
  }
}
}


