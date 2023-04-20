#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <THC/THCGeneral.h>
#include <torch/extension.h>
#include "layernorm.h"

using namespace std;

namespace {
    constexpr int kVec = 4;
    constexpr int kColwiseReduceTileSize = 32;
    constexpr int kCUDANumThreads = 256;
    constexpr int kCUDABlockReduceNumThreads=512;
}

namespace at{
namespace native{
template <typename index_t, typename scalar_t, typename acc_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_droppath_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd, float* mask,
                                            scalar_t* gamma,scalar_t* beta,PhiloxCudaState philox_args,index_t cols,acc_t keep_prob,scalar_t eps,bool scale_by_keep)
{
  index_t block_id = blockIdx.y*gridDim.x + blockIdx.x;
  using scalar_cuda_t = acc_type<scalar_t, true>;
  using welford_t = WelfordData<scalar_cuda_t, index_t, scalar_cuda_t>;
  using welfordop_t = WelfordOps<scalar_cuda_t, scalar_cuda_t, index_t, scalar_cuda_t, thrust::pair<scalar_cuda_t, scalar_cuda_t>>;
  __shared__ typename std::aligned_storage<sizeof(welford_t), alignof(welford_t)>::type val_shared[kCUDABlockReduceNumThreads/C10_WARP_SIZE];

  welford_t* val_shared_ptr = reinterpret_cast<welford_t*>(val_shared);
  welfordop_t welford_op;
  welford_t val;

  for (index_t j = threadIdx.x; j < cols; j += blockDim.x) {
    index_t thread_id = block_id * cols + j;
    val = welford_op.reduce(val, static_cast<scalar_cuda_t>(input[thread_id]));
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);

  __shared__ scalar_cuda_t s_mean;
  __shared__ scalar_cuda_t s_rstd;
  if (threadIdx.x == 0) {
    thrust::tie(s_rstd, s_mean) = welford_op.project(val);
    mean[block_id] = s_mean;
    s_rstd=c10::cuda::compat::rsqrt(s_rstd + static_cast<scalar_cuda_t>(eps));
    rstd[block_id] = s_rstd;
  }
  __syncthreads();
  
  int mask_id = blockIdx.y;
  curandStatePhilox4_32_10_t state;
  auto seeds = at::cuda::philox::unpack(philox_args);
  curand_init(std::get<0>(seeds),mask_id,std::get<1>(seeds),&state);
  
  float rand = curand_uniform(&state);
  float tmask = (rand < keep_prob);
  tmask = scale_by_keep ? tmask / keep_prob : tmask;
  
  const int tail_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tail_idx==0) mask[mask_id]=tmask;
  
  scalar_t tmean=static_cast<scalar_t>(s_mean);
  scalar_t trstd=static_cast<scalar_t>(s_rstd);
  scalar_cuda_t tinput, tgamma, tbeta;
  for (index_t j = threadIdx.x; j < cols; j += blockDim.x) {
    index_t thread_id = block_id * cols + j;
    scalar_cuda_t cmean=static_cast<scalar_cuda_t>(tmean);
    scalar_cuda_t crstd=static_cast<scalar_cuda_t>(trstd);
    tinput = static_cast<scalar_cuda_t>(input[thread_id]);
    tgamma = static_cast<scalar_cuda_t>(gamma[j]);
    tbeta = static_cast<scalar_cuda_t>(beta[j]);
    scalar_t tres = static_cast<scalar_t>(((tinput - cmean)*crstd * tgamma + tbeta));
    ret[thread_id] = tres * static_cast<scalar_t>(tmask); // drop output
  }
}

extern std::tuple<Tensor,Tensor,Tensor>
layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon);

std::tuple<Tensor,Tensor,Tensor,Tensor>
layernorm_droppath_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon, double keep_prob, bool scale_by_keep,bool training)
{
  const int batchsize = self.sizes().front();
  const int cols = self.sizes().back();
  const int rows = self.numel() / cols;
  const int grid_rows = rows / batchsize;

  Tensor ret, mean, rstd, mask;
  if (keep_prob == 1 || !training) {
    std::tie(ret, mean, rstd) = layernorm_forward(self, weight, bias, ln_epsilon);
  } else {
    ret = at::empty_like(self);
    mean = at::empty({rows},self.options());
    rstd = at::empty({rows},self.options());
    mask = at::empty({batchsize},self.options().dtype(at::kFloat));
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
    PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(4);
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "layernorm_forward",
      [&] {
        using acc_t = acc_type<scalar_t, true>;
        scalar_t eps = ln_epsilon;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data = ret.data_ptr<scalar_t>();
        scalar_t* mean_data = mean.data_ptr<scalar_t>();
        scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
        scalar_t* weight_data = weight.data_ptr<scalar_t>();
        scalar_t* bias_data = bias.data_ptr<scalar_t>();
        float* mask_data = mask.data_ptr<float>();
        acc_t kp = static_cast<acc_t>(keep_prob);
        layernorm_droppath_forward_kernel<int, scalar_t, acc_t><<<dim3(grid_rows, batchsize), kCUDABlockReduceNumThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                            self_data, ret_data, mean_data, rstd_data, mask_data,
                            weight_data, bias_data, rng_engine_inputs, cols, keep_prob, eps, scale_by_keep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }
  return std::tuple<Tensor,Tensor,Tensor,Tensor>(ret,mean,rstd,mask);
}

template <typename scalar_t>
__global__ void gb_with_drop_backward_simple_kernel(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                                   const scalar_t* rstd, const float* mask, scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < cols) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    const int M = batch*rows;
    for (int i = 0; i < M; ++i) {
      int bc = i / rows;
      int index = i * cols + j;
      scalar_t mask_dY = dY[index]*static_cast<scalar_t>(mask[bc]);
      sum1 += dg == nullptr ? T_ACC(0) : static_cast<T_ACC>(mask_dY) *
                            (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
      sum2 += mask_dY;
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
__global__ void gb_with_drop_backward_kernel(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,const float* mask,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (j < cols) {
    const int M = batch*rows;
    for (int i = threadIdx.y; i < M; i += blockDim.y * 2) {
      const int i1 = i;
      const int bc1 = i1 / rows;
      const int i2 = i + blockDim.y;
      const int bc2 = i2 / rows;
      const int index1 = i1 * cols + j;
      const int index2 = i2 * cols + j;
      scalar_t mask_dY1 = dY[index1]*static_cast<scalar_t>(mask[bc1]);
      dg_sum1 += dg == nullptr ? T_ACC(0)
                              : static_cast<T_ACC>(mask_dY1) *
              (static_cast<T_ACC>(X[index1]) - static_cast<T_ACC>(mean[i1])) *
              static_cast<T_ACC>(rstd[i1]);
      db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(mask_dY1);
      if (i2 < M) {
        scalar_t mask_dY2 = dY[index2]*static_cast<scalar_t>(mask[bc2]);
        dg_sum2 += dg == nullptr ? T_ACC(0)
                                : static_cast<T_ACC>(mask_dY2) *
                (static_cast<T_ACC>(X[index2]) - static_cast<T_ACC>(mean[i2])) *
                static_cast<T_ACC>(rstd[i2]);
        db_sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(mask_dY2);
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
  sum1 = WarpReduceSum<16>(sum1);
  sum2 = WarpReduceSum<16>(sum2);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < cols) {
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
  sum1 = WarpReduceSum<16>(sum1);
  sum2 = WarpReduceSum<16>(sum2);
  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < cols) {
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
__global__ void layernorm_droppath_backward_kernel(int cols,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,
                                         const scalar_t* rstd, const float* mask, scalar_t* dX)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int bc = blockIdx.y;
  const int i = blockIdx.x + gridDim.x*bc;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    const int index = i * cols + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    scalar_t mask_dY = dY[index]*static_cast<scalar_t>(mask[bc]);
    sum1 += static_cast<T_ACC>(mask_dY) * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += static_cast<T_ACC>(mask_dY) * gamma_v;
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  
  __shared__ T_ACC b;
  __shared__ T_ACC c;
  if (threadIdx.x == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(cols);
    T_ACC tmean = static_cast<T_ACC>(mean[i]);
    T_ACC trstd = static_cast<T_ACC>(rstd[i]);
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;
    b = a;
    c = -a * tmean - sum2 * trstd * s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    const int index = i * cols + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    scalar_t mask_dY = dY[index]*static_cast<scalar_t>(mask[bc]);
    dX[index] = static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(mask_dY) * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
  }
}

extern std::tuple<Tensor,Tensor,Tensor>
layernorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma);

std::tuple<Tensor,Tensor,Tensor>
layernorm_droppath_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma,const Tensor& mask, double keep_prob, bool training)
{
  Tensor dX, dgamma, dbeta;
  if (keep_prob == 1 || !training) {
    std::tie(dX, dgamma, dbeta) = layernorm_backward(grad, X,mean,rstd,gamma);
  } else {
    const int batchsize = grad.sizes().front();
    const int cols = grad.sizes().back();
    const int rows = grad.numel() / cols;
    const int grid_rows = rows / batchsize;
    auto pgrad = grad.expect_contiguous();
    dX = at::empty_like(*pgrad);
    dgamma = at::empty_like(gamma);
    dbeta = at::empty_like(gamma);
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
        float* mask_data = mask.data_ptr<float>();
        if (rows < 512) {
          const int B = (cols + kCUDANumThreads - 1) / kCUDANumThreads;
          gb_with_drop_backward_simple_kernel<scalar_t><<<B, kCUDANumThreads, 0, cuda_stream>>>(
                batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else{
          const int B =(cols + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
          gb_with_drop_backward_kernel<scalar_t><<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        layernorm_droppath_backward_kernel<scalar_t><<<dim3(grid_rows, batchsize), kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
                    cols, dY_data, X_data,gamma_data,mean_data,rstd_data,mask_data,dX_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

class fuseLayerNormDropPathFunction : public torch::autograd::Function<fuseLayerNormDropPathFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& weight,
    const torch::autograd::Variable& bias,
    double eps, double keep_prob, bool scale_by_keep, bool training) {
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["training"] = training;
    auto ret = layernorm_droppath_forward(x, weight,bias,eps, keep_prob, scale_by_keep, training);
    ctx->save_for_backward({x,std::get<1>(ret),std::get<2>(ret),weight,std::get<3>(ret)});
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
    auto mask = saved[4];
    double keep_prob = ctx->saved_data["keep_prob"].toDouble();
    bool training = ctx->saved_data["training"].toBool();
    auto ret = layernorm_droppath_backward(gradOutput[0],X,mean,rstd,gamma,mask,keep_prob,training);
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};
Tensor fuse_layernorm_droppath_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps, double keep_prob, bool scale_by_keep, bool training) {
   auto result = fuseLayerNormDropPathFunction::apply(x, weight,bias,eps,keep_prob,scale_by_keep,training);
   return result[0];
}

}
} // namespace at
