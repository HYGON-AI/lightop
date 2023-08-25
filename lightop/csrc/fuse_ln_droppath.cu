#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>
#include "layernorm.h"

using namespace std;

namespace {
    constexpr int kCUDANumThreads = 256;
    constexpr int kCUDABlockReduceNumThreads=512;
}

namespace at{
namespace native{

template <typename index_t, typename scalar_t, typename acc_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_droppath_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd, scalar_t* mask,
                                            scalar_t* gamma,scalar_t* beta,PhiloxCudaState philox_args,index_t cols,acc_t keep_prob,scalar_t eps,bool scale_by_keep)
{
  int threadidx=threadIdx.x;
  int maskID=blockIdx.y;
  int blockidx=blockIdx.x+maskID*gridDim.x;
  int block_cols=blockidx*cols;
  __shared__ scalar_t tmask;
  if(threadidx==0){
    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds),maskID,std::get<1>(seeds),&state);
    float rand = curand_uniform(&state);
    if(scale_by_keep)tmask=(rand<keep_prob)/keep_prob;
    else tmask=rand<keep_prob;
    if(blockIdx.x==0&&threadidx==0){
      mask[maskID]=tmask;
    }
  }
  __syncthreads();
  
  //if mask=0,we needn't calculate mean and rstd. ret always is 0.
  if(tmask==0){
    for (int j = threadidx; j < cols; j += blockDim.x) {
      int index = block_cols + j;
      ret[index] = static_cast<scalar_t>(0);
    }
    return;
  }
  //layernorm get mean and rstd
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];//kCUDABlockReduceNumThreads/C10_WARP_SIZE
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;

  for (int j = threadidx; j < cols; j += blockDim.x) {
    int index = block_cols + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(input[index]));
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  if (threadidx == 0) {
    thrust::tie(s_rstd, s_mean) = welford_op.project(val);
    mean[blockidx] = s_mean;
    s_rstd=c10::cuda::compat::rsqrt(s_rstd + static_cast<T_ACC>(eps));
    rstd[blockidx] = s_rstd;
  }
  __syncthreads();
  //layernorm  (x-mean)*rstd*gamma+beta
  scalar_t tmean=s_mean;
  scalar_t trstd=s_rstd;
  for (int j = threadidx; j < cols; j += blockDim.x) {
    int index = block_cols + j;
    scalar_t tret=((static_cast<T_ACC>(input[index]) - static_cast<T_ACC>(tmean))*static_cast<T_ACC>(trstd) 
                 * static_cast<T_ACC>(gamma[j]) + static_cast<T_ACC>(beta[j]));
    ret[index] = tret*tmask;
  }
}

template <typename index_t, typename scalar_t, typename acc_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_droppath_forward_small_col_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd, scalar_t* mask,
                                            scalar_t* gamma,scalar_t* beta,PhiloxCudaState philox_args,index_t cols,acc_t keep_prob,scalar_t eps,bool scale_by_keep)
{
  int threadidx=threadIdx.x;
  int maskID=blockIdx.y;
  int blockidx=blockIdx.x+maskID*gridDim.x;
  int block_cols=blockidx*cols;
  __shared__ scalar_t tmask;
  if(threadidx==0){
    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds),maskID,std::get<1>(seeds),&state);
    float rand = curand_uniform(&state);
    if(scale_by_keep)tmask=(rand<keep_prob)/keep_prob;
    else tmask=rand<keep_prob;
    if(threadidx==0&&blockIdx.x==0)mask[maskID]=tmask;
  }
  __syncthreads();
  //if mask=0,we needn't calculate mean and rstd. ret always is 0.
  if(tmask==0){
    if(threadidx<cols)ret[block_cols+threadidx] = static_cast<scalar_t>(0);
    return;
  }
  //layernorm get mean and rstd
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];//kCUDABlockReduceNumThreads/C10_WARP_SIZE
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int index=block_cols+threadidx;
  T_ACC t_input;
  if(threadidx<cols){
    t_input=input[index];
    val = welford_op.reduce(val, t_input);
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if(threadidx<cols){
    __shared__ T_ACC s_mean;
    __shared__ T_ACC s_rstd;
    if (threadidx == 0) {
      thrust::tie(s_rstd, s_mean) = welford_op.project(val);
      mean[blockidx] = s_mean;
      s_rstd=c10::cuda::compat::rsqrt(s_rstd + static_cast<T_ACC>(eps));
      rstd[blockidx] = s_rstd;
    }
    __syncthreads();
    //layernorm  (x-mean)*rstd*gamma+beta
    scalar_t tret=((t_input - static_cast<T_ACC>(s_mean))*static_cast<T_ACC>(s_rstd) * static_cast<T_ACC>(gamma[threadidx]) + static_cast<T_ACC>(beta[threadidx]));
    ret[index] = tret*tmask;
  }
}

std::tuple<Tensor,Tensor,Tensor> layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon);

std::tuple<Tensor,Tensor,Tensor,Tensor>
layernorm_droppath_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon, double keep_prob, bool scale_by_keep,bool training)
{
  AT_ASSERTM(keep_prob >=0.0f&&keep_prob<=1.0f, "keep_prob mast be between 0 and 1");
  AT_ASSERTM(self.is_cuda(), "input must on dcu");
  const int batchsize = self.sizes().front();
  AT_ASSERT(batchsize<=600*256,"input's first dim size is too big");
  const int elem_per_batch=self.numel()/batchsize;
  AT_ASSERT(elem_per_batch%4==0,"element size per batch must be 4*interger");
  const int cols = self.sizes().back();
  const int rows = self.numel() / cols;

  Tensor ret, mean, rstd, mask;
  if (keep_prob == 1 || !training) {
    std::tie(ret, mean, rstd) = layernorm_forward(self, weight, bias, ln_epsilon);
  } else {
    ret = at::empty_like(self);
    mean = at::empty({rows},self.options());
    rstd = at::empty({rows},self.options());
    mask = at::empty({batchsize},self.options());
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
    PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(4);
    }
    const int grid_rows = rows / batchsize;
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
        scalar_t* mask_data = mask.data_ptr<scalar_t>();
        acc_t kp = static_cast<acc_t>(keep_prob);
        if(cols>512){
           layernorm_droppath_forward_kernel<int, scalar_t, acc_t><<<dim3(grid_rows,batchsize), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                            self_data, ret_data, mean_data, rstd_data, mask_data,
                            weight_data, bias_data, rng_engine_inputs, cols, keep_prob, eps, scale_by_keep);
        }
        else{
          layernorm_droppath_forward_small_col_kernel<int, scalar_t, acc_t><<<dim3(grid_rows,batchsize), get_layernorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                            self_data, ret_data, mean_data, rstd_data, mask_data,
                            weight_data, bias_data, rng_engine_inputs, cols, keep_prob, eps, scale_by_keep);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }
  return std::tuple<Tensor,Tensor,Tensor,Tensor>(ret,mean,rstd,mask);
}

template <typename scalar_t>
__global__ void gb_with_drop_backward_simple_kernel(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                                   const scalar_t* rstd, const scalar_t* mask, scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < cols) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    const int M = batch*rows;
    for (int i = 0; i < M; ++i) {
      #pragma clang fp contract(off)
      int bc = i / rows;
      int index = i * cols + j;
      T_ACC mask_dY = dY[index]*mask[bc];
      sum1 += mask_dY * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
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
                                  const scalar_t* rstd,const scalar_t* mask,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < cols) {
    const int M = batch*rows;
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      //#pragma clang fp contract(off) // if need same output with pytorch ,then enable this line.
      const int bc = i / rows;
      const int index = i * cols + j;
      const T_ACC cmask_dY = dY[index] * mask[bc];
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
__global__ void gb_with_drop_backward_small_batch_kernel(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,const scalar_t* mask,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  __shared__ scalar_t s_mask[kColwiseReduceTileSize*kColwiseReduceTileSize];
  const int threadid=threadIdx.y*blockDim.x+threadIdx.x;
  if(threadid<batch)s_mask[threadid]=mask[threadid];
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < cols) {
    const int M = batch*rows;
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      //#pragma clang fp contract(off) // if need same output with pytorch ,then enable this line.
      const int bc = i / rows;
      const int index = i * cols + j;
      const T_ACC cmask_dY = dY[index] * s_mask[bc];
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
                                         const scalar_t* rstd, const scalar_t* mask, scalar_t* dX)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int bc = blockIdx.y;
  const int i = blockIdx.x + gridDim.x*bc;
  scalar_t tmask=mask[bc];
  if(tmask==0){
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
       const int index = i * cols + j;
       dX[index]=0;
    }
    return;
  }
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    const int index = i * cols + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    T_ACC mask_dY = dY[index]*tmask;
    sum1 += mask_dY * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += mask_dY * gamma_v;
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
    T_ACC mask_dY = dY[index]*tmask;
    dX[index] = static_cast<T_ACC>(rstd[i]) * mask_dY * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
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
    constexpr int kThreadY = kColwiseReduceTileSize;
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
        scalar_t* mask_data = mask.data_ptr<scalar_t>();
        if (rows < 512) {
          const int B = (cols + kCUDANumThreads - 1) / kCUDANumThreads;
          gb_with_drop_backward_simple_kernel<scalar_t><<<B, kCUDANumThreads, 0, cuda_stream>>>(
                batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else{
          const int B =(cols + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
          if(batchsize<=1024)
            gb_with_drop_backward_small_batch_kernel<scalar_t><<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          else
            gb_with_drop_backward_kernel<scalar_t><<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        layernorm_droppath_backward_kernel<scalar_t><<<dim3(grid_rows, batchsize), get_layernorm_blocksize(cols), 0, cuda_stream>>>(
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
