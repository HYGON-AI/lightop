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
    constexpr int VEC=4;
}

namespace at{
namespace native{

template <typename index_t, typename scalar_t, typename acc_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_droppath_forward_kernel(scalar_t* input,scalar_t* ret,float* mean,float* rstd, scalar_t* mask,
                                            scalar_t* gamma,scalar_t* beta,PhiloxCudaState philox_args,index_t cols,acc_t keep_prob,float eps,bool scale_by_keep)
{
  int j=threadIdx.x;
  int maskID=blockIdx.y;
  int i=blockIdx.x+maskID*gridDim.x;
  int block_cols=i*cols;
  __shared__ scalar_t tmask;
  if(j==0){
    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds),maskID,std::get<1>(seeds),&state);
    float rand = curand_uniform(&state);
    if(scale_by_keep)tmask=(rand<keep_prob)/keep_prob;
    else tmask=rand<keep_prob;
  }
  __syncthreads();
  int tcol=cols/VEC;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  
    //if mask=0,we needn't calculate mean and rstd. ret always is 0.
  if(tmask==0){
    const scalar_t zeros[]={0,0,0,0};
    if(j<tcol)Y[j] = *(VecTpye*)zeros;
    if(j==0&&blockIdx.x==0)mask[maskID]=tmask;
    return;
  }
  //layernorm get mean and rstd
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  if(j<tcol){
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, static_cast<T_ACC>(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (j == 0) {
    //layernorm get mean and rstd
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      py[i]=(static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
      py[i]*=tmask;
    }
    Y[j]=ty;
  }
  if(j==0&&blockIdx.x==0)mask[maskID]=tmask;
  if(j==0){
    mean[i] = tmean;
    rstd[i] = trstd;
  }
}

std::tuple<Tensor,Tensor,Tensor> layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon,bool training);

std::tuple<Tensor,Tensor,Tensor,Tensor>
layernorm_droppath_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon, double keep_prob, bool scale_by_keep,bool training)
{
  AT_ASSERTM(keep_prob >0.0f&&keep_prob<=1.0f, "keep_prob mast be between 0 and 1");
  AT_ASSERTM(self.is_cuda(), "input must on HCU");
  const int batchsize = self.sizes().front();
  const int elem_per_batch=self.numel()/batchsize;
  AT_ASSERT(elem_per_batch%4==0,"element size per batch must be 4*interger");
  const int cols = self.sizes().back();
  //last dim usually be 96,192,384
  AT_ASSERT(cols<=2048,"last dim size should be <= 2048");
  const int rows = self.numel() / cols;

  Tensor ret, mean, rstd, mask;
  if (keep_prob == 1 || !training) {
    std::tie(ret, mean, rstd) = layernorm_forward(self, weight, bias, ln_epsilon,training);
  } else {
    ret = at::empty_like(self);
    mean = at::empty({rows},self.options().dtype(at::ScalarType::Float));
    rstd = at::empty({rows},self.options().dtype(at::ScalarType::Float));
    mask = at::empty({batchsize},self.options());
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
    int rand_offset=at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor*
                              at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int gridx=((batchsize + 256 -1)/256);
    gridx=std::min(rand_offset/256, gridx);
    int64_t counter_offset = ((batchsize - 1)/(1024*gridx)+1)*4;
    PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }
    const int grid_rows = rows / batchsize;
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "layernorm_droppath_forward",
      [&] {
        using acc_t = acc_type<scalar_t, true>;
        float eps = ln_epsilon;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data = ret.data_ptr<scalar_t>();
        float* mean_data = mean.data_ptr<float>();
        float* rstd_data = rstd.data_ptr<float>();
        scalar_t* weight_data = weight.data_ptr<scalar_t>();
        scalar_t* bias_data = bias.data_ptr<scalar_t>();
        scalar_t* mask_data = mask.data_ptr<scalar_t>();
        acc_t kp = static_cast<acc_t>(keep_prob);
        layernorm_droppath_forward_kernel<int, scalar_t, acc_t><<<dim3(grid_rows,batchsize), get_layernorm_blocksize(cols/4), 0, at::cuda::getCurrentCUDAStream()>>>(
                            self_data, ret_data, mean_data, rstd_data, mask_data,weight_data, bias_data, rng_engine_inputs, cols, keep_prob, eps, scale_by_keep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }
  return std::tuple<Tensor,Tensor,Tensor,Tensor>(ret,mean,rstd,mask);
}

template <typename scalar_t>
__global__ void gb_with_drop_backward_kernel(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const float* mean,
                                  const float* rstd,const scalar_t* mask,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < cols) {
    const int M = batch*rows;
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      const int bc = i / rows;
      const int index = i * cols + j;
      const T_ACC cmask_dY = dY[index] * mask[bc];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - mean[i]) * rstd[i];
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

template <typename scalar_t,typename T_ACC>
__global__ void gb_with_drop_backward_kernel_part_a(int batch,int rows,int cols,const scalar_t* dY,const scalar_t* X,const float* mean,
                                  const float* rstd,const scalar_t* mask,T_ACC* dg,T_ACC* db)
{
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ scalar_t s_mask[256];
  const int threadid=threadIdx.y*blockDim.x+threadIdx.x;
  if(batch<=256){
    if(threadid<batch)s_mask[threadid]=mask[threadid];
    mask=s_mask;
    __syncthreads();
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  const int M = batch*rows;
  const int bidy=blockIdx.y;
  const int gdimy=gridDim.y;
  int col_block=(M-1)/gdimy+1;
  int off=col_block*blockIdx.y;
  if(bidy==gdimy-1){
    col_block=M-off;
  }
  if (j < cols) {
    for (int i = threadIdx.y ; i < col_block; i += blockDim.y) {
      const int i2=i+off;
      const int bc = i2 / rows;
      const int index = i2* cols + j;
      const T_ACC cmask_dY = dY[index] * mask[bc];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - mean[i2]) * rstd[i2];
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
        dg[j*gdimy+bidy] = sum1;
      }
      if (db != nullptr) {
        db[j*gdimy+bidy] = sum2;
      }
    }
  }
}

template <typename scalar_t>
__global__ void layernorm_droppath_backward_kernel(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,
                                         const float* rstd, const scalar_t* mask, scalar_t* dX)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC ds_shared[8];
  __shared__ T_ACC db_shared[8];
  __shared__ T_ACC b;
  __shared__ T_ACC c;
  const int maskID=blockIdx.y;;
  const int i = blockIdx.x+maskID*gridDim.x;
  const int j = threadIdx.x;
  const int index = i * N + j*VEC;
  int tcol=N/VEC;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  __shared__ scalar_t smask;
  if(j==0){
    smask=mask[maskID];
  }
  __syncthreads();
  scalar_t tmask=smask;
  if(tmask==0){
    const scalar_t zeros[]={0,0,0,0};
    if(j<tcol){
       *(VecTpye*)(dX+index) = *(VecTpye*)zeros;
    }
    return;
  }
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  T_ACC gamma_v;
  VecTpye tx;
  VecTpye tdy;
  VecTpye tdx;
  VecTpye tgamma;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* pdy=(scalar_t*)(&tdy);
  scalar_t* pdx=(scalar_t*)(&tdx);
  scalar_t* pgamma=(scalar_t*)(&tgamma);
  if (j < tcol) {
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+j*VEC);
    for(int i=0;i<VEC;i++){
      pdy[i]*=tmask;
      T_ACC mask_dY = static_cast<T_ACC>(pdy[i])*pgamma[i];
      sum1+=mask_dY*static_cast<T_ACC>(px[i]);
      sum2+=mask_dY;
    }
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);
  T_ACC trstd = rstd[i];
  if (threadIdx.x == 0) {
    T_ACC tmean = mean[i];
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;
    b = a;
    c = -a * tmean - sum2 * trstd * s;
  }
  __syncthreads();
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
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
      "layernorm_droppath_backward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
        scalar_t* X_data = X.data_ptr<scalar_t>();
        float* mean_data = mean.data_ptr<float>();
        float* rstd_data = rstd.data_ptr<float>();
        scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
        scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
        scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
        scalar_t* dX_data = dX.data_ptr<scalar_t>();
        scalar_t* mask_data = mask.data_ptr<scalar_t>();
        const int B =(cols + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
        if(cols>512||rows<4096){
          gb_with_drop_backward_kernel<scalar_t><<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
              batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else{
          int part_size=64;
          const auto part_grad_dtype = at::toAccumulateType(X.scalar_type(), true);
          Tensor part_grad_gamma = at::empty({cols,part_size}, gamma.options().dtype(part_grad_dtype));
          Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);
          T_ACC *pd_gamma = part_grad_gamma.template data_ptr<T_ACC>();
          T_ACC *pd_beta = part_grad_beta.template data_ptr<T_ACC>();
          gb_with_drop_backward_kernel_part_a<<<dim3(B,part_size), dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
              batchsize,grid_rows,cols,dY_data,X_data,mean_data,rstd_data,mask_data,pd_gamma,pd_beta);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          gb_backward_kernel_part_b<<<cols,part_size>>>(pd_gamma,pd_beta,dgamma_data,dbeta_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        
        layernorm_droppath_backward_kernel<scalar_t><<<dim3(grid_rows, batchsize), get_layernorm_blocksize(cols/4), 0, cuda_stream>>>(
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
    auto px=x.expect_contiguous();
    auto pweight=weight.expect_contiguous();
    auto pbias=bias.expect_contiguous();
    auto ret = layernorm_droppath_forward(*px, *pweight,*pbias,eps, keep_prob, scale_by_keep, training);
    ctx->save_for_backward({*px,std::get<1>(ret),std::get<2>(ret),weight,std::get<3>(ret)});
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
