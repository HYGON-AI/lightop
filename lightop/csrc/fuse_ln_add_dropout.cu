#include "layernorm.h"

namespace at{
namespace native{

#define VEC 4
typedef int IndexType ;
namespace {
  constexpr int kCUDANumThreads = 256;
  constexpr int kCUDABlockReduceNumThreads=512;
}

template <typename scalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_dropout_kernel_val(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
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
    scalar_t temp= input[index]+residual[index];
    ret[index]=temp;
    val = welford_op.reduce(val, static_cast<T_ACC>(temp));
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
    ret[index] = (static_cast<T_ACC>(ret[index]) - static_cast<T_ACC>(tmean))*static_cast<T_ACC>(trstd) 
                 * static_cast<T_ACC>(gamma[j]) + static_cast<T_ACC>(beta[j]);
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_dropout_kernel_train(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retdrop_data,scalar_t* mean,
                                            scalar_t* rstd,scalar_t* gamma,scalar_t* beta,scalar_t* mask,IndexType cols,
                                            IndexType totalElements,T_ACC p,scalar_t eps,PhiloxCudaState philox_args,IndexType rand_offset)
{
  //dropout+add
  IndexType i=blockIdx.x;
  IndexType tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  T_ACC pinv = T_ACC(1)/p;
  auto seeds = at::cuda::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t residual4[VEC];
  scalar_t mask4[VEC];
  for(int off=threadIdx.x;off<tcol;off+=blockDim.x){
    IndexType idx = i * tcol + off;
    hiprand_init(std::get<0>(seeds),idx%rand_offset,std::get<1>(seeds)+idx/rand_offset*VEC,&state);
    idx*=VEC;
    *(float4*)rand = hiprand_uniform4(&state);
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      mask4[ii] = (rand[ii]<p)*pinv;
      input4[ii] = input4[ii]*mask4[ii]+residual4[ii];
    }
    *(LoadT*)(retdrop_data+idx) = *(LoadT*)input4;
    *(LoadT*)(mask+idx) = *(LoadT*)mask4;
  }
  __syncthreads();
  // layernorm get mean and rstd
  using WelfordType = WelfordData<T_ACC, IndexType, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, IndexType, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[kCUDABlockReduceNumThreads/C10_WARP_SIZE];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  for (IndexType j = threadIdx.x; j < cols; j += blockDim.x) {
    IndexType index = i * cols + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(retdrop_data[index]));
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
    ret[index] = (static_cast<T_ACC>(retdrop_data[index]) - static_cast<T_ACC>(tmean))*static_cast<T_ACC>(trstd) 
                 * static_cast<T_ACC>(gamma[j]) + static_cast<T_ACC>(beta[j]);
  }
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
fused_ln_add_dropout_forward(const Tensor& self,const Tensor& other,const Tensor&weight,const Tensor& bias,
                             double ln_epsilon, double p, bool training)
{
    AT_ASSERTM(self.sizes() == other.sizes(), "x must be the same size as y");
    AT_ASSERTM(p >=0.0f&&p<1.0f, "p mast be between 0 and 1");
    AT_ASSERTM(self.is_cuda()&&other.is_cuda(), "x and y must on dcu");
    const IndexType cols=self.sizes().back();
    AT_ASSERT(cols>=512,"cols must >= 512");
    const IndexType nelem = self.numel();
    const IndexType rows=nelem/cols;
    Tensor ret = at::empty_like(self);
    Tensor retdrop;
    Tensor mean = at::empty({rows},self.options());
    Tensor rstd = at::empty({rows},self.options());
    PhiloxCudaState rng_engine_inputs;
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_ln_dropout_forward",
      [&] {
        p=1-p;
        using T_ACC = acc_type<scalar_t, true>;
        T_ACC pa = (T_ACC)(p);
        scalar_t eps = ln_epsilon;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* other_data =other.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* mean_data=mean.data_ptr<scalar_t>();
        scalar_t* rstd_data=rstd.data_ptr<scalar_t>();
        scalar_t* weight_data=weight.data_ptr<scalar_t>();
        scalar_t* bias_data=bias.data_ptr<scalar_t>();
        if(training)
        {
          retdrop = at::empty({nelem*2},self.options());
          scalar_t* retdrop_data=retdrop.data_ptr<scalar_t>();
          auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
          IndexType rand_offset=at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor*
                                    at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
          IndexType gridx=((nelem + 256 -1)/256);
          gridx=std::min(rand_offset/256, gridx);
          int64_t counter_offset = ((nelem - 1)/(1024*gridx)+1)*4;
          PhiloxCudaState rng_engine_inputs;
          {
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_cuda_state(counter_offset);
          }
          fused_ln_dropout_kernel_train<scalar_t,T_ACC><<<rows, kCUDABlockReduceNumThreads, 0, 
               at::cuda::getCurrentCUDAStream()>>>(self_data,other_data,ret_data,retdrop_data,mean_data,rstd_data,weight_data,
               bias_data,retdrop_data+nelem,cols,nelem,pa,eps,rng_engine_inputs,rand_offset);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else {
          fused_ln_dropout_kernel_val<scalar_t><<<rows, kCUDABlockReduceNumThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     self_data,other_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

      });
     return std::tuple<Tensor,Tensor,Tensor,Tensor>(ret,retdrop,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel(IndexType N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,
                                         const scalar_t* rstd, scalar_t* dX1,scalar_t* dX2,scalar_t* mask)
{
  LAYERNORMBACKWARD
  for (IndexType j = threadIdx.x; j < N; j += blockDim.x) {
    const IndexType index = i * N + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    scalar_t temp = static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(dY[index]) * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
    dX2[index] = temp;
    dX1[index] = temp*mask[index];
  }
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
fused_ln_add_dropout_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  IndexType N=grad.sizes().back();
  IndexType M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX1 = at::empty_like(*pgrad);
  Tensor dX2 = at::empty_like(dX1);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  constexpr int kThreadX = kColwiseReduceTileSize;
  constexpr int kThreadY = kColwiseReduceTileSize;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "fused_ln_dropout_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      scalar_t* mean_data = mean.data_ptr<scalar_t>();
      scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
      scalar_t* dX1_data = dX1.data_ptr<scalar_t>();
      scalar_t* dX2_data = dX2.data_ptr<scalar_t>();
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
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX1_data,dX2_data,X_data+grad.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor,Tensor>(dX1,dX2,dgamma,dbeta);
}
class fuselnAddDropoutFunction : public torch::autograd::Function<fuselnAddDropoutFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& y,
    const torch::autograd::Variable& weight,
    const torch::autograd::Variable& bias,
    double eps,double rate,bool training) {
    auto ret = fused_ln_add_dropout_forward(x,y, weight,bias,eps,rate,training);
    ctx->save_for_backward({std::get<1>(ret),std::get<2>(ret),std::get<3>(ret),weight});
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
    auto ret = fused_ln_add_dropout_backward(gradOutput[0],X,mean,rstd,gamma);
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),std::get<3>(ret),
            torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};
Tensor fuse_ln_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,
                                            const Tensor& bias,double eps,double rate,bool training) {
   auto result = fuselnAddDropoutFunction::apply(x,y, weight,bias,eps,rate,training);
   return result[0];
}
}
}


