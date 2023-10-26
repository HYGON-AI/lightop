#include "layernorm.h"
int get_layernorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else blocksize=128;
  return blocksize;
}

namespace at{
namespace native{

constexpr int kCUDANumThreads = 256;
constexpr int kCUDABlockReduceNumThreads=512;

template <typename scalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void layernorm_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,scalar_t eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  //layernorm get mean and rstd
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];//kCUDABlockReduceNumThreads/C10_WARP_SIZE
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  if(cols<=512){
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
      ret[index] = (t_input - static_cast<T_ACC>(s_mean))*static_cast<T_ACC>(s_rstd)  * static_cast<T_ACC>(gamma[threadidx]) + static_cast<T_ACC>(beta[threadidx]);
    }
  }
  else{
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
      ret[index] = (static_cast<T_ACC>(input[index]) - static_cast<T_ACC>(tmean))*static_cast<T_ACC>(trstd) 
                  * static_cast<T_ACC>(gamma[j]) + static_cast<T_ACC>(beta[j]);
    }
  }
}

//to avoid windows SFINAE errors
template <typename scalar_t, typename T_ACC>
__global__ __inline__ void vectorized_layer_norm_kernel(
  const int N,
  scalar_t eps,
  const  scalar_t* __restrict__ X,
  const  scalar_t* gamma,
  const  scalar_t* beta,
  scalar_t* mean,
  scalar_t* rstd,
  scalar_t* Y){
    vectorized_layer_norm_kernel_impl<scalar_t,T_ACC>(N, eps, X, gamma, beta, mean, rstd, Y);
  }

std::tuple<Tensor,Tensor,Tensor>
layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon)
{
    AT_ASSERTM(self.is_cuda(), "x must on dcu");
    const int cols=self.sizes().back();
    AT_ASSERT(cols>64,"cols must > 64");//torch layernorm don't support cols<=64 on dcu, so we too .
    const int nelem = self.numel();
    const int rows=nelem/cols;
    Tensor ret = at::empty_like(self);
    Tensor mean = at::empty({rows},self.options());
    Tensor rstd = at::empty({rows},self.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      self.scalar_type(),
      "layernorm_forward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* weight_data=weight.data_ptr<scalar_t>();
        scalar_t* bias_data=bias.data_ptr<scalar_t>();
        scalar_t  eps = ln_epsilon;
        scalar_t* mean_data=mean.data_ptr<scalar_t>();
        scalar_t* rstd_data=rstd.data_ptr<scalar_t>();
        // check if can take fast path- all tensors are properly aligned, N is less than 2^24 (to use float count),
        // N is multiple of vec_size (so that all rows are aligned if tensor is aligned)
        auto can_vectorize = [&](const scalar_t * ptr, int alignment){uint64_t addr = reinterpret_cast<uint64_t>(ptr); return addr % alignment == 0;};
        constexpr int num_vec_elems = vec_size;
        constexpr int alignment = num_vec_elems * sizeof(scalar_t);
        if ((std::is_same<scalar_t, float>::value || std::is_same<scalar_t, at::Half>::value || std::is_same<scalar_t, at::BFloat16>::value) &&
          cols <= 1ULL << std::numeric_limits<float>::digits && cols % num_vec_elems == 0 &&
          can_vectorize(self_data, alignment) && can_vectorize(ret_data, alignment))
        {
            const int warp_size = at::cuda::warp_size();
            const dim3 threads(warp_size, kCUDANumThreads / warp_size, 1);
            const dim3 blocks(rows);
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(threads.y % 2 == 0 || threads.y == 1);
            int nshared = threads.y > 1 ? threads.y * 3/2 *sizeof(T_ACC) : 0;
            vectorized_layer_norm_kernel<scalar_t,T_ACC><<<blocks, threads, nshared, at::cuda::getCurrentCUDAStream()>>>
            (cols, eps, self_data,weight_data, bias_data,mean_data, rstd_data, ret_data);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else
        {
            layernorm_forward_kernel<scalar_t><<<rows, get_layernorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      
      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,
                                         const scalar_t* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD

  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    dX[index] = static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(dY[index]) * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
  }
}

std::tuple<Tensor,Tensor,Tensor>
layernorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
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

      GammaBeta_Backward(nelem,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
      LayerNormBackward_kernel<scalar_t><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
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


