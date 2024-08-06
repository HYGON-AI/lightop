#include "layernorm.h"
int get_layernorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else if(cols>64)blocksize=128;
  else blocksize=64;
  return blocksize;
}

namespace at{
namespace native{

namespace {
  constexpr int VEC = 4;
}

template <typename scalar_t, int VEC=4>
__global__ void layernorm_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,scalar_t eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, static_cast<T_ACC>(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (threadidx == 0) {
    //layernorm get mean and rstd 
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    s_rstd=c10::cuda::compat::rsqrt(tmean + static_cast<T_ACC>(eps));
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      py[i]=(static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    Y[j]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      mean[blockidx] = tmean;
      rstd[blockidx] = trstd;
  }
}

template <typename scalar_t, int VEC=4,int blocksize=512>
__global__ void layernorm_forward_kernel_fast(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,scalar_t eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  constexpr int share_size=blocksize/C10_WARP_SIZE;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  int j=threadidx;
  if (j < tcol) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, static_cast<T_ACC>(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (threadidx == 0) {
    //layernorm get mean and rstd
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    s_rstd=c10::cuda::compat::rsqrt(tmean + static_cast<T_ACC>(eps));
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      py[i]=(static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    Y[j]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      mean[blockidx] = tmean;
      rstd[blockidx] = trstd;
  }
}

std::tuple<Tensor,Tensor,Tensor>
layernorm_forward(const Tensor& self,const Tensor& weight,const Tensor& bias,double ln_epsilon,bool training)
{
    AT_ASSERTM(self.is_cuda(), "x must on dcu");
    const int cols=self.sizes().back();
    AT_ASSERT(cols>=64,"cols must >= 64");//torch layernorm don't support cols<=64 on dcu, so we too .
    const int nelem = self.numel();
    const int rows=nelem/cols;
    Tensor ret = at::empty_like(self),mean,rstd;
    if(training){
      mean = at::empty({rows},self.options());
      rstd = at::empty({rows},self.options());
    }

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
        scalar_t* mean_data=NULL,*rstd_data=NULL;
        if(training){
          mean_data=mean.data_ptr<scalar_t>();
          rstd_data=rstd.data_ptr<scalar_t>();
        }
        if (cols % 16 == 0&&cols<=16384){
          if(cols<=1024){
                layernorm_forward_kernel_fast<scalar_t,8,128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
         } 
          else if(cols<=2048){
            layernorm_forward_kernel_fast<scalar_t,8,256><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
          }
          else if(cols<=4096){
            if(rows>1200){
                layernorm_forward_kernel_fast<scalar_t,8,512><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                          self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
            }
            else{
              layernorm_forward_kernel_fast<scalar_t,4,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                      self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
            }
          }
          else if(cols<=8192){
            layernorm_forward_kernel_fast<scalar_t,8,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
          }
          else {
            layernorm_forward_kernel_fast<scalar_t,16,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
          }
        }
        else{
            layernorm_forward_kernel<scalar_t,1><<<rows, get_layernorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t, int VEC>
__global__ void LayerNormBackward_kernel_vec_fast(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_4096
  #pragma unroll
  for(int i=0;i<VEC;i++){
    pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
  }
  *(VecTpye*)(dX+index) = tdx;
  if (j < tcol-512) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(pgamma_2[i])+b*static_cast<T_ACC>(px_2[i])+c;
    }
    *(VecTpye*)(dX+index_2) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void LayerNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
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
        if (N % 16 == 0 && N<=16384){
          if(N<=1024)
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          else if(N<=2048){
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else if(N<=4096){
            if(M>1200){
              LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
            else{
              LayerNormBackward_kernel_vec_fast<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
          }
          else if(N<=8192){
            LayerNormBackward_kernel_vec_fast<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else{
            LayerNormBackward_kernel_vec_fast<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
        }
        else{
            LayerNormBackward_kernel_vec<scalar_t,1><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
        }
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
    double eps,bool training) {
    auto px=x.expect_contiguous();
    auto pweight=weight.expect_contiguous();
    auto pbias=bias.expect_contiguous();
    auto ret = layernorm_forward(*px, *pweight,*pbias,eps,training);
    ctx->save_for_backward({*px,std::get<1>(ret),std::get<2>(ret),*pweight});
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
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};

Tensor fuse_layernorm_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps,bool training) {
   auto result = fuseLayerNormFunction::apply(x, weight,bias,eps,training);
   return result[0];
  }
}
}


