#include "rmsnorm.h"

namespace at{
namespace native{

namespace {
  constexpr int VEC = 4;
}


template <typename scalar_t, int VEC=4>
__global__ void rmsnorm_forward_kernel(scalar_t* input,scalar_t* ret,float* rstd, scalar_t* gamma,int cols,scalar_t eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC val_shared[8];
  __shared__ T_ACC s_rstd;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC val=0;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val += static_cast<T_ACC>(px[i])*static_cast<T_ACC>(px[i]);
    }
  }
  val = BlockReduceSum(val,val_shared);
  if (threadidx == 0) {
    s_rstd=c10::cuda::compat::rsqrt(val/cols + static_cast<T_ACC>(eps));
  }
  __syncthreads();
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      py[i]=static_cast<T_ACC>(px[i]) * static_cast<T_ACC>(gamma[jj])*trstd;
    }
    Y[j]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      rstd[blockidx] = trstd;
  }
}

template <typename scalar_t, int VEC=4,int blocksize=512>
__global__ void rmsnorm_forward_kernel_fast(scalar_t* input,scalar_t* ret,float* rstd, scalar_t* gamma,int cols,scalar_t eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  constexpr int sharesize=blocksize/C10_WARP_SIZE;
  __shared__ T_ACC val_shared[sharesize];
  __shared__ T_ACC s_rstd;
  
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  int j=threadidx;
  T_ACC val=0;
  if (j < tcol) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val += static_cast<T_ACC>(px[i])*static_cast<T_ACC>(px[i]);
    }
  }
  val = BlockReduceSum<T_ACC,sharesize>(val,val_shared);
  if (threadidx == 0) {
    s_rstd=c10::cuda::compat::rsqrt(val/cols + static_cast<T_ACC>(eps));
  }
  __syncthreads();
  trstd=s_rstd;
  // if(threadIdx.x==0)printf("%d,%f,%f\n",blockIdx.x,(float)trstd,s_rstd);
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      py[i]=static_cast<T_ACC>(px[i]) * static_cast<T_ACC>(gamma[jj])*trstd;
    }
    Y[j]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      rstd[blockidx] = trstd;
  }
}

std::tuple<Tensor,Tensor>rmsnorm_forward(const Tensor& self,const Tensor& weight,c10::optional<Tensor>& output_inplace,double ln_epsilon,bool training)
{
    AT_ASSERTM(self.is_cuda(), "x must on HCU");
    const int cols=self.sizes().back();
    AT_ASSERT(cols>=64,"cols must >= 64");//torch layernorm don't support cols<=64 on HCU, so we too .
    const int nelem = self.numel();
    const int rows=nelem/cols;
    Tensor ret = output_inplace? *(at::borrow_from_optional_tensor(output_inplace)):at::empty_like(self);
    Tensor rstd;
    if(training)rstd = at::empty({rows},self.options().dtype(at::ScalarType::Float));
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      self.scalar_type(),
      "rmsnorm_forward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* weight_data=weight.data_ptr<scalar_t>();
        scalar_t  eps = ln_epsilon;
        float* rstd_data=NULL;
        if(training)rstd_data=rstd.data_ptr<float>();
        if (cols % 16 == 0 && cols<=16384){
          if(cols<=128){
            rmsnorm_forward_kernel_fast<scalar_t,1,128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
          }
          else if(cols<=1024){
            rmsnorm_forward_kernel_fast<scalar_t,8,128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
          }
          else if(cols<=2048){
            rmsnorm_forward_kernel_fast<scalar_t,8,256><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
          }
          else if(cols<=4096){
            if(rows>1200){
              rmsnorm_forward_kernel_fast<scalar_t,8,512><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                      self_data,ret_data,rstd_data,weight_data,cols,eps);
            }
            else{
              rmsnorm_forward_kernel_fast<scalar_t,4,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                        self_data,ret_data,rstd_data,weight_data,cols,eps);
            }
          }
          else if(cols<=8192){
            rmsnorm_forward_kernel_fast<scalar_t,8,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
          }
          else{
            rmsnorm_forward_kernel_fast<scalar_t,16,1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
          }
        }
        else{
            rmsnorm_forward_kernel<scalar_t,1><<<rows, get_layernorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,ret_data,rstd_data,weight_data,cols,eps);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor>(ret,rstd);
}


template <typename scalar_t,int VEC=4>
__global__ void RMSNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* rstd, scalar_t* dX)
{
  RMSNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void RMSNormBackward_kernel_vec_fast(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* rstd, scalar_t* dX)
{
  RMSNORMBACKWARD_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void RMSNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* rstd, scalar_t* dX)
{
  RMSNORMBACKWARD_VEC_4096
  #pragma unroll
  for(int i=0;i<VEC;i++){
    pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
  }
  *(VecTpye*)(dX+index) = tdx;
  if (j < tcol-512) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(pgamma_2[i])+b*static_cast<T_ACC>(px_2[i]);
    }
    *(VecTpye*)(dX+index_2) = tdx;
  }
}

template <typename scalar_t,int VEC=4>
__global__ void RMSNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* rstd, scalar_t* dX)
{
  RMSNORMBACKWARD_VEC
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i]);
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}



std::tuple<Tensor,Tensor>
rmsnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "rmsnorm_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      
      Gamma_Backward(nelem,N,dY_data,X_data,rstd_data,dgamma_data);
      if (N % 16 == 0 && N<=16384){
        if(N<=1024)
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
        else if(N<=2048){
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
        }
        else if(N<=4096){
          if(M>1200){
            RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
          }
          else{
            RMSNormBackward_kernel_vec_fast<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
          }
        }
        else if(N<=8192){
          RMSNormBackward_kernel_vec_fast<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
        }
        else{
          RMSNormBackward_kernel_vec_fast<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
        }
      }
      else{
          RMSNormBackward_kernel_vec<scalar_t><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,rstd_data,dX_data);
      }     
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor>(dX,dgamma);
}

class RMSNormFunction : public torch::autograd::Function<RMSNormFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& weight,
    double eps,bool training) {
    auto px=x.expect_contiguous();
    auto pweight=weight.expect_contiguous();
    c10::optional<Tensor> out;
    auto ret = rmsnorm_forward(*px,*pweight,out,eps,training);
    ctx->save_for_backward({*px,std::get<1>(ret),*pweight});
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto saved = ctx->get_saved_variables();
    auto X = saved[0];
    auto rstd = saved[1];
    auto gamma = saved[2];
    auto ret = rmsnorm_backward(gradOutput[0],X,rstd,gamma);
    return {std::get<0>(ret),std::get<1>(ret),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};
Tensor rmsnorm_forward_autograd(const Tensor& x,const Tensor& weight,double eps,bool training) {
   auto result = RMSNormFunction::apply(x, weight,eps,training);
   return result[0];
  }
}
}


