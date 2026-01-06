#include "layernorm.h"
#include <c10/util/MaybeOwned.h>
#include <iostream> //lj
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


template <typename scalar_t>
struct TypeChange {
  using type = scalar_t;
};

template <>
struct TypeChange<c10::BFloat16> {
  using type = uint16_t;
};
template<typename scalar_t>
using __change_type = typename TypeChange<scalar_t>::type;


template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1); 
    u += 0x8000; 
    out = u>>16;
  }
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void layernorm_forward_kernel(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
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
    s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0){
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
  }
}

// template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
// __global__ void layernorm_forward_kernel_fast(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
//                                             const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
// {
//   int blockidx=blockIdx.x;
//   int threadidx=threadIdx.x;
//   int block_cols=blockidx*cols;
//   using T_ACC = acc_type<scalar_t, true>;
//   using WelfordType = WelfordData<T_ACC, int, T_ACC>;
//   using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
//   constexpr int share_size=blocksize/C10_WARP_SIZE;
//   __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[share_size];
//   __shared__ T_ACC s_mean;
//   __shared__ T_ACC s_rstd;
//   WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
//   WelfordOp welford_op;
//   WelfordType val;
//   using VecTpye=memory::aligned_vector<scalar_t, VEC>;
//   using new_type = __change_type<scalar_t>;
//   VecTpye *X=(VecTpye*)(input+block_cols);
//   VecTpye *Y=(VecTpye*)(ret+block_cols);
//   int tcol=cols/VEC;
//   T_ACC tmean;
//   T_ACC trstd;
//   VecTpye tx;
//   VecTpye ty;
//   new_type* px=(new_type*)(&tx);
//   new_type* py=(new_type*)(&ty);
//   int j=threadidx;
//   if (j < tcol) {
//     tx = X[j];
//     #pragma unroll
//     for(int i=0;i<VEC;i++){
//       val = welford_op.reduce(val, to_float(px[i]));
//     }
//   }
//   val = BlockReduce(val,welford_op,val_shared_ptr);
//   if (threadidx == 0) {
//     //layernorm get mean and rstd
//     thrust::tie(tmean, s_mean) = welford_op.project(val);
//     s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
//   }
//   __syncthreads();
//   tmean=s_mean;
//   trstd=s_rstd;
//   if (j < tcol) {
//     #pragma unroll
//     for(int i=0;i<VEC;i++){
//       int jj=j*VEC+i;
//       if constexpr(is_affine){
//         from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
//       }
//       else{
//         from_float(py[i],(to_float(px[i]) - tmean) * trstd);
//       }
//     }
//     Y[j]=ty;
//   }
//   if constexpr(is_training){
//     if(threadIdx.x==0){
//         mean[blockidx] = tmean;
//         rstd[blockidx] = trstd;
//     }
//   }
// }

template <typename scalar_t, int VEC,int blocksize,bool is_affine,bool is_training>
__global__ void layernorm_forward_kernel_fast(scalar_t* input,scalar_t* ret,float* mean,float* rstd,
                                            const scalar_t* gamma,const scalar_t* beta,int cols,float eps)
{
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  using T_ACC = acc_type<scalar_t, true>;
  constexpr int share_size=blocksize/C10_WARP_SIZE*2;
  __shared__ float val_shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye *Y=(VecTpye*)(ret+block_cols);
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  VecTpye ty;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(&ty);
  int j=threadidx;
  float sum = 0.0f;
  float sum_sq = 0.0f;
  if (j < tcol) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      float temp = to_float(px[i]);
      sum+=temp;
      sum_sq+=temp*temp;
    }
  }
  BlockReduceSum2<float,blocksize>(sum,sum_sq,val_shared);
  if (threadidx == 0) {
    //layernorm get mean and rstd
    float mean = sum / cols;
    float var = (sum_sq / cols) - (mean * mean);
    var = (var < 0.0f) ? 0.0f : var;
    s_rstd=c10::cuda::compat::rsqrt(var + eps);
    s_mean = mean;
  }
  __syncthreads();
  tmean=s_mean;
  trstd=s_rstd;
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j*VEC+i;
      if constexpr(is_affine){
        from_float(py[i],(to_float(px[i]) - tmean) * trstd * to_float(gamma[jj]) + to_float(beta[jj]));
      }
      else{
        from_float(py[i],(to_float(px[i]) - tmean) * trstd);
      }
    }
    Y[j]=ty;
  }
  if constexpr(is_training){
    if(threadIdx.x==0){
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
  }
}

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()



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
      mean = at::empty({rows},self.options().dtype(at::ScalarType::Float));
      rstd = at::empty({rows},self.options().dtype(at::ScalarType::Float));
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
        const scalar_t* weight_data = weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr;
        const scalar_t* bias_data = bias.defined() ? bias.const_data_ptr<scalar_t>() : nullptr;
        float  eps = ln_epsilon;
        float* mean_data=NULL,*rstd_data=NULL;
        if(training){
          mean_data=mean.data_ptr<float>();
          rstd_data=rstd.data_ptr<float>();
        }
        if ((cols<=1024)||(cols % 16 == 0&&cols<=16384)){
          BOOL_SWITCH(weight_data!=nullptr,is_affine,[&] {
            BOOL_SWITCH(training,is_training,[&] {
              if(cols<=1024){
                layernorm_forward_kernel_fast<scalar_t,8,128,is_affine,is_training><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
              } 
              else if(cols<=2048){
                layernorm_forward_kernel_fast<scalar_t,16,128,is_affine,is_training><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
              }
              else if(cols<=4096){
                layernorm_forward_kernel_fast<scalar_t,16,256,is_affine,is_training><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                              self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
              }
              else if(cols<=8192){
                layernorm_forward_kernel_fast<scalar_t,16,512,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
              }
              else{
                layernorm_forward_kernel_fast<scalar_t,16,1024,is_affine,is_training><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
              }
            });
          });
        }
        else {
          BOOL_SWITCH(weight_data!=nullptr,is_affine,[&] {
            BOOL_SWITCH(training,is_training,[&] {
              layernorm_forward_kernel<scalar_t,8,512,is_affine,is_training><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
            });
          });     
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
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
__global__ void LayerNormBackward_kernel_vec_fast(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
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
__global__ void LayerNormBackward_kernel_vec_fast_optional(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_2048_OPTIONAL
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*(gamma==nullptr?T_ACC(1):static_cast<T_ACC>(pgamma[i]))+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
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
__global__ void LayerNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
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

template <typename scalar_t,int VEC=4>
__global__ void LayerNormBackward_kernel_vec_optional(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,const float* rstd, scalar_t* dX)
{
  LAYERNORMBACKWARD_VEC_OPTIONAL
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    if (gamma!=nullptr) tgamma = *(VecTpye*)(gamma+off*VEC);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*(gamma==nullptr?T_ACC(1):static_cast<T_ACC>(pgamma[i]))+b*static_cast<T_ACC>(px[i])+c;
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
      float* mean_data = mean.data_ptr<float>();
      float* rstd_data = rstd.data_ptr<float>();
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

std::tuple<Tensor,Tensor,Tensor>
layernorm_backward_optional(const Tensor& grad, const Tensor& X, const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& gamma_opt,
                        bool weight_has_value, bool bias_has_value)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);

  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma_maybe = *gamma_maybe_owned;
  auto gamma = gamma_maybe.expect_contiguous();

  Tensor dgamma;
  Tensor dbeta;
  if(weight_has_value){
    dgamma = at::empty_like(*gamma);
  }
  if(bias_has_value){
    dbeta = at::empty_like(*gamma);
  }

  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "layernorm_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* mean_data = mean.data_ptr<float>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data =  gamma->defined() ? gamma->data_ptr<scalar_t>() : nullptr;
      scalar_t* dgamma_data = dgamma.defined() ? dgamma.data_ptr<scalar_t>() : nullptr;
      scalar_t* dbeta_data = dbeta.defined() ? dbeta.data_ptr<scalar_t>() : nullptr;
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      
      GammaBeta_Backward_optional(nelem,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
        if (N % 16 == 0 && N<=16384){
          if(N<=1024)
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          else if(N<=2048){
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else if(N<=4096){
            if(M>1200){
              LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
            else{
              LayerNormBackward_kernel_vec_fast_optional<scalar_t,4><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
            }
          }
          else if(N<=8192){
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,8><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          else{
            LayerNormBackward_kernel_vec_fast_optional<scalar_t,16><<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
        }
        else{
            LayerNormBackward_kernel_vec_optional<scalar_t,1><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
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
    const std::optional<torch::Tensor>& weight_opt,
    const std::optional<torch::Tensor>& bias_opt,
    double eps,bool training) {
    auto px=x.expect_contiguous();
    c10::MaybeOwned<torch::Tensor> weight_maybe_owned =
        at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<torch::Tensor> bias_maybe_owned =
        at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;
    auto pweight=weight.expect_contiguous();
    auto pbias=bias.expect_contiguous();
    
    std::tuple<Tensor,Tensor,Tensor> ret ;
    int cols = px->sizes().back();
    if(cols%8==0&&cols>=64){
      ret = layernorm_forward(*px, *pweight,*pbias,eps,training);
    }
    else{
      std::vector<int64_t> shape;
      shape.push_back(cols);
      ret = at::native::layer_norm_cuda(*px,shape,weight_opt,bias_opt,eps);
    }
    if(training){
      int flags = (weight_opt.has_value() ? 1 : 0) | (bias_opt.has_value() ? 2 : 0);
      torch::Tensor flags_tensor = torch::tensor(flags, torch::kInt32);
      std::vector<torch::Tensor> tensors_to_save = {*px,std::get<1>(ret),std::get<2>(ret),*pweight};
      tensors_to_save.push_back(flags_tensor);
      ctx->save_for_backward(tensors_to_save);
    }
    
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
    torch::Tensor flags_tensor = saved.back();
    int flags = flags_tensor.item<int>();
    bool weight_has_value = (flags & 1) != 0;
    bool bias_has_value = (flags & 2) != 0;
    std::tuple<Tensor,Tensor,Tensor> ret;
    if(weight_has_value){
      ret = layernorm_backward_optional(gradOutput[0],X,mean,rstd,gamma,weight_has_value,bias_has_value);
    }else{
      ret = layernorm_backward_optional(gradOutput[0],X,mean,rstd,std::nullopt,weight_has_value,bias_has_value);
    }
    if(weight_has_value && bias_has_value){
      return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable()};
    } else if(weight_has_value && !bias_has_value){
      return {std::get<0>(ret),std::get<1>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
    }else{
      return {std::get<0>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
    }
  }
};

Tensor fuse_layernorm_forward_autograd(const Tensor& x,const std::optional<Tensor>& weight,const std::optional<Tensor>& bias,
                                      double eps,bool training) {
   auto result = fuseLayerNormFunction::apply(x, weight,bias,eps,training);
   return result[0];
  }
}
}


