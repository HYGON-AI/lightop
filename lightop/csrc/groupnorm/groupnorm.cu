#include "groupnorm.h"

int get_groupnorm_blocksize(int cols)
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
  constexpr int kCUDANumThreads = 256;
}

template <typename scalar_t, int VEC=4>
__global__ void RowwiseMomentsCUDAKernel(scalar_t* input,scalar_t* mean,scalar_t* rstd,int cols,scalar_t eps)
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
  int tcol=cols/VEC;
  T_ACC tmean;
  T_ACC trstd;
  VecTpye tx;
  scalar_t* px=(scalar_t*)(&tx);
  for (int j = threadidx; j < tcol; j += blockDim.x) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, static_cast<T_ACC>(px[i]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if(threadIdx.x==0&&rstd!=NULL) {
    //layernorm get mean and rstd 
    thrust::tie(tmean, s_mean) = welford_op.project(val);
    mean[blockidx] = s_mean;
    rstd[blockidx] = c10::cuda::compat::rsqrt(tmean + static_cast<T_ACC>(eps));
  }
 
}


template <typename scalar_t>
__global__ void ComputeFusedParamsCUDAKernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,scalar_t* a_data,scalar_t* b_data,int N,int num_channels,int group)
{

  using T_ACC = acc_type<scalar_t, true>;
  const int64_t index = (blockIdx.x * blockDim.x + threadIdx.x);

  if (index < N*num_channels) {
      const int ng = index / (num_channels / group);
      const int c = index % num_channels;
      const T_ACC scale =static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
      a_data[index]=scale;
      b_data[index]=-scale * static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(beta[c]);
  }
  
}

template <typename scalar_t, int VEC=4>
__global__ void groupnorm_forward_kernel_1d(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
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

template <typename scalar_t, int VEC=4>
__global__ void groupnorm_forward_kernel_1d_2048(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
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
    //groupnorm get mean and rstd
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

template <typename scalar_t, int VEC=4>
__global__ void groupnorm_forward_kernel_1d_4096(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
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
  VecTpye tx2;
  VecTpye ty2;
  scalar_t* px2=(scalar_t*)(&tx2);
  scalar_t* py2=(scalar_t*)(&ty2);
  int j=threadidx;
  int j2=j+512;
  tx = X[j];
  #pragma unroll
  for(int i=0;i<VEC;i++){
    val = welford_op.reduce(val, static_cast<T_ACC>(px[i]));
  }
  if (j2 < tcol) {
    tx2 = X[j2];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      val = welford_op.reduce(val, static_cast<T_ACC>(px2[i]));
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
  #pragma unroll
  for(int i=0;i<VEC;i++){
    int jj=j*VEC+i;
    py[i]=(static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
  }
  Y[j]=ty;
  if (j2 < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      int jj=j2*VEC+i;
      py2[i]=(static_cast<T_ACC>(px2[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    Y[j2]=ty2;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      mean[blockidx] = tmean;
      rstd[blockidx] = trstd;
  }
}

std::tuple<Tensor,Tensor,Tensor>
groupnorm_forward(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double ln_epsilon,bool training)
{
    AT_ASSERTM(self.is_cuda(), "x must on dcu");
    AT_ASSERT(num_channels % num_groups == 0,"num_channels must be divisible by num_groups");
    const int nelem = self.numel();
    const int N=self.sizes().front();
    const int HxW=nelem/(N*num_channels);
    const int G = num_groups;
    const int D = num_channels / G;
    Tensor ret = at::empty_like(self),mean,rstd;
    if(training){
      mean = at::empty({N, G},self.options());
      rstd = at::empty({N, G},self.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16, 
      self.scalar_type(),
      "groupnorm_forward",
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
        if (HxW == 1) {
            int cols=D * HxW;
            if (cols % 4 == 0){
                if(cols<=2048)
                groupnorm_forward_kernel_1d_2048<scalar_t,4><<<N * G, get_groupnorm_blocksize(cols/4), 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                else if(cols<=4096){
                groupnorm_forward_kernel_1d_4096<scalar_t,4><<<N * G, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                      self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                else{
                groupnorm_forward_kernel_1d<scalar_t,4><<<N * G, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                                      self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
                }
                
            }
            else{
                groupnorm_forward_kernel_1d<scalar_t,1><<<N * G, get_groupnorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                                        self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,eps);
            
            }
        }
        else{
            int cols=D * HxW;
            const int64_t B = (N*num_channels + kCUDANumThreads - 1) / kCUDANumThreads;
            if (cols % 4 == 0){
              RowwiseMomentsCUDAKernel<scalar_t,4><<<N * G, get_groupnorm_blocksize(cols/4), 0, at::cuda::getCurrentCUDAStream()>>>(
                                    self_data,mean_data,rstd_data,cols,eps);
    
            }
            else{
                RowwiseMomentsCUDAKernel<scalar_t,1><<<N * G, get_groupnorm_blocksize(cols), 0, at::cuda::getCurrentCUDAStream()>>>(
                                      self_data,mean_data,rstd_data,cols,eps);
            }
           
            Tensor a = at::empty({N, num_channels}, self.options());
            Tensor b = at::empty({N, num_channels}, self.options());
            scalar_t* a_data = a.data_ptr<scalar_t>();
            scalar_t* b_data = b.data_ptr<scalar_t>();
            ComputeFusedParamsCUDAKernel<scalar_t><<<B, kCUDANumThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,a_data,b_data,N,num_channels,G);
            auto iter = TensorIteratorConfig()
                .check_all_same_dtype(std::is_same<scalar_t, T_ACC>::value)
                .resize_outputs(false)
                .add_owned_output(ret.view({N * num_channels, HxW}))
                .add_owned_input(self.view({N * num_channels, HxW}))
                .add_owned_input(a.view({N * num_channels, 1}))
                .add_owned_input(b.view({N * num_channels, 1}))
                .build();
            gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x, scalar_t a, scalar_t b) -> scalar_t {
              return static_cast<T_ACC>(a) * static_cast<T_ACC>(x) + static_cast<T_ACC>(b);
            });

        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}


template <typename scalar_t>
__global__ void GroupNormBackward_1d_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  GROUPNORMBACKWARD_1D_VEC_2048
  if (j < tcol) {
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t>
__global__ void GroupNormBackward_1d_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  GROUPNORMBACKWARD_1D_VEC_4096
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
__global__ void GroupNormBackward_1d_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX)
{
  GROUPNORMBACKWARD_1D_VEC
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

template <typename scalar_t>
__global__ void GroupNormBackward_kernel_vec_2048(int N,int HxW,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX,scalar_t* ds_data,scalar_t* db_data,scalar_t* c2_data,scalar_t* c3_data)
{
  GROUPNORMBACKWARD_VEC_2048
}

template <typename scalar_t>
__global__ void GroupNormBackward_kernel_vec_4096(int N,int HxW,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX,scalar_t* ds_data,scalar_t* db_data,scalar_t* c2_data,scalar_t* c3_data)
{
  GROUPNORMBACKWARD_VEC_4096
}


template <typename scalar_t,int VEC=4>
__global__ void GroupNormBackward_kernel_vec(int N,int HxW,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX,scalar_t* ds_data,scalar_t* db_data,scalar_t* c2_data,scalar_t* c3_data)
{
  GROUPNORMBACKWARD_VEC
}

std::tuple<Tensor,Tensor,Tensor>
groupnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  const int nelem = grad.numel();
  const int N=grad.sizes().front();
  const int num_channels=gamma.numel();
  const int HxW=nelem/(N*num_channels);
  const int G = mean.numel()/N;
  const int D = num_channels / G;
  auto pgrad = grad.expect_contiguous();
  Tensor dX = at::empty_like(*pgrad);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "groupnorm_backward",
    [&] {
      using T_ACC = acc_type<scalar_t, true>;
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      scalar_t* mean_data = mean.data_ptr<scalar_t>();
      scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
      scalar_t* dX_data = dX.data_ptr<scalar_t>();
      
      if (HxW == 1) {
          int cols=D * HxW;
          GN_GammaBeta_Backward_1d(nelem,num_channels,D,G,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);

          if (cols % 4 == 0){
              if(cols<=2048)
                GroupNormBackward_1d_kernel_vec_2048<scalar_t><<<N * G, get_groupnorm_blocksize(cols/4), 0, cuda_stream>>>(cols, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
        
              else if(cols<=4096){
                GroupNormBackward_1d_kernel_vec_4096<scalar_t><<<N * G, 512, 0, cuda_stream>>>(cols, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
              }
              else{
                GroupNormBackward_1d_kernel_vec<scalar_t><<<N * G, 512, 0, cuda_stream>>>(cols, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
              }
              
          }
          else{
              GroupNormBackward_1d_kernel_vec<scalar_t,1><<<N * G, get_groupnorm_blocksize(cols), 0, cuda_stream>>>(cols, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
          }
          

        }
        else{

            Tensor ds = at::empty({N, num_channels}, grad.options());
            Tensor db = at::empty({N, num_channels}, grad.options());
            Tensor c1 = at::empty({0}, grad.options());
            Tensor c2 = at::empty({N, G}, grad.options());
            Tensor c3 = at::empty({N, G}, grad.options());
            scalar_t* c2_data = c2.data_ptr<scalar_t>();
            scalar_t* c3_data = c3.data_ptr<scalar_t>();
            scalar_t* ds_data = ds.data_ptr<scalar_t>();
            scalar_t* db_data = db.data_ptr<scalar_t>();
            ComputeInternalGradientsCUDAKernel<scalar_t><<<N * num_channels, kCUDANumThreads, 0, cuda_stream>>>(
              HxW, dY_data, X_data, ds_data, db_data);
            int cols=D;  
            GN_GammaBeta_Backward(nelem,N,num_channels,D,G,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data,ds_data,db_data);

            if (cols % 4 == 0){
                if(cols<=2048)
                  GroupNormBackward_kernel_vec_2048<scalar_t><<<N * G, get_groupnorm_blocksize(cols/4), 0, cuda_stream>>>(cols,HxW, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data,ds_data,db_data,c2_data,c3_data);
          
                else if(cols<=4096){
                  GroupNormBackward_kernel_vec_4096<scalar_t><<<N * G, 512, 0, cuda_stream>>>(cols,HxW, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data,ds_data,db_data,c2_data,c3_data);
                }
                else{
                  GroupNormBackward_kernel_vec<scalar_t><<<N * G, 512, 0, cuda_stream>>>(cols,HxW, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data,ds_data,db_data,c2_data,c3_data);
                }
                
            }
            else{
                GroupNormBackward_kernel_vec<scalar_t,1><<<N * G, get_groupnorm_blocksize(cols), 0, cuda_stream>>>(cols,HxW, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data,ds_data,db_data,c2_data,c3_data);
            }

             auto iter1 = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<scalar_t, T_ACC>::value)
                      .add_output(c1)
                      .add_owned_input(rstd.view({N, G, 1}))
                      .add_owned_input(gamma.view({1, G, D}))
                      .build();
            gpu_kernel(iter1, [] GPU_LAMBDA(scalar_t rstd, scalar_t gamma) -> T_ACC {
              return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
            });

            auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<scalar_t, T_ACC>::value)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D, HxW}))
                      .add_owned_input(grad.view({N * G, D, HxW}))
                      .add_owned_input(X.view({N * G, D, HxW}))
                      .add_owned_input(c1.view({N * G, D, 1}))
                      .add_owned_input(c2.view({N * G, 1, 1}))
                      .add_owned_input(c3.view({N * G, 1, 1}))
                      .build();
            gpu_kernel(
                iter,
                [] GPU_LAMBDA(scalar_t dy, scalar_t x, scalar_t c1, scalar_t c2, scalar_t c3) -> scalar_t {
                  return static_cast<T_ACC>(c1) * static_cast<T_ACC>(dy) + static_cast<T_ACC>(c2) * static_cast<T_ACC>(x) +
                      static_cast<T_ACC>(c3);
                });
        }
      
     
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}


class fuseGroupNormFunction : public torch::autograd::Function<fuseGroupNormFunction> {

  public:
    static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& x,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& bias,
      int num_groups,int num_channels,double eps,bool training) {
      auto px=x.expect_contiguous();
      auto pweight=weight.expect_contiguous();
      auto pbias=bias.expect_contiguous();
      auto ret = groupnorm_forward(*px, *pweight,*pbias,num_groups,num_channels,eps,training);
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
      auto ret = groupnorm_backward(gradOutput[0],X,mean,rstd,gamma);
      return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};


Tensor fuse_groupnorm_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,bool training) {
   auto result = fuseGroupNormFunction::apply(x, weight,bias,num_groups,num_channels,eps,training);
   return result[0];
  }
}
}
