#include "act_groupnorm.h"
namespace at{
namespace native{
template <typename scalar_t,class act_op, int VEC=4>
__global__ void groupnorm_act_forward_kernel_fast(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,int spatial_size,int channel_size,scalar_t eps,act_op op)
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
      int jj=((block_cols+j*VEC+i) / spatial_size) % channel_size;
      py[i]=op.forward((static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]));
    }
    Y[j]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      mean[blockidx] = tmean;
      rstd[blockidx] = trstd;
  }
}


template <typename scalar_t,class act_op, int VEC=4>
__global__ void groupnorm_act_forward_kernel(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,int spatial_size,int N,int channel_size,act_op op)
{
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using T_ACC = acc_type<scalar_t, true>;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  int tid=(blockIdx.x*blockDim.x+threadIdx.x)*VEC;
  if(tid>=N)return;
  scalar_t tmean;
  scalar_t trstd;
  tmean=mean[tid/cols];
  trstd=rstd[tid/cols];
  int jj=tid%(spatial_size*channel_size)/spatial_size;
  tx = *(VecTpye*)(input+tid);
  #pragma unroll
  for(int i=0;i<VEC;i++){
    py[i]=op.forward((static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]));
  }
  *(VecTpye*)(ret+tid)=ty;
}

template <typename scalar_t,class act_op> 
void groupnorm_act_forward_impl(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double ln_epsilon,
                                    bool training,act_op op,Tensor& ret,Tensor& mean,Tensor&rstd)
{
    AT_ASSERT(self.is_cuda(), "x must on dcu");
    AT_ASSERT(num_channels % num_groups == 0,"num_channels must be divisible by num_groups");
    AT_ASSERT(self.scalar_type()!=at::kDouble,"Don't surpport FP64 tensor");
    const int nelem = self.numel();
    const int N=self.sizes().front();
    const int HxW=nelem/(N*num_channels);
    const int G = num_groups;
    const int D = num_channels / G;
    int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/512;
    int maxblock=at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;
    bool eval_large_col=false;
    int cols=D * HxW;
    int block_per_row=(cols-1)/2048+1;
    int blockdimy=(maxblock-1)/(N * G)+1;
    if(blockdimy>block_per_row) blockdimy=block_per_row;
    if(cols<80000)blockdimy=1;//don't splitk for small cols
    if(blockdimy>64)blockdimy=64;
    bool eval_fast=false;
    if(!training&&cols % 4 == 0&&(cols<=2048||cols<=4096&&cols%8==0||cols<=8192&&cols%16==0))eval_fast=true;
    if(!eval_fast){
      mean = at::empty({N, G},self.options());
      rstd = at::empty({N, G},self.options());
    }
    using T_ACC = acc_type<scalar_t, true>;
    scalar_t* self_data = self.data_ptr<scalar_t>();
    scalar_t* ret_data=ret.data_ptr<scalar_t>();
    scalar_t* weight_data=weight.data_ptr<scalar_t>();
    scalar_t* bias_data=bias.data_ptr<scalar_t>();
    scalar_t  eps = ln_epsilon;
    scalar_t* mean_data=NULL,*rstd_data=NULL;
    if(!eval_fast){
      mean_data=mean.data_ptr<scalar_t>();
      rstd_data=rstd.data_ptr<scalar_t>();
    }
    int spatial_size=HxW;
    int channel_size=num_channels;
    auto stream = at::cuda::getCurrentCUDAStream();
    PassThrough<scalar_t> passop;
    if (cols % 4 == 0){
        if(cols<=2048){
          groupnorm_act_forward_kernel_fast<scalar_t,act_op,4><<<N * G, get_groupnorm_blocksize(cols/4), 0, stream>>>(
                                  self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,spatial_size,channel_size,eps,op);
        }
        else if(cols<=4096&&cols%8==0){
          groupnorm_act_forward_kernel_fast<scalar_t,act_op,8><<<N * G, 512, 0, stream>>>(
                                self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,spatial_size,channel_size,eps,op);
        }
        else if(cols<=8192&&cols%16==0){
          groupnorm_act_forward_kernel_fast<scalar_t,act_op,16><<<N * G, 512, 0, stream>>>(
                                self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,spatial_size,channel_size,eps,op);
        }
        else{//very long last dim
            //we change torch native method to VEC 4.
          if(blockdimy==1){
            act_groupnorm_forward_mean_rstd<scalar_t,PassThrough<scalar_t>,4><<<N * G, 512, 0, stream>>>(self_data,mean_data,rstd_data,cols,eps,passop);
          }
          else{//use splitk method
              c10::TensorOptions options(DeviceType::CUDA);
              options.dtype(ScalarType::Int);
              Tensor wel_data=torch::empty(N * G * blockdimy * 4, options);
              Tensor semaphores=torch::empty(N * G, options);
              AT_CUDA_CHECK(cudaMemsetAsync(semaphores.data_ptr(), 0, N * G * sizeof(int),stream));
              act_groupnorm_forward_kernel_large_col_mean_rstd<scalar_t,PassThrough<scalar_t>,4><<<dim3(N * G,blockdimy), 512, 0, stream>>>(
                                self_data,mean_data,rstd_data,cols,blockdimy,eps,(int*)semaphores.data_ptr(),wel_data.data_ptr(),passop);
          }
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          int blocksize=(N * G *cols/4-1)/128+1;
          groupnorm_act_forward_kernel<scalar_t,act_op,4><<<blocksize,128,0,stream>>>(
                                  self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,spatial_size,N * G * cols,channel_size,op);
        }
    } 
    else{//torch native method
        act_groupnorm_forward_mean_rstd<scalar_t,PassThrough<scalar_t>,1><<<N * G, get_groupnorm_blocksize(cols), 0, stream>>>(self_data,mean_data,rstd_data,cols,eps,passop);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int blocksize=(N * G *cols-1)/128+1;
        groupnorm_act_forward_kernel<scalar_t,act_op,1><<<blocksize, 128, 0, stream>>>(self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,spatial_size,N * G * cols,channel_size,op);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t,class act_op> 
void groupnorm_act_backward_impl(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma,act_op op,Tensor& dX,Tensor& dgamma,Tensor& dbeta)
{

}

}
}

