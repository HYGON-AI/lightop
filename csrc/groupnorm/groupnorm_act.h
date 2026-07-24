#include "act_groupnorm.h"
namespace at{
namespace native{
template <typename scalar_t,class act_op, int VEC=4>
__global__ void groupnorm_act_forward_kernel_fast(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,int spatial_size,int channel_size,acc_type<scalar_t, true> eps,act_op op)
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
__global__ void groupnorm_act_forward_kernel_fast_NHWC(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,int spatial_size,int channel_size,acc_type<scalar_t, true> eps,act_op op)
{
  int blockidx=blockIdx.x;
  int blockidy=blockIdx.y;
  int threadidx=threadIdx.x;
  int D=cols/spatial_size;
  int G=channel_size/D;
  int block_cols=blockidx*cols*G + blockidy*D;
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
  int D2=D/VEC;
  int j=threadidx;
  int j2=j%D2+j/D2*(channel_size/VEC);
  if (j < tcol) {
    tx = X[j2];
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
      int jj=j*VEC%D+blockidy*D+i;
      py[i]=op.forward((static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]));
    }
    Y[j2]=ty;
  }
  if(threadIdx.x==0&&rstd!=NULL){
      int bid=blockidx*gridDim.y+blockidy;
      mean[bid] = tmean;
      rstd[bid] = trstd;
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

template <typename scalar_t,class act_op, int VEC=4>
__global__ void groupnorm_act_forward_kernel_NHWC(scalar_t* input,scalar_t* ret,scalar_t* mean,scalar_t* rstd,
                                            scalar_t* gamma,scalar_t* beta,int cols,int spatial_size,int N,int G,act_op op)
{
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using T_ACC = acc_type<scalar_t, true>;
  const int datas_per_batch=cols*G;
  const int channel_size = datas_per_batch/spatial_size;
  const int D=channel_size/G;
  VecTpye tx;
  VecTpye ty;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* py=(scalar_t*)(&ty);
  int tid=(blockIdx.x*blockDim.x+threadIdx.x)*VEC;
  if(tid>=N)return;
  int meanid=tid/datas_per_batch*G+tid%channel_size/D;
  int gammaid=tid%channel_size;
  auto tmean = static_cast<T_ACC>(mean[meanid]);
  auto trstd = static_cast<T_ACC>(rstd[meanid]);
  tx = *(VecTpye*)(input+tid);
  #pragma unroll
  for(int i=0;i<VEC;i++){
    py[i]=op.forward((static_cast<T_ACC>(px[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[gammaid+i]) + static_cast<T_ACC>(beta[gammaid+i]));
  }
  *(VecTpye*)(ret+tid)=ty;
}

template <typename scalar_t,class act_op> 
void groupnorm_act_forward_NHWC(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double gn_epsilon,
                                    bool training,act_op& op,Tensor& ret,Tensor& mean,Tensor&rstd)
{
    const int nelem = self.numel();
    const int N=self.sizes().front();
    const int HxW=nelem/(N*num_channels);
    const int G = num_groups;
    const int D = num_channels / G;
    const int cols=D * HxW;
    scalar_t* mean_data=NULL,*rstd_data=NULL;
    bool small=(D % 4 == 0&&cols<=2048)||(D % 2 == 0&&cols<=1024)||(cols<=512)||(D%8==0&&cols<=4096)||(D%16==0&&cols<=8192);
    if(training||!small){
      mean = at::empty({N, G},self.options());
      rstd = at::empty({N, G},self.options());
      mean_data=mean.data_ptr<scalar_t>();
      rstd_data=rstd.data_ptr<scalar_t>();
    }
    using T_ACC = acc_type<scalar_t, true>;
    scalar_t* self_data = self.data_ptr<scalar_t>();
    scalar_t* ret_data=ret.data_ptr<scalar_t>();
    scalar_t* weight_data=weight.data_ptr<scalar_t>();
    scalar_t* bias_data=bias.data_ptr<scalar_t>();
    T_ACC  eps = gn_epsilon;
    auto stream = at::cuda::getCurrentCUDAStream();
    PassThrough<scalar_t> passop;
    if(D % 4 == 0&&cols<=2048){
      groupnorm_act_forward_kernel_fast_NHWC<scalar_t,act_op,4><<<dim3(N , G), get_groupnorm_blocksize(cols/4), 0, stream>>>(
                              self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,num_channels,eps,op);
    }
    else if(D % 2 == 0&&cols<=1024){
      groupnorm_act_forward_kernel_fast_NHWC<scalar_t,act_op,2><<<dim3(N , G), get_groupnorm_blocksize(cols/2), 0, stream>>>(
                              self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,num_channels,eps,op);
    }
    else if(cols<=512){
      groupnorm_act_forward_kernel_fast_NHWC<scalar_t,act_op,1><<<dim3(N , G), get_groupnorm_blocksize(cols), 0, stream>>>(
                              self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,num_channels,eps,op);
    }
    else if(D%8==0&&cols<=4096){
      groupnorm_act_forward_kernel_fast_NHWC<scalar_t,act_op,8><<<dim3(N , G), 512, 0, stream>>>(
                            self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,num_channels,eps,op);
    }
    else if(D%16==0&&cols<=8192){
      groupnorm_act_forward_kernel_fast_NHWC<scalar_t,act_op,16><<<dim3(N , G), 512, 0, stream>>>(
                            self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,num_channels,eps,op);
    }
    else{//very long last dim
      if(D%4==0){
        act_groupnorm_forward_mean_rstd_NHWC<scalar_t,PassThrough<scalar_t>,4><<<dim3(N , G), 512, 0, stream>>>(self_data,mean_data,rstd_data,cols,HxW,num_channels,eps,passop);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int blocksize=(N * G *cols/4-1)/128+1;
        groupnorm_act_forward_kernel_NHWC<scalar_t,act_op,4><<<blocksize,128,0,stream>>>(
                                  self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,N * G * cols,G,op);
      }
      else if(D%2==0){
        act_groupnorm_forward_mean_rstd_NHWC<scalar_t,PassThrough<scalar_t>,2><<<dim3(N , G), 512, 0, stream>>>(self_data,mean_data,rstd_data,cols,HxW,num_channels,eps,passop);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int blocksize=(N * G *cols/2-1)/128+1;
        groupnorm_act_forward_kernel_NHWC<scalar_t,act_op,2><<<blocksize,128,0,stream>>>(
                                  self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,N * G * cols,G,op);
      }
      else{
        act_groupnorm_forward_mean_rstd_NHWC<scalar_t,PassThrough<scalar_t>,1><<<dim3(N , G), 512, 0, stream>>>(self_data,mean_data,rstd_data,cols,HxW,num_channels,eps,passop);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int blocksize=(N * G *cols-1)/128+1;
        groupnorm_act_forward_kernel_NHWC<scalar_t,act_op,1><<<blocksize,128,0,stream>>>(
                                  self_data,ret_data,mean_data,rstd_data,weight_data,bias_data,cols,HxW,N * G * cols,G,op);
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t,class act_op> 
void groupnorm_act_forward_impl(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double gn_epsilon,
                                    bool training,act_op op,Tensor& ret,Tensor& mean,Tensor&rstd)
{
    AT_ASSERT(self.is_cuda(), "x must on HCU");
    AT_ASSERT(num_channels % num_groups == 0,"num_channels must be divisible by num_groups");
    AT_ASSERT(self.scalar_type()!=at::kDouble,"Don't surpport FP64 tensor");
    if(self.suggest_memory_format()==at::MemoryFormat::ChannelsLast){
      groupnorm_act_forward_NHWC<scalar_t,act_op>(self,weight,bias,num_groups,num_channels,gn_epsilon,training,op,ret,mean,rstd);
      return;
    }
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
    T_ACC  eps = gn_epsilon;
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
__global__ void GN_act_GammaBetaBackward_1d(int M,int N,int depth,int group,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,scalar_t* dg,scalar_t* db,scalar_t* gamma,scalar_t* beta,act_op op)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      const int index = i * N + j;
      const int ng =i * group + j / depth;
      const int jj=index%N;
      T_ACC x=(static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[ng])) * static_cast<T_ACC>(rstd[ng]);
      const T_ACC cmask_dY = op.backward(dY[index],x * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]));
      dg_sum += cmask_dY * x;
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
    if (j < N) {
        dg[j] = sum1;
        db[j] = sum2;
    }
  }
}

template <typename scalar_t,class act_op,int VEC=4>
__global__ void GroupNorm_act_Backward_1d_kernel_vec_fast(int N,int channel_size,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,
                                                              const scalar_t* beta,const scalar_t* mean,const scalar_t* rstd, scalar_t* dX,act_op op)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC b;
  __shared__ T_ACC c;
  const int bid=blockIdx.x;
  const int j = threadIdx.x;
  int group=channel_size/N;
  const int index = bid * N + j*VEC;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  int tcol=N/VEC;
  VecTpye tx;
  VecTpye tdy;
  VecTpye tdx;
  VecTpye tgamma;
  VecTpye tbeta;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* pdy=(scalar_t*)(&tdy);
  scalar_t* pdx=(scalar_t*)(&tdx);
  scalar_t* pgamma=(scalar_t*)(&tgamma);
  scalar_t* pbeta=(scalar_t*)(&tbeta);
  T_ACC tmean = static_cast<T_ACC>(mean[bid]);
  T_ACC trstd = static_cast<T_ACC>(rstd[bid]);
  int gb_index=index%channel_size;
  if(j<tcol){
    tx=*(VecTpye*)(X+index);
    tdy=*(VecTpye*)(dY+index);
    tgamma=*(VecTpye*)(gamma+gb_index);
    tbeta=*(VecTpye*)(beta+gb_index);
    for(int i=0;i<VEC;i++){
      pdy[i]=op.backward(pdy[i],(static_cast<T_ACC>(px[i]) - tmean) * trstd*pgamma[i]+pbeta[i]);
      sum1 += static_cast<T_ACC>(pdy[i]) * static_cast<T_ACC>(px[i])* pgamma[i];
      sum2 += static_cast<T_ACC>(pdy[i]) * pgamma[i];
    }
  }
  if(blockDim.x>64){
    __shared__ T_ACC ds_shared[8];
    __shared__ T_ACC db_shared[8];
    sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  else{
    sum1 = WarpReduceSum<T_ACC>(sum1);
    sum2 = WarpReduceSum<T_ACC>(sum2);
  }
  const T_ACC s = T_ACC(1) / (static_cast<T_ACC>(N));
  if (threadIdx.x == 0) {
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;
    b = a;
    c = -a * tmean - sum2 * trstd * s;
  }
  __syncthreads();
  if(j<tcol){
    for(int i=0;i<VEC;i++){
       pdx[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t,class act_op,int VEC=4>
__global__ void GN_act_ComputeInternalGradientsCUDAKernel(int HxW,int num_channels,int D,const scalar_t* dY,const scalar_t* X,acc_type<scalar_t, true>* ds,acc_type<scalar_t, true>* db,
                                                           const scalar_t* gamma,const scalar_t* beta,const scalar_t* mean,const scalar_t* rstd,act_op op) {
  using T_ACC = acc_type<scalar_t, true>;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*HxW;
  int tcol=HxW/VEC;
  VecTpye *x=(VecTpye*)(X+block_cols);
  VecTpye *dy=(VecTpye*)(dY+block_cols);
  VecTpye tx;
  VecTpye tdy;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* pdy=(scalar_t*)(&tdy);
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  int gb_index=blockidx%num_channels;
  T_ACC tgamma=gamma[gb_index];
  T_ACC tbeta=beta[gb_index];
  int tr_index=blockidx/D;
  T_ACC tmean=mean[tr_index];
  T_ACC trstd=rstd[tr_index];
  for (int hw = threadIdx.x; hw < tcol; hw += blockDim.x) {
    const int index = blockidx * HxW + hw*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    for(int i=0;i<VEC;i++){
      T_ACC dy=op.backward(pdy[i],(static_cast<T_ACC>(px[i]) - tmean) * trstd*tgamma+tbeta);
      sum1+=dy*static_cast<T_ACC>(px[i]);
      sum2+=dy;
    }
  }
  __shared__ T_ACC ds_shared[8];
  __shared__ T_ACC db_shared[8];
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  
  if (threadIdx.x == 0) {
    ds[blockidx] = sum1;
    db[blockidx] = sum2;
  }
}

template <typename scalar_t,class act_op,int VEC=4>
__global__ void groupnorm_act_backworward_kernel(scalar_t* dY_data,scalar_t* X_data,scalar_t* dX_data,scalar_t* gamma_data,scalar_t* beta_data,scalar_t* mean,scalar_t* rstd,int cols,int HxW,int N,
                                                int channel_size,acc_type<scalar_t, true>* c2_data,acc_type<scalar_t, true>* c3_data,act_op op)
{
  using T_ACC = acc_type<scalar_t, true>;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  VecTpye tx;
  VecTpye tdy;
  VecTpye tdx;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* pdy=(scalar_t*)(&tdy);
  scalar_t* pdx=(scalar_t*)(&tdx);
  int tid=(blockIdx.x*blockDim.x+threadIdx.x)*VEC;
  if(tid>=N)return;
  int tid_col=tid/cols;
  scalar_t b=c2_data[tid_col];
  scalar_t c=c3_data[tid_col];
  T_ACC tmean=mean[tid_col];
  T_ACC trstd=rstd[tid_col];
  int jj=tid%(HxW*channel_size)/HxW;
  T_ACC tgamma=static_cast<T_ACC>(gamma_data[jj]);
  T_ACC tbeta=static_cast<T_ACC>(beta_data[jj]);
  tx = *(VecTpye*)(X_data+tid);
  tdy = *(VecTpye*)(dY_data+tid);
  #pragma unroll
  for(int i=0;i<VEC;i++){
    T_ACC dy=op.backward(pdy[i],(static_cast<T_ACC>(px[i]) - tmean) * trstd*tgamma+tbeta);
    pdx[i]=trstd*dy*tgamma+b*static_cast<T_ACC>(px[i])+c;
  }
  *(VecTpye*)(dX_data+tid)=tdx;
}

template <typename scalar_t,class act_op,int VEC=4>
__global__ void GroupNormBackward_act_kernel_small_cols(int N,int HxW,int channel_size,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const scalar_t* beta,const scalar_t* mean,
                                           const scalar_t* rstd, scalar_t* dX,acc_type<scalar_t, true>* ds_data,acc_type<scalar_t, true>* db_data,act_op op)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC ds_shared[8];
  __shared__ T_ACC db_shared[8];
  __shared__ T_ACC b;
  __shared__ T_ACC c;
  const int bid=blockIdx.x;
  const int j = threadIdx.x;
  const int D = N/HxW;
  int group=channel_size/D;
  const int index = blockIdx.x * N + j*VEC;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  int tcol=N/VEC;
  VecTpye tx;
  VecTpye tdy;
  VecTpye tdx;
  scalar_t* px=(scalar_t*)(&tx);
  scalar_t* pdy=(scalar_t*)(&tdy);
  scalar_t* pdx=(scalar_t*)(&tdx);
  for (int ii = threadIdx.x; ii < D; ii += blockDim.x) {
    const int index = bid * D + ii;
    const int jj = (bid%group) * D + ii;
    const T_ACC gamma_v =gamma[jj];
    sum1 += ds_data[index] * gamma_v;
    sum2 += db_data[index] * gamma_v;
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  const T_ACC s = T_ACC(1) / (static_cast<T_ACC>(N));
  T_ACC trstd = static_cast<T_ACC>(rstd[bid]);
  T_ACC tmean = static_cast<T_ACC>(mean[bid]);
  if (threadIdx.x == 0) {
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;
    b = a;
    c = -a * tmean - sum2 * trstd * s;
  }
  __syncthreads();
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = bid * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    int jj=index%(HxW*channel_size)/HxW;
    T_ACC tgamma = static_cast<T_ACC>(gamma[jj]);
    T_ACC tbeta = static_cast<T_ACC>(beta[jj]);
    for(int i=0;i<VEC;i++){
      T_ACC dy=op.backward(pdy[i],(static_cast<T_ACC>(px[i]) - tmean) * trstd * tgamma + tbeta);
      pdx[i]=trstd*dy*tgamma+b*static_cast<T_ACC>(px[i])+c;
    }
    *(VecTpye*)(dX+index) = tdx;
  }
}

template <typename scalar_t,class act_op> 
void groupnorm_act_backward_impl(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma,const Tensor& beta,act_op op,Tensor& dX,Tensor& dgamma,Tensor& dbeta)
{
    const int nelem = grad.numel();
    const int N=grad.sizes().front();
    const int num_channels=gamma.numel();
    const int HxW=nelem/(N*num_channels);
    const int G = mean.numel()/N;
    const int D = num_channels / G;
    auto pgrad = grad.expect_contiguous();
    auto stream=at::cuda::getCurrentCUDAStream();
    using T_ACC = acc_type<scalar_t, true>;
    scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
    scalar_t* X_data = X.data_ptr<scalar_t>();
    scalar_t* mean_data = mean.data_ptr<scalar_t>();
    scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
    scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
    scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
    scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
    scalar_t* dX_data = dX.data_ptr<scalar_t>();
    scalar_t* beta_data=beta.data_ptr<scalar_t>();
    const int B =(num_channels - 1) / kColwiseReduceTileSize + 1;
    PassThrough<scalar_t> passop;
    if (HxW == 1) {//HxW==1,D usually <= 128
      GN_act_GammaBetaBackward_1d<scalar_t,act_op><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
                          N,num_channels,D,G,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data,gamma_data,beta_data,op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if(D%4==0){
        if(D<=2048){
          GroupNorm_act_Backward_1d_kernel_vec_fast<scalar_t,act_op,4><<<N * G, get_groupnorm_blocksize(D/4), 0, stream>>>(D,num_channels, dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,op);
        }
        else if(D<=4096 && D%8==0){
          GroupNorm_act_Backward_1d_kernel_vec_fast<scalar_t,act_op,8><<<N * G, get_groupnorm_blocksize(D/8), 0, stream>>>(D,num_channels, dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,op);
        }
        else if(D<=8192 && D%16==0){
          GroupNorm_act_Backward_1d_kernel_vec_fast<scalar_t,act_op,16><<<N * G, get_groupnorm_blocksize(D/16), 0, stream>>>(D,num_channels, dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,op);
        }
        else {
          AT_ASSERT(false, "lightop groupnorm_backward error! channel/group is too big");
        }
      }
      else{
        AT_ASSERT(D<=512, "lightop groupnorm_backward error! channel/group is too big");
        GroupNorm_act_Backward_1d_kernel_vec_fast<scalar_t,act_op,1><<<N * G, get_groupnorm_blocksize(D), 0, stream>>>(D,num_channels, dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,op);
      }
    } 
    else{
        const auto kAccType =(X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)? kFloat: X.scalar_type();
        Tensor ds = at::empty({N, num_channels}, grad.options().dtype(kAccType));
        Tensor db = at::empty({N, num_channels}, grad.options().dtype(kAccType));
        T_ACC* ds_data = ds.data_ptr<T_ACC>();
        T_ACC* db_data = db.data_ptr<T_ACC>();
        if(HxW%4==0)
        { 
            GN_act_ComputeInternalGradientsCUDAKernel<scalar_t,act_op,4><<<N * num_channels, get_groupnorm_blocksize(HxW/4), 0, stream>>>(
            HxW,num_channels,D, dY_data, X_data, ds_data, db_data,gamma_data,beta_data,mean_data,rstd_data,op);
        }
        else
        {
            GN_act_ComputeInternalGradientsCUDAKernel<scalar_t,act_op,1><<<N * num_channels, get_groupnorm_blocksize(HxW), 0, stream>>>(
            HxW,num_channels,D, dY_data, X_data, ds_data, db_data,gamma_data,beta_data,mean_data,rstd_data,op);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        GN_GammaBetaBackward<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
                            N,num_channels,D,G,mean_data,rstd_data,dgamma_data,dbeta_data,ds_data,db_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int cols=D*HxW;
        if(cols>8192){
            Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
            Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
            int num_threads = (num_channels / G) < 512? 64: 512;
            T_ACC* c2_data = c2.data_ptr<T_ACC>();
            T_ACC* c3_data = c3.data_ptr<T_ACC>();
            ComputeBackwardFusedParamsCUDAKernel<scalar_t><<<dim3(N, G), num_threads, 0, stream>>>(num_channels,HxW,G, mean_data, rstd_data,gamma_data, ds_data, db_data, c2_data,c3_data);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if(cols%4==0){
              int blocksize=(N * G *cols/4-1)/128+1;
              groupnorm_act_backworward_kernel<scalar_t,act_op,4><<<blocksize,128,0,stream>>>(dY_data,X_data,dX_data,gamma_data,beta_data,mean_data,rstd_data,cols,HxW,N * G * cols,num_channels,c2_data,c3_data,op);
            }
            else{
              int blocksize=(N * G *cols-1)/128+1;
              groupnorm_act_backworward_kernel<scalar_t,act_op,1><<<blocksize,128,0,stream>>>(dY_data,X_data,dX_data,gamma_data,beta_data,mean_data,rstd_data,cols,HxW,N * G * cols,num_channels,c2_data,c3_data,op);
            }
        }
        else{
          if (cols % 4 == 0){
              GroupNormBackward_act_kernel_small_cols<scalar_t,act_op,4><<<N * G, get_groupnorm_blocksize(cols/4), 0, stream>>>(cols,HxW,num_channels,dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,ds_data,db_data,op);
          }
          else{
              GroupNormBackward_act_kernel_small_cols<scalar_t,act_op,1><<<N * G, get_groupnorm_blocksize(cols), 0, stream>>>(cols,HxW,num_channels, dY_data, X_data,gamma_data,beta_data,mean_data,rstd_data,dX_data,ds_data,db_data,op);
          }
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
}

