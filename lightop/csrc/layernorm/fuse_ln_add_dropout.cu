#include "layernorm.h"

namespace at{
namespace native{

namespace {
  constexpr int VEC = 4;
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_val_2048(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* bias,scalar_t* gamma,scalar_t* beta,int cols,float eps)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t residual4[VEC];
  T_ACC tmean;
  T_ACC trstd;
  int idx = i * tcol + j;
  if(j<tcol){
    idx*=VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=j*VEC+ii;
      input4[ii]+=bias[jj]+residual4[ii];
      val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (j == 0) {
      //layernorm get mean and rstd
      thrust::tie(tmean, s_mean) = welford_op.project(val);
      s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  if(j<tcol){
    tmean=s_mean;
    trstd=s_rstd;
    //layernorm  (x-mean)*rstd*gamma+beta
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii;
      output4[ii] = (static_cast<T_ACC>(input4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_val_4096(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* bias,scalar_t* gamma,scalar_t* beta,int cols,float eps)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  scalar_t input4[VEC];
  scalar_t input8[VEC];
  scalar_t output4[VEC];
  scalar_t output8[VEC];
  scalar_t residual4[VEC];
  scalar_t residual8[VEC];
  T_ACC tmean;
  T_ACC trstd;
  int idx1 = i * tcol + j;
  int idx2 = i * tcol + j + 512;
  idx1*=VEC;
  *(LoadT*)input4 = *(LoadT*)(input+idx1);
  *(LoadT*)residual4 = *(LoadT*)(residual+idx1);
  #pragma unroll
  for (int ii = 0; ii < VEC; ii++) {
    int jj=j*VEC+ii;
    input4[ii]+=bias[jj]+residual4[ii];
    val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]));
  }
  if(j<tcol-512){
    idx2*=VEC;
    *(LoadT*)input8 = *(LoadT*)(input+idx2);
    *(LoadT*)residual8 = *(LoadT*)(residual+idx2);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=j*VEC+ii+2048;
      input8[ii]+=bias[jj]+residual8[ii];
      val = welford_op.reduce(val, static_cast<T_ACC>(input8[ii]));
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
  #pragma unroll
  for(int ii=0;ii<VEC;ii++){
    int jj=j*VEC+ii;
    output4[ii] = (static_cast<T_ACC>(input4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
  }
  *(LoadT*)(ret+idx1)=*(LoadT*)output4;
  if(j<tcol-512){
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii+2048;
      output8[ii] = (static_cast<T_ACC>(input8[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(ret+idx2)=*(LoadT*)output8;
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_val(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* bias,scalar_t* gamma,scalar_t* beta,int cols,float eps)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t mask4[VEC];
  scalar_t residual4[VEC];
  T_ACC tmean;
  T_ACC trstd;
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = i * tcol + off;
    idx*=VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=off*VEC+ii;
      val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]+bias[jj]+residual4[ii]));
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
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = (i * tcol + off)*VEC;
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    //layernorm  (x-mean)*rstd*gamma+beta
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=off*VEC+ii;
      output4[ii] = (static_cast<T_ACC>(input4[ii]+bias[jj]+residual4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
}


template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_train_2048(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retdrop_data,float* mean,float* rstd,scalar_t* bias,scalar_t* gamma,
                                              scalar_t* beta,scalar_t* mask,int cols,T_ACC p,float eps,PhiloxCudaState philox_args,int rand_offset)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  T_ACC pinv = T_ACC(1)/p;
  auto seeds = at::cuda::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t output4[VEC];
  scalar_t mask4[VEC];
  scalar_t residual4[VEC];
  T_ACC tmean;
  T_ACC trstd;
  int idx = i * tcol + j;
  if(j<tcol){
    hiprand_init(std::get<0>(seeds),idx%rand_offset,std::get<1>(seeds)+idx/rand_offset*VEC,&state);
    idx*=VEC;
    *(float4*)rand = hiprand_uniform4(&state);
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=j*VEC+ii;
      mask4[ii] = (rand[ii]<p)*pinv;
      input4[ii] = (input4[ii]+bias[jj])*mask4[ii]+residual4[ii];
      val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]));
    }
  }
  val = BlockReduce(val,welford_op,val_shared_ptr);
  if (j == 0) {
      //layernorm get mean and rstd
      thrust::tie(tmean, s_mean) = welford_op.project(val);
      s_rstd=c10::cuda::compat::rsqrt(tmean + eps);
  }
  __syncthreads();
  if(j<tcol){
    tmean=s_mean;
    trstd=s_rstd;
    //layernorm  (x-mean)*rstd*gamma+beta
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii;
      output4[ii] = (static_cast<T_ACC>(input4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(retdrop_data+idx) = *(LoadT*)input4;
    *(LoadT*)(mask+idx) = *(LoadT*)mask4;
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
    if(threadIdx.x==0){
      mean[i] = tmean;
      rstd[i] = trstd;
    }
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_train_4096(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retdrop_data,float* mean,float* rstd,scalar_t* bias,scalar_t* gamma,
                                              scalar_t* beta,scalar_t* mask,int cols,T_ACC p,float eps,PhiloxCudaState philox_args,int rand_offset)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  T_ACC pinv = T_ACC(1)/p;
  auto seeds = at::cuda::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t input8[VEC];
  scalar_t output4[VEC];
  scalar_t output8[VEC];
  scalar_t mask4[VEC];
  scalar_t mask8[VEC];
  scalar_t residual4[VEC];
  scalar_t residual8[VEC];
  T_ACC tmean;
  T_ACC trstd;
  int idx1 = i * tcol + j;
  int idx2 = i * tcol + j + 512;
  hiprand_init(std::get<0>(seeds),idx1%rand_offset,std::get<1>(seeds)+idx1/rand_offset*VEC,&state);
  idx1*=VEC;
  *(float4*)rand = hiprand_uniform4(&state);
  *(LoadT*)input4 = *(LoadT*)(input+idx1);
  *(LoadT*)residual4 = *(LoadT*)(residual+idx1);
  #pragma unroll
  for (int ii = 0; ii < VEC; ii++) {
    int jj=j*VEC+ii;
    mask4[ii] = (rand[ii]<p)*pinv;
    input4[ii] = (input4[ii]+bias[jj])*mask4[ii]+residual4[ii];
    val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]));
  }
  if(j<tcol-512){
    hiprand_init(std::get<0>(seeds),idx2%rand_offset,std::get<1>(seeds)+idx2/rand_offset*VEC,&state);
    idx2*=VEC;
    *(float4*)rand = hiprand_uniform4(&state);
    *(LoadT*)input8 = *(LoadT*)(input+idx2);
    *(LoadT*)residual8 = *(LoadT*)(residual+idx2);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=j*VEC+ii+2048;
      mask8[ii] = (rand[ii]<p)*pinv;
      input8[ii] = (input8[ii]+bias[jj])*mask8[ii]+residual8[ii];
      val = welford_op.reduce(val, static_cast<T_ACC>(input8[ii]));
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
  #pragma unroll
  for(int ii=0;ii<VEC;ii++){
    int jj=j*VEC+ii;
    output4[ii] = (static_cast<T_ACC>(input4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
  }
  *(LoadT*)(retdrop_data+idx1) = *(LoadT*)input4;
  *(LoadT*)(mask+idx1) = *(LoadT*)mask4;
  *(LoadT*)(ret+idx1)=*(LoadT*)output4;
  if(j<tcol-512){
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=j*VEC+ii+2048;
      output8[ii] = (static_cast<T_ACC>(input8[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(retdrop_data+idx2) = *(LoadT*)input8;
    *(LoadT*)(mask+idx2) = *(LoadT*)mask8;
    *(LoadT*)(ret+idx2)=*(LoadT*)output8;
  }
  if(j==0){
    mean[i] = tmean;
    rstd[i] = trstd;
  }
}

template <typename scalar_t,typename T_ACC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void fused_ln_add_dropout_bias_kernel_train(scalar_t* input,scalar_t* residual,scalar_t* ret,scalar_t* retdrop_data,float* mean,float* rstd,scalar_t* bias,scalar_t* gamma,
                                         scalar_t* beta,scalar_t* mask,int cols,T_ACC p,float eps,PhiloxCudaState philox_args,int rand_offset)
{
  using WelfordType = WelfordData<T_ACC, int, T_ACC>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[8];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  WelfordOp welford_op;
  WelfordType val;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/VEC;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  T_ACC pinv = T_ACC(1)/p;
  auto seeds = at::cuda::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  float rand[VEC];
  scalar_t input4[VEC];
  scalar_t residual4[VEC];
  scalar_t output4[VEC];
  scalar_t mask4[VEC];
  T_ACC tmean;
  T_ACC trstd;
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = i * tcol + off;
    hiprand_init(std::get<0>(seeds),idx%rand_offset,std::get<1>(seeds)+idx/rand_offset*VEC,&state);
    idx*=VEC;
    *(float4*)rand = hiprand_uniform4(&state);
    *(LoadT*)input4 = *(LoadT*)(input+idx);
    *(LoadT*)residual4 = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      int jj=off*VEC+ii;
      mask4[ii] = (rand[ii]<p)*pinv;
      input4[ii] = (input4[ii]+bias[jj])*mask4[ii]+residual4[ii];
      val = welford_op.reduce(val, static_cast<T_ACC>(input4[ii]));
    }
    *(LoadT*)(retdrop_data+idx) = *(LoadT*)input4;
    *(LoadT*)(mask+idx) = *(LoadT*)mask4;
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
  for(int off=j;off<tcol;off+=blockDim.x){
    int idx = (i * tcol + off)*VEC;
    *(LoadT*)input4 = *(LoadT*)(retdrop_data+idx);
    //layernorm  (x-mean)*rstd*gamma+beta
    #pragma unroll
    for(int ii=0;ii<VEC;ii++){
      int jj=off*VEC+ii;
      output4[ii] = (static_cast<T_ACC>(input4[ii]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
  if(threadIdx.x==0){
    mean[i] = tmean;
    rstd[i] = trstd;
  }
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
fused_ln_add_dropout_bias_forward(const Tensor& self,const Tensor& other,const Tensor& attention_bias,const Tensor& weight,const Tensor& bias,
                             double ln_epsilon, double p, bool training)
{
    AT_ASSERTM(self.sizes() == other.sizes(), "x must be the same size as y");
    AT_ASSERTM(p >=0.0f&&p<1.0f, "p mast be between 0 and 1");
    AT_ASSERTM(self.is_cuda()&&other.is_cuda(), "x and y must on dcu");
    const int cols=self.sizes().back();
    AT_ASSERT(cols>=64,"cols must >= 64");
    AT_ASSERT(cols%4==0,"cols%4 must be 0");
    const int nelem = self.numel();
    const int rows=nelem/cols;
    Tensor ret = at::empty_like(self);
    Tensor retdrop;
    Tensor mean;
    Tensor rstd;
    PhiloxCudaState rng_engine_inputs;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_ln_add_dropout_bias_forward",
      [&] {
        p=1-p;
        using T_ACC = acc_type<scalar_t, true>;
        T_ACC pa = (T_ACC)(p);
        float eps = ln_epsilon;
        scalar_t* self_data = self.data_ptr<scalar_t>();
        scalar_t* other_data =other.data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* weight_data=weight.data_ptr<scalar_t>();
        scalar_t* bias_data=bias.data_ptr<scalar_t>();
        scalar_t* attention_bias_data=attention_bias.data_ptr<scalar_t>();
        if(training)
        {
          retdrop = at::empty({nelem*2},self.options());
          mean = at::empty({rows},self.options().dtype(at::ScalarType::Float));
          rstd = at::empty({rows},self.options().dtype(at::ScalarType::Float));
          scalar_t* retdrop_data=retdrop.data_ptr<scalar_t>();
          float* mean_data=mean.data_ptr<float>();
          float* rstd_data=rstd.data_ptr<float>();
          auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
          int rand_offset=at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor*
                                    at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
          int gridx=((nelem + 256 -1)/256);
          gridx=std::min(rand_offset/256, gridx);
          int64_t counter_offset = ((nelem - 1)/(1024*gridx)+1)*4;
          PhiloxCudaState rng_engine_inputs;
          {
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_cuda_state(counter_offset);
          }
          if(cols<=2048){
            fused_ln_add_dropout_bias_kernel_train_2048<scalar_t,T_ACC><<<rows, get_layernorm_blocksize(cols/4), 0, stream>>>(self_data,other_data,ret_data,
                retdrop_data,mean_data,rstd_data,attention_bias_data,weight_data,bias_data,retdrop_data+nelem,cols,pa,eps,rng_engine_inputs,rand_offset);
          }
          else if(cols<=4096){
            fused_ln_add_dropout_bias_kernel_train_4096<scalar_t,T_ACC><<<rows, 512, 0, stream>>>(self_data,other_data,ret_data,
                retdrop_data,mean_data,rstd_data,attention_bias_data,weight_data,bias_data,retdrop_data+nelem,cols,pa,eps,rng_engine_inputs,rand_offset);
          }
          else{
            fused_ln_add_dropout_bias_kernel_train<scalar_t,T_ACC><<<rows,512, 0, stream>>>(self_data,other_data,ret_data,
                retdrop_data,mean_data,rstd_data,attention_bias_data,weight_data,bias_data,retdrop_data+nelem,cols,pa,eps,rng_engine_inputs,rand_offset);
          }
        }
        else {
          if(cols<=2048){
            fused_ln_add_dropout_bias_kernel_val_2048<scalar_t,T_ACC><<<rows,  get_layernorm_blocksize(cols/4), 0, stream>>>(self_data,other_data,ret_data,attention_bias_data,weight_data,bias_data,cols,eps);
          }
          else if(cols<=4096){
            fused_ln_add_dropout_bias_kernel_val_4096<scalar_t,T_ACC><<<rows,  512, 0, stream>>>(self_data,other_data,ret_data,attention_bias_data,weight_data,bias_data,cols,eps);
          }
          else{
            fused_ln_add_dropout_bias_kernel_val<scalar_t,T_ACC><<<rows, 512, 0, stream>>>(self_data,other_data,ret_data,attention_bias_data,weight_data,bias_data,cols,eps);
          }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
     return std::tuple<Tensor,Tensor,Tensor,Tensor>(ret,retdrop,mean,rstd);
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_2048(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2,scalar_t* mask)
{
  LAYERNORMBACKWARD_VEC_2048
  VecTpye tmask;
  scalar_t* pmask=(scalar_t*)(&tmask);
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  if (j < tcol) {
    tmask = *(VecTpye*)(mask+index); 
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
      pdx[i]=pdx2[i]*pmask[i];
    }
    *(VecTpye*)(dX1+index) = tdx;
    *(VecTpye*)(dX2+index) = tdx2;
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec_4096(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2,scalar_t* mask)
{
  LAYERNORMBACKWARD_VEC_4096
  VecTpye tmask;
  VecTpye tmask_2;
  scalar_t* pmask=(scalar_t*)(&tmask);
  scalar_t* pmask_2=(scalar_t*)(&tmask_2);
  tmask = *(VecTpye*)(mask+index);
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  #pragma unroll
  for(int i=0;i<VEC;i++){
    pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
    pdx[i]=pdx2[i]*pmask[i];
  }
  *(VecTpye*)(dX1+index) = tdx;
  *(VecTpye*)(dX2+index) = tdx2;
  if (j < tcol-512) {
    tmask_2 = *(VecTpye*)(mask+index_2); 
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy_2[i])*static_cast<T_ACC>(pgamma_2[i])+b*static_cast<T_ACC>(px_2[i])+c;
      pdx[i]=pdx2[i]*pmask_2[i];
    }
    *(VecTpye*)(dX1+index_2) = tdx;
    *(VecTpye*)(dX2+index_2) = tdx2;
  }
}

template <typename scalar_t>
__global__ void LayerNormBackward_kernel_vec(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const float* mean,
                                         const float* rstd, scalar_t* dX1,scalar_t* dX2,scalar_t* mask)
{
  LAYERNORMBACKWARD_VEC
  VecTpye tmask;
  scalar_t* pmask=(scalar_t*)(&tmask);
  VecTpye tdx2;
  scalar_t* pdx2=(scalar_t*)(&tdx2);
  for(int off=j;off<tcol;off+=blockDim.x){
    const int index = i * N + off*VEC;
    tx = *(VecTpye*)(X+index);
    tdy = *(VecTpye*)(dY+index);
    tgamma = *(VecTpye*)(gamma+off*VEC);
    tmask = *(VecTpye*)(mask+index);
    #pragma unroll
    for(int i=0;i<VEC;i++){
      pdx2[i]=trstd*static_cast<T_ACC>(pdy[i])*static_cast<T_ACC>(pgamma[i])+b*static_cast<T_ACC>(px[i])+c;
      pdx[i]=pdx2[i]*static_cast<T_ACC>(pmask[i]);
    }
    *(VecTpye*)(dX1+index) = tdx;
    *(VecTpye*)(dX2+index) = tdx2;
  }
}


std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>
fused_ln_add_dropout_bias_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  int N=grad.sizes().back();
  int nelem=grad.numel();
  int M=grad.numel()/N;
  auto pgrad = grad.expect_contiguous();
  Tensor dX1 = at::empty_like(*pgrad);
  Tensor dX2 = at::empty_like(dX1);
  Tensor dgamma = at::empty_like(gamma);
  Tensor dbeta = at::empty_like(gamma);
  Tensor dbias = at::empty_like(gamma);
  auto cuda_stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "fused_ln_add_dropout_bias_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* X_data = X.data_ptr<scalar_t>();
      float* mean_data = mean.data_ptr<float>();
      float* rstd_data = rstd.data_ptr<float>();
      scalar_t* gamma_data = gamma.data_ptr<scalar_t>();
      scalar_t* dgamma_data = dgamma.data_ptr<scalar_t>();
      scalar_t* dbeta_data = dbeta.data_ptr<scalar_t>();
      scalar_t* dX1_data = dX1.data_ptr<scalar_t>();
      scalar_t* dX2_data = dX2.data_ptr<scalar_t>();
      scalar_t* dbias_data = dbias.data_ptr<scalar_t>();
      
      GammaBeta_Backward(nelem,N,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
      if(N<=2048){
        LayerNormBackward_kernel_vec_2048<scalar_t><<<M, get_layernorm_blocksize(N/4), 0, cuda_stream>>>(
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX1_data,dX2_data,X_data+grad.numel());
      }
      else if(N<=4096){
        LayerNormBackward_kernel_vec_4096<scalar_t><<<M, 512, 0, cuda_stream>>>(
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX1_data,dX2_data,X_data+grad.numel());
      }
      else{
        LayerNormBackward_kernel_vec<scalar_t><<<M, 512, 0, cuda_stream>>>(
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX1_data,dX2_data,X_data+grad.numel());
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      bias_reduce_sum(dbias_data,dX1_data,nelem,N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>(dX1,dX2,dbias,dgamma,dbeta);
}
class fuselnAddDropoutBiasFunction : public torch::autograd::Function<fuselnAddDropoutBiasFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& y,
    const torch::autograd::Variable& attention_bias,
    const torch::autograd::Variable& weight,
    const torch::autograd::Variable& bias,
    double eps,double rate,bool training) {
    auto px=x.expect_contiguous();
    auto py=y.expect_contiguous();
    auto pweight=weight.expect_contiguous();
    auto pbias=bias.expect_contiguous();
    auto ret = fused_ln_add_dropout_bias_forward(*px,*py,attention_bias,*pweight,*pbias,eps,rate,training);
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
    auto ret = fused_ln_add_dropout_bias_backward(gradOutput[0],X,mean,rstd,gamma);
    return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),std::get<3>(ret),std::get<4>(ret),
            torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};
Tensor fuse_ln_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& attention_bias,const Tensor& weight,
                                            const Tensor& bias,double eps,double rate,bool training) {
   auto result = fuselnAddDropoutBiasFunction::apply(x,y,attention_bias,weight,bias,eps,rate,training);
   return result[0];
}
}
}

