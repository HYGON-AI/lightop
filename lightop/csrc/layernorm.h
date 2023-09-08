#include "reduce.h"

using namespace std;

namespace at{
namespace native{

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      combine_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct WelfordOps {
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    return res_t(acc.m2 / acc.nf, static_cast<scalar_t>(acc.mean));
  }

  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      __shfl_down(acc.mean, offset)
      , __shfl_down(acc.m2, offset)
      , __shfl_down(acc.n, offset)
      , __shfl_down(acc.nf, offset)
    };
  }
};


template <typename scalar_t,typename T_ACC,int VEC>
__global__ void GammaBetaBackwardSimple(int length,int stride,scalar_t* dY,scalar_t* X,scalar_t* mean,
             scalar_t* rstd,T_ACC* block_gamma_gpu,T_ACC* block_beta_gpu,int *semaphores,scalar_t* dg,scalar_t* db)
{
    __shared__ int islast;
    __shared__ T_ACC smem[REDUCE_BLOCK_SIZE/C10_WARP_SIZE];
    using VecTpyeS=memory::aligned_vector<scalar_t, VEC>;
    using VecTpyeT=memory::aligned_vector<T_ACC, VEC>;
    const int tid=threadIdx.x;
    const int bidx=blockIdx.x;
    const int bidy=blockIdx.y;
    const int gdimx=gridDim.x;
    const int block_size=blockDim.x;
    int idx=(bidx*block_size+tid)*stride+bidy*VEC;
    int offset=block_size*gdimx*stride;
    T_ACC block_gamma_sum[VEC]={0};
    T_ACC block_beta_sum[VEC]={0};
    scalar_t gamma_temp[VEC];
    scalar_t beta_temp[VEC];
    //scalar_t mean_temp[VEC];
    //scalar_t rstd_temp[VEC];
    for(int i=idx;i<length;i+=offset){
        int j=i/stride;
        *(VecTpyeS*)gamma_temp=*(VecTpyeS*)(X+i);
        *(VecTpyeS*)beta_temp=*(VecTpyeS*)(dY+i);
        //*(VecTpyeS*)mean_temp=*(VecTpyeS*)(mean+j);
        //*(VecTpyeS*)rstd_temp=*(VecTpyeS*)(rstd+j);
        #pragma unroll
        for(int i=0;i<VEC;i++){
            block_gamma_sum[i]+=static_cast<T_ACC>(beta_temp[i]) * (static_cast<T_ACC>(gamma_temp[i]) - static_cast<T_ACC>(mean[j])) * static_cast<T_ACC>(rstd[j]);
            block_beta_sum[i]+=beta_temp[i];
        }
    }
    if(semaphores==NULL){
        #pragma unroll
        for(int i=0;i<VEC;i++){
            gamma_temp[i]=BlockReduceSum(block_gamma_sum[i],smem);
            beta_temp[i]=BlockReduceSum(block_beta_sum[i],smem);
        }
        if(tid==0) {
            *(VecTpyeS*)(dg+bidy*VEC)=*(VecTpyeS*)gamma_temp;
            *(VecTpyeS*)(db+bidy*VEC)=*(VecTpyeS*)beta_temp;
        }
        return;
    }
    #pragma unroll
    for(int i=0;i<VEC;i++){
        block_gamma_sum[i]=BlockReduceSum(block_gamma_sum[i],smem);
        block_beta_sum[i]=BlockReduceSum(block_beta_sum[i],smem);
    }
    if(tid==0){
        *(VecTpyeT*)(block_gamma_gpu+bidx*stride+bidy*VEC)=*(VecTpyeT*)block_gamma_sum;
        *(VecTpyeT*)(block_beta_gpu+bidx*stride+bidy*VEC)=*(VecTpyeT*)block_beta_sum;
        __threadfence();
        int value=atomicAdd(semaphores+bidy,1);
        islast=(value==gdimx-1);
    }
    __syncthreads();
    if(islast){
        T_ACC s_gamma[VEC]={0};
        T_ACC s_beta[VEC]={0};
        T_ACC t_gamma[VEC];
        T_ACC t_beta[VEC];
        scalar_t gamma_out[VEC];
        scalar_t beta_out[VEC];
        for(int i=tid;i<gdimx;i+=block_size){
            *(VecTpyeT*)t_gamma=*(VecTpyeT*)(block_gamma_gpu+i*stride+bidy*VEC);
            *(VecTpyeT*)t_beta=*(VecTpyeT*)(block_beta_gpu+i*stride+bidy*VEC);
            #pragma unroll
            for(int k=0;k<VEC;k++){
                s_gamma[k]+=t_gamma[k];
                s_beta[k]+=t_beta[k];
            }
        }
        if(gdimx<=64){
            #pragma unroll
            for(int i=0;i<VEC;i++){
                gamma_out[i]=WarpReduceSum(s_gamma[i]);
                beta_out[i]=WarpReduceSum(s_beta[i]);
            }
        }
        else{
            #pragma unroll
            for(int i=0;i<VEC;i++){
                gamma_out[i]=BlockReduceSum(s_gamma[i],smem);
                beta_out[i]=BlockReduceSum(s_beta[i],smem);
            }
        }
        if(tid==0) 
        {
            *(VecTpyeS*)(dg+bidy*VEC)=*(VecTpyeS*)gamma_out;
            *(VecTpyeS*)(db+bidy*VEC)=*(VecTpyeS*)beta_out;
        }
        
    }


}


template <typename scalar_t>
__global__ void GammaBetaBackward(int M,int N,const scalar_t* dY,const scalar_t* X,const scalar_t* mean,
                                  const scalar_t* rstd,scalar_t* dg,scalar_t* db)
{
  using T_ACC = acc_type<scalar_t, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    for (int i = threadIdx.y; i < M; i += blockDim.y) {
      
      //#pragma clang fp contract(off) // if need same output with pytorch ,then enable this line.
      const int index = i * N + j;
      const T_ACC cmask_dY = dY[index];
      dg_sum += cmask_dY * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
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

#define GammaBetaVEC_KERNEL(VEC)                                                                                                             \
    if(block_num*stride/VEC>maxblocks)block_num=(maxblocks-1)/(stride/VEC)+1;                                                                \
    if(block_num>32){                                                                                                                        \
        c10::TensorOptions options(DeviceType::CUDA);                                                                                        \
        options.dtype(ScalarType::Int);                                                                                                      \
        Tensor buffer_gamma=torch::empty(stride*block_num, options);                                                                         \
        Tensor buffer_beta=torch::empty(stride*block_num, options);                                                                          \
        Tensor semaphores=torch::empty(stride/VEC, options);                                                                                 \
        AT_CUDA_CHECK(cudaMemsetAsync(semaphores.data_ptr(), 0, stride/VEC*sizeof(int), stream));                                            \
        GammaBetaBackwardSimple<scalar_t,T_ACC,VEC><<<dim3(block_num,stride/VEC),block_size,0,stream>>>                                      \
         (length,stride,dY_data,X_data,mean_data,rstd_data,(T_ACC*)buffer_gamma.data_ptr(),(T_ACC*)buffer_beta.data_ptr(),                   \
          (int*)semaphores.data_ptr() ,dgamma_data,dbeta_data);                                                                              \
    }                                                                                                                                        \
    else {                                                                                                                                   \
        GammaBetaBackwardSimple<scalar_t,T_ACC,VEC><<<dim3(1,stride/VEC),block_size,0,stream>>>                                              \
        (length,stride,dY_data,X_data,mean_data,rstd_data,(T_ACC*)NULL,(T_ACC*)NULL,(int*)NULL ,dgamma_data,dbeta_data);                     \
    }                                                                                                                                        \


template <typename scalar_t>
void GammaBeta_Backward(int length,int stride,scalar_t* dY_data,scalar_t* X_data,scalar_t* mean_data,scalar_t* rstd_data,scalar_t* dgamma_data,scalar_t* dbeta_data)
{
    using T_ACC = acc_type<scalar_t, true>;
    auto stream = at::cuda::getCurrentCUDAStream();
    int N=length/stride;
    if(stride<=512||N<512){
        constexpr int block_size=REDUCE_BLOCK_SIZE;
        int block_num=((N-1)/block_size+1);
        int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
        int maxblocks=at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;
        if(stride%4!=0){
            GammaBetaVEC_KERNEL(1)
        }
        else if(stride==512 && N>=2000){
            GammaBetaVEC_KERNEL(8)
        }
        else{
            GammaBetaVEC_KERNEL(4)
        }
    }
    else{
        const int B =(stride - 1) / kColwiseReduceTileSize + 1;
        GammaBetaBackward<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(
                       N,stride,dY_data,X_data,mean_data,rstd_data,dgamma_data,dbeta_data);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}





}//namespace native
}//namesapace at




#define LAYERNORMBACKWARD                                                                \
  using T_ACC = acc_type<scalar_t, true>;                                                \
  __shared__ T_ACC ds_shared[8];                                                         \
  __shared__ T_ACC db_shared[8];                                                         \
 const int i = blockIdx.x;                                                               \
  T_ACC sum1 = 0;                                                                        \
  T_ACC sum2 = 0;                                                                        \
  for (int j = threadIdx.x; j < N; j += blockDim.x) {                                    \
    const int index = i * N + j;                                                         \
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);                                  \
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;      \
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;                                     \
  }                                                                                      \
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                         \
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                         \
  __shared__ T_ACC b;                                                                    \
  __shared__ T_ACC c;                                                                    \
  if (threadIdx.x == 0) {                                                                \
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                    \
    T_ACC tmean = static_cast<T_ACC>(mean[i]);                                           \
    T_ACC trstd = static_cast<T_ACC>(rstd[i]);                                           \
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                   \
    b = a;                                                                               \
    c = -a * tmean - sum2 * trstd * s;                                                   \
  }                                                                                      \
  __syncthreads();


int get_layernorm_blocksize(int cols);