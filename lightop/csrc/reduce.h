#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/autograd.h>
#include <THC/THCDeviceUtils.cuh>
namespace at{
namespace native{

constexpr int kColwiseReduceTileSize=32;
constexpr int REDUCE_BLOCK_SIZE=256;

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val,const ReduceOp& op,int max=32) {
  for (int offset = max; offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  const int block_size=blockDim.x;
  const int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduce(val, op);
  if(block_size==64)return val;
  __syncthreads();
  if (lid == 0&&wid<share_size) {
    shared[wid] = val;
  }
  __syncthreads();
  if (wid == 0&&lid<share_size) {
    val= shared[lid];
    val = WarpReduce(val,op,share_size/2);
  }
  return val;
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val,int max=32) {
  for (int offset = max; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  const int block_size=blockDim.x;
  const int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceSum(val);
  if(block_size==C10_WARP_SIZE)return val;
  __syncthreads();
  if (lid == 0&&wid<share_size) {
    shared[wid] = val;
  }
  __syncthreads();
  if (wid == 0&&lid<share_size) {
    val= shared[lid];
    val = WarpReduceSum(val,share_size/2);
  }
  return val;
}



template <typename scalar_t>
__global__ void col_wise_reduce(scalar_t *dst, const scalar_t *src,int M,int N){
    using T_ACC = acc_type<scalar_t, true>;
    __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize+1];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    T_ACC grad_sum = 0;
    if (j < N) {
        for (int i = threadIdx.y; i < M; i += blockDim.y) {
            grad_sum += src[i * N + j];
        }
    }
    g_shared[threadIdx.y][threadIdx.x] = grad_sum;
    __syncthreads();
    T_ACC sum = g_shared[threadIdx.x][threadIdx.y];
    sum = WarpReduceSum(sum,kColwiseReduceTileSize/2);
    if (threadIdx.x == 0) {
        const int j = blockIdx.x * blockDim.x + threadIdx.y;
        if (j < N) {
            dst[j] = sum;
        }
   }
}

template <typename scalar_t,typename T_ACC,int VEC>
__global__ void ReduceVEC(scalar_t* data_gpu,T_ACC* block_gpu,int *semaphores,scalar_t *sum_gpu,int length,int stride)
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
    T_ACC block_sum[VEC]={0};
    scalar_t sum_temp[VEC];
    for(int i=idx;i<length;i+=offset){
        *(VecTpyeS*)sum_temp=*(VecTpyeS*)(data_gpu+i);
        #pragma unroll
        for(int i=0;i<VEC;i++){
            block_sum[i]+=sum_temp[i];
        }
    }
    if(semaphores==NULL){
        #pragma unroll
        for(int i=0;i<VEC;i++){
            sum_temp[i]=BlockReduceSum(block_sum[i],smem);
        }
        if(tid==0) *(VecTpyeS*)(sum_gpu+bidy*VEC)=*(VecTpyeS*)sum_temp;
        return;
    }
    #pragma unroll
    for(int i=0;i<VEC;i++){
        block_sum[i]=BlockReduceSum(block_sum[i],smem);
    }
    if(tid==0){
        *(VecTpyeT*)(block_gpu+bidx*stride+bidy*VEC)=*(VecTpyeT*)block_sum;
        __threadfence();
        int value=atomicAdd(semaphores+bidy,1);
        islast=(value==gdimx-1);
    }
    __syncthreads();
    if(islast){
        T_ACC s[VEC]={0};
        T_ACC t[VEC];
        scalar_t t_out[VEC];
        for(int i=tid;i<gdimx;i+=block_size){
            *(VecTpyeT*)t=*(VecTpyeT*)(block_gpu+i*stride+bidy*VEC);
            #pragma unroll
            for(int k=0;k<VEC;k++){
                s[k]+=t[k];
            }
        }
        if(gdimx<=64){
            #pragma unroll
            for(int i=0;i<VEC;i++){
                t_out[i]=WarpReduceSum(s[i]);
            }
        }
        else{
            #pragma unroll
            for(int i=0;i<VEC;i++){
                t_out[i]=BlockReduceSum(s[i],smem);
            }
        }
        if(tid==0) *(VecTpyeS*)(sum_gpu+bidy*VEC)=*(VecTpyeS*)t_out;
    }
}

#define REDUCE_VEC_KERNEL(VEC)                                                                     \
    if(block_num*stride/VEC>maxblocks)block_num=(maxblocks-1)/(stride/VEC)+1;                      \
    if(block_num>32){                                                                              \
        c10::TensorOptions options(DeviceType::CUDA);                                              \
        options.dtype(ScalarType::Int);                                                            \
        Tensor buffer=torch::empty(stride*block_num, options);                                     \
        Tensor semaphores=torch::empty(stride/VEC, options);                                       \
        AT_CUDA_CHECK(cudaMemsetAsync(semaphores.data_ptr(), 0, stride/VEC*sizeof(int), stream));  \
        ReduceVEC<scalar_t,T_ACC,VEC><<<dim3(block_num,stride/VEC),block_size,0,stream>>>          \
         (data_gpu,(T_ACC*)buffer.data_ptr(),(int*)semaphores.data_ptr(),output_gpu,length,stride);\
    }                                                                                              \
    else {                                                                                         \
        ReduceVEC<scalar_t,T_ACC,VEC><<<dim3(1,stride/VEC),block_size,0,stream>>>                  \
        (data_gpu,(T_ACC *)NULL,(int*)NULL,output_gpu,length,stride);                              \
    }

//for example input size is batchsize*sqlen*vlen,output size is vlen
//stride is vlen, length is batchsize*sqlen*vlen
template <typename scalar_t>
void bias_reduce_sum(scalar_t* output_gpu, scalar_t* data_gpu,int length,int stride)
{
    using T_ACC = acc_type<scalar_t, true>;
    auto stream = at::cuda::getCurrentCUDAStream();
    int N=length/stride;
    if(stride<=512){
        constexpr int block_size=REDUCE_BLOCK_SIZE;
        int block_num=((N-1)/block_size+1);
        int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
        int maxblocks=at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;
        if(stride%4!=0){
            REDUCE_VEC_KERNEL(1)
        }
        else if(stride==512 && N>=2000){
            REDUCE_VEC_KERNEL(8)
        }
        else{
            REDUCE_VEC_KERNEL(4)
        }
    }
    else{
        const int B =(stride - 1) / kColwiseReduceTileSize + 1;
        col_wise_reduce<<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(output_gpu,data_gpu,N,stride);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}//namespace native
}//namespace at
