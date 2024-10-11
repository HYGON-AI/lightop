#include "reduce.h"


namespace at {
namespace native {


template <typename T>
__inline__ __device__ T WarpReduceSumMutiLine(T val,int start=32) {
  for (int offset = start; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename scalar_t,int VEC>
__global__ void ReduceSum(scalar_t* x, scalar_t* ret,int cols,int lines_per_warp,int warps,bool mean=false){
  
  int tid=threadIdx.x;
  int bid=blockIdx.x;
  x+=bid*cols*lines_per_warp*warps;
  int threads_per_line=64/lines_per_warp;
  int lineid=tid/threads_per_line;
  int line=tid%threads_per_line;
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  scalar_t invec[VEC];
  float sum=0;
  if(line<cols/VEC){
    *(VecTpye*)invec=*((VecTpye*)(x+lineid*cols+line*VEC));
    #pragma unroll
    for(int i=0;i<VEC;i++){
      sum+=invec[i];
    }
  }
  sum = WarpReduceSumMutiLine<float>(sum,threads_per_line/2);
  if(mean)sum/=cols;
  if(line==0){
    ret[bid*lines_per_warp*warps+lineid]=sum;
  }
}

template <typename scalar_t,int colwise_reduce_col,int colwise_reduce_row>
__global__ void col_wise_sum(scalar_t *dst, const scalar_t *src,int M,int N,bool mean=false){
    __shared__ float g_shared[colwise_reduce_row][colwise_reduce_col*2];
    src+=blockIdx.y*M*N;
    dst+=blockIdx.y*N;
    using VecTpye=memory::aligned_vector<scalar_t, 2>;
    scalar_t temp[2];
    const int j = blockIdx.x * blockDim.x*2 + threadIdx.x*2;
    float grad_sum[2] = {0};
    if (j < N) {
        for (int i = threadIdx.y; i < M; i += blockDim.y) {
          *(VecTpye*)temp=*((VecTpye*)(src+i * N + j));
          grad_sum[0] += temp[0];
          grad_sum[1] += temp[1];
        }
    }
    g_shared[threadIdx.y][threadIdx.x*2] = grad_sum[0];
    g_shared[threadIdx.y][threadIdx.x*2+1] = grad_sum[1];
    __syncthreads();
    if(threadIdx.y<2){
      float sum=0;
      for(int i=0;i<colwise_reduce_row;i++){
        sum+=g_shared[i][threadIdx.x*2+threadIdx.y];
      }
      if(j<N){
        if(mean)sum/=M;
        dst[j+threadIdx.y]=sum;
      }
    }
}

int get_lines_per_warp(int cols,int VEC){
  if(cols<=32)return 64*VEC/32;
  else if(cols<=64)return 64*VEC/64;
  else if(cols<=128)return 64*VEC/128;
  else if(cols<=256)return 64*VEC/256;
  else return 1;
}

#define VEC_SWITCH(cols, ...)                   \
  [&] {                                         \
    if (cols %16 == 0&& batch>100000 ) {        \
      constexpr static int VEC = 16;            \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int VEC = 8;             \
      return __VA_ARGS__();                     \
    }                                           \
  }()


Tensor sum_opt_impl(const Tensor& x,int dim,bool mean){
  std::vector<long> size(x.dim()-1);
  int k=0;
  int cols=x.sizes()[dim];
  int stride=1;
  int batch=1;
  for(int i=0;i<x.dim();i++){
    if(i==dim)continue;
    size[k] = x.sizes()[i];
    if(i>dim)stride*=size[k];
    else batch*=size[k];
    k++;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if(stride==1&&cols>512){
    if(mean)return at::mean(x,dim);
    return at::sum(x,dim);
  }
  Tensor ret=at::empty(size,x.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(
  at::ScalarType::Half,
  at::ScalarType::BFloat16, 
  x.scalar_type(),
  "sum_opt",
  [&] {
    scalar_t* x_data = x.data_ptr<scalar_t>();
    scalar_t* ret_data=ret.data_ptr<scalar_t>();
    if(stride==1){
      VEC_SWITCH(cols,[&] {
        int lines_per_warp=get_lines_per_warp(cols,VEC);
        int warps=2;
        int blocks=(ret.numel()-1)/(lines_per_warp*warps)+1;
        ReduceSum<scalar_t,VEC><<<blocks,64*warps,0,stream>>>(x_data,ret_data,cols,lines_per_warp,warps,mean);
      });
    }
    else{
      constexpr int colwise_reduce_col=32;
      const int B =(stride - 1) / (colwise_reduce_col*2) + 1;
      if(stride>64&&cols<=128){
        constexpr int colwise_reduce_row=16;
        col_wise_sum<scalar_t,colwise_reduce_col,colwise_reduce_row><<<dim3(B,batch), dim3(colwise_reduce_col, colwise_reduce_row), 0, stream>>>(ret_data,x_data,cols,stride,mean);
      }
      else{
        constexpr int colwise_reduce_row=32;
        col_wise_sum<scalar_t,colwise_reduce_col,colwise_reduce_row><<<dim3(B,batch), dim3(colwise_reduce_col, colwise_reduce_row), 0, stream>>>(ret_data,x_data,cols,stride,mean);        
      }
    }
  });
  return ret;
 
}

Tensor sum_opt(const Tensor& x,int dim){
  return sum_opt_impl(x,dim,false);
}
Tensor mean_opt(const Tensor& x,int dim){
  return sum_opt_impl(x,dim,true);
}

}
}