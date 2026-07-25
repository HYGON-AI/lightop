#include "reduce.h"

namespace at {
namespace native {

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

template <typename T>
__inline__ __device__ T WarpReduceSumMutiLine(T val,int start=32) {
  for (int offset = start; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename scalar_t,int VEC>
__global__ void ReduceSumMutiLine(scalar_t* x, scalar_t* ret,int cols,
                             int lines_per_warp,int warps,bool mean=false){
  int tid=threadIdx.x;
  int bid=blockIdx.x;
  x+=bid*cols*lines_per_warp*warps;
  int threads_per_line=64/lines_per_warp;
  int lineid=tid/threads_per_line;
  int line=tid%threads_per_line;
  using VecType=memory::aligned_vector<scalar_t, VEC>;
  scalar_t invec[VEC];
  float sum=0;
  if(line<cols/VEC){
    *(VecType*)invec=*((VecType*)(x+lineid*cols+line*VEC));
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

template <typename scalar_t,int VEC,int blocksize,bool mean>
__global__ void ReduceSum(scalar_t* input, scalar_t* ret,int cols){
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  constexpr int share_size=blocksize/C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  int tcol=cols/VEC;
  VecTpye tx;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(ret+blockidx);
  float sum=0;
  for (int j = threadidx; j < tcol; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      sum += to_float(px[i]);
    }
  }
  sum = BlockReduceSum<float,share_size>(sum,val_shared);
  if (threadidx == 0) {
    if constexpr(mean)sum/=cols;
    from_float(py[0],sum);
  }
}

template <typename scalar_t,int VEC,int blocksize,bool mean,int cols>
__global__ void ReduceSumFast(scalar_t* input, scalar_t* ret){
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  constexpr int share_size=blocksize/C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye tx;
  new_type* px=(new_type*)(&tx);
  new_type* py=(new_type*)(ret+blockidx);
  float sum=0;
  #pragma unroll
  for (int j = threadidx; j < cols/VEC; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      sum += to_float(px[i]);
    }
  }
  sum = BlockReduceSum<float,share_size>(sum,val_shared);
  if (threadidx == 0) {
    if constexpr(mean)sum/=cols;
    from_float(py[0],sum);
  }
}

template <typename scalar_t,int VEC,int blocksize,bool mean,int cols>
__global__ void ReduceSumFast(scalar_t* input, float* ret,int lines_per_block){
  int blockidx=blockIdx.x;
  int threadidx=threadIdx.x;
  int block_cols=blockidx*cols;
  constexpr int share_size=blocksize/C10_WARP_SIZE;
  __shared__ float val_shared[share_size];
  using VecTpye=memory::aligned_vector<scalar_t, VEC>;
  using new_type = __change_type<scalar_t>;
  VecTpye *X=(VecTpye*)(input+block_cols);
  VecTpye tx;
  new_type* px=(new_type*)(&tx);
  float sum=0;
  #pragma unroll
  for (int j = threadidx; j < cols/VEC; j += blocksize) {
    tx = X[j];
    #pragma unroll
    for(int i=0;i<VEC;i++){
      sum += to_float(px[i]);
    }
  }
  sum = BlockReduceSum<float,share_size>(sum,val_shared);
  if (threadidx == 0) {
    if constexpr(mean)sum/=cols;
    atomicAdd(ret+blockidx/lines_per_block,sum);
  }
}
template <typename scalar_t>
__global__ void type_covert(scalar_t* out,float *in,int size)
{
  for(int i=threadIdx.x;i<size;i+=256){
    using new_type = __change_type<scalar_t>;
    new_type* p =(new_type*)out;
    from_float(p[i],in[i]);
    in[i]=0.0f;
  }
}
template <typename scalar_t,int colwise_reduce_col,int colwise_reduce_row>
__global__ void col_wise_sum(scalar_t *dst, const scalar_t *src,int M,int N,bool mean=false){
    __shared__ float g_shared[colwise_reduce_row][colwise_reduce_col*2];
    src+=blockIdx.y*M*N;
    dst+=blockIdx.y*N;
    using VecType=memory::aligned_vector<scalar_t, 2>;
    scalar_t temp[2];
    const int j = blockIdx.x * blockDim.x*2 + threadIdx.x*2;
    float grad_sum[2] = {0};
    if (j < N) {
        for (int i = threadIdx.y; i < M; i += blockDim.y) {
          *(VecType*)temp=*((VecType*)(src+i * N + j));
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
  static float* tmp_out_ptr = nullptr;
  constexpr int temp_out_size = 100000;
  if(tmp_out_ptr == nullptr){
    hipMalloc(&tmp_out_ptr, temp_out_size);
    hipMemset(tmp_out_ptr,0,temp_out_size);
  }
  // printf("cols=%d,batch=%d,stride=%d\n",cols,batch,stride);
  // std::cout<<size<<std::endl;
  auto stream = at::cuda::getCurrentCUDAStream();
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
      if(cols<=512){
        VEC_SWITCH(cols,[&] {
          int lines_per_warp=get_lines_per_warp(cols,VEC);
          int warps=2;
          int blocks=(ret.numel()-1)/(lines_per_warp*warps)+1;
          ReduceSumMutiLine<scalar_t,VEC><<<blocks,64*warps,0,stream>>>(x_data,ret_data,cols,lines_per_warp,warps,mean);
        });
      }
      else{
        BOOL_SWITCH(mean,is_mean,[&] {
          if(cols==1024){
            constexpr int blocksize=128;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,1024><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data);
          }
          else if(cols==2048){
            constexpr int blocksize=256;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,2048><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data);
          }
          else if(cols==4096){
            constexpr int blocksize=256;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,4096><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data);
          }
          else if(cols==8192){
            constexpr int blocksize=256;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,8192><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data);
          }
          else if(cols==16384){
            constexpr int blocksize=256;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,16384><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data);
          }
          else if(cols%16384==0){
            constexpr int blocksize=256;
            constexpr int cols_per_block=4096;
            int lines_per_block=cols/cols_per_block;
            ReduceSumFast<scalar_t,8,blocksize,is_mean,cols_per_block><<<ret.numel()*lines_per_block,blocksize,0,stream>>>(x_data,tmp_out_ptr,lines_per_block);
            type_covert<scalar_t><<<1,256>>>(ret_data,tmp_out_ptr,ret.numel());
          }
          else{
            constexpr int blocksize=512;
            ReduceSum<scalar_t,8,blocksize,is_mean><<<ret.numel(),blocksize,0,stream>>>(x_data,ret_data,cols);
          }
        });
      }
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