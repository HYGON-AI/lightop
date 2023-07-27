#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include <THC/THCGeneral.h>
#include <torch/extension.h>
#include "layernorm.h"


namespace at{
namespace native{

static constexpr int VEC = 4;
template <typename acc_t>
__device__ acc_t bias_gelu(acc_t y) {
 
  return c10::cuda::compat::normcdf(y) * y;
  
}


template <typename acc_t>
__device__ acc_t bias_gelu_back(acc_t y, acc_t g) {
constexpr acc_t kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
const acc_t cdf = c10::cuda::compat::normcdf(y);
const acc_t pdf = c10::cuda::compat::exp((acc_t)-0.5 * y * y) * kBeta;
return g * (cdf + y * pdf);
}


template <typename scalar_t, typename acc_t>
__global__ void bias_gelu_forward(scalar_t *dst, const scalar_t *src, const scalar_t *bias, int N,int dim) {
    int idx=(int)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        int j=idx%dim;
        dst[idx] = (scalar_t)bias_gelu<acc_t>(src[idx] + bias[j]);
    }
}

template <typename scalar_t, typename acc_t>
__global__ void bias_gelu_forwardVEC(scalar_t *dst, const scalar_t *src, const scalar_t *bias, int N,int dim) {
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_src[VEC];
    scalar_t v_bias[VEC];
    scalar_t v_output[VEC];
    int idx=((int)blockIdx.x*blockDim.x+threadIdx.x)*VEC;
    if(idx<N){
        int j=idx%dim;
        *(LoadT*)(v_src)=*(LoadT*)(src+idx);
        *(LoadT*)(v_bias)=*(LoadT*)(bias+j);
        #pragma unroll
        for(int i=0;i<VEC;i++){
            v_output[i]=(scalar_t)bias_gelu<acc_t>((acc_t)(v_src[i]+v_bias[i]));
        }
        *(LoadT*)(dst+idx) = *(LoadT*)v_output;
    }
}

template <typename scalar_t, typename acc_t>
__global__ void bias_gelu_backward(scalar_t *dst, const scalar_t *src, const scalar_t *bias, const scalar_t *grad, int N,int dim) {
    int idx=(int)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        int j=idx%dim;
        dst[idx] = (scalar_t)bias_gelu_back<acc_t>((acc_t)(src[idx] + bias[j]),(acc_t)grad[idx]);
    }
}

template <typename scalar_t, typename acc_t>
__global__ void bias_gelu_backwardVEC(scalar_t *dst, const scalar_t *src, const scalar_t *bias, const scalar_t *grad, int N,int dim) {
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_src[VEC];
    scalar_t v_bias[VEC];
    scalar_t v_grad[VEC];
    scalar_t v_output[VEC];
    int idx=((int)blockIdx.x*blockDim.x+threadIdx.x)*VEC;
    if(idx<N){
        int j=idx%dim;
        *(LoadT*)(v_src)=*(LoadT*)(src+idx);
        *(LoadT*)(v_grad)=*(LoadT*)(grad+idx);
        *(LoadT*)(v_bias)=*(LoadT*)(bias+j);
        #pragma unroll
        for(int i=0;i<VEC;i++){
            v_output[i]=(scalar_t)bias_gelu_back<acc_t>((acc_t)(v_src[i]+v_bias[i]), (acc_t)v_grad[i]);
        }
        *(LoadT*)(dst+idx) = *(LoadT*)v_output;
    }
}

template <typename scalar_t>
__global__ void bias_gelu_backward_bias_simple(scalar_t *dst, const scalar_t *src,int bsz,int dim){
    using T_ACC = acc_type<scalar_t, true>;
    T_ACC grad= 0;
    for(int i=threadIdx.x;i<bsz;i+=blockDim.x){
        grad+=src[i*dim+blockIdx.x];
    }
    grad = WarpReduceSum(grad);
    if(threadIdx.x==0)dst[blockIdx.x] = grad;
    
}

template <typename scalar_t>
__global__ void bias_gelu_backward_bias(scalar_t *dst, const scalar_t *src,int M,int N){
    using T_ACC = acc_type<scalar_t, true>;
    __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
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
    sum = WarpReduceSum(sum,16);
    if (threadIdx.x == 0) {
        const int j = blockIdx.x * blockDim.x + threadIdx.y;
        if (j < N) {
            dst[j] = sum;
        }
   }
}

Tensor fused_bias_gelu_forward(const Tensor &x, const Tensor &bias) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int N = x.numel();
    int dim=x.sizes().back();
    auto dst_options = x.options().requires_grad(false);
    Tensor results = torch::empty(x.sizes(), dst_options);
    const int ThreadsPerBlock=256;
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "bias_gelu_forward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        if(dim%VEC==0){
            int gridsize=(N/VEC-1)/ThreadsPerBlock+1;
            bias_gelu_forwardVEC<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
                results.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                N,dim);
        }
        else{
            int gridsize=(N-1)/ThreadsPerBlock+1;
            bias_gelu_forward<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
                    results.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    N,dim);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return results;
}

std::tuple<Tensor,Tensor> fused_bias_gelu_backward(const Tensor &x, const Tensor &bias, const Tensor &grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int N = x.numel();
    int dim=x.sizes().back();
    int bsz=N/dim;
    auto dst_options = x.options().requires_grad(false);
    Tensor input_grad  = torch::empty(x.sizes(), dst_options);
    Tensor bias_grad;
    bool cal_bias=true;
    const int ThreadsPerBlock = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_grad.scalar_type(),
      "bias_gelu_backward",
      [&] {
        using T_ACC = acc_type<scalar_t, true>;
        if(dim%VEC==0){
            int gridsize=(N/VEC-1)/ThreadsPerBlock+1;
            bias_gelu_backwardVEC<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
            input_grad.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            N,dim);
        }
        else{
            int gridsize=(N-1)/ThreadsPerBlock+1;
            bias_gelu_backward<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
                input_grad.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                grad.data_ptr<scalar_t>(),
                N,dim);
        }
        
        if (bsz < 2048 && dim<2048 && N < 600000) {
            // For small batch size, do colwise reduce directly.
            bias_grad  = torch::empty({dim}, dst_options);
            bias_gelu_backward_bias_simple<scalar_t><<<dim, 64, 0, stream>>>(bias_grad.data_ptr<scalar_t>(),input_grad.data_ptr<scalar_t>(),bsz,dim);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else if(dim>=512 && bsz/dim<=32){
            bias_grad  = torch::empty({dim}, dst_options);
            const int B =(dim + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
            bias_gelu_backward_bias<scalar_t><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(bias_grad.data_ptr<scalar_t>(),input_grad.data_ptr<scalar_t>(),bsz,dim);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else {
            bias_grad=input_grad.view({-1,dim}).sum(0);
        }
        
    });
    return std::tuple<Tensor,Tensor>(input_grad,bias_grad);
}

}//namespace native
}//namespace at

