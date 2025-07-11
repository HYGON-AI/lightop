#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include <torch/extension.h>

#include "reduce.h"


namespace at{
namespace native{
static constexpr int VEC=4;

template <typename acc_t>
__device__ acc_t bias_gelu(acc_t x) {
    using opmath_t = at::opmath_type<acc_t>;
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
}


template <typename acc_t>
__device__ inline acc_t  bias_gelu_back(acc_t x, acc_t dy) {
    using opmath_t = at::opmath_type<acc_t>;
    constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
    constexpr opmath_t kAlpha = M_SQRT1_2;
    const opmath_t cdf =
        opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
    const opmath_t pdf =
        c10::cuda::compat::exp(
            opmath_t(-0.5) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x)) *
        kBeta;
    return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
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

Tensor fused_conv_bias_add_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const Tensor &extra_x) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int N = x.numel();
    int dim=x.sizes().back();
    auto dst_options = x.options().requires_grad(false);
    Tensor results = torch::empty(x.sizes(), dst_options);
    // const int ThreadsPerBlock=256;
    // AT_DISPATCH_FLOATING_TYPES_AND2(
    //   at::ScalarType::Half,
    //   at::ScalarType::BFloat16,
    //   x.scalar_type(),
    //   "conv_bias_add_forward",
    //   [&] {
    //     using T_ACC = acc_type<scalar_t, true>;
    //     if(dim%VEC==0){
    //         int gridsize=(N/VEC-1)/ThreadsPerBlock+1;
    //         bias_gelu_forwardVEC<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
    //             results.data_ptr<scalar_t>(),
    //             x.data_ptr<scalar_t>(),
    //             bias.data_ptr<scalar_t>(),
    //             N,dim);
    //     }
    //     else{
    //         int gridsize=(N-1)/ThreadsPerBlock+1;
    //         bias_gelu_forward<scalar_t, T_ACC><<<gridsize, ThreadsPerBlock, 0, stream>>>(
    //                 results.data_ptr<scalar_t>(),
    //                 x.data_ptr<scalar_t>(),
    //                 bias.data_ptr<scalar_t>(),
    //                 N,dim);
    //     }
    //     C10_CUDA_KERNEL_LAUNCH_CHECK();
    // });
    return results;
}

}//namespace native
}//namespace at

