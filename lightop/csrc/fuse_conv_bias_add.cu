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

Tensor fused_conv_bias_add_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const Tensor &extra_x, const Tensor &stride, const Tensor &padding, const Tensor &dilation, int groups) {
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

