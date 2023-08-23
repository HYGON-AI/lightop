#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <THC/THCGeneral.h>
#include <torch/extension.h>
#include "reduce.h"
using namespace std;

namespace at{
namespace native{

#define UNROLL 4
#define VEC 4
typedef unsigned int IndexType ;

template <typename scalar_t,typename accscalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fuse_bias_dropout_add_train_vec(
    cuda::detail::TensorInfo<scalar_t, IndexType> a,
    cuda::detail::TensorInfo<scalar_t, IndexType> a1,
    cuda::detail::TensorInfo<scalar_t, IndexType> a2,
    cuda::detail::TensorInfo<scalar_t, IndexType> b,
    cuda::detail::TensorInfo<uint8_t, IndexType> c,
    IndexType totalElements,
    IndexType dim,
    accscalar_t p,
    PhiloxCudaState philox_args,
    IndexType rand_offset) {

  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using MaskLoadT = memory::aligned_vector<uint8_t, VEC>;

  auto seeds = at::cuda::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  accscalar_t pinv = accscalar_t(1)/p;

  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType linearIndex=idx*VEC;
 
  float4 rand;
  if(linearIndex<totalElements)
  {
      scalar_t input4[VEC];
      scalar_t bias4[VEC];
      scalar_t residual4[VEC];
      uint8_t  mask4[VEC];
      IndexType t_dim=linearIndex%dim;
      hiprand_init(std::get<0>(seeds),
              idx%rand_offset,
              std::get<1>(seeds)+idx/rand_offset*VEC,
              &state);

      rand  = hiprand_uniform4(&state);
      rand.x = rand.x < p;
      rand.y = rand.y < p;
      rand.z = rand.z < p;
      rand.w = rand.w < p;
      LoadT *value = reinterpret_cast<LoadT*>(&input4);
      LoadT *value1 = reinterpret_cast<LoadT*>(&bias4);
      LoadT *value2 = reinterpret_cast<LoadT*>(&residual4);

      *value = *reinterpret_cast<LoadT*>(&a.data[linearIndex]);
      *value1 = *reinterpret_cast<LoadT*>(&a1.data[t_dim]);
      *value2 = *reinterpret_cast<LoadT*>(&a2.data[linearIndex]);

      #pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        
        input4[ii] = (input4[ii]+bias4[ii])*(&rand.x)[ii]*pinv+residual4[ii];
        mask4[ii] = (uint8_t)(&rand.x)[ii];
      }
      *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) = *reinterpret_cast<LoadT*>(&input4[0]);
      *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask4[0]);
   
  }

}

template <typename scalar_t,typename accscalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fuse_bias_dropout_add_eval_vec(
    cuda::detail::TensorInfo<scalar_t, IndexType> a,
    cuda::detail::TensorInfo<scalar_t, IndexType> a1,
    cuda::detail::TensorInfo<scalar_t, IndexType> a2,
    cuda::detail::TensorInfo<scalar_t, IndexType> b,
    IndexType totalElements,
    IndexType dim) {

  using LoadT = memory::aligned_vector<scalar_t, VEC>;

  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType linearIndex=idx*VEC;

  if(linearIndex<totalElements) {
    IndexType bias_idx=linearIndex%dim;
    scalar_t src[VEC];
    scalar_t src2[VEC];
    scalar_t bias[VEC];

    LoadT *value = reinterpret_cast<LoadT*>(&src);
    LoadT *value1 = reinterpret_cast<LoadT*>(&bias);
    LoadT *value2 = reinterpret_cast<LoadT*>(&src2);

    *value = *reinterpret_cast<LoadT*>(&a.data[linearIndex]);
    *value1 = *reinterpret_cast<LoadT*>(&a1.data[bias_idx]);
    *value2 = *reinterpret_cast<LoadT*>(&a2.data[linearIndex]);
    scalar_t r[VEC];
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      r[ii] = src[ii]+bias[ii]+src2[ii];
     
    }
    *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
  
  }
}

template <typename scalar_t,typename accscalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fuse_bias_dropout_add_eval(
    cuda::detail::TensorInfo<scalar_t, IndexType> a,
    cuda::detail::TensorInfo<scalar_t, IndexType> a1,
    cuda::detail::TensorInfo<scalar_t, IndexType> a2,
    cuda::detail::TensorInfo<scalar_t, IndexType> b,
    IndexType totalElements,
    IndexType dim) {

  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<totalElements) {
    IndexType bias_idx=idx%dim;

    b.data[idx]=a.data[idx]+a1.data[bias_idx]+a2.data[idx];

  }
}

 /**
* backward functions
* */
template<typename scalar_t,typename accscalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fuse_bias_dropout_add_kernel_backwardVEC(
        at::cuda::detail::TensorInfo<scalar_t, IndexType> input_grad,
        at::cuda::detail::TensorInfo<scalar_t, IndexType> grad,
        at::cuda::detail::TensorInfo<uint8_t, IndexType>  mask,
        IndexType totalElements,accscalar_t p) {
      
      using LoadT = memory::aligned_vector<scalar_t, VEC>;
      using MaskLoadT = memory::aligned_vector<uint8_t, VEC>;

      accscalar_t pinv = accscalar_t(1)/p;
      IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
      IndexType linearIndex=idx*VEC;
      if(linearIndex<totalElements)
      {
          scalar_t v_input_grad[VEC];
          scalar_t v_grad[VEC];
          uint8_t  mask4[VEC];
          LoadT *value = reinterpret_cast<LoadT*>(&v_input_grad);
          LoadT *value2 = reinterpret_cast<LoadT*>(&v_grad);
          MaskLoadT *value3 = reinterpret_cast<MaskLoadT*>(&mask4);

          *value = *reinterpret_cast<LoadT*>(&input_grad.data[linearIndex]);
          *value2 = *reinterpret_cast<LoadT*>(&grad.data[linearIndex]);
          *value3 = *reinterpret_cast<MaskLoadT*>(&mask.data[linearIndex]);
          #pragma unroll
          for (int ii = 0; ii < VEC; ii++) {

            v_input_grad[ii] = v_grad[ii]*(float)mask4[ii]*pinv;
          }
          *(reinterpret_cast<LoadT*>(&input_grad.data[linearIndex])) = *reinterpret_cast<LoadT*>(&v_input_grad[0]);
      
      }
        
    
}


std::tuple<Tensor,Tensor> fuse_bias_dropout_add_forward(const Tensor& self,const Tensor& bias, const Tensor& other,float p, bool training){
    AT_ASSERTM(self.sizes() == other.sizes(), "x must be the same size as y");
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(self.size(-1) == bias.size(-1) && other.size(-1) == bias.size(-1),
        "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(p >=0.0f&&p<1.0f, "p mast be between 0 and 1");
    AT_ASSERTM(self.is_cuda()&&bias.is_cuda()&&other.is_cuda(), "self and bias and other must on dcu");
    p=1-p;
   
    Tensor ret = at::empty_like(self);
    Tensor mask = at::empty_like(self, self.options().dtype(kByte));
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
    const IndexType nelem = self.numel();
    const IndexType dim=self.sizes().back();
    const IndexType rows=nelem/dim;
    const IndexType block_size = 256;
    IndexType rand_offset=at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor*
                                    at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    dim3 dim_block(block_size);
    dim3 grid((nelem + block_size -1)/block_size);
    grid.x = std::min(rand_offset / 256, grid.x);
  //number of times random will be generated per thread, to offset philox counter in thc random state
    int64_t counter_offset = ((nelem - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
    PhiloxCudaState rng_engine_inputs;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }
    //launcher<unsigned int>(self,other, ret, mask, p, nelem, rng_engine_inputs, grid, dim_block);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "fuse_bias_dropout_add_forward",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          accscalar_t pa = (accscalar_t)(p);
          auto self_info =
                cuda::detail::getTensorInfo<scalar_t, IndexType>(self);
          auto bias_info =
                cuda::detail::getTensorInfo<scalar_t, IndexType>(bias);
          auto other_info =
                cuda::detail::getTensorInfo<scalar_t, IndexType>(other);
          auto ret_info =
                cuda::detail::getTensorInfo<scalar_t, IndexType>(ret);
          auto mask_info =
                cuda::detail::getTensorInfo<uint8_t, IndexType>(mask);
          self_info.collapseDims();
          bias_info.collapseDims();
          other_info.collapseDims();
          ret_info.collapseDims();
          mask_info.collapseDims();
          if(training){
              int gridsize=(nelem/VEC-1)/block_size+1;
              fuse_bias_dropout_add_train_vec<scalar_t,accscalar_t>
                  <<<gridsize, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      bias_info,
                      other_info,
                      ret_info,
                      mask_info,
                      nelem,
                      dim,
                      pa,
                      rng_engine_inputs,
                      rand_offset);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
          else{
             
            int gridsize=(nelem/VEC-1)/block_size+1;
            fuse_bias_dropout_add_eval_vec<
                scalar_t,
                accscalar_t>
                <<<gridsize, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_info,
                    bias_info,
                    other_info,
                    ret_info,
                    nelem,
                    dim
                );
             
              C10_CUDA_KERNEL_LAUNCH_CHECK();

          }
        });

    return std::tuple<Tensor,Tensor>(ret, mask);   
}


std::tuple<Tensor,Tensor> fuse_bias_dropout_add_backward(const Tensor &grad,const Tensor &mask, float p) {
    p=1-p;
    const IndexType nelem = grad.numel();
    const IndexType dim=grad.sizes().back();
    auto dst_options = grad.options().requires_grad(false);
    Tensor input_grad  = torch::empty(grad.sizes(), dst_options);
    Tensor bias_grad  = torch::empty({dim}, dst_options);
    const IndexType block_size = 256;
    dim3 dim_block(block_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad.scalar_type(),
        "fuse_bias_dropout_add_backward",
        [&] {
            using accscalar_t = acc_type<scalar_t, true>;
            accscalar_t pa = (accscalar_t)(p);
            auto input_grad_info = cuda::detail::getTensorInfo<scalar_t, IndexType>(input_grad);
            auto grad_info = cuda::detail::getTensorInfo<scalar_t, IndexType>(grad);
            auto mask_info = cuda::detail::getTensorInfo<uint8_t, IndexType>(mask);
            // collapsed to 1d
            input_grad_info.collapseDims();
            grad_info.collapseDims();
            mask_info.collapseDims();
            int gridsize=(nelem/VEC-1)/block_size+1;
            fuse_bias_dropout_add_kernel_backwardVEC<scalar_t,accscalar_t>
            <<<gridsize, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    input_grad_info, grad_info, mask_info, nelem,pa);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            
            bias_reduce_sum(bias_grad.data_ptr<scalar_t>(),input_grad.data_ptr<scalar_t>(),nelem,dim);
      });
      return std::tuple<Tensor,Tensor>(input_grad,bias_grad);

}


}
}


