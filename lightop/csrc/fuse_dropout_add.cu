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

using namespace std;

namespace at{
namespace native{

#define UNROLL 4
#define VEC 4
typedef unsigned int IndexType ;

template <
    typename scalar_t,
    typename accscalar_t>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fused_dropout_kernel_vec(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> a,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> a2,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
    at::cuda::detail::TensorInfo<uint8_t, IndexType> c,
    IndexType totalElements,
    accscalar_t p,
    PhiloxCudaState philox_args) {

  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using MaskLoadT = memory::aligned_vector<uint8_t, VEC>;

  auto seeds = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);
  accscalar_t pinv = accscalar_t(1)/p;
  float4 rand;
  for (IndexType linearIndex = idx * VEC;
      linearIndex < totalElements;
      linearIndex += gridDim.x * blockDim.x * VEC) {
    scalar_t src[VEC];
    scalar_t src2[VEC];
    LoadT *value = reinterpret_cast<LoadT*>(&src);
    LoadT *value2 = reinterpret_cast<LoadT*>(&src2);
    rand = curand_uniform4(&state);
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;
    *value = *reinterpret_cast<LoadT*>(&a.data[linearIndex]);
    *value2 = *reinterpret_cast<LoadT*>(&a2.data[linearIndex]);

    scalar_t r[VEC];
    uint8_t mask[VEC];
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      r[ii] = src[ii]*(&rand.x)[ii]*pinv+src2[ii];
      mask[ii] = (uint8_t)(&rand.x)[ii];
    }
    *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    __syncthreads();
  }
}

template<typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const at::Tensor& src, const at::Tensor& mask, accscalar_t scale){
   auto iter = at::TensorIteratorConfig()
     .check_all_same_dtype(false)
     .add_output(ret)
     .add_input(src)
     .add_input(mask)
     .build();

   at::native::gpu_kernel(
       iter,
       [=]GPU_LAMBDA(const scalar_t src_val, const uint8_t mask_val) -> scalar_t {
          return (float)mask_val * src_val * scale;
       });
}

std::tuple<Tensor,Tensor> fused_add_dropout_forward(const Tensor& self,const Tensor& other, double p, bool training){
    AT_ASSERTM(self.sizes() == other.sizes(), "x must be the same size as y");
    AT_ASSERTM(p >=0.0f&&p<1.0f, "p mast be between 0 and 1");
    AT_ASSERTM(self.is_cuda()&&other.is_cuda(), "self and other must on dcu");
  p=1-p;
  if(training){
      Tensor ret = at::empty_like(self);
      Tensor mask = at::empty_like(self, self.options().dtype(kByte));
      auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
      const int64_t nelem = self.numel();
      const int64_t block_size = 256;
      unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
      dim3 dim_block(block_size);
      dim3 grid((nelem + block_size -1)/block_size);
      grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
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
          "fused_add_dropout_forward",
          [&] {
            using accscalar_t = acc_type<scalar_t, true>;
            accscalar_t pa = (accscalar_t)(p);
            auto self_info =
                cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
            auto other_info =
                cuda::detail::getTensorInfo<scalar_t, unsigned int>(other);
            auto ret_info =
                cuda::detail::getTensorInfo<scalar_t, unsigned int>(ret);
            auto mask_info =
                cuda::detail::getTensorInfo<uint8_t, unsigned int>(mask);
            fused_dropout_kernel_vec<
                scalar_t,
                accscalar_t>
                <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_info,
                    other_info,
                    ret_info,
                    mask_info,
                    nelem,
                    pa,
                    rng_engine_inputs);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
      return std::tuple<Tensor,Tensor>(ret, mask);
   }
   else{
      return std::tuple<Tensor,Tensor>(self+other, Tensor());
   }
}

Tensor masked_scale_cuda(const Tensor& self, const Tensor& mask, double scale){
   Tensor ret = at::empty_like(self, self.suggest_memory_format());
   TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
   AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, ret.scalar_type(), "masked_scale", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      accscalar_t pa = (accscalar_t)(scale);
      masked_scale_kernel<scalar_t>(ret, self, mask, pa);
  });
  return ret;
}

std::tuple<Tensor,Tensor> fused_add_dropout_backward(Tensor grad, Tensor mask, double p) {
    p=1-p;
  if (grad.requires_grad()) {
    // Use autograd-friendly backward if double backward is required
    Tensor ret=grad * (mask.type_as(grad) * (1. / p));
    return std::tuple<Tensor,Tensor>(ret,grad);
  } else {
    auto pgrad = grad.expect_contiguous();
    Tensor ret=masked_scale_cuda(*pgrad, mask, 1. / p);
    return std::tuple<Tensor,Tensor>(ret,grad);
  }
}

class fuseAddDropoutFunction : public torch::autograd::Function<fuseAddDropoutFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    const torch::autograd::Variable& y,
    double rate,bool training) {
    auto ret = fused_add_dropout_forward(x,y,rate,training);
    ctx->save_for_backward({std::get<1>(ret)});
    ctx->saved_data["rate"] = rate;
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto mask = ctx->get_saved_variables()[0];
    double rate = ctx->saved_data["rate"].toDouble();
    auto ret = fused_add_dropout_backward(gradOutput[0],mask,rate);
    return {std::get<0>(ret),std::get<1>(ret),torch::autograd::Variable(),torch::autograd::Variable(),};
  }
};
Tensor fuse_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,double rate,bool training) {
   auto result = fuseAddDropoutFunction::apply(x,y,rate,training);
   return result[0];
}

}
}


