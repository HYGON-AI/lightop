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
#include "cuda_meta.h"

namespace at{
namespace native{

#define VEC 4

template <typename scalar_t,typename T_ACC>
__global__ void drop_path_kernel(scalar_t* input,scalar_t* output, scalar_t* mask,T_ACC keep_prob, bool scale_by_keep,PhiloxCudaState philox_args,int N)
{
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*VEC;
    if(idx>=N)return;
    int maskID=blockIdx.y;
    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds),maskID,std::get<1>(seeds),&state);
    float rand = curand_uniform(&state);
    scalar_t tmask;
    if(scale_by_keep)tmask=(rand<keep_prob)/keep_prob;
    else tmask=rand<keep_prob;
    if(idx==0)mask[maskID]=tmask;
    idx+=N*maskID;
    if(tmask==0){
      const scalar_t zeros[]={0,0,0,0};
      *(LoadT*)(output+idx)=*(LoadT*)zeros;
    }
    else{
      scalar_t input4[VEC];
      scalar_t output4[VEC];
      *(LoadT*)input4=*(LoadT*)(input+idx);
      #pragma unroll
      for(int i=0;i<VEC;i++){
          output4[i]=input4[i]*tmask;
      }
      *(LoadT*)(output+idx)=*(LoadT*)output4;
    }
}

Tensor fused_drop_path_forward(const Tensor& self,const Tensor& ret, double p, bool training,bool scale_by_keep)
{
    if(!training)return Tensor();
    AT_ASSERTM(p >=0.0f&&p<1.0f, "p mast be between 0 and 1");
    AT_ASSERTM(self.is_cuda(), "input must on dcu");
    auto pself = self.expect_contiguous();
    const int batchsize=self.sizes().front();
    AT_ASSERT(batchsize<=600*256,"input's first dim size is too big");
    const int elem_per_batch=self.numel()/batchsize;
    AT_ASSERT(elem_per_batch%4==0,"element size per batch must be 4*interger");
    Tensor mask = at::empty({batchsize},pself->options());
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
    PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(4);
    }
    int blocksize=256;
    int gridx=(elem_per_batch/VEC-1)/blocksize+1;
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      pself->scalar_type(),
      "drop_path_forward",
      [&] {
        p=1-p;
        using T_ACC = acc_type<scalar_t, true>;
        T_ACC keep_prob = (T_ACC)(p);
        scalar_t* self_data = pself->data_ptr<scalar_t>();
        scalar_t* ret_data=ret.data_ptr<scalar_t>();
        scalar_t* mask_data=mask.data_ptr<scalar_t>();
        drop_path_kernel<scalar_t,T_ACC><<<dim3(gridx,batchsize), blocksize, 0, at::cuda::getCurrentCUDAStream()>>>(self_data,ret_data,mask_data,keep_prob,scale_by_keep,rng_engine_inputs,elem_per_batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

      });
     return mask;
}


template <typename scalar_t>
__global__ void drop_path_backward_kernel(scalar_t* ret,scalar_t* grad,scalar_t* mask,int N)
{
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x)*VEC;
  if(idx>=N)return;
  int maskID=blockIdx.y;
  idx+=N*maskID;
  scalar_t tmask=mask[maskID];
  if(tmask==0){
    const scalar_t zeros[]={0,0,0,0};
    *(LoadT*)(ret+idx)=*(LoadT*)zeros;
  }else{
    scalar_t grad4[VEC];
    scalar_t output4[VEC];
    *(LoadT*)grad4=*(LoadT*)(grad+idx);
    #pragma unroll
    for(int i=0;i<VEC;i++){
        output4[i]=grad4[i]*tmask;
    }
    *(LoadT*)(ret+idx)=*(LoadT*)output4;
  }
}

Tensor fused_drop_path_backward(const Tensor& grad, const Tensor& mask)
{
  auto pgrad = grad.expect_contiguous();
  auto ret = at::empty_like(*pgrad);
  const int batchsize=pgrad->sizes().front();
  const int elem_per_batch=pgrad->numel()/batchsize;
  int blocksize=256;
  int gridx=(elem_per_batch/VEC-1)/blocksize+1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    pgrad->scalar_type(),
    "fused_ln_dropout_backward",
    [&] {
      scalar_t* dY_data = pgrad->data_ptr<scalar_t>();
      scalar_t* ret_data = ret.data_ptr<scalar_t>();
      scalar_t* mask_data = mask.data_ptr<scalar_t>();
      drop_path_backward_kernel<scalar_t><<<dim3(gridx,batchsize), blocksize, 0, at::cuda::getCurrentCUDAStream()>>>(ret_data,dY_data, mask_data,elem_per_batch);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return ret;
}

struct TORCH_API structured_drop_path : public TensorIteratorBase {  
  void meta(const Tensor& self) {
    build_unary_op(maybe_get_output(), self);
  }
  Tensor impl(const Tensor& self,const Tensor& ret,double p, bool training,bool scale_by_keep){
    return fused_drop_path_forward(self, ret, p, training, scale_by_keep);
  }
};

TORCH_META_FUNC_FINAL_SET_OUTPUT(drop_path)

std::tuple<Tensor,Tensor> wrapper_drop_path(const Tensor& self,double p, bool training,bool scale_by_keep) {
  structured_drop_path_functional op;
  op.meta(self);
  auto mask=op.impl(self, *op.outputs_[0],p,training,scale_by_keep);
  return std::tuple<Tensor,Tensor>(std::move(op.outputs_[0]).take(),mask);
}


class drop_pathFunction : public torch::autograd::Function<drop_pathFunction> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::Variable& x,
    double p, bool training,bool scale_by_keep) {
    auto ret = wrapper_drop_path(x,p,training,scale_by_keep);
    ctx->save_for_backward({std::get<1>(ret)});
    return {std::get<0>(ret)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
    auto saved = ctx->get_saved_variables();
    auto mask = saved[0];
    auto ret = fused_drop_path_backward(gradOutput[0],mask);
    return {ret,torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};

Tensor drop_path_forward_autograd(const Tensor& x,double p, bool training,bool scale_by_keep) {
   auto result = drop_pathFunction::apply(x,p,training,scale_by_keep);
   return result[0];
}
}
}


