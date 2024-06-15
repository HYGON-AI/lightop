#include "act_groupnorm.h"
int get_groupnorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else if(cols>64)blocksize=128;
  else blocksize=64;
  return blocksize;
}

namespace at{
namespace native{

std::tuple<Tensor,Tensor,Tensor>
groupnorm_forward(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,bool training)
{
  Tensor ret = at::empty_like(self),mean,rstd;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16, 
    self.scalar_type(),
    "groupnorm_forward",
    [&] {
      PassThrough<scalar_t> passop;
      act_groupnorm_forward_impl<scalar_t,PassThrough<scalar_t>>(self,weight,bias,num_groups,num_channels,eps,training,passop,ret,mean,rstd);
    });
  return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}  

std::tuple<Tensor,Tensor,Tensor>
groupnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma)
{
  Tensor dX = at::empty_like(X),dgamma = at::empty_like(gamma),dbeta = at::empty_like(gamma);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "groupnorm_backward",
    [&] {
      PassThrough<scalar_t> passop; 
      act_groupnorm_backward_impl<scalar_t,PassThrough<scalar_t>>(grad,X,mean,rstd,gamma,passop,dX,dgamma,dbeta);
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

class fuseGroupNormFunction : public torch::autograd::Function<fuseGroupNormFunction> {

  public:
    static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& x,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& bias,
      int num_groups,int num_channels,double eps,bool training) {
      if(x.suggest_memory_format()==at::MemoryFormat::ChannelsLast){
        auto ret = groupnorm_forward(x, weight,bias,num_groups,num_channels,eps,training);
        if(training){
          ctx->save_for_backward({x,std::get<1>(ret),std::get<2>(ret),weight});
        }
        return {std::get<0>(ret)};
      }
      else{       
        auto px=x.expect_contiguous();
        auto ret = groupnorm_forward(*px, weight,bias,num_groups,num_channels,eps,training);
        if(training){
          ctx->save_for_backward({*px,std::get<1>(ret),std::get<2>(ret),weight});
        }
        return {std::get<0>(ret)}; 
      }  
  }
    static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
      auto saved = ctx->get_saved_variables();
      auto X = saved[0];
      AT_ASSERT(X.suggest_memory_format()!=at::MemoryFormat::ChannelsLast,"Lightop GroupNorm backward doesn't support ChannelsLast now.If you need the function,please contact us.");
      auto mean = saved[1];
      auto rstd = saved[2];
      auto gamma = saved[3];
      auto ret = groupnorm_backward(gradOutput[0],X,mean,rstd,gamma);
      return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};


Tensor fuse_groupnorm_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,bool training) {
   auto result = fuseGroupNormFunction::apply(x, weight,bias,num_groups,num_channels,eps,training);
   return result[0];
  }
}
}

