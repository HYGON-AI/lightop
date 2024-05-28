#include "act_groupnorm.h"

namespace at{
namespace native{

std::tuple<Tensor,Tensor,Tensor>
celu_gn_forward(const Tensor& self,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,double alpha,bool training)
{
  Tensor ret = at::empty_like(self),mean,rstd;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16, 
    self.scalar_type(),
    "celu_gn_forward",
    [&] {
      CELU<scalar_t> celuop;
      celuop.alpha=alpha;
      act_groupnorm_forward_impl<scalar_t,CELU<scalar_t>>(self,weight,bias,num_groups,num_channels,eps,training,celuop,ret,mean,rstd);
    });
    return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}
 
std::tuple<Tensor,Tensor,Tensor>
celu_gn_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma,double alpha)
{
  Tensor dX = at::empty_like(X),dgamma = at::empty_like(gamma),dbeta = at::empty_like(gamma);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "celu_gn_backward",
    [&] {
      CELU<scalar_t> celuop;
      celuop.alpha=alpha;
      act_groupnorm_backward_impl<scalar_t,CELU<scalar_t>>(grad,X,mean,rstd,gamma,celuop,dX,dgamma,dbeta);
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}

class fuseCeluGnFunction : public torch::autograd::Function<fuseCeluGnFunction> {

  public:
    static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& x,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& bias,
      int num_groups,int num_channels,double eps,double alpha,bool training) {
      auto px=x.expect_contiguous();
      auto pweight=weight.expect_contiguous();
      auto pbias=bias.expect_contiguous();
      auto ret = celu_gn_forward(*px, *pweight,*pbias,num_groups,num_channels,eps,alpha,training);
      ctx->save_for_backward({*px,std::get<1>(ret),std::get<2>(ret),*pweight});
      ctx->saved_data["alpha"] = alpha;
      return {std::get<0>(ret)};
  }
    static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& gradOutput) {
      auto saved = ctx->get_saved_variables();
      auto X = saved[0];
      auto mean = saved[1];
      auto rstd = saved[2];
      auto gamma = saved[3];
      double alpha = ctx->saved_data["alpha"].toDouble();
      auto ret = celu_gn_backward(gradOutput[0],X,mean,rstd,gamma,alpha);
      return {std::get<0>(ret),std::get<1>(ret),std::get<2>(ret),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable(),torch::autograd::Variable()};
  }
};


Tensor fuse_celu_gn_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,double alpha,bool training) {
   auto result = fuseCeluGnFunction::apply(x, weight,bias,num_groups,num_channels,eps,alpha,training);
   return result[0];
  }
}
}
